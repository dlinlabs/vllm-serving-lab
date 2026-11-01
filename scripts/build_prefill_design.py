"""Rebuild reviewable design/order and frozen-parameter CPU scenarios. No GPU collector."""
import argparse
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prefill_design import simulate,requests_for,scored_steps,COEF
from prefill_identifiability import features,calibration_support
import greybox_simulator as gb
import evaluate_low_cap_frozen as lf


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def cases():
    result=[]
    for d in (0,16):
        for length in (1024,2048,4096,7168):
            result.append(dict(id=f'single-L{length}-D{d}',family='single',background_decoders=d,foreground_tokens=[length],foreground_offsets_ms=[0],sentinel=None,split='calibration' if length in (1024,7168) else 'held_case'))
        for name,lengths in [('uniform',[4096]*4),('mixed',[1024,1024,7168,7168]),('reverse',[7168,7168,1024,1024])]:
            result.append(dict(id=f'composition-{name}-D{d}',family='composition',background_decoders=d,foreground_tokens=lengths,foreground_offsets_ms=[0]*4,sentinel=None,split='calibration' if name=='uniform' else 'held_case'))
        for offset in (50,400,800):
            result.append(dict(id=f'sentinel-four-at{offset}-D{d}',family='sentinel',background_decoders=d,foreground_tokens=([512,512,7680,7680] if offset in (50,400) else [4096]*4),foreground_offsets_ms=[0]*4,sentinel=dict(input_tokens=512,arrival_ms=offset),split='held_case'))
        result.append(dict(id=f'sentinel-one-at200-D{d}',family='sentinel',background_decoders=d,foreground_tokens=[7680],foreground_offsets_ms=[0],sentinel=dict(input_tokens=512,arrival_ms=200),split='held_case'))
    return result


def order(cells):
    rng=random.Random(20261022);ids=[c['id'] for c in cells];out=[]
    def add(stage,case,**kw):out.append(dict(trial_id=f'trial-{len(out)+1:03d}',stage=stage,case=case,**kw))
    deck=ids.copy();rng.shuffle(deck)
    for c in deck:add('warmup',c,observer='light',scored=False)
    for c,pairs in [('single-L1024-D0',20),('composition-uniform-D16',10)]:
        first=['off','light']*(pairs//2);rng.shuffle(first)
        for pair,mode in enumerate(first,1):
            for m in [mode,'light' if mode=='off' else 'off']:
                add('observer_check',c,pair=pair,observer=m,paired_content_key=f'{c}-{pair}',scored=False)
    for block in range(1,6):
        add('drift_reference','single-L1024-D16',block=block,position='start',observer='light',scored=False)
        deck=ids.copy();rng.shuffle(deck)
        for c in deck:
            split=next(x['split'] for x in cells if x['id']==c)
            role='train' if split=='calibration' and block<=3 else 'repeat_check' if split=='calibration' else 'held_case'
            add('main',c,block=block,observer='light',scored=True,analysis_role=role)
        add('drift_reference','single-L1024-D16',block=block,position='end',observer='light',scored=False)
    deck=[c['id'] for c in cells if c['family']=='sentinel'];rng.shuffle(deck)
    for c in deck:add('http_bridge',c,observer='light',scored=False,role='development_transport_bridge')
    for c in ('single-L7168-D16','composition-mixed-D16'):
        add('deep_diagnostic',c,replicate=1,observer='deep',profile_steps=5,scored=False)
    return out


def frozen_prediction(case,budget,coef,overhead):
    """Unchanged gb.simulate with a declared barrier-origin adapter; old overhead separate."""
    req=requests_for(case);d=case['background_decoders']
    at_zero=[r for r in req if r['at']==0];later=[r for r in req if r['at']>0]
    class Barrier:
        def cap(self,t):return 1e9
        def origin(self,t):
            return [(255.,0.)]*d,deque([float(r['tokens']),1.,0.,d,r['id']] for r in at_zero),float(len(at_zero))
    old=gb.B
    try:
        gb.B=budget
        result=gb.simulate(Barrier(),0.,[gb.Arrival(r['at'],r['at'],1.,r['tokens'],r['id']) for r in later],coef,horizon=10.,extra=5.)
    finally:gb.B=old
    return {r.request_id:dict(raw_engine_first_s=r.first,raw_ttft_s=r.first-r.sched,
        old_empirical_corrected_ttft_s=r.first-r.sched+float(overhead.mean(r.running))) for r in result}


def offset_contrasts(predictions):
    rows=[];lookup={p['case']:p['scenarios'] for p in predictions}
    for d in (0,16):
        # Only compare offsets with the SAME foreground composition.
        for a,b in ((50,400),):
            x=lookup[f'sentinel-four-at{a}-D{d}'];y=lookup[f'sentinel-four-at{b}-D{d}']
            for key in x:
                rows.append(dict(backgrounds=d,offsets_ms=[a,b],scenario=key,
                    target='sentinel_ttft_offset_difference',delta_s=x[key]['foreground']['sentinel']['ttft_s']-y[key]['foreground']['sentinel']['ttft_s'],
                    exceeds_planning_noise_bands={str(ms):abs(x[key]['foreground']['sentinel']['ttft_s']-y[key]['foreground']['sentinel']['ttft_s'])>3*ms/1000 for ms in (10,25,50)}))
    return rows


def noise_references(baseline):
    refs={f'{case}/{rid}':r['ttft_s'] for case in ('single-L1024-D0','composition-uniform-D16')
          for rid,r in baseline[case]['foreground'].items()}
    import math
    rows=[]
    for case,result in baseline.items():
        for rid,r in result['foreground'].items():
            ref=min(refs,key=lambda k:(abs(math.log(r['ttft_s']/refs[k])),k))
            rows.append(dict(case=case,request_id=rid,reference=ref,predicted_ttft_s=r['ttft_s'],
                reference_predicted_ttft_s=refs[ref],outside_reference_range=not min(refs.values())<=r['ttft_s']<=max(refs.values())))
    return dict(rule='Nearest absolute log-latency distance using raw-B2048 toy only; freeze before collection. Compute MAD across off trials per position, never pool request positions.',
        references=refs,targets=rows,
        sentinel_mapping={r['case']:r['reference'] for r in rows if r['request_id']=='sentinel'},
        caveat='Noise-range proxies, not cell-specific measured variance or power. Out-of-range targets and mode differences must be reported; do not tune mapping from held data.')


def build(output):
    output=Path(output)
    if output.exists():raise ValueError('Choose fresh output directory')
    output.mkdir(parents=True);cells=cases();trials=order(cells);coef,overhead,reference,_=lf.frozen_parameters()
    assert tuple(COEF)==tuple(coef)
    predictions=[];baseline={};coverage=[];gaps=[];calibration_rows={}
    for c in cells:
        scenarios={}
        for budget,policy,q in [(8192,'fifo',0.),(2048,'fifo',0.),(2048,'fifo',1.5e-9),(2048,'fifo',3.15e-9),(2048,'round_robin_512',0.)]:
            steps=[];v=simulate(requests_for(c),budget,backgrounds=c['background_decoders'],policy=policy,quadratic=q,steps_out=steps)
            key=f'B{budget}-{policy}-q{q}';scenarios[key]=v
            if budget==2048 and policy=='fifo' and q==0.:
                baseline[c['id']]=v
                if c['split']=='calibration':
                    # Only chunks after the foreground release enter calibration-coverage summaries.
                    window=scored_steps(steps,v)
                    coverage.extend(dict(case=c['id'],q=chunk[1],l=chunk[2]) for s in window for chunk in s['chunks'])
                    calibration_rows[c['id']]=[features(s) for s in window]
        exact={str(b):frozen_prediction(c,b,coef,overhead) for b in (8192,2048)}
        # Explicit estimands; never average long requests into the sentinel target.
        base=scenarios['B2048-fifo-q0.0']['foreground']
        contrasts=[]
        for key,v in scenarios.items():
            if key=='B2048-fifo-q0.0':continue
            ids=['sentinel'] if c['sentinel'] else list(base)
            for rid in ids:
                delta=v['foreground'][rid]['ttft_s']-base[rid]['ttft_s']
                contrasts.append(dict(scenario=key,request_id=rid,baseline='B2048-fifo-q0.0',
                    target='sentinel_ttft' if c['sentinel'] else 'request_ttft',delta_s=delta,
                    identification_hypothesis=('H2' if key in ('B2048-fifo-q1.5e-09','B2048-fifo-q3.15e-09') and
                        (c['id'].startswith('single-L7168-') or c['id'].startswith(('composition-mixed-','composition-reverse-')))
                        else 'H3' if key=='B2048-round_robin_512-q0.0' and c['sentinel'] and c['sentinel']['arrival_ms']!=800 else None),
                    exceeds_planning_noise_bands={str(ms):abs(delta)>3*ms/1000 for ms in (10,25,50)}))
        gaps.append(dict(case=c['id'],split=c['split'],contrasts=contrasts,
            foreground_tokens=c['foreground_tokens'],policy_role='negative_control_expected_fifo_rr_tie' if c['sentinel'] and c['sentinel']['arrival_ms']==800 else 'identification',
            nonlinear_role='low_signal_validation_only' if c['family']=='single' and c['foreground_tokens']==[2048] else 'sensitivity'))
        predictions.append(dict(case=c['id'],scenarios=scenarios,frozen_barrier_predictions=exact))
    counts=Counter(t['stage'] for t in trials);roles=Counter(t['analysis_role'] for t in trials if t['stage']=='main')
    linear_time=sum(baseline[t['case']]['wall_s'] for t in trials)
    # Fixed conservative allowance; no runtime observations tune this budget.
    transfer_attempts=[dict(attempt_id=f'transfer-{window}-r{rep}',window=window,replicate=rep,
        replay_start=start,score_start=start+20,score_end=start+80,lead_in_seconds=20,cap=32,
        attempts_allowed=1,drain_before_next=True)
        for window,rep,start in [('a',1,40),('b',1,160),('b',2,160),('a',2,40)]]
    transfer_seconds=sum(t['score_end']-t['replay_start'] for t in transfer_attempts)
    terminal_reserve=10*len(transfer_attempts)
    estimated=linear_time*1.35+len(trials)*.25+transfer_seconds+terminal_reserve+120
    fit_rows=np.array([[r['q'],r['q']*r['l']+r['q']*(r['q']+1)/2] for r in coverage])
    scaled=fit_rows/np.maximum(np.linalg.norm(fit_rows,axis=0),1e-12)
    write(output/'cases.json',dict(schema='prefill-service-design-cases-v8',status='draft_not_executable',cases=cells))
    write(output/'order.json',dict(schema='prefill-service-design-order-v8',seed=20261022,order=trials,
        transfer_slices=transfer_attempts))
    source={f:digest(ROOT/f) for f in ['prefill_design.py','prefill_identifiability.py','prefill_validation.py','scripts/build_prefill_design.py','greybox_simulator.py','evaluate_low_cap_frozen.py']}
    write(output/'predictions.json',dict(role='preregistered hypothetical CPU predictions; not measured GPU results',source_sha256=source,
        frozen_coefficients=coef,legacy_adapter='All D16 backgrounds already decoding with255 remaining outputs at barrier; context ignored by frozen code. Raw engine and old empirical TTFT correction are separate. Generic scenario includes background preparation for budgeting.',
        quadratic_coefficient_role='1.5e-9 and3.15e-9 are unvalidated sensitivities per full L^2, not fitted bounds; applied incrementally as2*k*attention_pairs',predictions=predictions))
    write(output/'discrimination.json',dict(role='raw-coefficient sensitivity only; revision9 noise-aware discrimination is in results/prefill-design-round7/noise-screen.json; not statistical power',cases=gaps,offset_contrasts=offset_contrasts(predictions),
        eligibility_rule='Freeze per-cell abs prediction gap >max(3*jitter[cell],25ms). Require BOTH mixed_burst and single_long groups with all constituent cells eligible. Average paired MAE gains within each block/group; both groups require mean gain >max(t(.95,4)*SD/sqrt(5),5ms) across five blocks. All8 cell regression guards retained. Unresolved or violated cross-block dependence prevents passing. Calibrated M0 and retained M1 required; M0 structural N/A.',
        primary_contrasts=[dict(id=f"{c['case']}|{x['scenario']}|{x['request_id']}",hypothesis=x['identification_hypothesis'],delta_s=x['delta_s'])
            for c in gaps for x in c['contrasts'] if x['identification_hypothesis']],
        resolution_rule='Before main, abs(expected delta)>3*calibration-only jitter; if an entire H2/H3 family is unresolved its identification claim is inconclusive. Stop main only when both families are unresolved. No cell replacement.',
        observed_microbenchmark_noise_available=False,jitter_references=noise_references(baseline),full_model_calibration_support=calibration_support(calibration_rows),calibration_chunk_coverage=dict(q_min=float(fit_rows[:,0].min()),q_max=float(fit_rows[:,0].max()),
            l_min=min(r['l'] for r in coverage),l_max=max(r['l'] for r in coverage),q_Z_rank=int(np.linalg.matrix_rank(scaled)),q_Z_scaled_condition=float(np.linalg.cond(scaled))),
        gate='Before main trials, compare declared contrasts with calibration-only observer noise; no held-case swaps. Stop unsupported precision claims rather than retune the matrix.'))
    write(output/'budget.json',dict(trial_counts=dict(counts),total_trials=len(trials),main_roles=dict(roles),
        frozen_linear_serial_trial_seconds=linear_time,planning_service_multiplier=1.35,idle_gap_seconds=.25,transfer_seconds=transfer_seconds,
        transfer_attempts=len(transfer_attempts),transfer_terminal_reserve_seconds=terminal_reserve,
        transfer_terminal_reserve_per_attempt_seconds=10,flush_and_frontend_reserve_seconds=120,
        estimated_active_seconds=estimated,hard_collection_limit_seconds=1800,fits_planning_budget=estimated<=1800,
        caveat='Planning estimate only. Stage statuses are independent; suite complete requires every stage. Preserve all failures/timeouts and never turn a selected subset into a complete suite.'))
    assert len(cells)==22 and len(trials)==212 and roles=={'train':18,'repeat_check':12,'held_case':80}
    assert estimated<=1800,'Proposed collection does not fit declared budget'
    return dict(cells=22,trials=len(trials),main_roles=dict(roles),estimated_active_seconds=estimated)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    print(json.dumps(build(p.parse_args().output_dir),indent=2))
