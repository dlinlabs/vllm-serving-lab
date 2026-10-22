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
from prefill_design import simulate,requests_for,COEF
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
            result.append(dict(id=f'sentinel-four-at{offset}-D{d}',family='sentinel',background_decoders=d,foreground_tokens=[4096]*4,foreground_offsets_ms=[0]*4,sentinel=dict(input_tokens=512,arrival_ms=offset),split='held_case'))
        result.append(dict(id=f'sentinel-one-at200-D{d}',family='sentinel',background_decoders=d,foreground_tokens=[4096],foreground_offsets_ms=[0],sentinel=dict(input_tokens=512,arrival_ms=200),split='held_case'))
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
            return [(511.,0.)]*d,deque([float(r['tokens']),1.,0.,d,r['id']] for r in at_zero),float(len(at_zero))
    old=gb.B
    try:
        gb.B=budget
        result=gb.simulate(Barrier(),0.,[gb.Arrival(r['at'],r['at'],1.,r['tokens'],r['id']) for r in later],coef,horizon=10.,extra=5.)
    finally:gb.B=old
    return {r.request_id:dict(raw_engine_first_s=r.first,raw_ttft_s=r.first-r.sched,
        old_empirical_corrected_ttft_s=r.first-r.sched+float(overhead.mean(r.running))) for r in result}


def build(output):
    output=Path(output)
    if output.exists():raise ValueError('Choose fresh output directory')
    output.mkdir(parents=True);cells=cases();trials=order(cells);coef,overhead,reference,_=lf.frozen_parameters()
    assert tuple(COEF)==tuple(coef)
    predictions=[];baseline={};coverage=[];gaps=[]
    for c in cells:
        scenarios={}
        for budget,policy,q in [(8192,'fifo',0.),(2048,'fifo',0.),(2048,'fifo',3.15e-9),(2048,'round_robin_512',0.)]:
            steps=[];v=simulate(requests_for(c),budget,backgrounds=c['background_decoders'],policy=policy,quadratic=q,steps_out=steps)
            key=f'B{budget}-{policy}-q{q}';scenarios[key]=v
            if budget==2048 and policy=='fifo' and q==0.:
                baseline[c['id']]=v
                if c['split']=='calibration':
                    # Only chunks after the foreground release enter calibration-coverage summaries.
                    coverage.extend(dict(case=c['id'],q=chunk[1],l=chunk[2]) for s in steps if s['start']>=v['release_s']-1e-12 for chunk in s['chunks'])
        exact={str(b):frozen_prediction(c,b,coef,overhead) for b in (8192,2048)}
        a=scenarios['B2048-fifo-q0.0']['foreground'];b=scenarios['B2048-fifo-q3.15e-09']['foreground']
        gap=float(np.median([b[k]['ttft_s']-a[k]['ttft_s'] for k in a]))
        gaps.append(dict(case=c['id'],split=c['split'],median_M2_minus_M0_ttft_s=gap,
            exceeds_planning_noise_bands={str(ms):abs(gap)>ms/1000 for ms in (10,25,50)}))
        predictions.append(dict(case=c['id'],scenarios=scenarios,frozen_barrier_predictions=exact))
    counts=Counter(t['stage'] for t in trials);roles=Counter(t['analysis_role'] for t in trials if t['stage']=='main')
    linear_time=sum(baseline[t['case']]['wall_s'] for t in trials)
    # Fixed conservative allowance; no runtime observations tune this budget.
    estimated=linear_time*1.35+len(trials)*.25+120+120
    fit_rows=np.array([[r['q'],r['q']*r['l']+r['q']*(r['q']+1)/2] for r in coverage])
    scaled=fit_rows/np.maximum(np.linalg.norm(fit_rows,axis=0),1e-12)
    write(output/'cases.json',dict(schema='prefill-service-design-cases-v2',status='draft_not_executable',cases=cells))
    write(output/'order.json',dict(schema='prefill-service-design-order-v2',seed=20261022,order=trials,
        transfer_slices=[dict(start=60,duration=60,cap=32),dict(start=180,duration=60,cap=32)]))
    source={f:digest(ROOT/f) for f in ['prefill_design.py','scripts/build_prefill_design.py','greybox_simulator.py','evaluate_low_cap_frozen.py']}
    write(output/'predictions.json',dict(role='preregistered hypothetical CPU predictions; not measured GPU results',source_sha256=source,
        frozen_coefficients=coef,legacy_adapter='All D16 backgrounds already decoding with511 remaining outputs at barrier; context ignored by frozen code. Raw engine and old empirical TTFT correction are separate. Generic scenario includes background preparation for budgeting.',
        quadratic_coefficient_role='3.15e-9 is an unvalidated sensitivity per full L^2, applied incrementally as2*k*attention_pairs',predictions=predictions))
    write(output/'discrimination.json',dict(role='planning sensitivity, not estimated statistical power',cases=gaps,
        observed_microbenchmark_noise_available=False,calibration_chunk_coverage=dict(q_min=float(fit_rows[:,0].min()),q_max=float(fit_rows[:,0].max()),
            l_min=min(r['l'] for r in coverage),l_max=max(r['l'] for r in coverage),q_Z_rank=int(np.linalg.matrix_rank(scaled)),q_Z_scaled_condition=float(np.linalg.cond(scaled))),
        gate='Before main trials, compare declared contrasts with calibration-only observer noise; no held-case swaps. Stop unsupported precision claims rather than retune the matrix.'))
    write(output/'budget.json',dict(trial_counts=dict(counts),total_trials=len(trials),main_roles=dict(roles),
        frozen_linear_serial_trial_seconds=linear_time,planning_service_multiplier=1.35,idle_gap_seconds=.25,transfer_seconds=120,flush_and_frontend_reserve_seconds=120,
        estimated_active_seconds=estimated,hard_collection_limit_seconds=1800,fits_planning_budget=estimated<=1800,
        caveat='A planning estimate, not runtime evidence. Timeout/failure preserves an incomplete suite; optional work is not enabled adaptively.'))
    assert len(cells)==22 and len(trials)==212 and roles=={'train':18,'repeat_check':12,'held_case':80}
    assert estimated<=1800,'Proposed collection does not fit declared budget'
    return dict(cells=22,trials=len(trials),main_roles=dict(roles),estimated_active_seconds=estimated)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    print(json.dumps(build(p.parse_args().output_dir),indent=2))
