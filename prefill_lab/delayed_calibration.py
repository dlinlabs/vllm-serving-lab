"""Frozen four-parameter delayed-arrival pilot; no HTTP/admission/MPC acceptance."""
import json
from pathlib import Path
import time
import numpy as np
from . import effective_calibration as base
from .async_predictor import rollout
from .contracts import audit_capture,digest,finite,save

PLAN=Path(__file__).resolve().parents[1]/'experiments/prefill-effective-arrivals-v2.json'
HELD={'held_delayed','held_anchor_before','held_anchor_after'}


def protocol():return json.loads(PLAN.read_text())


def validate_manifest(manifest):
    from .tokens import validate
    p=protocol()
    if (manifest['model'],manifest['revision'])!=(p['model']['name'],p['model']['revision']):
        raise ValueError('Delayed model/revision mismatch')
    return validate(manifest,p['cases'],p['order'])


def identities(case):
    result={f'foreground-{i}':(n,128,'foreground',at/1000.) for i,(n,at) in enumerate(zip(case['foreground_tokens'],case['foreground_offsets_ms']))}
    result.update({f'background-{i}':(1024,256,'background',0.) for i in range(case['background_decoders'])})
    if case['sentinel']:result['sentinel']=(512,128,'sentinel',case['sentinel']['arrival_ms']/1000.)
    return result


def extract(capture,trial,case,plan):
    if not case['sentinel']:
        return base.extract(capture,trial,case)
    if capture['trial_id']!=trial['trial_id'] or capture['observer']!=trial['observer']:
        raise ValueError('Delayed trial/observer mismatch')
    workers=[r for w in capture.get('workers',[]) for r in w['rows']]
    audit=audit_capture(capture,workers,require_steps=trial['observer']=='light')
    if not audit['valid'] or capture.get('pending_step_ids'):raise ValueError('Invalid delayed work capture')
    expected=identities(case);endpoints={};clocks={};seen=set();release=capture['release']
    if not finite(release):raise ValueError('Invalid foreground barrier')
    for rid,r in capture['requests'].items():
        prefix=f'plab/{trial["trial_id"]}/'
        if not rid.startswith(prefix):raise ValueError('Delayed request namespace mismatch')
        short=rid[len(prefix):]
        if short not in expected or short in seen:raise ValueError('Unexpected delayed request')
        seen.add(short);n,out,role,offset=expected[short]
        if (r['input_tokens'],r['max_tokens'],r['role'])!=(n,out,role):raise ValueError('Wrong delayed request specification')
        values=[r[k] for k in ('queue_posted','enqueue','first','finished')]
        if not all(finite(v) for v in values) or values!=sorted(values):raise ValueError('Invalid delayed request chronology')
        if role=='background':continue
        target=release+offset
        if not finite(r.get('target')) or abs(r['target']-target)>1e-9:raise ValueError('Declared target mismatch')
        lag=r['queue_posted']-target
        if (offset==0 and abs(lag)>1e-9) or lag < -1e-9 or abs(lag)>plan['posting_lag_limit_s']:
            raise ValueError('Posting lag exceeds frozen limit')
        if r['first']<=target:raise ValueError('Nonpositive target latency')
        endpoints[short]=dict(ttft_s=r['first']-target,completion_s=r['finished']-target)
        clocks[short]=dict(target=target,queue_posted=r['queue_posted'],enqueue=r['enqueue'],
            posting_lag_s=lag,queue_residence_s=r['enqueue']-r['queue_posted'],
            posted_ttft_s=r['first']-r['queue_posted'],target_ttft_s=r['first']-target)
    if seen!=set(expected):raise ValueError('Missing delayed request')
    done=max(r['finished'] for r in capture['requests'].values() if r['role']!='background')
    if not finite(capture.get('foreground_done')) or abs(capture['foreground_done']-done)>1e-9:
        raise ValueError('Delayed foreground completion mismatch')
    for r in capture['requests'].values():
        if r['role']=='background' and (r['first']>release or r['finished']<done):raise ValueError('Delayed background lifetime mismatch')
    return dict(trial_id=trial['trial_id'],case=trial['case'],stage=trial['stage'],block=trial['block'],observer=trial['observer'],
        endpoints=endpoints,clocks=clocks,capture_sha256=digest(capture),
        work_signature=base.work_signature(capture['steps']) if trial['observer']=='light' else None)


def forecasts(theta,plan,runtime):
    from scripts.predict_prefill_async import runtime_inputs
    result=base.predictions(theta,plan,runtime);scope,dispatch=runtime_inputs(runtime)
    for case in plan['cases']:
        if case['sentinel']:
            pred=rollout(case,base.costs(theta,plan),dispatch,scope=scope)
            result[case['id']]['sentinel_queue_residence_s']=pred['requests']['sentinel']['input_queue_wait_s']
    return result


def select(rows,plan,stages,observer):
    wanted={t['trial_id']:t for t in plan['order'] if t['stage'] in stages and t['observer']==observer}
    found={};cases={c['id']:c for c in plan['cases']}
    for r in rows:
        tid=r['trial_id']
        if tid not in wanted:continue
        if tid in found:raise ValueError('Duplicate delayed trial')
        t=wanted[tid]
        if any(r[k]!=t[k] for k in ('case','stage','observer','block')):raise ValueError('Delayed split membership mismatch')
        ids={rid for rid,v in identities(cases[t['case']]).items() if v[2]!='background'}
        if set(r['endpoints'])!=ids:raise ValueError('Missing delayed endpoint identities')
        for endpoint in r['endpoints'].values():
            if set(endpoint)!={'ttft_s','completion_s'} or not all(finite(v) and v>0 for v in endpoint.values()):
                raise ValueError('Invalid delayed endpoint values')
        if cases[t['case']]['sentinel']:
            clock=r['clocks']['sentinel']
            if not all(finite(clock[k]) for k in ('queue_residence_s','posting_lag_s','posted_ttft_s','target_ttft_s')):
                raise ValueError('Invalid sentinel clock diagnostics')
            if clock['queue_residence_s']<0 or abs(clock['posting_lag_s'])>plan['posting_lag_limit_s']:
                raise ValueError('Invalid sentinel queue/posting values')
        found[tid]=r
    if set(found)!=set(wanted):raise ValueError('Incomplete delayed trial split')
    return [found[k] for k in sorted(wanted)]


def quality(rows,plan):
    off=select(rows,plan,HELD,'off');light=select(rows,plan,HELD,'light');g=plan['gates']
    pairs={(r['stage'],r['case'],r['block']):r for r in light};checks=[];groups={}
    for r in off:
        pair=pairs[(r['stage'],r['case'],r['block'])]
        for rid,endpoint in r['endpoints'].items():
            for metric,value in endpoint.items():
                delta=abs(pair['endpoints'][rid][metric]-value)
                checks.append(dict(trial=r['trial_id'],request=rid,metric=metric,delta_s=delta,
                    passed=delta<=max(g['observer_absolute_s'],g['observer_relative']*value)))
                groups.setdefault((r['case'],rid,metric),[]).append(value)
    spreads=[]
    for (case,rid,metric),values in groups.items():
        floor=g['ttft_absolute_s'] if metric=='ttft_s' else g['completion_absolute_s']
        spread=max(values)-min(values)
        spreads.append(dict(case=case,request=rid,metric=metric,repeats=len(values),spread_s=spread,
            passed=spread<=max(floor,g['repeat_relative_spread']*float(np.median(values)))))
    return dict(passed=all(r['passed'] for r in checks+spreads),paired_observer=checks,repeat_spread=spreads)


def evaluate(rows,plan,runtime,frozen,predicted):
    if (frozen['sha256']!=digest({k:v for k,v in frozen.items() if k!='sha256'}) or
        frozen['plan_sha256']!=digest(plan) or frozen['runtime_sha256']!=digest(runtime)):
        raise ValueError('Delayed frozen identity changed')
    if digest(predicted)!=digest(forecasts(frozen['parameters_s'],plan,runtime)):raise ValueError('Delayed forecasts changed')
    if frozen['training_sha256']!=digest(base.select(rows,plan,{'train'},'off')):raise ValueError('Delayed training identity changed')
    return score(rows,plan,predicted,frozen['sha256'])


def score(rows,plan,predicted,frozen_sha):
    """Unchanged v2 gates shared with the separately identified clock model."""
    held=select(rows,plan,HELD,'off');light=select(rows,plan,HELD,'light')
    q=quality(rows,plan);delayed=[r for r in held if r['stage']=='held_delayed'];anchors=[r for r in held if r['stage']!='held_delayed']
    scores=base.endpoint_scores(delayed,predicted,plan);anchor_scores=base.endpoint_scores(anchors,predicted,plan)
    # Each before/after anchor must pass separately; averaging cannot hide drift.
    anchor_phases={s:base.endpoint_scores([r for r in anchors if r['stage']==s],predicted,plan) for s in ('held_anchor_before','held_anchor_after')}
    queue=[]
    for case in sorted({r['case'] for r in delayed}):
        values=[r['clocks']['sentinel']['queue_residence_s'] for r in delayed if r['case']==case]
        measured=float(np.median(values));estimate=predicted[case]['sentinel_queue_residence_s']
        limit=max(plan['queue_absolute_s'],plan['queue_relative']*measured)
        queue.append(dict(case=case,measured_median_s=measured,predicted_s=estimate,error_s=estimate-measured,
                          limit_s=limit,passed=abs(estimate-measured)<=limit))
    structure=base.structure_check(light,predicted) and all(p['background_outlives_foreground'] for p in predicted.values())
    passed=q['passed'] and scores['passed'] and all(p['passed'] for p in anchor_phases.values()) and structure and all(r['passed'] for r in queue)
    return dict(schema='effective-delayed-report-v2',candidate_passed=passed,quality=q,endpoint_scores=scores,
        anchor_scores=anchor_scores,anchor_phase_scores=anchor_phases,sentinel_queue_scores=queue,
        structure_and_background_lifetime_valid=structure,frozen_sha256=frozen_sha,
        primary_clock='declared target-to-first/final output',secondary_clock='actual queue-post-to-first and enqueue residence',
        clock_diagnostics=[dict(trial_id=r['trial_id'],case=r['case'],sentinel=r['clocks']['sentinel']) for r in delayed],
        signed_bias_requires_cap_uncertainty_qualification=True,model_accepted=False,physical_costs_identified=False,
        complete_experiment_passed=False,adaptive_cap_accepted=False,mpc_accepted=False)


def rejection_reasons(report):
    """Expose existing failures without changing any score or threshold."""
    reasons=[]
    if not report['quality']['passed']:reasons.append(dict(gate='quality'))
    if not report['structure_and_background_lifetime_valid']:reasons.append(dict(gate='structure_and_background_lifetime'))
    for key in ('endpoint_scores',):
        reasons.extend(dict(gate=key,**r) for r in report[key]['cells'] if not r['passed'])
    for phase,score in report['anchor_phase_scores'].items():
        reasons.extend(dict(gate=phase,**r) for r in score['cells'] if not r['passed'])
    reasons.extend(dict(gate='sentinel_queue',**r) for r in report['sentinel_queue_scores'] if not r['passed'])
    return reasons


async def run(client,http,url,manifest,output,*,event_clock=False):
    from .suite import controlled_trial,runtime_preflight,host_identity
    from .causal import dispatcher_parity
    from scripts.predict_prefill_async import runtime_inputs
    if event_clock:
        from . import event_clock_calibration as clock
    p=clock.protocol() if event_clock else protocol();output=Path(output);start=time.monotonic();deadline=start+p['hard_seconds'];rows=[];event_rows=[];frozen=predicted=None
    summary=dict(collection_complete=False,candidate_passed=False,model_accepted=False,scope=p['scope'],completed=[],failures=[])
    save(output/'calibration-plan.json',p)
    try:
        (clock.validate_manifest if event_clock else validate_manifest)(manifest);meta=await client.command('meta');runtime_preflight(meta,manifest);runtime_inputs(meta)
        if any(w.get('pipeline_stages') for w in meta['workers']):raise ValueError('Delayed pilot prohibits deep stage timers')
        save(output/'runtime.json',meta);save(output/'host-before.json',host_identity());cases={c['id']:c for c in p['cases']}
        for t in p['order']:
            if time.monotonic()>=deadline:raise TimeoutError('Delayed pilot1800s deadline')
            if t['stage'] in HELD and frozen is None:
                print('Delayed pilot: training-only fit and forecast freeze before held trials',flush=True)
                frozen=clock.fit(rows,event_rows,p,meta,deadline=deadline,origin='fresh_native_training_events') if event_clock else base.fit(rows,p,meta,deadline=deadline,origin='native_controlled_endpoints')
                save(output/'frozen.json',frozen);save(output/'costs.json',frozen['costs'])
                predicted=clock.predict_frozen(frozen,p,meta) if event_clock else forecasts(frozen['parameters_s'],p,meta);save(output/'pre-held-predictions.json',predicted)
                if not all(r['background_outlives_foreground'] for r in predicted.values()):raise ValueError('Predicted background lifetime outside delayed scope')
            tid=t['trial_id'];print(f'Delayed {len(summary["completed"])+1}/{len(p["order"])}: {tid} {t["stage"]} {t["case"]} {t["observer"]}',flush=True)
            capture=await controlled_trial(client,t,deadline,output/(tid+'.json'));save(output/(tid+'.json'),capture)
            row=extract(capture,t,cases[t['case']],p)
            if t['observer']=='light' and not dispatcher_parity(capture,meta['workers'][0]['dispatch_table'])['matches']:
                raise ValueError('Delayed native dispatcher mismatch')
            if event_clock and t['stage']=='train' and t['observer']=='light':
                event_rows.append(clock.event_labels(capture,t,cases[t['case']]));save(output/'training-events.json',event_rows)
            rows.append(row);save(output/'endpoints.json',rows);summary['completed'].append(tid);save(output/'calibration.json',summary)
        final=await client.command('meta');save(output/'runtime-final.json',final)
        if digest(meta)!=digest(final):raise ValueError('Delayed runtime drift')
        save(output/'host-after.json',host_identity());report=clock.evaluate(rows,event_rows,p,meta,frozen,predicted) if event_clock else evaluate(rows,p,meta,frozen,predicted);save(output/'held-report.json',report)
        if time.monotonic()>=deadline:raise TimeoutError('Delayed pilot deadline during analysis')
        summary.update(collection_complete=True,candidate_passed=report['candidate_passed'],rejection_reasons=rejection_reasons(report))
        if summary['rejection_reasons']:
            print('Delayed validation rejected: '+json.dumps(summary['rejection_reasons']),flush=True)
    except BaseException as exc:
        summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        summary['elapsed_seconds']=time.monotonic()-start;save(output/'calibration.json',summary)
    return summary
