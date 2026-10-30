"""TRAIN-first-entry return clock. A development family, not physical costs.

The two-constant predecessor remains immutable. This family changes the P
estimator, not the causal simulator, GPU coefficients, queue budget or domain.
"""
import hashlib
from pathlib import Path
import statistics
from . import submit_clock as submit,event_clock_calibration as clock
from .contracts import digest,finite


def training_labels(captures,plan):
    captures=list(captures);labels=submit.training_labels(captures,plan)
    wanted={t['trial_id']:t for t in plan['order'] if t['stage']=='train' and t['observer']=='light'}
    entries=[]
    for cap in captures:
        if cap.get('trial_id') not in wanted:continue
        release=cap.get('release')
        if not finite(release):raise ValueError('Invalid TRAIN entry release')
        if any(cap['steps'][i]['start']>cap['steps'][i+1]['start'] for i in range(len(cap['steps'])-1)):
            raise ValueError('Unordered TRAIN entry steps')
        # Membership comes from planned request roles and scheduled prompt work,
        # not a future probe's timing or observed receipt state.
        prefix=f'plab/{cap["trial_id"]}/foreground-'
        steps=[s for s in cap['steps'] if s['end']<=cap['foreground_done'] and any(
            w['request_id'].startswith(prefix) and w['position']<cap['requests'][w['request_id']]['input_tokens']
            for w in s['work'])]
        if not steps:raise ValueError('Missing TRAIN first foreground prefill')
        step=steps[0]
        if not step['prefill'] or step['start']<release:raise ValueError('Invalid TRAIN entry chronology')
        workers=[w for worker in cap['workers'] for w in worker['rows'] if w['step_id']==step['step_id']]
        if len(workers)!=1:raise ValueError('Invalid TRAIN entry worker identity')
        w=workers[0]
        entries.append(dict(trial_id=cap['trial_id'],step_id=step['step_id'],host_span_s=w['host_end']-w['host_start']))
    if len(entries)!=len(wanted):raise ValueError('Incomplete TRAIN entries')
    nominal=statistics.median(r['host_span_s'] for r in entries)
    if not labels['prefill_lower_s']<=nominal<=labels['prefill_upper_s']:raise ValueError('Entry outside TRAIN domain')
    labels.update(schema='entry-submit-label-freeze-v1',predecessor_prefill_s=labels['constants_s']['prefill'],
        entry_samples=sorted(entries,key=lambda r:r['trial_id']),
        rule='P=median first foreground prefill host span per TRAIN LIGHT trial; D and full-prefix P domain unchanged')
    labels['constants_s']=dict(labels['constants_s'],prefill=nominal)
    return labels


def first_commit_scores(captures,plan,runtime,model,labels):
    cases={t['case'] for t in plan['order'] if t['stage']=='train'}
    predicted=submit.own_rollouts(model['parameters_s'],dict(plan,cases=[c for c in plan['cases'] if c['id'] in cases]),
        runtime,labels['constants_s']['prefill'],labels['constants_s']['decode'])
    wanted={t['trial_id']:t for t in plan['order'] if t['stage']=='train' and t['observer']=='light'}
    by_case={c['id']:c for c in plan['cases']};checks=[]
    for cap in captures:
        if cap.get('trial_id') not in wanted:continue
        trial=wanted[cap['trial_id']];row=clock.event_labels(cap,trial,by_case[trial['case']])
        own=clock.event_predictions(predicted[trial['case']],row)
        i=next(i for i,v in enumerate(row['times_s']) if v[1]>0)
        observed=row['times_s'][i][1];estimate=float(own[i,1]);limit=max(plan['queue_absolute_s'],plan['queue_relative']*observed)
        checks.append(dict(trial_id=cap['trial_id'],case=trial['case'],observed_s=observed,predicted_s=estimate,
            error_s=estimate-observed,limit_s=limit,passed=abs(estimate-observed)<=limit))
    if len(checks)!=len(wanted):raise ValueError('Incomplete first-commit checks')
    return dict(passed=bool(checks) and all(c['passed'] for c in checks),checks=sorted(checks,key=lambda r:r['trial_id']),
        scope='TRAIN model readiness; not independent accuracy or physical CPU-stage identification')


def freeze(captures,plan,runtime,model):
    captures=list(captures)
    # Reuse the full original membership/event identity checks before changing
    # only the P estimator. The predecessor is not silently reinterpreted.
    predecessor=submit.freeze(captures,plan,runtime,model)
    labels=training_labels(captures,plan);g=labels['constants_s']
    predictions=clock.forecasts(submit.own_rollouts(model['parameters_s'],plan,runtime,g['prefill'],g['decode']))
    cells={};errors=[]
    for case in plan['cases']:
        if not case['sentinel']:continue
        try:cells[case['id']]=submit.phase_cell(case,model['parameters_s'],plan,runtime,labels,predictions[case['id']]['sentinel_queue_residence_s'])
        except (ValueError,StopIteration) as exc:errors.append(case['id']+': '+str(exc))
    first=first_commit_scores(captures,plan,runtime,model,labels)
    result=dict(schema='entry-submit-clock-freeze-v1',implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        predecessor_sha256=predecessor['sha256'],base_model_sha256=model['sha256'],plan_sha256=digest(plan),runtime_sha256=digest(runtime),
        labels=labels,predictions=predictions,f0_predictions=clock.predict_frozen(model,plan,runtime),cells=cells,errors=errors,first_commit_readiness=first,
        readiness_passed=not errors and first['passed'],probe_labels_used_for_fit=False,
        independent_confirmation_performed=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    result['sha256']=digest(result);return result


def verify(frozen,captures,plan,runtime,model):
    rebuilt=freeze(captures,plan,runtime,model)
    if digest(frozen)!=digest(rebuilt):raise ValueError('Entry clock freeze/TRAIN/runtime/plan changed')
    return rebuilt
