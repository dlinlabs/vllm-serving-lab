"""Independent 240-trial controlled queue confirmation; never cap/MPC approval."""
import asyncio
import copy
import json
from pathlib import Path
import statistics
import time
from . import queue_probe as probe,event_clock_calibration as clock,effective_calibration as base,entry_clock,queue_phase
from .contracts import canonical,digest,save,finite

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'experiments/prefill-queue-confirmation-v2.json'
HELD={'held_queue','held_anchor_before','held_anchor_after'}
ACCEPTANCE=('model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted')


def protocol():
    policy=json.loads(POLICY.read_text());source=probe.protocol()
    if digest(source)!=policy['base_probe_plan_sha256']:raise ValueError('Confirmation base plan changed')
    plan=copy.deepcopy(source);order=[]
    def add(t):
        t=copy.deepcopy(t);t['trial_id']=f'qc-{len(order)+1:03d}'
        t['paired_content_key']='queue-confirmation-v2-'+t['paired_content_key'];order.append(t)
    # All warmup, including delayed cells, precedes TRAIN and sealing.
    for stage in ('clock_warmup','warmup','train'):
        for t in source['order']:
            if t['stage']==stage:add(t)
    def anchors(stage,block):
        for d in (0,16):
            case=f'composition-uniform-D{d}'
            for arm in ('baseline','light'):
                add(dict(case=case,stage=stage,block=block,arm=arm,observer='off' if arm=='baseline' else 'light',
                    receipt_observer=False,paired_content_key=f'{stage}-{case}'))
    anchors('held_anchor_before',7)
    for t in source['order']:
        if t['stage']=='calibration_probe':add(dict(t,stage='held_queue'))
    anchors('held_anchor_after',8)
    for c in plan['cases']:
        if c['sentinel']:c['split']='independent_queue_confirmation'
    plan.update(schema='queue-confirmation-v2',scope=policy['scope'],order=order,
        hard_seconds=policy['hard_seconds'],policy_sha256=digest(policy))
    if len(order)!=240 or sum(t['stage']=='train' for t in order)!=48:raise ValueError('Confirmation denominator changed')
    if digest(plan)!=digest(json.loads((ROOT/'experiments/prefill-queue-confirmation-v2.plan.json').read_text())):
        raise ValueError('Numeric confirmation manifest differs')
    return plan


def validate_manifest(manifest):
    from .tokens import validate
    p=protocol()
    if (manifest['model'],manifest['revision'])!=(p['model']['name'],p['model']['revision']):raise ValueError('Confirmation model pin differs')
    validate(manifest,p['cases'],p['order']);paired={}
    for t in p['order']:
        rows=manifest['trials'][t['trial_id']];key=t['paired_content_key']
        if any(r['token_key']!=f'pair:{key}:{r["id"]}:{r["input_tokens"]}' for r in rows):raise ValueError('Confirmation token namespace differs')
        signature=digest([(r['id'],manifest['pools'][r['token_key']]['input_ids']) for r in rows])
        if key in paired and paired[key]!=signature:raise ValueError('Confirmation paired content differs')
        paired[key]=signature
    return digest(manifest)


def extract(cap,t,case,p):
    if case['sentinel']:return probe.extract(cap,t,case,p)
    row=probe.delayed.extract(cap,t,case,p);row.update(arm=t['arm'],receipt=None)
    if 'receipt' in cap:raise ValueError('Unexpected anchor/TRAIN receipt observer')
    return row


def membership(rows,p):
    wanted={t['trial_id']:t for t in p['order']}
    if len(rows)!=len(wanted) or {r['trial_id'] for r in rows}!=set(wanted):raise ValueError('Missing/duplicate/extra confirmation trials')
    for r in rows:
        t=wanted[r['trial_id']]
        if any(r[k]!=t[k] for k in ('case','stage','block','observer','arm')):raise ValueError('Confirmation membership differs')


def measurement(rows,p):
    # Reuse the original observer/control gates with an explicit analysis view;
    # raw held membership and the saved protocol never change.
    plan=copy.deepcopy(p);view=copy.deepcopy(rows)
    for t in plan['order']:
        if t['stage']=='held_queue':t['stage']='calibration_probe'
    for r in view:
        if r['stage']=='held_queue':r['stage']='calibration_probe'
    report=probe.analyze(view,plan)
    report.update(scope='original observer gates on an explicit held-analysis view; OFF phase unknown',held_plan_sha256=digest(p))
    return report


def evaluate(rows,p,frozen):
    # Nested metric insertion order must not change array ordering on disk replay.
    rows=json.loads(canonical(rows))
    membership(rows,p)
    if frozen['plan_sha256']!=digest(p) or not frozen['readiness_passed']:raise ValueError('Unready/different confirmation freeze')
    controls=measurement(rows,p);predicted=frozen['predictions'];checks=[];counts={}
    for row in rows:
        if row['stage']!='held_queue':continue
        cell=frozen['cells'].get(row['case'])
        if not cell:raise ValueError('Missing frozen confirmation cell')
        q=row['clocks']['sentinel']['queue_residence_s']
        if not finite(q) or q<0:raise ValueError('Invalid held queue residence')
        result=queue_phase.assign(q,cell,queue_phase.policy())
        point=queue_phase.assign(q,dict(branches=[dict(queue_s=predicted[row['case']]['sentinel_queue_residence_s'])]),queue_phase.policy())
        f0=queue_phase.assign(q,dict(branches=[dict(queue_s=frozen['f0_predictions'][row['case']]['sentinel_queue_residence_s'])]),queue_phase.policy())
        checks.append(dict(trial_id=row['trial_id'],case=row['case'],arm=row['arm'],block=row['block'],observed_s=q,phase=result,point=point,f0_point=f0))
        group=counts.setdefault(row['case'],{}).setdefault(row['arm'],{str(i):0 for i in range(len(cell['branches']))}|{'unresolved':0})
        group['unresolved' if result['branch'] is None else str(result['branch'])]+=1
    by_id={c['trial_id']:c for c in checks};coverage=[]
    for case,cell in sorted(frozen['cells'].items()):
        for branch in range(len(cell['branches'])):
            matching=[r for r in controls['receipt_pair_checks'] if r['case']==case and r['assessable'] and r['passed'] and all(
                by_id[r[k]]['phase']['passed'] and by_id[r[k]]['phase']['branch']==branch for k in ('receipt_trial','light_trial'))]
            coverage.append(dict(case=case,branch=branch,comparable_control_pairs=len(matching),
                passed=bool(matching),assignments_are_native_phase_labels=False))
    endpoints={};quality={}
    for stage in sorted(HELD):
        for arm in ('baseline','receipt','light'):
            selected=[r for r in rows if r['stage']==stage and r['arm']==arm]
            if selected:endpoints[stage+':'+arm]=base.endpoint_scores(selected,predicted,p)
        if stage!='held_queue':quality[stage]=base.quality(rows,p,{stage})
    # Anchors must remain near TRAIN on their own side, not merely agree with
    # each other after both have drifted.
    drift=[]
    for r in rows:
        if r['stage'] not in ('held_anchor_before','held_anchor_after') or r['arm']!='baseline':continue
        train=[a for a in rows if a['stage']=='train' and a['arm']=='baseline' and a['case']==r['case']]
        for rid,e in r['endpoints'].items():
            for metric,value in e.items():
                median=statistics.median(a['endpoints'][rid][metric] for a in train)
                floor=p['gates']['ttft_absolute_s' if metric=='ttft_s' else 'completion_absolute_s']
                limit=max(floor,p['gates']['repeat_relative_spread']*median)
                drift.append(dict(trial_id=r['trial_id'],request=rid,metric=metric,delta_s=value-median,limit_s=limit,passed=abs(value-median)<=limit))
    work=[dict(trial_id=r['trial_id'],matches=r['work_signature']==predicted[r['case']]['work_signature'],
        own_background_lifetime_passed=predicted[r['case']]['background_outlives_foreground'])
        for r in rows if r['stage'] in HELD and r['observer']=='light']
    queue=all(c['phase']['passed'] and c['phase']['branch'] is not None for c in checks)
    measure=controls['endpoint_observer_controls_passed'] and controls['same_state_queue_controls_passed']
    branch_coverage=bool(coverage) and all(r['passed'] for r in coverage)
    endpoint=bool(endpoints) and all(v['passed'] for v in endpoints.values())
    anchors=all(v['passed'] for v in quality.values()) and bool(drift) and all(r['passed'] for r in drift)
    structure=bool(work) and all(r['matches'] and r['own_background_lifetime_passed'] for r in work)
    passed=queue and measure and branch_coverage and endpoint and anchors and structure
    failed=not(queue and endpoint and anchors and structure and controls['endpoint_observer_controls_passed']) or any(
        r['passed'] is False for r in controls['receipt_pair_checks'])
    arm_scores={arm:dict(samples=sum(c['arm']==arm for c in checks),
        f0_point_failures=sum(c['arm']==arm and not c['f0_point']['passed'] for c in checks),
        entry_point_failures=sum(c['arm']==arm and not c['point']['passed'] for c in checks),
        phase_failures=sum(c['arm']==arm and (not c['phase']['passed'] or c['phase']['branch'] is None) for c in checks))
        for arm in ('baseline','receipt','light')}
    return dict(schema='queue-confirmation-report-v2',candidate_passed=passed,
        status='passed' if passed else 'failed' if failed else 'inconclusive',
        queue_budget_passed=queue,measurement_passed=measure,branch_controls_passed=branch_coverage,
        endpoints_passed=endpoint,anchors_passed=anchors,work_passed=structure,
        sample_checks=checks,arm_scores=arm_scores,branch_counts=counts,branch_coverage=coverage,measurement=controls,
        endpoint_scores=endpoints,anchor_quality=quality,anchor_drift=drift,work_checks=work,
        plan_sha256=digest(p),rows_sha256=digest(rows),frozen_sha256=frozen['sha256'],
        scope=p['scope'],baseline_native_phase_observed=False,**{k:False for k in ACCEPTANCE})


def seal_training(rows,captures,p,runtime,deadline=None):
    if any(r['stage'] in HELD for r in rows):raise ValueError('Held outcomes reached confirmation estimator')
    rows=json.loads(canonical(rows))
    events=[clock.event_labels(c,t,next(case for case in p['cases'] if case['id']==t['case']))
        for t in p['order'] if t['stage']=='train' and t['observer']=='light'
        for c in captures if c['trial_id']==t['trial_id']]
    model=clock.fit(rows,events,p,runtime,deadline=deadline,origin='fresh_native_training_events')
    frozen=entry_clock.freeze(captures,p,runtime,model)
    if not frozen['readiness_passed']:raise ValueError('TRAIN entry-clock/phase readiness failed; held phase not attempted')
    training=[r for r in rows if r['stage']=='train']
    if not base.endpoint_scores([r for r in training if r['observer']=='off'],frozen['predictions'],p)['passed'] or not base.structure_check(
        [r for r in training if r['observer']=='light'],frozen['predictions']):raise ValueError('Entry TRAIN endpoint/work gate failed')
    return model,frozen


async def run(client,http,url,manifest,output):
    from .suite import controlled_trial,runtime_preflight,host_identity
    from .causal import dispatcher_parity
    from scripts.predict_prefill_async import runtime_inputs
    from scripts.run_prefill_service import code_identity
    p=protocol();output=Path(output);start=time.monotonic();deadline=start+p['hard_seconds'];rows=[];captures=[];frozen=None;initial_sources=code_identity()
    summary=dict(collection_complete=False,candidate_passed=False,completed=[],failures=[],**{k:False for k in ACCEPTANCE})
    save(output/'queue-confirmation-plan.json',p)
    try:
        validate_manifest(manifest);runtime=await client.command('meta');runtime_preflight(runtime,manifest);runtime_inputs(runtime)
        if any(w.get('pipeline_stages') for w in runtime['workers']):raise ValueError('Unexpected deep stages')
        save(output/'runtime.json',runtime);save(output/'host-before.json',host_identity());cases={c['id']:c for c in p['cases']}
        for t in p['order']:
            if time.monotonic()>=deadline:raise TimeoutError('Confirmation deadline')
            if code_identity()!=initial_sources:raise ValueError('Confirmation source changed during collection')
            if t['stage'] in HELD and frozen is None:
                # Fit in a thread so the API connection can progress; no other
                # trial runs concurrently. Source and model are sealed once.
                model,frozen=await asyncio.wait_for(asyncio.to_thread(seal_training,rows,captures,p,runtime,deadline),max(.001,deadline-time.monotonic()))
                save(output/'base-model.json',model);save(output/'entry-freeze.json',frozen)
                seal=dict(schema='queue-confirmation-seal-v2',at_monotonic=time.monotonic(),next_trial=t['trial_id'],
                    completed=list(summary['completed']),source_sha256=code_identity(),manifest_sha256=digest(manifest),
                    model_sha256=model['sha256'],freeze_sha256=frozen['sha256'])
                save(output/'seal.json',seal)
            if frozen is not None and (digest(json.loads((output/'entry-freeze.json').read_text()))!=digest(frozen) or code_identity()!=seal['source_sha256']):
                raise ValueError('Confirmation freeze/source changed during held collection')
            print(f'Queue confirmation {len(rows)+1}/240: {t["trial_id"]} {t["stage"]} {t["arm"]}',flush=True)
            cap=await asyncio.wait_for(controlled_trial(client,t,deadline,output/(t['trial_id']+'.json')),max(.001,deadline-time.monotonic()))
            save(output/(t['trial_id']+'.json'),cap);row=extract(cap,t,cases[t['case']],p)
            if t['observer']=='light' and not dispatcher_parity(cap,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('Dispatcher mismatch')
            if t['stage']=='train' and t['observer']=='light':captures.append(cap)
            rows.append(row);summary['completed'].append(t['trial_id']);save(output/'confirmation-rows.json',rows);save(output/'confirmation.json',summary)
        final=await client.command('meta');save(output/'runtime-final.json',final)
        if digest(final)!=digest(runtime):raise ValueError('Runtime drift')
        save(output/'host-after.json',host_identity());report=evaluate(rows,p,frozen);save(output/'confirmation-report.json',report)
        summary.update(collection_complete=True,candidate_passed=report['candidate_passed'],status=report['status'])
    except BaseException as exc:summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:summary['elapsed_seconds']=time.monotonic()-start;save(output/'confirmation.json',summary)
    return summary
