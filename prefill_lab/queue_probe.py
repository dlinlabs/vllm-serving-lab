"""Frozen receipt-state calibration collection; no model/control acceptance."""
import asyncio
import copy
import itertools
import json
import statistics
from pathlib import Path
import time
from . import event_clock_calibration as clock,delayed_calibration as delayed
from .contracts import digest,save
from .receipt import validate as validate_receipt
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'experiments/prefill-queue-probe-v1.json'


def protocol():
    spec=json.loads(SPEC.read_text());base=clock.protocol()
    if digest(base)!=spec['base_clock_plan_sha256']:raise ValueError('Base plan changed')
    cases=copy.deepcopy(base['cases']);probes=[c for c in cases if c['sentinel']]
    for c in probes:c['split']='calibration_probe'
    order=[]
    for t in base['order']:
        if t['stage'] not in ('warmup','train'):continue
        trial=copy.deepcopy(t)
        trial.update(trial_id='qr-clock-'+t['trial_id'].split('-',1)[1],
            stage='clock_warmup' if t['stage']=='warmup' else 'train',
            arm='baseline' if t['observer']=='off' else 'light',receipt_observer=False,
            paired_content_key='queue-receipt-clock-v1-'+t['paired_content_key'])
        order.append(trial)
    permutations=list(itertools.permutations(('baseline','receipt','light')))
    for block in range(7):
        stage='warmup' if block==0 else 'calibration_probe'
        for case in probes[::(-1 if block%2 else 1)]:
            for arm in permutations[max(0,block-1)]:
                order.append(dict(trial_id=f'qr-{len(order)+1:03d}',case=case['id'],stage=stage,block=block,
                    arm=arm,observer='light' if arm=='light' else 'off',receipt_observer=arm!='baseline',
                    paired_content_key=f'queue-receipt-v1-{stage}-{block}-{case["id"]}'))
    return dict(schema='queue-receipt-probe-v1',scope=spec['scope'],cases=cases,order=order,
        model=base['model'],gates=base['gates'],features=base['features'],clock_spec=base['clock_spec'],
        hard_seconds=spec['hard_seconds'],posting_lag_limit_s=base['posting_lag_limit_s'],
        queue_absolute_s=base['queue_absolute_s'],queue_relative=base['queue_relative'],policy_sha256=digest(spec))


def validate_manifest(manifest):
    from .tokens import validate
    p=protocol()
    if (manifest['model'],manifest['revision'])!=(p['model']['name'],p['model']['revision']):raise ValueError('Model pin differs')
    validate(manifest,p['cases'],p['order']);groups={};pools={}
    for t in p['order']:
        rows=manifest['trials'][t['trial_id']];key=t['paired_content_key']
        signature=digest([(r['id'],manifest['pools'][r['token_key']]['input_ids']) for r in rows])
        if key in groups and groups[key]!=signature:raise ValueError('Arms must share exact tokens')
        groups[key]=signature
        if any(r['token_key']!=f'pair:{key}:{r["id"]}:{r["input_tokens"]}' for r in rows):raise ValueError('Fresh token namespace differs')
        pools.setdefault(t['block'],set()).update(r['token_key'] for r in rows)
    if any(pools[a]&pools[b] for a,b in itertools.combinations(pools,2)):raise ValueError('Blocks share pools')
    return digest(manifest)


def extract(capture,trial,case,plan):
    row=delayed.extract(capture,trial,case,plan);row['arm']=trial['arm'];row['receipt']=None
    if not trial['receipt_observer']:
        if 'receipt' in capture:raise ValueError('Baseline unexpectedly observed receipt')
        return row
    r=validate_receipt(capture['receipt'],capture)[f'plab/{trial["trial_id"]}/sentinel']
    state=dict(scheduled=r['schedules_since_release'],committed=r['commits_since_release'],
        pending_count=len(r['pending_sequences']),state=r['state'])
    events=capture['receipt']['events'];context={}
    for kind in ('schedule','commit'):
        before=[e for e in events if e['kind']==kind and e['at']<=r['posted']]
        after=[e for e in events if e['kind']==kind and e['at']>r['posted']]
        relative=lambda e:dict(sequence=e['sequence'],relative_to_post_s=e['at']-r['posted'])
        context[kind]=dict(previous=relative(before[-1]) if before else None,next=relative(after[0]) if after else None)
    release=next(e for e in events if e['kind']=='release')
    row['receipt']=dict(state=state,state_sha256=digest(state),post_context=context,
        after_release_s=r['at']-capture['release'],observation_seconds=r['observation_end']-r['at'],
        previous_commit_after_release_s=None if r['last_commit'] is None else r['last_commit']-capture['release'],
        previous_schedule_after_release_s=None if r['last_schedule'] is None else r['last_schedule']-capture['release'],
        release_observer_lag_s=release['at']-capture['release'])
    return row


def analyze(rows,plan):
    wanted={t['trial_id']:t for t in plan['order']}
    if len(rows)!=len(wanted) or {r['trial_id'] for r in rows}!=set(wanted):raise ValueError('Missing/duplicate/extra trials')
    for r in rows:
        t=wanted[r['trial_id']]
        if any(r[k]!=t[k] for k in ('case','stage','block','observer','arm')):raise ValueError('Membership differs')
        if (r['receipt'] is not None)!=t['receipt_observer']:raise ValueError('Receipt membership differs')
        if r['receipt'] is not None and digest(r['receipt']['state'])!=r['receipt']['state_sha256']:raise ValueError('State digest differs')
    groups={}
    for r in rows:
        if r['stage']=='calibration_probe':groups.setdefault((r['case'],r['block']),{})[r['arm']]=r
    controls=[];checks=[];samples=[];counts={};distributions=[]
    for (case,block),group in sorted(groups.items()):
        baseline=group['baseline']
        for arm in ('receipt','light'):
            for rid,endpoint in sorted(baseline['endpoints'].items()):
                for metric,value in sorted(endpoint.items()):
                    delta=abs(value-group[arm]['endpoints'][rid][metric])
                    limit=max(plan['gates']['observer_absolute_s'],plan['gates']['observer_relative']*value)
                    controls.append(dict(case=case,block=block,arm=arm,request=rid,metric=metric,delta_s=delta,limit_s=limit,passed=delta<=limit))
        a,b=group['receipt'],group['light'];qa,qb=(r['clocks']['sentinel']['queue_residence_s'] for r in (a,b))
        same=a['receipt']['state_sha256']==b['receipt']['state_sha256'];limit=max(plan['queue_absolute_s'],plan['queue_relative']*qa)
        checks.append(dict(case=case,block=block,receipt_trial=a['trial_id'],light_trial=b['trial_id'],
            category='same_receipt_state' if same else 'different_receipt_state',delta_s=abs(qa-qb),limit_s=limit,
            assessable=same,passed=abs(qa-qb)<=limit if same else None))
        for arm in ('baseline','receipt','light'):
            r=group[arm];samples.append(dict(trial_id=r['trial_id'],case=case,block=block,arm=arm,
                queue_s=r['clocks']['sentinel']['queue_residence_s'],posting_lag_s=r['clocks']['sentinel']['posting_lag_s'],receipt=r['receipt']))
    for r in samples:
        key='unobserved' if r['receipt'] is None else r['receipt']['state_sha256']
        group=counts.setdefault(r['case'],{}).setdefault(r['arm'],{});group[key]=group.get(key,0)+1
    for case in sorted(counts):
        for arm in ('baseline','receipt','light'):
            values=[r['queue_s'] for r in samples if r['case']==case and r['arm']==arm]
            distributions.append(dict(case=case,arm=arm,samples=len(values),minimum_s=min(values),median_s=statistics.median(values),maximum_s=max(values),spread_s=max(values)-min(values)))
    assessable=[r for r in checks if r['assessable']]
    coverage=[dict(case=case,comparable_pairs=sum(r['assessable'] for r in checks if r['case']==case),
        total_pairs=sum(r['case']==case for r in checks)) for case in sorted(counts)]
    uncovered=[r['case'] for r in coverage if r['comparable_pairs']==0]
    return dict(schema='queue-receipt-probe-report-v1',diagnostic_complete=True,
        endpoint_observer_controls_passed=all(r['passed'] for r in controls),
        same_state_queue_controls_passed=bool(assessable) and not uncovered and all(r['passed'] for r in assessable),
        state_control_coverage=coverage,unassessable_state_control_cases=uncovered,
        same_state_pairs=len(assessable),different_state_pairs=len(checks)-len(assessable),controls=controls,
        receipt_pair_checks=checks,samples=samples,state_counts=counts,queue_distributions=distributions,
        plan_sha256=digest(plan),rows_sha256=digest(rows),
        excluded_warmup_ids=[r['trial_id'] for r in rows if r['stage'] in ('warmup','clock_warmup')],
        clock_training_ids=[r['trial_id'] for r in rows if r['stage']=='train'],
        scope='calibration only; OFF phase unknown; receipt states are outcomes, not forecast inputs',
        candidate_passed=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


async def run(client,http,url,manifest,output):
    from .suite import controlled_trial,runtime_preflight,host_identity
    from .causal import dispatcher_parity
    from scripts.predict_prefill_async import runtime_inputs
    output=Path(output);p=protocol();start=time.monotonic();deadline=start+p['hard_seconds'];rows=[];events=[]
    summary=dict(diagnostic_complete=False,completed=[],failures=[],model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    save(output/'queue-probe-plan.json',p)
    try:
        validate_manifest(manifest);meta=await client.command('meta');runtime_preflight(meta,manifest);runtime_inputs(meta)
        if any(w.get('pipeline_stages') for w in meta['workers']):raise ValueError('Unexpected stage timers')
        save(output/'runtime.json',meta);save(output/'host-before.json',host_identity());cases={c['id']:c for c in p['cases']}
        for t in p['order']:
            if time.monotonic()>=deadline:raise TimeoutError('Queue probe deadline')
            print(f'Queue probe {len(rows)+1}/{len(p["order"])}: {t["trial_id"]} {t["case"]} {t["arm"]}',flush=True)
            cap=await asyncio.wait_for(controlled_trial(client,t,deadline,output/(t['trial_id']+'.json')),max(.001,deadline-time.monotonic()))
            save(output/(t['trial_id']+'.json'),cap);row=extract(cap,t,cases[t['case']],p)
            if t['observer']=='light' and not dispatcher_parity(cap,meta['workers'][0]['dispatch_table'])['matches']:raise ValueError('Dispatcher mismatch')
            if t['stage']=='train' and t['observer']=='light':
                events.append(clock.event_labels(cap,t,cases[t['case']]));save(output/'training-events.json',events)
            rows.append(row);summary['completed'].append(t['trial_id']);save(output/'queue-probe-rows.json',rows);save(output/'queue-probe.json',summary)
        final=await client.command('meta');save(output/'runtime-final.json',final)
        if digest(meta)!=digest(final):raise ValueError('Runtime drift')
        save(output/'host-after.json',host_identity());report=analyze(rows,p);save(output/'queue-probe-report.json',report)
        if time.monotonic()>=deadline:raise TimeoutError('Queue probe analysis deadline')
        summary.update(diagnostic_complete=True,**{k:report[k] for k in ('endpoint_observer_controls_passed','same_state_queue_controls_passed')})
    except BaseException as exc:summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:summary['elapsed_seconds']=time.monotonic()-start;save(output/'queue-probe.json',summary)
    return summary
