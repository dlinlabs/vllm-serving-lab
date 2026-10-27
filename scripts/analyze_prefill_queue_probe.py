"""Immutable receipt-probe analysis and TRAIN-only submit-clock development.

Reconstruct the exact 8bd96cb native source for audit/fitting before using new
analysis code. No raw file, old policy, original verdict or acceptance is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab import queue_probe,event_clock_calibration as clock,queue_phase,submit_clock,effective_calibration as base
from prefill_lab.contracts import digest,save
from scripts.analyze_prefill_effective_capture import archive_identity,source_paths

COMMIT='8bd96cbe07ba8e7f26aab9fd33ff31af5453e55f'
EXTRAS={'experiments/prefill-effective-arrivals-v2.json','experiments/prefill-event-clock-v1.json',
    'experiments/prefill-event-clock-confirmation-v1.json','experiments/prefill-live-http-v1.json',
    'experiments/prefill-queue-phase-v1.json','experiments/prefill-queue-probe-v1.json'}


def original_source(capture,destination):
    load=lambda n:json.loads((capture/n).read_text())
    launch=load('launch.json');sources=load('collection-sources.json')
    if launch['cpu_acceptance']['commit']!=COMMIT or launch['scope']!='queue_receipt_calibration_probe':
        raise ValueError('Unsupported receipt capture source/scope')
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=ROOT,text=True).splitlines()
    if set(sources)!=source_paths(names)|EXTRAS:raise ValueError('Native source inventory differs')
    for name,sha in sources.items():
        raw=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Git/native source mismatch: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return sources


def original_audit_and_fit(capture,output):
    with tempfile.TemporaryDirectory(prefix='pr32-receipt-source-') as temporary:
        source=Path(temporary);sources=original_source(capture,source)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source)+os.pathsep+os.environ.get('PYTHONPATH',''))
        def command(script,*args,timeout):
            r=subprocess.run([sys.executable,str(source/'scripts'/script),*map(str,args)],cwd=source,
                env=env,capture_output=True,text=True,timeout=timeout)
            if r.returncode:raise ValueError('Original source command failed: '+r.stdout+r.stderr)
            return json.loads(r.stdout)
        proof=command('check_prefill_queue_probe.py',capture,timeout=180)
        result=command('fit_prefill_queue_probe.py',capture,'--output',output/'native-fit',timeout=240)
        if proof['audit_valid'] is not True or result['fit_complete'] is not True:raise ValueError('Native audit/fit incomplete')
    return dict(commit=COMMIT,matched_source_files=len(sources),**proof),result


def score(rows,plan,predicted,cells):
    queue_probe.analyze(rows,plan)  # Preserve the entire frozen denominator.
    checks=[];counts={}
    for row in rows:
        if row['stage']!='calibration_probe':continue
        name=row['case'];q=row['clocks']['sentinel']['queue_residence_s'];cell=cells.get(name)
        point=predicted[name]['sentinel_queue_residence_s']
        a=queue_phase.assign(q,dict(branches=[dict(queue_s=point)]),queue_phase.policy())
        b=queue_phase.assign(q,cell,queue_phase.policy()) if cell else None
        checks.append(dict(trial_id=row['trial_id'],case=name,block=row['block'],arm=row['arm'],
            observed_s=q,predicted_point_s=point,point=a,phase=b))
        if cell:
            group=counts.setdefault(name,{}).setdefault(row['arm'],{str(i):0 for i in range(len(cell['branches']))}|{'unresolved':0})
            key='unresolved' if b['branch'] is None else str(b['branch']);group[key]+=1
    endpoints={arm:base.endpoint_scores([r for r in rows if r['stage']=='calibration_probe' and r['arm']==arm],predicted,plan)
        for arm in ('baseline','receipt','light')}
    structure=[]
    for row in rows:
        if row['stage']=='calibration_probe' and row['arm']=='light':
            structure.append(dict(trial_id=row['trial_id'],case=row['case'],matches=row['work_signature']==predicted[row['case']]['work_signature']))
    result=dict(sample_checks=checks,endpoint_scores=endpoints,work_signature_checks=structure,branch_counts=counts,
        unobserved_branches=[dict(case=name,arm=arm,branch=b) for name,arms in counts.items() for arm,values in arms.items() for b,n in values.items() if b!='unresolved' and n==0],
        queue_budget_passed=bool(checks) and all(c['phase'] is not None and c['phase']['passed'] and c['phase']['branch'] is not None for c in checks),
        scope='retrospective calibration-probe development; branch assignments are not measured OFF phase or independent confirmation',
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    result['counts']={arm:dict(samples=sum(c['arm']==arm for c in checks),
        point_failures=sum(c['arm']==arm and not c['point']['passed'] for c in checks),
        phase_failures=sum(c['arm']==arm and (c['phase'] is None or not c['phase']['passed'] or c['phase']['branch'] is None) for c in checks))
        for arm in ('baseline','receipt','light')}
    return result


def event_diagnosis(capture,plan,runtime,model,submit_freeze):
    rows=json.loads((capture/'queue-probe-rows.json').read_text());events=json.loads((capture/'training-events.json').read_text())
    predictions=clock.rollouts(model['parameters_s'],plan,runtime,clock.specification());training=[]
    gamma=submit_freeze['labels']['constants_s']
    submit_predictions=submit_clock.own_rollouts(model['parameters_s'],plan,runtime,gamma['prefill'],gamma['decode'])
    for row in events:
        if not row['case'].endswith('D16'):continue
        own=clock.event_predictions(predictions[row['case']],row)
        alternative=clock.event_predictions(submit_predictions[row['case']],row)
        i=next(i for i,times in enumerate(row['times_s']) if times[1]>0)
        training.append(dict(trial_id=row['trial_id'],case=row['case'],observed_first_post_release_commit_s=row['times_s'][i][1],
            modeled_first_post_release_commit_s=float(own[i,1]),error_s=float(own[i,1]-row['times_s'][i][1]),
            submit_nominal_first_post_release_commit_s=float(alternative[i,1]),submit_nominal_error_s=float(alternative[i,1]-row['times_s'][i][1])))
    trials=[]
    for t in plan['order']:
        if t['stage']!='calibration_probe' or not t['case'].endswith('at50-D16'):continue
        cap=json.loads((capture/(t['trial_id']+'.json')).read_text());release=cap['release']
        request=cap['requests'][f'plab/{t["trial_id"]}/sentinel']
        entry=dict(**{k:t[k] for k in ('trial_id','arm','case','block')},raw_capture_sha256=digest(cap),
            queue_s=request['enqueue']-request['queue_posted'],post_from_release_s=request['queue_posted']-release,
            enqueue_from_release_s=request['enqueue']-release,internal_state_observed='receipt' in cap)
        if 'receipt' in cap:
            receipt=next(e for e in cap['receipt']['events'] if e['kind']=='receipt')
            barrier=next(e for e in cap['receipt']['events'] if e['kind']=='release')
            entry.update(scheduled_at_receipt=receipt['scheduled'],committed_at_receipt=receipt['committed'],
                state_sha256=digest(receipt['state']),events=[dict(kind=e['kind'],sequence=e.get('sequence'),relative_s=e['at']-release)
                for e in cap['receipt']['events'] if -.025<=e['at']-release<=.15])
        trials.append(entry)
    return dict(training_first_pending_commits=training,at50_D16_trials=trials,
        conclusion='CPU return unavailability is missing from F0; cpu_after placement is an effective-model hypothesis, not a physical stage identification',
        baseline_internal_phase_observed=False,causal_perturbation_identified=False)


def run(capture,archive,output):
    capture,archive,output=map(lambda p:Path(p).resolve(),(capture,archive,output))
    if output==capture or capture in output.parents:raise ValueError('Output must be outside immutable evidence')
    identity,index=archive_identity(archive,capture);output.mkdir(parents=True,exist_ok=False)
    save(output/'input-files.json',index)
    proof,fit=original_audit_and_fit(capture,output)
    read=lambda n:json.loads((capture/n).read_text())
    plan=read('queue-probe-plan.json');runtime=read('runtime.json');rows=read('queue-probe-rows.json')
    model=json.loads((output/'native-fit/frozen.json').read_text())
    # Guard all timing/predictor sources used by the new analysis; unrelated
    # analysis files may differ, but a changed service law may not replay as F0.
    sources=read('collection-sources.json')
    for name in queue_phase.policy()['timing_source_sha256']:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sources[name]:raise ValueError('F0 timing source changed: '+name)
    train=[r for r in rows if r['stage']=='train'];predicted=clock.predict_frozen(model,plan,runtime)
    f0=queue_phase.freeze(train,plan,runtime,model,predicted);save(output/'f0-phase-freeze.json',f0)
    save(output/'f0-development-report.json',score(rows,plan,predicted,f0['cells']))
    captures=[read(t['trial_id']+'.json') for t in plan['order'] if t['stage']=='train' and t['observer']=='light']
    # Seal the TRAIN-only constants/domain and branch plan before reading probe
    # queue outcomes into the development scoring function.
    f1=submit_clock.freeze(captures,plan,runtime,model);save(output/'submit-development-freeze.json',f1)
    save(output/'submit-development-report.json',score(rows,plan,f1['predictions'],f1['cells']))
    save(output/'event-diagnosis.json',event_diagnosis(capture,plan,runtime,model,f1))
    save(output/'native-measurement-report.json',read('queue-probe-report.json'))
    summary=dict(schema='receipt-probe-development-analysis-v1',archive=identity,original_audit=proof,
        original_native_cpu_tests=read('native-cpu-acceptance/acceptance.json')['tests'],
        collection_elapsed_seconds=read('queue-probe.json')['elapsed_seconds'],
        f0_readiness_passed=f0['readiness_passed'],submit_readiness_passed=f1['readiness_passed'],
        development_only=True,held_confirmation_performed=False,model_selected=False,
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    save(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.capture,args.archive,args.output)))
