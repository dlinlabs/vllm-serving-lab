"""Read-only audit and phase diagnosis of the first native delayed pilot."""
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
from prefill_lab.contracts import save
from scripts.analyze_prefill_effective_capture import archive_identity,source_paths,endpoint_summary

COMMIT='39d0c7d6aa02e9667a5b67bc94740f5d2148264a'


def original_audit(capture):
    load=lambda n:json.loads((capture/n).read_text())
    launch=load('launch.json')
    if launch['cpu_acceptance']['commit']!=COMMIT:raise ValueError('Unsupported delayed source revision')
    sources=load('collection-sources.json')
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=ROOT,text=True).splitlines()
    expected=source_paths(names)|{'experiments/prefill-effective-arrivals-v2.json'}
    if set(sources)!=expected:raise ValueError('Delayed source inventory differs')
    with tempfile.TemporaryDirectory(prefix='pr32-arrival-original-') as temp:
        root=Path(temp)
        for name,sha in sources.items():
            raw=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)
            if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Delayed source hash differs: '+name)
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        result=subprocess.run([sys.executable,str(root/'scripts/check_prefill_arrivals.py'),'audit',str(capture)],
            cwd=root,env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')),
            text=True,capture_output=True,timeout=120)
        if result.returncode not in (0,2):raise ValueError('Original delayed audit failed: '+result.stdout+result.stderr)
        verdict=json.loads(result.stdout)
        if not verdict['audit_valid'] or verdict['model_accepted'] or (result.returncode==0)!=verdict['candidate_passed']:
            raise ValueError('Inconsistent audit status')
    return dict(commit=COMMIT,matched_source_files=len(sources),exit_status=result.returncode,**verdict)


def gates(report):
    result=dict(quality=report['quality']['passed'],endpoints=report['endpoint_scores']['passed'],
        structure_and_background_lifetime=report['structure_and_background_lifetime_valid'],
        anchor_before=report['anchor_phase_scores']['held_anchor_before']['passed'],
        anchor_after=report['anchor_phase_scores']['held_anchor_after']['passed'],
        sentinel_queue=all(r['passed'] for r in report['sentinel_queue_scores']))
    if all(result.values())!=report['candidate_passed']:raise ValueError('Gate verdict inconsistency')
    return result


def phase_diagnosis(capture,plan,frozen,runtime,failed):
    # Current timing implementation must remain identical to captured code.
    sources=json.loads((capture/'collection-sources.json').read_text())
    for name in ('prefill_lab/async_predictor.py','prefill_lab/effective_calibration.py','prefill_lab/causal.py',
                 'prefill_lab/contracts.py','scripts/predict_prefill_async.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sources[name]:raise ValueError('Timing diagnostic implementation differs')
    from prefill_lab.async_predictor import rollout
    from prefill_lab.effective_calibration import costs
    from scripts.predict_prefill_async import runtime_inputs
    scope,dispatch=runtime_inputs(runtime);result=[]
    for failure in failed:
        case=next(c for c in plan['cases'] if c['id']==failure['case'])
        pred=rollout(case,costs(frozen['parameters_s'],plan),dispatch,scope=scope)
        sentinel=pred['requests']['sentinel'];trials=[]
        for t in plan['order']:
            if t['stage']!='held_delayed' or t['case']!=case['id']:continue
            c=json.loads((capture/(t['trial_id']+'.json')).read_text());r=c['requests'][f'plab/{t["trial_id"]}/sentinel']
            item=dict(trial=t['trial_id'],observer=t['observer'],posting_lag_s=r['queue_posted']-r['target'],
                queue_residence_s=r['enqueue']-r['queue_posted'],enqueue_after_release_s=r['enqueue']-c['release'])
            if t['observer']=='light':
                i=min(range(len(c['steps'])),key=lambda i:abs(c['steps'][i]['start']-r['enqueue']))
                item.update(nearest_schedule_index=i,
                    schedule_minus_enqueue_s=c['steps'][i]['start']-r['enqueue'],
                    native_schedule_after_release_s=c['steps'][i]['start']-c['release'],
                    predicted_same_work_schedule_after_release_s=pred['steps'][i]['scheduled_s']-pred['release_s'])
            trials.append(item)
        result.append(dict(case=case['id'],predicted_enqueue_after_release_s=sentinel['enqueue_s']-pred['release_s'],trials=trials,
            interpretation='Matched work order with phase timing error; collapsed service law is implicated, individual physical cause is not identified'))
    return result


def analyze(capture,archive):
    identity,index=archive_identity(archive,capture);proof=original_audit(capture)
    load=lambda n:json.loads((capture/n).read_text());h=load('held-report.json');p=load('calibration-plan.json');f=load('frozen.json')
    failed=[r for r in h['sentinel_queue_scores'] if not r['passed']]
    result=dict(schema='native-delayed-analysis-v1',archive=identity,original_audit=proof,gates=gates(h),
        collection=load('calibration.json'),native_cpu_tests=load('native-cpu-acceptance/acceptance.json')['tests'],
        delayed_endpoint_summary=endpoint_summary(h['endpoint_scores']['cells']),queue_scores=h['sentinel_queue_scores'],
        phase_diagnosis=phase_diagnosis(capture,p,f,load('runtime.json'),failed) if h['structure_and_background_lifetime_valid'] else [],frozen_sha256=f['sha256'],
        candidate_passed=h['candidate_passed'],model_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    return result,index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','archive','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result,index=analyze(a.capture.resolve(),a.archive.resolve());save(a.output/'analysis.json',result);save(a.output/'input-files.json',index)
    print(json.dumps(dict(audit_valid=True,candidate_passed=result['candidate_passed'],gates=result['gates'])))


if __name__=='__main__':main()
