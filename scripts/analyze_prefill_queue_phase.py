"""Immutable, source-bound diagnosis of clock queue-phase rejection.

Retrospective development only. No fitting, exclusions, changed raw reports or
new model/HTTP/cap/MPC acceptance. OFF captures have no internal step trace.
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
from prefill_lab.contracts import save
from scripts.analyze_prefill_effective_capture import archive_identity,source_paths

SUPPORTED={'4c89188a6af64624ab76f3c39f17c01e2f31906f':'live_http_state_bridge',
           '33876f34b8e540c7008ecf5e64880289a60e2b84':'effective_event_clock_confirmation'}
EXTRAS={'experiments/prefill-effective-arrivals-v2.json','experiments/prefill-event-clock-v1.json',
        'experiments/prefill-event-clock-confirmation-v1.json'}
PREDICTOR=('prefill_lab/event_clock_calibration.py','prefill_lab/async_predictor.py',
    'prefill_lab/effective_calibration.py','prefill_lab/causal.py','prefill_lab/contracts.py',
    'scripts/predict_prefill_async.py','experiments/prefill-event-clock-v1.json')


def original_audit(capture):
    read=lambda name:json.loads((capture/name).read_text())
    launch=read('launch.json');commit=launch['cpu_acceptance']['commit']
    if commit not in SUPPORTED or launch['scope']!=SUPPORTED[commit]:raise ValueError('Unsupported source/scope')
    sources=read('collection-sources.json')
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
    expected=source_paths(names)|EXTRAS
    if SUPPORTED[commit]=='live_http_state_bridge':expected|={'experiments/prefill-live-http-v1.json'}
    if set(sources)!=expected:raise ValueError('Captured source inventory differs')
    with tempfile.TemporaryDirectory(prefix='pr32-original-queue-audit-') as temp:
        root=Path(temp)
        for name,sha in sources.items():
            raw=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Captured source hash differs: '+name)
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        code=('import json,sys;from pathlib import Path;from scripts.check_prefill_arrivals import audit;'
              'r=audit(Path(sys.argv[1]),event_clock=True,launch_scope='+
              (repr(launch['scope']) if launch['scope']=='live_http_state_bridge' else 'None')+');'
              'print(json.dumps(dict(audit_valid=True,candidate_passed=r["candidate_passed"],model_accepted=False)))')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
        result=subprocess.run([sys.executable,'-c',code,str(capture)],cwd=root,env=env,text=True,capture_output=True,timeout=180)
        if result.returncode:raise ValueError('Original-source audit failed: '+result.stdout+result.stderr)
        verdict=json.loads(result.stdout)
        if (verdict.get('audit_valid') is not True or verdict.get('model_accepted') is not False
                or type(verdict.get('candidate_passed')) is not bool):raise ValueError('Invalid original verdict')
    return dict(commit=commit,matched_source_files=len(sources),**verdict)


def analyze(capture,archive):
    from prefill_lab import event_clock_calibration as clock
    from prefill_lab.delayed_calibration import queue_quality
    identity,index=archive_identity(archive,capture);proof=original_audit(capture)
    read=lambda name:json.loads((capture/name).read_text())
    sources=read('collection-sources.json')
    for name in PREDICTOR:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sources[name]:raise ValueError('Predictor source changed: '+name)
    plan=read('calibration-plan.json');runtime=read('runtime.json');frozen=read('frozen.json');rows=read('endpoints.json')
    # All held cases are reported; the failed case is not used to refit anything.
    quality=queue_quality(rows,plan);rejected=read('calibration.json')['rejection_reasons']
    cases={c['id']:c for c in plan['cases']};diagnoses=[]
    for failure in rejected:
        if failure['gate']!='sentinel_queue':continue
        name=failure['case'];case=cases[name];one=dict(plan,cases=[case])
        pred=clock.rollouts(frozen['parameters_s'],one,runtime,clock.specification())[name];release=pred['release_s']
        signature=clock.forecasts({name:pred})[name]['work_signature']
        by_id={r['trial_id']:r for r in rows}
        at=case['sentinel']['arrival_ms']/1000.;window=(max(0,at-.2),at+.25)
        bounds=lambda start,end:window[0]<=start<=window[1] or window[0]<=end<=window[1]
        modeled=[dict(step=i,scheduled_s=s['scheduled_s']-release,committed_s=s['committed_s']-release,
            prefill=s['features']['prefill'],decode=s['features']['decode']) for i,s in enumerate(pred['steps'])
            if bounds(s['scheduled_s']-release,s['committed_s']-release)]
        trials=[]
        for t in plan['order']:
            if t['case']!=name:continue
            raw=read(t['trial_id']+'.json');sentinel=raw['requests']['plab/'+t['trial_id']+'/sentinel'];barrier=raw['release']
            trials.append(dict(**{k:t[k] for k in ('trial_id','observer','stage','block')},
                posting_lag_s=sentinel['queue_posted']-sentinel['target'],queue_residence_s=sentinel['enqueue']-sentinel['queue_posted'],
                enqueue_from_release_s=sentinel['enqueue']-barrier,ttft_from_target_s=sentinel['first']-sentinel['target'],
                work_signature_matches=by_id[t['trial_id']]['work_signature']==signature if t['observer']=='light' else None,
                internal_steps_observed=t['observer']=='light',steps=[dict(step=i,scheduled_s=s['start']-barrier,
                    committed_s=s['end']-barrier,prefill=s['prefill'],decode=s['decode']) for i,s in enumerate(raw['steps'])
                    if bounds(s['start']-barrier,s['end']-barrier)]))
        sweep=[]
        for scale in (.97,.98,.99,1.,1.01,1.02,1.03):
            changed=clock.rollouts([v*scale for v in frozen['parameters_s']],one,runtime,clock.specification())[name]
            s=changed['requests']['sentinel'];sweep.append(dict(all_coefficients_scale=scale,queue_residence_s=s['input_queue_wait_s']))
        diagnoses.append(dict(case=name,original_failure=failure,modeled_steps=modeled,trials=trials,
            coefficient_scaling_sweep=sweep,sweep_scope='retrospective sensitivity only; not fitted parameters or a qualified uncertainty bound',
            off_internal_poll_boundaries_measured=False))
    return dict(schema='clock-queue-phase-diagnosis-v1',archive=identity,original_audit=proof,
        native_cpu_tests=read('native-cpu-acceptance/acceptance.json')['tests'],original_collection=read('calibration.json'),
        http_trials_completed=read('bridge-summary.json')['completed'] if (capture/'bridge-summary.json').exists() else [],
        retrospective_queue_quality=quality,diagnoses=diagnoses,original_queue_scores=read('held-report.json')['sentinel_queue_scores'],
        refitted=False,raw_modified=False,local_gpu_execution_performed=False,model_accepted=False,http_accepted=False,
        adaptive_cap_accepted=False,mpc_accepted=False),index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result,index=analyze(a.capture.resolve(),a.archive.resolve())
    save(a.output/'analysis.json',result);save(a.output/'input-files.json',index)
    print(json.dumps(dict(original_audit=result['original_audit'],retrospective_queue_quality_passed=result['retrospective_queue_quality']['passed'],
        native_cpu_tests=result['native_cpu_tests'],http_trials_completed=len(result['http_trials_completed']),refitted=False)))


if __name__=='__main__':main()
