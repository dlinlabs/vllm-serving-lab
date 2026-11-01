"""Read-only, original-source audit of the first fresh native event-clock pilot.

No refitting, threshold changes, raw writes or model/control promotion. The
reported errors concern four-repeat position medians in this controlled scope.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import digest,save
from scripts.analyze_prefill_effective_capture import archive_identity,source_paths,endpoint_summary
from scripts.analyze_prefill_arrival_capture import gates

COMMIT='33876f34b8e540c7008ecf5e64880289a60e2b84'
EXPLICIT={'experiments/prefill-effective-arrivals-v2.json','experiments/prefill-event-clock-v1.json',
          'experiments/prefill-event-clock-confirmation-v1.json'}
TIMING=('prefill_lab/event_clock_calibration.py','prefill_lab/async_predictor.py','prefill_lab/effective_calibration.py',
        'prefill_lab/causal.py','prefill_lab/contracts.py','scripts/predict_prefill_async.py')


def original_audit(capture):
    load=lambda name:json.loads((capture/name).read_text())
    if load('launch.json')['cpu_acceptance']['commit']!=COMMIT:raise ValueError('Unsupported clock source revision')
    sources=load('collection-sources.json')
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=ROOT,text=True).splitlines()
    if set(sources)!=source_paths(names)|EXPLICIT:raise ValueError('Clock source inventory differs')
    with tempfile.TemporaryDirectory(prefix='pr32-original-clock-') as temp:
        root=Path(temp)
        for name,expected in sources.items():
            raw=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Clock source hash differs: '+name)
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
        result=subprocess.run([sys.executable,str(root/'scripts/check_prefill_event_clock.py'),str(capture)],
            cwd=root,env=env,text=True,capture_output=True,timeout=180)
        if result.returncode not in (0,2):raise ValueError('Original clock audit failed: '+result.stdout+result.stderr)
        verdict=json.loads(result.stdout)
        if (not verdict['audit_valid'] or verdict['model_accepted']
                or (result.returncode==0)!=verdict['candidate_passed']):raise ValueError('Inconsistent original clock verdict')
    return dict(commit=COMMIT,matched_source_files=len(sources),exit_status=result.returncode,**verdict)


def sentinel_clocks(rows):
    """Separate actual generator lag from engine queue residence; OFF only."""
    groups={}
    for row in rows:
        if row['stage']!='held_delayed' or row['observer']!='off':continue
        groups.setdefault(row['case'],[]).append(row['clocks']['sentinel'])
    return [dict(case=case,repeats=len(clocks),max_posting_lag_s=max(r['posting_lag_s'] for r in clocks),
        min_queue_residence_s=min(r['queue_residence_s'] for r in clocks),
        max_queue_residence_s=max(r['queue_residence_s'] for r in clocks)) for case,clocks in sorted(groups.items())]


def quality_summary(quality):
    result=dict(passed=quality['passed'],scope='incremental LIGHT over OFF control hooks, not uninstrumented production overhead')
    for metric in ('ttft_s','completion_s'):
        pairs=[r['delta_s'] for r in quality['paired_observer'] if r['metric']==metric]
        repeats=[r['spread_s'] for r in quality['repeat_spread'] if r['metric']==metric]
        result[metric]=dict(max_paired_delta_s=max(pairs),max_repeat_spread_s=max(repeats))
    return result


def own_prediction_checks(capture,plan,runtime,frozen,rows):
    sources=json.loads((capture/'collection-sources.json').read_text())
    for name in TIMING:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sources[name]:raise ValueError('Prediction source differs: '+name)
    from prefill_lab import event_clock_calibration as clock
    for name in ('experiments/prefill-event-clock-v1.json',):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sources[name]:raise ValueError('Clock specification differs')
    predicted=clock.predict_frozen(frozen,plan,runtime)
    light=[r for r in rows if r['observer']=='light']
    if any(r['work_signature']!=predicted[r['case']]['work_signature'] for r in light):
        raise ValueError('A LIGHT own-work signature differs, including warmups')
    events=json.loads((capture/'training-events.json').read_text())
    training_cases={r['case'] for r in rows if r['stage']=='train'}
    train_plan=dict(plan,cases=[c for c in plan['cases'] if c['id'] in training_cases])
    train_pred=clock.rollouts(frozen['parameters_s'],train_plan,runtime,clock.specification())
    recomputed=clock.event_scores(events,train_pred)
    if digest(recomputed)!=digest(frozen['train_event_scores']):raise ValueError('Stored training event scores differ')
    return dict(all_light_work_signatures_match=True,light_trials=len(light),
                training_event_scores_recomputed=True,training_event_scores=recomputed)


def analyze(capture,archive):
    identity,index=archive_identity(archive,capture);proof=original_audit(capture)
    load=lambda name:json.loads((capture/name).read_text())
    plan=load('calibration-plan.json');report=load('held-report.json');frozen=load('frozen.json');runtime=load('runtime.json');rows=load('endpoints.json')
    own=own_prediction_checks(capture,plan,runtime,frozen,rows)
    modes=Counter();requests=steps=0
    for t in plan['order']:
        raw=load(t['trial_id']+'.json');requests+=len(raw['requests']);steps+=len(raw['steps'])
        for worker in raw['workers']:modes.update(r['runtime_mode'] for r in worker['rows'])
    by_background={}
    for d in (0,16):
        ids={c['id'] for c in plan['cases'] if c['sentinel'] and c['background_decoders']==d}
        by_background[str(d)]=endpoint_summary([r for r in report['endpoint_scores']['cells'] if r['case'] in ids])
    result=dict(schema='native-event-clock-confirmation-analysis-v1',archive=identity,original_audit=proof,
        collection=load('calibration.json'),native_cpu_tests=load('native-cpu-acceptance/acceptance.json')['tests'],
        gates=gates(report),delayed_endpoint_summary=endpoint_summary(report['endpoint_scores']['cells']),
        by_background=by_background,queue_scores=report['sentinel_queue_scores'],sentinel_clock_ranges=sentinel_clocks(rows),
        anchors={phase:endpoint_summary(score['cells']) for phase,score in report['anchor_phase_scores'].items()},
        held_quality=quality_summary(report['quality']),training_quality=quality_summary(frozen['train_quality']),
        own_prediction_checks=own,counts=dict(request_instances=requests,light_executed_steps=steps,worker_modes=dict(modes),
            observers=dict(Counter(t['observer'] for t in plan['order'])),phases=dict(Counter(t['stage'] for t in plan['order']))),
        frozen=dict(sha256=frozen['sha256'],parameters=frozen['parameters'],parameters_s=frozen['parameters_s'],
            local_sensitivity=frozen['local_sensitivity'],optimizer_runs=frozen['optimizer_runs'],measurement_origin=frozen['measurement_origin']),
        runtime=dict(gpu=runtime['workers'][0]['device_name'],torch=runtime['workers'][0]['torch_version'],
            cuda=runtime['workers'][0]['cuda_version'],kv_blocks=runtime['num_gpu_blocks'],block_size=runtime['block_size'],
            source_pins=len(runtime['source']['files']),token_budget=runtime['max_num_batched_tokens'],max_seqs=runtime['max_num_seqs'],
            max_model_len=runtime['max_model_len']),
        candidate_passed=proof['candidate_passed'],native_collection_performed=True,local_gpu_execution_performed=False,
        refitted_for_analysis=False,physical_costs_identified=False,model_accepted=False,complete_experiment_passed=False,
        adaptive_cap_accepted=False,mpc_accepted=False)
    return result,index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result,index=analyze(a.capture.resolve(),a.archive.resolve())
    save(a.output/'analysis.json',result);save(a.output/'input-files.json',index)
    print(json.dumps(dict(audit_valid=result['original_audit']['audit_valid'],candidate_passed=result['candidate_passed'],
                         native_cpu_tests=result['native_cpu_tests'],gates=result['gates'])))


if __name__=='__main__':main()
