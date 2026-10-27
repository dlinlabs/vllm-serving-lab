"""Independently rebuild a complete controlled confirmation from raw evidence."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab import queue_confirmation as qc,event_clock_calibration as clock
from prefill_lab.contracts import digest,finite


def verify_seal(seal,plan,captures,source,manifest,model,frozen):
    boundary=next(i for i,t in enumerate(plan['order']) if t['stage'] in qc.HELD)
    if (seal['completed']!=[t['trial_id'] for t in plan['order'][:boundary]] or
            seal['next_trial']!=plan['order'][boundary]['trial_id'] or seal['source_sha256']!=source or
            seal['manifest_sha256']!=digest(manifest) or seal['model_sha256']!=model['sha256'] or seal['freeze_sha256']!=frozen['sha256']):
        raise ValueError('Seal identity/membership differs')
    at=seal['at_monotonic']
    if not finite(at) or not captures[boundary-1]['end']<=at<=captures[boundary]['start']:
        raise ValueError('Freeze was not sealed between TRAIN and held collection')


def audit(directory):
    from prefill_lab.suite import runtime_preflight
    from prefill_lab.causal import dispatcher_parity
    from scripts.run_prefill_service import code_identity
    from scripts.run_linux_cpu_tests import validate_report
    from scripts.predict_prefill_async import runtime_inputs
    root=Path(directory);load=lambda n:json.loads((root/n).read_text());plan=qc.protocol();summary=load('confirmation.json')
    if digest(load('queue-confirmation-plan.json'))!=digest(plan):raise ValueError('Confirmation plan differs')
    if not summary['collection_complete'] or summary['failures'] or summary['completed']!=[t['trial_id'] for t in plan['order']]:raise ValueError('Incomplete confirmation')
    manifest=load('token-manifest.json');qc.validate_manifest(manifest);runtime=load('runtime.json')
    if digest(runtime)!=digest(load('runtime-final.json')):raise ValueError('Runtime drift')
    runtime_preflight(runtime,manifest);runtime_inputs(runtime)
    if any(w.get('pipeline_stages') for w in runtime['workers']):raise ValueError('Unexpected deep stages')
    pins=json.loads((ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    if any(s['version']!='0.30.0' or s['files']!=pins for s in [runtime['source'],*[w['source'] for w in runtime['workers']]]):raise ValueError('Installed source pins differ')
    source=load('collection-sources.json');launch=load('launch.json');acceptance=launch['cpu_acceptance'];env=launch['environment']
    if (source!=code_identity() or launch['source_sha256']!=source or acceptance['source_sha256']!=source or
            launch['scope']!='independent_queue_confirmation' or not acceptance['valid'] or not acceptance['tracked_clean'] or
            load('native-cpu-acceptance/acceptance.json')!=acceptance):raise ValueError('Native source/CPU identity differs')
    if validate_report(root/'native-cpu-acceptance/junit.xml')['test_cases']!=acceptance['tests']:raise ValueError('Native CPU denominator differs')
    if env.get('PREFILL_LAB_QUEUE_CONFIRMATION')!='1' or any(env.get('PREFILL_LAB_'+k)!='0' for k in
            ('QUEUE_PROBE','STAGES','EFFECTIVE_CALIBRATION','DELAYED_ARRIVALS','EVENT_CLOCK','LIVE_STATE')):raise ValueError('Conflicting launch modes')
    captures=[];rows=[];cases={c['id']:c for c in plan['cases']};previous=None
    for t in plan['order']:
        cap=load(t['trial_id']+'.json')
        if not all(finite(cap.get(k)) for k in ('start','end')) or cap['start']>cap['end'] or (previous is not None and cap['start']<previous):raise ValueError('Trial chronology differs')
        previous=cap['end'];captures.append(cap);rows.append(qc.extract(cap,t,cases[t['case']],plan))
        if t['observer']=='light' and not dispatcher_parity(cap,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('Dispatcher mismatch')
    qc.membership(rows,plan)
    if digest(rows)!=digest(load('confirmation-rows.json')):raise ValueError('Rows differ from raw capture')
    # Refit the predeclared estimator on fresh TRAIN only, not merely a rehashed
    # persisted coefficient document. This includes first-entry worker spans.
    boundary=next(i for i,t in enumerate(plan['order']) if t['stage'] in qc.HELD)
    train_caps=[c for c,t in zip(captures,plan['order']) if t['stage']=='train' and t['observer']=='light']
    model,frozen=qc.seal_training(rows[:boundary],train_caps,plan,runtime,time.monotonic()+240)
    if digest(model)!=digest(load('base-model.json')) or digest(frozen)!=digest(load('entry-freeze.json')):raise ValueError('TRAIN refit/freeze differs')
    verify_seal(load('seal.json'),plan,captures,source,manifest,model,frozen)
    report=qc.evaluate(rows,plan,frozen)
    if digest(report)!=digest(load('confirmation-report.json')):raise ValueError('Confirmation report differs')
    if summary['candidate_passed']!=report['candidate_passed'] or summary['status']!=report['status'] or any(summary.get(k) is not False for k in qc.ACCEPTANCE):raise ValueError('Summary verdict differs')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    report=audit(parser.parse_args().directory)
    print(json.dumps(dict(audit_valid=True,status=report['status'],candidate_passed=report['candidate_passed'],
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)))
