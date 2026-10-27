"""Rebuild receipt diagnostics from raw captures using the exact source checkout."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab import queue_probe as probe
from prefill_lab.contracts import digest


def audit(directory):
    from prefill_lab.suite import runtime_preflight
    from prefill_lab.causal import dispatcher_parity
    from scripts.run_prefill_service import code_identity
    from scripts.run_linux_cpu_tests import validate_report
    from scripts.predict_prefill_async import runtime_inputs
    directory=Path(directory);load=lambda name:json.loads((directory/name).read_text())
    p=probe.protocol();summary=load('queue-probe.json')
    if digest(load('queue-probe-plan.json'))!=digest(p):raise ValueError('Plan differs')
    if not summary['diagnostic_complete'] or summary['failures'] or summary['completed']!=[t['trial_id'] for t in p['order']]:raise ValueError('Incomplete collection')
    runtime=load('runtime.json');manifest=load('token-manifest.json')
    if digest(runtime)!=digest(load('runtime-final.json')):raise ValueError('Runtime drift')
    probe.validate_manifest(manifest);runtime_preflight(runtime,manifest);runtime_inputs(runtime)
    if any(w.get('pipeline_stages') for w in runtime['workers']):raise ValueError('Unexpected deep stages')
    pins=json.loads((ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    for source in [runtime['source'],*[w['source'] for w in runtime['workers']]]:
        if source['version']!='0.30.0' or source['files']!=pins:raise ValueError('Installed pins differ')
    sources=load('collection-sources.json');launch=load('launch.json');acceptance=launch['cpu_acceptance']
    if sources!=code_identity() or launch['source_sha256']!=sources or acceptance['source_sha256']!=sources or launch['scope']!='queue_receipt_calibration_probe' or not acceptance['valid'] or not acceptance['tracked_clean']:
        raise ValueError('Source/CPU identity differs; use exact captured checkout')
    env=launch['environment']
    if env.get('PREFILL_LAB_QUEUE_PROBE')!='1' or any(env.get(k)!='0' for k in ('PREFILL_LAB_STAGES','PREFILL_LAB_EFFECTIVE_CALIBRATION','PREFILL_LAB_DELAYED_ARRIVALS','PREFILL_LAB_EVENT_CLOCK','PREFILL_LAB_LIVE_STATE')):raise ValueError('Launch modes differ')
    if load('native-cpu-acceptance/acceptance.json')!=acceptance:raise ValueError('CPU record differs')
    if validate_report(directory/'native-cpu-acceptance/junit.xml')['test_cases']!=acceptance['tests']:raise ValueError('CPU count differs')
    rows=[];events=[];cases={c['id']:c for c in p['cases']}
    for t in p['order']:
        cap=load(t['trial_id']+'.json');rows.append(probe.extract(cap,t,cases[t['case']],p))
        if t['stage']=='train' and t['observer']=='light':events.append(probe.clock.event_labels(cap,t,cases[t['case']]))
        if t['observer']=='light' and not dispatcher_parity(cap,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('Dispatcher differs')
    if digest(events)!=digest(load('training-events.json')):raise ValueError('Training events differ')
    if digest(rows)!=digest(load('queue-probe-rows.json')):raise ValueError('Rows differ from raw capture')
    report=probe.analyze(rows,p)
    if digest(report)!=digest(load('queue-probe-report.json')):raise ValueError('Report differs')
    if any(summary[k]!=report[k] for k in ('endpoint_observer_controls_passed','same_state_queue_controls_passed')):raise ValueError('Summary differs')
    if any(summary.get(k) is not False for k in ('model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted')):raise ValueError('Diagnostic cannot accept model/control')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    report=audit(p.parse_args().directory)
    print(json.dumps(dict(audit_valid=True,diagnostic_complete=True,model_accepted=False,
        endpoint_observer_controls_passed=report['endpoint_observer_controls_passed'],
        same_state_queue_controls_passed=report['same_state_queue_controls_passed'])))


if __name__=='__main__':main()
