"""Independent raw audit on the exact source used by the native bridge."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prefill_lab.contracts import digest
from prefill_lab.live_experiment import extract,validate_manifest,validate_pair
from prefill_lab.live_http import protocol,score,gateway_contract,gateway_drained,validate_warmup
from scripts.check_prefill_arrivals import audit as audit_clock


def audit(root):
    read=lambda name:json.loads((root/name).read_text())
    clock=audit_clock(root,event_clock=True,launch_scope='live_http_state_bridge')
    if not clock['candidate_passed']:raise ValueError('Clock phase did not pass')
    p=protocol();summary=read('bridge-summary.json')
    if (digest(read('bridge-plan.json'))!=digest(p) or not summary['collection_complete'] or summary['failures']
            or summary['completed']!=[t['trial_id'] for t in p['order']]):raise ValueError('Incomplete live HTTP plan')
    manifest=read('http-token-manifest.json');validate_manifest(manifest);validate_pair(read('token-manifest.json'),manifest);rows=[]
    frozen=read('frozen.json');runtime=read('runtime.json')
    journal=[json.loads(line) for line in (root/'gateway.jsonl').read_text().splitlines()]
    for t in p['order']:
        bundle=read('bridge/'+t['trial_id']+'.json');ids={r['trace_id'] for r in bundle['client']['rows']}
        if digest(read('bridge/'+t['trial_id']+'-client.json'))!=digest(bundle['client']):raise ValueError('Client sidecar differs from trial record')
        actual=[r for r in journal if r.get('request_id') in ids]
        if digest(actual)!=digest(bundle['gateway']):raise ValueError('Gateway journal differs from trial records')
        rows.append(extract(bundle,t,manifest,runtime,frozen))
    if digest(rows)!=digest(read('http-rows.json')):raise ValueError('HTTP rows differ from raw inputs')
    validate_warmup(rows,p)
    result=score(rows,p,read('transport-freeze.json'))
    if digest(result)!=digest(read('bridge-report.json')) or result['candidate_passed']!=summary['candidate_passed']:
        raise ValueError('HTTP verdict differs')
    expected={r['trace_id'] for t in p['order'] for r in read('bridge/'+t['trial_id']+'.json')['client']['rows']}
    if {r.get('request_id') for r in journal if r['event']=='received'}!=expected:raise ValueError('Unexpected gateway traffic')
    gateway_drained(read('gateway-final.json'),p)
    run_id=gateway_contract(read('gateway-start.json'))
    if run_id!=gateway_contract(read('gateway-final.json')) or any(r.get('run_id')!=run_id for r in journal):raise ValueError('Gateway run identity drift')
    status=read('telemetry-status.json')
    if status.get('run_id')!=run_id or status.get('status')!='closed' or status.get('dropped',1) or status.get('write_errors',1) or not status.get('diagnostics',{}).get('complete'):raise ValueError('Gateway telemetry did not close cleanly')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();r=audit(a.directory)
    print(json.dumps(dict(audit_valid=True,candidate_passed=r['candidate_passed'],model_accepted=False)))
    return 0 if r['candidate_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
