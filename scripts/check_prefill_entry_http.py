"""Independent raw audit of the combined controlled/entry HTTP experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prefill_lab import entry_http,entry_http_model,live_http
from prefill_lab.contracts import digest,finite
from scripts.check_prefill_queue_confirmation import audit as audit_controlled


def audit(root):
    from prefill_lab.causal import dispatcher_parity
    root=Path(root);load=lambda n:json.loads((root/n).read_text());controlled=root/'controlled'
    prerequisite=audit_controlled(controlled,entry_http=True)
    if not prerequisite['candidate_passed']:raise ValueError('Controlled prerequisite did not pass')
    for n in ('launch.json','collection-sources.json','token-manifest.json','native-cpu-acceptance/acceptance.json','native-cpu-acceptance/junit.xml','native-cpu-acceptance/pytest.log'):
        if (root/n).read_bytes()!=(controlled/n).read_bytes():raise ValueError('Controlled/parent evidence differs: '+n)
    p=entry_http.protocol();summary=load('entry-http-summary.json');manifest=load('http-token-manifest.json')
    if digest(load('entry-http-plan.json'))!=digest(p) or not summary['collection_complete'] or summary['failures'] or summary['completed']!=[t['trial_id'] for t in p['order']]:raise ValueError('Incomplete entry HTTP collection')
    entry_http.validate_manifest(manifest);entry_http.old.validate_pair(load('token-manifest.json'),manifest)
    config=load('entry-http-config.json');runtime=json.loads((controlled/'runtime.json').read_text());first=load('bridge/'+p['order'][0]['trial_id']+'.json')['capture']
    last_controlled=json.loads((controlled/'qc-240.json').read_text())
    if (config['controlled_finished_at']!=last_controlled['end'] or not config['configured_at']<=first['start'] or
            config['http_manifest_sha256']!=digest(manifest) or config['controlled_report_sha256']!=digest(prerequisite) or
            digest(config['base_model'])!=digest(json.loads((controlled/'base-model.json').read_text())) or
            digest(config['entry_freeze'])!=digest(json.loads((controlled/'entry-freeze.json').read_text()))):raise ValueError('Controlled-to-HTTP handoff differs')
    model=entry_http_model.costs(config,runtime,now=first['start'],domain=config['clock_domain'])
    if load('entry-http-configured.json')!=dict(configured=True,frozen_sha256=config['sha256']):raise ValueError('Native configuration acknowledgement differs')
    if digest(load('http-runtime-final.json'))!=digest(runtime):raise ValueError('HTTP runtime drift')
    journal=[json.loads(line) for line in (root/'gateway.jsonl').read_text().splitlines() if line.strip()];rows=[];captures=[]
    for t in p['order']:
        bundle=load('bridge/'+t['trial_id']+'.json');cap=bundle['capture'];captures.append(cap)
        if t['observer']=='light' and not dispatcher_parity(cap,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('HTTP dispatcher differs')
        if digest(load('bridge/'+t['trial_id']+'-client.json'))!=digest(bundle['client']):raise ValueError('HTTP client sidecar differs')
        ids={r['trace_id'] for r in bundle['client']['rows']}
        if digest([r for r in journal if r.get('request_id') in ids])!=digest(bundle['gateway']):raise ValueError('HTTP gateway journal differs')
        rows.append(entry_http.extract(bundle,t,manifest,runtime,config,model=model))
    if any(not all(finite(c.get(k)) for k in ('start','end')) or c['start']>c['end'] for c in captures) or any(a['end']>b['start'] for a,b in zip(captures,captures[1:])):raise ValueError('HTTP trial chronology differs')
    if any(r['clock_domain']!=config['clock_domain'] for r in rows):raise ValueError('HTTP milestones differ from the configured clock domain')
    if digest(rows)!=digest(load('http-rows.json')):raise ValueError('HTTP rows differ from raw evidence')
    live_http.validate_warmup(rows,p);transport=load('transport-freeze.json');seal=load('transport-seal.json')
    boundary=next(i for i,t in enumerate(p['order']) if t['stage'] not in ('warmup','train'))
    if (seal['completed']!=[t['trial_id'] for t in p['order'][:boundary]] or seal['next_trial']!=p['order'][boundary]['trial_id'] or
            seal['config_sha256']!=config['sha256'] or seal['transport_sha256']!=transport['sha256'] or seal['source_sha256']!=config['source_sha256'] or
            not finite(seal.get('at_monotonic')) or not captures[boundary-1]['end']<=seal['at_monotonic']<=captures[boundary]['start']):raise ValueError('Transport was not sealed before held HTTP trials')
    report=entry_http.score(rows,p,transport)
    if digest(report)!=digest(load('entry-http-report.json')) or summary['candidate_passed']!=report['candidate_passed'] or any(summary.get(k) is not False for k in entry_http.qc.ACCEPTANCE):raise ValueError('Entry HTTP report/summary differs')
    expected={r['trace_id'] for t in p['order'] for r in load('bridge/'+t['trial_id']+'.json')['client']['rows']}
    if {r.get('request_id') for r in journal if r['event']=='received'}!=expected:raise ValueError('Unexpected gateway traffic')
    live_http.gateway_drained(load('gateway-final.json'),p);run_id=live_http.gateway_contract(load('gateway-start.json'))
    if run_id!=live_http.gateway_contract(load('gateway-final.json')) or any(r.get('run_id')!=run_id for r in journal):raise ValueError('Gateway run identity drift')
    status=load('telemetry-status.json')
    if status.get('run_id')!=run_id or status.get('status')!='closed' or status.get('dropped',1) or status.get('write_errors',1) or not status.get('diagnostics',{}).get('complete'):raise ValueError('Gateway telemetry did not close cleanly')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    result=audit(parser.parse_args().directory)
    print(json.dumps(dict(audit_valid=True,candidate_passed=result['candidate_passed'],model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)))
