"""Read-only audit of a captured PR32 suite; never fit or qualify new labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import audit_capture,digest,save
from prefill_lab.causal import dispatcher_parity
from prefill_lab.analysis import observer_gate,rows_from_capture
from prefill_lab.tokens import validate


def intervals(rows):
    ordered=sorted((r['start'],r['end']) for r in rows)
    if not ordered:return dict(count=0,sum_s=0.,union_s=0.,span_s=0.,sum_over_union=None,max_inflight=0)
    union=0.;left,right=ordered[0]
    for a,b in ordered[1:]:
        if a>right:union+=right-left;left,right=a,b
        else:right=max(right,b)
    union+=right-left
    edges=sorted([(a,1) for a,b in ordered]+[(b,-1) for a,b in ordered])
    inflight=maximum=0
    for _,delta in edges:inflight+=delta;maximum=max(maximum,inflight)
    total=sum(b-a for a,b in ordered)
    return dict(count=len(rows),sum_s=total,union_s=union,span_s=max(b for a,b in ordered)-ordered[0][0],
                sum_over_union=total/union if union else None,max_inflight=maximum)


def profile_window(capture,count=5):
    workers=[r for w in capture['workers'] for r in w['rows']]
    proof=capture['workers'][0].get('deep_profile',{})
    by_id={r['step_id']:r for r in workers}
    selected=[by_id[key] for key in proof['step_ids']] if proof.get('schema')=='foreground-profile-v1' else workers[:count]
    roles={k:v['role'] for k,v in capture['requests'].items()}
    foreground=lambda row:any(roles.get(rid)!='background' for rid in row['counts'])
    return dict(worker_steps=len(workers),first_foreground_worker_step=next((i+1 for i,r in enumerate(workers) if foreground(r)),None),
                profile_window_policy=proof.get('schema','legacy-first-five-calls'),profiled_steps=len(selected),
                foreground_steps_in_profile=sum(foreground(r) for r in selected),
                profiled_modes=dict(Counter(r['runtime_mode'] for r in selected)),
                profiled_actual_tokens=[r['actual_tokens'] for r in selected])


def analyze(capture,output,archive=None):
    capture=Path(capture);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    load=lambda name:json.loads((capture/name).read_text())
    launch=load('launch.json');acceptance=launch['cpu_acceptance'];ref=acceptance['commit']
    if not acceptance.get('valid') or not acceptance.get('tracked_clean'):raise ValueError('Capture lacks clean native acceptance')
    def source(path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
    declared=load('collection-sources.json')
    mismatches=[p for p,h in declared.items() if hashlib.sha256(source(p)).hexdigest()!=h]
    if mismatches or declared!=acceptance['source_sha256'] or declared!=launch['source_sha256']:
        raise ValueError('Capture/native acceptance source identity mismatch: '+str(mismatches))
    order_data=json.loads(source('experiments/prefill-service-v1.order.json'))
    cases=json.loads(source('experiments/prefill-service-v1.cases.json'))['cases'];order=order_data['order']
    tokens=load('token-manifest.json');validate(tokens,cases,order)
    meta=load('runtime.json');final=load('runtime-final.json');suite=load('suite.json')
    if digest(meta)!=digest(final) or meta['token_manifest_sha256']!=digest(tokens):raise ValueError('Runtime/token identity changed')
    source_map=json.loads(source('experiments/prefill-vllm-v030-source-map.json'))
    if meta['source']['files']!=source_map['sha256']:raise ValueError('Installed vLLM source mismatch')
    trace_raw=source('results/trace-experiment-96-r1/trace.json');trace=json.loads(trace_raw)
    if hashlib.sha256(trace_raw).hexdigest()!=load('trace-identity.json')['sha256']:raise ValueError('Frozen Azure trace mismatch')
    report=dict(capture_commit=ref,source_files_verified=len(declared),installed_source_files=len(source_map['sha256']),
        recorded_native_cpu_acceptance={k:acceptance[k] for k in ('valid','tests','tracked_clean')},
        runtime={k:meta[k] for k in ('async_scheduling','max_concurrent_batches','scheduler_class','max_num_batched_tokens')},
        scope='Offline collection/label/profile audit; no post-hoc model selection or GPU scope qualification',
        model_accepted=False,gpu_validated=False,complete_experiment_passed=False,
        suite={k:suite.get(k) for k in ('complete','elapsed_seconds','calibration_status','failures','block_dependence_status')},
        stages={k:dict(integrity=v['integrity'],completed=len(v['completed_trial_ids']),planned=len(v['planned_trial_ids'])) for k,v in suite['stages'].items()},
        trial_audits=[],training_wall_labels=[],deep_profiles=[],transfers=[])
    observer=[]
    for trial in order:
        c=load(trial['trial_id']+'.json');workers=[r for w in c['workers'] for r in w['rows']]
        audit=audit_capture(c,workers,require_steps=trial['observer']!='off')
        parity=dispatcher_parity(c,meta['workers'][0]['dispatch_table']) if trial['observer']!='off' else dict(matches=True)
        report['trial_audits'].append(dict(trial_id=trial['trial_id'],work_valid=audit['valid'],dispatcher_valid=parity['matches'],
            steps=audit['step_count'],async_overlap=audit['async_overlap'],problems=audit['problems']))
        if trial['stage']=='observer_check':observer.append({k:c[k] for k in ('trial_id','status','errors','requests','release','start','end')})
        if trial.get('analysis_role')=='train':
            rows=rows_from_capture(c)
            report['training_wall_labels'].append(dict(trial_id=trial['trial_id'],case=trial['case'],block=trial['block'],**intervals(rows)))
        if trial['stage']=='deep_diagnostic':
            window=profile_window(c)
            path=next((capture/'deep-traces').glob(trial['trial_id']+'-worker-*.json'))
            es=json.loads(path.read_text())['traceEvents'];gpu=[e for e in es if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
            report['deep_profiles'].append(dict(trial_id=trial['trial_id'],case=trial['case'],**window,
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trace_events=len(es),
                categories=dict(Counter(e.get('cat','metadata') for e in es)),
                gpu_streams=sorted({e['args']['stream'] for e in gpu}),
                step_id_markers=sum(e.get('name','').startswith('prefill_lab_step/') for e in es),
                qualifies_foreground_gpu_scope=False))
    measured_observer=observer_gate(observer,order)
    report['observer_gate_recomputed']=dict(status=measured_observer['status'],rows=[{k:v for k,v in r.items() if k!='log_ratios'} for r in measured_observer['rows']])
    report['drift']=suite['drift'];report['model_freeze']=load('model-freeze.json');report['held_report']=load('held-report.json')
    if digest(report['model_freeze'])!=suite['model_freeze_sha256']:raise ValueError('Freeze identity mismatch')
    for attempt in order_data['transfer_slices']:
        tid=attempt['attempt_id'];d=load(tid+'.json');rows=d['rows'];c=d['capture']
        audit=audit_capture(c,[r for w in c['workers'] for r in w['rows']],forced_outputs=False)
        from prefill_lab.suite import transfer_offer_audit
        offers=transfer_offer_audit(rows,c,audit)
        outcome=load(tid+'-runner/outcome.json');gc=load(tid+'-runner/gc-audit.json')
        packet_path=capture/(tid+'-runner/input.json')
        if hashlib.sha256(packet_path.read_bytes()).hexdigest()!=outcome['input_sha256']:raise ValueError('Transfer input identity mismatch')
        packet=json.loads(packet_path.read_text())
        selected=[r for r in trace['requests'] if attempt['replay_start']<=r['arrival_seconds']<attempt['score_end']]
        if packet['attempt']!=attempt or packet['trace']!=dict(model=trace['model'],requests=selected):raise ValueError('Transfer does not match frozen trace slice')
        from prefill_lab.transfer_runner import audit_transfer_gc
        gc_recomputed=audit_transfer_gc(capture/(tid+'-runner'),load(tid+'-runner/gc-window.json'))
        gc_valid=gc['valid'] and gc_recomputed['valid'] and outcome['status']=='complete' and gc_recomputed['process_id']==outcome['process_id']
        report['transfers'].append(dict(attempt_id=tid,requests=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
            offer_audit=offers,worker_audit=audit,runner_gc_valid=gc_valid,runner_gc_recomputed=gc_recomputed,runner_gc_peak_rss_mib=gc['peak_rss_mib'],
            runner_gc_disabled_seconds=gc['disabled_seconds']))
    report['recomputed_collection_integrity']=all(t['work_valid'] and t['dispatcher_valid'] for t in report['trial_audits']) and all(t['offer_audit']['valid'] and t['runner_gc_valid'] for t in report['transfers'])
    manifest=[dict(path=str(p.relative_to(capture)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(capture.rglob('*')) if p.is_file()]
    save(output/'input-files.json',manifest)
    if archive:report['archive_sha256']=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    report['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(output/'analysis.json',report)
    print(json.dumps({k:report[k] for k in ('capture_commit','source_files_verified','recomputed_collection_integrity','training_wall_labels','deep_profiles','transfers')},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path)
    a=p.parse_args();analyze(a.capture,a.output,a.archive)


if __name__=='__main__':main()
