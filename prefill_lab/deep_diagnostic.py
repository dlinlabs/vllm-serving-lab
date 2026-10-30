"""Targeted foreground profiling; never a substitute for a complete suite/model gate."""
import hashlib
import json
from pathlib import Path
import time

from .contracts import audit_capture,digest,save
from .causal import dispatcher_parity


def profile_audit(capture,directory):
    issues=[];workers=capture.get('workers',[])
    result=dict(valid=False,issues=issues,cuda_scope_qualified=False,model_accepted=False)
    if len(workers)!=1:return dict(result,issues=['Expected one profiling worker'])
    worker=workers[0];proof=worker.get('deep_profile',{});rows=worker['rows']
    roles={k:v['role'] for k,v in capture['requests'].items()}
    foreground=lambda row:any(roles.get(rid) in ('foreground','sentinel') for rid in row['counts'])
    first=next((i for i,row in enumerate(rows) if foreground(row)),None)
    if proof.get('schema')!='foreground-profile-v1' or proof.get('state')!='complete':issues.append('Missing/incomplete foreground profile')
    expected=[] if first is None else [r['step_id'] for r in rows[first:first+5]]
    if len(expected)!=5 or proof.get('step_ids')!=expected:issues.append('Profile does not cover first five foreground-window steps')
    try:
        artifact=proof['artifact'];name=artifact['name']
        if Path(name).name!=name or not name.startswith(capture['trial_id']+'-worker-'):raise ValueError('Profile artifact name mismatch')
        raw=(Path(directory)/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=artifact['sha256']:raise ValueError('Profile artifact hash mismatch')
        events=json.loads(raw)['traceEvents']
        names=[e['name'] for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('prefill_lab_step/')]
        if sorted(names)!=sorted('prefill_lab_step/'+key for key in expected):issues.append('Trace lacks exact per-step CPU markers')
        if not any(e.get('cat')=='kernel' for e in events):issues.append('Trace lacks GPU kernels')
        result['trace_sha256']=artifact['sha256']
    except (OSError,ValueError,KeyError,TypeError):issues.append('Missing/invalid profile artifact')
    result.update(valid=not issues,first_foreground_worker_step=None if first is None else first+1,step_ids=expected,
                  interpretation='Foreground window/marker integrity only; stream/event coverage and async model still require analysis')
    return result


async def run(client,http,url,manifest,output,*,stages=False):
    from .suite import inputs,runtime_preflight,host_identity,controlled_trial,metrics_sample
    cases,order,_=inputs();output=Path(output);started=time.monotonic();deadline=started+180
    meta=await client.command('meta');preflight=runtime_preflight(meta,manifest)
    if stages:
        from .pipeline import preflight as stage_preflight
        stage_preflight(meta)
    save(output/'runtime.json',meta);save(output/'host.json',host_identity())
    deep=[t for t in order if t['stage']=='deep_diagnostic']
    warm=[next(t for t in order if t['stage']=='warmup' and t['case']==d['case']) for d in deep]
    if stages:
        protocol=json.loads((Path(__file__).resolve().parents[1]/'experiments/prefill-stage-diagnostic-v1.json').read_text())
        if protocol['planned_trial_ids']!=[t['trial_id'] for t in warm+deep]:
            raise ValueError('Stage diagnostic does not match its frozen development order')
        save(output/'stage-protocol.json',protocol)
    summary=dict(scope='targeted_stage_diagnostic_only' if stages else 'targeted_foreground_diagnostic_only',diagnostic_complete=False,complete_experiment_passed=False,
        model_accepted=False,gpu_validated=False,cuda_scope_qualified=False,preflight=preflight,
        planned=[t['trial_id'] for t in warm+deep],completed=[],profiles=[],failures=[])
    if stages:summary['pipeline_audits']=[]
    save(output/'diagnostic.json',summary)
    try:
        for trial in warm+deep:
            if time.monotonic()>=deadline:raise TimeoutError('Targeted diagnostic180s deadline expired')
            tid=trial['trial_id'];print('Diagnostic '+tid+': '+trial['stage']+' '+trial['case'],flush=True)
            save(output/(tid+'-metrics-before.json'),await metrics_sample(http,url))
            capture=await controlled_trial(client,trial,deadline,output/(tid+'.json'))
            capture['end']=max(r['finished'] for r in capture['requests'].values());save(output/(tid+'.json'),capture)
            save(output/(tid+'-metrics-after.json'),await metrics_sample(http,url))
            audit=audit_capture(capture,[r for w in capture['workers'] for r in w['rows']])
            audit['dispatcher']=dispatcher_parity(capture,meta['workers'][0]['dispatch_table'])
            if trial['stage']=='deep_diagnostic':
                audit['profile']=profile_audit(capture,output/'deep-traces');summary['profiles'].append(dict(trial_id=tid,**audit['profile']))
            if stages:
                from .pipeline import audit as stage_audit
                audit['pipeline']=stage_audit(capture,output/'deep-traces')
                summary['pipeline_audits'].append(dict(trial_id=tid,**audit['pipeline']))
            save(output/(tid+'-audit.json'),audit)
            if not audit['valid'] or not audit['dispatcher']['matches'] or not audit.get('profile',{'valid':True})['valid'] or not audit.get('pipeline',{'valid':True})['valid']:
                raise RuntimeError('Diagnostic work/dispatcher/profile integrity failed: '+tid)
            summary['completed'].append(tid);save(output/'diagnostic.json',summary)
        final=await client.command('meta');save(output/'runtime-final.json',final)
        if digest(meta)!=digest(final):raise RuntimeError('Diagnostic backend identity changed')
        if time.monotonic()>deadline:raise TimeoutError('Targeted diagnostic180s deadline expired')
        summary['diagnostic_complete']=True
    except BaseException as exc:
        summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        summary['elapsed_seconds']=time.monotonic()-started;save(output/'diagnostic.json',summary)
    return summary
