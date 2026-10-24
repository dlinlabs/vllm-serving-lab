"""One loaded OpenAI-server engine, immutable inputs, bounded one-attempt stages."""
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import signal
import subprocess
import sys
import time

import httpx

from .analysis import observer_gate,freeze_models,drift_report
from .causal import dispatcher_parity
from .scoring import freeze_eligibility
from .contracts import audit_capture,digest,save
from .tokens import peak_kv_tokens,validate

ROOT=Path(__file__).resolve().parents[1]


def inputs():
    load=lambda name:json.loads((ROOT/f'experiments/prefill-service-v1.{name}.json').read_text())
    return load('cases')['cases'],load('order')['order'],load('order')['transfer_slices']


def runtime_preflight(meta,manifest):
    expected=dict(max_model_len=8192,max_num_batched_tokens=2048,tensor_parallel_size=1,
                  pipeline_parallel_size=1,data_parallel_size=1,prefix_cache=False,speculative=False,
                  enforce_eager=False,enable_chunked_prefill=True,kv_transfer=False)
    problems=[f'{k}: observed {meta.get(k)!r}, expected {v!r}' for k,v in expected.items() if meta.get(k)!=v]
    if meta.get('max_num_seqs',0)<21:problems.append('Need21 active sequences')
    if meta.get('max_num_scheduled_tokens') not in (None,2048):problems.append('Unexpected scheduler input budget')
    if meta.get('scheduler_policy')!='fcfs' or meta.get('long_prefill_token_threshold')!=0:
        problems.append('Unexpected scheduling policy/chunk threshold')
    if meta.get('revision')!=manifest['revision'] or meta.get('tokenizer_revision')!=manifest['revision']:
        problems.append('Model/tokenizer revision mismatch')
    if meta.get('token_manifest_sha256')!=digest(manifest):problems.append('Engine token manifest mismatch')
    block=meta.get('block_size');blocks=meta.get('num_gpu_blocks')
    if type(block) is not int or type(blocks) is not int or block<=0 or blocks<=0:
        problems.append('Unavailable resolved KV capacity')
    else:
        peak=max(peak_kv_tokens(rows,block) for rows in manifest['trials'].values())
        if peak> .9*block*blocks:problems.append('Less than10% block-rounded KV headroom')
    workers=meta.get('workers',[])
    if len(workers)!=1 or workers[0].get('uniform_decode_query_len')!=1:
        problems.append('Expected one non-speculative reviewed worker')
    if workers and (workers[0].get('torch_version','').split('+')[0]!='2.13.0' or workers[0].get('cuda_version')!='13.0'):
        problems.append('Expected Torch2.13.0/CUDA13.0 runtime')
    if workers and '4090' not in workers[0].get('device_name',''):problems.append('Plan requires RTX4090')
    if problems:raise ValueError('; '.join(problems))
    return dict(valid=True,peak_kv_tokens=peak,capacity_tokens=block*blocks,
                identity_sha256=digest(meta),gpu_validated=False)


def host_identity():
    import importlib.metadata
    result=dict(python=sys.version,platform=platform.platform(),pid=os.getpid(),
                affinity=sorted(os.sched_getaffinity(0)),packages={n:importlib.metadata.version(n) for n in ('vllm','torch','numpy','scipy','httpx')})
    for path in ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max'):
        result[path]=Path(path).read_text().strip() if Path(path).exists() else None
    query=['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit,temperature.gpu','--format=csv,noheader,nounits']
    result['nvidia_smi']=subprocess.check_output(query,text=True,timeout=10).strip()
    return result


def stage_book(order,transfers):
    result={}
    for stage in ('warmup','observer_check','main','drift_reference','http_bridge','transfer','deep_diagnostic'):
        ids=[t['trial_id'] for t in order if t['stage']==stage] if stage!='transfer' else [t['attempt_id'] for t in transfers]
        result[stage]=dict(planned_trial_ids=ids,attempted_trial_ids=[],completed_trial_ids=[],
                           integrity='not_started',deadline=False,failure_reason=None)
    return result


class Client:
    def __init__(self,http,url):self.http,self.url=http,url
    async def command(self,operation,**payload):
        response=await self.http.post(self.url+'/prefill-lab/command',json=dict(operation=operation,**payload))
        response.raise_for_status();return response.json()


async def metrics_sample(http,url):
    sent=time.monotonic();body=bytearray()
    async with http.stream('GET',url+'/metrics') as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if len(body)+len(chunk)>1024*1024 or time.monotonic()-sent>10:
                raise RuntimeError('Exporter body size/time bound exceeded')
            body.extend(chunk)
    return dict(sent=sent,received=time.monotonic(),raw_metrics=body.decode())


async def controlled_trial(client,trial,deadline,partial_path=None):
    limit=min(deadline,time.monotonic()+45)
    await client.command('start',trial_id=trial['trial_id'])
    try:
        while True:
            status=await client.command('status')
            if status['status']=='complete':break
            if status['status']=='invalid':raise RuntimeError('Controlled trial invalidated; see partial capture')
            if time.monotonic()>=limit:raise TimeoutError('Trial45s/shared deadline expired')
            await asyncio.sleep(.01)
        while True:
            capture=await client.command('finish')
            if capture.get('status')!='gpu_events_pending':return capture
            if time.monotonic()>=limit:raise TimeoutError('Unresolved delayed CUDA events')
            await asyncio.sleep(.01)
    except BaseException:
        try:
            await client.command('abort',reason='Controlled trial interrupted or timed out')
            until=time.monotonic()+2
            while time.monotonic()<until:
                try:
                    partial=await client.command('finish')
                    if partial.get('status')!='gpu_events_pending':
                        if partial_path:save(partial_path,partial)
                        break
                except httpx.HTTPStatusError:pass
                await asyncio.sleep(.01)
        except Exception:pass
        raise


async def completion(http,url,payload,scheduled,row,ready=None):
    await asyncio.sleep(max(0,scheduled-time.monotonic()))
    row.update(scheduled=scheduled,sent=time.monotonic(),first=None,finished=None,status='attempted')
    try:
        async with http.stream('POST',url,json=payload) as response:
            row['http_status']=response.status_code
            row['gateway_request_id']=response.headers.get('x-request-id')
            if response.status_code!=200:
                row['status']='rejected' if response.status_code==503 else 'failed'
                return
            async for line in response.aiter_lines():
                if not line.startswith('data: '):continue
                value=line[6:]
                if value=='[DONE]':row['stream_complete']=True;continue
                item=json.loads(value);row['response_id']=item.get('id')
                choices=item.get('choices',[])
                if any(c.get('text') or c.get('delta',{}).get('content') for c in choices):
                    if row['first'] is None:
                        row['first']=time.monotonic()
                        if ready:ready.set()
                if item.get('usage'):row['usage']=item['usage']
            row['status']='complete' if row.get('stream_complete') else 'failed'
    except asyncio.CancelledError:
        row['status']='cancelled';raise
    except (httpx.HTTPError,ValueError) as exc:
        row.update(status='failed',error=type(exc).__name__)
    finally:row['finished']=time.monotonic()


async def http_bridge(client,http,url,trial,manifest,deadline,partial_path=None):
    tid=trial['trial_id'];await client.command('external_begin',trial_id=tid,observer='light')
    rows=[];tasks=[];ready=[]
    def launch(r,at,event=None):
        payload=dict(model=manifest['model'],prompt=manifest['pools'][r['token_key']]['input_ids'],
                     max_tokens=r['max_tokens'],temperature=0,ignore_eos=True,stream=True,
                     stream_options=dict(include_usage=True),request_id=f'plab-http-{tid}-{r["id"]}',
                     vllm_xargs=dict(prefill_lab_role=r['role']))
        row=dict(id=r['id'],status='not_started');rows.append(row)
        task=asyncio.create_task(completion(http,url+'/v1/completions',payload,at,row,event));tasks.append(task)
    try:
        for r in manifest['trials'][tid]:
            if r['role']=='background':
                event=asyncio.Event();ready.append(event);launch(r,time.monotonic(),event)
        if ready:await asyncio.wait_for(asyncio.gather(*(e.wait() for e in ready)),min(45,max(.001,deadline-time.monotonic())))
        release=time.monotonic()
        for r in manifest['trials'][tid]:
            if r['role']!='background':launch(r,release+r['at'])
        await asyncio.wait_for(asyncio.gather(*tasks),min(45,max(.001,deadline-time.monotonic())))
        if any(r['status']!='complete' for r in rows):raise RuntimeError('HTTP bridge request failure')
        capture=await finish_external(client,deadline)
        return dict(capture=capture,client=rows,release=release,diagnostic_only=True,
                    transport_path='shared HTTP client -> OpenAI completions -> EngineCore; gateway bypassed',gateway_transport_qualified=False)
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if partial_path:save(partial_path,dict(client=rows,diagnostic_only=True))


async def finish_external(client,deadline):
    while time.monotonic()<deadline:
        result=await client.command('external_end')
        if result.get('status')!='gpu_events_pending':return result
        await asyncio.sleep(.01)
    raise TimeoutError('External stage drain/CUDA deadline expired')


async def transfer(client,http,url,attempt,trace,deadline,output):
    from .transfer_runner import isolated_transfer
    return await isolated_transfer(http,url,attempt,trace,deadline,output)


def transfer_offer_audit(rows,capture,native_audit):
    failures=[r for r in rows if r['status'] not in ('complete','rejected')]
    if not native_audit['valid']:failures.append(dict(error='Transfer work/worker integrity failed'))
    maximum=max((r['sent']-r['scheduled'] for r in rows),default=0)
    completed=sum(r['status']=='complete' for r in rows)
    if len(capture['requests'])!=completed:
        failures.append(dict(error='External/unreconciled engine requests during transfer'))
    return dict(valid=not failures and maximum<=.05,max_offer_lag_s=maximum,
                max_allowed_offer_lag_s=.05,completed_requests=completed,
                engine_requests=len(capture['requests']),failures=failures)


async def transfer_local(client,http,url,attempt,trace,deadline,output):
    tid=attempt['attempt_id'];await client.command('external_begin',trial_id=tid,observer='light')
    start=time.monotonic();rows=[];tasks=[];observations=[];stop=asyncio.Event()
    async def sample():
        while not stop.is_set():
            sent=time.monotonic()
            try:
                a=await http.get(url+'/metrics');b=await http.get('http://127.0.0.1:8080/metrics')
                a.raise_for_status();b.raise_for_status()
                observations.append(dict(sent=sent,received=time.monotonic(),raw_metrics=a.text,gateway=b.json()))
            except httpx.HTTPError as exc:observations.append(dict(sent=sent,error=type(exc).__name__))
            try:await asyncio.wait_for(stop.wait(),1.)
            except asyncio.TimeoutError:pass
    sampler=asyncio.create_task(sample())
    try:
        for request in trace['requests']:
            at=request['arrival_seconds']
            if not attempt['replay_start']<=at<attempt['score_end']:continue
            scheduled=start+at-attempt['replay_start']
            await asyncio.sleep(max(0,scheduled-time.monotonic()))
            if time.monotonic()>deadline or len([t for t in tasks if not t.done()])>=1000:
                raise TimeoutError('Transfer offer deadline/pending bound exceeded')
            row=dict(trace_id=request['trace_id'],source_input_tokens=request['input_tokens'],
                     scored=attempt['score_start']<=at<attempt['score_end'],status='not_started')
            rows.append(row)
            payload=dict(model=trace['model'],messages=[dict(role='user',content=request['prompt'])],
                         max_tokens=128,temperature=0,stream=True,stream_options=dict(include_usage=True))
            tasks.append(asyncio.create_task(completion(http,'http://127.0.0.1:8080/v1/chat/completions',payload,scheduled,row)))
        await asyncio.sleep(max(0,start+80-time.monotonic()))
        await asyncio.wait_for(asyncio.gather(*tasks),min(10,max(.001,deadline-time.monotonic())))
        result=await finish_external(client,min(deadline,start+90))
        native_audit=audit_capture(result,[r for w in result['workers'] for r in w['rows']],forced_outputs=False)
        offers=transfer_offer_audit(rows,result,native_audit)
        save(output/f'{tid}.json',dict(attempt=attempt,rows=rows,observations=observations,capture=result,
                                     offered_valid=offers['valid'],offer_audit=offers,work_audit=native_audit,
                                     output_policy='cap128, natural EOS; source GeneratedTokens not used',
                                     lead_in_seconds=20,score_start=start+20,score_end=start+80,
                                     boundary_state_note='Raw gateway/engine snapshots straddle score boundaries; no empty-boundary assumption'))
        if not offers['valid']:
            raise RuntimeError(f'Transfer integrity/load fidelity failed: max offer lag {1000*offers["max_offer_lag_s"]:.3f}ms (limit50ms); integrity failures {len(offers["failures"])}')
        return result
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        stop.set();sampler.cancel()
        await asyncio.gather(sampler,return_exceptions=True)
        # Preserve partial offers even on timeout/cancellation.
        if not (output/f'{tid}.json').exists():save(output/f'{tid}.json',dict(attempt=attempt,rows=rows,observations=observations,incomplete=True))


async def run(client,http,url,manifest,trace,output,*,hard_seconds=1800):
    cases,order,transfers=inputs();validate(manifest,cases,order)
    meta=await client.command('meta');preflight=runtime_preflight(meta,manifest)
    save(output/'runtime.json',meta);save(output/'host.json',host_identity())
    stages=stage_book(order,transfers);captures=[];started=time.monotonic();deadline=started+hard_seconds
    summary=dict(schema='prefill-suite-v1',stages=stages,start=started,deadline=deadline,preflight=preflight,
                 complete=False,model_accepted=False,gpu_validated=False,failures=[])
    dispatch=meta['workers'][0]['dispatch_table']
    def record():save(output/'suite.json',summary)
    record();current=None
    async def trial(t0):
        nonlocal current
        current=stages[t0['stage']];tid=t0['trial_id']
        if time.monotonic()>=deadline:raise TimeoutError('Hard1800s active deadline expired')
        current['attempted_trial_ids'].append(tid);current['integrity']='running';record()
        if http is not None:save(output/f'{tid}-metrics-before.json',await metrics_sample(http,url))
        if t0['stage']=='http_bridge':
            bridge=await http_bridge(client,http,url,t0,manifest,deadline,output/f'{tid}-http.json');capture=bridge['capture']
            save(output/f'{tid}-http.json',bridge)
        else:capture=await controlled_trial(client,t0,deadline,output/f'{tid}.json')
        if http is not None:save(output/f'{tid}-metrics-after.json',await metrics_sample(http,url))
        if t0['stage']=='drift_reference':save(output/f'{tid}-host.json',host_identity())
        capture['end']=max(r['finished'] for r in capture['requests'].values())
        save(output/f'{tid}.json',capture);captures.append(capture)
        worker=[r for w in capture['workers'] for r in w['rows']]
        integrity=audit_capture(capture,worker,require_steps=t0['observer']!='off')
        if t0['observer']!='off':integrity['dispatcher']=dispatcher_parity(capture,dispatch)
        save(output/f'{tid}-audit.json',integrity)
        if not integrity['valid'] or not integrity.get('dispatcher',{}).get('matches',True):
            raise RuntimeError('Capture work/parity integrity failed; see audit')
        current['completed_trial_ids'].append(tid)
        if current['completed_trial_ids']==current['planned_trial_ids']:current['integrity']='complete_valid'
        await asyncio.sleep(.25)
        if time.monotonic()>deadline:raise TimeoutError('Active budget exceeded including drain/idle')
        record()
    try:
        # Collection order preserves the frozen observer/main/deep structure exactly.
        for t0 in order:
            if t0['stage'] in ('http_bridge','deep_diagnostic'):continue
            if t0['stage'] in ('main','drift_reference') and stages['observer_check']['integrity']=='complete_valid' and 'observer_gate' not in summary:
                summary['observer_gate']=observer_gate(captures,order);record()
                if summary['observer_gate']['status']!='passed':raise RuntimeError('Observer failed before main')
            await trial(t0)
            if t0['stage']=='drift_reference' and t0.get('block')==3 and t0.get('position')=='end':
                try:
                    frozen=freeze_eligibility(freeze_models(captures,order),cases,order,captures,dispatch)
                except Exception as exc:
                    # Freeze is analysis, not collection. Missing archived inputs,
                    # imports or unexpected analysis errors must preserve blocks4–5.
                    # Cancellation/KeyboardInterrupt remain outside this boundary.
                    frozen=dict(status='inconclusive',reason=str(exc),error_type=type(exc).__name__,
                                failed_stage='model_freeze',model_accepted=False,held_labels_used=False)
                save(output/'model-freeze.json',frozen);summary['model_freeze_sha256']=digest(frozen)
                summary['calibration_status']=frozen.get('noise_readiness',{}).get('status','inconclusive');record()
        summary['drift']=drift_report(captures,order);record()
        for t0 in order:
            if t0['stage']=='http_bridge':await trial(t0)
        current=stages['transfer']
        for attempt in transfers:
            if time.monotonic()+90>deadline:raise TimeoutError('Insufficient frozen transfer/drain budget')
            current['attempted_trial_ids'].append(attempt['attempt_id']);current['integrity']='running';record()
            await transfer(client,http,url,attempt,trace,deadline,output)
            current['completed_trial_ids'].append(attempt['attempt_id']);record()
        current['integrity']='complete_valid'
        for t0 in order:
            if t0['stage']=='deep_diagnostic':await trial(t0)
        final_meta=await client.command('meta')
        save(output/'runtime-final.json',final_meta)
        if digest(final_meta)!=digest(meta):raise RuntimeError('Backend/source/config identity changed during suite')
        summary['complete']=all(s['integrity']=='complete_valid' for s in stages.values())
        # Physical model/MPC acceptance requires additional measured gate evidence.
        summary['structural_status']='inconclusive'
        summary['block_dependence_status']='unresolved'
    except (Exception,asyncio.CancelledError) as exc:
        summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        # Interleaved stages (main/drift) can both have started. Close every
        # unfinished running stage, while preserving all completed evidence.
        for stage in stages.values():
            if stage['integrity']!='running' and stage is not current:continue
            stage['integrity']='incomplete' if isinstance(exc,(TimeoutError,asyncio.CancelledError)) or stage['completed_trial_ids']!=stage['planned_trial_ids'] else 'complete_invalid'
            stage['failure_reason']=str(exc);stage['deadline']=isinstance(exc,TimeoutError)
        try:
            await client.command('abort',reason=type(exc).__name__)
            until=time.monotonic()+2
            while time.monotonic()<until:
                try:
                    partial=await client.command('finish')
                    if partial.get('status')!='gpu_events_pending':
                        save(output/'partial-lab-capture.json',partial);break
                except httpx.HTTPError:pass
                await asyncio.sleep(.01)
        except Exception:pass
    finally:
        summary['elapsed_seconds']=time.monotonic()-started
        if summary['elapsed_seconds']>hard_seconds:summary['complete']=False
        record()
    return summary
