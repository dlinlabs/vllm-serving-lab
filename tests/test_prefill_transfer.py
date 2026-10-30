"""Real process isolation with synthetic HTTP; no GPU-performance claims."""
import asyncio
import gc
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace as NS

import pytest

from prefill_lab import transfer_runner as runner


def test_real_transfer_restores_before_capture_fetch_and_preserves_frozen_payload(monkeypatch,tmp_path):
    import httpx
    from prefill_lab import suite
    now=[100.];events=[];real_sleep=asyncio.sleep;payloads=[]
    class Guard:
        active=False
        async def prepare(self):self.active=True;events.append('prepare')
        async def supervise(self,work):
            try:return await work
            finally:self.active=False;events.append('restore')
    guard=Guard()
    async def sleep(delay):
        if delay>10:
            await real_sleep(.01)  # Let zero-offset HTTP tasks run, then skip the80s hold.
            now[0]+=delay
        else:await real_sleep(0)
    async def handler(request):
        if request.method=='GET':return httpx.Response(200,json={})
        assert guard.active
        payloads.append(json.loads(request.content))
        return httpx.Response(200,text='data: {"choices":[{"text":"ok"}]}\n\ndata: [DONE]\n\n')
    class Client:
        async def command(self,operation,**kwargs):
            assert not guard.active
            events.append(operation)
            return dict(workers=[],requests={str(i):{} for i in range(3)})
    monkeypatch.setattr(suite,'time',NS(monotonic=lambda:now[0]))
    monkeypatch.setattr(suite.asyncio,'sleep',sleep)
    monkeypatch.setattr(suite,'audit_capture',lambda *a,**kw:dict(valid=True))
    trace=dict(model='m',requests=[dict(arrival_seconds=40,trace_id=str(i),input_tokens=3,prompt='exact-'+str(i)) for i in range(3)])
    attempt=dict(attempt_id='t',replay_start=40,score_start=60,score_end=120)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await suite.transfer_local(Client(),http,'http://engine',attempt,trace,1000.,tmp_path,gc_control=guard)
    asyncio.run(scenario())
    assert events==['external_begin','prepare','restore','external_end']
    assert [p['messages'][0]['content'] for p in payloads]==['exact-0','exact-1','exact-2']
    assert all(p['max_tokens']==128 and 'ignore_eos' not in p and p['model']=='m' for p in payloads)
    saved=json.loads((tmp_path/'t.json').read_text())
    assert saved['offer_start']==100 and saved['score_start']==120 and saved['score_end']==180
    assert saved['drain_end']==180 and saved['offered_valid']


@pytest.mark.parametrize('lag,valid',[(.006963420775718987,True),(.05,True),
    (.050001,False),(.0543741552,False),(.20411953289294615,False)])
def test_reported_offer_lags_keep_the_original_fifty_ms_gate(lag,valid):
    from prefill_lab.suite import transfer_offer_audit
    rows=[dict(status='complete',scheduled=0.,sent=lag),dict(status='rejected',scheduled=0.,sent=0.)]
    capture=dict(requests={'r':{}})
    result=transfer_offer_audit(rows,capture,dict(valid=True))
    assert result['valid'] is valid and result['max_allowed_offer_lag_s']==.05
    assert not transfer_offer_audit(rows,capture,dict(valid=False))['valid']
    assert not transfer_offer_audit(rows,dict(requests={}),dict(valid=True))['valid']
    rows[1]['status']='failed'
    assert not transfer_offer_audit(rows,capture,dict(valid=True))['valid']


# Inject only the actual 80s/network workload. Exercise the real child worker,
# diagnostics, credential handoff, outcome and parent process lifecycle.
HARNESS='''
import asyncio,gc,json,os,sys,time
from pathlib import Path
import httpx
import runner_gc_control
from prefill_lab import suite
from prefill_lab.contracts import save
from prefill_lab.transfer_runner import main

# Unit-test seam only. Production keeps the native PID/procfs-guarded reader.
test_mode=json.loads(Path(sys.argv[sys.argv.index('--input')+1]).read_text())['trace']['model']
probe_count=0
def probe():
    global probe_count
    probe_count+=1
    if test_mode=='probe_fail' or (test_mode=='after_disable' and probe_count>=3):
        return dict(available=False)
    rss=2048*1048576 if test_mode=='rss' and probe_count>=4 else 1024
    return dict(available=True,pid=os.getpid(),process_created_unix=123.,
        sample_start=time.monotonic(),sample_end=time.monotonic(),rss_bytes=rss)
runner_gc_control.resources=probe

async def workload(client,http,url,attempt,trace,deadline,output,*,gc_control):
    await gc_control.prepare()
    assert not gc.isenabled()
    assert http.headers['authorization']=='Bearer test-secret'
    assert [r['trace_id'] for r in trace['requests']]==['a','b','c']
    start=time.monotonic()+.03
    save(output/'child-ready.json',{'at':start})
    rows=[];bounds={}
    async def traffic():
        if trace['model']=='cancel':await asyncio.sleep(60)
        async def respond(request):
            if trace['model']=='cycles':
                for _ in range(10000):
                    cycle=[];cycle.append(cycle)
            return httpx.Response(200,text='data: {"choices":[{"text":"ok"}]}\\n\\ndata: [DONE]\\n\\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as local:
            tasks=[]
            for i,r in enumerate(trace['requests']):
                row={'trace_id':r['trace_id']};rows.append(row)
                tasks.append(asyncio.create_task(suite.completion(local,'http://mock',{},start+i*.02,row)))
            await asyncio.gather(*tasks)
        bounds['drain_end']=time.monotonic()
    try:
        await gc_control.supervise(traffic())
        assert gc.isenabled()  # Capture postprocessing is outside the guarded window.
        if trace['model']=='postprocess':time.sleep(.65)
        if trace['model']=='invalid':
            save(output/(attempt['attempt_id']+'.json'),{'rows':rows,'offered_valid':False})
            raise RuntimeError('synthetic offer lag204ms exceeds unchanged50ms gate')
        save(output/(attempt['attempt_id']+'.json'),dict(rows=rows,offered_valid=True,
            score_start=start+20,score_end=start+.04,
            offer_start=start,drain_end=bounds['drain_end'],
            capture={'finished':time.monotonic(),'process':'child'}))
    except asyncio.CancelledError:
        save(output/'child-partial.json',{'rows':rows,'cancelled':True})
        raise

suite.transfer_local=workload
main()
'''


def setup(monkeypatch,tmp_path,mode='ok'):
    real_create=asyncio.create_subprocess_exec;processes=[]
    async def create(*args,**kwargs):
        assert args[:3]==(runner.sys.executable,'-m','prefill_lab.transfer_runner')
        assert 'test-secret' not in ' '.join(map(str,args))
        assert kwargs['env'][runner.KEY_ENV]=='Bearer test-secret'
        assert kwargs['cwd']==runner.ROOT
        process=await real_create(runner.sys.executable,'-c',HARNESS,*args[3:],**kwargs)
        processes.append(process);return process
    monkeypatch.setattr(runner.asyncio,'create_subprocess_exec',create)
    trace=dict(model=mode,requests=[dict(trace_id=tid,arrival_seconds=at,prompt='exact-'+tid,
        input_tokens=3) for tid,at in [('before',39),('a',40),('b',41),('c',42),('after',120)]])
    attempt=dict(attempt_id='transfer-a-r1',replay_start=40,score_start=60,score_end=120)
    http=NS(headers={'authorization':'Bearer test-secret'})
    return lambda:runner.isolated_transfer(http,'http://127.0.0.1:8000',attempt,trace,
        time.monotonic()+30,tmp_path),processes


async def wait_ready(output):
    until=time.monotonic()+15
    while not (output/'child-ready.json').exists():
        if time.monotonic()>until:raise TimeoutError('Synthetic child did not become ready')
        await asyncio.sleep(.01)


@pytest.mark.parametrize('mode',['ok','cycles'])
def test_generator_continues_while_controller_loop_is_blocked(monkeypatch,tmp_path,mode):
    call,processes=setup(monkeypatch,tmp_path,mode);policy=(gc.isenabled(),gc.get_threshold())
    async def scenario():
        task=asyncio.create_task(call())
        try:
            await wait_ready(tmp_path)
            time.sleep(.4)  # Intentional controller-only stall, longer than50ms.
            unblocked=time.monotonic()
            result=await task
            assert result['finished']<unblocked
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    asyncio.run(scenario())
    directory=tmp_path/'transfer-a-r1-runner'
    outcome=json.loads((directory/'outcome.json').read_text())
    health=json.loads((directory/'runner_observations_status.json').read_text())
    assert outcome['status']=='complete' and outcome['process_id']!=os.getpid()
    assert health['process_id']==outcome['process_id'] and health['health']['complete']
    assert outcome['gc_policy']=='bounded_runner_defer' and processes[0].returncode==0
    control=json.loads((directory/'runner_gc_control.json').read_text())
    assert control['completed'] and control['restored_enabled'] and control['initial_enabled']
    verdict=json.loads((directory/'gc-audit.json').read_text())
    assert verdict['valid'] and verdict['runner_diagnostics']['valid']
    assert (gc.isenabled(),gc.get_threshold())==policy
    assert runner.KEY_ENV not in os.environ


@pytest.mark.parametrize('mode',['probe_fail','after_disable','rss'])
def test_native_resource_failures_reject_transfer_and_restore_gc(monkeypatch,tmp_path,mode):
    call,processes=setup(monkeypatch,tmp_path,mode)
    with pytest.raises(RuntimeError,match='RSS'):asyncio.run(call())
    assert len(processes)==1 and processes[0].returncode!=0
    directory=tmp_path/'transfer-a-r1-runner'
    control=json.loads((directory/'runner_gc_control.json').read_text())
    assert not control['completed'] and control['restored_enabled'] and control['failure']
    assert control['initial_thresholds']==control['restored_thresholds']
    if mode=='probe_fail':
        assert control['disabled_at'] is None and not (tmp_path/'child-ready.json').exists()
    elif mode=='after_disable':
        assert control['disabled_at'] is not None and not (tmp_path/'child-ready.json').exists()
    else:
        assert (tmp_path/'child-ready.json').exists()
        assert not (directory/'gc-audit.json').exists()  # Offered fidelity cannot override guard failure.


def test_slow_capture_postprocessing_runs_after_gc_restore(monkeypatch,tmp_path):
    call,_=setup(monkeypatch,tmp_path,'postprocess');asyncio.run(call())
    directory=tmp_path/'transfer-a-r1-runner'
    verdict=json.loads((directory/'gc-audit.json').read_text())
    data=json.loads((tmp_path/'transfer-a-r1.json').read_text())
    control=json.loads((directory/'runner_gc_control.json').read_text())
    assert verdict['valid'] and data['capture']['finished']-control['restore_started_at']>=.65


def test_transfer_gc_evidence_cannot_be_upgraded_after_corruption(monkeypatch,tmp_path):
    call,_=setup(monkeypatch,tmp_path);asyncio.run(call())
    directory=tmp_path/'transfer-a-r1-runner';path=directory/'runner_gc_control.json'
    original=path.read_text();manifest=json.loads((directory/'gc-window.json').read_text())
    assert runner.audit_transfer_gc(directory,manifest)['valid']
    for damage in ('restoration','rss','policy','pid','spec','completed','coverage'):
        data=json.loads(original)
        if damage=='restoration':data['restored_enabled']=False
        elif damage=='rss':data['samples'][-1]['rss_bytes']=runner.GC_SPEC['max_rss_bytes']+1
        elif damage=='policy':data['samples'][-1]['gc_enabled']=True
        elif damage=='pid':data['process_id']+=1
        elif damage=='spec':data['spec']['max_disabled_seconds']+=1
        elif damage=='completed':data['completed']=False
        elif damage=='coverage':data['restore_started_at']=manifest['measurement_start']
        path.write_text(json.dumps(data))
        assert not runner.audit_transfer_gc(directory,manifest)['valid'],damage
    path.write_text(original)
    observations=directory/'runner_observations.jsonl'
    with observations.open('a') as stream:
        stream.write(json.dumps(dict(event='gc_span',span_start=manifest['measurement_start'],
            span_end=manifest['measurement_end']))+'\n')
    assert not runner.audit_transfer_gc(directory,manifest)['valid']


def test_failed_fidelity_is_retained_without_retry(monkeypatch,tmp_path):
    call,processes=setup(monkeypatch,tmp_path,'invalid')
    with pytest.raises(RuntimeError,match='unchanged50ms gate'):asyncio.run(call())
    assert len(processes)==1 and processes[0].returncode!=0
    assert not json.loads((tmp_path/'transfer-a-r1.json').read_text())['offered_valid']
    outcome=json.loads((tmp_path/'transfer-a-r1-runner/outcome.json').read_text())
    assert outcome['status']=='failed' and outcome['error_type']=='RuntimeError'


def test_controller_cancellation_stops_owned_generator_and_keeps_partial(monkeypatch,tmp_path):
    call,processes=setup(monkeypatch,tmp_path,'cancel')
    async def scenario():
        task=asyncio.create_task(call())
        await wait_ready(tmp_path);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(scenario())
    assert len(processes)==1 and processes[0].returncode is not None
    assert json.loads((tmp_path/'child-partial.json').read_text())['cancelled']
    outcome=json.loads((tmp_path/'transfer-a-r1-runner/outcome.json').read_text())
    assert outcome['status']=='failed' and outcome['error_type']=='CancelledError'
    control=json.loads((tmp_path/'transfer-a-r1-runner/runner_gc_control.json').read_text())
    assert control['restored_enabled'] and not control['completed']
    assert control['initial_thresholds']==control['restored_thresholds']
