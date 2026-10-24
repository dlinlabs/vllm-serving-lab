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


@pytest.mark.parametrize('lag,valid',[(.006963420775718987,True),(.05,True),
    (.050001,False),(.20411953289294615,False)])
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
import asyncio,json,time
from pathlib import Path
import httpx
from prefill_lab import suite
from prefill_lab.contracts import save
from prefill_lab.transfer_runner import main

async def workload(client,http,url,attempt,trace,deadline,output):
    assert http.headers['authorization']=='Bearer test-secret'
    assert [r['trace_id'] for r in trace['requests']]==['a','b','c']
    start=time.monotonic()+.03
    save(output/'child-ready.json',{'at':start})
    rows=[]
    try:
        if trace['model']=='cancel':await asyncio.sleep(60)
        async def respond(request):
            return httpx.Response(200,text='data: {"choices":[{"text":"ok"}]}\\n\\ndata: [DONE]\\n\\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as local:
            tasks=[]
            for i,r in enumerate(trace['requests']):
                row={'trace_id':r['trace_id']};rows.append(row)
                tasks.append(asyncio.create_task(suite.completion(local,'http://mock',{},start+i*.02,row)))
            await asyncio.gather(*tasks)
        if trace['model']=='invalid':
            save(output/(attempt['attempt_id']+'.json'),{'rows':rows,'offered_valid':False})
            raise RuntimeError('synthetic offer lag204ms exceeds unchanged50ms gate')
        save(output/(attempt['attempt_id']+'.json'),dict(rows=rows,offered_valid=True,
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


def test_generator_continues_while_controller_loop_is_blocked(monkeypatch,tmp_path):
    call,processes=setup(monkeypatch,tmp_path);policy=(gc.isenabled(),gc.get_threshold())
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
    assert outcome['gc_policy']=='automatic_unchanged' and processes[0].returncode==0
    assert (gc.isenabled(),gc.get_threshold())==policy
    assert runner.KEY_ENV not in os.environ


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
