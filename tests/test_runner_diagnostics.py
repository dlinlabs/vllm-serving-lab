import asyncio
import gc
import json
import os
from pathlib import Path
import time
import uuid

import httpx
import pytest

from runner_diagnostics import RunnerDiagnostics, audit_runner, FILES, SCHEMA, LOOP_INTERVAL
from run_trace_pilot import replay
from test_trace_replay import trace, sse


def manifest(run_id, start, end):
    return dict(run_id=run_id, runner_process_id=os.getpid(), measurement_start=start,
        measurement_end=end, config=dict(runner_diagnostics=True),
        runner_diagnostics=dict(schema_version=SCHEMA, enabled=True, process_id=os.getpid(),
                                interval_seconds=LOOP_INTERVAL),
        code_sha256={'runner_diagnostics.py': 'a'*64})


def capture(tmp_path, *, stalled=False):
    async def scenario():
        run_id=uuid.uuid4().hex
        observer=RunnerDiagnostics(tmp_path,run_id)
        original_policy=(gc.isenabled(),gc.get_threshold())
        await observer.start()
        start=time.monotonic()
        await asyncio.sleep(.025)
        if stalled:
            # The diagnostic callback observes a genuine GC cycle while a later
            # test callback deliberately occupies the loop thread inside that cycle.
            def slow_gc(phase, info):
                if phase=='start' and info['generation']==2:
                    until=time.monotonic()+.07
                    while time.monotonic()<until:
                        pass
            gc.callbacks.append(slow_gc)
            try:
                gc.collect(2)
            finally:
                gc.callbacks.remove(slow_gc)
        await asyncio.sleep(.025)
        end=time.monotonic()
        await observer.close()
        assert observer.diagnostics.callback not in gc.callbacks
        assert (gc.isenabled(),gc.get_threshold())==original_policy
        assert all(t.done() for t in observer.tasks)
        await observer.close()  # Idempotent; no duplicate shutdown records.
        return manifest(run_id,start,end),observer
    return asyncio.run(scenario())


def test_gc_loop_cpu_identity_and_policy_preservation(tmp_path):
    m,_=capture(tmp_path,stalled=True)
    verdict=audit_runner(tmp_path,m)
    assert verdict['valid'],verdict
    assert verdict['gc_spans']>=1 and verdict['max_loop_lag']>.05
    rows=[json.loads(s) for s in (tmp_path/FILES[0]).read_text().splitlines()]
    span=next(r for r in rows if r['event']=='gc_span' and r['generation']==2)
    assert span['on_loop_thread'] and span['span_end']-span['span_start']>=.07
    assert 0<=span['callback_thread_cpu_seconds']<=span['span_end']-span['span_start']+.001
    delayed=[r for r in rows if r['event']=='event_loop_lag' and r['lag_seconds']>.05]
    assert any(min(r['resumed_at'],span['span_end'])>max(r['scheduled_at'],span['span_start']) for r in delayed)


def test_mock_replay_retains_all_offered_and_exposes_loop_block(tmp_path):
    async def scenario():
        run_id=uuid.uuid4().hex
        observer=RunnerDiagnostics(tmp_path,run_id)
        await observer.start()
        start=time.monotonic()+.02
        data=trace((0,.05,.051,.15),.22)
        rows=[]
        async def handler(request):
            await asyncio.sleep(.015)
            return sse(request.headers['x-benchmark-request-id'])
        async def block():
            await asyncio.sleep(.05)
            time.sleep(.09)  # Synthetic loop blockage; not a GPU performance test.
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False) as client:
                await asyncio.gather(replay(client,data,start,run_id,1,20,rows),block())
        finally:
            await observer.close()
        assert len(rows)==4 and all(r['success'] for r in rows)
        assert max(r['started_at']-r['scheduled_at'] for r in rows)>.05
        verdict=audit_runner(tmp_path,manifest(run_id,start,start+.22),rows)
        assert verdict['valid'] and verdict['max_loop_lag']>.05,verdict
    asyncio.run(scenario())


def test_cancelled_replay_can_close_without_callback_or_task_leak(tmp_path):
    async def scenario():
        run_id=uuid.uuid4().hex
        observer=RunnerDiagnostics(tmp_path,run_id)
        await observer.start()
        start=time.monotonic()
        async def handler(request):
            await asyncio.sleep(10)
            return sse(request.headers['x-benchmark-request-id'])
        rows=[]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task=asyncio.create_task(replay(client,trace((0,),.1),start,run_id,1,10,rows))
            await asyncio.sleep(.03)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await observer.close()
        assert observer.diagnostics.callback not in gc.callbacks
        assert all(t.done() for t in observer.tasks)
        assert json.loads((tmp_path/FILES[1]).read_text())['health']['complete']
    asyncio.run(scenario())


@pytest.mark.parametrize('damage',[
    'missing_log','missing_status','malformed_log','malformed_status','foreign_pid','foreign_run',
    'disabled','missing_declaration','missing_fingerprint','missing_monitor','bad_monitor_time',
    'missing_cpu','short_coverage','lost','write_error','worker_error','count_mismatch',
    'missing_boundary','bad_gc_sequence','bad_gc_time','missing_callback','loop_stopped_early',
    'malformed_config','cpu_coverage'])
def test_damage_fails_closed_without_relabelling_load(tmp_path,damage):
    m,_=capture(tmp_path,stalled=damage.startswith('bad_gc'))
    log,status=tmp_path/FILES[0],tmp_path/FILES[1]
    rows=[json.loads(s) for s in log.read_text().splitlines()];state=json.loads(status.read_text())
    if damage=='missing_log': log.unlink()
    elif damage=='missing_status': status.unlink()
    elif damage=='malformed_log': log.write_text('{bad json')
    elif damage=='malformed_status': status.write_text('[]')
    elif damage=='foreign_pid': rows[0]['process_id']+=1
    elif damage=='foreign_run': rows[0]['run_id']='foreign'
    elif damage=='disabled': m['config']['runner_diagnostics']=False
    elif damage=='missing_declaration': del m['runner_diagnostics']
    elif damage=='missing_fingerprint': m['code_sha256'].clear()
    elif damage=='missing_monitor': rows=[r for r in rows if r['event']!='event_loop_lag']
    elif damage=='bad_monitor_time': next(r for r in rows if r['event']=='event_loop_lag')['lag_seconds']=-1
    elif damage=='missing_cpu': rows=[r for r in rows if r['event']!='diagnostic_cpu']
    elif damage=='short_coverage': m['measurement_end']=state['closed_at']+1
    elif damage=='lost': state['health']['lost_callbacks']=1
    elif damage=='write_error': state['write_errors']=1
    elif damage=='worker_error': state['task_errors']=['RuntimeError']
    elif damage=='count_mismatch': state['counts']['gc_span']=999
    elif damage=='missing_boundary': rows=[r for r in rows if r['event']!='diagnostics_stopped']
    elif damage=='bad_gc_sequence': next(r for r in rows if r['event']=='gc_span')['start_sequence']=9999
    elif damage=='bad_gc_time': next(r for r in rows if r['event']=='gc_span')['span_end']=float('nan')
    elif damage=='missing_callback': state['health']['registration_missing']=True
    elif damage=='loop_stopped_early': next(r for r in rows if r['event']=='loop_monitor_stopped')['monotonic']=m['measurement_start']
    elif damage=='malformed_config': m['config']=[]
    elif damage=='cpu_coverage': next(r for r in rows if r['event']=='diagnostic_cpu')['sampled_at']=m['measurement_start']+1
    if damage not in ['missing_log','malformed_log']: log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if damage not in ['missing_status','malformed_status']: status.write_text(json.dumps(state))
    result=audit_runner(tmp_path,m)
    assert result is not None and not result['valid'] and result['issues']


def test_legacy_and_disabled_do_not_add_report_fields(tmp_path):
    assert audit_runner(tmp_path,{}) is None
    assert audit_runner(tmp_path,{'config':{'runner_diagnostics':False}}) is None
    (tmp_path/FILES[0]).write_text('')
    assert not audit_runner(tmp_path,{})['valid']


def test_startup_write_failure_stops_before_monitor_and_removes_callback(tmp_path,monkeypatch):
    async def scenario():
        observer=RunnerDiagnostics(tmp_path,uuid.uuid4().hex)
        async def failed_flush(path):
            observer.telemetry.write_errors+=1
        monkeypatch.setattr(observer.telemetry,'flush',failed_flush)
        with pytest.raises(RuntimeError,match='initial persistence'):
            await observer.start()
        assert not observer.tasks
        with pytest.raises(RuntimeError):
            await observer.close()
        assert observer.diagnostics.callback not in gc.callbacks
    asyncio.run(scenario())


@pytest.mark.parametrize('failure',['worker','flush','callback_overflow'])
def test_collection_failure_is_not_a_successful_shutdown(tmp_path,monkeypatch,failure):
    async def scenario():
        observer=RunnerDiagnostics(tmp_path,uuid.uuid4().hex,capacity=2 if failure=='callback_overflow' else 4096)
        await observer.start()
        if failure=='worker':
            observer.tasks[0].cancel()
        elif failure=='flush':
            def failed_write(*args,**kwargs): raise OSError('injected write failure')
            monkeypatch.setattr(Path,'open',failed_write)
        else:
            for _ in range(4): gc.collect(0)
        try:
            with pytest.raises((RuntimeError,OSError)):
                await observer.close()
        finally:
            assert observer.diagnostics.callback not in gc.callbacks
    asyncio.run(scenario())
