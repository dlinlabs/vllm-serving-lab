import asyncio
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import httpx
import pytest

from runner_diagnostics import RunnerDiagnostics, audit_runner, SCHEMA, LOOP_INTERVAL
from runner_gc_control import RunnerGCControl, audit_gc_control, gc_spec, FILE, MAX_SAMPLE_GAP
from run_trace_pilot import replay
from test_trace_replay import trace, sse
from trace_control import make_plan, read_plan, CONTROL_PROTOCOL, PLAN_FILE
from trace_workload import sha256_file
from types import SimpleNamespace


def config():
    return dict(runner_gc_policy='defer', runner_max_rss_mib=64,
                runner_diagnostics=True, sampler_mode='process', engine_observations=True,
                gateway_diagnostics=False, drain_timeout=.3,
                budget=96., budget_schedule=None, ttft_slo=.5,e2e_slo=None,
                max_scheduling_lag=.05,max_budget_lag=.05)


def probe(rss=1024):
    def read():
        return dict(available=True,pid=os.getpid(),process_created_unix=123.,
                    sample_start=time.monotonic(),sample_end=time.monotonic(),rss_bytes=rss)
    return read


def captured(tmp_path):
    async def scenario():
        run_id=uuid.uuid4().hex
        observer=RunnerDiagnostics(tmp_path,run_id)
        await observer.start()
        data=trace((0,.03,.06),.12)
        cfg=config();spec=gc_spec(cfg,data['duration_seconds'])
        # Inject only the resource-reader seam for unit tests; the production
        # CLI always uses the unchanged native PID/procfs guarded reader.
        guard=RunnerGCControl(tmp_path,run_id,spec,_probe=probe())
        rows=[]
        try:
            await guard.prepare()
            start=time.monotonic()+.02
            async def handler(request):
                await asyncio.sleep(.01)
                return sse(request.headers['x-benchmark-request-id'])
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await guard.supervise(replay(client,data,start,run_id,.3,10,rows))
        finally:
            guard.close()
            await observer.close()
        manifest=dict(config=cfg,run_id=run_id,runner_process_id=os.getpid(),
            measurement_start=start,measurement_end=start+data['duration_seconds'],
            runner_gc_control=spec,
            runner_diagnostics=dict(schema_version=SCHEMA,enabled=True,process_id=os.getpid(),
                                    interval_seconds=LOOP_INTERVAL),
            code_sha256={'runner_gc_control.py':'b'*64,'runner_diagnostics.py':'a'*64})
        (tmp_path/'trace.json').write_text(json.dumps(data))
        return manifest,rows,guard
    return asyncio.run(scenario())


def test_mock_replay_covers_arrivals_and_drain_and_restores_gc(tmp_path):
    old=(gc.isenabled(),gc.get_threshold())
    manifest,rows,guard=captured(tmp_path)
    assert len(rows)==3 and all(r['success'] for r in rows)
    assert (gc.isenabled(),gc.get_threshold())==old
    runner=audit_runner(tmp_path,manifest,rows)
    assert runner['valid'],runner
    result=audit_gc_control(tmp_path,manifest,rows,runner)
    assert result['valid'] and result['restored'] and result['samples']>=4,result
    assert guard.state['precollect_end']<=guard.state['disabled_at']<=manifest['measurement_start']
    assert max(r['finished_at'] for r in rows)<=guard.state['restore_started_at']


@pytest.mark.parametrize('change',[
    {'runner_max_rss_mib':None},{'runner_max_rss_mib':0},{'runner_max_rss_mib':-1},
    {'runner_max_rss_mib':True},{'runner_max_rss_mib':float('nan')},
    {'runner_max_rss_mib':float('inf')},{'runner_max_rss_mib':1e308},
    {'runner_gc_policy':'unknown'},{'runner_diagnostics':False},
    {'sampler_mode':'thread'},{'engine_observations':False},
    {'drain_timeout':600}, {'runner_gc_policy':'automatic','runner_max_rss_mib':64}])
def test_invalid_modes_and_bounds_are_rejected(change):
    cfg=config();cfg.update(change)
    with pytest.raises(ValueError):gc_spec(cfg,.12)


def test_automatic_default_has_no_intervention_or_artifacts(tmp_path):
    assert gc_spec({},.12) is None
    assert audit_gc_control(tmp_path,dict(config={})) is None
    (tmp_path/FILE).write_text('{}')
    assert not audit_gc_control(tmp_path,dict(config={}))['valid']


def test_plan_binds_gc_policy_and_memory_limit(tmp_path):
    cfg=config();data=trace();path=tmp_path/'trace.json';path.write_text(json.dumps(data))
    manifest=dict(config=cfg,trace_sha256=sha256_file(path),control_protocol=CONTROL_PROTOCOL)
    plan=make_plan(cfg,data,manifest['trace_sha256'],[dict(at_seconds=0,budget=96)])
    (tmp_path/PLAN_FILE).write_text(json.dumps(plan));manifest['replay_plan_sha256']=sha256_file(tmp_path/PLAN_FILE)
    assert read_plan(tmp_path,manifest,data)['runner_gc_control']['max_rss_bytes']==64*1048576
    manifest['config']['runner_max_rss_mib']=128
    with pytest.raises(ValueError,match='differs'):read_plan(tmp_path,manifest,data)


@pytest.mark.parametrize('failure',['rss','probe','work','cancel','repeated_cancel','policy','gap','duration'])
def test_failure_cancels_work_and_restores_state(tmp_path,failure):
    async def scenario():
        old=(gc.isenabled(),gc.get_threshold())
        spec=gc_spec(config(),.12)
        guard=RunnerGCControl(tmp_path,uuid.uuid4().hex,spec,_probe=probe())
        stopped=asyncio.Event()
        await guard.prepare()
        async def work():
            try:
                if failure=='work':raise ValueError('injected work failure')
                if failure=='policy':gc.enable()
                if failure=='gap':
                    await asyncio.sleep(.02)
                    time.sleep(MAX_SAMPLE_GAP+.1)
                if failure=='repeated_cancel':
                    try:await asyncio.sleep(10)
                    finally:await asyncio.sleep(.05)
                else:await asyncio.sleep(10)
            finally:stopped.set()
        if failure=='rss':guard.probe=probe(65*1048576)
        if failure=='probe':guard.probe=lambda:dict(available=False)
        if failure=='duration':guard.spec=dict(spec,max_disabled_seconds=.01)
        task=asyncio.create_task(guard.supervise(work()))
        if failure in ['cancel','repeated_cancel']:
            await asyncio.sleep(.02);task.cancel()
            if failure=='repeated_cancel':
                await asyncio.sleep(.01);task.cancel()
        with pytest.raises((RuntimeError,ValueError,asyncio.CancelledError)):
            await task
        guard.close()
        assert (gc.isenabled(),gc.get_threshold())==old
        assert stopped.is_set() and not guard.state['completed']
    asyncio.run(scenario())


def test_initially_disabled_gc_is_rejected_without_enabling_it(tmp_path):
    async def scenario():
        gc.disable()
        guard=RunnerGCControl(tmp_path,uuid.uuid4().hex,gc_spec(config(),.12),_probe=probe())
        try:
            with pytest.raises(RuntimeError,match='already disabled'):await guard.prepare()
            guard.close()
            assert not gc.isenabled()
        finally:gc.enable()
    asyncio.run(scenario())


def test_preflight_resource_failure_does_not_disable_gc(tmp_path):
    async def scenario():
        guard=RunnerGCControl(tmp_path,uuid.uuid4().hex,gc_spec(config(),.12),_probe=lambda:dict(available=False))
        try:
            with pytest.raises(RuntimeError,match='RSS probe'):await guard.prepare()
        finally:guard.close()
        assert gc.isenabled() and guard.state['disabled_at'] is None
    asyncio.run(scenario())


def test_child_process_keeps_its_own_gc_policy(tmp_path):
    async def scenario():
        guard=RunnerGCControl(tmp_path,uuid.uuid4().hex,gc_spec(config(),.12),_probe=probe())
        try:
            await guard.prepare()
            assert not gc.isenabled()
            assert subprocess.check_output([sys.executable,'-c','import gc; print(gc.isenabled())'],text=True).strip()=='True'
        finally:guard.close()
        assert gc.isenabled()
    asyncio.run(scenario())


@pytest.mark.parametrize('damage',['missing','malformed','policy','limit','pid','restore',
    'time','completed','failure','no_samples','rss','gap','gc_enabled','gc_span','runner_unqualified','fingerprint'])
def test_corrupt_evidence_fails_closed(tmp_path,damage):
    m,clients,_=captured(tmp_path);p=tmp_path/FILE;data=json.loads(p.read_text())
    runner=audit_runner(tmp_path,m,clients)
    if damage=='missing':p.unlink()
    elif damage=='malformed':p.write_text('[]')
    elif damage=='policy':m['config']['runner_gc_policy']='automatic'
    elif damage=='limit':m['config']['runner_max_rss_mib']=128
    elif damage=='pid':data['process_id']+=1
    elif damage=='restore':data['restored_enabled']=False
    elif damage=='time':data['restore_started_at']=m['measurement_start']
    elif damage=='completed':data['completed']=False
    elif damage=='failure':data['failure']='RSS limit exceeded'
    elif damage=='no_samples':data['samples']=[]
    elif damage=='rss':data['samples'][-1]['rss_bytes']=65*1048576
    elif damage=='gap':data['samples'][-1]['sample_end']+=1
    elif damage=='gc_enabled':data['samples'][2]['gc_enabled']=True
    elif damage=='gc_span':
        with (tmp_path/'runner_observations.jsonl').open('a') as h:
            h.write(json.dumps(dict(event='gc_span',span_start=m['measurement_start'],span_end=m['measurement_start']+.01))+'\n')
    elif damage=='runner_unqualified':runner['valid']=False
    elif damage=='fingerprint':m['code_sha256'].clear()
    if damage not in ['missing','malformed']:p.write_text(json.dumps(data))
    result=audit_gc_control(tmp_path,m,clients,runner)
    assert result is not None and not result['valid'],result


@pytest.mark.parametrize('outcome',['success','rss','preflight','cancel','persist'])
def test_runner_orchestration_restores_before_gateway_cleanup(tmp_path,monkeypatch,outcome):
    """Orchestration unit test with mock resources/services, not native acceptance."""
    import run_trace_pilot
    import diagnostic_sampling
    import runner_gc_control
    async def scenario():
        alive=False
        began=asyncio.Event()
        callbacks=tuple(gc.callbacks)
        calls=0
        def read():
            nonlocal calls
            calls+=1
            if outcome=='preflight':return dict(available=False)
            return probe(65*1048576 if outcome=='rss' and calls>3 else 1024)()
        monkeypatch.setattr(runner_gc_control,'resources',read)
        if outcome=='persist':
            original=Path.write_text
            def write(path,*args,**kwargs):
                if path.name==FILE+'.tmp':raise OSError('injected GC evidence write failure')
                return original(path,*args,**kwargs)
            monkeypatch.setattr(Path,'write_text',write)
        class Process:
            pid=os.getpid()
            returncode=None
            def poll(self):return self.returncode
            def terminate(self):
                assert gc.isenabled()
                assert (tmp_path/'runner_observations_status.json').exists()
                self.returncode=0
            def wait(self,timeout=None):return 0
        def launch(*a,**k):
            nonlocal alive
            alive=True
            return Process()
        class Sampler:
            def __init__(self,*a,**k):self.state={}
            def start(self):pass
            async def wait_ready(self,*a):pass
            def request_stop(self):pass
            async def close(self,*a):self.state['stopped']=True
        async def handler(request):
            if request.url.path=='/v1/models':
                return httpx.Response(200,json={'data':[{'id':'mock-model','max_model_len':8192}]})
            if request.url.path=='/metrics':
                if not alive:raise httpx.ConnectError('not started')
                return httpx.Response(200,json=dict(tokenizer_available=True,
                    telemetry_run_id='12345678123456781234567812345678',gateway_diagnostics=False,
                    current_in_flight=0,current_admitted_cost=0.,state_sampled_at=time.monotonic()))
            if request.url.path=='/admin/budget':
                assert gc.isenabled()  # Initial budget acknowledgement precedes GC deferral.
                return httpx.Response(200,json=dict(applied_at=time.monotonic(),budget=96))
            if not gc.isenabled():
                began.set()
                if outcome in ['rss','cancel']:await asyncio.sleep(10)
            return sse(request.headers['x-benchmark-request-id'])
        real=httpx.AsyncClient
        monkeypatch.setattr(run_trace_pilot.httpx,'AsyncClient',lambda **k:real(transport=httpx.MockTransport(handler),**k))
        monkeypatch.setattr(run_trace_pilot,'subprocess',SimpleNamespace(Popen=launch,STDOUT=-2))
        monkeypatch.setattr(diagnostic_sampling,'SamplerWorker',Sampler)
        args=SimpleNamespace(**dict(config(),budget=96,vllm_url='http://mock',startup_timeout=1,
                                   max_pending=10,request_timeout=1,warmup_requests=1))
        m={};task=asyncio.create_task(run_trace_pilot.run(args,trace((0,.03,.06),.12),tmp_path,m))
        if outcome=='cancel':
            await asyncio.wait_for(began.wait(),2);task.cancel()
        if outcome=='success':await task
        else:
            with pytest.raises((RuntimeError,OSError,asyncio.CancelledError)):await task
        assert gc.isenabled() and tuple(gc.callbacks)==callbacks
        assert (tmp_path/'client.json').is_file()
        if outcome!='persist':
            evidence=json.loads((tmp_path/FILE).read_text())
            assert evidence['restored_enabled'] is True
            assert evidence['completed'] is (outcome=='success')
    asyncio.run(scenario())
