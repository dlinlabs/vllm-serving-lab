"""Busy continuation and same-domain HTTP algebra; synthetic, no GPU claims."""
import copy
import json
import asyncio
from types import SimpleNamespace as NS
import pytest
import httpx
from test_dynamic_budget import gateway

from prefill_lab import live_predictor as live,event_clock_calibration as ec,live_http as http
from prefill_lab.async_predictor import Request,Scope,simulate
from prefill_lab.delayed_calibration import protocol
from prefill_lab.contracts import digest

D=dict(schema='v2-dispatch-candidates-v1',graphs_captured=False,candidates={})
MODEL=ec.costs([.01,.1,.002,.003,.001,.004],protocol(),ec.specification())


def empty():return dict(schema='live-engine-state-v1',clock_domain='test',observed_at=10.,last_commit_at=None,running=[],waiting=[],requests={},pending=[])


def busy(ready=False):
    s=empty();s.update(last_commit_at=9.998,running=['a'],requests={'a':dict(prompt_tokens=1024,max_tokens=4,num_tokens=1025,computed=1025,placeholders=1)},
        pending=[dict(work=[dict(request_id='a',position=1024,tokens=1)],scheduled_at=9.99,ready=ready)])
    return s


@pytest.mark.parametrize('tokens',[1,1024,4096,7168])
def test_empty_continuation_matches_original_independent_engine(tokens):
    request=Request('new',tokens,4);before=empty();result=live.forecast(before,request,MODEL,D,until_first=False)
    old=simulate([request],MODEL,D)
    assert result['ttft_from_cut_s']==pytest.approx(old['requests']['new']['ttft_s'])
    assert result['completion_from_cut_s']==pytest.approx(old['requests']['new']['finished_s'])
    assert result['output_tokens']=={'new':4} and before==empty()
    assert not result['model_accepted'] and not result['http_accepted'] and not result['mpc_accepted']


def test_pending_output_progress_and_future_labels_are_not_zeroed():
    state=busy();result=live.forecast(state,Request('new',1024,2),MODEL,D,until_first=False)
    assert result['initial_pending_residual_s']>0 and result['output_tokens']=={'a':4,'new':2}
    assert state==busy()
    ready=live.forecast(busy(True),Request('new',1024,2),MODEL,D)
    assert ready['initial_pending_residual_s']==0


@pytest.mark.parametrize('fault',['future','placeholder','progress','work','duplicate','scope','overdue','label'])
def test_incoherent_or_unsupported_busy_state_fails_closed(fault):
    s=busy()
    if fault=='future':s['pending'][0]['scheduled_at']=11.
    if fault=='placeholder':s['requests']['a']['placeholders']=0
    if fault=='progress':s['requests']['a']['num_tokens']+=1
    if fault=='work':s['pending'][0]['work'][0]['position']-=1
    if fault=='duplicate':s['waiting']=['a']
    if fault=='scope':s['pending']*=2
    if fault=='overdue':s['last_commit_at']=9.;s['pending'][0]['scheduled_at']=9.
    if fault=='label':s['measured_next_output']=11.
    with pytest.raises(ValueError):live.forecast(s,Request('new',1024,1),MODEL,D)


def test_kv_capacity_is_conservative_not_allocator_observation():
    with pytest.raises(ValueError,match='KV headroom'):live.forecast(empty(),Request('n',1024,128),MODEL,D,scope=Scope(kv_blocks=70))


def joined():
    tid='a'*32;client=dict(trace_id=tid,clock_domain='test',sent=1.,first=2.,finished=3.,status='complete',stream_complete=True)
    event=lambda name,t,**kw:dict(event=name,monotonic=t,request_id=tid,run_id='g',**kw)
    events=[event('received',1.1),event('accepted',1.2),event('backend_sent',1.3),event('first_content',1.9),event('completed',3.1),event('bridge_clock',1.11,clock_domain='test')]
    record=dict(trace_id=tid,clock_domain='test',cut=1.4,engine_enqueue=1.5,client_sent=1.,status='off')
    engine=dict(trace_id=tid,first=1.8,finished=2.9)
    return client,events,record,engine


def test_http_durations_close_without_summing_overlapping_spans():
    args=joined();r=http.join(*args)
    assert sum(r['parts'].values())==pytest.approx(r['client_ttft_s'])
    assert r['egress_s']==pytest.approx(.2) and r['engine_from_cut_s']==pytest.approx(.4)


@pytest.mark.parametrize('fault',['clock','duplicate','missing','order','identity','incomplete','future_forecast'])
def test_http_join_rejects_mixed_clocks_ids_and_future_information(fault):
    c,g,r,e=joined()
    if fault=='clock':r['clock_domain']='another-namespace'
    if fault=='duplicate':g.append(copy.deepcopy(g[0]))
    if fault=='missing':g.pop(0)
    if fault=='order':e['first']=2.1
    if fault=='identity':e['trace_id']='b'*32
    if fault=='incomplete':c['stream_complete']=False
    if fault=='future_forecast':r.update(status='predicted',snapshot={'observed_at':1.4},forecast={'state_sha256':digest({'observed_at':1.4})},prediction_finished=1.9)
    with pytest.raises(ValueError):http.join(c,g,r,e)


def test_sse_probe_ignores_role_only_and_handles_fragmentation():
    async def run():
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',b'data: {"choices":[{"text":"he',b'llo"}]}\n\n',b'data: [DONE]\n\n']:yield chunk
        def handle(request):return httpx.Response(200,stream=Body(),headers={'content-type':'text/event-stream','x-request-id':'a'*32})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            row=await http.probe(client,'http://test',{},'a'*32,0.,domain='test')
        assert row['status']=='complete' and row['sent']<=row['first']<=row['finished']
    asyncio.run(run())


def synthetic_http_rows():
    p=http.protocol();rows=[]
    for t in p['order']:
        rows.append(dict(**{k:t[k] for k in ('trial_id','case','stage','block','observer')},egress_s=.002,engine_from_cut_s=.1,
            elapsed_to_cut_s=.01,client_ttft_s=.112,forecast_available=t['observer']=='light',structure_matches=True,
            predicted_engine_s=.1,snapshot_width_s=.0001,prediction_age_s=.001))
    return p,rows


def test_transport_held_poison_isolation_and_no_unavailable_prediction_exclusion():
    p,rows=synthetic_http_rows();f=http.fit_transport(rows,p)
    poisoned=copy.deepcopy(rows)
    for r in poisoned:
        if r['stage']!='train':r['egress_s']=None
    assert http.fit_transport(poisoned,p)==f
    result=http.score(rows,p,f);assert result['candidate_passed'] and not result['model_accepted'] and not result['http_accepted']
    next(r for r in rows if r['stage']=='held' and r['observer']=='light')['forecast_available']=False
    result=http.score(rows,p,f);assert not result['candidate_passed'] and result['unavailable_predictions']==1


@pytest.mark.parametrize('fault',['missing','duplicate','training','forecast','structure','age','width','clock_freeze'])
def test_bridge_failure_is_retained_as_failed_gate_or_integrity_error(fault):
    p,rows=synthetic_http_rows();f=http.fit_transport(rows,p)
    r=next(r for r in rows if r['stage']=='held' and r['observer']=='light')
    if fault=='missing':rows.remove(r)
    if fault=='duplicate':rows.append(copy.deepcopy(r))
    if fault=='training':next(r for r in rows if r['stage']=='train' and r['observer']=='off')['egress_s']+=.1
    if fault=='forecast':r['predicted_engine_s']+=.1
    if fault=='structure':r['structure_matches']=False
    if fault=='age':r['prediction_age_s']=.006
    if fault=='width':r['snapshot_width_s']=.002
    if fault=='clock_freeze':f['egress_s']+=.1
    if fault in ('missing','duplicate','training','clock_freeze'):
        with pytest.raises(ValueError):http.score(rows,p,f)
    else:assert not http.score(rows,p,f)['candidate_passed']


def test_native_pending_readiness_never_uses_future_done_for_gpu_work(monkeypatch):
    import sys
    from prefill_lab.live_runtime import copy_ready
    class Output:pass
    class Future:
        def done(self):raise AssertionError('not GPU readiness')
    value=Future();value.async_output=Output();value.async_output.copy_event=NS(query=lambda:True)
    monkeypatch.setitem(sys.modules,'vllm.v1.executor.uniproc_executor',NS(AsyncOutputFuture=Future))
    monkeypatch.setitem(sys.modules,'vllm.v1.worker.gpu.async_utils',NS(AsyncOutput=Output))
    assert copy_ready(value,[{'tokens':1}]) is True


def test_tracker_captures_before_native_add_without_future_labels():
    from prefill_lab.live_runtime import LiveTracker
    output=object();future=object()
    native=NS(request_id='a',num_computed_tokens=1025,num_tokens=1025,num_output_placeholders=1,num_prompt_tokens=1024,max_tokens=128)
    core=NS(scheduler=NS(running=[native],waiting=[],requests={'a':native}),batch_queue=[(future,output,None)])
    tracker=LiveTracker(core,MODEL,Scope(),D,clock=lambda:10.,domain='test',ready_probe=lambda *a:True)
    tracker.pending[id(output)]=dict(work=[dict(request_id='a',position=1024,tokens=1)],scheduled_at=9.99)
    req=NS(request_id='probe',num_prompt_tokens=512,max_tokens=128,sampling_params=NS(ignore_eos=True,extra_args=dict(
        prefill_lab_trace_id='a'*32,prefill_lab_client_sent=9.9,prefill_lab_clock_domain='test',prefill_lab_probe=True)))
    tracker.before_add(req);row=tracker.records.rows[0]
    assert row['status']=='predicted' and row['engine_enqueue'] is None
    assert 'probe' not in row['snapshot']['requests'] and row['forecast']['ttft_from_cut_s']>0
    assert tracker.pending[id(output)]['scheduled_at']==9.99


def test_raw_gateway_endpoint_is_absent_by_default(gateway):
    assert not any(getattr(r,'path',None)=='/v1/completions' for r in gateway.app.routes)


def test_raw_gateway_forwarding_ids_and_fragmented_first_content(gateway,monkeypatch):
    async def scenario():
        from telemetry import Telemetry
        import prefill_lab.live_http as clocks
        monkeypatch.setattr(gateway,'ADMISSION_MODE','protected');gateway.sem=asyncio.Semaphore(32)
        monkeypatch.setattr(gateway,'MAX_OUTSTANDING',32)
        monkeypatch.setattr(gateway,'PREFILL_RAW_COMPLETIONS',True);monkeypatch.setattr(clocks,'clock_domain',lambda:'test')
        monkeypatch.setattr(gateway,'BENCHMARK_REQUEST_IDS',True);monkeypatch.setattr(gateway,'SHARED_UPSTREAM_CLIENT',True)
        gateway.telemetry=Telemetry();gateway.telemetry.enabled=True
        real=httpx.AsyncClient;seen=[]
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"text":"hi"}]}\n\n';yield b'data: [DONE]\n\n'
        def backend(request):
            seen.append(request);return httpx.Response(200,stream=Body(),headers={'content-type':'text/event-stream'})
        gateway.app.state.upstream_client=real(transport=httpx.MockTransport(backend))
        from starlette.requests import Request as HttpRequest
        payload=dict(model='test',prompt=[1,2],max_tokens=128,temperature=0,ignore_eos=True,stream=True,request_id='a'*32,
            stream_options={'include_usage':True},vllm_xargs={'prefill_lab_trace_id':'a'*32})
        raw=json.dumps(payload).encode()
        async def receive():return {'type':'http.request','body':raw,'more_body':False}
        req=HttpRequest(dict(type='http',method='POST',path='/v1/completions',headers=[(b'x-benchmark-request-id',b'a'*32)]),receive)
        response=await gateway.proxy_raw_completions(req);chunks=[c async for c in response.body_iterator];await response.background()
        assert str(seen[0].url).endswith('/v1/completions') and seen[0].content==raw
        assert response.headers['x-request-id']=='a'*32 and b'hi' in b''.join(chunks)
        assert len([r for r in gateway.telemetry.records if r['event']=='first_content'])==1
        assert next(r for r in gateway.telemetry.records if r['event']=='bridge_clock')['clock_domain']=='test'
        await gateway.app.state.upstream_client.aclose()
    asyncio.run(scenario())


def test_live_host_uses_two_manifests_and_archives_failures(monkeypatch,tmp_path):
    import subprocess
    from pathlib import Path
    from scripts import prefill_host as host
    from test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');prepared=[];calls=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path));monkeypatch.setattr(host,'prepare',lambda args:prepared.append(args))
    def fail(args,**kwargs):
        calls.append(args);output=Path(args[args.index('--output')+1]);output.mkdir();(output/'bridge-summary.json').write_text('{}')
        raise subprocess.CalledProcessError(2,args)
    monkeypatch.setattr(host.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',live_http=True)
    assert len(prepared)==2 and prepared[0].event_clock_only and prepared[1].live_http_tokens
    assert '--live-http-only' in calls[0] and '--http-manifest' in calls[0]
    assert len(list((tmp_path/'pr32-live-http').glob('*.tar.gz')))==1


def test_observation_is_immutable_and_has_no_future_schedule_labels():
    s=busy();before=copy.deepcopy(s)
    result=live.forecast(s,Request('n',512,128),MODEL,D)
    assert s==before and result['state_sha256']==digest(s)
    assert all('measured' not in key for key in result)


def test_busy_continuation_matches_from_empty_at_a_synthetic_poll_boundary():
    # Independent original from-empty simulator supplies a SYNTHETIC reference.
    # After a's prompt commits, its first decode batch is already pending.
    old=simulate([Request('a',1024,4),Request('n',512,2,.005)],MODEL,D)
    cut=old['requests']['n']['enqueue_s'];state=empty()
    state.update(observed_at=cut,last_commit_at=cut,running=['a'],
        requests={'a':dict(prompt_tokens=1024,max_tokens=4,num_tokens=1025,computed=1025,placeholders=1)},
        pending=[dict(work=[dict(request_id='a',position=1024,tokens=1)],scheduled_at=0.,ready=False)])
    result=live.forecast(state,Request('n',512,2),MODEL,D,until_first=False)
    assert result['ttft_from_cut_s']==pytest.approx(old['requests']['n']['first_s']-cut)
    assert result['completion_from_cut_s']==pytest.approx(old['requests']['n']['finished_s']-cut)
    assert result['output_tokens']=={'a':4,'n':2}


@pytest.mark.parametrize('fail_generator',[False,True])
def test_native_http_phase_freezes_transport_before_held_and_retains_failure(monkeypatch,tmp_path,fail_generator):
    from prefill_lab import live_experiment as experiment,delayed_calibration,causal,suite
    from prefill_lab.contracts import save
    p,rows=synthetic_http_rows();runtime={'workers':[{'dispatch_table':D}]};seen=[]
    gw=dict(admission_mode='protected',max_in_flight=32,max_waiting_capacity=0,shared_upstream_client=True,
            prefill_raw_completions=True,gateway_diagnostics=True,telemetry_dropped_records=0,telemetry_write_errors=0,telemetry_run_id='test',accepted_requests=http.expected_requests(p),
            rejected_requests=0,failed_requests=0,current_in_flight=0,current_waiting=0)
    save(tmp_path/'gateway-start.json',gw)
    async def get(*args):return NS(json=lambda:gw)
    http_client=NS(get=get)
    async def calibration(*args,**kw):
        assert kw['event_clock'];save(tmp_path/'frozen.json',{'sha256':'mock'});save(tmp_path/'runtime.json',runtime)
        return {'candidate_passed':True}
    monkeypatch.setattr(delayed_calibration,'run',calibration);monkeypatch.setattr(experiment,'validate_manifest',lambda m:None);monkeypatch.setattr(experiment,'validate_pair',lambda *m:None)
    monkeypatch.setattr(experiment,'gateway_records',lambda *a:[]);monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:{'matches':True})
    monkeypatch.setattr(experiment,'extract',lambda b,t,*args:next(r for r in rows if r['trial_id']==t['trial_id']))
    async def finish(*args):return {}
    monkeypatch.setattr(suite,'finish_external',finish)
    async def generator(path,tid,url,output,deadline):
        if fail_generator:raise RuntimeError('generator stopped')
        t=next(t for t in p['order'] if t['trial_id']==tid)
        if t['stage'] not in ('train','warmup'):assert (tmp_path/'transport-freeze.json').exists()
        return {'rows':[]}
    monkeypatch.setattr(experiment,'isolated_trial',generator)
    async def command(op,**kw):
        seen.append(op)
        return runtime if op=='meta' else {}
    if fail_generator:
        with pytest.raises(RuntimeError,match='generator stopped'):asyncio.run(experiment.run(NS(command=command),http_client,'',{}, {},tmp_path,'',99999999))
        assert 'abort' in seen and (tmp_path/'bridge/lh-001-partial.json').exists()
        assert not json.loads((tmp_path/'bridge-summary.json').read_text())['collection_complete']
        return
    result=asyncio.run(experiment.run(NS(command=command),http_client,'',{}, {},tmp_path,'',99999999))
    assert result['collection_complete'] and len(result['completed'])==98 and not result['model_accepted']
    assert seen.index('live_configure')<seen.index('external_begin')


def test_isolated_generator_is_terminated_and_joined_on_cancellation(monkeypatch,tmp_path):
    import time
    from prefill_lab.live_experiment import isolated_trial
    async def scenario():
        entered=asyncio.Event();closed=[]
        class Child:
            returncode=None
            async def wait(self):
                entered.set()
                if self.returncode is None:await asyncio.Event().wait()
                closed.append('joined');return self.returncode
            def terminate(self):self.returncode=-15;closed.append('terminated')
            def kill(self):self.returncode=-9
        child=Child()
        async def create(*args,**kw):return child
        monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
        task=asyncio.create_task(isolated_trial(tmp_path/'tokens.json','trial','http://test',tmp_path/'client.json',time.monotonic()+10))
        await entered.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert closed==['terminated','joined']
    asyncio.run(scenario())


@pytest.mark.parametrize('audit_fails',[False,True])
def test_successful_live_collection_requires_post_shutdown_audit(monkeypatch,tmp_path,audit_fails):
    import subprocess
    from pathlib import Path
    from scripts import prefill_host as host
    from test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path));monkeypatch.setattr(host,'prepare',lambda a:None)
    def invoke(args,**kwargs):
        calls.append(args)
        if len(calls)==1:
            output=Path(args[args.index('--output')+1]);output.mkdir();(output/'bridge-summary.json').write_text('{}')
        else:
            assert args[1].endswith('check_prefill_live_http.py') and kwargs['timeout']==180
            if audit_fails:raise subprocess.CalledProcessError(1,args)
    monkeypatch.setattr(host.subprocess,'run',invoke)
    if audit_fails:
        with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',live_http=True)
    else:host.collect_host(tmp_path/'inputs',live_http=True)
    assert len(calls)==2 and len(list((tmp_path/'pr32-live-http').glob('*.tar.gz')))==1


@pytest.mark.parametrize('key',['model','revision','tokenizer_sha256','tokenizer_files'])
def test_http_inputs_cannot_silently_use_a_different_tokenizer(key):
    from prefill_lab.live_experiment import validate_pair
    a=dict(model='model',revision='rev',tokenizer_sha256='hash',tokenizer_files={'file':'hash'})
    b=copy.deepcopy(a);validate_pair(a,b);b[key]='different'
    with pytest.raises(ValueError):validate_pair(a,b)


@pytest.mark.parametrize('problem',['missing','duplicate','unavailable','work_mismatch'])
def test_warmup_failure_cannot_be_hidden_by_later_passing_trials(problem):
    p,rows=synthetic_http_rows();assert http.validate_warmup(rows,p)
    warm=next(r for r in rows if r['stage']=='warmup')
    if problem=='missing':rows.remove(warm)
    elif problem=='duplicate':rows.append(copy.deepcopy(warm))
    elif problem=='unavailable':warm['forecast_available']=False
    else:warm['structure_matches']=False
    with pytest.raises(ValueError,match='warmup'):http.validate_warmup(rows,p)
