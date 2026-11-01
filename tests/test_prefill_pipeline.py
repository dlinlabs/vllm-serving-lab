"""Native-shaped CPU fixtures for V2 pipeline hooks; no Torch/CUDA qualification."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from prefill_lab import pipeline,plugin
from prefill_lab.contracts import scheduled_work,step_key
from tests.test_prefill_collector import output


def fixture(monkeypatch,tmp_path,mode='light',capacity=8192):
    state=NS(events=[],trace=[],starts=0,stops=0,steps=0,native_copies=0,native_gets=0,fail_sample=None)
    class Event:
        ready=False
        def __init__(self,**kwargs):state.events.append(self)
        def record(self,stream):self.stream=stream
        def query(self):return self.ready
        def elapsed_time(self,end):return 1.
    class Scope:
        def __init__(self,name):self.name=name
        def __enter__(self):return self
        def __exit__(self,*exc):state.trace.append(dict(cat='user_annotation',name=self.name))
    class Profile:
        def start(self):state.starts+=1
        def stop(self):state.stops+=1
        def step(self):state.steps+=1
        def export_chrome_trace(self,path):Path(path).write_text(json.dumps(dict(traceEvents=state.trace+[dict(cat='kernel',name='mock')])) )
    main,copy_stream=NS(cuda_stream=0),NS(cuda_stream=22)
    torch=NS(cuda=NS(Event=Event,current_stream=lambda:main),profiler=NS(
        profile=lambda **kwargs:Profile(),record_function=Scope,ProfilerActivity=NS(CPU=0,CUDA=1)))
    rec=plugin.WorkerRecorder(torch,'stages',mode,stages=True)
    rec.stages.buffer.capacity=capacity
    monkeypatch.setenv('PREFILL_LAB_ARTIFACT_DIR',str(tmp_path))
    class Tensor:
        def numel(self):return 1
        def element_size(self):return 8
    class NativeOutput:
        def __init__(self,model_runner_output,sampler_output,num_sampled_tokens,main_stream,copy_stream,check_ep_fault=False,routed_experts=None):
            state.native_copies+=1;self.model_runner_output=model_runner_output
        def get_output(self):state.native_gets+=1;return self.model_runner_output
    class Runner:
        def __init__(self):self.main_stream=main;self.copy_stream=copy_stream;self.batch=None
        def sample(self,hidden_states,input_batch,grammar_output):
            if state.fail_sample is not None:raise state.fail_sample
            return NS(sampled_token_ids=Tensor(),logprobs_tensors=None,num_nans=None,sampling_mask_tensors=None)
        def postprocess_sampled(self):pass
        def sample_tokens(self,grammar_output):
            sampled=self.sample(None,self.batch,grammar_output)
            result=NativeOutput(NS(req_ids=self.batch.req_ids,prompt_logprobs_dict={}),sampled,Tensor(),self.main_stream,self.copy_stream)
            self.postprocess_sampled()
            return result
    pipeline.install(torch,Runner,NativeOutput,lambda:rec)
    runner=Runner();captures=[];requests={}
    def forward(i):
        rid='plab/stages/foreground-'+str(i)
        native=output([(rid,0,4,4)],[rid]);started=time.monotonic()
        entry=rec.begin(native);pipeline.bind_forward(runner,rec,entry);rec.end(entry)
        runner.batch=NS(req_ids=[rid],num_reqs=1,logits_indices=Tensor())
        step=dict(step_id=step_key(native),start=started,end=None,work=scheduled_work(native))
        captures.append(step);requests[rid]=dict(role='foreground',output_tokens=1,first=None)
        return step,runner.sample_tokens(None)
    def commit(step,native):
        out=native.get_output();rid=out.req_ids[0]
        with rec.stages.span(step['step_id'],'commit'):
            rec.stages.committed(step['step_id'],{0:NS(outputs=[NS(request_id=rid,new_token_ids=[9])])})
        step['end']=time.monotonic();requests[rid]['first']=step['end']
    def capture():
        for e in state.events:e.ready=True
        return dict(trial_id='stages',observer=mode,steps=captures,requests=requests,workers=[rec.resolve()])
    return NS(rec=rec,state=state,forward=forward,commit=commit,capture=capture,runner=runner,NativeOutput=NativeOutput,capture_requests=requests)


def test_hooks_capture_native_stages_without_new_synchronization(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path);step,obj=f.forward(0);f.commit(step,obj);c=f.capture()
    assert pipeline.audit(c)['valid']
    row=c['workers'][0]['pipeline']['rows'][0]
    assert row['primary_copy_bytes']==16 and not row['extra_copy_payloads']
    assert row['sampled_requests']==1 and row['logits_rows']==1
    assert row['stages']['copy_enqueue']['stream_handle']==22
    assert row['stages']['post_update']['stream_handle']==0
    assert f.state.native_copies==f.state.native_gets==1
    assert all(not hasattr(e,'synchronize') for e in f.state.events)
    assert not c['workers'][0]['pipeline']['cuda_scope_qualified']


def test_two_inflight_outputs_keep_their_own_step_binding(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path);a,oa=f.forward(0);b,ob=f.forward(1)
    f.commit(a,oa);f.commit(b,ob);c=f.capture()
    assert a['end']>b['start'] and pipeline.audit(c)['valid']
    rows=c['workers'][0]['pipeline']['rows']
    assert rows[0]['output_request_ids']==['plab/stages/foreground-0']
    assert rows[1]['output_request_ids']==['plab/stages/foreground-1']


def test_fifth_sampling_is_profiled_before_stop(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path,'deep')
    for i in range(5):
        s,o=f.forward(i)
        if i<4:assert f.state.stops==0
        else:assert f.state.stops==1
        f.commit(s,o)
    c=f.capture();proof=c['workers'][0]['deep_profile']
    assert proof['stage_step_ids']==proof['step_ids'] and len(proof['step_ids'])==5
    assert pipeline.audit(c,tmp_path)['valid']
    assert (f.state.starts,f.state.stops,f.state.steps)==(1,1,5)
    assert proof['profile_stop_span']['export_return']>=proof['profile_stop_span']['stop_return']


def test_copy_events_can_remain_pending_after_forward_resolves(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path);s,o=f.forward(0);f.commit(s,o)
    for e in f.state.events:e.ready=True
    copy_end=next(end for row,start,end in f.rec.stages.pending if row is f.rec.stages.by_id[s['step_id']]['stages']['copy_enqueue'])
    copy_end.ready=False
    monkeypatch.setattr(plugin,'_WORKER',f.rec)
    result=plugin.finish_worker_recorder()
    assert result['pending']==1 and plugin._WORKER is f.rec and not f.rec.active
    copy_end.ready=True;result=plugin.finish_worker_recorder()
    assert result['pending']==0 and plugin._WORKER is None


def test_core_commit_hook_joins_original_step_and_retains_first_token_clock(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path);step,obj=f.forward(0);result=obj.get_output()
    rid=result.req_ids[0]
    results={0:NS(outputs=[NS(request_id=rid,new_token_ids=[9])])}
    scheduler=NS(schedule=lambda:None,update_from_output=lambda *args:results)
    core=NS(scheduler=scheduler)
    monkeypatch.setattr(plugin,'installed_sources',lambda:{})
    monkeypatch.setattr(plugin.sys,'setswitchinterval',lambda value:None)
    monkeypatch.setattr(plugin,'_WORKER',f.rec)
    plugin.core_runtime(core)
    def observed(*args):
        step['end']=time.monotonic();f.capture_requests[rid]['first']=step['end']
    core._prefill_trial=NS(mode='light',after_output=observed)
    native=output([(rid,0,4,4)],[rid])
    assert scheduler.update_from_output(native,result) is results
    assert pipeline.audit(f.capture())['valid']


@pytest.mark.parametrize('damage',['missing','duplicate_step','work','missing_stage','gpu_nan','cpu_order','request_ids','first','output_count','bytes','error','thread','overflow','main_stream','copy_stream','promoted','forward_event'])
def test_pipeline_corruption_cannot_pass(monkeypatch,tmp_path,damage):
    f=fixture(monkeypatch,tmp_path);s,o=f.forward(0);f.commit(s,o);c=f.capture()
    p=c['workers'][0]['pipeline'];r=p['rows'][0]
    if damage=='missing':p['rows']=[]
    elif damage=='duplicate_step':p['rows'].append(copy.deepcopy(r))
    elif damage=='work':r['counts']={}
    elif damage=='missing_stage':del r['stages']['post_update']
    elif damage=='gpu_nan':r['stages']['copy_enqueue']['gpu_elapsed_s']=float('nan')
    elif damage=='cpu_order':r['stages']['commit']['host_start']=r['stages']['forward']['host_start']-1
    elif damage=='request_ids':r['output_request_ids']=['another-step']
    elif damage=='first':next(iter(c['requests'].values()))['first']+=1
    elif damage=='output_count':r['committed_outputs']={}
    elif damage=='bytes':r['primary_copy_bytes']=0
    elif damage=='error':r['stages']['output_ready']['error_type']='RuntimeError'
    elif damage=='thread':r['stages']['commit']['thread_id']=-1
    elif damage=='overflow':p['overflow']=1
    elif damage=='main_stream':r['stages']['post_update']['stream_handle']=1234
    elif damage=='copy_stream':r['stages']['copy_enqueue']['stream_handle']=0
    elif damage=='promoted':p['model_accepted']=True
    elif damage=='forward_event':r['stages']['forward']['gpu_elapsed_s']+=1
    assert not pipeline.audit(c)['valid']


def test_fifth_stage_profile_marker_is_required(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path,'deep')
    for i in range(5):s,o=f.forward(i);f.commit(s,o)
    c=f.capture();c['workers'][0]['deep_profile']['stage_step_ids'].pop()
    assert not pipeline.audit(c,tmp_path)['valid']


def test_profile_stop_cannot_precede_last_sampling(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path,'deep')
    for i in range(5):s,o=f.forward(i);f.commit(s,o)
    c=f.capture();c['workers'][0]['deep_profile']['profile_stop_span']['host_start']=0.
    assert not pipeline.audit(c,tmp_path)['valid']


def test_sampling_without_forward_and_capacity_overflow_fail(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path,capacity=1)
    with pytest.raises(RuntimeError,match='matching forward'):f.runner.sample_tokens(None)
    s,o=f.forward(0);f.commit(s,o)
    with pytest.raises(RuntimeError,match='overflow'):f.forward(1)


def test_native_sampling_exception_propagates_and_invalidates_stage(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path)
    def fail(*args):raise RuntimeError('native sampling failure')
    monkeypatch.setattr(f.runner,'sample',fail)
    with pytest.raises(RuntimeError,match='native sampling failure'):f.forward(0)
    assert f.runner._prefill_stage_binding is None
    row=next(iter(f.rec.stages.by_id.values()))
    assert row['stages']['sample_tokens']['error_type']=='RuntimeError'


def test_cancellation_retains_partial_rows_and_cannot_complete_profile(monkeypatch,tmp_path):
    import asyncio
    f=fixture(monkeypatch,tmp_path,'deep');f.state.fail_sample=asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):f.forward(0)
    assert f.runner._prefill_stage_binding is None
    f.rec.stop_profile();c=f.capture()
    assert c['workers'][0]['deep_profile']['state']=='incomplete'
    row=c['workers'][0]['pipeline']['rows'][0]
    assert row['stages']['sample_tokens']['error_type']=='CancelledError'
    assert not pipeline.audit(c,tmp_path)['valid']


def test_no_stage_recorder_leaves_native_hooks_idle(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path);f.rec.stages=None
    f.runner.batch=NS(req_ids=['ordinary'],num_reqs=1,logits_indices=NS(numel=lambda:1))
    result=f.runner.sample_tokens(None);result.get_output()
    assert f.state.native_copies==f.state.native_gets==1 and not f.state.events


@pytest.mark.parametrize('damage',[None,'v1','executor','pid','slots','disabled','missing_pid'])
def test_stage_preflight_requires_exact_reviewed_process_boundary(damage):
    m=dict(workers=[dict(adapter='V2',pipeline_stages=True,pid=10)],process_id=10,
        executor_class='vllm.v1.executor.uniproc_executor.UniProcExecutor',async_scheduling=True,max_concurrent_batches=2)
    if damage=='v1':m['workers'][0]['adapter']='V1'
    elif damage=='executor':m['executor_class']='other'
    elif damage=='pid':m['process_id']=11
    elif damage=='slots':m['max_concurrent_batches']=1
    elif damage=='disabled':m['workers'][0]['pipeline_stages']=False
    elif damage=='missing_pid':m['process_id']=m['workers'][0]['pid']=None
    if damage is None:pipeline.preflight(m)
    else:
        with pytest.raises(ValueError):pipeline.preflight(m)


def test_stage_host_delegates_with_distinct_output_and_exact_acceptance(monkeypatch,tmp_path):
    from scripts import prefill_host as host
    from tests.test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.subprocess,'run',lambda args,**kw:calls.append(args))
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    host.collect_host(tmp_path/'inputs',stages=True)
    args=calls[0]
    assert '--stage-diagnostics-only' in args and '--diagnostics-only' not in args
    assert 'pr32-stage-diagnostics' in args[args.index('--output')+1]
    monkeypatch.setattr(host,'commit',lambda:'z'*40)
    with pytest.raises(RuntimeError,match='exact checkout'):host.collect_host(tmp_path/'inputs',stages=True)
    assert len(calls)==1


@pytest.mark.parametrize('invalid',[False,True])
def test_stage_diagnostic_requires_pipeline_gate_and_preserves_development_scope(monkeypatch,tmp_path,invalid):
    import asyncio
    from prefill_lab import deep_diagnostic as diagnostic,suite
    meta=dict(workers=[dict(adapter='V2',pipeline_stages=True,pid=10,dispatch_table={})],process_id=10,
        executor_class='vllm.v1.executor.uniproc_executor.UniProcExecutor',async_scheduling=True,max_concurrent_batches=2)
    class Client:
        async def command(self,*args,**kwargs):return meta
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:{'valid':True})
    monkeypatch.setattr(suite,'host_identity',lambda:{})
    async def metrics(*args):return {}
    history=[]
    async def controlled(client,trial,*args):
        history.append(trial['trial_id']);return dict(requests={'r':dict(finished=1)},workers=[dict(rows=[])])
    monkeypatch.setattr(suite,'metrics_sample',metrics)
    monkeypatch.setattr(suite,'controlled_trial',controlled)
    monkeypatch.setattr(diagnostic,'audit_capture',lambda *a:{'valid':True})
    monkeypatch.setattr(diagnostic,'dispatcher_parity',lambda *a:{'matches':True})
    monkeypatch.setattr(diagnostic,'profile_audit',lambda *a:{'valid':True})
    monkeypatch.setattr(pipeline,'audit',lambda *a:dict(valid=not invalid,cuda_scope_qualified=False,model_accepted=False))
    if invalid:
        with pytest.raises(RuntimeError):asyncio.run(diagnostic.run(Client(),None,'url',{},tmp_path,stages=True))
    else:asyncio.run(diagnostic.run(Client(),None,'url',{},tmp_path,stages=True))
    d=json.loads((tmp_path/'diagnostic.json').read_text())
    assert d['scope']=='targeted_stage_diagnostic_only' and d['diagnostic_complete'] is (not invalid)
    assert len(history)==(1 if invalid else 4) and len(d['pipeline_audits'])==len(history)
    assert not d['model_accepted'] and not d['gpu_validated'] and not d['complete_experiment_passed']
    assert not (tmp_path/'suite.json').exists() and not (tmp_path/'model-freeze.json').exists()
    assert json.loads((tmp_path/'stage-protocol.json').read_text())['planned_trial_ids']==['trial-014','trial-007','trial-211','trial-212']


def test_diagnostic_cli_modes_are_mutually_exclusive(monkeypatch):
    import sys
    from scripts.run_prefill_service import main
    argv=['run','collect','--diagnostics-only','--stage-diagnostics-only']
    for key in ['manifest','tokenizer','trace','cpu-acceptance','output']:argv+=['--'+key,'dummy']
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==2
