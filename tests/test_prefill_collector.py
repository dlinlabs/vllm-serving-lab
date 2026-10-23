"""Executable CPU contracts for the installed-version collector, not GPU evidence."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest

from prefill_lab.contracts import Buffer,audit_capture,features,fifo_reference,scheduled_work,step_key,digest
from prefill_lab.runtime import Trial,ExternalCapture,INTERNAL_REQUEST
from prefill_lab.tokens import materialize,validate,peak_kv_tokens
from prefill_lab.analysis import observer_gate,basis,fit,predict,freeze_models
from prefill_lab.causal import rollout,dispatcher_parity
from prefill_lab.plugin import WorkerRecorder


class Tokenizer:
    all_special_ids=[0,1]
    def get_vocab(self):return {str(i):i for i in range(20)}


def small_plan():
    cases=[dict(id='single',background_decoders=0,foreground_tokens=[4],foreground_offsets_ms=[0],sentinel=None)]
    order=[dict(trial_id=f't{i}',case='single',stage='observer_check',observer=m,paired_content_key='pair1') for i,m in enumerate(('off','light'))]
    return cases,order


def test_exact_token_manifest_pairing_and_tamper_guards():
    cs,order=small_plan()
    a=materialize(cs,order,Tokenizer(),model='model',revision='a'*40,tokenizer_sha256='hash')
    b=materialize(cs,order,Tokenizer(),model='model',revision='a'*40,tokenizer_sha256='hash')
    assert a==b and a['trials']['t0']==a['trials']['t1']
    assert all(0 not in p['input_ids'] and 1 not in p['input_ids'] for p in a['pools'].values())
    broken=copy.deepcopy(a);next(iter(broken['pools'].values()))['input_ids'][0]+=1
    with pytest.raises(ValueError,match='pool'):validate(broken,cs,order)
    broken=copy.deepcopy(a);broken['trials']['t0'][0]['max_tokens']=127
    with pytest.raises(ValueError,match='frozen'):validate(broken,cs,order)
    with pytest.raises(ValueError,match='immutable'):materialize(cs,order,Tokenizer(),model='model',revision='main',tokenizer_sha256='x')
    assert peak_kv_tokens([dict(input_tokens=17,max_tokens=16)],16)==48


def output(work,new=None):
    new=set(new or [])
    return NS(num_scheduled_tokens={rid:n for rid,pos,n,length in work},
              total_num_scheduled_tokens=sum(n for _,_,n,_ in work),
              scheduled_new_reqs=[NS(req_id=rid,num_computed_tokens=pos,prompt_token_ids=[2]*length) for rid,pos,n,length in work if rid in new],
              scheduled_cached_reqs=NS(req_ids=[rid for rid,_,_,_ in work if rid not in new],num_computed_tokens=[pos for rid,pos,_,_ in work if rid not in new]),
              preempted_req_ids=set(),scheduled_spec_decode_tokens={})


def test_partial_chunks_and_async_stable_ids_are_exact():
    a=output([('r',0,2048,7168)],['r']);b=output([('r',2048,2048,7168)])
    assert step_key(a)!=step_key(b)
    row=features(scheduled_work(b),{'r':7168})
    assert row['prefill']==2048 and row['attention_pairs']==2048*2048+2048*2049/2
    d=output([('d',1030,1,1024)])
    assert features(scheduled_work(d),{'d':1024})['decode_context']==1030
    a.total_num_scheduled_tokens+=1
    with pytest.raises(ValueError,match='total'):scheduled_work(a)


def test_restricted_reference_preserves_running_order_and_async_terminal_guard():
    def r(computed,n,prompt,max_tokens=128,placeholders=0):
        return dict(computed=computed,num_tokens=n,prompt_tokens=prompt,max_tokens=max_tokens,placeholders=placeholders)
    state=dict(running=['long','decode','terminal'],waiting=['short'],requests={
        'long':r(2048,7168,7168),'decode':r(1024,1025,1024),
        'terminal':r(1151,1151,1024,128,1),'short':r(0,512,512)})
    assert fifo_reference(state,2048,32)=={'long':2048}
    assert fifo_reference(state,4096,32)=={'long':4096}
    state['requests']['long']['computed']=7168
    assert fifo_reference(state,2048,32)=={'decode':1,'short':512}


class Clock:
    def __init__(self):self.now=1.
    def __call__(self):return self.now
    def advance(self,x=.01):self.now+=x


class FakeCore:
    def __init__(self):
        self.scheduler=NS(running=[],waiting=[],requests={})
        self.vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=8,max_num_seqs=32,long_prefill_token_threshold=0))
        self.added=[];self.aborted=[]
    def add_request(self,row):
        self.added.append(row)
        r=NS(request_id=row['id'],num_computed_tokens=0,num_tokens=len(row['input_ids']),
             num_output_placeholders=0,num_prompt_tokens=len(row['input_ids']),max_tokens=row['max_tokens'])
        self.scheduler.requests[r.request_id]=r;self.scheduler.waiting.append(r)
    def abort_requests(self,ids):self.aborted.extend(ids)


def test_trial_background_barrier_atomic_batch_output_isolation_and_lifecycle():
    clock=Clock();core=FakeCore()
    rows=[dict(id='background-0',role='background',input_ids=[2]*2,max_tokens=5,at=0),
          dict(id='foreground-0',role='foreground',input_ids=[2]*4,max_tokens=2,at=0)]
    trial=Trial(core,dict(trial_id='t',observer='light',requests=rows,epoch='x'),lambda r:r,clock=clock)
    trial.begin();assert len(core.added)==1 and trial.release is None
    bg='plab/t/background-0'
    before=trial.before_schedule();o=output([(bg,0,2,2)],[bg]);trial.after_schedule(o,before)
    core.scheduler.waiting=[];core.scheduler.running=[core.scheduler.requests[bg]]
    core.scheduler.requests[bg].num_computed_tokens=2;core.scheduler.requests[bg].num_tokens=3
    clock.advance();batch=NS(outputs=[NS(request_id=bg,new_token_ids=[7],finish_reason=None)])
    trial.after_output(o,{0:batch})
    assert batch.outputs==[] and trial.release==clock.now and len(core.added)==2
    fg='plab/t/foreground-0'
    assert trial.requests[fg]['enqueue']==trial.release
    for rid,max_tokens in ((fg,2),(bg,5)):
        trial.requests[rid]['output_tokens']=max_tokens;trial.requests[rid]['finished']=clock.now
    # Force a private request output plus an unrelated frontend output through the filter.
    outsider=NS(request_id='ordinary',new_token_ids=[9],finish_reason=None)
    batch=NS(outputs=[outsider]);trial.after_output(output([]),{0:batch})
    assert batch.outputs==[outsider] and trial.status=='complete'
    assert trial.capture()['release_state']['requests'][bg]['num_tokens']==3


def test_overflow_and_cancellation_preserve_invalid_evidence():
    b=Buffer(1);b.append(1)
    with pytest.raises(RuntimeError):b.append(2)
    assert b.rows==[1] and b.overflow==1
    core=FakeCore();clock=Clock()
    trial=Trial(core,dict(trial_id='t',observer='light',requests=[dict(id='foreground-0',role='foreground',input_ids=[2]*4,max_tokens=2,at=0)],epoch='e'),lambda x:x,clock=clock,capacity=1)
    trial.begin();rid=core.added[0]['id']
    for pos in (0,2):
        o=output([(rid,pos,2,4)],[rid] if pos==0 else None)
        trial.after_schedule(o,trial.before_schedule())
    assert trial.steps.overflow==1 and trial.errors
    trial.abort('cancelled');assert core.aborted==[rid]
    assert trial.capture()['status']=='invalid'


class Event:
    ready=False
    def record(self,stream):pass
    def query(self):return self.ready
    def elapsed_time(self,other):return 12.


class FakeTorch:
    cuda=NS(Event=lambda **kw:Event(),current_stream=lambda:'stream')


def test_cuda_events_resolve_later_without_global_or_event_synchronize():
    recorder=WorkerRecorder(FakeTorch(),'t','light')
    entry=recorder.begin(output([('r',0,4,4)],['r']));recorder.end(entry)
    assert recorder.resolve()['pending']==1
    entry[2].ready=True
    result=recorder.resolve();assert result['pending']==0 and result['rows'][0]['gpu_forward_s']==.012
    assert not result['cuda_scope_qualified']
    off=WorkerRecorder(FakeTorch(),'off','off');assert off.begin(output([('r',0,4,4)],['r'])) is None


def good_capture():
    o=output([('plab/t/foreground-0',0,4,4)],['plab/t/foreground-0']);key=step_key(o)
    work=scheduled_work(o)
    c=dict(status='complete',errors=[],overflow=0,release=1.,foreground_done=1.1,
           requests={'plab/t/foreground-0':dict(role='foreground',input_tokens=4,max_tokens=1,output_tokens=1,finished=1.1,first=1.1)},
           steps=[dict(step_id=key,start=1.,end=1.1,work=work,parity=dict(matches=True),**features(work,{'plab/t/foreground-0':4}))])
    w=dict(step_id=key,counts={'plab/t/foreground-0':4},executed_counts={'plab/t/foreground-0':4},executed_positions={'plab/t/foreground-0':0},actual_tokens=4,runtime_mode='NONE',padded_tokens=4,microbatching=False,gpu_forward_s=.09)
    c['workers']=[dict(rows=[w])]
    return c,w


@pytest.mark.parametrize('damage', ['missing','duplicate','progress','count','regression','nan','short'])
def test_audit_rejects_missing_duplicate_mismatched_and_incomplete_evidence(damage):
    c,w=good_capture();rows=[w]
    assert audit_capture(c,rows)['valid']
    if damage=='missing':rows=[]
    if damage=='duplicate':rows=[w,w]
    if damage=='progress':c['steps'][0]['work'][0]['position']=1
    if damage=='count':w['counts']={}
    if damage=='regression':c['steps'][0]['parity']['matches']=False
    if damage=='nan':c['steps'][0]['end']=float('nan')
    if damage=='short':c['requests']['plab/t/foreground-0']['output_tokens']=0
    assert not audit_capture(c,rows)['valid']


def test_dispatcher_prediction_uses_only_own_work_and_frozen_grid():
    c,w=good_capture();table={'4:0':['NONE',4,None,False]}
    assert dispatcher_parity(c,table)['matches']
    w['padded_tokens']=8;assert not dispatcher_parity(c,table)['matches']
    model=dict(features=['mode:NONE','decode','prefill'],coefficients=[.001,.0001,.001])
    case=dict(background_decoders=0,foreground_tokens=[4],foreground_offsets_ms=[0],sentinel=None)
    table={f'{n}:{u}':['NONE',n,None,False] for n in range(1,9) for u in (0,1)}
    result=rollout(case,model,table)
    assert result['foreground']['foreground-0']['ttft_s']==pytest.approx(.005)
    # No observed future labels/chunks parameter is accepted by the rollout API.
    with pytest.raises(TypeError):rollout(case,model,table,future_steps=[])


def test_mode_aware_fit_trains_scaling_on_only_supplied_cells():
    rng=np.random.default_rng(4);training={}
    for i in range(4):
        rows=[]
        for j in range(30):
            d,p=map(float,rng.uniform(1,10,2));mode='NONE' if j%2 else 'FULL';padding=j%3
            rows.append(dict(mode=mode,decode=d,prefill=p,padding_excess=padding,y=.01+.002*d+.004*p+.003*padding))
        training[str(i)]=rows
    encoding=basis(training);model=fit(training,'M0',encoding)
    assert np.max([abs(predict(r,model)-r['y']) for rows in training.values() for r in rows])<1e-12
    changed=copy.deepcopy(training);changed['held']=[dict(mode='UNKNOWN',y=1e9)]
    assert fit({k:v for k,v in changed.items() if k!='held'},'M0',encoding)==model
    with pytest.raises(ValueError,match='Mode'):predict(dict(mode='UNKNOWN'),model)
    with pytest.raises(ValueError,match='18'):freeze_models([],[])


def test_observer_cannot_pass_missing_trials():
    assert observer_gate([],[])['status']=='inconclusive'


def test_external_capture_keeps_frontend_outputs_and_records_engine_work():
    core=FakeCore();capture=ExternalCapture(core,'bridge','light')
    r=NS(request_id='cmpl-http-0',sampling_params=NS(extra_args={'prefill_lab_role':'foreground'}),num_prompt_tokens=4,max_tokens=1)
    capture.register_request(r);core.add_request(dict(id=r.request_id,input_ids=[2]*4,max_tokens=1))
    o=output([(r.request_id,0,4,4)],[r.request_id])
    capture.after_schedule(o,capture.before_schedule())
    item=NS(request_id=r.request_id,new_token_ids=[2],finish_reason='length');batch=NS(outputs=[item])
    capture.after_output(o,{0:batch})
    assert batch.outputs==[item] and capture.requests[r.request_id]['output_tokens']==1
    assert not capture.inflight


def test_runtime_preflight_requires_resolved_configuration_token_binding_and_kv_headroom():
    from prefill_lab.suite import runtime_preflight
    cs,order=small_plan();manifest=materialize(cs,order,Tokenizer(),model='m',revision='a'*40,tokenizer_sha256='h')
    meta=dict(max_model_len=8192,max_num_batched_tokens=2048,max_num_seqs=32,tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,prefix_cache=False,speculative=False,enforce_eager=False,enable_chunked_prefill=True,kv_transfer=False,scheduler_policy='fcfs',long_prefill_token_threshold=0,max_num_scheduled_tokens=2048,revision='a'*40,tokenizer_revision='a'*40,token_manifest_sha256=digest(manifest),block_size=16,num_gpu_blocks=100,workers=[dict(device_name='NVIDIA GeForce RTX 4090',uniform_decode_query_len=1,torch_version='2.13.0+cu130',cuda_version='13.0')])
    assert runtime_preflight(meta,manifest)['valid']
    for field,value in (('prefix_cache',True),('max_model_len',4096),('num_gpu_blocks',1),('token_manifest_sha256','changed')):
        bad=dict(meta);bad[field]=value
        with pytest.raises(ValueError):runtime_preflight(bad,manifest)


def test_endpoint_auth_and_frozen_trial_selection(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from prefill_lab.endpoint import Endpoint
    cs,order=small_plan();manifest=materialize(cs,order,Tokenizer(),model='m',revision='a'*40,tokenizer_sha256='h')
    plugin=Endpoint();plugin.manifest=manifest;plugin.trials={t['trial_id']:t for t in order};calls=[]
    class Core:
        async def call_utility_async(self,method,payload):calls.append((method,payload));return dict(started=True)
    plugin.engine=NS(engine_core=Core());plugin.argv=['server']
    app=FastAPI();plugin.attach_router(app);monkeypatch.setenv('PREFILL_LAB_KEY','x'*64)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://test') as client:
            assert (await client.post('/prefill-lab/command',json={'operation':'meta'})).status_code==401
            client.headers['authorization']='Bearer '+'x'*64
            assert (await client.post('/prefill-lab/command',json={'operation':'start','trial_id':'missing'})).status_code==400
            assert (await client.post('/prefill-lab/command',json={'operation':'start','trial_id':'t0','input_ids':[1]})).status_code==400
            assert (await client.post('/prefill-lab/command',json={'operation':'start','trial_id':'t0'})).status_code==200
    asyncio.run(exercise());assert len(calls)==1
    assert calls[0][0]=='prefill_lab_command' and len(calls[0][1]['trial']['requests'][0]['input_ids'])==4


@pytest.mark.parametrize('transfer_failure',[False,True])
def test_suite_order_budget_and_partial_stage_retention(monkeypatch,tmp_path,transfer_failure):
    import prefill_lab.suite as suite
    cs,_=small_plan()
    order=[dict(trial_id='warm',case='single',stage='warmup',observer='light'),
           dict(trial_id='observer',case='single',stage='observer_check',observer='light'),
           dict(trial_id='main',case='single',stage='main',observer='light'),
           dict(trial_id='drift',case='single',stage='drift_reference',observer='light',block=3,position='end'),
           dict(trial_id='bridge',case='single',stage='http_bridge',observer='light'),
           dict(trial_id='deep',case='single',stage='deep_diagnostic',observer='deep')]
    transfers=[dict(attempt_id='transfer')]
    manifest=materialize(cs,order,Tokenizer(),model='m',revision='a'*40,tokenizer_sha256='h')
    monkeypatch.setattr(suite,'inputs',lambda:(cs,order,transfers))
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:dict(valid=True))
    monkeypatch.setattr(suite,'host_identity',lambda:dict(cpu='mock'))
    monkeypatch.setattr(suite,'audit_capture',lambda *a,**kw:dict(valid=True))
    monkeypatch.setattr(suite,'dispatcher_parity',lambda *a:dict(matches=True))
    monkeypatch.setattr(suite,'observer_gate',lambda *a:dict(status='passed'))
    monkeypatch.setattr(suite,'freeze_models',lambda *a:dict(noise_readiness=dict(status='within_tested_noise_range')))
    monkeypatch.setattr(suite,'drift_report',lambda *a:dict(rows=[]))
    history=[]
    async def controlled(client,t,deadline,partial_path=None):
        history.append(t['trial_id']);c,w=good_capture();c.update(trial_id=t['trial_id'],start=1.,end=1.1)
        return c
    async def bridge(client,http,url,t,m,d,partial_path=None):return dict(capture=await controlled(client,t,d))
    async def transfer(*a):
        history.append('transfer')
        if transfer_failure:raise TimeoutError('Transfer drain deadline')
    async def sleep(*a):pass
    monkeypatch.setattr(suite,'controlled_trial',controlled);monkeypatch.setattr(suite,'http_bridge',bridge)
    monkeypatch.setattr(suite,'transfer',transfer);monkeypatch.setattr(suite.asyncio,'sleep',sleep)
    class Client:
        async def command(self,operation,**kw):return dict(workers=[dict(dispatch_table={})])
    summary=asyncio.run(suite.run(Client(),None,'http://test',manifest,{},tmp_path))
    assert history==(['warm','observer','main','drift','bridge','transfer'] if transfer_failure else ['warm','observer','main','drift','bridge','transfer','deep'])
    assert summary['stages']['main']['integrity']=='complete_valid'
    assert summary['complete']==(not transfer_failure) and not summary['model_accepted']
    if transfer_failure:
        assert summary['stages']['transfer']['integrity']=='incomplete'
        assert summary['stages']['deep_diagnostic']['attempted_trial_ids']==[]


def test_unknown_or_nonfinite_worker_timing_is_not_a_gpu_label():
    from prefill_lab.analysis import rows_from_capture
    c,w=good_capture()
    with pytest.raises(ValueError,match='qualified'):rows_from_capture(c,'gpu')
    w['runtime_mode']=None
    with pytest.raises(ValueError,match='Unavailable'):rows_from_capture(c)


def test_worker_event_buffer_is_bounded_and_invalidates_overflow():
    recorder=WorkerRecorder(FakeTorch(),'t','light');recorder.records.capacity=1
    for pos in (0,4):recorder.end(recorder.begin(output([('r',pos,4,8)],['r'] if pos==0 else None)))
    assert len(recorder.pending)==1 and recorder.records.overflow==1


def test_delayed_arrival_posts_to_core_queue_without_mutating_scheduler_thread():
    import queue
    core=FakeCore();core.input_queue=queue.Queue();clock=Clock()
    trial=Trial(core,dict(trial_id='t',observer='off',requests=[
        dict(id='foreground-0',role='foreground',input_ids=[2]*4,max_tokens=2,at=0),
        dict(id='sentinel',role='sentinel',input_ids=[2]*2,max_tokens=2,at=.001)],epoch='e'),lambda x:x,clock=clock)
    trial.begin();kind,payload=core.input_queue.get(timeout=1)
    assert kind==INTERNAL_REQUEST and payload[0]=='e' and len(core.added)==1
    assert payload[1]==['plab/t/sentinel']
    trial.abort('test finished')


def test_engine_hooks_and_worker_records_reconcile_end_to_end_on_cpu(monkeypatch):
    import prefill_lab.plugin as plugin
    clock=Clock();core=FakeCore()
    class Scheduler:
        def __init__(self):self.running=[];self.waiting=[];self.requests={};self.new=set()
        def schedule(self):
            work=[]
            for rid,r in self.requests.items():
                pos=r.num_computed_tokens;n=r.num_tokens-pos
                if n:work.append((rid,pos,n,r.num_prompt_tokens))
            o=output(work,self.new);self.new.clear()
            for rid,n in o.num_scheduled_tokens.items():self.requests[rid].num_computed_tokens+=n
            return o
        def update_from_output(self,o,*args):
            rows=[]
            for rid in o.num_scheduled_tokens:
                r=self.requests[rid];r.num_tokens+=1
                count=r.num_tokens-r.num_prompt_tokens
                done=count==r.max_tokens
                rows.append(NS(request_id=rid,new_token_ids=[7],finish_reason='length' if done else None))
                if done:del self.requests[rid];self.running=[v for v in self.running if v.request_id!=rid]
                elif r in self.waiting:self.waiting.remove(r);self.running.append(r)
            return {0:NS(outputs=rows)}
    core.scheduler=Scheduler()
    original_add=core.add_request
    def add(row):original_add(row);core.scheduler.new.add(row['id'])
    core.add_request=add
    monkeypatch.setattr(plugin,'installed_sources',lambda:dict(mock=True))
    plugin.core_runtime(core)
    trial=Trial(core,dict(trial_id='integrated',observer='light',requests=[dict(id='foreground-0',role='foreground',input_ids=[2]*4,max_tokens=2,at=0)],epoch='x'),lambda r:r,clock=clock)
    core._prefill_trial=trial;trial.begin();recorder=WorkerRecorder(FakeTorch(),'integrated','light')
    while core.scheduler.requests:
        scheduled=core.scheduler.schedule();event=recorder.begin(scheduled)
        event[0].update(runtime_mode='NONE',padded_tokens=sum(scheduled.num_scheduled_tokens.values()),
                        executed_counts=dict(scheduled.num_scheduled_tokens),actual_tokens=sum(scheduled.num_scheduled_tokens.values()),
                        executed_positions={r['request_id']:r['position'] for r in scheduled_work(scheduled)})
        recorder.end(event);event[2].ready=True;clock.advance()
        result=core.scheduler.update_from_output(scheduled)
        assert result[0].outputs==[]
    captured=trial.capture();workers=recorder.resolve()
    assert captured['status']=='complete' and len(captured['steps'])==2
    assert audit_capture(captured,workers['rows'])['valid']
    workers['rows'].pop()
    assert not audit_capture(captured,workers['rows'])['valid']


def test_installed_source_guard_refuses_unreviewed_code(monkeypatch,tmp_path):
    import prefill_lab.plugin as plugin
    import sys
    declaration=json.loads((Path(plugin.__file__).resolve().parents[1]/'experiments/prefill-vllm-v030-source-map.json').read_text())
    for name in declaration['sha256']:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('changed source')
    monkeypatch.setitem(sys.modules,'vllm',NS(__file__=str(tmp_path/'__init__.py')))
    with pytest.raises(RuntimeError,match='sources differ'):plugin.installed_sources()


def test_native_cpu_gate_does_not_turn_procfs_failure_into_acceptance(monkeypatch,tmp_path):
    from scripts.run_prefill_service import cpu_check
    from diagnostic_sampling import ObservationSampler
    monkeypatch.setattr(ObservationSampler,'process_resources',staticmethod(lambda pid:dict(available=False,error_reason='procfs_pid_namespace_unverified')))
    assert cpu_check(tmp_path/'evidence')==1
    result=json.loads((tmp_path/'evidence/acceptance.json').read_text())
    assert not result['valid'] and result['tests']==0


def test_natural_eos_can_have_short_output_but_abort_cannot_pass():
    c,w=good_capture();r=next(iter(c['requests'].values()));r['max_tokens']=128;r['finish_reason']='stop'
    assert audit_capture(c,[w],forced_outputs=False)['valid']
    assert not audit_capture(c,[w],forced_outputs=True)['valid']
    r['finish_reason']='abort'
    assert not audit_capture(c,[w],forced_outputs=False)['valid']


def test_v2_candidates_preserve_order_query_caps_and_prefill_uniformity():
    from prefill_lab.dispatch import freeze_v2_manager,v2_dispatch
    def desc(mode,tokens,reqs,uniform=None,query=None,ubatches=1):
        return NS(cg_mode=NS(name=mode),num_tokens=tokens,num_reqs=reqs,uniform_token_count=uniform,max_query_len=query,num_active_loras=0,num_ubatches=ubatches)
    manager=NS(_graphs_captured=True,_candidates={(3,0):[
        desc('FULL',4,4,1,1),desc('PIECEWISE',8,None)],(6,0):[
        desc('FULL',8,4,None,1),desc('PIECEWISE',8,None)]})
    frozen=freeze_v2_manager(manager)
    assert v2_dispatch(frozen,3,3,1,False)==('FULL',4,4,True)
    assert v2_dispatch(frozen,1,3,3,True)==('PIECEWISE',8,None,False)
    assert v2_dispatch(frozen,3,6,2,False)==('PIECEWISE',8,None,False)
    assert v2_dispatch(frozen,2,2048,1024,True)==('NONE',2048,2,False)
    # A single-token prefill must not masquerade as a uniform-decode graph.
    assert v2_dispatch(frozen,3,3,1,True)==('PIECEWISE',8,None,False)
    manager._candidates[(3,0)][0].num_tokens=100
    assert frozen['candidates']['3'][0]['num_tokens']==4
    model=dict(features=['mode:FULL','mode:PIECEWISE','decode','prefill'],coefficients=[.001,.002,.001,.003])
    from prefill_lab.causal import mode_row
    assert mode_row(dict(chunks=[],decode=3,prefill=0,decode_context=300,attention_pairs=0),frozen)['padding_excess']==1


def test_complete_calibration_freeze_uses_18_trials_and_never_reads_held_labels():
    from scripts.build_prefill_design import cases
    from prefill_planning import rollout as toy_rollout
    from prefill_design import COEF
    from prefill_lab.analysis import freeze_models
    captures=[];order=[];truth=[*COEF,1e-7,3e-9]
    for case in [c for c in cases() if c['split']=='calibration']:
        result,steps=toy_rollout(case,truth)
        lengths={f'foreground-{i}':n for i,n in enumerate(case['foreground_tokens'])}
        for block in (1,2,3):
            tid=f"{case['id']}-{block}";rows=[];workers=[]
            for i,s in enumerate(steps):
                key=f'{tid}-{i}'
                work=[dict(request_id=f'plab/{tid}/{rid}',position=int(pos),tokens=int(q),prompt_length=lengths[rid]) for rid,q,pos in s['chunks']]
                if s['decode']:
                    count=int(s['decode']);total=int(s['decode_context'])
                    work.extend(dict(request_id=f'plab/{tid}/decoder-{j}',position=total//count+(j<total%count),tokens=1,prompt_length=0) for j in range(count))
                rows.append(dict(s,step_id=key,attention_pairs=sum(q*pos+q*(q+1)/2 for _,q,pos in s['chunks']),work=work))
                workers.append(dict(step_id=key,runtime_mode='NONE',padded_tokens=s['prefill']+s['decode'],microbatching=False))
            captures.append(dict(trial_id=tid,status='complete',errors=[],overflow=0,release=result['release_s'],foreground_done=result['scored_step_window']['end_s'],steps=rows,workers=[dict(rows=workers)]))
            order.append(dict(trial_id=tid,case=case['id'],block=block,analysis_role='train'))
    a=freeze_models(captures,order)
    assert len(a['training_capture_sha256'])==18 and a['selected']=='M2'
    assert a['noise_readiness']['status']=='within_tested_noise_range'
    assert a['fits']['M2']['coefficients']==pytest.approx(truth,rel=1e-6,abs=1e-11)
    assert 'fit' in a['plain_M0_ablation'] and not a['unique_context_mechanism_claimed']
    bad=dict(trial_id='held',steps='not even valid observations',errors=['fatal'])
    order.append(dict(trial_id='held',case='sentinel',block=1,analysis_role='held_case'))
    assert freeze_models([*captures,bad],order)==a
