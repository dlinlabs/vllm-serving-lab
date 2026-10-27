"""Opt-in vLLM0.30.0 hooks. No scheduler replacement or eager/async override."""
import contextvars
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import time
import sys
import uuid
from contextlib import nullcontext

from .contracts import Buffer, digest, scheduled_work, step_key, wire_value
from .runtime import INTERNAL_REQUEST, Trial, ExternalCapture

_ACTIVE = contextvars.ContextVar('prefill_worker_step',default=None)
_WORKER = None
_REGISTERED = False


def installed_sources():
    import hashlib
    import vllm
    root=Path(vllm.__file__).resolve().parent
    declaration=json.loads((Path(__file__).resolve().parents[1]/'experiments/prefill-vllm-v030-source-map.json').read_text())
    actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in declaration['sha256']}
    if actual!=declaration['sha256']:raise RuntimeError('Installed vLLM sources differ from reviewed0.30.0 hooks')
    return dict(version=importlib.metadata.version('vllm'),files=actual,upstream=declaration['upstream'])


def request_factory(row):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    params=SamplingParams(temperature=0,max_tokens=row['max_tokens'],ignore_eos=True,detokenize=False)
    return Request(row['id'],row['input_ids'],params,None)


def core_runtime(core):
    if getattr(core,'_prefill_runtime_installed',False):return
    installed_sources()
    # Timer threads must acquire the GIL well inside the fixed5ms posting limit.
    sys.setswitchinterval(5e-4)
    core._prefill_trial=None
    core._prefill_external=False
    scheduler=core.scheduler
    schedule,update=scheduler.schedule,scheduler.update_from_output
    @functools.wraps(schedule)
    def observed_schedule(*args,**kwargs):
        trial=core._prefill_trial
        before=trial.before_schedule() if trial and trial.mode!='off' else None
        live=getattr(core,'_prefill_live',None)
        live_started=time.monotonic() if live and live.observed else None
        output=schedule(*args,**kwargs)
        receipt=getattr(trial,'receipt',None)
        if receipt:
            try:receipt.schedule(output)
            except Exception as exc:trial.errors.append('Receipt schedule failed: '+str(exc))
        if live and live.observed:
            try:live.scheduled(output,live_started)
            except Exception as exc:live.errors.append(str(exc))
        if trial and trial.mode!='off':
            try:trial.after_schedule(output,before)
            except (ValueError,KeyError,RuntimeError) as exc:trial.errors.append(f'Step capture failed: {exc}')
        return output
    @functools.wraps(update)
    def observed_update(output,*args,**kwargs):
        trial=core._prefill_trial
        rec=_WORKER
        stages=rec.stages if rec and rec.active and rec.stages and trial and trial.mode!='off' and output.num_scheduled_tokens else None
        key=step_key(output) if stages else None
        with stages.span(key,'commit',profiler=rec.profile) if stages else nullcontext():
            results=update(output,*args,**kwargs)
            if stages:stages.committed(key,results)
        receipt=getattr(trial,'receipt',None)
        if receipt:
            try:receipt.commit(output)
            except Exception as exc:trial.errors.append('Receipt commit failed: '+str(exc))
        live=getattr(core,'_prefill_live',None)
        if live and live.observed:
            try:live.committed(output)
            except Exception as exc:live.errors.append(str(exc))
        if trial:
            try:trial.after_output(output,results)
            except Exception as exc:
                # Catch instrumentation failures only; native update() above must
                # still surface its own errors. Keep the engine and raw results.
                trial.errors.append(f'Output capture failed ({type(exc).__name__}): {exc}')
                trial.status='invalid';trial.cancel.set()
                if not isinstance(trial,ExternalCapture):
                    # A failed observer may not have completed its private-output
                    # filter. These IDs do not exist in the frontend processor.
                    for batch in results.values():
                        batch.outputs=[item for item in batch.outputs if item.request_id not in trial.requests]
        return results
    scheduler.schedule, scheduler.update_from_output=observed_schedule,observed_update
    core._prefill_runtime_installed=True


def revision_identity(revision):
    # HF ResolvedRevision is a str subclass whose value can still be an alias
    # (e.g. main). Preserve its resolved commit and emit an exact builtin str:
    # vLLM's msgspec utility transport rejects custom str subclasses.
    return None if revision is None else str(getattr(revision,'resolved',revision))


def resolved_config(core):
    c=core.vllm_config;s=c.scheduler_config;p=c.parallel_config
    return dict(model=c.model_config.model,revision=revision_identity(c.model_config.revision),
                tokenizer_revision=revision_identity(c.model_config.tokenizer_revision),
                max_model_len=c.model_config.max_model_len,max_num_batched_tokens=s.max_num_batched_tokens,
                max_num_seqs=s.max_num_seqs,enable_chunked_prefill=s.enable_chunked_prefill,
                long_prefill_token_threshold=s.long_prefill_token_threshold,
                max_num_partial_prefills=getattr(s,'max_num_partial_prefills',None),max_long_partial_prefills=getattr(s,'max_long_partial_prefills',None),
                scheduler_class=f'{type(core.scheduler).__module__}.{type(core.scheduler).__name__}',
                scheduler_policy=str(s.policy),max_num_scheduled_tokens=s.max_num_scheduled_tokens,async_scheduling=core.async_scheduling,
                max_concurrent_batches=core.batch_queue_size,
                tensor_parallel_size=p.tensor_parallel_size,pipeline_parallel_size=p.pipeline_parallel_size,
                data_parallel_size=p.data_parallel_size,prefix_cache=c.cache_config.enable_prefix_caching,
                block_size=c.cache_config.block_size,num_gpu_blocks=c.cache_config.num_gpu_blocks,
                speculative=c.speculative_config is not None,enforce_eager=c.model_config.enforce_eager,
                dtype=str(c.model_config.dtype),attention_config=str(c.attention_config),
                compilation_config=str(c.compilation_config),kv_transfer=c.kv_transfer_config is not None,
                source=installed_sources(),gil_switch_interval_s=sys.getswitchinterval(),
                process_id=os.getpid(),executor_class=f'{type(core.model_executor).__module__}.{type(core.model_executor).__name__}')


def wire_reply(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        return wire_value(function(*args,**kwargs))
    return wrapped


@wire_reply
def command(core,payload):
    core_runtime(core)
    op=payload['operation'];trial=core._prefill_trial
    if op=='meta':
        result=resolved_config(core)
        result['workers']=core.model_executor.collective_rpc('prefill_lab_worker_meta')
        return result
    if op=='live_configure':
        if os.environ.get('PREFILL_LAB_LIVE_STATE')!='1' or trial or core.scheduler.has_requests() or core.batch_queue:
            raise RuntimeError('Live configuration requires an empty opt-in engine')
        from . import event_clock_calibration as clock
        from scripts.predict_prefill_async import runtime_inputs
        supplied=payload['runtime'];actual=resolved_config(core)
        actual['workers']=core.model_executor.collective_rpc('prefill_lab_worker_meta')
        if any(supplied.get(k)!=v for k,v in actual.items()):raise ValueError('Live runtime differs from native engine')
        frozen=payload['frozen']
        if frozen.get('measurement_origin')!='fresh_native_training_events':raise ValueError('Live phase requires fresh same-host training')
        clock.predict_frozen(frozen,clock.protocol(),supplied)
        scope,dispatch=runtime_inputs(supplied)
        core._prefill_live_config=(clock.costs(frozen['parameters_s'],clock.protocol(),clock.specification()),scope,dispatch)
        return dict(configured=True,frozen_sha256=frozen['sha256'])
    if op=='start':
        if trial or core.scheduler.has_requests() or core.batch_queue:
            raise RuntimeError('Start requires an empty engine and a fully fetched prior trial')
        spec=dict(payload['trial'],epoch=uuid.uuid4().hex)
        if spec['observer'] not in ('off','light','deep'):raise ValueError('Invalid observer mode')
        # A single worker, same loaded model; this is not per-trial model startup.
        core.model_executor.collective_rpc('prefill_lab_worker_begin',args=(spec['trial_id'],spec['observer']))
        trial=Trial(core,spec,request_factory);core._prefill_trial=trial
        trial.begin();return dict(started=True,trial_id=trial.id)
    if op=='status':return dict(status='idle' if trial is None else trial.status)
    if op=='abort':
        if trial:trial.abort(payload.get('reason','runner cancelled'))
        return dict(aborted=True)
    if op=='finish':
        if not trial:raise RuntimeError('No active trial')
        if core.scheduler.has_requests() or core.batch_queue:raise RuntimeError('Drain before fetching records')
        worker=core.model_executor.collective_rpc('prefill_lab_worker_finish')
        result=trial.capture();result['workers']=worker
        if any(w['pending'] for w in worker):return dict(status='gpu_events_pending')
        if trial.status!='complete':result['errors'].append('Trial did not complete')
        core._prefill_trial=None
        return result
    if op=='external_begin':
        if trial or core.scheduler.has_requests() or core.batch_queue:raise RuntimeError('External stage needs drained engine')
        core._prefill_external=True
        if os.environ.get('PREFILL_LAB_LIVE_STATE')=='1':
            from .live_runtime import LiveTracker
            config=getattr(core,'_prefill_live_config',None)
            if config is None:raise RuntimeError('Configure fresh live model first')
            core._prefill_live=LiveTracker(core,*config,observed=payload['observer']=='light')
        core._prefill_trial=ExternalCapture(core,payload['trial_id'],payload.get('observer','light'),payload.get('expected_foreground'))
        core.model_executor.collective_rpc('prefill_lab_worker_begin',args=(payload['trial_id'],payload.get('observer','light')))
        return dict(external=True)
    if op=='external_end':
        if core.scheduler.has_requests() or core.batch_queue:raise RuntimeError('External requests did not drain')
        if not isinstance(trial,ExternalCapture):raise RuntimeError('No external capture')
        workers=core.model_executor.collective_rpc('prefill_lab_worker_finish')
        if any(w['pending'] for w in workers):return dict(status='gpu_events_pending')
        trial.status='invalid' if trial.status=='invalid' or trial.errors else 'complete';result=trial.capture();result['workers']=workers
        if getattr(core,'_prefill_live',None):
            result['live']=core._prefill_live.capture();core._prefill_live=None
        core._prefill_trial=None;core._prefill_external=False
        return result
    raise ValueError('Unknown prefill operation')


class WorkerRecorder:
    def __init__(self,torch,trial_id,mode,*,stages=False):
        self.torch,self.trial_id,self.mode=torch,trial_id,mode
        self.records=Buffer();self.pending=[];self.profile=None;self.active=True
        self.profile_steps=0;self.profile_state='waiting_for_foreground';self.profile_step_ids=[]
        self.profile_ranges={};self.profile_artifact=None
        self.profile_stage_step_ids=[];self.profile_stop_span=None
        from .pipeline import StageRecorder
        self.stages=StageRecorder(torch) if stages else None

    def begin(self,output,*,dummy_run=False):
        if dummy_run or not self.active or not output.num_scheduled_tokens or self.mode=='off':return None
        torch=self.torch
        if self.mode=='deep' and self.profile_state=='waiting_for_foreground':
            foreground=any(rid.rsplit('/',1)[-1]=='sentinel' or rid.rsplit('/',1)[-1].startswith('foreground-')
                           for rid in output.num_scheduled_tokens)
            if foreground:
                self.profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True)
                self.profile.start();self.profile_state='recording'
        key=step_key(output)
        if self.profile:
            scope=torch.profiler.record_function('prefill_lab_step/'+key);scope.__enter__()
            self.profile_ranges[key]=scope
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        stream=torch.cuda.current_stream();start.record(stream)
        row=dict(step_id=key,counts=dict(output.num_scheduled_tokens),
                 executed_counts=None,executed_positions=None,actual_tokens=None,
                 host_start=time.monotonic(),host_end=None,gpu_forward_s=None,
                 query_uniformity=len(set(output.num_scheduled_tokens.values()))==1,
                 runtime_mode=None,padded_tokens=None,padded_requests=None,uniform=None,
                 capture_key=None,graph_actions={},attention_backends=None,
                 cuda_stream_id=getattr(stream,'cuda_stream',None),
                 cuda_scope='current-stream execute_model preparation and forward; sampling/multi-stream scope unqualified')
        if self.stages:self.stages.forward_begin(row,start,end,stream)
        return row,start,end,stream

    def end(self,entry):
        if entry is None:return
        row,start,end,stream=entry
        end.record(stream);row['host_end']=time.monotonic()
        if self.stages:self.stages.forward_end(row)
        if self.mode=='deep':
            scope=self.profile_ranges.pop(row['step_id'],None)
            if scope is not None:scope.__exit__(None,None,None)
        if len(self.pending)>=self.records.capacity:
            self.records.overflow+=1
        else:self.pending.append(entry)
        if self.profile and self.profile_steps<5:
            self.profile_step_ids.append(row['step_id'])
            self.profile_steps+=1
            if not self.stages:
                self.profile.step()
                if self.profile_steps==5:self.stop_profile()

    def profile_after_sample(self,key):
        if not self.profile:return
        if self.profile_step_ids[len(self.profile_stage_step_ids)]!=key:
            raise RuntimeError('Profile sample/forward identity mismatch')
        self.profile_stage_step_ids.append(key);self.profile.step()
        if len(self.profile_stage_step_ids)==5:self.stop_profile()

    def stop_profile(self):
        if not self.profile:return
        start=time.monotonic()
        self.profile.stop()
        stopped=time.monotonic()
        directory=Path(os.environ['PREFILL_LAB_ARTIFACT_DIR']);directory.mkdir(parents=True,exist_ok=True)
        self.profile.export_chrome_trace(str(directory/f'{self.trial_id}-worker-{os.getpid()}.json'))
        import hashlib
        path=directory/f'{self.trial_id}-worker-{os.getpid()}.json'
        self.profile_artifact=dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.profile_state='complete' if self.profile_steps==5 and (not self.stages or self.profile_stage_step_ids==self.profile_step_ids) else 'incomplete'
        self.profile_stop_span=dict(host_start=start,stop_return=stopped,export_return=time.monotonic())
        self.profile=None

    def resolve(self):
        remaining=[]
        for row,start,end,stream in self.pending:
            try:
                if not end.query():
                    remaining.append((row,start,end,stream));continue
                elapsed=start.elapsed_time(end)/1000
                from .contracts import finite
                if not finite(elapsed) or elapsed<0:raise ValueError('Invalid CUDA elapsed span')
                row['gpu_forward_s']=elapsed
            except (RuntimeError,ValueError) as exc:
                row['gpu_error']=type(exc).__name__
            try:self.records.append(row)
            except RuntimeError:pass
        self.pending=remaining
        result=dict(rows=self.records.rows,pending=len(remaining),overflow=self.records.overflow,cuda_scope_qualified=False)
        if self.stages:
            result['pipeline']=self.stages.capture();result['pending']+=result['pipeline']['pending']
        if self.mode=='deep':result['deep_profile']=dict(schema='foreground-profile-v1',state=self.profile_state,
            step_ids=list(self.profile_step_ids),artifact=self.profile_artifact,cuda_scope_qualified=False)
        if self.mode=='deep' and self.stages:
            result['deep_profile'].update(stage_step_ids=list(self.profile_stage_step_ids),profile_stop_span=self.profile_stop_span)
        return result


def guarded_add(core,request,original,*args,**kwargs):
    """Invalid traffic terminates only its request, never the EngineCore loop."""
    trial=getattr(core,'_prefill_trial',None)
    accepted=True
    if isinstance(trial,ExternalCapture):accepted=trial.register_request(request)
    elif trial and (request.request_id not in trial.requests or request.request_id in trial.enqueued):
        trial.errors.append('External traffic during controlled trial');trial.status='invalid';accepted=False
    if not accepted:
        rid=request.request_id
        # Duplicates may already be scheduled: abort that ID before replying.
        if rid in core.scheduler.requests:core.abort_requests([rid])
        core._send_abort_outputs_to_client([rid],getattr(request,'client_index',0))
        if rid in trial.requests:trial.requests[rid].update(finished=time.monotonic(),finish_reason='abort')
        return
    live=getattr(core,'_prefill_live',None)
    if live:
        try:live.before_add(request)
        except Exception as exc:live.errors.append(str(exc))
    result=original(core,request,*args,**kwargs)
    if live:
        try:live.after_add(request)
        except Exception as exc:live.errors.append(str(exc))
    return result


def handle_arrival(core,payload):
    epoch,ids,posted=payload;trial=getattr(core,'_prefill_trial',None)
    if isinstance(trial,ExternalCapture) or not trial or trial.epoch!=epoch or trial.status!='running':return
    try:trial.inject(ids,posted)
    except (ValueError,KeyError,RuntimeError) as exc:trial.abort(f'Arrival injection failed: {exc}')


def finish_worker_recorder():
    global _WORKER
    if _WORKER is None:raise RuntimeError('Worker recorder not initialized')
    _WORKER.active=False  # Disable immediately, even if CUDA resolution needs another poll.
    _WORKER.stop_profile();result=_WORKER.resolve()
    if not result['pending']:_WORKER=None
    return result


def forward_context_or_none():
    from vllm.forward_context import get_forward_context,is_forward_context_available
    return get_forward_context() if is_forward_context_available() else None


def attention_backends(runner):
    groups=getattr(runner,'attn_groups',None)
    if groups is None:return None
    return sorted({str(g.backend) for group in groups for g in group}) or None


def register():
    global _REGISTERED
    if _REGISTERED or os.environ.get('PREFILL_LAB_ENABLED')!='1':return
    if importlib.metadata.version('vllm').split('+')[0]!='0.30.0':
        raise RuntimeError('Prefill instrumentation requires reviewed vLLM0.30.0')
    installed_sources()
    from vllm.v1.engine.core import EngineCore,EngineCoreProc
    original_add=EngineCore.add_request
    def add(core,request,*args,**kwargs):
        return guarded_add(core,request,original_add,*args,**kwargs)
    EngineCore.add_request=add
    EngineCore.prefill_lab_command=command
    original_handle=EngineCoreProc._handle_client_request
    def handle(core,kind,payload):
        if kind==INTERNAL_REQUEST:
            handle_arrival(core,payload)
            return
        return original_handle(core,kind,payload)
    EngineCoreProc._handle_client_request=handle

    import torch
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    @wire_reply
    def worker_begin(worker,trial_id,mode):
        global _WORKER
        if _WORKER and (_WORKER.pending or _WORKER.stages and _WORKER.stages.pending):raise RuntimeError('Unresolved CUDA events from previous trial')
        if not isinstance(worker.model_runner,(GPUModelRunner,GPUModelRunnerV2)):raise RuntimeError('Unreviewed model runner; no graph/async override is allowed')
        stages=os.environ.get('PREFILL_LAB_STAGES')=='1'
        if stages and (mode=='off' or not isinstance(worker.model_runner,GPUModelRunnerV2)):
            raise RuntimeError('Stage diagnostic requires observed V2 calls')
        _WORKER=WorkerRecorder(torch,trial_id,mode,stages=stages)
        return dict(pid=os.getpid(),source=installed_sources())
    @wire_reply
    def worker_finish(worker):
        return finish_worker_recorder()
    @wire_reply
    def worker_meta(worker):
        if not isinstance(worker.model_runner,(GPUModelRunner,GPUModelRunnerV2)):
            raise RuntimeError('Unsupported installed model runner; preserve runtime defaults')
        runner=worker.model_runner
        table={}
        if isinstance(runner,GPUModelRunnerV2):
            from .dispatch import freeze_v2_manager
            if runner.cudagraph_manager is None:raise RuntimeError('V2 dispatcher not initialized')
            table=freeze_v2_manager(runner.cudagraph_manager)
            query_len=runner.decode_query_len
        else:
            query_len=runner.uniform_decode_query_len
            for uniform in (False,True):
                for count in range(1,2049):
                    mode,desc=runner.cudagraph_dispatcher.dispatch(count,uniform_decode=uniform)
                    table[f'{count}:{int(uniform)}']=[mode.name,desc.num_tokens,desc.num_reqs,desc.uniform]
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        return dict(dispatch_table=table,uniform_decode_query_len=query_len,pid=os.getpid(),gpu_uuid=str(getattr(props,'uuid',None)),
                    pipeline_stages=os.environ.get('PREFILL_LAB_STAGES')=='1',
                    model_runner=f'{type(runner).__module__}.{type(runner).__name__}',adapter='V2' if isinstance(runner,GPUModelRunnerV2) else 'V1',device_name=props.name,
                    total_memory=props.total_memory,torch_version=str(torch.__version__),
                    cuda_version=torch.version.cuda,source=installed_sources())
    Worker.prefill_lab_worker_meta=worker_meta
    Worker.prefill_lab_worker_begin=worker_begin
    Worker.prefill_lab_worker_finish=worker_finish
    execute=GPUModelRunner.execute_model
    @functools.wraps(execute)
    def observed_execute(runner,output,*args,**kwargs):
        entry=_WORKER.begin(output) if _WORKER else None
        token=_ACTIVE.set(entry[0] if entry else None)
        try:return execute(runner,output,*args,**kwargs)
        finally:
            _ACTIVE.reset(token)
            if _WORKER:_WORKER.end(entry)
    GPUModelRunner.execute_model=observed_execute
    execute_v2=GPUModelRunnerV2.execute_model
    @functools.wraps(execute_v2)
    def observed_execute_v2(runner,output,*args,**kwargs):
        dummy=kwargs.get('dummy_run',args[1] if len(args)>1 else False)
        entry=_WORKER.begin(output,dummy_run=dummy) if _WORKER else None
        if entry and _WORKER.stages:
            from .pipeline import bind_forward
            bind_forward(runner,_WORKER,entry)
        if entry:entry[0]['attention_backends']=attention_backends(runner)
        token=_ACTIVE.set(entry[0] if entry else None)
        try:return execute_v2(runner,output,*args,**kwargs)
        finally:
            _ACTIVE.reset(token)
            if _WORKER:_WORKER.end(entry)
    GPUModelRunnerV2.execute_model=observed_execute_v2
    if os.environ.get('PREFILL_LAB_STAGES')=='1':
        from .pipeline import install
        from vllm.v1.worker.gpu.async_utils import AsyncOutput
        install(torch,GPUModelRunnerV2,AsyncOutput,lambda:_WORKER)
    determine=GPUModelRunner._determine_batch_execution_and_padding
    @functools.wraps(determine)
    def observed_padding(runner,*args,**kwargs):
        result=determine(runner,*args,**kwargs);row=_ACTIVE.get()
        if row is not None:
            ids=list(runner.input_batch.req_ids)[:kwargs['num_reqs']]
            counts=kwargs['num_scheduled_tokens_np'].tolist()
            row.update(executed_counts=dict(zip(ids,map(int,counts))),
                       executed_positions=dict(zip(ids,map(int,runner.input_batch.num_computed_tokens_cpu[:len(ids)]))),actual_tokens=int(sum(counts)))
            mode,desc,ubatch,_,_=result
            row.update(runtime_mode=mode.name,padded_tokens=desc.num_tokens,padded_requests=desc.num_reqs,
                       uniform=desc.uniform,capture_key=str(desc),microbatching=bool(ubatch),
                       attention_backends=sorted({str(g.backend) for groups in runner.attn_groups for g in groups}) if runner.attn_groups and isinstance(runner.attn_groups[0],list) else None)
        return result
    GPUModelRunner._determine_batch_execution_and_padding=observed_padding
    graph_call=CUDAGraphWrapper.__call__
    @functools.wraps(graph_call)
    def observed_graph(wrapper,*args,**kwargs):
        row=_ACTIVE.get()
        if row is not None:
            ctx=forward_context_or_none()
            desc=None if ctx is None else ctx.batch_descriptor
            if ctx is not None and ctx.cudagraph_runtime_mode==wrapper.runtime_mode and desc is not None:
                entry=wrapper.concrete_cudagraph_entries.get(desc)
                action='capture' if entry is None or entry.cudagraph is None else 'replay'
                key=f'{wrapper.runtime_mode.name}:{desc}:{action}'
                row['graph_actions'][key]=row['graph_actions'].get(key,0)+1
        return graph_call(wrapper,*args,**kwargs)
    CUDAGraphWrapper.__call__=observed_graph
    prepare_v2=GPUModelRunnerV2.prepare_inputs
    @functools.wraps(prepare_v2)
    def observed_prepare_v2(runner,output,state,desc):
        row=_ACTIVE.get()
        if row is not None:
            ids=list(state.req_ids)
            row.update(executed_counts=dict(zip(ids,map(int,state.num_scheduled_tokens))),
                       executed_positions=dict(zip(ids,map(int,runner.req_states.num_computed_tokens_np[state.idx_mapping_np]))),
                       actual_tokens=int(state.num_tokens),worker_position_scope='CPU mirror used for input preparation')
        return prepare_v2(runner,output,state,desc)
    GPUModelRunnerV2.prepare_inputs=observed_prepare_v2
    dispatch_v2=CudaGraphManager.dispatch
    @functools.wraps(dispatch_v2)
    def observed_dispatch_v2(manager,*args,**kwargs):
        desc=dispatch_v2(manager,*args,**kwargs);row=_ACTIVE.get()
        if row is not None:
            row.update(runtime_mode=desc.cg_mode.name,padded_tokens=desc.num_tokens,
                       padded_requests=desc.num_reqs,uniform=desc.uniform_token_count is not None,
                       capture_key=str(desc),microbatching=desc.num_ubatches>1)
        return desc
    CudaGraphManager.dispatch=observed_dispatch_v2
    fullgraph=CudaGraphManager.run_fullgraph
    @functools.wraps(fullgraph)
    def observed_fullgraph(manager,desc):
        result=fullgraph(manager,desc);row=_ACTIVE.get()
        if row is not None:
            key=f'FULL:{desc}:replay';row['graph_actions'][key]=row['graph_actions'].get(key,0)+1
        return result
    CudaGraphManager.run_fullgraph=observed_fullgraph
    breakable=BreakableCUDAGraphWrapper.__call__
    @functools.wraps(breakable)
    def observed_breakable(wrapper,*args,**kwargs):
        row=_ACTIVE.get()
        if row is not None:
            ctx=forward_context_or_none()
            desc=None if ctx is None else ctx.batch_descriptor
            if ctx is not None and ctx.cudagraph_runtime_mode.name!='NONE' and (wrapper.runtime_mode is None or wrapper.runtime_mode==ctx.cudagraph_runtime_mode):
                entry=wrapper.entries.get(desc)
                action='capture' if entry is None or entry.capture is None else 'replay'
                key=f'breakable:{desc}:{action}'
                row['graph_actions'][key]=row['graph_actions'].get(key,0)+1
        return breakable(wrapper,*args,**kwargs)
    BreakableCUDAGraphWrapper.__call__=observed_breakable
    _REGISTERED=True
