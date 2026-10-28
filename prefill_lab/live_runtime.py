"""Opt-in engine-thread live snapshots and causal forecasts before native ADD.

No CUDA synchronize, Future.result, token materialization, or future schedule is
used. Errors retain an unavailable forecast and never change native admission.
"""
import copy
import time

from .async_predictor import Request
from .contracts import Buffer,digest,scheduled_work,finite
from .runtime import snapshot as scheduler_snapshot
from .live_http import clock_domain,trace_id
from .live_predictor import forecast


def copy_ready(future,work):
    if not work:return bool(future.done())  # Empty execute Future has no GPU sampling payload.
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    if not isinstance(future,AsyncOutputFuture) or not isinstance(future.async_output,AsyncOutput):
        raise ValueError('Unreviewed pending output representation')
    return bool(future.async_output.copy_event.query())


class LiveTracker:
    def __init__(self,core,model,scope,dispatch,*,observed=True,clock=time.monotonic,domain=None,ready_probe=copy_ready,configuration_sha256=None):
        self.core,self.model,self.scope,self.dispatch=core,model,scope,dispatch
        self.observed,self.clock,self.domain,self.ready_probe=observed,clock,domain or clock_domain(),ready_probe
        self.configuration_sha256=configuration_sha256
        self.pending={};self.last_commit=None;self.records=Buffer(64);self.errors=[];self.by_id={}

    def scheduled(self,output,started):
        if not self.observed:return
        if getattr(output,'preempted_req_ids',None) or getattr(output,'scheduled_spec_decode_tokens',None):
            raise ValueError('Preemption/speculation invalidates live state')
        work=[{k:w[k] for k in ('request_id','position','tokens')} for w in scheduled_work(output)]
        self.pending[id(output)]=dict(work=work,scheduled_at=started)
        if len(self.pending)>2:raise ValueError('Live pending capacity exceeded')

    def committed(self,output):
        if not self.observed:return
        if self.pending.pop(id(output),None) is None:raise ValueError('Unknown live output batch')
        self.last_commit=self.clock()

    def snapshot(self):
        native=list(reversed(self.core.batch_queue));state=scheduler_snapshot(self.core.scheduler);jobs=[]
        if len(native)>=2:raise ValueError('Snapshot is not an EngineCore pre-ADD boundary')
        for future,output,_ in native:
            item=self.pending.get(id(output))
            if item is None:raise ValueError('Unobserved in-flight history')
            jobs.append(dict(copy.deepcopy(item),ready=self.ready_probe(future,item['work'])))
        if len(jobs)!=len(self.pending):raise ValueError('Native and tracked backlog differ')
        return dict(schema='live-engine-state-v1',clock_domain=self.domain,observed_at=self.clock(),
                    last_commit_at=self.last_commit,**state,pending=jobs)

    def _begin_record(self,request):
        metadata=request.sampling_params.extra_args or {};identifier=trace_id(metadata.get('prefill_lab_trace_id'))
        if (metadata.get('prefill_lab_clock_domain')!=self.domain or not finite(metadata.get('prefill_lab_client_sent'))
                or metadata['prefill_lab_client_sent']>self.clock() or not request.sampling_params.ignore_eos
                or request.max_tokens not in (128,256)):raise ValueError('Live request clock/forced length differs')
        row=dict(trace_id=identifier,engine_id=request.request_id,clock_domain=self.domain,cut=self.clock(),
                 engine_enqueue=None,status='off',forecast=None,client_sent=metadata['prefill_lab_client_sent'])
        self.records.append(row);self.by_id[request.request_id]=row
        return row,self.observed and metadata.get('prefill_lab_probe') is True

    def before_add(self,request):
        row,probe=self._begin_record(request)
        if not probe:return
        row['status']='unavailable';started=self.clock()
        try:
            state=self.snapshot();row.update(snapshot=state,cut=state['observed_at'],snapshot_width_s=state['observed_at']-started)
            result=forecast(state,Request(request.request_id,request.num_prompt_tokens,request.max_tokens),
                            self.model,self.dispatch,scope=self.scope)
            row.update(forecast=result,status='predicted')
        except Exception as exc:row.update(error=type(exc).__name__+': '+str(exc))
        finally:row['prediction_finished']=self.clock()

    def after_add(self,request):
        row=self.by_id.get(request.request_id)
        if row is not None:
            row['engine_enqueue']=self.clock()
            trial=self.core._prefill_trial
            if request.request_id in trial.requests:trial.requests[request.request_id]['enqueue']=row['engine_enqueue']

    def capture(self):
        result=dict(schema='live-probe-records-v1',records=self.records.rows,overflow=self.records.overflow,
                    errors=self.errors,clock_domain=self.domain,observed=self.observed,
                    cost_sha256=digest(self.model.to_dict()),dispatcher_sha256=digest(self.dispatch))
        if self.configuration_sha256 is not None:result['configuration_sha256']=self.configuration_sha256
        return result


class EnvelopeTracker(LiveTracker):
    """Shadow-only intervals; errors and elapsed compute time remain in evidence."""
    def __init__(self,core,predictor,**kwargs):
        super().__init__(core,predictor._lower,predictor._scope,predictor._dispatch,**kwargs)
        self.predictor=predictor
        self.support=predictor.support
        self.anchor_events=Buffer()
        self._anchor_ids={}
        self._next_batch=0

    def scheduled(self,output,started):
        super().scheduled(output,started)
        if not self.observed:return
        self._next_batch+=1
        self._anchor_ids[id(output)]=self._next_batch
        job=self.pending[id(output)]
        self.anchor_events.append(dict(kind='scheduled',batch=self._next_batch,
            scheduled_at=started,observed_at=self.clock(),work=copy.deepcopy(job['work'])))

    def committed(self,output):
        super().committed(output)
        if not self.observed:return
        batch=self._anchor_ids.pop(id(output))
        self.anchor_events.append(dict(kind='committed',batch=batch,observed_at=self.last_commit))

    def before_add(self,request):
        row,probe=self._begin_record(request)
        if not probe:return
        row.update(status='unavailable',prediction_started=self.clock())
        row['cut']=row['prediction_started']
        try:
            state=self.snapshot()
            row.update(snapshot=state,cut=state['observed_at'],
                snapshot_width_s=state['observed_at']-row['prediction_started'],evaluation_at=self.clock(),
                anchor_count=len(self.anchor_events.rows))
            result=self.predictor.forecast(state,Request(request.request_id,request.num_prompt_tokens,request.max_tokens),
                now=row['evaluation_at'],domain=self.domain)
            row.update(forecast=result,status='predicted' if result['available'] else 'unavailable')
        except (ValueError,RuntimeError) as exc:
            row.update(error=type(exc).__name__+': '+str(exc))
        finally:
            row['prediction_finished']=self.clock()
            row['completion_age_s']=row['prediction_finished']-row['cut']
            row['observer_elapsed_s']=row['prediction_finished']-row['prediction_started']
            row['completion_age_passed']=row['completion_age_s']<=self.support['policy']['max_age_s']
            row['fallback_required']=not row['completion_age_passed'] or row['forecast'] is None or row['forecast']['fallback_required']

    def capture(self):
        result=super().capture()
        result.update(schema='envelope-shadow-records-v1',support_sha256=self.support['sha256'],
            anchor_events=self.anchor_events.rows,anchor_overflow=self.anchor_events.overflow,
            actuation_authorized=False,native_coverage_qualified=False)
        return result
