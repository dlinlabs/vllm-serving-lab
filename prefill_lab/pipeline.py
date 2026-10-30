"""Opt-in V2 stage diagnostics. No fitted costs, predictive rollout or GPU qualification."""
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import inspect
import os
import threading
import time

from .contracts import Buffer, finite

ACTIVE = ContextVar('prefill_pipeline_stage', default=None)
GPU_STAGES = ('forward', 'sample_tokens', 'logits_sample', 'copy_enqueue', 'post_update')
STAGES = GPU_STAGES + ('output_ready', 'commit')


def bind_forward(runner,recorder,entry):
    if entry is None or recorder.stages is None:return
    if getattr(runner,'_prefill_stage_binding',None) is not None:
        raise RuntimeError('Previous forward has not sampled')
    runner._prefill_stage_binding=(recorder,entry[0]['step_id'])


class StageRecorder:
    def __init__(self, torch, capacity=8192, clock=time.monotonic):
        self.torch, self.clock = torch, clock
        self.buffer, self.by_id, self.pending, self.errors = Buffer(capacity), {}, [], []

    def forward_begin(self, row, start, end, stream):
        key = row['step_id']
        if key in self.by_id:raise RuntimeError('Duplicate pipeline step')
        record = dict(step_id=key, counts=dict(row['counts']), stages={}, process_id=os.getpid())
        self.buffer.append(record);self.by_id[key] = record
        record['stages']['forward'] = dict(host_start=row['host_start'], host_end=None,
            thread_id=threading.get_ident(), stream_handle=getattr(stream,'cuda_stream',None), gpu_elapsed_s=None)
        self.pending.append((record['stages']['forward'], start, end))

    def forward_end(self, row):
        self.by_id[row['step_id']]['stages']['forward']['host_end'] = row['host_end']

    @contextmanager
    def span(self, key, name, stream=None, profiler=None):
        record = self.by_id[key]
        if name in record['stages']:raise RuntimeError('Duplicate pipeline stage: '+name)
        scope = None
        if profiler is not None:
            scope = self.torch.profiler.record_function('prefill_lab_stage/'+name+'/'+key)
            scope.__enter__()
        row = dict(host_start=self.clock(),host_end=None,thread_id=threading.get_ident())
        record['stages'][name] = row
        pair = None
        if stream is not None:
            start,end = self.torch.cuda.Event(enable_timing=True),self.torch.cuda.Event(enable_timing=True)
            start.record(stream);pair=(start,end)
            row.update(stream_handle=getattr(stream,'cuda_stream',None),gpu_elapsed_s=None)
        try:
            yield record
        except BaseException as exc:
            row['error_type']=type(exc).__name__
            raise
        finally:
            if pair is not None:
                pair[1].record(stream);self.pending.append((row,*pair))
            row['host_end']=self.clock()
            if scope is not None:scope.__exit__(None,None,None)

    def committed(self, key, results):
        counts = {}
        for batch in results.values():
            for item in batch.outputs:
                n=len(item.new_token_ids)
                if n:counts[item.request_id]=counts.get(item.request_id,0)+n
        self.by_id[key]['committed_outputs']=counts

    def capture(self):
        remaining=[]
        for row,start,end in self.pending:
            try:
                if not end.query():remaining.append((row,start,end));continue
                value=start.elapsed_time(end)/1000
                if not finite(value) or value<0:raise ValueError('Invalid pipeline CUDA span')
                row['gpu_elapsed_s']=value
            except (RuntimeError,ValueError) as exc:
                row['error_type']=type(exc).__name__
        self.pending=remaining
        return dict(schema='pipeline-stage-v1',process_id=os.getpid(),rows=self.buffer.rows,pending=len(remaining),
            overflow=self.buffer.overflow,errors=list(self.errors),cuda_scope_qualified=False,model_accepted=False,
            gpu_label_scope='Stream-local elapsed spans, including enqueue gaps and copy-stream waits; nested spans are not additive')


def binding(runner, current):
    rec=current()
    if rec is None or rec.stages is None or not rec.active:return None
    bound=getattr(runner,'_prefill_stage_binding',None)
    if bound is None or bound[0] is not rec:raise RuntimeError('Sampling lacks its matching forward step')
    return bound


def install(torch, runner_class, output_class, current):
    """Install against the source-pinned V2/AsyncOutput classes only."""
    native_tokens,native_sample,native_post=runner_class.sample_tokens,runner_class.sample,runner_class.postprocess_sampled
    native_init,native_output=output_class.__init__,output_class.get_output
    signature=inspect.signature(native_init)

    @functools.wraps(native_tokens)
    def tokens(runner,*args,**kwargs):
        bound=binding(runner,current)
        if bound is None:return native_tokens(runner,*args,**kwargs)
        rec,key=bound;token=ACTIVE.set(bound)
        try:
            with rec.stages.span(key,'sample_tokens',runner.main_stream,rec.profile):
                result=native_tokens(runner,*args,**kwargs)
            if not isinstance(result,output_class):raise RuntimeError('Expected native V2 AsyncOutput')
            rec.profile_after_sample(key)
            return result
        finally:
            ACTIVE.reset(token);runner._prefill_stage_binding=None

    def stage_method(native,name):
        @functools.wraps(native)
        def wrapped(runner,*args,**kwargs):
            bound=ACTIVE.get()
            if bound is None:return native(runner,*args,**kwargs)
            rec,key=bound
            with rec.stages.span(key,name,runner.main_stream,rec.profile) as record:
                if name=='logits_sample':
                    batch=inspect.signature(native).bind(runner,*args,**kwargs).arguments['input_batch']
                    record.update(sample_request_ids=list(batch.req_ids),sampled_requests=int(batch.num_reqs),
                        logits_rows=int(batch.logits_indices.numel()))
                return native(runner,*args,**kwargs)
        return wrapped

    @functools.wraps(native_init)
    def initialize(output,*args,**kwargs):
        bound=ACTIVE.get()
        if bound is None:return native_init(output,*args,**kwargs)
        rec,key=bound;params=signature.bind(output,*args,**kwargs).arguments
        ids=params['model_runner_output'].req_ids
        with rec.stages.span(key,'copy_enqueue',params['copy_stream'],rec.profile) as record:
            sampled=params['sampler_output'];num=params['num_sampled_tokens']
            record.update(copy_request_ids=list(ids),primary_copy_bytes=int(
                sampled.sampled_token_ids.numel()*sampled.sampled_token_ids.element_size()+num.numel()*num.element_size()),
                extra_copy_payloads=bool(sampled.logprobs_tensors is not None or sampled.num_nans is not None
                    or sampled.sampling_mask_tensors is not None or params.get('routed_experts') is not None
                    or params.get('check_ep_fault',False) or params['model_runner_output'].prompt_logprobs_dict))
            result=native_init(output,*args,**kwargs)
        output._prefill_stage_binding=bound
        return result

    @functools.wraps(native_output)
    def get_output(output,*args,**kwargs):
        bound=getattr(output,'_prefill_stage_binding',None)
        if bound is None:return native_output(output,*args,**kwargs)
        rec,key=bound
        with rec.stages.span(key,'output_ready',profiler=rec.profile) as record:
            result=native_output(output,*args,**kwargs)
            record['output_request_ids']=list(result.req_ids)
        return result

    runner_class.sample_tokens=tokens
    runner_class.sample=stage_method(native_sample,'logits_sample')
    runner_class.postprocess_sampled=stage_method(native_post,'post_update')
    output_class.__init__,output_class.get_output=initialize,get_output


def preflight(meta):
    workers=meta['workers']
    if (len(workers)!=1 or workers[0].get('adapter')!='V2'
        or workers[0].get('pipeline_stages') is not True or type(meta.get('process_id')) is not int or meta['process_id']<1
        or meta['process_id']!=workers[0].get('pid')
        or meta.get('executor_class')!='vllm.v1.executor.uniproc_executor.UniProcExecutor'
        or meta.get('async_scheduling') is not True or meta.get('max_concurrent_batches')!=2):
        raise ValueError('Stage diagnostic requires reviewed V2, same-process UniProcExecutor and two async slots')


def audit(capture, directory=None):
    """Integrity only; original CUDA/full-model acceptance is never promoted."""
    issues=[]
    result=dict(valid=False,issues=issues,cuda_scope_qualified=False,model_accepted=False)
    try:
        workers=capture['workers']
        if len(workers)!=1:raise ValueError('Need one stage worker')
        proof=workers[0]['pipeline'];rows=proof['rows'];steps=capture['steps']
        if proof['schema']!='pipeline-stage-v1' or proof['pending'] or proof['overflow'] or proof['errors']:
            raise ValueError('Incomplete/overflowed pipeline evidence')
        if proof.get('cuda_scope_qualified') is not False or proof.get('model_accepted') is not False:
            raise ValueError('Pipeline evidence cannot grant GPU/model acceptance')
        if len(rows)!=len(steps) or len({r['step_id'] for r in rows})!=len(rows):
            raise ValueError('Missing/duplicate pipeline steps')
        if [r['step_id'] for r in rows]!=[s['step_id'] for s in steps]:
            raise ValueError('Pipeline submission order mismatch')
        forwards={r['step_id']:r for r in workers[0]['rows']}
        by_id={r['step_id']:r for r in rows};emitted={};first={}
        for step in steps:
            row=by_id[step['step_id']];stages=row['stages']
            if row['counts']!={w['request_id']:w['tokens'] for w in step['work']} or set(stages)!=set(STAGES):
                raise ValueError('Pipeline work/stage mismatch')
            for name,s in stages.items():
                if s.get('error_type') or not finite(s['host_start']) or not finite(s['host_end']) or s['host_end']<s['host_start']:
                    raise ValueError('Invalid pipeline CPU stage')
                if name in GPU_STAGES and (not finite(s['gpu_elapsed_s']) or s['gpu_elapsed_s']<0):
                    raise ValueError('Missing/invalid pipeline CUDA span')
            if len({s['thread_id'] for s in stages.values()})!=1 or type(row['process_id']) is not int or row['process_id']<1 or row['process_id']!=proof['process_id']:
                raise ValueError('Pipeline CPU process/thread identity mismatch')
            f,t,l,c,p,o,k=(stages[n] for n in STAGES)
            original=forwards[row['step_id']]
            if (f['host_start'],f['host_end'],f['gpu_elapsed_s'])!=(original['host_start'],original['host_end'],original['gpu_forward_s']):
                raise ValueError('Pipeline forward does not match original worker events')
            main=f['stream_handle']
            if type(main) is not int or any(stages[n]['stream_handle']!=main for n in ('sample_tokens','logits_sample','post_update')):
                raise ValueError('Pipeline main-stream identity mismatch')
            if type(c['stream_handle']) is not int or c['stream_handle']==main:
                raise ValueError('Pipeline requires the native separate copy stream')
            if not (step['start']<=f['host_start']<=f['host_end']<=t['host_start']<=l['host_start']
                <=l['host_end']<=c['host_start']<=c['host_end']<=p['host_start']<=p['host_end']<=t['host_end']
                <=o['host_start']<=o['host_end']<=k['host_start']<=k['host_end']<=step['end']):
                raise ValueError('Pipeline CPU dependency order mismatch')
            for field in ('sample_request_ids','copy_request_ids','output_request_ids'):
                ids=row[field]
                if len(ids)!=len(set(ids)) or set(ids)!=set(row['counts']):raise ValueError('Pipeline request identity mismatch')
            if row['sampled_requests']!=len(row['counts']) or type(row['logits_rows']) is not int or row['logits_rows']<1:
                raise ValueError('Invalid sampling work')
            if type(row['primary_copy_bytes']) is not int or row['primary_copy_bytes']<1:
                raise ValueError('Invalid primary copy bytes')
            for rid,n in row['committed_outputs'].items():
                if rid not in row['counts'] or type(n) is not int or n<1:raise ValueError('Invalid commit output')
                emitted[rid]=emitted.get(rid,0)+n;first.setdefault(rid,step['end'])
        for rid,r in capture['requests'].items():
            if emitted.get(rid,0)!=r['output_tokens'] or first.get(rid)!=r['first']:
                raise ValueError('Committed output/first-token accounting mismatch')
        if capture['observer']=='deep':
            from .deep_diagnostic import profile_audit
            import json
            from pathlib import Path
            if directory is None or not profile_audit(capture,directory)['valid']:raise ValueError('Missing foreground profile')
            profile=workers[0]['deep_profile'];keys=profile['step_ids']
            if profile.get('stage_step_ids')!=keys:raise ValueError('Profiler did not include fifth token/output stage')
            stop=profile['profile_stop_span'];fifth=by_id[keys[-1]]['stages']
            if not all(finite(stop.get(k)) for k in ('host_start','stop_return','export_return')) or not (
                fifth['sample_tokens']['host_end']<=stop['host_start']<=stop['stop_return']<=stop['export_return']<=fifth['output_ready']['host_start']):
                raise ValueError('Profiler stop/export boundary mismatch')
            events=json.loads((Path(directory)/profile['artifact']['name']).read_text())['traceEvents']
            from collections import Counter
            found=Counter(e['name'] for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('prefill_lab_stage/'))
            expected=Counter('prefill_lab_stage/'+name+'/'+key for key in keys for name in ('sample_tokens','logits_sample','copy_enqueue','post_update'))
            # output_ready/commit may also appear before profiler stop; check exact measured GPU enqueue stages.
            found={name:n for name,n in found.items() if name.split('/')[1] in ('sample_tokens','logits_sample','copy_enqueue','post_update')}
            if found!=expected:raise ValueError('Missing/duplicate/unexpected token/output profile marker')
        result.update(valid=True,step_count=len(rows),interpretation='Stage identity/order/completion only; no full CUDA scope or predictive-model acceptance')
    except (ValueError,KeyError,TypeError,OSError) as exc:
        issues.append(str(exc))
    return result
