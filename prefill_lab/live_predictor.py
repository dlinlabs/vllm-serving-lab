"""Busy-engine continuation of the six-term effective lane, CPU only.

Input is an EngineCore pre-ADD boundary, never future schedules/completions.
Pending service uses a declared commit-anchored effective approximation. A CUDA
copy-event query is an observed readiness bit, NOT Future.done() or a duration.
No online/native/HTTP/cap/MPC qualification is implied by synthetic completion.
"""
import copy
import math

from .async_predictor import Request,Scope,CostLaw
from .causal import mode_row
from .contracts import digest,features,finite,fifo_reference
from .event_clock_calibration import EventClockCosts,MixedCostLaw
from .submit_clock import SubmitCosts,SubmitLaw


def service(model,row):
    return model.launches['forward']['gpu_tail'](row)


def validate_costs(model):
    if type(model) not in (EventClockCosts,SubmitCosts):raise ValueError('Unsupported busy effective cost family')
    for launch,parts in model.launches.items():
        for name,law in parts.items():
            if (launch,name)==('forward','gpu_tail'):
                if type(law) is not MixedCostLaw:raise ValueError('Expected six-term GPU lane')
            elif (launch,name)==('forward','cpu_after') and type(model) is SubmitCosts:
                if type(law) is not SubmitLaw or law.coefficients:raise ValueError('Expected explicit submit-return constants')
            elif type(law) is not CostLaw or any(law.coefficients.values()):
                raise ValueError('Non-lane physical costs are outside busy continuation scope')
    if any(type(law) is not CostLaw or any(law.coefficients.values()) for law in model.scalar.values()):
        raise ValueError('Non-lane physical costs are outside busy continuation scope')


def feature_row(work,requests,dispatch):
    row=features(work,{rid:r['prompt_tokens'] for rid,r in requests.items()})
    row=mode_row(row,dispatch) if work else dict(row,mode='EMPTY',padding_excess=0)
    row['num_requests']=len(work)
    return row


def validate(snapshot,scope=Scope()):
    if (set(snapshot)!={'schema','clock_domain','observed_at','last_commit_at','running','waiting','requests','pending'}
            or snapshot['schema']!='live-engine-state-v1' or not isinstance(snapshot['clock_domain'],str)
            or not snapshot['clock_domain'] or not finite(snapshot['observed_at']) or snapshot['observed_at']<0):
        raise ValueError('Invalid live snapshot/clock schema')
    cut=snapshot['observed_at'];last=snapshot['last_commit_at']
    if last is not None and (not finite(last) or not 0<=last<=cut):raise ValueError('Future/invalid commit anchor')
    requests=snapshot['requests'];ids=snapshot['running']+snapshot['waiting']
    if (not isinstance(requests,dict) or len(ids)!=len(set(ids)) or set(ids)!=set(requests)
            or len(snapshot['running'])>scope.max_seqs or len(snapshot['pending'])>=scope.batch_slots):
        raise ValueError('Incomplete/duplicate/non-boundary live state')
    inflight={rid:0 for rid in ids};emits={rid:0 for rid in ids}
    for job in snapshot['pending']:
        if (set(job)!={'work','scheduled_at','ready'} or type(job['ready']) is not bool
                or not finite(job['scheduled_at']) or not 0<=job['scheduled_at']<=cut):
            raise ValueError('Invalid/future pending job')
        seen=set()
        for w in job['work']:
            if (set(w)!={'request_id','position','tokens'} or w['request_id'] not in requests
                    or w['request_id'] in seen or any(type(w[k]) is not int for k in ('position','tokens'))
                    or w['position']<0 or w['tokens']<=0):raise ValueError('Invalid pending work')
            rid=w['request_id'];r=requests[rid];seen.add(rid);inflight[rid]+=w['tokens']
            if w['position']+w['tokens']!=r['computed']:raise ValueError('Pending position/progress mismatch')
            p=r['prompt_tokens']
            if w['position']<p and w['position']+w['tokens']>p:raise ValueError('Mixed/speculative pending work')
            if w['position']>=p and w['tokens']!=1:raise ValueError('Speculative pending decode')
            emits[rid]+=int(w['position']+w['tokens']>=p)
        if sum(w['tokens'] for w in job['work'])>scope.token_budget:raise ValueError('Pending budget exceeded')
        if not job['work'] and not job['ready']:raise ValueError('Unknown empty-future readiness')
    for rid,r in requests.items():
        if (not isinstance(rid,str) or not rid or set(r)!={'prompt_tokens','max_tokens','num_tokens','computed','placeholders'}
                or any(type(v) is not int or v<0 for v in r.values()) or r['prompt_tokens']<1 or r['max_tokens']<1
                or r['prompt_tokens']+r['max_tokens']>scope.max_model_len):raise ValueError('Invalid request progress')
        out=r['num_tokens']-r['prompt_tokens'];committed=r['computed']-inflight[rid]
        if (not 0<=out<r['max_tokens'] or r['placeholders']!=emits[rid] or committed<0
                or (out>0 and committed!=r['prompt_tokens']+out-1)
                or (out==0 and committed>=r['prompt_tokens'])
                or (rid in snapshot['waiting'] and (r['computed'] or out or r['placeholders']))):
            raise ValueError('Impossible committed/in-flight progress')
    return True


def forecast(snapshot,proposed,model,dispatch,*,scope=Scope(),until_first=True,max_iterations=20000):
    validate(snapshot,scope)
    if type(max_iterations) is not int or max_iterations<1:raise ValueError('Invalid busy iteration bound')
    validate_costs(model)
    if not isinstance(proposed,Request) or proposed.at_s!=0 or proposed.id in snapshot['requests']:
        raise ValueError('One new unique request at the pre-ADD cut is required')
    if proposed.prompt_tokens+proposed.max_tokens>scope.max_model_len:raise ValueError('New request exceeds context')
    if dispatch.get('schema')!='v2-dispatch-candidates-v1':raise ValueError('Unsupported dispatcher')
    state=copy.deepcopy({k:snapshot[k] for k in ('running','waiting','requests')})
    initial_outputs={rid:r['num_tokens']-r['prompt_tokens'] for rid,r in state['requests'].items()}
    rows=state['requests'];rows[proposed.id]=dict(prompt_tokens=proposed.prompt_tokens,max_tokens=proposed.max_tokens,
        num_tokens=proposed.prompt_tokens,computed=0,placeholders=0)
    initial_outputs[proposed.id]=0;state['waiting'].append(proposed.id)
    # No prefix/speculation/preemption: reserve all known forced lengths. This
    # is a conservative KV capacity bound, NOT measured allocator utilization.
    reserved=sum(math.ceil((r['prompt_tokens']+r['max_tokens'])/scope.block_size) for r in rows.values())
    if reserved>.9*scope.kv_blocks:raise ValueError('Less than 10% conservative KV headroom')
    queue=[];now=gpu=0.;steps=[];out=dict(initial_outputs);first=None;finished=None
    for job in snapshot['pending']:
        row=feature_row(job['work'],rows,dispatch);duration=service(model,row) if job['work'] else 0.
        anchor=max(job['scheduled_at'],snapshot['last_commit_at'] or job['scheduled_at'])
        residual=0. if job['ready'] else anchor+duration-snapshot['observed_at']
        if residual<0:raise ValueError('Unready pending batch overdue under effective clock; prediction unavailable')
        gpu=max(gpu,residual)
        queue.append(dict(work=copy.deepcopy(job['work']),emits=[w['request_id'] for w in job['work']
            if w['position']+w['tokens']>=rows[w['request_id']]['prompt_tokens']],ready=gpu,step=None))
    pending_residual=gpu
    for iteration in range(max_iterations):
        if not rows and not queue:break
        executed=False
        if rows:
            counts=fifo_reference(state,scope.token_budget,scope.max_seqs)
            work=[dict(request_id=rid,position=rows[rid]['computed'],tokens=n) for rid,n in counts.items()]
            row=feature_row(work,rows,dispatch);emits=[];executed=bool(work)
            for rid,n in counts.items():
                r=rows[rid]
                if rid in state['waiting']:state['waiting'].remove(rid);state['running'].append(rid)
                r['computed']+=n
                if r['computed']>=r['num_tokens']+r['placeholders']:r['placeholders']+=1;emits.append(rid)
            gpu=max(now,gpu)+(service(model,row) if work else 0.)
            step=dict(work=work,scheduled_s=now,committed_s=None) if work else None
            if step is not None:steps.append(step)
            queue.append(dict(work=work,emits=emits,ready=gpu if work else now,step=step))
            # A pending job has already returned before the observed pre-ADD
            # cut. Charge CPU-return unavailability only to future submissions.
            # It overlaps GPU service, rather than being added to GPU duration.
            if work:now+=model.launches['forward']['cpu_after'](row)
            if len(queue)<scope.batch_slots:
                if not executed and rows:now+=.001
                continue
        job=queue.pop(0);now=max(now,job['ready'])
        if job['step'] is not None:job['step']['committed_s']=now
        for rid in job['emits']:
            r=rows[rid];r['placeholders']-=1;r['num_tokens']+=1;out[rid]+=1
            if rid==proposed.id and first is None:first=now
            if out[rid]==r['max_tokens']:
                if r['placeholders']:raise RuntimeError('Terminal output still has pending placeholder')
                if rid==proposed.id:finished=now
                state['running'].remove(rid);del rows[rid]
        if not executed and rows:now+=.001
        if until_first and first is not None:break
    else:raise RuntimeError('Busy prediction iteration bound exceeded')
    if first is None:raise RuntimeError('New request never produced an output')
    return dict(schema='live-engine-forecast-v1',request_id=proposed.id,ttft_from_cut_s=first,
        completion_from_cut_s=finished,initial_pending_residual_s=pending_residual,kv_reserved_upper_blocks=reserved,
        steps=steps,output_tokens=out,state_sha256=digest(snapshot),cost_sha256=digest(model.to_dict()),
        timing_assumption='copy-ready bit or commit-anchored effective pending debt; not exclusive GPU service',
        stopped_at_first=until_first,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
