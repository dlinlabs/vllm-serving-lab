"""Release-state-only FIFO rollout with a frozen installed dispatcher lookup.

No measured future chunks, modes, output lengths or timing labels are accepted.
A dispatcher mismatch in real captured steps invalidates its scope of use.
"""
from collections import deque
from .analysis import predict


def mode_row(step,dispatch_table):
    counts=[q for _,q,_ in step['chunks']]+[1]*step['decode']
    total=sum(counts)
    uniform=bool(counts) and max(counts)==1
    if dispatch_table.get('schema')=='v2-dispatch-candidates-v1':
        from .dispatch import v2_dispatch
        mode,padded,requests,desc_uniform=v2_dispatch(dispatch_table,len(counts),total,max(counts),step['prefill']>0)
    else:
        mode,padded,requests,desc_uniform=dispatch_table[f'{total}:{int(uniform)}']
    return dict(step,mode=mode,padding_excess=padded-total)


def rollout(case,model,dispatch_table,release_state=None):
    """Use only known barrier state and offered arrivals; generated lengths are forced."""
    bg=[]
    for i in range(case['background_decoders']):
        name=f'background-{i}'
        observed=None
        if release_state is not None:
            observed=next((v for k,v in release_state['requests'].items() if k.endswith('/'+name)),None)
            if observed is None:raise ValueError('Missing measured release background state')
        context=1025 if observed is None else observed['num_tokens']
        remaining=256-(context-1024)
        if not 0<remaining<256:raise ValueError('Background is not decoding at release')
        bg.append(dict(id=name,context=context,left=remaining))
    offered=[dict(id=f'foreground-{i}',tokens=n,out=128,at=a/1000.) for i,(n,a) in enumerate(zip(case['foreground_tokens'],case['foreground_offsets_ms']))]
    if case['sentinel']:offered.append(dict(id='sentinel',tokens=512,out=128,at=case['sentinel']['arrival_ms']/1000.))
    offered.sort(key=lambda r:r['at']);index=0;queue=deque();decoders=bg;now=0.;outcomes={};steps=[]
    while index<len(offered) or queue or any(not r['id'].startswith('background-') for r in decoders):
        if len(steps)>20000:raise RuntimeError('Causal rollout failed to drain')
        if not queue and not decoders and index<len(offered):now=max(now,offered[index]['at'])
        while index<len(offered) and offered[index]['at']<=now+1e-12:
            r=offered[index];queue.append(dict(r,remaining=r['tokens'],position=0));index+=1
        budget=2048-len(decoders);chunks=[];finished=[]
        while queue and budget:
            r=queue[0];q=min(budget,r['remaining']);chunks.append((r['id'],q,r['position']))
            r['position']+=q;r['remaining']-=q;budget-=q
            if not r['remaining']:finished.append(queue.popleft())
            else:break
        step=dict(decode=len(decoders),decode_context=sum(r['context'] for r in decoders),
                  prefill=sum(q for _,q,_ in chunks),attention_pairs=sum(q*l+q*(q+1)/2 for _,q,l in chunks),chunks=chunks)
        if not step['decode'] and not chunks:
            now=offered[index]['at'];continue
        duration=predict(mode_row(step,dispatch_table),model)
        if duration<=0:raise ValueError('Nonpositive fitted step cost')
        end=now+duration;steps.append(dict(step,start=now,end=end))
        for r in decoders:r['left']-=1;r['context']+=1
        decoders=[r for r in decoders if r['left']]
        for r in finished:
            outcomes[r['id']]=dict(ttft_s=end-r['at'],first_s=end)
            decoders.append(dict(id=r['id'],context=r['tokens']+1,left=r['out']-1))
        now=end
    return dict(foreground=outcomes,steps=steps,scope='serial FIFO planning conditional on measured initial state; actual scheduler/mode parity required')


def dispatcher_parity(capture,dispatch_table):
    workers={r['step_id']:r for w in capture['workers'] for r in w['rows']}
    mismatches=[]
    for step in capture['steps']:
        worker=workers.get(step['step_id'])
        if worker is None:mismatches.append(step['step_id']);continue
        expected=mode_row(step,dispatch_table)
        if worker['runtime_mode']!=expected['mode'] or worker['padded_tokens']-step['decode']-step['prefill']!=expected['padding_excess'] or worker.get('microbatching'):
            mismatches.append(step['step_id'])
    return dict(matches=not mismatches,mismatched_step_ids=mismatches)
