"""CPU-only, explicitly approximate preregistration scenarios; not a vLLM collector."""
from collections import deque
import math

COEF = (.00918083968026616, .0003168517008371403, .00005341461305169013)


def simulate(requests, budget, *, backgrounds=0, policy='fifo', quadratic=0., steps_out=None, background_output=256, itl_window=None, coefficients=None):
    """Executed FIFO/round-robin toy service, variable request output lengths.

    Foreground offsets are relative to the barrier after all backgrounds first decode.
    Quadratic adds 2*k*attention_pairs; over chunks this is k*(L^2+L), not k*L^2 per chunk.
    All predictions are raw engine elapsed time; frontend/calibration is not fabricated.
    """
    coef=tuple(COEF if coefficients is None else coefficients)
    if len(coef)!=3 or not all(math.isfinite(x) and x>=0 for x in coef) or coef[0]<=0:
        raise ValueError('Invalid step coefficients')
    if budget not in (2048, 8192) or policy not in ('fifo', 'round_robin_512'):
        raise ValueError('Unsupported scenario')
    if (not isinstance(backgrounds,int) or backgrounds < 0 or backgrounds >= budget
            or not math.isfinite(quadratic) or quadratic < 0
            or not isinstance(background_output,int) or background_output < 1):
        raise ValueError('Invalid service parameters')
    if itl_window is not None and (len(itl_window) != 2 or not all(math.isfinite(t) for t in itl_window) or itl_window[0] >= itl_window[1]):
        raise ValueError('Invalid ITL window')
    itl=dict(count=0,gap_100_150ms=0,gap_150_200ms=0,gap_above200ms=0,max_gap_s=None)
    records = [dict(r) for r in requests]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate request IDs')
    for r in records:
        if (not math.isfinite(r['tokens']) or r['tokens'] <= 0
                or not isinstance(r.get('output',128),int) or r.get('output',128) < 1
                or r['at'] < 0 or not math.isfinite(r['at'])):
            raise ValueError('Invalid request')
    bgs = [dict(id=f'background-{i}',tokens=1024,output=background_output,at=0.,role='background') for i in range(backgrounds)]
    all_records = bgs + records
    if len({r['id'] for r in all_records}) != len(all_records):
        raise ValueError('Background identity collision')
    queue, running, outcomes = deque(), [], {}
    pending = sorted(bgs, key=lambda r:r['at']); ai=0; tau=0.; barrier=None if bgs else 0.; steps=0
    def enqueue(r):
        queue.append(dict(r,remaining=float(r['tokens']),computed=0.,arrival=r['at']))
    if not bgs:
        pending = sorted(records, key=lambda r:r['at'])
    while ai < len(pending) or queue or running:
        if steps > 200000:
            raise RuntimeError('Scenario did not drain')
        if not queue and not running and ai < len(pending):
            tau=max(tau,pending[ai]['at'])
        while ai < len(pending) and pending[ai]['at'] <= tau+1e-12:
            enqueue(pending[ai]);ai+=1
        available=budget-len(running); prompt=0.; z=0.; finished=[]; chunks=[]
        visits=len(queue)
        for _ in range(visits):
            item=queue.popleft()
            if available <= 0:
                queue.appendleft(item);break
            q=min(available,item['remaining'],512 if policy=='round_robin_512' else math.inf)
            chunks.append((item['id'],q,item['computed']))
            z+=q*item['computed']+q*(q+1)/2
            item['computed']+=q;item['remaining']-=q;available-=q;prompt+=q
            if item['remaining']<=0:finished.append(item)
            elif policy=='fifo':
                queue.appendleft(item);break
            else:queue.append(item)
        end=tau+coef[0]+coef[1]*len(running)+coef[2]*prompt+2*quadratic*z
        if steps_out is not None:
            steps_out.append(dict(start=tau,end=end,prefill=prompt,decode=len(running),
                                  decode_context=sum(r['context'] for r in running),chunks=chunks))
        for r in running:
            previous=outcomes[r['id']]['last']
            if itl_window is not None and itl_window[0] < end <= itl_window[1]:
                gap=end-previous;itl['count']+=1
                itl['gap_100_150ms']+=int(.1 < gap <= .15)
                itl['gap_150_200ms']+=int(.15 < gap <= .2)
                itl['gap_above200ms']+=int(gap > .2)
                itl['max_gap_s']=max(itl['max_gap_s'] or 0.,gap)
            outcomes[r['id']]['last']=end
            r['remaining_output']-=1;r['context']+=1
        for r in running:
            if r['remaining_output']==0:outcomes[r['id']]['completed']=end
        running=[r for r in running if r['remaining_output']>0]
        for r in finished:
            outcomes[r['id']]=dict(arrival=r['arrival'],first=end,last=end,completed=end if r.get('output',128)==1 else None,role=r.get('role','foreground'))
            if r.get('output',128)>1:running.append(dict(id=r['id'],remaining_output=r.get('output',128)-1,context=r['tokens']+1))
        # Requests arriving during execution wait for the next scheduling boundary.
        while ai < len(pending) and pending[ai]['at'] < end:
            enqueue(pending[ai]);ai+=1
        tau=end;steps+=1
        if barrier is None and all(r['id'] in outcomes for r in bgs):
            barrier=tau
            pending=sorted([dict(r,at=r['at']+barrier) for r in records],key=lambda r:r['at']);ai=0
    barrier=barrier or 0.
    primary={r['id']:dict(first_s=outcomes[r['id']]['first']-barrier,
                         ttft_s=outcomes[r['id']]['first']-outcomes[r['id']]['arrival'],
                         prefill_complete_s=outcomes[r['id']]['first']-barrier,
                         completed_s=outcomes[r['id']]['completed']-barrier)
             for r in records}
    return dict(budget=budget,policy=policy,quadratic=quadratic,backgrounds=backgrounds,
                foreground=primary,scored_step_window=dict(start_s=barrier,end_s=max((outcomes[r['id']]['completed'] for r in records),default=barrier)),itl=itl if itl_window is not None else None,background_output=background_output,wall_s=tau,release_s=barrier,steps=steps,
                assumptions='serial running-first toy engine; first token at prefill completion; no frontend or fitted residual correction')


def requests_for(case):
    req=[dict(id=f'foreground-{i}',tokens=t,at=at/1000,output=128)
         for i,(t,at) in enumerate(zip(case['foreground_tokens'],case['foreground_offsets_ms']))]
    if case.get('sentinel'):
        s=case['sentinel'];req.append(dict(id='sentinel',tokens=s['input_tokens'],at=s['arrival_ms']/1000,output=128,role='sentinel'))
    return req


def contrast_eligibility(candidate, baseline, jitter_by_cell):
    """Freeze calibration-only, per-cell resolution before opening held timing."""
    if (not isinstance(jitter_by_cell,dict) or set(candidate)!=set(baseline)
            or set(candidate)!=set(jitter_by_cell) or len(candidate)!=8):
        raise ValueError('Expected eight matching sentinel predictions and per-cell jitter')
    if not all(math.isfinite(v) and v>=0 for v in [*candidate.values(),*baseline.values(),*jitter_by_cell.values()]):
        raise ValueError('Invalid predicted latency/jitter')
    thresholds={k:max(3*jitter_by_cell[k],.025) for k in candidate}
    eligible=sorted(k for k in candidate if abs(candidate[k]-baseline[k])>thresholds[k])
    return dict(threshold_s_by_cell=thresholds,eligible_cells=eligible,minimum_eligible=4,
        comparison_evaluable=len(eligible)>=4,
        strict_wins_required=math.ceil(.75*len(eligible)) if len(eligible)>=4 else None,
        no_regression_cells=sorted(candidate),
        note='Eligibility frozen before held timing; fewer than4 eligible cells is inconclusive, not passed.')


def scored_steps(steps, result):
    """Release through last foreground completion; exclude preparation/background drain.

    Simulation endpoints are copied exactly. Measured collector endpoints need an
    explicit clock/step-ID contract; this helper is not that GPU collector.
    """
    window=result['scored_step_window']
    return [s for s in steps if s['start'] >= window['start_s'] and s['end'] <= window['end_s']]
