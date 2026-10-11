"""Offline diagnosis of a single-host budget pilot; never modifies raw input.

Usage: python analyze_budget_dynamics.py INPUT_DIR --output-dir OUTPUT_DIR
Requires numpy and matplotlib. All percentiles use linear interpolation.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def q(values, p):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.quantile(values, p)) if values else None


def stats(values):
    return {name: q(values, p) for name, p in
            [('p50', .5), ('p95', .95), ('p99', .99), ('max', 1)]}


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def correlation(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ['client.json', 'warmup.json', 'gateway.jsonl', 'manifest.json', 'budget_changes.json']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    manifest = json.loads((root / 'manifest.json').read_text())
    clients = json.loads((root / 'client.json').read_text())['requests']
    warmup = json.loads((root / 'warmup.json').read_text())['requests']
    records = [json.loads(s) for s in (root / 'gateway.jsonl').read_text().splitlines() if s.strip()]
    records = [r for r in records if r['run_id'] == manifest['run_id']]
    start, end = manifest['measurement_start'], manifest['measurement_end']
    duration = end - start
    histories = defaultdict(dict)
    issues = []
    for r in records:
        if 'request_id' in r:
            if r['event'] in histories[r['request_id']]:
                issues.append('Duplicate lifecycle event')
            histories[r['request_id']][r['event']] = r
    ids = [r.get('request_id') for r in clients + warmup]
    if None in ids or len(ids) != len(set(ids)) or set(ids) != set(histories):
        issues.append('Missing, duplicate, or unmatched request IDs')
    joined = []
    for c in clients + warmup:
        h = histories.get(c['request_id'], {})
        outcome = 'completed' if c['success'] else 'rejected' if c['rejected'] else 'failed'
        if [k for k in h if k in ('completed', 'rejected', 'failed', 'cancelled')] != [outcome]:
            issues.append('Outcome mismatch')
        if outcome == 'completed':
            required = ['received', 'accepted', 'backend_sent', 'first_content', 'completed']
            if any(k not in h for k in required):
                issues.append('Missing successful lifecycle event')
                continue
            ts = [h[k]['monotonic'] for k in required]
            if ts != sorted(ts):
                issues.append('Lifecycle order violation')
        if c in warmup:
            continue
        row = dict(request_id=c['request_id'], workload=c['workload'], outcome=outcome,
                   scheduled_s=c['scheduled_at'] - start, started_s=c['started_at'] - start,
                   received_s=h['received']['monotonic'] - start,
                   terminal_s=h[outcome]['monotonic'] - start,
                   scheduling_lag_s=c['started_at'] - c['scheduled_at'],
                   ingress_s=h['received']['monotonic'] - c['started_at'],
                   client_ttft_s=c['ttft'], client_e2e_s=c['end_to_end'],
                   input_tokens=c['input_tokens'], output_tokens=c['output_tokens'],
                   max_output_tokens=c['max_output_tokens'], cost=None, admission_budget=None,
                   received_to_accepted_s=None, accepted_to_sent_s=None,
                   sent_to_first_s=None, gateway_ttft_s=None, delivery_s=None)
        if outcome == 'completed':
            a, sent, first = (h[k]['monotonic'] for k in ['accepted', 'backend_sent', 'first_content'])
            row.update(cost=h['accepted']['request_cost'], admission_budget=h['accepted']['budget'],
                       received_to_accepted_s=a-h['received']['monotonic'],
                       accepted_to_sent_s=sent-a, sent_to_first_s=first-sent,
                       gateway_ttft_s=first-h['received']['monotonic'],
                       delivery_s=c['started_at']+c['ttft']-first)
        joined.append(row)
    snapshots = sorted([r for r in records if r['event'] == 'snapshot'], key=lambda x: x['monotonic'])
    last = snapshots[-1]
    if manifest['status'] != 'collected' or last['active_requests'] or abs(last['current_admitted_cost']) > 1e-6:
        issues.append('Incomplete collection or final drain')
    if any(r.get('telemetry_dropped_records', 0) or r.get('telemetry_write_errors', 0)
           or r.get('tokenizer_fallback_requests', 0) for r in snapshots):
        issues.append('Telemetry loss or tokenizer fallback')
    config = manifest['config']
    changes = [r for r in records if r['event'] == 'budget_updated']
    if [r['budget'] for r in changes] != config['budgets']:
        issues.append('Budget sequence mismatch')
    if len(clients) != math.ceil(config['rps'] * duration):
        issues.append('Measurement count mismatch')
    # Independently compare lifecycle reconstruction with every observed snapshot.
    check_active = {}
    cost_errors = []
    for r in records:
        kind, rid = r['event'], r.get('request_id')
        if kind == 'accepted':
            check_active[rid] = [r['request_cost'], False]
        elif kind == 'first_content' and rid in check_active:
            check_active[rid][1] = True
        elif kind in ('completed', 'failed', 'cancelled', 'rejected'):
            check_active.pop(rid, None)
        elif kind == 'snapshot':
            if (r['current_in_flight'] != len(check_active)
                    or r['awaiting_first_content'] != sum(not v[1] for v in check_active.values())
                    or r['streaming_requests'] != sum(v[1] for v in check_active.values())):
                issues.append('Snapshot/lifecycle state mismatch')
            cost_errors.append(abs(r['current_admitted_cost'] - sum(v[0] for v in check_active.values())))
    if max(cost_errors, default=0) > 1e-6:
        issues.append('Snapshot/lifecycle cost mismatch')
    if issues:
        raise ValueError(sorted(set(issues)))
    successful = [r for r in joined if r['outcome'] == 'completed']
    segments = ['scheduling_lag_s', 'ingress_s', 'received_to_accepted_s', 'accepted_to_sent_s',
                'sent_to_first_s', 'delivery_s', 'gateway_ttft_s', 'client_ttft_s', 'client_e2e_s']
    decomposition = {k: stats([r[k] for r in successful]) for k in segments}
    residual = [r['client_ttft_s'] - sum(r[k] for k in
                ['ingress_s', 'received_to_accepted_s', 'accepted_to_sent_s', 'sent_to_first_s', 'delivery_s'])
                for r in successful]
    top = sorted(successful, key=lambda x: x['client_ttft_s'], reverse=True)[:int(np.ceil(.01*len(successful)))]
    top_contributions = {k: float(np.mean([r[k] for r in top])) for k in
                        ['ingress_s', 'received_to_accepted_s', 'accepted_to_sent_s', 'sent_to_first_s', 'delivery_s', 'client_ttft_s']}
    lifecycle = [r for r in records if start <= r['monotonic'] <= end and 'request_id' in r]
    gaps = []
    for a, b in zip(lifecycle, lifecycle[1:]):
        gap = b['monotonic'] - a['monotonic']
        if gap > .2:
            t = a['monotonic']-start
            gaps.append(dict(start_s=t, end_s=b['monotonic']-start, gap_s=gap,
                             last_event=a['event'], next_event=b['event'],
                             nearest_budget_change_s=min(abs(t-(c['monotonic']-start)) for c in changes[1:]),
                             snapshots_inside_gap=sum(a['monotonic'] < r['monotonic'] < b['monotonic'] for r in snapshots),
                             client_starts_during_gap=sum(a['monotonic'] < r['started_at'] < b['monotonic'] for r in clients)))
    spikes = [(g['start_s']-1, g['end_s']+2) for g in gaps]
    def near_spike(t):
        return any(a <= t <= b for a, b in spikes)
    stage_rows = []
    for i, budget in enumerate(config['budgets']):
        lo, hi = i*config['step_seconds'], (i+1)*config['step_seconds']
        rs = [r for r in joined if lo <= r['scheduled_s'] < hi]
        ss = [r for r in rs if r['outcome']=='completed']
        sn = [r for r in snapshots if start+lo <= r['monotonic'] < start+hi]
        stage_rows.append(dict(stage=i+1, start_s=lo, end_s=hi, budget=budget,
                               attempted=len(rs), succeeded=len(ss), rejected=len(rs)-len(ss),
                               rejection_pct=100*(len(rs)-len(ss))/len(rs),
                               client_ttft_p50_s=q([r['client_ttft_s'] for r in ss], .5),
                               client_ttft_p99_s=q([r['client_ttft_s'] for r in ss], .99),
                               gateway_ttft_p99_s=q([r['gateway_ttft_s'] for r in ss], .99),
                               client_e2e_p99_s=q([r['client_e2e_s'] for r in ss], .99),
                               snapshot_mean_cost=float(np.mean([r['current_admitted_cost'] for r in sn])),
                               snapshot_mean_streaming=float(np.mean([r['streaming_requests'] for r in sn])),
                               snapshot_mean_awaiting=float(np.mean([r['awaiting_first_content'] for r in sn]))))
    workloads = []
    for name in sorted({r['workload'] for r in joined}):
        rs=[r for r in joined if r['workload']==name]; ss=[r for r in rs if r['outcome']=='completed']
        workloads.append(dict(workload=name, attempted=len(rs), success=len(ss),
                              rejection_pct=100*(len(rs)-len(ss))/len(rs),
                              mean_output_tokens=float(np.mean([r['output_tokens'] for r in ss])),
                              output_cap=rs[0]['max_output_tokens'],
                              mean_reserved_cost=float(np.mean([r['cost'] for r in ss])),
                              client_ttft_p99_s=q([r['client_ttft_s'] for r in ss],.99),
                              client_e2e_p99_s=q([r['client_e2e_s'] for r in ss],.99)))
    # Reconstruct state at exact second boundaries from lifecycle events. Only
    # events at/before a boundary inform its features; future labels are separate.
    active, received, budget = {}, {}, config['budgets'][0]
    ordered = sorted(records, key=lambda r:r['monotonic'])
    index=0; aligned=[]
    for sec in range(int(duration)+1):
        boundary=start+sec
        while index<len(ordered) and ordered[index]['monotonic']<=boundary:
            r=ordered[index]; kind=r['event']; rid=r.get('request_id')
            if kind=='received': received[rid]=r['monotonic']
            elif kind=='accepted': active[rid]={'cost':r['request_cost'], 'received':received[rid], 'first':False}
            elif kind=='first_content' and rid in active: active[rid]['first']=True
            elif kind in ('completed','failed','cancelled','rejected'): active.pop(rid,None)
            elif kind=='budget_updated': budget=r['budget']
            index+=1
        prev=[r for r in joined if sec-1<=r['received_s']<sec]
        prior_snap=[r for r in snapshots if r['monotonic']<=boundary]
        ss=[r for r in successful if sec<=r['started_s']<sec+1]
        aligned.append(dict(second=sec, budget=budget,
                            state_cost=sum(r['cost'] for r in active.values()),
                            state_awaiting=sum(not r['first'] for r in active.values()),
                            state_streaming=sum(r['first'] for r in active.values()),
                            state_oldest_age_s=max((boundary-r['received'] for r in active.values() if not r['first']),default=0),
                            last_snapshot_age_s=boundary-prior_snap[-1]['monotonic'] if prior_snap else None,
                            previous_second_received=len(prev),
                            previous_second_rejected=sum(r['outcome']=='rejected' for r in prev),
                            future_second_success_count=len(ss),
                            future_second_client_ttft_p90_s=q([r['client_ttft_s'] for r in ss],.9),
                            future_second_ingress_p90_s=q([r['ingress_s'] for r in ss],.9),
                            future_second_gateway_ttft_p90_s=q([r['gateway_ttft_s'] for r in ss],.9)))
    # Descriptive correlations only. Overlapping windows, fixed traffic and
    # budget trends prevent causal or out-of-sample predictive conclusions.
    predictor_rows=[]
    features=['current_admitted_cost','awaiting_first_content','streaming_requests','oldest_awaiting_first_seconds']
    for lead in [0,1,2,5]:
        for target in ['client_ttft_s','gateway_ttft_s','ingress_s']:
            pairs=[]
            for sn in snapshots:
                t=sn['monotonic']-start
                if t<0 or t+lead+5>duration: continue
                rs=[r for r in successful if t+lead<r['started_s']<=t+lead+5]
                y=q([r[target] for r in rs],.9)
                if y is not None: pairs.append((sn,y,t))
            for feature in features:
                a=[r[0][feature] for r in pairs]; b=[r[1] for r in pairs]
                # Remove phase means to expose confounding by budget level/time.
                da=[];db=[]
                for phase in range(len(config['budgets'])):
                    sub=[r for r in pairs if int(r[2]//config['step_seconds'])==phase]
                    if not sub:continue
                    ma=np.mean([r[0][feature] for r in sub]);mb=np.mean([r[1] for r in sub])
                    da.extend(r[0][feature]-ma for r in sub);db.extend(r[1]-mb for r in sub)
                predictor_rows.append(dict(feature=feature,target=target,lead_s=lead,label_window_s=5,n=len(pairs),
                                           pearson=correlation(a,b),phase_demeaned_pearson=correlation(da,db)))
    # Downward steps: exact reservation history until cost fits the new limit.
    step_rows=[]
    for change in changes[1:]:
        t=change['monotonic']; value=change['current_admitted_cost']; fitted=t
        applicable=sorted((r for r in records if r['monotonic']>t and r['event'] in ('accepted','completed','failed','cancelled')), key=lambda r:r['monotonic'])
        if value>change['budget']:
            for r in applicable:
                if r['event']=='accepted': value+=r['request_cost']
                elif 'accepted' in histories[r['request_id']]: value-=histories[r['request_id']]['accepted']['request_cost']
                if value<=change['budget']:
                    fitted=r['monotonic'];break
        local=[r for r in records if t<=r['monotonic']<t+5 and r['event']=='rejected']
        step_rows.append(dict(applied_s=t-start, old_budget=change['previous_budget'],new_budget=change['budget'],
                              cost_at_update=change['current_admitted_cost'],
                              seconds_until_cost_within_limit=fitted-t,
                              rejected_first_5s=len(local)))
    save_csv(output/'requests.csv',joined)
    save_csv(output/'aligned_seconds.csv',aligned)
    save_csv(output/'stages.csv',stage_rows)
    save_csv(output/'workloads.csv',workloads)
    save_csv(output/'event_gaps.csv',gaps)
    save_csv(output/'budget_responses.csv',step_rows)
    save_csv(output/'signal_correlations.csv',predictor_rows)
    summary=dict(run_id=manifest['run_id'],input_sha256=hashes,valid=True,
                 requests=len(joined),success=len(successful),rejected=len(joined)-len(successful),
                 decomposition=decomposition,top_1pct_count=len(top),top_1pct_mean_segments=top_contributions,
                 snapshot_state_check_count=len(cost_errors),max_snapshot_cost_error=max(cost_errors, default=0),
                 max_decomposition_residual_s=max(abs(x) for x in residual),
                 scheduling_lag_all=stats([r['scheduling_lag_s'] for r in joined]),
                 snapshot_interval=stats([r['interval_seconds'] for r in snapshots if start<=r['monotonic']<=end]),
                 stages=stage_rows,workloads=workloads,event_gaps=gaps,budget_responses=step_rows,
                 outside_gap_windows_client_ttft=stats([r['client_ttft_s'] for r in successful if not near_spike(r['started_s'])]),
                 outside_gap_windows_success_count=sum(not near_spike(r['started_s']) for r in successful),
                 drain_seconds=max(r['terminal_s'] for r in joined)-duration)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot(output,joined,aligned,gaps,changes,start)
    report(output,summary,predictor_rows,root)
    print(json.dumps(dict(valid=True,output=str(output),success=len(successful),event_gaps=len(gaps),
                          top_1pct_mean_segments=top_contributions,budget_responses=step_rows),indent=2))


def plot(output,joined,aligned,gaps,changes,start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    successful=[r for r in joined if r['outcome']=='completed']
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    x=[r['started_s'] for r in successful]
    for ax,key,label,color in zip(axes[:3],['client_ttft_s','ingress_s','gateway_ttft_s'],
                                  ['Client TTFT','Client start → gateway handler','Gateway handler → first content'],
                                  ['#315ca8','#d06c1a','#16877e']):
        ax.scatter(x,[r[key] for r in successful],s=5,alpha=.3,color=color)
        ax.set_ylabel(label+' (s)');ax.grid(alpha=.2)
    axes[3].plot([r['second'] for r in aligned],[r['state_awaiting'] for r in aligned],label='Awaiting first content')
    axes[3].plot([r['second'] for r in aligned],[r['state_streaming'] for r in aligned],label='Streaming')
    axes[3].set_ylabel('Requests');axes[3].legend(loc='upper left');axes[3].grid(alpha=.2)
    for ax in axes:
        for g in gaps:ax.axvspan(g['start_s'],g['end_s'],color='#b83939',alpha=.15)
        for c in changes[1:]:ax.axvline(c['monotonic']-start,color='gray',ls='--',lw=1)
        ax.set_xlim(0,aligned[-1]['second'])
    axes[-1].set_xlabel('Client start time / state boundary relative to measurement start (s)')
    fig.suptitle('pilot-02: latency decomposition; red bands = gateway event gaps; dashed = budget updates')
    fig.tight_layout();fig.savefig(output/'latency_diagnosis.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    sec=[r['second'] for r in aligned]
    axes[0].step(sec,[r['budget'] for r in aligned],where='post',label='Budget',color='#d06c1a')
    axes[0].plot(sec,[r['state_cost'] for r in aligned],label='Reserved cost');axes[0].set_ylabel('Cost units')
    axes[1].plot(sec,[r['previous_second_received'] for r in aligned],label='Gateway received in preceding 1 s')
    axes[1].plot(sec,[r['previous_second_rejected'] for r in aligned],label='Gateway rejected in preceding 1 s')
    for ax in axes:
        ax.legend();ax.grid(alpha=.2)
        for g in gaps:ax.axvspan(g['start_s'],g['end_s'],color='#b83939',alpha=.15)
    axes[1].set_ylabel('Requests');axes[1].set_xlabel('Seconds since measurement start')
    fig.suptitle('pilot-02: exact-boundary lifecycle reconstruction (not interpolated snapshots)')
    fig.tight_layout();fig.savefig(output/'budget_diagnosis.png',dpi=150);plt.close(fig)


def report(output,s,predictors,root):
    def table(columns,rows):
        return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
    d=s['decomposition']; top=s['top_1pct_mean_segments']
    text=f'''# pilot-02 dynamics diagnosis

Run ID: `{s['run_id']}`. Source code commit: `e1e2cf9716f460cabdcc661db9df677656e4817f`.

## Scope and integrity

Single RTX 3090; Qwen/Qwen3-4B-Instruct-2507; max_model_len 8192; vLLM 0.30.0; Python 3.12.3. Open-loop 30 RPS for 150 seconds, budgets 80/96/112/96/80 for 30 seconds each, after 15 warmup requests at 1 RPS and a separate drain. Temperature 0; natural EOS. All 4,515 warmup + measurement IDs are unique and match gateway lifecycles. Measurement: 4,500 attempted, 3,402 completed, 1,098 rejected (24.4%), zero failed. Telemetry loss, write errors and tokenizer fallback are zero; final active work is zero. Last measurement request terminates {s['drain_seconds']:.3f} seconds after arrivals stop. Raw input is not modified; SHA-256 provenance is in summary.json.

## Latency is concentrated in two different paths

Each successful request is joined by ID and decomposed as:

`client TTFT = client start→gateway received + received→accepted + accepted→backend_sent + backend_sent→gateway first_content + gateway first_content→client first_content`

Scheduling lag is separate because client TTFT starts at actual client start. All processes share this host's monotonic clock. Gateway emit timestamps approximate observer timestamps; no engine-internal timestamps exist. Maximum per-request decomposition residual is {s['max_decomposition_residual_s']:.3g} seconds.

'''
    text+=table(['Segment','P50 ms','P95 ms','P99 ms'],[(key,*(f'{d[key][p]*1000:.1f}' for p in ['p50','p95','p99'])) for key in ['ingress_s','received_to_accepted_s','accepted_to_sent_s','sent_to_first_s','delivery_s','gateway_ttft_s','client_ttft_s']])
    text+=f'''
Quantiles from different segments cannot be added or subtracted. Instead, the slowest {s['top_1pct_count']} successful requests (ceil of 1%) have mean client TTFT {top['client_ttft_s']:.3f}s, mean ingress {top['ingress_s']:.3f}s ({100*top['ingress_s']/top['client_ttft_s']:.1f}% of aggregate TTFT), and mean backend_sent→first_content {top['sent_to_first_s']:.3f}s. The latter includes upstream connection/HTTP, vLLM processing/queueing, and gateway observation scheduling; it is not pure GPU time.

![Per-request latency diagnosis](latency_diagnosis.png)

## Five gateway lifecycle event gaps

'''
    text+=table(['Gap start s','Gap end s','Duration s','Client starts inside','Distance to nearest budget step s'],[(f"{g['start_s']:.3f}",f"{g['end_s']:.3f}",f"{g['gap_s']:.3f}",g['client_starts_during_gap'],f"{g['nearest_budget_change_s']:.1f}") for g in s['event_gaps']])
    intervals=np.diff([g['start_s'] for g in s['event_gaps']])
    text+=f'''
Gap-start spacings are {', '.join(f'{v:.2f}s' for v in intervals)}. They do not coincide with budget steps. No snapshots were emitted inside any of the five gaps. Requests continue to start in the separate load process during these gaps; scheduling lag over all arrivals has P99 {s['scheduling_lag_all']['p99']*1000:.1f}ms and maximum {s['scheduling_lag_all']['max']*1000:.1f}ms. A temporary slowdown of gateway processing/host scheduling is consistent with the evidence. The exact cause is unproven: no CPU profiles, event-loop lag, GC timing, or vLLM internal timing were recorded. Transport queueing and resource contention are still possible contributors.

Gateway code creates a fresh httpx.AsyncClient for each admitted request. accepted→backend_sent has a median {d['accepted_to_sent_s']['p50']*1000:.1f}ms. This span includes client construction and request building, but does not separately time each operation. Reusing a lifespan-owned client is a plausible experiment; it is not an established fix for the recurring gaps. Other hypotheses include GC pauses and synchronous work. Instrument them before drawing causal conclusions.

As a sensitivity analysis only, excluding starts from 1 second before each event-gap start to 2 seconds after its end leaves {s['outside_gap_windows_success_count']} successes with client P99 TTFT {s['outside_gap_windows_client_ttft']['p99']:.3f}s. These windows were chosen after observing this run. This is not a replacement performance result and exclusions must not be used to claim MPC improvement.

## Budget effects and downward-step behavior

Rows are cohorts by scheduled arrival time, with latency on successful requests only. They include transitions and spikes. Snapshot state means are unweighted sample means, not time-weighted integrals. One run cannot establish steady state, causal latency effects, an optimal budget, or a settling-time constant.

'''
    text+=table(['Window s','Budget','Reject %','Client P99 TTFT s','Client P99 E2E s','Mean cost','Mean streaming'],[(f"{r['start_s']}-{r['end_s']}",r['budget'],f"{r['rejection_pct']:.1f}",f"{r['client_ttft_p99_s']:.3f}",f"{r['client_e2e_p99_s']:.3f}",f"{r['snapshot_mean_cost']:.1f}",f"{r['snapshot_mean_streaming']:.1f}") for r in s['stages']])
    text+='\nHigher budgets coincide with more reserved work and more streaming sequences. Rejection is not monotonic across stages, and client tail latency is confounded by the event gaps. No throughput benefit from increasing budget is established.\n\n'
    text+=table(['Applied s','Budget change','Cost at update','Time until cost ≤ new budget s'],[(f"{r['applied_s']:.3f}",f"{r['old_budget']:.0f} → {r['new_budget']:.0f}",f"{r['cost_at_update']:.2f}",f"{r['seconds_until_cost_within_limit']:.3f}") for r in s['budget_responses']])
    text+='''
The short downward-step delay measures reservation release until cost first fits the new limit, not latency settling. Lowering the budget preserves running requests; new requests are rejected until their projected cost fits. Current admitted cost is the original cost reservation held until completion, not remaining GPU work.

![Budget and gateway arrival behavior](budget_diagnosis.png)

## Output cap and workload selection

'''
    text+=table(['Workload','Reject %','Output cap','Mean actual output','Mean reserved cost'],[(r['workload'],f"{r['rejection_pct']:.1f}",r['output_cap'],f"{r['mean_output_tokens']:.1f}",f"{r['mean_reserved_cost']:.3f}") for r in s['workloads']])
    text+='''
long_output does not generate 512 tokens on average; natural EOS ends it much earlier. Admission still reserves its 512-token cap. This is evidence of cap/realized-length mismatch and unequal rejection by class. It does not establish true GPU cost, and successful-output distributions are selected by admission. Rejected requests have no realized token usage. Do not replace output caps with future actual output lengths during online evaluation.

## Candidate MPC signals

Signal correlations use only snapshot features available at snapshot time and the P90 latency of successful client starts in a future 5-second window. Lead is the delay before the label window starts. Correlations are descriptive: windows overlap, samples are not independent, only one fixed arrival/mix schedule exists, and phase/time trends confound the results. Phase demeaning removes phase means, not all confounders.

'''
    text+=table(['Feature','1-second lead client P90 r','Within-phase r'],[(r['feature'],f"{r['pearson']:.3f}",f"{r['phase_demeaned_pearson']:.3f}") for r in predictors if r['lead_s']==1 and r['target']=='client_ttft_s'])
    text+=f'''
No signal is validated for out-of-sample MPC prediction by these correlations. Awaiting/oldest age may react to an ongoing slowdown rather than warn before it; gateway states cannot see requests still delayed before handler receipt. Cost and streaming counts carry the controlled operating level but are not independent measurements of GPU work. Estimated prompt/output features for rejected traffic are absent from gateway events, so total offered token pressure cannot be recovered exactly from this telemetry alone. Exact-second state reconstruction is an offline view of lifecycle history, not proof that an external controller could fetch fresh state during a gateway stall. Snapshot state counts match lifecycle reconstruction for all {s['snapshot_state_check_count']} snapshots, with maximum cost difference {s['max_snapshot_cost_error']:.3g} (floating-point rounding).

## Next experiment

1. Add gateway event-loop lag, process CPU/RSS and GC pause measurements; keep the real workload and original trace as the baseline. Run an isolated repeated A/B of per-request versus lifespan-owned upstream HTTP clients, keeping pool limits and all other workload/runtime choices documented and controlled. Preserve every request and include failures.
2. After diagnosing the path, collect repeated budget excitation with non-periodic step order/hold durations, and separately vary RPS and request mix. Keep a held-out run for prediction validation; evaluate 1–5-second forecasts against persistence. Identify a usable operating range before enabling MPC.
3. Add online summaries of total offered prompt tokens/output caps before admission (including rejected requests), and define the latency SLO before calculating SLO goodput. Gateway first-content state is a proxy, not actual prefill/decode queue state.
4. Start control with a 1-second interval only as a hypothesis. Current snapshot intervals drift and can be stale; use actual dt and a stale-data fallback. Run shadow mode, measure controller overhead, then compare fixed budget and simple feedback against MPC on identical traces and repeated trials.

## Reproduce

```bash
python analyze_budget_dynamics.py results/pilot-02 --output-dir results/pilot-02-analysis
```

Requires numpy and matplotlib. Artifacts: requests.csv (ID-joined latency decomposition), aligned_seconds.csv (state at second boundaries and explicitly future labels), stages.csv, workloads.csv, event_gaps.csv, budget_responses.csv, signal_correlations.csv, summary.json, and two figures. Missing numerical values are blank in CSV. All raw input hashes are included in summary.json; the script refuses incomplete lifecycle data. This report is one-run diagnosis, not MPC performance evidence.
'''
    (output/'report.md').write_text(text)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    analyze(args.input_dir,args.output_dir)
