"""Frozen ITL bucket comparisons; exporter counters are not GPU step timers."""
import json
import math
from pathlib import Path
import re
import numpy as np

COUNT_FLOOR=5.
JOINT_FACTOR_LIMIT=2.  # engineering discrepancy flag, not a statistical acceptance test


def snapshot(raw):
    series={}
    for labels,value in re.findall(r'^vllm:inter_token_latency_seconds_bucket\{([^}]+)\}\s+(\S+)',raw,re.M):
        pairs=re.findall(r'(\w+)="((?:\\.|[^"\\])*)"',labels)
        if len(dict(pairs))!=len(pairs) or 'le' not in dict(pairs):raise ValueError('Malformed ITL labels')
        bound=float(dict(pairs)['le']);count=float(value)
        if math.isnan(bound) or bound<=0 or not math.isfinite(count) or count<0:raise ValueError('Invalid ITL counter')
        key=tuple(sorted((k,v) for k,v in pairs if k!='le'))
        buckets=series.setdefault(key,{})
        if bound in buckets:raise ValueError('Duplicate ITL series/bucket')
        buckets[bound]=count
    if not series:raise ValueError('Missing ITL histogram')
    bounds=sorted(next(iter(series.values())))
    if bounds[-1]!=math.inf:raise ValueError('Missing infinite ITL bucket')
    for v in series.values():
        if sorted(v)!=bounds or any(v[b]<v[a] for a,b in zip(bounds,bounds[1:])):
            raise ValueError('Inconsistent/nonmonotone ITL buckets')
    return series,bounds


def exporter_histogram(path,start,end):
    """First/last scrapes inside the declared window; preserve actual endpoints.

    Reject resets/identity/series changes and failed internal samples instead of
    silently bridging them. Snapshot timestamps still only approximate when the
    exporter observed a token, so this is not exact event-level alignment.
    """
    path=Path(path);t0=json.loads((path/'manifest.json').read_text())['measurement_start']
    samples=[];previous=None;identity=None;epoch=None
    for line in (path/'engine_observations.jsonl').read_text().splitlines():
        r=json.loads(line);t=r['sent_at']-t0
        if not start<=t<=end:continue
        if (r.get('error_type') or r.get('http_status',200)!=200 or not r.get('raw_metrics')
                or r.get('counter_resets') or r.get('disappeared_counters') or r.get('exporter_identity_changed')):
            raise ValueError('Incomplete/reset ITL observation window')
        current,bounds=snapshot(r['raw_metrics'])
        marker=r.get('exporter_identity');ep=r.get('counter_epoch')
        if not marker or ep is None:raise ValueError('Missing ITL identity/epoch')
        if previous is not None:
            if t<=samples[-1][0] or marker!=identity or ep!=epoch or set(current)!=set(previous):
                raise ValueError('ITL time/identity/series changed')
            if any(set(current[k])!=set(previous[k]) or any(current[k][b]<previous[k][b] for b in current[k]) for k in current):
                raise ValueError('ITL buckets changed or reset')
        previous=current;identity=marker;epoch=ep
        samples.append((t,current))
    if len(samples)<2:raise ValueError('Insufficient ITL window samples')
    a,b=samples[0],samples[-1]
    cumulative=np.array([sum(b[1][k][x]-a[1][k][x] for k in a[1]) for x in bounds])
    counts=np.diff(np.r_[0.,cumulative])
    if np.any(counts<0):raise ValueError('Nonmonotone ITL increments')
    return dict(requested_window_s=[start,end],actual_window_s=[a[0],b[0]],samples=len(samples),
        upper_bounds_s=[None if x==math.inf else x for x in bounds],counts=counts.tolist(),total=float(cumulative[-1]),
        endpoint_note='First/last exporter sent timestamps; no interpolation; None upper bound means +Inf')


def simulated_histogram(steps,window,bounds):
    """Weight each step duration by pre-existing decoders, one ITL per decoder.

    This is valid only for this toy's running-first/no-skipped-decode rule.
    """
    edges=np.array([math.inf if b is None else b for b in bounds],float)
    if not len(edges) or edges[-1]!=math.inf or np.any(np.diff(edges)<=0):raise ValueError('Invalid histogram bounds')
    counts=np.zeros(len(edges),dtype=int)
    for s in steps:
        if window[0]<s['end']<=window[1]:
            gap=s['end']-s['start']
            counts[np.searchsorted(edges,gap,side='left')]+=s['decode']
    return dict(upper_bounds_s=bounds,counts=counts.tolist(),total=int(counts.sum()),window_s=list(window))


def discrepancy(observed,predicted):
    if observed['upper_bounds_s']!=predicted['upper_bounds_s']:raise ValueError('Histogram bucket mismatch')
    a=np.array(observed['counts'],float);b=np.array(predicted['counts'],float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<0 or a.sum()<=0 or b.sum()<=0:
        raise ValueError('Invalid/empty histogram')
    # Every disjoint interval starting at25ms is scored, including +Inf; all bins are archived.
    lower=np.r_[0.,[x for x in observed['upper_bounds_s'][:-1]]]
    mask=lower>=.025
    logs=np.log((np.maximum(b,COUNT_FLOOR)/b.sum())/(np.maximum(a,COUNT_FLOOR)/a.sum()))
    score=float(np.max(np.abs(logs[mask])))
    return dict(count_floor=COUNT_FLOOR,scored_lower_bound_s=.025,per_bin_log_rate_ratio=logs.tolist(),
        scored_bin_indices=np.flatnonzero(mask).tolist(),max_abs_log_rate_ratio=score,
        mean_abs_log_rate_ratio=float(np.mean(np.abs(logs[mask]))),worst_bin_rate_factor=math.exp(score),
        joint_factor_limit=JOINT_FACTOR_LIMIT,large_joint_discrepancy=score>math.log(JOINT_FACTOR_LIMIT),
        automatic_model_acceptance=False,
        note='Normalized by total gaps; count-floor5; factor2 engineering flag, not independent-bin significance or model identification')
