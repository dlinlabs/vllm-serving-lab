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
    counts=np.zeros(len(edges),dtype=int);step_counts=np.zeros(len(edges),dtype=int);decoders=[]
    for s in steps:
        if window[0]<s['end']<=window[1]:
            gap=s['end']-s['start']
            index=np.searchsorted(edges,gap,side='left')
            counts[index]+=s['decode']
            if s['decode']>0:
                step_counts[index]+=1;decoders.append(s['decode'])
    return dict(upper_bounds_s=bounds,counts=counts.tolist(),total=int(counts.sum()),window_s=list(window),
        decode_bearing_steps=len(decoders),decode_bearing_steps_by_bin=step_counts.tolist(),
        mean_gaps_per_decode_bearing_step=float(np.mean(decoders)) if decoders else None,
        gaps_per_step_range=[min(decoders),max(decoders)] if decoders else None,
        clustering_note='Toy step counts only. Exporter gaps are clustered; these are not measured real steps or effective sample size.')


def discrepancy(observed,predicted,*,count_floor=COUNT_FLOOR,scored_bin_indices=None):
    """Descriptive sensitivity only; caller must label the shared floor/reference."""
    if not math.isfinite(count_floor) or count_floor<=0:raise ValueError('Invalid count floor')
    if observed['upper_bounds_s']!=predicted['upper_bounds_s']:raise ValueError('Histogram bucket mismatch')
    a=np.array(observed['counts'],float);b=np.array(predicted['counts'],float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<0 or a.sum()<=0 or b.sum()<=0:
        raise ValueError('Invalid/empty histogram')
    # Every disjoint interval starting at25ms is scored, including +Inf; all bins are archived.
    lower=np.r_[0.,[x for x in observed['upper_bounds_s'][:-1]]]
    mask=lower>=.025
    if scored_bin_indices is not None:
        allowed=np.zeros(len(a),dtype=bool);allowed[scored_bin_indices]=True;mask &= allowed
    if not mask.any():raise ValueError('No scored histogram bins')
    logs=np.log((np.maximum(b,count_floor)/b.sum())/(np.maximum(a,count_floor)/a.sum()))
    score=float(np.max(np.abs(logs[mask])))
    return dict(count_floor=count_floor,scored_lower_bound_s=.025,per_bin_log_rate_ratio=logs.tolist(),
        scored_bin_indices=np.flatnonzero(mask).tolist(),max_abs_log_rate_ratio=score,
        mean_abs_log_rate_ratio=float(np.mean(np.abs(logs[mask]))),worst_bin_rate_factor=math.exp(score),
        joint_factor_limit=JOINT_FACTOR_LIMIT,large_joint_discrepancy=score>math.log(JOINT_FACTOR_LIMIT),
        automatic_model_acceptance=False,
        note='Normalized by total gaps. Factor2 is an uncalibrated engineering flag, not acceptance or independent-bin significance.')


def shared_step_reference(histogram,reference_id='B2048-fifo-k0.0',equivalent_steps=5.):
    n=histogram.get('decode_bearing_steps',0)
    if n<=0 or not math.isfinite(equivalent_steps) or equivalent_steps<=0:raise ValueError('Missing/invalid toy step reference')
    mean=histogram['total']/n
    if not math.isfinite(mean) or mean<=0:raise ValueError('Empty toy gap reference')
    return dict(reference_id=reference_id,equivalent_steps=equivalent_steps,
        mean_toy_gaps_per_step=mean,count_floor=equivalent_steps*mean,
        note='Fixed linear-B2048 reference for this window, shared by ALL scenarios. Equivalent-step normalization only; not observed steps, independent samples, or a statistical noise threshold.')


def support_and_magnitude(observed,predicted,reference,common_positive_bins):
    floor=reference['count_floor']
    magnitude=discrepancy(observed,predicted,count_floor=floor)
    flags=[]
    for i in magnitude['scored_bin_indices']:
        a=observed['counts'][i];b=predicted['counts'][i]
        if (a==0)!=(b==0):
            flags.append(dict(bin_index=i,observed_count=a,predicted_count=b,
                direction='missing_predicted_support' if b==0 else 'extra_predicted_support',
                material_at_shared_floor=max(a,b)>=floor,
                observed_equivalent_steps=a/reference['mean_toy_gaps_per_step'],
                predicted_equivalent_steps=b/reference['mean_toy_gaps_per_step']))
    return dict(shared_reference=reference,support_mismatches=flags,
        material_support_mismatch=any(r['material_at_shared_floor'] for r in flags),
        magnitude=magnitude,
        common_positive_sensitivity=discrepancy(observed,predicted,count_floor=1.,scored_bin_indices=common_positive_bins) if common_positive_bins else None,
        legacy_floor_sensitivity={str(n):discrepancy(observed,predicted,count_floor=float(n)) for n in (1,5)},
        automatic_model_acceptance=False,
        note='Primary evidence retains zero-bin support flags and all-bin magnitude. Common-positive bins exclude support mismatches and are sensitivity ONLY. Neither score nor factor2 accepts a model.')
