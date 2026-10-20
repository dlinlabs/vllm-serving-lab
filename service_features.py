"""Causal service context and counter-rate observations on audited v1 tables."""
import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from analyze_trace_dynamics import dump, finite, hashes
from analyze_trace_pilot import TERMINAL
from identification_features import Window
from transition_features import HORIZONS, PROTOCOL, build_run

SERVICE_PROTOCOL = 'service-context-development-v2'
COUNTERS = ('prompt_tokens', 'generation_tokens', 'preemptions')
MAX_COUNTER_AGE_S = 3.
CONTEXT = ('awaiting_input_tokens', 'streaming_input_tokens', 'awaiting_cost', 'streaming_cost',
           'awaiting_age_sum_s', 'streaming_age_sum_s', 'offered_input_tokens_5s')
RATE_FIELDS = tuple(field for key in COUNTERS for field in (key + '_rate', key + '_missing'))


def counter_values(point, key):
    metrics = point.get('metrics')
    metric = metrics.get(key) if isinstance(metrics, dict) else None
    if not isinstance(metric, dict) or metric.get('available') is not True or metric.get('type') != 'counter':
        return None
    samples = metric.get('samples')
    if not isinstance(samples, list) or not samples:
        return None
    values = {}
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        name, labels, value = sample.get('name'), sample.get('labels'), sample.get('value')
        if (not isinstance(name, str) or not name.endswith('_total') or not isinstance(labels, dict) or
                not finite(value) or value < 0 or any(not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items())):
            return None
        identity = (name, tuple(sorted(labels.items())))
        if identity in values:
            return None
        values[identity] = value
    return values


def counter_series(points, start):
    if not finite(start):
        raise ValueError('Counter origin must be finite')
    series = []
    for point in points:
        finished, received, sent = (point.get(k) for k in ('finished_at', 'received_at', 'sent_at'))
        if not finite(finished):
            raise ValueError('Counter sample has no finite availability time')
        epoch = point.get('counter_epoch')
        valid = (point.get('http_status') == 200 and not point.get('error_type') and
                 finite(sent) and finite(received) and sent <= received <= finished and
                 type(epoch) is int and epoch >= 0 and isinstance(point.get('exporter_identity'), dict) and
                 bool(point['exporter_identity']) and not point.get('counter_resets') and
                 not point.get('exporter_identity_changed') and not point.get('disappeared_counters'))
        series.append(dict(available_s=finished - start, sent_s=sent - start if finite(sent) else None,
            received_s=received - start if finite(received) else None, valid=valid, epoch=epoch,
            exporter=point.get('exporter_identity'),
            values={key: counter_values(point, key) if valid else None for key in COUNTERS}))
    series.sort(key=lambda p: p['available_s'])
    return series


def rate_features(series, available_times, moment):
    if not finite(moment):
        raise ValueError('Counter decision time must be finite')
    result = {field: (True if field.endswith('_missing') else None) for field in RATE_FIELDS}
    index = bisect_right(available_times, moment) - 1
    if index < 1:
        return result
    previous, current = series[index - 1:index + 1]
    if (not current['valid'] or not previous['valid'] or current['epoch'] != previous['epoch'] or
            current['exporter'] != previous['exporter'] or moment - current['sent_s'] > MAX_COUNTER_AGE_S):
        return result
    gap = current['received_s'] - previous['received_s']
    send_gap = current['sent_s'] - previous['sent_s']
    rtts = [sample['received_s'] - sample['sent_s'] for sample in (previous, current)]
    if (not 0 < gap <= MAX_COUNTER_AGE_S or not 0 < send_gap <= MAX_COUNTER_AGE_S or
            any(not 0 <= rtt <= MAX_COUNTER_AGE_S for rtt in rtts) or
            previous['received_s'] > current['sent_s']):
        return result
    for key in COUNTERS:
        a, b = previous['values'][key], current['values'][key]
        if a is None or b is None or a.keys() != b.keys():
            continue
        changes = [b[label] - a[label] for label in a]
        if any(delta < 0 for delta in changes):
            continue  # per-series check: aggregation must not hide one engine's reset
        result[key + '_rate'] = sum(changes) / gap
        result[key + '_missing'] = False
    return result


def contexts(records, start, moments):
    """State token/cost composition available at t, not future per-request outcomes."""
    events = sorted(((r['monotonic'] - start, i, r) for i, r in enumerate(records)
                     if finite(r.get('monotonic')) and r.get('request_id') is not None), key=lambda r: r[:2])
    if any(r.get('event') in ('accepted', 'rejected') and
           (type(r.get('estimated_input_tokens')) is not int or r['estimated_input_tokens'] < 0)
           for _, _, r in events):
        raise ValueError('Admission decisions need nonnegative integer input-token estimates')
    offered = Window([(m, r['estimated_input_tokens']) for m, _, r in events
                      if r.get('event') in ('accepted', 'rejected') and finite(r.get('estimated_input_tokens'))])
    active, position, rows = {}, 0, []
    previous_moment = -float('inf')
    for moment in moments:
        if not finite(moment) or moment < previous_moment:
            raise ValueError('Context times must be finite and nondecreasing')
        previous_moment = moment
        while position < len(events) and events[position][0] <= moment:
            at, _, event = events[position]
            position += 1
            rid, kind = event['request_id'], event.get('event')
            if kind == 'accepted':
                tokens, cost = event.get('estimated_input_tokens'), event.get('request_cost')
                if rid in active or type(tokens) is not int or tokens < 0 or not finite(cost) or cost < 0:
                    raise ValueError('Invalid admission context')
                active[rid] = dict(input=tokens, cost=cost, admitted=at, first=None)
            elif kind == 'first_content':
                if rid not in active or active[rid]['first'] is not None:
                    raise ValueError('Invalid first-content context')
                active[rid]['first'] = at
            elif kind in TERMINAL:
                active.pop(rid, None)
        value = dict.fromkeys(CONTEXT, 0.)
        for entry in active.values():
            prefix = 'awaiting' if entry['first'] is None else 'streaming'
            value[prefix + '_input_tokens'] += entry['input']
            value[prefix + '_cost'] += entry['cost']
            value[prefix + '_age_sum_s'] += moment - (entry['admitted'] if entry['first'] is None else entry['first'])
        value['offered_input_tokens_5s'] = offered.total(moment - 5., moment)
        rows.append(value)
    return rows


def augment(root):
    meta, rows = build_run(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    start = manifest['measurement_start']
    records = [json.loads(s) for s in (root / 'gateway.jsonl').read_text().splitlines() if s.strip()]
    points = [json.loads(s) for s in (root / 'engine_observations.jsonl').read_text().splitlines() if s.strip()]
    if any(p.get('run_id') != manifest['run_id'] for p in points):
        raise ValueError('Engine sample belongs to another run')
    series = counter_series(points, start)
    times = [p['available_s'] for p in series]
    context = contexts(records, start, [r['t_s'] for r in rows])
    missing = Counter()
    for row, extra in zip(rows, context):
        if abs(extra['awaiting_cost'] + extra['streaming_cost'] - row['features']['reserved_cost']) > 1e-6:
            raise ValueError('Context costs do not reconcile with audited reservation')
        rates = rate_features(series, times, row['t_s'])
        row['features'].update(extra, **rates)
        missing.update(key for key in COUNTERS if rates[key + '_missing'])
    if hashes(root) != meta['raw_sha256']:
        raise ValueError('Capture changed during service-feature extraction')
    meta['service_context'] = dict(verified_rows=len(rows), missing_rate_rows=dict(missing), counter_max_age_s=MAX_COUNTER_AGE_S)
    return meta, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2, 4))
    args = parser.parse_args()
    sources = ('service_features.py', 'transition_features.py', 'identification_features.py',
               'analyze_trace_dynamics.py', 'analyze_trace_pilot.py', 'trace_control.py')
    code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    roots, output = [r.resolve() for r in args.runs], args.output_dir.resolve()
    if len({r.name for r in roots}) != len(roots) or output.exists() or any(r == output or r in output.parents for r in roots):
        parser.error('Use unique run names and a fresh output outside raw captures')
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(augment, root): i for i, root in enumerate(roots)}
            for future in as_completed(pending):
                index = pending[future]
                results[index] = future.result()
                print('Qualified service context: ' + roots[index].name, flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    entries = []
    if code != {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}:
        parser.error('Extraction source changed while processing captures')
    if len({meta['raw_sha256']['manifest.json'] for meta, _ in results.values()}) != len(results):
        parser.error('Duplicate raw capture under different names')
    output.mkdir(parents=True)
    for index in range(len(roots)):
        meta, rows = results[index]
        data = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows).encode()
        filename = meta['name'] + '.jsonl'
        (output / filename).write_bytes(data)
        entries.append(dict(meta=meta, file=filename, rows=len(rows), sha256=hashlib.sha256(data).hexdigest()))
    dump(output / 'dataset.json', dict(protocol=PROTOCOL, service_protocol=SERVICE_PROTOCOL,
         horizons=list(HORIZONS), step_s=1., runs=entries, acceptance=None, c_used=False,
         code_sha256=code))


if __name__ == '__main__':
    main()
