"""Development-only one-second flow labels for a planned-cap transition model.

Reuses the immutable capture auditor and causal feature reconstruction. Neither
realized future application times nor future engine observations are model inputs.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from analyze_trace_dynamics import dump, finite, hashes
from analyze_trace_pilot import TERMINAL
from identification_features import Window, extract
from trace_control import validate_schedule

PROTOCOL = 'service-transition-development-v1'
HORIZONS = (1, 2, 5)
FLOWS = ('admitted', 'first', 'exit_awaiting', 'exit_streaming', 'cost_in', 'cost_out')
STATES = ('awaiting_first_content', 'streaming', 'reserved_cost')
ASSUMPTION = ('Premeasurement cap plan with instantaneous application at each integer-second '
              'step; observed future application times are excluded. Non-state covariates '
              'are held at their decision-time values throughout each rollout.')


def flow_windows(records, start):
    """Count disjoint admitted-request flows, including premeasurement survivors."""
    pairs, active = defaultdict(list), {}
    ordered = sorted(enumerate(records), key=lambda p: (p[1].get('monotonic', -math.inf), p[0]))
    for _, row in ordered:
        kind, rid, moment = row.get('event'), row.get('request_id'), row.get('monotonic')
        if rid is None or not finite(moment):
            continue
        moment -= start
        if kind == 'accepted':
            cost = row.get('request_cost')
            if rid in active or not finite(cost) or cost < 0:
                raise ValueError('Invalid admission in flow reconstruction')
            active[rid] = [False, cost]
            pairs['admitted'].append((moment, 1.))
            pairs['cost_in'].append((moment, cost))
        elif kind == 'first_content':
            if rid not in active or active[rid][0]:
                raise ValueError('Invalid first-content flow')
            active[rid][0] = True
            pairs['first'].append((moment, 1.))
        elif kind in TERMINAL and rid in active:
            streaming, cost = active.pop(rid)
            pairs['exit_streaming' if streaming else 'exit_awaiting'].append((moment, 1.))
            pairs['cost_out'].append((moment, cost))
    return {key: Window(pairs[key]) for key in FLOWS}


def planned_caps(schedule, moment, horizon):
    """Left-endpoint controls, zero-delay assumption, not realized future controls."""
    if not finite(moment) or moment < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError('Invalid rollout origin or horizon')
    offsets = [step['at_seconds'] for step in schedule]
    return [schedule[bisect_right(offsets, moment + k) - 1]['budget'] for k in range(horizon)]


def conservation(features, flows):
    a, f, eq, ed, ci, co = (flows[key] for key in FLOWS)
    q, d, r = (features[key] for key in STATES)
    return dict(zip(STATES, (q + a - f - eq, d + f - ed, r + ci - co)))


def build_run(root):
    meta, rows, _ = extract(root, step=1., horizons=HORIZONS, windows=(1., 5.), engine_max_age=3.)
    if not meta['state_reconstruction']['valid'] or meta['e2e_slo'] is not None:
        raise ValueError('Requires reconstructed state and the original TTFT-only objective')
    plan = json.loads((root / 'replay_plan.json').read_text())
    schedule = validate_schedule(plan['budget_schedule'], meta['duration_s'])
    manifest = json.loads((root / 'manifest.json').read_text())
    records = [json.loads(s) for s in (root / 'gateway.jsonl').read_text().splitlines() if s.strip()]
    windows = flow_windows(records, manifest['measurement_start'])
    verified = 0
    for row in rows:
        moment = row['t_s']
        row['planned_caps'] = planned_caps(schedule, moment, max(HORIZONS))
        row['next_flows'] = {key: window.total(moment, moment + 1.) for key, window in windows.items()}
        if row['labels']['1'] is not None:
            rebuilt = conservation(row['features'], row['next_flows'])
            if any(abs(rebuilt[key] - row['labels']['1'][key]) > 1e-6 for key in STATES):
                raise ValueError(f'Flow/state conservation mismatch at {moment}s')
            verified += 1
        # Future realized control is evidence in the raw capture, never an input here.
        del row['control']
    if hashes(root) != meta['raw_sha256']:
        raise ValueError('Raw capture changed during transition extraction')
    meta.update(budget_mode=plan['budget_mode'], conservation_verified_rows=verified)
    return meta, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    roots, output = [p.resolve() for p in args.runs], args.output_dir.resolve()
    if len({p.name for p in roots}) != len(roots):
        parser.error('Run names must be unique')
    if output.exists() or any(output == root or root in output.parents for root in roots):
        parser.error('Choose a fresh output directory outside the captures')
    results = []
    try:
        for root in roots:
            results.append(build_run(root))
            print(f'Qualified and reconstructed: {root.name}', flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True)
    manifests = []
    for meta, rows in results:
        filename = meta['name'] + '.jsonl'
        data = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows).encode()
        (output / filename).write_bytes(data)
        manifests.append(dict(meta=meta, file=filename, rows=len(rows), sha256=hashlib.sha256(data).hexdigest()))
    sources = ('transition_features.py', 'identification_features.py', 'analyze_trace_dynamics.py',
               'analyze_trace_pilot.py', 'trace_control.py')
    dump(output / 'dataset.json', dict(protocol=PROTOCOL, horizons=list(HORIZONS), step_s=1.,
         control_assumption=ASSUMPTION, runs=manifests,
         code_sha256={key: hashlib.sha256(Path(__file__).with_name(key).read_bytes()).hexdigest() for key in sources},
         acceptance=None, c_used=False, note='Retrospective development data. No causal policy claim.'))


if __name__ == '__main__':
    main()
