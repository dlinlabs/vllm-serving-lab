"""Synthetic design checks with refitted baselines; never GPU evidence or power.

The small causal simulator below implements the declared serial FIFO toy only.
It must pass parity checks against prefill_design.simulate, but that does not
establish parity with vLLM. Measured scheduler/mode-aware rollout is still pending.
"""
from collections import deque
import copy
import math

import numpy as np
from scipy.optimize import nnls

from prefill_design import COEF, requests_for, scored_steps
from prefill_identifiability import features
from prefill_validation import comparison_plan, structural_error_check

FAMILIES = {'M0': (0, 1, 2), 'M1': (0, 1, 2, 3),
            'M0Z': (0, 1, 2, 4), 'M2': (0, 1, 2, 3, 4)}


def rollout(case, coefficients):
    """Causal FIFO toy: use only arrivals, own progress and five fixed costs."""
    coef = np.asarray(coefficients, float)
    if coef.shape != (5,) or not np.isfinite(coef).all() or (coef < 0).any() or coef[0] <= 0:
        raise ValueError('Expected five nonnegative finite costs and positive intercept')
    d = case['background_decoders']
    if not isinstance(d, int) or not 0 <= d < 2048:
        raise ValueError('Invalid background count')
    foreground = requests_for(case)
    backgrounds = [dict(id=f'background-{i}', tokens=1024, output=256, at=0.) for i in range(d)]
    pending = backgrounds if d else foreground
    pending = sorted(pending, key=lambda r: r['at'])
    index = 0
    queue, running, outcomes, steps = deque(), [], {}, []
    now, release = 0., None if d else 0.

    def enqueue(r):
        queue.append(dict(r, remaining=r['tokens'], computed=0.))

    while index < len(pending) or queue or running:
        if len(steps) > 200000:
            raise RuntimeError('Toy did not drain')
        if not queue and not running and index < len(pending):
            now = max(now, pending[index]['at'])
        while index < len(pending) and pending[index]['at'] <= now + 1e-12:
            enqueue(pending[index]); index += 1
        budget, chunks, finished = 2048 - len(running), [], []
        for _ in range(len(queue)):
            item = queue.popleft()
            if budget <= 0:
                queue.appendleft(item); break
            q = min(budget, item['remaining'])
            chunks.append((item['id'], q, item['computed']))
            item['computed'] += q; item['remaining'] -= q; budget -= q
            if item['remaining'] == 0:
                finished.append(item)
            else:
                queue.appendleft(item); break
        step = dict(start=now, prefill=sum(q for _, q, _ in chunks),
                    decode=len(running), decode_context=sum(r['context'] for r in running), chunks=chunks)
        end = now + float(np.dot(features(step), coef))
        step['end'] = end; steps.append(step)
        for r in running:
            r['remaining_output'] -= 1; r['context'] += 1
            if not r['remaining_output']:
                outcomes[r['id']]['completed'] = end
        running = [r for r in running if r['remaining_output']]
        for r in finished:
            outcomes[r['id']] = dict(first=end, arrival=r['at'], completed=None)
            running.append(dict(id=r['id'], remaining_output=r['output'] - 1, context=r['tokens'] + 1))
        while index < len(pending) and pending[index]['at'] < end:
            enqueue(pending[index]); index += 1
        now = end
        if release is None and all(r['id'] in outcomes for r in backgrounds):
            release = now
            pending = sorted([dict(r, at=r['at'] + release) for r in foreground], key=lambda r: r['at'])
            index = 0
    targets = {r['id']: dict(ttft_s=outcomes[r['id']]['first'] - outcomes[r['id']]['arrival'],
                            first_s=outcomes[r['id']]['first'] - release,
                            completed_s=outcomes[r['id']]['completed'] - release) for r in foreground}
    result = dict(foreground=targets, release_s=release, wall_s=now,
                  scored_step_window=dict(start_s=release, end_s=max(outcomes[r['id']]['completed'] for r in foreground)))
    return result, scored_steps(steps, result)


def collect_training(cases, truth):
    """Exclude every held cell before generating responses or choosing scales."""
    return {c['id']: rollout(c, truth) for c in cases if c['split'] == 'calibration'}


def fit(training, family):
    columns = FAMILIES[family]
    if len(training) < 2:
        raise ValueError('Need multiple calibration cells')
    x, y = [], []
    for _, steps in training.values():
        weight = 1 / math.sqrt(len(steps))
        x.extend(np.asarray([features(s) for s in steps])[:, columns] * weight)
        y.extend(np.asarray([s['end'] - s['start'] for s in steps]) * weight)
    x, y = np.asarray(x), np.asarray(y)
    scale = np.linalg.norm(x, axis=0)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (scale <= 0).any():
        raise ValueError('Invalid or unsupported training rows')
    normalized = x / scale
    if np.linalg.matrix_rank(normalized) != len(columns) or np.linalg.cond(normalized) > 1e4:
        raise ValueError('Unsupported training matrix')
    coef = np.zeros(5)
    coef[list(columns)] = nnls(normalized, y)[0] / scale
    return coef


def conditional_errors(observed, coefficients):
    result, steps = observed
    predicted = np.asarray([np.dot(features(s), coefficients) for s in steps])
    actual = np.asarray([s['end'] - s['start'] for s in steps])
    prefill = np.asarray([s['prefill'] > 0 for s in steps])
    cumulative = []
    for r in result['foreground'].values():
        mask = np.asarray([s['end'] <= result['release_s'] + r['first_s'] + 1e-12 for s in steps])
        cumulative.append(abs(float(predicted[mask].sum() - actual[mask].sum())))
    return {'decode': float(np.abs(predicted[~prefill] - actual[~prefill]).mean()),
            'prefill': float(np.abs(predicted[prefill] - actual[prefill]).mean()),
            'cumulative': float(np.mean(cumulative))}


def select(training):
    """Training-only LOCO decisions. M0Z is an attribution ablation, not a selector."""
    errors = {family: {} for family in FAMILIES}
    for held, observed in training.items():
        remaining = {k: v for k, v in training.items() if k != held}
        for family in FAMILIES:
            errors[family][held] = conditional_errors(observed, fit(remaining, family))

    def retain(candidate, baseline, primary, guards):
        for target in primary:
            a = np.mean([v[target] for v in errors[candidate].values()])
            b = np.mean([v[target] for v in errors[baseline].values()])
            # Exact synthetic fits have numerical residuals, not learnable error.
            if b <= 1e-12 or b - a < .1 * b:
                return False
        return all(errors[candidate][case][target] - b <= max(.1 * b, .001)
                   for case, values in errors[baseline].items() for target, b in values.items() if target in guards)

    base = 'M1' if retain('M1', 'M0', ('decode',), ('prefill', 'cumulative')) else 'M0'
    selected = 'M2' if retain('M2', base, ('prefill', 'cumulative'), ('decode', 'prefill', 'cumulative')) else base
    return dict(selected=selected, retained_base=base, loco_errors=errors)


def synthetic_check(cases, *, quadratic, context_cost=0., scale=1., jitter_s=0.):
    if not all(math.isfinite(v) for v in (quadratic, context_cost, scale, jitter_s)) or min(quadratic, context_cost, jitter_s) < 0 or scale <= 0:
        raise ValueError('Invalid synthetic scenario')
    truth = np.asarray([*COEF, context_cost, 2 * quadratic]) * scale
    training = collect_training(cases, truth)
    selection = select(training)
    fitted = {f: fit(training, f) for f in FAMILIES}
    sentinels = [c for c in cases if c['sentinel']]

    def predictions(coef):
        return {c['id']: rollout(c, coef)[0]['foreground']['sentinel']['ttft_s'] for c in sentinels}

    observed = predictions(truth)
    predicted = {f: predictions(coef) for f, coef in fitted.items()}
    raw = predictions(np.asarray([*COEF, 0., 0.]))
    family = selection['selected']
    plan = comparison_plan(family, selection['retained_base'], predicted[family], predicted['M0'], raw,
                           {k: jitter_s for k in observed}, calibrated_m1=predicted['M1'])
    # Five identical deterministic entries exercise the 8x5 component contract.
    # They are NOT independently generated replicates or uncertainty estimates.
    errors = {f: {k: [v - observed[k]] * 5 for k, v in pred.items()} for f, pred in predicted.items()}
    comparators = {name: errors[name.removeprefix('calibrated_')] for name in plan['required_structural_comparators']}
    gate = structural_error_check(plan, errors[family], comparators)
    return dict(quadratic=quadratic, context_cost=context_cost, scale=scale, jitter_s=jitter_s,
                selection=selection, coefficients={f: v.tolist() for f, v in fitted.items()},
                training_cells=sorted(training), truth_s=observed, predictions_s=predicted,
                raw_B2048_s=raw, comparison_plan=plan, gate=gate,
                synthetic=True, statistical_power_estimated=False, complete_experiment_passed=False,
                repeated_errors='five identical deterministic values; not independent replicates')


def legacy_cases(cases):
    original = copy.deepcopy(cases)
    for c in original:
        if c['sentinel'] and len(c['foreground_tokens']) == 4:
            c['foreground_tokens'] = [4096] * 4
    return original
