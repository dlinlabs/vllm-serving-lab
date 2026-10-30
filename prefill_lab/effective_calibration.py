"""Measured endpoint calibration of a reduced effective-service candidate.

No exclusive physical stage costs are inferred. Only request timestamps are
labels; schedules/modes/states are generated independently by causal rollout.
This pilot cannot accept the physical pipeline, adaptive cap, or MPC.
"""
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from .async_predictor import CostLaw, PipelineCosts, LAUNCHES, PARTS, SCALARS, rollout
from .contracts import audit_capture, digest, finite, save

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/prefill-effective-calibration-v1.json'
PARAMETERS = ('bias', 'prefill', 'decode_context', 'attention_pairs')


def protocol():
    return json.loads(PLAN.read_text())


def validate_manifest(manifest):
    from .tokens import validate
    plan=protocol()
    if (manifest['model'],manifest['revision']) != (plan['model']['name'],plan['model']['revision']):
        raise ValueError('Calibration model/revision differs from frozen plan')
    return validate(manifest,plan['cases'],plan['order'])


def costs(theta, plan):
    """One effective server lane, NOT a measured forward GPU service law.

    CPU submission/copy/commit are collapsed into this effective service. Their
    independent overlap is deliberately not identified. Keep native two-slot
    scheduling/barrier/idle semantics; held endpoint error tests this reduction.
    """
    if set(plan['features']) != set(PARAMETERS) or len(theta) != len(PARAMETERS):
        raise ValueError('Wrong effective parameter count')
    laws = {k: float(v) / plan['features'][k] for k, v in zip(PARAMETERS, theta)}
    launches = {s: {p: CostLaw() for p in PARTS} for s in LAUNCHES}
    launches['forward']['gpu_tail'] = CostLaw(laws)
    return PipelineCosts(launches, {s: CostLaw() for s in SCALARS},
                         'effective endpoint fit; not exclusive GPU/CPU stage costs')


def predictions(theta, plan, runtime):
    from scripts.predict_prefill_async import runtime_inputs
    scope, dispatch = runtime_inputs(runtime)
    if (scope.token_budget, scope.max_seqs, scope.max_model_len) != (2048, 32, 8192):
        raise ValueError('Effective pilot requires B2048/seq32/context8192')
    model = costs(theta, plan)
    result = {}
    for case in plan['cases']:
        pred = rollout(case, model, dispatch, scope=scope)
        result[case['id']] = dict(
            endpoints={rid: dict(ttft_s=r['ttft_s'], completion_s=r['finished_s']-r['arrival_s'])
                       for rid, r in pred['requests'].items() if r['role'] != 'background'},
            background_outlives_foreground=pred['background_outlives_foreground'],
            work_signature=work_signature(pred['steps']))
    return result


def work_signature(steps):
    return digest([[(w['request_id'].rsplit('/',1)[-1],w['position'],w['tokens']) for w in s['work']] for s in steps])


def structure_check(rows, predicted):
    checked = [r for r in rows if r['observer']=='light']
    return bool(checked) and all(r.get('work_signature')==predicted[r['case']]['work_signature'] for r in checked)


def extract(capture, trial, case):
    """Audit native work, then extract queue-post/commit endpoints, never spans."""
    if capture['trial_id'] != trial['trial_id'] or capture['observer'] != trial['observer']:
        raise ValueError('Capture trial/observer mismatch')
    workers = [r for w in capture.get('workers', []) for r in w['rows']]
    audit = audit_capture(capture, workers, require_steps=trial['observer'] != 'off')
    if capture.get('pending_step_ids'):raise ValueError('Unresolved scheduler outputs')
    if not audit['valid']:
        raise ValueError('Invalid capture: ' + '; '.join(audit['problems']))
    expected = {f'background-{i}': (1024, 256, 'background') for i in range(case['background_decoders'])}
    expected.update({f'foreground-{i}': (n, 128, 'foreground') for i, n in enumerate(case['foreground_tokens'])})
    rows = {}
    for rid, r in capture['requests'].items():
        prefix = f'plab/{trial["trial_id"]}/'
        if not rid.startswith(prefix):raise ValueError('Wrong trial request namespace')
        short = rid[len(prefix):]
        if short not in expected or (r['input_tokens'], r['max_tokens'], r['role']) != expected[short]:
            raise ValueError('Request differs from frozen case')
        times = [r[k] for k in ('queue_posted', 'enqueue', 'first', 'finished')]
        if not all(finite(v) for v in times) or times != sorted(times):
            raise ValueError('Invalid request endpoint timestamps')
        if r['role'] != 'background':
            if not finite(capture['release']) or abs(r['queue_posted'] - capture['release']) > 1e-9:
                raise ValueError('Pilot requires simultaneous foreground posting')
            rows[short] = dict(ttft_s=r['first']-r['queue_posted'], completion_s=r['finished']-r['queue_posted'])
    if len(capture['requests']) != len(expected) or len(rows) != len(case['foreground_tokens']):
        raise ValueError('Missing request endpoints')
    foreground_done=max(r['finished'] for r in capture['requests'].values() if r['role']!='background')
    if not finite(capture.get('foreground_done')) or abs(capture['foreground_done']-foreground_done)>1e-9:
        raise ValueError('Foreground completion boundary mismatch')
    for r in capture['requests'].values():
        if r['role']=='background' and (r['first']>capture['release'] or r['finished']<foreground_done):
            raise ValueError('Background barrier/lifetime mismatch')
    if any(v <= 0 for r in rows.values() for v in r.values()):
        raise ValueError('Nonpositive request latency')
    return dict(trial_id=trial['trial_id'], case=trial['case'], observer=trial['observer'],
                stage=trial['stage'], block=trial['block'], endpoints=rows, capture_sha256=digest(capture),
                work_signature=work_signature(capture['steps']) if trial['observer']=='light' else None)


def select(rows, plan, stages, observer):
    """Plan owns membership; unrelated labels are never inspected by fitting."""
    wanted = {t['trial_id']: t for t in plan['order'] if t['stage'] in stages and t['observer'] == observer}
    found = {}
    for row in rows:
        tid = row['trial_id']
        if tid not in wanted:continue
        if tid in found:raise ValueError('Duplicate calibration trial')
        t = wanted[tid]
        if any(row[k] != t[k] for k in ('case', 'observer', 'stage', 'block')):
            raise ValueError('Row contradicts frozen membership')
        case = next(c for c in plan['cases'] if c['id'] == t['case'])
        if set(row['endpoints']) != {f'foreground-{i}' for i in range(len(case['foreground_tokens']))}:
            raise ValueError('Incomplete endpoint identities')
        for value in row['endpoints'].values():
            if set(value) != {'ttft_s', 'completion_s'} or not all(finite(x) and x > 0 for x in value.values()):
                raise ValueError('Invalid endpoint labels')
        found[tid] = row
    if set(found) != set(wanted):raise ValueError('Incomplete frozen trial split')
    return [found[k] for k in sorted(wanted)]


def quality(rows, plan, stages):
    """Paired observer effects and OFF repeat spread, separate from prediction."""
    off = select(rows, plan, stages, 'off');light = select(rows, plan, stages, 'light')
    by_pair = {(r['stage'], r['case'], r['block']): r for r in light}
    gates = plan['gates'];checks = [];groups = {}
    for r in off:
        other = by_pair[(r['stage'], r['case'], r['block'])]
        for rid, endpoint in r['endpoints'].items():
            for metric, value in endpoint.items():
                key = (r['case'], rid, metric)
                groups.setdefault(key, []).append(value)
                delta = abs(other['endpoints'][rid][metric] - value)
                checks.append(dict(trial=r['trial_id'], request=rid, metric=metric, delta_s=delta,
                    passed=delta <= max(gates['observer_absolute_s'], gates['observer_relative'] * value)))
    repeat = []
    for key, values in groups.items():
        median = float(np.median(values));spread = max(values)-min(values)
        floor = gates['ttft_absolute_s'] if key[2] == 'ttft_s' else gates['completion_absolute_s']
        repeat.append(dict(case=key[0], request=key[1], metric=key[2], repeats=len(values),
                           spread_s=spread, passed=spread <= max(floor, gates['repeat_relative_spread']*median)))
    return dict(passed=all(x['passed'] for x in checks+repeat), paired_observer=checks, repeat_spread=repeat,
                scope='OFF retains control hooks; this is incremental LIGHT overhead, not uninstrumented production overhead')


def endpoint_scores(rows, predicted, plan):
    cells = {}
    for r in rows:
        for rid, endpoint in r['endpoints'].items():
            for metric, value in endpoint.items():cells.setdefault((r['case'], rid, metric), []).append(value)
    scores = []
    for (case, rid, metric), values in sorted(cells.items()):
        observed = float(np.median(values));estimate = predicted[case]['endpoints'][rid][metric]
        floor = plan['gates']['ttft_absolute_s' if metric == 'ttft_s' else 'completion_absolute_s']
        limit = max(floor, plan['gates']['relative_error'] * observed)
        scores.append(dict(case=case, request=rid, metric=metric, repeats=len(values),
                           measured_median_s=observed, predicted_s=estimate, error_s=estimate-observed,
                           limit_s=limit, passed=abs(estimate-observed) <= limit))
    return dict(passed=all(s['passed'] for s in scores), cells=scores)


def fit(rows, plan, runtime, *, deadline=None, origin='synthetic'):
    """OFF training endpoints only. Four effective parameters, bounded multistart."""
    if origin not in ('synthetic','native_controlled_endpoints'):raise ValueError('Unknown label origin')
    training = select(rows, plan, {'train'}, 'off')
    train_quality = quality(rows, plan, {'train'})
    if not train_quality['passed']:raise ValueError('Training observer/repeat gate failed')
    # Do not even simulate held cases while choosing parameters.
    train_cases = {r['case'] for r in training}
    training_plan = dict(plan, cases=[c for c in plan['cases'] if c['id'] in train_cases])
    keys, labels, weights = [], [], []
    for r in training:
        for rid, endpoint in sorted(r['endpoints'].items()):
            for metric, value in sorted(endpoint.items()):
                keys.append((r['case'], rid, metric));labels.append(value)
                floor = plan['gates']['ttft_absolute_s' if metric == 'ttft_s' else 'completion_absolute_s']
                weights.append(1 / (max(value, floor) * math.sqrt(len(r['endpoints']) * 2)))
    labels, weights = np.array(labels), np.array(weights)
    cache = {}
    def residual(theta):
        if deadline is not None and time.monotonic() >= deadline:raise TimeoutError('Calibration deadline expired')
        key = tuple(theta)
        if key not in cache:
            pred = predictions(theta, training_plan, runtime)
            cache[key] = (np.array([pred[c]['endpoints'][rid][m] for c,rid,m in keys])-labels)*weights
        return cache[key]
    fits = [least_squares(residual, x, bounds=plan['parameter_bounds_s'], max_nfev=plan['fit_max_nfev'],
                          diff_step=1e-4, ftol=1e-9, xtol=1e-9, gtol=1e-9) for x in plan['starts_s']]
    if any(not f.success for f in fits):raise ValueError('Bounded optimizer did not converge')
    best = min(fits, key=lambda f: float(np.dot(f.fun, f.fun)))
    theta = best.x
    if any(np.max(np.abs(f.x-theta)/np.maximum(theta, 1e-6)) > .05 for f in fits):
        raise ValueError('Multistart parameter disagreement; effective model not identified')
    # RMS column scaling gives an explicit local sensitivity rank, not global proof.
    jac = best.jac;scale = np.linalg.norm(jac, axis=0)
    if np.any(scale <= 1e-12):raise ValueError('Zero sensitivity parameter')
    singular = np.linalg.svd(jac/scale, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jac/scale));condition = float(singular[0]/singular[-1])
    if rank != len(theta) or condition > plan['gates']['condition_limit']:
        raise ValueError('Effective sensitivity rank/condition gate failed')
    lower, upper = plan['parameter_bounds_s']
    if np.any(theta < lower*1.01) or np.any(theta > upper*.99):
        raise ValueError('Parameter hits bound; do not treat it as identified')
    pred = predictions(theta, training_plan, runtime)
    if not structure_check(select(rows, plan, {'train'}, 'light'), pred):
        raise ValueError('Training own-rollout work differs from native evidence')
    scores = endpoint_scores(training, pred, plan)
    if not scores['passed']:raise ValueError('Training endpoint error gate failed')
    frozen = dict(schema='effective-async-freeze-v1', data_origin=origin, plan_sha256=digest(plan), runtime_sha256=digest(runtime),
                  training_sha256=digest(training), parameters_s=theta.tolist(), costs=costs(theta, plan).to_dict(),
                  local_sensitivity=dict(rank=rank, condition=condition, singular_values=singular.tolist()),
                  train_scores=scores, train_quality=train_quality,
                  status='effective_candidate_frozen', physical_costs_identified=False, model_accepted=False,
                  held_labels_used=False)
    frozen['costs']['provenance']=origin+': '+frozen['costs']['provenance']
    frozen['sha256'] = digest(frozen)
    return frozen


def evaluate(rows, plan, runtime, frozen, predicted):
    if (frozen['sha256'] != digest({k:v for k,v in frozen.items() if k != 'sha256'})
            or frozen['plan_sha256'] != digest(plan) or frozen['runtime_sha256'] != digest(runtime)):
        raise ValueError('Frozen model/plan/runtime identity changed')
    # Predictions must be the exact pre-held forecast, never supplied observed paths.
    if digest(predicted) != digest(predictions(frozen['parameters_s'], plan, runtime)):
        raise ValueError('Held prediction changed')
    held = select(rows, plan, {'held','held_reference'}, 'off')
    checks = quality(rows, plan, {'held','held_reference'})
    scores = endpoint_scores(held, predicted, plan)
    topology = (all(p['background_outlives_foreground'] for p in predicted.values())
                and structure_check(select(rows, plan, {'held','held_reference'}, 'light'), predicted))
    return dict(schema='effective-async-report-v1', data_origin=frozen['data_origin'], candidate_passed=checks['passed'] and scores['passed'] and topology,
                quality=checks, endpoint_scores=scores, structure_and_background_lifetime_valid=topology,
                frozen_sha256=frozen['sha256'], model_accepted=False, physical_costs_identified=False,
                complete_experiment_passed=False, adaptive_cap_accepted=False, mpc_accepted=False,
                scope=plan['scope'])


async def run(client, http, url, manifest, output):
    """Bounded native pilot: warmup, train, freeze+forecast, then fresh held trials."""
    from .suite import controlled_trial, runtime_preflight, host_identity
    from .causal import dispatcher_parity
    from scripts.predict_prefill_async import runtime_inputs
    plan = protocol();output = Path(output);started = time.monotonic();deadline = started+plan['hard_seconds']
    summary = dict(collection_complete=False, candidate_passed=False, model_accepted=False,
                   scope=plan['scope'], completed=[], failures=[])
    rows = [];frozen = predicted = None
    save(output/'calibration-plan.json', plan)
    try:
        validate_manifest(manifest)
        meta = await client.command('meta');runtime_preflight(meta, manifest);runtime_inputs(meta)
        if any(w.get('pipeline_stages') for w in meta['workers']):raise ValueError('Pilot must not enable stage profiler')
        save(output/'runtime.json', meta);save(output/'host-before.json', host_identity())
        by_case = {c['id']:c for c in plan['cases']}
        for trial in plan['order']:
            if time.monotonic() >= deadline:raise TimeoutError('Native pilot1800s deadline expired')
            if trial['stage'].startswith('held') and frozen is None:
                print('Calibration: fitting training endpoints; held trials have not started', flush=True)
                frozen = fit(rows, plan, meta, deadline=deadline, origin='native_controlled_endpoints')
                save(output/'frozen.json', frozen)
                save(output/'costs.json', frozen['costs'])
                predicted = predictions(frozen['parameters_s'], plan, meta)
                save(output/'pre-held-predictions.json', predicted)
                if not all(p['background_outlives_foreground'] for p in predicted.values()):
                    raise ValueError('Predicted D16 background lifetime outside pilot scope')
            tid = trial['trial_id']
            print(f'Calibration {len(summary["completed"])+1}/{len(plan["order"])}: {tid} {trial["stage"]} {trial["case"]} {trial["observer"]}', flush=True)
            capture = await controlled_trial(client, trial, deadline, output/(tid+'.json'))
            save(output/(tid+'.json'), capture)
            row = extract(capture, trial, by_case[trial['case']])
            if trial['observer'] != 'off':
                parity = dispatcher_parity(capture, meta['workers'][0]['dispatch_table'])
                if not parity['matches']:raise ValueError('Native dispatcher parity failed')
            rows.append(row);save(output/'endpoints.json', rows)
            summary['completed'].append(tid);save(output/'calibration.json', summary)
        final = await client.command('meta');save(output/'runtime-final.json', final)
        if digest(meta) != digest(final):raise ValueError('Runtime changed during calibration')
        save(output/'host-after.json', host_identity())
        report = evaluate(rows, plan, meta, frozen, predicted)
        save(output/'held-report.json', report)
        if time.monotonic() >= deadline:raise TimeoutError('Native pilot deadline expired during analysis')
        summary.update(collection_complete=True, candidate_passed=report['candidate_passed'])
    except BaseException as exc:
        summary['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic()-started
        save(output/'calibration.json', summary)
    return summary
