"""Multi-step forecast baselines on causal identification features.

Scores persistence, training-mean and ridge (ARX-style, conditioned on the realized budget
path) forecasts of labels written by identification_features.py on held-out time and
held-out runs. Conditioning on realized control makes this system identification, not a
deployable planned-control forecast. Without designated test runs the captures are development
data: the numbers choose what to measure next and do not establish a model for control. With
--acceptance, every capture must match its declared trace, budget plan and replay settings, the
predeclared test runs are excluded from every development split and scored once against the
declared limits, and the exit status reports the outcome (0 passed, 3 fail, 4 inconclusive,
5 not established: limits not frozen or no collection-time binding to this exact protocol).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_trace_dynamics import dump
from experiment_protocol import capture_mismatches, load_protocol, predeclaration_problems
from identification_features import CONTROL_CONDITIONING, CONTROL_LIMITS, PROTOCOL, SCHEMA_VERSION, label_key

REPORT_VERSION = 1
# Process exit status per acceptance outcome; argparse uses 2 for rejected inputs.
EXIT_STATUS = dict(passed=0, fail=3, inconclusive=4, not_established=5)
# Error direction that would make an admission controller too optimistic; None when ambiguous.
TARGETS = {
    'awaiting_first_content': 'under', 'oldest_awaiting_age_s': 'under', 'reserved_cost': 'under',
    'in_flight': 'under', 'streaming': None,
    'received': 'under', 'accepted': None, 'rejected': None, 'first_content': 'over', 'completed': 'over',
    'cohort_offered': 'under', 'cohort_ttft_met': 'over', 'cohort_admitted_ttft_missed': 'under',
    'cohort_joint_met': 'over',
}
STATE = {'awaiting_first_content', 'oldest_awaiting_age_s', 'reserved_cost', 'in_flight', 'streaming'}
# Trailing-window feature whose rate the persistence forecast extends over the horizon.
RATE_SOURCE = {'received': 'received', 'accepted': 'accepted', 'rejected': 'rejected',
               'first_content': 'first_content', 'completed': 'completed', 'cohort_offered': 'received',
               'cohort_ttft_met': 'resolved_ttft_met',
               'cohort_admitted_ttft_missed': 'resolved_admitted_ttft_missed',
               'cohort_joint_met': 'resolved_joint_met'}
MODELS = ('persistence', 'mean', 'ridge')
SUBSETS = ('all', 'near_budget_change', 'near_budget_increase', 'near_budget_decrease')
SINCE_CAP_S = 30.
NEAR_CHANGE_S = 10.
ACCEPTANCE_METRICS = ('skill_mae.ridge_vs_persistence', 'skill_mae.ridge_vs_mean', 'models.ridge.mae',
                      'models.ridge.p99_abs', 'models.ridge.optimistic_p99', 'budget_effect_per_unit')
BOUNDS = ('min', 'max', 'greater_than')


def load(directory, allow_unqualified=False):
    summary = json.loads((directory / 'features_summary.json').read_text())
    if summary.get('protocol') != PROTOCOL or summary.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported feature table protocol or schema version')
    runs = {}
    for run in summary['runs']:
        if not allow_unqualified and not (run['qualified_for_comparison'] and run['state_reconstruction']['valid']):
            raise ValueError(f"{run['name']} is unqualified or its gateway state was not reconstructed")
        path = directory / f"{run['name']}-features.jsonl"
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'{path.name} has no rows')
        runs[run['name']] = dict(meta=run, rows=rows, sha256=hashlib.sha256(data).hexdigest())
    if not runs:
        raise ValueError('The feature table has no runs')
    return summary, runs


def feature_names(runs):
    names = None
    for run in runs.values():
        for row in run['rows']:
            keys = sorted(row['features'])
            if names is None:
                names = keys
            elif keys != names:
                raise ValueError('Feature columns differ between rows; rebuild all runs with the same options')
    return names


def design(rows, names, key):
    """Features at t plus the realized budget path over (t, t+h]; nulls stay NaN for imputation."""
    matrix = np.full((len(rows), len(names) + 2), np.nan)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            value = row['features'][name]
            if name.startswith('since_'):
                value = SINCE_CAP_S if value is None else min(value, SINCE_CAP_S)
            if value is not None:
                matrix[i, j] = float(value)
        control = row['control'][key]
        matrix[i, -2:] = control['budget_mean'], control['budget_end']
    return matrix


def target(rows, key, name):
    values = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        label = row['labels'].get(key)
        if label is not None and row['label_window_complete'][key] and label.get(name) is not None:
            values[i] = label[name]
    return values


def known_at(rows, key, horizon, name):
    """When a label is fully observed: t+h, or the latest cohort outcome for cohort labels."""
    times = np.array([row['t_s'] + horizon for row in rows])
    if name.startswith('cohort_'):
        times = np.array([max(t, row['labels'][key]['cohort_resolved_s']) if row['labels'].get(key) else math.inf
                          for t, row in zip(times, rows)])
    return times


def near_change(rows, key):
    """Rows within NEAR_CHANGE_S after a cap change, or whose horizon contains a realized change."""
    mask = []
    for row in rows:
        since, budget, control = row['features']['since_budget_value_change_s'], row['features']['budget'], row['control'][key]
        mask.append((since is not None and since <= NEAR_CHANGE_S) or
                    control['budget_end'] != budget or control['budget_mean'] != budget)
    return np.array(mask, dtype=bool)


def change_directions(rows, key):
    """near_change split into (increase, decrease) masks by the direction of the past or in-horizon change."""
    increase, decrease = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
    previous = latest = None
    for i, row in enumerate(rows):
        features, control = row['features'], row['control'][key]
        budget = features['budget']
        if budget is None:
            continue
        if previous is not None and budget != previous:
            latest = budget > previous
        previous = budget
        since = features['since_budget_value_change_s']
        if latest is not None and since is not None and since <= NEAR_CHANGE_S:
            (increase if latest else decrease)[i] = True
        for realized in (control['budget_end'], control['budget_mean']):
            if realized is not None and realized != budget:
                (increase if realized > budget else decrease)[i] = True
    return increase, decrease


def persistence(rows, name, horizon, windows):
    if name in STATE:
        return np.array([float(row['features'][name]) for row in rows])
    window = min(windows, key=lambda w: (abs(w - horizon), w))
    column = f'{RATE_SOURCE[name]}_{window:g}s'
    return np.array([row['features'][column] * horizon / window for row in rows], dtype=float)


class Ridge:
    """Standardized ridge with training-mean imputation; predictions clipped at zero."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, x, y):
        present = ~np.isnan(x)
        counts = present.sum(axis=0)
        self.center = np.divide(np.where(present, x, 0.).sum(axis=0), counts,
                                out=np.zeros(x.shape[1]), where=counts > 0)
        z = np.where(present, x, self.center)
        self.scale = z.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        z = (z - self.center) / self.scale
        self.intercept = float(y.mean())
        self.coef = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]), z.T @ (y - self.intercept))
        return self

    def predict(self, x):
        z = (np.where(np.isnan(x), self.center, x) - self.center) / self.scale
        return np.maximum(0., self.intercept + z @ self.coef)


def quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else None


def metrics(prediction, actual, direction):
    if not len(actual):
        return dict(n=0)
    error = prediction - actual
    absolute = np.abs(error)
    result = dict(n=int(len(actual)), mae=float(absolute.mean()), rmse=float(np.sqrt((error ** 2).mean())),
                  bias=float(error.mean()), p90_abs=quantile(absolute, .9), p99_abs=quantile(absolute, .99),
                  max_abs=float(absolute.max()))
    if direction is not None:
        optimistic = np.maximum(0., -error if direction == 'under' else error)
        result.update(optimistic_direction=direction, optimistic_fraction=float((optimistic > 0).mean()),
                      optimistic_p90=quantile(optimistic, .9), optimistic_p99=quantile(optimistic, .99),
                      optimistic_max=float(optimistic.max()))
    return result


def skill(model, reference):
    if not model.get('n') or not reference.get('mae'):
        return None
    return 1. - model['mae'] / reference['mae']


def splits(runs, fraction, test_runs=()):
    """(split, training masks by run, evaluation masks by run) per evaluation design.

    Designated test runs never enter the chronological or leave-one-run-out splits; they are only
    scored by 'designated_test', trained on every other run.
    """
    test_runs = set(test_runs)

    def chronological(prepared):
        cut = {run: fraction * item['duration'] for run, item in prepared.items()}
        return ({run: (item['known'] <= cut[run]) & (run not in test_runs) for run, item in prepared.items()},
                {run: (item['t'] >= cut[run]) & (run not in test_runs) for run, item in prepared.items()})
    yield 'chronological', chronological
    development = [run for run in runs if run not in test_runs]
    if len(development) > 1:
        for held in development:
            def leave_out(prepared, held=held):
                return ({run: np.full(len(item['t']), run != held and run not in test_runs)
                         for run, item in prepared.items()},
                        {run: np.full(len(item['t']), run == held) for run, item in prepared.items()})
            yield f'leave_one_run_out:{held}', leave_out
    if test_runs:
        def designated(prepared):
            return ({run: np.full(len(item['t']), run not in test_runs) for run, item in prepared.items()},
                    {run: np.full(len(item['t']), run in test_runs) for run, item in prepared.items()})
        yield 'designated_test', designated


def evaluate(runs, *, horizons, windows, fraction=2 / 3, alpha=1., top=6, test_runs=()):
    names = feature_names(runs)
    columns = names + ['control.budget_mean', 'control.budget_end']
    present = set()
    for run in runs.values():
        for row in run['rows']:
            for label in row['labels'].values():
                present.update(label or ())
    targets = [name for name in TARGETS if name in present]
    results, coefficients, training, effects = [], [], [], []
    for horizon in horizons:
        key = label_key(horizon)
        base = {}
        for run, item in runs.items():
            increase, decrease = change_directions(item['rows'], key)
            base[run] = dict(rows=item['rows'], x=design(item['rows'], names, key),
                             t=np.array([row['t_s'] for row in item['rows']]), duration=item['meta']['duration_s'],
                             subsets=dict(all=np.ones(len(item['rows']), dtype=bool),
                                          near_budget_change=near_change(item['rows'], key),
                                          near_budget_increase=increase, near_budget_decrease=decrease))
        for name in targets:
            direction = TARGETS[name]
            prepared = {run: dict(item, y=target(item['rows'], key, name),
                                  known=known_at(item['rows'], key, horizon, name),
                                  persistence=persistence(item['rows'], name, horizon, windows))
                        for run, item in base.items()}
            for split, masks in splits(runs, fraction, test_runs):
                train, test = masks(prepared)
                train = {run: train[run] & ~np.isnan(item['y']) for run, item in prepared.items()}
                test = {run: test[run] & ~np.isnan(item['y']) for run, item in prepared.items()}
                y = np.concatenate([item['y'][train[run]] for run, item in prepared.items()])
                training.append(dict(split=split, horizon_s=horizon, target=name, n_train=int(len(y)),
                                     train_runs=sorted(run for run in runs if train[run].any())))
                if len(y) < len(columns) + 1:
                    continue
                model = Ridge(alpha).fit(np.vstack([item['x'][train[run]] for run, item in prepared.items()]), y)
                # Linear response of the forecast to holding a different cap over (t, t+h], before clipping.
                control_columns = [columns.index('control.budget_mean'), columns.index('control.budget_end')]
                effects.append(dict(split=split, horizon_s=horizon, target=name,
                                    per_unit=float(sum(model.coef[i] / model.scale[i] for i in control_columns))))
                if split == 'chronological':
                    order = np.argsort(-np.abs(model.coef))[:top]
                    coefficients.append(dict(horizon_s=horizon, target=name, intercept=model.intercept,
                                             standardized=[dict(feature=columns[i], coef=float(model.coef[i]))
                                                           for i in order]))
                scopes = {run: [run] for run in runs if test[run].any()}
                if split in ('chronological', 'designated_test') and len(scopes) > 1:
                    scopes['pooled'] = list(scopes)
                for scope, members in scopes.items():
                    for subset in SUBSETS:
                        pick = {run: test[run] & prepared[run]['subsets'][subset] for run in members}
                        actual = np.concatenate([prepared[run]['y'][pick[run]] for run in members])
                        predictions = dict(
                            persistence=np.concatenate([prepared[run]['persistence'][pick[run]] for run in members]),
                            mean=np.full(len(actual), float(y.mean())),
                            ridge=np.concatenate([model.predict(prepared[run]['x'][pick[run]]) for run in members]))
                        scores = {model_name: metrics(predictions[model_name], actual, direction)
                                  for model_name in MODELS}
                        results.append(dict(split=split, scope=scope, horizon_s=horizon, target=name, subset=subset,
                                            models=scores,
                                            skill_mae=dict(
                                                mean_vs_persistence=skill(scores['mean'], scores['persistence']),
                                                ridge_vs_persistence=skill(scores['ridge'], scores['persistence']),
                                                ridge_vs_mean=skill(scores['ridge'], scores['mean']))))
    return dict(feature_columns=columns, targets=targets, results=results,
                ridge_coefficients=coefficients, budget_effects=effects, training=training)


def lookup(row, path):
    value = row
    for part in path.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def load_acceptance(path, horizons):
    """Validate the protocol's declarations and acceptance section; returns (document, file sha256).

    The hash is of the LF-normalized bytes, matching the committed file on any checkout.
    """
    document, sha256 = load_protocol(path)
    spec = document.get('acceptance')
    if not isinstance(spec, dict) or spec.get('schema_version') != 1:
        raise ValueError('The protocol file needs an acceptance section with schema_version 1')
    if spec.get('status') not in ('proposed', 'frozen'):
        raise ValueError("Acceptance status must be 'proposed' or 'frozen'")
    if spec.get('model') != 'ridge':
        raise ValueError("Acceptance model must be 'ridge'")
    test_runs, training_runs = spec.get('test_runs') or [], spec.get('training_runs') or []
    if not test_runs or len(set(test_runs)) != len(test_runs):
        raise ValueError('Acceptance needs distinct test_runs')
    if not training_runs or len(set(training_runs)) != len(training_runs) or set(training_runs) & set(test_runs):
        raise ValueError('Acceptance needs distinct training_runs disjoint from test_runs')
    roles = {role: {run['name'] for run in document['runs'] if run['role'] == role}
             for role in ('development', 'test')}
    if set(training_runs) != roles['development'] or set(test_runs) != roles['test']:
        raise ValueError('Acceptance training_runs and test_runs must be the declared development and test runs')
    pre = spec.get('preconditions', {})
    if set(pre) != {'binding_headroom', 'min_binding_fraction', 'min_subset_n'}:
        raise ValueError('Acceptance preconditions need binding_headroom, min_binding_fraction and min_subset_n')
    ids = [criterion.get('id') for criterion in spec.get('criteria', [])]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise ValueError('Acceptance criteria need unique ids')
    for criterion in spec['criteria']:
        bounds = [name for name in BOUNDS if name in criterion]
        if (len(bounds) != 1 or criterion.get('metric') not in ACCEPTANCE_METRICS
                or criterion.get('target') not in TARGETS or float(criterion.get('horizon_s', -1)) not in horizons
                or (criterion['metric'] != 'budget_effect_per_unit' and criterion.get('subset') not in SUBSETS)):
            raise ValueError(f"Invalid acceptance criterion {criterion['id']}")
    return document, sha256


def analysis_mismatches(analysis, parameters, split_fraction, alpha):
    """Names of feature/forecast settings that differ from the protocol's declared analysis."""
    analysis = analysis or {}
    declared = dict(analysis.get('features', {}), **analysis.get('forecast', {}))
    actual = dict(parameters, split_fraction=split_fraction, alpha=alpha)

    def same(a, b):
        a, b = np.atleast_1d(np.asarray(a, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
        return a.shape == b.shape and np.allclose(a, b)
    return sorted(name for name, value in declared.items() if name not in actual or not same(value, actual[name]))


def judge(spec, evaluation, runs):
    """Score the designated test against predeclared limits: pass, fail or inconclusive."""
    pre, test_runs = spec['preconditions'], spec['test_runs']
    scope = 'pooled' if len(test_runs) > 1 else test_runs[0]
    rows = [row for run in test_runs for row in runs[run]['rows']]
    binding = float(np.mean([row['features'].get('budget_headroom') is not None and
                             row['features']['budget_headroom'] < pre['binding_headroom'] for row in rows]))
    effects = {(e['horizon_s'], e['target']): e['per_unit'] for e in evaluation['budget_effects']
               if e['split'] == 'designated_test'}
    outcomes = []
    for criterion in spec['criteria']:
        horizon, name = float(criterion['horizon_s']), criterion['target']
        if criterion['metric'] == 'budget_effect_per_unit':
            value, n = effects.get((horizon, name)), None
        else:
            row = next((row for row in evaluation['results'] if row['split'] == 'designated_test'
                        and row['scope'] == scope and row['horizon_s'] == horizon and row['target'] == name
                        and row['subset'] == criterion['subset']), None)
            value = lookup(row, criterion['metric']) if row else None
            n = row['models']['ridge'].get('n', 0) if row else 0
        bound = next(option for option in BOUNDS if option in criterion)
        limit = criterion[bound]
        if value is None or (n is not None and n < pre['min_subset_n']):
            outcome = 'insufficient'
        else:
            passed = value >= limit if bound == 'min' else value <= limit if bound == 'max' else value > limit
            outcome = 'pass' if passed else 'fail'
        outcomes.append(dict(id=criterion['id'], horizon_s=horizon, target=name, subset=criterion.get('subset'),
                             metric=criterion['metric'], bound=bound, limit=limit, value=value, n=n,
                             outcome=outcome))
    binding_met = binding >= pre['min_binding_fraction']
    states = {item['outcome'] for item in outcomes}
    verdict = ('inconclusive' if not binding_met else 'fail' if 'fail' in states else
               'inconclusive' if 'insufficient' in states else 'pass')
    return dict(scope=scope, test_runs=test_runs, binding_fraction=binding, binding_met=binding_met,
                criteria=outcomes, verdict=verdict)


def fmt(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def markdown(report):
    designated = report['evaluation_role'] == 'designated_test'
    changing, total = report['captures_with_cap_changes'], len(report['inputs'])
    qualification = report['input_qualification']
    lines = [f"# Forecast baselines ({'designated test' if designated else 'development data only'})", '']
    if report['warnings']:
        lines.extend(['> **Warning: unqualified evidence.** ' + ' '.join(report['warnings']), ''])
    lines.extend([
        f"Model readiness: **{report['model_readiness']}**. Control conditioning: "
        f"**{report['control_conditioning']}**. Rows overlap in time and are not independent; the control "
        'columns are the realized budget path over (t, t+h], so ridge is system identification conditioned '
        f'on the applied cap, not a deployable planned-control forecast; {changing} of {total} captures change '
        'the cap. Skill is 1 - MAE ratio against persistence (P) and the training mean (M); optimistic errors '
        'are those that would make admission look safer than it was.', '',
        f"Inputs (--allow-unqualified {'set' if qualification['allow_unqualified'] else 'not set'}): " +
        ', '.join(f"{name} (qualified {'yes' if run['qualified_for_comparison'] else 'NO'}, state reconstructed "
                  f"{'yes' if run['state_reconstruction_valid'] else 'NO'})"
                  for name, run in report['inputs'].items()) + '.', ''])
    results = report['evaluation']['results']

    def table(title, rows):
        rows = [row for row in rows if row['models']['ridge'].get('n')]
        if not rows:
            return
        lines.extend([f'## {title}', '', '| h (s) | target | n | MAE persistence | MAE mean | MAE ridge | '
                      'ridge skill vs P / M | p99 abs P / R | optimistic p99 P / R |',
                      '|---|---|---|---|---|---|---|---|---|'])
        for row in rows:
            p, m, r = (row['models'][name] for name in MODELS)
            skills = row['skill_mae']
            lines.append(f"| {row['horizon_s']:g} | {row['target']} | {r['n']} | {fmt(p['mae'])} | {fmt(m['mae'])} | "
                         f"{fmt(r['mae'])} | {fmt(skills['ridge_vs_persistence'])} / {fmt(skills['ridge_vs_mean'])} | "
                         f"{fmt(p['p99_abs'])} / {fmt(r['p99_abs'])} | "
                         f"{fmt(p.get('optimistic_p99'))} / {fmt(r.get('optimistic_p99'))} |")
        lines.append('')
    acceptance = report.get('acceptance')
    if acceptance:
        judged = acceptance['judgement']
        predeclaration = acceptance['predeclaration']
        lines.extend([f"## Acceptance ({acceptance['status']} limits): **{judged['verdict']}**", '',
                      f"Outcome: **{acceptance['outcome']}** (exit status {acceptance['exit_status']}). "
                      f"Predeclaration: {'verified' if predeclaration['verified'] else 'UNVERIFIED'}"
                      + (f" ({'; '.join(predeclaration['problems'])})" if predeclaration['problems'] else '') + '.',
                      '',
                      f"Test runs: {', '.join(judged['test_runs'])}; binding fraction "
                      f"{fmt(judged['binding_fraction'])} (needs >= "
                      f"{fmt(acceptance['preconditions']['min_binding_fraction'])}); protocol sha256 "
                      f"`{acceptance['protocol_sha256']}`.", '',
                      '| id | h (s) | target | subset | metric | limit | value | n | outcome |',
                      '|---|---|---|---|---|---|---|---|---|'])
        symbols = dict(min='>=', max='<=', greater_than='>')
        for item in judged['criteria']:
            lines.append(f"| {item['id']} | {item['horizon_s']:g} | {item['target']} | {item['subset'] or '-'} | "
                         f"{item['metric']} | {symbols[item['bound']]} {item['limit']:g} | {fmt(item['value'], 3)} | "
                         f"{'-' if item['n'] is None else item['n']} | {item['outcome']} |")
        lines.append('')
    for split, label in (('designated_test', 'Designated test'), ('chronological', 'Chronological hold-out')):
        scopes = sorted({row['scope'] for row in results if row['split'] == split})
        pooled = 'pooled' if 'pooled' in scopes else scopes[0] if scopes else None
        for subset in SUBSETS:
            table(f'{label} ({pooled}, {subset})',
                  [row for row in results if row['split'] == split and row['scope'] == pooled
                   and row['subset'] == subset])
    for split in sorted({row['split'] for row in results if row['split'].startswith('leave_one_run_out:')}):
        table(f"Held-out run {split.split(':', 1)[1]} (all)",
              [row for row in results if row['split'] == split and row['subset'] == 'all'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=Path, help='Directory written by identification_features.py')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--horizons', type=float, nargs='+', help='Subset of the table horizons (default: all)')
    parser.add_argument('--split-fraction', type=float, default=2 / 3,
                        help='Chronological cut as a fraction of each run duration')
    parser.add_argument('--alpha', type=float, default=1., help='Ridge penalty on standardized features')
    parser.add_argument('--allow-unqualified', action='store_true')
    parser.add_argument('--test-runs', nargs='+', metavar='NAME',
                        help='Runs excluded from every development split and scored only as the designated test')
    parser.add_argument('--acceptance', type=Path,
                        help='Protocol JSON with a predeclared acceptance section (implies its test_runs)')
    args = parser.parse_args()
    if not 0 < args.split_fraction < 1:
        parser.error('--split-fraction must be between 0 and 1')
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error('--alpha must be finite and positive')
    if args.acceptance and args.allow_unqualified:
        parser.error('--acceptance cannot be scored on unqualified captures')
    try:
        summary, runs = load(args.features.resolve(), args.allow_unqualified)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    parameters = summary['parameters']
    available = [float(h) for h in parameters['horizons']]
    horizons = sorted(set(args.horizons or available))
    if any(h not in available for h in horizons):
        parser.error(f'--horizons must be a subset of the table horizons {available}')
    spec = document = protocol_sha256 = None
    if args.acceptance:
        try:
            document, protocol_sha256 = load_acceptance(args.acceptance, horizons)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        spec = document['acceptance']
        if args.test_runs and set(args.test_runs) != set(spec['test_runs']):
            parser.error('--test-runs differs from the test_runs declared in --acceptance')
        if set(runs) != set(spec['training_runs']) | set(spec['test_runs']):
            parser.error(f"The feature table runs {sorted(runs)} differ from the declared training and test runs")
        mismatched = analysis_mismatches(document.get('analysis'), parameters, args.split_fraction, args.alpha)
        if mismatched:
            parser.error(f"Settings differ from the protocol's declared analysis: {', '.join(mismatched)}")
        problems = [problem for name in sorted(runs)
                    for problem in capture_mismatches(document, name, runs[name]['meta'].get('capture'))]
        if problems:
            parser.error('Captures differ from the protocol declaration: ' + '; '.join(problems))
    test_runs = list(spec['test_runs'] if spec else args.test_runs or [])
    if len(set(test_runs)) != len(test_runs) or any(run not in runs for run in test_runs):
        parser.error(f'Test runs must be distinct runs of the feature table: {sorted(runs)}')
    if test_runs and len(test_runs) == len(runs):
        parser.error('At least one run must remain for training')
    evaluation = evaluate(runs, horizons=horizons, windows=[float(w) for w in parameters['windows']],
                          fraction=args.split_fraction, alpha=args.alpha, test_runs=test_runs)
    acceptance = None
    if spec:
        judgement = judge(spec, evaluation, runs)
        unverified = [problem for name in spec['test_runs'] for problem in
                      predeclaration_problems(document, protocol_sha256, name, runs[name]['meta'].get('capture'))]
        verdict = judgement['verdict']
        outcome = verdict if verdict != 'pass' else 'not_established' if unverified else 'passed'
        acceptance = dict(protocol=str(args.acceptance), protocol_sha256=protocol_sha256, status=spec['status'],
                          training_runs=spec['training_runs'], preconditions=spec['preconditions'],
                          predeclaration=dict(verified=not unverified, problems=unverified),
                          judgement=judgement, outcome=outcome, exit_status=EXIT_STATUS[outcome])
    passed = acceptance is not None and acceptance['outcome'] == 'passed'
    inputs = {name: dict(features_sha256=run['sha256'], raw_sha256=run['meta']['raw_sha256'],
                         qualified_for_comparison=run['meta']['qualified_for_comparison'],
                         state_reconstruction_valid=run['meta']['state_reconstruction']['valid'],
                         duration_s=run['meta']['duration_s'], e2e_slo=run['meta']['e2e_slo'],
                         e2e_slo_source=run['meta']['e2e_slo_source'],
                         budgets=sorted({a['budget'] for a in run['meta']['table']['budget_applications']}
                                        | {run['meta']['initial_budget']}),
                         capture=run['meta'].get('capture'))
              for name, run in runs.items()}
    changing = sum(len(item['budgets']) > 1 for item in inputs.values())
    development = len(runs) - len(test_runs)
    caveats = [f'{development} development capture(s) selected this design; they are not a test set.',
               'Neighboring rows overlap, so n overstates independent evidence.',
               'Ridge conditions on realized control (recorded application times), not on a plan known at t.',
               'Persistence for cohort outcomes uses receipt-anchored proxies, not scheduled-arrival labels.']
    if changing < 2:
        caveats.insert(2, f'Only {changing} capture(s) change the cap, so budget effects are weakly identified.')
    if test_runs:
        caveats.append(f"Designated test runs {', '.join(test_runs)} are excluded from every development split.")
    if acceptance:
        caveats.append('budget_effect_per_unit describes the model fitted on the training runs (a sign check), '
                       'not a test-set measurement.')
        if acceptance['status'] != 'frozen':
            caveats.append('The acceptance limits are not frozen, so the verdict cannot establish readiness.')
        if not acceptance['predeclaration']['verified']:
            caveats.append('Predeclaration is unverified: a test run lacks a record of collection under this exact '
                           'frozen protocol, so a pass does not establish readiness.')
    splits_doc = dict(chronological='Train on rows whose label is fully known by the cut in every development '
                                    'run; evaluate rows at or after the cut. Scored per run and pooled.',
                      leave_one_run_out='Train on all other development runs; evaluate the whole held-out run.')
    if test_runs:
        splits_doc['designated_test'] = 'Train on every non-test run; evaluate the whole test runs once.'
    here = Path(__file__).parent
    unqualified = [name for name, run in runs.items()
                   if not (run['meta']['qualified_for_comparison'] and run['meta']['state_reconstruction']['valid'])]
    report = dict(
        report_version=REPORT_VERSION, evaluation_role='designated_test' if test_runs else 'development',
        model_readiness='offline_acceptance_passed' if passed else 'not_established',
        control_conditioning=CONTROL_CONDITIONING, control_limits=CONTROL_LIMITS,
        input_qualification=dict(allow_unqualified=args.allow_unqualified, all_qualified=not unqualified,
                                 unqualified_runs=unqualified),
        warnings=[f"--allow-unqualified admitted {name}, which failed capture qualification or gateway state "
                  'reconstruction; its scores describe invalid evidence.' for name in unqualified],
        feature_protocol=PROTOCOL, feature_parameters=parameters,
        parameters=dict(horizons=horizons, split_fraction=args.split_fraction, alpha=args.alpha,
                        since_cap_s=SINCE_CAP_S, near_change_s=NEAR_CHANGE_S, test_runs=test_runs),
        inputs=inputs, captures_with_cap_changes=changing,
        analysis_code_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                              for name in ('forecast_baselines.py', 'identification_features.py',
                                           'experiment_protocol.py')},
        models=dict(persistence='Current state; flows and cohorts extend the trailing rate from the window '
                                'nearest the horizon (cohort outcomes use receipt-anchored resolved proxies).',
                    mean='Training-set mean of the target.',
                    ridge='Direct per-horizon ridge on standardized features plus the realized budget mean/end '
                          'over (t, t+h] from recorded application times; nulls imputed with training means; '
                          'clipped at zero.'),
        splits=splits_doc, evaluation=evaluation, acceptance=acceptance, caveats=caveats)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'forecast_report.json', report)
    (output / 'forecast_summary.md').write_text(markdown(report) + '\n', encoding='utf-8')
    print(output / 'forecast_summary.md')
    if acceptance:
        print(f"acceptance ({acceptance['status']}): {acceptance['judgement']['verdict']}; "
              f"outcome {acceptance['outcome']} (exit status {acceptance['exit_status']})")
        return acceptance['exit_status']
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
