"""Development-only multi-step forecast baselines on causal identification features.

Scores persistence, training-mean and ridge (ARX-style, conditioned on the realized budget
path) forecasts of labels written by identification_features.py on held-out time and
held-out runs. Conditioning on realized control makes this system identification, not a
deployable planned-control forecast. The existing captures are development data: the numbers
choose what to measure next and do not establish a model for control.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_trace_dynamics import dump
from identification_features import CONTROL_CONDITIONING, CONTROL_LIMITS, PROTOCOL, SCHEMA_VERSION, label_key

REPORT_VERSION = 1
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
SINCE_CAP_S = 30.
NEAR_CHANGE_S = 10.


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


def splits(runs, fraction):
    """(split, held-out run or None, training masks by run, evaluation masks by run) per evaluation design."""
    def chronological(prepared):
        cut = {run: fraction * item['duration'] for run, item in prepared.items()}
        return ({run: item['known'] <= cut[run] for run, item in prepared.items()},
                {run: item['t'] >= cut[run] for run, item in prepared.items()})
    yield 'chronological', chronological
    if len(runs) > 1:
        for held in runs:
            def leave_out(prepared, held=held):
                return ({run: np.full(len(item['t']), run != held) for run, item in prepared.items()},
                        {run: np.full(len(item['t']), run == held) for run, item in prepared.items()})
            yield f'leave_one_run_out:{held}', leave_out


def evaluate(runs, *, horizons, windows, fraction=2 / 3, alpha=1., top=6):
    names = feature_names(runs)
    columns = names + ['control.budget_mean', 'control.budget_end']
    present = set()
    for run in runs.values():
        for row in run['rows']:
            for label in row['labels'].values():
                present.update(label or ())
    targets = [name for name in TARGETS if name in present]
    results, coefficients, training = [], [], []
    for horizon in horizons:
        key = label_key(horizon)
        base = {run: dict(rows=item['rows'], x=design(item['rows'], names, key),
                          t=np.array([row['t_s'] for row in item['rows']]), near=near_change(item['rows'], key),
                          duration=item['meta']['duration_s']) for run, item in runs.items()}
        for name in targets:
            direction = TARGETS[name]
            prepared = {run: dict(item, y=target(item['rows'], key, name),
                                  known=known_at(item['rows'], key, horizon, name),
                                  persistence=persistence(item['rows'], name, horizon, windows))
                        for run, item in base.items()}
            for split, masks in splits(runs, fraction):
                train, test = masks(prepared)
                train = {run: train[run] & ~np.isnan(item['y']) for run, item in prepared.items()}
                test = {run: test[run] & ~np.isnan(item['y']) for run, item in prepared.items()}
                y = np.concatenate([item['y'][train[run]] for run, item in prepared.items()])
                training.append(dict(split=split, horizon_s=horizon, target=name, n_train=int(len(y)),
                                     train_runs=sorted(run for run in runs if train[run].any())))
                if len(y) < len(columns) + 1:
                    continue
                model = Ridge(alpha).fit(np.vstack([item['x'][train[run]] for run, item in prepared.items()]), y)
                if split == 'chronological':
                    order = np.argsort(-np.abs(model.coef))[:top]
                    coefficients.append(dict(horizon_s=horizon, target=name, intercept=model.intercept,
                                             standardized=[dict(feature=columns[i], coef=float(model.coef[i]))
                                                           for i in order]))
                scopes = {run: [run] for run in runs if test[run].any()}
                if split == 'chronological' and len(scopes) > 1:
                    scopes['pooled'] = list(scopes)
                for scope, members in scopes.items():
                    for subset in ('all', 'near_budget_change'):
                        pick = {run: test[run] & (prepared[run]['near'] if subset != 'all' else True)
                                for run in members}
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
                ridge_coefficients=coefficients, training=training)


def fmt(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def markdown(report):
    lines = ['# Forecast baselines (development data only)', '']
    qualification = report['input_qualification']
    if report['warnings']:
        lines.extend(['> **Warning: unqualified evidence.** ' + ' '.join(report['warnings']), ''])
    lines.extend([
        f"Model readiness: **{report['model_readiness']}**. Control conditioning: "
        f"**{report['control_conditioning']}**. Rows overlap in time and are not independent; the control "
        'columns are the realized budget path over (t, t+h], so ridge is system identification conditioned '
        'on the applied cap, not a deployable planned-control forecast; only one capture changes the cap. '
        'Skill is 1 - MAE ratio against persistence (P) and the training mean (M); optimistic errors are '
        'those that would make admission look safer than it was.', '',
        f"Inputs (--allow-unqualified {'set' if qualification['allow_unqualified'] else 'not set'}): " +
        ', '.join(f"{name} (qualified {'yes' if run['qualified_for_comparison'] else 'NO'}, state reconstructed "
                  f"{'yes' if run['state_reconstruction_valid'] else 'NO'})"
                  for name, run in report['inputs'].items()) + '.', ''])
    results = report['evaluation']['results']

    def table(title, rows):
        lines.extend([f'## {title}', '', '| h (s) | target | n | MAE persistence | MAE mean | MAE ridge | '
                      'ridge skill vs P / M | p99 abs P / R | optimistic p99 P / R |',
                      '|---|---|---|---|---|---|---|---|---|'])
        for row in rows:
            p, m, r = (row['models'][name] for name in MODELS)
            if not r.get('n'):
                continue
            skills = row['skill_mae']
            lines.append(f"| {row['horizon_s']:g} | {row['target']} | {r['n']} | {fmt(p['mae'])} | {fmt(m['mae'])} | "
                         f"{fmt(r['mae'])} | {fmt(skills['ridge_vs_persistence'])} / {fmt(skills['ridge_vs_mean'])} | "
                         f"{fmt(p['p99_abs'])} / {fmt(r['p99_abs'])} | "
                         f"{fmt(p.get('optimistic_p99'))} / {fmt(r.get('optimistic_p99'))} |")
        lines.append('')
    scopes = sorted({row['scope'] for row in results if row['split'] == 'chronological'})
    pooled = 'pooled' if 'pooled' in scopes else scopes[0] if scopes else None
    for subset in ('all', 'near_budget_change'):
        table(f'Chronological hold-out ({pooled}, {subset})',
              [row for row in results if row['split'] == 'chronological' and row['scope'] == pooled
               and row['subset'] == subset])
    for split in sorted({row['split'] for row in results if row['split'] != 'chronological'}):
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
    args = parser.parse_args()
    if not 0 < args.split_fraction < 1:
        parser.error('--split-fraction must be between 0 and 1')
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error('--alpha must be finite and positive')
    try:
        summary, runs = load(args.features.resolve(), args.allow_unqualified)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    parameters = summary['parameters']
    available = [float(h) for h in parameters['horizons']]
    horizons = sorted(set(args.horizons or available))
    if any(h not in available for h in horizons):
        parser.error(f'--horizons must be a subset of the table horizons {available}')
    evaluation = evaluate(runs, horizons=horizons, windows=[float(w) for w in parameters['windows']],
                          fraction=args.split_fraction, alpha=args.alpha)
    here = Path(__file__).parent
    unqualified = [name for name, run in runs.items()
                   if not (run['meta']['qualified_for_comparison'] and run['meta']['state_reconstruction']['valid'])]
    report = dict(
        report_version=REPORT_VERSION, evaluation_role='development', model_readiness='not_established',
        control_conditioning=CONTROL_CONDITIONING, control_limits=CONTROL_LIMITS,
        input_qualification=dict(allow_unqualified=args.allow_unqualified, all_qualified=not unqualified,
                                 unqualified_runs=unqualified),
        warnings=[f"--allow-unqualified admitted {name}, which failed capture qualification or gateway state "
                  'reconstruction; its scores describe invalid evidence.' for name in unqualified],
        feature_protocol=PROTOCOL, feature_parameters=parameters,
        parameters=dict(horizons=horizons, split_fraction=args.split_fraction, alpha=args.alpha,
                        since_cap_s=SINCE_CAP_S, near_change_s=NEAR_CHANGE_S),
        inputs={name: dict(features_sha256=run['sha256'], raw_sha256=run['meta']['raw_sha256'],
                           qualified_for_comparison=run['meta']['qualified_for_comparison'],
                           state_reconstruction_valid=run['meta']['state_reconstruction']['valid'],
                           duration_s=run['meta']['duration_s'], e2e_slo=run['meta']['e2e_slo'],
                           e2e_slo_source=run['meta']['e2e_slo_source'],
                           budgets=sorted({a['budget'] for a in run['meta']['table']['budget_applications']}
                                          | {run['meta']['initial_budget']}))
                for name, run in runs.items()},
        analysis_code_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                              for name in ('forecast_baselines.py', 'identification_features.py')},
        models=dict(persistence='Current state; flows and cohorts extend the trailing rate from the window '
                                'nearest the horizon (cohort outcomes use receipt-anchored resolved proxies).',
                    mean='Training-set mean of the target.',
                    ridge='Direct per-horizon ridge on standardized features plus the realized budget mean/end '
                          'over (t, t+h] from recorded application times; nulls imputed with training means; '
                          'clipped at zero.'),
        splits=dict(chronological='Train on rows whose label is fully known by the cut in every run; '
                                  'evaluate rows at or after the cut. Scored per run and pooled.',
                    leave_one_run_out='Train on all other runs; evaluate the whole held-out run.'),
        evaluation=evaluation,
        caveats=['Three captures from one development set; they selected this design and are not a test set.',
                 'Neighboring rows overlap, so n overstates independent evidence.',
                 'Only one capture changes the cap, so budget effects are weakly identified.',
                 'Ridge conditions on realized control (recorded application times), not on a plan known at t.',
                 'Persistence for cohort outcomes uses receipt-anchored proxies, not scheduled-arrival labels.'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'forecast_report.json', report)
    (output / 'forecast_summary.md').write_text(markdown(report) + '\n', encoding='utf-8')
    print(output / 'forecast_summary.md')


if __name__ == '__main__':
    main()
