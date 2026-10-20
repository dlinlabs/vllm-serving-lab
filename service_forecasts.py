"""Fixed service-context ablations and cap-response diagnostics; no controller."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from analyze_trace_dynamics import dump, finite
from forecast_baselines import Ridge, metrics, skill
from service_features import CONTEXT, RATE_FIELDS, SERVICE_PROTOCOL
from transition_baselines import BASE, ENGINE, OPTIMISTIC, TARGETS, eligible, load, serialize, splits, vector
from transition_features import HORIZONS, STATES

MODELS = ('direct_ridge', 'context_ridge', 'gauge_ridge', 'counter_ridge', 'service_quadratic', 'service_no_plan')
CONTEXT_COLUMNS = CONTEXT + ('oldest_awaiting_age_s', 'applied_budget', 'applied_headroom')
PLAN_COLUMNS = ('cap_first', 'cap_mean', 'cap_last')
INTERACTIONS = ('cap_mean_square', 'cap_first_square', 'mean_cap_delta',
                'cap_x_awaiting_cost', 'cap_x_streaming_cost', 'cap_x_offered_cost')
CANDIDATE_CAPS = (16., 32., 64., 96.)
SUPPORT_COLUMNS = STATES + ('awaiting_input_tokens', 'streaming_input_tokens', 'offered_input_tokens_5s')


def columns(kind):
    if kind not in MODELS:
        raise ValueError('Unknown service model')
    if kind == 'direct_ridge':
        return BASE + ('cap_mean', 'cap_last')
    static = tuple(k for k in BASE if k not in ('budget', 'budget_headroom')) + CONTEXT_COLUMNS
    if kind in ('gauge_ridge', 'counter_ridge', 'service_quadratic', 'service_no_plan'):
        static += ENGINE
    if kind in ('counter_ridge', 'service_quadratic', 'service_no_plan'):
        static += RATE_FIELDS
    return static + (() if kind == 'service_no_plan' else PLAN_COLUMNS) + (INTERACTIONS if kind == 'service_quadratic' else ())


def design(features, caps, kind):
    if not caps or any(not finite(c) or c <= 0 for c in caps):
        raise ValueError('Caps must be finite and positive')
    if kind == 'direct_ridge':
        return vector(features, caps[0]) + [sum(caps) / len(caps), caps[-1]]
    values = dict(features)
    values.update(in_flight=features['awaiting_first_content'] + features['streaming'],
        applied_budget=features['budget'], applied_headroom=features['budget_headroom'])
    mean = sum(caps) / len(caps)
    values.update(cap_first=caps[0], cap_mean=mean, cap_last=caps[-1],
        cap_mean_square=sum(c * c for c in caps) / len(caps), cap_first_square=caps[0] ** 2,
        mean_cap_delta=mean - features['budget'], cap_x_awaiting_cost=mean * features['awaiting_cost'],
        cap_x_streaming_cost=mean * features['streaming_cost'], cap_x_offered_cost=mean * features['offered_cost_5s'])
    result = [math.nan if values.get(key) is None else float(values[key]) for key in columns(kind)]
    if np.isinf(result).any():
        raise ValueError('Infinite predictor')
    return result


class DirectModel:
    def __init__(self, horizon, kind):
        if horizon not in HORIZONS:
            raise ValueError('Unsupported horizon')
        self.horizon, self.kind = horizon, kind
        self.names = columns(kind)

    def fit(self, rows):
        if len(rows) <= len(self.names):
            raise ValueError('Too few labeled rows for the fixed service model')
        matrix = np.array([design(r['features'], r['planned_caps'][:self.horizon], self.kind) for r in rows])
        if np.isinf(matrix).any():
            raise ValueError('Infinite predictors')
        self.heads = {}
        for target in TARGETS:
            y = np.array([r['labels'][str(self.horizon)][target] for r in rows])
            if not np.isfinite(y).all() or (y < 0).any():
                raise ValueError('Invalid target')
            self.heads[target] = Ridge(1.).fit(matrix, y)
        caps = [c for r in rows for c in r['planned_caps'][:self.horizon]]
        self.cap_range = min(caps), max(caps)
        return self

    def predict(self, features, caps, *, allow_extrapolation=False):
        if len(caps) != self.horizon:
            raise ValueError('Cap plan length must equal the trained horizon')
        if any(not finite(c) or c <= 0 for c in caps):
            raise ValueError('Invalid candidate cap')
        if not allow_extrapolation and any(not self.cap_range[0] <= c <= self.cap_range[1] for c in caps):
            raise ValueError('Cap outside training range; explicit diagnostic extrapolation required')
        x = np.array([design(features, caps, self.kind)])
        return {target: float(head.predict(x)[0]) for target, head in self.heads.items()}

    def export(self):
        return dict(kind=self.kind, horizon_s=self.horizon, columns=list(self.names), cap_range=list(self.cap_range),
                    heads={key: serialize(value) for key, value in self.heads.items()})


class SupportIndex:
    """Descriptive training coverage, never a causal validity or safety certificate."""
    def __init__(self, rows, horizon):
        self.rows, self.horizon = rows, horizon
        matrix = np.array([[r['features'][k] for k in SUPPORT_COLUMNS] for r in rows], dtype=float)
        self.center, self.scale = matrix.mean(axis=0), matrix.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        self.groups = {}
        for cap in CANDIDATE_CAPS:
            chosen = [i for i, row in enumerate(rows) if all(c == cap for c in row['planned_caps'][:horizon])]
            self.groups[cap] = (chosen, (matrix[chosen] - self.center) / self.scale)

    def at(self, features, cap):
        indices, x = self.groups[cap]
        query = (np.array([features[key] for key in SUPPORT_COLUMNS]) - self.center) / self.scale
        return dict(training_hold_rows=len(indices),
            same_applied_cap_rows=sum(self.rows[i]['features']['budget'] == features['budget'] for i in indices),
            nearest_normalized_state_distance=float(np.linalg.norm(x - query, axis=1).min()) if indices else None)


def load_service(directory):
    manifest, runs = load(directory)
    if manifest.get('service_protocol') != SERVICE_PROTOCOL:
        raise ValueError('Service-context dataset required')
    for run in runs.values():
        if run['meta'].get('service_context', {}).get('verified_rows') != len(run['rows']):
            raise ValueError('Unverified service context')
        for row in run['rows']:
            f = row['features']
            if any(not finite(f.get(key)) or f[key] < 0 for key in CONTEXT):
                raise ValueError('Missing/nonfinite service context')
            for key in ('prompt_tokens', 'generation_tokens', 'preemptions'):
                value, missing = f.get(key + '_rate'), f.get(key + '_missing')
                if type(missing) is not bool or (missing and value is not None) or (not missing and (not finite(value) or value < 0)):
                    raise ValueError('Counter value and missing flag disagree')
    return manifest, runs


def evaluate(runs):
    scores, fitted, predictions, responses = [], {}, [], []
    for split, cuts, testing in splits(runs):
        fitted[split] = dict(training_cutoffs={n: cut if finite(cut) else None for n, cut in cuts.items()}, models={})
        for horizon in HORIZONS:
            training = [row for name, cut in cuts.items() for row in eligible(runs[name]['rows'], horizon, cut, True)]
            models = {kind: DirectModel(horizon, kind).fit(training) for kind in MODELS}
            fitted[split]['models'][str(horizon)] = dict(rows=len(training), models={k: m.export() for k, m in models.items()})
            support = SupportIndex(training, horizon) if split == 'chronological' and horizon == 5 else None
            for name in testing:
                cutoff = cuts.get(name, 0.) if split == 'chronological' else 0.
                rows = [r for r in eligible(runs[name]['rows'], horizon, reward=True) if r['t_s'] >= cutoff]
                if not rows:
                    continue
                actual = {target: np.array([r['labels'][str(horizon)][target] for r in rows]) for target in TARGETS}
                forecasts = {kind: {target: [] for target in TARGETS} for kind in MODELS}
                near, extrapolation = [], []
                for row in rows:
                    caps, f = row['planned_caps'][:horizon], row['features']
                    since = f['since_budget_value_change_s']
                    near.append((since is not None and since <= 10.) or any(c != f['budget'] for c in caps))
                    low, high = models['direct_ridge'].cap_range
                    extrapolation.append(any(c < low or c > high for c in caps))
                    for kind, model in models.items():
                        forecast = model.predict(f, caps, allow_extrapolation=True)
                        for target in TARGETS:
                            forecasts[kind][target].append(forecast[target])
                    if support is not None:
                        coverage = {str(c): support.at(f, c) for c in CANDIDATE_CAPS}
                        choices = {kind: {str(c): model.predict(f, [c] * horizon, allow_extrapolation=True)['cohort_ttft_met'] / horizon
                                         for c in CANDIDATE_CAPS} for kind, model in models.items()}
                        responses.append(dict(run=name, t_s=row['t_s'], applied_cap=f['budget'], support=coverage,
                            candidate_goodput_rate=choices,
                            best_candidates={kind: [cap for cap, value in values.items() if abs(value - max(values.values())) <= 1e-9]
                                             for kind, values in choices.items()}))
                for subset, mask in (('all', np.ones(len(rows), dtype=bool)), ('near_cap_change', np.array(near))):
                    for target in TARGETS:
                        values = {kind: metrics(np.array(forecasts[kind][target])[mask], actual[target][mask], OPTIMISTIC[target]) for kind in MODELS}
                        scores.append(dict(split=split, run=name, horizon_s=horizon, subset=subset, target=target,
                            diagnostic_extrapolation_rows=int(np.array(extrapolation)[mask].sum()), models=values,
                            skill_vs_direct={kind: skill(values[kind], values['direct_ridge']) for kind in MODELS[1:]}))
                if split == 'chronological':
                    predictions.extend(dict(run=name, t_s=row['t_s'], horizon_s=horizon,
                        actual={key: float(actual[key][i]) for key in TARGETS},
                        predictions={kind: {key: forecasts[kind][key][i] for key in TARGETS} for kind in MODELS}) for i, row in enumerate(rows))
    return dict(protocol=SERVICE_PROTOCOL, acceptance=None, c_used=False, scores=scores, fitted=fitted,
        chronological_predictions=predictions, cap_responses=responses,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        note='Direct multi-horizon forecasts, not a recursive transition simulator or MPC. Cap-response rankings are unvalidated predictions; no counterfactual outcomes exist for a common origin state.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source, output = args.dataset.resolve(), args.output_dir.resolve()
    if output.exists() or source == output or source in output.parents:
        parser.error('Choose a fresh output outside the dataset')
    try:
        manifest, runs = load_service(source)
        report = evaluate(runs)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    report.update(dataset_sha256=hashlib.sha256((source / 'dataset.json').read_bytes()).hexdigest(), source_runs=manifest['runs'],
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                     ('service_features.py', 'service_forecasts.py', 'transition_baselines.py', 'forecast_baselines.py')})
    output.mkdir(parents=True)
    dump(output / 'service_report.json', report)
    print(json.dumps(dict(runs=len(runs), scores=len(report['scores']), cap_response_origins=len(report['cap_responses'])), indent=2))


if __name__ == '__main__':
    main()
