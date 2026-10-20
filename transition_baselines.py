"""Evaluate planned-cap flow rollouts; retrospective development, not MPC acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from analyze_trace_dynamics import dump, finite
from forecast_baselines import Ridge, metrics, skill
from transition_features import ASSUMPTION, FLOWS, HORIZONS, PROTOCOL, STATES, conservation

BASE = STATES + ('in_flight', 'budget', 'budget_headroom', 'received_5s', 'offered_cost_5s',
    'accepted_input_tokens_5s', 'first_content_5s', 'completed_5s',
    'resolved_ttft_met_5s', 'resolved_admitted_ttft_missed_5s')
ENGINE = ('engine_waiting', 'engine_running', 'engine_kv_cache_usage', 'engine_missing')
TARGETS = STATES + ('cohort_ttft_met',)
MODELS = ('persistence', 'mean', 'direct_ridge', 'flow_ridge', 'flow_ridge_engine')
OPTIMISTIC = dict(awaiting_first_content='under', reserved_cost='under', streaming=None, cohort_ttft_met='over')


def vector(features, cap, engine=False):
    """Allowlist excludes future labels, realized applications, run identity and time."""
    values = dict(features, budget=cap, budget_headroom=cap - features['reserved_cost'],
                  in_flight=features['awaiting_first_content'] + features['streaming'])
    return [float(values[key]) if values.get(key) is not None else math.nan
            for key in BASE + (ENGINE if engine else ())]


def advance(state, prediction):
    """Project predicted aggregate flows onto nonnegative conservation constraints.

    This is not an exact admission simulator: request-level cost composition and
    within-second arrival/release ordering remain unmodeled. Budget is not a clamp
    on reserved cost because lowering the cap cannot cancel existing work.
    """
    if len(prediction) != len(FLOWS) or any(not finite(v) for v in prediction):
        raise ValueError('Flow prediction must contain six finite values')
    if any(not finite(state[key]) or state[key] < 0 for key in STATES):
        raise ValueError('State must be finite and nonnegative')
    raw = np.array(prediction, dtype=float)
    a, first, eq, ed, ci, co = np.maximum(raw, 0.)
    q, d, r = (state[key] for key in STATES)
    first = min(first, q + a)
    eq = min(eq, q + a - first)
    ed = min(ed, d + first)
    ci = ci if a > 0 else 0.
    co = min(co, r + ci) if eq + ed > 0 else 0.
    if q + a - first - eq + d + first - ed <= 1e-12:
        co = r + ci
    projected = dict(zip(FLOWS, (a, first, eq, ed, ci, co)))
    result = conservation(state, projected)
    result = {key: max(0., value) for key, value in result.items()}
    return result, float(np.abs(raw - np.array(list(projected.values()))).sum())


def eligible(rows, horizon, cutoff=math.inf, reward=False):
    key = str(horizon)
    result = []
    for row in rows:
        label = row['labels'][key]
        known = row['t_s'] + horizon
        if label is None or not row['label_window_complete'][key]:
            continue
        if reward:
            if not finite(label.get('cohort_ttft_met')) or not finite(label.get('cohort_resolved_s')):
                continue
            known = max(known, label['cohort_resolved_s'])
        if known <= cutoff:
            result.append(row)
    return result


def fit_head(rows, labels, engine=False):
    if len(rows) <= len(BASE) + (len(ENGINE) if engine else 0):
        raise ValueError('Too few labeled training rows')
    x = np.array([vector(row['features'], row['planned_caps'][0], engine) for row in rows])
    y = np.array(labels, dtype=float)
    if not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError('Invalid training values')
    return Ridge(1.).fit(x, y)


def serialize(head):
    return dict(alpha=head.alpha, center=head.center.tolist(), scale=head.scale.tolist(),
                intercept=head.intercept, coef=head.coef.tolist())


class FlowModel:
    def __init__(self, engine=False):
        self.engine = engine

    def fit(self, flow_rows, reward_rows):
        self.heads = {key: fit_head(flow_rows, [r['next_flows'][key] for r in flow_rows], self.engine)
                      for key in FLOWS}
        self.reward = fit_head(reward_rows, [r['labels']['1']['cohort_ttft_met'] for r in reward_rows], self.engine)
        self.cap_range = (min(r['planned_caps'][0] for r in flow_rows), max(r['planned_caps'][0] for r in flow_rows))
        return self

    def forecast(self, features, caps, *, allow_extrapolation=False):
        if not 1 <= len(caps) <= max(HORIZONS) or any(not finite(c) or c <= 0 for c in caps):
            raise ValueError('Expected one to five finite positive caps')
        if not allow_extrapolation and any(not self.cap_range[0] <= c <= self.cap_range[1] for c in caps):
            raise ValueError('Cap outside fitted range; diagnostic extrapolation must be explicit')
        current = dict(features)
        total_reward, projection, outputs = 0., 0., []
        for cap in caps:
            x = np.array([vector(current, cap, self.engine)])
            total_reward += float(self.reward.predict(x)[0])
            flows = [float(self.heads[key].predict(x)[0]) for key in FLOWS]
            state, correction = advance(current, flows)
            current.update(state)
            projection += correction
            outputs.append(dict(state, cohort_ttft_met=total_reward, projection_l1=projection))
        return outputs

    def export(self):
        return dict(engine=self.engine, columns=list(BASE + (ENGINE if self.engine else ())),
                    cap_range=list(self.cap_range), heads={k: serialize(v) for k, v in self.heads.items()},
                    reward=serialize(self.reward))


def load(directory):
    manifest = json.loads((directory / 'dataset.json').read_text())
    if manifest.get('protocol') != PROTOCOL or manifest.get('horizons') != list(HORIZONS) or manifest.get('step_s') != 1.:
        raise ValueError('Unsupported transition dataset')
    runs = {}
    for entry in manifest['runs']:
        meta, filename = entry['meta'], entry['file']
        if filename != meta['name'] + '.jsonl' or Path(filename).name != filename or meta['name'] in runs:
            raise ValueError('Unsafe or duplicate run filename')
        if not (meta['qualified_for_comparison'] and meta['state_reconstruction']['valid']):
            raise ValueError('Unqualified capture')
        data = (directory / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Transition table fingerprint mismatch')
        rows = [json.loads(s) for s in data.decode().splitlines() if s.strip()]
        if not rows or len(rows) != entry['rows'] or [r['t_s'] for r in rows] != list(range(len(rows))):
            raise ValueError('Invalid transition time grid')
        for row in rows:
            if any(not finite(row['features'][k]) or row['features'][k] < 0 for k in STATES):
                raise ValueError('Invalid state')
            if len(row['planned_caps']) != max(HORIZONS) or any(not finite(c) or c <= 0 for c in row['planned_caps']):
                raise ValueError('Invalid planned caps')
        runs[meta['name']] = dict(meta=meta, rows=rows)
    if len(runs) < 2:
        raise ValueError('At least two qualified captures are required')
    if (any(not finite(r['meta']['ttft_slo']) or r['meta']['ttft_slo'] <= 0 or
            r['meta'].get('e2e_slo') is not None for r in runs.values()) or
            len({r['meta']['ttft_slo'] for r in runs.values()}) != 1):
        raise ValueError('Captures must share one positive TTFT-only objective')
    return manifest, runs


def splits(runs):
    names = list(runs)
    yield 'chronological', {n: runs[n]['meta']['duration_s'] * 2 / 3 for n in names}, names
    for held in names:
        yield 'leave_out:' + held, {n: math.inf for n in names if n != held}, [held]
    for source, target in (('fixed', 'scheduled'), ('scheduled', 'fixed')):
        training = {n: math.inf for n in names if runs[n]['meta']['budget_mode'] == source}
        testing = [n for n in names if runs[n]['meta']['budget_mode'] == target]
        if training and testing:
            yield source + '_to_' + target, training, testing


def evaluate(runs):
    scores, diagnostics, fitted = [], [], {}
    for split, cuts, test_names in splits(runs):
        flow_rows = [r for n, cut in cuts.items() for r in eligible(runs[n]['rows'], 1, cut)]
        reward_rows = [r for n, cut in cuts.items() for r in eligible(runs[n]['rows'], 1, cut, True)]
        flow_models = {name: FlowModel(name == 'flow_ridge_engine').fit(flow_rows, reward_rows)
                       for name in ('flow_ridge', 'flow_ridge_engine')}
        fitted[split] = dict(training_cutoffs={n: cut if finite(cut) else None for n, cut in cuts.items()},
                             flow_rows=len(flow_rows), reward_rows=len(reward_rows),
                             models={name: model.export() for name, model in flow_models.items()})
        for horizon in HORIZONS:
            train = [r for n, cut in cuts.items() for r in eligible(runs[n]['rows'], horizon, cut, True)]
            # Direct baseline uses planned mean/end controls too, never realized control.
            def direct_x(rows):
                return np.array([vector(r['features'], r['planned_caps'][0]) +
                                 [sum(r['planned_caps'][:horizon]) / horizon, r['planned_caps'][horizon - 1]]
                                 for r in rows])
            direct, means = {}, {}
            for target in TARGETS:
                y = np.array([r['labels'][str(horizon)][target] for r in train], dtype=float)
                direct[target] = Ridge(1.).fit(direct_x(train), y)
                means[target] = float(y.mean())
            fitted[split].setdefault('direct_heads', {})[str(horizon)] = {
                key: serialize(head) for key, head in direct.items()}
            for name in test_names:
                cutoff = cuts.get(name, 0.) if split == 'chronological' else 0.
                test = [r for r in eligible(runs[name]['rows'], horizon, reward=True) if r['t_s'] >= cutoff]
                if not test:
                    continue
                predictions = {model: {key: [] for key in TARGETS} for model in MODELS}
                projection, extrapolated, near = {}, [], []
                for row in test:
                    caps = row['planned_caps'][:horizon]
                    low, high = flow_models['flow_ridge'].cap_range
                    extrapolated.append(any(c < low or c > high for c in caps))
                    since = row['features']['since_budget_value_change_s']
                    near.append((since is not None and since <= 10.) or any(c != row['features']['budget'] for c in caps))
                    for model_name, model in flow_models.items():
                        forecast = model.forecast(row['features'], caps, allow_extrapolation=True)[-1]
                        projection.setdefault(model_name, []).append(forecast['projection_l1'])
                        for target in TARGETS:
                            predictions[model_name][target].append(forecast[target])
                    for target in TARGETS:
                        value = (row['features'][target] if target in STATES else
                                 row['features']['resolved_ttft_met_5s'] * horizon / 5.)
                        predictions['persistence'][target].append(value)
                        predictions['mean'][target].append(means[target])
                for target in TARGETS:
                    predictions['direct_ridge'][target] = direct[target].predict(direct_x(test)).tolist()
                for subset, mask in (('all', np.ones(len(test), dtype=bool)), ('near_cap_change', np.array(near))):
                    for target in TARGETS:
                        truth = np.array([r['labels'][str(horizon)][target] for r in test])
                        result = {m: metrics(np.array(predictions[m][target])[mask], truth[mask], OPTIMISTIC[target]) for m in MODELS}
                        scores.append(dict(split=split, run=name, horizon_s=horizon, subset=subset, target=target,
                            diagnostic_extrapolation_rows=int(np.array(extrapolated)[mask].sum()), models=result,
                            skill_vs_persistence={m: skill(result[m], result['persistence']) for m in MODELS[1:]}))
                if split == 'chronological':
                    diagnostics.extend(dict(run=name, t_s=row['t_s'], horizon_s=horizon,
                        actual={key: row['labels'][str(horizon)][key] for key in TARGETS},
                        predictions={m: {key: predictions[m][key][i] for key in TARGETS} for m in MODELS},
                        projection_l1={m: values[i] for m, values in projection.items()}) for i, row in enumerate(test))
    return dict(protocol=PROTOCOL, acceptance=None, c_used=False, control_assumption=ASSUMPTION,
                ttft_slo=next(iter(runs.values()))['meta']['ttft_slo'],
                runtime=dict(python=platform.python_version(), numpy=np.__version__),
                scores=scores, chronological_predictions=diagnostics, fitted=fitted,
                limitations=['Retrospective development; overlapping windows are not independent repeats.',
                    'Frozen history/engine covariates approximate endogenous feedback, not a complete simulator.',
                    'Flow conservation does not enforce request-level cost composition or exact admission ordering.',
                    'Projection L1 sums corrections across request and cost units; descriptive only, not a physical norm.',
                    'Cap range is not proof of action support; low-cap switch dynamics were not collected.',
                    'Cross-host A/B transfer is confounded; model error is not causal policy value.',
                    'Explicit diagnostic extrapolation is scored and counted, never silently accepted.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output, source = args.output_dir.resolve(), args.dataset.resolve()
    if output.exists() or output == source or source in output.parents:
        parser.error('Choose a fresh output directory outside the dataset')
    try:
        manifest, runs = load(source)
        report = evaluate(runs)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    report['dataset_sha256'] = hashlib.sha256((source / 'dataset.json').read_bytes()).hexdigest()
    report['source_runs'] = manifest['runs']
    report['analysis_code_sha256'] = {key: hashlib.sha256(Path(__file__).with_name(key).read_bytes()).hexdigest()
                                    for key in ('transition_baselines.py', 'forecast_baselines.py')}
    output.mkdir(parents=True)
    dump(output / 'transition_report.json', report)
    print(json.dumps(dict(runs=len(runs), score_rows=len(report['scores']), output=str(output)), indent=2))


if __name__ == '__main__':
    main()
