"""Frozen-model diagnostic oracles, never deployable forecasts or causal attribution."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_trace_dynamics import dump
from forecast_baselines import metrics
from transition_baselines import BASE, ENGINE, OPTIMISTIC, TARGETS, FlowModel, advance, eligible, load, serialize, vector
from transition_features import FLOWS, STATES, conservation

MODES = ('free', 'state_reset', 'history_refresh', 'engine_refresh', 'history_engine_refresh',
         'all_observations_reset', 'admission_flow_oracle', 'first_flow_oracle',
         'exit_flow_oracle', 'all_flows_oracle', 'arrival_history_refresh',
         'activity_history_refresh', 'outcome_history_refresh', 'activity_outcome_refresh')
HISTORY = tuple(k for k in BASE if k not in STATES + ('in_flight', 'budget', 'budget_headroom'))
ARRIVAL_HISTORY = ('received_5s', 'offered_cost_5s')
ACTIVITY_HISTORY = ('accepted_input_tokens_5s', 'first_content_5s', 'completed_5s')
OUTCOME_HISTORY = ('resolved_ttft_met_5s', 'resolved_admitted_ttft_missed_5s')
HISTORY_GROUPS = {'arrival_history_refresh': ARRIVAL_HISTORY, 'activity_history_refresh': ACTIVITY_HISTORY,
                  'outcome_history_refresh': OUTCOME_HISTORY,
                  'activity_outcome_refresh': ACTIVITY_HISTORY + OUTCOME_HISTORY}
ORACLE_FLOWS = {'admission_flow_oracle': ('admitted', 'cost_in'), 'first_flow_oracle': ('first',),
                'exit_flow_oracle': ('exit_awaiting', 'exit_streaming', 'cost_out'),
                'all_flows_oracle': FLOWS}


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError('Archived model fields differ')
        for k in a:
            close(a[k], b[k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError('Archived model dimensions differ')
        for x, y in zip(a, b):
            close(x, y)
    elif isinstance(a, (float, int)) and not isinstance(a, bool) and isinstance(b, (float, int)) and not isinstance(b, bool):
        if not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError('Archived model numeric mismatch')
    elif a != b:
        raise ValueError('Archived model value mismatch')


def rollout(model, rows, origin, mode):
    """Replace observations/flows with truth only in explicitly named oracles.

    Cap inputs always come from the origin's premeasurement plan. Refreshes use
    future measured data, including endogenous histories: they cannot establish
    deployable improvements or additively attribute physical causes.
    """
    if mode not in MODES:
        raise ValueError('Unknown diagnostic mode')
    first = rows[origin]
    caps = first['planned_caps'][:5]
    if len(caps) != 5 or origin + 5 > len(rows):
        raise ValueError('Five complete diagnostic steps required')
    current = dict(first['features'])
    total_reward, actual_reward, outputs = 0., 0., []
    for j, cap in enumerate(caps):
        observed = rows[origin + j]
        truth = observed['features']
        if mode in ('state_reset', 'all_observations_reset'):
            current.update({k: truth[k] for k in STATES})
        if mode in ('history_refresh', 'history_engine_refresh', 'all_observations_reset'):
            current.update({k: truth[k] for k in HISTORY})
        current.update({k: truth[k] for k in HISTORY_GROUPS.get(mode, ())})
        if mode in ('engine_refresh', 'history_engine_refresh', 'all_observations_reset'):
            current.update({k: truth[k] for k in ENGINE})
        x = np.array([vector(current, cap, model.engine)])
        reward = float(model.reward.predict(x)[0])
        learned = {k: float(model.heads[k].predict(x)[0]) for k in FLOWS}
        used = dict(learned)
        for k in ORACLE_FLOWS.get(mode, ()):
            used[k] = observed['next_flows'][k]
        state, correction = advance(current, [used[k] for k in FLOWS])
        total_reward += reward
        actual_reward += observed['labels']['1']['cohort_ttft_met']
        actual = {k: observed['labels']['1'][k] for k in STATES}
        actual['cohort_ttft_met'] = actual_reward
        outputs.append(dict(step=j+1, prediction=dict(state, cohort_ttft_met=total_reward), actual=actual,
            reward=reward, actual_reward=observed['labels']['1']['cohort_ttft_met'],
            learned_flows=learned, actual_flows=observed['next_flows'], projection_l1=correction))
        current.update(state)
    for h in (1, 2, 5):
        close(outputs[h-1]['actual'], {k: first['labels'][str(h)][k] for k in TARGETS})
    return outputs


def tags(run, rows, origin):
    row = rows[origin]
    since = row['features']['since_budget_value_change_s']
    near = (since is not None and since <= 10.) or any(c != row['features']['budget'] for c in row['planned_caps'][:5])
    result = ['all', 'near_cap_change' if near else 'away_from_cap_change', run['meta']['budget_mode']]
    origin_cap = row['features']['budget']
    for c in row['planned_caps'][:5]:
        if c != origin_cap:
            result.append('upcoming_up' if c > origin_cap else 'upcoming_down')
            break
    return result


def evaluate(runs, archived):
    cuts = {n: run['meta']['duration_s']*2/3 for n, run in runs.items()}
    flow_train = [r for n, cut in cuts.items() for r in eligible(runs[n]['rows'], 1, cut)]
    reward_train = [r for n, cut in cuts.items() for r in eligible(runs[n]['rows'], 1, cut, True)]
    models = {'flow_ridge': FlowModel(False).fit(flow_train, reward_train),
              'flow_ridge_engine': FlowModel(True).fit(flow_train, reward_train)}
    for name, model in models.items():
        close(model.export(), archived['fitted']['chronological']['models'][name])
    original = {(r['run'], r['t_s']): r for r in archived['chronological_predictions'] if r['horizon_s']==5}
    grouped, flow_errors, records, examples = {}, {}, [], []
    verified = 0
    for name, run in runs.items():
        rows = run['rows']
        selected = [r for r in eligible(rows, 5, reward=True) if r['t_s'] >= cuts[name]]
        for row in selected:
            origin = int(row['t_s'])
            if rows[origin] is not row:
                raise ValueError('Expected audited integer-second grid')
            subsets = tags(run, rows, origin) + ['run:'+name]
            for model_name, model in models.items():
                finals = {}
                for mode in MODES:
                    outputs = rollout(model, rows, origin, mode)
                    finals[mode] = outputs[-1]['prediction']
                    if mode == 'free':
                        close(finals[mode], original[name, row['t_s']]['predictions'][model_name])
                        close(sum(o['projection_l1'] for o in outputs), original[name,row['t_s']]['projection_l1'][model_name])
                        verified += 1
                    if mode == 'all_flows_oracle':
                        close({k: finals[mode][k] for k in STATES}, {k: row['labels']['5'][k] for k in STATES})
                    for output in outputs:
                        for subset in subsets:
                            key = model_name, mode, subset, output['step']
                            grouped.setdefault(key, []).append(output)
                        if mode == 'all_observations_reset':
                            flow_errors.setdefault(model_name, {})[name, origin+output['step']-1] = output
                repeat = sum(float(model.reward.predict(np.array([vector(row['features'],c,model.engine)]))[0])
                             for c in row['planned_caps'][:5])
                records.append(dict(run=name, t_s=row['t_s'], model=model_name, subsets=subsets,
                    actual={k: row['labels']['5'][k] for k in TARGETS}, predictions=finals,
                    origin_repeat_reward=repeat))
    summaries = []
    for (model, mode, subset, step), values in sorted(grouped.items()):
        scores = {k: metrics(np.array([v['prediction'][k] for v in values]),
                             np.array([v['actual'][k] for v in values]), OPTIMISTIC[k]) for k in TARGETS}
        summaries.append(dict(model=model, mode=mode, subset=subset, step=step,
            scores=scores, goodput_rate_mae=scores['cohort_ttft_met']['mae']/step,
            projected_steps=sum(v['projection_l1']>1e-9 for v in values)))
    heads, correlations, repeat_scores = {}, {}, {}
    for name, unique in flow_errors.items():
        values = list(unique.values())
        heads[name] = {k: metrics(np.array([v['learned_flows'][k] for v in values]),
                                  np.array([v['actual_flows'][k] for v in values]), None) for k in FLOWS}
        residuals = np.array([[v['learned_flows'][k]-v['actual_flows'][k] for k in FLOWS] for v in values])
        correlations[name] = {a:{b:(float(np.corrcoef(residuals[:,i],residuals[:,j])[0,1])
                                   if min(residuals[:,i].std(),residuals[:,j].std())>1e-12 else None)
                                for j,b in enumerate(FLOWS)} for i,a in enumerate(FLOWS)}
        selected = [r for r in records if r['model']==name]
        predicted = np.array([r['origin_repeat_reward'] for r in selected])/5
        actual = np.array([r['actual']['cohort_ttft_met'] for r in selected])/5
        repeat_scores[name] = dict(scores=metrics(predicted,actual,'over'),
            mean_absolute_difference_from_free_rate=float(np.abs(predicted-np.array([
                r['predictions']['free']['cohort_ttft_met']/5 for r in selected])).mean()))
    for model in models:
        chosen = sorted((r for r in records if r['model']==model),
                        key=lambda r: abs(r['predictions']['free']['cohort_ttft_met']-r['actual']['cohort_ttft_met']),reverse=True)[:12]
        examples.extend(chosen)
    return dict(protocol='flow-error-diagnosis-v1', acceptance=None, c_used=False,
        models_unchanged=True, archived_coefficients_match=True, verified_free_rollouts=verified,
        scores=summaries, one_step_flow_errors_with_observed_inputs=heads,
        observed_input_flow_residual_correlations=correlations,
        origin_only_repeat_goodput=repeat_scores,
        examples=examples, predictions=records,
        limitations=['Development diagnosis on already examined captures; no independent acceptance.',
          'Every mode except free uses future measured observations or flows and is not deployable.',
          'Oracle groups overlap and are not additive, exclusive, or physical causal attribution.',
          'Flow replacement still passes through the original projection; constraints can couple heads.',
          'Step errors use the same five-second eligible origins; shorter-step counts differ from the original full horizon tests.',
          'No new cap interventions, workload windows, GPU measurements or future-data training.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--archived-report',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    source, archive, output = args.dataset.resolve(), args.archived_report.resolve(), args.output_dir.resolve()
    if output.exists() or output==source or source in output.parents:
        parser.error('Use fresh output outside the dataset')
    sources = ('diagnose_flow_rollout.py','transition_baselines.py','transition_features.py','forecast_baselines.py')
    checksums = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources}
    manifest, runs = load(source)
    raw = archive.read_bytes()
    archived = json.loads(gzip.decompress(raw))
    report = evaluate(runs, archived)
    if checksums != {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources}:
        raise ValueError('Source changed during diagnosis')
    report.update(code_sha256=checksums, dataset_sha256=hashlib.sha256((source/'dataset.json').read_bytes()).hexdigest(),
        archived_report_sha256=hashlib.sha256(raw).hexdigest(), source_runs=manifest['runs'])
    output.mkdir(parents=True)
    dump(output/'diagnosis.json',report)
    for model in ('flow_ridge','flow_ridge_engine'):
        print(model)
        for score in report['scores']:
            if score['model']==model and score['subset']=='all' and score['step']==5:
                print(score['mode'], 'G/rate MAE',round(score['goodput_rate_mae'],6),
                      'Q',round(score['scores']['awaiting_first_content']['mae'],6),
                      'R',round(score['scores']['reserved_cost']['mae'],6))


if __name__=='__main__':
    main()
