"""Exploratory A/B-only model diagnostics; no capture or production source changes."""
import copy
import hashlib
import json
import math
import os
import sys
from bisect import bisect_left
from pathlib import Path

CODE = Path(os.environ.get('SHARED_CLIENT_CODE_ROOT', Path(__file__).resolve().parents[3])).resolve()
BASE = Path(os.environ['SHARED_CLIENT_ANALYSIS_OUTPUT']).resolve()
RAW = Path(os.environ['SHARED_CLIENT_RAW_ROOT']).resolve()
sys.path.insert(0, str(CODE))
import numpy as np
import forecast_baselines as f
from identification_features import GatewayReplay

TARGETS = ('awaiting_first_content', 'cohort_ttft_met', 'cohort_admitted_ttft_missed')
_, originals = f.load(BASE / 'features')
original_design = f.design


class PhaseReplay(GatewayReplay):
    def _apply(self, moment, kind, record):
        super()._apply(moment, kind, record)
        if kind == 'accepted':
            self.active[record['request_id']].update(
                input_tokens=record['estimated_input_tokens'],
                output_cap=record['max_output_tokens'], admitted_at=moment)

    def phase(self, t):
        state = self.state(t)
        pending = [e for e in self.active.values() if e['admitted'] and e['first'] is None]
        streaming = [e for e in self.active.values() if e['admitted'] and e['first'] is not None]
        ages = [t-e['first'] for e in streaming]
        pending_ages = [t-e['received'] for e in pending]
        inputs = sum(e['input_tokens'] for e in pending)
        result = dict(
            phase_pending_input_tokens=inputs,
            phase_pending_reserved_cost=sum(e['cost'] for e in pending),
            phase_pending_age_mean=float(np.mean(pending_ages)) if pending_ages else 0.,
            phase_pending_token_weighted_age=sum(e['input_tokens']*(t-e['received']) for e in pending)/max(inputs, 1),
            phase_pending_over_receipt_slo=sum(a>.5 for a in pending_ages),
            phase_streaming_input_tokens=sum(e['input_tokens'] for e in streaming),
            phase_streaming_reserved_cost=sum(e['cost'] for e in streaming),
            phase_streaming_output_caps=sum(e['output_cap'] for e in streaming),
            phase_streaming_age_mean=float(np.mean(ages)) if ages else 0.,
            phase_streaming_age_max=max(ages, default=0.),
            phase_streaming_age_min=min(ages, default=0.),
            phase_positive_headroom=max(state['budget_headroom'], 0.),
            phase_over_budget_cost=max(-state['budget_headroom'], 0.))
        edges = (0., 1., 2., 4., 6., 8., math.inf)
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            result[f'phase_streaming_age_bin_{i}'] = sum(lo<=a<hi for a in ages)
        return result


def known_plan(row, schedule, h, delay=.05):
    """At t use observed cap, future predeclared commands and a fixed 50 ms delay.

    No future application time is accessed. Commands already past the assumed
    application time are represented by the observed current cap.
    """
    t = row['t_s']
    cap = row['features']['budget']
    cursor, area = t, 0.
    for step in schedule[1:]:
        at = step['at_seconds'] + delay
        if t < at <= t+h:
            area += (at-cursor)*cap
            cursor, cap = at, step['budget']
    area += (t+h-cursor)*cap
    return dict(budget_mean=area/h, budget_end=cap)


enriched = copy.deepcopy(originals)
provenance = {}
for name, run in enriched.items():
    side = 'a' if name == 'shared-client-a-r1' else 'b'
    root = RAW / f'shared-{side}' / 'results' / name
    manifest = json.loads((root/'manifest.json').read_text())
    start = manifest['measurement_start']
    records = [json.loads(s) for s in (root/'gateway.jsonl').read_text().splitlines()]
    initial = manifest['initial_metrics']
    replay = PhaseReplay(records, start, initial['admission_budget'], initial.get('budget_version', 0))
    schedule = json.loads((root/'replay_plan.json').read_text())['budget_schedule']
    trace = json.loads((root/'trace.json').read_text())
    arrivals = [r['arrival_seconds'] for r in trace['requests']]
    input_prefix = np.r_[0., np.cumsum([r['input_tokens'] for r in trace['requests']])]
    original_rows = originals[name]['rows']
    for i, row in enumerate(run['rows']):
        t = row['t_s']
        row['phase'] = replay.phase(t)
        row['history'] = {
            f'lag_{lag}s_{key}': original_rows[i-int(lag/.5)]['features'][key] if i>=int(lag/.5) else None
            for lag in (1, 2, 5)
            for key in ('reserved_cost','awaiting_first_content','streaming','completed_1s','accepted_1s','resolved_ttft_met_1s')}
        row['planned'] = {str(h):known_plan(row,schedule,h) for h in (1,2,5)}
        left, right = bisect_left(arrivals,t), bisect_left(arrivals,t+5)
        # Diagnostic upper-information comparison only, never a causal online feature.
        row['oracle'] = dict(diagnostic_actual_future_offered_5s=right-left,
                             diagnostic_actual_future_input_tokens_5s=float(input_prefix[right]-input_prefix[left]))
        assert row['phase']['phase_streaming_age_min'] >= 0.
        assert row['features']['streaming'] == sum(row['phase'][f'phase_streaming_age_bin_{k}'] for k in range(6))
        assert abs(row['phase']['phase_pending_reserved_cost']+row['phase']['phase_streaming_reserved_cost']-row['features']['reserved_cost'])<1e-6
    provenance[name] = dict(runtime=manifest['git_commit'], gateway_sha256=hashlib.sha256((root/'gateway.jsonl').read_bytes()).hexdigest(),
                            snapshot_counts=run['meta']['state_reconstruction']['snapshots'])


def configure(groups):
    runs = copy.deepcopy(enriched)
    for run in runs.values():
        for row in run['rows']:
            for group in groups:
                row['features'].update(row[group])
    return runs


def design_planned(rows,names,key):
    matrix = original_design(rows,names,key)
    for i,row in enumerate(rows):
        v = row['planned'][key]
        matrix[i,-2:] = v['budget_mean'],v['budget_end']
    return matrix


def predict_detail(runs, evaluation, design):
    names=f.feature_names(runs)
    results=[]
    for h in (1.,2.,5.):
        key=f.label_key(h)
        for target in TARGETS:
            prepared={}
            for name,item in runs.items():
                rows=item['rows']
                prepared[name]=dict(rows=rows,duration=item['meta']['duration_s'],
                    t=np.array([r['t_s'] for r in rows]), x=design(rows,names,key),
                    y=f.target(rows,key,target), known=f.known_at(rows,key,h,target))
            for split,masks in f.splits(runs,2/3):
                train,test=masks(prepared)
                train={name:train[name]&~np.isnan(item['y']) for name,item in prepared.items()}
                test={name:test[name]&~np.isnan(item['y']) for name,item in prepared.items()}
                model=f.Ridge(1.).fit(np.vstack([item['x'][train[name]] for name,item in prepared.items()]),
                    np.concatenate([item['y'][train[name]] for name,item in prepared.items()]))
                for name,item in prepared.items():
                    pick=test[name]
                    if not pick.any():continue
                    pred=model.predict(item['x'][pick]); actual=item['y'][pick]
                    rr=[row for row,p in zip(item['rows'],pick) if p]
                    for row,p,a in zip(rr,pred,actual):
                        results.append(dict(split=split,run=name,horizon_s=h,target=target,
                            t_s=row['t_s'],budget=row['features']['budget'],prediction=float(p),actual=float(a),error=float(p-a),
                            future_offered=row['labels'][key]['cohort_offered'],
                            phase=row.get('phase'),since_change=row['features']['since_budget_value_change_s']))
    return results


def main():
    output=BASE/'diagnostics'
    output.mkdir(exist_ok=True)
    variants=[('baseline_realized',(),False),('phase_realized',('phase',),False),
              ('history_realized',('history',),False),('phase_history_realized',('phase','history'),False),
              ('no_future_control',(),False),
              ('baseline_planned',(),True),('phase_planned',('phase',),True),
              ('phase_history_planned',('phase','history'),True),
              ('oracle_arrivals_diagnostic_only',('oracle',),False)]
    compact={}
    for name,groups,planned in variants:
        runs=configure(groups)
        design=design_planned if planned else original_design
        if name == 'no_future_control':
            def design(rows,names,key):
                matrix=original_design(rows,names,key)
                matrix[:,-2:]=0.
                return matrix
        f.design=design
        evaluation=f.evaluate(runs,horizons=[1.,2.,5.] if not groups==('oracle',) else [5.],windows=[1.,5.],fraction=2/3,alpha=1.)
        (output/f'{name}.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        detail=predict_detail(runs,evaluation,design) if name in ('baseline_realized','phase_planned','phase_history_planned') else []
        if detail:(output/f'{name}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
        wanted=[r for r in evaluation['results'] if r['target'] in TARGETS and r['subset'] in ('all','near_budget_change')
                and (r['split']!='chronological' or r['scope']=='pooled')]
        compact[name]=wanted
        print(name, 'features',len(evaluation['feature_columns']),flush=True)
        for r in wanted:
            if r['horizon_s']==5 and r['subset']=='all':
                print(r['split'],r['target'],'MAE',round(r['models']['ridge']['mae'],3),
                    'skill',round(r['skill_mae']['ridge_vs_persistence'],3),flush=True)
    f.design=original_design
    (output/'summary.json').write_text(json.dumps(dict(development_only=True,C_used=False,provenance=provenance,
        code_commit='d0ccb2c0d3711e1b2a69bb4ec248c61961ce3586',
        planned_delay_assumption_s=.05,oracle_is_not_deployable=True,results=compact),indent=2)+'\n')


if __name__ == '__main__':
    main()
