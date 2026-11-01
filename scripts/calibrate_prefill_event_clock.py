"""Develop a training-only batch clock on the retained, audited v2 archive.

Always keep the historical rejected model. Freeze before opening retrospective
scoring rows; no native GPU run or confirmatory acceptance is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab import effective_calibration as base,delayed_calibration as delayed,event_clock_calibration as clock
from prefill_lab.contracts import digest,save
from scripts.analyze_prefill_arrival_capture import original_audit
from scripts.analyze_prefill_effective_capture import archive_identity,endpoint_summary


def training_inputs(capture,plan):
    """Only frozen TRAIN membership can cause a native file to be opened."""
    rows=[];events=[];cases={c['id']:c for c in plan['cases']}
    for t in plan['order']:
        if t['stage']!='train':continue
        raw=json.loads((capture/(t['trial_id']+'.json')).read_text())
        row=base.extract(raw,t,cases[t['case']]);rows.append(row)
        if t['observer']=='light':events.append(clock.event_labels(raw,t,cases[t['case']]))
    return rows,events


def development_scores(rows,plan,predicted):
    """Same old numeric rules, explicit retrospective label, no accepted verdict."""
    off=delayed.select(rows,plan,delayed.HELD,'off');light=delayed.select(rows,plan,delayed.HELD,'light')
    held=[r for r in off if r['stage']=='held_delayed'];queue=[]
    for case in sorted({r['case'] for r in held}):
        observed=float(np.median([r['clocks']['sentinel']['queue_residence_s'] for r in held if r['case']==case]))
        value=predicted[case]['sentinel_queue_residence_s'];limit=max(plan['queue_absolute_s'],plan['queue_relative']*observed)
        queue.append(dict(case=case,measured_median_s=observed,predicted_s=value,error_s=value-observed,
                          limit_s=limit,passed=abs(value-observed)<=limit))
    endpoints=base.endpoint_scores(held,predicted,plan)
    anchors={s:base.endpoint_scores([r for r in off if r['stage']==s],predicted,plan)
             for s in ('held_anchor_before','held_anchor_after')}
    quality=delayed.quality(rows,plan)
    structure=base.structure_check(light,predicted) and all(r['background_outlives_foreground'] for r in predicted.values())
    return dict(scope='retrospective development; delayed data was previously inspected; not fresh held validation',
        endpoint_scores=endpoints,endpoint_summary=endpoint_summary(endpoints['cells']),sentinel_queue_scores=queue,
        anchor_phase_scores=anchors,quality=quality,structure_and_background_lifetime_valid=structure,
        development_rules_satisfied=quality['passed'] and structure and endpoints['passed']
            and all(r['passed'] for r in queue) and all(r['passed'] for r in anchors.values()),
        model_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


def cross_validation(rows,events,plan,runtime,spec):
    result=[]
    for block in sorted({t['block'] for t in plan['order'] if t['stage']=='train'}):
        subset=dict(plan,order=[t for t in plan['order'] if not(t['stage']=='train' and t['block']==block)])
        print('Training cross-validation: omit block '+str(block),flush=True)
        frozen=clock.fit(rows,events,subset,runtime,spec,origin='retained_native_training_development')
        cases={r['case'] for r in rows};train_plan=dict(plan,cases=[c for c in plan['cases'] if c['id'] in cases])
        pred=clock.rollouts(frozen['parameters_s'],train_plan,runtime,spec)
        omitted=[r for r in events if r['block']==block]
        before=[r for r in rows if r['block']==block and r['observer']=='off']
        result.append(dict(excluded_block=block,parameters_s=frozen['parameters_s'],local_sensitivity=frozen['local_sensitivity'],
            event_scores=clock.event_scores(omitted,pred),endpoint_scores=base.endpoint_scores(before,clock.forecasts(pred),plan)))
    return result


def source_identity():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('prefill_lab/event_clock_calibration.py','prefill_lab/async_predictor.py','prefill_lab/effective_calibration.py',
         'prefill_lab/contracts.py','prefill_lab/causal.py','scripts/predict_prefill_async.py',
         'scripts/calibrate_prefill_event_clock.py',
         'experiments/prefill-event-clock-v1.json','experiments/prefill-event-clock-confirmation-v1.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();capture=a.capture.resolve();a.output.mkdir(parents=True,exist_ok=False)
    sources=source_identity()
    identity,index=archive_identity(a.archive.resolve(),capture);proof=original_audit(capture)
    read=lambda name:json.loads((capture/name).read_text())
    plan=read('calibration-plan.json');runtime=read('runtime.json');spec=clock.specification()
    rows,events=training_inputs(capture,plan)
    print('Fitting six effective coefficients from TRAIN OFF endpoints and TRAIN LIGHT event times',flush=True)
    frozen=clock.fit(rows,events,plan,runtime,spec,origin='retained_native_training_development')
    save(a.output/'spec.json',spec);save(a.output/'frozen.json',frozen)
    pred=clock.predict_frozen(frozen,plan,runtime,spec);save(a.output/'development-forecasts.json',pred)
    print('Frozen. Checking four training-only leave-one-block-out fits',flush=True)
    folds=cross_validation(rows,events,plan,runtime,spec);save(a.output/'training-cross-validation.json',folds)
    # The development forecasts and chosen coefficients already exist on disk.
    # Only now load the previously examined delayed scoring rows.
    development=development_scores(read('endpoints.json'),plan,pred);save(a.output/'development-report.json',development)
    original=read('frozen.json');cases={r['case'] for r in rows};train_plan=dict(plan,cases=[c for c in plan['cases'] if c['id'] in cases])
    from prefill_lab.async_predictor import rollout
    from scripts.predict_prefill_async import runtime_inputs
    scope,dispatch=runtime_inputs(runtime)
    baseline={c['id']:rollout(c,base.costs(original['parameters_s'],plan),dispatch,scope=scope) for c in train_plan['cases']}
    baseline_scores=clock.event_scores(events,baseline)
    if source_identity()!=sources:raise ValueError('Clock source changed during development analysis')
    result=dict(schema='effective-event-clock-development-result-v1',input_archive=identity,original_audit=proof,
        source_sha256=sources,original_frozen_sha256=original['sha256'],original_candidate_passed=False,frozen_sha256=frozen['sha256'],
        training_trials=len(rows),training_light_trials=len(events),baseline_train_event_scores=baseline_scores,
        training_event_scores=frozen['train_event_scores'],parameters_s=frozen['parameters_s'],
        cross_validation_endpoint_rules_satisfied=all(r['endpoint_scores']['passed'] for r in folds),
        development_rules_satisfied=development['development_rules_satisfied'],
        native_gpu_run_performed=False,fresh_confirmation_required=True,model_accepted=False,
        physical_costs_identified=False,adaptive_cap_accepted=False,mpc_accepted=False)
    save(a.output/'analysis.json',result);save(a.output/'input-files.json',index)
    print(json.dumps({k:result[k] for k in ('development_rules_satisfied','model_accepted','parameters_s')}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
