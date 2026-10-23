"""Offline held scoring consumes an already frozen calibration-only model file."""
import json
from pathlib import Path
import numpy as np
from .causal import rollout
from .analysis import rows_from_capture,predict
from .contracts import digest,save
from prefill_grouped_validation import ALL_CELLS,grouped_comparison_plan,grouped_structural_check,BLOCK_IDS
from prefill_validation import position_jitter,assign_jitter


def score(output,cases,order,freeze,dispatch,reference_mapping):
    output=Path(output)
    # Training provenance must still match; never refit after looking at held timings.
    for tid,sha in freeze['training_capture_sha256'].items():
        if digest(json.loads((output/f'{tid}.json').read_text()))!=sha:raise ValueError('Calibration capture changed after freeze')
    lookup={c['id']:c for c in cases}
    predictions={f:{k:[] for k in ALL_CELLS} for f in freeze['fits']}
    measured={k:[] for k in ALL_CELLS};bias={};conditional=[];off={k:[] for k in ('single-L1024-D0','composition-uniform-D16')}
    for trial in order:
        if trial['stage']=='observer_check' and trial['observer']=='off':
            c=json.loads((output/f'{trial["trial_id"]}.json').read_text())
            values=sorted((rid,r) for rid,r in c['requests'].items() if r['role']=='foreground')
            off[trial['case']].append([r['first']-c['release'] for _,r in values])
    jitter=assign_jitter(position_jitter(off),reference_mapping)
    blocks={k:[] for k in ALL_CELLS}
    for trial in order:
        if trial.get('analysis_role')!='held_case':continue
        c=json.loads((output/f'{trial["trial_id"]}.json').read_text());case=lookup[trial['case']]
        rows=rows_from_capture(c)
        conditional.append(dict(trial_id=trial['trial_id'],models={f:float(np.mean([abs(predict(r,m)-r['y']) for r in rows])) for f,m in freeze['fits'].items()}))
        if trial['case'] not in measured:continue
        sentinel=next(r for r in c['requests'].values() if r['role']=='sentinel')
        measured[trial['case']].append(sentinel['first']-sentinel['queue_posted'])
        blocks[trial['case']].append(trial['block'])
        for family,model in freeze['fits'].items():
            p=rollout(case,model,dispatch,c['release_state'])['foreground']['sentinel']['ttft_s']
            predictions[family][trial['case']].append(p)
    if any(tuple(blocks[k])!=BLOCK_IDS for k in ALL_CELLS):raise ValueError('Missing/reordered held blocks')
    average={f:{k:float(np.mean(v)) for k,v in cells.items()} for f,cells in predictions.items()}
    selected=freeze['selected'];base=freeze['retained_base']
    # Raw diagnostic is archived separately; structural comparators are always refitted models.
    from scripts.build_prefill_design import frozen_prediction
    import evaluate_low_cap_frozen as lf
    coefficients,overhead,_,_=lf.frozen_parameters()
    raw={k:frozen_prediction(lookup[k],2048,coefficients,overhead)['sentinel']['raw_ttft_s'] for k in ALL_CELLS}
    plan=grouped_comparison_plan(selected,base,average[selected],average['M0'],raw,jitter,calibrated_m1=average.get('M1'))
    candidate={k:(np.array(predictions[selected][k])-measured[k]).tolist() for k in ALL_CELLS}
    comparators={name:{k:(np.array(predictions[name.removeprefix('calibrated_')][k])-measured[k]).tolist() for k in ALL_CELLS} for name in plan['required_structural_comparators']}
    structural=grouped_structural_check(plan,candidate,comparators,block_ids=BLOCK_IDS,
                                      block_dependence_status=freeze['block_dependence_status'])
    readiness=freeze['noise_readiness']
    if readiness['status']!='within_tested_noise_range' and structural['structural_status']=='components_passed':
        structural['structural_status']='inconclusive'
        structural['blocked_by']='calibration_noise_readiness'
    for k in ALL_CELLS:
        error=abs(float(np.median(candidate[k])));limit=max(.1*float(np.median(measured[k])),.025)
        bias[k]=dict(median_bias_s=error,limit_s=limit,passed=error<=limit)
    rng=np.random.default_rng(20261023);ordering=[]
    for d in (0,16):
        a,b=f'sentinel-four-at50-D{d}',f'sentinel-four-at400-D{d}'
        delta=np.array(measured[a])-measured[b]
        boot=np.median(delta[rng.integers(0,5,size=(10000,5))],axis=1)
        low,high=np.quantile(boot,[.025,.975]);different=(low>0 or high<0) and abs(np.median(delta))>.025
        predicted=average[selected][a]-average[selected][b]
        ordering.append(dict(background=d,interval_s=[float(low),float(high)],distinguishable=bool(different),
                             direction_matches=bool(np.sign(predicted)==np.sign(np.median(delta))) if different else None))
    return dict(schema='prefill-held-report-v1',model_freeze_sha256=digest(freeze),plan=plan,
                predictions=predictions,measured_ttft_s=measured,structural=structural,bias=bias,ordering=ordering,
                calibration_noise_readiness=readiness,
                conditional_step_mae=conditional,complete_experiment_passed=False,mpc_evaluated=False,
                interpretation='Structural components remain inconclusive until calibration-only dependence is qualified; bias/order/integrity are separate gates')
