"""Calibration-only eligibility and paired held-error checks, not an MPC controller."""
import math
import numpy as np
from prefill_design import contrast_eligibility

OBSERVER_SHAPES={'single-L1024-D0':(20,1),'composition-uniform-D16':(10,4)}


def position_jitter(off_trials):
    if set(off_trials)!=set(OBSERVER_SHAPES):raise ValueError('Missing observer reference cell')
    result={}
    for name,shape in OBSERVER_SHAPES.items():
        a=np.asarray(off_trials[name],float)
        if a.shape!=shape or not np.isfinite(a).all() or (a<=0).any():raise ValueError('Incomplete/nonfinite off-mode trials')
        median=np.median(a,axis=0);sigma=1.4826*np.median(np.abs(a-median),axis=0)
        for i in range(shape[1]):result[f'{name}/foreground-{i}']=dict(median_s=float(median[i]),jitter_s=float(sigma[i]),off_trials=shape[0])
    return result


def assign_jitter(position_stats,mapping):
    """Mapping is preregistered from toy latency ranges, never from held errors."""
    out={}
    for cell,reference in mapping.items():
        if reference not in position_stats:raise ValueError('Missing preregistered position reference')
        value=position_stats[reference]['jitter_s']
        if not math.isfinite(value) or value<0:raise ValueError('Invalid position jitter')
        out[cell]=value
    return out


def comparison_plan(selected_family,retained_base,candidate,calibrated_m0,raw_frozen,jitter,calibrated_m1=None):
    """Only calibration predictions enter this plan; no held measurements allowed."""
    if selected_family not in ('M0','M1','M2') or retained_base not in ('M0','M1'):
        raise ValueError('Invalid selected family/base')
    if selected_family=='M0' and (retained_base!='M0' or candidate!=calibrated_m0):
        raise ValueError('Selected M0 must equal the calibrated M0 comparator')
    if retained_base=='M1' and calibrated_m1 is None:raise ValueError('Retained M1 comparator required')
    if selected_family=='M1' and (retained_base!='M1' or candidate!=calibrated_m1):
        raise ValueError('Selected M1 must equal the calibrated M1 comparator')
    stale=contrast_eligibility(candidate,raw_frozen,jitter)
    structural={}
    if selected_family!='M0':
        structural['calibrated_M0']=contrast_eligibility(candidate,calibrated_m0,jitter)
        if selected_family=='M2' and retained_base=='M1':
            structural['calibrated_M1']=contrast_eligibility(candidate,calibrated_m1,jitter)
    return dict(selected_family=selected_family,retained_base=retained_base,
        structure_status='not_applicable' if selected_family=='M0' else 'not_evaluated',
        required_structural_comparators=structural,raw_B2048_staleness_only=stale,
        actual_held_data_used=False,
        note='Same training data, weights, timing/accounting basis and freeze boundary for calibrated comparators. Raw historical coefficients cannot support a structural claim.')


def paired_error_check(eligibility,candidate_errors,baseline_errors):
    """Gate3 component only: paired block errors for ALL eight cells, five blocks.

    Bias/order gates, capture integrity and provenance/future-isolation checks are
    separate prerequisites; a passing component cannot pass the complete experiment.
    """
    keys=eligibility['no_regression_cells']
    thresholds=eligibility['threshold_s_by_cell']
    if len(keys)!=8 or len(set(keys))!=8 or set(thresholds)!=set(keys) or any(not math.isfinite(v) or v<=0 for v in thresholds.values()):
        raise ValueError('Eight unique cells with finite positive win margins required')
    if set(keys)!=set(candidate_errors) or set(keys)!=set(baseline_errors):raise ValueError('All8 paired error cells required')
    a=np.asarray([candidate_errors[k] for k in keys],float);b=np.asarray([baseline_errors[k] for k in keys],float)
    if a.shape!=(8,5) or b.shape!=(8,5) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected8x5 finite paired errors')
    ma=np.abs(a).mean(axis=1);mb=np.abs(b).mean(axis=1)
    rows={k:dict(candidate_mae_s=float(ma[i]),baseline_mae_s=float(mb[i]),
        improvement_s=float(mb[i]-ma[i]),win_margin_s=thresholds[k],
        paired_absolute_error_deltas_s=(np.abs(a[i])-np.abs(b[i])).tolist()) for i,k in enumerate(keys)}
    regressions=[k for i,k in enumerate(keys) if ma[i]-mb[i]>max(.1*mb[i],.025)]
    wins=[k for i,k in enumerate(keys) if k in eligibility['eligible_cells'] and mb[i]-ma[i]>thresholds[k]
          and not math.isclose(mb[i]-ma[i],thresholds[k],rel_tol=1e-12,abs_tol=1e-12)]
    status='failed' if regressions else 'passed' if eligibility['comparison_evaluable'] and len(wins)>=eligibility['strict_wins_required'] else 'inconclusive'
    return dict(component_status=status,wins=wins,regressions=regressions,per_cell=rows,
        win_rule='baseline MAE - candidate MAE > calibration-only per-cell eligibility threshold; numerical equality is a non-win',
        unresolved_cells=[k for k in eligibility['eligible_cells'] if k not in wins and k not in regressions],
        complete_experiment_passed=False,statistical_significance_claimed=False)


def structural_error_check(plan,candidate_errors,comparator_errors):
    required=plan['required_structural_comparators']
    if plan['structure_status']=='not_applicable':
        if required or comparator_errors:raise ValueError('M0 has no structural comparator gate')
        return dict(structural_status='not_applicable',complete_experiment_passed=False)
    if not required or 'calibrated_M0' not in required:raise ValueError('Missing required structural baseline')
    if set(comparator_errors)!=set(required):raise ValueError('All calibrated structural comparators required; raw baseline is insufficient')
    results={name:paired_error_check(eligibility,candidate_errors,comparator_errors[name]) for name,eligibility in required.items()}
    states=[r['component_status'] for r in results.values()]
    status='failed' if 'failed' in states else 'inconclusive' if 'inconclusive' in states else 'components_passed'
    return dict(structural_status=status,comparators=results,complete_experiment_passed=False)
