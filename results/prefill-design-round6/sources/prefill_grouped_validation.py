"""Revision8 structural components: two fixed contrasts, five paired blocks.

The t bound assumes independent, approximately normal block-level differences.
It is an engineering screen, not a calibrated full-experiment error guarantee.
"""
import math
import numpy as np
from scipy.stats import t

from prefill_validation import comparison_plan

CONTRAST_GROUPS = {
    'mixed_burst': tuple(f'sentinel-four-at{o}-D{d}' for d in (0, 16) for o in (50, 400)),
    'single_long': tuple(f'sentinel-one-at200-D{d}' for d in (0, 16)),
}
CONTROL_CELLS = tuple(f'sentinel-four-at800-D{d}' for d in (0, 16))
ALL_CELLS = tuple(sorted(sum(CONTRAST_GROUPS.values(), ()) + CONTROL_CELLS))
BLOCK_IDS = (1, 2, 3, 4, 5)
T95_DF4 = float(t.ppf(.95, 4))


def grouped_comparison_plan(*args, **kwargs):
    """Freeze prediction-only eligibility; never choose groups using held errors."""
    plan = comparison_plan(*args, **kwargs)
    plan['protocol'] = 'prefill-grouped-paired-v8'
    plan['contrast_groups'] = {k: list(v) for k, v in CONTRAST_GROUPS.items()}
    for eligibility in plan['required_structural_comparators'].values():
        if set(eligibility['no_regression_cells']) != set(ALL_CELLS):
            raise ValueError('Expected the eight preregistered cells')
        eligible = set(eligibility['eligible_cells'])
        # All constituent cells must exceed their own resolution, including both D levels.
        groups = [g for g, members in CONTRAST_GROUPS.items() if set(members) <= eligible]
        for obsolete in ('minimum_eligible', 'comparison_evaluable', 'strict_wins_required'):
            eligibility.pop(obsolete)
        eligibility.update(eligible_groups=groups, minimum_distinct_groups=2,
                           comparison_evaluable=len(groups) == 2,
                           note='Two fixed structural groups required; offsets/D levels are not independent contrasts.')
    return plan


def paired_components(eligibility, candidate_errors, baseline_errors):
    """Vectorized (...,8,5) scoring, used by production wrapper AND simulations.

    Arrays have canonical ALL_CELLS order. Average cells within each block BEFORE
    computing mean/SD over five blocks; never use n=number of cells times five.
    """
    if set(eligibility['no_regression_cells']) != set(ALL_CELLS):
        raise ValueError('All eight cells required')
    thresholds = eligibility['threshold_s_by_cell']
    if set(thresholds) != set(ALL_CELLS) or any(not math.isfinite(v) or v < .025 for v in thresholds.values()):
        raise ValueError('Invalid frozen eligibility thresholds')
    eligible = set(eligibility['eligible_cells'])
    if not eligible <= set(ALL_CELLS):
        raise ValueError('Unknown eligible cell')
    groups = [g for g, members in CONTRAST_GROUPS.items() if set(members) <= eligible]
    if eligibility['eligible_groups'] != groups or eligibility['comparison_evaluable'] != (len(groups) == 2):
        raise ValueError('Invalid frozen contrast groups')
    a, b = np.asarray(candidate_errors, float), np.asarray(baseline_errors, float)
    if a.shape != b.shape or a.shape[-2:] != (8, 5) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite paired (...,8,5) errors')
    a, b = np.abs(a), np.abs(b)
    per_cell_gain = (b-a).mean(axis=-1)
    regressions = -per_cell_gain > np.maximum(.1*b.mean(axis=-1), .025)
    group_results = {}
    wins = []
    for name, members in CONTRAST_GROUPS.items():
        indices = [ALL_CELLS.index(k) for k in members]
        gains = (b-a)[..., indices, :].mean(axis=-2)
        mean = gains.mean(axis=-1)
        sd = gains.std(axis=-1, ddof=1)
        lower = mean - T95_DF4*sd/math.sqrt(5)
        win = (lower > 1e-12) & (name in groups)
        wins.append(win)
        group_results[name] = dict(block_gains_s=gains, mean_gain_s=mean,
                                   sd_gain_s=sd, one_sided_95_lower_s=lower, win=win)
    failed = regressions.any(axis=-1)
    passed = np.logical_and.reduce(wins) & ~failed & (len(groups) == 2)
    return dict(passed=passed, failed=failed, per_cell_gain_s=per_cell_gain,
                regressions=regressions, groups=group_results)


def grouped_error_check(eligibility, candidate_errors, baseline_errors, *, block_ids, block_dependence_status="unresolved"):
    if block_dependence_status not in ("qualified", "unresolved", "violated"):
        raise ValueError("Invalid block-dependence status")
    if tuple(block_ids) != BLOCK_IDS:
        raise ValueError('Errors must have the five canonical paired block IDs in order')
    if set(candidate_errors) != set(ALL_CELLS) or set(baseline_errors) != set(ALL_CELLS):
        raise ValueError('All eight cells including controls required')
    a = np.asarray([candidate_errors[k] for k in ALL_CELLS], float)
    b = np.asarray([baseline_errors[k] for k in ALL_CELLS], float)
    if a.shape != (8, 5) or b.shape != (8, 5):
        raise ValueError('Expected 8x5 paired errors')
    r = paired_components(eligibility, a, b)
    groups = {g: {k: v.tolist() for k, v in row.items()} for g, row in r['groups'].items()}
    nominal = 'failed' if r['failed'] else 'passed' if r['passed'] else 'inconclusive'
    status = 'inconclusive' if nominal == 'passed' and block_dependence_status != 'qualified' else nominal
    return dict(component_status=status, nominal_component_status=nominal,
                block_dependence_status=block_dependence_status,
                eligible_groups=eligibility['eligible_groups'], groups=groups,
                regressions=[k for k, failed in zip(ALL_CELLS, r['regressions']) if failed],
                per_cell_gain_s=dict(zip(ALL_CELLS, r['per_cell_gain_s'].tolist())),
                block_ids=list(block_ids), block_sample_count=5,
                complete_experiment_passed=False, statistical_significance_claimed=False)


def grouped_structural_check(plan, candidate_errors, comparator_errors, *, block_ids, block_dependence_status="unresolved"):
    if plan.get('protocol') != 'prefill-grouped-paired-v8' or plan.get('contrast_groups') != {k: list(v) for k, v in CONTRAST_GROUPS.items()}:
        raise ValueError('Missing revision8 frozen plan')
    required = plan['required_structural_comparators']
    if plan['structure_status'] == 'not_applicable':
        if required or comparator_errors:
            raise ValueError('M0 has no structural gate')
        return dict(structural_status='not_applicable', complete_experiment_passed=False)
    if not required or 'calibrated_M0' not in required or set(comparator_errors) != set(required):
        raise ValueError('Every calibrated structural comparator is required')
    if plan['selected_family'] == 'M2' and plan['retained_base'] == 'M1' and 'calibrated_M1' not in required:
        raise ValueError('Retained M1 comparator is required')
    results = {k: grouped_error_check(e, candidate_errors, comparator_errors[k], block_ids=block_ids, block_dependence_status=block_dependence_status)
               for k, e in required.items()}
    statuses = [r['component_status'] for r in results.values()]
    status = 'failed' if 'failed' in statuses else 'inconclusive' if 'inconclusive' in statuses else 'components_passed'
    return dict(structural_status=status, comparators=results, complete_experiment_passed=False)


def calibration_noise_readiness(repeated_durations, *, expected_strata):
    """Matched work/mode/position strata, one duration per calibration block1–3.

    Caller must freeze stratum identity before opening labels and verify actual
    work/mode/padding identity. Missing strata never disappear from the check.
    This only checks the noise envelope; it cannot approve a GPU/model experiment.
    """
    if not expected_strata or set(repeated_durations) != set(expected_strata):
        return dict(status='inconclusive', reason='missing or unexpected matched strata', model_accepted=False)
    cvs = {}
    for key in expected_strata:
        a = np.asarray(repeated_durations[key], float)
        if a.shape != (3,) or not np.isfinite(a).all() or (a <= 0).any():
            return dict(status='inconclusive', reason='need three finite positive matched-trial durations', model_accepted=False)
        cvs[key] = float(a.std(ddof=1)/a.mean())
    maximum = max(cvs.values())
    return dict(status='stop_structural_claim' if maximum > .05 else 'within_tested_noise_range',
                max_cv=maximum, cv_by_stratum=cvs, warning_above_2pct=maximum > .02,
                threshold_cv=.05, model_accepted=False,
                note='Three repeats give an imprecise CV; within range is not go/acceptance or a power guarantee.')
