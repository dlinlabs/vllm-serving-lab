import copy
import json
import numpy as np
import pytest

from prefill_design import COEF
from prefill_planning import collect_training, select
from prefill_grouped_validation import (ALL_CELLS, BLOCK_IDS, CONTRAST_GROUPS, T95_DF4,
    grouped_comparison_plan, grouped_error_check, grouped_structural_check,
    paired_components, calibration_noise_readiness)
from prefill_noise import (cases_at_revision, fit_scenario, freeze_plan, training_labels,
                          fit_labels, select_labels, held_noise, evaluate_draws)
from scripts.build_prefill_design import cases, ROOT, digest


def plan_for_gaps(single=.06, mixed=.06):
    baseline = {k: .5 for k in ALL_CELLS}
    candidate = {k: .5+(single if 'one-' in k else mixed if 'at800' not in k else 0.) for k in ALL_CELLS}
    return grouped_comparison_plan('M2', 'M0', candidate, baseline, baseline, {k: .01 for k in ALL_CELLS})


def test_four_eligible_cells_are_only_one_group():
    plan = plan_for_gaps(single=.02)
    e = plan['required_structural_comparators']['calibrated_M0']
    assert len(e['eligible_cells']) == 4 and e['eligible_groups'] == ['mixed_burst']
    a = {k: [0.]*5 for k in ALL_CELLS}; b = {k: [.1]*5 for k in ALL_CELLS}
    assert grouped_error_check(e, a, b, block_ids=BLOCK_IDS)['component_status'] == 'inconclusive'


def test_paired_t_uses_five_block_means_not_twenty_cell_observations():
    e = plan_for_gaps()['required_structural_comparators']['calibrated_M0']
    gains = np.array([.01,.02,.03,.04,.05])
    a = {k: [0.]*5 for k in ALL_CELLS}; b = {k: gains.tolist() for k in ALL_CELLS}
    r = grouped_error_check(e, a, b, block_ids=BLOCK_IDS, block_dependence_status='qualified')
    assert r['component_status'] == 'passed'
    for g in CONTRAST_GROUPS:
        assert r['groups'][g]['one_sided_95_lower_s'] == pytest.approx(gains.mean()-T95_DF4*gains.std(ddof=1)/np.sqrt(5))
    assert r['block_sample_count'] == 5


def test_adding_correlated_member_disagreement_does_not_create_more_samples():
    e = plan_for_gaps()['required_structural_comparators']['calibrated_M0']
    a = np.zeros((8,5)); b = np.full((8,5),.04)
    # Opposite within-block member deviations cancel before computing block SD.
    b[ALL_CELLS.index(CONTRAST_GROUPS['mixed_burst'][0])] += [.01,-.01,0,.01,-.01]
    b[ALL_CELLS.index(CONTRAST_GROUPS['mixed_burst'][1])] -= [.01,-.01,0,.01,-.01]
    r = paired_components(e,a,b)
    assert r['groups']['mixed_burst']['sd_gain_s'] == pytest.approx(0.,abs=1e-15)


def test_equal_or_negative_gains_cannot_pass_and_controls_keep_regression_guard():
    e = plan_for_gaps()['required_structural_comparators']['calibrated_M0']
    a = {k: [.01]*5 for k in ALL_CELLS}; b = copy.deepcopy(a)
    assert grouped_error_check(e,a,b,block_ids=BLOCK_IDS)['component_status'] == 'inconclusive'
    b = {k: [.08]*5 for k in ALL_CELLS}
    a['sentinel-four-at800-D0'] = [.2]*5
    r = grouped_error_check(e,a,b,block_ids=BLOCK_IDS)
    assert r['component_status'] == 'failed' and r['regressions'] == ['sentinel-four-at800-D0']


def test_missing_blocks_nonfinite_errors_and_forged_group_support_fail_closed():
    e = plan_for_gaps()['required_structural_comparators']['calibrated_M0']
    a = {k: [0.]*5 for k in ALL_CELLS}; b = {k: [.1]*5 for k in ALL_CELLS}
    with pytest.raises(ValueError):grouped_error_check(e,a,b,block_ids=(1,2,3,4,4))
    broken = copy.deepcopy(a); broken[ALL_CELLS[0]][0] = float('nan')
    with pytest.raises(ValueError):grouped_error_check(e,broken,b,block_ids=BLOCK_IDS)
    broken = copy.deepcopy(e); broken['eligible_groups'] = ['mixed_burst']
    with pytest.raises(ValueError):grouped_error_check(broken,a,b,block_ids=BLOCK_IDS)


def test_both_calibrated_comparators_are_required_when_m1_retained():
    raw = {k: .4 for k in ALL_CELLS}; m1 = {k: .5 for k in ALL_CELLS}; m2 = {k: .6 for k in ALL_CELLS}
    plan = grouped_comparison_plan('M2','M1',m2,raw,raw,{k:.01 for k in ALL_CELLS},calibrated_m1=m1)
    errors = {k:[0.]*5 for k in ALL_CELLS}
    with pytest.raises(ValueError):grouped_structural_check(plan,errors,{'calibrated_M0':errors},block_ids=BLOCK_IDS)


def test_current_design_has_two_groups_at10ms_but_not12_or15():
    r = fit_scenario(cases(),1.5e-9)
    assert freeze_plan(r,.010)['required_structural_comparators']['calibrated_M0']['eligible_groups'] == ['mixed_burst','single_long']
    for sigma in (.012,.015):
        e = freeze_plan(r,sigma)['required_structural_comparators']['calibrated_M0']
        assert e['eligible_groups'] == ['mixed_burst']
        assert evaluate_draws(r,sigma,draws=100,seed=7,profile='iid')['passed'] == 0
    for c in cases():
        if c['sentinel'] and len(c['foreground_tokens']) == 1:assert c['foreground_tokens'] == [7680]
        assert all(n+128 <= 8192 for n in c['foreground_tokens'])


def test_noisy_labels_preserve_work_and_use_three_trials_without_held_leakage():
    cs = cases(); truth = [*COEF,0.,3e-9]
    training = collect_training(cs,truth); before = copy.deepcopy(training)
    labels = training_labels(training,.05,44,'shared_trial')
    assert training == before
    assert all(v.shape == (3,len(training[k][1])) and (v>0).all() for k,v in labels.items())
    assert any(np.std(v,axis=0).max()>0 for v in labels.values())
    changed = copy.deepcopy(cs)
    for c in changed:
        if c['split'] != 'calibration':c['foreground_tokens'] = [17]*len(c['foreground_tokens'])
    other = collect_training(changed,truth)
    assert other == training
    assert np.array_equal(fit_labels(training,labels,'M2'),fit_labels(other,training_labels(other,.05,44,'shared_trial'),'M2'))


def test_zero_label_noise_matches_existing_loco_and_fit():
    training = collect_training(cases(),[*COEF,0.,3e-9])
    labels = training_labels(training,0.,7)
    a,b = select(training),select_labels(training,labels)
    assert (a['selected'],a['retained_base']) == (b['selected'],b['retained_base'])
    assert fit_labels(training,labels,'M2') == pytest.approx([*COEF,0.,3e-9],abs=1e-12)


def test_held_noise_is_real_reproducible_and_has_declared_correlations():
    x = held_noise(7,10000,.01,'shared_block_ar1')
    assert np.array_equal(x,held_noise(7,10000,.01,'shared_block_ar1'))
    assert x.std() == pytest.approx(.01,rel=.03)
    assert np.corrcoef(x[:,0,0],x[:,1,0])[0,1] == pytest.approx(.8,abs=.03)
    assert np.corrcoef(x[:,0,0],x[:,0,1])[0,1] == pytest.approx(.6,abs=.03)


def test_repeat_cv_missing_or_noisy_strata_cannot_approve_model():
    assert calibration_noise_readiness({},expected_strata=['x'])['status'] == 'inconclusive'
    assert calibration_noise_readiness({'x':[1,1]},expected_strata=['x'])['status'] == 'inconclusive'
    r = calibration_noise_readiness({'x':[.9,1.,1.1]},expected_strata=['x'])
    assert r['status'] == 'stop_structural_claim' and not r['model_accepted']
    r = calibration_noise_readiness({'x':[.99,1.,1.01]},expected_strata=['x'])
    assert r['status'] == 'within_tested_noise_range' and not r['model_accepted']


def test_archived_noise_screen_matches_sources_and_enforces_structure_limits():
    d = json.loads((ROOT/'results/prefill-design-round6/noise-screen.json').read_text())
    for name,sha in d['source_sha256'].items():assert digest(ROOT/name) == sha
    assert not d['measured_power'] and not d['complete_experiment_passed']
    assert d['draws'] == 20000 and d['training_seeds'] == 40
    assert all(r['passed'] == 0 for r in d['held_noise'] if r['revision'] == 7 or r['sigma'] >= .012)
    assert all(r['passed'] == 0 for r in d['held_noise'] + d['null_controls'] if r['profile'] == 'shared_block_ar1')
    assert all(sum(r[k] for k in ('passed','failed','inconclusive','not_applicable')) == r['draws'] for r in d['held_noise'] + d['null_controls'])
    assert len(d['label_noise']) == 12
    assert all(len(r['records']) == 40 for r in d['label_noise'])


def test_unqualified_or_violated_block_dependence_cannot_pass():
    e = plan_for_gaps()['required_structural_comparators']['calibrated_M0']
    a = {k:[0.]*5 for k in ALL_CELLS}; b = {k:[.1]*5 for k in ALL_CELLS}
    for status in ('unresolved','violated'):
        r = grouped_error_check(e,a,b,block_ids=BLOCK_IDS,block_dependence_status=status)
        assert r['nominal_component_status'] == 'passed'
        assert r['component_status'] == 'inconclusive'
    r = grouped_error_check(e,a,b,block_ids=BLOCK_IDS)
    assert r['component_status'] == 'inconclusive'


def test_vectorized_monte_carlo_matches_public_block_gate():
    fitted = fit_scenario(cases(), 1.5e-9)
    plan = freeze_plan(fitted, .010)
    draws, seed = 30, 73
    for profile, dependence in (('iid','qualified'), ('shared_block_ar1','violated')):
        noise = held_noise(seed,draws,.010,profile)
        counts = dict(passed=0,failed=0,inconclusive=0)
        for sample in noise:
            measured = {k:fitted['truth'][k]+sample[i] for i,k in enumerate(ALL_CELLS)}
            candidate = {k:fitted['predictions']['M2'][k]-measured[k] for k in ALL_CELLS}
            baseline = {k:fitted['predictions']['M0'][k]-measured[k] for k in ALL_CELLS}
            result = grouped_structural_check(plan,candidate,{'calibrated_M0':baseline},
                                             block_ids=BLOCK_IDS,block_dependence_status=dependence)
            status = result['structural_status']
            counts['passed' if status == 'components_passed' else status] += 1
        simulated = evaluate_draws(fitted,.010,draws=draws,seed=seed,profile=profile)
        assert {k:simulated[k] for k in counts} == counts
