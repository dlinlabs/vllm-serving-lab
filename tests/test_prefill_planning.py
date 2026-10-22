import copy
import json
import numpy as np
import pytest

from prefill_design import COEF, requests_for, simulate
from prefill_planning import (FAMILIES, collect_training, fit, legacy_cases,
                              rollout, synthetic_check)
from scripts.build_prefill_design import cases, ROOT, digest


@pytest.mark.parametrize('k', [0., 1.5e-9, 3.15e-9])
def test_causal_planner_matches_original_fifo_toy(k):
    for c in cases():
        original = simulate(requests_for(c), 2048, backgrounds=c['background_decoders'], quadratic=k)
        actual, _ = rollout(c, [*COEF, 0., 2*k])
        assert actual['wall_s'] == pytest.approx(original['wall_s'], abs=1e-10)
        assert actual['release_s'] == pytest.approx(original['release_s'], abs=1e-10)
        for rid, row in actual['foreground'].items():
            for field, value in row.items():
                assert value == pytest.approx(original['foreground'][rid][field], abs=1e-10)


def test_held_inputs_cannot_change_fit_or_scaling():
    cs = cases(); changed = copy.deepcopy(cs)
    for c in changed:
        if c['split'] != 'calibration':
            c['foreground_tokens'] = [17]*len(c['foreground_tokens'])
    truth = [*COEF, 1e-7, 3e-9]
    a, b = collect_training(cs, truth), collect_training(changed, truth)
    assert a == b
    for family in FAMILIES:
        assert np.array_equal(fit(a, family), fit(b, family))


def test_exact_nonlinear_truth_cannot_resolve_legacy_sentinels_after_refitting():
    r = synthetic_check(legacy_cases(cases()), quadratic=1.5e-9)
    assert r['selection']['selected'] == 'M2'
    assert r['comparison_plan']['required_structural_comparators']['calibrated_M0']['eligible_cells'] == []
    assert r['gate']['structural_status'] == 'inconclusive'
    gaps = [abs(r['predictions_s']['M2'][c] - r['predictions_s']['M0'][c]) for c in r['truth_s']]
    assert min(gaps) > .019 and max(gaps) < .023


@pytest.mark.parametrize('jitter,expected', [(0., 'components_passed'), (.010, 'components_passed'), (.015, 'inconclusive')])
def test_replacement_resolves_only_declared_noise_range(jitter, expected):
    r = synthetic_check(cases(), quadratic=1.5e-9, jitter_s=jitter)
    assert r['gate']['structural_status'] == expected
    assert not r['statistical_power_estimated'] and not r['complete_experiment_passed']
    if expected == 'components_passed':
        eligible = r['comparison_plan']['required_structural_comparators']['calibrated_M0']['eligible_cells']
        assert len(eligible) == 4
        assert all('-at50-' in c or '-at400-' in c for c in eligible)


@pytest.mark.parametrize('scale', [.9, 1., 1.1])
def test_linear_recalibration_is_not_structural_success(scale):
    r = synthetic_check(cases(), quadratic=0., scale=scale)
    assert r['selection']['selected'] == 'M0'
    assert r['gate']['structural_status'] == 'not_applicable'


def test_context_and_z_ablation_recover_separate_synthetic_terms():
    truth = np.asarray([*COEF, 1e-7, 3e-9])
    training = collect_training(cases(), truth)
    assert fit(training, 'M2') == pytest.approx(truth, rel=1e-7, abs=1e-13)
    assert fit(training, 'M0Z')[3] == 0
    assert fit(training, 'M1')[4] == 0


def test_composition_swap_preserves_load_and_keeps_independent_policy_control():
    cs = cases()
    for c in cs:
        if c['sentinel'] and len(c['foreground_tokens']) == 4:
            assert sum(c['foreground_tokens']) == 16384
            assert c['foreground_tokens'] == ([4096]*4 if c['sentinel']['arrival_ms'] == 800 else [1024,1024,7168,7168])


def test_committed_planning_has_current_sources_and_all_comparators():
    data = json.loads((ROOT/'results/prefill-design-round5/calibrated-planning.json').read_text())
    for path, sha in data['source_sha256'].items():
        assert digest(ROOT/path) == sha
    assert len(data['scenarios']) == 12 and len(data['controls']) == 4
    for r in data['scenarios'] + data['controls']:
        assert not r['complete_experiment_passed']
        if r['selection']['selected'] == 'M2' and r['selection']['retained_base'] == 'M1':
            assert 'calibrated_M1' in r['gate']['comparators']
