"""Exact ordering and full-action-set fixtures; no native performance claims."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import pytest

from test_policy_planner16 import (
    api, exact_oracle, frozen_clock, history_case, install_tape, no_ranked_output,
    original, planner, planning_case,
)
from test_role7_planner_controls import CASES, ONE_MOVE_FROM_16, PATHS, tiny_good


ABSORPTION_CASES = tuple(name for name in sorted(CASES) if not name.startswith("A0_"))


def table_panel(case, weights):
    return tuple(replace(
        case.scenario, weight=weight,
        model={**case.scenario.model, "oracle_world": index})
        for index, weight in enumerate(weights))


def accepted_table(case, monkeypatch, weights, scores, walls, *, policy="mpc",
                   budgets=(16, 32)):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    calls = install_tape(monkeypatch, (0.05, 0.25), scores, walls)
    result = planner(case, policy, budgets=budgets).plan(
        table_panel(case, weights), clock_now=case.clock_now)
    assert result["status"] == "model_plan_only"
    assert result["evaluations_attempted"] == len(calls) == len(budgets) ** 2 * len(weights)
    return result


@pytest.mark.parametrize("name", ABSORPTION_CASES)
@pytest.mark.parametrize("policy", ("mpc", "strong_one_move"))
def test_nonzero_common_terms_accept_exact_order_despite_identical_display_floats(
        planning_case, monkeypatch, name, policy):
    tiny, (scores, walls) = CASES[name]
    weights = (1.0, tiny)
    result = accepted_table(
        planning_case, monkeypatch, weights, scores, walls, policy=policy)
    order, ranks = exact_oracle(PATHS, 16, weights, scores, walls)
    assert [tuple(row["path"]) for row in result["rankings"]] == order
    expected = order[0] if policy == "mpc" else min(
        ONE_MOVE_FROM_16, key=ranks.__getitem__)
    assert tuple(result["selected_plan"]["path"]) == expected == (32, 32)
    assert len({(row["expected_natural_slo_good"], row["expected_cohort_wall_s"])
                for row in result["rankings"]}) == 1
    for row in result["rankings"]:
        rank = ranks[tuple(row["path"])]
        good = Fraction(*map(int, row["exact_expected_natural_slo_good"]))
        wall = Fraction(*map(int, row["exact_expected_cohort_wall_s"]))
        assert (-good, wall) == rank[:2]
        assert len(row["per_scenario"]) == 2
    assert result["scenario_weights_original"] == list(weights)
    assert not result["first_action"]["execute_permitted"]


@pytest.mark.parametrize("name", ABSORPTION_CASES)
def test_training_uses_exact_per_panel_means_and_keeps_frozen_choice(
        planning_case, monkeypatch, name):
    case = planning_case
    tiny, (scores, walls) = CASES[name]
    weights = (1.0, tiny)
    training = accepted_table(case, monkeypatch, weights, scores, walls)
    choice = api.freeze_fixed_choice([training], split="train")
    assert choice.budget == 32
    fixed = planner(case, "train_selected_fixed", training_choice=choice)
    heldout_scores = [{path: 5 if path == (16, 16) else 0 for path in PATHS}]
    heldout_walls = [{path: 1.0 for path in PATHS}]
    heldout = accepted_table(case, monkeypatch, (1.0,), heldout_scores, heldout_walls)
    assert heldout["best_fixed"]["path"] == [16, 16]
    result = fixed.plan(table_panel(case, (1.0,)), clock_now=case.clock_now)
    assert result["status"] == "model_plan_only"
    assert result["selected_plan"]["path"] == [32, 32]
    assert result["rankings"] == heldout["rankings"]


@pytest.mark.parametrize("budgets", (
    (16, 24, 32), (32, 24, 16), (32.0, 24.0, 16.0),
))
def test_train_selected_constant_covers_canonical_full_shared_set(
        planning_case, monkeypatch, budgets):
    case = planning_case
    paths = tuple(product(budgets, repeat=2))
    scores = [{path: 4 if path == (24, 24) else 1 for path in paths}]
    walls = [{path: 0.25 for path in paths}]
    training = accepted_table(case, monkeypatch, (1.0,), scores, walls, budgets=budgets)
    choice = api.freeze_fixed_choice([training], split="train")
    assert choice.candidate_budgets == (16, 24, 32)
    assert [row[0] for row in choice.candidate_objectives] == [16, 24, 32]
    assert choice.budget == 24 and type(choice.budget) is int
    fixed = planner(case, "train_selected_fixed", budgets=budgets, training_choice=choice)
    heldout_scores = [{path: 5 if path == (32, 32) else 0 for path in paths}]
    heldout = accepted_table(
        case, monkeypatch, (1.0,), heldout_scores, walls, budgets=budgets)
    assert heldout["best_fixed"]["path"] == [32, 32]
    result = fixed.plan(table_panel(case, (1.0,)), clock_now=case.clock_now)
    assert result["status"] == "model_plan_only"
    assert result["selected_plan"]["path"] == [24, 24]
    assert result["rankings"] == heldout["rankings"]


def test_partial_candidate_artifact_cannot_claim_full_action_set(planning_case, monkeypatch):
    budgets = (16, 24, 32)
    paths = tuple(product(budgets, repeat=2))
    training = accepted_table(
        planning_case, monkeypatch, (1.0,),
        [{path: 5 if path == (16, 16) else 0 for path in paths}],
        [{path: 1.0 for path in paths}], budgets=budgets)
    choice = api.freeze_fixed_choice([training], split="train")
    weakened = replace(
        choice, candidate_budgets=(16, 32),
        candidate_objectives=tuple(row for row in choice.candidate_objectives if row[0] != 24))
    with pytest.raises(ValueError, match="training_configuration"):
        planner(planning_case, "train_selected_fixed", budgets=budgets, training_choice=weakened)


def test_training_recomputes_exact_keys_instead_of_trusting_cached_values(
        planning_case, monkeypatch):
    tiny, (scores, walls) = CASES[ABSORPTION_CASES[1]]
    result = accepted_table(planning_case, monkeypatch, (1.0, tiny), scores, walls)
    changed = deepcopy(result)
    changed["rankings"][0]["exact_expected_natural_slo_good"] = ["5", "1"]
    with pytest.raises(ValueError, match="exact_objective_mismatch"):
        api.freeze_fixed_choice([changed], split="train")


@pytest.mark.parametrize("stage", ("exact_mean", "exact_sort"))
def test_exact_ordering_work_is_inside_full_solver_deadline(planning_case, monkeypatch, stage):
    case = planning_case
    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(api, "time", clock)
    monkeypatch.setattr(original, "time", clock)
    name = "_exact_mean" if stage == "exact_mean" else "_rank"
    operation = getattr(api, name)

    def delayed(*args, **kwargs):
        result = operation(*args, **kwargs)
        now[0] = 0.3
        return result

    monkeypatch.setattr(api, name, delayed)
    result = planner(case).plan((case.scenario,), clock_now=case.clock_now)
    no_ranked_output(result)
    assert result["timing_refused"] and result["solver_elapsed_s"] == 0.3
    assert result["evaluations_attempted"] == result["evaluations_expected"] == 4


def test_version_change_during_exact_ordering_refuses_whole_result(planning_case, monkeypatch):
    case, version = planning_case, [1]
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    operation = api._exact_mean

    def changed(*args, **kwargs):
        result = operation(*args, **kwargs)
        version[0] = 2
        return result

    monkeypatch.setattr(api, "_exact_mean", changed)
    result = planner(case).plan_acquisition(
        api.NativePanelAcquisition((case.scenario,), case.clock_now(), 1),
        version_now=lambda: version[0])
    no_ranked_output(result)
    assert any("observation_changed" in reason for reason in result["policy_guard_refusals"])


def test_exact_primary_improvement_precedes_a_large_secondary_wall_penalty(
        planning_case, monkeypatch):
    scores, _ = tiny_good()
    walls = [{path: 2.0 if path == (32, 32) else 1.0 for path in PATHS}
             for _ in range(2)]
    weights = (1.0, 2.0 ** -60)
    result = accepted_table(planning_case, monkeypatch, weights, scores, walls)
    order, _ = exact_oracle(PATHS, 16, weights, scores, walls)
    assert tuple(result["selected_plan"]["path"]) == order[0] == (32, 32)
    assert result["selected_plan"]["expected_cohort_wall_s"] == 2.0
    assert len({row["expected_natural_slo_good"] for row in result["rankings"]}) == 1
