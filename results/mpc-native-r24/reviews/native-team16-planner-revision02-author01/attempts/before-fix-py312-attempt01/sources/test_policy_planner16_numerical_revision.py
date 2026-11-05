"""Exact small numerical refusal controls; no native qualification evidence."""

from dataclasses import replace
import math
import sys

import pytest

from test_policy_planner16 import (
    api, exact_oracle, frozen_clock, history_case, install_tape, no_ranked_output,
    planner, planning_case, policies,
)


def numerical_panel(case, monkeypatch, tiny_weight, target_wall, other_wall,
                    policy="mpc", **settings):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    paths = [(16, 16), (16, 32), (32, 16), (32, 32)]
    weights = (1.0, tiny_weight)
    scores = [{path: 0 for path in paths} for _ in range(2)]
    walls = [
        {path: 0.0 for path in paths},
        {path: target_wall if path == (32, 32) else other_wall for path in paths},
    ]
    calls = install_tape(monkeypatch, (0.05, 0.25), scores, walls)
    panel = tuple(replace(
        case.scenario, weight=weight,
        model={**case.scenario.model, "oracle_world": index})
        for index, weight in enumerate(weights))
    expected, _ = exact_oracle(paths, 16, weights, scores, walls)
    assert expected[0] == (32, 32)
    result = planner(case, policy, **settings).plan(panel, clock_now=case.clock_now)
    assert result["evaluations_attempted"] == result["evaluations_expected"] == len(calls) == 8
    return result, expected, weights


@pytest.mark.parametrize("policy,settings", policies())
def test_parent_counterexample_refuses_the_whole_common_panel_for_every_arm(
        planning_case, monkeypatch, policy, settings):
    result, _, _ = numerical_panel(
        planning_case, monkeypatch, 32 * math.ulp(0.0), 0.9375, 1.0,
        policy, **settings)
    no_ranked_output(result)
    assert "policy16_unrepresentable_weighted_objective" in result["policy_guard_refusals"]


@pytest.mark.parametrize("tiny_weight,target_wall,other_wall", (
    pytest.param(32 * math.ulp(0.0), 0.75, 0.8125, id="subnormal-round-down-tie"),
    pytest.param(
        4 * sys.float_info.min, math.nextafter(1.0, 0.0), 1.0,
        id="below-boundary-rounds-up-to-min-normal"),
    pytest.param(
        4 * math.nextafter(sys.float_info.min, math.inf),
        math.nextafter(1.0, 0.0), 1.0,
        id="above-boundary-rounds-down-to-min-normal"),
))
def test_nonzero_inexact_terms_at_both_sides_of_underflow_boundary_refuse(
        planning_case, monkeypatch, tiny_weight, target_wall, other_wall):
    result, _, _ = numerical_panel(
        planning_case, monkeypatch, tiny_weight, target_wall, other_wall)
    no_ranked_output(result)
    assert "policy16_unrepresentable_weighted_objective" in result["policy_guard_refusals"]


@pytest.mark.parametrize("tiny_weight,target_wall,other_wall", (
    pytest.param(32 * math.ulp(0.0), 0.875, 1.0, id="exact-positive-subnormal"),
    pytest.param(4 * sys.float_info.min, 1.0, 2.0, id="exact-minimum-normal"),
))
def test_representable_tiny_draws_are_retained_and_keep_exact_oracle_order(
        planning_case, monkeypatch, tiny_weight, target_wall, other_wall):
    result, expected, weights = numerical_panel(
        planning_case, monkeypatch, tiny_weight, target_wall, other_wall)
    assert result["status"] == "model_plan_only"
    assert [tuple(row["path"]) for row in result["rankings"]] == expected
    assert tuple(result["selected_plan"]["path"]) == (32, 32)
    assert result["scenario_weights_original"] == list(weights)
    assert result["scenario_count"] == 2
    assert all(len(row["per_scenario"]) == 2 for row in result["rankings"])
    assert not result["first_action"]["execute_permitted"]
