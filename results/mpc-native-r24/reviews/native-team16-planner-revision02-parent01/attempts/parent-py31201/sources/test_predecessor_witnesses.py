"""Deliberately failing safety/objective regressions on the unchanged predecessor."""

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

import native_response_planner as original
import policy_planner15 as predecessor
from test_r24_historical_scenario import history_case, planning_case


def frozen_clock(monkeypatch, planner_class):
    clock = SimpleNamespace(monotonic=lambda: 0.0)
    for module in (original, predecessor, sys.modules[planner_class.__module__]):
        monkeypatch.setattr(module, "time", clock)


def make_planner(case, planner_class):
    return planner_class(
        cost_spec=case.planner().cost_spec, budgets=(16, 32),
        step_offsets_s=(0.05, 0.25), solver_budget_s=0.04,
        actuation_budget_s=0.005, policy="mpc")


def bounded_materialization(case, monkeypatch, planner_class):
    frozen_clock(monkeypatch, planner_class)
    yielded = []

    def too_many():
        for index in range(17):
            yielded.append(index)
            yield case.scenario
        raise AssertionError("read eighteenth item after the panel was already oversized")

    with pytest.raises(ValueError, match="scenarios|panel"):
        make_planner(case, planner_class).plan(too_many(), clock_now=case.clock_now)
    assert len(yielded) == 17


def external_world_mutation(case, monkeypatch, planner_class):
    frozen_clock(monkeypatch, planner_class)
    expected = make_planner(case, planner_class).plan(
        [case.scenario], clock_now=case.clock_now)
    assert expected["status"] == "model_plan_only"
    simulate = original.simulate
    calls = []

    def updating_external_world(*args, **kwargs):
        result = simulate(*args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            case.scenario.model["effective_readiness_coefficients"]["decode_tokens"] = 0.8
        return result

    monkeypatch.setattr(original, "simulate", updating_external_world)
    actual = make_planner(case, planner_class).plan(
        [case.scenario], clock_now=case.clock_now)
    assert actual["status"] == "model_plan_only"
    assert len(calls) == 4
    assert actual["rankings"] == expected["rankings"], "paths used different model snapshots"


def internal_world_mutation(case, monkeypatch, planner_class):
    frozen_clock(monkeypatch, planner_class)
    simulate = original.simulate
    calls = []

    def mutating_evaluator(*args, **kwargs):
        result = simulate(*args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            args[2]["effective_readiness_coefficients"]["decode_tokens"] = 0.8
        return result

    monkeypatch.setattr(original, "simulate", mutating_evaluator)
    result = make_planner(case, planner_class).plan(
        [case.scenario], clock_now=case.clock_now)
    assert result["status"] == "refused", "mixed-world panel was ranked"
    assert result["first_action"] is None
    assert "rankings" not in result and "selected_plan" not in result


def positive_weight_scale_invariance(case, monkeypatch, planner_class):
    frozen_clock(monkeypatch, planner_class)
    expected = make_planner(case, planner_class).plan(
        [case.scenario], clock_now=case.clock_now)
    actual = make_planner(case, planner_class).plan(
        [replace(case.scenario, weight=float.fromhex("0x0.0000000000001p-1022"))],
        clock_now=case.clock_now)
    assert actual["status"] == expected["status"] == "model_plan_only"
    assert actual["rankings"] == expected["rankings"], "positive weight changed a one-draw objective"


def test_predecessor_bounded_materialization(planning_case, monkeypatch):
    bounded_materialization(planning_case, monkeypatch, predecessor.PolicyPlanner15)


def test_predecessor_external_world_isolation(planning_case, monkeypatch):
    external_world_mutation(planning_case, monkeypatch, predecessor.PolicyPlanner15)


def test_predecessor_internal_world_integrity(planning_case, monkeypatch):
    internal_world_mutation(planning_case, monkeypatch, predecessor.PolicyPlanner15)


def test_predecessor_weight_scale_invariance(planning_case, monkeypatch):
    positive_weight_scale_invariance(planning_case, monkeypatch, predecessor.PolicyPlanner15)
