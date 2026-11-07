"""Role7 read-only controls against one frozen team16 planner package.

run_controls.py selects the package (original or revision01) by sys.path order. The
defect controls encode the expected-correct behaviour (refuse, or keep the exact
rational-oracle order), so a FAILED control is the evidence; characterization controls
assert observed scope facts. Synthetic score tables only: not native evidence, not a
qualification, speed or win claim.
"""
from dataclasses import replace
from itertools import product
import json
import math

import pytest

from test_policy_planner16 import (
    api, exact_oracle, frozen_clock, history_case, install_tape, planner, planning_case,
)

PATHS = [(16, 16), (16, 32), (32, 16), (32, 32)]
ONE_MOVE_FROM_16 = {(16, 16), (16, 32), (32, 32)}


def observe(name, **fields):
    print("ROLE7_OBSERVATION " + json.dumps({"control": name, **fields}, sort_keys=True),
          flush=True)


def hexed(value):
    return value.hex() if type(value) is float else value


def tiny_wall(common_wall):
    scores = [{p: 0 for p in PATHS} for _ in range(2)]
    walls = [{p: common_wall for p in PATHS},
             {p: 0.875 if p == (32, 32) else 1.0 for p in PATHS}]
    return scores, walls


def tiny_good():
    scores = [{p: 1 for p in PATHS}, {p: 1 if p == (32, 32) else 0 for p in PATHS}]
    walls = [{p: 1.0 for p in PATHS} for _ in range(2)]
    return scores, walls


CASES = {
    "A0_exact_subnormal_draw_zero_common_term_positive_control": (
        32 * math.ulp(0.0), tiny_wall(0.0)),
    "A1_exact_subnormal_draw_nonzero_common_wall": (32 * math.ulp(0.0), tiny_wall(1.0)),
    "A2_normal_2pow-60_draw_nonzero_common_wall": (2.0 ** -60, tiny_wall(1.0)),
    "A3_normal_2pow-60_draw_primary_natural_good_key": (2.0 ** -60, tiny_good()),
}


@pytest.mark.parametrize("policy", ("mpc", "strong_one_move"))
@pytest.mark.parametrize("name", sorted(CASES))
def test_A_tiny_positive_draw_refuses_or_keeps_rational_oracle_choice(
        planning_case, monkeypatch, name, policy):
    tiny, (scores, walls) = CASES[name]
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    calls = install_tape(monkeypatch, (0.05, 0.25), scores, walls)
    weights = (1.0, tiny)
    panel = tuple(replace(
        planning_case.scenario, weight=weight,
        model={**planning_case.scenario.model, "oracle_world": index})
        for index, weight in enumerate(weights))
    order, ranks = exact_oracle(PATHS, 16, weights, scores, walls)
    oracle = order[0] if policy == "mpc" else min(ONE_MOVE_FROM_16, key=ranks.__getitem__)
    result = planner(planning_case, policy).plan(panel, clock_now=planning_case.clock_now)
    selected = tuple(result["selected_plan"]["path"]) if "selected_plan" in result else None
    observe(name, policy=policy, status=result["status"], selected=selected, oracle=oracle,
            refusals=result.get("policy_guard_refusals", []),
            scaled_weights=[hexed(w) for w in result.get("scenario_weights_scaled", [])],
            float_good_wall={str(tuple(row["path"])): [
                hexed(row["expected_natural_slo_good"]), hexed(row["expected_cohort_wall_s"])]
                for row in result.get("rankings", [])})
    assert len(calls) == len(PATHS) * len(weights)
    assert oracle == (32, 32)
    if result["status"] == "refused":
        assert result["first_action"] is None and selected is None
    else:
        assert selected == oracle, (
            "positive-weight draw absorbed into an artificial tie that the transition/"
            "path-sum tie-break resolves away from the exact rational-oracle choice")


def test_B_train_selected_fixed_covers_every_constant_policy_of_shared_action_set(
        planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    budgets, offsets = (16, 24, 32), (0.05, 0.25)
    paths = list(product(budgets, repeat=len(offsets)))
    install_tape(monkeypatch, offsets, [{p: 3 if p == (24, 24) else 1 for p in paths}],
                 [{p: 0.25 for p in paths}])
    scenario = replace(planning_case.scenario,
                       model={**planning_case.scenario.model, "oracle_world": 0})
    train = planner(planning_case, budgets=budgets, step_offsets_s=offsets).plan(
        [scenario], clock_now=planning_case.clock_now)
    assert train["status"] == "model_plan_only"
    strongest = tuple(train["best_fixed"]["path"])
    chosen = executed = error = None
    try:
        choice = api.freeze_fixed_choice([train], split="train")
    except ValueError as exc:
        error = str(exc)
    else:
        chosen = (choice.budget,) * len(offsets)
        heldout = planner(planning_case, "train_selected_fixed", budgets=budgets,
                          step_offsets_s=offsets, training_choice=choice).plan(
            [scenario], clock_now=planning_case.clock_now)
        executed = (heldout["status"], tuple(heldout["selected_plan"]["path"])
                    if "selected_plan" in heldout else None)
    observe("B_three_budget_train_selected_fixed", strongest_constant=strongest,
            chosen=chosen, executed=executed, error=error,
            mpc_best=tuple(train["best_multistep"]["path"]))
    assert strongest == (24, 24)
    assert error is not None or chosen == strongest, (
        "accepted 3-budget contract, but train-selected fixed silently omits constant-24 "
        "and freezes a weaker constant than the strongest one in the shared action set")


def test_C_characterize_default_acquisition_clock_excludes_queue_delay(
        planning_case, monkeypatch):
    """Scope fact, not a defect claim: the adapter's default entry is acquisition time."""
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    acquired_at = case.clock_now()
    dequeued_at = api.PlanningEntry(acquired_at.clock_domain, acquired_at.at + 10.0)
    default = planner(case).plan_acquisition(
        api.NativePanelAcquisition((case.scenario,), acquired_at, 1), version_now=lambda: 1)
    explicit = planner(case).plan_acquisition(
        api.NativePanelAcquisition((case.scenario,), acquired_at, 1), version_now=lambda: 1,
        clock_now=lambda: dequeued_at)
    observe("C_acquisition_queue_delay_scope", queue_delay_s=10.0,
            default_status=default["status"],
            default_first_action_source_age_s=default.get("first_action_source_age_s"),
            default_solver_elapsed_s=default.get("solver_elapsed_s"),
            explicit_status=explicit["status"],
            explicit_timing_refused=explicit.get("timing_refused"),
            explicit_refusals=explicit.get("policy_guard_refusals", []))
    assert default["status"] == "model_plan_only"
    assert default["first_action_source_age_s"] <= 0.5
    assert explicit["status"] == "refused" and explicit["timing_refused"] is True
