from dataclasses import replace

from mpc_lab.contracts import (
    FutureOffer, ModelParameters, Plan, PlanningConfig, Scenario, Snapshot,
)
from mpc_lab.controller import choose_plan
from mpc_lab.model import rollout


def test_changed_dynamics_change_prediction_and_cannot_reuse_old_support_identity():
    fast = ModelParameters(prefill_s_per_token=0.00001)
    slow = replace(fast, prefill_s_per_token=0.01)
    snapshot = Snapshot("cpu", "test-clock", 0, 16, 0, (), fast.identity)
    scenario = Scenario((FutureOffer("new", 0, 1000, 1, 1),))
    plan = Plan((16, 16, 16))
    config = PlanningConfig()
    assert rollout(snapshot, plan, scenario, fast, config).score_lower == 1
    assert rollout(snapshot, plan, scenario, slow, config).score_lower == 0
    assert fast.identity != slow.identity
    decision = choose_plan(
        snapshot, slow, (scenario,), config, now=0, clock_domain="test-clock",
    )
    assert decision.fallback and decision.reason == "identity_mismatch"


def test_real_model_and_planner_handle_failure_before_first_without_success_credit():
    params = ModelParameters()
    snapshot = Snapshot("cpu", "test-clock", 0, 16, 0, (), params.identity)
    scenario = Scenario((FutureOffer("failure", 0.1, 128, 128, 0, success=False),))
    decision = choose_plan(
        snapshot, params, (scenario,), PlanningConfig(), now=0, clock_domain="test-clock",
    )
    assert not decision.fallback
    assert decision.evaluated_plans == 8
    assert decision.score_lower == decision.score_upper == 0
