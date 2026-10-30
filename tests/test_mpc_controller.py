"""Controller checks use injected rollouts, not native serving measurements."""

from dataclasses import replace
import math
import sys
from types import ModuleType

import pytest

from mpc_lab import controller
from mpc_lab.contracts import (
    CostSpec,
    FutureOffer,
    ModelParameters,
    Outcome,
    PlanningConfig,
    RequestMark,
    RolloutError,
    RolloutResult,
    Scenario,
    Snapshot,
)


@pytest.fixture
def parameters():
    return ModelParameters()


def request(parameters, request_id="old", prompt_tokens=512, **changes):
    mark = RequestMark(
        request_id=request_id,
        cost=parameters.cost_spec.estimate(prompt_tokens, 128),
        prompt_tokens=prompt_tokens,
        max_tokens=128,
        received_at=8.0,
        admitted_at=8.1,
        scheduled_at=7.9,
    )
    return replace(mark, **changes)


@pytest.fixture
def snapshot(parameters):
    return Snapshot(
        "run", "test-boot", 10.0, 32.0, 0,
        (request(parameters),), parameters.identity,
    )


@pytest.fixture
def scenarios():
    return (
        Scenario((FutureOffer("new", 0.1, 512, 128, 128),)),
        Scenario((FutureOffer("new", 0.2, 1024, 128, 64),), service_scale=1.5),
    )


def prediction(lower=0.0, upper=None, events=1):
    return RolloutResult((), (), lower, lower if upper is None else upper, events)


def choose(snapshot, parameters, scenarios, config=None, predictor=None, **kwargs):
    return controller.choose_plan(
        snapshot, parameters, scenarios,
        config or PlanningConfig(max_wall_s=10),
        now=kwargs.pop("now", 10.0),
        clock_domain=kwargs.pop("clock_domain", "test-boot"),
        predictor=predictor or (lambda *args, **kw: prediction()),
        **kwargs,
    )


def test_failure_before_first_content_is_a_valid_zero_output_scenario(snapshot, parameters):
    scenario = Scenario((FutureOffer("failed", 0.1, 512, 128, 0, success=False),))
    decision = choose(snapshot, parameters, (scenario,))
    assert not decision.fallback
    assert decision.evaluated_plans == 8


def test_controller_matches_positive_work_model_parameter_domain(snapshot, parameters, scenarios):
    zero_slope = replace(parameters, prefill_s_per_token=0)
    state = replace(snapshot, identity=zero_slope.identity)
    assert not choose(state, zero_slope, scenarios).fallback
    zero_overhead = replace(parameters, first_overhead_s=0)
    state = replace(snapshot, identity=zero_overhead.identity)
    decision = choose(state, zero_overhead, scenarios)
    assert decision.fallback and decision.reason == "invalid_parameters"


@pytest.mark.parametrize("now", [1.0, 10.0, 100.0])
def test_clock_domain_precedes_chronology(snapshot, parameters, scenarios, now):
    cross_boot = replace(snapshot, clock_domain="another-boot")
    assert controller.validate_snapshot(
        cross_boot, now=now, clock_domain="test-boot", expected_identity=snapshot.identity,
    ) == "clock_domain_mismatch"
    decision = choose(cross_boot, parameters, scenarios, now=now)
    assert decision.fallback and decision.reason == "clock_domain_mismatch"


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"complete": False}, "incomplete_snapshot"),
        ({"complete": 1}, "incomplete_snapshot"),
        ({"run_id": ""}, "missing_run_id"),
        ({"budget_version": True}, "invalid_snapshot_version"),
        ({"sequence": -1}, "invalid_snapshot_version"),
        ({"budget": math.inf}, "invalid_budget"),
        ({"budget": True}, "invalid_budget"),
        ({"sampled_at": math.nan}, "invalid_snapshot_time"),
        ({"sampled_at": True}, "invalid_snapshot_time"),
        ({"sampled_at": 11.0}, "future_snapshot"),
        ({"sampled_at": 7.0}, "stale_snapshot"),
        ({"requests": []}, "invalid_ledger"),
    ],
)
def test_snapshot_metadata_validation(snapshot, changes, reason):
    assert controller.validate_snapshot(
        replace(snapshot, **changes), now=10, clock_domain="test-boot",
        expected_identity=snapshot.identity,
    ) == reason


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"request_id": ""}, "invalid_request_id"),
        ({"cost": False}, "invalid_request_cost"),
        ({"cost": math.nan}, "invalid_request_cost"),
        ({"prompt_tokens": True}, "invalid_request_tokens"),
        ({"max_tokens": 0}, "invalid_request_tokens"),
        ({"received_at": -1}, "invalid_request_time"),
        ({"admitted_at": False}, "invalid_request_time"),
        ({"admitted_at": 7.0}, "invalid_request_chronology"),
        ({"admitted_at": 11.0}, "invalid_request_chronology"),
        ({"first_at": 8.0}, "invalid_request_chronology"),
        ({"first_at": 11.0}, "invalid_request_chronology"),
        ({"first_at": True}, "invalid_request_chronology"),
        ({"scheduled_at": 8.01}, "invalid_request_chronology"),
        ({"scheduled_at": math.inf}, "invalid_request_chronology"),
    ],
)
def test_request_phase_and_numeric_validation(snapshot, changes, reason):
    invalid = replace(snapshot, requests=(replace(snapshot.requests[0], **changes),))
    assert controller.validate_snapshot(
        invalid, now=10, clock_domain="test-boot", expected_identity=snapshot.identity,
    ) == reason


def test_age_boundary_missing_and_duplicate_ledger(snapshot):
    validate = lambda value, **kw: controller.validate_snapshot(
        value, clock_domain="test-boot", expected_identity=snapshot.identity, **kw,
    )
    assert validate(snapshot, now=12) is None
    assert validate(snapshot, now=12.001) == "stale_snapshot"
    assert validate(snapshot, now=True) == "invalid_snapshot_time"
    assert validate(snapshot, now=10, max_age_s=False) == "invalid_max_age"
    assert validate(None, now=10) == "missing_snapshot"
    assert validate(
        replace(snapshot, requests=snapshot.requests * 2), now=10,
    ) == "invalid_request_id"
    huge = replace(snapshot.requests[0], cost=10**1000)
    assert validate(replace(snapshot, requests=(huge,)), now=10) == "invalid_request_cost"
    overflow = (
        replace(snapshot.requests[0], cost=1e308),
        replace(snapshot.requests[0], request_id="other", cost=1e308),
    )
    assert validate(replace(snapshot, requests=overflow), now=10) == "invalid_reserved_cost"


@pytest.mark.parametrize(
    "field",
    ["model_version", "calibration_version", "support_hash", "cost_hash", "implementation_hash"],
)
def test_every_identity_component_is_checked(snapshot, parameters, scenarios, field):
    changed = replace(snapshot, identity=replace(snapshot.identity, **{field: "changed"}))
    result = choose(changed, parameters, scenarios)
    assert result.fallback and result.reason == "identity_mismatch"


def test_cost_formula_matches_marks_and_nondefault_spec(snapshot, parameters, scenarios):
    inconsistent = replace(
        snapshot, requests=(replace(snapshot.requests[0], cost=2.0),),
    )
    result = choose(inconsistent, parameters, scenarios)
    assert result.fallback and result.reason == "cost_spec_mismatch"

    custom = replace(parameters, cost_spec=CostSpec(input_per_token=0.001))
    custom_snapshot = replace(
        snapshot, identity=custom.identity, requests=(request(custom),),
    )
    assert not choose(custom_snapshot, custom, scenarios).fallback
    assert choose(snapshot, custom, scenarios).reason == "identity_mismatch"

    overflowing = replace(parameters, cost_spec=CostSpec(input_per_token=1e308))
    overflow_snapshot = replace(snapshot, identity=overflowing.identity)
    assert choose(overflow_snapshot, overflowing, scenarios).reason == "cost_spec_mismatch"


@pytest.mark.parametrize("steps, count", [(1, 2), (3, 8), (5, 32)])
def test_complete_plan_counts_shared_draws_fixed_window_and_ties(
    snapshot, parameters, scenarios, steps, count,
):
    calls = []

    def predictor(observed, plan, scenario, model, config, *, deadline):
        assert observed is snapshot and model is parameters
        assert config.evaluation_window_s == 5 and config.tail_s == 10
        assert plan.slot_s == 1 and plan.continuation_budget == config.fallback_budget
        assert math.isfinite(deadline)
        calls.append((plan.actions, scenario))
        return prediction()

    result = choose(
        snapshot, parameters, scenarios,
        PlanningConfig(control_steps=steps, max_wall_s=10),
        predictor=predictor,
    )
    assert not result.fallback and result.reason == "selected"
    assert result.evaluated_plans == count and result.scenario_count == len(scenarios)
    assert result.plan.actions == (snapshot.budget,) * steps
    assert len(calls) == count * len(scenarios)
    assert len({actions for actions, _ in calls}) == count
    for index, (_, scenario) in enumerate(calls):
        assert scenario is scenarios[index % len(scenarios)]


def test_tie_prefers_hold_then_fewer_changes(snapshot, parameters, scenarios):
    def predictor(observed, plan, scenario, model, config, *, deadline):
        # All hold-first plans have a future switch; prefer the one with fewer.
        score = int(plan.actions in ((32.0, 16.0, 16.0), (32.0, 16.0, 32.0), (16.0,) * 3))
        return prediction(score)

    result = choose(snapshot, parameters, scenarios, predictor=predictor)
    assert result.plan.actions == (32.0, 16.0, 16.0)


@pytest.mark.parametrize(
    "actions, expected",
    [
        ((16.0, 32.0), (32.0, 16.0, 16.0)),
        ((32.0, 16.0), (32.0, 32.0, 16.0)),
    ],
)
def test_remaining_ties_use_configured_action_order(
    snapshot, parameters, scenarios, actions, expected,
):
    def predictor(observed, plan, scenario, model, config, *, deadline):
        return prediction(int(plan.actions in ((32.0, 16.0, 16.0), (32.0, 32.0, 16.0))))

    result = choose(
        snapshot, parameters, scenarios,
        PlanningConfig(actions=actions, max_wall_s=10), predictor=predictor,
    )
    assert result.plan.actions == expected


def test_old_cohorts_and_eventual_success_are_scored_by_predictor(
    snapshot, parameters, scenarios,
):
    snapshot = replace(snapshot, requests=(replace(snapshot.requests[0], first_at=8.2),))

    def predictor(observed, plan, scenario, model, config, *, deadline):
        old = Outcome(
            "old", 7.9, True, 8.2, 10.5,
            plan.actions[0] == 16, observed.requests[0].cost, 512, 128,
        )
        new = Outcome("new", 10.1, True, 10.2, 10.6, True, 1.0, 512, 128)
        outcomes = (old, new)
        lower = sum(outcome.good(config.slo_s) for outcome in outcomes)
        return RolloutResult(outcomes, ((0.0, observed.reserved),), lower, lower, 4)

    result = choose(snapshot, parameters, scenarios, predictor=predictor)
    assert result.budget == 16
    assert result.score_lower == result.score_upper == 2


def test_lower_not_unresolved_upper_selects_plan(snapshot, parameters, scenarios):
    def predictor(observed, plan, scenario, model, config, *, deadline):
        return prediction(1, 1) if plan.actions[0] == 16 else prediction(0, 2)

    result = choose(snapshot, parameters, scenarios, predictor=predictor)
    assert result.budget == 16 and result.score_lower == result.score_upper == 1

    uncertain = choose(
        snapshot, parameters, scenarios, predictor=lambda *a, **kw: prediction(0, 2),
    )
    assert uncertain.budget == snapshot.budget
    assert uncertain.score_lower == 0 and uncertain.score_upper == 2


def test_scores_are_scenario_means(snapshot, parameters, scenarios):
    def predictor(observed, plan, scenario, model, config, *, deadline):
        return prediction(1 if scenario is scenarios[0] else 0, 2)

    result = choose(snapshot, parameters, scenarios, predictor=predictor)
    assert result.score_lower == 0.5 and result.score_upper == 2


def test_reduction_does_not_clamp_or_evict(snapshot, parameters, scenarios):
    marks = tuple(request(parameters, str(i), 4000) for i in range(5))
    snapshot = replace(snapshot, requests=marks)
    reserved = snapshot.reserved
    assert reserved > 16

    def predictor(observed, plan, scenario, model, config, *, deadline):
        assert observed is snapshot and observed.requests is marks
        assert observed.reserved == reserved
        return prediction(int(plan.actions[0] == 16))

    result = choose(snapshot, parameters, scenarios, predictor=predictor)
    assert result.budget == 16 and not result.fallback
    assert snapshot.requests == marks and snapshot.reserved == reserved
    assert controller.validate_snapshot(
        replace(snapshot, budget=16), now=10, clock_domain="test-boot",
        expected_identity=parameters.identity,
    ) is None


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"max_events": True}, "invalid_limits"),
        ({"max_active_requests": 0}, "invalid_limits"),
        ({"max_plans": -1}, "invalid_limits"),
        ({"max_scenarios": 1.5}, "invalid_limits"),
        ({"control_steps": False}, "invalid_limits"),
        ({"max_wall_s": math.nan}, "invalid_limits"),
        ({"max_wall_s": 0}, "invalid_limits"),
        ({"max_snapshot_age_s": math.inf}, "invalid_limits"),
        ({"fallback_budget": True}, "invalid_limits"),
        ({"slot_s": 0.5}, "unsupported_config"),
        ({"evaluation_window_s": 3}, "unsupported_config"),
        ({"tail_s": 0.5}, "unsupported_config"),
        ({"control_steps": 2}, "unsupported_config"),
        ({"slo_s": 1}, "unsupported_config"),
        ({"actions": (16, True)}, "invalid_actions"),
        ({"actions": (16, 16)}, "invalid_actions"),
        ({"actions": ()}, "invalid_actions"),
        ({"actions": (16, 24)}, "unsupported_budget"),
        ({"fallback_budget": 24}, "unsupported_budget"),
        ({"max_plans": 7}, "plan_limit_exceeded"),
        ({"max_scenarios": 1}, "scenario_limit_exceeded"),
    ],
)
def test_invalid_or_unsupported_limits_fall_back(
    snapshot, parameters, scenarios, changes, reason,
):
    def never(*args, **kwargs):
        pytest.fail("invalid configuration reached the predictor")

    result = choose(
        snapshot, parameters, scenarios,
        replace(PlanningConfig(max_wall_s=10), **changes), predictor=never,
    )
    assert result.fallback and result.reason == reason
    assert result.plan is None and result.evaluated_plans == 0
    assert type(result.budget) is float and math.isfinite(result.budget)


@pytest.mark.parametrize(
    "changes",
    [
        {"prefill_s_per_token": True},
        {"decode_s_per_token": -1},
        {"first_overhead_s": math.inf},
        {"decode_interference": math.nan},
        {"max_prompt_tokens": False},
        {"supported_budgets": (16, True)},
    ],
)
def test_invalid_model_parameters(snapshot, parameters, scenarios, changes):
    result = choose(snapshot, replace(parameters, **changes), scenarios)
    assert result.fallback and result.reason == "invalid_parameters"


@pytest.mark.parametrize(
    "offer_changes",
    [
        {"request_id": "old"},
        {"offset_s": -1},
        {"offset_s": True},
        {"prompt_tokens": False},
        {"prompt_tokens": 32769},
        {"max_tokens": 4097},
        {"output_tokens": 129},
        {"output_tokens": 0},
        {"success": 1},
        {"ingress_delay_s": math.inf},
        {"egress_delay_s": -1},
    ],
)
def test_invalid_future_offers(snapshot, parameters, scenarios, offer_changes):
    invalid = replace(
        scenarios[0], offers=(replace(scenarios[0].offers[0], **offer_changes),),
    )
    result = choose(snapshot, parameters, (invalid,))
    assert result.fallback and result.reason == "invalid_scenario"
    assert result.evaluated_plans == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"service_scale": False},
        {"service_scale": math.nan},
        {"old_output_fraction": 1.1},
        {"old_output_fraction": 0},
        {"old_success": 1},
        {"ingress_delay_s": True},
        {"egress_delay_s": -1},
    ],
)
def test_invalid_scenario_state(snapshot, parameters, scenarios, changes):
    result = choose(snapshot, parameters, (replace(scenarios[0], **changes),))
    assert result.fallback and result.reason == "invalid_scenario"


def test_missing_unordered_duplicate_scenarios_and_event_limits(snapshot, parameters, scenarios):
    assert choose(snapshot, parameters, ()).reason == "missing_scenarios"
    assert choose(snapshot, parameters, list(scenarios)).reason == "missing_scenarios"
    offer = scenarios[0].offers[0]
    for offers in (
        (offer, offer),
        (replace(offer, request_id="a", offset_s=1), offer),
    ):
        result = choose(snapshot, parameters, (Scenario(offers),))
        assert result.reason == "invalid_scenario"
    crowded = Scenario((offer, replace(offer, request_id="other")))
    result = choose(
        snapshot, parameters, (crowded,),
        PlanningConfig(max_events=1, max_wall_s=10),
    )
    assert result.reason == "event_limit_exceeded"
    active = replace(snapshot, requests=(snapshot.requests[0], request(parameters, "other")))
    assert choose(
        active, parameters, scenarios, PlanningConfig(max_active_requests=1, max_wall_s=10),
    ).reason == "active_request_limit_exceeded"
    oversized = replace(snapshot, requests=(None,) * 513)
    assert choose(oversized, parameters, scenarios).reason == "active_request_limit_exceeded"


@pytest.mark.parametrize(
    "result, reason",
    [
        (None, "invalid_prediction"),
        (prediction(True), "invalid_prediction"),
        (prediction(math.nan), "invalid_prediction"),
        (prediction(1, 0), "invalid_prediction"),
        (prediction(0, math.inf), "invalid_prediction"),
        (prediction(0, 100), "invalid_prediction"),
        (prediction(events=True), "invalid_prediction"),
        (prediction(events=20001), "event_limit_exceeded"),
    ],
)
def test_invalid_predictor_results_fail_closed(snapshot, parameters, scenarios, result, reason):
    decision = choose(snapshot, parameters, scenarios, predictor=lambda *a, **kw: result)
    assert decision.fallback and decision.reason == reason
    assert decision.score_lower is None and decision.score_upper is None


def test_deadline_passed_inside_predictor_rejects_partial_search(
    monkeypatch, snapshot, parameters, scenarios,
):
    clock = [100.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    calls = []

    def predictor(observed, plan, scenario, model, config, *, deadline):
        calls.append((plan, scenario))
        if len(calls) == 3:
            clock[0] = deadline
        return prediction(1)

    result = choose(
        snapshot, parameters, scenarios, PlanningConfig(max_wall_s=0.1),
        predictor=predictor,
    )
    assert result.fallback and result.reason == "deadline_exceeded"
    assert result.evaluated_plans == 1 and len(calls) == 3
    assert result.plan is None and result.score_lower is None


def test_deadline_before_first_predictor(monkeypatch, snapshot, parameters, scenarios):
    times = iter([100.0, 101.0])
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(times, 101.0))

    def never(*args, **kwargs):
        pytest.fail("expired decision reached the predictor")

    result = choose(
        snapshot, parameters, scenarios, PlanningConfig(max_wall_s=0.1), predictor=never,
    )
    assert result.fallback and result.reason == "deadline_exceeded"
    assert result.evaluated_plans == 0


@pytest.mark.parametrize(
    "error, reason",
    [
        (RolloutError("max_events"), "max_events"),
        (TimeoutError(), "deadline_exceeded"),
        (RuntimeError("broken predictor"), "predictor_error"),
    ],
)
def test_predictor_errors_are_diagnostic_fallbacks(snapshot, parameters, scenarios, error, reason):
    def predictor(*args, **kwargs):
        raise error

    result = choose(snapshot, parameters, scenarios, predictor=predictor)
    assert result.fallback and result.reason == reason
    assert result.budget == 16 and result.plan is None


def test_default_predictor_is_lazy(monkeypatch, snapshot, parameters, scenarios):
    model = ModuleType("mpc_lab.model")
    model.rollout = lambda *args, **kwargs: prediction()
    monkeypatch.setitem(sys.modules, "mpc_lab.model", model)
    result = controller.choose_plan(
        snapshot, parameters, scenarios, PlanningConfig(max_wall_s=10),
        now=10, clock_domain="test-boot",
    )
    assert not result.fallback and result.evaluated_plans == 8
    monkeypatch.setitem(sys.modules, "mpc_lab.model", None)
    result = controller.choose_plan(
        snapshot, parameters, scenarios, PlanningConfig(max_wall_s=10),
        now=10, clock_domain="test-boot",
    )
    assert result.reason == "predictor_unavailable"


def test_default_clock_uses_local_domain_not_snapshot(monkeypatch, snapshot, parameters, scenarios):
    clock = ModuleType("prefill_lab.live_http")
    clock.clock_domain = lambda: "local-boot"
    monkeypatch.setitem(sys.modules, "prefill_lab.live_http", clock)
    monkeypatch.setattr(controller.time, "monotonic", lambda: 10.0)
    result = controller.choose_plan(
        snapshot, parameters, scenarios, PlanningConfig(), predictor=lambda *a, **kw: prediction(),
    )
    assert result.fallback and result.reason == "clock_domain_mismatch"


def test_feedback_uses_only_observed_awaiting_backlog(snapshot, parameters):
    def at_backlog(prompt, budget=32, first_at=None):
        return replace(
            snapshot, budget=budget,
            requests=(request(parameters, prompt_tokens=prompt, first_at=first_at),),
        )

    assert controller.feedback_budget(at_backlog(2000, 16)) == 32
    assert controller.feedback_budget(at_backlog(6000)) == 16
    assert controller.feedback_budget(at_backlog(4000, 16)) == 16
    assert controller.feedback_budget(at_backlog(4000, 32)) == 32
    assert controller.feedback_budget(at_backlog(8000, 16, first_at=9)) == 32
    assert controller.feedback_budget(at_backlog(4000, 96)) == 32
    with pytest.raises(ValueError, match="invalid_feedback_config"):
        controller.feedback_budget(snapshot, backlog_low=True)
    with pytest.raises(ValueError, match="invalid_feedback_config"):
        controller.feedback_budget(snapshot, low=32, high=16)
    with pytest.raises(ValueError, match="incomplete_snapshot"):
        controller.feedback_budget(replace(snapshot, complete=False))
