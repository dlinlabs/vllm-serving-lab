"""Bounded shadow recommendations; a fallback budget is not a delivered action."""

from __future__ import annotations

from dataclasses import fields
from itertools import product
import math
import time

from .contracts import (
    CostSpec,
    Decision,
    FutureOffer,
    Identity,
    ModelParameters,
    Plan,
    PlanningConfig,
    RequestMark,
    RolloutError,
    RolloutResult,
    Scenario,
    Snapshot,
)


def _number(value: object, *, minimum: float = 0, positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > minimum if positive else value >= minimum)
    except OverflowError:
        return False


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _identity_valid(identity: object) -> bool:
    return isinstance(identity, Identity) and all(
        _text(getattr(identity, field.name)) for field in fields(Identity)
    )


def validate_snapshot(
    snapshot: Snapshot,
    *,
    now: float,
    clock_domain: str,
    expected_identity: Identity,
    max_age_s: float = 2.0,
) -> str | None:
    """Validate the observed ledger, without treating reservation as a cap clamp.

    Cost-formula consistency additionally needs ModelParameters and is checked
    by choose_plan; an Identity alone cannot reconstruct an arbitrary CostSpec.
    """
    if not isinstance(snapshot, Snapshot):
        return "missing_snapshot"
    if not _text(clock_domain) or not _text(snapshot.clock_domain):
        return "missing_clock_domain"
    if snapshot.clock_domain != clock_domain:
        return "clock_domain_mismatch"
    if not _identity_valid(expected_identity) or not _identity_valid(snapshot.identity):
        return "invalid_identity"
    if snapshot.identity != expected_identity:
        return "identity_mismatch"
    if snapshot.complete is not True:
        return "incomplete_snapshot"
    if not _text(snapshot.run_id):
        return "missing_run_id"
    if not _integer(snapshot.sequence) or not _integer(snapshot.budget_version):
        return "invalid_snapshot_version"
    if not _number(snapshot.budget, positive=True):
        return "invalid_budget"
    if not _number(now) or not _number(snapshot.sampled_at):
        return "invalid_snapshot_time"
    if not _number(max_age_s):
        return "invalid_max_age"
    if snapshot.sampled_at > now:
        return "future_snapshot"
    if now - snapshot.sampled_at > max_age_s:
        return "stale_snapshot"
    if type(snapshot.requests) is not tuple:
        return "invalid_ledger"
    seen = set()
    for request in snapshot.requests:
        if not isinstance(request, RequestMark):
            return "invalid_request"
        if not _text(request.request_id) or request.request_id in seen:
            return "invalid_request_id"
        seen.add(request.request_id)
        if not _number(request.cost, positive=True):
            return "invalid_request_cost"
        if not _integer(request.prompt_tokens) or not _integer(request.max_tokens, 1):
            return "invalid_request_tokens"
        if not _number(request.received_at) or not _number(request.admitted_at):
            return "invalid_request_time"
        if not request.received_at <= request.admitted_at <= snapshot.sampled_at:
            return "invalid_request_chronology"
        if request.first_at is not None and (
            not _number(request.first_at)
            or not request.admitted_at <= request.first_at <= snapshot.sampled_at
        ):
            return "invalid_request_chronology"
        if request.scheduled_at is not None and (
            not _number(request.scheduled_at) or request.scheduled_at > request.received_at
        ):
            return "invalid_request_chronology"
    try:
        if not _number(snapshot.reserved):
            return "invalid_reserved_cost"
    except (OverflowError, ValueError):
        return "invalid_reserved_cost"
    return None


def _parameters_valid(parameters: ModelParameters) -> bool:
    if not isinstance(parameters, ModelParameters) or not isinstance(parameters.cost_spec, CostSpec):
        return False
    cost = parameters.cost_spec
    return (
        all(_number(getattr(cost, name)) for name in ("intercept", "input_per_token", "output_per_token"))
        and _number(cost.minimum, positive=True)
        and _integer(cost.default_max_tokens, 1)
        and _text(cost.tokenizer_identity)
        and _text(cost.estimator_version)
        and _number(parameters.prefill_s_per_token)
        and _number(parameters.decode_s_per_token, positive=True)
        and _number(parameters.first_overhead_s, positive=True)
        and _number(parameters.decode_interference)
        and _integer(parameters.max_prompt_tokens, 1)
        and _integer(parameters.max_output_tokens, 1)
        and _text(parameters.model_version)
        and _text(parameters.calibration_version)
        and type(parameters.supported_budgets) is tuple
        and bool(parameters.supported_budgets)
        and all(_number(budget, positive=True) for budget in parameters.supported_budgets)
        and len(set(parameters.supported_budgets)) == len(parameters.supported_budgets)
    )


def _config_reason(config: PlanningConfig, parameters: ModelParameters) -> str | None:
    if not isinstance(config, PlanningConfig):
        return "invalid_config"
    if any(
        not _integer(getattr(config, name), 1)
        for name in ("control_steps", "max_events", "max_active_requests", "max_plans", "max_scenarios")
    ):
        return "invalid_limits"
    if any(
        not _number(getattr(config, name), positive=True)
        for name in ("slot_s", "evaluation_window_s", "tail_s", "fallback_budget", "slo_s", "max_wall_s")
    ) or not _number(config.max_snapshot_age_s):
        return "invalid_limits"
    if (
        config.control_steps not in (1, 3, 5)
        or config.slot_s != 1.0
        or config.evaluation_window_s != 5.0
        or config.tail_s != 10.0
        or config.slo_s != 0.5
    ):
        return "unsupported_config"
    if (
        type(config.actions) is not tuple
        or not config.actions
        or any(not _number(action, positive=True) for action in config.actions)
        or len(set(config.actions)) != len(config.actions)
    ):
        return "invalid_actions"
    if any(action not in parameters.supported_budgets for action in config.actions) or (
        config.fallback_budget not in parameters.supported_budgets
    ):
        return "unsupported_budget"
    if len(config.actions) ** config.control_steps > config.max_plans:
        return "plan_limit_exceeded"
    return None


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RolloutError("deadline_exceeded")


def _scenario_reason(
    scenario: Scenario,
    snapshot: Snapshot,
    parameters: ModelParameters,
    config: PlanningConfig,
    deadline: float,
) -> str | None:
    if not isinstance(scenario, Scenario) or type(scenario.offers) is not tuple:
        return "invalid_scenario"
    if (
        not _number(scenario.service_scale, positive=True)
        or not _number(scenario.old_output_fraction, positive=True)
        or scenario.old_output_fraction > 1
        or type(scenario.old_success) is not bool
        or not _number(scenario.ingress_delay_s)
        or not _number(scenario.egress_delay_s)
    ):
        return "invalid_scenario"
    if len(scenario.offers) > config.max_events:
        return "event_limit_exceeded"
    seen = {request.request_id for request in snapshot.requests}
    previous = 0.0
    for offer in scenario.offers:
        _check_deadline(deadline)
        if not isinstance(offer, FutureOffer):
            return "invalid_scenario"
        if not _text(offer.request_id) or offer.request_id in seen:
            return "invalid_scenario"
        seen.add(offer.request_id)
        if (
            not _number(offer.offset_s)
            or offer.offset_s < previous
            or not _integer(offer.prompt_tokens)
            or offer.prompt_tokens > parameters.max_prompt_tokens
            or not _integer(offer.max_tokens, 1)
            or offer.max_tokens > parameters.max_output_tokens
            or not _integer(offer.output_tokens)
            or offer.output_tokens > offer.max_tokens
            or type(offer.success) is not bool
            or (offer.success and offer.output_tokens == 0)
            or not _number(offer.ingress_delay_s)
            or not _number(offer.egress_delay_s)
        ):
            return "invalid_scenario"
        try:
            parameters.cost_spec.estimate(offer.prompt_tokens, offer.max_tokens)
        except (ValueError, OverflowError):
            return "invalid_scenario"
        previous = offer.offset_s
    return None


def choose_plan(
    snapshot: Snapshot,
    parameters: ModelParameters,
    scenarios: tuple[Scenario, ...],
    config: PlanningConfig,
    *,
    now: float | None = None,
    clock_domain: str | None = None,
    predictor=None,
) -> Decision:
    """Compare complete plans on the same supplied draws and fixed cohort window.

    The predictor owns old/new-cohort and eventual-success scoring. Its lower
    count is optimized; upper counts are reported, never treated as resolved
    success. Deadlines are cooperative checks, not synchronous preemption.
    """
    started = time.monotonic()
    evaluated = 0
    scenario_count = len(scenarios) if type(scenarios) is tuple else 0
    identity = (
        snapshot.identity
        if isinstance(snapshot, Snapshot) and _identity_valid(snapshot.identity)
        else Identity("unavailable", "unavailable", "unavailable", "unavailable", "unavailable")
    )

    def fallback(reason: str) -> Decision:
        budget = getattr(config, "fallback_budget", None)
        if not _number(budget, positive=True):
            budget = getattr(snapshot, "budget", None)
        if not _number(budget, positive=True):
            budget = 16.0
        return Decision(
            float(budget), None, reason, True, None, None,
            max(0.0, time.monotonic() - started), evaluated, scenario_count, identity,
        )

    if not _parameters_valid(parameters):
        return fallback("invalid_parameters")
    try:
        identity = parameters.identity
    except (OSError, TypeError, ValueError, OverflowError):
        return fallback("identity_unavailable")
    reason = _config_reason(config, parameters)
    if reason:
        return fallback(reason)
    deadline = started + config.max_wall_s
    if not _number(deadline, positive=True):
        return fallback("invalid_limits")
    if (
        isinstance(snapshot, Snapshot)
        and type(snapshot.requests) is tuple
        and len(snapshot.requests) > config.max_active_requests
    ):
        return fallback("active_request_limit_exceeded")
    if clock_domain is None:
        try:
            from prefill_lab.live_http import clock_domain as local_clock_domain
            clock_domain = local_clock_domain()
        except (OSError, ImportError):
            return fallback("clock_domain_unavailable")
    reason = validate_snapshot(
        snapshot, now=time.monotonic() if now is None else now,
        clock_domain=clock_domain, expected_identity=identity,
        max_age_s=config.max_snapshot_age_s,
    )
    if reason:
        return fallback(reason)
    if snapshot.budget not in parameters.supported_budgets:
        return fallback("unsupported_budget")
    if type(scenarios) is not tuple or not scenarios:
        return fallback("missing_scenarios")
    if len(scenarios) > config.max_scenarios:
        return fallback("scenario_limit_exceeded")

    try:
        _check_deadline(deadline)
        for request in snapshot.requests:
            _check_deadline(deadline)
            if request.prompt_tokens > parameters.max_prompt_tokens or request.max_tokens > parameters.max_output_tokens:
                return fallback("unsupported_request")
            try:
                expected_cost = parameters.cost_spec.estimate(request.prompt_tokens, request.max_tokens)
            except (ValueError, OverflowError):
                return fallback("cost_spec_mismatch")
            if not math.isclose(request.cost, expected_cost, rel_tol=1e-12, abs_tol=1e-12):
                return fallback("cost_spec_mismatch")
        for scenario in scenarios:
            _check_deadline(deadline)
            reason = _scenario_reason(scenario, snapshot, parameters, config, deadline)
            if reason:
                return fallback(reason)
        if predictor is None:
            try:
                from .model import rollout
            except ImportError:
                return fallback("predictor_unavailable")
            predictor = rollout

        best = None
        best_key = None
        best_scores = None
        for actions in product(config.actions, repeat=config.control_steps):
            _check_deadline(deadline)
            plan = Plan(actions, config.slot_s, config.fallback_budget)
            lowers, uppers = [], []
            for scenario in scenarios:
                _check_deadline(deadline)
                result = predictor(snapshot, plan, scenario, parameters, config, deadline=deadline)
                _check_deadline(deadline)
                if not isinstance(result, RolloutResult) or (
                    not _number(result.score_lower)
                    or not _number(result.score_upper)
                    or result.score_upper < result.score_lower
                    or not _integer(result.event_count)
                ):
                    return fallback("invalid_prediction")
                if result.event_count > config.max_events:
                    return fallback("event_limit_exceeded")
                cohort_size = len(snapshot.requests) + sum(
                    offer.offset_s < config.evaluation_window_s for offer in scenario.offers
                )
                if result.score_upper > cohort_size:
                    return fallback("invalid_prediction")
                lowers.append(result.score_lower)
                uppers.append(result.score_upper)
            lower = math.fsum(lowers) / len(scenarios)
            upper = math.fsum(uppers) / len(scenarios)
            evaluated += 1
            changes = sum(
                before != after for before, after in zip((snapshot.budget,) + actions, actions)
            )
            key = (lower, actions[0] == snapshot.budget, -changes)
            if best_key is None or key > best_key:
                best, best_key, best_scores = plan, key, (lower, upper)
        _check_deadline(deadline)
    except RolloutError as exc:
        return fallback(str(exc) or "rollout_error")
    except TimeoutError:
        return fallback("deadline_exceeded")
    except Exception:
        return fallback("predictor_error")

    return Decision(
        best.actions[0], best, "selected", False, best_scores[0], best_scores[1],
        max(0.0, time.monotonic() - started), evaluated, scenario_count, identity,
    )


def feedback_budget(
    snapshot: Snapshot,
    *,
    low: float = 16.0,
    high: float = 32.0,
    backlog_low: int = 2000,
    backlog_high: int = 6000,
) -> float:
    """Same-state hysteresis; inside the band hold the bounded current budget."""
    if (
        not _number(low, positive=True)
        or not _number(high, positive=True)
        or low >= high
        or not _integer(backlog_low)
        or not _integer(backlog_high)
        or backlog_low >= backlog_high
    ):
        raise ValueError("invalid_feedback_config")
    if not isinstance(snapshot, Snapshot):
        raise ValueError("missing_snapshot")
    reason = validate_snapshot(
        snapshot, now=snapshot.sampled_at, clock_domain=snapshot.clock_domain,
        expected_identity=snapshot.identity, max_age_s=0,
    )
    if reason:
        raise ValueError(reason)
    if snapshot.prompt_backlog >= backlog_high:
        return float(low)
    if snapshot.prompt_backlog <= backlog_low:
        return float(high)
    return float(min(high, max(low, snapshot.budget)))
