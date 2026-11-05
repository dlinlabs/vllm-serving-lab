"""Causal R21 prefix search; prediction owns full-horizon reactive continuation."""

from itertools import product
import math
import time

from mpc_lab.contracts import finite as _finite
from mpc_local.contracts import (
    Arrival, Choice, Config, Forecast, Prediction, Predictor, PublicRequest, PublicState,
)


_ACTIONS = (16.0, 32.0)
_HISTORY_S = 5.0
_FORECAST_SEED = 99173


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _config_reason(config):
    if not isinstance(config, Config):
        return "invalid_config"
    if (
        type(config.actions) is not tuple or len(config.actions) != 2
        or any(not _finite(a) for a in config.actions)
        or set(config.actions) != set(_ACTIONS) or type(config.moves) is not int
        or config.moves != 2
    ):
        return "unsupported_config"
    positive = (
        config.slot_s, config.horizon_s, config.slo_ttft_s, config.slo_tpot_s,
        config.http_deadline_s, config.solver_budget_s, config.reactive_threshold,
    )
    if any(not _finite(v) or v <= 0 for v in positive):
        return "invalid_config"
    if any(not _finite(v) or v < 0 for v in (config.tail_s, config.max_snapshot_age_s)):
        return "invalid_config"
    if config.offered_end_s is not None and (
        not _finite(config.offered_end_s) or config.offered_end_s < 0
    ):
        return "invalid_offered_end"
    if config.horizon_s < config.moves * config.slot_s:
        return "short_horizon"
    return None


def _state_reason(state, max_age_s, offered_end_s=None):
    if not isinstance(state, PublicState) or state.complete is not True:
        return "incomplete_state"
    if (
        not _finite(state.now) or not _finite(state.budget)
        or state.budget not in _ACTIONS or not _integer(state.version)
        or not isinstance(state.run_id, str) or not state.run_id
        or not isinstance(state.clock_domain, str) or not state.clock_domain
        or type(state.requests) is not tuple or type(state.arrivals) is not tuple
    ):
        return "invalid_state"
    sampled_at = state.sampled_at
    if sampled_at is None:
        if state.clock_domain != "simulation":
            return "missing_sample_time"
        sampled_at = state.now
    if not _finite(sampled_at) or sampled_at > state.now:
        return "invalid_sample_time"
    if state.now - sampled_at > max_age_s:
        return "stale_state"
    seen = set()
    for request in state.requests:
        if not isinstance(request, PublicRequest):
            return "invalid_request"
        if (
            not isinstance(request.request_id, str) or not request.request_id
            or request.request_id in seen or not _integer(request.prompt_tokens)
            or not _integer(request.max_tokens, 1)
            or not _finite(request.cost) or request.cost <= 0
            or not _finite(request.scheduled_at) or not _finite(request.admitted_at)
            or not request.scheduled_at <= request.admitted_at <= sampled_at
            or not _integer(request.observed_output_tokens)
            or request.observed_output_tokens > request.max_tokens
        ):
            return "invalid_request"
        seen.add(request.request_id)
        if request.first_at is None:
            if request.observed_output_tokens:
                return "invalid_progress"
        elif (
            not _finite(request.first_at)
            or not request.admitted_at <= request.first_at <= sampled_at
            or request.observed_output_tokens == 0
        ):
            return "invalid_progress"
        output_at = request.output_sampled_at
        if output_at is None:
            if state.clock_domain != "simulation":
                return "missing_output_sample_time"
            output_at = sampled_at
        if (
            not _finite(output_at)
            or not request.admitted_at <= output_at <= sampled_at
            or (request.first_at is not None and request.first_at > output_at)
        ):
            return "invalid_output_sample_time"
        terminal = request.terminal_at
        finish = request.finish_reason
        if terminal is None:
            if finish is not None:
                return "invalid_terminal"
        elif (
            not _finite(terminal) or not request.admitted_at <= terminal <= sampled_at
            or (request.first_at is not None and terminal < request.first_at)
            or not isinstance(finish, str) or not finish
            or (finish == "length" and request.observed_output_tokens != request.max_tokens)
        ):
            return "invalid_terminal"
    for arrival in state.arrivals:
        if (
            not isinstance(arrival, Arrival) or not _finite(arrival.at)
            or arrival.at > sampled_at or not _integer(arrival.prompt_tokens)
            or not _integer(arrival.max_tokens, 1)
        ):
            return "invalid_arrival_history"
    try:
        if not math.isfinite(math.fsum(r.cost for r in state.requests)):
            return "invalid_stock"
    except OverflowError:
        return "invalid_stock"
    if offered_end_s is not None and (
        any(a.at >= offered_end_s for a in state.arrivals)
        or any(r.scheduled_at >= offered_end_s for r in state.requests)
    ):
        return "invalid_offered_history"
    return None


def _forecast(state):
    recent = sorted(
        (a for a in state.arrivals if state.now - _HISTORY_S < a.at <= state.now),
        key=lambda a: (a.at, a.prompt_tokens, a.max_tokens),
    )
    return Forecast(
        len(recent) / _HISTORY_S,
        tuple((a.prompt_tokens, a.max_tokens) for a in recent),
        _FORECAST_SEED,
    )


def forecast(state: PublicState) -> Forecast:
    """Fixed trailing-five-second empirical law; no history means no guessed marks."""
    reason = _state_reason(state, Config().max_snapshot_age_s)
    if reason:
        raise ValueError(reason)
    return _forecast(state)


def reactive(state: PublicState, config: Config) -> float:
    """Retain each owner's entire supplied cost, including after first content."""
    reason = _config_reason(config) or _state_reason(
        state, config.max_snapshot_age_s, config.offered_end_s,
    )
    if reason:
        raise ValueError(reason)
    return 16.0 if math.fsum(r.cost for r in state.requests) > config.reactive_threshold else 32.0


def _prediction_valid(prediction, state):
    if not isinstance(prediction, Prediction):
        return False
    values = (
        prediction.cohort_good, prediction.window_good, prediction.admitted,
        prediction.rejected, prediction.final_stock, prediction.cost_in, prediction.cost_out,
    )
    if not (
        all(_finite(v) and v >= 0 for v in values)
        and prediction.window_good <= prediction.cohort_good
        and prediction.cohort_good <= prediction.admitted + len(state.requests)
    ):
        return False
    try:
        represented = math.fsum(
            [r.cost for r in state.requests] + [prediction.cost_in, -prediction.cost_out]
        )
    except OverflowError:
        return False
    # Same floating-point tolerance as the source owner ledger, not a model-error allowance.
    return math.isclose(prediction.final_stock, represented, rel_tol=1e-12, abs_tol=1e-9)


def choose(
    state: PublicState, model: Predictor, config: Config = Config(),
    forecast: Forecast | None = None, mode: str = "mpc",
) -> Choice:
    """Enumerate complete-objective prefixes, never just immediate one-move reward.

    Short-prefix predicted-stock continuation is the Predictor contract.
    Deadlines are measured before/after calls; synchronous prediction is not
    preempted. An overrun returns an explicit late fallback, not a partial winner.
    Invalid inputs have an empty plan (fenced). Solver failures after valid
    inputs carry only a fallback candidate, never dispatch permission. The
    caller must revalidate the producing snapshot, phase clocks, run/identity,
    CAS and command expiry after elapsed solve time before any application.
    """
    started = time.perf_counter()
    scores = []

    def fallback(reason, *, solver_failure=False):
        return Choice(16.0, (16.0,) if solver_failure else (), tuple(scores),
                      reason, time.perf_counter() - started)

    reason = _config_reason(config)
    if reason:
        return fallback(reason)
    deadline = started + config.solver_budget_s
    if mode not in ("mpc", "one_move"):
        return fallback("invalid_mode")
    reason = _state_reason(state, config.max_snapshot_age_s, config.offered_end_s)
    if reason:
        return fallback(reason)
    common_forecast = _forecast(state)
    # An injected forecast is a reusable causal cache, not an oracle override.
    if forecast is not None and (
        type(forecast) is not Forecast or not _finite(forecast.rate)
        or type(forecast.seed) is not int or type(forecast.marks) is not tuple
        or any(
            type(mark) is not tuple or len(mark) != 2
            or not _integer(mark[0]) or not _integer(mark[1], 1)
            for mark in forecast.marks
        )
        or forecast != common_forecast
    ):
        return fallback("noncausal_forecast")
    if forecast is not None:
        common_forecast = forecast
    best_plan, best_key = None, None
    for plan in product(_ACTIONS, repeat=2 if mode == "mpc" else 1):
        if time.perf_counter() >= deadline:
            return fallback("solver_deadline", solver_failure=True)
        try:
            prediction = model.predict(state, plan, common_forecast, config)
        except Exception:
            if time.perf_counter() >= deadline:
                return fallback("solver_deadline", solver_failure=True)
            return fallback("prediction_error", solver_failure=True)
        if not _prediction_valid(prediction, state):
            if time.perf_counter() >= deadline:
                return fallback("solver_deadline", solver_failure=True)
            return fallback("invalid_prediction", solver_failure=True)
        score = prediction.cohort_good / config.horizon_s
        if not _finite(score):
            return fallback("invalid_prediction", solver_failure=True)
        scores.append((plan, score))
        if time.perf_counter() >= deadline:
            return fallback("solver_deadline", solver_failure=True)
        changes = sum(a != b for a, b in zip((state.budget,) + plan, plan))
        key = (score, plan[0] == state.budget, -changes)
        if best_key is None or key > best_key:
            best_plan, best_key = plan, key
    elapsed = time.perf_counter() - started
    if elapsed >= config.solver_budget_s:
        return fallback("solver_deadline", solver_failure=True)
    return Choice(best_plan[0], best_plan, tuple(scores), "ok", elapsed)
