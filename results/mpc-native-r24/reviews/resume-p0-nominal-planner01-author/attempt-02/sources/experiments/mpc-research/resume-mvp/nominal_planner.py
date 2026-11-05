"""CPU-only Gateway nominal planning; a proposal is neither a permit nor an ACK.

The existing cpu-lifecycle-v1 / synthetic-unqualified-v1 simulator supplies
every forecast. Remaining work, output length, success, transport delay and
future arrivals are explicit nominal assumptions, not observed Native state.
The caller supplies ONE provenance-bound development/past-information scenario;
provenance declarations are not proof that its producer avoided evaluation data.

This bounded derivation reuses the accepted planner's exact binary-input
ordering, complete-panel and full-action reference ideas, not its Native API or
qualification. The objective here is resolved-success TTFT<=0.5 COUNT, not
natural_slo_good. Ties prefer holding the first action, fewer transitions,
smaller budget sum, then the path. After three slots, EVERY policy holds its
last planned budget; thus Fixed and OneMove do not acquire a hidden tail switch.

Clock/read_state callbacks must be trustworthy, bounded and in the snapshot's
local monotonic clock domain. Guards are cooperative, not hard preemption.
The coordinator must recheck state/expiry, honor not_before, obtain its OWN
authorization and record actual application. No HTTP, actuation or RNG is here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import product
import math
import time
from typing import Callable

from mpc_lab.contracts import (
    CostSpec, FutureOffer, Identity, ModelParameters, Outcome, PlanningConfig,
    RolloutResult, Scenario, Snapshot, digest,
)
from mpc_lab.controller import validate_snapshot
from mpc_lab.model import rollout


PLANNER_VERSION = "resume-p0-gateway-nominal-planner-v1"
OBJECTIVE_ID = "gateway-resolved-success-ttft0.5-lower-count-hold-switch-sum-lex-v1"
BUDGETS = (16.0, 32.0)
PATHS = tuple(product(BUDGETS, repeat=3))
MAX_SOURCE_AGE_S = 0.5
MAX_OFFERS = 512
CONFIG = PlanningConfig(
    max_plans=8, max_scenarios=1, max_snapshot_age_s=MAX_SOURCE_AGE_S,
)
CONTRACT_SHA256 = digest({
    "planner": PLANNER_VERSION, "objective": OBJECTIVE_ID,
    "config": asdict(CONFIG), "budgets": BUDGETS, "scenario_count": 1,
    "max_offers": MAX_OFFERS, "continuation": "hold-last-planned-budget",
    "arithmetic": "exact-rational-frozen-binary-inputs",
})


class PlanningRefused(ValueError):
    """Atomic refusal: no selected action or partial-panel winner is returned."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PlanningRefused(reason)


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _sha(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class StateVersion:
    run_id: str
    clock_domain: str
    identity: Identity
    budget_version: int
    sequence: int
    budget: float

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot) -> StateVersion:
        return cls(
            snapshot.run_id, snapshot.clock_domain, snapshot.identity,
            snapshot.budget_version, snapshot.sequence, snapshot.budget,
        )


@dataclass(frozen=True)
class CausalScenario:
    scenario: Scenario
    conditioned_on: StateVersion
    source_sha256: str
    information_cutoff_at: float
    available_at: float
    origin: str  # development | past_observations; never evaluation_future
    purpose: str  # train | evaluation | engineering
    objective_id: str = OBJECTIVE_ID


@dataclass(frozen=True)
class DecisionWindow:
    entered_at: float
    solve_deadline: float
    first_action_at: float
    action_deadline: float


@dataclass(frozen=True)
class TimedPlan:
    actions: tuple[float, ...]
    received_budget: float
    sampled_at: float
    first_action_at: float

    @property
    def slot_s(self) -> float:
        return CONFIG.slot_s

    @property
    def continuation_budget(self) -> float:
        return self.actions[-1]

    @property
    def action_times(self) -> tuple[float, ...]:
        return tuple(self.first_action_at + i * self.slot_s for i in range(3))

    def budget_at(self, offset_s: float) -> float:
        budget = self.received_budget
        for at, action in zip(self.action_times, self.actions):
            if offset_s < at - self.sampled_at:
                break
            budget = action
        return budget

    @property
    def transitions(self) -> int:
        return sum(
            left != right
            for left, right in zip((self.received_budget,) + self.actions, self.actions)
        )


@dataclass(frozen=True)
class Candidate:
    plan: TimedPlan
    forecast: RolloutResult


@dataclass(frozen=True)
class ForecastPanel:
    snapshot: Snapshot
    scenario: CausalScenario
    window: DecisionWindow
    started_at: float
    candidates: tuple[Candidate, ...]
    objective_id: str = OBJECTIVE_ID
    contract_sha256: str = CONTRACT_SHA256


@dataclass(frozen=True)
class FixedChoice:
    budget: float
    identity: Identity
    training_panel_sha256: tuple[str, ...]
    candidate_totals: tuple[tuple[float, Fraction], ...]
    objective_id: str = OBJECTIVE_ID
    contract_sha256: str = CONTRACT_SHA256


@dataclass(frozen=True)
class FirstAction:
    budget: float
    not_before: float
    not_after: float
    source: StateVersion


@dataclass(frozen=True)
class NominalDecision:
    policy: str
    plan: TimedPlan
    first_action: FirstAction
    predicted_lower: Fraction | None
    predicted_upper: Fraction | None
    evaluated_paths: int
    objective_id: str = OBJECTIVE_ID
    planner_version: str = PLANNER_VERSION
    contract_sha256: str = CONTRACT_SHA256


def _rank(candidate: Candidate) -> tuple:
    plan = candidate.plan
    return (
        -Fraction(candidate.forecast.score_lower),
        plan.actions[0] != plan.received_budget,
        plan.transitions,
        sum(map(Fraction, plan.actions)),
        plan.actions,
    )


class _Guard:
    def __init__(
        self, snapshot: Snapshot, identity: Identity, window: DecisionWindow,
        clock_domain: str, read_state: Callable[[], StateVersion],
        clock: Callable[[], float], parameters: ModelParameters,
    ):
        _require(type(window) is DecisionWindow, "invalid_decision_window")
        _require(callable(read_state) and callable(clock), "missing_guard_callbacks")
        self.snapshot, self.identity, self.window = snapshot, identity, window
        self.clock_domain, self.read_state, self.clock = clock_domain, read_state, clock
        self.parameters, self.ledger_checked = parameters, False
        _require(all(_number(v) for v in asdict(window).values()), "invalid_decision_window")
        self.last = window.entered_at

    def _time(self) -> float:
        now = self.clock()
        _require(_number(now) and now >= self.last, "nonmonotonic_clock")
        self.last = now
        reason = validate_snapshot(
            self.snapshot, now=now, clock_domain=self.clock_domain,
            expected_identity=self.identity, max_age_s=MAX_SOURCE_AGE_S,
        )
        _require(reason is None, reason or "invalid_snapshot")
        if not self.ledger_checked:
            _require(len(self.snapshot.requests) <= CONFIG.max_active_requests, "active_limit")
            for request in self.snapshot.requests:
                _require(
                    request.prompt_tokens <= self.parameters.max_prompt_tokens
                    and request.max_tokens <= self.parameters.max_output_tokens,
                    "unsupported_request_tokens",
                )
                expected_cost = self.parameters.cost_spec.estimate(request.prompt_tokens, request.max_tokens)
                _require(
                    math.isclose(request.cost, expected_cost, rel_tol=1e-12, abs_tol=0.0),
                    "ledger_cost_mismatch",
                )
            self.ledger_checked = True
        w = self.window
        _require(
            w.entered_at < w.solve_deadline <= w.entered_at + CONFIG.max_wall_s
            and w.solve_deadline <= w.first_action_at < w.action_deadline
            and w.action_deadline <= self.snapshot.sampled_at + MAX_SOURCE_AGE_S,
            "invalid_decision_window",
        )
        _require(now < w.solve_deadline, "solve_deadline_exhausted")
        _require(self.snapshot.budget in BUDGETS, "unsupported_received_budget")
        times = tuple(w.first_action_at + i * CONFIG.slot_s for i in range(3))
        _require(all(a < b for a, b in zip(times, times[1:])), "nonadvancing_action_times")
        return now

    def check(self) -> float:
        self._time()
        try:
            current = self.read_state()
        except Exception as exc:
            raise PlanningRefused("state_read_failed") from exc
        _require(type(current) is StateVersion, "invalid_current_state")
        expected = StateVersion.from_snapshot(self.snapshot)
        _require(current.clock_domain == expected.clock_domain, "state_clock_changed")
        _require(current.identity == expected.identity, "state_identity_changed")
        _require(current.run_id == expected.run_id, "state_run_changed")
        _require(
            type(current.budget_version) is int and type(current.sequence) is int
            and (current.budget_version, current.sequence)
            == (expected.budget_version, expected.sequence),
            "state_version_changed",
        )
        _require(_number(current.budget) and current.budget == expected.budget, "state_budget_changed")
        return self._time()


@dataclass(frozen=True, init=False)
class NominalPlanner:
    parameters: ModelParameters
    identity: Identity

    def __init__(self, parameters: ModelParameters = ModelParameters()):
        _require(type(parameters) is ModelParameters, "invalid_parameters")
        _require(type(parameters.cost_spec) is CostSpec, "invalid_cost_spec")
        _require(
            parameters == replace(ModelParameters(), cost_spec=parameters.cost_spec),
            "v1_parameters_not_frozen",
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "identity", parameters.identity)

    def _input(self, snapshot: Snapshot, source: CausalScenario, now: float) -> None:
        _require(type(source) is CausalScenario, "missing_typed_causal_scenario")
        _require(source.objective_id == OBJECTIVE_ID, "objective_mismatch")
        _require(
            source.origin in ("development", "past_observations")
            and source.purpose in ("train", "evaluation", "engineering"),
            "unapproved_scenario_provenance",
        )
        _require(_sha(source.source_sha256), "invalid_scenario_source_hash")
        _require(
            type(source.conditioned_on) is StateVersion
            and source.conditioned_on == StateVersion.from_snapshot(snapshot),
            "scenario_state_mismatch",
        )
        _require(
            _number(source.information_cutoff_at) and _number(source.available_at)
            and source.information_cutoff_at <= snapshot.sampled_at
            and source.information_cutoff_at <= source.available_at <= now,
            "noncausal_scenario_information",
        )
        sc = source.scenario
        _require(
            type(sc) is Scenario and type(sc.offers) is tuple
            and len(sc.offers) <= MAX_OFFERS, "invalid_or_oversized_scenario",
        )
        _require(
            all(
                type(offer) is FutureOffer and _number(offer.offset_s)
                and offer.offset_s < CONFIG.evaluation_window_s + CONFIG.tail_s
                for offer in sc.offers
            ),
            "invalid_scenario_horizon",
        )

    def _result(self, result: RolloutResult, snapshot: Snapshot, source: CausalScenario) -> None:
        _require(type(result) is RolloutResult, "invalid_forecast_result")
        expected = {r.request_id for r in snapshot.requests}
        expected.update(
            o.request_id for o in source.scenario.offers
            if o.offset_s < CONFIG.evaluation_window_s
        )
        _require(
            type(result.outcomes) is tuple
            and all(type(o) is Outcome for o in result.outcomes)
            and len(result.outcomes) == len(expected)
            and {o.request_id for o in result.outcomes} == expected,
            "incomplete_forecast_cohort",
        )
        lower, upper = result.score_lower, result.score_upper
        _require(
            _number(lower) and _number(upper)
            and Fraction(lower).denominator == Fraction(upper).denominator == 1
            and 0 <= lower <= upper <= len(expected)
            and lower == sum(o.good(CONFIG.slo_s) for o in result.outcomes)
            and upper <= lower + sum(o.success is None for o in result.outcomes),
            "objective_count_mismatch",
        )
        _require(
            type(result.event_count) is int and 0 < result.event_count <= CONFIG.max_events,
            "invalid_forecast_event_count",
        )

    def _panel(self, panel: ForecastPanel, check: Callable[[], float] | None = None) -> None:
        _require(type(panel) is ForecastPanel, "missing_typed_panel")
        _require(
            panel.objective_id == OBJECTIVE_ID and panel.contract_sha256 == CONTRACT_SHA256,
            "objective_or_contract_mismatch",
        )
        _require(panel.snapshot.identity == self.identity, "identity_mismatch")
        self._input(panel.snapshot, panel.scenario, panel.started_at)
        rows = panel.candidates
        _require(
            type(rows) is tuple and len(rows) == len(PATHS)
            and all(
                type(row) is Candidate and type(row.plan) is TimedPlan
                and type(row.plan.actions) is tuple
                and len(row.plan.actions) == CONFIG.control_steps
                and all(_number(b) and b in BUDGETS for b in row.plan.actions)
                for row in rows
            )
            and {row.plan.actions for row in rows} == set(PATHS),
            "incomplete_candidate_panel",
        )
        for row in rows:
            if check is not None:
                check()
            _require(
                row.plan == TimedPlan(
                    row.plan.actions, panel.snapshot.budget, panel.snapshot.sampled_at,
                    panel.window.first_action_at,
                ),
                "candidate_time_or_budget_mismatch",
            )
            self._result(row.forecast, panel.snapshot, panel.scenario)

    def evaluate(
        self, snapshot: Snapshot, source: CausalScenario, window: DecisionWindow, *,
        clock_domain: str, read_state: Callable[[], StateVersion],
        clock: Callable[[], float] = time.monotonic,
    ) -> ForecastPanel:
        guard = _Guard(snapshot, self.identity, window, clock_domain, read_state, clock, self.parameters)
        started = guard.check()
        self._input(snapshot, source, started)
        rows = []
        for actions in PATHS:
            guard.check()
            plan = TimedPlan(actions, snapshot.budget, snapshot.sampled_at, window.first_action_at)
            try:
                result = rollout(
                    snapshot, plan, source.scenario, self.parameters, CONFIG,
                    deadline=window.solve_deadline,
                )
            except Exception as exc:
                raise PlanningRefused("forecast_refused:" + str(exc)) from exc
            guard.check()
            self._result(result, snapshot, source)
            rows.append(Candidate(plan, result))
        panel = ForecastPanel(snapshot, source, window, started, tuple(rows))
        self._panel(panel, guard.check)
        guard.check()
        return panel

    def select(
        self, panel: ForecastPanel, *, policy: str, clock_domain: str,
        read_state: Callable[[], StateVersion], clock: Callable[[], float] = time.monotonic,
    ) -> NominalDecision:
        _require(type(panel) is ForecastPanel, "missing_typed_panel")
        _require(policy in ("mpc", "one_move"), "use_frozen_choice_for_fixed_execution")
        guard = _Guard(
            panel.snapshot, self.identity, panel.window, clock_domain, read_state, clock, self.parameters,
        )
        guard.check()
        self._panel(panel, guard.check)
        rows = panel.candidates
        if policy == "one_move":
            rows = tuple(row for row in rows if row.plan.transitions <= 1)
        best = min(rows, key=_rank)
        decision = self._decision(
            policy, best.plan, panel.snapshot, panel.window,
            Fraction(best.forecast.score_lower), Fraction(best.forecast.score_upper), len(PATHS),
        )
        guard.check()
        return decision

    def plan(
        self, snapshot: Snapshot, source: CausalScenario, window: DecisionWindow, *,
        policy: str, clock_domain: str, read_state: Callable[[], StateVersion],
        clock: Callable[[], float] = time.monotonic,
    ) -> NominalDecision:
        _require(policy in ("mpc", "one_move"), "use_frozen_choice_for_fixed_execution")
        panel = self.evaluate(
            snapshot, source, window, clock_domain=clock_domain, read_state=read_state, clock=clock,
        )
        return self.select(
            panel, policy=policy, clock_domain=clock_domain, read_state=read_state, clock=clock,
        )

    def freeze_fixed(self, training: tuple[ForecastPanel, ...]) -> FixedChoice:
        """Train-only reference selection; never select a benchmark cap on Eval."""
        _require(type(training) is tuple and 1 <= len(training) <= 16, "bounded_train_panels_required")
        totals = {budget: Fraction(0) for budget in BUDGETS}
        hashes = []
        for panel in training:
            self._panel(panel)
            _require(panel.scenario.purpose == "train", "fixed_selection_requires_train")
            hashes.append(digest(asdict(panel)))
            for row in panel.candidates:
                if len(set(row.plan.actions)) == 1:
                    totals[row.plan.actions[0]] += Fraction(row.forecast.score_lower)
        _require(len(set(hashes)) == len(hashes), "duplicate_train_panel")
        budget = min(BUDGETS, key=lambda b: (-totals[b], Fraction(b)))
        return FixedChoice(budget, self.identity, tuple(hashes), tuple(totals.items()))

    def fixed_action(
        self, snapshot: Snapshot, choice: FixedChoice, window: DecisionWindow, *,
        clock_domain: str, read_state: Callable[[], StateVersion],
        clock: Callable[[], float] = time.monotonic,
    ) -> NominalDecision:
        """Runtime Fixed does not run or wait for the comparative model solver."""
        guard = _Guard(snapshot, self.identity, window, clock_domain, read_state, clock, self.parameters)
        guard.check()
        _require(
            type(choice) is FixedChoice and choice.identity == self.identity
            and choice.objective_id == OBJECTIVE_ID and choice.contract_sha256 == CONTRACT_SHA256,
            "invalid_fixed_choice",
        )
        _require(
            type(choice.training_panel_sha256) is tuple
            and 1 <= len(choice.training_panel_sha256) <= 16
            and all(_sha(h) for h in choice.training_panel_sha256)
            and type(choice.candidate_totals) is tuple
            and len(choice.candidate_totals) == len(BUDGETS)
            and tuple(b for b, _ in choice.candidate_totals) == BUDGETS
            and all(type(score) is Fraction and score >= 0 for _, score in choice.candidate_totals),
            "incomplete_fixed_training_reference",
        )
        expected = min(choice.candidate_totals, key=lambda row: (-row[1], Fraction(row[0])))[0]
        _require(choice.budget == expected, "fixed_choice_mismatch")
        plan = TimedPlan((choice.budget,) * 3, snapshot.budget, snapshot.sampled_at, window.first_action_at)
        decision = self._decision("fixed", plan, snapshot, window, None, None, 0)
        guard.check()
        return decision

    @staticmethod
    def _decision(policy, plan, snapshot, window, lower, upper, evaluated):
        return NominalDecision(
            policy, plan,
            FirstAction(
                plan.actions[0], window.first_action_at, window.action_deadline,
                StateVersion.from_snapshot(snapshot),
            ),
            lower, upper, evaluated,
        )
