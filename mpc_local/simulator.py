"""CPU discrete-event workload and service simulator for local R21 MPC research.

This simulator models GPU serving dynamics with:
- Deterministic workload generation (heterogeneous prompt/max tokens and natural output lengths)
- Strict non-evicting whole-cost retention in reservation stock
- Causal public state emission (zero future truth or hidden labels exposed to policies)
- Finite multi-phase service: prefill (TTFT), batched decode (TPOT contention), and terminal cleanup
- Full all-assigned denominator scoring (including HTTP 503 capacity rejects)
- Dual service models:
  * v2_event_driven (default): dynamic post-admission load sharing with event-driven decode rate
  * v1_admission_fixed (ablation): admission-time fixed service duration
- Canonical terminal-inclusive token interval scoring matching independent reference scorer
- Actuator TTL enforcement: commands delayed >= 100ms TTL expire without application
- Privileged offline training record generation adhering to R21 canonical schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import time
import random
from typing import Any, Callable

from mpc_lab.contracts import CostSpec
from mpc_local.contracts import Arrival, Choice, Config, PublicRequest, PublicState


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkItem:
    """An offered request with public arrival metadata and privileged offline service truth."""
    request_id: str
    arrival_s: float
    prompt_tokens: int
    max_tokens: int
    cost: float
    # Privileged offline ground truth — NEVER exposed to online policies:
    true_output_tokens: int
    seed: int
    regime: str = "mixed"
    natural_output_tokens: int | None = None
    finish_reason: str = "stop"  # "stop" | "length"


@dataclass(frozen=True)
class RequestOutcome:
    """Terminal outcome of an offered request under simulation."""
    request_id: str
    arrival_s: float
    prompt_tokens: int
    max_tokens: int
    cost: float
    admitted: bool
    rejection_reason: str | None = None
    admitted_at: float | None = None
    first_at: float | None = None
    last_token_at: float | None = None
    terminal_at: float | None = None
    owner_release_at: float | None = None
    owner_release_s: float = 0.0
    owner_release_extra_s: float = 0.0
    api_terminal_overhead_s: float | None = None
    output_tokens: int = 0
    natural_output_tokens: int | None = None
    finish_reason: str | None = None  # "stop" | "length" | None
    fate: str = "rejected"  # "complete" | "rejected" | "timeout" | "capped" | "unknown"
    ttft_s: float | None = None
    token_interval_s: float | None = None  # Canonical terminal-inclusive (terminal_at - first_at) / (output_tokens - 1)
    decode_step_s: float | None = None     # Pure decode step (last_token_at - first_at) / (output_tokens - 1)
    cleanup_s: float | None = None
    ttft_slo_met: bool = False
    tpot_slo_met: bool = False
    output_min_met: bool = False
    deadline_met: bool = False
    cohort_good: bool = False
    window_good: bool = False

    def to_reference_row(self) -> dict[str, Any]:
        """Convert to independent reference scorer row format."""
        row: dict[str, Any] = {
            "request_id": self.request_id,
            "scheduled_at": self.arrival_s,
            "fate": self.fate,
        }
        if self.admitted and self.fate in ("complete", "capped"):
            row.update({
                "first_at": self.first_at,
                "terminal_at": self.terminal_at,
                "output_tokens": self.output_tokens,
                "finish_reason": self.finish_reason or "stop",
            })
        return row


# ---------------------------------------------------------------------------
# Workload Generator & Tape Hashing
# ---------------------------------------------------------------------------

def compute_public_tape_hash(items: list[WorkItem]) -> str:
    """Exact deterministic SHA-256 hash of public arrival tape (no rounding)."""
    canonical = [
        {
            "id": w.request_id,
            "arrival_s": w.arrival_s,
            "prompt_tokens": w.prompt_tokens,
            "max_tokens": w.max_tokens,
            "cost": w.cost,
        }
        for w in items
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_truth_tape_hash(items: list[WorkItem]) -> str:
    """Exact deterministic SHA-256 hash of complete privileged truth tape (no rounding)."""
    canonical = [
        {
            "id": w.request_id,
            "arrival_s": w.arrival_s,
            "prompt_tokens": w.prompt_tokens,
            "max_tokens": w.max_tokens,
            "cost": w.cost,
            "seed": w.seed,
            "regime": w.regime,
            "natural_output_tokens": w.natural_output_tokens,
            "true_output_tokens": w.true_output_tokens,
            "finish_reason": w.finish_reason,
        }
        for w in items
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_tape_hash(items: list[WorkItem]) -> str:
    """Default tape hash (returns exact public arrival tape hash)."""
    return compute_public_tape_hash(items)


def generate(seed: int, duration_s: float = 20.0, regime: str = "mixed") -> list[WorkItem]:
    """Deterministically generate a sequence of WorkItems for an episode.

    Args:
        seed: Random seed for arrival times and prompt/output geometry.
        duration_s: Duration of offered arrival window in seconds.
        regime: Workload regime ('mixed', 'p4096-m128', 'p128-m128', 'p128-m256', 'p4096-m256').

    Returns:
        Deterministic list of WorkItem instances sorted by arrival_s.
    """
    rng = random.Random(seed)
    cost_spec = CostSpec()

    regime_tables: dict[str, list[tuple[int, int, float]]] = {
        "mixed": [
            (128, 128, 0.40),
            (128, 256, 0.30),
            (4096, 128, 0.20),
            (4096, 256, 0.10),
        ],
        "p4096-m128": [(4096, 128, 1.0)],
        "p128-m128": [(128, 128, 1.0)],
        "p128-m256": [(128, 256, 1.0)],
        "p4096-m256": [(4096, 256, 1.0)],
    }

    if regime not in regime_tables:
        raise ValueError(f"unknown_regime: {regime}")

    table = regime_tables[regime]
    shapes = [(p, m) for p, m, _ in table]
    weights = [w for _, _, w in table]

    arrival_rate = 8.0
    items: list[WorkItem] = []
    current_time = 0.0
    req_idx = 0

    while True:
        interval = rng.expovariate(arrival_rate)
        current_time += interval
        if current_time > duration_s:
            break

        shape = rng.choices(shapes, weights=weights, k=1)[0]
        prompt_tokens, max_tokens = shape
        cost = cost_spec.estimate(prompt_tokens, max_tokens)

        natural_output = int(rng.expovariate(1.0 / 48.0)) + 1
        if natural_output >= max_tokens:
            true_output_tokens = max_tokens
            finish_reason = "length"
        else:
            true_output_tokens = max(1, natural_output)
            finish_reason = "stop"

        # Request ID is purely sequential to prevent online policies from inferring seed
        request_id = f"req_{req_idx:05d}"
        items.append(
            WorkItem(
                request_id=request_id,
                arrival_s=current_time,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                cost=cost,
                true_output_tokens=true_output_tokens,
                seed=seed,
                regime=regime,
                natural_output_tokens=natural_output,
                finish_reason=finish_reason,
            )
        )
        req_idx += 1

    return items


# ---------------------------------------------------------------------------
# Simulator Implementation
# ---------------------------------------------------------------------------

@dataclass
class _PendingCommand:
    send_time: float
    expires_at: float
    applied_at: float
    budget: float
    version: int
    decision_index: int


@dataclass
class _V2RequestState:
    item: WorkItem
    admitted_at: float
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    prefill_done_at: float | None = None
    first_at: float | None = None
    last_token_at: float | None = None
    terminal_at: float | None = None
    owner_release_at: float | None = None
    # Decode progress tracking
    tokens_needed: int = 0
    tokens_remaining: float = 0.0
    # Integral metrics during decode
    accum_stock_tokens: float = 0.0
    accum_nactive_tokens: float = 0.0
    accum_decode_tokens: float = 0.0
    stock_at_admit: float = 0.0
    nactive_at_admit: int = 0
    phase: str = "prefill"  # "prefill" -> "decode" -> "cleanup" -> "complete"

    def tokens_at(self, t: float) -> int:
        if self.first_at is None or t < self.first_at:
            return 0
        if self.output_tokens <= 1:
            return self.output_tokens
        if self.last_token_at is not None and t >= self.last_token_at:
            return self.output_tokens
        emitted = 1 + int(self.tokens_needed - self.tokens_remaining)
        return min(self.output_tokens, max(1, emitted))


def simulate(
    items: list[WorkItem],
    policy: Callable[[PublicState], Choice | float] | float,
    duration_s: float = 20.0,
    control_delay_s: float = 0.0,
    solver_delay_s: float = 0.0,
    config: Config | None = None,
    service_model: str = "v2_event_driven",
    extra_owner_release_s: float = 0.0,
) -> dict[str, Any]:
    """Execute discrete-event simulation of workload items under a control policy.

    Args:
        items: List of WorkItems to simulate.
        policy: Callable taking PublicState and returning Choice or budget float,
                or a constant budget float.
        duration_s: Duration of active arrivals window in seconds.
        control_delay_s: Transport/CAS delay before a budget change takes effect.
        solver_delay_s: Time elapsed during solver optimization.
        config: Simulation and planning configuration (defaults to standard Config).
        service_model: 'v2_event_driven' (dynamic load sharing) or
                       'v1_admission_fixed' (ablation).

    Returns:
        Dict with 'outcomes', 'decisions', 'training_records', and 'metrics'.
    """
    cfg = config or Config()
    if cfg.offered_end_s is None:
        cfg = replace(cfg, offered_end_s=duration_s)
    if callable(policy):
        original_policy = policy

        def measured_policy(state):
            started = time.perf_counter()
            result = original_policy(state)
            elapsed = time.perf_counter() - started
            if isinstance(result, Choice):
                if result.status == "ok" and result.budget not in cfg.actions:
                    raise ValueError("unsupported_policy_budget")
                return replace(result, elapsed_s=max(result.elapsed_s, elapsed))
            if type(result) not in (int, float) or result not in cfg.actions:
                raise ValueError("invalid_policy_result")
            return Choice(float(result), (float(result),), (), "ok", elapsed)

        policy = measured_policy
    tape_hash = compute_tape_hash(items)
    truth_tape_hash = compute_truth_tape_hash(items)

    current_budget = cfg.actions[0] if cfg.actions else 16.0
    budget_version = 1
    pending_command: _PendingCommand | None = None

    solver_failures = 0
    control_failures = 0

    first_overhead_s = 0.025
    prefill_s_per_token = 0.000020  # ~50k tok/s
    prefill_contention_rate = 0.020  # +2% per active running request
    base_decode_interval_s = 0.016   # ~62.5 tok/s baseline decode
    decode_contention_rate = 0.0008  # +0.8ms per stock unit in memory bandwidth contention
    base_cleanup_s = 0.020           # HTTP/Gateway terminal cleanup

    arrivals_by_time = sorted(items, key=lambda x: x.arrival_s)
    arrival_idx = 0
    num_items = len(arrivals_by_time)

    slot_s = cfg.slot_s
    next_control_tick = 0.0
    sim_time = 0.0
    sim_end_time = duration_s + cfg.tail_s

    past_arrivals: list[Arrival] = []
    outcomes: list[RequestOutcome] = []
    decisions: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []

    if service_model == "v1_admission_fixed":
        # -------------------------------------------------------------------
        # V1 Admission-Fixed Service Engine (Ablation)
        # -------------------------------------------------------------------
        @dataclass
        class _V1Execution:
            item: WorkItem
            admitted_at: float
            first_at: float
            last_token_at: float
            terminal_at: float
            owner_release_at: float
            output_tokens: int
            finish_reason: str
            cleanup_s: float
            stock_at_admit: float
            nactive_at_admit: int

            def tokens_at(self, t: float) -> int:
                if t < self.first_at:
                    return 0
                if self.output_tokens <= 1 or t >= self.last_token_at:
                    return self.output_tokens
                fraction = (t - self.first_at) / (self.last_token_at - self.first_at)
                tokens_emitted = 1 + int(fraction * (self.output_tokens - 1))
                return min(self.output_tokens, max(1, tokens_emitted))

        active_v1: list[_V1Execution] = []

        while sim_time <= sim_end_time:
            # 1. Apply pending command if due
            if pending_command is not None and sim_time >= pending_command.applied_at:
                current_budget = pending_command.budget
                budget_version = pending_command.version
                decisions[pending_command.decision_index].update(applied=True, status="applied")
                pending_command = None

            # 2. Retire completed executions
            active_v1 = [r for r in active_v1 if r.owner_release_at > sim_time]
            current_stock = math.fsum(r.item.cost for r in active_v1)

            # 3. Control tick
            if sim_time >= next_control_tick and sim_time <= duration_s:
                public_requests = tuple(
                    PublicRequest(
                        request_id=r.item.request_id,
                        prompt_tokens=r.item.prompt_tokens,
                        max_tokens=r.item.max_tokens,
                        cost=r.item.cost,
                        scheduled_at=r.item.arrival_s,
                        admitted_at=r.admitted_at,
                        first_at=r.first_at if sim_time >= r.first_at else None,
                        observed_output_tokens=r.tokens_at(sim_time),
                        terminal_at=r.terminal_at if sim_time >= r.terminal_at else None,
                        finish_reason=r.finish_reason if sim_time >= r.terminal_at else None,
                        output_sampled_at=sim_time,
                    )
                    for r in active_v1
                )
                state = PublicState(
                    now=sim_time,
                    budget=current_budget,
                    version=budget_version,
                    requests=public_requests,
                    arrivals=tuple(past_arrivals),
                    run_id="local_sim",
                    clock_domain="simulation",
                    complete=True,
                    sampled_at=sim_time,
                )

                chosen_budget = current_budget
                plan_tuple: tuple[float, ...] | None = None
                choice_scores: tuple[Any, ...] = ()
                planner_status = "not_invoked"
                status = "ok"

                dispatch_delay = solver_delay_s
                if solver_delay_s > cfg.solver_budget_s:
                    solver_failures += 1
                    status = "solver_budget_exceeded_fallback"
                    chosen_budget = current_budget
                    dispatch_delay = solver_delay_s
                elif callable(policy):
                    try:
                        res = policy(state)
                        if isinstance(res, Choice):
                            planner_status = res.status
                            plan_tuple = res.plan
                            choice_scores = res.scores
                            status = res.status
                            dispatch_delay = max(dispatch_delay, res.elapsed_s)
                            if res.elapsed_s > cfg.solver_budget_s:
                                solver_failures += 1
                                status = "solver_budget_exceeded_fallback"
                                chosen_budget = current_budget
                            elif res.status == "ok":
                                chosen_budget = res.budget
                            else:
                                solver_failures += 1
                                chosen_budget = current_budget
                        elif isinstance(res, (int, float)):
                            chosen_budget = float(res)
                    except Exception as exc:
                        solver_failures += 1
                        chosen_budget = current_budget
                        status = f"policy_error_{type(exc).__name__}"
                elif isinstance(policy, (int, float)):
                    chosen_budget = float(policy)

                send_time = sim_time + dispatch_delay
                expires_at = send_time + 0.100  # 100ms TTL
                applied_at = send_time + control_delay_s

                # Strict inequality: applied_at < expires_at
                is_expired = applied_at >= expires_at
                if status != "ok":
                    decisions.append({
                        "time": sim_time, "budget": chosen_budget, "applied": False,
                        "version": budget_version, "status": status,
                        "plan": plan_tuple, "current_stock": current_stock,
                    })
                elif is_expired:
                    control_failures += 1
                    decisions.append({
                        "time": round(sim_time, 4),
                        "send_time": round(send_time, 4),
                        "expires_at": round(expires_at, 4),
                        "applied_at": round(applied_at, 4),
                        "budget": chosen_budget,
                        "applied": False,
                        "version": budget_version,
                        "current_stock": round(current_stock, 4),
                        "nactive": len(active_v1),
                        "plan": plan_tuple,
                        "status": "expired_unapplied",
                    })
                else:
                    if applied_at <= sim_time:
                        current_budget = chosen_budget
                        budget_version += 1
                        decisions.append({
                            "time": round(sim_time, 4),
                            "send_time": round(send_time, 4),
                            "expires_at": round(expires_at, 4),
                            "applied_at": round(applied_at, 4),
                            "budget": chosen_budget,
                            "applied": True,
                            "version": budget_version,
                            "current_stock": round(current_stock, 4),
                            "nactive": len(active_v1),
                            "plan": plan_tuple,
                            "status": status,
                        })
                    else:
                        pending_command = _PendingCommand(
                            send_time=send_time,
                            expires_at=expires_at,
                            applied_at=applied_at,
                            budget=chosen_budget,
                            version=budget_version + 1,
                            decision_index=len(decisions),
                        )
                        decisions.append({
                            "time": round(sim_time, 4),
                            "send_time": round(send_time, 4),
                            "expires_at": round(expires_at, 4),
                            "applied_at": round(applied_at, 4),
                            "budget": chosen_budget,
                            "applied": False,
                            "version": budget_version + 1,
                            "current_stock": round(current_stock, 4),
                            "nactive": len(active_v1),
                            "plan": plan_tuple,
                            "status": "pending",
                        })

                decisions[-1].update(planner_status=planner_status, scores=choice_scores,
                                     solver_elapsed_s=dispatch_delay)
                next_control_tick += slot_s

            # 4. Process arrivals
            while arrival_idx < num_items and arrivals_by_time[arrival_idx].arrival_s <= sim_time:
                item = arrivals_by_time[arrival_idx]
                arrival_idx += 1

                past_arrivals.append(
                    Arrival(
                        at=item.arrival_s,
                        prompt_tokens=item.prompt_tokens,
                        max_tokens=item.max_tokens,
                    )
                )

                stock_at_admit = current_stock
                nactive_at_admit = len(active_v1)

                if current_stock + item.cost <= current_budget:
                    admitted_at = item.arrival_s
                    prefill_scale = 1.0 + prefill_contention_rate * nactive_at_admit
                    prefill_duration = first_overhead_s + (item.prompt_tokens * prefill_s_per_token) * prefill_scale
                    first_at = admitted_at + prefill_duration
                    ttft_s = first_at - item.arrival_s

                    decode_step = base_decode_interval_s + decode_contention_rate * (current_stock + item.cost)
                    num_decode_steps = max(0, item.true_output_tokens - 1)
                    decode_duration = num_decode_steps * decode_step
                    last_token_at = first_at + decode_duration

                    cleanup_s = base_cleanup_s
                    terminal_at = last_token_at + cleanup_s

                    finish_reason = item.finish_reason
                    if terminal_at > sim_end_time:
                        fate = "unknown"
                    elif (terminal_at - item.arrival_s) > cfg.http_deadline_s:
                        fate = "timeout"
                    elif finish_reason == "length":
                        fate = "capped"
                    else:
                        fate = "complete"

                    output_min_met = (item.true_output_tokens > 1)
                    ttft_slo_met = (ttft_s <= cfg.slo_ttft_s)
                    token_interval_s = (terminal_at - first_at) / (item.true_output_tokens - 1) if output_min_met else None
                    tpot_slo_met = (token_interval_s is not None and token_interval_s <= cfg.slo_tpot_s)
                    deadline_met = ((terminal_at - item.arrival_s) <= cfg.http_deadline_s)

                    cohort_good = (
                        fate == "complete"
                        and output_min_met
                        and ttft_slo_met
                        and tpot_slo_met
                        and deadline_met
                    )
                    window_good = cohort_good and (terminal_at <= duration_s)

                    execution = _V1Execution(
                        item=item,
                        admitted_at=admitted_at,
                        first_at=first_at,
                        last_token_at=last_token_at,
                        terminal_at=terminal_at,
                        owner_release_at=max(last_token_at, terminal_at + extra_owner_release_s),
                        output_tokens=item.true_output_tokens,
                        finish_reason=finish_reason,
                        cleanup_s=cleanup_s,
                        stock_at_admit=stock_at_admit,
                        nactive_at_admit=nactive_at_admit,
                    )
                    active_v1.append(execution)
                    current_stock += item.cost

                    api_terminal_overhead_s = base_cleanup_s
                    owner_release_at = max(last_token_at, terminal_at + extra_owner_release_s)
                    owner_release_extra_s = owner_release_at - terminal_at
                    cleanup_s = max(0.0, owner_release_at - last_token_at)

                    training_records.append({
                        "seed": item.seed,
                        "prompt_tokens": item.prompt_tokens,
                        "max_tokens": item.max_tokens,
                        "output_tokens": item.true_output_tokens,
                        "natural_output_tokens": item.natural_output_tokens or item.true_output_tokens,
                        "finish_reason": finish_reason,
                        "stock_at_admission": round(stock_at_admit, 4),
                        "nactive_at_admission": nactive_at_admit,
                        "ttft_s": round(ttft_s, 6),
                        "token_interval_s": round(token_interval_s, 6) if token_interval_s is not None else None,
                        "decode_step_s": round(decode_step, 6) if item.true_output_tokens > 1 else None,
                        "cleanup_s": round(cleanup_s, 6),
                        "api_terminal_overhead_s": round(api_terminal_overhead_s, 6),
                        "api_terminal_s": round(api_terminal_overhead_s, 6),
                        "owner_release_extra_s": round(owner_release_extra_s, 6),
                        "owner_release_s": round(owner_release_extra_s, 6),
                        "decode_load_stock": round(current_stock, 4),
                        "decode_load_nactive": nactive_at_admit + 1,
                    })

                    outcomes.append(
                        RequestOutcome(
                            request_id=item.request_id,
                            arrival_s=item.arrival_s,
                            prompt_tokens=item.prompt_tokens,
                            max_tokens=item.max_tokens,
                            cost=item.cost,
                            admitted=True,
                            rejection_reason=None,
                            admitted_at=admitted_at,
                            first_at=first_at,
                            last_token_at=last_token_at,
                            terminal_at=terminal_at,
                            owner_release_at=owner_release_at,
                            owner_release_s=round(owner_release_extra_s, 6),
                            owner_release_extra_s=round(owner_release_extra_s, 6),
                            api_terminal_overhead_s=round(api_terminal_overhead_s, 6),
                            output_tokens=item.true_output_tokens,
                            natural_output_tokens=item.natural_output_tokens or item.true_output_tokens,
                            finish_reason=finish_reason,
                            fate=fate,
                            ttft_s=ttft_s,
                            token_interval_s=token_interval_s,
                            decode_step_s=decode_step,
                            cleanup_s=cleanup_s,
                            ttft_slo_met=ttft_slo_met,
                            tpot_slo_met=tpot_slo_met,
                            output_min_met=output_min_met,
                            deadline_met=deadline_met,
                            cohort_good=cohort_good,
                            window_good=window_good,
                        )
                    )
                else:
                    outcomes.append(
                        RequestOutcome(
                            request_id=item.request_id,
                            arrival_s=item.arrival_s,
                            prompt_tokens=item.prompt_tokens,
                            max_tokens=item.max_tokens,
                            cost=item.cost,
                            admitted=False,
                            natural_output_tokens=item.natural_output_tokens or item.true_output_tokens,
                            finish_reason=item.finish_reason,
                            fate="rejected",
                            rejection_reason="capacity_exceeded_503",
                        )
                    )

            # Advance simulation time
            next_event_time = sim_end_time + 1.0
            if arrival_idx < num_items:
                next_event_time = min(next_event_time, arrivals_by_time[arrival_idx].arrival_s)
            if next_control_tick <= duration_s:
                next_event_time = min(next_event_time, next_control_tick)
            if pending_command is not None and pending_command.applied_at > sim_time:
                next_event_time = min(next_event_time, pending_command.applied_at)
            for req in active_v1:
                if req.owner_release_at > sim_time:
                    next_event_time = min(next_event_time, req.owner_release_at)
                if req.terminal_at > sim_time:
                    next_event_time = min(next_event_time, req.terminal_at)

            sim_time = sim_time + 0.001 if next_event_time <= sim_time else next_event_time

    else:
        # -------------------------------------------------------------------
        # V2 Dynamic Event-Driven Service Engine (Default)
        # -------------------------------------------------------------------
        active_states: list[_V2RequestState] = []
        completed_states: list[_V2RequestState] = []
        prefill_queue: list[_V2RequestState] = []
        prefill_engine_free_at = 0.0

        while sim_time <= sim_end_time:
            # 1. Apply pending budget command if applied_at <= sim_time
            if pending_command is not None and sim_time >= pending_command.applied_at:
                current_budget = pending_command.budget
                budget_version = pending_command.version
                decisions[pending_command.decision_index].update(applied=True, status="applied")
                pending_command = None

            retained_active = []
            for r in active_states:
                if r.phase == "prefill" and r.prefill_done_at is not None and r.prefill_done_at <= sim_time:
                    r.phase = "decode"
                    if r.tokens_needed == 0:
                        r.last_token_at = r.prefill_done_at
                        r.terminal_at = r.last_token_at + base_cleanup_s
                        r.owner_release_at = max(r.last_token_at, r.terminal_at + extra_owner_release_s)
                        r.phase = "cleanup"
                if r.owner_release_at is not None and r.owner_release_at <= sim_time:
                    r.phase = "complete"
                    completed_states.append(r)
                else:
                    retained_active.append(r)
            active_states = retained_active
            current_stock = math.fsum(r.item.cost for r in active_states)

            # 2. Control tick
            if sim_time >= next_control_tick and sim_time <= duration_s:
                public_requests = tuple(
                    PublicRequest(
                        request_id=r.item.request_id,
                        prompt_tokens=r.item.prompt_tokens,
                        max_tokens=r.item.max_tokens,
                        cost=r.item.cost,
                        scheduled_at=r.item.arrival_s,
                        admitted_at=r.admitted_at,
                        first_at=r.first_at if (r.first_at is not None and sim_time >= r.first_at) else None,
                        observed_output_tokens=r.tokens_at(sim_time),
                        terminal_at=r.terminal_at if (r.terminal_at is not None and sim_time >= r.terminal_at) else None,
                        finish_reason=r.finish_reason if (r.terminal_at is not None and sim_time >= r.terminal_at) else None,
                        output_sampled_at=sim_time,
                    )
                    for r in active_states
                )
                state = PublicState(
                    now=sim_time,
                    budget=current_budget,
                    version=budget_version,
                    requests=public_requests,
                    arrivals=tuple(past_arrivals),
                    run_id="local_sim",
                    clock_domain="simulation",
                    complete=True,
                    sampled_at=sim_time,
                )

                chosen_budget = current_budget
                plan_tuple: tuple[float, ...] | None = None
                choice_scores: tuple[Any, ...] = ()
                planner_status = "not_invoked"
                status = "ok"

                dispatch_delay = solver_delay_s
                if solver_delay_s > cfg.solver_budget_s:
                    solver_failures += 1
                    status = "solver_budget_exceeded_fallback"
                    chosen_budget = current_budget
                    dispatch_delay = solver_delay_s
                elif callable(policy):
                    try:
                        res = policy(state)
                        if isinstance(res, Choice):
                            planner_status = res.status
                            plan_tuple = res.plan
                            choice_scores = res.scores
                            status = res.status
                            dispatch_delay = max(dispatch_delay, res.elapsed_s)
                            if res.elapsed_s > cfg.solver_budget_s:
                                solver_failures += 1
                                status = "solver_budget_exceeded_fallback"
                                chosen_budget = current_budget
                            elif res.status == "ok":
                                chosen_budget = res.budget
                            else:
                                solver_failures += 1
                                chosen_budget = current_budget
                        elif isinstance(res, (int, float)):
                            chosen_budget = float(res)
                    except Exception as exc:
                        solver_failures += 1
                        chosen_budget = current_budget
                        status = f"policy_error_{type(exc).__name__}"
                elif isinstance(policy, (int, float)):
                    chosen_budget = float(policy)

                send_time = sim_time + dispatch_delay
                expires_at = send_time + 0.100  # 100ms TTL
                applied_at = send_time + control_delay_s

                # Strict inequality: applied_at < expires_at
                is_expired = applied_at >= expires_at
                if status != "ok":
                    decisions.append({
                        "time": sim_time, "budget": chosen_budget, "applied": False,
                        "version": budget_version, "status": status,
                        "plan": plan_tuple, "current_stock": current_stock,
                    })
                elif is_expired:
                    control_failures += 1
                    decisions.append({
                        "time": round(sim_time, 4),
                        "send_time": round(send_time, 4),
                        "expires_at": round(expires_at, 4),
                        "applied_at": round(applied_at, 4),
                        "budget": chosen_budget,
                        "applied": False,
                        "version": budget_version,
                        "current_stock": round(current_stock, 4),
                        "nactive": len(active_states),
                        "plan": plan_tuple,
                        "status": "expired_unapplied",
                    })
                else:
                    if applied_at <= sim_time:
                        current_budget = chosen_budget
                        budget_version += 1
                        decisions.append({
                            "time": round(sim_time, 4),
                            "send_time": round(send_time, 4),
                            "expires_at": round(expires_at, 4),
                            "applied_at": round(applied_at, 4),
                            "budget": chosen_budget,
                            "applied": True,
                            "version": budget_version,
                            "current_stock": round(current_stock, 4),
                            "nactive": len(active_states),
                            "plan": plan_tuple,
                            "status": status,
                        })
                    else:
                        pending_command = _PendingCommand(
                            send_time=send_time,
                            expires_at=expires_at,
                            applied_at=applied_at,
                            budget=chosen_budget,
                            version=budget_version + 1,
                            decision_index=len(decisions),
                        )
                        decisions.append({
                            "time": round(sim_time, 4),
                            "send_time": round(send_time, 4),
                            "expires_at": round(expires_at, 4),
                            "applied_at": round(applied_at, 4),
                            "budget": chosen_budget,
                            "applied": False,
                            "version": budget_version + 1,
                            "current_stock": round(current_stock, 4),
                            "nactive": len(active_states),
                            "plan": plan_tuple,
                            "status": "pending",
                        })

                decisions[-1].update(planner_status=planner_status, scores=choice_scores,
                                     solver_elapsed_s=dispatch_delay)
                next_control_tick += slot_s

            # 3. Process new arrivals occurring at or before sim_time
            while arrival_idx < num_items and arrivals_by_time[arrival_idx].arrival_s <= sim_time:
                item = arrivals_by_time[arrival_idx]
                arrival_idx += 1

                past_arrivals.append(
                    Arrival(
                        at=item.arrival_s,
                        prompt_tokens=item.prompt_tokens,
                        max_tokens=item.max_tokens,
                    )
                )

                stock_at_admit = current_stock
                nactive_at_admit = len(active_states)

                if current_stock + item.cost <= current_budget:
                    tokens_needed = max(0, item.true_output_tokens - 1)
                    req_state = _V2RequestState(
                        item=item,
                        admitted_at=item.arrival_s,
                        prompt_tokens=item.prompt_tokens,
                        output_tokens=item.true_output_tokens,
                        finish_reason=item.finish_reason,
                        tokens_needed=tokens_needed,
                        tokens_remaining=float(tokens_needed),
                        stock_at_admit=stock_at_admit,
                        nactive_at_admit=nactive_at_admit,
                        phase="prefill",
                    )
                    active_states.append(req_state)
                    prefill_queue.append(req_state)
                    current_stock += item.cost
                else:
                    outcomes.append(
                        RequestOutcome(
                            request_id=item.request_id,
                            arrival_s=item.arrival_s,
                            prompt_tokens=item.prompt_tokens,
                            max_tokens=item.max_tokens,
                            cost=item.cost,
                            admitted=False,
                            natural_output_tokens=item.natural_output_tokens or item.true_output_tokens,
                            finish_reason=item.finish_reason,
                            fate="rejected",
                            rejection_reason="capacity_exceeded_503",
                        )
                    )

            # 4. Schedule prefill executions FIFO
            while prefill_queue and prefill_queue[0].prefill_done_at is None:
                next_req = prefill_queue[0]
                start_p = max(sim_time, prefill_engine_free_at, next_req.admitted_at)
                n_decoding = sum(1 for r in active_states if r.phase == "decode")
                prefill_scale = 1.0 + prefill_contention_rate * n_decoding
                duration_p = first_overhead_s + (next_req.prompt_tokens * prefill_s_per_token) * prefill_scale
                next_req.first_at = start_p + duration_p
                next_req.prefill_done_at = next_req.first_at
                prefill_engine_free_at = next_req.first_at
                prefill_queue.pop(0)

            # 5. Check if any prefill completes at sim_time
            for r in active_states:
                if r.phase == "prefill" and r.prefill_done_at is not None and r.prefill_done_at <= sim_time:
                    r.phase = "decode"
                    if r.tokens_needed == 0:
                        r.last_token_at = sim_time
                        r.terminal_at = sim_time + base_cleanup_s
                        r.owner_release_at = max(r.last_token_at, r.terminal_at + extra_owner_release_s)
                        r.phase = "cleanup"

            # 6. Check if any cleanup completes at sim_time
            retained_active: list[_V2RequestState] = []
            for r in active_states:
                if r.owner_release_at is not None and r.owner_release_at <= sim_time:
                    r.phase = "complete"
                    completed_states.append(r)
                else:
                    retained_active.append(r)
            active_states = retained_active
            current_stock = math.fsum(r.item.cost for r in active_states)

            # 7. Compute next event time and advance continuous decode progress
            next_event_time = sim_end_time + 1.0

            if arrival_idx < num_items:
                next_event_time = min(next_event_time, arrivals_by_time[arrival_idx].arrival_s)
            if next_control_tick <= duration_s:
                next_event_time = min(next_event_time, next_control_tick)
            if pending_command is not None and pending_command.applied_at > sim_time:
                next_event_time = min(next_event_time, pending_command.applied_at)

            for r in active_states:
                if r.phase == "prefill" and r.prefill_done_at is not None and r.prefill_done_at > sim_time:
                    next_event_time = min(next_event_time, r.prefill_done_at)

            for r in active_states:
                if r.phase == "cleanup" and r.terminal_at is not None and r.terminal_at > sim_time:
                    next_event_time = min(next_event_time, r.terminal_at)
                if r.owner_release_at is not None and r.owner_release_at > sim_time:
                    next_event_time = min(next_event_time, r.owner_release_at)

            decoding_reqs = [r for r in active_states if r.phase == "decode"]
            if decoding_reqs:
                inst_step = base_decode_interval_s + decode_contention_rate * current_stock
                for r in decoding_reqs:
                    if r.tokens_remaining > 0:
                        finish_in_s = r.tokens_remaining * inst_step
                        next_event_time = min(next_event_time, sim_time + finish_in_s)

            if next_event_time <= sim_time:
                next_event_time = sim_time + 0.0001

            dt = next_event_time - sim_time
            if decoding_reqs and dt > 0.0:
                inst_step = base_decode_interval_s + decode_contention_rate * current_stock
                tokens_per_s = 1.0 / inst_step
                tokens_consumed = dt * tokens_per_s
                n_active_now = len(active_states)

                for r in decoding_reqs:
                    emitted = min(r.tokens_remaining, tokens_consumed)
                    r.accum_stock_tokens += emitted * current_stock
                    r.accum_nactive_tokens += emitted * n_active_now
                    r.accum_decode_tokens += emitted

                    if r.tokens_remaining <= tokens_consumed + 1e-9:
                        r.tokens_remaining = 0.0
                        r.last_token_at = next_event_time
                        r.terminal_at = next_event_time + base_cleanup_s
                        r.owner_release_at = max(r.last_token_at, r.terminal_at + extra_owner_release_s)
                        r.phase = "cleanup"
                    else:
                        r.tokens_remaining -= tokens_consumed

            sim_time = next_event_time

        all_admitted = completed_states + active_states
        for r in all_admitted:
            item = r.item
            first_at = r.first_at or r.admitted_at
            ttft_s = first_at - item.arrival_s
            last_token_at = r.last_token_at or sim_time
            terminal_at = r.terminal_at or (last_token_at + base_cleanup_s)
            cleanup_s = base_cleanup_s
            finish_reason = r.finish_reason

            if terminal_at > sim_end_time:
                fate = "unknown"
            elif (terminal_at - item.arrival_s) > cfg.http_deadline_s:
                fate = "timeout"
            elif finish_reason == "length":
                fate = "capped"
            else:
                fate = "complete"

            output_min_met = (item.true_output_tokens > 1)
            ttft_slo_met = (ttft_s <= cfg.slo_ttft_s)
            token_interval_s = (terminal_at - first_at) / (item.true_output_tokens - 1) if output_min_met else None
            decode_step_s = (last_token_at - first_at) / (item.true_output_tokens - 1) if output_min_met else None
            tpot_slo_met = (token_interval_s is not None and token_interval_s <= cfg.slo_tpot_s)
            deadline_met = ((terminal_at - item.arrival_s) <= cfg.http_deadline_s)

            cohort_good = (
                fate == "complete"
                and output_min_met
                and ttft_slo_met
                and tpot_slo_met
                and deadline_met
            )
            window_good = cohort_good and (terminal_at <= duration_s)

            avg_stock = r.accum_stock_tokens / r.accum_decode_tokens if r.accum_decode_tokens else 0.0
            avg_nactive = r.accum_nactive_tokens / r.accum_decode_tokens if r.accum_decode_tokens else 0.0

            api_terminal_overhead_s = base_cleanup_s
            owner_release_at = max(last_token_at, terminal_at + extra_owner_release_s)
            owner_release_extra_s = owner_release_at - terminal_at
            cleanup_s = max(0.0, owner_release_at - last_token_at)

            training_records.append({
                "seed": item.seed,
                "prompt_tokens": item.prompt_tokens,
                "max_tokens": item.max_tokens,
                "output_tokens": item.true_output_tokens,
                "natural_output_tokens": item.natural_output_tokens or item.true_output_tokens,
                "finish_reason": finish_reason,
                "stock_at_admission": round(r.stock_at_admit, 4),
                "nactive_at_admission": r.nactive_at_admit,
                "ttft_s": round(ttft_s, 6),
                "token_interval_s": round(token_interval_s, 6) if token_interval_s is not None else None,
                "decode_step_s": round(decode_step_s, 6) if decode_step_s is not None else None,
                "cleanup_s": round(cleanup_s, 6),
                "api_terminal_overhead_s": round(api_terminal_overhead_s, 6),
                "api_terminal_s": round(api_terminal_overhead_s, 6),
                "owner_release_extra_s": round(owner_release_extra_s, 6),
                "owner_release_s": round(owner_release_extra_s, 6),
                # System-level instantaneous load experienced during decode (inclusive of own request)
                "decode_load_stock": avg_stock,
                "decode_load_nactive": avg_nactive,
            })

            outcomes.append(
                RequestOutcome(
                    request_id=item.request_id,
                    arrival_s=item.arrival_s,
                    prompt_tokens=item.prompt_tokens,
                    max_tokens=item.max_tokens,
                    cost=item.cost,
                    admitted=True,
                    rejection_reason=None,
                    admitted_at=r.admitted_at,
                    first_at=first_at,
                    last_token_at=last_token_at,
                    terminal_at=terminal_at,
                    owner_release_at=owner_release_at,
                    owner_release_s=round(owner_release_extra_s, 6),
                    owner_release_extra_s=round(owner_release_extra_s, 6),
                    api_terminal_overhead_s=round(api_terminal_overhead_s, 6),
                    output_tokens=item.true_output_tokens,
                    natural_output_tokens=item.natural_output_tokens or item.true_output_tokens,
                    finish_reason=finish_reason,
                    fate=fate,
                    ttft_s=ttft_s,
                    token_interval_s=token_interval_s,
                    decode_step_s=decode_step_s,
                    cleanup_s=cleanup_s,
                    ttft_slo_met=ttft_slo_met,
                    tpot_slo_met=tpot_slo_met,
                    output_min_met=output_min_met,
                    deadline_met=deadline_met,
                    cohort_good=cohort_good,
                    window_good=window_good,
                )
            )

    for decision in decisions:
        if decision["status"] == "pending":
            decision["status"] = "pending_unapplied_end"
    outcomes.sort(key=lambda x: x.arrival_s)

    all_assigned = len(outcomes)
    admitted_outcomes = [o for o in outcomes if o.admitted]
    rejected_outcomes = [o for o in outcomes if not o.admitted]
    cohort_good_outcomes = [o for o in outcomes if o.cohort_good]
    window_good_outcomes = [o for o in outcomes if o.window_good]
    timeouts = [o for o in outcomes if o.admitted and o.fate == "timeout"]
    capped = [o for o in outcomes if o.admitted and o.fate == "capped"]
    unknown = [o for o in outcomes if o.admitted and o.fate == "unknown"]

    ttft_values = [o.ttft_s for o in admitted_outcomes if o.ttft_s is not None]
    tpot_values = [o.token_interval_s for o in admitted_outcomes if o.token_interval_s is not None]

    mean_ttft = float(sum(ttft_values) / len(ttft_values)) if ttft_values else 0.0
    mean_tpot = float(sum(tpot_values) / len(tpot_values)) if tpot_values else 0.0

    p95_ttft = float(sorted(ttft_values)[int(0.95 * len(ttft_values))]) if ttft_values else 0.0
    p95_tpot = float(sorted(tpot_values)[int(0.95 * len(tpot_values))]) if tpot_values else 0.0

    completed_outcomes = [o for o in admitted_outcomes if o.fate in ("complete", "capped")]
    drain_tail_s = 0.0
    if completed_outcomes:
        last_term = max(o.terminal_at for o in completed_outcomes if o.terminal_at is not None)
        drain_tail_s = max(0.0, last_term - duration_s)

    metrics = {
        "arrival_tape_hash": tape_hash,
        "truth_tape_hash": truth_tape_hash,
        "all_assigned": all_assigned,
        "admitted": len(admitted_outcomes),
        "rejected": len(rejected_outcomes),
        "completed": len(completed_outcomes),
        "timeouts": len(timeouts),
        "capped": len(capped),
        "unknown": len(unknown),
        "cohort_good": len(cohort_good_outcomes),
        "window_good": len(window_good_outcomes),
        # Goodput units: SLO-good requests per offered second (req/s)
        "cohort_goodput_req_s": round(len(cohort_good_outcomes) / duration_s, 4),
        "window_goodput_req_s": round(len(window_good_outcomes) / duration_s, 4),
        "all_assigned_good_fraction": round(len(cohort_good_outcomes) / all_assigned, 4) if all_assigned else 0.0,
        "goodput_per_offered_s": round(len(cohort_good_outcomes) / duration_s, 4),
        "window_goodput_per_offered_s": round(len(window_good_outcomes) / duration_s, 4),
        "mean_ttft_s": round(mean_ttft, 6),
        "p95_ttft_s": round(p95_ttft, 6),
        "mean_tpot_s": round(mean_tpot, 6),
        "p95_tpot_s": round(p95_tpot, 6),
        "drain_tail_s": round(drain_tail_s, 4),
        "solver_failures": solver_failures,
        "control_failures": control_failures,
        "service_model": service_model,
    }

    return {
        "outcomes": outcomes,
        "decisions": decisions,
        "training_records": training_records,
        "metrics": metrics,
        "config": cfg,
    }
