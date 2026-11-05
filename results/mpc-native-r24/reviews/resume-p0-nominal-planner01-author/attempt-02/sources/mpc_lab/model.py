"""Compact event/lifecycle CPU development surrogate for admission planning.

This is NOT a vLLM scheduler replica, an identified GPU service law or native
qualification. Its service coefficients are synthetic until intervention
fitting. It exists so planners and baselines share one coherent plant:

* exact gateway reservation accounting (admit iff ``R + c <= budget``, the
  whole cost is held until the terminal event and released exactly once,
  a budget decrease never evicts or clips an admitted reservation);
* one request lifecycle generating every flow: FIFO prefill phase ->
  first content -> decode phase -> terminal (success or failure);
* explicit latent remaining work per request, advanced at a shared,
  piecewise-constant service slowdown, so first-content, decode progress and
  reservation release are coupled rather than independent regressions.

Service law (all work in nominal seconds, divided by the current slowdown):

    slowdown(t) = service_scale * (1 + decode_interference * D(t))
    prefill work of request i = first_overhead_s + prefill_s_per_token * p_i
    decode work of request i  = decode_s_per_token * (output_i - 1)

Prefill is a single FIFO server (head-of-line only); decoders progress in
parallel. ``D(t)`` is the number of requests in the decode phase.

Time conventions (all absolute values in ``snapshot.clock_domain``):

* gateway receipt of a future offer = sampled_at + offset_s + ingress_delay_s;
* client-scheduled time of a future offer = sampled_at + offset_s;
* ``Outcome.first_at`` is the *client-observed* first content, i.e. gateway
  first content plus the egress delay, and ``Outcome.terminal_at`` is the
  *client-observed* terminal, i.e. gateway terminal plus the same egress delay
  (so ``first_at <= terminal_at`` always). The reservation is released at the
  *gateway* terminal (``reserved_trace``), not at the client terminal. A client
  event later than the rollout cut is not reported: ``first_at``/
  ``terminal_at`` are ``None`` and ``success`` stays unresolved (``None``) even
  if the gateway already finished. ``Outcome.good`` therefore compares client
  time to scheduled time, matching the scored label.
* an old request without ``scheduled_at`` is assigned
  ``received_at - scenario.ingress_delay_s``; this is an explicit scenario
  assumption, not a measurement.

Old-request latent work (declared belief, not a measurement):

* awaiting old requests are assumed to have made no prefill progress
  (pessimistic member of the aliased set), in admitted_at FIFO order;
* old streaming requests have total output
  ``max(1, ceil(old_output_fraction * max_tokens))``; tokens already emitted are
  estimated from elapsed time since ``first_at`` at the nominal undisturbed
  decode rate; at least one token always remains;
* all old requests share ``scenario.old_success``.

Sealed-ledger consistency: every old ``RequestMark.cost`` must equal
``parameters.cost_spec.estimate(prompt_tokens, max_tokens)`` (relative
tolerance 1e-12) because ``identity.cost_hash`` claims that formula; otherwise
``RolloutError("ledger_cost_mismatch")``.

Future offer failure semantics: ``success=False`` with ``output_tokens=k``
fails after emitting ``k`` tokens; ``k == 0`` fails at prefill completion
without first content.

Scoring cohort (each request scored exactly once): every old (snapshot)
request plus every future offer with ``0 <= offset_s < evaluation_window_s``.
Offers with ``evaluation_window_s <= offset_s < evaluation_window_s + tail_s``
are simulated and contend for reservation/service, but are not rewarded and
are not returned in ``outcomes``. The rollout ends at
``sampled_at + evaluation_window_s + tail_s``.

``score_lower`` counts resolved good outcomes. ``score_upper`` additionally
counts unresolved cohort members that could still be good under the scenario
(scenario success flag true and the on-time first-content condition not yet
violated). Both are counts of successful requests.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field

from .contracts import (
    FutureOffer,
    ModelParameters,
    Outcome,
    Plan,
    PlanningConfig,
    RolloutError,
    RolloutResult,
    Scenario,
    Snapshot,
    finite,
)

__all__ = ["rollout", "generate_scenarios"]

_EPS = 1e-12
_DEADLINE_CHECK_EVERY = 64
_LEDGER_REL_TOL = 1e-12
# generate_scenarios caps, tied to the documented PlanningConfig defaults.
MAX_GENERATED_SCENARIOS = PlanningConfig().max_scenarios
MAX_GENERATED_OFFERS = PlanningConfig().max_events  # total across all scenarios


@dataclass
class _Req:
    rid: str
    cost: float
    prompt: int
    max_tokens: int
    scheduled: float
    egress: float
    success: bool
    scored: bool
    output: int | None = None  # future offers only; None for old
    arrival: float | None = None  # gateway receipt (future offers)
    accepted: bool = False
    phase: str = "pending"  # pending|prefill|decode|done|rejected
    work: float = 0.0  # remaining nominal seconds of current phase
    decode_work: float = 0.0  # nominal decode seconds after first content
    emits_first: bool = True
    first_client: float | None = None
    terminal: float | None = None
    final_success: bool | None = None
    released: bool = False


@dataclass
class _Sim:
    now: float
    reserved: float
    trace: list = field(default_factory=list)
    events: int = 0


def _fail(reason: str) -> None:
    raise RolloutError(reason)


def _validate_parameters(p: ModelParameters) -> None:
    if not finite(p.prefill_s_per_token) or p.prefill_s_per_token < 0:
        _fail("invalid_prefill_s_per_token")
    if not finite(p.decode_s_per_token, positive=True):
        _fail("invalid_decode_s_per_token")
    if not finite(p.first_overhead_s, positive=True):
        _fail("invalid_first_overhead_s")  # guarantees positive prefill work
    if not finite(p.decode_interference) or p.decode_interference < 0:
        _fail("invalid_decode_interference")
    if not p.supported_budgets or any(not finite(b, positive=True) for b in p.supported_budgets):
        _fail("invalid_supported_budgets")
    for name in ("max_prompt_tokens", "max_output_tokens"):
        v = getattr(p, name)
        if type(v) is not int or v <= 0:
            _fail(f"invalid_{name}")


def _validate_config(c: PlanningConfig) -> None:
    for name in ("slot_s", "evaluation_window_s", "slo_s"):
        if not finite(getattr(c, name), positive=True):
            _fail(f"invalid_{name}")
    if not finite(c.tail_s) or c.tail_s < 0:
        _fail("invalid_tail_s")
    for name in ("max_events", "max_active_requests"):
        v = getattr(c, name)
        if type(v) is not int or v <= 0:
            _fail(f"invalid_{name}")


def _validate_plan(plan: Plan, p: ModelParameters, c: PlanningConfig) -> None:
    if not finite(plan.slot_s, positive=True):
        _fail("invalid_plan_slot")
    budgets = tuple(plan.actions) + (plan.continuation_budget,)
    if not plan.actions:
        _fail("empty_plan")
    for b in budgets:
        if not finite(b, positive=True):
            _fail("nonfinite_plan_budget")
        if b not in p.supported_budgets:
            _fail("unsupported_plan_budget")


def _validate_snapshot(s: Snapshot, p: ModelParameters, c: PlanningConfig) -> None:
    if not s.complete:
        _fail("incomplete_snapshot")
    if not finite(s.sampled_at) or not finite(s.budget, positive=True):
        _fail("invalid_snapshot_values")
    if s.identity.cost_hash != p.cost_spec.fingerprint:
        _fail("cost_identity_mismatch")
    if len(s.requests) > c.max_active_requests:
        _fail("active_limit")
    seen = set()
    for r in s.requests:
        if r.request_id in seen:
            _fail("duplicate_request_id")
        seen.add(r.request_id)
        if not finite(r.cost, positive=True) or r.cost < p.cost_spec.minimum:
            _fail("invalid_request_cost")
        if type(r.prompt_tokens) is not int or not 0 <= r.prompt_tokens <= p.max_prompt_tokens:
            _fail("unsupported_prompt_tokens")
        if type(r.max_tokens) is not int or not 1 <= r.max_tokens <= p.max_output_tokens:
            _fail("unsupported_max_tokens")
        times = (r.received_at, r.admitted_at)
        if any(not finite(t) for t in times) or not r.received_at <= r.admitted_at <= s.sampled_at:
            _fail("noncausal_request_times")
        if r.first_at is not None and (not finite(r.first_at) or not r.admitted_at <= r.first_at <= s.sampled_at):
            _fail("noncausal_first_at")
        if r.scheduled_at is not None and (not finite(r.scheduled_at) or r.scheduled_at > r.received_at):
            _fail("noncausal_scheduled_at")
        expected = p.cost_spec.estimate(r.prompt_tokens, r.max_tokens)
        if not math.isclose(r.cost, expected, rel_tol=_LEDGER_REL_TOL, abs_tol=0.0):
            _fail("ledger_cost_mismatch")


def _validate_scenario(sc: Scenario, s: Snapshot, p: ModelParameters) -> None:
    if not finite(sc.service_scale, positive=True):
        _fail("invalid_service_scale")
    if not finite(sc.old_output_fraction, positive=True) or sc.old_output_fraction > 1:
        _fail("invalid_old_output_fraction")
    if type(sc.old_success) is not bool:
        _fail("invalid_old_success")
    for d in (sc.ingress_delay_s, sc.egress_delay_s):
        if not finite(d) or d < 0:
            _fail("invalid_transport_delay")
    ids = {r.request_id for r in s.requests}
    last = 0.0
    for o in sc.offers:
        if not isinstance(o, FutureOffer):
            _fail("invalid_offer")
        if o.request_id in ids:
            _fail("duplicate_request_id")
        ids.add(o.request_id)
        if not finite(o.offset_s) or o.offset_s < 0:
            _fail("invalid_offer_offset")
        if o.offset_s < last:
            _fail("unsorted_offers")
        last = o.offset_s
        if type(o.prompt_tokens) is not int or not 0 <= o.prompt_tokens <= p.max_prompt_tokens:
            _fail("unsupported_prompt_tokens")
        if type(o.max_tokens) is not int or not 1 <= o.max_tokens <= p.max_output_tokens:
            _fail("unsupported_max_tokens")
        if type(o.output_tokens) is not int or not 0 <= o.output_tokens <= o.max_tokens:
            _fail("output_exceeds_declared_max")
        if type(o.success) is not bool or (o.success and o.output_tokens < 1):
            _fail("invalid_offer_success")
        for d in (o.ingress_delay_s, o.egress_delay_s):
            if not finite(d) or d < 0:
                _fail("invalid_transport_delay")


def _build(snapshot, scenario, params, config):
    t0 = snapshot.sampled_at
    end = t0 + config.evaluation_window_s + config.tail_s
    old_fifo, decoders, pending = [], [], []
    nominal = params.decode_s_per_token * scenario.service_scale
    for r in sorted(snapshot.requests, key=lambda r: (r.admitted_at, r.request_id)):
        sched = r.scheduled_at if r.scheduled_at is not None else r.received_at - scenario.ingress_delay_s
        total = max(1, math.ceil(scenario.old_output_fraction * r.max_tokens))
        q = _Req(r.request_id, r.cost, r.prompt_tokens, r.max_tokens, sched,
                 scenario.egress_delay_s, scenario.old_success, True, accepted=True)
        if r.first_at is None:
            q.phase = "prefill"
            q.work = params.first_overhead_s + params.prefill_s_per_token * r.prompt_tokens
            q.decode_work = params.decode_s_per_token * (total - 1)
            old_fifo.append(q)
        else:
            q.first_client = r.first_at + scenario.egress_delay_s
            emitted = 1 + int((t0 - r.first_at) / nominal)
            q.phase = "decode"
            q.work = params.decode_s_per_token * max(1, total - emitted)
            decoders.append(q)
    for o in scenario.offers:
        if o.offset_s >= config.evaluation_window_s + config.tail_s:
            continue
        sched = t0 + o.offset_s
        q = _Req(o.request_id, params.cost_spec.estimate(o.prompt_tokens, o.max_tokens),
                 o.prompt_tokens, o.max_tokens, sched, o.egress_delay_s, o.success,
                 o.offset_s < config.evaluation_window_s, output=o.output_tokens,
                 arrival=sched + o.ingress_delay_s)
        q.emits_first = o.output_tokens >= 1
        q.decode_work = params.decode_s_per_token * max(0, o.output_tokens - 1)
        pending.append(q)
    pending.sort(key=lambda q: (q.arrival, q.rid))
    return end, old_fifo, decoders, pending


def rollout(snapshot: Snapshot, plan: Plan, scenario: Scenario, parameters: ModelParameters,
            config: PlanningConfig, *, deadline: float | None = None) -> RolloutResult:
    """Simulate one plan under one shared scenario; see module docstring.

    ``deadline`` is an absolute ``time.monotonic()`` value. Exceeding it, the
    event limit or the active-request limit raises ``RolloutError`` with a
    stable reason (``deadline_exceeded``, ``event_limit``, ``active_limit``);
    no truncated result is ever returned.
    """
    _validate_parameters(parameters)
    _validate_config(config)
    _validate_plan(plan, parameters, config)
    _validate_snapshot(snapshot, parameters, config)
    _validate_scenario(scenario, snapshot, parameters)
    if deadline is not None and (not finite(deadline)):
        _fail("invalid_deadline")
    if deadline is not None and time.monotonic() > deadline:
        _fail("deadline_exceeded")

    t0 = snapshot.sampled_at
    end, fifo, decoders, pending = _build(snapshot, scenario, parameters, config)
    cohort = [q for q in fifo + decoders + pending if q.scored]
    sim = _Sim(now=t0, reserved=snapshot.reserved)
    sim.trace.append((t0, sim.reserved))
    pi = 0

    def release(q: _Req, t: float, success: bool) -> None:
        if q.released:  # pragma: no cover - structural guard
            _fail("double_release")
        q.released = True
        q.phase = "done"
        q.terminal = t
        q.final_success = success
        # Mirror gateway.finish_request arithmetic.
        sim.reserved = max(0.0, sim.reserved - q.cost)
        sim.trace.append((t, sim.reserved))

    def start_decode(q: _Req, t: float) -> None:
        if not q.emits_first:  # failed before first content
            release(q, t, False)
            return
        q.first_client = t + q.egress
        q.work = q.decode_work
        if q.work <= _EPS:
            release(q, t, q.success)
        else:
            q.phase = "decode"
            decoders.append(q)

    while True:
        sim.events += 1
        if sim.events > config.max_events:
            _fail("event_limit")
        if deadline is not None and sim.events % _DEADLINE_CHECK_EVERY == 0 and time.monotonic() > deadline:
            _fail("deadline_exceeded")

        slow = scenario.service_scale * (1.0 + parameters.decode_interference * len(decoders))
        t_pre = sim.now + fifo[0].work * slow if fifo else math.inf
        t_dec = sim.now + min(q.work for q in decoders) * slow if decoders else math.inf
        t_arr = pending[pi].arrival if pi < len(pending) else math.inf
        t_next = min(t_pre, t_dec, t_arr)
        if t_next > end:
            _advance(sim, fifo, decoders, end, slow)
            break
        _advance(sim, fifo, decoders, t_next, slow)

        # Tie order: service completions (releases) before arrivals.
        if t_dec <= t_next:
            done = [q for q in decoders if q.work <= _EPS]
            if not done:  # numerical guard: finish the minimum
                done = [min(decoders, key=lambda q: q.work)]
            for q in done:
                decoders.remove(q)
                release(q, t_next, q.success)
            continue
        if t_pre <= t_next:
            q = fifo.pop(0)
            q.work = 0.0
            start_decode(q, t_next)
            continue
        q = pending[pi]
        pi += 1
        budget = plan.budget_at(t_next - t0)
        # Mirror gateway.begin_request: reject iff R + c > budget.
        projected = sim.reserved + q.cost
        if projected > budget:
            q.phase = "rejected"
            q.terminal = t_next
            q.final_success = False
            continue
        q.accepted = True
        sim.reserved = projected
        sim.trace.append((t_next, sim.reserved))
        q.phase = "prefill"
        q.work = parameters.first_overhead_s + parameters.prefill_s_per_token * q.prompt
        fifo.append(q)
        if len(fifo) + len(decoders) > config.max_active_requests:
            _fail("active_limit")

    outcomes, lower, upper = [], 0, 0
    for q in cohort:
        # Client-observed view: gateway time + egress, censored at the cut.
        first = q.first_client if q.first_client is not None and q.first_client <= end else None
        client_terminal = None
        if q.phase in ("done", "rejected"):
            client_terminal = q.terminal + q.egress
        resolved = client_terminal is not None and client_terminal <= end
        o = Outcome(q.rid, q.scheduled, q.accepted, first,
                    client_terminal if resolved else None,
                    q.final_success if resolved else None, q.cost, q.prompt, q.max_tokens)
        outcomes.append(o)
        eventual = q.final_success if q.final_success is not None else q.success
        if o.good(config.slo_s):
            lower += 1
            upper += 1
        elif o.success is None and eventual:
            if q.first_client is not None:
                possible = q.first_client - q.scheduled <= config.slo_s
            else:
                earliest = max(end, q.arrival if q.arrival is not None else end) + q.egress
                possible = earliest - q.scheduled <= config.slo_s
            upper += int(possible)
    return RolloutResult(tuple(outcomes), tuple(sim.trace), float(lower), float(upper), sim.events)


def _advance(sim: _Sim, fifo, decoders, t: float, slow: float) -> None:
    dt = t - sim.now
    if dt < 0:  # pragma: no cover - structural guard
        _fail("time_reversal")
    if dt > 0:
        done = dt / slow
        if fifo:
            fifo[0].work = max(0.0, fifo[0].work - done)
        for q in decoders:
            q.work = max(0.0, q.work - done)
    sim.now = t


def _rng(seed: int, snapshot: Snapshot, index: int) -> random.Random:
    key = f"{seed}|{snapshot.run_id}|{snapshot.sequence}|{index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def generate_scenarios(snapshot: Snapshot, *, arrival_rate: float, prompt_tokens: tuple[int, ...],
                       max_tokens: tuple[int, ...], count: int, seed: int, duration_s: float,
                       service_scales: tuple[float, ...] = (1.0,)) -> tuple[Scenario, ...]:
    """Seeded, past-conditioned scenario draws (no benchmark future tape).

    Inputs are past summaries supplied by the caller: an offer rate estimate
    and empirical prompt/max_tokens marks. Arrivals are Poisson on
    ``[0, duration_s)``; marks are drawn independently with replacement.
    Without a qualified EOS model, ``output_tokens = max_tokens`` (no oracle
    EOS) and ``success=True``; transport delays are zero unless the caller
    replaces them. Service scales are assigned round-robin. The draw depends
    only on (seed, run_id, sequence, index), so plans share common scenarios.

    Generation is bounded: ``count`` may not exceed ``MAX_GENERATED_SCENARIOS``
    (``scenario_count_limit``), the total number of offers over all scenarios
    may not exceed ``MAX_GENERATED_OFFERS`` (``offer_limit``), and an
    inter-arrival too small to advance the float clock raises
    ``nonadvancing_arrivals``. All are ``ValueError``; nothing is truncated.
    """
    if not finite(arrival_rate) or arrival_rate < 0:
        raise ValueError("invalid_arrival_rate")
    if not finite(duration_s, positive=True):
        raise ValueError("invalid_duration")
    if type(count) is not int or count <= 0:
        raise ValueError("invalid_count")
    if type(seed) is not int:
        raise ValueError("invalid_seed")
    if not prompt_tokens or any(type(p) is not int or p < 0 for p in prompt_tokens):
        raise ValueError("invalid_prompt_marks")
    if not max_tokens or any(type(m) is not int or m <= 0 for m in max_tokens):
        raise ValueError("invalid_max_token_marks")
    if not service_scales or any(not finite(s, positive=True) for s in service_scales):
        raise ValueError("invalid_service_scales")
    if count > MAX_GENERATED_SCENARIOS:
        raise ValueError("scenario_count_limit")
    out, total = [], 0
    for k in range(count):
        rng = _rng(seed, snapshot, k)
        offers, t, j = [], 0.0, 0
        while arrival_rate > 0:
            dt = rng.expovariate(arrival_rate)
            if dt > 0 and t + dt == t:
                raise ValueError("nonadvancing_arrivals")
            t += dt
            if t >= duration_s:
                break
            total += 1
            if total > MAX_GENERATED_OFFERS:
                raise ValueError("offer_limit")
            m = rng.choice(max_tokens)
            offers.append(FutureOffer(f"future:{snapshot.sequence}:{k}:{j}", t,
                                      rng.choice(prompt_tokens), m, m))
            j += 1
        out.append(Scenario(tuple(offers), service_scale=service_scales[k % len(service_scales)]))
    return tuple(out)
