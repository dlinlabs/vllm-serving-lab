"""Pure native-data finite-plan challenger; NOT an actuator or calibrated guarantee.

Required prefill parameters are an explicitly sourced approximation, not inferred
from decode theta. No physical defaults are supplied. EOS cells contain TRAIN-only
completed outputs, available before a declared cutoff; active requests condition
that distribution on output already observed, never on their future terminal.

Eight 3x1s plans share three deterministic joint scenarios: low/median/high arrival
rates paired with EOS quartiles. Workmix is stratified, not a future offer tape.
Score is mean modeled completions with TTFT<=.5s AND request-mean TPOT<=.05s;
rejections score zero. TPOT=(terminal-first)/(sampled_EOS-1), not inter-token P99.
Old requests with known scheduled arrival/first time are scored once per rollout,
so future interference can spoil their TPOT. These are counterfactual forecasts:
never sum overlapping receding-horizon scores as measured benchmark goodput.
All plans use the same 2s offered continuation at Fixed16, then drain completely
(bounded at 35s, otherwise fallback). No partial-drain reward or terminal weights.
This reduces horizon-edge gaming, NOT an infinite-horizon stability proof.

FIFO prefill consumes overhead+physical_tokens*coefficient, then parallel decode
uses 1000/(theta0+thetaD*D+thetaP*Pk) tokens/s/request. Pk counts all admitted
prefill prompts /1000; ready-first-token jobs count in D. Both are approximations
to native scheduling, not identified causal laws. Unknown old prefill progress
restarts full prefill conservatively; computed_tokens is deliberately unused.
Transport delays, failures, KV stalls and prefill batching are not modeled.
Caller must qualify/calibrate externally; both acceptance flags remain false.
Deadline checks are cooperative wall-clock checks, not OS real-time scheduling:
an overrun never returns a candidate, but preemption can delay the fallback.
"""
from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import math
from pathlib import Path
import time

from mpc_lab.contracts import CostSpec, digest, finite
from mpc_lab.controller import validate_snapshot
from state_sample import ServingObservation

VERSION = "native-finite-plan-research-v1"
RULES = dict(actions=(16, 32), steps=3, slot_s=1, offered_window_s=5,
             continuation_budget=16, drain_limit_s=35, slo_s=.5,
             deadline_s=.1, max_age_s=.5, max_offers=512, max_events=20000,
             mean_tpot_s=.05, single_token_tpot_s=0.,
             quantiles=(.25, .5, .75), objective="mean-offered-composite-good-completions",
             tie="keep-current-first,then-fewer-changes,then-lexicographic",
             native_calibration_accepted=False, performance_accepted=False,
             model_error_calibrated=False, actuation_authority=False)


@dataclass(frozen=True)
class EOSCell:
    pclass: str
    physical_prompt_tokens: int
    cost_prompt_tokens: int
    max_tokens: int
    train_outputs: tuple[int, ...]

    def eos(self, quantile, observed=0):
        surviving = sorted(n for n in self.train_outputs if n > observed)
        if not surviving:
            raise ValueError("unknown_conditional_EOS_support")
        return surviving[min(int(quantile * len(surviving)), len(surviving) - 1)]


def train_eos_cell(pclass, physical_prompt_tokens, cost_prompt_tokens, max_tokens,
                   records, *, cutoff):
    """Records: split, available_at, output_tokens. Filter unavailable labels."""
    records = tuple(records)
    if not finite(cutoff):
        raise ValueError("invalid_train_cutoff")
    if any(r["split"] != "train" or not finite(r["available_at"]) for r in records):
        raise ValueError("non_train_or_invalid_EOS_record")
    outputs = tuple(r["output_tokens"] for r in records if r["available_at"] <= cutoff)
    return EOSCell(pclass, physical_prompt_tokens, cost_prompt_tokens, max_tokens, outputs)


@dataclass(frozen=True)
class NativeModel:
    cost_spec: CostSpec
    theta_ms: tuple[float, float, float]
    prefill_overhead_s: float
    prefill_s_per_token: float
    cells: tuple[EOSCell, ...]
    max_decode: int
    max_prefill_ktokens: float
    decode_source: str
    prefill_source: str
    eos_source: str
    observer_sha256: str
    training_source_sha256: str
    train_cutoff: str
    prefill_identified: bool = False

    def __post_init__(self):
        if (not isinstance(self.cost_spec, CostSpec) or type(self.theta_ms) is not tuple
                or len(self.theta_ms) != 3 or not finite(self.theta_ms[0], positive=True)
                or any(not finite(x) or x < 0 for x in self.theta_ms)
                or any(not finite(x) or x < 0 for x in
                       (self.prefill_overhead_s, self.prefill_s_per_token))
                or self.prefill_overhead_s + self.prefill_s_per_token <= 0
                or type(self.max_decode) is not int or self.max_decode < 1
                or not finite(self.max_prefill_ktokens, positive=True)
                or type(self.prefill_identified) is not bool):
            raise ValueError("invalid_native_physics")
        for source in (self.decode_source, self.prefill_source, self.eos_source,
                       self.observer_sha256, self.training_source_sha256, self.train_cutoff):
            if not isinstance(source, str) or not source.strip():
                raise ValueError("required_native_source")
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("missing_train_EOS")
        keys = set()
        for c in self.cells:
            key = (c.physical_prompt_tokens, c.max_tokens)
            if (not c.pclass or key in keys or type(c.train_outputs) is not tuple
                    or not c.train_outputs
                    or any(type(x) is not int or x < 0 for x in
                           (c.physical_prompt_tokens, c.cost_prompt_tokens))
                    or type(c.max_tokens) is not int or c.max_tokens < 1
                    or any(type(n) is not int or not 1 <= n <= c.max_tokens
                           for n in c.train_outputs)):
                raise ValueError("invalid_or_ambiguous_train_EOS")
            keys.add(key)

    def spec(self):
        return dict(version=VERSION, calibration_version="native-unaccepted-v1",
                    rules=RULES.copy(), parameters=asdict(self),
                    cost_hash=self.cost_spec.fingerprint,
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @property
    def identity(self):
        """Own sealed digest, deliberately NOT ModelParameters/normal MPC identity."""
        return digest(self.spec())

    def cell(self, physical_tokens, max_tokens):
        for c in self.cells:
            if (c.physical_prompt_tokens, c.max_tokens) == (physical_tokens, max_tokens):
                return c
        raise ValueError("unknown_work_support")


@dataclass(frozen=True)
class Forecast:
    observed_through: float
    clock_domain: str
    source: str
    rates: tuple[float, float, float]
    mix: tuple[float, ...]  # proportions in model.cells order, derived from past offers


@dataclass(frozen=True)
class Offer:
    at: float
    cell_index: int
    eos: int


@dataclass(frozen=True)
class Scenario:
    offers: tuple[Offer, ...]
    eos_quantile: float


def scenarios(model, forecast, check=lambda: None):
    if (not forecast.source or type(forecast.rates) is not tuple
            or len(forecast.rates) != 3
            or any(not finite(x) or x < 0 for x in forecast.rates)
            or tuple(sorted(forecast.rates)) != forecast.rates
            or len(forecast.mix) != len(model.cells)
            or any(not finite(x) or x < 0 for x in forecast.mix)
            or not math.isclose(math.fsum(forecast.mix), 1, abs_tol=1e-12)):
        raise ValueError("invalid_past_forecast")
    result = []
    for rate, q in zip(forecast.rates, RULES["quantiles"]):
        check()
        if rate * 5 > RULES["max_offers"]:
            raise ValueError("forecast_capacity_limit")
        count = int(rate * 5)
        offers, deficits = [], [0.] * len(model.cells)
        for i in range(count):
            check()
            deficits = [d + w for d, w in zip(deficits, forecast.mix)]
            index = max(range(len(deficits)), key=deficits.__getitem__)
            deficits[index] -= 1
            offers.append(Offer((i + .5) / rate, index, model.cells[index].eos(q)))
        result.append(Scenario(tuple(offers), q))
    return tuple(result)


@dataclass
class _Job:
    cost: float
    physical: int
    prefill: float
    output: float
    eos: int
    arrival: float
    score_eligible: bool
    first: float | None = None
    old: bool = False


@dataclass(frozen=True)
class Rollout:
    good: int
    offered: int
    rejected: int
    admitted: int
    terminal: int
    reserved_trace: tuple[tuple[float, float], ...]
    scenario_hash: str
    old_good: int
    future_good: int


def rollout(model, observation, scenario, actions, check=lambda: None, *, now=None):
    """Event lifecycle; no cap decrease can evict/reprice an existing reservation."""
    if len(actions) != 3 or any(a not in (16, 32) for a in actions):
        raise ValueError("infeasible_plan")
    jobs = []
    marks = {m.request_id: m for m in observation.gateway.requests}
    origin = observation.native_received_at if now is None else now
    for w in sorted(observation.observations,
                    key=lambda w: (marks[w.request_id].admitted_at, w.request_id)):
        check()
        c = model.cell(w.physical_prompt_tokens, w.max_output_tokens)
        mark = marks[w.request_id]
        jobs.append(_Job(w.charged_cost, w.physical_prompt_tokens,
                         model.prefill_overhead_s + model.prefill_s_per_token * w.physical_prompt_tokens
                         if w.output_tokens == 0 else 0., float(w.output_tokens),
                         c.eos(scenario.eos_quantile, w.output_tokens),
                         (mark.scheduled_at if mark.scheduled_at is not None else mark.received_at) - origin,
                         mark.scheduled_at is not None and (w.output_tokens == 0 or mark.first_at is not None),
                         mark.first_at - origin if mark.first_at is not None else None, old=True))
    trace = [(0., math.fsum(j.cost for j in jobs))]
    t, index, good, rejected, admitted, terminal = 0., 0, 0, 0, 0, 0
    events, old_good = 0, 0
    while jobs or index < len(scenario.offers):
        check()
        events += 1
        if events > RULES["max_events"]:
            raise ValueError("event_limit")
        # Completions precede coincident offers: release the original cost exactly once.
        for j in tuple(jobs):
            if j.prefill <= 0 and j.output >= 1 and j.first is None:
                j.first = t
            if j.prefill <= 0 and j.output >= j.eos:
                mean_tpot = (t - j.first) / (j.eos - 1) if j.eos > 1 else 0.
                qualified = bool(j.score_eligible and j.first is not None
                                 and 0 <= j.first - j.arrival <= .5 and mean_tpot <= .05)
                good += qualified
                old_good += qualified and j.old
                terminal += 1
                jobs.remove(j)
                trace.append((t, math.fsum(k.cost for k in jobs)))
        while index < len(scenario.offers) and scenario.offers[index].at <= t:
            offer = scenario.offers[index]
            c = model.cells[offer.cell_index]
            cost = model.cost_spec.estimate(c.cost_prompt_tokens, c.max_tokens)
            budget = actions[int(t)] if t < 3 else 16
            if math.fsum(j.cost for j in jobs) + cost <= budget:
                jobs.append(_Job(cost, c.physical_prompt_tokens,
                                 model.prefill_overhead_s + model.prefill_s_per_token * c.physical_prompt_tokens,
                                 0., offer.eos, offer.at, True))
                admitted += 1
                trace.append((t, math.fsum(j.cost for j in jobs)))
            else:
                rejected += 1
            index += 1
        if not jobs and index == len(scenario.offers):
            break
        prefill = [j for j in jobs if j.prefill > 0]
        decode = [j for j in jobs if j.prefill <= 0]
        pk = sum(j.physical for j in prefill) / 1000
        if len(decode) > model.max_decode or pk > model.max_prefill_ktokens:
            raise ValueError("unknown_physics_support")
        s = model.theta_ms[0] + model.theta_ms[1] * len(decode) + model.theta_ms[2] * pk
        if not finite(s, positive=True):
            raise ValueError("nonfinite_service")
        mu = 1000 / s
        arrival = scenario.offers[index].at if index < len(scenario.offers) else math.inf
        dt = min(arrival - t, prefill[0].prefill if prefill else math.inf,
                 min(((min(j.eos, 1 if j.first is None else j.eos) - j.output) / mu
                      for j in decode), default=math.inf))
        if not finite(dt, positive=True) or t + dt > RULES["drain_limit_s"]:
            raise ValueError("infeasible_or_undrained")
        if prefill:
            prefill[0].prefill = max(0., prefill[0].prefill - dt)
        for j in decode:
            target = min(j.eos, 1 if j.first is None else j.eos)
            # Snap only the event being reached, avoiding roundoff zero-time loops.
            j.output = float(target) if dt >= (target - j.output) / mu else j.output + dt * mu
        t = arrival if dt == arrival - t else t + dt
    return Rollout(good, len(scenario.offers), rejected, admitted, terminal,
                   tuple(trace), digest(asdict(scenario)), old_good, good - old_good)


def _validate(model, observation, forecast, now, domain):
    if not isinstance(observation, ServingObservation):
        raise ValueError("missing_serving_observation")
    g = observation.gateway
    if len(g.requests) > RULES["max_offers"]:
        raise ValueError("state_capacity_limit")
    reason = validate_snapshot(g, now=now, clock_domain=domain,
                               expected_identity=g.identity, max_age_s=.5)
    if reason:
        raise ValueError(reason)
    if (g.identity.cost_hash != model.cost_spec.fingerprint
            or observation.observer_sha256 != model.observer_sha256 or not observation.native_run_id):
        raise ValueError("native_source_or_cost_identity")
    if forecast.clock_domain != domain:
        raise ValueError("forecast_clock_domain")
    receipts = (observation.gateway_received_at, observation.native_received_at, forecast.observed_through)
    if (any(not finite(x) or not 0 <= now - x <= .5 for x in receipts)
            or not g.sampled_at <= receipts[0] <= receipts[1]):
        raise ValueError("stale_or_future_receipts")
    if (g.budget not in (16, 32) or type(observation.observations) is not tuple
            or observation.running != len(observation.observations) or observation.waiting != 0
            or not finite(observation.kv_cache_usage) or not 0 <= observation.kv_cache_usage <= 1
            or type(observation.pending_batches) is not int or observation.pending_batches < 0
            or len(observation.observations) != len(g.requests)):
        raise ValueError("incomplete_native_state")
    marks = {m.request_id: m for m in g.requests}
    if {w.request_id for w in observation.observations} != set(marks):
        raise ValueError("incomplete_work_join")
    for w in observation.observations:
        m = marks[w.request_id]
        if (type(w.physical_prompt_tokens) is not int or w.physical_prompt_tokens < 0
                or type(w.output_tokens) is not int or not 0 <= w.output_tokens < m.max_tokens):
            raise ValueError("unknown_physical_or_output")
        c = model.cell(w.physical_prompt_tokens, w.max_output_tokens)
        if (w.cost_prompt_tokens != m.prompt_tokens or w.max_output_tokens != m.max_tokens
                or w.gateway_received_at != m.received_at or w.charged_cost != m.cost
                or not math.isclose(m.cost, model.cost_spec.estimate(m.prompt_tokens, m.max_tokens),
                                    rel_tol=1e-12, abs_tol=0)
                or (m.first_at is not None and w.output_tokens == 0)):
            raise ValueError("work_cost_or_progress_mismatch")
        c.eos(.5, w.output_tokens)


@dataclass(frozen=True)
class NativeDecision:
    actions: tuple[int, ...]
    score: float | None
    reason: str
    deadline_exceeded: bool
    latency_s: float
    model_identity: str
    scenario_hashes: tuple[str, ...] = ()
    plan_scores: tuple[tuple[tuple[int, ...], float], ...] = ()
    native_calibration_accepted: bool = False
    performance_accepted: bool = False
    actuation_authority: bool = False

    @property
    def budget(self):
        return self.actions[0]


def choose_plan(model, observation, forecast, *, now, clock_domain, clock=time.monotonic):
    """Return research recommendations only. Entire validation/search shares 100ms."""
    start = clock()
    identity, shared_hashes, scores = "", (), []

    def check():
        elapsed = clock() - start
        if not finite(elapsed) or elapsed < 0 or elapsed >= RULES["deadline_s"]:
            raise TimeoutError("deadline")

    try:
        identity = model.identity
        check()
        _validate(model, observation, forecast, now, clock_domain)
        shared = scenarios(model, forecast, check)
        shared_hashes = tuple(digest(asdict(s)) for s in shared)
        for actions in product((16, 32), repeat=3):
            check()
            results = [rollout(model, observation, s, actions, check, now=now) for s in shared]
            score = math.fsum(r.good for r in results) / len(results)
            if not finite(score):
                raise ValueError("nonfinite_score")
            scores.append((actions, score))
        current = observation.gateway.budget
        best = min(scores, key=lambda row: (-row[1], row[0][0] != current,
                   sum(a != b for a, b in zip((current,) + row[0], row[0])), row[0]))
        check()
        reason, actions, score = "research_only_unqualified", best[0], best[1]
    except (ValueError, TypeError, AttributeError, OverflowError, TimeoutError) as exc:
        reason, actions, score = str(exc), (16, 16, 16), None
    elapsed = clock() - start
    if not finite(elapsed) or elapsed < 0 or elapsed >= RULES["deadline_s"]:
        reason, actions, score = "deadline", (16, 16, 16), None
    return NativeDecision(actions, score, reason, reason == "deadline", elapsed,
                          identity, shared_hashes, tuple(scores))
