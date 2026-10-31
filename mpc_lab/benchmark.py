"""Exploratory CPU comparison against a deliberately different synthetic plant."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .contracts import ModelParameters, Outcome, PlanningConfig, RequestMark, Snapshot


@dataclass(frozen=True)
class TraceOffer:
    request_id: str
    scheduled_at: float
    prompt_tokens: int
    max_tokens: int
    output_tokens: int
    success: bool


def make_trace(seed: int, workload: str, duration_s: float) -> tuple[TraceOffer, ...]:
    if workload not in ("stationary", "burst", "shift") or duration_s <= 0:
        raise ValueError("invalid_trace_configuration")
    rng = random.Random(seed)
    offers = []
    now = 0.0
    while True:
        now += rng.expovariate(40.0)
        if now >= duration_s:
            break
        rate = 14.0
        if workload == "burst":
            rate = 36.0 if duration_s / 3 <= now < 2 * duration_s / 3 else 8.0
        if rng.random() > rate / 40:
            continue
        prompts = (128, 512, 2048)
        if workload == "shift" and now >= duration_s / 2:
            prompts = (2048, 4096, 8192)
        prompt = rng.choice(prompts)
        maximum = rng.choice((64, 128, 256))
        offers.append(TraceOffer(
            f"offer-{len(offers)}", now, prompt, maximum,
            rng.randint(maximum // 4, maximum), rng.random() >= 0.02,
        ))
    return tuple(offers)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "maximum": max(values) if values else None,
    }


def outcome_metrics(outcomes: tuple[Outcome, ...], duration_s: float) -> dict:
    accepted = [o for o in outcomes if o.accepted]
    good = sum(o.good() for o in outcomes)
    successful = [o for o in accepted if o.success is True and o.terminal_at is not None]
    return {
        "offered": len(outcomes),
        "accepted": len(accepted),
        "successful": len(successful),
        "good": good,
        "accepted_throughput_rps": len(accepted) / duration_s,
        "successful_goodput_rps": good / duration_s,
        "rejection_rate": (len(outcomes) - len(accepted)) / max(1, len(outcomes)),
        "failure_rate": sum(o.success is False for o in accepted) / max(1, len(outcomes)),
        "unresolved": sum(o.success is None for o in accepted),
        "all_offered_slo_nonattainment": (len(outcomes) - good) / max(1, len(outcomes)),
        "ttft_s": distribution([o.first_at - o.scheduled_at for o in accepted if o.first_at is not None]),
        "e2e_success_s": distribution([o.terminal_at - o.scheduled_at for o in successful]),
    }


def run_episode(
    trace: tuple[TraceOffer, ...],
    duration_s: float,
    policy: str,
    seed: int,
    *,
    feedback_thresholds: tuple[int, int] = (2000, 6000),
    config: PlanningConfig = PlanningConfig(),
) -> dict:
    from .controller import choose_plan, feedback_budget
    from .model import generate_scenarios

    if policy not in ("fixed-16", "fixed-32", "feedback", "one-move", "mpc"):
        raise ValueError("unknown_policy")
    params = ModelParameters()
    identity = params.identity
    budget = 32.0 if policy == "fixed-32" else 16.0
    version = 0
    active: dict[str, dict] = {}
    outcomes: dict[str, Outcome] = {}
    observed: list[TraceOffer] = []
    index = 0
    step = 0.025
    control_tick = 0
    fallback = Counter()
    times, cpu_times, actions, backlogs = [], [], [], []
    tpots = []
    # This plant is a time-grid fluid queue, not model.rollout or its service law.
    for tick in range(math.ceil((duration_s + 20.0) / step)):
        now = tick * step
        interval_end = (tick + 1) * step
        if now >= duration_s and not active and index == len(trace):
            break
        if now < duration_s and now + 1e-9 >= control_tick:
            marks = tuple(RequestMark(
                rid, row["cost"], row["offer"].prompt_tokens, row["offer"].max_tokens,
                row["received"], row["received"], row["first"], row["offer"].scheduled_at,
            ) for rid, row in active.items())
            snapshot = Snapshot("cpu-episode", "cpu-clock", now, budget, version, marks, identity, control_tick)
            backlogs.append((now, snapshot.prompt_backlog))
            wall, cpu = time.monotonic(), time.process_time()
            next_budget = budget
            if policy == "feedback":
                next_budget = feedback_budget(
                    snapshot, backlog_low=feedback_thresholds[0], backlog_high=feedback_thresholds[1],
                )
            elif policy in ("one-move", "mpc"):
                recent = [o for o in observed if o.scheduled_at >= now - 2.0]
                rate = len(recent) / min(2.0, now) if now > 0 else 14.0
                forecast_seed = int.from_bytes(
                    hashlib.sha256(f"forecast:{seed}:{control_tick}".encode()).digest()[:8], "big",
                )
                scenarios = generate_scenarios(
                    snapshot, arrival_rate=rate,
                    prompt_tokens=tuple(o.prompt_tokens for o in recent) or (128, 512, 2048),
                    max_tokens=tuple(o.max_tokens for o in recent) or (64, 128, 256),
                    count=4, seed=forecast_seed,
                    duration_s=config.evaluation_window_s + config.tail_s,
                    service_scales=(0.8, 1.0, 1.3),
                )
                planning = replace(config, control_steps=1) if policy == "one-move" else config
                decision = choose_plan(
                    snapshot, params, scenarios, planning, now=now, clock_domain="cpu-clock",
                )
                next_budget = decision.budget
                if decision.fallback:
                    fallback[decision.reason] += 1
            times.append(time.monotonic() - wall)
            cpu_times.append(time.process_time() - cpu)
            if next_budget != budget:
                version += 1
            budget = next_budget
            actions.append(budget)
            control_tick += 1

        while index < len(trace) and trace[index].scheduled_at <= now:
            offer = trace[index]
            index += 1
            observed.append(offer)
            cost = params.cost_spec.estimate(offer.prompt_tokens, offer.max_tokens)
            if math.fsum(r["cost"] for r in active.values()) + cost <= budget:
                active[offer.request_id] = {
                    "offer": offer, "cost": cost, "received": now, "first": None,
                    "prompt_left": float(offer.prompt_tokens + 200),
                    "output_left": float(max(0, offer.output_tokens - 1)),
                }
            else:
                outcomes[offer.request_id] = Outcome(
                    offer.request_id, offer.scheduled_at, False, None, None, False,
                    cost, offer.prompt_tokens, offer.max_tokens,
                )

        streaming = [r for r in active.values() if r["first"] is not None]
        awaiting = [r for r in active.values() if r["first"] is None]
        remaining_prefill = 14000.0 * step / (1.0 + 0.08 * len(streaming))
        for row in awaiting:
            spent = min(row["prompt_left"], remaining_prefill)
            row["prompt_left"] -= spent
            remaining_prefill -= spent
            if row["prompt_left"] <= 1e-9:
                row["first"] = interval_end
            if remaining_prefill <= 0:
                break
        backlog = sum(r["offer"].prompt_tokens for r in awaiting)
        decode_tokens = 100.0 * step / (1.0 + 0.04 * len(streaming) + backlog / 100000.0)
        for rid, row in list(active.items()):
            if row in streaming:
                row["output_left"] -= decode_tokens
            if row["first"] is not None and row["output_left"] <= 1e-9:
                offer = row["offer"]
                terminal = interval_end
                outcomes[rid] = Outcome(
                    rid, offer.scheduled_at, True, row["first"], terminal, offer.success,
                    row["cost"], offer.prompt_tokens, offer.max_tokens,
                )
                if offer.output_tokens > 1:
                    tpots.append((terminal - row["first"]) / (offer.output_tokens - 1))
                del active[rid]

    for rid, row in active.items():
        offer = row["offer"]
        outcomes[rid] = Outcome(
            rid, offer.scheduled_at, True, row["first"], None, None,
            row["cost"], offer.prompt_tokens, offer.max_tokens,
        )
    if len(outcomes) != len(trace):
        raise AssertionError("offered_cohort_accounting_incomplete")
    metrics = outcome_metrics(tuple(outcomes.values()), duration_s)
    recovery = None
    before = [w for t, w in backlogs if t < duration_s / 3]
    if before:
        threshold = quantile(before, 0.95)
        after = [(t, w) for t, w in backlogs if t >= 2 * duration_s / 3]
        for (t, w), (_, next_w) in zip(after, after[1:]):
            if w <= threshold and next_w <= threshold:
                recovery = t - 2 * duration_s / 3
                break
    metrics.update({
        "policy": policy, "seed": seed,
        "request_tpot_s": distribution(tpots),
        "controller_wall_s": distribution(times),
        "controller_cpu_total_s": sum(cpu_times),
        "decisions": len(actions),
        "fallback_reasons": dict(fallback),
        "budget_switches": sum(a != b for a, b in zip(actions, actions[1:])),
        "budget_total_variation": sum(abs(a - b) for a, b in zip(actions, actions[1:])),
        "backlog_recovery_s": recovery,
        "gpu_utilization": None,
    })
    return metrics


def paired_interval(differences: list[float]) -> dict:
    rng = random.Random(5813)
    means = sorted(
        statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000)
    )
    return {
        "mean_difference_rps": statistics.mean(differences),
        "exploratory_seed_block_percentile95": [quantile(means, 0.025), quantile(means, 0.975)],
        "seed_blocks": len(differences),
        "confirmatory": False,
    }


def benchmark(duration_s: float = 12.0, seeds: tuple[int, ...] = (10001, 10002, 10003, 10004, 10005)) -> dict:
    workloads = ("stationary", "burst", "shift")
    development = []
    thresholds = ((512, 2048), (2000, 6000), (4000, 10000))
    for seed in (1, 2):
        for workload in workloads:
            trace = make_trace(seed, workload, duration_s)
            for policy in ("fixed-16", "fixed-32"):
                development.append({**run_episode(trace, duration_s, policy, seed), "workload": workload})
            for low, high in thresholds:
                row = run_episode(trace, duration_s, "feedback", seed, feedback_thresholds=(low, high))
                development.append({**row, "workload": workload, "thresholds": [low, high]})
    fixed = max(("fixed-16", "fixed-32"), key=lambda p: statistics.mean(
        r["successful_goodput_rps"] for r in development if r["policy"] == p
    ))
    feedback = max(thresholds, key=lambda pair: statistics.mean(
        r["successful_goodput_rps"] for r in development if r.get("thresholds") == list(pair)
    ))
    episodes = []
    for seed in seeds:
        for workload in workloads:
            trace = make_trace(seed, workload, duration_s)
            policies = ["fixed-16", "fixed-32", "feedback", "one-move", "mpc"]
            random.Random(f"arm-order:{seed}:{workload}").shuffle(policies)
            for policy in policies:
                episodes.append({
                    **run_episode(trace, duration_s, policy, seed, feedback_thresholds=feedback),
                    "workload": workload,
                })
    comparisons = {}
    for baseline in (fixed, "feedback", "one-move"):
        differences = []
        for seed in seeds:
            def mean(policy: str) -> float:
                return statistics.mean(
                    r["successful_goodput_rps"] for r in episodes
                    if r["seed"] == seed and r["policy"] == policy
                )
            differences.append(mean("mpc") - mean(baseline))
        comparisons[baseline] = paired_interval(differences)
    return {
        "schema": "mpc-exploratory-cpu-benchmark-v1",
        "scope": "synthetic CPU only; different truth/forecast models; not GPU/HTTP acceptance",
        "duration_s": duration_s, "dt_s": 0.025, "drain_limit_s": 20,
        "warmup": "identical empty plant; no separate warmup",
        "development_seeds": [1, 2], "heldout_seeds": list(seeds),
        "protocol_amendment": "reviewer used a mock prefix of original seed101; fresh seeds frozen before real comparison",
        "selected_fixed": fixed, "selected_feedback_thresholds": list(feedback),
        "planning_config": asdict(PlanningConfig()), "forecast_scenarios": 4,
        "identity": asdict(ModelParameters().identity),
        "statistical_limit": "five seed blocks; exploratory intervals, no multiplicity/superiority claim",
        "development": development, "episodes": episodes, "paired_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark(), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
