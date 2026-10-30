"""CPU-only development diagnostics and offline HTTP evidence checks; no network."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mpc_lab import benchmark, controller, model
from mpc_lab.contracts import (
    FutureOffer, ModelParameters, Plan, PlanningConfig, Scenario, Snapshot,
    implementation_hash,
)


def audit_http(rows, expected_ids, clock_domain, slo_s=0.5):
    """Keep the entire offered cohort; invalid evidence cannot earn goodput."""
    expected = set(expected_ids)
    if len(expected) != len(expected_ids) or not clock_domain or not 0 < slo_s < math.inf:
        raise ValueError("invalid_offered_manifest")
    result = {"planned": len(expected), "good": 0, "rejected": 0,
              "failed": 0, "unresolved": 0, "errors": []}
    seen = set()
    for row in rows:
        rid = row.get("request_id")
        if rid not in expected or rid in seen:
            result["errors"].append({"request_id": rid, "reason": "cohort_identity"})
            continue
        seen.add(rid)
        if row.get("clock_domain") != clock_domain:
            result["errors"].append({"request_id": rid, "reason": "clock_domain"})
            continue
        status = row.get("status")
        end = row.get("terminal_at")
        if end is None and status in ("timeout", "unresolved"):
            end = row.get("observed_until_at")
        times = [row.get("scheduled_at"), row.get("sent_at"), end]
        content = [row.get("first_content_at"), row.get("last_content_at")]
        def finite(x):
            try:
                return type(x) in (float, int) and math.isfinite(x) and x >= 0
            except OverflowError:
                return False
        if not all(finite(t) for t in times) or times != sorted(times):
            result["errors"].append({"request_id": rid, "reason": "client_chronology"})
            continue
        if any(t is not None for t in content) and (
            not all(finite(t) for t in content)
            or not times[1] <= content[0] <= content[1] <= times[2]
        ):
            result["errors"].append({"request_id": rid, "reason": "content_chronology"})
            continue
        if status == "rejected":
            if row.get("http_status") != 503 or content != [None, None]:
                result["errors"].append({"request_id": rid, "reason": "rejection_evidence"})
            else:
                result["rejected"] += 1
        elif status == "complete":
            if (row.get("http_status") != 200 or row.get("stream_done") is not True
                    or row.get("stream_error") is not False
                    or row.get("finish_reason") not in ("stop", "length")
                    or row.get("response_request_id") != rid or content[0] is None):
                result["errors"].append({"request_id": rid, "reason": "completion_evidence"})
            else:
                result["good"] += content[0] - times[0] <= slo_s
        elif status in ("failed", "cancelled"):
            result["failed"] += 1
        elif status in ("timeout", "unresolved"):
            result["unresolved"] += 1
        else:
            result["errors"].append({"request_id": rid, "reason": "unknown_status"})
    for rid in sorted(expected - seen):
        result["errors"].append({"request_id": rid, "reason": "missing_offered_request"})
    result["valid"] = not result["errors"]
    result["all_offered_nonattainment"] = (
        (len(expected) - result["good"]) / len(expected) if expected else None
    )
    return result


def exact_plan_comparison(snapshot, scenarios, parameters, config):
    """Untimed model-only diagnostic, not a deadline-relaxed policy."""
    scores = []
    for actions in product(config.actions, repeat=config.control_steps):
        outcomes = [
            model.rollout(snapshot, Plan(actions, config.slot_s, config.fallback_budget),
                          scenario, parameters, config)
            for scenario in scenarios
        ]
        changes = sum(a != b for a, b in zip((snapshot.budget,) + actions, actions))
        scores.append({
            "actions": list(actions),
            "lower": statistics.mean(r.score_lower for r in outcomes),
            "upper": statistics.mean(r.score_upper for r in outcomes),
            "events": sum(r.event_count for r in outcomes),
            "changes": changes,
        })
    key = lambda r: (r["lower"], r["actions"][0] == snapshot.budget, -r["changes"])
    best = max(scores, key=key)
    one = [r for r in scores if all(b == config.fallback_budget for b in r["actions"][1:])]
    best_one = max(one, key=key)
    return {"plans": scores, "best": best, "one_move_equivalent": best_one,
            "first_action_differs": best["actions"][0] != best_one["actions"][0],
            "open_loop_gain": best["lower"] - best_one["lower"]}


def identification_manifest():
    episodes = []
    for seed in range(1, 7):
        split = "train" if seed <= 3 else {4: "calibration", 5: "validation", 6: "test"}[seed]
        for cell, long_prompt, max_tokens in (
            ("short-short", False, 64), ("long-short", True, 64),
            ("short-long", False, 512), ("long-long", True, 512),
        ):
            rng = random.Random(f"native-id:{cell}:{seed}")
            boundaries, at, step = [], 0, 0
            while at < 90:
                boundaries.append((at, rng.choice((16, 32))))
                at += (2, 5, 10)[step % 3]
                step += 1
            for arm in (0, 1):
                episodes.append({
                    "episode_id": f"{split}-{seed}-{cell}-{arm}", "split": split,
                    "offered_tape_key": f"{cell}:{seed}",
                    "arrival_tape_key": f"arrival:{seed}",
                    "prompt_tokens": (7168 if seed == 6 else 4096) if long_prompt else 128,
                    "max_tokens": max_tokens, "natural_eos": True,
                    "offers_s": 90, "maximum_drain_s": 20,
                    "schedule": [{"at_s": t, "budget": b if arm == 0 else 48 - b}
                                 for t, b in boundaries],
                })
    ordered = []
    for split in ("train", "calibration", "validation", "test"):
        group = [e for e in episodes if e["split"] == split]
        random.Random(f"order:20261030:{split}").shuffle(group)
        ordered.extend(group)
    return {
        "execution_enabled": False, "approved": False,
        "scope": "Offline design template; no requests or cloud operations",
        "gpu": "one RTX4090", "context_limit": 8192,
        "rate_reference": "Unfilled until authorized TRAIN-only capacity scout",
        "train_calibration_validation_rate_phases": [
            {"start_s": 0, "rate_multiple": 0.6}, {"start_s": 30, "rate_multiple": 1.1},
            {"start_s": 60, "rate_multiple": 0.6},
        ],
        "test_rate_phases": [
            {"start_s": 0, "rate_multiple": 0.6}, {"start_s": 20, "rate_multiple": 1.3},
            {"start_s": 35, "rate_multiple": 0.6}, {"start_s": 55, "rate_multiple": 1.3},
            {"start_s": 70, "rate_multiple": 0.6},
        ],
        "time_budget_s": {"setup": 1200, "qualification_scout": 900,
                          "episodes_and_drain": 5280, "artifact_export": 600,
                          "contingency": 1020, "total": 9000},
        "episodes": ordered,
    }


def fixed_oracle_diagnostic(trace, seed, workload):
    """Known-future/EOS ablation isolates synthetic service-law error, not deployable."""
    parameters = ModelParameters()
    config = replace(PlanningConfig(), evaluation_window_s=12.0, tail_s=20.0)
    scenario = Scenario(tuple(
        FutureOffer(o.request_id, o.scheduled_at, o.prompt_tokens, o.max_tokens,
                    o.output_tokens, o.success) for o in trace
    ))
    rows = []
    for budget in (16.0, 32.0):
        truth = benchmark.run_episode(trace, 12.0, f"fixed-{int(budget)}", seed)
        snapshot = Snapshot("oracle-diagnostic", "cpu-clock", 0.0, budget, 0,
                            (), parameters.identity)
        predicted = model.rollout(
            snapshot, Plan((budget,), continuation_budget=budget), scenario, parameters, config,
        )
        rows.append({"budget": budget, "truth_good": truth["good"],
                     "predicted_lower": predicted.score_lower,
                     "predicted_upper": predicted.score_upper})
    return {"seed": seed, "workload": workload, "arms": rows,
            "truth_delta_32_minus_16": rows[1]["truth_good"] - rows[0]["truth_good"],
            "model_delta_32_minus_16": rows[1]["predicted_lower"] - rows[0]["predicted_lower"]}


def diagnose():
    records, captures, fixed = [], [], []
    original = controller.choose_plan
    context = {}

    def measured(snapshot, parameters, scenarios, config, **kwargs):
        calls = []

        def timed_rollout(*args, **kw):
            started = time.perf_counter()
            row = {}
            try:
                prediction = model.rollout(*args, **kw)
                row["events"] = prediction.event_count
                return prediction
            except Exception as exc:
                row["error"] = str(exc)
                raise
            finally:
                row["wall_s"] = time.perf_counter() - started
                calls.append(row)

        decision = original(snapshot, parameters, scenarios, config,
                            predictor=timed_rollout, **kwargs)
        records.append({
            **context, "tick": snapshot.sequence, "active": len(snapshot.requests),
            "prompt_backlog": snapshot.prompt_backlog,
            "offers": sum(len(s.offers) for s in scenarios),
            "decision": {"budget": decision.budget, "reason": decision.reason,
                         "elapsed_s": decision.elapsed_s,
                         "complete_plans": decision.evaluated_plans},
            "rollout_calls": len(calls),
            "rollout_wall_s": sum(r["wall_s"] for r in calls),
            "completed_events": sum(r.get("events", 0) for r in calls),
        })
        if context["seed"] == 1 and snapshot.sequence in (0, 4, 8):
            captures.append((dict(context), snapshot, scenarios, parameters, config))
        return decision

    with patch.object(controller, "choose_plan", measured):
        for seed in (1, 2):
            for workload in ("stationary", "burst", "shift"):
                context.update(seed=seed, workload=workload)
                trace = benchmark.make_trace(seed, workload, 12.0)
                benchmark.run_episode(trace, 12.0, "mpc", seed)
                fixed.append(fixed_oracle_diagnostic(trace, seed, workload))
    replays = [
        {**ctx, "tick": state.sequence,
         **exact_plan_comparison(state, scenarios, params, config)}
        for ctx, state, scenarios, params, config in captures
    ]
    return {
        "scope": "Development seeds1/2 only. Instrumented timing is not the frozen benchmark.",
        "implementation_hash": implementation_hash(),
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_parameters_changed": False,
        "heldout_used": False,
        "deadline_s": PlanningConfig().max_wall_s,
        "instrumented_decisions": records,
        "instrumented_reasons": dict(Counter(r["decision"]["reason"] for r in records)),
        "untimed_replay_scope": "Same snapshots/scenarios; model score only, never a deployed action",
        "untimed_replays": replays,
        "known_future_fixed_scope": "Oracle future/EOS for service-law diagnosis only, not a baseline",
        "known_future_fixed": fixed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = identification_manifest() if args.manifest_only else diagnose()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
