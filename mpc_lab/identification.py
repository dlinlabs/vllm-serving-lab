"""Offline identification scoring. No model fitting, deployment or GPU execution."""

from collections import defaultdict
from dataclasses import asdict
import math
import statistics

from .contracts import digest, finite
from .controller import validate_snapshot


def reserved_cost_at(initial, frames, final, target_at, *, dropped_records, write_errors):
    domain, identity = initial.clock_domain, initial.identity
    if final.clock_domain != domain or any(f.get("clock_domain") != domain for f in frames):
        raise ValueError("clock_domain_mismatch")
    if any(type(v) is not int or v != 0 for v in (dropped_records, write_errors)):
        raise ValueError("trace_loss")
    for snapshot in (initial, final):
        reason = validate_snapshot(snapshot, now=snapshot.sampled_at, clock_domain=domain,
                                   expected_identity=identity, max_age_s=0)
        if reason:
            raise ValueError(reason)
    if (initial.run_id != final.run_id or not finite(target_at)
            or not initial.sampled_at <= target_at <= final.sampled_at):
        raise ValueError("unobserved_target")
    sequence, at, cost = initial.sequence, initial.sampled_at, initial.reserved
    value = cost
    for frame in frames:
        if (frame.get("event") != "mpc_ledger" or frame.get("run_id") != initial.run_id
                or frame.get("identity") != asdict(identity)
                or frame.get("complete") is not True
                or type(frame.get("sequence")) is not int
                or frame["sequence"] != sequence + 1):
            raise ValueError("ledger_identity_or_gap")
        t, r = frame.get("effective_at"), frame.get("reserved_cost")
        if (not finite(t) or not at <= t <= final.sampled_at
                or not finite(r) or r < 0):
            raise ValueError("ledger_chronology")
        sequence, at, cost = frame["sequence"], t, r
        if t <= target_at:
            value = r
    if sequence != final.sequence or not math.isclose(cost, final.reserved, abs_tol=1e-9):
        raise ValueError("final_watermark_mismatch")
    return value


def primary_forecast_mae(rows, *, expected_decisions):
    episodes, seen, blocks = defaultdict(list), set(), {}
    expected = set(expected_decisions)
    if len(expected) != len(expected_decisions):
        raise ValueError("duplicate_manifest")
    for row in rows:
        if row["target"] != "reserved_cost_1s" or row["truth_complete"] is not True:
            raise ValueError("unqualified_primary_target")
        key = row["seed_block"], row["episode_id"]
        decision_key = (*key, row["decision_id"])
        if decision_key in seen:
            raise ValueError("duplicate_decision")
        seen.add(decision_key)
        origin, target, issued = row["origin_at"], row["target_at"], row["prediction_issued_at"]
        if (any(not finite(v) for v in (origin, target, issued)) or origin < 0
                or not origin <= issued < target
                or not math.isclose(target - origin, 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("noncausal_primary_horizon")
        block_key = row["split"], row["cell"]
        if key[0] in blocks and blocks[key[0]] != block_key:
            raise ValueError("seed_leakage_or_cell_change")
        blocks[key[0]] = block_key
        values = row["predicted_reserved"], row["actual_reserved"]
        if any(not finite(v) or v < 0 for v in values):
            raise ValueError("invalid_forecast")
        episodes[key].append(abs(values[0] - values[1]) / 32.0)
    if seen != expected:
        raise ValueError("forecast_cohort_mismatch")
    seeds = defaultdict(list)
    for (seed, _), errors in episodes.items():
        seeds[seed].append(statistics.mean(errors))
    scores = {seed: statistics.mean(values) for seed, values in seeds.items()}
    return {"independent_seed_blocks": len(scores), "seed_scores": scores,
            "normalized_mae": statistics.mean(scores.values()) if scores else None}


def brier_bounds(probability, row, *, domain, predicted_at, slo_s=0.5):
    if row.get("clock_domain") != domain:
        raise ValueError("clock_domain_mismatch")
    if not finite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid_probability")
    start, sent = row.get("scheduled_at"), row.get("sent_at")
    if (not finite(predicted_at) or not finite(start) or not finite(sent)
            or start < 0 or predicted_at < 0 or not start <= sent
            or predicted_at > sent or not finite(slo_s, positive=True)):
        raise ValueError("noncausal_prediction")
    first = row.get("first_content_at")
    end = row.get("terminal_at")
    if end is None:
        end = row.get("observed_until_at")
    if not finite(end) or end < sent:
        raise ValueError("terminal_chronology")
    if first is not None and (not finite(first) or not sent <= first <= end):
        raise ValueError("content_chronology")
    timely = first is not None and first - start <= slo_s
    status = row.get("status")
    if status == "complete":
        if (row.get("http_status") != 200 or row.get("stream_done") is not True
                or row.get("stream_error") is not False
                or row.get("finish_reason") not in ("stop", "length")
                or row.get("response_request_id") != row.get("request_id")
                or first is None):
            raise ValueError("unqualified_completion")
        label = int(timely)
    elif status == "rejected":
        if row.get("admission_rejection") is not True or row.get("http_status") != 503:
            raise ValueError("unqualified_rejection")
        label = 0
    elif status in ("failed", "cancelled"):
        label = 0
    elif status in ("timeout", "unresolved"):
        observed = row.get("observed_until_at", row.get("terminal_at"))
        if not finite(observed) or observed < sent or (first is not None and first > observed):
            raise ValueError("unobserved_censoring")
        label = 0 if ((first is not None and not timely)
                      or (first is None and observed - start > slo_s)) else None
    else:
        raise ValueError("unknown_outcome")
    losses = [(probability - label) ** 2] if label is not None else [
        probability ** 2, (1 - probability) ** 2,
    ]
    return {"label": label, "lower": min(losses), "upper": max(losses)}


def cohort_brier(predictions, rows, expected_ids, *, domain, slo_s=0.5):
    def index(records):
        result = {}
        for record in records:
            key = record["request_id"]
            if key in result:
                raise ValueError("duplicate_request")
            result[key] = record
        return result
    forecasts, outcomes = index(predictions), index(rows)
    expected = set(expected_ids)
    if (len(expected) != len(expected_ids) or set(forecasts) != expected
            or set(outcomes) != expected):
        raise ValueError("offered_cohort_mismatch")
    scores = []
    for rid in expected_ids:
        forecast = forecasts[rid]
        if forecast.get("clock_domain") != domain:
            raise ValueError("clock_domain_mismatch")
        scores.append(brier_bounds(forecast["probability"], outcomes[rid], domain=domain,
                                   predicted_at=forecast["predicted_at"], slo_s=slo_s))
    return {"planned": len(scores), "unresolved": sum(s["label"] is None for s in scores),
            "lower": statistics.mean(s["lower"] for s in scores) if scores else None,
            "upper": statistics.mean(s["upper"] for s in scores) if scores else None}


def freeze_baseline(scores, *, train_seeds):
    train = set(train_seeds)
    if not train or len(train) != len(train_seeds):
        raise ValueError("invalid_training_manifest")
    candidates = defaultdict(dict)
    for row in scores:
        seed = row["validation_seed_block"]
        fits = set(row["fit_seed_blocks"])
        if (row["split"] != "train" or seed not in train or seed in fits
                or not fits <= train or len(fits) != len(row["fit_seed_blocks"])):
            raise ValueError("training_fold_leakage")
        value = row["normalized_mae"]
        candidate = row["candidate"]
        if not isinstance(candidate, str) or not candidate or not finite(value) or value < 0:
            raise ValueError("invalid_candidate_score")
        if seed in candidates[candidate]:
            raise ValueError("duplicate_fold")
        candidates[candidate][seed] = value
    if not candidates or any(set(folds) != train for folds in candidates.values()):
        raise ValueError("incomplete_training_folds")
    means = {candidate: statistics.mean(folds.values()) for candidate, folds in candidates.items()}
    result = {"train_seeds": sorted(train), "grouped_cv_mae": means,
              "selected": min(means, key=lambda key: (means[key], key))}
    return {**result, "selection_hash": digest(result)}
