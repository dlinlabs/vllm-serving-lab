"""TRAIN-only finite-tail accounting, not a plant model or controller.

Reconstruct client event prefixes retrospectively; terminal status and output
length cannot affect a bound before terminal_at. This does not establish when a
Gateway controller could receive those events. Only two named TRAIN members are
read, without extracting archives. No GPU/network dependencies.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


MEMBERS = {
    "native-id-train/interventions.json": range(160001, 160005),
    "native-id-low-train/interventions.json": range(160005, 160009),
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def score_bounds(rows, cutoff):
    """Bound eventual composite good COUNT for an immutable offered cohort.

    TTFT <= 0.5s; request-mean TPOT <= 0.05s (one-token TPOT = 0).
    Missing a first-content deadline is irreversible. A timely first-content
    event alone never earns success credit. Pending terminal failures/EOS remain
    unknown, including when their final labels exist in the offline input.
    """
    if not finite(cutoff):
        raise ValueError("invalid cutoff")
    good = pending = possible = expired = 0
    for row in rows:
        if row["terminal_at"] <= cutoff:
            if row["status"] == "complete":
                first = row["first_content_at"]
                n = row["usage"]["completion_tokens"]
                tpot = (row["terminal_at"] - first) / (n - 1) if n > 1 else 0
                good += first - row["scheduled_at"] <= 0.5 and tpot <= 0.05
            continue
        pending += 1
        first = row["first_content_at"]
        observed_first = first is not None and first <= cutoff
        missed = (
            first - row["scheduled_at"] > 0.5
            if observed_first
            else cutoff >= row["scheduled_at"] + 0.5
        )
        expired += missed
        possible += not missed
    return {
        "offered": len(rows),
        "lower": good,
        "upper": good + possible,
        "pending_terminal": pending,
        "pending_possible_good": possible,
        "pending_ttft_impossible": expired,
    }


def contrast(a, b):
    """Interval for arm1 minus arm0; certainty is accounting-only, not causal."""
    lower, upper = b["lower"] - a["upper"], b["upper"] - a["lower"]
    verdict = (
        "arm1" if lower > 0 else
        "arm0" if upper < 0 else
        "tie" if lower == upper == 0 else "unresolved"
    )
    return {"lower": lower, "upper": upper, "verdict": verdict}


def validate_episode(episode, seeds, document):
    """Require complete TRAIN cohorts; do not silently drop unfavorable rows."""
    if (episode["split"] != "train" or episode["seed_block"] not in seeds
            or type(episode["arm"]) is not int or episode["arm"] not in (0, 1)
            or not finite(episode["start"]) or not finite(episode["offers_s"])
            or episode["offers_s"] <= 0):
        raise ValueError("unsupported episode")
    rows, ids = episode["rows"], episode["expected_ids"]
    if (len(ids) != len(set(ids)) or len(rows) != len(ids)
            or {r["request_id"] for r in rows} != set(ids)):
        raise ValueError("incomplete or duplicate offered cohort")
    tape = document["offered_tapes"][str(episode["seed_block"])]
    if len(tape) != len(ids):
        raise ValueError("offered tape length")
    scheduled = dict(zip(ids, tape))
    for row in rows:
        start, terminal = row["scheduled_at"], row["terminal_at"]
        if (not finite(start) or not finite(terminal) or terminal < start
                or row["clock_domain"] != document["clock_domain"]
                or not 0 <= start - episode["start"] < episode["offers_s"]
                or abs(start - (episode["start"] + scheduled[row["request_id"]])) > 1e-8):
            raise ValueError("invalid clock or offered time")
        if row["status"] == "complete":
            first, n = row["first_content_at"], row["usage"]["completion_tokens"]
            if (not finite(first) or not start <= first <= terminal
                    or type(n) is not int or not 1 <= n <= row["requested_max_tokens"]
                    or not row["stream_done"] or row["stream_error"]
                    or row["finish_reason"] != "stop"):
                raise ValueError("invalid natural-EOS outcome")
        elif row["status"] != "rejected" or not row["admission_rejection"]:
            raise ValueError("failed or incomplete collection; diagnostic unavailable")


def read_train(path):
    episodes, sources = [], {}
    with tarfile.open(path, "r:gz") as archive:
        for name, seeds in MEMBERS.items():
            matches = [m for m in archive.getmembers() if m.name == name]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError("missing, duplicate or non-file TRAIN member")
            with archive.extractfile(matches[0]) as stream:
                payload = stream.read()
            document = json.loads(payload)
            if document.get("completed") is not True:
                raise ValueError("incomplete collection")
            keys = [(e["seed_block"], e["arm"]) for e in document["episodes"]]
            if len(keys) != 8 or set(keys) != {(s, a) for s in seeds for a in (0, 1)}:
                raise ValueError("TRAIN paired blocks differ")
            for episode in document["episodes"]:
                validate_episode(episode, seeds, document)
            episodes.extend(document["episodes"])
            sources[name] = hashlib.sha256(payload).hexdigest()
    return episodes, sources


def diagnose(episodes, *, cohort_start=0.0, cohort_duration=90.0,
             tails=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)):
    if (not finite(cohort_start) or cohort_start < 0
            or not finite(cohort_duration) or cohort_duration <= 0
            or not tails or any(not finite(t) or t < 0 for t in tails)):
        raise ValueError("invalid cohort or tails")
    end = cohort_start + cohort_duration
    if any(end > e["offers_s"] for e in episodes):
        raise ValueError("cohort exceeds recorded offered window")
    groups = {}
    for episode in episodes:
        rows = [r for r in episode["rows"]
                if cohort_start <= r["scheduled_at"] - episode["start"] < end]
        key = episode["seed_block"]
        group = groups.setdefault(key, {})
        if episode["arm"] in group:
            raise ValueError("duplicate arm")
        group[episode["arm"]] = (episode, rows)
    pairs = []
    for seed, arms in sorted(groups.items()):
        if set(arms) != {0, 1}:
            raise ValueError("missing paired arm")
        e0, r0 = arms[0]
        e1, r1 = arms[1]
        if len(r0) != len(r1) or e0["cell"] != e1["cell"]:
            raise ValueError("unmatched paired cohorts")
        eventual = [
            score_bounds(rows, max((r["terminal_at"] for r in rows), default=e["start"]))
            for e, rows in (arms[0], arms[1])
        ]
        cuts = []
        for tail in sorted(set(tails)):
            bounds = [score_bounds(rows, e["start"] + end + tail)
                      for e, rows in (arms[0], arms[1])]
            cuts.append({"tail_s": tail, "arms": bounds,
                         "arm1_minus_arm0": contrast(*bounds)})
        pairs.append({
            "seed": seed, "cell": e0["cell"], "offered_per_arm": len(r0),
            "eventual_arms": eventual, "eventual_arm1_minus_arm0": contrast(*eventual),
            "cuts": cuts,
        })
    return {
        "schema": "astra-train-tail-accounting-v1",
        "scope": "retrospective client prefixes; assigned sequences, NOT MPC or same-state effects",
        "cohort_start_s": cohort_start, "cohort_duration_s": cohort_duration,
        "count_to_cohort_goodput_divisor_s": cohort_duration,
        "pair_count": len(pairs), "pairs": pairs,
        "native_model_accepted": False, "performance_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(
        "results/mpc-native-4090-scout/native-identification-train-evidence.tar.gz"))
    parser.add_argument("--cohort-start-s", type=float, default=0)
    parser.add_argument("--cohort-duration-s", type=float, default=90)
    parser.add_argument("--tails-s", type=float, nargs="+", default=[0, 0.5, 1, 2, 5, 10])
    args = parser.parse_args()
    episodes, sources = read_train(args.archive)
    result = diagnose(episodes, cohort_start=args.cohort_start_s,
                      cohort_duration=args.cohort_duration_s, tails=args.tails_s)
    result["read_members_sha256"] = sources
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
