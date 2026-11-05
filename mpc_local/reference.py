"""Independent all-assigned HTTP scoring, not a model or simulator oracle."""

import math


def score(rows, duration_s, *, ttft_limit=0.5, tpot_limit=0.05):
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("invalid_duration")
    counts = dict(offered=0, complete=0, rejected=0, timeout=0, capped=0,
                  unknown=0, cohort_good=0, window_good=0)
    seen = set()
    for row in rows:
        uid = row["request_id"]
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("duplicate_or_invalid_assignment")
        seen.add(uid)
        start = row["scheduled_at"]
        if not math.isfinite(start) or not 0 <= start < duration_s:
            raise ValueError("outside_assigned_window")
        fate = row["fate"]
        if fate not in ("complete", "rejected", "timeout", "capped", "unknown"):
            raise ValueError("unclassified_fate")
        counts["offered"] += 1
        counts[fate] += 1
        if fate != "complete":
            continue
        first, end, output = row["first_at"], row["terminal_at"], row["output_tokens"]
        if (not math.isfinite(end) or end < start
                or type(output) is not int or output < 0):
            raise ValueError("invalid_complete_fate")
        if first is None:
            continue
        if not math.isfinite(first) or not start <= first <= end:
            raise ValueError("invalid_first_content")
        good = (
            row["finish_reason"] == "stop" and output > 1
            and first - start <= ttft_limit
            and (end - first) / (output - 1) <= tpot_limit
        )
        counts["cohort_good"] += int(good)
        counts["window_good"] += int(good and end <= duration_s)
    return {
        **counts, "cohort_goodput": counts["cohort_good"] / duration_s,
        "window_goodput": counts["window_good"] / duration_s,
        "all_assigned_good_fraction": counts["cohort_good"] / counts["offered"]
        if counts["offered"] else 0.0,
    }
