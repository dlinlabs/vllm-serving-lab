"""Endpoint-observed natural-EOS scoring, including the complete offered cohort."""

from collections import Counter
import math


RULE = {
    "schema": "r23-natural-endpoint-slo-v1",
    "denominator": "All retained scheduled arrivals, including not_launched/rejected/failed",
    "ttft_from": "scheduled_at", "ttft_limit_s": 0.5,
    "multi_token_output_delay": "(terminal_at-first_content_at)/(completion_tokens-1)",
    "multi_token_output_delay_limit_s": 0.05,
    "one_token": "TPOT undefined; require terminal_at-scheduled_at <=0.5s, plus TTFT<=0.5s",
    "natural_completion": "status complete, ignore_eos false, stream_done, no stream_error, finish_reason stop, G>=1",
    "capped": "finish_reason length is censored and receives no natural-completion SLO credit",
    "scope": "HTTP observations/proxies, not per-token clocks or serial/exclusive GPU service",
}


def natural_good(row):
    if (row.get("status") != "complete" or row.get("finish_reason") != "stop"
            or row.get("requested_ignore_eos") is not False
            or row.get("stream_done") is not True or row.get("stream_error") is not False):
        return False
    usage = row.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 1:
        return False
    scheduled, first, terminal = (row.get(key) for key in
                                  ("scheduled_at", "first_content_at", "terminal_at"))
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (scheduled, first, terminal)):
        return False
    if not scheduled <= first <= terminal or first - scheduled > RULE["ttft_limit_s"]:
        return False
    if tokens == 1:
        return terminal - scheduled <= RULE["ttft_limit_s"]
    return (terminal - first) / (tokens - 1) <= RULE["multi_token_output_delay_limit_s"]


def score(rows, duration_s, *, offered_start, accounting_started_at, accounting_finished_at):
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("positive_measured_offered_window_required")
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in
            (offered_start, accounting_started_at, accounting_finished_at))
            or not accounting_started_at <= offered_start
            or accounting_finished_at < offered_start + duration_s):
        raise ValueError("complete_policy_and_tail_accounting_interval_required")
    good = sum(natural_good(row) for row in rows)
    window_end = offered_start + duration_s
    in_window = sum(natural_good(row) and row["terminal_at"] <= window_end for row in rows)
    return {
        "rule": RULE, "all_offered": len(rows),
        "statuses": dict(Counter(row.get("status", "missing") for row in rows)),
        "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in rows)),
        "one_token_complete": sum(
            row.get("status") == "complete" and (row.get("usage") or {}).get("completion_tokens") == 1
            for row in rows
        ),
        "censored_cap_completions": sum(
            row.get("status") == "complete" and row.get("finish_reason") == "length" for row in rows
        ),
        "natural_slo_good": good, "offered_window_s": duration_s,
        "cohort_slo_goodput_rps": good / duration_s,
        "in_window_slo_goodput_rps": in_window / duration_s,
        "policy_and_tail_wall_s": accounting_finished_at - accounting_started_at,
        "policy_and_tail_wall_slo_goodput_rps": good / (accounting_finished_at - accounting_started_at),
        "after_offer_tail_and_accounting_s": accounting_finished_at - window_end,
        "all_offered_good_fraction": good / len(rows) if rows else None,
        "tail_scope": "All cohort outcomes, including completions after offered window; tail remains measured",
    }
