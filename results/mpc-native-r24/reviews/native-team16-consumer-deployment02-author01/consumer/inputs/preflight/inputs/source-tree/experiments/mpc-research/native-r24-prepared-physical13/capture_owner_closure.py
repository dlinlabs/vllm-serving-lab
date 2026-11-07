"""Narrow cold-start closure before any trial admission, not a drain heuristic."""

from copy import deepcopy
from dataclasses import dataclass
import re

from live_owner_prefix import number
from clocked_owner_cut import SOURCE_FIELDS


@dataclass(frozen=True)
class InitialCaptureClosure:
    native_run_id: str
    gateway_run_id: str
    clock_domain: str
    capture_id: str
    capture_started_at: float
    initial_cut_at: float
    initial_sequence: int
    native_sources: dict
    baseline_publications: dict


def bootstrap_warm_ids(body, native_startup):
    phase = body["native_phase"]
    warm = native_startup["readiness"]["warmup_response"]
    publications = phase["publications"]
    if (phase["lifecycle"] or body["scheduler_state"]["requests"]
            or body["queue_oldest_first"] or len(publications) != 1
            or not isinstance(warm["id"], str) or not warm["id"]
            or any(re.fullmatch(re.escape(warm["id"]) + r"-[0-9a-f]{8}", rid) is None
                   for rid in publications)
            or any(number(row["put_finished_at"]) > number(phase["capture"]["started_at"])
                   for row in publications.values())):
        raise ValueError("capture_closure_unjoined_or_live_bootstrap_baseline")
    return set(publications)


def bound_baseline_ids(closure, phase, native_startup, *, expected_gateway_run_id):
    warm = native_startup["readiness"]["warmup_response"]
    if (not isinstance(closure, InitialCaptureClosure)
            or closure.native_run_id != phase["run_id"]
            or closure.gateway_run_id != expected_gateway_run_id
            or closure.clock_domain != phase["clock_domain"]
            or closure.capture_id != phase["capture"]["trial_id"]
            or closure.capture_started_at != phase["capture"]["started_at"]
            or closure.initial_cut_at > number(phase["source_at"])
            or closure.native_sources != {key: phase.get(key) for key in SOURCE_FIELDS}
            or len(closure.baseline_publications) != 1
            or any(re.fullmatch(re.escape(warm["id"]) + r"-[0-9a-f]{8}", rid) is None
                   for rid in closure.baseline_publications)
            or any(phase["publications"].get(rid) != row
                   for rid, row in closure.baseline_publications.items())
            or closure.baseline_publications.keys() & phase["lifecycle"].keys()):
        raise ValueError("capture_closure_current_baseline_binding")
    return set(closure.baseline_publications)


def establish_initial_capture_closure(cut, *, native_startup, gateway_startup):
    """Consume only trusted live acquisition outputs plus owned startup receipts.

    The exclusive managed sender must not start traffic before this returns.
    Gateway's only preceding ledger change must be the original same-budget
    readiness CAS; any admission, even already released, advances the sequence.
    """
    phase = cut["native_observation"]["native_phase"]
    body = cut["native_observation"]
    prefix = cut["owner_prefix"]
    current = prefix["snapshot"]
    native_ready = native_startup["readiness"]
    gateway_ready = gateway_startup["readiness"]
    before = gateway_ready["snapshot"]
    assignment = gateway_ready["same_budget_ack"]
    ack, command = assignment["acknowledgment"], assignment["command"]
    capture = phase["capture"]
    domain = cut["common_clock_proof"]["clock_domain"]
    if (cut["schema"] != "r24-clocked-owner-cut-v1"
            or cut["common_clock_proof"].get("clock_process_binding_only") is not True
            or cut["common_clock_proof"]["native_run_id"] != phase["run_id"]
            or cut["common_clock_proof"]["gateway_run_id"] != prefix["run_id"]
            or prefix.get("gateway_prefix_closed") is not True
            or body.get("complete") is not True or phase.get("complete") is not True
            or phase["clock_domain"] != domain or prefix["clock_domain"] != domain
            or cut["cut_at"] != phase["source_at"]
            or prefix["cut_at"] != cut["cut_at"]
            or prefix["owners"] or current["requests"] or before["requests"]
            or type(before["sequence"]) is not int or before["sequence"] != 0
            or type(before["budget_version"]) is not int or before["budget_version"] != 0
            or type(current["sequence"]) is not int or current["sequence"] != 1
            or current["budget_version"] != 1 or prefix["closed_through_sequence"] != 1
            or assignment["status"] != "applied" or ack["status"] != "applied"
            or ack["operation_id"] != command["operation_id"]
            or command["expected_version"] != 0 or ack["budget_version"] != 1
            or ack["budget"] != ack["previous_budget"]
            or ack["budget"] != before["budget"] or current["budget"] != before["budget"]
            or any(row["run_id"] != prefix["run_id"] or row["clock_domain"] != domain
                   for row in (before, current, ack, command))
            or phase["lifecycle"] or body["scheduler_state"]["requests"]
            or body["queue_oldest_first"]
            or capture["status"] != "running"
            or not number(ack["applied_at"]) <= number(capture["started_at"]) <= number(cut["cut_at"])):
        raise ValueError("capture_closure_nonempty_or_nonfresh_bootstrap")
    frames = native_ready["frames"]
    if (len(frames) != 2 or any(
            frame["complete"] is not True or frame["run_id"] != phase["run_id"]
            or frame["clock_domain"] != domain
            or any(frame[key] != 0 for key in
                   ("running", "waiting", "pending_batches", "frontend_unfinished"))
            or number(frame["sampled_at"]) > capture["started_at"] for frame in frames)):
        raise ValueError("capture_closure_owned_initial_drain")
    warm = native_ready["warmup_response"]
    if (len(warm["choices"]) != 1 or warm["choices"][0]["finish_reason"] not in ("stop", "length")
            or type(warm["usage"]["completion_tokens"]) is not int
            or warm["usage"]["completion_tokens"] < 1
            or not isinstance(warm["id"], str) or not warm["id"]):
        raise ValueError("capture_closure_known_warm_http_terminal_required")
    publications = phase["publications"]
    if (len(publications) != 1
            or set(publications) != set(cut["baseline_native_ids"])
            or any(re.fullmatch(re.escape(warm["id"]) + r"-[0-9a-f]{8}", rid) is None
                   for rid in publications)
            or any(number(row["put_finished_at"]) > capture["started_at"]
                   for row in publications.values())):
        raise ValueError("capture_closure_unjoined_baseline_publication")
    bootstrap_warm_ids(body, native_startup)
    return InitialCaptureClosure(
        phase["run_id"], prefix["run_id"], domain, capture["trial_id"],
        capture["started_at"], cut["cut_at"], 1,
        {key: phase[key] for key in SOURCE_FIELDS}, deepcopy(publications),
    )
