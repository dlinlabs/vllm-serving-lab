"""Live Gateway ledger closure only; not a native common-clock or phase sensor."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from mpc_lab import contracts as gateway_contracts
from mpc_lab.contracts import CostSpec, implementation_hash
from mpc_lab.runtime import snapshot_from_dict


PRODUCT_SHA256 = {
    "gateway.py": "0b1a80ade258d8875ba80a072c4f360bfa5d4a7ac312bd222c0571380b341570",
    "telemetry.py": "46d6fb923e558f6435d269dd6fa5cfd5b0d71eb1fd91cbb3aaca8a82385e6370",
}
MAX_PREFIX_BYTES = 32 * 1024 * 1024
MAX_PREFIX_ROWS = 100000


@dataclass(frozen=True)
class PublicRequest:
    prompt_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class Receipt:
    snapshot_sent_at: float
    snapshot_received_at: float
    prefix_started_at: float
    prefix_received_at: float
    snapshot_sha256: str
    prefix_sha256: str


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or positive and value == 0):
        raise ValueError("owner_prefix_nonfinite_or_negative_value")
    return value


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("owner_prefix_duplicate_json_key")
            result[key] = value
        return result

    def floating(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("owner_prefix_nonfinite_json")
        return value

    def constant(_):
        raise ValueError("owner_prefix_nonfinite_json")

    return json.loads(raw, object_pairs_hook=pairs, parse_float=floating, parse_constant=constant)


def _verified_source_implementation():
    root = Path(gateway_contracts.__file__).resolve().parents[1]
    for path, expected in PRODUCT_SHA256.items():
        if sha256((root / path).read_bytes()) != expected:
            raise ValueError("owner_prefix_unreviewed_product_source")
    return implementation_hash()


class OwnerSourceAudit:
    """Single-thread boundary audit; optional exact-input replay reuse within one plan."""

    def __init__(self, *, reuse_received_prefix=False):
        if type(reuse_received_prefix) is not bool:
            raise ValueError("owner_prefix_explicit_replay_reuse_required")
        self._state = "new"
        self._thread = None
        self._implementation = None
        self._reuse_received_prefix = reuse_received_prefix
        self._prefix_memo = None

    def __copy__(self):
        raise ValueError("owner_prefix_source_audit_not_transferable")

    def __deepcopy__(self, memo):
        raise ValueError("owner_prefix_source_audit_not_transferable")

    def __reduce_ex__(self, protocol):
        raise ValueError("owner_prefix_source_audit_not_transferable")

    def __enter__(self):
        if self._state != "new":
            raise ValueError("owner_prefix_source_audit_single_use")
        self._state = "closed"
        self._thread = threading.get_ident()
        self._implementation = _verified_source_implementation()
        self._state = "active"
        return self

    def _active(self):
        if self._state != "active" or threading.get_ident() != self._thread:
            raise ValueError("owner_prefix_source_audit_inactive_or_wrong_thread")

    def check(self, expected_identity):
        self._active()
        if (not isinstance(expected_identity, dict)
                or expected_identity.get("implementation_hash") != self._implementation):
            raise ValueError("owner_prefix_implementation_binding")

    def __exit__(self, *_):
        self._active()
        try:
            if _verified_source_implementation() != self._implementation:
                raise ValueError("owner_prefix_source_changed_during_audit")
        finally:
            self._prefix_memo = None
            self._state = "closed"


def validate_live_owner_prefix(
    snapshot_raw, prefix_raw, *, receipt, now, max_age_s, cut_at,
    expected_run_id, expected_clock_domain, expected_identity, initial_budget,
    public_memory, native_ids=(), native_run_id=None, expected_native_run_id=None,
    include_intervening_events=False, source_audit=None,
):
    """Validate received bytes. Authentication/trial liveness are caller obligations.

    cut_at is explicitly in Gateway time, not implicitly a native/collector clock.
    Native IDs are correlation claims only: absence does not establish any phase.
    """
    if source_audit is None:
        if (not isinstance(expected_identity, dict)
                or expected_identity.get("implementation_hash") != _verified_source_implementation()):
            raise ValueError("owner_prefix_implementation_binding")
    elif type(source_audit) is OwnerSourceAudit:
        source_audit.check(expected_identity)
    else:
        raise ValueError("owner_prefix_typed_source_audit_required")
    if not isinstance(public_memory, dict):
        raise ValueError("owner_prefix_public_feature_memory_required")
    if type(include_intervening_events) is not bool:
        raise ValueError("owner_prefix_explicit_intervening_event_flag_required")
    if not isinstance(snapshot_raw, bytes) or not isinstance(prefix_raw, bytes):
        raise ValueError("owner_prefix_received_bytes_required")
    if len(snapshot_raw) > MAX_PREFIX_BYTES or len(prefix_raw) > MAX_PREFIX_BYTES:
        raise ValueError("owner_prefix_capacity")
    if (sha256(snapshot_raw) != receipt.snapshot_sha256
            or sha256(prefix_raw) != receipt.prefix_sha256):
        raise ValueError("owner_prefix_receipt_digest")
    times = [receipt.snapshot_sent_at, receipt.snapshot_received_at,
             receipt.prefix_started_at, receipt.prefix_received_at, now]
    for value in [*times, cut_at, max_age_s, initial_budget]:
        number(value)
    if times != sorted(times) or max_age_s <= 0 or initial_budget <= 0:
        raise ValueError("owner_prefix_receipt_chronology")
    replay_key = None
    if (source_audit is not None and source_audit._reuse_received_prefix
            and type(snapshot_raw) is bytes and type(prefix_raw) is bytes
            and type(receipt) is Receipt and type(public_memory) is dict
            and type(expected_identity) is dict and type(native_ids) is tuple and not native_ids
            and type(expected_run_id) is str and type(expected_clock_domain) is str
            and type(receipt.snapshot_sha256) is str and type(receipt.prefix_sha256) is str
            and all(type(key) is str and type(value) is str
                    for key, value in expected_identity.items())
            and native_run_id is None and expected_native_run_id is None
            and all(type(rid) is str and type(public) is PublicRequest
                    and type(public.prompt_tokens) is int and type(public.max_tokens) is int
                    for rid, public in public_memory.items())):
        # Bind every replay input, not just receipt digests; preserve scalar types in JSON.
        metadata = [
            asdict(receipt), now, max_age_s, cut_at, expected_run_id, expected_clock_domain,
            expected_identity, initial_budget, include_intervening_events,
            sorted((rid, public.prompt_tokens, public.max_tokens)
                   for rid, public in public_memory.items()),
        ]
        try:
            replay_key = (
                snapshot_raw, prefix_raw, json.dumps(metadata, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError):
            pass
        cached = source_audit._prefix_memo
        if replay_key is not None and cached is not None and cached[0] == replay_key:
            return deepcopy(cached[1])
    document = strict_json(snapshot_raw)
    snapshot = snapshot_from_dict(document)
    if (snapshot.run_id != expected_run_id or snapshot.clock_domain != expected_clock_domain
            or asdict(snapshot.identity) != expected_identity or not snapshot.complete):
        raise ValueError("owner_prefix_snapshot_epoch_clock_identity_or_complete")
    # Strict inequality closes equal-time events too; sequence alone closes only
    # the snapshot's own (timestamp, sequence) boundary, not an inclusive time cut.
    if cut_at >= snapshot.sampled_at:
        raise ValueError("owner_prefix_strictly_later_snapshot_required")
    age_upper = now - receipt.snapshot_sent_at + snapshot.sampled_at - cut_at
    if age_upper > max_age_s:
        raise ValueError("owner_prefix_stale_cut_or_receipt")
    if "cost_spec" not in document:
        raise ValueError("owner_prefix_cost_spec_required")
    cost_spec = CostSpec(**document["cost_spec"])
    if prefix_raw and not prefix_raw.endswith(b"\n"):
        raise ValueError("owner_prefix_truncated_row")
    lines = prefix_raw.splitlines()
    if len(lines) > MAX_PREFIX_ROWS:
        raise ValueError("owner_prefix_capacity")
    owners, admissions, first_seen = {}, {}, {}
    cut_owners, cut_admissions = {}, {}
    intervening = []
    covered_cut = False
    budget, version = initial_budget, 0
    cut_budget, cut_version = budget, version
    sequence, previous_at, previous_emit = 0, 0, 0
    for line in lines:
        row = strict_json(line)
        if (not isinstance(row, dict) or row.get("run_id") != expected_run_id
                or not isinstance(row.get("event"), str)):
            raise ValueError("owner_prefix_row_epoch_or_shape")
        emitted = number(row.get("monotonic"))
        if emitted < previous_emit:
            raise ValueError("owner_prefix_emission_order")
        previous_emit = emitted
        if row["event"] != "mpc_ledger":
            continue
        sequence += 1
        at = number(row.get("effective_at"))
        if (type(row.get("sequence")) is not int or row["sequence"] != sequence
                or type(row.get("ledger_schema_version")) is not int
                or row["ledger_schema_version"] != 2 or row.get("complete") is not True
                or row.get("clock_domain") != expected_clock_domain
                or row.get("identity") != expected_identity
                or not previous_at <= at <= emitted):
            raise ValueError("owner_prefix_ledger_gap_duplicate_clock_or_order")
        previous_at = at
        if sequence > snapshot.sequence:
            if at < snapshot.sampled_at:
                raise ValueError("owner_prefix_later_backdated_event")
            continue
        if emitted > snapshot.sampled_at:
            raise ValueError("owner_prefix_required_row_after_snapshot")
        kind, rid = row.get("change_kind"), row.get("request_id")
        if kind == "control":
            budget = number(row.get("budget"), positive=True)
            version += 1
        elif kind == "admission":
            if not isinstance(rid, str) or re.fullmatch("[0-9a-f]{32}", rid) is None:
                raise ValueError("owner_prefix_gateway_id")
            public = public_memory.get(rid)
            if (not isinstance(public, PublicRequest)
                    or any(type(value) is not int or value <= 0
                           for value in (public.prompt_tokens, public.max_tokens))):
                raise ValueError("owner_prefix_public_feature_memory_required")
            cost = number(row.get("request_cost"), positive=True)
            if (rid in admissions or not math.isclose(cost, cost_spec.estimate(
                    public.prompt_tokens, public.max_tokens), rel_tol=1e-12, abs_tol=1e-9)
                    or math.fsum(owner["cost"] for owner in owners.values()) + cost > budget + 1e-9):
                raise ValueError("owner_prefix_admission_conflict")
            fact = dict(request_id=rid, cost=cost, **asdict(public),
                        admission_effective_at=at, budget_at_admission=budget,
                        budget_version_at_admission=version)
            admissions[rid] = fact
            owners[rid] = fact
        elif kind in ("terminal", "first_content"):
            if (rid not in owners or type(row.get("request_cost")) not in (int, float)
                    or row["request_cost"] != owners[rid]["cost"]):
                raise ValueError("owner_prefix_terminal_or_first_owner_conflict")
            fact = owners[rid]
            if kind == "terminal":
                del owners[rid]
            elif rid in first_seen:
                raise ValueError("owner_prefix_duplicate_first")
            else:
                first_seen[rid] = at
        else:
            raise ValueError("owner_prefix_unknown_change")
        if (type(row.get("budget_version")) is not int or row["budget_version"] != version
                or number(row.get("budget"), positive=True) != budget
                or type(row.get("running")) is not int or row["running"] != len(owners)
                or not math.isclose(number(row.get("reserved_cost")),
                                    math.fsum(owner["cost"] for owner in owners.values()),
                                    rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError("owner_prefix_conservation_or_budget_conflict")
        if at <= cut_at:
            covered_cut = True
            cut_owners, cut_admissions = dict(owners), dict(admissions)
            cut_budget, cut_version = budget, version
        elif include_intervening_events:
            intervening.append({
                "sequence": sequence, "effective_at": at, "emitted_at": emitted,
                "change_kind": kind, "request_id": rid if kind != "control" else None,
                "budget": budget, "budget_version": version,
                "public_reservation": dict(fact) if kind != "control" else None,
            })
    if sequence < snapshot.sequence:
        raise ValueError("owner_prefix_required_sequence_missing")
    if not covered_cut:
        raise ValueError("owner_prefix_cut_predates_observed_epoch")
    marks = {mark.request_id: mark for mark in snapshot.requests}
    if set(marks) != set(owners) or budget != snapshot.budget or version != snapshot.budget_version:
        raise ValueError("owner_prefix_snapshot_inventory_or_budget_conflict")
    for rid, mark in marks.items():
        fact = owners[rid]
        if (mark.cost != fact["cost"] or mark.prompt_tokens != fact["prompt_tokens"]
                or mark.max_tokens != fact["max_tokens"]
                or mark.admitted_at > fact["admission_effective_at"]
                or (mark.first_at is not None) != (rid in first_seen)
                or mark.first_at is not None and mark.first_at > first_seen[rid]):
            raise ValueError("owner_prefix_snapshot_mark_conflict")
    native_ids = tuple(native_ids)
    joined = {}
    if native_ids or native_run_id is not None or expected_native_run_id is not None:
        if (not isinstance(expected_native_run_id, str) or not expected_native_run_id
                or native_run_id != expected_native_run_id):
            raise ValueError("owner_prefix_native_epoch")
    for native_id in native_ids:
        match = re.fullmatch(r"chatcmpl-([0-9a-f]{32})-[0-9a-f]{8}", native_id) if isinstance(native_id, str) else None
        if match is None or match[1] not in cut_admissions or match[1] in joined:
            raise ValueError("owner_prefix_unjoined_or_duplicate_native_id")
        joined[match[1]] = native_id
    result = {
        "schema": "r24-gateway-live-owner-prefix-v1",
        "run_id": snapshot.run_id, "clock_domain": snapshot.clock_domain,
        "cut_at": cut_at, "snapshot_at": snapshot.sampled_at,
        "closed_through_sequence": snapshot.sequence,
        "receipt": asdict(receipt), "source_age_upper_s": age_upper,
        "age_as_of_receiver_clock": now,
        "budget": cut_budget, "budget_version": cut_version,
        "owners": [dict(fact, native_id=joined.get(rid))
                   for rid, fact in sorted(cut_owners.items())],
        "released_native_ids": [nid for rid, nid in joined.items() if rid not in cut_owners],
        "native_run_id": native_run_id,
        "gateway_prefix_closed": True, "native_common_cut_verified": False,
        "native_absence_verified": False, "trial_liveness_verified": False,
        "deployable": False, "native_mpc_permitted": False,
        "scope": "Gateway-time owner facts from received prefix; ledger admission linearization, not exact RequestMark time. No native phase, readiness, cancellation, GPU retirement or residual inferred.",
    }
    if include_intervening_events:
        result["intervening_gateway_events"] = {
            "schema": "r24-received-gateway-event-window-v1",
            "run_id": snapshot.run_id, "clock_domain": snapshot.clock_domain,
            "after_cut_at": cut_at, "snapshot_at": snapshot.sampled_at,
            "through_snapshot_sequence": snapshot.sequence,
            "prefix_received_at": receipt.prefix_received_at,
            "snapshot_sha256": receipt.snapshot_sha256, "prefix_sha256": receipt.prefix_sha256,
            "events": intervening,
            "native_state_advanced": False, "native_mpc_permitted": False,
            "scope": (
                "Validated Gateway ledger events strictly after the inclusive cut, "
                "through the snapshot sequence only. Equal-time later sequences and "
                "events after this receipt boundary are not covered. first_content "
                "and terminal are Gateway facts, not client/native first, EOS, "
                "GPU retirement or a propagated native state."
            ),
        }
    if replay_key is not None:
        source_audit._prefix_memo = (replay_key, deepcopy(result))
    return result


async def collect_live_owner_prefix(http, telemetry_path, *, clock=time.monotonic, **kwargs):
    """One live GET and bounded local prefix read; never flush, drain or retry.

    The caller supplies an authenticated client and the owned single-writer log.
    This does not acquire native clock/trial evidence or accept saved snapshots.
    """
    sent = clock()
    response = await http.get("/admin/mpc/state")
    response.raise_for_status()
    snapshot_raw, received = response.content, clock()

    def read_prefix():
        with Path(telemetry_path).open("rb") as handle:
            return handle.read(MAX_PREFIX_BYTES + 1)

    started = clock()
    raw = await asyncio.to_thread(read_prefix)
    ended = clock()
    receipt = Receipt(sent, received, started, ended, sha256(snapshot_raw), sha256(raw))
    checking = clock()
    result = validate_live_owner_prefix(
        snapshot_raw, raw, receipt=receipt, now=checking, **kwargs)
    finished = number(clock())
    if finished < checking:
        raise ValueError("owner_prefix_receipt_chronology")
    result["validation_elapsed_s"] = finished - checking
    result["source_age_upper_s"] += finished - checking
    result["age_as_of_receiver_clock"] = finished
    if result["source_age_upper_s"] > kwargs["max_age_s"]:
        raise ValueError("owner_prefix_stale_after_validation")
    return result
