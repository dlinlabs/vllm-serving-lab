"""Join phases at the historical native cut, not at decision/receipt time."""

from dataclasses import asdict, dataclass
import re

from capture_owner_closure import bound_baseline_ids
from live_owner_prefix import PublicRequest, number
from prefill_lab.contracts import digest


@dataclass(frozen=True)
class SourceOwner:
    request_id: str
    native_id: str | None
    phase: str
    prompt_tokens: int
    max_tokens: int
    cost: float
    admitted_at: float
    historical_admission_budget: float
    publication_interval: tuple | None
    enqueue_at: float | None
    native_first_at: float | None
    native_finished_at: float | None


@dataclass(frozen=True)
class SourceOwnerPhases:
    source_at: float
    available_at: float
    classified_at: float
    clock_domain: str
    native_run_id: str
    gateway_run_id: str
    capture_id: str
    source_budget: float
    source_budget_version: int
    received_prefix_through_sequence: int
    cut_canonical_sha256: str
    initial_closure_canonical_sha256: str
    owners: tuple[SourceOwner, ...]
    running_ids: tuple[str, ...]
    waiting_ids: tuple[str, ...]
    ghost_ids: tuple[str, ...]
    released_native_ids: tuple[str, ...]
    released_ghost_ids: tuple[str, ...]

    def evidence(self):
        return {
            **asdict(self), "schema": "r24-source-time-owner-phases-v1",
            "requires_source_age_propagation": True,
            "decision_time_state_verified": False,
            "deployable": False, "native_mpc_permitted": False,
            "scope": (
                "Trusted clocked acquisition plus bound initial closure and actual-offer "
                "public memory. Phase/owner facts at historical native source T only. "
                "Later Gateway receipt closes T; it does not advance native state to "
                "decision time or provide ingress/release/HTTP residuals."
            ),
        }


def classify_source_owners(cut, *, initial_closure, native_startup, public_memory,
                           now, expected_clock_domain, max_age_s=0.5):
    """Caller supplies authenticated acquisition outputs, not arbitrary inventories."""
    body, prefix = cut["native_observation"], cut["owner_prefix"]
    phase, common = body["native_phase"], cut["common_clock_proof"]
    source, available, now, limit = map(number, (
        cut["cut_at"], cut["validated_at"], now, max_age_s))
    if (cut["schema"] != "r24-clocked-owner-cut-v1" or not 0 < limit <= 0.5
            or not source <= available <= now or now - source > limit
            or body.get("complete") is not True or phase.get("complete") is not True
            or body.get("refusal_reasons") or phase.get("refusal_reasons")
            or phase["source_at"] != source or prefix["cut_at"] != source
            or prefix.get("gateway_prefix_closed") is not True
            or not source < number(prefix["snapshot_at"]) <= available
            or common.get("clock_process_binding_only") is not True
            or common["native_run_id"] != phase["run_id"]
            or common["gateway_run_id"] != prefix["run_id"]
            or prefix["native_run_id"] != phase["run_id"]
            or any(row["clock_domain"] != expected_clock_domain
                   for row in (phase, prefix, common))):
        raise ValueError("source_owner_cut_clock_epoch_age_or_completeness")
    baseline = bound_baseline_ids(
        initial_closure, phase, native_startup, expected_gateway_run_id=prefix["run_id"])
    if set(cut["baseline_native_ids"]) != baseline:
        raise ValueError("source_owner_unbound_baseline")
    lifecycle, publications = phase["lifecycle"], phase["publications"]
    state = body["scheduler_state"]
    running, waiting = tuple(state["running"]), tuple(state["waiting"])
    active = set(running) | set(waiting)
    queued = {w["request_id"] for q in body["queue_oldest_first"] for w in q["work"]}
    ghosts = queued - active
    if (len(running) != len(set(running)) or len(waiting) != len(set(waiting))
            or set(running) & set(waiting) or active != set(state["requests"])
            or active != {rid for rid, row in lifecycle.items() if row["finished"] is None}
            or not queued <= lifecycle.keys()
            or ghosts != set(body["queued_ghost_request_ids"])
            or not lifecycle.keys() <= publications.keys()):
        raise ValueError("source_owner_native_inventory_conflict")
    native = (set(publications) | set(lifecycle)) - baseline
    wire_ids = {}
    for nid in sorted(native):
        match = re.fullmatch(r"chatcmpl-([0-9a-f]{32})-[0-9a-f]{8}", nid)
        if match is None or match[1] in wire_ids:
            raise ValueError("source_owner_native_uuid_join")
        wire_ids[match[1]] = nid
        public = public_memory.get(match[1])
        if not isinstance(public, PublicRequest):
            raise ValueError("source_owner_public_memory_required")
        row = lifecycle.get(nid)
        if row is not None and (
                row["input_tokens"] != public.prompt_tokens or row["max_tokens"] != public.max_tokens):
            raise ValueError("source_owner_native_public_spec_conflict")
    owners = prefix["owners"]
    ids = [row["request_id"] for row in owners]
    if len(ids) != len(set(ids)):
        raise ValueError("source_owner_duplicate_owner")
    joined = {row["native_id"] for row in owners if row["native_id"] is not None}
    released = set(prefix["released_native_ids"])
    if (len(released) != len(prefix["released_native_ids"]) or joined & released
            or joined | released != native):
        raise ValueError("source_owner_prefix_native_inventory_conflict")
    for nid in released:
        if nid not in lifecycle or lifecycle[nid]["finished"] is None:
            raise ValueError("source_owner_released_native_not_finished")
    result = []
    for owner in owners:
        rid, nid = owner["request_id"], owner["native_id"]
        public = public_memory.get(rid)
        if (not isinstance(public, PublicRequest) or nid != wire_ids.get(rid)
                or owner["prompt_tokens"] != public.prompt_tokens
                or owner["max_tokens"] != public.max_tokens):
            raise ValueError("source_owner_public_or_native_join_conflict")
        admitted = number(owner["admission_effective_at"])
        cost, budget = number(owner["cost"], positive=True), number(
            owner["budget_at_admission"], positive=True)
        if not initial_closure.capture_started_at <= admitted <= source:
            raise ValueError("source_owner_admission_outside_closed_capture")
        publication = publications.get(nid)
        interval = None
        life = lifecycle.get(nid)
        if publication is not None:
            interval = (number(publication["put_started_at"]),
                        number(publication["put_finished_at"]))
            if not admitted <= interval[0] <= interval[1] <= source:
                raise ValueError("source_owner_publication_chronology")
        if nid is None:
            kind = "pre_publication"
        elif life is None:
            kind = "published"
        elif nid in running:
            kind = "active"
        elif nid in waiting:
            kind = "waiting"
        elif life["finished"] is not None:
            kind = "issued_ghost" if nid in ghosts else "native_finished"
        else:
            raise ValueError("source_owner_unresolved_native_phase")
        result.append(SourceOwner(
            rid, nid, kind, public.prompt_tokens, public.max_tokens, cost, admitted, budget,
            interval, life["enqueue"] if life else None,
            life["first"] if life else None, life["finished"] if life else None))
    return SourceOwnerPhases(
        source, available, now, expected_clock_domain, phase["run_id"], prefix["run_id"],
        phase["capture"]["trial_id"], prefix["budget"], prefix["budget_version"],
        prefix["closed_through_sequence"], digest(cut), digest(asdict(initial_closure)),
        tuple(result), running, waiting, tuple(sorted(ghosts)),
        tuple(sorted(released)), tuple(sorted(ghosts & released)))
