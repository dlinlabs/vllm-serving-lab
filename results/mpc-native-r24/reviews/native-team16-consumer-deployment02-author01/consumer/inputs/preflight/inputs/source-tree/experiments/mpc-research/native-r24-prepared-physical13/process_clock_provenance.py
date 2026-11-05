"""Read-only owned-process clock proof under trusted host/endpoint provenance."""

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import time


CLOCK_SOURCE_SHA256 = "cbbd1f338e0d078d5c5eacdfbcec725dcfe2ed9b46c1ee1ed68ad2f26de24e1f"
BOOT = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
MAX_ANCESTRY = 32


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise ValueError("process_clock_positive_integer_required")
    return value


def ticks(value):
    if isinstance(value, str) and re.fullmatch("[0-9]{1,20}", value):
        value = int(value)
    return positive_int(value)


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("process_clock_invalid_time")
    return value


def domain_parts(domain):
    match = re.fullmatch(rf"linux-monotonic:({BOOT}):(time:\[[0-9]+\])", domain) if isinstance(domain, str) else None
    if match is None:
        raise ValueError("process_clock_kernel_domain_required")
    return match.groups()


@dataclass(frozen=True)
class Process:
    pid: int
    parent_pid: int
    process_group: int
    session: int
    start_ticks: int
    time_namespace: str
    pid_namespace: str


def parse_stat(raw, pid):
    """Field22 follows the last ')' even when comm contains spaces or ')'."""
    try:
        prefix, fields = raw[:raw.index("(")].strip(), raw[raw.rindex(")") + 1:].split()
        if (prefix != str(positive_int(pid)) or len(fields) < 20
                or fields[0] not in {"R", "S", "D", "T", "t", "I", "W", "P"}):
            raise ValueError
        parent, group, session, start = (int(fields[i]) for i in (1, 2, 3, 19))
        if parent < 0 or any(value <= 0 for value in (group, session, start)):
            raise ValueError
        return pid, parent, group, session, start
    except (ValueError, IndexError) as exc:
        raise ValueError("process_clock_invalid_or_dead_proc_stat") from exc


class LinuxProc:
    """A trusted host procfs view; alternate roots/self_pid are for pure tests."""
    def __init__(self, root="/proc", *, self_pid=None):
        self.root = Path(root)
        self.self_pid = os.getpid() if self_pid is None else positive_int(self_pid)
        if self_pid is None and os.readlink(self.root / "self") != str(self.self_pid):
            raise ValueError("process_clock_procfs_pid_view_mismatch")

    def text(self, path):
        with path.open() as stream:
            raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("process_clock_proc_record_capacity")
        return raw

    def boot_id(self):
        boot = self.text(self.root / "sys/kernel/random/boot_id").strip()
        if re.fullmatch(BOOT, boot) is None:
            raise ValueError("process_clock_invalid_boot")
        return boot

    def read(self, pid):
        path = self.root / str(positive_int(pid))
        first = parse_stat(self.text(path / "stat"), pid)
        time_ns = os.readlink(path / "ns/time")
        pid_ns = os.readlink(path / "ns/pid")
        second = parse_stat(self.text(path / "stat"), pid)
        if first != second:
            raise ValueError("process_clock_proc_identity_changed_during_read")
        if (re.fullmatch(r"time:\[[0-9]+\]", time_ns) is None
                or re.fullmatch(r"pid:\[[0-9]+\]", pid_ns) is None):
            raise ValueError("process_clock_invalid_namespace")
        return Process(*first, time_ns, pid_ns)


def startup_binding(receipt, role):
    """Normalize existing owned_engine ready.json, not a caller-supplied epoch."""
    if (not isinstance(receipt, dict) or receipt.get("native_ready") is not True
            or receipt.get("cleanup_completed") is not False
            or "finished_at" in receipt or receipt.get("returncode") is not None):
        raise ValueError("process_clock_live_owned_startup_required")
    pid, start = positive_int(receipt["pid"]), ticks(receipt["start_ticks"])
    if positive_int(receipt.get("process_group")) != pid:
        raise ValueError("process_clock_owned_session_required")
    readiness = receipt["readiness"]
    if role == "gateway":
        frames = [readiness["snapshot"]]
        if (readiness.get("gateway_run_id") != frames[0].get("run_id")
                or readiness.get("clock_domain") != frames[0].get("clock_domain")):
            raise ValueError("process_clock_gateway_readiness_conflict")
    elif role == "native":
        frames = readiness["frames"]
    else:
        raise ValueError("process_clock_unknown_role")
    if not isinstance(frames, list) or not frames:
        raise ValueError("process_clock_readiness_frames_required")
    epoch, domain = frames[0]["run_id"], frames[0]["clock_domain"]
    if not isinstance(epoch, str) or not epoch:
        raise ValueError("process_clock_epoch_required")
    domain_parts(domain)
    if any(frame.get("run_id") != epoch or frame.get("clock_domain") != domain
           or frame.get("complete") is not True for frame in frames):
        raise ValueError("process_clock_readiness_epoch_or_domain_conflict")
    return pid, start, epoch, domain


def metadata_identity(document):
    if not isinstance(document, dict) or set(document) != {"pid", "start_ticks", "pid_namespace"}:
        raise ValueError("process_clock_core_process_metadata_required")
    pid, start = positive_int(document["pid"]), ticks(document["start_ticks"])
    namespace = document["pid_namespace"]
    if not isinstance(namespace, str) or re.fullmatch(r"pid:\[[0-9]+\]", namespace) is None:
        raise ValueError("process_clock_core_pid_namespace_required")
    return pid, start, namespace


def self_process_metadata(proc=None):
    """Minimal future core-utility field; existing clock_domain supplies boot/time."""
    proc = LinuxProc() if proc is None else proc
    process = proc.read(proc.self_pid)
    return dict(pid=process.pid, start_ticks=process.start_ticks,
                pid_namespace=process.pid_namespace)


def inspect_chain(proc, leader_pid, leader_start, target_pid, boot, time_namespace):
    if proc.boot_id() != boot:
        raise ValueError("process_clock_wrong_host_or_boot")
    collector = proc.read(proc.self_pid)
    if collector.time_namespace != time_namespace:
        raise ValueError("process_clock_collector_namespace")
    chain, seen = [], set()
    pid = target_pid
    for _ in range(MAX_ANCESTRY):
        if pid in seen or pid <= 0:
            raise ValueError("process_clock_not_owned_descendant")
        seen.add(pid)
        process = proc.read(pid)
        if (process.time_namespace != time_namespace
                or process.pid_namespace != collector.pid_namespace):
            raise ValueError("process_clock_namespace_mismatch")
        if process.process_group != leader_pid or process.session != leader_pid:
            raise ValueError("process_clock_wrong_owned_session")
        if chain and process.start_ticks > chain[-1].start_ticks:
            raise ValueError("process_clock_parent_identity_conflict")
        chain.append(process)
        if pid == leader_pid:
            if process.start_ticks != leader_start:
                raise ValueError("process_clock_leader_pid_reused")
            return collector, tuple(chain)
        pid = process.parent_pid
    raise ValueError("process_clock_ancestry_capacity")


def prove_owned_process_clock(
    startup_receipt, *, role, acquire, native_process=None, proc=None,
    clock=time.monotonic, max_age_s=0.5,
):
    """Bracket a fresh authenticated acquisition with trusted host procfs reads.

    Startup receipts, direct endpoint authentication and installed source receipts
    must be trusted by the caller. No hostile kernel/procfs or transient namespace
    switching/ABA is covered. Gateway routing assumes its owned single-process
    launcher serves the endpoint; native core identity is never guessed.
    """
    finite(max_age_s)
    if max_age_s == 0:
        raise ValueError("process_clock_positive_age_required")
    source = Path(__file__).resolve().parents[3] / "prefill_lab/live_http.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != CLOCK_SOURCE_SHA256:
        raise ValueError("process_clock_unreviewed_clock_source")
    leader_pid, leader_start, epoch, domain = startup_binding(startup_receipt, role)
    boot, time_ns = domain_parts(domain)
    if role == "native":
        target_pid, target_start, target_pid_ns = metadata_identity(native_process)
    else:
        if native_process is not None:
            raise ValueError("process_clock_unexpected_native_metadata")
        target_pid, target_start, target_pid_ns = leader_pid, leader_start, None
    proc = LinuxProc() if proc is None else proc
    started = finite(clock())
    before = inspect_chain(proc, leader_pid, leader_start, target_pid, boot, time_ns)
    target = before[1][0]
    if (target.start_ticks != target_start
            or target_pid_ns is not None and target.pid_namespace != target_pid_ns):
        raise ValueError("process_clock_core_identity_mismatch")
    sent = finite(clock())
    frame = acquire()
    received = finite(clock())
    after = inspect_chain(proc, leader_pid, leader_start, target_pid, boot, time_ns)
    if before != after or proc.boot_id() != boot:
        raise ValueError("process_clock_identity_or_namespace_changed")
    if (not isinstance(frame, dict) or frame.get("run_id") != epoch
            or frame.get("clock_domain") != domain):
        raise ValueError("process_clock_live_epoch_or_domain_mismatch")
    if role == "native":
        if metadata_identity(frame.get("process_identity")) != (
                target_pid, target_start, target_pid_ns):
            raise ValueError("process_clock_reply_core_identity_changed")
        source_at = finite(frame.get("source_at"))
        source_finished = finite(frame.get("sample_finished_at"))
    else:
        source_at = source_finished = finite(frame.get("sampled_at"))
    finished = finite(clock())
    if not started <= sent <= source_at <= source_finished <= received <= finished:
        raise ValueError("process_clock_acquisition_chronology")
    if finished - source_at > max_age_s:
        raise ValueError("process_clock_stale_after_validation")
    return {
        "schema": "r24-owned-process-clock-proof-v1", "role": role,
        "run_id": epoch, "clock_domain": domain, "boot_id": boot,
        "clock_source_sha256": CLOCK_SOURCE_SHA256,
        "collector": asdict(after[0]), "owned_chain": [asdict(p) for p in after[1]],
        "checked_from": started, "request_sent_at": sent,
        "source_at": source_at, "sample_finished_at": source_finished,
        "received_at": received, "validated_at": finished,
        "source_age_s": finished - source_at,
        "endpoint_pid_binding": (
            "authenticated_core_metadata" if role == "native"
            else "trusted_owned_single_process_gateway"),
        "process_clock_verified_under_trust_model": True,
        "capture_verified": False, "owner_inventory_verified": False,
        "native_mpc_permitted": False,
        "remaining_obligations": [
            "Caller-authenticated startup, endpoint and installed-source provenance",
            "No unobserved transient namespace/identity ABA during acquisition",
            "Native complete capture, initial-drain and common-cut inventory closure",
            "Recheck proof compatibility and age through later planning/dispatch",
        ],
    }


def join_process_clocks(gateway, native, *, now, max_age_s):
    """Join only trusted local helper outputs; this is not signature verification."""
    finite(now)
    finite(max_age_s)
    if max_age_s == 0:
        raise ValueError("process_clock_positive_age_required")
    for proof, role in ((gateway, "gateway"), (native, "native")):
        if (proof.get("schema") != "r24-owned-process-clock-proof-v1"
                or proof.get("role") != role
                or proof.get("process_clock_verified_under_trust_model") is not True
                or proof.get("clock_source_sha256") != CLOCK_SOURCE_SHA256):
            raise ValueError("process_clock_local_proofs_required")
        if (not finite(proof["source_at"]) <= finite(proof["validated_at"]) <= now
                or now - proof["source_at"] > max_age_s):
            raise ValueError("process_clock_join_stale_or_unordered")
    if any(gateway[key] != native[key] for key in ("boot_id", "clock_domain", "collector")):
        raise ValueError("process_clock_join_host_namespace_or_collector_mismatch")
    return {
        "schema": "r24-owned-common-clock-proof-v1",
        "clock_domain": gateway["clock_domain"],
        "gateway_run_id": gateway["run_id"], "native_run_id": native["run_id"],
        "validated_at": now,
        "oldest_source_age_s": now - min(gateway["source_at"], native["source_at"]),
        "clock_process_binding_only": True, "capture_verified": False,
        "owner_inventory_verified": False, "native_mpc_permitted": False,
    }
