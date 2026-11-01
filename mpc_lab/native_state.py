"""Source-bound reported state; token headroom is not remaining EOS work."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Identity, Snapshot, finite
from .runtime import snapshot_from_dict


@dataclass(frozen=True)
class NativeBinding:
    clock_domain: str
    gateway_run_id: str
    native_run_id: str
    observer_sha256: str
    identity: Identity
    vllm_version: str = "0.30.0"


@dataclass(frozen=True)
class NativeProgress:
    engine_request_id: str
    prompt_tokens: int
    computed_tokens: int
    output_tokens: int
    maximum_output_tokens: int


@dataclass(frozen=True)
class NativeState:
    gateway: Snapshot
    sampled_at: float
    received_at: float
    running: int
    waiting: int
    pending_batches: int
    frontend_unfinished: int
    kv_cache_usage: float
    progress: tuple[NativeProgress, ...]

    @property
    def visible_prefill_remaining(self) -> int:
        return sum(max(0, p.prompt_tokens - p.computed_tokens) for p in self.progress)

    @property
    def visible_output_headroom(self) -> int:
        return sum(max(0, p.maximum_output_tokens - p.output_tokens) for p in self.progress)


@dataclass(frozen=True)
class NativeEstimate:
    state: NativeState | None
    reason: str | None


def estimate_native_state(
    frame: object, binding: NativeBinding, *, now: float,
    max_age_s: float = 0.5, max_frontend_width_s: float = 0.05,
) -> NativeEstimate:
    """Do not join different request namespaces or invent waiting-request work."""
    def refuse(reason: str) -> NativeEstimate:
        return NativeEstimate(None, reason)

    if not isinstance(frame, dict) or not isinstance(frame.get("gateway_snapshot"), dict):
        return refuse("native_state_schema")
    gateway = frame["gateway_snapshot"]
    version = frame.get("vllm_version")
    if (frame.get("clock_domain") != binding.clock_domain
            or gateway.get("clock_domain") != binding.clock_domain):
        return refuse("clock_domain_mismatch")
    if (frame.get("run_id") != binding.native_run_id
            or gateway.get("run_id") != binding.gateway_run_id
            or frame.get("observer_sha256") != binding.observer_sha256
            or frame.get("schema") != "pr33-native-read-only-v1"
            or not isinstance(version, str) or version.partition("+")[0] != binding.vllm_version):
        return refuse("native_source_mismatch")
    if (not finite(now) or now < 0 or not finite(max_age_s, positive=True)
            or not finite(max_frontend_width_s, positive=True)):
        return refuse("invalid_native_cut")
    try:
        snapshot = snapshot_from_dict(gateway)
        if snapshot.identity != binding.identity:
            return refuse("native_identity_mismatch")
        if frame["complete"] is not True or not snapshot.complete:
            return refuse("incomplete_native_state")
        sampled, frontend = frame["sampled_at"], frame["frontend_sampled_at"]
        received, gateway_received = frame["received_at"], frame["gateway_received_at"]
        if (any(not finite(v) for v in (sampled, frontend, received, gateway_received))
                or not 0 <= sampled <= frontend <= received <= now
                or not snapshot.sampled_at <= gateway_received <= received
                or now - sampled > max_age_s or now - snapshot.sampled_at > max_age_s
                or frontend - sampled > max_frontend_width_s):
            return refuse("native_state_timing")
        counts = [frame[k] for k in (
            "running", "waiting", "pending_batches", "frontend_unfinished",
        )]
        usage = frame["kv_cache_usage"]
        if (any(type(v) is not int or v < 0 or not finite(v) for v in counts)
                or not finite(usage) or not 0 <= usage <= 1):
            return refuse("native_state_values")
        rows = frame["requests"]
        if not isinstance(rows, list) or len(rows) != counts[0] or len(rows) > 512:
            return refuse("native_running_inventory")
        progress, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                return refuse("native_progress_values")
            request_id = row["request_id"]
            values = [row[k] for k in (
                "prompt_tokens", "computed_tokens", "output_tokens", "maximum_output_tokens",
            )]
            if (not isinstance(request_id, str) or not request_id or request_id in seen
                    or any(type(v) is not int or v < 0 or not finite(v) for v in values)
                    or values[-1] == 0):
                return refuse("native_progress_values")
            seen.add(request_id)
            progress.append(NativeProgress(request_id, *values))
        return NativeEstimate(NativeState(
            snapshot, sampled, received, *counts, usage, tuple(progress),
        ), None)
    except (KeyError, ValueError, TypeError, OverflowError):
        return refuse("invalid_native_state")
