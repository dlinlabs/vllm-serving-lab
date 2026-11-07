"""Default-off live publication/lifecycle evidence, not a joined owner state."""

import hashlib
import hmac
import inspect
import math
import os
from pathlib import Path
import threading
import time

import input_post_observer
import process_clock_provenance
import queue_observer


_BaseLedger = input_post_observer.InputPostLedger
SCHEMA = "r24-native-live-phase-v1"


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class LiveInputPostLedger(_BaseLedger):
    def __init__(self, *args, **kwargs):
        self.generation = 0
        self.writer_thread = threading.get_ident()
        self.writer_violation = False
        super().__init__(*args, **kwargs)

    def put(self, item, block=True, timeout=None):
        is_add = isinstance(item, tuple) and len(item) == 2 and item[0] == self.add_type
        if not is_add:
            return super().put(item, block=block, timeout=timeout)
        if threading.get_ident() != self.writer_thread:
            self.writer_violation = True
        self.generation += 1
        try:
            return super().put(item, block=block, timeout=timeout)
        finally:
            self.generation += 1


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def phase_snapshot(core):
    from prefill_lab.runtime import ExternalCapture

    ledger = getattr(core, "_pr33_r23_input_post_ledger", None)
    before = ledger.generation if isinstance(ledger, LiveInputPostLedger) else None
    started = time.monotonic()
    result = queue_observer.queue_snapshot(core)
    base = result["base"]
    phase = {
        "schema": SCHEMA, "source_sha256": source_hash(),
        "input_post_source_sha256": input_post_observer.source_hash(),
        "queue_observer_sha256": queue_observer.source_hash(),
        "core_source_sha256": getattr(core, "_pr33_r23_input_post_core_sha256", None),
        "clock_domain": base["clock_domain"], "run_id": base["run_id"],
        "process_identity": None,
        "source_at": started, "capture": None, "lifecycle": {}, "publications": {},
        "writer_generation_before": before, "writer_generation_after": None,
        "queue_schedule_starts": {}, "missing_schedule_starts": [],
        "queue_age_prefix_complete": False, "refusal_reasons": [],
        "native_mpc_permitted": False,
        "scope": (
            "Serialized core lifecycle plus coherently retained input-publication brackets; "
            "all retained publication history is returned without assuming a capture-time "
            "boundary excludes pending inputs. Schedule starts are CPU scheduler entry, "
            "not GPU start/readiness. No Gateway owner join, source-age propagation, "
            "residual ingress model or deployment qualification."
        ),
    }
    reasons = phase["refusal_reasons"]
    try:
        phase["process_identity"] = process_clock_provenance.self_process_metadata()
    except (OSError, ValueError):
        reasons.append("process_identity_unavailable")
    if not result["complete"]:
        reasons.append("incomplete_queue_snapshot")
    if phase["core_source_sha256"] != input_post_observer.CORE_SHA256:
        reasons.append("unreviewed_core_source")
    capture = getattr(core, "_prefill_trial", None)
    if not isinstance(capture, ExternalCapture):
        reasons.append("no_active_external_capture")
    else:
        phase["capture"] = {
            "trial_id": capture.id, "started_at": capture.started, "status": capture.status,
        }
        if (capture.status != "running" or capture.errors or capture.steps.overflow
                or not _finite(capture.started) or capture.started > started):
            reasons.append("invalid_external_capture")
        if len(capture.requests) > input_post_observer.LIMIT:
            reasons.append("lifecycle_retention_limit")
        else:
            for rid, row in capture.requests.items():
                copied = {name: row.get(name) for name in (
                    "input_tokens", "max_tokens", "output_tokens", "enqueue",
                    "first", "finished", "finish_reason",
                )}
                phase["lifecycle"][rid] = copied
                times = [copied[key] for key in ("enqueue", "first", "finished")
                         if copied[key] is not None]
                if (not isinstance(rid, str) or not rid
                        or copied["enqueue"] is None
                        or any(not _finite(t) or t > started for t in times)
                        or times != sorted(times)
                        or any(type(copied[key]) is not int or copied[key] < minimum
                               for key, minimum in (
                                   ("input_tokens", 1), ("max_tokens", 1),
                                   ("output_tokens", 0),
                               ))):
                    reasons.append("invalid_lifecycle_prefix")
            unfinished = {rid for rid, row in phase["lifecycle"].items()
                          if row["finished"] is None}
            if unfinished != set(result["scheduler_state"]["requests"]):
                reasons.append("lifecycle_scheduler_inventory_mismatch")
            queued_ids = {work["request_id"] for batch in result["queue_oldest_first"]
                          for work in batch["work"]}
            if queued_ids - phase["lifecycle"].keys():
                reasons.append("queued_work_outside_capture_lifecycle")
            for batch in result["queue_oldest_first"]:
                if not batch["work"]:
                    continue
                key = batch["step_id"]
                row = capture.inflight.get(key)
                at = row.get("start") if row is not None else None
                if not _finite(at) or at > started:
                    phase["missing_schedule_starts"].append(key)
                else:
                    phase["queue_schedule_starts"][key] = at
            phase["queue_age_prefix_complete"] = not phase["missing_schedule_starts"]
    if not isinstance(ledger, LiveInputPostLedger):
        reasons.append("live_publication_guard_not_installed")
    elif before % 2:
        reasons.append("input_publication_in_flight")
    elif not ledger.lock.acquire(blocking=False):
        reasons.append("input_post_writer_busy")
    else:
        try:
            phase["publications"] = {rid: dict(row) for rid, row in ledger.records.items()}
            if ledger.duplicates or ledger.invalid_adds or ledger.overflow:
                reasons.append("invalid_publication_history")
            for row in phase["publications"].values():
                a, b = row["put_started_at"], row["put_finished_at"]
                if not _finite(a) or not _finite(b) or not a <= b <= started:
                    reasons.append("publication_outside_source_prefix")
            for rid, row in phase["lifecycle"].items():
                publication = phase["publications"].get(rid)
                if publication is None:
                    reasons.append("native_lifecycle_missing_publication")
                elif (_finite(row["enqueue"])
                      and publication["put_started_at"] > row["enqueue"]):
                    reasons.append("publication_after_native_enqueue")
        finally:
            ledger.lock.release()
    if isinstance(ledger, LiveInputPostLedger):
        after = ledger.generation
        phase["writer_generation_after"] = after
        if before != after or after % 2:
            reasons.append("publication_generation_changed")
        if ledger.writer_violation:
            reasons.append("publication_writer_thread_changed")
    phase["refusal_reasons"] = sorted(set(reasons))
    phase["complete"] = not phase["refusal_reasons"]
    phase["sample_finished_at"] = time.monotonic()
    result["native_phase"] = phase
    return result


def register():
    if os.environ.get("PR33_R24_PHASE_OBSERVER_ENABLED") != "1":
        return
    if (os.environ.get("PR33_R23_INPUT_POST_OBSERVER_ENABLED") != "1"
            or os.environ.get("PR33_R23_QUEUE_OBSERVER_ENABLED") != "1"):
        raise ValueError("phase_observer_requires_input_and_queue_observers")
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    input_post_observer.verify_core_abi(EngineCoreProc)
    if getattr(EngineCoreProc, "_pr33_r24_phase_instrumented", False):
        return
    input_post_observer.InputPostLedger = LiveInputPostLedger
    input_post_observer.register()
    EngineCore.pr33_r24_phase_observe = phase_snapshot
    EngineCoreProc._pr33_r24_phase_instrumented = True


class Endpoint:
    name = "pr33_r24_phase_observer"
    required_tasks = ("generate",)

    async def init_state(self, engine_client, state, args):
        if os.environ.get("PR33_R24_PHASE_OBSERVER_ENABLED") != "1":
            return
        if not hasattr(engine_client.engine_core, "call_utility_async"):
            raise RuntimeError("unsupported_native_utility_channel")
        self.engine = engine_client

    def attach_router(self, app):
        if os.environ.get("PR33_R24_PHASE_OBSERVER_ENABLED") != "1":
            return
        from fastapi import HTTPException, Request

        @app.get("/pr33-native/phase-state")
        async def state(request: Request):
            secret = os.environ.get("PR33_NATIVE_OBSERVER_KEY", "")
            if len(secret) < 32 or not hmac.compare_digest(
                    request.headers.get("x-observer-key", ""), secret):
                raise HTTPException(401, "Observer credential required")
            result = await self.engine.engine_core.call_utility_async("pr33_r24_phase_observe")
            count = self.engine.get_num_unfinished_requests()
            if inspect.isawaitable(count):
                count = await count
            if type(count) is not int or count < 0:
                raise HTTPException(503, "Unavailable frontend count")
            base = result["base"]
            base["frontend_unfinished"] = count
            base["frontend_sampled_at"] = time.monotonic()
            base["orphaned"] = (
                0 if base["running"] == base["waiting"] == base["pending_batches"] == count == 0
                else None
            )
            return result
