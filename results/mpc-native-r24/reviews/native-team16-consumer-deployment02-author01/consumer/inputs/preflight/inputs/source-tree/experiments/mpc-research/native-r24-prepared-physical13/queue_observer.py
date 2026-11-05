"""Default-off engine-thread queue observation without inspecting any future."""

import hashlib
import hmac
import inspect
import os
from pathlib import Path
import time


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def queue_snapshot(core):
    from prefill_lab.contracts import scheduled_work, step_key, wire_value
    from prefill_lab.runtime import snapshot
    from pr33_native_observer import observe

    started = time.monotonic()
    base = observe(core)
    scheduler = core.scheduler
    state = snapshot(scheduler)
    queue = []
    for entry in reversed(core.batch_queue if core.batch_queue is not None else ()):
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise ValueError("unreviewed_native_queue_entry")
        output = entry[1]
        if (getattr(output, "preempted_req_ids", None)
                or getattr(output, "scheduled_spec_decode_tokens", None)
                or getattr(output, "pending_structured_output_tokens", False)):
            raise ValueError("unsupported_queued_preemption_speculation_or_structure")
        queue.append({"step_id": step_key(output), "work": scheduled_work(output)})
    reasons = []
    skipped = [r.request_id for r in scheduler.skipped_waiting]
    streaming = scheduler.num_waiting_for_streaming_input
    if skipped or streaming:
        reasons.append("skipped_or_streaming_waiting_outside_scope")
    if core.batch_queue is None or core.batch_queue_size != 2 or len(queue) > 2:
        reasons.append("reviewed_two_slot_queue_required")
    if len(state["requests"]) > 512:
        reasons.append("scheduler_inventory_limit")
    represented = state["running"] + state["waiting"] + skipped
    if (len(represented) != len(set(represented))
            or set(represented) != set(state["requests"])):
        reasons.append("scheduler_inventory_not_fully_represented")
    if base["pending_batches"] != len(queue):
        reasons.append("native_queue_inventory_mismatch")
    ghost_ids = sorted({w["request_id"] for q in queue for w in q["work"]}
                       - set(state["requests"]))
    return wire_value({
        "schema": "r23-native-queue-read-only-v1", "queue_observer_sha256": source_hash(),
        "base": base, "scheduler_state": state, "queue_oldest_first": queue,
        "queued_ghost_request_ids": ghost_ids, "skipped_waiting": skipped,
        "streaming_waiting": streaming, "sample_started_at": started,
        "sample_finished_at": time.monotonic(), "complete": not reasons and base["complete"],
        "refusal_reasons": reasons,
        "scope": "Engine-thread scheduler and issued work only; futures are neither queried nor awaited; no remaining GPU time, exclusive service or exact warm-state qualification",
        "native_mpc_permitted": False,
    })


def register():
    prepared = os.environ.get("PR33_R24_PREPARED_FORWARD_ENABLED", "0")
    copy_ready = os.environ.get("PR33_R24_PREPARED_COPY_READY_ENABLED", "0")
    if copy_ready not in ("0", "1"):
        raise ValueError("prepared_copy_boolean_launch_flag_required")
    if copy_ready == "1" and prepared != "1":
        raise ValueError("prepared_copy_forward_parent_required")
    if prepared not in ("0", "1"):
        raise ValueError("prepared_worker_boolean_launch_flag_required")
    if os.environ.get("PR33_R23_QUEUE_OBSERVER_ENABLED") != "1":
        if prepared == "1":
            raise ValueError("prepared_worker_queue_parent_required")
        return
    if prepared == "1":
        from prepared_forward_plugin import register as register_prepared
        register_prepared()
    if copy_ready == "1":
        from prepared_copy_ready_plugin import register as register_copy
        register_copy()
    from prefill_lab.plugin import installed_sources
    from vllm.v1.engine.core import EngineCore

    installed_sources()
    EngineCore.pr33_r23_queue_observe = queue_snapshot


class Endpoint:
    name = "pr33_r23_queue_observer"
    required_tasks = ("generate",)

    async def init_state(self, engine_client, state, args):
        if os.environ.get("PR33_R23_QUEUE_OBSERVER_ENABLED") != "1":
            return
        if not hasattr(engine_client.engine_core, "call_utility_async"):
            raise RuntimeError("unsupported_native_utility_channel")
        self.engine = engine_client

    def attach_router(self, app):
        if os.environ.get("PR33_R23_QUEUE_OBSERVER_ENABLED") != "1":
            return
        from fastapi import HTTPException, Request

        @app.get("/pr33-native/queue-state")
        async def state(request: Request):
            secret = os.environ.get("PR33_NATIVE_OBSERVER_KEY", "")
            if len(secret) < 32 or not hmac.compare_digest(
                    request.headers.get("x-observer-key", ""), secret):
                raise HTTPException(401, "Observer credential required")
            result = await self.engine.engine_core.call_utility_async("pr33_r23_queue_observe")
            count = self.engine.get_num_unfinished_requests()
            if inspect.isawaitable(count):
                count = await count
            if type(count) is not int or count < 0:
                raise HTTPException(503, "Unavailable frontend count")
            result["base"]["frontend_unfinished"] = count
            result["base"]["frontend_sampled_at"] = time.monotonic()
            base = result["base"]
            base["orphaned"] = (
                0 if base["running"] == base["waiting"] == base["pending_batches"] == count == 0
                else None
            )
            return result
