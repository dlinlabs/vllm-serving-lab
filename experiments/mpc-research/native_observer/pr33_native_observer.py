"""Opt-in read-only vLLM0.30 utility observer; no scheduler/kernel replacement."""
import hashlib
import hmac
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import re
import time
import uuid

from prefill_lab.live_http import clock_domain


def sampling_ignore_eos(request):
    value = getattr(getattr(request, "sampling_params", None), "ignore_eos", None)
    return value if type(value) is bool else None


def sampling_identifier(request, key):
    extra = getattr(getattr(request, "sampling_params", None), "extra_args", None)
    value = extra.get(key) if isinstance(extra, dict) else None
    return value if isinstance(value, str) and re.fullmatch("[0-9a-f]{32}", value) else None


def observe(core):
    scheduler = core.scheduler
    requests = list(scheduler.running)
    waiting = len(scheduler.waiting) + len(scheduler.skipped_waiting)
    streaming = scheduler.num_waiting_for_streaming_input
    if not hasattr(core, "_pr33_observer_run"):
        core._pr33_observer_run = uuid.uuid4().hex
    usage = float(scheduler.kv_cache_manager.usage)
    if not math.isfinite(usage) or not 0 <= usage <= 1:
        raise ValueError("invalid_native_kv_usage")
    if type(streaming) is not int or streaming < 0:
        raise ValueError("invalid_native_streaming_count")
    return {
        "schema": "pr33-native-read-only-v1",
        "run_id": core._pr33_observer_run,
        "clock_domain": clock_domain(),
        "sampled_at": time.monotonic(),
        "running": len(requests),
        "waiting": waiting + streaming,
        "pending_batches": len(core.batch_queue) if core.batch_queue is not None else 0,
        "kv_cache_usage": usage,
        "complete": len(requests) <= 512,
        "requests": [
            {"request_id": r.request_id, "prompt_tokens": r.num_prompt_tokens,
             "computed_tokens": r.num_computed_tokens, "output_tokens": r.num_output_tokens,
             "maximum_output_tokens": r.max_tokens, "ignore_eos": sampling_ignore_eos(r),
             "cohort_id": sampling_identifier(r, "pr33_cohort_id"),
             "gateway_request_id": sampling_identifier(r, "pr33_gateway_request_id")}
            for r in requests[:512]
        ],
        "vllm_version": importlib.metadata.version("vllm"),
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def register():
    if os.environ.get("PR33_NATIVE_OBSERVER_ENABLED") != "1":
        return
    if importlib.metadata.version("vllm").split("+")[0] != "0.30.0":
        raise RuntimeError("Observer requires reviewed vLLM0.30.0")
    from prefill_lab.plugin import installed_sources
    installed_sources()
    from vllm.v1.engine.core import EngineCore
    EngineCore.pr33_native_observe = observe


class Endpoint:
    name = "pr33_native_observer"
    required_tasks = ("generate",)

    async def init_state(self, engine_client, state, args):
        if os.environ.get("PR33_NATIVE_OBSERVER_ENABLED") != "1":
            return
        if not hasattr(engine_client.engine_core, "call_utility_async"):
            raise RuntimeError("Unsupported native utility channel")
        self.engine = engine_client

    def attach_router(self, app):
        if os.environ.get("PR33_NATIVE_OBSERVER_ENABLED") != "1":
            return
        from fastapi import HTTPException, Request

        @app.get("/pr33-native/state")
        async def state(request: Request):
            secret = os.environ.get("PR33_NATIVE_OBSERVER_KEY", "")
            if len(secret) < 32 or not hmac.compare_digest(
                request.headers.get("x-observer-key", ""), secret,
            ):
                raise HTTPException(401, "Observer credential required")
            result = await self.engine.engine_core.call_utility_async("pr33_native_observe")
            count = self.engine.get_num_unfinished_requests()
            if inspect.isawaitable(count):
                count = await count
            if type(count) is not int or count < 0:
                raise HTTPException(503, "Unavailable frontend count")
            result["frontend_unfinished"] = count
            result["frontend_sampled_at"] = time.monotonic()
            result["orphaned"] = (
                0 if result["running"] == result["waiting"] == result["pending_batches"] == count == 0
                else None
            )
            return result
