"""Default-off brackets around core input publication, never future readiness."""

import functools
import hashlib
import hmac
import inspect
import os
from pathlib import Path
import queue
import threading
import time


LIMIT = 8192
CORE_SHA256 = "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5"


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class InputPostLedger:
    def __init__(self, input_queue, add_type, *, limit=LIMIT):
        if not isinstance(input_queue, queue.Queue) or input_queue.maxsize != 0:
            raise ValueError("reviewed_unbounded_core_input_queue_required")
        if type(limit) is not int or limit < 1:
            raise ValueError("positive_input_post_retention_limit_required")
        self.add_type, self.limit = add_type, limit
        self.records, self.duplicates = {}, set()
        self.invalid_adds, self.overflow = 0, False
        self.lock = threading.Lock()
        self.original_put = input_queue.put
        input_queue.put = self.put

    def put(self, item, block=True, timeout=None):
        is_add = isinstance(item, tuple) and len(item) == 2 and item[0] == self.add_type
        rid = None
        if is_add:
            value = item[1]
            if isinstance(value, tuple) and len(value) == 2:
                rid = getattr(value[0], "request_id", None)
            if not isinstance(rid, str) or not rid:
                with self.lock:
                    self.invalid_adds += 1
                rid = None
        started = time.monotonic() if rid is not None else None
        result = self.original_put(item, block=block, timeout=timeout)
        if rid is not None:
            finished = time.monotonic()
            with self.lock:
                if rid in self.records:
                    self.duplicates.add(rid)
                elif len(self.records) >= self.limit:
                    self.overflow = True
                else:
                    self.records[rid] = {"put_started_at": started, "put_finished_at": finished}
        return result

    def selected(self, ids):
        if (not isinstance(ids, list) or len(ids) > self.limit
                or any(not isinstance(rid, str) or not rid for rid in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("bounded_unique_native_ids_required")
        if not self.lock.acquire(blocking=False):
            return {"complete": False, "refusal": "input_post_writer_busy", "records": {}}
        try:
            missing = sorted(set(ids) - self.records.keys())
            duplicated = sorted(set(ids) & self.duplicates)
            return {
                "complete": not missing and not duplicated and not self.overflow and not self.invalid_adds,
                "records": {rid: dict(self.records[rid]) for rid in ids if rid in self.records},
                "missing_ids": missing, "duplicate_ids": duplicated,
                "overflow": self.overflow, "invalid_adds": self.invalid_adds,
                "retention_limit": self.limit,
            }
        finally:
            self.lock.release()


def records(core, ids):
    from pr33_native_observer import observe
    base = observe(core)
    if not base["complete"] or base["running"] or base["waiting"] or base["pending_batches"]:
        raise ValueError("input_post_export_requires_native_drain")
    ledger = getattr(core, "_pr33_r23_input_post_ledger", None)
    if not isinstance(ledger, InputPostLedger):
        raise ValueError("input_post_receiver_not_instrumented")
    return {
        "schema": "r23-native-input-post-records-v1", "source_sha256": source_hash(),
        "clock_domain": base["clock_domain"], "run_id": base["run_id"],
        "observer_sha256": base["observer_sha256"],
        "core_source_sha256": core._pr33_r23_input_post_core_sha256,
        "sampled_at": base["sampled_at"], **ledger.selected(ids),
        "scope": "Input queue publication lies inside each host bracket; no exact point timestamp, future/CUDA readiness, transport calibration or MPC qualification",
        "native_mpc_permitted": False,
    }


def verify_core_abi(core_class):
    source = inspect.getsourcefile(core_class)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != CORE_SHA256:
        raise ValueError("unreviewed_input_post_core_source")
    return CORE_SHA256


def register():
    if os.environ.get("PR33_R23_INPUT_POST_OBSERVER_ENABLED") != "1":
        return
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from prefill_lab.plugin import installed_sources
    installed_sources()
    core_sha = verify_core_abi(EngineCoreProc)
    if getattr(EngineCoreProc, "_pr33_r23_input_post_instrumented", False):
        return
    original = EngineCoreProc.process_input_sockets

    @functools.wraps(original)
    def process_inputs(core, *args, **kwargs):
        core._pr33_r23_input_post_ledger = InputPostLedger(core.input_queue, EngineCoreRequestType.ADD)
        core._pr33_r23_input_post_core_sha256 = core_sha
        return original(core, *args, **kwargs)

    EngineCoreProc.process_input_sockets = process_inputs
    EngineCore.pr33_r23_input_post_records = records
    EngineCoreProc._pr33_r23_input_post_instrumented = True


class Endpoint:
    name = "pr33_r23_input_post_observer"
    required_tasks = ("generate",)

    async def init_state(self, engine_client, state, args):
        if os.environ.get("PR33_R23_INPUT_POST_OBSERVER_ENABLED") != "1":
            return
        if not hasattr(engine_client.engine_core, "call_utility_async"):
            raise RuntimeError("unsupported_native_utility_channel")
        self.engine = engine_client

    def attach_router(self, app):
        if os.environ.get("PR33_R23_INPUT_POST_OBSERVER_ENABLED") != "1":
            return
        from fastapi import HTTPException, Request

        @app.post("/pr33-native/input-post-records")
        async def snapshot(request: Request):
            secret = os.environ.get("PR33_NATIVE_OBSERVER_KEY", "")
            if len(secret) < 32 or not hmac.compare_digest(
                    request.headers.get("x-observer-key", ""), secret):
                raise HTTPException(401, "Observer credential required")
            body = await request.json()
            if (not isinstance(body, dict) or set(body) != {"request_ids"}
                    or not isinstance(body["request_ids"], list) or len(body["request_ids"]) > LIMIT
                    or any(not isinstance(rid, str) or not rid for rid in body["request_ids"])
                    or len(body["request_ids"]) != len(set(body["request_ids"]))):
                raise HTTPException(400, "Bounded unique native IDs required")
            count = self.engine.get_num_unfinished_requests()
            if inspect.isawaitable(count):
                count = await count
            if type(count) is not int or count != 0:
                raise HTTPException(409, "Frontend drain required")
            return await self.engine.engine_core.call_utility_async(
                "pr33_r23_input_post_records", body["request_ids"])
