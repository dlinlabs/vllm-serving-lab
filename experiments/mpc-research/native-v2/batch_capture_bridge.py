"""Default-off, TRAIN-only facade over the existing PR32 passive capture."""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_plan(plan):
    if (plan.get("schema") != "native-v2-batch-capture-plan-v1"
            or plan.get("partition") != "train"
            or not re.fullmatch(r"[0-9a-f]{40}", plan.get("source_commit", ""))
            or not isinstance(plan.get("trials"), list)
            or not 1 <= len(plan["trials"]) <= 512):
        raise ValueError("A frozen TRAIN capture plan is required")
    trials = {}
    for row in plan["trials"]:
        tid = row.get("trial_id", "")
        if (not re.fullmatch(r"nv2-[a-z0-9-]{1,100}", tid) or tid in trials
                or row.get("observer") not in ("off", "light")
                or type(row.get("seed")) is not int
                or not 170001 <= row["seed"] <= 170012):
            raise ValueError("Invalid,duplicate or non-TRAIN capture trial")
        trials[tid] = row["observer"]
    return trials


def capture_errors(capture, observer):
    errors = list(capture.get("errors", []))
    if (capture.get("schema") != "prefill-capture-v1"
            or capture.get("status") != "complete"
            or capture.get("overflow", 0) or capture.get("pending_step_ids")):
        errors.append("engine_capture_incomplete")
    workers = capture.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        return errors + ["single_worker_capture_required"]
    worker = workers[0]
    if worker.get("pending") or worker.get("overflow"):
        errors.append("worker_capture_incomplete")
    steps = capture.get("steps", [])
    rows = worker.get("rows", [])
    engine_ids = [r["step_id"] for r in steps]
    worker_ids = [r["step_id"] for r in rows]
    if (len(engine_ids) != len(set(engine_ids))
            or len(worker_ids) != len(set(worker_ids))
            or set(engine_ids) != set(worker_ids)):
        errors.append("request_batch_identity_mismatch")
    for row in rows:
        duration = row.get("gpu_forward_s")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0 or row.get("gpu_error")):
            errors.append("unresolved_gpu_forward_span")
    if observer == "off" and (steps or rows):
        errors.append("off_capture_contains_batch_instrumentation")
    if observer == "light" and capture.get("requests") and not steps:
        errors.append("missing_observed_batches")
    if any(r.get("finished") is None for r in capture.get("requests", {}).values()):
        errors.append("missing_request_terminal")
    return errors


class BatchCapture:
    def __init__(self, core, plan, command, *, clock=time.monotonic, domain=None):
        self.core, self.command, self.clock = core, command, clock
        self.trials = validate_plan(plan)
        self.plan_hash = hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.plan = plan
        self.domain = domain
        self.used, self.active, self.invalid = set(), None, False
        self.qualified = False
        self.run_id, self.worker_identity = uuid.uuid4().hex, None
        self.completed = None

    def invoke(self, payload):
        if (not isinstance(payload, dict)
                or set(payload) - {"operation", "trial_id", "capture_sha256"}
                or payload.get("operation") not in ("metadata", "begin", "end", "status", "ack")):
            raise ValueError("Only passive capture operations are supported")
        op = payload["operation"]
        allowed = {"operation", "trial_id", "capture_sha256"} if op == "ack" else (
            {"operation", "trial_id"} if op == "begin" else {"operation"})
        if set(payload) - allowed:
            raise ValueError("Unexpected capture command fields")
        common = dict(schema="native-v2-batch-capture-v1",
                      bridge_sha256=source_hash(), plan_sha256=self.plan_hash,
                      source_commit=self.plan["source_commit"],
                      run_id=self.run_id, clock_domain=self.domain,
                      command_started_at=self.clock(),
                      measurement_scope="Current-stream preparation/forward CUDA span;"
                        "engine/worker host timestamps. Not kernel busy or whole-pipeline GPU service.")
        if op == "metadata":
            if self.active is not None or self.completed is not None:
                raise ValueError("Metadata qualification requires idle acknowledged capture state")
            self.qualified = False
            result = self.command(self.core, {"operation": "meta"})
            clocks = self.core.model_executor.collective_rpc("pr33_native_capture_clock")
            if (result.get("tensor_parallel_size") != 1
                    or result.get("pipeline_parallel_size") != 1
                    or len(clocks) != 1
                    or any(c.get("clock_domain") != self.domain
                           or c.get("bridge_sha256") != source_hash()
                           or type(c.get("process_id")) is not int
                           or c["process_id"] <= 0 for c in clocks)):
                raise ValueError("Single-worker source/clock qualification failed")
            identity = [(c["process_id"], c["clock_domain"], c["bridge_sha256"]) for c in clocks]
            if self.worker_identity is not None and identity != self.worker_identity:
                self.invalid = True
                raise ValueError("Worker source/clock/epoch changed")
            self.worker_identity = identity
            self.qualified = True
            return {**common, "runtime": result, "worker_clocks": clocks}
        if op == "status":
            return {**common, "active_trial": self.active,
                    "invalid": self.invalid, "used_trials": sorted(self.used),
                    "unacknowledged_trial": self.completed["trial_id"] if self.completed else None}
        if op == "ack":
            if (self.completed is None
                    or payload.get("trial_id") != self.completed["trial_id"]
                    or payload.get("capture_sha256") != self.completed["capture_sha256"]):
                raise ValueError("ACK must identify the retained exact capture")
            self.completed = None
            return {**common, "acknowledged": True}
        if op == "begin":
            tid = payload.get("trial_id")
            if (not self.qualified or self.invalid or self.active is not None
                    or self.completed is not None
                    or tid not in self.trials or tid in self.used):
                raise ValueError("Unqualified,active,invalid,unfrozen or reused capture")
            clocks = self.core.model_executor.collective_rpc("pr33_native_capture_clock")
            identity = [(c.get("process_id"), c.get("clock_domain"), c.get("bridge_sha256"))
                        for c in clocks]
            if identity != self.worker_identity:
                self.invalid = True
                raise ValueError("Worker source/clock/epoch changed before capture")
            self.active = tid
            self.used.add(tid)
            try:
                result = self.command(self.core, dict(operation="external_begin",
                    trial_id=tid, observer=self.trials[tid]))
                if result.get("external") is not True:
                    raise RuntimeError("Core capture begin was not acknowledged")
            except Exception:
                self.invalid = True
                raise
            return {**common, "trial_id": tid, "begin": result}
        if self.active is None:
            if self.completed is not None:
                return self.completed
            raise ValueError("No active capture")
        clocks = self.core.model_executor.collective_rpc("pr33_native_capture_clock")
        identity = [(c.get("process_id"), c.get("clock_domain"), c.get("bridge_sha256"))
                    for c in clocks]
        epoch_changed = identity != self.worker_identity
        self.invalid = self.invalid or epoch_changed
        result = self.command(self.core, {"operation": "external_end"})
        if isinstance(result, dict) and result.get("status") == "gpu_events_pending":
            return {**common, "trial_id": self.active, "status": "gpu_events_pending"}
        try:
            errors = capture_errors(result, self.trials[self.active])
            if result.get("trial_id") != self.active:
                errors.append("trial_identity_mismatch")
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            errors = ["malformed_capture_" + type(error).__name__]
        if epoch_changed:
            errors.append("worker_source_clock_epoch_changed")
        self.invalid = self.invalid or bool(errors)
        trial, self.active = self.active, None
        self.completed = {
            **common, "trial_id": trial, "capture_valid": not self.invalid,
            "validation_errors": errors, "capture": result,
            "capture_sha256": hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "retention_scope": "In-memory replay until ACK;not process-crash durability."
                " Collector must persist/fsync and verify the capture before ACK."}
        return self.completed


def worker_clock(worker):
    from prefill_lab.live_http import clock_domain
    return dict(clock_domain=clock_domain(), sampled_at=time.monotonic(),
                process_id=os.getpid(), bridge_sha256=source_hash())


def native_command(core, payload):
    if not hasattr(core, "_pr33_batch_capture"):
        from prefill_lab.live_http import clock_domain
        from prefill_lab.plugin import command
        plan = json.loads(Path(os.environ["PR33_NATIVE_BATCH_PLAN"]).read_text())
        core._pr33_batch_capture = BatchCapture(core, plan, command, domain=clock_domain())
    return core._pr33_batch_capture.invoke(payload)


def register():
    if os.environ.get("PR33_NATIVE_BATCH_CAPTURE_ENABLED") != "1":
        return
    if (os.environ.get("PREFILL_LAB_ENABLED") != "1"
            or any(os.environ.get("PREFILL_LAB_" + name, "0") != "0" for name in (
                "STAGES", "LIVE_STATE", "ENVELOPE_SHADOW", "EFFECTIVE_CALIBRATION",
                "DELAYED_ARRIVALS", "EVENT_CLOCK", "QUEUE_PROBE", "QUEUE_CONFIRMATION",
                "ENTRY_HTTP"))):
        raise RuntimeError("Batch capture requires a dedicated passive-only launch")
    from prefill_lab.plugin import register as install_reviewed_hooks, wire_reply
    install_reviewed_hooks()
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_worker import Worker
    EngineCore.pr33_native_batch = wire_reply(native_command)
    Worker.pr33_native_capture_clock = wire_reply(worker_clock)


class Endpoint:
    name = "pr33_native_batch"
    required_tasks = ("generate",)

    async def init_state(self, engine_client, state, args):
        if os.environ.get("PR33_NATIVE_BATCH_CAPTURE_ENABLED") == "1":
            self.engine = engine_client

    def attach_router(self, app):
        if os.environ.get("PR33_NATIVE_BATCH_CAPTURE_ENABLED") != "1":
            return
        from fastapi import HTTPException, Request
        globals()["Request"] = Request

        @app.post("/pr33-native/batch")
        async def capture(request: Request):
            secret = os.environ.get("PR33_NATIVE_OBSERVER_KEY", "")
            if len(secret) < 32 or not hmac.compare_digest(
                    request.headers.get("x-observer-key", ""), secret):
                raise HTTPException(401, "Observer credential required")
            payload = await request.json()
            if (not isinstance(payload, dict)
                    or set(payload) - {"operation", "trial_id", "capture_sha256"}
                    or payload.get("operation") not in ("metadata", "begin", "end", "status", "ack")):
                raise HTTPException(400, "Invalid passive capture command")
            return await self.engine.engine_core.call_utility_async("pr33_native_batch", payload)
