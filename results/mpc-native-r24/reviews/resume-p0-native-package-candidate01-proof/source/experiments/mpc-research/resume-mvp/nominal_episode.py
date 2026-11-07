"""Default-OFF, in-process engineering wiring; not service ownership or qualification.

Prepare locally, then choose one future monotonic START and construct the runtime
with that start and the returned Train provider/binding. Await run_episode inside
the caller's NominalServices scope. The caller owns the runtime HTTP client.
Client/metrics acceptance and Mayor exact-source approval are external prerequisites.
Neither collection nor cap ACKs establish admission effect; full wall ends only
after the caller's actual cleanup. Cancellation/joins remain cooperative.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _directory in (ROOT, HERE):
    _path = str(_directory)
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import nominal_client as recorder
import nominal_runtime as runtime_api
from nominal_permit import NominalDemoPermit

SCHEMA = "resume-p0-nominal-engineering-episode-v1"
PROBE = ROOT / "results/mpc-native-r24/reviews/resume-p0-probes02/inputs/admission-boundary.json"
PROBE_SHA256 = "6d938ba18240f3e0f8cb677064849a73316ee7a8dc918526c68cba37f02da6be"
TRAIN = ROOT / "results/mpc-native-r24/reviews/resume-p0-workload-author01/tapes-v1/train.tape.json"
POLICIES = {"mpc": "mpc", "one_move": "strong_one_move"}
SOURCE_PINS = {
    "nominal_runtime.py": "2d7a01dbd335297439d31ec00d144db4d4a771b190d8b47a03e1ddd3a5f80959",
    "nominal_client.py": "933e4161d4eadd7b9730f4d122387aad15932343bcc18de7580e9ea92b86fe95",
    "nominal_metrics.py": "49c76b8bcda2560ed9c24770a41c3fa876082cd0e19d171daf025300452a60ec",
    "nominal_planner.py": "e492a2548f378dcf0511abc56bde4a6bc06bdc9c62854e9dd1053b5f95ec2ea1",
    "nominal_permit.py": "05b7d0aebcede0851300ee70e9947e50ea6c94e6bd3ed2dc1283e10c5fcf6503",
    "nominal_scenario.py": "b15ccb25c4db2164a6bd44f3a4ec956a8e2aff1cfaa37040459948c4d7d477fc",
    "nominal_workload.py": "614ae110e901110f875383b9ee0a66b08fe75e3a58a923b1d30dbd9c21d88bd4",
    "nominal_services.py": "5249221b9bd795c8265039502746fe30a394e774a52040f9fcf6261146919bed",
}


class EpisodeRefused(ValueError):
    pass


class EpisodePersistenceError(RuntimeError):
    def __init__(self, receipt):
        super().__init__("episode evidence persistence failed; receipt retained on exception")
        self.receipt = receipt


def _require(condition, reason):
    if not condition:
        raise EpisodeRefused(reason)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _error(error):
    return {"type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(error))}


def _write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sources():
    actual = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    _require(actual == SOURCE_PINS, "frozen_dependency_source_changed")
    actual["nominal_episode.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    actual["public_implementation"] = runtime_api.implementation_hash()
    return actual


@dataclass
class PreparedEpisode:
    directory: Path
    recorder: recorder.PreparedCapture
    policy: str
    entered_at: float
    prepared_at: float
    sources: dict
    scenario_factory: object = None
    scenario_binding: dict | None = None
    error: dict | None = None
    input_sha256: str | None = None
    used: bool = False


def _inputs(prepared):
    return runtime_api.digest({
        "capture": prepared.recorder.capture, "trace": prepared.recorder.trace,
        "ready": prepared.recorder.ready, "started": prepared.recorder.started,
        "policy": prepared.policy, "scenario_binding": prepared.scenario_binding,
        "sources": prepared.sources, "preparation_error": prepared.error,
        "entered_at": prepared.entered_at, "prepared_at": prepared.prepared_at,
    })


def prepare_episode(*, output_dir, binding_path, gateway_url, backend_url, policy="mpc",
                    trace_path=PROBE, train_path=TRAIN, request_timeout=120.0,
                    tokenizer_loader=recorder.load_tokenizer):
    """Local-only preparation. Persist the peer envelope even when ready=False; choose START afterward."""
    entered = time.monotonic()
    directory = Path(output_dir).absolute()
    directory.mkdir(mode=0o700)
    sources = _sources()
    capture = recorder.prepare_capture(
        trace_path=trace_path, binding_path=binding_path, arm=POLICIES.get(policy, policy),
        purpose="engineering", gateway_url=gateway_url, backend_url=backend_url,
        request_timeout=request_timeout, tokenizer_loader=tokenizer_loader,
    )
    prepared = PreparedEpisode(directory, capture, policy, entered, entered, sources)
    try:
        _require(capture.ready, "recorder_local_preparation:" + str(capture.capture["error"]))
        _require(policy in POLICIES, "only_mpc_or_one_move_engineering")
        _require(capture.capture["trace_sha256"] == PROBE_SHA256
                 and len(capture.trace["requests"]) == 5, "exact_frozen_five_request_probe_required")
        for endpoint in (gateway_url, backend_url):
            runtime_api._root_endpoint(endpoint)
        prepared.scenario_factory, prepared.scenario_binding = runtime_api.load_train_scenario(
            train_path, purpose="engineering")
    except Exception as error:
        prepared.error = _error(error)
    prepared.prepared_at = time.monotonic()
    prepared.input_sha256 = _inputs(prepared)
    capture_sha = recorder.write_capture(directory / "prepared-capture.json", capture.capture)
    _write(directory / "preparation.json", {
        "schema": SCHEMA, "entered_at": entered, "prepared_at": prepared.prepared_at,
        "error": prepared.error, "recorder_ready": capture.ready,
        "input_sha256": prepared.input_sha256, "prepared_capture_sha256": capture_sha,
        "sources": sources, "scenario_binding": prepared.scenario_binding,
        "no_HTTP_or_offers": True,
    })
    return prepared


def _eligible(prepared, runtime, owner, start, deadline, reserve, enabled):
    _require(enabled is True, "experimental_episode_disabled")
    _require(prepared.error is None and prepared.recorder.ready, "local_preparation_refused")
    _require(not prepared.recorder.started and _inputs(prepared) == prepared.input_sha256,
             "prepared_input_changed_or_reused")
    _require(_sources() == prepared.sources, "prepared_sources_changed")
    _require(type(runtime) is runtime_api.NominalRuntime and runtime.enabled is True,
             "enabled_nominal_runtime_required")
    _require(type(runtime.permit) is NominalDemoPermit and runtime.permit.approved is True,
             "separate_approved_nominal_permit_required")
    _require(runtime.clock is time.monotonic and runtime.sleeper is asyncio.sleep,
             "actual_shared_monotonic_clock_required")
    _require(all(_finite(value) for value in (start, deadline, reserve)) and reserve >= 0,
             "finite_original_deadline_and_reserve_required")
    now = time.monotonic()
    _require(prepared.prepared_at <= now < start and runtime.start_at == start,
             "shared_start_past_or_mismatched")
    _require(deadline - reserve >= start + max(prepared.recorder.trace["duration_seconds"],
                                               runtime.limits.episode_s),
             "original_work_deadline_truncates_episode")
    permit = runtime.permit
    _require(_finite(permit.not_before) and _finite(permit.expires_at)
             and 0 <= permit.not_before <= start < permit.expires_at
             and 0 < permit.expires_at - permit.not_before <= 60
             and start + runtime.limits.episode_s <= min(prepared.entered_at + 60, permit.expires_at),
             "shared_start_outside_existing_runtime_envelope")
    _require(runtime.policy == prepared.policy and runtime.scenario_factory is prepared.scenario_factory
             and runtime_api.digest(runtime.protocol["scenario_binding"])
             == runtime_api.digest(prepared.scenario_binding),
             "runtime_Train_provider_or_policy_changed")
    _require(runtime_api._root_endpoint(prepared.recorder.capture["config"]["gateway_url"])
             == runtime.protocol["endpoint"], "controller_recorder_gateway_mismatch")
    _require(prepared.recorder.capture["clock_domain"] == recorder.clock_domain(),
             "recorder_clock_domain_changed")
    runtime._binding()
    owner.check()


async def _controller(prepared, runtime, start):
    # Do not let the controller act while the recorder's backend preflight is unresolved.
    while prepared.recorder.capture["records"] is None:
        remaining = start - time.monotonic()
        _require(remaining > 0, "models_preflight_not_before_shared_start")
        await asyncio.sleep(min(0.005, remaining))
    _require(prepared.recorder.capture["offered_start"] == start, "recorder_changed_shared_start")
    startup = await runtime_api.wait_for_start(runtime, cli_entered=prepared.entered_at)
    return {"startup": startup, "runtime": await runtime.run()}


async def _worker(call):
    try:
        return await call(), None
    except BaseException as error:
        return None, error


def _outcome(task):
    if task.cancelled():
        return None, asyncio.CancelledError("cancelled_before_worker_start")
    try:
        return task.result()
    except BaseException as error:
        return None, error


def _failure(name, result, runtime):
    if not isinstance(result, dict):
        return "missing_worker_result"
    if name == "recorder":
        if result.get("status") != "collected" or result.get("error") is not None:
            return "recorder_returned_" + str(result.get("status"))
        if not isinstance(result.get("records"), list) or len(result["records"]) != 5:
            return "recorder_incomplete_offered_records"
        return None
    value = result.get("runtime", {})
    steps = value.get("steps", [])
    if (value.get("failure") is not None or value.get("pending") is not False
            or value.get("pending_outcome") is not None or len(steps) != runtime.limits.iterations
            or any(row.get("status") not in ("hold", "applied_reobserved")
                   or row.get("deadline_met") is not True for row in steps)):
        return "controller_returned_failure_refusal_pending_or_incomplete"
    return None


async def run_episode(prepared, runtime, *, owner, offered_start, enabled=False, transport=None):
    """Use owner.deadline, owner.reserve and owner.check() from the existing synchronous NominalServices scope.

    Only the two asyncio workers are owned here. Every outcome is persisted before
    cancellation/KeyboardInterrupt is re-raised. Returned 'collected' is not PASS.
    """
    _require(type(prepared) is PreparedEpisode and not prepared.used, "unused_prepared_episode_required")
    prepared.used = True
    deadline, reserve = getattr(owner, "deadline", None), getattr(owner, "reserve", None)
    receipt = {
        "schema": SCHEMA, "status": "refused", "entered_at": time.monotonic(),
        "offered_start": offered_start if _finite(offered_start) else None,
        "deadline": deadline if _finite(deadline) else None,
        "cleanup_reserve_s": reserve if _finite(reserve) else None,
        "runtime_clock_domain": runtime_api.clock_domain(), "recorder_clock_domain": recorder.clock_domain(),
        "runtime_journal": str(runtime.store.path) if type(runtime) is runtime_api.NominalRuntime else None,
        "workers": {}, "failures": [], "persistence_errors": [], "cancel_requested": [],
        "admission_effect": None, "ordered_gateway_ledger": None, "full_wall": None,
        "qualification": None, "qualification_note": "engineering collection only; Mayor-owned evidence absent",
        "sources": prepared.sources,
    }
    tasks, interruption = {}, None
    try:
        _eligible(prepared, runtime, owner, offered_start, deadline, reserve, enabled)
        tasks["recorder"] = asyncio.create_task(_worker(lambda: recorder.capture_replay(
            prepared.recorder, offered_start=offered_start, deadline=deadline, cleanup_reserve_s=reserve,
            clock_domain=recorder.clock_domain(), transport=transport)), name="nominal-episode:recorder")
        tasks["controller"] = asyncio.create_task(_worker(lambda: _controller(prepared, runtime, offered_start)),
                                                   name="nominal-episode:controller")
        receipt["status"] = "collected"
        pending = set(tasks.values())
        while pending:
            _require(owner.deadline == deadline and owner.reserve == reserve, "owner_original_budget_changed")
            owner.check()
            remaining = deadline - reserve - time.monotonic()
            _require(remaining > 0, "original_work_deadline_exhausted")
            done, pending = await asyncio.wait(pending, timeout=min(remaining, 0.05),
                                               return_when=asyncio.FIRST_COMPLETED)
            for name, task in tasks.items():
                if task not in done:
                    continue
                result, error = _outcome(task)
                reason = "worker_exception" if error is not None else _failure(name, result, runtime)
                if reason:
                    receipt["failures"].append({"worker": name, "reason": reason})
                    if error is not None and not isinstance(error, Exception):
                        interruption = error
            if receipt["failures"]:
                receipt["status"] = "failed"
                break
    except BaseException as error:
        receipt["status"] = (
            "refused" if not tasks and isinstance(error, (EpisodeRefused, runtime_api.RuntimeRefused)) else "failed")
        receipt["failures"].append({"worker": "coordinator", "error": _error(error)})
        if not isinstance(error, Exception):
            interruption = error
    finally:
        for name, task in tasks.items():
            if not task.done():
                receipt["cancel_requested"].append(name)
                task.cancel()
        if tasks:
            joined = asyncio.gather(*tasks.values(), return_exceptions=True)
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError as error:
                    interruption = interruption or error
            for name, task in tasks.items():
                result, error = _outcome(task)
                receipt["workers"][name] = {
                    "joined": task.done(), "result": result, "error": _error(error) if error else None,
                }
        receipt["joined_at"] = time.monotonic()
        receipt["all_workers_joined"] = all(task.done() for task in tasks.values())
        if interruption is not None:
            receipt["status"] = "interrupted"
        if _finite(deadline) and receipt["joined_at"] > deadline:
            receipt["failures"].append({"worker": "coordinator", "reason": "original_deadline_overrun"})
            if interruption is None:
                receipt["status"] = "failed"
        try:
            if tasks:
                owner.check()
            _require(_sources() == prepared.sources, "source_changed_during_episode")
        except Exception as error:
            receipt["failures"].append({"worker": "postcheck", "error": _error(error)})
            if interruption is None:
                receipt["status"] = "failed"
        trace = prepared.recorder.trace
        offered_ids = [request["trace_id"] for request in trace["requests"]] if trace else None
        records = prepared.recorder.capture["records"]
        recorded_ids = {row["trace_id"] for row in records} if records is not None else set()
        receipt["offered_request_ids"] = offered_ids
        receipt["missing_record_ids"] = ([name for name in offered_ids if name not in recorded_ids]
                                         if offered_ids is not None else None)
        try:
            receipt["capture_sha256"] = recorder.write_capture(
                prepared.directory / "capture.json", prepared.recorder.capture)
        except Exception as error:
            receipt["persistence_errors"].append(_error(error))
            receipt["status"] = "failed" if interruption is None else "interrupted"
        try:
            _write(prepared.directory / "episode.json", receipt)
        except Exception as error:
            receipt["persistence_errors"].append(_error(error))
        if interruption is not None:
            if receipt["persistence_errors"]:
                interruption.add_note("Episode persistence failed; receipt attached as episode_receipt.")
            interruption.episode_receipt = receipt
            raise interruption
        if receipt["persistence_errors"]:
            raise EpisodePersistenceError(receipt)
    return receipt
