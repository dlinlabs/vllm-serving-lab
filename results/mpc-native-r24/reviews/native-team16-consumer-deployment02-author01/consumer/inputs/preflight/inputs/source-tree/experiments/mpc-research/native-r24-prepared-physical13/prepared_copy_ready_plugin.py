"""Default-off copy-ready sidecar on the existing authenticated batch route."""

from copy import deepcopy
from functools import wraps
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

from prepared_copy_ready_hook import (
    CaptureContext, METROLOGY_MODES, PINS, SCHEMA as HOOK_SCHEMA,
    check_file, install, require, validate_metrology,
)
from copy_ready_intervals import CopyReadyIntervals, V2CopyEventBinding


FLAG = "PR33_R24_PREPARED_COPY_READY_ENABLED"
MARKER = "_r24_prepared_copy_ready_registration"
STATE = "_r24_prepared_copy_ready_state"
EXPORT_PREFIX = "pr33_r24_prepared_copy_ready_export:"
SCHEMA = "r24-prepared-copy-ready-extension-v2"
SIDECAR_SCHEMA = "r24-prepared-copy-ready-sidecar-v2"
FILES = ("prepared_copy_ready_plugin.py", "prepared_copy_ready_hook.py",
         "copy_ready_intervals.py", "queue_observer.py", "launch_queue_service.py",
         "collect_azure.py")
_REGISTRATION = None


def declaration(metrology="off"):
    require(type(metrology) is str and metrology in METROLOGY_MODES,
            "getter_metrology_mode_required")
    directory = Path(__file__).parent
    value = {
        "schema": SCHEMA, "enabled": True, "observer": "light",
        "sources": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in FILES},
        "native_sources": dict(PINS),
    }
    if metrology != "off":
        value["getter_metrology_mode"] = metrology
    return value


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("prepared_copy_ready_extension")) is dict,
            "exact_copy_ready_extension_required")
    extension = plan["prepared_copy_ready_extension"]
    require(extension == declaration(extension.get("getter_metrology_mode", "off")),
            "exact_copy_ready_extension_required")
    from prepared_forward_plugin import validate_plan as validate_prepared
    validate_prepared(plan)
    require(not any(plan.get(name) for name in (
        "host_trace_extension", "copy_ready_extension", "phase_frontier_extension",
        "input_post_extension")), "prepared_copy_exclusive_route_required")


def register():
    global _REGISTRATION
    flag = os.environ.get(FLAG, "0")
    require(flag in ("0", "1"), "boolean_launch_flag_required")
    if flag == "0":
        return
    require(all(os.environ.get(name) == "1" for name in (
        "PR33_R24_PREPARED_FORWARD_ENABLED", "PR33_R23_QUEUE_OBSERVER_ENABLED",
        "PR33_NATIVE_BATCH_CAPTURE_ENABLED", "PR33_NATIVE_OBSERVER_ENABLED",
        "PREFILL_LAB_ENABLED")), "prepared_batch_parent_required")
    require(all(os.environ.get(name, "0") == "0" for name in (
        "PR33_R24_HOST_TRACE_ENABLED", "PR33_R24_COPY_READY_ENABLED",
        "PR33_R24_COPY_READY_ZERO_TOKENS_ENABLED",
        "PR33_R24_PHASE_OBSERVER_ENABLED", "PR33_R23_INPUT_POST_OBSERVER_ENABLED")),
        "conflicting_copy_ready_route")
    plan = json.loads(Path(os.environ["PR33_NATIVE_BATCH_PLAN"]).read_bytes())
    validate_plan(plan)
    directory = Path(__file__).parent.resolve()
    for reference, name in ((CaptureContext, "prepared_copy_ready_hook.py"),
                            (install, "prepared_copy_ready_hook.py"),
                            (validate_metrology, "prepared_copy_ready_hook.py"),
                            (CopyReadyIntervals, "copy_ready_intervals.py"),
                            (V2CopyEventBinding, "copy_ready_intervals.py")):
        module = sys.modules.get(reference.__module__)
        require(getattr(module, reference.__name__, None) is reference
                and Path(module.__file__).resolve() == directory / name,
                "imported_copy_helper_identity_required")
        check_file(module.__file__, declaration()["sources"][name])
    if _REGISTRATION is not None:
        _REGISTRATION.check_owner()
        require(_REGISTRATION.bound_metrology == plan["prepared_copy_ready_extension"].get(
            "getter_metrology_mode", "off"), "registered_getter_metrology_mode_changed")
        return
    facade = importlib.import_module("batch_capture_bridge")
    check_file(facade.__file__, PINS["batch_capture_bridge"])
    registration = Registration(facade, metrology=plan["prepared_copy_ready_extension"].get(
        "getter_metrology_mode", "off"))
    registration.install()
    _REGISTRATION = registration


class Registration:
    def __init__(self, facade, *, metrology="off"):
        require(type(metrology) is str and metrology in METROLOGY_MODES,
                "getter_metrology_mode_required")
        self.metrology = self.bound_metrology = metrology
        self.facade, self.cls = facade, facade.BatchCapture
        require(getattr(self.cls, MARKER, None) is None, "foreign_batch_registration")
        self.original = self.cls.invoke
        require(self.original.__module__ == facade.__name__
                and self.original.__code__.co_filename == facade.__file__,
                "pristine_batch_invoke_required")

        @wraps(self.original)
        def invoke(bridge, payload):
            if (type(payload) is dict and payload.get("operation") == "status"
                    and type(payload.get("trial_id")) is str
                    and payload["trial_id"].startswith(EXPORT_PREFIX)):
                require(set(payload) == {"operation", "trial_id"},
                        "exact_export_fields_required")
                trial_id = payload["trial_id"][len(EXPORT_PREFIX):]
                require(0 < len(trial_id) <= 128, "bounded_export_trial_required")
                state = getattr(bridge.core, STATE, None)
                return (state.export(bridge, trial_id) if type(state) is State else
                        refusal("no_copy_ready_capture", trial_id))
            result = self.original(bridge, payload)
            if type(payload) is dict and payload.get("operation") == "begin":
                try:
                    self.check_owner()
                    state = getattr(bridge.core, STATE, None)
                    if state is None:
                        state = State(self, bridge)
                        setattr(bridge.core, STATE, state)
                    require(type(state) is State and state.registration is self,
                            "copy_ready_state_ownership_changed")
                    state.begin(bridge)
                except BaseException as error:
                    state = getattr(bridge.core, STATE, None)
                    if type(state) is State:
                        state.fail("begin", error)
            elif type(payload) is dict and payload.get("operation") == "ack":
                state = getattr(bridge.core, STATE, None)
                if type(state) is State:
                    try:
                        state.acknowledge(bridge, payload, result)
                    except BaseException as error:
                        state.fail("ack", error)
            return result

        self.callback = invoke

    def install(self):
        try:
            self.cls.invoke = self.callback
            setattr(self.cls, MARKER, self)
        except BaseException:
            self.remove()
            raise

    def check_owner(self):
        require(self.cls.invoke is self.callback and vars(self.cls).get(MARKER) is self
                and self.metrology == self.bound_metrology,
                "batch_registration_ownership_changed")

    def remove(self):
        if self.cls.invoke is self.callback:
            self.cls.invoke = self.original
        if vars(self.cls).get(MARKER) is self:
            delattr(self.cls, MARKER)


def refusal(reason, trial_id):
    return {
        "schema": SIDECAR_SCHEMA, "trial_id": trial_id,
        "whole_capture_complete": False, "discard_required": True, "refusal": reason,
        "native_mpc_permitted": False, "deployable": False,
    }


class State:
    def __init__(self, registration, bridge):
        self.registration, self.bridge = registration, bridge
        self.hook = self.context = self.snapshot = None
        self.epoch = 0
        self.bound_extension = None
        self.last_ack = None
        self.errors = []
        self.discard_required = False

    def fail(self, operation, error):
        name = type.__dict__["__name__"].__get__(type(error))
        reason = operation + ":" + str.__getitem__(name, slice(None, 64))
        if reason not in self.errors and len(self.errors) < 16:
            self.errors.append(reason)
        self.discard_required = True

    def begin(self, bridge):
        require(bridge is self.bridge and not self.discard_required
                and (self.hook is None or self.hook.removed and not self.hook.discard_required),
                "previous_copy_capture_not_clean")
        require(os.environ.get(FLAG) == "1", "copy_ready_disabled")
        self.epoch += 1
        self.errors, self.snapshot = [], None
        self.last_ack = None
        self.bound_extension = declaration(self.registration.metrology)
        self.context = CaptureContext(bridge.core, bridge, self.epoch)
        self.hook = install(bridge.core, self.context, enabled=True,
                            metrology=self.registration.metrology)

    def acknowledge(self, bridge, payload, result):
        self.registration.check_owner()
        require(bridge is self.bridge and self.context is not None
                and payload.get("trial_id") == self.context.capture.id
                and type(payload.get("capture_sha256")) is str
                and re.fullmatch(r"[0-9a-f]{64}", payload["capture_sha256"])
                and result.get("acknowledged") is True
                and result.get("clock_domain") == self.context.context[0]
                and result.get("run_id") == bridge.run_id
                and result.get("plan_sha256") == bridge.plan_hash
                and bridge.active is None and bridge.completed is None,
                "native_capture_ack_digest_required")
        self.last_ack = (payload["trial_id"], payload["capture_sha256"])

    def export(self, bridge, trial_id):
        if self.snapshot is not None and trial_id == self.snapshot["trial_id"]:
            return deepcopy(self.snapshot)
        try:
            require(os.environ.get(FLAG) == "1", "copy_ready_disabled")
            require(declaration(self.registration.metrology) == self.bound_extension,
                    "copy_ready_sources_changed")
            self.registration.check_owner()
            require(bridge is self.bridge and self.hook is not None
                    and self.context.capture.id == trial_id, "current_capture_required")
            require(bridge.active is None and bridge.completed is None
                    and trial_id in bridge.used and self.last_ack is not None
                    and self.last_ack[0] == trial_id, "post_end_ack_barrier_required")
            result = self.hook.close()
            self.discard_required = self.discard_required or result["discard_required"]
            self.snapshot = {
                "schema": SIDECAR_SCHEMA, "trial_id": trial_id,
                "extension": deepcopy(self.bound_extension), "hook": result, "errors": list(self.errors),
                "clock_domain": self.context.context[0],
                "native_run_id": self.context.context[1],
                "bridge_run_id": bridge.run_id, "plan_sha256": bridge.plan_hash,
                "capture_sha256": self.last_ack[1],
                "whole_capture_complete": result["whole_capture_complete"]
                                          and not self.errors and not self.discard_required,
                "discard_required": self.discard_required,
                "native_mpc_permitted": False, "deployable": False,
            }
        except BaseException as error:
            self.fail("export", error)
            return refusal(self.errors[-1], trial_id)
        return deepcopy(self.snapshot)


def validate_export(value, capture, extension, *, clock_domain, bridge_run_id, plan_sha256):
    require(type(value) is dict and value.get("schema") == SIDECAR_SCHEMA
            and value.get("trial_id") == capture.get("trial_id")
            and value.get("extension") == extension
            and value.get("clock_domain") == clock_domain
            and value.get("bridge_run_id") == bridge_run_id
            and value.get("plan_sha256") == plan_sha256
            and type(value.get("capture_sha256")) is str
            and value["capture_sha256"] == capture.get("capture_sha256")
            == hashlib.sha256(json.dumps(capture.get("capture"), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            and type(value.get("native_run_id")) is str and bool(value["native_run_id"])
            and value.get("whole_capture_complete") is True
            and value.get("discard_required") is False and value.get("errors") == []
            and value.get("native_mpc_permitted") is False and value.get("deployable") is False,
            "whole_copy_capture_export_required")
    hook = value.get("hook")
    require(type(hook) is dict and hook.get("schema") == HOOK_SCHEMA
            and hook.get("whole_capture_complete") is True and hook.get("hook_removed") is True
            and hook.get("discard_required") is False and hook.get("errors") == [],
            "removed_clean_parent_hook_required")
    native = capture.get("capture")
    require(type(native) is dict and type(native.get("steps")) is list,
            "native_capture_inventory_required")
    steps = native["steps"]
    require(all(type(row) is dict and type(row.get("step_id")) is str for row in steps),
            "native_step_identity_required")
    ids = [row["step_id"] for row in steps]
    require(bool(ids) and len(ids) == len(set(ids))
            and hook.get("entered_step_ids") == hook.get("consumed_step_ids") == ids,
            "complete_ordered_copy_inventory_required")
    intervals = hook.get("intervals")
    require(type(intervals) is dict
            and intervals.get("schema") == "r24-async-copy-ready-intervals-v1"
            and intervals.get("capture_id") == value["trial_id"]
            and intervals.get("clock_domain") == clock_domain
            and type(intervals.get("thread_id")) is int and intervals["thread_id"] > 0
            and intervals.get("observations_complete") is True and intervals.get("errors") == []
            and type(intervals.get("records")) is dict and set(intervals["records"]) == set(ids),
            "complete_same_clock_copy_intervals_required")
    sources = intervals.get("source_sha256")
    require(type(sources) is dict and len(sources) == 2
            and set(sources.values()) == {PINS["vllm.v1.executor.uniproc_executor"],
                                         PINS["vllm.v1.worker.gpu.async_utils"]},
            "native_copy_source_pair_required")
    routes = hook.get("routes")
    require(type(routes) is list and 0 < len(routes) <= 8192
            and all(type(hook.get(name)) is int and hook[name] == len(routes)
                    for name in ("entered_route_count", "completed_route_count"))
            and type(hook.get("completed_route_sequences")) is list
            and all(type(sequence) is int for sequence in hook["completed_route_sequences"])
            and hook.get("completed_route_sequences") == list(range(len(routes))),
            "complete_native_route_counts_required")
    copy_ids = []
    previous_execute = previous_update = -1
    for sequence, route in enumerate(routes):
        require(type(route) is dict and type(route.get("sequence")) is int
                and route["sequence"] == sequence and route.get("kind") in ("copy", "no_copy")
                and route.get("status") == "consumed"
                and type(route.get("step_id")) is str and bool(route["step_id"])
                and all(type(route.get(name)) is int and route[name] >= 0
                        for name in ("scheduled_tokens", "scheduled_requests")),
                "native_route_shape_required")
        times = [route.get(name) for name in (
            "execute_started_at", "execute_finished_at", "update_started_at", "update_finished_at")]
        require(all(type(at) in (int, float) and math.isfinite(at) and at >= 0 for at in times)
                and times == sorted(times) and previous_execute <= times[0]
                and previous_update <= times[2], "native_route_time_chain_required")
        previous_execute, previous_update = times[0], times[3]
        if route["kind"] == "no_copy":
            require(route["scheduled_tokens"] == route["scheduled_requests"] == 0,
                    "empty_native_no_copy_route_required")
        else:
            require(route["scheduled_tokens"] > 0 and route["scheduled_requests"] > 0
                    and route["step_id"] in intervals["records"], "positive_native_copy_route_required")
            copy_ids.append(route["step_id"])
            record = intervals["records"][route["step_id"]]
            require(type(record) is dict, "copy_interval_row_required")
            at = record.get("native_update_begin_at")
            require(type(at) in (int, float) and math.isfinite(at) and times[2] <= at <= times[3],
                    "copy_update_route_join_required")
    require(copy_ids == ids, "whole_positive_route_inventory_required")
    for row in intervals["records"].values():
        keys = ("sample_started_at", "query_started_at", "query_finished_at",
                "native_get_output_begin_at", "native_get_output_return_at",
                "native_update_begin_at", "ready_lower_at", "ready_upper_at")
        require(type(row) is dict and all(type(row.get(key)) in (int, float)
                and math.isfinite(row[key]) and row[key] >= 0 for key in keys),
                "finite_native_copy_clocks_required")
        require(row["sample_started_at"] <= row["query_started_at"]
                <= row["query_finished_at"] <= row["native_get_output_begin_at"]
                <= row["native_get_output_return_at"] <= row["native_update_begin_at"]
                and type(row.get("query_ready")) is bool
                and row.get("ready_lower_inclusive") is row["query_ready"]
                and row["ready_lower_at"] == (row["sample_started_at"] if row["query_ready"]
                                              else row["query_started_at"])
                and row["ready_upper_at"] == min(row["native_get_output_return_at"],
                    row["query_finished_at"] if row["query_ready"]
                    else row["native_get_output_return_at"])
                and (row["ready_lower_at"] < row["ready_upper_at"]
                     or row["ready_lower_inclusive"] is True),
                "native_copy_bracket_order_required")
    mode = extension.get("getter_metrology_mode", "off")
    if mode == "off":
        require("getter_metrology" not in hook, "undeclared_getter_metrology")
    else:
        validate_metrology(hook.get("getter_metrology"), routes, intervals["records"], mode=mode)
    return {"schema": "r24-prepared-copy-ready-collector-gate-v2", "valid": True,
            "step_count": len(ids), "native_run_id": value["native_run_id"],
            "native_route_count": len(routes), "no_copy_routes": len(routes) - len(copy_ids),
            "native_mpc_permitted": False, "deployable": False}
