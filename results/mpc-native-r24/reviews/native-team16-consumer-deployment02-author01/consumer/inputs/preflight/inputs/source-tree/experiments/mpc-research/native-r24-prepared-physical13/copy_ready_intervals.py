"""Default-off intervals for existing async token-copy events; no native wiring."""

from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
import threading
from weakref import WeakValueDictionary, ref

EXECUTOR_SHA256 = "3cb039b80fe9b03dc1122f93c1f8a4098ef1355104b317ff6df25358f4349494"
RUNNER_SHA256 = "87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51"
V2_ASYNC_SHA256 = "77e17a4570ead2be30ae9b00888cf077a3b873018c12ca0549b853bfda02c1ba"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _at(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("copy_ready_finite_monotonic_clock_required")
    return value


class CopyEventBinding:
    def __init__(self, future_type, output_type):
        self.future_type, self.output_type = future_type, output_type
        self.sources = {}
        for cls, name, expected in (
            (future_type, "AsyncOutputFuture", EXECUTOR_SHA256),
            (output_type, "AsyncGPUModelRunnerOutput", RUNNER_SHA256),
        ):
            path = inspect.getsourcefile(cls)
            if cls.__name__ != name or path is None or _sha(path) != expected:
                raise ValueError("unreviewed_async_copy_event_type")
            self.sources[path] = expected

    def guard(self):
        if any(_sha(path) != expected for path, expected in self.sources.items()):
            raise ValueError("async_copy_event_source_changed")

    def event(self, future):
        if type(future) is not self.future_type:
            raise ValueError("exact_async_output_future_required")
        output = future.async_output
        if type(output) is not self.output_type:
            raise ValueError("exact_async_gpu_output_required")
        return output.async_copy_ready_event


class V2CopyEventBinding(CopyEventBinding):
    def __init__(self, future_type, output_type):
        self.future_type, self.output_type = future_type, output_type
        self.sources = {}
        for cls, name, expected in (
            (future_type, "AsyncOutputFuture", EXECUTOR_SHA256),
            (output_type, "AsyncOutput", V2_ASYNC_SHA256),
        ):
            path = inspect.getsourcefile(cls)
            if cls.__name__ != name or path is None or _sha(path) != expected:
                raise ValueError("unreviewed_v2_copy_event_type")
            self.sources[path] = expected

    def event(self, future):
        if type(future) is not self.future_type or future.single_value is not True:
            raise ValueError("exact_single_v2_async_output_future_required")
        output = future.async_output
        if type(output) is not self.output_type:
            raise ValueError("exact_v2_async_output_required")
        return output.copy_event


class CopyReadyIntervals:
    """One query per output, then the native update boundary supplies an upper bound."""

    def __init__(self, binding, *, capture_id, clock_domain, clock, limit=512):
        if (not isinstance(binding, CopyEventBinding) or not callable(clock)
                or isinstance(binding, V2CopyEventBinding) and type(binding) is not V2CopyEventBinding
                or any(type(v) is not str or not 0 < len(v) <= 128
                       for v in (capture_id, clock_domain))
                or type(limit) is not int
                or not 1 <= limit <= (8192 if type(binding) is V2CopyEventBinding else 512)):
            raise ValueError("explicit_bounded_copy_ready_capture_required")
        self.binding, self.clock = binding, clock
        self.bound_clock = clock
        self.capture_id, self.clock_domain = capture_id, clock_domain
        self.limit, self.owner_thread = limit, threading.get_ident()
        self.records, self.errors = {}, []
        self.v2_objects = {}
        self.v2_seen = {key: WeakValueDictionary() for key in ("future", "output", "event")}
        self.last_at = 0.0
        self.closed = False

    def _guard(self):
        if self.closed:
            raise ValueError("copy_ready_capture_already_closed")
        if type(self.binding) is V2CopyEventBinding and self.errors:
            raise ValueError("copy_ready_v2_capture_invalid")
        if threading.get_ident() != self.owner_thread or self.clock is not self.bound_clock:
            raise ValueError("copy_ready_thread_or_clock_changed")

    def _now(self):
        at = _at(self.clock())
        if at < self.last_at:
            raise ValueError("copy_ready_clock_regressed")
        self.last_at = at
        return at

    def _fail(self, operation, exc):
        reason = operation + ":" + type(exc).__name__
        if reason not in self.errors and len(self.errors) < 16:
            self.errors.append(reason)

    def register(self, step_id, future, *, sample_started_at):
        try:
            self._guard()
            if (type(step_id) is not str or not 0 < len(step_id) <= 128
                    or step_id in self.records or len(self.records) >= self.limit
                    or type(self.binding) is V2CopyEventBinding and step_id in self.v2_objects):
                raise ValueError("copy_ready_step_inventory_or_limit")
            started = _at(sample_started_at)
            event = self.binding.event(future)
            if type(self.binding) is V2CopyEventBinding:
                self.binding.guard()
                objects = {"future": future, "output": future.async_output, "event": event}
                if any(id(obj) in self.v2_seen[key] for key, obj in objects.items()):
                    raise ValueError("copy_ready_reused_v2_future_output_or_event")
                for key, obj in objects.items():
                    self.v2_seen[key][id(obj)] = obj
                self.v2_objects[step_id] = tuple(ref(objects[k]) for k in ("future", "output", "event"))
            query_started = self._now()
            if started > query_started:
                raise ValueError("copy_ready_sample_clock_order")
            ready = event.query()
            query_finished = self._now()
            if type(ready) is not bool:
                raise ValueError("copy_ready_boolean_query_required")
            self.records[step_id] = {
                "sample_started_at": started,
                "query_started_at": query_started,
                "query_finished_at": query_finished,
                "query_ready": ready,
                "ready_lower_at": started if ready else query_started,
                "ready_lower_inclusive": ready,
                "ready_upper_at": query_finished if ready else None,
                "native_update_begin_at": None,
            }
        except BaseException as exc:
            self._fail("register", exc)

    def consumed(self, step_id):
        try:
            self._guard()
            row = self.records[step_id]
            if type(self.binding) is V2CopyEventBinding:
                objects = [reference() for reference in self.v2_objects[step_id]]
                if (any(obj is None for obj in objects)
                        or objects[0].async_output is not objects[1]
                        or self.binding.event(objects[0]) is not objects[2]):
                    raise ValueError("copy_ready_v2_output_event_ownership_changed")
                if (objects[0]._state != "FINISHED" or objects[0]._exception is not None
                        or objects[0]._result is not objects[1].model_runner_output):
                    raise ValueError("copy_ready_v2_normal_native_consumption_required")
            if row["native_update_begin_at"] is not None:
                raise ValueError("copy_ready_duplicate_consumption")
            at = self._now()
            row["native_update_begin_at"] = at
            if row["ready_upper_at"] is None:
                row["ready_upper_at"] = at
        except BaseException as exc:
            self._fail("consumed", exc)

    def finish(self, expected_step_ids):
        try:
            self._guard()
            self.binding.guard()
            expected = tuple(expected_step_ids)
            if (not expected or any(type(key) is not str for key in expected)
                    or len(set(expected)) != len(expected)
                    or set(expected) != self.records.keys()):
                raise ValueError("copy_ready_exact_native_step_inventory")
            if any(row["native_update_begin_at"] is None for row in self.records.values()):
                raise ValueError("copy_ready_unconsumed_output")
        except BaseException as exc:
            self._fail("finish", exc)
        self.closed = True
        return {
            "schema": "r24-async-copy-ready-intervals-v1",
            "capture_id": self.capture_id, "clock_domain": self.clock_domain,
            "thread_id": self.owner_thread,
            "observations_complete": not self.errors,
            "records": deepcopy(self.records), "errors": list(self.errors),
            "source_sha256": dict(self.binding.sources),
            "scope": (
                "Existing token-D2H-copy event, not GPU compute or exclusive service. "
                "One nonblocking query per output; no added result/get_output/synchronize, "
                "event record, background polling or native wiring. False query gives "
                "strict lower bound at query START, true gives upper at query END. "
                "Native update begin is an upper only under pinned normal result-consumption "
                "ordering. Sample begin lower requires the fresh event created inside that "
                "same sample call. Runtime wiring, event identity, capture/step/clock "
                "binding and query overhead are unqualified; file pins are not authentication."
            ),
            "native_mpc_permitted": False, "deployable": False,
        }
