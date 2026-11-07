"""Opt-in native forward-region recording; never exclusive GPU service."""

from functools import wraps
import math
import threading
import time


SCHEMA = "r24-prepared-forward-region-v3"
SCOPE = (
    "Native forward_start after input preparation through immediately before "
    "kv_connector.finish_forward;stream-local model/dispatch interval,not exclusive "
    "GPU service,sampling,copy,absolute GPU-clock calibration or deployment qualification."
    "Any measurement exception,failed region or unresolved final close refuses the whole run;"
    "measurement exceptions may suppress native dispatch and must not be caught as a successful capture."
)
RUNNER_SOURCE_SHA256 = "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce"


class ForwardRegionRecorder:
    def __init__(self, torch, current_row, *, clock=time.monotonic, capacity=8192):
        self.torch, self.current_row, self.clock, self.capacity = torch, current_row, clock, capacity
        self.pending = {}
        self.errors = []
        self._lock = threading.RLock()
        self._closed = False
        self.entered_count = 0
        self.completed_count = 0
        self._completed_regions = {}

    def enter(self):
        with self._lock:
            try:
                return self._enter()
            except Exception as error:
                self.errors.append(type(error).__name__)
                raise

    def _enter(self):
        if self._closed:
            raise ValueError("prepared_forward_run_closed")
        row = self.current_row()
        if row is None:
            return
        if "prepared_forward_region" in row or len(self.pending) >= self.capacity:
            raise ValueError("prepared_forward_duplicate_or_capacity")
        entered = self.clock()
        if type(entered) not in (int, float) or not math.isfinite(entered):
            raise ValueError("prepared_forward_invalid_cpu_clock")
        stream = self.torch.cuda.current_stream()
        stream_id = getattr(stream, "cuda_stream", None)
        if type(stream_id) is not int or stream_id < 0:
            raise ValueError("prepared_forward_stream_identity_required")
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record(stream)
        record = {
            "schema": SCHEMA, "scope": SCOPE, "qualified": False,
            "region_id": self.entered_count, "step_id": row.get("step_id"),
            "host_enter_at": entered, "host_leave_at": None,
            "thread_id": threading.get_ident(), "stream_id": stream_id,
            "gpu_elapsed_s": None, "gpu_ready_lower_at": entered, "gpu_ready_upper_at": None,
            "status": "entered",
        }
        row["prepared_forward_region"] = record
        self.pending[id(row)] = (
            record, start, end, stream, row, (self.entered_count, row.get("step_id")))
        self.entered_count += 1

    def leave(self):
        with self._lock:
            try:
                return self._leave()
            except Exception as error:
                self.errors.append(type(error).__name__)
                raise

    def _leave(self):
        if self._closed:
            raise ValueError("prepared_forward_run_closed")
        row = self.current_row()
        if row is None:
            return
        entry = self.pending.get(id(row))
        if entry is None or entry[0]["status"] != "entered":
            raise ValueError("prepared_forward_missing_or_duplicate_leave")
        record, _, end, stream, _, _ = entry
        current = self.torch.cuda.current_stream()
        if (record["thread_id"] != threading.get_ident()
                or current != stream):
            record.update(status="failed", error_type="prepared_forward_thread_or_stream_changed")
            raise ValueError("prepared_forward_thread_or_stream_changed")
        end.record(stream)
        record["host_leave_at"] = self.clock()
        record["status"] = "pending"

    def wrap(self, forward_start, finish_forward):
        @wraps(forward_start)
        def start(*args, **kwargs):
            self.enter()
            return forward_start(*args, **kwargs)

        @wraps(finish_forward)
        def finish(*args, **kwargs):
            self.leave()
            return finish_forward(*args, **kwargs)

        return start, finish

    def resolve(self):
        with self._lock:
            return self._resolve()

    def _resolve(self):
        for key, entry in list(self.pending.items()):
            record, start, end, stream, _, identity = entry
            if record["status"] == "entered":
                continue
            if record["status"] == "failed":
                self.pending.pop(key)
                continue
            try:
                before_query = self.clock()
                ready = end.query()
                after_query = self.clock()
                if type(ready) is not bool:
                    raise ValueError("prepared_forward_query_type")
                if (not all(type(t) in (int, float) and math.isfinite(t) for t in (
                        record["host_enter_at"], record["host_leave_at"], before_query, after_query))
                        or not record["host_enter_at"] <= record["host_leave_at"] <= before_query
                        <= after_query or record["gpu_ready_lower_at"] > after_query):
                    raise ValueError("prepared_forward_invalid_cpu_clock")
                if not ready:
                    # A false query can race with completion before its CPU return.
                    record["gpu_ready_lower_at"] = max(record["gpu_ready_lower_at"], before_query)
                    continue
                elapsed = start.elapsed_time(end)
                if type(elapsed) not in (int, float):
                    raise ValueError("prepared_forward_invalid_elapsed")
                value = elapsed / 1000
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("prepared_forward_invalid_elapsed")
                record.update(
                    status="complete", gpu_elapsed_s=value, gpu_ready_upper_at=after_query)
                self.pending.pop(key)
                self._completed_regions[identity[0]] = identity[1]
                self.completed_count += 1
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__)
                self.errors.append(type(error).__name__)
                self.pending.pop(key)
        return self._status()

    def _status(self):
        return {
            "schema": SCHEMA, "scope": SCOPE, "qualified": False,
            "pending": len(self.pending), "errors": list(self.errors),
            "in_flight": sum(entry[0]["status"] == "entered" for entry in self.pending.values()),
            "closed": self._closed, "whole_run_refused": bool(self.errors),
            "entered_count": self.entered_count, "completed_count": self.completed_count,
            "completed_regions": [
                {"region_id": identity, "step_id": step}
                for identity, step in sorted(self._completed_regions.items())
            ],
            "native_mpc_permitted": False, "deployable": False,
        }

    def close(self):
        """Finalize only after native execution has returned or been quiesced."""
        with self._lock:
            self._closed = True
            self._resolve()
            for key, entry in list(self.pending.items()):
                if entry[0]["status"] == "entered":
                    entry[0].update(status="failed", error_type="missing_native_finish")
                    self.errors.append("missing_native_finish")
                    self.pending.pop(key)
            if self.pending and "unresolved_gpu_event_on_close" not in self.errors:
                self.errors.append("unresolved_gpu_event_on_close")
            return self._status()
