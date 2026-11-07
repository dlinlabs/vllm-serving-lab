"""Whole-capture measurement gate; neither physical nor policy qualification."""

import math

from prepared_forward_region import SCHEMA, SCOPE


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def gate(capture, *, native_errors):
    reasons = []
    if not isinstance(capture, dict):
        capture = {}
        reasons.append("worker_capture_required")
    if type(native_errors) is not list or native_errors:
        reasons.append("empty_native_exception_log_required")
    if any(type(capture.get(key)) is not int or capture[key] != 0
           for key in ("pending", "overflow")):
        reasons.append("complete_nonoverflow_worker_capture_required")
    if capture.get("cuda_scope_qualified") is not False:
        reasons.append("unqualified_worker_cuda_scope_required")
    if capture.get("worker_must_be_discarded") is not False:
        reasons.append("discarded_worker_cannot_supply_accepted_capture")
    status = capture.get("prepared_forward")
    if not isinstance(status, dict):
        status = {}
    if (status.get("schema") != SCHEMA or status.get("scope") != SCOPE
            or status.get("qualified") is not False or status.get("closed") is not True
            or status.get("whole_run_refused") is not False
            or status.get("native_mpc_permitted") is not False
            or status.get("deployable") is not False
            or status.get("errors") != []
            or type(status.get("errors")) is not list
            or any(type(status.get(key)) is not int or status[key] != 0
                   for key in ("pending", "in_flight"))):
        reasons.append("closed_complete_unrefused_region_run_required")
    rows = capture.get("rows")
    if type(rows) is not list or not rows:
        rows = []
        reasons.append("nonempty_full_worker_rows_required")
    bindings = status.get("completed_regions")
    inventory = {}
    if type(bindings) is list:
        for binding in bindings:
            if not isinstance(binding, dict):
                reasons.append("completed_region_binding_required")
                continue
            identity, step = binding.get("region_id"), binding.get("step_id")
            if (type(identity) is not int or identity < 0 or identity in inventory
                    or type(step) is not str or not step):
                reasons.append("unique_completed_region_binding_required")
                continue
            inventory[identity] = step
    if (type(bindings) is not list
            or any(type(status.get(key)) is not int or status[key] != len(rows)
                   for key in ("entered_count", "completed_count"))
            or len(inventory) != len(rows)
            or len(bindings or []) != len(rows)
            or set(inventory) != set(range(len(rows)))):
        reasons.append("exact_entered_completed_row_inventory_required")
    seen = set()
    seen_regions = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"row_{index}_object_required")
            continue
        step = row.get("step_id")
        if type(step) is not str or not step or step in seen:
            reasons.append(f"row_{index}_unique_step_required")
        else:
            seen.add(step)
        region = row.get("prepared_forward_region")
        if not isinstance(region, dict):
            region = {}
        identity = region.get("region_id")
        if (type(identity) is not int or identity < 0 or identity in seen_regions
                or inventory.get(identity) != step or region.get("step_id") != step):
            reasons.append(f"row_{index}_unique_bound_region_required")
        else:
            seen_regions.add(identity)
        if (region.get("schema") != SCHEMA or region.get("scope") != SCOPE
                or region.get("qualified") is not False or region.get("status") != "complete"
                or region.get("error_type") is not None or row.get("gpu_error") is not None):
            reasons.append(f"row_{index}_complete_unqualified_region_required")
        elapsed = region.get("gpu_elapsed_s")
        if not _finite(elapsed) or elapsed <= 0:
            reasons.append(f"row_{index}_positive_relative_cuda_elapsed_required")
        clocks = [row.get("host_start"), region.get("host_enter_at"),
                  region.get("host_leave_at"), row.get("host_end")]
        ready = [region.get("gpu_ready_lower_at"), region.get("gpu_ready_upper_at")]
        if (not all(_finite(value) for value in clocks + ready)
                or clocks != sorted(clocks) or not clocks[1] <= ready[0] <= ready[1]
                or clocks[2] > ready[1]):
            reasons.append(f"row_{index}_cpu_clock_and_ready_bracket_required")
    return {
        "schema": "r24-prepared-forward-capture-gate-v2",
        "accepted": not reasons, "refusal_reasons": reasons, "worker_rows": len(rows),
        "scope": SCOPE, "physical_model_qualified": False,
        "native_mpc_permitted": False, "deployable": False,
    }


class Resolver:
    """Call only from the native worker's quiescent finish RPC, after active=False."""

    def __init__(self, worker, region, hook, *, native_errors):
        self.worker, self.region, self.hook = worker, region, hook
        self.native_errors = native_errors
        self.original = worker.resolve
        self.discard_required = False

    def resolve(self):
        if self.worker.active is not False:
            raise ValueError("prepared_capture_quiescent_finish_rpc_required")
        status = self.region.resolve()
        capture = self.original()
        if type(capture.get("pending")) is not int or capture["pending"] < 0:
            raise ValueError("prepared_capture_native_pending_invalid")
        if ((status["closed"] and status["pending"])
                or any(error in status["errors"] for error in (
                    "unresolved_gpu_event_on_close", "hook_cleanup_failed"))):
            self.discard_required = True
        if (type(capture.get("pending")) is int and capture["pending"] == 0
                and status["pending"] == status["in_flight"]):
            status = self.region.close()
            if not status["pending"] and "hook_cleanup_failed" not in status["errors"]:
                try:
                    self.hook.remove()
                except Exception:
                    self.discard_required = True
                    self.region.errors.append("hook_cleanup_failed")
                    status = self.region.resolve()
        capture = {**capture, "prepared_forward": status,
                   "worker_must_be_discarded": self.discard_required}
        capture["pending"] = capture["pending"] + status["pending"]
        capture["prepared_forward_gate"] = gate(capture, native_errors=self.native_errors)
        return capture
