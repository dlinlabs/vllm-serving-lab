"""Fresh owned native acquisition, then a later flushed Gateway prefix at its cut."""

import asyncio
import re
import time

from flushed_owner_prefix import collect_flushed_owner_prefix
from live_owner_prefix import strict_json
from process_clock_provenance import join_process_clocks, prove_owned_process_clock


SOURCE_FIELDS = {
    "source_sha256", "input_post_source_sha256",
    "queue_observer_sha256", "core_source_sha256",
}


async def collect_clocked_owner_cut(
    *, native_acquire, native_startup, native_process, native_sources,
    expected_capture_id, gateway_sync, gateway_async, gateway_startup,
    telemetry_path, initial_closure=None, bootstrap=False, clock=time.monotonic, **prefix_options,
):
    """Callbacks/clients and startup receipts must be authenticated and source-bound.

    The synchronous clients run in worker threads so the existing before/acquire/
    after procfs proof brackets real fresh calls without blocking traffic tasks.
    No initial-capture closure or native absence/residual inference is performed.
    """
    if (not isinstance(native_sources, dict) or set(native_sources) != SOURCE_FIELDS
            or any(not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None
                   for value in native_sources.values())
            or not isinstance(expected_capture_id, str) or not expected_capture_id):
        raise ValueError("clocked_cut_explicit_native_sources_required")
    if "baseline_native_ids" in prefix_options:
        raise ValueError("clocked_cut_unbound_baseline_forbidden")
    native_box = {}

    def acquire_native():
        body = native_acquire()
        native_box["body"] = body
        return body["native_phase"]

    native_proof = await asyncio.to_thread(
        prove_owned_process_clock, native_startup, role="native",
        acquire=acquire_native, native_process=native_process,
        clock=clock, max_age_s=prefix_options["max_age_s"],
    )
    body = native_box["body"]
    phase = body["native_phase"]
    if (body.get("complete") is not True or phase.get("complete") is not True
            or phase.get("schema") != "r24-native-live-phase-v1"
            or any(phase.get(key) != value for key, value in native_sources.items())
            or not isinstance(phase.get("capture"), dict)
            or phase["capture"].get("trial_id") != expected_capture_id
            or phase["capture"].get("status") != "running"):
        raise ValueError("clocked_cut_native_source_capture_or_completeness")
    from capture_owner_closure import bootstrap_warm_ids, bound_baseline_ids

    if bootstrap:
        if initial_closure is not None:
            raise ValueError("clocked_cut_bootstrap_or_bound_closure_required")
        baseline = bootstrap_warm_ids(body, native_startup)
    elif initial_closure is not None:
        baseline = bound_baseline_ids(
            initial_closure, phase, native_startup,
            expected_gateway_run_id=prefix_options["expected_run_id"],
        )
    else:
        baseline = set()
    native_ids = (set(phase.get("publications", {})) | set(phase.get("lifecycle", {}))) - baseline
    if {"native_ids", "native_run_id", "expected_native_run_id", "cut_at"} & prefix_options.keys():
        raise ValueError("clocked_cut_derived_native_join_cannot_be_overridden")
    gateway_box = {}

    class ClockBoundGateway:
        async def get(self, path):
            if path != "/admin/mpc/state" or gateway_box:
                raise ValueError("clocked_cut_single_fresh_gateway_get_required")

            def acquire():
                response = gateway_sync.get(path)
                response.raise_for_status()
                gateway_box["response"] = response
                return strict_json(response.content)

            proof = await asyncio.to_thread(
                prove_owned_process_clock, gateway_startup, role="gateway",
                acquire=acquire, clock=clock, max_age_s=prefix_options["max_age_s"],
            )
            gateway_box["proof"] = proof
            return gateway_box["response"]

        async def post(self, path, **kwargs):
            return await gateway_async.post(path, **kwargs)

    owner = await collect_flushed_owner_prefix(
        ClockBoundGateway(), telemetry_path, clock=clock,
        cut_at=phase["source_at"],
        native_ids=tuple(sorted(native_ids)),
        native_run_id=phase["run_id"],
        expected_native_run_id=native_proof["run_id"],
        **prefix_options,
    )
    checked = clock()
    common = join_process_clocks(
        gateway_box["proof"], native_proof, now=checked,
        max_age_s=prefix_options["max_age_s"],
    )
    if (common["clock_domain"] != owner["clock_domain"]
            or common["native_run_id"] != phase["run_id"]
            or common["gateway_run_id"] != owner["run_id"]):
        raise ValueError("clocked_cut_process_epoch_or_clock_join")
    return {
        "schema": "r24-clocked-owner-cut-v1",
        "cut_at": phase["source_at"], "validated_at": checked,
        "oldest_source_age_s": common["oldest_source_age_s"],
        "native_observation": body, "owner_prefix": owner,
        "baseline_native_ids": sorted(baseline),
        "native_process_proof": native_proof,
        "gateway_process_proof": gateway_box["proof"],
        "common_clock_proof": common,
        "capture_initialization_verified": False,
        "native_common_cut_verified": False,
        "native_absence_verified": False,
        "deployable": False, "native_mpc_permitted": False,
        "scope": "Native-clock cut and strictly later Gateway received prefix, under owned-process/source trust. Initial-drain/capture history closure, owner-to-native phase classification and residual scenarios remain separate obligations.",
    }
