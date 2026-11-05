"""Default-off acquisition boundary; reuse the unchanged owner-cut contracts."""

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import tomllib

import frozen01 as frozen


class Refused(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def number(value):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            "nonfinite_or_negative_clock")
    return value


def document(name):
    return json.loads((frozen.HERE / name).read_bytes())


def permissions():
    return dict(native_execution_authorized=False, native_mpc_permitted=False,
                qualified_model=False, prediction_qualified=False, deployable=False,
                actual_application_at=None, exact_application_time_gate=False,
                acquisition_token=None, request_token=None)


def source_feasibility():
    frozen.verify()
    proposed = document("PROPOSED_NATIVE_PLAN.json")
    current = document("inputs/control-launch02/startup-instrumentation.json")
    copied = deepcopy(current)
    for key in ("input_post_extension", "phase_frontier_extension"):
        copied[key] = proposed[key]
    copy_plugin = frozen.module("prepared_copy_ready_plugin")
    try:
        copy_plugin.validate_plan(copied)
    except ValueError as error:
        conflict = str(error)
    else:
        raise Refused("expected_current_route_exclusion_not_reproduced")
    require(conflict == "prepared_copy:prepared_copy_exclusive_route_required",
            "unexpected_current_route_failure")
    frozen.module("prepared_forward_plugin").validate_plan(proposed)
    frozen.module("batch_capture_bridge").validate_plan(proposed)
    require("prepared_copy_ready_extension" not in proposed,
            "proposed_route_must_not_bypass_exclusion")
    declaration = document("inputs/source-tree/experiments/prefill-vllm-v030-source-map.json")
    prior = document("inputs/native02/runtime-identity.json")["installed_vllm_sources"]
    require(prior["files"] == declaration["sha256"]
            and prior["version"] == declaration["version"]
            and prior["files"]["v1/engine/core.py"] == frozen.sources()["core_source_sha256"],
            "pinned_SDK_source_disagreement")
    registry = tomllib.loads((frozen.NATIVE / "pyproject.toml").read_text())["project"]["entry-points"]
    required = {
        "vllm.general_plugins": {
            "pr33_r23_input_post_observer": "input_post_observer:register",
            "pr33_r24_phase_observer": "phase_observer:register",
        },
        "vllm.endpoint_plugins": {
            "pr33_r23_input_post_observer": "input_post_observer:Endpoint",
            "pr33_r24_phase_observer": "phase_observer:Endpoint",
        },
    }
    require(all(registry[group].get(key) == value
                for group, values in required.items() for key, value in values.items()),
            "source_package_missing_phase_entrypoints")
    return dict(
        schema="team16-owner-acquisition-source-feasibility-v1",
        status="SOURCE_FEASIBLE_WITH_PENDING_LAUNCH_AMENDMENTS",
        current_control02_route_compatible=False, current_route_refusal=conflict,
        proposed_plan_static_guards_passed=True,
        source_phase_hooks_present=True, observer_source_shim_required=False,
        pinned_SDK_files=len(prior["files"]), pinned_SDK_core=prior["files"]["v1/engine/core.py"],
        SDK_evidence_scope="Exact prior Native02 hashes and source ABI guards; not a current installed SDK/entrypoint inspection.",
        installed_entrypoints_verified=False,
        live_launcher_ready=False, mayor_amendment_approval_required=True,
        native_sources=frozen.sources(), required_entrypoints=required,
        gateway_requirement="Flush process_identity must equal the actual owned single-process Gateway leader, not safe_exec.",
        readiness_requirement="Retain actual warmup/full frames; bind base readiness frames and actual same-budget0->1 CAS before capture.",
        **permissions())


def receipt_body(record, *, method, path, cutoff):
    api = frozen.control()
    require(record.get("method") == method and record.get("path") == path
            and record.get("http_status") == 200, "HTTP_schema_or_status")
    require(record.get("auth_values_captured") is False
            and isinstance(record.get("response_headers"), dict)
            and set(record["response_headers"]) <= {"content-type", "x-request-id"},
            "credential_safe_capture_required")
    sent, received, cutoff = map(number, (
        record.get("client_send_at"), record.get("client_receive_at"), cutoff))
    require(sent <= number(record.get("client_headers_at")) <= received <= cutoff,
            "receipt_after_permitted_cutoff")
    return api.body(record)


def bootstrap_inputs(startup, *, cutoff):
    api = frozen.control()
    settings = frozen.settings()
    native, gateway = startup["native_owned"], startup["gateway_owned"]
    frames = startup["drain_records"]
    require(len(frames) == 2, "two_actual_full_readiness_frames_required")
    full = []
    for record in frames:
        receipt_body(record, method="GET", path=settings["final_queue_path"], cutoff=cutoff)
        frame, empty = api.full_queue_observation(
            record, settings, native_run=full[0]["base"]["run_id"] if full else None)
        require(empty, "initial_physical_stock_not_empty")
        full.append(frame)
    require(native["readiness"]["frames"] == [row["base"] for row in full],
            "base_readiness_not_bound_to_full_receipts")
    warm = receipt_body(startup["warmup_record"], method="POST",
                        path="/v1/chat/completions", cutoff=cutoff)
    require(native["readiness"]["warmup_response"] == warm,
            "warmup_response_binding")
    require(startup["warmup_record"]["client_clock_domain"] == full[0]["base"]["clock_domain"]
            and json.loads(startup["warmup_record"]["request_body"])["max_tokens"] == 16
            and type(warm["usage"]["completion_tokens"]) is int
            and 1 <= warm["usage"]["completion_tokens"] <= 16,
            "warmup_clock_or_declared_cap")
    require(startup["warmup_record"]["client_receive_at"] <= frames[0]["client_send_at"]
            and frames[0]["client_receive_at"] <= frames[1]["client_send_at"],
            "warmup_drain_chronology")
    records = [startup[name] for name in (
        "gateway_initial_record", "cas_command_record", "gateway_confirm_record")]
    for record, method, path in zip(records, ("GET", "POST", "GET"),
                                    (settings["state_path"], settings["command_path"],
                                     settings["state_path"])):
        receipt_body(record, method=method, path=path, cutoff=cutoff)
    before = api.snapshot(records[0])
    require(before.budget == 32 and before.budget_version == 0
            and before.sequence == 0 and not before.requests,
            "fresh_empty_gateway32_version0_required")
    request = json.loads(records[1]["request_body"])
    step = dict(before_record=0, command_record=1, after_records=[2], budget=32,
                control="fresh", expected_http=200, operation_id=request["operation_id"])
    confirmation = api.evaluate_step(step, records, before, settings)
    for record in records:
        require(number(record["client_receive_at"]) <= number(cutoff),
                "readiness_receipt_after_cutoff")
    initial = api.body(records[0])
    ack = api.body(records[1])
    expected = dict(status="applied", command=request, acknowledgment=ack)
    require(gateway["readiness"]["snapshot"] == initial
            and gateway["readiness"]["same_budget_ack"] == expected,
            "readiness_CAS_wire_binding")
    require(confirmation["actual_applied_at"] is None
            and confirmation["actual_application_before_deadline"] == "established_by_interval_upper",
            "CAS_interval_unknown_or_late")
    require(api.snapshot(records[2], binding=before).sequence == 1,
            "unexpected_activity_before_capture")
    plan = document("PROPOSED_NATIVE_PLAN.json")
    expected_plan = api.sha(api.canonical(plan))
    bridge_sha = hashlib.sha256((frozen.SOURCE / (
        "experiments/mpc-research/native-v2/batch_capture_bridge.py")).read_bytes()).hexdigest()
    batch = []
    for key, operation in (("capture_metadata_record", "metadata"), ("capture_begin_record", "begin")):
        record = startup[key]
        reply = receipt_body(record, method="POST", path="/pr33-native/batch", cutoff=cutoff)
        payload = json.loads(record["request_body"])
        expected_payload = {"operation": operation}
        if operation == "begin":
            expected_payload["trial_id"] = plan["trials"][0]["trial_id"]
        require(payload == expected_payload, "capture_command_binding")
        require(reply.get("schema") == "native-v2-batch-capture-v1"
                and reply.get("bridge_sha256") == bridge_sha
                and reply.get("plan_sha256") == expected_plan
                and reply.get("source_commit") == plan["source_commit"]
                and reply.get("clock_domain") == before.clock_domain
                and record.get("client_clock_domain") == before.clock_domain
                and isinstance(reply.get("run_id"), str) and bool(reply["run_id"])
                and number(record["client_send_at"]) <= number(reply.get("command_started_at"))
                <= number(record["client_receive_at"]), "capture_source_epoch_or_clock")
        batch.append(reply)
    meta, begin = batch
    expected_sdk = document("inputs/source-tree/experiments/prefill-vllm-v030-source-map.json")
    runtime = meta["runtime"]
    require(runtime["source"]["files"] == expected_sdk["sha256"]
            and runtime["source"]["version"] == expected_sdk["version"]
            and type(runtime["tensor_parallel_size"]) is int
            and type(runtime["pipeline_parallel_size"]) is int
            and runtime["tensor_parallel_size"] == runtime["pipeline_parallel_size"] == 1,
            "fresh_capture_metadata_SDK_binding")
    workers = runtime["workers"]
    clocks = meta["worker_clocks"]
    require(len(workers) == len(clocks) == 1
            and workers[0]["source"] == runtime["source"]
            and clocks[0]["clock_domain"] == before.clock_domain
            and clocks[0]["bridge_sha256"] == bridge_sha
            and type(clocks[0]["process_id"]) is int and clocks[0]["process_id"] > 0,
            "capture_worker_source_clock_binding")
    require(meta["run_id"] == begin["run_id"]
            and begin.get("begin") == {"external": True}
            and begin.get("trial_id") == plan["trials"][0]["trial_id"],
            "capture_begin_or_epoch_binding")
    begin_record = startup["capture_begin_record"]
    confirmation_available = records[2]["client_receive_at"]
    require(confirmation_available <= startup["capture_metadata_record"]["client_send_at"]
            and startup["capture_metadata_record"]["client_receive_at"] <= begin_record["client_send_at"]
            and frames[-1]["client_receive_at"] <= begin_record["client_send_at"],
            "capture_before_confirmed_readiness")
    return dict(
        confirmation=confirmation, confirmation_available_at=confirmation_available,
        full_readiness=full, metadata=meta, begin_record=begin_record,
        capture_id=plan["trials"][0]["trial_id"])


def inspect_cut(cut, *, consume_at, deadline):
    now, deadline = number(consume_at), number(deadline)
    source, available = number(cut["cut_at"]), number(cut["validated_at"])
    require(source <= available <= now <= deadline and now - source <= 0.5,
            "cut_stale_future_or_after_absolute_deadline")
    prefix = cut["owner_prefix"]
    require(prefix["gateway_prefix_closed"] is True
            and prefix.get("acquisition") == "explicit_serialized_flush",
            "flushed_ledger_prefix_required")
    barrier = prefix["barrier"]
    times = [number(prefix["barrier_sent_at"]),
             *(number(barrier[name]) for name in (
                 "source_at", "flush_started_at", "flush_finished_at", "sample_finished_at")),
             number(prefix["barrier_received_at"]), available]
    require(times == sorted(times), "flush_source_outside_collector_brackets")
    proc = frozen.module("process_clock_provenance")
    metadata = proc.metadata_identity(prefix["barrier"].get("process_identity"))
    chain = cut["gateway_process_proof"]["owned_chain"]
    require(bool(chain) and metadata == (
        chain[0]["pid"], chain[0]["start_ticks"], chain[0]["pid_namespace"]),
        "gateway_serving_pid_not_owned_proof_target")
    phase = cut["native_observation"]["native_phase"]
    require(all(phase.get(key) == value for key, value in frozen.sources().items()),
            "native_phase_exact_source_binding")
    require(phase.get("complete") is True
            and isinstance(phase.get("lifecycle"), dict)
            and isinstance(phase.get("publications"), dict),
            "missing_or_incomplete_phase_history")
    stock = cut["native_observation"]
    return dict(
        source_at=source, available_at=available, consumed_at=now,
        full_stock=deepcopy({key: stock[key] for key in (
            "base", "scheduler_state", "queue_oldest_first", "queued_ghost_request_ids",
            "skipped_waiting", "streaming_waiting")}),
        gateway_budget_version=prefix["snapshot"]["budget_version"],
        **permissions())


async def collect_initial(
    *, enabled=False, startup=None, native_get=None, journal=None, deadline=None,
    clock=time.monotonic, **source_options,
):
    if enabled is not True:
        return dict(status="disabled", **permissions())
    require(isinstance(journal, dict) and not journal, "new_owned_journal_required")
    journal.update(status="started", schema="team16-owner-bootstrap-boundary-v1")
    try:
        frozen.verify()
        require(number(clock()) < number(deadline), "initial_deadline_exhausted")
        bound = bootstrap_inputs(startup, cutoff=clock())
        require(source_options["native_sources"] == frozen.sources()
                and source_options["expected_capture_id"] == bound["capture_id"]
                and source_options["max_age_s"] == 0.5,
                "unchanged_native_source_capture_age_contract")
        require(bound["metadata"]["runtime"]["process_id"] ==
                source_options["native_process"]["pid"], "capture_core_process_binding")
        settings = {**frozen.settings(), "final_queue_path": "/pr33-native/phase-state"}

        def acquire():
            record = native_get()
            require(record.get("auth_values_captured") is False
                    and isinstance(record.get("response_headers"), dict)
                    and set(record["response_headers"]) <= {"content-type", "x-request-id"},
                    "credential_safe_capture_required")
            journal["native_record"] = deepcopy(record)
            receipt_body(record, method="GET", path=settings["final_queue_path"], cutoff=clock())
            frame, _ = frozen.control().full_queue_observation(
                record, settings, native_run=startup["native_owned"]["readiness"]["frames"][0]["run_id"],
                deadline=deadline)
            phase = frame["native_phase"]
            require(record["client_send_at"] <= number(phase["source_at"])
                    <= number(phase["sample_finished_at"]) <= record["client_receive_at"],
                    "phase_source_outside_collector_brackets")
            return frame

        cut = await frozen.module("clocked_owner_cut").collect_clocked_owner_cut(
            native_acquire=acquire, native_startup=startup["native_owned"],
            gateway_startup=startup["gateway_owned"], bootstrap=True,
            clock=clock, **source_options)
        journal["cut"] = deepcopy(cut)
        view = inspect_cut(cut, consume_at=clock(), deadline=deadline)
        phase = cut["native_observation"]["native_phase"]
        started = number(phase["capture"]["started_at"])
        begin = bound["begin_record"]
        upper = number(bound["confirmation"]["actual_application_interval"][1])
        require(bound["confirmation_available_at"] <= begin["client_send_at"] <= started
                <= begin["client_receive_at"] <= cut["cut_at"]
                and upper <= started, "confirmed_receipt_interval_must_precede_capture")
        closure = frozen.module("capture_owner_closure").establish_initial_capture_closure(
            cut, native_startup=startup["native_owned"], gateway_startup=startup["gateway_owned"])
        journal.update(
            status="initial_closure_contract_satisfied",
            initial_closure=asdict(closure), observation=view,
            confirmation=bound["confirmation"], **permissions())
        require(number(clock()) <= deadline, "post_validation_absolute_deadline")
        return closure, journal
    except BaseException as error:
        journal.update(status="refused", error_type=type(error).__name__,
                       reason=str(error), **permissions())
        raise


def historical_native02():
    api, settings = frozen.control(), frozen.settings()
    legacy = api.native_state(document("inputs/native02/http/035.json"), settings)
    full = [api.full_queue_observation(document(f"inputs/native02/http/{name}.json"),
                                       settings, native_run=legacy["run_id"])
            for name in ("036", "037")]
    return dict(
        scope="READ_ONLY_PRIOR_ACCEPTED_NATIVE02_SUBSET_NOT_NEW_NATIVE_EVIDENCE",
        legacy_pending_batches=legacy["pending_batches"],
        full_empty=[empty for _, empty in full],
        native_phase_present=["native_phase" in frame for frame, _ in full],
        owner_bootstrap_admissible=False,
        refusal="No native_phase/capture/process-clock/flushed owner-prefix/initial readiness-CAS closure in this prior evidence.",
        **permissions())
