"""Causal, all-population Azure development replay against an owned native stack."""

import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

from run_h16 import ROOT, digest, write
from score_http import score


async def export_prepared_copy(native, output, tid, closed_capture, extension, *,
                               clock_domain, bridge_run_id, plan_sha256, previous_native_run):
    from prepared_copy_ready_plugin import EXPORT_PREFIX, validate_export
    from collect_batch_qualification import write_new

    sidecar_path = output / f"{tid}-prepared-copy-ready.json"
    receipt_path = output / f"{tid}-prepared-copy-receipt.json"
    receipt = {
        "native_capture_file_sha256": digest(output / f"{tid}-capture.json"),
        "valid": False, "discard_owned_stack_required": True,
        "scope": "Any refusal aborts the entire attempt; nonzero collector "
                 "exit triggers owned_engine teardown, never worker reuse.",
    }
    try:
        sent = time.monotonic()
        response = await native.post("/pr33-native/batch", json={
            "operation": "status", "trial_id": EXPORT_PREFIX + tid,
        })
        write_new(sidecar_path, {
            "sent_at": sent, "received_at": time.monotonic(),
            "status": response.status_code, "body": response.text,
        })
        receipt["sidecar_sha256"] = digest(sidecar_path)
        response.raise_for_status()
        exported = response.json()
        gate = validate_export(exported, closed_capture, extension,
            clock_domain=clock_domain, bridge_run_id=bridge_run_id, plan_sha256=plan_sha256)
        if previous_native_run is not None and gate["native_run_id"] != previous_native_run:
            raise ValueError("prepared_copy_native_epoch_changed")
        receipt.update(valid=True, discard_owned_stack_required=False, gate=gate)
    except BaseException as error:
        receipt["refusal"] = type(error).__name__
        write_new(receipt_path, receipt)
        raise
    write_new(receipt_path, receipt)
    return gate["native_run_id"], receipt_path


async def run(args):
    import httpx
    from mpc_lab.actuator import GuardedActuator, IdentificationPermit
    from mpc_lab.contracts import CostSpec, Decision, ModelParameters
    from mpc_lab.http_capture import capture_chat, drain_evidence
    from mpc_lab.runtime import local_clock_domain, snapshot_from_dict
    from mpc_local.control import SOURCE_HASHES
    from trace_workload import load_trace, verify_prompts
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "experiments/mpc-research/native-v2"))
    from collect_batch_qualification import CaptureClient

    plan = json.loads(args.plan.read_text())
    if plan.get("prepared_forward_extension") is not None:
        from prepared_forward_plugin import validate_plan
        validate_plan(plan)
    copy_extension = plan.get("prepared_copy_ready_extension")
    if copy_extension is not None:
        from prepared_copy_ready_plugin import validate_plan as validate_copy_plan
        validate_copy_plan(plan)
    queue_extension = plan.get("queue_observer_extension")
    input_extension = plan.get("input_post_extension")
    phase_extension = plan.get("phase_frontier_extension")
    if phase_extension and (not queue_extension or not input_extension):
        raise ValueError("phase_requires_queue_and_input_extensions")
    if phase_extension and any(
        trial.get("queue_observer_poll", True) is not True for trial in plan["trials"]
    ):
        raise ValueError("phase_requires_queue_poll_every_trial")
    config = json.loads(args.config.read_text())
    protocol = json.loads(args.protocol.read_text())
    trace = load_trace(args.trace)
    if digest(args.trace) != plan["trace_sha256"] or any(
        trial["duration_s"] > trace["duration_seconds"] for trial in plan["trials"]
    ):
        raise ValueError("frozen_trace_or_window_changed")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("fresh_result_directory_required")
    output.mkdir(mode=0o700)
    shared_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], local_files_only=True)
    verify_prompts(trace, tokenizer)
    parameters = ModelParameters(
        cost_spec=CostSpec(**protocol["cost_spec"]), model_version="native-id-unfitted-v1",
        calibration_version="unfitted-not-validated",
    )
    domain = local_clock_domain()
    observer_sha = SOURCE_HASHES["experiments/mpc-research/native_observer/pr33_native_observer.py"]
    index = {
        "schema": "r23-azure-native-development-v1", "completed": False,
        "trace_sha256": digest(args.trace), "plan_sha256": digest(args.plan),
        "collector_sha256": digest(__file__), "score_source_sha256": digest(Path(__file__).with_name("score_http.py")),
        "clock_domain": domain, "model_identity": asdict(parameters.identity), "episodes": [],
        "scope": "Development trace-shaped synthetic prompts; fixed caps, natural EOS; no original text or historical G",
        "mpc_active": False, "untouched_heldout": False,
    }
    copy_native_run = None

    def persist_index():
        (output / "index.json").write_text(json.dumps(index, sort_keys=True, indent=2, allow_nan=False))

    persist_index()
    try:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8080", timeout=5, trust_env=False,
            headers={"x-admin-api-key": config["ADMIN_API_KEY"]},
        ) as admin, httpx.AsyncClient(
            base_url="http://127.0.0.1:8000", timeout=5, trust_env=False,
            headers={"x-observer-key": config["ADMIN_API_KEY"]},
        ) as native, httpx.AsyncClient(
            base_url="http://127.0.0.1:8080", timeout=50, trust_env=False,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
        ) as traffic:
            response = await admin.get("/admin/mpc/state")
            write(output / "initial-gateway-reply.json", {"status": response.status_code, "body": response.text})
            response.raise_for_status()
            initial = snapshot_from_dict(response.json())
            if initial.identity != parameters.identity or initial.clock_domain != domain:
                raise ValueError("initial_identity_or_clock")
            driver = GuardedActuator(
                admin, parameters, output / "actuator-pending.json", enabled=True, max_age_s=0.5,
                permit=IdentificationPermit(
                    parameters.identity, approved=True, run_id=initial.run_id,
                    clock_domain=domain, expires_at=time.monotonic() + 3600,
                ),
            )
            capture = CaptureClient(native, output, plan, domain)
            await capture.call("metadata")
            input_calls, input_epoch = 0, None

            async def export_input_posts(ids):
                nonlocal input_calls, input_epoch
                sent = time.monotonic()
                response = await native.post("/pr33-native/input-post-records",
                                             json={"request_ids": ids})
                received = time.monotonic()
                input_calls += 1
                path = output / f"input-post-utility-{input_calls:04d}.json"
                write(path, {"request_ids": ids, "sent_at": sent, "received_at": received,
                             "status": response.status_code, "body": response.text})
                response.raise_for_status()
                value = response.json()
                if (value.get("schema") != "r23-native-input-post-records-v1"
                        or value.get("source_sha256") != input_extension["source_sha256"]
                        or value.get("core_source_sha256") != input_extension["vllm_core_sha256"]
                        or value.get("observer_sha256") != observer_sha
                        or value.get("clock_domain") != domain
                        or not isinstance(value.get("run_id"), str) or not value["run_id"]
                        or input_epoch is not None and value["run_id"] != input_epoch
                        or not sent <= value["sampled_at"] <= received
                        or value.get("complete") is not True
                        or set(value.get("records", {})) != set(ids)):
                    raise ValueError("input_post_export_source_epoch_clock_or_inventory")
                input_epoch = value["run_id"]
                return value, path

            if input_extension:
                await export_input_posts([])
            index["shared_initialization"] = {
                "started_at": shared_started, "finished_at": time.monotonic(),
                "scope": "Tokenizer/clients/initial identity and capture metadata; common to all arms",
            }
            engine_run = None
            for trial in plan["trials"]:
                queue_poll = bool(queue_extension) and trial.get("queue_observer_poll", True)
                tid = trial["trial_id"]
                accounting_start = time.monotonic()
                selected = [row for row in trace["requests"] if row["arrival_seconds"] < trial["duration_s"]]
                ids = trial["request_ids"]
                if len(ids) != len(selected) or len(set(ids)) != len(ids):
                    raise ValueError("frozen_trial_population")
                episode = {
                    **trial, "gateway_run_id": initial.run_id, "expected_ids": ids,
                    "native_frames": [], "actions": [], "public_offer_count": 0,
                    "accounting_started_at": accounting_start,
                }
                rows = {rid: {"request_id": rid, "status": "not_launched"} for rid in ids}
                public_inputs = {rid: item["input_tokens"] for rid, item in zip(ids, selected)}
                stream = (output / f"{tid}.events.jsonl").open("x")
                last_sync = time.monotonic()

                def event(kind, value):
                    nonlocal last_sync
                    stream.write(json.dumps({"kind": kind, "value": value}, allow_nan=False) + "\n")
                    if time.monotonic() - last_sync >= 1:
                        stream.flush()
                        os.fsync(stream.fileno())
                        last_sync = time.monotonic()

                async def sample():
                    nonlocal engine_run
                    gateway_reply = await admin.get("/admin/mpc/state")
                    gateway_received = time.monotonic()
                    event("gateway_reply", {"status": gateway_reply.status_code, "body": gateway_reply.text,
                                            "received_at": gateway_received})
                    gateway_reply.raise_for_status()
                    native_sent = time.monotonic()
                    native_reply = await native.get(
                        "/pr33-native/phase-state" if phase_extension else
                        "/pr33-native/queue-state" if queue_poll else "/pr33-native/state")
                    received = time.monotonic()
                    event("native_reply", {"status": native_reply.status_code, "body": native_reply.text,
                                           "sent_at": native_sent, "received_at": received})
                    native_reply.raise_for_status()
                    gateway, frame = gateway_reply.json(), native_reply.json()
                    queue_observation = None
                    if queue_poll:
                        queue_observation = {k: v for k, v in frame.items() if k != "base"}
                        if (frame.get("schema") != "r23-native-queue-read-only-v1"
                                or frame.get("queue_observer_sha256") != queue_extension["source_sha256"]
                                or frame.get("complete") is not True
                                or not frame["sample_started_at"] <= frame["base"]["sampled_at"]
                                <= frame["sample_finished_at"] <= received):
                            raise ValueError("queue_observer_source_scope_or_clock")
                        frame = frame["base"]
                    engine_run = engine_run or frame["run_id"]
                    if (gateway["run_id"] != initial.run_id or frame["run_id"] != engine_run
                            or gateway["clock_domain"] != domain or frame["clock_domain"] != domain
                            or frame["observer_sha256"] != observer_sha):
                        raise ValueError("native_source_epoch_or_clock_changed")
                    current = snapshot_from_dict(gateway)
                    if current.identity != parameters.identity:
                        raise ValueError("gateway_model_identity_changed")
                    if held_ack is not None and (
                        current.budget != held_ack["budget"]
                        or current.budget_version != held_ack["budget_version"]
                    ):
                        raise ValueError("fixed_cap_or_version_changed")
                    available = (
                        frame["complete"] and gateway["complete"]
                        and 0 <= received - frame["sampled_at"] <= 0.5
                        and 0 <= frame["frontend_sampled_at"] - frame["sampled_at"] <= 0.05
                        and 0 <= received - gateway["sampled_at"] <= 0.5
                    )
                    result = {**frame, "gateway_snapshot": gateway, "received_at": received,
                              "gateway_received_at": gateway_received, "available": available}
                    if queue_observation is not None:
                        result["queue_observation"] = queue_observation
                    if phase_extension:
                        phase = queue_observation.get("native_phase")
                        if (not isinstance(phase, dict)
                                or phase.get("schema") != phase_extension["schema"]
                                or phase.get("source_sha256") != phase_extension["source_sha256"]
                                or phase.get("input_post_source_sha256") != input_extension["source_sha256"]
                                or phase.get("queue_observer_sha256") != queue_extension["source_sha256"]
                                or phase.get("core_source_sha256") != input_extension["vllm_core_sha256"]
                                or phase.get("clock_domain") != domain
                                or phase.get("run_id") != engine_run
                                or type(phase.get("complete")) is not bool
                                or not isinstance(phase.get("refusal_reasons"), list)
                                or phase["complete"] != (not phase["refusal_reasons"])
                                or phase.get("native_mpc_permitted") is not False
                                or not native_sent <= phase["source_at"]
                                <= phase["sample_finished_at"] <= received):
                            raise ValueError("phase_source_epoch_clock_or_refusal_contract")
                        result["phase_available"] = phase["complete"] and available
                        result["phase_observation_cost"] = {
                            "sent_at": native_sent, "received_at": received,
                            "round_trip_s": received - native_sent,
                            "body_bytes": len(native_reply.content),
                            "scope": "Observed round trip/payload, not a disabled-relative causal overhead",
                        }
                    return result

                async def drain():
                    checks = []
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        frame = await sample()
                        check = drain_evidence(
                            snapshot_from_dict(frame["gateway_snapshot"]), frame,
                            identity=parameters.identity, gateway_run=initial.run_id,
                            engine_run=engine_run, domain=domain, now=time.monotonic(),
                            pending_operation=driver.path.exists(),
                        )
                        clear = (check["drained"] and frame["available"]
                                 and frame["pending_batches"] == frame["frontend_unfinished"] == 0)
                        if clear:
                            if not checks or frame["sampled_at"] > checks[-1]["sampled_at"]:
                                checks.append({"sampled_at": frame["sampled_at"], "evidence": check})
                        else:
                            checks = []
                        if len(checks) == 2:
                            return checks
                        await asyncio.sleep(0.1)
                    raise ValueError("strict_native_drain_not_established")

                stop = asyncio.Event()
                observer = work = None
                tasks = []
                held_ack = None

                def record(row):
                    rows[row["request_id"]] = row
                    write(output / f'{row["request_id"]}-http.json', row)
                    event("http_outcome", row)

                def observe(kind, row):
                    if kind == "offered":
                        episode["public_offer_count"] += 1
                        event("public_client_offer", {
                            "request_id": row["request_id"], "offered_at": row["sent_at"],
                            "observed_at": row["published_at"],
                            "prompt_tokens": public_inputs[row["request_id"]],
                            "max_tokens": row["requested_max_tokens"],
                            "scope": "Causal client-offered history, not Gateway-received or future tape",
                        })
                    else:
                        event("http_observation", {"kind": kind, **row})

                async def monitor():
                    nonlocal held_ack
                    last_gpu = -math.inf
                    steps = iter(trial.get("budget_steps", []))
                    next_step = next(steps, None)
                    while not stop.is_set():
                        if next_step and time.monotonic() >= episode["start"] + next_step["offset_s"]:
                            scheduled = episode["start"] + next_step["offset_s"]
                            invoked = time.monotonic()
                            reply = await admin.get("/admin/mpc/state")
                            reply.raise_for_status()
                            state = snapshot_from_dict(reply.json())
                            if (state.budget != held_ack["budget"]
                                    or state.budget_version != held_ack["budget_version"]):
                                raise ValueError("fixed_cap_or_version_changed")
                            decision = Decision(
                                next_step["budget"], None, "identification_assignment",
                                False, None, None, 0, 0, 0, parameters.identity,
                            )
                            result = await driver.submit(state, decision)
                            action = {"reply": result, "scheduled_at": scheduled,
                                      "invoked_at": invoked, "received_at": time.monotonic()}
                            episode["actions"].append(action)
                            event("scheduled_identification_control", action)
                            if (result["status"] != "applied"
                                    or result["acknowledgment"]["budget"] != next_step["budget"]):
                                raise ValueError("inflight_control_unacknowledged_raw_preserved")
                            held_ack = result["acknowledgment"]
                            next_step = next(steps, None)
                        frame = await sample()
                        episode["native_frames"].append(frame)
                        if time.monotonic() - last_gpu >= 5:
                            process = await asyncio.create_subprocess_exec(
                                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                "--format=csv,noheader", stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            try:
                                stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
                                event("gpu_observation", {"received_at": time.monotonic(),
                                      "returncode": process.returncode, "stdout": stdout.decode(),
                                      "stderr": stderr.decode()})
                            finally:
                                if process.returncode is None:
                                    process.kill()
                                    await process.wait()
                            last_gpu = time.monotonic()
                        await asyncio.sleep(0.1)

                try:
                    episode["initial_drain"] = await drain()
                    acknowledgment = None
                    for _ in range(3):
                        fence = await driver.reconcile()
                        if fence["status"] != "pending":
                            state_reply = await admin.get("/admin/mpc/state")
                            state_reply.raise_for_status()
                            state = snapshot_from_dict(state_reply.json())
                            decision = Decision(trial["budget"], None, "identification_assignment",
                                                False, None, None, 0, 0, 0, parameters.identity)
                            result = await driver.submit(state, decision)
                            episode["actions"].append({"fence": fence, "reply": result,
                                                       "received_at": time.monotonic()})
                            if result["status"] == "applied":
                                acknowledgment = result["acknowledgment"]
                                held_ack = acknowledgment
                                break
                        await asyncio.sleep(0.11)
                    if acknowledgment is None or acknowledgment["budget"] != trial["budget"]:
                        raise ValueError("fixed_cap_not_acknowledged")
                    episode["preassigned_drain"] = await drain()
                    await capture.call("begin", trial_id=tid)
                    if phase_extension:
                        health = await sample()
                        episode["phase_initial_health"] = health
                        phase = health["queue_observation"]["native_phase"]
                        if (not health["phase_available"]
                                or phase["capture"]["trial_id"] != tid
                                or phase["lifecycle"] or health["pending_batches"]):
                            raise ValueError("initial_live_phase_health_failed_raw_preserved")
                    start = time.monotonic() + 0.5
                    episode["start"] = start
                    for rid, item in zip(ids, selected):
                        scheduled = start + item["arrival_seconds"]
                        rows[rid]["scheduled_at"] = scheduled
                        payload = {
                            "model": trace["model"], "messages": [{"role": "user", "content": item["prompt"]}],
                            "temperature": 0, "max_tokens": item["max_tokens"],
                        }
                        tasks.append(asyncio.create_task(capture_chat(
                            traffic, "/v1/chat/completions", payload, rid, scheduled,
                            enabled=True, domain=domain, record=record, observe=observe,
                        )))

                    async def offered_window():
                        await asyncio.gather(*tasks)
                        await asyncio.sleep(max(0, start + trial["duration_s"] - time.monotonic()))

                    observer = asyncio.create_task(monitor())
                    work = asyncio.create_task(offered_window())
                    done, _ = await asyncio.wait((observer, work), return_when=asyncio.FIRST_COMPLETED)
                    if observer in done:
                        observer.result()
                        raise ValueError("native_observer_stopped_early")
                    await work
                    stop.set()
                    await observer
                    applied_actions = [
                        a for a in episode["actions"] if a["reply"]["status"] == "applied"
                    ]
                    if len(applied_actions) != 1 + len(trial.get("budget_steps", [])):
                        raise ValueError("scheduled_control_inventory_incomplete")
                    episode["final_drain"] = await drain()
                    if phase_extension:
                        final_health = await sample()
                        episode["phase_final_health"] = final_health
                        phase = final_health["queue_observation"]["native_phase"]
                        if not final_health["phase_available"]:
                            raise ValueError("final_live_phase_health_failed_raw_preserved")
                        history, utility = await export_input_posts(list(phase["publications"]))
                        if history["records"] != phase["publications"]:
                            raise ValueError("live_phase_publication_history_mismatches_drained_export")
                        episode["phase_publication_history"] = {
                            "utility_artifact": utility.name, "utility_sha256": digest(utility),
                            "export": history,
                        }
                    closed_capture = await capture.finish(tid)
                    if copy_extension is not None:
                        try:
                            copy_native_run, receipt_path = await export_prepared_copy(
                                native, output, tid, closed_capture, copy_extension,
                                clock_domain=domain, bridge_run_id=capture.run_id,
                                plan_sha256=capture.plan_sha, previous_native_run=copy_native_run)
                        except BaseException:
                            episode["prepared_copy_capture_refused"] = True
                            raise
                        episode["prepared_copy_receipt"] = receipt_path.name
                        episode["prepared_copy_receipt_sha256"] = digest(receipt_path)
                    if plan.get("prepared_forward_extension") is not None:
                        from native_physics_support import validate_prepared_worker
                        prepared_workers = closed_capture["capture"].get("workers")
                        if type(prepared_workers) is not list or len(prepared_workers) != 1:
                            raise ValueError("native_physics_single_prepared_worker_required")
                        try:
                            validate_prepared_worker(prepared_workers[0])
                        except ValueError as error:
                            episode["prepared_capture_refused"] = str(error)
                            write(output / f"{tid}-prepared-refusal.json", {
                                "reason": str(error),
                                "native_capture_file_sha256": digest(output / f"{tid}-capture.json"),
                                "discard_owned_stack_required": True,
                                "scope": "Abort this entire attempt;collector nonzero exit causes "
                                         "owned_engine teardown,not worker reuse or successful retry.",
                            })
                            raise
                    if input_extension:
                        exported, utility = await export_input_posts(
                            list(closed_capture["capture"]["requests"]))
                        record_path = output / f"{tid}-input-post.json"
                        write(record_path, {
                            "schema": "r23-native-input-post-receipt-v1",
                            "extension": input_extension, "export": exported,
                            "native_capture_file_sha256": digest(output / f"{tid}-capture.json"),
                            "utility_artifact": utility.name, "utility_sha256": digest(utility),
                        })
                        episode["input_post_artifact"] = record_path.name
                        episode["input_post_sha256"] = digest(record_path)
                    if any(row.get("observation_error") for row in rows.values()):
                        raise ValueError("causal_public_observation_failed")
                    if episode["public_offer_count"] != len(ids):
                        raise ValueError("scheduled_population_not_all_offered")
                    for rid, row in rows.items():
                        if row["status"] == "complete":
                            usage = row.get("usage")
                            if (not isinstance(usage, dict)
                                    or usage.get("prompt_tokens") != public_inputs[rid]
                                    or type(usage.get("completion_tokens")) is not int
                                    or not 1 <= usage["completion_tokens"] <= trace["output_cap"]):
                                raise ValueError("completed_http_usage_or_prompt_mismatch")
                    episode["accounting_finished_at"] = time.monotonic()
                    episode["score"] = score(
                        list(rows.values()), trial["duration_s"], offered_start=start,
                        accounting_started_at=accounting_start,
                        accounting_finished_at=episode["accounting_finished_at"],
                    )
                    episode["completed"] = True
                finally:
                    stop.set()
                    for task in [*tasks, work, observer]:
                        if task is not None and not task.done():
                            task.cancel()
                    await asyncio.gather(*(task for task in [*tasks, work, observer] if task is not None),
                                         return_exceptions=True)
                    episode["rows"] = list(rows.values())
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    write(output / f"{tid}.json", episode)
                    index["episodes"].append({"trial_id": tid, "artifact": f"{tid}.json",
                                              "completed": episode.get("completed", False)})
                    persist_index()
                gate = plan.get("prompt_behavior_gate")
                if gate and gate["after_trial"] == tid:
                    complete = [r for r in rows.values() if r["status"] == "complete"]
                    natural = [r for r in complete if r.get("finish_reason") == "stop"]
                    tiny = [r for r in natural if r["usage"]["completion_tokens"] <= 2]
                    enough_stops = (
                        bool(complete) and len(natural) / len(complete)
                        >= gate["minimum_natural_stop_fraction_of_completions"]
                    )
                    not_degenerate = (
                        bool(complete) and len(tiny) / len(complete)
                        <= gate["maximum_natural_g_le2_fraction_of_completions"]
                    )
                    write(output / (tid + "-prompt-gate.json"), {
                        "rule": gate, "all_offered": len(rows), "complete": len(complete),
                        "natural_stop": len(natural), "natural_g_le2": len(tiny),
                        "passed": enough_stops and not_degenerate,
                        "scope": "Diagnostic stopping only; original all-offered SLO rule is unchanged",
                    })
                    if not enough_stops or not not_degenerate:
                        raise ValueError("prospective_prompt_behavior_gate_failed_raw_preserved")
            index["completed"] = True
    finally:
        persist_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "config", "protocol", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
