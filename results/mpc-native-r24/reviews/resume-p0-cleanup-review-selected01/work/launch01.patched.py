#!/usr/bin/env python3
"""Default-off, one-stack Mayor-owned fixed32 acquisition launch."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition_common01 as c
from owned01 import Owned
from authorization02 import require_activation


def alarm(signum, frame):
    raise TimeoutError("owned_absolute_phase_deadline")


def arm(deadline):
    c.require(deadline > time.monotonic(), "phase_deadline_already_exhausted")
    signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())


def fresh_host(prereg):
    row = c.load(Path(prereg["fresh_host_receipt"]))
    expected = c.load(c.BASE / "inputs/host-readonly01.json")
    c.require(row["scope"] == "READ_ONLY_PREREQUISITES_NOT_NATIVE_EXECUTION"
              and row["boot_id"] == c.domain().split(":")[1]
              and 0 <= time.time() - row["received_at_unix"] <= 3600
              and row["compute_apps"] == [] and row["listeners_8000_8080"] == {}
              and row["gpu"] == ["GPU-8610519d-3680-ae5f-cdfc-40037bfb5013, 1, 0"]
              and row["installed_SDK_matches_source_map"] is True
              and row["installed_SDK_sha256"] == expected["installed_SDK_sha256"]
              and row["entrypoints"] == expected["entrypoints"]
              and row["source13_first_find_spec_origins"] == expected["source13_first_find_spec_origins"]
              and row["installed_or_modified_sources"] is False
              and row["native_modules_imported"] is False and row["servers_or_inference_started"] is False,
              "fresh_same_boot_readonly_prerequisites")
    return row


def acquire_lock(path=c.LOCK):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        c.require(info.st_uid == os.getuid() and info.st_nlink == 1
                  and info.st_mode & 0o777 == 0o600, "private_GPU_lock_identity")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def marker(directory, name, started):
    path = directory / name
    if not path.exists():
        return None
    value = c.load(path)
    now = time.monotonic()
    key = "finished_at" if name == "protocol-finished.json" else "started_at"
    point = value[key]
    c.require(type(point) in (float, int) and started <= point <= now
              and value["clock_domain"] == c.domain(), "future_unknown_phase_marker")
    if name == "active-start.json":
        c.require(point <= started + 180 and value["deadline"] == point + 20,
                  "active_marker_outside_startup")
    return value


def monitor(owned, client, started, cancelled, directory=c.PUBLIC):
    try:
        while True:
            owned.remember()
            registration = directory / "consumer-registration02.json"
            acknowledgement = directory / "consumer-registered02.json"
            if registration.exists() and not acknowledgement.exists():
                row = c.load(registration)
                c.require(row["schema"] == "team16-cut-consumer02-registration-v1"
                          and row["parent"]["pid"] == client["process"].pid
                          and row["parent"]["start_ticks"] == owned.identities[client["process"].pid]
                          and row["consumer_source_freeze_sha256"] == c.sha(c.CONSUMER / "CONTROL_FREEZE.json"),
                          "consumer_source_parent_registration")
                c.publish(acknowledgement, owned.register_leaf(row, client))
            active = marker(directory, "active-start.json", started)
            finished = marker(directory, "protocol-finished.json", started)
            now = time.monotonic()
            deadline = active["started_at"] + 23 if active else started + 180
            if finished:
                c.require(finished["finished_at"] <= deadline
                          and finished["active_started_at"] == (active["started_at"] if active else None),
                          "late_or_inconsistent_protocol_completion")
                return dict(anchor=finished["finished_at"], cancel=False, normal=True)
            if client["process"].poll() is not None:
                return dict(anchor=now, cancel=False, normal=False, failure="client_exited_without_completion")
            if any(root["process"].poll() is not None for root in owned.roots if root is not client):
                return dict(anchor=now, cancel=True, normal=False, failure="service_exited_during_acquisition")
            if cancelled or now >= deadline:
                return dict(anchor=now, cancel=True, normal=False, failure="signal_or_phase_deadline")
            arm(min(started + 203, deadline))
            time.sleep(0.05)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def output_inventory(directory):
    return [dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size, sha256=c.sha(path))
            for path in sorted(directory.rglob("*")) if path.is_file()]


def persist_close(directory, *, anchor, hard_deadline, io_deadline, joined):
    c.require(joined and time.monotonic() < io_deadline, "owned_join_before_actual_close_IO")
    staging = c.STATE / "consumer-close-returned02.json"
    info = staging.lstat()
    c.require(not staging.is_symlink() and info.st_uid == os.getuid()
              and info.st_mode & 0o777 == 0o600 and info.st_nlink == 1, "private_close_receipt")
    raw = staging.read_bytes()
    path = directory / "consumer-actual-returned-close02.json"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    c.require(time.monotonic() < io_deadline and c.sha(path) == c.sha_bytes(raw), "close_original_IO_deadline")
    value = __import__("json").loads(raw)
    receipt = value["receipt"]
    c.require(value["schema"] == "team16-actual-consumer02-close-return-v1"
              and value["clock_domain"] == c.domain() and value["shared_teardown_anchor"] == anchor
              and value["original_host_cleanup_cutoff"] == hard_deadline
              and receipt["started_at"] == anchor and receipt["deadline"] == min(anchor + 40, hard_deadline)
              and receipt["status"] == "complete" and receipt["all_joined"]
              and anchor <= receipt["finished_at"] <= receipt["cleanup_return_guard_at"]
              <= value["actual_caller_observed_return_at"] < receipt["deadline"],
              "actual_returned_close_not_pre_return_component_file")
    c.require(time.monotonic() < io_deadline and c.sha(path) == c.sha_bytes(raw), "close_original_IO_deadline")
    return dict(receipt=receipt, staged_sha256=c.sha_bytes(raw), persisted_sha256=c.sha(path),
                persisted_at=time.monotonic(), original_IO_deadline=io_deadline)


def all_offered(directory, client_started):
    planned = c.load(c.OFFERS)["rows"]
    path = directory / "client/result.json"
    reported = c.load(path) if path.exists() else None
    raw = []
    for path in sorted((directory / "client/http").glob("*.json"), key=lambda p: int(p.stem)):
        row = c.load(path)
        if row.get("path") == "/v1/chat/completions":
            raw.append(dict(path=str(path.relative_to(directory)), request_id=row.get("request_id"),
                            role=row.get("role"), http_status=row.get("http_status"),
                            error_type=row.get("error_type")))
    rows = (reported["offers"] if reported else
            [dict(row, fate="unobserved" if client_started else "not_sent") for row in planned])
    c.require(len(rows) == 2 and [r["request_id"] for r in rows] == [r["request_id"] for r in planned],
              "unknown_offer_output")
    return dict(planned_managed=2, offers=rows, observed_raw_offer_receipts=raw,
                warmup_send_attempted=(directory / "client/warmup-offered.json").exists(),
                planned_warmup=1, missing_receipt_not_proof_of_no_send=True,
                maximum_requested_completion_tokens=48)


def incomplete_all_offered(directory, *, deadline):
    planned = c.load(c.OFFERS)["rows"]
    c.require(len(planned) == 2, "two_original_planned_offers")
    inventory, errors, truncated = [], [], False
    groups = ("client/http-refusals", "client/http-reservations", "client/http", "client/consumer")
    for group in groups:
        if time.monotonic() >= deadline:
            truncated = True
            break
        folder = directory / group
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if len(inventory) >= 1900 or time.monotonic() >= deadline:
                        truncated = True
                        break
                    if not entry.name.endswith(".json") or (
                            group == "client/consumer" and "refus" not in entry.name):
                        continue
                    row = dict(path=group + "/" + entry.name, content_not_claimed_complete=True)
                    try:
                        row.update(bytes_observed=entry.stat(follow_symlinks=False).st_size,
                                   symlink=entry.is_symlink())
                    except OSError as error:
                        row["stat_error_type"] = type(error).__name__
                    inventory.append(row)
        except OSError as error:
            errors.append(dict(directory=group, error_type=type(error).__name__))
    return dict(schema="team16-all-offered-incomplete-v1", status="INCOMPLETE",
        planned_managed=2, offers=[dict(row, fate="unobserved", fate_known=False) for row in planned],
        planned_warmup=1, warmup_send_attempted=None, actual_observation_count=None,
        observed_final_fates=None, client_result_read=False, inventory_complete=False,
        raw_and_refusal_directory_references=list(groups), inventory_snapshot=inventory,
        inventory_errors=errors, inventory_truncated_or_deadline=truncated,
        empty_inventory_not_zero_observations=True, missing_receipt_not_proof_of_no_send=True,
        maximum_requested_completion_tokens=48, actual_application_at=None,
        scope="Nonjoined failure snapshot;mutable result and receipt bodies are not parsed or called complete.")


def cleanup_owned_stage(owned, client, started, supervision, *, directory, secret):
    signal.setitimer(signal.ITIMER_REAL, 0)
    now = time.monotonic()
    hard_deadline = started + 243
    state = dict(schema="team16-cleanup-successor-attempt-v1", entered_at=now,
        original_hard_deadline=hard_deadline, errors=[], cleanup=None,
        owned_exists=owned is not None, cleanup_attempted=False, cleanup_call_finished=False,
        cleanup_returned=False)
    anchor, cancel = now, True
    try:
        recorded = supervision["anchor"]
        c.require(type(recorded) in (float, int) and math.isfinite(recorded)
                  and started <= recorded <= now, "invalid_recorded_cleanup_anchor")
        state["recorded_anchor"] = recorded
        if supervision.get("anchor_origin") == "initial_unset":
            state["anchor_resolution"] = "exception_fallback_initial_unset_true_current_time"
            state["errors"].append(dict(phase="anchor", reason="initial_anchor_was_not_a_shared_teardown"))
        else:
            anchor = recorded
            state["anchor_resolution"] = "recorded_shared_anchor_preserved"
        cancel = supervision.get("cancel") is not False
    except BaseException as error:
        state["anchor_resolution"] = "exception_fallback_invalid_anchor_true_current_time"
        state["errors"].append(dict(phase="anchor", **c.failure(error, secret)))
    deadline = min(anchor + 40, hard_deadline)
    state.update(anchor=anchor, cleanup_deadline=deadline, deadline_exhausted_at_entry=now >= deadline)
    if now >= deadline:
        state["errors"].append(dict(phase="anchor", reason="recorded_cleanup_budget_exhausted_no_reset"))
    try:
        c.publish(directory / "host-cleanup-anchor02.json", dict(
            anchor=anchor, clock_domain=c.domain(), original_host_cleanup_cutoff=hard_deadline))
    except BaseException as error:
        state["errors"].append(dict(phase="anchor_publication_or_clock", **c.failure(error, secret)))
    # A raising phase alarm must not preempt the explicit bounded owned stop/reap path.
    signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        if owned is not None:
            state["cleanup_attempted"] = True
            state["cleanup"] = owned.cleanup(anchor, client=client,
                cancelled=cancel or bool(state["errors"]), hard_deadline=hard_deadline)
            state["cleanup_returned"] = True
        else:
            state["cleanup"] = dict(all_joined=True, capture_complete=True, surviving_owned_pids=[],
                started_at=anchor, finished_at=time.monotonic(), readers=[], events=[])
    except BaseException as error:
        state["errors"].append(dict(phase="owned_cleanup", **c.failure(error, secret)))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state["cleanup_call_finished"] = True
        state["attempt_finished_at"] = time.monotonic()
    known = {} if owned is None else dict(owned.identities)
    if owned is not None:
        for root in owned.roots:
            known.setdefault(root["process"].pid, None)
    identities = []
    for pid, ticks in known.items():
        row = dict(pid=pid, start_ticks=ticks, state="unknown")
        try:
            current = owned.helper.snapshot(pid)
            if ticks is not None and (current is None or current[0] != ticks):
                row["state"] = "old_identity_gone"
            elif ticks is not None:
                row["state"] = "same_identity_present"
        except BaseException as error:
            row["error_type"] = type(error).__name__
        identities.append(row)
    state["owned_identity_report"] = identities
    state["remaining_owned"] = [row for row in identities if row["state"] != "old_identity_gone"]
    cleanup = state["cleanup"]
    if cleanup and cleanup.get("errors"):
        state["errors"].extend(cleanup["errors"])
    state["actual_joined"] = bool(cleanup and cleanup["all_joined"] and not state["remaining_owned"])
    state["cleanup_in_budget"] = bool(cleanup and now < deadline and cleanup["finished_at"] <= deadline)
    state["joined_in_budget"] = state["actual_joined"] and state["cleanup_in_budget"]
    finished = cleanup["finished_at"] if cleanup else state["attempt_finished_at"]
    state["IO_deadline"] = min(started + 260, finished + 17)
    if not state["joined_in_budget"]:
        try:
            c.require(time.monotonic() < state["IO_deadline"], "incomplete_offered_original_IO_exhausted")
            c.write(directory / "all-offered.json",
                    incomplete_all_offered(directory, deadline=state["IO_deadline"]))
            state["incomplete_all_offered_written"] = True
        except BaseException as error:
            state["errors"].append(dict(phase="incomplete_all_offered", **c.failure(error, secret)))
        try:
            c.require(time.monotonic() < state["IO_deadline"], "failure_marker_original_IO_exhausted")
            c.write(directory / "DO_NOT_EXPORT.json", dict(
                reason="owned_cleanup_incomplete_or_budget_missed", component_pass=False,
                actual_joined=state["actual_joined"], joined_in_budget=False,
                remaining_owned=state["remaining_owned"], actual_application_at=None))
        except BaseException as error:
            state["errors"].append(dict(phase="failure_marker", **c.failure(error, secret)))
    try:
        c.require(time.monotonic() < state["IO_deadline"], "cleanup_receipt_original_IO_exhausted")
        c.write(directory / "cleanup-attempt02.json", state)
    except BaseException as error:
        state["errors"].append(dict(phase="cleanup_receipt", **c.failure(error, secret)))
    try:
        print(__import__("json").dumps(dict(
            schema=state["schema"], cleanup_attempted=state["cleanup_attempted"],
            cleanup_call_finished=state["cleanup_call_finished"], cleanup_returned=state["cleanup_returned"],
            remaining_owned=state["remaining_owned"], joined_in_budget=state["joined_in_budget"],
            errors=state["errors"])), flush=True)
    except BaseException as error:
        state["errors"].append(dict(phase="cleanup_stdout", error_type=type(error).__name__))
    return state


def release_after_cleanup(lock, state):
    c.require(state["cleanup_call_finished"]
              and (not state["owned_exists"] or state["cleanup_attempted"])
              and "remaining_owned" in state, "owned_cleanup_attempt_and_identity_report_before_unlock")
    signal.setitimer(signal.ITIMER_REAL, 0)
    fcntl.flock(lock, fcntl.LOCK_UN)
    os.close(lock)


def publish_outputs(directory, result, *, joined, secret, deadline, no_key_created=False):
    c.require(time.monotonic() < deadline, "output_deadline_exhausted")
    result["component_pass_is_provisional_until_export_gate"] = True
    c.write(directory / "supervision.json", result)
    safe = joined and c.privacy_scan((directory,), secret, no_key_created=no_key_created)
    if safe:
        c.write(directory / "result-index.json", dict(
            schema="team16-cut-consumer-deployment02-raw-index-v1", files=output_inventory(directory),
            output=str(directory), component_pass=result["component_pass"],
            all_offered="all-offered.json", raw_http="client/http/", ownership="supervision.json",
            prior_owner_component_not_reexecuted=True, actual_application_at=None, native_mpc_permitted=False,
            excluded_final_files=["result-index.json", "export-gate.json", "DO_NOT_EXPORT.json",
                                  "safe-export-before-config-close02.json","private-config-close02.json"]))
        safe = c.privacy_scan((directory,), secret, no_key_created=no_key_created)
    removed = False
    if safe:
        removed = close_private_config(directory, joined=joined, secret=secret, deadline=deadline,
                                       no_key_created=no_key_created)
        safe = c.privacy_scan((directory,), secret, no_key_created=no_key_created)
    c.require(time.monotonic() < deadline, "output_absolute_deadline")
    c.write(directory / "export-gate.json", dict(
        safe_to_export=bool(safe), all_helpers_joined=joined,
        private_config_removed_after_join_and_safe_scan=removed,
        component_pass=bool(result["component_pass"] and safe and removed),
        completed_at=time.monotonic(), absolute_deadline=deadline,
        privacy_scope="Raw bytes plus decoded response bodies; malformed/unknown JSON refuses export",
        actual_application_at=None, native_mpc_permitted=False))
    return bool(result["component_pass"] and safe and removed)


def close_private_config(directory, *, joined, secret, deadline, no_key_created=False):
    c.require(joined and time.monotonic() < deadline, "join_before_private_config_close")
    c.require(c.privacy_scan((directory,), secret, no_key_created=no_key_created),
              "safe_export_before_private_config_close")
    c.write(directory / "safe-export-before-config-close02.json", dict(
        all_helpers_joined=True, safe_to_export=True, checked_at=time.monotonic(),
        deadline=deadline, scope="Closed output set including exact actual-returned-close receipt."))
    existed = c.CONFIG.exists()
    if existed:
        config = c.private_config(c.CONFIG)
        c.require(config["ADMIN_API_KEY"] == secret, "owned_private_key_identity")
        c.CONFIG.unlink()
    else:
        c.require(no_key_created, "created_config_missing_before_safe_close")
    removed = not c.CONFIG.exists()
    c.write(directory / "private-config-close02.json", dict(
        private_config_existed=existed, removed=removed, all_helpers_joined=joined,
        safe_export_before_removal=True, finished_at=time.monotonic(), deadline=deadline))
    c.require(removed and time.monotonic() < deadline, "private_config_close_original_IO17")
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute-on-existing-gpu", action="store_true")
    parser.add_argument("--mayor-preregistration", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic() if args.execute_on_existing_gpu else None
    activation = require_activation(args.execute_on_existing_gpu, args.mayor_preregistration, role="launch")
    if activation is None:
        print('{"status":"disabled","native_execution":false,"actual_application_at":null}')
        return 0
    c.remote_only()
    prereg = activation["document"]
    c.verify_code()
    c.require(not any(path.exists() or path.is_symlink() for path in (c.STATE, c.PUBLIC)),
              "exclusive_one_attempt_spent_namespace")
    lock = acquire_lock()
    c.STATE.mkdir(mode=0o700)
    c.PUBLIC.mkdir(mode=0o700)
    for name in ("t", "python-cache-evidence", "native-artifacts", "origins-consumer02"):
        (c.STATE / name).mkdir(mode=0o700)
    sys.pycache_prefix = str(c.STATE / "python-cache-evidence")
    signal.signal(signal.SIGALRM, alarm)
    arm(started + 180)
    cancelled = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: cancelled.append(number))
    result = dict(schema="team16-owned-cut-consumer02-supervision-v1", started_at=started,
                  clock_domain=c.domain(), preregistration=prereg, errors=[],
                  component_pass=False, actual_application_at=None, native_mpc_permitted=False,
                  prediction_qualified=False, exact_point_gate=False, copy_ready_qualification=False)
    result["private_config_creation_attempted"] = False
    owned, client, secret = None, None, ""
    supervision = dict(anchor=started, cancel=True, normal=False, anchor_origin="initial_unset")
    try:
        result["fresh_host_receipt"] = fresh_host(prereg)
        environment = c.clock_helpers().native_environment(c.STATE, c.STATE / "python-cache-evidence")
        environment.update(MPC_OWNER_LEDGER_ENABLED="true",
                           TELEMETRY_JSONL=str(c.PUBLIC / "gateway/gateway-telemetry.jsonl"),
                           TELEMETRY_STATUS_JSON=str(c.PUBLIC / "gateway/gateway-telemetry-status.json"),
                           CUDA_VISIBLE_DEVICES="GPU-8610519d-3680-ae5f-cdfc-40037bfb5013",
                           VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1")
        os.environ["TMPDIR"] = environment["TMPDIR"]
        prelaunch = c.module("_owner_adapter_TMPDIR_prelaunch",
                            c.PREFLIGHT / "inputs/control02/prelaunch01.py")
        receipt = prelaunch.make_receipt(c.STATE / "t", c.STATE)
        c.write(c.PUBLIC / "tmpdir-prelaunch.json", receipt)
        result["source_before"] = c.source_snapshot()
        result["import_prerequisites"] = c.import_prerequisites()
        result["gpu_before"] = c.clock_helpers().gpu_inventory()
        c.require_idle(result["gpu_before"])
        c.require(c.domain().split(":")[1] == result["fresh_host_receipt"]["boot_id"],
                  "host_boot_changed_from_reviewed_receipt")
        c.require(c.ALIAS.is_symlink() and c.ALIAS.resolve() == c.MODEL, "immutable_existing_model_alias")
        result["private_config_creation_attempted"] = True
        config = c.create_config()
        secret = config["ADMIN_API_KEY"]
        owned = Owned(secret)
        for service in ("native", "gateway"):
            c.require(not cancelled and time.monotonic() < started + 180, "cancelled_or_late_before_service")
            owned.spawn([str(c.PYTHON), "-I", "-B", str(c.HERE / "service01.py"),
                         "--owned-service", "--service", service, "--owner-pid", str(os.getpid()),
                         "--mayor-preregistration", activation["approval_path"],
                         "--approval-sha256", activation["approval_sha256"],
                         "--owner-start-ticks", str(c.owner().snapshot(os.getpid())[0])],
                         directory=c.PUBLIC / service, cwd=c.HERE, env=environment)
        c.write(c.STATE / "client-consumer-invocation02.json", dict(
                host_started_at=started, startup_deadline=started + 180, outer_deadline=started + 260,
                cleanup_hard_deadline=started + 243, clock_domain=c.domain(),
                consumer_manifest_sha256=c.CONSUMER_SEAL,
                approval_sha256=activation["approval_sha256"],
                owner_pid=os.getpid(), owner_start_ticks=c.owner().snapshot(os.getpid())[0]))
        c.require(not cancelled and time.monotonic() < started + 180, "cancelled_or_late_before_client")
        client = owned.spawn([str(c.PYTHON), "-I", "-B", str(c.HERE / "client01.py"), "--owned-client",
                              "--mayor-preregistration", activation["approval_path"],
                              "--approval-sha256", activation["approval_sha256"]],
                             directory=c.PUBLIC / "client-process", cwd=c.PREFLIGHT, env=environment)
        supervision = monitor(owned, client, started, cancelled)
        if "failure" in supervision:
            result["errors"].append(supervision["failure"])
    except BaseException as error:
        result["errors"].append(type(error).__name__)
        supervision = dict(anchor=time.monotonic(), cancel=True, normal=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_state = cleanup_owned_stage(
            owned, client, started, supervision, directory=c.PUBLIC, secret=secret)
        try:
            anchor, cleanup = cleanup_state["anchor"], cleanup_state["cleanup"]
            joined = cleanup_state["joined_in_budget"]
            result.update(cleanup=cleanup, cleanup_attempt=cleanup_state)
            result["errors"].extend(cleanup_state["errors"])
            result["private_config_close_pending_safe_export"] = c.CONFIG.exists()
            result["client_returncode"] = client["process"].returncode if client else None
            io_deadline = cleanup_state["IO_deadline"]
            try:
                arm(io_deadline)
                c.require(joined, "DO_NOT_EXPORT_owned_helpers_remain_or_cleanup_budget_missed")
                c.write(c.PUBLIC / "all-offered.json", all_offered(c.PUBLIC, client is not None))
                result["source_after"] = c.source_snapshot()
                result["sources_unchanged"] = result.get("source_before") == result["source_after"]
                result["gpu_after"] = c.clock_helpers().gpu_inventory()
                c.require_idle(result["gpu_after"])
                c.verify_code()
                result["candidate_unchanged"] = True
                c.write(c.PUBLIC / "origin-events-consumer02.json", c.load_origins(c.ORIGINS))
                result["privacy_pass"] = c.privacy_scan((c.PUBLIC,), secret) if secret else owned is None
                c.require(result["privacy_pass"], "DO_NOT_EXPORT_CREDENTIAL_REFUSAL")
                if client:
                    returned = persist_close(c.PUBLIC, anchor=anchor, hard_deadline=started + 243,
                                             io_deadline=io_deadline, joined=joined)
                    result["actual_consumer_close"] = dict(
                        staged_sha256=returned["staged_sha256"], persisted_sha256=returned["persisted_sha256"],
                        persisted_at=returned["persisted_at"], original_IO_deadline=io_deadline)
                    report = c.load(c.PUBLIC / "client/result.json")
                    close = c.load(c.PUBLIC / "client/client-cleanup.json")
                    result["component_pass"] = bool(
                        report["component_pass"] and close["joined"] and result["client_returncode"] == 0
                        and returned["receipt"]["component_pass"] is True
                        and (c.PUBLIC / "consumer-registered02.json").exists()
                        and supervision["normal"] and cleanup["capture_complete"]
                        and result["sources_unchanged"] and not result["errors"] and not cancelled)
            except BaseException as error:
                result["errors"].append("postflight:" + type(error).__name__)
            result.update(finished_at=time.monotonic(), signals=cancelled)
            result["phase_bounds"] = bool(
                cleanup and cleanup["finished_at"] <= min(anchor + 40, started + 243)
                and result["finished_at"] <= started + 260
                and result["finished_at"] <= cleanup["finished_at"] + 17)
            result["component_pass"] = result["component_pass"] and result["phase_bounds"] and not result["errors"]
            try:
                result["component_pass"] = publish_outputs(
                    c.PUBLIC, result, joined=joined, secret=secret, deadline=io_deadline,
                    no_key_created=not result["private_config_creation_attempted"] and owned is None)
            except BaseException as error:
                result["component_pass"] = False
                if not (c.PUBLIC / "DO_NOT_EXPORT.json").exists():
                    c.write(c.PUBLIC / "DO_NOT_EXPORT.json",
                            dict(reason="final_output_refusal", error_type=type(error).__name__))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            release_after_cleanup(lock, cleanup_state)
    print(__import__("json").dumps(dict(output=str(c.PUBLIC), component_pass=result["component_pass"],
                                      owned_joined=joined, actual_application_at=None)))
    return 0 if result["component_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
