#!/usr/bin/env python3
"""Default-off, one-stack Mayor-owned fixed32 acquisition launch."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition_common01 as c
from owned01 import Owned


def alarm(signum, frame):
    raise TimeoutError("owned_absolute_phase_deadline")


def arm(deadline):
    c.require(deadline > time.monotonic(), "phase_deadline_already_exhausted")
    signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())


def authority(path):
    value = c.load(path)
    c.require(set(value) == {"schema", "execute_once", "composition_manifest_sha256", "protocol_sha256",
                            "gpu_uuid", "source_root", "existing_Mayor_exclusive_GPU_lock_held",
                            "preregistration_commit", "consumer_manifest_sha256",
                            "consumer_Mayor_engineering_acceptance", "consumer_independent_review_acceptance",
                            "composition_Mayor_acceptance", "fresh_host_receipt", "fresh_host_receipt_sha256"}
              and value.get("schema") == "mayor-cut-consumer-composition-preregistration-v1"
              and value.get("execute_once") is True
              and value.get("composition_manifest_sha256") == c.sha(c.HERE / "SHA256SUMS")
              and value.get("protocol_sha256") == c.sha(c.HERE / "PROTOCOL.json")
              and value.get("consumer_manifest_sha256") == c.CONSUMER_SEAL
              and value.get("gpu_uuid") == "GPU-8610519d-3680-ae5f-cdfc-40037bfb5013"
              and value.get("source_root") == str(c.ROOT)
              and value.get("existing_Mayor_exclusive_GPU_lock_held") is True
              and re.fullmatch("[a-f0-9]{40}", value.get("preregistration_commit", "")),
              "new_Mayor_preregistration_required_not_development_approval")
    for key in ("consumer_Mayor_engineering_acceptance", "consumer_independent_review_acceptance",
                "composition_Mayor_acceptance"):
        proof = value[key]
        c.require(set(proof) == {"path", "sha256", "accepted"}
                  and proof["accepted"] is True and re.fullmatch("[a-f0-9]{64}", proof["sha256"])
                  and c.sha(Path(proof["path"])) == proof["sha256"],
                  "exact_separate_acceptance_required:" + key)
    c.require(c.sha(Path(value["fresh_host_receipt"])) == value["fresh_host_receipt_sha256"],
              "fresh_readonly_host_receipt_binding")
    return value


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
    while True:
        owned.remember()
        registration = directory / "consumer-registration01.json"
        acknowledgement = directory / "consumer-registered01.json"
        if registration.exists() and not acknowledgement.exists():
            row = c.load(registration)
            c.require(row["schema"] == "team16-cut-consumer-registration-v1"
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


def output_inventory(directory):
    return [dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size, sha256=c.sha(path))
            for path in sorted(directory.rglob("*")) if path.is_file()]


def persist_close(directory, *, anchor, hard_deadline, io_deadline, joined):
    c.require(joined and time.monotonic() < io_deadline, "owned_join_before_actual_close_IO")
    staging = c.STATE / "consumer-close-returned01.json"
    info = staging.lstat()
    c.require(not staging.is_symlink() and info.st_uid == os.getuid()
              and info.st_mode & 0o777 == 0o600 and info.st_nlink == 1, "private_close_receipt")
    raw = staging.read_bytes()
    path = directory / "consumer-actual-returned-close01.json"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    c.require(time.monotonic() < io_deadline and c.sha(path) == c.sha_bytes(raw), "close_original_IO_deadline")
    value = __import__("json").loads(raw)
    receipt = value["receipt"]
    c.require(value["schema"] == "team16-actual-consumer-close-return-v1"
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


def publish_outputs(directory, result, *, joined, secret, deadline, no_key_created=False):
    c.require(time.monotonic() < deadline, "output_deadline_exhausted")
    result["component_pass_is_provisional_until_export_gate"] = True
    c.write(directory / "supervision.json", result)
    safe = joined and c.privacy_scan((directory,), secret, no_key_created=no_key_created)
    if safe:
        c.write(directory / "result-index.json", dict(
            schema="team16-cut-consumer-composition-raw-index-v1", files=output_inventory(directory),
            output=str(directory), component_pass=result["component_pass"],
            all_offered="all-offered.json", raw_http="client/http/", ownership="supervision.json",
            prior_owner_component_not_reexecuted=True, actual_application_at=None, native_mpc_permitted=False,
            excluded_final_files=["result-index.json", "export-gate.json", "DO_NOT_EXPORT.json"]))
        safe = c.privacy_scan((directory,), secret, no_key_created=no_key_created)
    c.require(time.monotonic() < deadline, "output_absolute_deadline")
    c.write(directory / "export-gate.json", dict(
        safe_to_export=bool(safe), all_helpers_joined=joined,
        component_pass=bool(result["component_pass"] and safe),
        completed_at=time.monotonic(), absolute_deadline=deadline,
        privacy_scope="Raw bytes plus decoded response bodies; malformed/unknown JSON refuses export",
        actual_application_at=None, native_mpc_permitted=False))
    return bool(result["component_pass"] and safe)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute-on-existing-gpu", action="store_true")
    parser.add_argument("--mayor-preregistration", type=Path)
    args = parser.parse_args(argv)
    if not args.execute_on_existing_gpu:
        print('{"status":"disabled","native_execution":false,"actual_application_at":null}')
        return 0
    raise ValueError("successor_CPU_patch_no_live_entrypoint")
    started = time.monotonic()
    c.remote_only()
    c.require(args.mayor_preregistration is not None, "Mayor_preregistration_required")
    prereg = authority(args.mayor_preregistration)
    c.verify_code()
    c.require(not any(path.exists() or path.is_symlink() for path in (c.STATE, c.PUBLIC)),
              "exclusive_one_attempt_spent_namespace")
    lock = acquire_lock()
    c.STATE.mkdir(mode=0o700)
    c.PUBLIC.mkdir(mode=0o700)
    for name in ("t", "python-cache-evidence", "native-artifacts", "origins-consumer01"):
        (c.STATE / name).mkdir(mode=0o700)
    sys.pycache_prefix = str(c.STATE / "python-cache-evidence")
    signal.signal(signal.SIGALRM, alarm)
    arm(started + 180)
    cancelled = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: cancelled.append(number))
    result = dict(schema="team16-owned-cut-consumer-supervision-v1", started_at=started,
                  clock_domain=c.domain(), preregistration=prereg, errors=[],
                  component_pass=False, actual_application_at=None, native_mpc_permitted=False,
                  prediction_qualified=False, exact_point_gate=False, copy_ready_qualification=False)
    result["private_config_creation_attempted"] = False
    owned, client, secret = None, None, ""
    supervision = dict(anchor=started, cancel=True, normal=False)
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
                         "--service", service, "--owner-pid", str(os.getpid()),
                         "--owner-start-ticks", str(c.owner().snapshot(os.getpid())[0])],
                         directory=c.PUBLIC / service, cwd=c.HERE, env=environment)
        c.write(c.STATE / "client-consumer-invocation01.json", dict(
                host_started_at=started, startup_deadline=started + 180, outer_deadline=started + 260,
                cleanup_hard_deadline=started + 243, clock_domain=c.domain(),
                consumer_manifest_sha256=c.CONSUMER_SEAL,
                owner_pid=os.getpid(), owner_start_ticks=c.owner().snapshot(os.getpid())[0]))
        c.require(not cancelled and time.monotonic() < started + 180, "cancelled_or_late_before_client")
        client = owned.spawn([str(c.PYTHON), "-I", "-B", str(c.HERE / "client01.py"), "--owned-client"],
                             directory=c.PUBLIC / "client-process", cwd=c.PREFLIGHT, env=environment)
        supervision = monitor(owned, client, started, cancelled)
        if "failure" in supervision:
            result["errors"].append(supervision["failure"])
    except BaseException as error:
        result["errors"].append(type(error).__name__)
        supervision = dict(anchor=time.monotonic(), cancel=True, normal=False)
    finally:
        anchor = supervision["anchor"]
        cleanup = None
        try:
            c.publish(c.PUBLIC / "host-cleanup-anchor01.json", dict(
                anchor=anchor, clock_domain=c.domain(), original_host_cleanup_cutoff=started + 243))
            arm(min(anchor + 40, started + 243))
            if owned:
                cleanup = owned.cleanup(anchor, client=client, cancelled=supervision["cancel"],
                                        hard_deadline=started + 243)
            else:
                cleanup = dict(all_joined=True, capture_complete=True, surviving_owned_pids=[],
                               started_at=anchor, finished_at=time.monotonic(), readers=[], events=[])
        except BaseException as error:
            result["errors"].append("cleanup:" + type(error).__name__)
        result["cleanup"] = cleanup
        joined = bool(cleanup and cleanup["all_joined"])
        if joined and c.CONFIG.exists():
            try:
                c.CONFIG.unlink()
            except OSError as error:
                result["errors"].append("private_cleanup:" + type(error).__name__)
        result["private_config_removed_after_join"] = not c.CONFIG.exists()
        result["client_returncode"] = client["process"].returncode if client else None
        io_deadline = min(started + 260, (cleanup["finished_at"] if cleanup else time.monotonic()) + 17)
        try:
            arm(io_deadline)
            c.require(joined, "DO_NOT_EXPORT_owned_helpers_remain")
            c.write(c.PUBLIC / "all-offered.json", all_offered(c.PUBLIC, client is not None))
            result["source_after"] = c.source_snapshot()
            result["sources_unchanged"] = result.get("source_before") == result["source_after"]
            result["gpu_after"] = c.clock_helpers().gpu_inventory()
            c.require_idle(result["gpu_after"])
            c.verify_code()
            result["candidate_unchanged"] = True
            c.write(c.PUBLIC / "origin-events-consumer01.json", c.load_origins(c.ORIGINS))
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
                    and (c.PUBLIC / "consumer-registered01.json").exists()
                    and supervision["normal"] and cleanup["capture_complete"]
                    and result["sources_unchanged"] and not result["errors"] and not cancelled
                    and result["private_config_removed_after_join"])
        except BaseException as error:
            result["errors"].append("postflight:" + type(error).__name__)
        result.update(finished_at=time.monotonic(), signals=cancelled)
        result["phase_bounds"] = bool(
            cleanup and cleanup["finished_at"] <= anchor + 40
            and result["finished_at"] <= started + 260
            and result["finished_at"] <= cleanup["finished_at"] + 17)
        result["component_pass"] = result["component_pass"] and result["phase_bounds"] and not result["errors"]
        try:
            result["component_pass"] = publish_outputs(
                c.PUBLIC, result, joined=joined, secret=secret, deadline=io_deadline,
                no_key_created=not result["private_config_creation_attempted"] and owned is None)
        except BaseException as error:
            result["component_pass"] = False
            c.write(c.PUBLIC / "DO_NOT_EXPORT.json",
                    dict(reason="final_output_refusal", error_type=type(error).__name__))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)
    print(__import__("json").dumps(dict(output=str(c.PUBLIC), component_pass=result["component_pass"],
                                      owned_joined=joined, actual_application_at=None)))
    return 0 if result["component_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
