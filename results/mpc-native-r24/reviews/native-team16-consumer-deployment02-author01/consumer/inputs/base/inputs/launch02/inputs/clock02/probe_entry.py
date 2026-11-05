"""One exclusive Mayor launch; Source13 PID/start-tick/pidfd cleanup, no old entry rerun."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_common import (ENTRY_SHA, EXECUTION_SEAL, EXPERIMENT, GPU_UUID, HERE, NATIVE_PYTHON,
                          PUBLIC_RESULT, SOURCE13, SOURCE_MANIFEST, SOURCE_ROOT,
                          load, load_module, require, sha, verify_candidate, write_new)


def gpu_inventory():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name",
         "--format=csv,noheader"], text=True, timeout=10).strip()
    return dict(gpu=gpu, compute_apps=apps)


def require_idle(inventory):
    row = [value.strip() for value in inventory["gpu"].split(",")]
    require(len(inventory["gpu"].splitlines()) == 1 and len(row) == 4
            and row[0] == GPU_UUID and "4090" in row[1] and int(row[2]) <= 1
            and not inventory["compute_apps"], "sole_existing_idle_GPU_required")
    for port in (8000, 8080):
        with socket.socket() as connection:
            connection.settimeout(1)
            require(connection.connect_ex(("127.0.0.1", port)) != 0, "preexisting_native_port")


def credentials_absent(directory, secret):
    for path in Path(directory).rglob("*"):
        if path.is_file():
            require(not path.name.endswith(".private.json"), "private_material_in_public_evidence")
            if secret and secret in path.read_bytes():
                return False
    return True


def native_environment(state, cache):
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH",
               "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES")
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    artifacts = state / "native-artifacts"
    env.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
        PYTHONPYCACHEPREFIX=str(cache), TMPDIR=str(state / "t"),
        TORCHINDUCTOR_CACHE_DIR=str(artifacts / "torchinductor"),
        TRITON_CACHE_DIR=str(artifacts / "triton"), CUDA_CACHE_PATH=str(artifacts / "cuda"),
    )
    return env


def uuid_ipc_preflight(tmpdir):
    directory = Path(tmpdir)
    require(directory.is_absolute() and "\0" not in str(directory), "absolute_private_TMPDIR_required")
    # The sealed SDK0.30.0 failure records TMPDIR/<hyphenated UUID>, not UUID.hex.
    example = directory / str(uuid.UUID(int=0))
    count = len(os.fsencode(example))
    require(count <= 107, "UUID36_IPC_pathname_exceeds_107_bytes:" + str(count))
    return dict(schema="team16-native-copy-clock-ipc-preflight-v1", tmpdir=str(directory),
                pathname_bytes=count, max_pathname_bytes=107, uuid_basename_bytes=36,
                evidence="CPU_PATHNAME_ARITHMETIC_NOT_NATIVE_CAPTURE",
                ipc_uri_prefix_excluded=True, example_not_created=True)


def source_binding(cache):
    require(sha(SOURCE13 / "source-closure.json") == SOURCE_MANIFEST,
            "source13_manifest_changed")
    require(sha(SOURCE13 / "research_source_guard.py") == sha(
        HERE / "inputs/source13/research_source_guard.py"), "source13_guard_changed")
    guard = load_module("_supervised_source13_source_guard", SOURCE13 / "research_source_guard.py")
    previous = sys.pycache_prefix
    try:
        sys.pycache_prefix = str(cache)
        return guard.verify(SOURCE13, (SOURCE13 / "source-closure.json").read_bytes(),
                            EXECUTION_SEAL)
    finally:
        sys.pycache_prefix = previous


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute-on-existing-gpu", action="store_true", required=True)
    parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize
            and HERE.parent == SOURCE_ROOT.parent and HERE.name == EXPERIMENT
            and Path.cwd().resolve() == HERE,
            "Mayor_only_exact_remote_namespace_isolated_entry")
    require(Path(sys.prefix).resolve() == NATIVE_PYTHON.parent.parent.resolve(),
            "existing_native_interpreter_required")
    prereg = verify_candidate()
    from probe_replay import raw_index, replay
    plan = load(HERE / "native-plan.json")
    state = HERE / "owned-run"
    require(not state.exists() and not PUBLIC_RESULT.exists(), "exclusive_one_attempt_no_retry")
    state.mkdir(mode=0o700)
    PUBLIC_RESULT.mkdir(mode=0o700)
    cache = state / "python-cache-evidence"
    cache.mkdir(mode=0o700)
    artifact_cache = state / "native-artifacts"
    artifact_cache.mkdir(mode=0o700)
    (state / "t").mkdir(mode=0o700)
    entry_path = HERE / "inputs/reference/source13-entry02.py"
    require(sha(entry_path) == ENTRY_SHA, "unchanged_source13_cleanup_helpers")
    owner = None
    cancelled, errors, owned, survivors = [], [], {}, []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: cancelled.append(signum))
    result = dict(schema="team16-native-copy-clock-supervision-v1",
                  prereg_sha256=sha(HERE / "prereg.json"),
                  child_started=False, clean_owned_shutdown=False, returncode=None,
                  native_mpc_permitted=False, deployable=False)
    process, root_fd, timed_out = None, None, False
    try:
        env = native_environment(state, cache)
        result["ipc_path_preflight"] = uuid_ipc_preflight(env["TMPDIR"])
        owner = load_module("_source13_owned_PID_helpers_only", entry_path)
        owner.enable_subreaper()
        result["source_before"] = source_binding(cache)
        result["gpu_before"] = gpu_inventory()
        require_idle(result["gpu_before"])
        command = [str(NATIVE_PYTHON), "-I", "-B", "-X", "pycache_prefix=" + str(cache),
                   str(HERE / "probe_native.py"), "--owned"]
        result["command"] = command
        with (PUBLIC_RESULT / "native-child.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=HERE, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            result["child_started"] = True
            root_fd = os.pidfd_open(process.pid)
            owner.remember(process.pid, owned, parent=os.getpid())
            deadline = time.monotonic() + prereg["bounds"]["native_child_deadline_s"]
            result["started_at"], result["deadline_at"] = time.monotonic(), deadline
            while process.poll() is None and time.monotonic() < deadline and not cancelled:
                owner.remember_adopted(owned)
                time.sleep(0.2)
            timed_out = process.poll() is None and not cancelled
    except BaseException as error:
        errors.append(type(error).__name__)
    finally:
        if process is not None:
            try:
                survivors = owner.terminate_owned(owned, process, grace=30)
            except BaseException as error:
                errors.append("owned_cleanup:" + type(error).__name__)
                if root_fd is not None:
                    try:
                        signal.pidfd_send_signal(root_fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                survivors = [pid for pid in owned if owner.alive(pid, owned)]
            try:
                result["returncode"] = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                errors.append("root_join_timeout")
            if root_fd is not None:
                os.close(root_fd)
        try:
            result["source_after"] = source_binding(cache)
            result["sources_unchanged"] = result.get("source_before") == result["source_after"]
            require(result["sources_unchanged"], "supervised_source_before_after_mismatch")
        except BaseException as error:
            errors.append("source_postflight:" + type(error).__name__)
        try:
            result["gpu_after"] = gpu_inventory()
            require(not result["gpu_after"]["compute_apps"], "post_probe_compute_processes_remain")
            for port in (8000, 8080):
                with socket.socket() as connection:
                    connection.settimeout(1)
                    require(connection.connect_ex(("127.0.0.1", port)) != 0,
                            "post_probe_owned_port_still_bound")
        except BaseException as error:
            errors.append("hardware_postflight:" + type(error).__name__)
        try:
            private_config = state / "control.private.json"
            secret = load(private_config)["ADMIN_API_KEY"].encode() if private_config.exists() else b""
            result["public_evidence_credentials_absent"] = credentials_absent(PUBLIC_RESULT, secret)
            require(result["public_evidence_credentials_absent"], "DO_NOT_EXPORT_CREDENTIAL_REFUSAL")
            if not survivors and (process is None or process.returncode is not None) and private_config.exists():
                private_config.unlink()
            result["private_config_removed_after_owned_join"] = not private_config.exists()
        except BaseException as error:
            errors.append("privacy_postflight:" + type(error).__name__)
        try:
            require(not any(cache.iterdir()), "private_python_cache_not_empty")
            verify_candidate()
            result["candidate_sources_unchanged"] = True
        except BaseException as error:
            result["candidate_sources_unchanged"] = False
            errors.append("candidate_postflight:" + type(error).__name__)
        result.update(
            timed_out=timed_out, interrupted_signals=cancelled, errors=errors,
            surviving_owned_pids=survivors, child_subreaper=True,
            owned_pid_start_ticks={str(pid): tick for pid, tick in owned.items()},
            finished_at=time.monotonic(),
        )
        result["clean_owned_shutdown"] = (process is not None and process.returncode is not None
            and not survivors and not errors and not cancelled and not timed_out)
        write_new(PUBLIC_RESULT / "supervision.json", result)
    if result.get("public_evidence_credentials_absent") is True:
        if result.get("candidate_sources_unchanged") is True:
            try:
                report = replay(PUBLIC_RESULT, plan)
            except BaseException as error:
                report = dict(schema="team16-native-copy-clock-replay-refused-v1", valid=False,
                              refusal_type=type(error).__name__, raw_failures_retained=True)
        else:
            report = dict(schema="team16-native-copy-clock-replay-refused-v1", valid=False,
                          refusal_type="candidate_sources_changed", raw_failures_retained=True)
        write_new(PUBLIC_RESULT / "replay.json", report)
        write_new(PUBLIC_RESULT / "raw-index.json", raw_index(PUBLIC_RESULT))
        accepted = report["valid"] and result["returncode"] == 0
    else:
        accepted = False
    print(json.dumps(dict(schema="team16-native-copy-clock-launch-receipt-v1",
                          raw_result=str(PUBLIC_RESULT), component_capture_valid=accepted,
                          sources_and_law_qualified=False, owned_survivors=survivors)))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
