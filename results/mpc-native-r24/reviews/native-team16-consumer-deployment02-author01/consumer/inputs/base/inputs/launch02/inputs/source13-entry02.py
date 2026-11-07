"""One bounded protected Source13 recorded-control help/capture;named tmux,owned-PID cleanup."""

import hashlib
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path("/workspace/pr32-r23/source")
SOURCE = ROOT / "experiments/mpc-research/native-r24-prepared-physical13"
UPLOAD = ROOT.parent / "r24-native-prepared-physical-upload13"
PYTHON = "/workspace/pr32-r23/venv/bin/python"
MANIFEST = "94fa09623feed131fb24c4c3054483f56387e64d48a59f9de556f4bbefab1de8"
EXECUTION = "961b8520efd2656b6494895030fa43b0f873b664"
PROTOCOLS = {'brackets': '26a6cfdaba4559ab2add7d895a63758fbad457d421adeebf5291760b3b09d626', 'off': '242f2f01d0a2a3386b29f396a79e4b4aa12160554d99c24a3829bffdafec53fa', 'query': '841019abd4e5373402a9519b599e2d2083cf2f83f3bccac88ba9973bbcec2e83'}
MODES = ('off', 'brackets', 'query')
RECORDED = {'collect_azure.py': '4b92fa2c68d8f6f4586538469aacc3d6d82ef5db817b19f8a690f5e0d2866784', 'collect_azure_recorded.py': 'c2775b43f3d2f9f2d937d661cd331d5e51320d8b5a896944d13b109db2159653', 'native_physics_analysis.py': '4ca647d32a6d118a6b409365a4c1fc3a1b6402493af649a1fe80296286829553', 'prepared_copy_ready_analysis.py': '59a39aa67c5f1bde7813ffe08d8085ec6b08ad4572f9bc2af98b9331f20545da', 'recorded-control-protocol01.json': '533eabc12598c2ea4f54a5c6e1f71069b4db348d84092dde6b1b42257af236f6', 'recorded_control_analysis.py': '9c1c4d5936afc5562544d2c997ccd5e2e4cbe7ad6a66fb517a4bc5fb375c22d7', 'recording_control.py': 'a31acf65a4f3876aa800da913e5a8cd7decb0b15df94cad2aa2414b6b3cf857e', 'research_source_guard.py': '5269b620c09b8714ae8d34c4973ebce9b2873a015821f9f062b52d6f3091bd88', 'run_azure.py': '110acdf0744cb7292e013f9ff7dfee7e4d6c1ac6f210089f1a8915b64e4ec1df', 'run_azure_recorded.py': '91aa250087df650f7a4ed9ca95275e136fe9e1dc155055227bf354567dff5fcf', 'run_native_shadow.py': '862aed708cf74c01af4b81d955b3ec6a864f226fb6bcf5e1abf3bdd77afa39b9'}
ROUTE = "run_native_shadow.py->run_azure_recorded.py->collect_azure_recorded.py"
TMUX_SESSION = "pr32-prepared-physical13"
SUBREAPER_ENABLED = False


def snapshot(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), fields[0], int(fields[1])
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity(pid):
    current = snapshot(pid)
    return None if current is None else current[:2]


def remember(pid, owned, parent=None):
    pending = [(pid, parent)]
    seen = set()
    while pending:
        pid, parent = pending.pop()
        current = snapshot(pid)
        if current is None or (pid, current[0]) in seen:
            continue
        if parent is not None:
            ancestor = snapshot(parent)
            if (current[2] != parent or ancestor is None
                    or parent != os.getpid() and ancestor[0] != owned.get(parent)):
                continue
        elif pid in owned and owned[pid] != current[0]:
            continue
        seen.add((pid, current[0]))
        owned[pid] = current[0]
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        pending.extend((int(child), pid) for child in reversed(children))


def alive(pid, owned):
    current = identity(pid)
    return current is not None and current[0] == owned[pid] and current[1] not in ("Z", "X")


def enable_subreaper():
    global SUBREAPER_ENABLED
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prepared13_entry_child_subreaper_required")
    enabled = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise ValueError("prepared13_entry_child_subreaper_not_enabled")
    SUBREAPER_ENABLED = True


def remember_adopted(owned):
    if not SUBREAPER_ENABLED:
        raise ValueError("prepared13_entry_child_subreaper_required")
    pid = os.getpid()
    for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
        remember(int(child), owned, parent=pid)


def terminate_owned(owned, process, grace=90):
    remember_adopted(owned)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 30)):
        deadline = time.monotonic() + wait
        sent = set()
        while time.monotonic() < deadline:
            remember_adopted(owned)
            for pid in reversed(list(owned)):
                key = (pid, owned[pid])
                if key in sent or not alive(pid, owned):
                    continue
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    if alive(pid, owned):
                        signal.pidfd_send_signal(fd, sig)
                        sent.add(key)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
            process.poll()
            for pid in owned:
                if pid != process.pid or process.returncode is not None:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            if not any(alive(pid, owned) for pid in owned):
                remember_adopted(owned)
                if not any(alive(pid, owned) for pid in owned):
                    break
            time.sleep(0.2)
    remember_adopted(owned)
    return [pid for pid in owned if alive(pid, owned)]


def main():
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize:
        raise ValueError("prepared13_entry_isolated_unoptimized_no_bytecode_required")
    if (len(sys.argv) != 3 or sys.argv[1] not in ("help", "capture")
            or sys.argv[2] not in MODES):
        raise ValueError("prepared13_entry_exact_operation_and_mode_required")
    mode, getter = sys.argv[1], sys.argv[2]
    PROTOCOL = PROTOCOLS[getter]
    stem = "single-" + getter
    entry_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ValueError("prepared13_entry_owned_pidfd_required")
    fd = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)
    receipt_path = UPLOAD / (stem + "-" + mode + "-entry-receipt.json")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("prepared13_entry_exclusive_receipt_required")
    if (not UPLOAD.is_dir() or any(p.is_symlink() for p in (UPLOAD, *UPLOAD.parents))
            or not SOURCE.is_dir() or any(p.is_symlink() for p in (SOURCE, *SOURCE.parents))):
        raise ValueError("prepared13_entry_plain_source_and_upload_required")
    if hashlib.sha256((SOURCE / "source-closure.json").read_bytes()).hexdigest() != MANIFEST:
        raise ValueError("prepared13_entry_exact_manifest_required")
    qualified = {
        ROOT / "results/mpc-native-r23/batch-qualification-766f5f70e0024767828041a496a652cf/outcome.json":
            "91ea608734a5d0006d134f8587d946a2ed6e72043c0367c83f831fb4b570351a",
        ROOT / "results/mpc-native-r23/runtime/e0a01-independent-replay.json":
            "9b45898ac892d1c52267623e19e2b7dc1c3a9630c61cdd28a571c6971bbf795e",
        SOURCE / "physical13/trace.json":
            "6d142f59c30d386711b8f91cbcacfd042b79675ef82d1505bedff3bce5cdb71b",
        SOURCE / ("physical13/protocol-" + getter + ".json"): PROTOCOL,
        **{SOURCE / name: digest for name, digest in RECORDED.items()},
    }
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != v for p, v in qualified.items()):
        raise ValueError("prepared13_entry_exact_qualified_inputs_required")
    if mode == "capture":
        if not os.environ.get("TMUX_PANE"):
            raise ValueError("prepared13_entry_named_tmux_required")
        session = subprocess.check_output([
            "tmux", "display-message", "-p", "-t", os.environ["TMUX_PANE"], "#S"],
            text=True, timeout=10).strip()
        if session != TMUX_SESSION:
            raise ValueError("prepared13_entry_named_tmux_required")
        def completed(value, operation):
            return (type(value) is dict and value.get("mode") == operation
                    and value.get("attempt") == "single"
                    and value.get("getter_metrology_mode") == getter
                    and value.get("returncode") == 0 and value.get("cache_empty") is True
                    and value.get("timed_out") is False
                    and all(value.get(key) == [] for key in (
                        "interrupted_signals", "surviving_owned_pids", "supervision_errors"))
                    and type(value.get("new_runs")) is list
                    and len(value["new_runs"]) == (operation == "capture")
                    and value.get("manifest_sha256") == MANIFEST
                    and value.get("execution_seal") == EXECUTION
                    and value.get("protocol_sha256") == PROTOCOL
                    and value.get("recorded_sha256") == RECORDED
                    and value.get("recorded_control_route") == ROUTE
                    and value.get("entry_sha256") == entry_sha256
                    and value.get("child_subreaper") is True)
        help_receipt = json.loads((UPLOAD / (stem + "-help-entry-receipt.json")).read_bytes())
        if not completed(help_receipt, "help"):
            raise ValueError("prepared13_entry_qualified_mode_matched_help_required")
        if any(UPLOAD.glob("*-capture-entry*")):
            raise ValueError("prepared13_entry_single_capture_attempt_only")
    cache = Path(tempfile.mkdtemp(prefix="r24-prepared13-" + stem + "-" + mode + "-", dir=ROOT.parent))
    os.chmod(cache, 0o700)
    args = [
        PYTHON, "-I", "-B", "-X", "pycache_prefix=" + str(cache),
        str(SOURCE / "run_native_shadow.py"),
        "--research-source-manifest-sha256", MANIFEST, "--execution-commit", EXECUTION,
        "--native-physics-support-protocol", str(SOURCE / ("physical13/protocol-" + getter + ".json")),
        "--step-panel", "--prepared-forward", "--prepared-copy-ready",
        "--getter-metrology", getter,
        "--recorded-control-protocol", str(SOURCE / "recorded-control-protocol01.json"),
    ]
    if mode == "help":
        args.append("--help")
    elif mode == "capture":
        args += [
            "--e0-receipt", str(next(iter(qualified))),
            "--e0-replay", str(ROOT / "results/mpc-native-r23/runtime/e0a01-independent-replay.json"),
            "--trace", str(SOURCE / "physical13/trace.json"),
            "--trace-sha256", qualified[SOURCE / "physical13/trace.json"],
            "--development-window", "minute05",
            "--preparation", str(SOURCE / "physical13/preparation.json"),
            "--queue-shadow", "--execute",
        ]
    cancelled = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: cancelled.append(signum))
    before_runs = set((ROOT / "results/mpc-native-r23").glob("azure-dev-*"))
    owned = {}
    supervision_errors = []
    enable_subreaper()
    with (UPLOAD / (stem + "-" + mode + "-entry.log")).open("x") as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        root_fd = None
        timed_out = False
        survivors = []
        returncode = None
        try:
            root_fd = os.pidfd_open(process.pid)
            remember(process.pid, owned, parent=os.getpid())
            deadline = time.monotonic() + (60 if mode == "help" else 1500)
            while process.poll() is None and time.monotonic() < deadline and not cancelled:
                remember_adopted(owned)
                time.sleep(0.2)
            timed_out = process.poll() is None and not cancelled
        except Exception as error:
            supervision_errors.append(type(error).__name__)
        finally:
            try:
                survivors = terminate_owned(owned, process)
            except Exception as error:
                supervision_errors.append("cleanup:" + type(error).__name__)
                if root_fd is not None:
                    try:
                        signal.pidfd_send_signal(root_fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    survivors = [pid for pid in owned if alive(pid, owned)]
                except Exception as error:
                    supervision_errors.append("survivors:" + type(error).__name__)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                returncode = None
            finally:
                if root_fd is not None:
                    os.close(root_fd)
    receipt = {
        "schema": "r24-prepared-physical13-recorded-control-entry-v1",
        "mode": mode, "attempt": "single", "getter_metrology_mode": getter, "argv": args, "returncode": returncode, "timed_out": timed_out,
        "interrupted_signals": cancelled, "owned_pid_start_ticks": owned,
        "child_subreaper": True, "supervision_errors": supervision_errors,
        "surviving_owned_pids": survivors, "tmux_session": TMUX_SESSION if mode == "capture" else None,
        "cache": str(cache), "cache_empty": not any(cache.iterdir()),
        "new_runs": sorted(str(p) for p in
            set((ROOT / "results/mpc-native-r23").glob("azure-dev-*")) - before_runs),
        "manifest_sha256": MANIFEST, "execution_seal": EXECUTION,
        "entry_sha256": entry_sha256,
        "protocol_sha256": PROTOCOL, "native_mpc_permitted": False, "deployable": False,
        "recorded_sha256": RECORDED, "recorded_control_route": ROUTE,
        "scope": "One bounded Source13 recorded-control eight-arm attempt pending separate "
                 "native qualification;not the halted Source12 six-block design or "
                 "MPC/model/PIT efficacy.",
    }
    with receipt_path.open("x") as file:
        file.write(json.dumps(receipt, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return (returncode if returncode is not None and not timed_out and not cancelled
            and not survivors and not supervision_errors else 1)


if __name__ == "__main__":
    raise SystemExit(main())
