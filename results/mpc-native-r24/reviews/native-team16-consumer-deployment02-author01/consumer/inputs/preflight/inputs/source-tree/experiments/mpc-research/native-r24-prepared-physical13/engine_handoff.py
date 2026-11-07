"""Owned engine handoff after free-port qualification; no provider lifecycle calls."""

import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def require_free_ports(ports=(8000, 8080)):
    for port in ports:
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"preexisting_service_on_port:{port}")


def process_start_ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return raw[raw.rfind(")") + 2:].split()[19]


def stop_owned_group(process, start_ticks, *, grace_s=30):
    if process.poll() is not None:
        # A dead leader may still have descendants in its original isolated session.
        try:
            current_ticks = process_start_ticks(process.pid)
        except FileNotFoundError:
            current_ticks = None
        if current_ticks is not None and current_ticks != start_ticks:
            raise RuntimeError("owned_engine_pid_reused")
    elif process_start_ticks(process.pid) != start_ticks or os.getpgid(process.pid) != process.pid:
        raise RuntimeError("owned_engine_process_identity_changed")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        current_ticks = process_start_ticks(process.pid)
    except FileNotFoundError:
        current_ticks = None
    if current_ticks is not None and current_ticks != start_ticks:
        raise RuntimeError("owned_engine_pid_reused_before_kill")
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


@contextlib.contextmanager
def owned_engine(command, *, evidence, env, ready, cwd, grace_s=30):
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    with (evidence / "process.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ticks = process_start_ticks(process.pid)
        receipt = {"pid": process.pid, "process_group": process.pid,
                   "start_ticks": ticks, "command": command, "started_at": time.monotonic(),
                   "native_ready": False, "cleanup_completed": False}
        try:
            readiness = ready(process)
            if process.poll() is not None:
                raise ValueError("owned_engine_exited_during_readiness")
            receipt.update(native_ready=True, readiness=readiness)
            (evidence / "ready.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
            yield process, receipt
        finally:
            try:
                stop_owned_group(process, ticks, grace_s=grace_s)
                receipt["cleanup_completed"] = True
            finally:
                receipt.update(returncode=process.poll(), finished_at=time.monotonic())
                with (evidence / "lifecycle.json").open("x") as stream:
                    json.dump(receipt, stream, sort_keys=True, indent=2)
