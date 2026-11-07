"""Only owned PID/start-tick/pidfd trees; capture-time parent redaction."""

import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time

import common01 as c


class Reader:
    def __init__(self, pipe, output, secret, *, limit=16 * 1024 * 1024):
        self.pipe, self.output, self.secret, self.limit = pipe, output, secret, limit
        self.stop = threading.Event()
        self.error, self.bytes, self.eof = None, 0, False
        self.redact = c.redaction().redacted_chunks
        self.thread = threading.Thread(target=self.run, name="owner-redactor-" + output.parent.name)
        self.thread.start()

    def chunks(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.pipe, selectors.EVENT_READ)
            while not self.stop.is_set():
                if not selector.select(0.05):
                    continue
                chunk = os.read(self.pipe.fileno(), 8192)
                if not chunk:
                    self.eof = True
                    return
                yield chunk

    def run(self):
        try:
            with self.output.open("xb") as stream:
                for clean in self.redact(self.chunks(), self.secret):
                    self.bytes += len(clean)
                    if self.bytes > self.limit:
                        self.error = "stdout_capture_capacity"
                        continue
                    stream.write(clean)
                    stream.flush()
        except BaseException as error:
            self.error = type(error).__name__
        finally:
            self.pipe.close()

    def join(self, deadline):
        self.thread.join(max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            self.stop.set()
            self.thread.join(max(0, deadline - time.monotonic()))
        return dict(joined=not self.thread.is_alive(), eof=self.eof,
                    error_type=self.error, captured_bytes=self.bytes)


class Owned:
    def __init__(self, secret, *, helper=None):
        self.helper = c.owner() if helper is None else helper
        self.helper.enable_subreaper()
        self.secret, self.identities, self.roots = secret, {}, []

    def spawn(self, command, *, directory, cwd, env):
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        root = dict(process=process, reader=None, fd=None, directory=directory)
        self.roots.append(root)
        root["reader"] = Reader(process.stdout, directory / "process.log", self.secret.encode())
        # A registration failure must still leave the direct child reachable for cleanup.
        self.helper.remember(process.pid, self.identities, parent=os.getpid())
        root["fd"] = os.pidfd_open(process.pid)
        ticks = self.identities[process.pid]
        c.require(os.getpgid(process.pid) == process.pid, "owned_session_leader_required")
        receipt = dict(pid=process.pid, start_ticks=str(ticks), process_group=process.pid,
                       started_at=time.monotonic(), native_ready=False, cleanup_completed=False,
                       command=[str(part) for part in command])
        root["receipt"] = receipt
        c.write(directory / "started.json", receipt)
        return root

    def remember(self):
        self.helper.remember_adopted(self.identities)

    def signal(self, root, sig):
        process = root["process"]
        if root["fd"] is not None and self.helper.alive(process.pid, self.identities):
            try:
                signal.pidfd_send_signal(root["fd"], sig)
                return True
            except ProcessLookupError:
                pass
        return False

    def cleanup(self, anchor, *, client=None, cancelled=False, allocation=(7, 30, 2, 1),
                hard_deadline=None):
        natural, reap, join, reserve = allocation
        deadline = min(anchor + sum(allocation), hard_deadline or float("inf"))
        called_at = time.monotonic()
        events = []
        if cancelled and client is not None:
            events.append(dict(action="cancel_owned_client", sent=self.signal(client, signal.SIGTERM)))
        services_signalled = False
        while time.monotonic() < min(anchor + natural, deadline - reap - join - reserve):
            self.remember()
            if client is None or client["process"].poll() is not None:
                if not services_signalled:
                    for root in self.roots:
                        if root is not client:
                            events.append(dict(action="terminate_owned_service",
                                               pid=root["process"].pid,
                                               sent=self.signal(root, signal.SIGTERM)))
                    services_signalled = True
            if all(root["process"].poll() is not None for root in self.roots) and not any(
                    self.helper.alive(pid, self.identities) for pid in self.identities):
                break
            time.sleep(0.02)
        self.remember()
        survivors = []
        if self.roots:
            # The frozen helper's grace=0 gives at most30s SIGKILL/reap.
            if any(self.helper.alive(pid, self.identities) for pid in self.identities):
                events.append(dict(action="frozen_owned_reap"))
                c.require(reap == 30 or getattr(self.helper, "synthetic", False),
                          "frozen_reap_budget_not_silently_changed")
                survivors = self.helper.terminate_owned(
                    self.identities, self.roots[-1]["process"], grace=0)
        join_deadline = min(deadline - reserve, time.monotonic() + join)
        readers = []
        for root in self.roots:
            try:
                root["process"].wait(timeout=max(0.001, join_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                survivors.append(root["process"].pid)
            if root["reader"]:
                readers.append(root["reader"].join(join_deadline))
            elif root["process"].stdout:
                root["process"].stdout.close()
            if root["fd"] is not None:
                os.close(root["fd"])
                root["fd"] = None
            receipt = dict(root.get("receipt", {}), returncode=root["process"].returncode,
                           cleanup_completed=root["process"].returncode is not None,
                           finished_at=time.monotonic())
            c.write(root["directory"] / "lifecycle.json", receipt)
        # Reap adopted zombies individually; never wait/kill arbitrary PIDs.
        for pid in self.identities:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        survivors = sorted(set(survivors) | {
            pid for pid in self.identities if self.helper.alive(pid, self.identities)})
        return dict(started_at=anchor, called_at=called_at, finished_at=time.monotonic(), deadline=deadline,
                    events=events, readers=readers, surviving_owned_pids=survivors,
                    owned_pid_start_ticks=self.identities,
                    all_joined=not survivors and all(row["joined"] for row in readers)
                    and all(root["process"].returncode is not None for root in self.roots),
                    capture_complete=all(row["eof"] and not row["error_type"] for row in readers))
