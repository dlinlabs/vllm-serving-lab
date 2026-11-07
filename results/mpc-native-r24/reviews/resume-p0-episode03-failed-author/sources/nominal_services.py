"""Default-OFF Linux ownership of exactly engine+gateway, not a Native permit.

Use an isolated, cooperating owner process: no unrelated children may be
created during this context. Supply the EXACT two Mayor-preregistered lock
paths; there are no production lock defaults. Readiness must be cooperative
and bound every operation by its supplied remaining budget. A watchdog owns
service cleanup, but this is not a hard-real-time or owner-crash OS guarantee.
Any incomplete/poisoned marker requires separate Mayor reconciliation.
"""

import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time
import traceback


class OwnershipError(RuntimeError):
    pass


@dataclass(frozen=True)
class Service:
    name: str
    argv: tuple
    env: dict
    safe_argv: tuple | None = None


SAFE_ENV = frozenset(("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"))


def _identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text[text.rfind(")") + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), ppid=int(fields[1]),
                pgid=int(fields[2]), sid=int(fields[3]))


def _same(identity):
    now = _identity(identity["pid"])
    return now is not None and now["start_ticks"] == identity["start_ticks"]


def _processes():
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                row = _identity(int(path.name))
            except PermissionError:
                continue
            if row:
                rows[row["pid"]] = row
    return rows


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    if value is None:
        result = ctypes.c_int()
        rc = libc.prctl(37, ctypes.byref(result), 0, 0, 0)
    else:
        rc = libc.prctl(36, int(value), 0, 0, 0)
    if rc:
        raise OSError(ctypes.get_errno(), "subreaper unavailable")
    return result.value if value is None else None


def _durable(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > 131072:
        raise OwnershipError("ownership record too large")
    pending = path.with_name(path.name + ".pending")
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_marker(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 131072:
            raise OwnershipError("invalid bounded ownership marker")
        data = stream.read(131073)
    if len(data) > 131072:
        raise OwnershipError("ownership marker grew")
    return json.loads(data)


class NominalServices:
    def __init__(self, services, *, lock_paths, run_dir, deadline, ready,
                 execute=False, cleanup_reserve=1.0, term_grace=0.25):
        self.services, self.paths = tuple(services), tuple(Path(p) for p in lock_paths)
        self.run_dir, self.deadline, self.ready = Path(run_dir), deadline, ready
        self.execute, self.reserve, self.grace = execute, cleanup_reserve, term_grace
        self._guard, self._stop = threading.RLock(), threading.Event()
        self._fds, self._pidfds, self._known, self._roots = [], {}, {}, {}
        self._errors, self._unknown = [], []
        self._thread = self._log = self._old_subreaper = None
        self._pending, self._status, self._expired = None, "OFF", False
        self._uncertain = self._started = self._failed = False

    def _error(self, where, error):
        with self._guard:
            self._errors.append(where + ":" + type(error).__name__)
            if self._log:
                self._log.write(where + "\n")
                traceback.print_exception(type(error), error, error.__traceback__, file=self._log)
                self._log.flush()

    def _store(self):
        with self._guard:
            gone = self._gone()
            if self._status == "STOPPED" and (not gone or self._uncertain):
                self._status = "POISON"
            record = dict(schema="nominal-service-ownership-v1", status=self._status,
                          owner=self._owner, source_sha256=self._source,
                          lock_paths=[str(p) for p in self.paths], run_dir=str(self.run_dir),
                          deadline_monotonic=self.deadline, pending_spawn=self._pending,
                          services=self._metadata, known=list(self._known.values()),
                          unknown=self._unknown, errors=self._errors[:32],
                          error_count=len(self._errors), expired=self._expired,
                          service_failure=self._failed,
                          known_identities_gone=gone,
                          universal_OS_proof=False, native_eligibility=False)
            for path in (*self._markers, self.run_dir / "ownership.json"):
                _durable(path, record)

    def _gone(self):
        gone = True
        for row in tuple(self._known.values()):
            try:
                gone = not _same(row) and gone
            except BaseException as error:
                gone, self._uncertain = False, True
                self._error("identity verification", error)
        return gone

    def _remember(self, row):
        pid = row["pid"]
        if pid in self._known:
            if self._known[pid]["start_ticks"] != row["start_ticks"]:
                raise OwnershipError("PID identity changed inside scope")
            return
        if len(self._known) >= 128:
            raise OwnershipError("known process inventory limit")
        self._known[pid] = row
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            if _same(row):
                raise
        else:
            if _same(row):
                self._pidfds[pid] = fd
            else:
                os.close(fd)

    def _discover(self):
        rows = _processes()
        with self._guard:
            live = {pid for pid, row in self._known.items()
                    if pid in rows and row["start_ticks"] == rows[pid]["start_ticks"]}
            sessions = {self._known[pid]["sid"] for pid in live}
            changed = True
            before = len(self._known)
            while changed:
                changed = False
                for pid, row in rows.items():
                    if pid not in self._known and (
                            row["ppid"] in live or row["sid"] in sessions):
                        self._remember(row)
                        live.add(pid)
                        sessions.add(row["sid"])
                        changed = True
            if self._pending is None:
                unknown = [row for pid, row in rows.items()
                           if row["ppid"] == os.getpid() and pid not in live]
                if unknown:
                    self._unknown = unknown
                    self._uncertain = True
                    self._stop.set()
            if len(self._known) != before:
                self._store()

    def _signal(self, row, sig):
        if time.monotonic() >= self.deadline:
            raise OwnershipError("no signalling budget remains")
        if _same(row):
            try:
                signal.pidfd_send_signal(self._pidfds[row["pid"]], sig)
            except ProcessLookupError:
                pass

    def _reap(self):
        roots = {p.pid for p in self._roots.values()}
        for process in tuple(self._roots.values()):
            try:
                process.poll()
            except BaseException as error:
                self._uncertain = True
                self._error("root reap", error)
        for pid, row in tuple(self._known.items()):
            if pid not in roots and pid in self._pidfds:
                try:
                    os.waitid(os.P_PIDFD, self._pidfds[pid], os.WEXITED | os.WNOHANG)
                except ChildProcessError:
                    pass
                except BaseException as error:
                    self._uncertain = True
                    self._error("descendant reap", error)

    def _cleanup(self):
        term_end = min(time.monotonic() + self.grace, self.deadline)
        finish = self.deadline - min(0.1, self.reserve / 4)
        sent = set()
        while time.monotonic() < finish:
            try:
                self._discover()
                self._reap()
            except BaseException as error:
                self._uncertain = True
                self._error("discover/reap", error)
            sig = signal.SIGTERM if time.monotonic() < term_end else signal.SIGKILL
            for pid, row in tuple(self._known.items()):
                key = (pid, sig)
                if key not in sent:
                    sent.add(key)
                    try:
                        self._signal(row, sig)
                    except BaseException as error:
                        self._uncertain = True
                        self._error(f"signal:{pid}:{sig}", error)
            if self._pending is None and self._gone():
                try:
                    self._discover()
                    self._reap()
                    if self._gone():
                        break
                except BaseException as error:
                    self._uncertain = True
                    self._error("final discovery", error)
            time.sleep(min(0.01, max(0, finish - time.monotonic())))
        self._reap()
        gone = self._gone()
        self._status = ("STOPPED" if gone and not self._pending and not self._uncertain
                        and not self._unknown else "POISON")
        self._store()

    def _monitor(self):
        try:
            while not self._stop.wait(0.02):
                self._discover()
                if time.monotonic() >= self.deadline - self.reserve:
                    self._expired = True
                    break
                if any(p.poll() is not None for p in tuple(self._roots.values())):
                    self._failed = True
                    self._error("service_exit", OwnershipError("owned service exited"))
                    break
        except BaseException as error:
            self._uncertain = self._failed = True
            self._error("monitor", error)
        finally:
            self._stop.set()
            try:
                self._cleanup()
            except BaseException as error:
                self._status, self._uncertain = "POISON", True
                self._error("cleanup", error)
                try:
                    self._store()
                except BaseException as failure:
                    self._error("poison_persistence; RUNNING/pending remains", failure)

    def check(self):
        if time.monotonic() >= self.deadline - self.reserve:
            self._expired = True
            self._stop.set()
        if self._stop.is_set():
            raise OwnershipError("service scope stopped or work deadline exhausted")

    def __enter__(self):
        if self.execute is not True:
            raise OwnershipError("default OFF: explicit execute=True required")
        if (tuple(s.name for s in self.services) != ("engine", "gateway")
                or len(self.paths) != 2 or len(set(self.paths)) != 2
                or not all(p.is_absolute() for p in self.paths)
                or not math.isfinite(self.deadline) or not 0 < self.grace < self.reserve
                or time.monotonic() >= self.deadline - self.reserve):
            raise OwnershipError("invalid scope, locks or original deadline")
        self.paths = tuple(sorted(self.paths))
        self._markers = tuple(p.with_name(p.name + ".nominal-state.json") for p in self.paths)
        try:
            inodes = set()
            for path in self.paths:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
                self._fds.append(fd)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) in inodes:
                    raise OwnershipError("two distinct regular lock files required")
                inodes.add((info.st_dev, info.st_ino))
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in self._markers:
                if path.with_name(path.name + ".pending").exists() or path.is_symlink():
                    raise OwnershipError("unresolved ownership marker")
                if path.exists():
                    prior = _read_marker(path)
                    if (prior.get("status") != "STOPPED"
                            or prior.get("schema") != "nominal-service-ownership-v1"
                            or prior.get("lock_paths") != [str(p) for p in self.paths]
                            or prior.get("pending_spawn") is not None or prior.get("unknown") != []
                            or prior.get("known_identities_gone") is not True
                            or type(prior.get("known")) is not list or len(prior["known"]) > 128
                            or any(type(row) is not dict
                                   or type(row.get("pid")) is not int or row["pid"] < 1
                                   or type(row.get("start_ticks")) is not int or row["start_ticks"] < 0
                                   for row in prior["known"])
                            or any(_same(row) for row in prior["known"])):
                        raise OwnershipError("POISON/RUNNING/unknown: Mayor reconciliation required")
            if any(row["ppid"] == os.getpid() for row in _processes().values()):
                raise OwnershipError("isolated owner process required")
            self.run_dir.mkdir(mode=0o700)
            fd = os.open(self.run_dir / "private-errors.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._log = os.fdopen(fd, "w")
            self._owner = _identity(os.getpid())
            self._source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            self._metadata = []
            for service in self.services:
                if (type(service.argv) not in (tuple, list) or not service.argv
                        or not all(type(a) is str and a and "\0" not in a for a in service.argv)
                        or not all(type(k) is str and type(v) is str for k, v in service.env.items())):
                    raise OwnershipError("fixed argv and explicit string environment required")
                safe = service.safe_argv or ("<redacted>",) * len(service.argv)
                if len(safe) != len(service.argv) or any(
                        shown not in (actual, "<redacted>") for shown, actual in zip(safe, service.argv)):
                    raise OwnershipError("safe argv must match or redact each argument")
                self._metadata.append(dict(name=service.name, argv=list(safe),
                                           root_identity=None,
                                           env={k: v for k, v in service.env.items() if k in SAFE_ENV},
                                           redacted_env_entries=len(set(service.env) - SAFE_ENV)))
            self._old_subreaper = _subreaper()
            probe = os.pidfd_open(os.getpid())
            os.close(probe)
            if not hasattr(signal, "pidfd_send_signal") or not hasattr(os, "P_PIDFD"):
                raise OwnershipError("stable pidfd signal/reap support required")
            _subreaper(True)
            self._status, self._started = "RUNNING", True
            self._store()
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()
            for index, service in enumerate(self.services):
                with self._guard:
                    self.check()
                    self._pending = service.name
                    self._store()
                    fd = os.open(self.run_dir / (service.name + ".log"),
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    try:
                        process = subprocess.Popen(list(service.argv), env=dict(service.env),
                                                   start_new_session=True, close_fds=True,
                                                   stdin=subprocess.DEVNULL, stdout=fd, stderr=fd)
                    finally:
                        os.close(fd)
                    self._roots[service.name] = process
                    row = _identity(process.pid)
                    if row is None or row["pgid"] != process.pid or row["sid"] != process.pid:
                        raise OwnershipError("spawn identity unknown")
                    self._remember(row)
                    self._metadata[index]["root_identity"] = row
                    self._pending = None
                    self._store()
            while True:
                self.check()
                if any(p.poll() is not None for p in self._roots.values()):
                    self._failed = True
                    raise OwnershipError("known service startup failure")
                if self.ready(self, max(0, self.deadline - self.reserve - time.monotonic())):
                    self.check()
                    return self
                self._stop.wait(0.02)
        except BaseException as error:
            self._error("enter", error)
            self._finish()
            raise

    def _finish(self):
        self._stop.set()
        if self._thread:
            self._thread.join(max(0, self.deadline - time.monotonic()))
        if self._thread and self._thread.is_alive():
            self._status, self._uncertain = "POISON", True
            self._store()
            raise OwnershipError("owner cleanup thread not joined; locks retained")
        try:
            if self._started and self._thread is None:
                self._status = "POISON"
                self._store()
            if self._old_subreaper is not None:
                _subreaper(self._old_subreaper)
        finally:
            for fd in (*self._pidfds.values(), *self._fds):
                os.close(fd)
            self._pidfds.clear()
            self._fds.clear()
            if self._log:
                self._log.close()
                self._log = None
        if self._status == "POISON":
            raise OwnershipError("POISON: incomplete ownership; inspect private evidence")

    def __exit__(self, kind, error, tb):
        if time.monotonic() >= self.deadline - self.reserve:
            self._expired = True
        if error is not None:
            self._error("body", error)
        self._finish()
        if kind is None and (self._expired or self._failed):
            raise OwnershipError("original work deadline exhausted or service failed")
