"""Independent owned caller authority, read by the consumer through a real RO FD."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import struct
import time
import uuid

import common01 as c

FORMAT = struct.Struct("!8s16sQQQQQddd32s")
MAGIC = b"CBGEN002"
KINDS = ("bootstrap_confirmed", "offer_intent", "offer_dispatched", "offer_outcome",
         "public_history_published", "control_history_published")


@contextmanager
def locked(fd, operation):
    try:
        fcntl.flock(fd, operation | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ValueError("generation_authority_busy") from error
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def file_identity(fd):
    info = os.fstat(fd)
    c.require(stat.S_ISREG(info.st_mode) and info.st_mode & 0o777 == 0o600
              and info.st_nlink == 1 and info.st_uid == os.getuid()
              and info.st_size == FORMAT.size, "generation_file_identity")
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, uid=info.st_uid)


class Reader:
    def __init__(self, fd, binding):
        self.fd, self.binding = fd, binding
        c.require(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
                  "generation_readonly_descriptor_required")
        c.require(file_identity(fd) == binding["file"], "generation_authority_inode")

    def read(self):
        started = time.monotonic()
        with locked(self.fd, fcntl.LOCK_SH):
            c.require(file_identity(self.fd) == self.binding["file"], "generation_authority_changed")
            raw = os.pread(self.fd, FORMAT.size + 1, 0)
        c.require(len(raw) == FORMAT.size, "generation_authority_partial")
        magic, run, pid, ticks, generation, lease, lease_generation, begin, active, deadline, domain = FORMAT.unpack(raw)
        b = self.binding
        c.require(magic == MAGIC and run.hex() == b["run_id"]
                  and (pid, ticks) == (b["parent"]["process"]["pid"], b["parent"]["process"]["start_ticks"])
                  and domain.hex() == c.sha(b["parent"]["clock_domain"].encode())
                  and lease in (0, 1), "generation_authority_binding")
        c.require(all(c.number(v) == v for v in (begin, active, deadline))
                  and (lease == 0 and lease_generation == 0 and begin == active == deadline == 0
                       or lease == 1 and lease_generation <= generation
                       and begin < deadline <= min(begin + c.PIPELINE, active)),
                  "generation_lease_binding")
        return dict(run_id=b["run_id"], generation=generation, lease=lease, begin=begin,
                    lease_generation=None if lease == 0 else lease_generation,
                    active_deadline=active, deadline=deadline,
                    read_started_at=started, read_finished_at=time.monotonic(),
                    authority_file=b["file"])

    def check(self, expected, *, require_lease=True):
        c.require(type(expected) is int and expected >= 0, "known_caller_generation_required")
        value = self.read()
        c.require(value["generation"] == expected
                  and (not require_lease or value["lease"] == 1 and value["lease_generation"] == expected),
                  "caller_generation_changed")
        return value


class Authority:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "caller-authority.bin"
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.run_id = uuid.uuid4().hex
        self.parent = c.current_identity()
        self.generation, self.lease, self.lease_generation = 0, 0, 0
        self.begin = self.active = self.deadline = 0.0
        self.events, self.attempts = [], []
        self.poisoned = False
        self.invalidated_at = self.invalidation_reason = None
        self._commit()
        self.read_fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        self.binding = dict(run_id=self.run_id, parent=self.parent, file=file_identity(self.fd),
                            namespace=c.NAMESPACE + ":caller-publication-v2")
        self.reader = Reader(self.read_fd, self.binding)
        c.write(self.directory / "authority-created.json", self.binding)

    def poison(self, reason):
        self.poisoned = True
        if self.invalidated_at is None:
            self.invalidated_at, self.invalidation_reason = time.monotonic(), reason

    def _commit(self, *, generation=None):
        process = self.parent["process"]
        raw = FORMAT.pack(MAGIC, bytes.fromhex(self.run_id), process["pid"], process["start_ticks"],
                          self.generation if generation is None else generation,
                          self.lease, self.lease_generation, self.begin, self.active, self.deadline,
                          bytes.fromhex(c.sha(self.parent["clock_domain"].encode())))
        try:
            with locked(self.fd, fcntl.LOCK_EX):
                c.require(os.pwrite(self.fd, raw, 0) == len(raw), "generation_short_write")
                os.fsync(self.fd)
        except BaseException as error:
            self.poison(str(error) if isinstance(error, ValueError) else type(error).__name__)
            raise

    def publish(self, kind, evidence):
        attempt = dict(sequence=len(self.attempts), kind=kind if kind in KINDS else None,
                       observed_at=time.monotonic(), previous_generation=self.generation,
                       desired_generation=self.generation + 1, evidence_sha256=None, committed=False)
        self.attempts.append(attempt)
        try:
            attempt["evidence_sha256"] = c.sha(c.canonical(evidence))
            c.require(not self.poisoned, "caller_authority_poisoned")
            c.require(kind in KINDS, "caller_publication_event_required")
            c.require(self.generation < 2 ** 64 - 1, "caller_generation_exhausted")
            self._commit(generation=attempt["desired_generation"])
            self.generation = attempt["desired_generation"]
            attempt.update(committed=True, published_at=time.monotonic())
            event = dict(generation=self.generation, kind=kind, observed_at=attempt["observed_at"],
                         published_at=attempt["published_at"], evidence_sha256=attempt["evidence_sha256"])
            self.events.append(event)
            c.write(self.directory / f"publication-{self.generation:04d}.json", event)
            attempt["status"] = "counted_and_retained"
        except BaseException as error:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
            self.poison(reason)
            attempt.update(status="refused", error_type=type(error).__name__, reason=reason,
                           invalidated_at=self.invalidated_at, lease=self.lease,
                           lease_generation=None if self.lease == 0 else self.lease_generation)
            try:
                c.write(self.directory / f"publication-refused-{attempt['sequence']:04d}.json", attempt)
            except BaseException as retention_error:
                attempt["retention_error_type"] = type(retention_error).__name__
            raise
        return self.generation

    def start(self, active_deadline):
        c.require(not self.poisoned, "caller_authority_poisoned")
        c.require(self.lease == 0, "one_authoritative_lease_only")
        self.begin = time.monotonic()
        self.active = c.number(active_deadline)
        self.deadline = min(self.begin + c.PIPELINE, self.active)
        c.require(self.deadline > self.begin, "active_deadline_exhausted")
        self.lease = 1
        self.lease_generation = self.generation
        self._commit()
        return self.reader.check(self.generation)

    def tighten(self, source_at):
        c.require(not self.poisoned, "caller_authority_poisoned")
        deadline = min(self.deadline, c.number(source_at) + .5)
        c.require(deadline > self.begin, "source_deadline_before_boundary")
        if deadline < self.deadline:
            self.deadline = deadline
            self._commit()

    def check(self, generation):
        c.require(not self.poisoned, "caller_authority_poisoned")
        current = self.reader.check(generation)
        c.require(self.generation == generation, "caller_generation_not_published")
        return current

    def close(self):
        for name in ("fd", "read_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
