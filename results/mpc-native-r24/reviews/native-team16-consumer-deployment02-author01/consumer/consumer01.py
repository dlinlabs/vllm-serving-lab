"""Default-off one-shot source consumer; trusted caller publication, never control."""

import asyncio
from dataclasses import asdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import common01 as c
from generation01 import Authority, KINDS
import ipc01 as ipc


class CutConsumer:
    def __init__(self, output, *, enabled=False, evidence_mode="owned-native", redaction_secret="",
                 host_deadline=None, startup_deadline=None):
        self.output = Path(output).resolve()
        self.enabled = enabled is True
        self.mode, self.secret = evidence_mode, redaction_secret
        self.host_deadline, self.host_startup_deadline = host_deadline, startup_deadline
        self.process = self.reader = self.authority = None
        self.pidfd = self.send_fd = self.recv_fd = self.finish_fd = None
        self.identities, self.clocks, self.errors = {}, {}, []
        self.received = bytearray()
        self.used, self.active_at, self.lease = False, None, None
        self.release = self.cleanup = None
        self.closed = False
        self._close_task = None
        self.publication_refusals = []

    def _chain(self):
        api = c.proc_api()
        domain = self.authority.parent["clock_domain"]
        boot, ns = api.domain_parts(domain)
        collector, chain = api.inspect_chain(api.LinuxProc(), self.process.pid,
            self.identities[self.process.pid], self.process.pid, boot, ns)
        return dict(collector=asdict(collector), owned_chain=[asdict(row) for row in chain],
                    boot_id=boot, clock_domain=domain)

    async def start(self):
        if not self.enabled:
            return dict(status="disabled", **c.permissions())
        c.require(self.mode in ("owned-native", "cpu-source-fixture") and self.process is None,
                  "one_owned_consumer_start")
        c.require(type(self.secret) is str and bool(self.secret), "nonempty_capture_redaction_token")
        c.require(self.mode != "owned-native" or len(self.secret) == 64,
                  "capture_redaction_secret_required_for_native_input")
        c.require(self.mode != "owned-native" or (
            self.host_deadline is not None and self.host_startup_deadline is not None),
            "native_host_absolute_budgets_required")
        c.verify_code()
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.hard_deadline = min(self.started + 260, c.number(self.host_deadline)
                                 if self.host_deadline is not None else self.started + 260)
        self.outer_startup_deadline = min(self.started + 180, c.number(self.host_startup_deadline)
            if self.host_startup_deadline is not None else self.started + 180)
        self.authority = Authority(self.output)
        child_output = self.output / "worker"
        child_output.mkdir(mode=0o700)
        request_read, self.send_fd = os.pipe()
        self.recv_fd, response_write = os.pipe()
        finish_read, self.finish_fd = os.pipe()
        for fd in (self.send_fd, self.recv_fd, self.finish_fd):
            os.set_blocking(fd, False)
        config = dict(authority=self.authority.binding, evidence_mode=self.mode,
                      worker_output=str(child_output), startup_deadline=min(
                          self.started + 5, self.outer_startup_deadline), hard_deadline=self.hard_deadline)
        c.write(self.output / "worker-config.json", config)
        argv = [sys.executable, "-I", "-B", str(c.HERE / "worker01.py"), "--enabled",
                "--config", str(self.output / "worker-config.json"), "--read-fd", str(request_read),
                "--write-fd", str(response_write), "--authority-fd", str(self.authority.read_fd),
                "--finish-fd", str(finish_read)]
        environment = {name: os.environ[name] for name in ("PATH", "HOME", "LANG", "LC_ALL") if name in os.environ}
        environment.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                           VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1")
        try:
            self.process = subprocess.Popen(
                argv, cwd=c.PREFLIGHT, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(request_read, response_write, self.authority.read_fd, finish_read))
            c.ownership().remember(self.process.pid, self.identities, parent=os.getpid())
            self.pidfd = os.pidfd_open(self.process.pid)
            self.reader = c.reader_type()(self.process.stdout, child_output / "process.log", self.secret.encode())
            c.require(self.process.pid in self.identities, "owned_consumer_registration")
            before_at = time.monotonic()
            before = self._chain()
            await ipc.write_async(self.send_fd, b"R", config["startup_deadline"], {})
            raw = bytearray()
            ready, _, _ = await ipc.receive_async(self.recv_fd, config["startup_deadline"], raw)
            received_at = time.monotonic()
            after = self._chain()
            c.require(before == after and ready["schema"] == "team16-consumer-ready-v1"
                      and ready["identity"]["process"] == after["owned_chain"][0]
                      and ready["identity"]["clock_domain"] == after["clock_domain"]
                      and ready["identity"]["boot_id"] == after["boot_id"]
                      and before_at <= ready["ready_at"] <= received_at
                      and ready["run_id"] == self.authority.run_id
                      and ready["evidence_mode"] == self.mode
                      and ready["source_freeze_sha256"] == c.sha((c.HERE / "CONTROL_FREEZE.json").read_bytes()),
                      "actual_consumer_ready_process_source_clock_binding")
            self.ready = ready
            c.raw_write(self.output / "ready-frame.bin", bytes(raw))
            c.write(self.output / "started.json", dict(
                argv=argv, pid=self.process.pid, start_ticks=self.identities[self.process.pid],
                before=before, after=after, checked_from=before_at, received_at=received_at,
                source="actual_readonly_procfs_not_source_fixture", ready=ready))
            return ready
        except BaseException as error:
            self.errors.append(dict(phase="startup", error_type=type(error).__name__, reason=str(error)))
            await self.close(cancelled=True)
            raise
        finally:
            for fd in (request_read, response_write, finish_read):
                os.close(fd)

    def activate(self):
        c.require(self.enabled and self.process is not None and self.active_at is None
                  and time.monotonic() < self.outer_startup_deadline, "one_actual_active_start")
        self.active_at = time.monotonic()
        c.require(self.active_at + 20 + 3 + 40 + 17 <= self.hard_deadline, "remaining_outer_phase_budget")
        c.write(self.output / "active.json", dict(started_at=self.active_at, deadline=self.active_at + 20))
        return self.active_at

    def publish(self, kind, evidence):
        before = 0 if self.authority is None else len(self.authority.attempts)
        try:
            c.require(self.enabled and self.authority is not None and not self.closed, "owned_caller_required")
            return self.authority.publish(kind, evidence)
        except Exception as error:
            attempt = self.authority.attempts[-1] if self.authority is not None \
                and len(self.authority.attempts) > before else None
            return self.invalidate_publication(kind, evidence, error, attempt=attempt)

    def invalidate_publication(self, kind, evidence, error, *, attempt=None):
        reason = str(error) if isinstance(error, ValueError) else type(error).__name__
        if self.authority is not None:
            self.authority.poison(reason)
        row = dict(status="consumer_refused_producer_continues", counted=False,
                   sequence=len(self.publication_refusals), error_type=type(error).__name__,
                   reason=reason, refused_at=time.monotonic(), authority_poisoned=True,
                   source_state_is_current_state=False, **c.permissions())
        row["event_kind"] = kind if kind in KINDS else None
        try:
            row["evidence_sha256"] = c.sha(c.canonical(evidence))
        except Exception as encoding_error:
            row["evidence_sha256"] = None
            row["evidence_encoding_error_type"] = type(encoding_error).__name__
        if self.authority is not None:
            row["authority_invalidated_at"] = self.authority.invalidated_at
            if attempt is not None:
                row["publication_attempt"] = dict(attempt)
                row["counted"] = row["publication_attempt"]["committed"]
        self.publication_refusals.append(row)
        self.errors.append(dict(phase="publication", sequence=row["sequence"], reason=reason))
        if self.cleanup is not None:
            self.cleanup["component_pass"] = False
        try:
            c.write(self.output / f"consumer-publication-refused-{row['sequence']:04d}.json", row)
        except Exception as retention_error:
            row["retention_error_type"] = type(retention_error).__name__
        return row

    def begin(self, *, ordinal, pending_at_trigger):
        if not self.enabled:
            return dict(status="disabled", **c.permissions())
        c.require(not self.used and not self.closed, "one_ordinal2_attempt_no_fallback")
        self.used = True
        c.write(self.output / "handoff-intent.json", dict(ordinal=ordinal, pending_at_trigger=pending_at_trigger))
        c.require(ordinal == 2 and type(ordinal) is int and pending_at_trigger is True,
                  "designated_ordinal2_unavailable_no_substitution")
        c.require(self.active_at is not None, "active_stage_required")
        self.lease = self.authority.start(self.active_at + 20)
        self.clocks["collection_boundary_started_at"] = self.lease["begin"]
        self.process_before = self._chain()
        self.clocks["consumer_process_checked_before_at"] = time.monotonic()
        c.require(time.monotonic() < self.lease["deadline"], "begin_absolute_deadline")
        c.require(os.write(self.send_fd, b"B") == 1, "consumer_BEGIN_write")
        self.clocks["collector_call_entered_at"] = time.monotonic()
        return dict(self.lease)

    def _fresh(self):
        value = self.authority.check(self.lease["generation"])
        c.require(time.monotonic() < value["deadline"], "parent_absolute_deadline")
        return value

    async def handoff(self, source, *, require_exact_application_time=False):
        if not self.enabled:
            return dict(status="disabled", **c.permissions())
        c.require(self.lease is not None and "collector_returned_at" not in self.clocks,
                  "one_started_collection_return_only")
        self.clocks["collector_returned_at"] = time.monotonic()
        receipt, value = {}, None
        try:
            c.no_secret(source, self.secret)
            self.authority.tighten(source["current"]["journal"]["cut"]["cut_at"])
            lease = self._fresh()
            value = dict(schema="team16-cut-consumer-input-v1", namespace=c.NAMESPACE,
                         run_id=self.authority.run_id, ordinal=2,
                         generation=self.lease["generation"], boundary_started_at=self.lease["begin"],
                         active_deadline=lease["active_deadline"], deadline=lease["deadline"],
                         collector_returned_at=self.clocks["collector_returned_at"], source=source,
                         require_exact_application_time=require_exact_application_time)
            self.clocks["serialization_started_at"] = time.monotonic()
            raw = c.canonical(value)
            self.clocks["serialization_finished_at"] = time.monotonic()
            c.require(len(raw) <= c.MAX_BYTES, "consumer_input_capacity_no_trimming")
            c.raw_write(self.output / "offered-input.bin", raw)
            self._fresh()
            self.clocks["enqueue_send_started_at"] = time.monotonic()
            framed = ipc.frame(raw, sent=self.clocks["enqueue_send_started_at"],
                serial_started=self.clocks["serialization_started_at"],
                serial_finished=self.clocks["serialization_finished_at"])
            await ipc.write_async(self.send_fd, framed, lease["deadline"], receipt)
            self.clocks["enqueue_send_finished_at"] = time.monotonic()
            os.close(self.send_fd)
            self.send_fd = None
            result, _, header = await ipc.receive_async(self.recv_fd, lease["deadline"], self.received)
            self.clocks["parent_result_receive_at"] = time.monotonic()
            self._fresh()
            c.require(result["schema"] == "team16-consumer-result-v1" and result["status"] == "validated",
                      "consumer_refused_or_unknown_output:" + str(result.get("reason")))
            c.require(set(result) == {
                "schema", "status", "run_id", "identity", "input_sha256", "header", "authority_reads",
                "clocks", "full_stock", "full_stock_sha256", "source_cut_sha256", "source_at",
                "raw_inventory", "source_phases", "initial_closure_sha256", "initial_point_gate",
                "source_state_is_current_state", "evidence_mode"} | set(c.permissions()),
                "unknown_consumer_output_fields")
            c.require(result["run_id"] == self.authority.run_id and result["identity"] == self.ready["identity"]
                      and result["input_sha256"] == c.sha(raw), "consumer_result_input_identity")
            expected_stock = {key: source["current"]["journal"]["cut"]["native_observation"][key]
                              for key in ("base", "scheduler_state", "queue_oldest_first",
                                          "queued_ghost_request_ids", "skipped_waiting", "streaming_waiting")}
            c.require(c.canonical(result["full_stock"]) == c.canonical(expected_stock)
                      and result["full_stock_sha256"] == c.sha(c.canonical(expected_stock))
                      and result["source_cut_sha256"] == c.sha(c.canonical(source["current"]["journal"]["cut"]))
                      and c.canonical({name:result[name] for name in c.permissions()}) == c.canonical(c.permissions())
                      and result["source_state_is_current_state"] is False
                      and result["initial_point_gate"] is False,
                      "consumer_stock_hash_or_permission_changed")
            phase = result["source_phases"]
            original_cut = source["current"]["journal"]["cut"]
            c.require(result["source_at"] == original_cut["cut_at"] and result["evidence_mode"] == self.mode
                      and phase["source_at"] == original_cut["cut_at"]
                      and phase["clock_domain"] == self.authority.parent["clock_domain"]
                      and phase["capture_id"] == c.CAPTURE
                      and phase["gateway_run_id"] == original_cut["owner_prefix"]["run_id"]
                      and phase["native_run_id"] == original_cut["native_observation"]["native_phase"]["run_id"]
                      and phase["cut_canonical_sha256"] == result["source_cut_sha256"],
                      "consumer_result_source_epoch_binding")
            for which in ("initial", "current"):
                bundle = source[which]
                expected = {key: dict(bytes=len(c.raw64(bundle[key + "_base64"])),
                                     sha256=c.sha(c.raw64(bundle[key + "_base64"])))
                            for key in ("snapshot", "prefix", "barrier")}
                native = c.raw64(bundle["journal"]["native_record"]["response_body_base64"])
                expected["native_HTTP_body"] = dict(bytes=len(native), sha256=c.sha(native))
                c.require(result["raw_inventory"][which] == expected, "consumer_raw_inventory_changed")
            clock = result["clocks"]
            self.clocks["consumer_result_send_at"] = header["enqueue_send_started_at"]
            self.clocks["consumer_result_serialization_started_at"] = header["serialization_started_at"]
            self.clocks["consumer_result_serialization_finished_at"] = header["serialization_finished_at"]
            times = [self.clocks["collector_returned_at"], self.clocks["serialization_started_at"],
                     self.clocks["serialization_finished_at"], self.clocks["enqueue_send_started_at"],
                     clock["consumer_recv_started_at"], clock["consumer_dequeue_at"],
                     clock["consumer_validation_started_at"], clock["consumer_validation_finished_at"],
                     header["serialization_started_at"], header["serialization_finished_at"],
                     header["enqueue_send_started_at"], self.clocks["parent_result_receive_at"]]
            c.require(times == sorted(times) and times[-1] < lease["deadline"],
                      "actual_consumer_phase_chronology")
            reads = result["authority_reads"]
            c.require(len(reads) == 3 and all(
                row["run_id"] == self.authority.run_id
                and row["generation"] == self.lease["generation"]
                and row["lease_generation"] == self.lease["generation"]
                and row["authority_file"] == self.authority.binding["file"]
                and clock["consumer_validation_started_at"] <= row["read_started_at"]
                <= row["read_finished_at"] <= header["enqueue_send_started_at"]
                for row in reads), "actual_cross_process_generation_reads")
            after = self._chain()
            c.require(after == self.process_before, "consumer_process_clock_changed")
            self.clocks["consumer_process_checked_after_at"] = time.monotonic()
            authority = self._fresh()
            c.write(self.output / "send-receipt.json", dict(receipt, clocks=self.clocks,
                                                         input_sha256=c.sha(raw)))
            c.raw_write(self.output / "result-at-handoff-return.bin", bytes(self.received))
            self.clocks["release_journal_started_at"] = time.monotonic()
            c.write(self.output / "release-candidate.json", dict(
                status="validated_pending_final_guard_not_released", result=result,
                parent_clocks=self.clocks, generation=authority,
                process_before=self.process_before, process_after=after))
            self.clocks["release_journal_finished_at"] = time.monotonic()
            final_generation = self._fresh()
            self.clocks["post_journal_release_at"] = time.monotonic()
            c.require(self.clocks["post_journal_release_at"] < lease["deadline"],
                      "post_journal_absolute_deadline")
            c.require(os.write(self.finish_fd, b"F") == 1, "final_guard_finish_signal")
            c.require(time.monotonic() < lease["deadline"], "post_finish_signal_deadline")
            self.release = dict(status="validated_pending_cleanup", source=result,
                clocks=dict(self.clocks), deadline=lease["deadline"], generation=final_generation,
                measured_in_pipeline_reported_later=True, component_pass=False, **c.permissions())
            self.clocks["handoff_return_guard_at"] = time.monotonic()
            c.require(self.clocks["handoff_return_guard_at"] < lease["deadline"], "final_return_deadline")
            self.release["clocks"] = dict(self.clocks)
            return self.release
        except BaseException as error:
            self.release = None
            self.errors.append(dict(phase="handoff", error_type=type(error).__name__, reason=str(error)))
            c.write(self.output / "handoff-refused.json", dict(
                error_type=type(error).__name__, reason=str(error), clocks=self.clocks))
            if not (self.output / "send-receipt.json").exists():
                c.write(self.output / "send-receipt.json", dict(receipt, clocks=self.clocks,
                    input_sha256=None if value is None else c.sha(c.canonical(value))))
            if not (self.output / "result-at-handoff-return.bin").exists():
                c.raw_write(self.output / "result-at-handoff-return.bin", bytes(self.received))
            raise

    def _signal(self, sig):
        if self.pidfd is not None and c.ownership().alive(self.process.pid, self.identities):
            try:
                signal.pidfd_send_signal(self.pidfd, sig)
                return True
            except ProcessLookupError:
                pass
        return False

    def _drain(self):
        if self.recv_fd is None:
            return
        while True:
            try:
                part = os.read(self.recv_fd, 65536)
            except BlockingIOError:
                return
            if not part:
                return
            if len(self.received) + len(part) > c.MAX_BYTES + ipc.HEADER.size:
                self.errors.append(dict(phase="drain", reason="result_drain_capacity",
                                        refused_bytes=len(part), refused_sha256=c.sha(part)))
                self._signal(signal.SIGKILL)
                return
            self.received.extend(part)

    async def close(self, *, cancelled=False, teardown_anchor=None, hard_deadline=None):
        if not self.enabled:
            return dict(status="disabled", all_joined=True, **c.permissions())
        if self.closed:
            return self.cleanup
        if self.authority is None:
            return dict(status="not_started", all_joined=self.process is None, **c.permissions())
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(
                cancelled=cancelled, teardown_anchor=teardown_anchor, hard_deadline=hard_deadline))
        cancellation = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancellation = True
        result = self._close_task.result()
        if cancellation:
            self.errors.append(dict(phase="cleanup_caller", reason="cancelled_after_owned_join"))
            self.cleanup["component_pass"] = False
            c.write(self.output / "cleanup-caller-cancelled.json", self.cleanup)
            raise asyncio.CancelledError
        return result

    async def _close(self, *, cancelled, teardown_anchor, hard_deadline):
        anchor = time.monotonic() if teardown_anchor is None else c.number(teardown_anchor)
        deadline = min(anchor + 40, self.hard_deadline,
                       self.hard_deadline if hard_deadline is None else c.number(hard_deadline))
        events = []
        for name in ("send_fd", "finish_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        if self.process is not None:
            if cancelled:
                events.append(dict(action="cancel_owned_consumer", sent=self._signal(signal.SIGTERM)))
            while self.process.poll() is None and time.monotonic() < min(anchor + 7, deadline - 33):
                self._drain()
                await asyncio.sleep(.005)
            if self.process.poll() is None:
                events.append(dict(action="kill_owned_consumer", sent=self._signal(signal.SIGKILL)))
            while self.process.poll() is None and time.monotonic() < min(anchor + 37, deadline - 3):
                self._drain()
                await asyncio.sleep(.005)
            self._drain()
            if self.process.poll() is None:
                self.errors.append(dict(phase="cleanup", reason="owned_consumer_unjoined_at_absolute_teardown"))
            else:
                self.process.wait(timeout=0)
        reader = None if self.reader is None else self.reader.join(min(anchor + 39, deadline - 1))
        gone = self.process is None or not c.ownership().alive(self.process.pid, self.identities)
        postflight_path = self.output / "worker/postflight.json"
        try:
            postflight = c.load(postflight_path) if postflight_path.exists() else None
            c.require(postflight is None or (
                type(postflight) is dict and postflight.get("status") in ("verified", "refused")),
                "unknown_worker_postflight")
        except (ValueError, OSError) as error:
            self.errors.append(dict(phase="cleanup_postflight", error_type=type(error).__name__))
            postflight = dict(status="refused", error_type=type(error).__name__)
        joined = gone and (reader is None or reader["joined"])
        normal = self.process is not None and self.process.returncode == 0 and postflight is not None \
            and postflight["status"] == "verified" and reader is not None and reader["eof"] and not reader["error_type"]
        for name in ("pidfd", "recv_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        self.authority.close()
        self.closed = True
        candidate = bool(self.release and not self.errors and normal and joined
                         and not self.authority.poisoned and not self.publication_refusals)
        self.cleanup = dict(started_at=anchor, finished_at=time.monotonic(), deadline=deadline,
            allocation_s=[7, 30, 2, 1], events=events, reader=reader,
            owned_pid_start_ticks=self.identities, all_joined=joined,
            returncode=None if self.process is None else self.process.returncode,
            postflight=postflight, component_pass=False, candidate_component_pass=candidate,
            status="cleanup_evidence_pending_final_return_guard",
            release=self.release, errors=self.errors, publication_refusals=self.publication_refusals,
            authority_poisoned=self.authority.poisoned,
            authority_invalidated_at=self.authority.invalidated_at, **c.permissions())
        c.raw_write(self.output / "result-after-drain.bin", bytes(self.received))
        c.write(self.output / "component.json", self.cleanup)
        c.require(joined and time.monotonic() < deadline, "cleanup_incomplete_or_late")
        returned_at = time.monotonic()
        c.require(returned_at < deadline, "cleanup_final_return_late")
        candidate = candidate and not self.authority.poisoned and not self.publication_refusals
        self.cleanup.update(status="complete", component_pass=candidate, cleanup_return_guard_at=returned_at)
        return self.cleanup
