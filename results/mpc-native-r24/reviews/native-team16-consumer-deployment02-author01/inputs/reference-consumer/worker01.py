"""Default-off, one read-only owned consumer. No Native/model/actuator entrypoint."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import selectors
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common01 as c
from generation01 import Reader
import ipc01 as ipc
from validate01 import prepare, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--read-fd", type=int)
    parser.add_argument("--write-fd", type=int)
    parser.add_argument("--authority-fd", type=int)
    parser.add_argument("--finish-fd", type=int)
    args = parser.parse_args()
    if not args.enabled:
        print(c.canonical(dict(status="disabled", **c.permissions())).decode())
        return 0
    c.require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,
              "isolated_source_bound_consumer")
    config = c.load(args.config)
    output = Path(config["worker_output"])
    os.chdir(c.PREFLIGHT)
    c.verify_code(literal_cwd=True)
    def audit(event, detail):
        if event in ("socket.connect", "socket.bind", "socket.getaddrinfo", "subprocess.Popen",
                     "os.fork", "os.posix_spawn", "os.system"):
            raise RuntimeError("consumer_forbidden_side_effect")
        if event == "import" and str(detail[0]).split(".")[0] in ("vllm", "torch", "triton", "paramiko"):
            raise RuntimeError("consumer_native_import_forbidden")
    sys.addaudithook(audit)
    preparation = prepare()
    own = c.current_identity()
    parent = config["authority"]["parent"]
    api = c.proc_api()
    proc = api.LinuxProc()
    actual_parent = proc.read(os.getppid())
    c.require(asdict(actual_parent) == parent["process"]
              and own["clock_domain"] == parent["clock_domain"]
              and own["process"]["pid"] == own["process"]["session"] == own["process"]["process_group"],
              "consumer_actual_parent_session_domain")
    # A separate open file description prevents shared-flock unlock races with parent readers.
    authority_fd = os.open(f"/proc/self/fd/{args.authority_fd}", os.O_RDONLY | os.O_CLOEXEC)
    os.close(args.authority_fd)
    authority = Reader(authority_fd, config["authority"])
    initial_authority = authority.read()
    c.require(initial_authority["lease"] == 0, "worker_ready_before_collection")
    for fd in (args.read_fd, args.write_fd, args.finish_fd):
        os.set_blocking(fd, False)
    c.require(ipc.read_sync(args.read_fd, 1, lambda: config["startup_deadline"], bytearray()) == b"R",
              "fresh_startup_challenge")
    ready = dict(schema="team16-consumer-ready-v1", identity=own,
                 run_id=config["authority"]["run_id"], source_freeze_sha256=c.sha(
                     (c.HERE / "CONTROL_FREEZE.json").read_bytes()),
                 ready_at=time.monotonic(), evidence_mode=config["evidence_mode"], preparation=preparation)
    serial_started = time.monotonic()
    raw_ready = c.canonical(ready)
    serial_finished = time.monotonic()
    at = time.monotonic()
    ipc.write_sync(args.write_fd, ipc.frame(raw_ready, sent=at, serial_started=serial_started, serial_finished=serial_finished),
                   config["startup_deadline"])
    retained = bytearray()
    clocks = dict(consumer_wait_started_at=time.monotonic())
    reply, status, result_sent = None, 2, False
    try:
        begin = ipc.read_sync(args.read_fd, 1, lambda: config["hard_deadline"], bytearray())
        c.require(begin == b"B", "one_consumer_BEGIN_required")
        lease = authority.read()
        c.require(lease["lease"] == 1, "authoritative_collection_lease_required")
        fixed_generation = lease["lease_generation"]
        c.require(lease["generation"] == fixed_generation, "caller_generation_changed_before_BEGIN")
        limit = lambda: min(lease["deadline"], authority.read()["deadline"])
        first = []
        header = ipc.parse_header(ipc.read_sync(args.read_fd, ipc.HEADER.size, limit, retained, first_read=first))
        clocks["consumer_recv_started_at"] = first[0]
        raw = ipc.read_sync(args.read_fd, header["size"], limit, retained)
        clocks["consumer_dequeue_at"] = time.monotonic()
        c.require(c.sha(raw) == header["sha256"], "IPC_input_body_hash")
        with selectors.DefaultSelector() as selector:
            selector.register(args.read_fd, selectors.EVENT_READ)
            remaining = limit() - time.monotonic()
            c.require(remaining > 0 and bool(selector.select(remaining)), "IPC_exact_one_frame_EOF_deadline")
            c.require(os.read(args.read_fd, 1) == b"", "IPC_duplicate_or_trailing_bytes")
        value = ipc.strict_json(raw)
        c.require(c.canonical(value) == raw, "canonical_immutable_input")
        c.raw_write(output / "consumed-input.bin", raw)
        c.require(value["generation"] == fixed_generation, "frame_generation_not_lease_start")
        c.require(value["boundary_started_at"] == lease["begin"]
                  and value["active_deadline"] == lease["active_deadline"]
                  and value["deadline"] == authority.read()["deadline"]
                  and value["collector_returned_at"] <= header["serialization_started_at"]
                  <= header["serialization_finished_at"] <= header["enqueue_send_started_at"]
                  <= clocks["consumer_recv_started_at"] <= clocks["consumer_dequeue_at"]
                  < value["deadline"], "consumer_whole_pipeline_chronology")
        clocks["consumer_validation_started_at"] = time.monotonic()
        before = authority.check(fixed_generation)
        result = validate(value, config=config, now=clocks["consumer_validation_started_at"])
        clocks["consumer_validation_finished_at"] = time.monotonic()
        after = authority.check(fixed_generation)
        c.require(clocks["consumer_validation_finished_at"] < value["deadline"]
                  and c.current_identity() == own
                  and asdict(proc.read(os.getppid())) == parent["process"], "consumer_late_or_identity_changed")
        reply = dict(schema="team16-consumer-result-v1", status="validated", run_id=ready["run_id"],
                     identity=own, input_sha256=c.sha(raw), header=header,
                     authority_reads=[before, after], clocks=dict(clocks), **result)
        reply["authority_reads"].append(authority.check(fixed_generation))
        serial_started = time.monotonic()
        response = c.canonical(reply)
        serial_finished = time.monotonic()
        sent = time.monotonic()
        c.require(sent < value["deadline"], "consumer_late_result")
        ipc.write_sync(args.write_fd, ipc.frame(
            response, sent=sent, serial_started=serial_started,
            serial_finished=serial_finished), value["deadline"])
        result_sent = True
        c.write(output / "send-receipt.json", dict(
            serialization_started_at=serial_started, serialization_finished_at=serial_finished,
            consumer_result_send_at=sent, consumer_result_send_finished_at=time.monotonic(),
            body_sha256=c.sha(response)))
        c.require(ipc.read_sync(args.finish_fd, 1, lambda: config["hard_deadline"], bytearray()) == b"F",
                  "parent_final_guard_not_released")
        status = 0
    except BaseException as error:
        reply = dict(schema="team16-consumer-result-v1", status="refused",
                     reason=str(error) if isinstance(error, (ValueError, TimeoutError, EOFError))
                     else type(error).__name__, error_type=type(error).__name__,
                     run_id=ready["run_id"], identity=own, clocks=clocks, **c.permissions())
        serial_started = time.monotonic()
        response = c.canonical(reply)
        serial_finished = time.monotonic()
        at = time.monotonic()
        if not result_sent:
            try:
                ipc.write_sync(args.write_fd, ipc.frame(response, sent=at, serial_started=serial_started,
                               serial_finished=serial_finished),
                               min(config["hard_deadline"], at + .2))
            except (BrokenPipeError, OSError, TimeoutError):
                pass
    finally:
        c.raw_write(output / "received-frame.bin", bytes(retained))
        c.write(output / "result.json", reply)
        try:
            proof = c.verify_code(literal_cwd=True)
            c.write(output / "postflight.json", dict(status="verified", source=proof,
                                                     finished_at=time.monotonic()))
        except BaseException as error:
            status = 2
            c.write(output / "postflight.json", dict(status="refused", error_type=type(error).__name__))
        for fd in (args.read_fd, args.write_fd, args.finish_fd, authority_fd):
            os.close(fd)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
