"""Two-offer fixed32 protocol using unchanged source closure/cut contracts."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition_common01 as c
from transport01 import Capture
from coverage01 import assess as assess_inflight


class Wire:
    """Source worker callbacks wait on bounded async I/O, not blocking sockets."""
    def __init__(self, output, secret, domain, native, gateway, *, clock=time.monotonic, consumer=None):
        self.pre = c.preflight()
        self.api = self.pre.frozen.control()
        self.clock, self.native, self.gateway = clock, native, gateway
        self.consumer, self.raw_cuts = consumer, {}
        self.loop, self.pending = asyncio.get_running_loop(), set()
        self.pending_lock = threading.Lock()
        settings = dict(http_timeout_s=0.2, maximum_http_exchanges=1900, body_cap_bytes=1048576)
        self.capture = Capture(output, settings, secret, domain, clock=clock, api=self.api,
                               publisher=None if consumer is None else consumer.publish)

    async def call(self, target, method, path, *, role, payload=None, auth=True,
                   identifier=None, deadline, limit=0.2, started=None, before_send=None):
        return await self.capture.exchange(
            self.native if target == "native" else self.gateway, method, path,
            role=role, payload=payload, auth=auth, identifier=identifier,
            deadline=deadline, limit=limit, started=started, before_send=before_send)

    def sync(self, target, path, *, deadline, role):
        future = asyncio.run_coroutine_threadsafe(self.call(
            target, "GET", path, role=role, deadline=deadline), self.loop)
        with self.pending_lock:
            self.pending.add(future)
        try:
            return future.result(timeout=min(0.2, max(0.001, deadline - self.clock())))
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("owned_sync_HTTP_200ms") from None
        finally:
            with self.pending_lock:
                self.pending.discard(future)

    def response(self, record):
        import httpx
        self.api.body(record)
        return httpx.Response(record["http_status"],
                              content=base64.b64decode(record["response_body_base64"]),
                              request=httpx.Request(record["method"], "http://127.0.0.1" + record["path"]))

    def gateway_clients(self, deadline, receipts):
        wire = self

        class Sync:
            def get(self, path):
                c.require(path == "/admin/mpc/state", "only_source_gateway_GET")
                record = wire.sync("gateway", path, deadline=deadline, role="cut-gateway")
                receipts["snapshot"] = deepcopy(record)
                return wire.response(record)

        class Async:
            async def post(self, path, **kwargs):
                c.require(path == "/admin/mpc/flush-prefix", "only_source_gateway_flush")
                record = await wire.call(
                    "gateway", "POST", path, role="cut-flush", payload=kwargs["json"],
                    deadline=deadline)
                receipts["barrier"] = deepcopy(record)
                return wire.response(record)

        return Sync(), Async()

    async def close(self):
        with self.pending_lock:
            pending = tuple(self.pending)
        for future in pending:
            future.cancel()
        errors = []
        try:
            await self.capture.close()
        except BaseException as error:
            errors.append(dict(phase="capture", **c.failure(error, self.capture.secret)))
        clients = [asyncio.create_task(client.aclose()) for client in (self.native, self.gateway)]
        try:
            _, outstanding = await asyncio.wait(clients, timeout=0.2)
        except BaseException as error:
            errors.append(dict(phase="client_wait", **c.failure(error, self.capture.secret)))
            outstanding = {task for task in clients if not task.done()}
        for task in outstanding:
            task.cancel()
        if outstanding:
            _, outstanding = await asyncio.wait(outstanding, timeout=0.2)
            errors.append(dict(phase="client_close", error_type="client_close_deadline"))
        for task in clients:
            if task.done():
                try:
                    task.result()
                except BaseException as error:
                    errors.append(dict(phase="client", **c.failure(error, self.capture.secret)))
        joined = not outstanding and not self.capture.active
        c.write(self.capture.output / "client-transports-cleanup.json", dict(
            joined=joined, errors=errors, active_HTTP_tasks=len(self.capture.active),
            client_tasks_remaining=len(outstanding), source_callback_executor_join_is_separate=True,
            finished_at=self.clock()))
        c.require(joined and not errors, "owned_transport_cleanup_refused")


def persist_prefix(output, name, cut, telemetry):
    size = cut["owner_prefix"]["prefix_bytes"]
    c.require(type(size) is int and 0 <= size <= 33554432, "prefix_capacity")
    with Path(telemetry).open("rb") as handle:
        raw = handle.read(size)
    c.require(len(raw) == size and c.sha_bytes(raw) ==
              cut["owner_prefix"]["receipt"]["prefix_sha256"], "retained_prefix_byte_binding")
    path = output / (name + "-prefix.jsonl")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=path.name, bytes=size, sha256=c.sha(path))


async def collect_one(wire, startup, output, name, *, deadline, memory, process,
                      telemetry, closure=None):
    pre, api = wire.pre, wire.api
    receipts = {}
    gateway_sync, gateway_async = wire.gateway_clients(deadline, receipts)
    snapshot = api.body(startup["gateway_initial_record"])
    options = dict(
        native_process=process, native_sources=pre.frozen.sources(),
        expected_capture_id=c.load(c.PLAN)["trials"][0]["trial_id"],
        gateway_sync=gateway_sync, gateway_async=gateway_async,
        telemetry_path=telemetry, expected_run_id=snapshot["run_id"],
        expected_clock_domain=snapshot["clock_domain"], expected_identity=snapshot["identity"],
        initial_budget=32, public_memory=memory, max_age_s=0.5)
    journal = {}
    native_get = lambda: wire.sync(
        "native", "/pr33-native/phase-state", deadline=deadline, role=name + "-native")
    try:
        async with asyncio.timeout(max(0, deadline - wire.clock())):
            if closure is None:
                closure, journal = await c.collect_initial(
                    enabled=True, startup=startup, native_get=native_get, journal=journal,
                    deadline=deadline, clock=wire.clock, **options)
                cut = journal["cut"]
            else:
                def native_body():
                    record = native_get()
                    journal["native_record"] = deepcopy(record)
                    pre.receipt_body(record, method="GET", path="/pr33-native/phase-state",
                                     cutoff=wire.clock())
                    settings = dict(pre.frozen.settings(), final_queue_path="/pr33-native/phase-state")
                    frame, _ = api.full_queue_observation(
                        record, settings, native_run=startup["native_owned"]["readiness"]["frames"][0]["run_id"],
                        deadline=deadline)
                    phase = frame["native_phase"]
                    c.require(record["client_send_at"] <= phase["source_at"] <=
                              phase["sample_finished_at"] <= record["client_receive_at"],
                              "phase_receipt_source_bracket")
                    return frame
                cut = await pre.frozen.module("clocked_owner_cut").collect_clocked_owner_cut(
                    native_acquire=native_body, native_startup=startup["native_owned"],
                    gateway_startup=startup["gateway_owned"], initial_closure=closure,
                    clock=wire.clock, **options)
                journal.update(cut=cut, status="received")
            observation = pre.inspect_cut(cut, consume_at=wire.clock(), deadline=deadline)
            phases = pre.frozen.module("source_owner_phases").classify_source_owners(
                cut, initial_closure=closure, native_startup=startup["native_owned"],
                public_memory=memory, now=wire.clock(), expected_clock_domain=snapshot["clock_domain"])
            journal.update(status="source_cut_received", observation=observation,
                           phases=phases.evidence(), closure=asdict(closure),
                           prefix=persist_prefix(output, name, cut, telemetry))
            c.write(output / (name + ".json"), journal)
            prefix_raw = (output / journal["prefix"]["path"]).read_bytes()
            raw_bundle = dict(journal=deepcopy(journal),
                snapshot_base64=receipts["snapshot"]["response_body_base64"],
                barrier_base64=receipts["barrier"]["response_body_base64"],
                prefix_base64=base64.b64encode(prefix_raw).decode())
            c.require(api.sha(base64.b64decode(raw_bundle["snapshot_base64"], validate=True)) ==
                      cut["owner_prefix"]["receipt"]["snapshot_sha256"]
                      and api.sha(base64.b64decode(raw_bundle["barrier_base64"], validate=True)) ==
                      cut["owner_prefix"]["barrier_sha256"],
                      "actual_per_call_HTTP_bytes_not_reconstruction")
            c.write(output / (name + "-raw-binding.json"), dict(
                raw_bundle=raw_bundle, native_record=journal["native_record"]["sequence"],
                gateway_record=receipts["snapshot"]["sequence"], barrier_record=receipts["barrier"]["sequence"]))
            wire.raw_cuts[name] = raw_bundle
            # Receipt/serialization never advances the physical source time.
            pre.inspect_cut(cut, consume_at=wire.clock(), deadline=deadline)
            return closure, journal
    except BaseException as error:
        journal.update(status="refused", **c.failure(error, wire.capture.secret))
        path = output / (name + "-refused.json")
        if not path.exists():
            c.write(path, journal)
        raise


def ready_copy(started, readiness):
    return dict(started, native_ready=True, cleanup_completed=False, readiness=readiness)


async def startup(wire, directory, started, deadline, *, sleep=asyncio.sleep):
    api, records = wire.api, wire.capture.records
    status = dict(warmup_status="not_sent", startup_samples=[], complete=False)
    c.write(directory / "startup-planned.json", dict(warmup_offers=1, max_tokens=16))
    try:
        for target in ("native", "gateway"):
            for attempt in range(180):
                try:
                    record = await wire.call(
                        target, "GET", "/health" if target == "native" else "/admin/mpc/state",
                        role="startup-health-" + target, auth=target == "gateway", deadline=deadline)
                except TimeoutError:
                    c.require(wire.clock() < deadline, "startup_health_absolute_deadline")
                    await sleep(min(1, max(0, deadline - wire.clock())))
                    continue
                if record["http_status"] == 200 and record["body_complete"]:
                    if target == "gateway":
                        value = api.snapshot(record)
                        c.require(value.budget == 32 and value.budget_version == 0
                                  and value.sequence == 0 and not value.requests, "initial_gateway_not_fresh32")
                    break
                await sleep(min(1, max(0, deadline - wire.clock())))
            else:
                raise ValueError("startup_health_count_exhausted")
        models = await wire.call("native", "GET", "/v1/models", role="model", auth=False, deadline=deadline)
        c.require(c.MODEL_ID in {r["id"] for r in api.body(models)["data"]}, "native_model_alias")
        payload = dict(model=c.MODEL_ID, messages=[dict(role="user", content="Count from one to five.")],
                       temperature=0, max_tokens=16)
        publish(wire, "offer_intent", dict(phase="warmup_intent", payload=payload))
        status["warmup_status"] = "send_attempted"
        c.write(directory / "warmup-offered.json", dict(request=payload, send_attempted=True,
                                                      sent_intent_at=wire.clock()))
        warm = await wire.call("native", "POST", "/v1/chat/completions", role="one-warmup",
                               payload=payload, auth=False, deadline=deadline, limit=60)
        reply = api.body(warm)
        c.require(warm["http_status"] == 200 and type(reply["usage"]["completion_tokens"]) is int
                  and 1 <= reply["usage"]["completion_tokens"] <= 16, "warmup_response_cap")
        publish(wire, "offer_outcome", dict(phase="warmup_validated_response", record=warm))
        status["warmup_status"] = "received"
        empty, epoch = [], None
        for number in range(1800):
            if number:
                await sleep(0.1)
            record = await wire.call("native", "GET", "/pr33-native/queue-state",
                                      role="startup-queue", deadline=deadline)
            sample = dict(record=record["sequence"], empty=None)
            status["startup_samples"].append(sample)
            frame, vacant = api.full_queue_observation(
                record, wire.pre.frozen.settings(), native_run=epoch, deadline=deadline)
            epoch = frame["base"]["run_id"]
            sample["empty"] = vacant
            empty = empty + [record] if vacant else []
            if len(empty) == 2:
                break
        c.require(len(empty) == 2, "startup_full_stock_not_drained")
        initial = await wire.call("gateway", "GET", "/admin/mpc/state", role="initial32", deadline=deadline)
        before = api.snapshot(initial)
        c.require(before.budget == 32 and before.budget_version == before.sequence == 0
                  and not before.requests, "initial_gateway_not_fresh32")
        native = ready_copy(started["native"], dict(
            frames=[api.body(row)["base"] for row in empty], warmup_response=reply))
        publish(wire, "public_history_published", dict(phase="native_startup_readiness", readiness=native))
        c.write(directory.parent / "native/ready.json", native)
        status["complete"] = True
        return dict(native_owned=native, warmup_record=warm, drain_records=empty,
                    gateway_initial_record=initial)
    finally:
        status["finished_at"] = wire.clock()
        c.write(directory / "startup-result.json", status)


async def bootstrap(wire, initial, started, directory, deadline):
    api = wire.api
    snapshot = api.body(initial["gateway_initial_record"])
    command = dict(schema=snapshot["schema"], run_id=snapshot["run_id"],
                   clock_domain=snapshot["clock_domain"], identity=snapshot["identity"],
                   expected_version=0, budget=32, not_after=snapshot["sampled_at"] + 0.5,
                   operation_id=c.protocol()["readiness_operation_id"])
    publish(wire, "control_history_published", dict(phase="CAS_intent", command=command))
    c.write(directory / "CAS-planned.json", dict(command=command, count=1))
    ack = await wire.call("gateway", "POST", "/admin/mpc/budget", role="same-budget-CAS",
                          payload=command, deadline=deadline)
    confirmation = await wire.call("gateway", "GET", "/admin/mpc/state",
                                   role="confirm-version1", deadline=deadline)
    gateway = ready_copy(started["gateway"], dict(
        snapshot=snapshot, gateway_run_id=snapshot["run_id"], clock_domain=snapshot["clock_domain"],
        same_budget_ack=dict(status="applied", command=command, acknowledgment=api.body(ack))))
    publish(wire, "control_history_published", dict(
        phase="CAS_receipts", acknowledgment=ack, confirmation=confirmation))
    initial.update(gateway_owned=gateway, cas_command_record=ack, gateway_confirm_record=confirmation)
    # Validate genuine interval/receipt confirmation before issuing metadata or begin.
    before = api.snapshot(initial["gateway_initial_record"])
    interval = api.evaluate_step(
        dict(before_record=0, command_record=1, after_records=[2], budget=32,
             control="fresh", expected_http=200, operation_id=command["operation_id"]),
        [initial["gateway_initial_record"], ack, confirmation], before, wire.pre.frozen.settings())
    c.require(interval["actual_applied_at"] is None
              and interval["actual_application_before_deadline"] == "established_by_interval_upper"
              and api.snapshot(confirmation, binding=before).sequence == 1, "CAS_confirmation")
    c.write(directory / "CAS-confirmation.json", interval)
    c.write(directory.parent / "gateway/ready.json", gateway)
    meta = await wire.call("native", "POST", "/pr33-native/batch", role="capture-metadata",
                           payload={"operation": "metadata"}, deadline=deadline)
    publish(wire, "control_history_published", dict(phase="capture_metadata_received", record=meta))
    initial["capture_metadata_record"] = meta
    metadata = wire.pre.receipt_body(meta, method="POST", path="/pr33-native/batch", cutoff=wire.clock())
    expected_sdk = c.load(c.SOURCE / "experiments/prefill-vllm-v030-source-map.json")
    expected_bridge = c.sha(c.SOURCE / "experiments/mpc-research/native-v2/batch_capture_bridge.py")
    c.require(metadata.get("schema") == "native-v2-batch-capture-v1"
              and metadata.get("bridge_sha256") == expected_bridge
              and metadata.get("plan_sha256") == api.sha(api.canonical(c.load(c.PLAN)))
              and metadata.get("source_commit") == c.load(c.PLAN)["source_commit"]
              and metadata.get("clock_domain") == snapshot["clock_domain"]
              and isinstance(metadata.get("run_id"), str) and bool(metadata["run_id"])
              and meta["client_send_at"] <= metadata["command_started_at"] <= meta["client_receive_at"]
              and metadata["runtime"]["source"]["files"] == expected_sdk["sha256"]
              and metadata["runtime"]["source"]["version"] == expected_sdk["version"]
              and type(metadata["runtime"]["tensor_parallel_size"]) is int
              and type(metadata["runtime"]["pipeline_parallel_size"]) is int
              and metadata["runtime"]["tensor_parallel_size"] == metadata["runtime"]["pipeline_parallel_size"] == 1
              and type(metadata["runtime"]["process_id"]) is int and metadata["runtime"]["process_id"] > 0
              and len(metadata["runtime"]["workers"]) == len(metadata["worker_clocks"]) == 1
              and metadata["runtime"]["workers"][0]["source"] == metadata["runtime"]["source"]
              and metadata["worker_clocks"][0]["clock_domain"] == snapshot["clock_domain"]
              and metadata["worker_clocks"][0]["bridge_sha256"] == expected_bridge
              and type(metadata["worker_clocks"][0]["process_id"]) is int
              and metadata["worker_clocks"][0]["process_id"] > 0,
              "metadata_source_before_capture_begin")
    proc = wire.pre.frozen.module("process_clock_provenance").LinuxProc().read(metadata["runtime"]["process_id"])
    process = dict(pid=proc.pid, start_ticks=proc.start_ticks, pid_namespace=proc.pid_namespace)
    begin = await wire.call("native", "POST", "/pr33-native/batch", role="capture-begin",
                            payload=dict(operation="begin", trial_id=c.load(c.PLAN)["trials"][0]["trial_id"]),
                            deadline=deadline)
    publish(wire, "control_history_published", dict(phase="capture_begin_received", record=begin))
    initial["capture_begin_record"] = begin
    c.bootstrap_inputs(initial, cutoff=wire.clock())
    publish(wire, "bootstrap_confirmed", initial)
    c.write(directory / "bootstrap-inputs.json", initial)
    return process


def publish(wire, kind, evidence):
    if wire.consumer is not None:
        return wire.consumer.publish(kind, evidence)


def consumer_bundle(wire, initial, closure, memory, name):
    return dict(startup=deepcopy(initial), initial=deepcopy(wire.raw_cuts["initial-cut"]),
        current=deepcopy(wire.raw_cuts[name]), closure=asdict(closure),
        public_memory={rid:asdict(row) for rid,row in memory.items()})


async def final_drain(wire, directory, *, native_run, deadline, sleep=asyncio.sleep):
    result = dict(status="incomplete", deadline=deadline, samples=[], legacy_failure=None)
    try:
        legacy = await wire.call("native", "GET", "/pr33-native/state", role="legacy-final", deadline=deadline)
        frame = wire.api.native_state(legacy, wire.pre.frozen.settings(), native_run=native_run)
        result["legacy_record"] = legacy["sequence"]
        if not all(frame[key] == 0 for key in ("running", "waiting", "pending_batches", "frontend_unfinished")):
            result["legacy_failure"] = "legacy_final_native_not_drained_at_first_snapshot"
        consecutive = 0
        for number in range(9):
            if number:
                await sleep(0.25)
            row = await wire.call("native", "GET", "/pr33-native/queue-state",
                                   role="final-full-queue", deadline=deadline)
            item = dict(record=row["sequence"], empty=None)
            result["samples"].append(item)
            _, empty = wire.api.full_queue_observation(
                row, wire.pre.frozen.settings(), native_run=native_run, deadline=deadline)
            item["empty"] = empty
            consecutive = consecutive + 1 if empty else 0
            if consecutive == 2:
                result["status"] = "two_complete_empty_observations"
                return result
        raise ValueError("final_full_stock_not_drained")
    except BaseException as error:
        result.update(status="refused", **c.failure(error, wire.capture.secret))
        raise
    finally:
        result["finished_at"] = wire.clock()
        c.write(directory / "final-drain.json", result)


async def finish_capture(wire, directory, metadata, *, deadline):
    end = await wire.call("native", "POST", "/pr33-native/batch", role="capture-end",
                          payload={"operation": "end"}, deadline=deadline)
    document = wire.api.body(end)
    expected = wire.api.body(metadata)
    c.require(all(document.get(key) == expected[key] for key in
                  ("schema", "bridge_sha256", "plan_sha256", "source_commit", "run_id", "clock_domain"))
              and document.get("capture_valid") is True
              and end["client_send_at"] <= document["command_started_at"] <= end["client_receive_at"] <= deadline
              and document.get("validation_errors") == []
              and document.get("trial_id") == c.load(c.PLAN)["trials"][0]["trial_id"],
              "unknown_pending_or_invalid_capture_end")
    capture = document["capture"]
    raw = wire.api.canonical(capture)
    c.require(wire.api.sha(raw) == document["capture_sha256"]
              and wire.pre.frozen.module("batch_capture_bridge").capture_errors(capture, "light") == [],
              "capture_complete_source_binding")
    attempts = [row for row in wire.capture.records if row.get("role", "").startswith("managed-")
                and row["path"] == "/v1/chat/completions"]
    offered = {row["request_id"] for row in attempts}
    completed = set()
    for row in attempts:
        try:
            fate = wire.api.offer_fate(row, row["request_id"])
        except (ValueError, KeyError, TypeError):
            continue
        if fate in ("completed_stop", "completed_length"):
            completed.add(row["request_id"])
    identifiers = []
    for identifier in capture["requests"]:
        match = re.fullmatch(r"chatcmpl-([a-f0-9]{32})-[a-f0-9]{8}", identifier)
        c.require(match is not None, "capture_unknown_native_identifier")
        identifiers.append(match[1])
    c.require(len(identifiers) == len(set(identifiers)) and set(identifiers) <= offered
              and completed <= set(identifiers), "capture_offered_request_binding")
    path = directory / "retained-capture.json"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    c.require(c.sha(path) == document["capture_sha256"] and wire.clock() < deadline,
              "capture_retention_or_final_deadline")
    ack = await wire.call("native", "POST", "/pr33-native/batch", role="capture-ack",
                          payload=dict(operation="ack", trial_id=document["trial_id"],
                                       capture_sha256=document["capture_sha256"]), deadline=deadline)
    answer = wire.api.body(ack)
    c.require(answer.get("acknowledged") is True
              and ack["client_send_at"] <= answer["command_started_at"] <= ack["client_receive_at"] <= deadline
              and all(answer.get(key) == expected[key] for key in
                  ("schema", "bridge_sha256", "plan_sha256", "source_commit", "run_id", "clock_domain"))
              and wire.clock() <= deadline, "capture_ACK_source_or_deadline")
    return dict(capture_sha256=document["capture_sha256"], ack_record=ack["sequence"])


async def managed_offer(wire, directory, row, initial, closure, previous, *, deadline,
                        memory, process, telemetry, payload, sleep=asyncio.sleep):
    ordinal, identifier = row["ordinal"], row["request_id"]
    started = asyncio.get_running_loop().create_future()
    attempt = dict(schema="team16-inflight-cut-attempt-v1", request_id=identifier,
                   trigger_delay_s=c.protocol()["inflight"]["trigger_delay_s"],
                   status="transport_not_started", nonempty_overlap=False, cut=None,
                   pending_at_trigger=None, actual_application_at=None, native_mpc_permitted=False)
    task = asyncio.create_task(wire.call(
        "gateway", "POST", "/v1/chat/completions", role="managed-" + str(ordinal),
        payload=payload, auth=False, identifier=identifier, deadline=deadline, limit=20,
        started=started, before_send=lambda: wire.pre.inspect_cut(
            previous["cut"], consume_at=wire.clock(), deadline=deadline)),
        name="owned-managed-" + str(ordinal))
    journal, returned_at, transport_error, cut_error, response = None, None, None, None, None
    interrupted, assessment_error = True, None
    try:
        done, _ = await asyncio.wait((started, task), timeout=max(0, deadline - wire.clock()),
                                      return_when=asyncio.FIRST_COMPLETED)
        c.require(done, "managed_dispatch_absolute_deadline")
        if started.done() and not started.cancelled():
            dispatch = started.result()
            row.update(record=dispatch["sequence"], fate="send_attempted")
            attempt.update(dispatch=dispatch, scheduled_trigger_at=dispatch["client_send_at"] +
                           attempt["trigger_delay_s"])
            await sleep(max(0, attempt["scheduled_trigger_at"] - wire.clock()))
            attempt["trigger_at"] = wire.clock()
            c.require(attempt["trigger_at"] <= deadline, "inflight_trigger_after_absolute_deadline")
            pending = not task.done() and wire.capture.records[dispatch["sequence"]]["client_receive_at"] is None
            attempt["pending_at_trigger"] = pending
            lease = None
            if ordinal == 2:
                try:
                    c.require(wire.consumer is not None, "owned_consumer_required")
                    lease = wire.consumer.begin(ordinal=2, pending_at_trigger=pending)
                except Exception as error:
                    attempt["consumer"] = dict(status="unavailable_or_refused",
                                               **c.failure(error, wire.capture.secret))
            if pending:
                try:
                    name = "offer-" + str(ordinal) + "-inflight-cut"
                    cut_deadline = min(deadline, wire.clock() + 0.5)
                    closure, journal = await collect_one(wire, initial, directory, name,
                        deadline=cut_deadline, memory=memory,
                        process=process, telemetry=telemetry, closure=closure)
                    returned_at = wire.clock()
                    attempt.update(cut=name + ".json", cut_returned_at=returned_at,
                                   cut_file_sha256=c.sha(directory / (name + ".json")))
                    if ordinal == 2 and lease is not None:
                        try:
                            bundle = consumer_bundle(wire, initial, closure, memory, name)
                            outcome = await wire.consumer.handoff(bundle)
                            c.require(returned_at <= outcome["clocks"]["collector_returned_at"]
                                      <= outcome["clocks"]["handoff_return_guard_at"] < lease["deadline"],
                                      "composition_collector_to_handoff_chronology")
                            attempt["consumer"] = dict(
                                status=outcome["status"], source_collector_return_observed_at=returned_at,
                                consumer_handoff_entry_at=outcome["clocks"]["collector_returned_at"],
                                handoff_return_guard_at=outcome["clocks"]["handoff_return_guard_at"],
                                deadline=outcome["deadline"], generation=outcome["generation"],
                                scope="Different measured return/entry points;neither is a planner/application clock.")
                        except Exception as error:
                            attempt["consumer"] = dict(status="refused", **c.failure(error, wire.capture.secret))
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    cut_error = c.failure(error, wire.capture.secret)
                    attempt.update(status="refused_or_unknown", error=cut_error)
                    if ordinal == 2:
                        attempt["consumer"] = dict(status="refused", **cut_error)
            elif not pending:
                attempt["status"] = "early_complete_no_cut"
        try:
            response = await task
        except asyncio.CancelledError:
            raise
        except Exception as error:
            transport_error = c.failure(error, wire.capture.secret)
        interrupted = False
    finally:
        if not task.done():
            task.cancel()
        _, outstanding = await asyncio.wait((task,), timeout=0.2)
        if not started.done():
            started.cancel()
        task_error = None
        if task.done():
            try:
                task.result()
            except BaseException as error:
                task_error = type(error).__name__
        matching = [record for record in wire.capture.records if record["request_id"] == identifier]
        try:
            publish(wire, "offer_outcome", dict(phase="caller_outcome_classification",
                    request_id=identifier, records=[item["sequence"] for item in matching]))
        except Exception as error:
            failure = c.failure(error, wire.capture.secret)
            attempt.setdefault("publication_failures", []).append(failure)
            attempt["consumer"] = dict(status="refused", **failure)
        if matching:
            c.require(len(matching) == 1, "managed_receipt_duplicate")
            response = matching[0]
            row["record"] = response["sequence"]
            if not response["dispatch_attempted"]:
                row["fate"] = "refused_before_dispatch"
            else:
                try:
                    row["fate"] = wire.api.offer_fate(response, identifier)
                except (ValueError, KeyError, TypeError) as error:
                    row.update(fate="received_unclassifiable", outcome_error=c.failure(error, wire.capture.secret))
        else:
            refused = [record["sequence"] for record in wire.capture.refusals if record["request_id"] == identifier]
            row.update(fate="refused_before_reservation", reservation_refusals=refused)
        if interrupted or transport_error or response is None:
            attempt.update(status="refused_or_unknown", nonempty_overlap=False,
                           interrupted=interrupted, task_error_type=task_error)
        row["inflight"] = deepcopy(attempt)
        c.write(directory / ("offer-" + str(ordinal) + "-transport-join.json"), dict(
            joined=not outstanding, task_error_type=task_error, record=row["record"],
            finished_at=wire.clock(), source_callback_tasks_are_separately_owned=True))
        # An interrupted attempt still retains its trigger/refusal. A later complete
        # assessment is a distinct file, never a rewrite of this receipt.
        c.write(directory / ("offer-" + str(ordinal) + "-inflight-attempt.json"), attempt)
        if interrupted:
            c.write(directory / ("offer-" + str(ordinal) + "-inflight.json"), attempt)
        c.require(not outstanding, "managed_transport_not_joined")
    if journal is not None and response is not None:
        try:
            attempt.update(assess_inflight(journal, response, identifier,
                                          returned_at=returned_at, deadline=deadline))
        except Exception as error:
            assessment_error = c.failure(error, wire.capture.secret)
            attempt.update(status="refused_or_unknown", nonempty_overlap=False,
                           assessment_error=assessment_error)
    if transport_error:
        attempt.update(status="refused_or_unknown", transport_error=transport_error, nonempty_overlap=False)
    row["inflight"] = deepcopy(attempt)
    c.write(directory / ("offer-" + str(ordinal) + "-inflight.json"), attempt)
    if response is not None and response["http_status"] is not None:
        closure, previous = await collect_one(wire, initial, directory, "offer-" + str(ordinal) + "-cut",
            deadline=min(deadline, wire.clock() + 0.5), memory=memory, process=process,
            telemetry=telemetry, closure=closure)
        row["cut"] = "offer-" + str(ordinal) + "-cut.json"
    c.require(not cut_error and not transport_error and not assessment_error
              and row["fate"] in ("completed_stop", "completed_length"), "managed_offer_or_source_refused")
    return closure, previous


async def protocol_run(wire, directory, started, startup_deadline, tokenizer):
    rows = [dict(row, fate="not_sent", record=None, cut=None)
            for row in c.load(c.OFFERS)["rows"]]
    result = dict(schema="team16-cut-consumer-result-v1", offers=rows, failures=[],
                  native_mpc_permitted=False, actual_application_at=None, component_pass=False,
                  planner_entry_at=None, decision_at=None, qualified_model=False,
                  prediction_qualified=False, observer_deployable=False)
    initial, closure, active_at, began = None, None, None, False
    try:
        c.require(wire.consumer is not None, "owned_consumer_REQUIRED_for_composed_protocol")
        initial = await startup(wire, directory, started, startup_deadline)
        active_at = wire.consumer.activate()
        c.require(active_at <= startup_deadline, "startup_late")
        deadline = active_at + 20
        c.publish(directory.parent / "active-start.json", dict(
            started_at=active_at, deadline=deadline, clock_domain=wire.api.clock_domain()))
        process = await bootstrap(wire, initial, started, directory, deadline)
        began = True
        origins = c.load_origins(c.ORIGINS)
        c.require_origins(origins, native_pid=process["pid"], gateway_pid=started["gateway"]["pid"])
        c.write(directory / "running-origins-consumer01.json", origins)
        memory = {}
        telemetry = c.PUBLIC / "gateway/gateway-telemetry.jsonl"
        closure, journal = await collect_one(wire, initial, directory, "initial-cut",
            deadline=min(deadline, wire.clock() + 0.5), memory=memory, process=process, telemetry=telemetry)
        for row in rows:
            wire.pre.inspect_cut(journal["cut"], consume_at=wire.clock(), deadline=deadline)
            payload = deepcopy(c.load(c.OFFERS)["payload"])
            c.require(payload["messages"] == [dict(role="user", content=row["prompt"])]
                      and payload["max_tokens"] == row["max_tokens"] == 16, "frozen_managed_payload")
            count = c.estimate_prompt(payload, tokenizer)
            publish(wire, "public_history_published", dict(
                phase="declared_public_request", request_id=row["request_id"], prompt_tokens=count, max_tokens=16))
            memory[row["request_id"]] = wire.pre.frozen.module("live_owner_prefix").PublicRequest(count, 16)
            publish(wire, "offer_intent", dict(
                phase="managed_intent", ordinal=row["ordinal"], request_id=row["request_id"], prompt_tokens=count))
            row.update(send_intent_at=wire.clock(), prompt_tokens=count)
            c.write(directory / ("offer-" + str(row["ordinal"]) + "-intent.json"), deepcopy(row))
            wire.pre.inspect_cut(journal["cut"], consume_at=wire.clock(), deadline=deadline)
            closure, journal = await managed_offer(wire, directory, row, initial, closure, journal,
                deadline=deadline, memory=memory, process=process, telemetry=telemetry, payload=payload)
        result["initial_closure"] = asdict(closure)
    except BaseException as error:
        result["failures"].append(dict(phase="startup_or_active", **c.failure(error, wire.capture.secret)))
    finally:
        if active_at is not None:
            end = wire.clock()
            result["active_started_at"], result["active_finished_at"] = active_at, end
            final_deadline = min(active_at + 23, end + 3)
            if end > active_at + 20:
                result["failures"].append(dict(phase="active", error_type="late_active_completion"))
            try:
                if initial is not None:
                    await final_drain(wire, directory, native_run=wire.api.body(
                        initial["drain_records"][0])["base"]["run_id"], deadline=final_deadline)
                    if began:
                        result["capture_retention"] = await finish_capture(
                            wire, directory, initial["capture_metadata_record"], deadline=final_deadline)
            except BaseException as error:
                result["failures"].append(dict(phase="final", **c.failure(error, wire.capture.secret)))
        result["engineering_pass"] = bool(not result["failures"] and closure is not None
            and all(row["cut"] and row["fate"] in ("completed_stop", "completed_length") for row in rows)
            and result.get("capture_retention"))
        result["nonempty_overlap_count"] = sum(row.get("inflight", {}).get("nonempty_overlap") is True
                                               for row in rows)
        result["coverage_complete"] = result["nonempty_overlap_count"] == 2
        handoff = rows[1].get("inflight", {}).get("consumer", {})
        result["consumer_handoff_received"] = handoff.get("status") == "validated_pending_cleanup"
        result["consumer_publication_refusals"] = deepcopy(getattr(wire.consumer, "publication_refusals", []))
        result["component_pass"] = result["engineering_pass"] and result["consumer_handoff_received"] \
            and not result["consumer_publication_refusals"]
        result["component_pass_provisional_until_actual_consumer_returned_close"] = True
        result["coverage_scope"] = "Prior2of2 metric retained descriptively,not this consumer question's success gate;no retry/substitution."
        result["finished_at"] = wire.clock()
        c.write(directory / "result.json", result)
        c.publish(directory.parent / "protocol-finished.json", dict(
            finished_at=result["finished_at"], active_started_at=active_at,
            component_pass=result["component_pass"], clock_domain=wire.api.clock_domain()))
    return result


async def register_consumer(consumer, ready, directory, startup_deadline):
    registration = dict(
        schema="team16-cut-consumer-registration-v1", identity=ready["identity"]["process"],
        clock_domain=ready["identity"]["clock_domain"], run_id=ready["run_id"],
        parent=c.consumer_module("common01").current_identity()["process"],
        consumer_source_freeze_sha256=ready["source_freeze_sha256"])
    c.publish(directory / "consumer-registration01.json", registration)
    path = directory / "consumer-registered01.json"
    while time.monotonic() < startup_deadline:
        if path.exists():
            ack = c.load(path)
            identity = registration["identity"]
            c.require(ack["pid"] == identity["pid"] and ack["start_ticks"] == identity["start_ticks"]
                      and ack["client_pid"] == os.getpid() and ack["run_id"] == registration["run_id"]
                      and ack["clock_domain"] == registration["clock_domain"],
                      "host_consumer_registration_mismatch")
            return ack
        await asyncio.sleep(.01)
    raise TimeoutError("host_consumer_registration_startup_deadline")


def invocation_check(invocation):
    start = invocation["host_started_at"]
    c.require(type(start) in (float, int) and start <= time.monotonic() < invocation["startup_deadline"]
              and invocation["startup_deadline"] == start + 180
              and invocation["outer_deadline"] == start + 260
              and invocation["cleanup_hard_deadline"] == start + 243
              and invocation["clock_domain"] == c.domain()
              and invocation["consumer_manifest_sha256"] == c.CONSUMER_SEAL,
              "original_host_absolute_budgets_and_dependency_required")


def shared_teardown_anchor(directory, consumer, active_at=None):
    path = directory / "protocol-finished.json"
    host = directory / "host-cleanup-anchor01.json"
    if host.exists():
        marker = c.load(host)
        c.require(marker["clock_domain"] == c.domain(), "host_cleanup_clock_domain")
        anchor = marker["anchor"]
    elif path.exists():
        anchor = c.load(path)["finished_at"]
    else:
        anchor = time.monotonic()
    if not path.exists():
        c.publish(path, dict(finished_at=anchor, active_started_at=active_at, component_pass=False,
                            clock_domain=c.domain(), reason="client_startup_or_protocol_refusal"))
    c.require(anchor <= time.monotonic(), "future_shared_cleanup_anchor")
    return anchor


def make_consumer(output, invocation, secret):
    class HostBoundConsumer(c.consumer_module("consumer01").CutConsumer):
        async def close(self, *, cancelled=False, teardown_anchor=None, hard_deadline=None):
            anchor = shared_teardown_anchor(output.parent, self, self.active_at)
            c.require(teardown_anchor is None or teardown_anchor == anchor, "one_shared_cleanup_anchor")
            return await super().close(cancelled=cancelled, teardown_anchor=anchor, hard_deadline=min(
                invocation["cleanup_hard_deadline"],
                float("inf") if hard_deadline is None else hard_deadline))
    return HostBoundConsumer(
        output / "consumer", enabled=True, evidence_mode="owned-native", redaction_secret=secret,
        host_deadline=invocation["outer_deadline"], startup_deadline=invocation["startup_deadline"])


async def client_session(output, invocation, started, secret):
    boundary = make_consumer(output, invocation, secret)
    wire, result, failed = None, None, False
    try:
        ready = await boundary.start()
        await register_consumer(boundary, ready, output.parent, invocation["startup_deadline"])
        import httpx
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(c.ALIAS, local_files_only=True)
        wire = Wire(output, secret, c.domain(),
            httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=0.2, trust_env=False),
            httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=0.2, trust_env=False),
            consumer=boundary)
        result = await protocol_run(wire, output, started, invocation["startup_deadline"], tokenizer)
    except BaseException as error:
        failed = True
        c.write(output / "client-session-refused.json", c.failure(error, secret))
    finally:
        anchor = shared_teardown_anchor(output.parent, boundary, boundary.active_at)
        if wire is not None:
            try:
                await wire.close()
            except BaseException as error:
                failed = True
                c.write(output / "client-transport-close-refused.json", c.failure(error, secret))
        try:
            closed = await boundary.close(cancelled=failed, teardown_anchor=anchor,
                                          hard_deadline=invocation["cleanup_hard_deadline"])
            returned_at = time.monotonic()
            staging = c.STATE / "consumer-close-returned01.json"
            c.write(staging, dict(
                schema="team16-actual-consumer-close-return-v1", receipt=closed,
                actual_caller_observed_return_at=returned_at, shared_teardown_anchor=anchor,
                original_host_cleanup_cutoff=invocation["cleanup_hard_deadline"],
                clock_domain=c.domain(), host_IO_persistence_required=True))
            os.chmod(staging, 0o600)
        except BaseException as error:
            failed = True
            c.write(output / "consumer-close-refused.json", c.failure(error, secret))
            closed = None
    return dict(component_pass=bool(result and result["component_pass"] and not failed
                                   and closed and closed["component_pass"]),
                protocol=result, close_return_received=closed is not None,
                actual_application_at=None, native_mpc_permitted=False)


def bounded_loop(coroutine, directory, *, hard_deadline=None):
    """Process-owned executor; an unjoinable worker means refusal, never implicit wait."""
    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="owner-source")
    loop.set_default_executor(executor)
    asyncio.set_event_loop(loop)
    task = loop.create_task(coroutine)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(sig, task.cancel)
    outcome, failure = None, None
    try:
        outcome = loop.run_until_complete(task)
    except BaseException as error:
        failure = type(error).__name__
    finally:
        cutoffs = [float("inf") if hard_deadline is None else hard_deadline]
        for name, key in (("host-cleanup-anchor01.json", "anchor"),
                          ("protocol-finished.json", "finished_at")):
            path = directory.parent / name
            if path.exists():
                cutoffs.append(c.load(path)[key] + 40)
        cutoff = min(cutoffs)
        remaining = asyncio.all_tasks(loop)
        for pending in remaining:
            pending.cancel()
        if remaining:
            loop.run_until_complete(asyncio.wait(remaining, timeout=min(.2, max(0, cutoff - time.monotonic()))))
        executor.shutdown(wait=False, cancel_futures=True)
        until = min(time.monotonic() + 0.5, cutoff)
        for thread in tuple(executor._threads):
            thread.join(max(0, until - time.monotonic()))
        alive = [thread.name for thread in executor._threads if thread.is_alive()]
        still_pending = len(asyncio.all_tasks(loop))
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            loop.remove_signal_handler(sig)
        loop.close()
        asyncio.set_event_loop(None)
        c.write(directory / "client-cleanup.json", dict(
            worker_threads_remaining=alive, async_tasks_remaining=still_pending,
            error_type=failure, joined=not alive and not still_pending, finished_at=time.monotonic(),
            shared_teardown_deadline=None if cutoff == float("inf") else cutoff))
    return outcome, not alive and not still_pending and failure is None and time.monotonic() < cutoff


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--owned-client", action="store_true")
    args = parser.parse_args(argv)
    if not args.owned_client:
        print('{"status":"disabled","native_execution":false,"actual_application_at":null}')
        return 0
    raise ValueError("successor_CPU_patch_no_live_entrypoint")
    c.remote_only()
    sys.pycache_prefix = str(c.STATE / "python-cache-evidence")
    c.verify_code()
    os.chdir(c.PREFLIGHT)
    c.preflight().frozen.verify()
    config = c.private_config()
    invocation = c.load(c.STATE / "client-consumer-invocation01.json")
    invocation_check(invocation)
    c.owned_parent(invocation["owner_pid"], invocation["owner_start_ticks"])
    c.write(c.STATE / "client-consumer-single-entry01.json", dict(pid=os.getpid()))
    started = {name: c.load(c.PUBLIC / name / "started.json") for name in ("native", "gateway")}
    output = c.PUBLIC / "client"
    output.mkdir(mode=0o700)

    result, joined = bounded_loop(
        client_session(output, invocation, started, config["ADMIN_API_KEY"]), output,
        hard_deadline=invocation["cleanup_hard_deadline"])
    return 0 if joined and result and result["component_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
