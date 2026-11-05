"""Explicitly synthetic CPU producer fixture; no socket, SDK, model or native process."""

import ast
import asyncio
import base64
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from typing import Optional

import frozen01 as frozen


BOOT = "11111111-1111-1111-1111-111111111111"
DOMAIN = "linux-monotonic:" + BOOT + ":time:[42]"
WARM = "chatcmpl-" + "f" * 32


class Clock:
    def __init__(self):
        self.value = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.value += 0.0001
            return self.value

    def advance(self, delta):
        with self.lock:
            self.value += delta


def wire(method, path, value, *, sent, received, payload=None):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return dict(
        method=method, path=path, http_status=200, client_clock_domain=DOMAIN,
        client_send_at=sent, client_headers_at=received, client_receive_at=received,
        request_body=None if payload is None else json.dumps(payload, sort_keys=True),
        response_body_base64=base64.b64encode(raw).decode(),
        response_body_sha256=hashlib.sha256(raw).hexdigest(),
        error_type=None, body_complete=True, credential_removed=False,
        auth_values_captured=False, response_headers={},
    )


def rewrite(record, value):
    replacement = wire(record["method"], record["path"], value,
                       sent=record["client_send_at"], received=record["client_receive_at"])
    for key in ("response_body_base64", "response_body_sha256"):
        record[key] = replacement[key]


class Fixture:
    def __init__(self, monkeypatch, directory):
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        self.httpx, self.directory = httpx, directory
        self.clock, self.records = Clock(), []
        self.phase = frozen.module("phase_observer")
        self.proc_api = frozen.module("process_clock_provenance")
        self.cut_api = frozen.module("clocked_owner_cut")
        self.prefix_api = frozen.module("gateway_prefix_bridge")
        self.contracts = frozen.module("mpc_lab.contracts")
        self.runtime = frozen.module("mpc_lab.runtime")
        self.capture_api = frozen.module("prefill_lab.runtime")
        telemetry = frozen.module("telemetry")
        phase = self.phase
        self.native_process = dict(pid=111, start_ticks=30, pid_namespace="pid:[7]")
        self.gateway_process = dict(pid=200, start_ticks=40, pid_namespace="pid:[7]")
        process = self.proc_api.Process
        self.proc = SimpleNamespace(
            self_pid=900, boot_id=lambda: BOOT,
            rows={
                900: process(900, 1, 900, 900, 5, "time:[42]", "pid:[7]"),
                100: process(100, 900, 100, 100, 10, "time:[42]", "pid:[7]"),
                110: process(110, 100, 100, 100, 20, "time:[42]", "pid:[7]"),
                111: process(111, 110, 100, 100, 30, "time:[42]", "pid:[7]"),
                200: process(200, 900, 200, 200, 40, "time:[42]", "pid:[7]"),
            })
        self.proc.read = lambda pid: self.proc.rows[pid]
        original_proof = self.proc_api.prove_owned_process_clock
        monkeypatch.setattr(self.cut_api, "prove_owned_process_clock",
                            lambda *args, **kwargs: original_proof(*args, proc=self.proc, **kwargs))
        monkeypatch.setattr(self.proc_api, "self_process_metadata", lambda: dict(self.native_process))
        monkeypatch.setattr(self.prefix_api, "self_process_metadata", lambda: dict(self.gateway_process))
        times = SimpleNamespace(monotonic=self.clock, time=lambda: 1000 + self.clock())
        for module in (phase, phase.input_post_observer, self.capture_api,
                       self.prefix_api, telemetry):
            monkeypatch.setattr(module, "time", times)
        self.input_queue = queue.Queue()
        self.ledger = phase.LiveInputPostLedger(self.input_queue, "ADD")
        self.core = SimpleNamespace(
            scheduler=SimpleNamespace(running=[], waiting=[], requests={}),
            _pr33_r23_input_post_ledger=self.ledger,
            _pr33_r23_input_post_core_sha256=phase.input_post_observer.CORE_SHA256,
        )
        self.routes, self.ghosts = [], []
        monkeypatch.setattr(phase.queue_observer, "queue_snapshot", lambda _: self.queue_frame())
        self.log = telemetry.Telemetry()
        self.log.enabled = self.log.owner_ledger_enabled = True
        self.log.run_id = "synthetic-gateway"
        spec = self.contracts.CostSpec(intercept=16, input_per_token=0, output_per_token=0)
        identity = self.contracts.Identity(
            "synthetic", "unqualified", "unqualified", spec.fingerprint,
            self.contracts.implementation_hash())
        namespace = dict(
            __file__=str(frozen.SOURCE / "gateway.py"), asyncio=asyncio, math=math,
            asdict=asdict, dataclass=dataclass, Optional=Optional, Request=Request,
            RequestMark=self.contracts.RequestMark, Snapshot=self.contracts.Snapshot,
            replace=replace, JSONResponse=JSONResponse, telemetry=self.log,
            state=defaultdict(float, admission_budget=32, budget_version=0),
            state_lock=asyncio.Lock(), mpc_requests={}, mpc_operations={}, MPC_OPERATION_LIMIT=16,
            mpc_now=self.clock, mpc_sequence=0, MPC_LAB_ENABLED=True,
            MPC_OWNER_LEDGER_ENABLED=True, ADMISSION_MODE="cost_aware",
            mpc_identity=identity, mpc_clock_domain=DOMAIN, mpc_cost_spec=spec,
            mpc_access=lambda _: None, snapshot_to_dict=self.runtime.snapshot_to_dict,
            GUARDED_CAPABILITY=self.contracts.GUARDED_CAPABILITY,
            SCHEMA=self.contracts.SCHEMA, digest=self.contracts.digest,
            TELEMETRY_JSONL=str(directory / "gateway.jsonl"),
        )
        tree = ast.parse((frozen.SOURCE / "gateway.py").read_text())
        selected = {"AdmissionDecision", "begin_request", "finish_request",
                    "mpc_changed", "mpc_first_content", "mpc_snapshot", "mpc_update_budget"}
        tree.body = [node for node in tree.body if getattr(node, "name", None) in selected]
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                node.decorator_list = []
        exec(compile(tree, str(frozen.SOURCE / "gateway.py"), "exec"), namespace)
        self.gateway = namespace
        self.app = FastAPI()
        self.app.add_api_route("/admin/mpc/state", namespace["mpc_snapshot"], methods=["GET"])
        self.prefix_api.install_gateway_prefix_bridge(self.app, enabled=True)
        self.mutate_native = None
        self.public_memory = {}
        self.journal = {}
        self.plan = json.loads((frozen.HERE / "PROPOSED_NATIVE_PLAN.json").read_bytes())

    def queue_frame(self):
        running = [r.request_id for r in self.core.scheduler.running]
        waiting = [r.request_id for r in self.core.scheduler.waiting]
        started = self.clock()
        frame = dict(
            schema="r23-native-queue-read-only-v1",
            queue_observer_sha256=frozen.sources()["queue_observer_sha256"],
            sample_started_at=started, complete=True, refusal_reasons=[],
            scheduler_state=dict(running=running, waiting=waiting,
                                 requests={rid: {} for rid in self.core.scheduler.requests}),
            queue_oldest_first=deepcopy(self.routes), queued_ghost_request_ids=list(self.ghosts),
            skipped_waiting=[], streaming_waiting=0,
            base=dict(
                schema="pr33-native-read-only-v1",
                observer_sha256="77633dc7652066e45b7680d297d3e36ddb97a3e12222de9f526a76b9a01d66b4",
                run_id="synthetic-native", clock_domain=DOMAIN, sampled_at=self.clock(),
                complete=True, running=len(running), waiting=len(waiting),
                pending_batches=len(self.routes), frontend_unfinished=len(running) + len(waiting),
                requests=[dict(request_id=rid) for rid in running],
                orphaned=0 if not running and not waiting and not self.routes else None),
        )
        frame["sample_finished_at"] = self.clock()
        return frame

    def native_record(self, phase=False):
        sent = self.clock()
        frame = self.phase.phase_snapshot(self.core) if phase else self.queue_frame()
        frame["base"]["frontend_sampled_at"] = self.clock()
        if self.mutate_native and phase:
            self.mutate_native(frame)
        result = wire("GET", "/pr33-native/phase-state" if phase else "/pr33-native/queue-state",
                      frame, sent=sent, received=self.clock())
        self.records.append(deepcopy(result))
        return result

    async def gateway_record(self):
        sent = self.clock()
        body = await self.gateway["mpc_snapshot"](None)
        return wire("GET", "/admin/mpc/state", body, sent=sent, received=self.clock())

    async def initialize(self):
        sent = self.clock()
        self.ledger.put(("ADD", (SimpleNamespace(request_id=WARM + "-11111111"), 0)))
        self.input_queue.get_nowait()
        warm = dict(id=WARM, choices=[dict(finish_reason="stop")],
                    usage=dict(completion_tokens=1))
        warm_record = wire("POST", "/v1/chat/completions", warm, sent=sent,
                           received=self.clock(), payload={"max_tokens": 16})
        drain = [self.native_record(), self.native_record()]
        initial = await self.gateway_record()
        body = frozen.control().body(initial)
        request = dict(
            schema=self.contracts.SCHEMA, run_id=body["run_id"], clock_domain=DOMAIN,
            identity=body["identity"], expected_version=0,
            not_after=body["sampled_at"] + 0.5, operation_id="synthetic-bootstrap32", budget=32)

        class Request:
            async def json(self):
                return request

        sent = self.clock()
        ack = await self.gateway["mpc_update_budget"](Request())
        command = wire("POST", "/admin/mpc/budget", ack, sent=sent,
                       received=self.clock(), payload=request)
        confirm = await self.gateway_record()
        sdk = json.loads((frozen.SOURCE / "experiments/prefill-vllm-v030-source-map.json").read_bytes())
        source = dict(version=sdk["version"], files=sdk["sha256"], upstream=sdk["upstream"])
        bridge = frozen.module("batch_capture_bridge")
        common = dict(
            schema="native-v2-batch-capture-v1", bridge_sha256=bridge.source_hash(),
            plan_sha256=frozen.control().sha(frozen.control().canonical(self.plan)),
            source_commit=self.plan["source_commit"], run_id="synthetic-batch-capture",
            clock_domain=DOMAIN,
        )
        sent = self.clock()
        metadata = wire("POST", "/pr33-native/batch", dict(
            **common, command_started_at=self.clock(),
            runtime=dict(source=source, tensor_parallel_size=1, pipeline_parallel_size=1,
                         process_id=111, workers=[dict(source=deepcopy(source))]),
            worker_clocks=[dict(clock_domain=DOMAIN, process_id=111,
                                bridge_sha256=bridge.source_hash())]),
            sent=sent, received=self.clock(), payload={"operation": "metadata"})
        sent = self.clock()
        self.capture = self.capture_api.ExternalCapture(
            self.core, self.plan["trials"][0]["trial_id"], "light")
        self.core._prefill_trial = self.capture
        begin = wire("POST", "/pr33-native/batch", dict(
            **common, command_started_at=self.clock(), trial_id=self.capture.id,
            begin={"external": True}), sent=sent, received=self.clock(),
            payload={"operation": "begin", "trial_id": self.capture.id})
        native_owned = dict(
            pid=100, process_group=100, start_ticks="10", native_ready=True,
            cleanup_completed=False, started_at=99,
            readiness=dict(frames=[frozen.control().body(r)["base"] for r in drain],
                           warmup_response=warm))
        gateway_owned = dict(
            pid=200, process_group=200, start_ticks="40", native_ready=True,
            cleanup_completed=False, started_at=99,
            readiness=dict(snapshot=body, gateway_run_id=body["run_id"], clock_domain=DOMAIN,
                           same_budget_ack=dict(status="applied", command=request, acknowledgment=ack)))
        self.startup = dict(
            native_owned=native_owned, gateway_owned=gateway_owned,
            warmup_record=warm_record, drain_records=drain,
            gateway_initial_record=initial, cas_command_record=command,
            gateway_confirm_record=confirm, capture_metadata_record=metadata,
            capture_begin_record=begin,
        )
        self.deadline = self.clock() + 0.5
        return self

    def options(self):
        fixture = self

        class SyncGateway:
            def get(self, path):
                assert path == "/admin/mpc/state"
                raw = asyncio.run(fixture.gateway["mpc_snapshot"](None))
                return fixture.httpx.Response(200, json=raw,
                    request=fixture.httpx.Request("GET", "http://synthetic" + path))

        class BarrierGateway:
            async def post(self, path, **kwargs):
                async with fixture.httpx.AsyncClient(
                    transport=fixture.httpx.ASGITransport(app=fixture.app),
                    base_url="http://synthetic") as client:
                    return await client.post(path, **kwargs)

        return dict(
            native_process=self.native_process, native_sources=frozen.sources(),
            expected_capture_id=self.capture.id,
            gateway_sync=SyncGateway(), gateway_async=BarrierGateway(),
            telemetry_path=self.directory / "gateway.jsonl",
            expected_run_id=self.log.run_id, expected_clock_domain=DOMAIN,
            expected_identity=asdict(self.gateway["mpc_identity"]),
            initial_budget=32, public_memory=self.public_memory, max_age_s=0.5)

    async def initial(self):
        from preflight01 import collect_initial
        return await collect_initial(
            enabled=True, startup=self.startup, native_get=lambda: self.native_record(True),
            journal=self.journal, deadline=self.deadline, clock=self.clock, **self.options())

    async def later(self, closure):
        return await self.cut_api.collect_clocked_owner_cut(
            native_acquire=lambda: frozen.control().body(self.native_record(True)),
            native_startup=self.startup["native_owned"],
            gateway_startup=self.startup["gateway_owned"], initial_closure=closure,
            clock=self.clock, **self.options())

    async def issue_then_release_with_physical_stock(self):
        rid = "a" * 32
        nid = "chatcmpl-" + rid + "-22222222"
        decision = await self.gateway["begin_request"](
            16, False, mpc_mark=(rid, self.clock(), 5, 16))
        assert decision.accepted
        self.public_memory[rid] = frozen.module("live_owner_prefix").PublicRequest(5, 16)
        request = SimpleNamespace(
            request_id=nid, num_prompt_tokens=5, max_tokens=16,
            sampling_params=SimpleNamespace(extra_args={}), prompt_token_ids=[1, 2, 3, 4, 5])
        self.ledger.put(("ADD", (request, 0)))
        published = self.input_queue.get_nowait()
        assert self.capture.register_request(published[1][0])
        output = SimpleNamespace(
            num_scheduled_tokens={nid: 1}, total_num_scheduled_tokens=1,
            scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(
                req_ids=[nid], num_computed_tokens=[5]))
        key = frozen.module("prefill_lab.contracts").step_key(output)
        self.capture.inflight[key] = dict(start=self.clock(), end=None)
        self.capture.after_output(output, {0: SimpleNamespace(outputs=[
            SimpleNamespace(request_id=nid, new_token_ids=[6], finish_reason="stop")])})
        await self.gateway["finish_request"](16, request_id=rid)
        self.routes = [
            dict(step_id="a" * 64, work=[dict(
                request_id=nid, position=6, tokens=1, prompt_length=5)]),
            dict(step_id="b" * 64, work=[]),
        ]
        self.ghosts = [nid]
        return rid, nid

    def preserve(self, name, value):
        with (self.directory / name).open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
