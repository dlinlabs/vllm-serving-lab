"""Synthetic in-process composition controls; no real service or tokenizer.

The original five-request probe bytes/timing are unchanged. The tokenizer interface
and fingerprint hook are CPU fixtures, not asset/tokenizer verification. Real
runtime/planner/client APIs run against MockTransport; no natural metrics finalize.
"""

import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import unittest
from unittest import mock

import httpx

import nominal_episode as episode
import nominal_client as recorder
import nominal_runtime as runtime_api
import test_nominal_client as client_fixtures
from mpc_lab.contracts import GUARDED_CAPABILITY, SCHEMA, RequestMark, Snapshot, digest
from mpc_lab.runtime import snapshot_to_dict
from nominal_permit import NominalDemoPermit

CPU_SOURCE = digest({"scope": "episode02-CPU-only-no-live-authority"})
CPU_KEY = "CPU-only-episode-key-not-a-live-secret"


class OwnerFixture:
    """Only the published budget/check interface; no service, process, lock or thread."""

    def __init__(self, deadline, reserve=0.5):
        self.deadline, self.reserve, self.checks = deadline, reserve, []

    def check(self):
        now = time.monotonic()
        self.checks.append(now)
        if now >= self.deadline - self.reserve:
            raise RuntimeError("CPU owner original work deadline exhausted")


class TokenizerFixture(client_fixtures.FakeTokenizer):
    def __init__(self, trace):
        self.counts = {row["prompt"]: row["input_tokens"] for row in trace["requests"]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and return_dict
        return {"input_ids": [0] * self.counts[messages[0]["content"]]}


class TrackedTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.closed = False

    async def aclose(self):
        self.closed = True
        await super().aclose()


class GatewayFixture:
    def __init__(self, planner, trace, *, state_failure=False, models="ok", hold_s=0.35):
        self.planner, self.trace = planner, trace
        self.state_failure, self.models, self.hold_s = state_failure, models, hold_s
        self.run_id = "CPU-only-episode02-gateway"
        self.budget, self.version, self.sequence = 16.0, 0, 0
        self.active, self.gets, self.commands, self.requests = {}, [], [], []
        self.model_gets, self.closed_streams = 0, 0
        self.burst = asyncio.Event()

    async def handle(self, request):
        now = time.monotonic()
        path = request.url.path
        if path == recorder.MODELS_PATH:
            self.model_gets += 1
            if self.models == "failed":
                raise RuntimeError("CPU model preflight transport implementation failure")
            model = json.loads(json.dumps(client_fixtures.MODELS))
            if self.models == "refused":
                model["data"][0]["root"] = "/CPU/wrong-local-asset"
            return httpx.Response(200, json=model)
        if path.startswith("/admin/mpc/"):
            assert request.headers["x-admin-api-key"] == CPU_KEY
            if request.method == "GET":
                self.gets.append({"at": now, "active": len(self.active), "budget": self.budget,
                                  "budget_version": self.version, "sequence": self.sequence})
                if self.state_failure:
                    return httpx.Response(503, text="CPU observer refused")
                snapshot = Snapshot(
                    self.run_id, runtime_api.clock_domain(), now, self.budget, self.version,
                    tuple(self.active.values()), self.planner.identity, self.sequence,
                )
                document = snapshot_to_dict(snapshot)
                document.update(capabilities=[GUARDED_CAPABILITY],
                                cost_spec=asdict(self.planner.parameters.cost_spec))
                return httpx.Response(200, json=document)
            command = json.loads(request.content)
            assert command["expected_version"] == self.version and command["not_after"] > now
            assert command["run_id"] == self.run_id and command["clock_domain"] == runtime_api.clock_domain()
            previous = self.budget
            self.budget, self.version = command["budget"], self.version + 1
            self.commands.append({"at": now, "command": command})
            return httpx.Response(200, json={
                "schema": SCHEMA, "status": "applied", "operation_id": command["operation_id"],
                "run_id": self.run_id, "clock_domain": runtime_api.clock_domain(),
                "budget_version": self.version, "budget": self.budget, "previous_budget": previous,
                "applied_at": now,
            })
        assert request.method == "POST" and path == recorder.CHAT_PATH
        body = json.loads(request.content)
        self.requests.append({"at": now, "body": body})
        if len(self.requests) == 5:
            self.burst.set()
        identifier = "CPU-gateway-request-" + str(len(self.requests))
        cost = self.planner.parameters.cost_spec.estimate(3072, 128)
        if sum(mark.cost for mark in self.active.values()) + cost > self.budget:
            return httpx.Response(503, headers={"x-request-id": identifier}, json={
                "detail": recorder.ADMISSION_DETAIL, "estimated_request_cost": cost,
                "estimated_input_tokens": 3072, "estimated_output_tokens": 128,
                "max_admitted_cost": self.budget, "budget_version": self.version,
            })
        self.active[identifier] = RequestMark(identifier, cost, 3072, 128, now, now)
        self.sequence += 1

        async def stream():
            try:
                yield (client_fixtures.chunk("CPU") + "\n\n").encode()
                await asyncio.sleep(self.hold_s)
                yield (client_fixtures.chunk(" fixture", "stop") + "\n\n").encode()
                yield (client_fixtures.chunk(usage={"prompt_tokens": 3072, "completion_tokens": 2,
                                                   "total_tokens": 3074}, choices=False) + "\n\n").encode()
                yield (client_fixtures.DONE + "\n\n").encode()
            finally:
                self.active.pop(identifier, None)
                self.sequence += 1
                self.closed_streams += 1
        return httpx.Response(200, headers={"content-type": "text/event-stream", "x-request-id": identifier},
                              content=stream())


class NominalEpisodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        base = os.environ.get("NOMINAL_EPISODE_TEST_DIR")
        if not base:
            raise RuntimeError("NOMINAL_EPISODE_TEST_DIR must be a fresh project-local evidence directory")
        self.directory = Path(base) / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=False)
        self.trace = json.loads(episode.PROBE.read_bytes())
        self.clients, self.transports, self.tasks, self.gateways = [], [], [], []
        self.planner = runtime_api.NominalPlanner()
        self.signal_handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        for patch in (
            mock.patch.object(runtime_api, "source_digest", return_value=CPU_SOURCE),
            mock.patch.object(recorder.tw, "tokenizer_fingerprint",
                              return_value=self.trace["provenance"]["tokenizer_sha256"]),
            mock.patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                              side_effect=AssertionError("real HTTP forbidden in episode CPU controls")),
            mock.patch.object(recorder, "build_run", side_effect=AssertionError("no engineering finalization")),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def write(self, name, value):
        path = self.directory / name
        with path.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    def prepare(self, name, *, bad_binding=False):
        binding = client_fixtures.make_binding(
            tokenizer_sha256=self.trace["provenance"]["tokenizer_sha256"])
        if bad_binding:
            binding["served_alias"] = "CPU-wrong-alias"
        binding_path = self.directory / (name + "-binding.json")
        binding_path.write_text(json.dumps(binding))
        return episode.prepare_episode(
            output_dir=self.directory / name, binding_path=binding_path,
            gateway_url=client_fixtures.GATEWAY, backend_url=client_fixtures.BACKEND,
            tokenizer_loader=lambda path: TokenizerFixture(self.trace),
        )

    def gateway(self, **kwargs):
        value = GatewayFixture(self.planner, self.trace, **kwargs)
        self.gateways.append(value)
        return value

    def transport(self, gateway):
        value = TrackedTransport(gateway.handle)
        self.transports.append(value)
        return value

    def runtime(self, prepared, gateway, start, *, iterations=3, poll_s=0.6, episode_s=3.0):
        client = httpx.AsyncClient(
            base_url=client_fixtures.GATEWAY, transport=self.transport(gateway),
            follow_redirects=False, trust_env=False,
        )
        self.clients.append(client)
        value = runtime_api.NominalRuntime(
            client, self.planner, prepared.scenario_factory, prepared.directory / "runtime-journal.json",
            admin_key=CPU_KEY, scenario_binding=prepared.scenario_binding,
            enabled=True, start_at=start,
            limits=runtime_api.RuntimeLimits(iterations=iterations, episode_s=episode_s, poll_s=poll_s),
        )
        value.permit = NominalDemoPermit(
            self.planner.identity, gateway.run_id, runtime_api.clock_domain(), CPU_SOURCE,
            value.protocol_sha256, time.monotonic() - 0.1, time.monotonic() + 15.0, approved=True,
        )
        return value

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), 2)
        for client in self.clients:
            await client.aclose()
        for transport in self.transports:
            await transport.aclose()
        remaining = [task.get_name() for task in asyncio.all_tasks()
                     if task is not asyncio.current_task() and not task.done()]
        self.write("OWNED_CLEANUP.json", {
            "clients_closed": [client.is_closed for client in self.clients],
            "transports_closed": [transport.closed for transport in self.transports],
            "test_owned_tasks_joined": [task.done() for task in self.tasks],
            "remaining_loop_tasks": remaining,
            "active_CPU_stream_ledgers": [len(gateway.active) for gateway in self.gateways],
            "signal_handlers_unchanged": self.signal_handlers == (
                signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)),
            "processes_or_services_started": [], "not_a_Native_cleanup_receipt": True,
        })
        self.assertEqual(remaining, [])
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertTrue(all(not gateway.active for gateway in self.gateways))
        self.assertEqual(self.signal_handlers, (
            signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))

    async def test_shared_start_full_outcomes_and_real_controller_feedback(self):
        prepared = self.prepare("complete")
        self.assertIsNone(prepared.error)
        self.assertEqual(prepared.recorder.capture["binding_evidence"]["tokenizer"]["prompts_verified"], 5)
        start = time.monotonic() + 0.3
        owner = OwnerFixture(start + 6.0)
        gateway = self.gateway()
        runtime = self.runtime(prepared, gateway, start)
        result = await episode.run_episode(
            prepared, runtime, owner=owner, offered_start=start, enabled=True,
            transport=self.transport(gateway),
        )
        self.assertEqual(result["status"], "collected", result["failures"])
        self.assertTrue(result["all_workers_joined"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["missing_record_ids"], [])
        self.assertIsNone(result["admission_effect"])
        self.assertIsNone(result["full_wall"])
        self.assertIsNone(result["qualification"])
        capture = json.loads((prepared.directory / "capture.json").read_bytes())
        self.assertEqual(capture, prepared.recorder.capture)
        self.assertEqual(capture["offered_start"], start)
        self.assertEqual(capture["offered_end"], start + 5.0)
        self.assertEqual(capture["drain_deadline"], owner.deadline - owner.reserve)
        self.assertEqual(capture["config"]["caller_deadline"], owner.deadline)
        self.assertEqual(capture["config"]["cleanup_reserve_s"], owner.reserve)
        self.assertEqual(capture["trace_sha256"], episode.PROBE_SHA256)
        self.assertEqual(len(capture["records"]), 5)
        self.assertEqual([row["scheduled_at"] for row in capture["records"]], [start + 1.25] * 5)
        self.assertEqual(capture["rows"], [recorder.normalized_row(row) for row in capture["records"]])
        self.assertEqual(capture["summary"]["offered"], 5)
        self.assertEqual(capture["summary"]["sent"], 5)
        self.assertTrue(all(row["fate"] in ("complete", "rejected") for row in capture["records"]))
        self.assertEqual(len(gateway.requests), 5)
        self.assertTrue(all(row["at"] >= start + 1.25 for row in gateway.requests))
        self.assertGreaterEqual(gateway.gets[0]["at"], start)
        self.assertTrue(any(row["active"] > 0 for row in gateway.gets))
        value = result["workers"]["controller"]["result"]
        self.assertEqual(value["startup"]["requested_start_at"], start)
        self.assertEqual(len(value["runtime"]["steps"]), 3)
        self.assertIsNone(value["runtime"]["failure"])
        state = json.loads(runtime.store.path.read_bytes())
        self.assertEqual(len([row for row in state["events"] if row["event"] == "Planned"]), 3)
        self.assertTrue(all(row["suffix_execution_permitted"] is False for row in state["events"]
                            if row["event"] == "Planned"))
        self.write("COMPOSITION_EVIDENCE.json", {
            "status": result["status"], "original_deadline": owner.deadline, "reserve": owner.reserve,
            "offered_start": start, "runtime_GETs": gateway.gets, "CPU_budget_commands": gateway.commands,
            "requests": [{"at": row["at"]} for row in gateway.requests], "summary": capture["summary"],
            "scope": "CPU-only real API/planner execution; not tokenizer validation or Native admission proof",
        })

    async def test_default_off_preparation_and_configuration_refusals_do_not_start(self):
        receipts = []
        for kind in ("off", "ready_false", "past", "different_start", "truncated", "input_changed",
                     "runtime_endpoint_changed"):
            with self.subTest(kind=kind):
                prepared = self.prepare(kind, bad_binding=kind == "ready_false")
                start = time.monotonic() + (0.25 if kind != "past" else -0.2)
                gateway = self.gateway()
                runtime = self.runtime(prepared, gateway, start + (0.1 if kind == "different_start" else 0))
                owner = OwnerFixture(start + (4.0 if kind == "truncated" else 6.0))
                if kind == "input_changed":
                    prepared.recorder.capture["config"]["request_timeout_s"] = 123.0
                if kind == "runtime_endpoint_changed":
                    runtime.client.base_url = "http://127.0.0.1:8081"
                result = await episode.run_episode(
                    prepared, runtime, owner=owner, offered_start=start, enabled=kind != "off",
                    transport=self.transport(gateway),
                )
                self.assertEqual(result["status"], "refused", result)
                self.assertEqual(result["workers"], {})
                self.assertEqual((gateway.model_gets, len(gateway.gets), len(gateway.requests)), (0, 0, 0))
                self.assertEqual(json.loads((prepared.directory / "capture.json").read_bytes()),
                                 prepared.recorder.capture)
                self.assertFalse(runtime.store.path.exists())
                receipts.append({"kind": kind, "status": result["status"], "failures": result["failures"],
                                 "capture_status": prepared.recorder.capture["status"]})
        self.write("NO_START_REFUSALS.json", receipts)

    async def test_returned_failure_envelopes_cancel_peer_and_preserve_results(self):
        receipts = []
        for kind in ("controller_refused", "recorder_refused", "recorder_failed"):
            with self.subTest(kind=kind):
                prepared = self.prepare(kind)
                start = time.monotonic() + 0.25
                gateway = self.gateway(state_failure=kind == "controller_refused",
                                       models=kind.removeprefix("recorder_") if kind.startswith("recorder") else "ok")
                runtime = self.runtime(prepared, gateway, start, iterations=2, poll_s=0.01)
                result = await episode.run_episode(
                    prepared, runtime, owner=OwnerFixture(start + 6), offered_start=start, enabled=True,
                    transport=self.transport(gateway),
                )
                self.assertEqual(result["status"], "failed", result)
                self.assertTrue(result["all_workers_joined"])
                self.assertEqual(set(result["workers"]), {"recorder", "controller"})
                self.assertEqual(gateway.requests, [])
                if kind == "controller_refused":
                    value = result["workers"]["controller"]["result"]["runtime"]
                    self.assertIsNone(value["failure"])
                    self.assertEqual([row["status"] for row in value["steps"]], ["refused", "refused"])
                    self.assertEqual(prepared.recorder.capture["status"], "interrupted")
                    self.assertEqual([row["fate"] for row in prepared.recorder.capture["records"]],
                                     ["not_launched"] * 5)
                else:
                    value = result["workers"]["recorder"]
                    self.assertIsNone(value["error"])
                    self.assertEqual(value["result"]["status"], kind.removeprefix("recorder_"))
                    self.assertEqual(gateway.gets, [])
                self.assertEqual(json.loads((prepared.directory / "capture.json").read_bytes()),
                                 prepared.recorder.capture)
                receipts.append({"kind": kind, "status": result["status"], "failures": result["failures"],
                                 "cancel_requested": result["cancel_requested"]})
        self.write("RETURNED_FAILURES.json", receipts)

    async def test_caller_cancellation_joins_workers_and_persists_partial_outcomes(self):
        prepared = self.prepare("cancel")
        start = time.monotonic() + 0.3
        gateway = self.gateway(hold_s=20)
        runtime = self.runtime(prepared, gateway, start, iterations=6, poll_s=0.4, episode_s=4.5)
        task = asyncio.create_task(episode.run_episode(
            prepared, runtime, owner=OwnerFixture(start + 6), offered_start=start, enabled=True,
            transport=self.transport(gateway),
        ), name="CPU-episode-cancellation-control")
        self.tasks.append(task)
        await asyncio.wait_for(gateway.burst.wait(), 3)
        await asyncio.sleep(0.03)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        with self.assertRaises(asyncio.CancelledError):
            await task
        result = json.loads((prepared.directory / "episode.json").read_bytes())
        capture = json.loads((prepared.directory / "capture.json").read_bytes())
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["all_workers_joined"])
        self.assertEqual(result["missing_record_ids"], [])
        self.assertEqual(capture["status"], "interrupted")
        self.assertEqual(len(capture["records"]), 5)
        self.assertTrue(any(row["fate"] == "cancelled" for row in capture["records"]))
        self.assertEqual(capture["rows"], [recorder.normalized_row(row) for row in capture["records"]])
        self.assertEqual(len(gateway.requests), 5)
        self.assertEqual(gateway.active, {})
        self.assertTrue(runtime.store.path.is_file())
        with runtime.store.locked():
            self.assertIsNotNone(runtime.store.read())
        frozen = hashlib.sha256((prepared.directory / "capture.json").read_bytes()).hexdigest()
        await asyncio.sleep(0.05)
        self.assertEqual(frozen, hashlib.sha256((prepared.directory / "capture.json").read_bytes()).hexdigest())
        self.write("CANCELLATION_EVIDENCE.json", {
            "status": result["status"], "workers": result["workers"],
            "fates": [row["fate"] for row in capture["records"]], "all_workers_joined": True,
            "runtime_lock_reacquired": True, "service_or_Native_cleanup_claim": False,
        })
