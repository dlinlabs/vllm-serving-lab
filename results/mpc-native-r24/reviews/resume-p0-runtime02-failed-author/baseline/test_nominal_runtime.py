"""CPU MockTransport controls; only test_real_planner_feedback uses real forecasts.

Other controls use explicit literal typed decisions to isolate transport/journal
failure boundaries. In-memory CPU permits use a non-live source digest; no
operator approval file is produced. All artifacts stay in the supplied directory.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
from unittest import mock

import httpx

from mpc_lab.contracts import (
    GUARDED_CAPABILITY, SCHEMA, FutureOffer, ModelParameters, RequestMark,
    Scenario, Snapshot, digest,
)
from mpc_lab.runtime import snapshot_to_dict
import nominal_planner as p
import nominal_runtime as r
from nominal_permit import NominalDemoPermit


CPU_SOURCE = digest({"not_live_authority": "runtime-CPU-MockTransport-only"})
CPU_KEY = "CPU-only-admin-fixture-not-a-live-secret"


class Clock:
    def __init__(self):
        self.value = 100.0
        self.sleeps = []

    def __call__(self):
        return self.value

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds
        await asyncio.sleep(0)


class Gateway:
    """A literal CAS/expiry transport fixture, not a Native server or benchmark."""

    def __init__(self, planner, clock):
        self.planner, self.clock = planner, clock
        self.run_id = "CPU-only-nominal-runtime"
        self.domain = r.clock_domain()
        self.budget, self.version, self.sequence = 16.0, 0, 0
        self.requests = ()
        self.gets, self.posts, self.acks, self.receipts = 0, [], [], []
        self.mutate_get = None
        self.post_mode = "apply"
        self.post_started = asyncio.Event()
        self.post_block = asyncio.Event()
        self.cancelled_posts = 0
        self.feedback = False
        self.failed_reobservation = False
        self.journal = None

    def mark(self, suffix="observed"):
        now = self.clock()
        return RequestMark(
            "CPU-" + suffix, self.planner.parameters.cost_spec.estimate(64, 4000),
            64, 4000, now, now,
        )

    def snapshot(self):
        return Snapshot(
            self.run_id, self.domain, self.clock(), self.budget, self.version,
            self.requests, self.planner.identity, self.sequence,
        )

    async def handle(self, request):
        assert request.headers["x-admin-api-key"] == CPU_KEY
        assert request.url.host == "cpu.invalid"
        if request.method == "GET":
            assert request.url.path == "/admin/mpc/state"
            self.gets += 1
            if self.failed_reobservation and self.acks:
                return httpx.Response(503, content=b"CPU explicit reobservation failure")
            document = snapshot_to_dict(self.snapshot())
            document.update(capabilities=[GUARDED_CAPABILITY],
                            cost_spec=asdict(self.planner.parameters.cost_spec))
            if self.mutate_get:
                changed = self.mutate_get(self.gets, document)
                if isinstance(changed, bytes):
                    return httpx.Response(200, content=changed)
                if changed is not None:
                    document = changed
            self.receipts.append({"method": "GET", "at": self.clock(), "document": document})
            return httpx.Response(200, json=document)
        assert request.method == "POST" and request.url.path == "/admin/mpc/budget"
        command = json.loads(request.content)
        durable = json.loads(self.journal.read_bytes())
        prepared = durable["events"][-1]
        assert prepared["event"] == "PendingPrepared"
        assert durable["pending"]["command"] == command
        assert durable["pending"]["ack"] is None
        assert durable["issued_commands"] == durable["pending"]["ordinal"] == len(self.posts) + 1
        assert prepared["exact_POST_body_sha256"] == hashlib.sha256(request.content).hexdigest()
        assert not self.journal.with_name(self.journal.name + ".partial").exists()
        assert prepared["not_before"] <= self.clock() < command["not_after"]
        assert command["not_after"] <= prepared["deadline"] <= prepared["entered_at"] + 0.5
        assert command["run_id"] == self.run_id and command["clock_domain"] == self.domain
        assert command["identity"] == asdict(self.planner.identity)
        assert command["expected_version"] == self.version
        self.posts.append(command)
        self.receipts.append({"method": "POST", "at": self.clock(), "command": command,
                              "durable_ordinal_before_send": durable["issued_commands"]})
        self.post_started.set()
        if self.post_mode == "block":
            try:
                await self.post_block.wait()
            except asyncio.CancelledError:
                self.cancelled_posts += 1
                raise
        if self.post_mode == "timeout":
            raise httpx.ReadTimeout("CPU unknown write " + CPU_KEY, request=request)
        previous = self.budget
        self.budget = command["budget"]
        self.version += 1
        self.sequence += 1
        if self.feedback and self.budget == 32.0 and not self.requests:
            self.requests = (self.mark("admitted-after-cap"),)
            self.sequence += 1
        ack = {
            "schema": SCHEMA, "status": "applied", "run_id": self.run_id,
            "clock_domain": self.domain, "operation_id": command["operation_id"],
            "previous_budget": previous, "budget": self.budget,
            "budget_version": self.version, "applied_at": self.clock(),
        }
        self.acks.append(ack)
        if self.post_mode == "malformed_ack":
            return httpx.Response(200, json={**ack, "operation_id": "CPU-wrong-operation"})
        return httpx.Response(200, json=ack)


class NominalRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        base = os.environ.get("NOMINAL_RUNTIME_TEST_DIR")
        if not base:
            raise RuntimeError("NOMINAL_RUNTIME_TEST_DIR must name a fresh project-local evidence directory")
        self.directory = Path(base) / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=False)
        self.clients, self.tasks, self.results, self.gateways = [], [], [], []
        self.planner = p.NominalPlanner()
        self.source_patch = mock.patch.object(r, "source_digest", return_value=CPU_SOURCE)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.network_patch = mock.patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request",
            side_effect=AssertionError("external HTTP transport forbidden in CPU controls"),
        )
        self.network_patch.start()
        self.addCleanup(self.network_patch.stop)

    def write(self, name, value):
        with (self.directory / name).open("x", encoding="utf-8") as stream:
            json.dump(r._plain(value), stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), 2)
        for client in self.clients:
            await client.aclose()
        self.write("OWNED_CLEANUP.json", {
            "scope": "CPU test AsyncClients and explicit asyncio tasks only",
            "runtime_children_started": [], "owned_helper_processes": [],
            "clients_closed": [client.is_closed for client in self.clients],
            "tasks_joined": [task.done() for task in self.tasks],
            "engine_or_campaign_cleanup_claim": False,
        })
        self.write("TRANSPORT_RECEIPTS.json", [gateway.receipts for gateway in self.gateways])
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertTrue(all(task.done() for task in self.tasks))

    def gateway(self, clock):
        gateway = Gateway(self.planner, clock)
        self.gateways.append(gateway)
        return gateway

    def factory(self, snapshot):
        return p.CausalScenario(
            Scenario(()), p.StateVersion.from_snapshot(snapshot),
            digest({"CPU-literal-causal-provider": True}), snapshot.sampled_at,
            snapshot.sampled_at, "development", "engineering",
        )

    def runtime(self, gateway, *, name="journal.json", enabled=True, permit=True,
                limits=None, policy="mpc", factory=None, max_commands=8):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(gateway.handle), base_url="http://cpu.invalid",
            follow_redirects=False, trust_env=False,
        )
        self.clients.append(client)
        sleeper = gateway.clock.sleep if isinstance(gateway.clock, Clock) else asyncio.sleep
        runtime = r.NominalRuntime(
            client, self.planner, self.factory if factory is None else factory,
            self.directory / name, admin_key=CPU_KEY,
            scenario_binding={"CPU_only": True, "provider": "literal-transport-or-declared-real-model-control"},
            enabled=enabled, policy=policy,
            limits=limits or r.RuntimeLimits(iterations=1, episode_s=3.0),
            clock=gateway.clock, sleeper=sleeper,
        )
        gateway.journal = runtime.store.path
        if permit:
            runtime.permit = NominalDemoPermit(
                self.planner.identity, gateway.run_id, gateway.domain,
                CPU_SOURCE, runtime.protocol_sha256, gateway.clock(),
                gateway.clock() + 10, max_commands=max_commands, approved=True,
            )
        return runtime

    async def execute(self, runtime):
        result = await runtime.run()
        self.results.append(result)
        self.write(f"result-{len(self.results):02d}.json", result)
        return result

    def events(self, runtime, name=None):
        events = json.loads(runtime.store.path.read_bytes())["events"]
        return [event for event in events if name is None or event["event"] == name]

    @contextmanager
    def literal(self, *, hold=False, advance=0.0, seen=None):
        def plan(planner, snapshot, source, window, **kwargs):
            if seen is not None:
                seen.append(snapshot)
            if advance:
                kwargs["clock"].__self__.clock.value += advance
            budget = snapshot.budget if hold else (32.0 if snapshot.budget == 16.0 else 16.0)
            proposal = p.TimedPlan((budget,) * 3, snapshot.budget, snapshot.sampled_at, window.first_action_at)
            return p.NominalDecision(
                kwargs["policy"], proposal,
                p.FirstAction(budget, window.first_action_at, window.action_deadline,
                              p.StateVersion.from_snapshot(snapshot)),
                Fraction(0), Fraction(0), 8,
            )

        with mock.patch.object(p.NominalPlanner, "plan", new=plan):
            yield

    async def test_real_planner_feedback(self):
        for policy in ("mpc", "one_move"):
            with self.subTest(policy=policy):
                gateway = self.gateway(time.monotonic)
                gateway.feedback = True
                seen = []
                source_id = digest({"CPU_frozen_development_forecast": [0.2, 64, 4000, 1]})

                def provider(snapshot):
                    seen.append(snapshot)
                    return p.CausalScenario(
                        Scenario((FutureOffer("CPU-forecast-offer", 0.2, 64, 4000, 1),)),
                        p.StateVersion.from_snapshot(snapshot), source_id,
                        snapshot.sampled_at, snapshot.sampled_at, "development", "engineering",
                    )

                runtime = self.runtime(
                    gateway, name=policy + ".json", policy=policy, factory=provider,
                    limits=r.RuntimeLimits(iterations=2, episode_s=3.0),
                )
                with mock.patch.object(p, "rollout", wraps=p.rollout) as real_rollout:
                    result = await self.execute(runtime)
                self.assertIsNone(result["failure"])
                self.assertEqual([row["status"] for row in result["steps"]], ["applied_reobserved", "hold"])
                self.assertEqual(real_rollout.call_count, 16)
                self.assertEqual(len(gateway.posts), 1)
                self.assertEqual(gateway.gets, 4)
                self.assertEqual([snapshot.budget for snapshot in seen], [16.0, 32.0])
                self.assertEqual([len(snapshot.requests) for snapshot in seen], [0, 1])
                self.assertEqual(seen[1].requests[0].request_id, "CPU-admitted-after-cap")
                planned = self.events(runtime, "Planned")
                self.assertEqual([event["decision"]["predicted_lower"]["numerator"] for event in planned], [1, 0])
                self.assertNotEqual(planned[0]["snapshot_sha256"], planned[1]["snapshot_sha256"])
                self.assertFalse(result["Native_qualified"])
                self.assertFalse(result["ledger_CAS"])
                self.write(policy + "-REAL_ROLLOUT.json", {
                    "scope": "literal CPU forecast + MockTransport feedback, not Native",
                    "real_rollout_calls": real_rollout.call_count, "planning_observations": seen,
                    "plans": planned, "actual_MockTransport_budget_changes": len(gateway.posts),
                })

    async def test_every_first_action_replans_and_reobserves(self):
        clock = Clock()
        gateway = self.gateway(clock)
        gateway.feedback = True
        runtime = self.runtime(gateway, limits=r.RuntimeLimits(iterations=3, episode_s=3.0))
        seen = []
        with self.literal(seen=seen):
            result = await self.execute(runtime)
        self.assertEqual([row["status"] for row in result["steps"]], ["applied_reobserved"] * 3)
        self.assertEqual([command["budget"] for command in gateway.posts], [32.0, 16.0, 32.0])
        self.assertEqual([snapshot.budget_version for snapshot in seen], [0, 1, 2])
        self.assertEqual(gateway.gets, 9)
        self.assertEqual(len(self.events(runtime, "CapApplied")), 3)
        self.assertEqual(len(self.events(runtime, "CapReobserved")), 3)
        self.assertTrue(all(event["suffix_execution_permitted"] is False for event in self.events(runtime, "Planned")))
        self.assertEqual(len(clock.sleeps), 3)
        self.assertTrue(all(abs(delay - 0.1) < 1e-10 for delay in clock.sleeps))
        self.assertEqual(len({command["operation_id"] for command in gateway.posts}), 3)
        self.assertIsNone(result["pending_outcome"])

    async def test_default_off_missing_permit_and_hold_are_not_application(self):
        for name, enabled, permit, hold in (
            ("off", False, True, False), ("missing", True, False, False), ("hold", True, True, True),
        ):
            with self.subTest(name=name):
                gateway = self.gateway(Clock())
                runtime = self.runtime(gateway, name=name + ".json", enabled=enabled, permit=permit)
                with self.literal(hold=hold):
                    result = await self.execute(runtime)
                self.assertEqual(len(gateway.posts), 0)
                self.assertEqual(result["issued_commands"], 0)
                self.assertEqual(len(self.events(runtime, "CapApplied")), 0)
                self.assertIs(result["steps"][0]["cap_change_applied"], False)
        with mock.patch.object(r.httpx, "AsyncClient", side_effect=AssertionError("client must not start")):
            with self.assertRaises(SystemExit) as stopped:
                r.main(["--journal", str(self.directory / "no-authority.json"), "--experimental-nominal-demo"])
            self.assertEqual(stopped.exception.code, 2)

    async def test_original_entry_covers_get_solve_and_durable_persistence(self):
        for phase in ("get", "solve", "fsync"):
            with self.subTest(phase=phase):
                clock = Clock()
                gateway = self.gateway(clock)
                runtime = self.runtime(gateway, name=phase + ".json")
                if phase == "get":
                    def delayed_get(number, document):
                        clock.value += 0.12
                        document["sampled_at"] = clock()
                    gateway.mutate_get = delayed_get
                save = r._Journal.save

                def delayed_save(store, state):
                    save(store, state)
                    if phase == "fsync" and state["events"] and state["events"][-1]["event"] == "PendingPrepared":
                        clock.value += 0.51

                with self.literal(advance=0.11 if phase == "solve" else 0), \
                        mock.patch.object(r._Journal, "save", new=delayed_save):
                    result = await self.execute(runtime)
                self.assertEqual(gateway.posts, [])
                self.assertEqual(result["steps"][0]["entered_at"], 100.0)
                self.assertEqual(result["steps"][0]["deadline"], 100.5)
                expected = {
                    "get": "entry_GET_exhausted_solver_window",
                    "solve": "planner_finished_after_original_solve_deadline",
                    "fsync": "stale_snapshot",
                }[phase]
                self.assertEqual(result["steps"][0]["reason"], expected)
                if phase == "fsync":
                    self.assertEqual(result["pending_outcome"], "unknown")
                    self.assertIsNone(result["steps"][0]["cap_change_applied"])
                    self.assertEqual(result["issued_commands"], 1)

    async def test_precommit_run_clock_identity_cas_sequence_and_ledger_changes_refuse(self):
        for kind in ("run", "clock", "identity", "version", "sequence", "ledger"):
            with self.subTest(kind=kind):
                clock = Clock()
                gateway = self.gateway(clock)

                def mutate(number, document):
                    if number != 2:
                        return
                    if kind == "run":
                        document["run_id"] = "CPU-different-run"
                    elif kind == "clock":
                        document["clock_domain"] = "CPU-foreign-clock"
                    elif kind == "identity":
                        document["identity"]["support_hash"] = "f" * 64
                    elif kind == "version":
                        document["budget_version"] += 1
                    elif kind == "sequence":
                        document["sequence"] += 1
                    elif kind == "ledger":
                        document["requests"] = [asdict(gateway.mark())]

                gateway.mutate_get = mutate
                runtime = self.runtime(gateway, name=kind + ".json")
                with self.literal():
                    result = await self.execute(runtime)
                self.assertEqual(gateway.gets, 2)
                self.assertEqual(gateway.posts, [])
                self.assertEqual(result["issued_commands"], 0)
                self.assertEqual(result["steps"][0]["status"], "refused")
                self.assertEqual(len(self.events(runtime, "PendingPrepared")), 0)

    async def test_snapshot_capability_costs_freshness_and_strict_raw_json(self):
        for kind in ("capability", "cost_spec", "second_mark_cost", "stale", "predates_GET",
                     "incomplete", "duplicate", "foreign_agreement", "secret_echo"):
            with self.subTest(kind=kind):
                clock = Clock()
                gateway = self.gateway(clock)
                if kind == "foreign_agreement":
                    gateway.domain = "CPU-permit-and-server-foreign"

                def mutate(number, document):
                    if kind == "capability":
                        document["capabilities"] = []
                    elif kind == "cost_spec":
                        document["cost_spec"]["intercept"] += 0.01
                    elif kind == "second_mark_cost":
                        first = gateway.mark("first")
                        second = replace(gateway.mark("second"), cost=first.cost + 1)
                        document["requests"] = [asdict(first), asdict(second)]
                    elif kind == "stale":
                        document["sampled_at"] -= 0.6
                    elif kind == "predates_GET":
                        document["sampled_at"] -= 0.001
                    elif kind == "incomplete":
                        document["complete"] = False
                    elif kind == "duplicate":
                        return json.dumps(document).encode()[:-1] + b',"budget":32}'
                    elif kind == "secret_echo":
                        document["CPU_unexpected_echo"] = CPU_KEY

                gateway.mutate_get = mutate
                runtime = self.runtime(gateway, name=kind + ".json")
                with self.literal():
                    result = await self.execute(runtime)
                self.assertEqual(gateway.posts, [])
                self.assertEqual(result["steps"][0]["status"], "refused")
                self.assertEqual(self.events(runtime, "Planned"), [])
                raw = runtime.store.path.read_text()
                self.assertNotIn(CPU_KEY, raw)
                if kind == "secret_echo":
                    self.assertTrue(self.events(runtime, "HTTP")[0]["redacted"])
                if kind == "second_mark_cost":
                    self.assertEqual(result["steps"][0]["reason"], "immutable_ledger_cost_mismatch")
                if kind == "predates_GET":
                    self.assertEqual(result["steps"][0]["reason"], "snapshot_predates_current_GET_entry")

    async def test_unknown_timeout_and_malformed_ack_need_observed_restart_fences(self):
        for kind in ("timeout_CAS", "timeout_expiry", "malformed_ack"):
            with self.subTest(kind=kind), self.literal():
                clock = Clock()
                gateway = self.gateway(clock)
                gateway.post_mode = "malformed_ack" if kind == "malformed_ack" else "timeout"
                name = kind + ".json"
                first = self.runtime(gateway, name=name)
                initial = await self.execute(first)
                self.assertEqual(initial["pending_outcome"], "unknown")
                self.assertIsNone(initial["steps"][0]["cap_change_applied"])
                self.assertEqual(len(gateway.posts), 1)
                if kind != "malformed_ack":
                    unchanged = await self.execute(self.runtime(gateway, name=name))
                    self.assertTrue(unchanged["pending"])
                    self.assertEqual(len(gateway.posts), 1)
                    old_sample = clock()
                    clock.value += 0.2
                    gateway.mutate_get = lambda number, document: document.update(sampled_at=old_sample)
                    stale = await self.execute(self.runtime(gateway, name=name))
                    self.assertTrue(stale["pending"])
                    self.assertEqual(len(gateway.posts), 1)
                    gateway.mutate_get = None
                    if kind == "timeout_CAS":
                        gateway.version += 1
                        gateway.budget = 32.0
                fence_runtime = self.runtime(gateway, name=name)
                fenced = await self.execute(fence_runtime)
                expected = "unknown_fenced_by_server_expiry" if kind == "timeout_expiry" else "unknown_fenced_by_CAS"
                self.assertEqual(fenced["steps"][0]["status"], expected)
                self.assertFalse(fenced["steps"][0]["cap_application_known"])
                self.assertFalse(fenced["pending"])
                self.assertEqual(fenced["issued_commands"], 1)
                self.assertEqual(len(gateway.posts), 1)
                self.assertEqual(self.events(fence_runtime, "CapApplied"), [])
                gateway.post_mode = "apply"
                final = await self.execute(self.runtime(gateway, name=name))
                self.assertEqual(final["issued_commands"], 2)
                self.assertEqual(len(gateway.posts), 2)
                self.assertNotEqual(gateway.posts[0]["operation_id"], gateway.posts[1]["operation_id"])

    async def test_ack_without_reobservation_stays_pending_and_ack_is_not_erased(self):
        gateway = self.gateway(Clock())
        gateway.failed_reobservation = True
        with self.literal():
            first = self.runtime(gateway)
            failed = await self.execute(first)
            self.assertEqual(failed["pending_outcome"], "ACK_requires_reobservation")
            self.assertIs(failed["steps"][0]["cap_change_applied"], True)
            self.assertEqual(failed["steps"][0]["actual_cap_applied_at"], gateway.acks[0]["applied_at"])
            self.assertEqual(len(self.events(first, "CapApplied")), 1)
            self.assertEqual(self.events(first, "CapReobserved"), [])
            gateway.failed_reobservation = False
            recovered = await self.execute(self.runtime(gateway))
        self.assertEqual(recovered["steps"][0]["status"], "applied_reobserved")
        self.assertEqual(len(gateway.posts), 1)
        self.assertFalse(recovered["pending"])

    async def test_command_count_and_episode_deadline_survive_restart(self):
        clock = Clock()
        gateway = self.gateway(clock)
        limits = r.RuntimeLimits(iterations=1, episode_s=1.0)
        with self.literal():
            for expected in (1, 2):
                result = await self.execute(self.runtime(gateway, limits=limits, max_commands=2))
                self.assertEqual(result["issued_commands"], expected)
                self.assertEqual(result["episode_deadline"], 101.0)
            refused = await self.execute(self.runtime(gateway, limits=limits, max_commands=2))
            self.assertEqual(refused["steps"][0]["reason"], "nominal_command_limit")
            self.assertEqual(len(gateway.posts), 2)
            gets = gateway.gets
            clock.value += 2
            expired = await self.execute(self.runtime(gateway, limits=limits, max_commands=2))
        self.assertEqual(expired["failure"], "persistent_episode_deadline_exhausted")
        self.assertEqual(expired["issued_commands"], 2)
        self.assertEqual(expired["episode_deadline"], 101.0)
        self.assertEqual(gateway.gets, gets)
        self.assertEqual(len(gateway.posts), 2)

    async def test_fsync_failure_preserves_partial_and_refuses_restart_before_post(self):
        gateway = self.gateway(Clock())
        runtime = self.runtime(gateway)
        save = r._Journal.save

        def fail_fsync(store, state):
            if state["events"] and state["events"][-1]["event"] == "PendingPrepared":
                with mock.patch.object(r.os, "fsync", side_effect=OSError("CPU injected fsync EIO")):
                    return save(store, state)
            return save(store, state)

        with self.literal(), mock.patch.object(r._Journal, "save", new=fail_fsync):
            failed = await self.execute(runtime)
        self.assertIn("CPU injected fsync EIO", failed["failure"])
        self.assertEqual(gateway.posts, [])
        self.assertTrue(runtime.store.partial.is_file())
        partial = json.loads(runtime.store.partial.read_bytes())
        self.assertEqual(partial["issued_commands"], 1)
        self.assertIsNone(partial["pending"]["ack"])
        gets = gateway.gets
        restart = await self.execute(self.runtime(gateway))
        self.assertIn("unclean_journal_commit_preserved", restart["failure"])
        self.assertEqual(gateway.gets, gets)
        self.assertEqual(gateway.posts, [])
        self.assertTrue(runtime.store.partial.exists())

    async def test_concurrent_owner_refuses_and_cancelled_post_releases_its_lock(self):
        gateway = self.gateway(Clock())
        gateway.post_mode = "block"
        with self.literal():
            owner = self.runtime(gateway)
            task = asyncio.create_task(owner.run())
            self.tasks.append(task)
            try:
                await asyncio.wait_for(gateway.post_started.wait(), 1)
                competing = await self.execute(self.runtime(gateway))
                self.assertIn("BlockingIOError", competing["failure"])
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            gateway.post_mode = "timeout"
            restarted = await self.execute(self.runtime(gateway))
        self.assertTrue(restarted["pending"])
        self.assertEqual(restarted["issued_commands"], 1)
        self.assertEqual(gateway.cancelled_posts, 1)
        self.assertEqual(len(gateway.posts), 1)
        self.assertEqual(len(self.events(owner, "Cancelled")), 1)
        self.assertEqual(self.events(owner, "CapApplied"), [])

    async def test_source_configuration_and_unresolved_journal_cannot_be_substituted(self):
        gateway = self.gateway(Clock())
        runtime = self.runtime(gateway, name="source.json")
        with self.literal(), mock.patch.object(
                r, "source_digest", side_effect=(CPU_SOURCE, CPU_SOURCE, "f" * 64)):
            changed = await self.execute(runtime)
        self.assertEqual(gateway.posts, [])
        self.assertEqual(changed["steps"][0]["reason"], "nominal_source_or_protocol_mismatch")
        self.assertEqual(changed["pending_outcome"], "unknown")

        altered = self.runtime(gateway, name="configuration.json")
        altered.limits = replace(altered.limits, episode_s=10.0)
        gets = gateway.gets
        result = await self.execute(altered)
        self.assertIn("runtime_configuration_or_parameter_binding_changed", result["failure"])
        self.assertEqual(gateway.gets, gets)

        raw = runtime.store.path.read_bytes()
        with (self.directory / "before-corruption.json").open("xb") as stream:
            stream.write(raw)
        document = json.loads(raw)
        document["pending"] = None
        runtime.store.path.write_text(json.dumps(document))
        corrupted = await self.execute(self.runtime(gateway, name="source.json"))
        self.assertIn("journal_unresolved_command_cannot_disappear", corrupted["failure"])
        self.assertEqual(gateway.posts, [])
        self.assertEqual(gateway.gets, gets)


if __name__ == "__main__":
    unittest.main()
