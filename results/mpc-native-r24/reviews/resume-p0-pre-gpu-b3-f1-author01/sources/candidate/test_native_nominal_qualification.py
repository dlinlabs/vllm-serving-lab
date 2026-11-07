"""CPU-only caller accounting, guarded MockTransport observations and persistence controls."""

import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import types
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import native_nominal_qualification as driver
import nominal_client
import nominal_episode
import nominal_runtime
from mpc_lab.contracts import ModelParameters
from nominal_planner import NominalPlanner


ROOT = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"])
ROOT = ROOT.with_name(ROOT.name + "-qualification")
ROOT.mkdir(parents=True, exist_ok=True)
REAL_CLIENT = httpx.AsyncClient
REAL_CAPTURE_WRITE = nominal_client.write_capture
REAL_WRITE = driver.services.write


class CallerFixture:
    """Literal capture/episode boundaries; real caller, binding, codec and MockTransport GETs."""

    def __init__(self, case, name):
        self.case = case
        self.output = ROOT / name
        (self.output / "services").mkdir(parents=True)
        self.protocol_path = Path(os.environ["NATIVE_SMOKE_PROTOCOL"]).resolve()
        self.root = self.protocol_path.parents[3]
        self.argv = ["CPU-vllm", "serve", driver.services.MODEL]
        (self.output / "services/ownership.json").write_text(json.dumps({
            "services": [{"name": "engine", "argv": self.argv}]}))
        parameters = ModelParameters()
        self.report = {"source": {
            "cost_spec": asdict(parameters.cost_spec), "identity": asdict(NominalPlanner(parameters).identity),
            "clock_domain": nominal_runtime.clock_domain(),
            "source_pins": {str(self.protocol_path): hashlib.sha256(self.protocol_path.read_bytes()).hexdigest()}},
            "configuration": {"engine": self.argv}}
        self.owner = types.SimpleNamespace(deadline=time.monotonic() + 480, reserve=30, check=self.check)
        self.original_deadline = self.owner.deadline
        self.clients, self.calls, self.metrics, self.observed_starts = [], [], [], []
        self.prepare_error = self.state_error = self.warmup_error = self.episode_error = None
        self.fail_writes, self.fail_capture_writes = set(), set()
        self.warmup_calls = self.episode_calls = self.checks = 0
        self.expire_after_metrics = None
        self.bad_result = None
        self.record_count = 5
        domain = nominal_client.clock_domain()
        self.warmup = types.SimpleNamespace(ready=True, capture={
            "scope": "CPU-only fixture", "clock_domain": domain, "status": "failed",
            "error": nominal_client.NOT_RUN_ERROR, "records": None})
        self.prepared = types.SimpleNamespace(
            error=None, recorder=types.SimpleNamespace(
                capture={"scope": "CPU-only fixture", "clock_domain": domain, "status": "failed",
                         "error": nominal_client.NOT_RUN_ERROR, "records": None},
                trace={"duration_seconds": 5, "requests": [{"trace_id": f"CPU-{i}"} for i in range(5)]}),
            scenario_binding={"scope": "CPU-only caller boundary"}, scenario_factory=lambda _: None)

    def check(self):
        self.checks += 1
        if self.expire_after_metrics is not None and sum(call["path"] == "/metrics" for call in self.calls) >= self.expire_after_metrics:
            raise driver.services.ProbeError("CPU original owner work budget exhausted")

    @staticmethod
    def zero_metrics():
        return dict(current_in_flight=0, current_waiting=0, current_admitted_cost=0,
                    tokenizer_fallback_requests=0)

    def handle(self, request):
        self.case.assertTrue((self.output / "qualification-source-binding.json").is_file())
        self.case.assertTrue(all(value > 0 for value in request.extensions["timeout"].values()))
        self.calls.append({"path": request.url.path, "timeout": request.extensions["timeout"]})
        if request.url.path == "/metrics":
            value = self.metrics.pop(0) if self.metrics else self.zero_metrics()
            return httpx.Response(200, json=value)
        self.case.assertEqual(request.url.path, "/admin/mpc/state")
        if self.state_error is not None:
            raise self.state_error
        return httpx.Response(200, json={
            **{name: self.report["source"][name] for name in ("identity", "clock_domain", "cost_spec")},
            "run_id": "CPU-only-qualified-caller-fixture"})

    def client(self, **kwargs):
        value = REAL_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(value)
        return value

    async def warmup_replay(self, prepared, **kwargs):
        self.warmup_calls += 1
        self.case.assertTrue((self.output / "qualification-source-binding.json").is_file())
        self.case.assertEqual(kwargs["deadline"], self.original_deadline)
        self.case.assertEqual(kwargs["cleanup_reserve_s"], 30)
        self.case.assertEqual(kwargs["clock_domain"], prepared.capture["clock_domain"])
        self.case.assertNotEqual(kwargs["clock_domain"], self.report["source"]["clock_domain"])
        prepared.capture.update(
            status="collected", error=None,
            records=[{"fate": "complete", "stream_done": True, "finish_reason": "stop",
                      "sent_at": time.monotonic()} for _ in range(3)],
            summary={"prompt_tokens_mismatch": 0})
        if self.warmup_error is not None:
            prepared.capture["status"] = "interrupted"
            raise self.warmup_error
        return prepared.capture

    def episode_prepare(self, **kwargs):
        self.case.assertEqual(kwargs["trace_path"], self.root / driver.PROBES / "admission-boundary.json")
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.prepared

    async def episode_run(self, prepared, runtime, *, owner, offered_start, enabled):
        self.episode_calls += 1
        self.case.assertTrue(enabled)
        self.case.assertIs(owner, self.owner)
        self.case.assertEqual(runtime.start_at, offered_start)
        self.case.assertEqual(runtime.permit.expires_at - runtime.permit.not_before, 20)
        self.case.assertEqual(runtime.permit.max_commands, 8)
        self.case.assertEqual(offered_start, runtime.permit.not_before + 1)
        self.case.assertLessEqual(offered_start + 5, min(owner.deadline - owner.reserve, runtime.permit.expires_at))
        binding = json.loads((self.output / "runtime-binding.json").read_text())
        self.case.assertEqual(binding["description"]["protocol_sha256"], runtime.protocol_sha256)
        self.case.assertEqual(binding["caller_sha256"], hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest())
        self.case.assertEqual(binding["adapter_sha256"], driver.ADAPTER_SHA256)
        self.observed_starts.append(offered_start)
        fates = ("complete", "rejected", "unknown", "complete", "not_launched")
        prepared.recorder.capture.update(status="collected", error=None, records=[
            {"trace_id": f"CPU-{i}", "fate": fates[i],
             "scheduled_at": offered_start + 1.25, "sent_at": offered_start + 1.25 if i < 4 else None}
            for i in range(self.record_count)])
        if self.episode_error is not None:
            prepared.recorder.capture["status"] = "interrupted"
            raise self.episode_error
        return self.bad_result if self.bad_result is not None else {
            "status": "collected", "failures": [], "all_workers_joined": True,
            "persistence_errors": [], "missing_record_ids": [], "qualification": None}

    def write(self, path, *args, **kwargs):
        if path.name in self.fail_writes:
            raise OSError("CPU receipt write failure: " + path.name)
        return REAL_WRITE(path, *args, **kwargs)

    def capture_write(self, path, capture):
        if path.name in self.fail_capture_writes:
            raise OSError("CPU raw capture write failure: " + path.name)
        return REAL_CAPTURE_WRITE(path, capture)

    def run(self):
        async def invoke():
            try:
                await driver.collect(owner=self.owner, root=self.root, output=self.output,
                                     key="CPU-key", report=self.report)
            except BaseException as error:
                return error
            finally:
                self.case.assertTrue(all(client.is_closed for client in self.clients))
                self.case.assertEqual([task for task in asyncio.all_tasks()
                                       if task is not asyncio.current_task() and not task.done()], [])
            return None

        with ExitStack() as stack:
            for target, name, value in (
                (httpx, "AsyncClient", self.client),
                (httpx.AsyncHTTPTransport, "handle_async_request", AsyncMock(side_effect=AssertionError("real HTTP forbidden"))),
                (nominal_client, "prepare_capture", lambda **_: self.warmup),
                (nominal_client, "capture_replay", self.warmup_replay),
                (nominal_client, "write_capture", self.capture_write),
                (nominal_episode, "prepare_episode", self.episode_prepare),
                (nominal_episode, "run_episode", self.episode_run),
                (driver.services, "write", self.write),
            ):
                stack.enter_context(patch.object(target, name, new=value))
            error = asyncio.run(invoke())
        (self.output / "CPU-CALLER-CHECK.json").write_text(json.dumps({
            "error": None if error is None else driver.error_record(error),
            "calls": self.calls, "checks": self.checks, "warmup_calls": self.warmup_calls,
            "episode_calls": self.episode_calls, "clients_closed": [c.is_closed for c in self.clients],
            "remaining_tasks": [], "real_HTTP_or_services": False,
            "report": self.report,
        }, sort_keys=True, allow_nan=False))
        return error


class Controls(unittest.TestCase):
    def early_exit(self, name, prepared_error=None):
        protocol_path = Path(os.environ["NATIVE_SMOKE_PROTOCOL"]).resolve()
        root, output = protocol_path.parents[3], ROOT / name
        (output / "services").mkdir(parents=True)
        argv = ["CPU-vllm", "serve", driver.services.MODEL]
        (output / "services/ownership.json").write_text(json.dumps({
            "services": [{"name": "engine", "argv": argv}]}))
        report = {"source": {
            "cost_spec": {}, "identity": {}, "clock_domain": "CPU-source-clock",
            "source_pins": {str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest()}},
            "configuration": {"engine": argv}, "offered_requests_from_driver": None}
        warmup = types.SimpleNamespace(ready=True, capture={
            "clock_domain": "CPU-recorder-clock", "status": "collected", "error": None,
            "records": [{"fate": "complete", "stream_done": True, "finish_reason": "stop",
                         "sent_at": time.monotonic()} for _ in range(3)],
            "summary": {"prompt_tokens_mismatch": 0}})
        prepared = types.SimpleNamespace(
            error=prepared_error, recorder=types.SimpleNamespace(
                trace={"duration_seconds": 5, "requests": [{"trace_id": f"CPU-{i}"} for i in range(5)]},
                capture={"clock_domain": "CPU-recorder-clock", "status": "failed",
                         "error": "not_run", "records": None}))

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def aclose(self):
                pass

            async def get(self, path, **_kwargs):
                if path == "/metrics":
                    return httpx.Response(200, request=httpx.Request("GET", "http://CPU/metrics"), json={
                        "current_in_flight": 0, "current_waiting": 0,
                        "current_admitted_cost": 0, "tokenizer_fallback_requests": 0})
                raise RuntimeError("CPU pre-control GET failure")

        with patch.object(nominal_client, "prepare_capture", return_value=warmup), \
                patch.object(nominal_client, "capture_replay", new=AsyncMock(return_value=warmup.capture)), \
                patch.object(nominal_episode, "prepare_episode", return_value=prepared), \
                patch.object(nominal_episode, "run_episode", new=AsyncMock()) as run_episode, \
                patch.object(httpx, "AsyncClient", new=lambda **_: Client()):
            expected = driver.services.ProbeError if prepared_error else RuntimeError
            with self.assertRaises(expected):
                asyncio.run(driver.collect(
                    owner=types.SimpleNamespace(check=lambda: None, deadline=time.monotonic() + 480, reserve=30),
                    root=root, output=output, key="", report=report))
            run_episode.assert_not_awaited()
        self.assertEqual(report["offered_requests_from_driver"], 0)
        receipt = json.loads((output / "episode-not-run.json").read_text())
        self.assertEqual(receipt["records_allocated"], 0)
        self.assertEqual(receipt["sent"], 0)
        self.assertEqual(receipt["prepared_error"], prepared_error)
        self.assertTrue((output / "episode-not-run-capture.json").is_file())
        return report

    def test_pre_control_error_keeps_not_run_capture_and_zero_offers(self):
        report = self.early_exit("pre-control-error")
        self.assertEqual(report["episode_trace_requests"], 5)

    def test_preparation_error_keeps_primary_reason_and_zero_offers(self):
        self.early_exit("prepare-error", {"message": "CPU preparation refused"})

    def test_prepare_exception_and_none_recorder_keep_not_run_reason(self):
        for kind in ("raised", "no-recorder"):
            with self.subTest(kind=kind):
                fixture = CallerFixture(self, kind)
                primary = RuntimeError("CPU prepare raised original reason")
                if kind == "raised":
                    fixture.prepare_error = primary
                else:
                    fixture.prepared.error = {"message": "CPU local preparation refused"}
                    fixture.prepared.recorder = None
                error = fixture.run()
                if kind == "raised":
                    self.assertIs(error, primary)
                else:
                    self.assertIsInstance(error, driver.services.ProbeError)
                    self.assertIn("CPU local preparation refused", str(error))
                receipt = json.loads((fixture.output / "episode-not-run.json").read_text())
                self.assertEqual((receipt["records_allocated"], receipt["sent"]), (0, 0))
                self.assertEqual(fixture.report["offered_requests_from_driver"], 0)
                self.assertEqual(fixture.episode_calls, 0)
                absent = json.loads((fixture.output / "episode-not-run-capture.json").read_text())
                self.assertFalse(absent["raw_v2_capture_available"])

    def test_get_failure_survives_capture_and_error_receipt_write_failures(self):
        fixture = CallerFixture(self, "get-plus-write-error")
        primary = RuntimeError("CPU original GET failure")
        fixture.state_error = primary
        fixture.fail_capture_writes.add("episode-not-run-capture.json")
        fixture.fail_writes.add("episode-not-run-capture.write-error.json")
        self.assertIs(fixture.run(), primary)
        self.assertTrue(any("write" in str(row) for row in fixture.report["caller_errors"]))
        self.assertIn("write-error receipt also failed", str(fixture.report["caller_errors"]))
        self.assertEqual(fixture.report["offered_requests_from_driver"], 0)
        self.assertEqual(fixture.report["episode_trace_requests"], 5)
        receipt = json.loads((fixture.output / "episode-not-run.json").read_text())
        self.assertEqual(receipt["primary_error"]["message"], str(primary))

    def test_warmup_interrupts_survive_capture_persistence_failure(self):
        for name, primary in (("warmup-KI", KeyboardInterrupt("CPU original warmup interrupt")),
                              ("warmup-cancel", asyncio.CancelledError("CPU original warmup cancellation"))):
            with self.subTest(name=name):
                fixture = CallerFixture(self, name)
                fixture.warmup_error = primary
                fixture.fail_capture_writes.add("warmup-capture.json")
                self.assertIs(fixture.run(), primary)
                self.assertEqual(fixture.episode_calls, 0)
                self.assertEqual(fixture.report["offered_requests_from_driver"], 0)
                self.assertTrue((fixture.output / "warmup-capture.write-error.json").is_file())
                self.assertTrue((fixture.output / "episode-not-run.json").is_file())

    def test_complete_collect_keeps_counts_domains_bindings_and_all_drain_readings(self):
        fixture = CallerFixture(self, "complete")
        zero = fixture.zero_metrics()
        fixture.metrics = [{**zero, "current_in_flight": 1, "current_admitted_cost": 3.2}, zero,
                           {**zero, "current_waiting": 1}, zero]
        self.assertIsNone(fixture.run())
        self.assertEqual(fixture.report["warmup_offered_requests"], 3)
        self.assertEqual(fixture.report["episode_trace_requests"], 5)
        self.assertEqual(fixture.report["offered_requests_from_driver"], 5)
        self.assertEqual(fixture.report["episode_sent_requests"], 4)
        self.assertEqual(fixture.report["qualification_caller"]["status"], "COLLECTED")
        raw = json.loads((fixture.output / "caller-episode-capture.json").read_text())
        self.assertEqual(raw, fixture.prepared.recorder.capture)
        self.assertIn("unknown", [row["fate"] for row in raw["records"]])
        self.assertEqual(len(list(fixture.output.glob("*-metrics-*.json"))), 4)
        self.assertEqual(fixture.report["caller_drains"]["pre-episode"]["readings"], 2)
        self.assertEqual(fixture.report["caller_drains"]["final"]["readings"], 2)
        self.assertEqual(fixture.report["qualification_caller"]["deadline"], fixture.original_deadline)
        self.assertIsNone(fixture.report["qualification_caller"]["full_wall"])

    def test_episode_cancel_preserves_allocated_partial_counts_and_primary(self):
        fixture = CallerFixture(self, "episode-cancel")
        primary = asyncio.CancelledError("CPU episode interrupted after allocation")
        fixture.episode_error, fixture.record_count = primary, 2
        self.assertIs(fixture.run(), primary)
        self.assertEqual(fixture.report["episode_planned_requests"], 5)
        self.assertEqual(fixture.report["offered_requests_from_driver"], 2)
        self.assertEqual(fixture.report["episode_sent_requests"], 2)
        self.assertEqual(fixture.report["qualification_caller"]["status"], "INTERRUPTED")
        raw = json.loads((fixture.output / "caller-episode-capture.json").read_text())
        self.assertEqual(len(raw["records"]), 2)

    def test_drain_and_fallback_failure_close_without_losing_readings(self):
        for kind in ("pre-fallback", "post-fallback", "busy-until-budget"):
            with self.subTest(kind=kind):
                fixture = CallerFixture(self, kind)
                zero = fixture.zero_metrics()
                if kind == "pre-fallback":
                    fixture.metrics = [{**zero, "tokenizer_fallback_requests": 1}]
                elif kind == "post-fallback":
                    fixture.metrics = [zero, {**zero, "tokenizer_fallback_requests": 1}]
                else:
                    fixture.metrics = [{**zero, "current_in_flight": 1}] * 2
                    fixture.expire_after_metrics = 2
                error = fixture.run()
                self.assertIsInstance(error, driver.services.ProbeError)
                self.assertEqual(fixture.episode_calls, int(kind == "post-fallback"))
                self.assertTrue(list(fixture.output.glob("*-metrics-*.json")))
                self.assertIsNotNone(fixture.report["qualification_caller"]["primary_error"])

    def test_drain_tolerates_only_idle_cost_residue(self):
        for name, cost, in_flight, waiting, idle in (
            ("epsilon-idle", 2.7755575615628914e-17, 0, 0, True),
            ("non-negligible-cost", 1e-6, 0, 0, False),
            ("one-in-flight", 0.0, 1, 0, False),
            ("one-waiting", 0.0, 0, 1, False),
        ):
            with self.subTest(case=name):
                async def exercise():
                    output = ROOT / ("drain-residue-" + name)
                    output.mkdir()
                    metrics = dict(current_in_flight=in_flight, current_waiting=waiting,
                                   current_admitted_cost=cost, tokenizer_fallback_requests=0)
                    report, calls, checks = {"caller_errors": []}, [], []

                    def check():
                        checks.append(time.monotonic())

                    def handle(request):
                        self.assertEqual(request.method, "GET")
                        self.assertEqual(request.url.path, "/metrics")
                        self.assertTrue(all(value > 0 for value in request.extensions["timeout"].values()))
                        calls.append(time.monotonic())
                        return httpx.Response(200, json=metrics)

                    error = None
                    async with REAL_CLIENT(base_url="http://CPU-drain", trust_env=False,
                                           transport=httpx.MockTransport(handle)) as client:
                        started = time.monotonic()
                        owner = types.SimpleNamespace(deadline=started + 0.6, reserve=0.1, check=check)
                        try:
                            await driver.drain(
                                client, owner=owner, deadline=owner.deadline, reserve=owner.reserve,
                                output=output, key="", report=report, phase="final")
                        except (driver.services.ProbeError, TimeoutError) as caught:
                            error = caught
                    returned = time.monotonic()
                    pending = [task.get_name() for task in asyncio.all_tasks()
                               if task is not asyncio.current_task() and not task.done()]
                    (output / "DRAIN-CHECK.json").write_text(json.dumps({
                        "case": name, "metrics": metrics, "expected_idle": idle,
                        "started": started, "returned": returned, "deadline": owner.deadline,
                        "cleanup_reserve_s": owner.reserve, "calls": calls, "owner_checks": checks,
                        "error": None if error is None else driver.error_record(error),
                        "client_closed": client.is_closed, "remaining_tasks": pending,
                        "report": report, "native_qualification": False,
                    }, sort_keys=True, allow_nan=False))
                    self.assertTrue(client.is_closed)
                    self.assertEqual(pending, [])
                    self.assertTrue(checks)
                    if idle:
                        self.assertIsNone(error)
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(report["caller_drains"]["final"]["readings"], 1)
                    else:
                        self.assertIsNotNone(error)
                        self.assertGreaterEqual(len(calls), 2)
                        self.assertGreaterEqual(returned, owner.deadline - owner.reserve)
                        self.assertNotIn("caller_drains", report)
                        if isinstance(error, driver.services.ProbeError):
                            self.assertIn("original work deadline exhausted", str(error))

                asyncio.run(exercise())

    def test_protocol_pin_and_positive_remaining_budget_fail_before_traffic(self):
        for kind in ("protocol-drift", "expired"):
            with self.subTest(kind=kind):
                fixture = CallerFixture(self, kind)
                if kind == "protocol-drift":
                    fixture.report["source"]["source_pins"][str(fixture.protocol_path)] = "0" * 64
                else:
                    fixture.owner.deadline = time.monotonic() - 1
                    fixture.original_deadline = fixture.owner.deadline
                self.assertIsInstance(fixture.run(), driver.services.ProbeError)
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.warmup_calls, 0)
                self.assertEqual(fixture.report["episode_sent_requests"], 0)

    def test_oversized_window_refuses_before_permit_or_episode(self):
        for kind in ("window-exceeds-permit", "window-exceeds-owner"):
            with self.subTest(kind=kind):
                fixture = CallerFixture(self, kind)
                if kind == "window-exceeds-permit":
                    fixture.prepared.recorder.trace["duration_seconds"] = 20
                else:
                    fixture.owner.deadline = time.monotonic() + 34
                    fixture.original_deadline = fixture.owner.deadline
                error = fixture.run()
                self.assertIsInstance(error, driver.services.ProbeError)
                self.assertIn("permit expiry", str(error))
                self.assertEqual(fixture.episode_calls, 0)
                self.assertFalse((fixture.output / "runtime-binding.json").exists())

    def test_non_json_result_and_terminal_write_failure_fail_closed(self):
        for kind in ("nonfinite-result", "mixed-keys", "terminal-write"):
            with self.subTest(kind=kind):
                fixture = CallerFixture(self, kind)
                if kind == "nonfinite-result":
                    fixture.bad_result = {"untrusted": float("inf")}
                elif kind == "mixed-keys":
                    fixture.bad_result = {1: "not-string", "other": "mixed"}
                else:
                    fixture.fail_writes.add("qualification-caller.json")
                error = fixture.run()
                self.assertIsNotNone(error)
                json.dumps(fixture.report, sort_keys=True, allow_nan=False)
                self.assertEqual(fixture.report["qualification_caller"]["status"], "FAILED")
                self.assertEqual(fixture.report["offered_requests_from_driver"], 5)
                self.assertTrue((fixture.output / "caller-episode-capture.json").is_file())

    def test_default_off_never_enters_native_body(self):
        forbidden = AssertionError("OFF performed an effect")
        with patch.object(driver, "body", side_effect=forbidden), \
                patch.object(driver.services, "source_info", side_effect=forbidden), \
                patch.object(driver.services, "gpu_receipt", side_effect=forbidden), \
                patch.object(Path, "mkdir", side_effect=forbidden):
            self.assertEqual(driver.main([]), 2)

    def test_nonfinite_capture_write_keeps_error_receipt_and_fails(self):
        path = ROOT / "nonfinite-capture.json"
        with self.assertRaises(ValueError):
            driver.persist_capture(nominal_client, path, {"nonconforming": float("inf")}, "")
        receipt = json.loads(path.with_suffix(".write-error.json").read_text())
        self.assertTrue(receipt["capture_unscorable"])
        self.assertEqual(receipt["type"], "ValueError")
        self.assertFalse(path.exists())
