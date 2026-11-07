"""CPU-only entry and failed-capture persistence controls."""

import asyncio
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


ROOT = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"]) / "qualification"
ROOT.mkdir(parents=True, exist_ok=True)


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
