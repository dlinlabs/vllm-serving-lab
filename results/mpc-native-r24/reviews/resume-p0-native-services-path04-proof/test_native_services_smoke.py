"""Mocked CPU controls; no real service, HTTP, asset, GPU or source-model execution."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import native_services_smoke as driver


ROOT = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"])
assert not ROOT.is_absolute()
ROOT.mkdir()
PROTOCOL = json.loads(Path(os.environ["NATIVE_SMOKE_PROTOCOL"]).read_text())
KEY = "CPU-MOCK-ONLY-NOT-AN-ACTUAL-ADMIN-KEY"
IDENTITY = dict(model_version="cpu-lifecycle-v1", calibration_version="synthetic-unqualified-v1",
                support_hash="MOCK-derived-ModelParameters-support", cost_hash="MOCK-cost",
                code_hash="MOCK-source")
SPEC = dict(intercept=.259, input_per_token=.000758, output_per_token=.0052083333,
            minimum=.1, default_max_tokens=128,
            tokenizer_identity=driver.TOKENIZER
            + f";model={driver.MODEL};available=True;fallback=chars4-chat-v1")
EXPECTED = dict(identity=IDENTITY, cost_spec=SPEC, cost_spec_fingerprint="MOCK-cost",
                clock_domain="MOCK-clock", evidence_kind="CPU_MOCK_ONLY_NOT_HOST_VALIDATION")


class Controls(unittest.TestCase):
    def test_07_virtualenv_tool_resolution(self):
        venv = (ROOT / "tool-venv-bin").resolve()
        venv.mkdir()
        ninja = venv / "ninja"
        ninja.write_text("#!/bin/sh\nexit 0\n")
        ninja.chmod(0o700)
        with patch.object(driver.sys, "executable", str(venv / "python3.12")), \
                patch.dict(os.environ, PATH="/usr/bin:/bin"):
            config = driver.configuration(Path("/source"), Path("/output"), PROTOCOL, IDENTITY)
        for service in ("engine", "gateway"):
            path = config["safe_environment"][service]["PATH"]
            self.assertEqual(shutil.which("ninja", path=path), str(ninja))
            self.assertTrue(path.endswith(os.pathsep + "/usr/bin:/bin"))

    def test_01_off_no_side_effects(self):
        forbidden = AssertionError("OFF performed an effect")
        with patch.object(Path, "mkdir", side_effect=forbidden), \
                patch.object(Path, "read_text", side_effect=forbidden), \
                patch.object(driver, "source_info", side_effect=forbidden), \
                patch.object(driver, "gpu_receipt", side_effect=forbidden), \
                patch.object(driver, "http_client", side_effect=forbidden), \
                patch.object(driver, "owner_scope", side_effect=forbidden), \
                patch.object(driver.secrets, "token_hex", side_effect=forbidden), \
                patch.object(driver.os, "umask", side_effect=forbidden), \
                patch.object(driver.subprocess, "Popen", side_effect=forbidden):
            self.assertEqual(driver.main([]), 2)

    def test_02_production_binding_and_bootstrap_refusal(self):
        config = driver.configuration(Path("/source"), Path("/output"), PROTOCOL, IDENTITY)
        self.assertEqual(config["lock_paths"], [
            "/workspace/pr32-r23/r24-team16-mayor-gpu.lock",
            "/workspace/pr32-r23/native-component-GPU-8610519d.lock"])
        self.assertEqual(config["engine"], [
            str(Path(sys.executable).parent / "vllm"), "serve",
            "/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507", "--served-model-name",
            "Qwen/Qwen3-4B-Instruct-2507", "--host", "127.0.0.1", "--port", "18000",
            "--scheduling-policy", "fcfs", "--no-enable-prefix-caching",
            "--max-model-len", "8192", "--max-num-seqs", "256",
            "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", "1"])
        self.assertEqual(config["gateway"], [sys.executable, "-m", "uvicorn", "gateway:app",
                                             "--app-dir", "/source", "--host", "127.0.0.1",
                                             "--port", "18001", "--workers", "1"])
        env = config["safe_environment"]["gateway"]
        self.assertEqual(env["MPC_SUPPORT_HASH"], IDENTITY["support_hash"])
        self.assertNotEqual(env["MPC_SUPPORT_HASH"], "unqualified")
        self.assertEqual(env["MODEL_NAME"], driver.MODEL)
        self.assertEqual(env["VLLM_URL"], "http://127.0.0.1:18000")
        self.assertEqual(env["MPC_TOKENIZER_IDENTITY"], driver.TOKENIZER)
        self.assertEqual(env["MAX_ADMITTED_COST"], "32")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertNotIn("ADMIN_API_KEY", json.dumps(config))
        for field in ("lock", "optimization"):
            altered = copy.deepcopy(PROTOCOL)
            if field == "lock":
                altered["hardware"]["lock_paths"][1] = "/other/lock"
            else:
                altered["backend"]["required_common_settings"]["enforce_eager"] = True
            with self.subTest(field=field), self.assertRaises(driver.ProbeError):
                driver.configuration(Path("/source"), Path("/output"), altered, IDENTITY)
        stale = types.ModuleType("mpc_lab")
        stale.__file__ = "/stale-editable/mpc_lab/__init__.py"
        old_path = sys.path[:]
        try:
            with patch.dict(sys.modules, {"mpc_lab": stale}), self.assertRaisesRegex(
                    driver.ProbeError, "stale cached SDK"):
                driver.source_info(Path("/source"))
            self.assertEqual(sys.path[0], "/source")
        finally:
            sys.path[:] = old_path

    def exercise(self, name, invalid=False, poison=False, interrupt=False):
        base = ROOT / name
        base.mkdir()
        source = (base / "source").resolve()
        (source / driver.RELATIVE).mkdir(parents=True)
        (source / driver.RELATIVE / "nominal_protocol.json").write_text(json.dumps(PROTOCOL))
        output = (base / "output").resolve()
        args = argparse.Namespace(execute=True, source_root=str(source), output_root=str(output),
                                  seconds=20, cleanup_reserve=5, term_grace=1)
        metrics = dict(tokenizer_available=True, tokenizer_fallback_requests=0,
                       tokenizer_model=driver.MODEL, admission_mode="cost_aware",
                       max_admitted_cost=32, initial_admission_budget=32, budget_version=0,
                       accepted_requests=0, rejected_requests=0, failed_requests=0,
                       current_in_flight=0, current_waiting=0, current_admitted_cost=0,
                       shared_upstream_client=True)
        state = dict(complete=True, identity=copy.deepcopy(IDENTITY), cost_spec=SPEC,
                     clock_domain=EXPECTED["clock_domain"])
        if invalid:
            state["identity"]["code_hash"] = "MOCK-WRONG-ACTUAL-SOURCE"
        payloads = {"/health": None, "/v1/models": {"data": [{"id": driver.ALIAS}]},
                    "/metrics": metrics, "/admin/mpc/state": state}
        trace, active = [], [False]
        test = self

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def get(self, url, *, timeout, headers):
                test.assertTrue(active[0])
                test.assertGreater(timeout, 0)
                test.assertLessEqual(timeout, 2)
                path = "/" + url.split("/", 3)[3]
                port = "18000" if path in ("/health", "/v1/models") else "18001"
                test.assertTrue(url.startswith(f"http://127.0.0.1:{port}/"))
                test.assertIn(path, payloads)
                test.assertEqual(headers, {"x-admin-api-key": KEY} if path == "/admin/mpc/state" else {})
                trace.append("GET " + path)
                return types.SimpleNamespace(status_code=200, text=json.dumps(payloads[path]),
                                             json=lambda: payloads[path])

        class Owner:
            _thread = None

            def __init__(self, ready):
                self.ready = ready
                self.closed = False

            def _gone(self):
                return self.closed and not poison

            def check(self):
                test.assertTrue(active[0])
                if interrupt:
                    raise KeyboardInterrupt("MOCK INTERRUPT " + KEY)

            def close(self):
                self.closed, active[0] = True, False
                trace.append("owner exit")
                (output / "services").mkdir()
                receipt = dict(status="POISON" if poison else "STOPPED",
                               known_identities_gone=not poison, unknown=[], pending_spawn=None,
                               lock_paths=sorted(driver.LOCKS), evidence_kind="CPU_MOCK_ONLY")
                (output / "services/ownership.json").write_text(json.dumps(receipt))
                (output / "telemetry-status.json").write_text(json.dumps(dict(
                    status="closed", mpc_ledger_complete=True, dropped=0, write_errors=0,
                    mpc_remaining_requests=0, mpc_unflushed_records=0, mpc_final_sequence=0,
                    mpc_producer_final_sequence=0)))

            def __enter__(self):
                trace.append("owner enter")
                active[0] = True
                try:
                    test.assertTrue(self.ready(self, 15))
                    return self
                except Exception:
                    self.close()
                    raise

            def __exit__(self, *_):
                self.close()
                if poison:
                    raise driver.ProbeError("MOCK POISON " + KEY)

        def gpu(_deadline):
            test.assertFalse(active[0], "GPU subprocess attempted inside ownership context")
            trace.append("gpu")
            return dict(idle_selected_gpu=True, evidence_kind="CPU_MOCK_ONLY")

        def scope(config, actual_output, deadline, ready, key, actual_args):
            test.assertEqual(key, KEY)
            test.assertEqual(actual_output, output)
            test.assertEqual(config["safe_environment"]["gateway"]["MPC_SUPPORT_HASH"], IDENTITY["support_hash"])
            return Owner(ready)

        with patch.object(driver, "source_info", return_value=copy.deepcopy(EXPECTED)), \
                patch.object(driver, "existing_assets", return_value=True), \
                patch.object(driver, "gpu_receipt", side_effect=gpu), \
                patch.object(driver, "http_client", side_effect=Client), \
                patch.object(driver, "owner_scope", side_effect=scope), \
                patch.object(driver.secrets, "token_hex", return_value=KEY), \
                patch.object(driver.subprocess, "Popen", side_effect=AssertionError("real subprocess")), \
                patch.object(driver.subprocess, "run", side_effect=AssertionError("real GPU subprocess")):
            exit_code = driver.run(args)
        report_text = (output / "report.json").read_text()
        report = json.loads(report_text)
        self.assertNotIn(KEY, report_text)
        self.assertNotIn(KEY, (output / "events.jsonl").read_text())
        self.assertFalse(report["native_qualification"])
        self.assertFalse(report["inference_or_MPC_or_benchmark_result"])
        self.assertEqual(report["offered_requests_from_driver"], 0)
        self.assertEqual(report["admin_POSTs"], 0)
        self.assertEqual([x for x in trace if x.startswith("GET ")],
                         ["GET /health", "GET /v1/models", "GET /metrics", "GET /admin/mpc/state"])
        self.assertFalse(active[0])
        (base / "mock-receipt.json").write_text(json.dumps(dict(
            evidence_kind="CPU_MOCK_ONLY_NOT_HOST_VALIDATION", trace=trace, exit=exit_code)))
        return exit_code, report, trace

    def test_03_mocked_success_and_privacy(self):
        code, report, trace = self.exercise("success")
        self.assertEqual(code, 0)
        self.assertTrue(report["lifecycle_passed"])
        self.assertEqual(trace[0], "gpu")
        self.assertEqual(trace[-2:], ["owner exit", "gpu"])
        self.assertLess(report["ownership_exit_returned"]["monotonic"], report["deadline_monotonic"])

    def test_04_source_identity_failure_keeps_cleanup(self):
        code, report, trace = self.exercise("source-mismatch", invalid=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["lifecycle_passed"])
        self.assertIn("identity mismatch", report["errors"][0]["message"])
        self.assertTrue(report["known_closure_checked"]["closed"])
        self.assertEqual(trace[-2:], ["owner exit", "gpu"])

    def test_05_poison_skips_post_gpu(self):
        code, report, trace = self.exercise("poison", poison=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["lifecycle_passed"])
        self.assertEqual(report["ownership_receipt"]["status"], "POISON")
        self.assertIsNone(report["post_gpu"])
        self.assertEqual(trace.count("gpu"), 1)

    def test_06_interruption_is_failure_with_cleanup(self):
        with self.assertRaises(KeyboardInterrupt):
            self.exercise("interrupt", interrupt=True)
        output = ROOT / "interrupt/output"
        report_text = (output / "report.json").read_text()
        report = json.loads(report_text)
        self.assertFalse(report["lifecycle_passed"])
        self.assertTrue(report["known_closure_checked"]["closed"])
        self.assertEqual(report["errors"][0]["type"], "KeyboardInterrupt")
        self.assertNotIn("ownership_exit_returned", report)
        self.assertTrue(report["post_gpu"]["idle_selected_gpu"])
        self.assertNotIn(KEY, report_text)
        self.assertNotIn(KEY, (output / "events.jsonl").read_text())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    tests = list(suite)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    errors = {test.id(): message for test, message in result.errors}
    failures = {test.id(): message for test, message in result.failures}
    summary = dict(tests=result.testsRun, errors=len(errors), failures=len(failures),
                   evidence_kind="CPU_MOCK_ONLY_NOT_HOST_VALIDATION", python=sys.version,
                   actual_servers_HTTP_GPU_NVML_SSH_model_or_asset_calls=0)
    (ROOT / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
    xml = ET.Element("testsuite", tests=str(result.testsRun), failures=str(len(failures)), errors=str(len(errors)))
    for test in tests:
        case = ET.SubElement(xml, "testcase", name=test._testMethodName)
        for kind, table in (("error", errors), ("failure", failures)):
            if test.id() in table:
                ET.SubElement(case, kind).text = table[test.id()]
    (ROOT / "junit.xml").write_bytes(ET.tostring(xml, encoding="utf-8", xml_declaration=True))
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
