import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from batch_capture_bridge import BatchCapture, Endpoint, capture_errors, register, source_hash, validate_plan


def plan():
    return dict(schema="native-v2-batch-capture-plan-v1", partition="train",
                source_commit="a" * 40, trials=[
                    dict(trial_id="nv2-test-light", observer="light", seed=170001),
                    dict(trial_id="nv2-test-off", observer="off", seed=170002)])


def captured():
    return dict(schema="prefill-capture-v1", trial_id="nv2-test-light",
                status="complete", errors=[], overflow=0, pending_step_ids=[],
                requests={"physical-id": {"finished": 3.}},
                steps=[{"step_id": "identity-includes-request-and-position"}],
                workers=[dict(pending=0, overflow=0, rows=[
                    dict(step_id="identity-includes-request-and-position",
                         gpu_forward_s=.012, cuda_scope_qualified=False)])])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.reply = captured()
        self.clocks = [dict(clock_domain="boot-A:namespace-A",
                            process_id=12, bridge_sha256=source_hash())]
        self.core = SimpleNamespace(model_executor=SimpleNamespace(
            collective_rpc=lambda name: self.clocks))

        def command(core, payload):
            self.calls.append(payload)
            if payload["operation"] == "meta":
                return dict(tensor_parallel_size=1, pipeline_parallel_size=1)
            if payload["operation"] == "external_begin":
                return {"external": True}
            return copy.deepcopy(self.reply)
        self.bridge = BatchCapture(self.core, plan(), command, clock=lambda: 2.,
                                   domain="boot-A:namespace-A")

    def begin(self):
        self.bridge.invoke({"operation": "metadata"})
        return self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-light"})

    def test_default_off_register_does_not_import_or_install_gpu_hooks(self):
        with patch.dict(os.environ, {"PR33_NATIVE_BATCH_CAPTURE_ENABLED": "0"}):
            self.assertIsNone(register())
        with patch.dict(os.environ, {"PR33_NATIVE_BATCH_CAPTURE_ENABLED": "1",
                                    "PREFILL_LAB_ENABLED": "0"}):
            with self.assertRaises(RuntimeError):
                register()

    def test_heldout_and_duplicate_trials_rejected_before_any_runtime_operation(self):
        bad = plan()
        bad["trials"][0]["seed"] = 170301
        with self.assertRaises(ValueError):
            validate_plan(bad)
        bad = plan()
        bad["trials"].append(bad["trials"][0])
        with self.assertRaises(ValueError):
            validate_plan(bad)
        self.assertEqual(self.calls, [])

    def test_no_unfrozen_or_unqualified_begin_and_no_synthetic_operation(self):
        for payload in ({"operation": "start"}, {"operation": "abort"},
                        {"operation": "begin", "trial_id": "nv2-test-light"}):
            with self.assertRaises(ValueError):
                self.bridge.invoke(payload)
        self.assertEqual(self.calls, [])

    def test_worker_clock_mismatch_prevents_batch_capture(self):
        self.clocks[0]["clock_domain"] = "another-boot"
        with self.assertRaises(ValueError):
            self.bridge.invoke({"operation": "metadata"})
        with self.assertRaises(ValueError):
            self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-light"})

    def test_metadata_scope_and_begin_reuse_existing_external_capture(self):
        result = self.begin()
        self.assertEqual(self.calls[-1], dict(operation="external_begin",
                         trial_id="nv2-test-light", observer="light"))
        self.assertIn("Not kernel busy", result["measurement_scope"])
        self.assertEqual(result["clock_domain"], "boot-A:namespace-A")

    def test_pending_gpu_events_do_not_clear_or_complete_the_trial(self):
        self.begin()
        self.reply = {"status": "gpu_events_pending"}
        result = self.bridge.invoke({"operation": "end"})
        self.assertEqual(result["status"], "gpu_events_pending")
        self.assertNotIn("capture_valid", result)
        self.assertEqual(self.bridge.active, "nv2-test-light")
        self.reply = captured()
        result = self.bridge.invoke({"operation": "end"})
        self.assertTrue(result["capture_valid"])

    def test_valid_capture_does_not_upgrade_forward_span_to_pipeline_qualification(self):
        self.begin()
        result = self.bridge.invoke({"operation": "end"})
        self.assertTrue(result["capture_valid"])
        self.assertFalse(result["capture"]["workers"][0]["rows"][0]["cuda_scope_qualified"])
        with self.assertRaises(ValueError):
            self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-light"})

    def test_identity_gap_overflow_or_gpu_error_retains_raw_and_blocks_next_trial(self):
        for mutation in ("identity", "overflow", "gpu_error"):
            self.setUp()
            self.begin()
            if mutation == "identity":
                self.reply["workers"][0]["rows"][0]["step_id"] = "another-step"
            elif mutation == "overflow":
                self.reply["workers"][0]["overflow"] = 1
            else:
                self.reply["workers"][0]["rows"][0]["gpu_error"] = "RuntimeError"
            result = self.bridge.invoke({"operation": "end"})
            self.assertFalse(result["capture_valid"])
            self.assertEqual(result["capture"], self.reply)
            with self.assertRaises(ValueError):
                self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-off"})

    def test_failed_begin_is_not_replayed_or_falsely_reported_idle(self):
        self.bridge.invoke({"operation": "metadata"})
        def fail(*args):
            raise RuntimeError("worker begin failed after core capture was created")
        self.bridge.command = fail
        with self.assertRaises(RuntimeError):
            self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-light"})
        state = self.bridge.invoke({"operation": "status"})
        self.assertTrue(state["invalid"])
        self.assertEqual(state["active_trial"], "nv2-test-light")
        with self.assertRaises(ValueError):
            self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-light"})

    def test_off_does_not_accept_accidental_batch_instrumentation(self):
        self.assertIn("off_capture_contains_batch_instrumentation",
                      capture_errors(captured(), "off"))
        capture = captured()
        capture.update(requests={}, steps=[])
        capture["workers"][0]["rows"] = []
        self.assertEqual(capture_errors(capture, "off"), [])

    def test_worker_epoch_change_invalidates_but_retains_collected_data(self):
        self.begin()
        self.clocks[0]["process_id"] = 13
        result = self.bridge.invoke({"operation": "end"})
        self.assertFalse(result["capture_valid"])
        self.assertIn("worker_source_clock_epoch_changed", result["validation_errors"])
        self.assertEqual(result["capture"], self.reply)

    def test_worker_restart_after_metadata_cannot_start_a_new_capture(self):
        self.bridge.invoke({"operation": "metadata"})
        self.clocks[0]["process_id"] = 13
        calls = len(self.calls)
        with self.assertRaises(ValueError):
            self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-light"})
        self.assertEqual(len(self.calls), calls)
        self.assertTrue(self.bridge.invalid)

    def test_missing_terminal_is_not_silently_complete(self):
        capture = captured()
        capture["requests"]["physical-id"]["finished"] = None
        self.assertIn("missing_request_terminal", capture_errors(capture, "light"))

    def test_end_replays_exact_capture_until_ack_and_blocks_overwrite(self):
        self.begin()
        result = self.bridge.invoke({"operation": "end"})
        calls = len(self.calls)
        self.assertEqual(self.bridge.invoke({"operation": "end"}), result)
        self.assertEqual(len(self.calls), calls)
        with self.assertRaises(ValueError):
            self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-off"})
        with self.assertRaises(ValueError):
            self.bridge.invoke(dict(operation="ack", trial_id="nv2-test-light",
                                    capture_sha256="wrong-copy"))
        self.bridge.invoke(dict(operation="ack", trial_id="nv2-test-light",
                                capture_sha256=result["capture_sha256"]))
        self.bridge.invoke({"operation": "begin", "trial_id": "nv2-test-off"})
        self.assertEqual(self.bridge.active, "nv2-test-off")

    def test_malformed_completed_data_is_retained_instead_of_lost_in_validation(self):
        self.begin()
        self.reply["workers"][0]["rows"][0].pop("step_id")
        result = self.bridge.invoke({"operation": "end"})
        self.assertFalse(result["capture_valid"])
        self.assertEqual(result["capture"], self.reply)
        self.assertEqual(self.bridge.invoke({"operation": "end"}), result)


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        from fastapi import FastAPI
        self.utility = AsyncMock(return_value={"passive_reply": True})
        endpoint = Endpoint()
        endpoint.engine = SimpleNamespace(engine_core=SimpleNamespace(
            call_utility_async=self.utility))
        app = FastAPI()
        self.environment = patch.dict(os.environ, {
            "PR33_NATIVE_BATCH_CAPTURE_ENABLED": "1",
            "PR33_NATIVE_OBSERVER_KEY": "test-only-not-a-real-credential-00000000"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        endpoint.attach_router(app)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)
        self.headers = {"x-observer-key": "test-only-not-a-real-credential-00000000"}

    async def test_authentication_precedes_any_backend_utility(self):
        result = await self.client.post("/pr33-native/batch", json={"operation": "metadata"})
        self.assertEqual(result.status_code, 401)
        self.utility.assert_not_called()

    async def test_nonpassive_operations_cannot_reach_the_engine(self):
        result = await self.client.post("/pr33-native/batch", headers=self.headers,
                                        json={"operation": "start"})
        self.assertEqual(result.status_code, 400)
        self.utility.assert_not_called()

    async def test_authenticated_scalar_tree_reaches_only_passive_utility(self):
        result = await self.client.post("/pr33-native/batch", headers=self.headers,
                                        json={"operation": "metadata"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {"passive_reply": True})
        self.utility.assert_awaited_once_with("pr33_native_batch", {"operation": "metadata"})


if __name__ == "__main__":
    unittest.main()
