import unittest

from astra_revision import bridge_for, completed, diagnostic, fixture


class BridgeAcceptanceWitnessTests(unittest.TestCase):
    def test_matching_keys_do_not_detect_wrong_executed_work(self):
        result = diagnostic()
        for field, witness in result["matching_keys_not_work_parity"].items():
            with self.subTest(field=field):
                self.assertTrue(witness["capture_valid"])
                self.assertEqual(witness["validation_errors"], [])
                self.assertTrue(witness["raw_mismatch_retained"])

    def test_end_replay_is_exact_without_another_native_end(self):
        bridge, calls, result = completed(fixture())
        before = list(calls)
        self.assertEqual(bridge.invoke({"operation": "end"}), result)
        self.assertEqual(calls, before)
        self.assertFalse(result["capture"]["workers"][0]["rows"][0]["cuda_scope_qualified"])

    def test_lost_ack_reply_requires_status_reconciliation(self):
        bridge, _, result = completed(fixture())
        ack = {"operation": "ack", "trial_id": "nv2-astra-check",
               "capture_sha256": result["capture_sha256"]}
        bridge.invoke(ack)
        with self.assertRaisesRegex(ValueError, "retained exact capture"):
            bridge.invoke(ack)
        status = bridge.invoke({"operation": "status"})
        self.assertEqual(status["run_id"], result["run_id"])
        self.assertIsNone(status["unacknowledged_trial"])
        self.assertIsNone(status["active_trial"])
        self.assertIn("nv2-astra-check", status["used_trials"])

    def test_worker_change_between_metadata_and_begin_is_rejected(self):
        bridge, calls, clocks = bridge_for(fixture())
        clocks[0]["process_id"] = 8
        before = list(calls)
        with self.assertRaisesRegex(ValueError, "epoch changed before"):
            bridge.invoke({"operation": "begin", "trial_id": "nv2-astra-check"})
        self.assertEqual(calls, before)
        self.assertTrue(bridge.invalid)


if __name__ == "__main__":
    unittest.main()
