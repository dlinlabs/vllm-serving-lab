"""Focused witnesses of what the bridge flag does and does not qualify."""

import unittest

from sol_revision import diagnostic


class BridgeReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = diagnostic()

    def test_equal_step_ids_do_not_validate_actual_executed_work(self):
        case = self.receipt["scope_cases"]["executed_work_mismatch"]
        self.assertTrue(case["bridge_capture_valid"])
        self.assertEqual(case["bridge_validation_errors"], [])
        self.assertTrue(case["raw_retained"])
        self.assertFalse(case["offline_audit"]["valid"])
        self.assertIn("Scheduler/worker work mismatch", case["offline_audit"]["problems"])

    def test_restricted_scheduler_parity_needs_a_separate_scope_gate(self):
        case = self.receipt["scope_cases"]["restricted_parity_mismatch"]
        self.assertTrue(case["bridge_capture_valid"])
        self.assertTrue(case["raw_retained"])
        self.assertFalse(case["offline_audit"]["valid"])
        self.assertIn("Installed schedule differs from restricted FIFO reference",
                      case["offline_audit"]["problems"])

    def test_matching_natural_eos_capture_passes_existing_audit(self):
        self.assertTrue(self.receipt["matching_capture_audit"]["valid"])

    def test_lost_end_and_ack_responses_have_distinct_reconciliation_paths(self):
        case = self.receipt["lost_response_workflow"]
        self.assertTrue(case["end_replays_exact_packet"])
        self.assertEqual(case["external_end_calls"], 1)
        self.assertTrue(case["duplicate_ack_rejected"])
        self.assertTrue(case["same_run"])
        self.assertIsNone(case["unacknowledged_trial"])
        self.assertTrue(case["consumed_trial_still_recorded"])


if __name__ == "__main__":
    unittest.main()
