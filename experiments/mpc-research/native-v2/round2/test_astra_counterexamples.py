from fractions import Fraction
import unittest

from astra_counterexamples import diagnostic, exact_signflip_p


class FrozenClaimWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnostic()

    def test_peer_protocol_accepts_broken_guardrails_and_duplicate_blocks(self):
        result = self.result["statistical_contract"]
        self.assertEqual(result["peer_verdict"], "CONFIRMED_SUPERIOR")
        self.assertEqual(result["eight_copies_of_one_block_verdict"], "CONFIRMED_SUPERIOR")
        self.assertLess(result["peer_solver_overrun_rate"], 0.01)
        self.assertGreater(result["overrun_rate_if_ten_decisions_per_block"], 0.01)
        self.assertGreater(result["rejection_fraction_B"], result["rejection_fraction_A"] * 1.05)

    def test_zero_null_significance_is_not_margin_evidence(self):
        result = self.result["statistical_contract"]
        self.assertAlmostEqual(result["mean_difference"], 0.5875)
        self.assertEqual(result["peer_p"], 1 / 256)
        self.assertEqual(result["reference_two_sided_zero_null_p"], 2 / 256)
        self.assertGreater(result["reference_one_sided_margin_null_p"], 0.05)
        self.assertEqual(exact_signflip_p([Fraction(0)] * 8), 1)

    def test_invalid_prefix_and_nonfinite_probe_are_accepted_by_frozen_peer(self):
        result = self.result["timing_contract"]
        self.assertTrue(result["invalid_trace_validated"])
        self.assertLess(result["negative_ttft_s"], 0)
        self.assertTrue(result["peer_composite_good"])
        self.assertTrue(result["nan_probe_accepted"])

    def test_warmup_must_not_remove_live_predecessors(self):
        result = self.result["warmup_history"]
        self.assertEqual(result, {"peer_target_ahead": 0, "actual_target_ahead": 1})

    def test_randomizing_assignment_does_not_randomize_composition(self):
        result = self.result["composition_confounding"]
        self.assertEqual(result["composition_regression_slope"], 0.5)
        self.assertEqual(result["true_do_composition_effect"], 0)

    def test_cap_reduction_preserves_existing_reservations(self):
        result = self.result["mathematical_claims"]
        self.assertGreater(result["reservation_after_cap_reduction"], result["peer_current_cap_bound"])
        self.assertEqual(result["reservation_after_cap_reduction"], result["correct_historical_cap_bound"])

    def test_batch_spans_do_not_partition_request_residence(self):
        result = self.result["mathematical_claims"]["missing_request_timing"]
        self.assertEqual(result["actual_ttft_ms"], 200)
        self.assertEqual(result["peer_named_terms_sum_ms"], 70)
        self.assertEqual(result["actual_ttft_ms"] - result["peer_named_terms_sum_ms"],
                         result["missing_bridge_wait_sample_ms"])

    def test_memory_and_unweighted_disagreement_do_not_prove_value(self):
        result = self.result["mathematical_claims"]
        memory = result["long_memory_no_value"]
        self.assertGreater(memory["request_residence_s"], memory["control_slot_s"])
        self.assertTrue(memory["all_budget_plans_accept_both"])
        self.assertEqual(memory["extra_good_from_more_moves"], 0)
        rare = result["rare_disagreement"]
        self.assertLess(rare["disagreement_fraction"], 0.05)
        self.assertGreater(rare["extra_good_per_second"], 0.5)


if __name__ == "__main__":
    unittest.main()
