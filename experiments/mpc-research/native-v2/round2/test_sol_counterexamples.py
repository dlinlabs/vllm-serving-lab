"""Reproduce concrete review witnesses against immutable peer source blobs."""

import unittest

from sol_counterexamples import (
    decomposition_witness, opus_metric_witness, persistent_but_no_multistep_gain,
    reservation_witness, statistical_witness, tail_witness,
)


class FrozenPeerWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.statistics = statistical_witness()

    def test_one_sided_result_contradicts_prespecified_two_sided_rule(self):
        case = self.statistics
        self.assertEqual(case["constructed_differences_req_s"], [1.] * 7 + [-1.])
        self.assertEqual(case["peer_result"]["permutation_p_value"], 9 / 256)
        self.assertEqual(case["prespecified_two_sided_sign_flip_p"], 18 / 256)
        self.assertEqual(case["peer_result"]["verdict"], "CONFIRMED_SUPERIOR")
        self.assertGreater(case["prespecified_two_sided_sign_flip_p"], .05)

    def test_duplicate_block_is_not_rejected_as_pseudoreplication(self):
        result = self.statistics["duplicate_same_block_result"]
        self.assertEqual(result["n_blocks"], 8)
        self.assertEqual(result["permutation_p_value"], 1 / 256)
        self.assertEqual(result["verdict"], "CONFIRMED_SUPERIOR")

    def test_rejection_and_solver_denominator_guards_are_not_enforced(self):
        case = self.statistics
        self.assertTrue(case["bad_guard_result"]["guards_passed"])
        self.assertEqual(case["bad_guard_result"]["solver_overrun_rate"], .002)
        self.assertEqual(case["bad_guard_rejection_rate_ratio_B_over_A"], 100)
        self.assertGreater(
            case["bad_guard_true_overrun_fraction_if_10_solves_per_block"], .01)

    def test_cap_reduction_invalidates_current_budget_invariant(self):
        case = reservation_witness()
        self.assertGreater(case["reserved_after_non_evicting_reduction"],
                           case["gemini_current_budget_bound"])
        self.assertEqual(case["reserved_after_non_evicting_reduction"],
                         case["correct_history_budget_bound"])

    def test_missing_lifecycle_intervals_break_exact_decomposition(self):
        case = decomposition_witness()
        self.assertAlmostEqual(case["actual_ttft_s"], .45)
        self.assertAlmostEqual(case["claimed_decomposition_sum_s"], .30)
        self.assertAlmostEqual(case["omitted_forwarding_interchunk_sampling_s"], .15)

    def test_long_state_memory_is_not_sufficient_for_multistep_gain(self):
        case = persistent_but_no_multistep_gain()
        self.assertGreater(case["reservation_holding_slots"], 1)
        self.assertEqual(case["high_plan_admissions_per_slot"], [20, 12, 20])
        self.assertEqual(case["best_three_move_good_count"], 52)
        self.assertEqual(case["one_move_with_common_high_backup_good_count"], 52)

    def test_clock_origin_and_disagreement_frequency_do_not_bound_goodput(self):
        case = opus_metric_witness()
        self.assertEqual(case["peer_send_origin_slo_fraction"], 1)
        self.assertEqual(case["scheduled_origin_slo_fraction"], 0)
        self.assertLess(case["shadow_disagreement_fraction"], .05)
        self.assertGreater(
            case["possible_relative_gain_despite_one_percent_disagreement"], .20)

    def test_sound_tail_bounds_do_not_justify_maximizing_lower_bound(self):
        case = tail_witness()
        self.assertEqual(case["early_A"]["offered"], case["early_B"]["offered"])
        self.assertGreater(case["early_B"]["lower"], case["early_A"]["lower"])
        self.assertEqual(case["early_B_minus_A"]["verdict"], "unresolved")
        self.assertGreater(case["eventual_A"], case["eventual_B"])
        difference = case["eventual_B"] - case["eventual_A"]
        self.assertLessEqual(case["early_B_minus_A"]["lower"], difference)
        self.assertGreaterEqual(case["early_B_minus_A"]["upper"], difference)


if __name__ == "__main__":
    unittest.main()
