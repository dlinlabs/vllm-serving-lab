"""Focused tests of the finite-witness diagnostic, not native controller tests."""

from dataclasses import replace
import math
import unittest

from sol_counterexamples import (
    DecodeWorld, available_at_decision, diagnostic, release_certificate,
)


class ReleaseAliasingTests(unittest.TestCase):
    def setUp(self):
        self.a = DecodeWorld((8., 8.), (1, 1), (16, 16), (1., 1.), (1, 9))
        self.b = replace(self.a, remaining_tokens=(5, 5))

    def certificate(self, worlds=None, horizon=2):
        return release_certificate(
            worlds or (self.a, self.b), horizon_s=horizon,
            token_rate=1, offer_cost=4, budgets=(16, 32))

    def test_equal_observed_and_total_work_does_not_determine_release(self):
        result = self.certificate()
        self.assertEqual(self.a.projection, self.b.projection)
        self.assertEqual(result["reserved_cost_by_world"], [8, 16])
        self.assertEqual(result["minimax_absolute_point_error_cost"], 4)
        self.assertEqual(result["offer_accepted_by_budget"], [
            {"budget": 16, "accepted_by_world": [True, False]},
            {"budget": 32, "accepted_by_world": [True, True]},
        ])
        for prediction in (0, 8, 12, 16, 20):
            self.assertGreaterEqual(
                max(abs(prediction - r) for r in (8, 16)), 4)

    def test_whole_release_at_eos_and_no_release_before(self):
        self.assertEqual(self.a.reserved_after(.999, 1), 16)
        self.assertEqual(self.a.reserved_after(1, 1), 8)
        self.assertEqual(self.a.reserved_after(9, 1), 0)
        self.assertEqual(self.b.reserved_after(5, 1), 0)
        self.assertEqual(self.a.reserved_after(.5, 2), 8)

    def test_horizon_can_remove_or_change_ambiguity(self):
        self.assertEqual(self.certificate(horizon=0)[
            "minimax_absolute_point_error_cost"], 0)
        self.assertEqual(self.certificate(horizon=9)[
            "minimax_absolute_point_error_cost"], 0)
        self.assertEqual(self.certificate(horizon=5)[
            "reserved_cost_by_world"], [8, 0])

    def test_different_projection_is_not_an_alias_certificate(self):
        for other in (replace(self.b, ages_s=(2., 1.)),
                      replace(self.b, remaining_tokens=(5, 6)),
                      replace(self.b, costs=(7., 8.))):
            with self.subTest(other=other), self.assertRaises(ValueError):
                self.certificate((self.a, other))

    def test_invalid_or_over_cap_worlds_are_rejected(self):
        for changes in (dict(remaining_tokens=(0, 10)),
                        dict(remaining_tokens=(16, 1)),
                        dict(output_so_far=(True, 1)),
                        dict(costs=(math.nan, 8)),
                        dict(costs=(-1, 8)),
                        dict(ages_s=(math.inf, 1)),
                        dict(output_caps=(16,)),
                        dict(remaining_tokens=[1, 9])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.a, **changes)
        for horizon in (-1, math.inf, math.nan):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                self.certificate(horizon=horizon)


class AvailabilityTests(unittest.TestCase):
    def available(self, **changes):
        values = dict(sampled_at=1., received_at=1.2, decision_at=1.1,
                      source_clock="boot", decision_clock="boot")
        return available_at_decision(**(values | changes))

    def test_old_source_sample_arriving_later_is_not_available(self):
        self.assertFalse(self.available())
        self.assertTrue(self.available(decision_at=1.2))

    def test_clock_mismatch_and_impossible_receipt_rejected(self):
        for changes in (dict(source_clock="other"),
                        dict(source_clock="", decision_clock=""),
                        dict(received_at=.5), dict(decision_at=math.nan)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.available(**changes)

    def test_diagnostic_labels_scope_and_no_native_gain(self):
        result = diagnostic()
        self.assertFalse(result["native_data_used"])
        self.assertFalse(result["native_performance_claim"])
        self.assertFalse(result["availability_counterexample"]["qualified_online_input"])


if __name__ == "__main__":
    unittest.main()
