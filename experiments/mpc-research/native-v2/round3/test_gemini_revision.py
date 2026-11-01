"""Unit tests for gemini_revision.py."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from gemini_revision import (
    MacroscopicQueuePredictor,
    RevisedEvaluationBlock,
    RevisedStatisticalProtocol,
    check_reservation_invariant,
    validate_request_outcome,
)


class TestReservationInvariant(unittest.TestCase):
    def test_budget_cut_remains_valid(self):
        # Start at 0, admit under 32, then budget cut to 16 while 28 cost is active.
        # Under old broken formula: R <= max(0, 16) = 16 -> FAILED.
        # Under corrected history-max formula: R <= max(0, 32) + 4.05 = 36.05 -> VALID.
        is_valid, bound = check_reservation_invariant(
            r_initial=0.0,
            applied_history=[32.0, 16.0],
            r_current=28.0,
            max_cost_step=4.05,
        )
        self.assertTrue(is_valid)
        self.assertEqual(bound, 36.05)

    def test_exceeding_history_max_fails(self):
        # History max is 16.0. Current R is 25.0 > 16.0 + 4.05 = 20.05.
        is_valid, bound = check_reservation_invariant(
            r_initial=0.0,
            applied_history=[16.0],
            r_current=25.0,
            max_cost_step=4.05,
        )
        self.assertFalse(is_valid)


class TestRevisedStatisticalProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = RevisedStatisticalProtocol(
            min_superiority_margin_req_s=0.5,
            max_solver_overrun_rate=0.01,
            max_relative_rejection_increase=0.05,
        )

    def test_k_less_than_six_raises(self):
        blocks = [
            RevisedEvaluationBlock(
                block_id=f"b-{i}", seed=170001+i, duration_s=90.0,
                control_decisions=90, offered_requests=500,
                policy_a_good_completions=200, policy_a_rejections=100, policy_a_solver_overruns=0,
                policy_b_good_completions=260, policy_b_rejections=100, policy_b_solver_overruns=0,
            )
            for i in range(4)
        ]
        with self.assertRaises(ValueError) as ctx:
            self.protocol.evaluate_blocks(blocks)
        self.assertIn("requires K >= 6", str(ctx.exception))

    def test_confirmed_superior_k6(self):
        # 6 blocks, each +60 good completions in 90s (+0.667 req/s > 0.5 margin)
        blocks = [
            RevisedEvaluationBlock(
                block_id=f"b-{i}", seed=170001+i, duration_s=90.0,
                control_decisions=90, offered_requests=500,
                policy_a_good_completions=200, policy_a_rejections=100, policy_a_solver_overruns=0,
                policy_b_good_completions=260, policy_b_rejections=100, policy_b_solver_overruns=0,
            )
            for i in range(6)
        ]
        res = self.protocol.evaluate_blocks(blocks)
        self.assertEqual(res["verdict"], "CONFIRMED_SUPERIOR")
        self.assertTrue(res["margin_satisfied"])
        self.assertTrue(res["statistically_significant"])
        # Exactly 2 out of 64 permutations have abs(mean) >= observed -> p = 2/64 = 0.03125
        self.assertAlmostEqual(res["two_sided_permutation_p_value"], 2 / 64)

    def test_decision_normalized_overrun_rejection(self):
        # 6 blocks, 90 decisions each = 540 decisions.
        # 10 overruns in policy B -> 10 / 540 = 0.0185 > 0.01 ceiling.
        blocks = [
            RevisedEvaluationBlock(
                block_id=f"b-{i}", seed=170001+i, duration_s=90.0,
                control_decisions=90, offered_requests=1000,
                policy_a_good_completions=200, policy_a_rejections=100, policy_a_solver_overruns=0,
                policy_b_good_completions=260, policy_b_rejections=100,
                policy_b_solver_overruns=(2 if i < 5 else 0), # Total 10 overruns
            )
            for i in range(6)
        ]
        res = self.protocol.evaluate_blocks(blocks)
        self.assertFalse(res["overrun_guard_passed"])
        self.assertEqual(res["verdict"], "EQUIVALENT_OR_INSUFFICIENT")

    def test_rejection_guard_triggers(self):
        # Policy B has 150 rejections vs Policy A 100 rejections -> +50% increase > 5% ceiling
        blocks = [
            RevisedEvaluationBlock(
                block_id=f"b-{i}", seed=170001+i, duration_s=90.0,
                control_decisions=90, offered_requests=500,
                policy_a_good_completions=200, policy_a_rejections=100, policy_a_solver_overruns=0,
                policy_b_good_completions=260, policy_b_rejections=150, policy_b_solver_overruns=0,
            )
            for i in range(6)
        ]
        res = self.protocol.evaluate_blocks(blocks)
        self.assertFalse(res["rejection_guard_passed"])
        self.assertEqual(res["verdict"], "EQUIVALENT_OR_INSUFFICIENT")


class TestValidateRequestOutcome(unittest.TestCase):
    def test_valid_stop_outcome(self):
        self.assertTrue(
            validate_request_outcome(
                scheduled_arrival=1.0,
                first_content_at=1.1,
                terminal_at=2.0,
                completion_tokens=50,
                finish_reason="stop",
                stream_error=False,
            )
        )

    def test_error_finish_reason_rejected(self):
        # Timely, but finished due to error/truncation -> MUST NOT count as good
        self.assertFalse(
            validate_request_outcome(
                scheduled_arrival=1.0,
                first_content_at=1.1,
                terminal_at=1.2,
                completion_tokens=5,
                finish_reason="error",
                stream_error=True,
            )
        )


class TestMacroscopicQueuePredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = MacroscopicQueuePredictor()

    def test_short_prompt_unaffected_by_queue(self):
        # Short prompt is flat ~ 66ms, safe with 0, 1, 2 ahead
        self.assertTrue(self.predictor.can_admit_safely(prompt_tokens=128, long_prefills_ahead=0))
        self.assertTrue(self.predictor.can_admit_safely(prompt_tokens=128, long_prefills_ahead=2))

    def test_long_prompt_delayed_by_queue(self):
        # 0 ahead: 0.310s <= 0.5s -> safe
        self.assertTrue(self.predictor.can_admit_safely(prompt_tokens=4096, long_prefills_ahead=0))
        # 1 ahead: 0.310 + 0.185 = 0.495s <= 0.5s -> safe
        self.assertTrue(self.predictor.can_admit_safely(prompt_tokens=4096, long_prefills_ahead=1))
        # 2 ahead: 0.310 + 0.370 = 0.680s > 0.5s -> UNSAFE!
        self.assertFalse(self.predictor.can_admit_safely(prompt_tokens=4096, long_prefills_ahead=2))


if __name__ == "__main__":
    unittest.main()
