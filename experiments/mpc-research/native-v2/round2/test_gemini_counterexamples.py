"""Unit tests for gemini_counterexamples.py."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from gemini_counterexamples import (
    ActiveStream,
    BiAffineModel,
    RooflineHingeModel,
    compare_batch_scaling,
    evaluate_measurement_strategies,
    evaluate_reservation_release,
)


class TestRooflineModel(unittest.TestCase):
    def test_bi_affine_failure_on_large_prefill(self):
        bi_affine = BiAffineModel()
        # Prefill 2048 tokens: 12.31 + 0.00873 * 2048 = 30.189 ms
        pred = bi_affine.predict_ms(prefill_tokens=2048, decode_requests=0)
        actual = 112.46 # Profiled median kernel busy for p2048
        # Error is massive (> 80ms)
        self.assertGreater(abs(pred - actual), 80.0)

    def test_roofline_accuracy_across_regimes(self):
        roofline = RooflineHingeModel()
        # Decode 1: 8.50 + 0.35 * 1 = 8.85 ms (close to actual 9.38 ms)
        self.assertAlmostEqual(roofline.predict_ms(0, 1), 8.85, places=2)
        # Prefill 128: 8.50 + 0 (below knee 128) = 8.50 ms (close to actual 10.36 ms)
        self.assertAlmostEqual(roofline.predict_ms(128, 0), 8.50, places=2)
        # Prefill 2048: 8.50 + 0.051 * (2048 - 128) = 8.50 + 97.92 = 106.42 ms (close to actual 112.46 ms)
        self.assertAlmostEqual(roofline.predict_ms(2048, 0), 106.42, places=2)

    def test_batch_scaling_comparison_function(self):
        comp = compare_batch_scaling()
        self.assertIn("comparison", comp)
        self.assertIn("decode_1", comp["comparison"])
        self.assertIn("prefill_2048", comp["comparison"])
        self.assertIn("Bi-affine model fitted on decode survivors fails", comp["conclusion"])


class TestReservationReleaseAmbiguity(unittest.TestCase):
    def test_identical_backlog_produces_divergent_admissions(self):
        # Two worlds with IDENTICAL total tokens (12) and IDENTICAL total reserved cost (16.0)
        # World A: 2 short requests (1 token remaining) + 1 long (10 tokens remaining)
        world_a = [
            ActiveStream("req-a1", cost=4.0, remaining_tokens=1),
            ActiveStream("req-a2", cost=4.0, remaining_tokens=1),
            ActiveStream("req-a3", cost=8.0, remaining_tokens=10),
        ]
        # World B: 3 balanced requests (4 tokens remaining each)
        world_b = [
            ActiveStream("req-b1", cost=4.0, remaining_tokens=4),
            ActiveStream("req-b2", cost=4.0, remaining_tokens=4),
            ActiveStream("req-b3", cost=8.0, remaining_tokens=4),
        ]

        result = evaluate_reservation_release(
            streams_world_a=world_a,
            streams_world_b=world_b,
            horizon_steps=2,
            tokens_per_step=1,
            budget=16.0,
            incoming_offer_cost=4.0,
        )

        # After 2 steps:
        # World A: req-a1 and req-a2 completed -> released 8.0 cost. Future reserved = 8.0.
        # Can admit offer of cost 4.0 (8.0 + 4.0 = 12.0 <= 16.0).
        self.assertEqual(result["world_a"]["released_cost"], 8.0)
        self.assertEqual(result["world_a"]["future_reserved_cost"], 8.0)
        self.assertTrue(result["world_a"]["can_admit_offer"])

        # World B: no requests completed -> released 0.0 cost. Future reserved = 16.0.
        # Cannot admit offer of cost 4.0 (16.0 + 4.0 = 20.0 > 16.0).
        self.assertEqual(result["world_b"]["released_cost"], 0.0)
        self.assertEqual(result["world_b"]["future_reserved_cost"], 16.0)
        self.assertFalse(result["world_b"]["can_admit_offer"])

        # Proves decisions diverge despite identical aggregate backlog!
        self.assertTrue(result["decisions_diverge"])
        self.assertEqual(result["cost_ambiguity"], 8.0)


class TestMeasurementFacadeEvaluation(unittest.TestCase):
    def test_mayor_facade_superiority(self):
        strategies = evaluate_measurement_strategies()
        mayor_facade = strategies["mayor_default_off_facade"]
        cupti = strategies["runtime_cupti_tracing"]
        sync = strategies["blocking_cuda_sync"]

        self.assertTrue(mayor_facade.is_non_blocking_during_serving)
        self.assertLess(mayor_facade.estimated_host_overhead_us_per_batch, 10.0)
        self.assertTrue(mayor_facade.requires_post_drain_resolution)
        self.assertIn("RECOMMENDED", mayor_facade.verdict)

        self.assertIn("REJECTED", cupti.verdict)
        self.assertIn("PROHIBITED", sync.verdict)


if __name__ == "__main__":
    unittest.main()
