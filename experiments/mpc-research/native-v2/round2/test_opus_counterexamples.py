import math
from pathlib import Path
import unittest

import opus_counterexamples as m

ROOT = Path(__file__).resolve().parents[4]
PROFILE = ROOT / "results/mpc-native-4090-scout/native-profile-diagnostic.json"
ARCHIVE = ROOT / "results/mpc-native-4090-scout/native-identification-train-evidence.tar.gz"


class LedgerAndStatistics(unittest.TestCase):
    def test_cap_reduction_breaks_current_budget_invariant_but_not_history_bound(self):
        w = m.reservation_invariant_witness()
        self.assertEqual(w["R"], 30.0)
        self.assertFalse(w["gemini_holds"])
        self.assertTrue(w["history_holds"])

    def test_four_blocks_cannot_reach_005_even_when_every_difference_favours_b(self):
        self.assertEqual(m.sign_flip_p([1.0, 2.0, 3.0, 4.0]), m.sign_flip_p_floor(4))
        self.assertGreater(m.sign_flip_p_floor(4), 0.05)
        self.assertGreater(m.sign_flip_p_floor(5, two_sided=True), 0.05)
        self.assertLess(m.sign_flip_p_floor(5), 0.05)
        self.assertEqual(m.sign_flip_p([1.0, -1.0, 2.0, -2.0], two_sided=True), 1.0)


class HorizonWitness(unittest.TestCase):
    def test_memory_longer_than_slot_does_not_imply_multistep_value(self):
        r = m.horizon_witness()["results"]
        self.assertEqual(r["memory_two_slots/optimal_dp"]["disagreements"], 0)

    def test_value_of_extra_moves_comes_from_a_suboptimal_backup(self):
        r = m.horizon_witness()["results"]
        strict = [e for e in r["memory_two_slots/fixed_high"]["examples"]
                  if len(e["n1"]) == 1 and len(e["n3"]) == 1 and e["n1"] != e["n3"]]
        self.assertTrue(strict)

    def test_without_carry_over_any_backup_gives_identical_first_actions(self):
        r = m.horizon_witness()["results"]
        for b in ("optimal_dp", "fixed_low", "fixed_high"):
            self.assertEqual(r[f"memoryless/{b}"]["disagreements"], 0)


class ComputeFloor(unittest.TestCase):
    def test_qwen3_4b_matmul_parameter_count(self):
        self.assertAlmostEqual(m.matmul_params() / 1e9, 3.633, places=3)

    def test_profiled_chunk_is_feasible_and_gemini_thetaP_is_not(self):
        self.assertLess(m.required_tflops(2048, 2048, 103.0), m.RTX4090_BF16_FP32ACC_TFLOPS)
        it = m.gemini_affine_iteration_ms(0, 4096)
        self.assertGreater(m.required_tflops(2048, 2048, it), 2 * m.RTX4090_BF16_FP32ACC_TFLOPS)

    def test_floor_is_monotone_in_tokens_and_context(self):
        self.assertLess(m.flop_floor_ms(512, 512), m.flop_floor_ms(1024, 1024))
        self.assertLess(m.flop_floor_ms(512, 512), m.flop_floor_ms(512, 4096))
        with self.assertRaises(ValueError):
            m.prefill_flops(10, 5)


class DecodeWall(unittest.TestCase):
    def test_iteration_counter_recovers_synthetic_law(self):
        rows = [dict(seed_block=b, prompt_tokens=128, iterations=n, decode=d, kv_tokens=k,
                     dt_ms=n * (9.0 + 0.05 * d + 0.2 * k / 1000))
                for b in (1, 2) for n in (3, 7) for d in (1, 5, 17) for k in (300, 4000)]
        fit = m.fit_decode_wall(rows)
        self.assertAlmostEqual(fit["intercept_ms"], 9.0, places=4)
        self.assertAlmostEqual(fit["per_decode_request_ms"], 0.05, places=5)
        self.assertAlmostEqual(fit["per_kilo_kv_ms"], 0.2, places=4)

    @unittest.skipUnless(ARCHIVE.exists() and PROFILE.exists(), "TRAIN artifacts absent")
    def test_train_frames_refute_round1_host_gap_and_profiled_decode_slope(self):
        import json
        law = m.r1.fit_law(json.loads(PROFILE.read_text())["batches"], 32)
        rows = m.decode_wall_intervals(m.r1.load_train_archive(ARCHIVE))
        self.assertGreater(len(rows), 1000)
        lobo = m.decode_wall_lobo(rows, law)
        best = lobo["unprofiled_fit_lobo"]["mae_ms"]
        for other in ("profiled_kernel_busy_law", "profiled_law_plus_round1_F3_host_gap",
                      "token_model_theta0_thetaD"):
            self.assertLess(best, lobo[other]["mae_ms"])
        self.assertLess(m.fit_decode_wall(rows)["per_decode_request_ms"], law["per_decode_request_ms"] / 3)


class LogLinear(unittest.TestCase):
    @unittest.skipUnless(PROFILE.exists(), "profile artifact absent")
    def test_hinge_beats_loglinear_out_of_composition_and_loglinear_breaks_flop_floor(self):
        import json
        r = m.loglinear_vs_hinge_loco(json.loads(PROFILE.read_text())["batches"])
        self.assertLess(r["hinge"]["mae_ms"], r["loglinear"]["mae_ms"])
        mid = next(c for c in r["curve"] if c["prefill_tokens"] == 1024)
        self.assertLess(mid["loglinear_ms"], mid["flop_floor_ms"])
        self.assertTrue(math.isfinite(mid["hinge_ms"]))


if __name__ == "__main__":
    unittest.main()
