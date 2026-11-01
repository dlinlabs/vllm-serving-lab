"""Focused tests for opus_batch_models (stdlib unittest; also collected by pytest).

Run from repo root:
    python3 -m unittest discover -s experiments/mpc-research/native-v2/round1 -p 'test_opus_*.py'
"""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opus_batch_models as m  # noqa: E402

PROFILE = Path(__file__).resolve().parents[4] / "results/mpc-native-4090-scout/native-profile-diagnostic.json"


def batch(s, d=0, kv=0, qk=0, pre=0, busy=None):
    b = dict(scheduled_tokens=s, decode_requests=d, decode_key_tokens_sum=kv,
             prefill_query_key_product_sum=qk, prefill_requests=pre, prefill_query_tokens=s - d)
    if busy is not None:
        b["kernel_busy_union_us"] = busy
    return b


def hinge_law(knee=128, a=8.0, slope=50.0):
    return dict(knee_tokens=knee, target="kernel_busy_union_us", intercept_ms=a, per_decode_request_ms=0.0,
                per_kilo_decode_kv_ms=0.0, per_kilo_token_above_knee_ms=slope, per_mega_prefill_qk_ms=0.0)


class NNLS(unittest.TestCase):
    def test_negative_unconstrained_coefficient_is_clamped(self):
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [3, 2, 1, 0]   # unconstrained slope -1
        beta = m.nnls(X, y)
        self.assertAlmostEqual(beta[1], 0.0)
        self.assertAlmostEqual(beta[0], 1.5)


class IterationLaw(unittest.TestCase):
    def synthetic(self, sizes):
        law = hinge_law()
        return [batch(s, busy=1000 * m.predict_ms(law, batch(s))) for s in sizes for _ in range(2)]

    def test_knee_recovered_only_when_design_brackets_it(self):
        dense = m.knee_profile(self.synthetic([1, 4, 64, 96, 128, 160, 192, 256, 512, 2048]))
        self.assertEqual(dense["best_knee"], 128)
        self.assertTrue(dense["knee_identified"])
        # Same truth, but no batch sizes between 4 and 2048: knee must not be claimed.
        sparse = m.knee_profile(self.synthetic([1, 4, 2048]))
        self.assertFalse(sparse["knee_identified"])

    def test_batch_law_beats_per_request_serial_on_shared_memory_bound_batches(self):
        # Truth: decode steps cost the same floor regardless of batch size.
        law = hinge_law()
        rows = []
        for d in (1, 4, 8, 16, 32):
            for _ in range(3):
                rows.append(batch(d, d=d, busy=1000 * m.predict_ms(law, batch(d, d=d))))
        for s in (1024, 2048):   # two above-knee cells so either held-out one is identifiable
            rows += [batch(s, pre=1, busy=1000 * m.predict_ms(law, batch(s)))] * 3
        loco = m.leave_one_condition_out(rows, 128)
        self.assertLess(loco["hinge_batch_law"]["mae_ms"], 1e-6)
        self.assertGreater(loco["per_request_serial"]["mae_ms"], 5.0)

    @unittest.skipUnless(PROFILE.exists(), "profile artifact not present")
    def test_real_profile_rejects_per_request_serial_service(self):
        batches = json.loads(PROFILE.read_text())["batches"]
        knee = m.knee_profile(batches)
        self.assertFalse(knee["knee_identified"])   # no batches between 13 and 128 tokens
        loco = m.leave_one_condition_out(batches, knee["best_knee"])
        self.assertLess(loco["hinge_batch_law"]["mae_ms"], loco["per_request_serial"]["mae_ms"] / 5)


class Capacity(unittest.TestCase):
    def test_batching_admits_more_completions_than_exclusive_serial(self):
        law = hinge_law(knee=128, a=10.0, slope=50.0)
        batch_bound = m.batch_completion_bound(90, 128, 45, law)
        serial = m.serial_completion_bound(90, 0.0449)
        self.assertEqual(serial, 2004)
        self.assertGreater(batch_bound, 2449)   # observed 2449 is feasible under batching

    def test_compute_bound_prompts_cannot_share_and_bound_is_monotone(self):
        law = hinge_law(knee=128, a=8.0, slope=50.0)
        long_bound = m.batch_completion_bound(90, 4096, 50, law)
        # ~ (4096-128)*50us per request of non-shareable compute -> < 90/0.198
        self.assertLessEqual(long_bound, int(90 / ((4096 - 128) * 50e-6)) + 1)
        t = [m.batch_time_lower_bound_s(n, 4096, 50, law) for n in (1, 10, 100)]
        self.assertEqual(t, sorted(t))

    def test_serial_service_must_be_positive(self):
        with self.assertRaises(ValueError):
            m.serial_completion_bound(90, 0)


class PrefillInterference(unittest.TestCase):
    def test_chunk_plan_respects_decode_first_token_budget(self):
        self.assertEqual(m.chunk_plan(4096, 1), [(2047, 2047), (2047, 4094), (2, 4096)])
        with self.assertRaises(ValueError):
            m.chunk_plan(10, 2048)

    def test_tpot_insertion_limit_matches_closed_form(self):
        law = hinge_law(knee=0, a=0.0, slope=50.0)   # chunk ms = 50*q/1000
        extra = sum(max(0.0, 50 * (q + 1) / 1000 - 12) for q, _ in m.chunk_plan(4096, 1))
        k = m.max_insertions_under_tpot(law, 12.0, 50, 4096, decode_running=1)
        self.assertEqual(k, math.floor((50 - 12) * 49 / extra))
        self.assertEqual(m.interference_tpot_ms(law, 12.0, 1, 5, 4096), 0.0)

    def test_tpot_limit_is_infinite_when_prefill_is_cheaper_than_decode(self):
        law = hinge_law(knee=4096, a=5.0, slope=0.0)
        self.assertEqual(m.max_insertions_under_tpot(law, 12.0, 50, 128), math.inf)

    def test_backlog_is_additive_in_compute_bound_regime(self):
        law = hinge_law(knee=0, a=0.0, slope=50.0)
        one = m.prefill_backlog_s(law, [4096])
        self.assertAlmostEqual(m.prefill_backlog_s(law, [4096] * 3), 3 * one)


class RemainingWork(unittest.TestCase):
    def test_mean_residual_life_of_deterministic_length(self):
        mrl = m.mean_residual_life([10, 10, 10])
        self.assertEqual(mrl[0], 10)
        self.assertEqual(mrl[7], 3)

    def test_age_is_informative_for_narrow_lengths_held_out(self):
        rows = m.age_value_held_out({1: [40, 45, 50] * 5, 2: [41, 46, 49] * 5})
        for r in rows:
            self.assertLess(r["mae_age_conditional"], r["mae_unconditional"] / 2)

    def test_paired_agreement_counts_batch_dependent_eos(self):
        out = m.paired_length_agreement([(40, 40), (40, 52), (30, 30), (30, 31)])
        self.assertEqual(out["identical_fraction"], 0.5)
        self.assertEqual(out["over_10pct_fraction"], 0.25)
        with self.assertRaises(ValueError):
            m.paired_length_agreement([])


class Guards(unittest.TestCase):
    def test_heldout_seed_is_refused(self):
        doc = dict(episodes=[dict(seed_block=160022, split="train")])
        with self.assertRaisesRegex(ValueError, "held-out"):
            m.train_episodes(doc)
        with self.assertRaisesRegex(ValueError, "Train"):
            m.train_episodes(dict(episodes=[dict(seed_block=160001, split="validation")]))

    def test_ttft_by_prefill_ahead_counts_only_unfinished_earlier_sends(self):
        rows = [dict(status="complete", sent_at=1.0, first_content_at=1.5),
                dict(status="complete", sent_at=1.1, first_content_at=1.9),
                dict(status="complete", sent_at=1.6, first_content_at=2.0),
                dict(status="rejected", sent_at=1.2, first_content_at=None)]
        ep = dict(prompt_tokens=4096, start=0.0, rows=rows)
        table = {r["ahead"]: r for r in m.ttft_by_prefill_ahead([ep], 4096)}
        self.assertEqual(table[0]["requests"], 1)
        self.assertEqual(table[1]["requests"], 2)   # 1.1 sees 1.0 pending; 1.6 sees 1.1 pending
        self.assertAlmostEqual(table[1]["within_slo_fraction"], 0.5)   # 0.8s miss, 0.4s hit

    def test_kv_fraction_through_origin(self):
        frames = [dict(complete=True, kv_cache_usage=0.1, requests=[dict(prompt_tokens=900, output_tokens=100)]),
                  dict(complete=True, kv_cache_usage=0.2, requests=[dict(prompt_tokens=1900, output_tokens=100)])]
        self.assertAlmostEqual(m.kv_fraction_per_token(frames)["tokens_at_full"], 10000)


if __name__ == "__main__":
    unittest.main()
