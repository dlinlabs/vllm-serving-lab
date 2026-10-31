"""Focused Round3 checks. Each case fails under the original claim it revises."""
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opus_revision as m  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
R = ROOT / "results/mpc-native-4090-scout"
PROFILE = R / "native-profile-diagnostic.json"
ARCHIVE = R / "native-identification-train-evidence.tar.gz"


def _bridge_available():
    try:
        m.load_bridge()
        return True
    except (subprocess.CalledProcessError, ValueError, OSError):
        return False


@unittest.skipUnless(PROFILE.exists(), "profile artifact missing")
class NestedKnee(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.out = m.nested_knee_loco(json.loads(PROFILE.read_text())["batches"])

    def test_round1_loco_mae_was_optimistic_because_knee_leaked(self):
        # Round1 reported 1.36 ms as held-out; with the knee chosen inside each fold it is worse.
        self.assertAlmostEqual(self.out["round1_leaked_knee"]["mae_ms"], 1.36, places=2)
        self.assertGreater(self.out["nested"]["mae_ms"], 1.25 * self.out["round1_leaked_knee"]["mae_ms"])
        self.assertGreater(len(self.out["nested"]["fold_knees"]), 1)

    def test_hinge_still_beats_loglinear_after_nesting(self):
        self.assertLess(self.out["nested"]["mae_ms"], self.out["loglinear_loco"]["mae_ms"])


@unittest.skipUnless(ARCHIVE.exists(), "TRAIN archive missing")
class BufferCapacity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bound = m.step_count_lower_bound(m.r1.load_train_archive(ARCHIVE))

    def test_short_prompt_90s_episode_nearly_fills_8192_before_prefill_and_drain(self):
        # "8192 rows suffice for a 90 s TRAIN tape" is not supported: the lower bound alone reaches >95%.
        self.assertEqual(len(self.bound["episodes"]), 16)
        self.assertGreater(self.bound["by_cell"]["p128-m256"]["max_fraction"], 0.95)
        self.assertLess(self.bound["by_cell"]["p128-m256"]["max"], m.BUFFER_CAPACITY)  # a bound, not an overflow claim

    def test_half_buffer_tape_limit_is_below_90s(self):
        self.assertLess(m.max_safe_tape_s(self.bound), 60.0)


@unittest.skipUnless(_bridge_available(), "reviewed bridge object 1cbe251 unavailable")
class BridgeWitnesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = m.bridge_witnesses()

    def test_unexpected_request_passes_bridge_but_fails_manifest_reconciliation(self):
        self.assertEqual(self.w["unexpected_request"]["bridge_errors"], [])
        self.assertTrue(self.w["unexpected_request"]["reconciliation_errors"])

    def test_metadata_reaches_engine_while_trial_active(self):
        got = self.w["metadata_during_active_trial"]
        self.assertTrue(got["accepted"])
        self.assertEqual(got["engine_meta_calls_during_trial"], 1)

    def test_identical_ack_retry_after_lost_response_is_rejected(self):
        got = self.w["ack_retry_after_lost_response"]
        self.assertTrue(got["retry"].startswith("ValueError"))
        self.assertIsNone(got["status_unacknowledged"])  # recoverable only via a status round-trip

    def test_reconciliation_detects_missing_and_duplicate_offers(self):
        good = m._capture({"a": dict(finished=1.0, trace_id="x"), "b": dict(finished=2.0, trace_id="y")})
        self.assertEqual(m.reconcile_requests(good, ["x", "y"]), [])
        self.assertTrue(m.reconcile_requests(good, ["x", "y", "z"]))
        dup = m._capture({"a": dict(finished=1.0, trace_id="x"), "b": dict(finished=2.0, trace_id="x")})
        self.assertTrue(m.reconcile_requests(dup, ["x"]))
        chat = m._capture({"chatcmpl-" + "a" * 32 + "-0000abcd": dict(finished=1.0)})
        self.assertEqual(m.reconcile_requests(chat, ["a" * 32]), [])


if __name__ == "__main__":
    unittest.main()
