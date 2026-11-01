import copy
import io
import json
import tarfile
import unittest
from unittest.mock import patch

from astra_horizon_diagnostic import contrast, diagnose, read_train, score_bounds


def row(name, scheduled=0.0, first=0.1, terminal=1.0, tokens=21):
    return dict(request_id=name, scheduled_at=scheduled, first_content_at=first,
                terminal_at=terminal, status="complete",
                usage={"completion_tokens": tokens})


def episode(arm, rows):
    return dict(seed_block=160001, arm=arm, rows=rows, start=0.0,
                cell="p128-m128", offers_s=90)


class TailAccountingTests(unittest.TestCase):
    def test_future_terminal_labels_do_not_leak(self):
        a = row("a", terminal=8, tokens=200)
        b = dict(a, status="rejected", usage=None)
        self.assertEqual(score_bounds([a], 2), score_bounds([b], 2))
        self.assertEqual(score_bounds([a], 2)["lower"], 0)
        self.assertEqual(score_bounds([a], 2)["upper"], 1)

    def test_future_first_time_does_not_leak(self):
        a = row("a", first=0.4)
        b = dict(a, first_content_at=0.9)
        c = dict(a, first_content_at=None)
        self.assertEqual(score_bounds([a], 0.2), score_bounds([b], 0.2))
        self.assertEqual(score_bounds([a], 0.2), score_bounds([c], 0.2))
        self.assertEqual(score_bounds([b], 0.5)["upper"], 0)

    def test_first_deadline_and_terminal_boundaries(self):
        rows = [row("a", first=0.5), row("b", first=0.50001)]
        bounds = score_bounds(rows, 0.5)
        self.assertEqual((bounds["lower"], bounds["upper"]), (0, 1))
        self.assertEqual(bounds["pending_ttft_impossible"], 1)
        self.assertEqual(score_bounds(rows, 1)["lower"], 1)

    def test_tpot_failure_rejection_and_one_token_convention(self):
        rejected = dict(row("reject"), status="rejected", usage=None)
        rows = [row("slow", terminal=5), row("one", terminal=5, tokens=1), rejected]
        self.assertEqual(score_bounds(rows, 5)["lower"], 1)
        self.assertEqual(score_bounds(rows, 0.2)["lower"], 0)

    def test_truncated_ranking_can_reverse(self):
        a = [row("a0", terminal=0.4), row("a1", terminal=3, tokens=100),
             row("a2", terminal=3, tokens=100)]
        b = [row("b0", terminal=0.4), row("b1", terminal=0.4)]
        early = contrast(score_bounds(a, 1), score_bounds(b, 1))
        self.assertEqual(early, dict(lower=-1, upper=1, verdict="unresolved"))
        self.assertEqual(contrast(score_bounds(a, 3), score_bounds(b, 3))["verdict"], "arm0")

    def test_bound_is_monotone_and_contains_eventual_score(self):
        rows = [row("good"), row("slow", terminal=5),
                row("miss", first=0.7), dict(row("reject"), status="rejected", usage=None)]
        truth = score_bounds(rows, 6)["lower"]
        previous = (0, len(rows))
        for cut in (0, 0.1, 0.5, 0.7, 1, 2, 5, 6):
            bounds = score_bounds(rows, cut)
            self.assertLessEqual(bounds["lower"], truth)
            self.assertGreaterEqual(bounds["upper"], truth)
            self.assertGreaterEqual(bounds["lower"], previous[0])
            self.assertLessEqual(bounds["upper"], previous[1])
            previous = (bounds["lower"], bounds["upper"])

    def test_common_cohort_half_open_boundary_and_tail_only_changes_scoring(self):
        rows = [row("left", scheduled=1, first=1.1, terminal=3, tokens=100),
                row("right", scheduled=2, first=2.1, terminal=3)]
        result = diagnose([episode(0, rows), episode(1, copy.deepcopy(rows))],
                          cohort_start=1, cohort_duration=1, tails=(0, 2))
        pair = result["pairs"][0]
        self.assertEqual(pair["offered_per_arm"], 1)
        self.assertEqual([c["arms"][0]["lower"] for c in pair["cuts"]], [0, 1])
        self.assertEqual(pair["cuts"][0]["arm1_minus_arm0"]["verdict"], "unresolved")
        self.assertEqual(pair["cuts"][1]["arm1_minus_arm0"]["verdict"], "tie")

    def test_invalid_windows_and_unpaired_arms_fail_closed(self):
        for kwargs in (dict(tails=(-1,)), dict(tails=(float("nan"),)),
                       dict(cohort_duration=0), dict(cohort_duration=91)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                diagnose([episode(0, [])], **kwargs)
        with self.assertRaisesRegex(ValueError, "missing paired"):
            diagnose([episode(0, [])])

    def test_reserved_seed_is_rejected_before_outcome_access(self):
        payload = json.dumps(dict(completed=True, episodes=[
            dict(seed_block=160021, arm=0)
        ])).encode()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo("native-id-train/interventions.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
            with patch("astra_horizon_diagnostic.tarfile.open", return_value=archive):
                with self.assertRaisesRegex(ValueError, "TRAIN paired blocks differ"):
                    read_train("unused")


if __name__ == "__main__":
    unittest.main()
