import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import prepare_probes as probes


TRAIN = probes.ROOT / "results/mpc-native-r24/reviews/resume-p0-tokenizer02-raw/returned/train.replay.json"


class PrepareProbesTests(unittest.TestCase):
    def test_derivation_preserves_measured_prompts_and_declares_engineering_scope(self):
        raw = TRAIN.read_bytes()
        train = json.loads(raw)
        original = {row["trace_id"]: row for row in train["requests"]}
        outputs = probes.build_probes(raw)
        for name, count, offsets in (
            ("warmup.json", 3, [0.0, 1.0, 2.0]),
            ("qualification.json", 4, [0.25, 1.25, 2.25, 3.25]),
            ("admission-boundary.json", 5, [1.25] * 5),
        ):
            trace = probes.tw.validate_trace(json.loads(outputs[name]))
            self.assertEqual(len(trace["requests"]), count)
            self.assertEqual([row["arrival_seconds"] for row in trace["requests"]], offsets)
            self.assertEqual(trace["provenance"]["purpose"], "engineering")
            self.assertTrue(trace["provenance"]["not_a_held_out_or_performance_comparison"])
            for row, source_id in zip(trace["requests"], trace["provenance"]["source_request_ids"]):
                for key in ("prompt", "input_tokens", "max_tokens"):
                    self.assertEqual(row[key], original[source_id][key])
                self.assertNotEqual(row["trace_id"], source_id)
        self.assertEqual(
            [row["input_tokens"] for row in json.loads(outputs["warmup.json"])["requests"]],
            [256, 1024, 3072],
        )
        boundary = json.loads(outputs["admission-boundary.json"])
        self.assertEqual([row["input_tokens"] for row in boundary["requests"]], [3072] * 5)
        expected = [row["trace_id"] for row in train["requests"] if row["input_tokens"] == 3072][:5]
        self.assertEqual(boundary["provenance"]["source_request_ids"], expected)
        self.assertEqual(outputs, probes.build_probes(raw))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), probes.TRAIN_REPLAY_SHA256)

    def test_nonfrozen_or_tampered_source_refused(self):
        raw = TRAIN.read_bytes()
        for changed in (raw + b"\n", b"{}", raw.replace(b'"duration_seconds":5.0', b'"duration_seconds":6.0')):
            self.assertNotEqual(raw, changed)
            with self.assertRaisesRegex(ValueError, "frozen"):
                probes.build_probes(changed)

    def test_cli_writes_new_directory_and_never_clobbers(self):
        base = Path(os.environ["NOMINAL_PROBE_SCRATCH"])
        self.assertTrue(base.is_absolute())
        output = base / "probes"
        command = [
            sys.executable, "-B", "-I", str(probes.HERE / "prepare_probes.py"),
            "--train-replay", str(TRAIN), "--output-dir", str(output),
        ]
        first = subprocess.run(command, capture_output=True, timeout=15)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        second = subprocess.run(command, capture_output=True, timeout=15)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        self.assertEqual(sorted(before), ["admission-boundary.json", "qualification.json", "warmup.json"])


if __name__ == "__main__":
    unittest.main()
