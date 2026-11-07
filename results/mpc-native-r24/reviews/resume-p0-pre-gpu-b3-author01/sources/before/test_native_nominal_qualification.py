"""CPU-only entry and failed-capture persistence controls."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import native_nominal_qualification as driver
import nominal_client


ROOT = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"]) / "qualification"
ROOT.mkdir(parents=True, exist_ok=True)


class Controls(unittest.TestCase):
    def test_default_off_never_enters_native_body(self):
        forbidden = AssertionError("OFF performed an effect")
        with patch.object(driver, "body", side_effect=forbidden), \
                patch.object(driver.services, "source_info", side_effect=forbidden), \
                patch.object(driver.services, "gpu_receipt", side_effect=forbidden), \
                patch.object(Path, "mkdir", side_effect=forbidden):
            self.assertEqual(driver.main([]), 2)

    def test_nonfinite_capture_write_keeps_error_receipt_and_fails(self):
        path = ROOT / "nonfinite-capture.json"
        with self.assertRaises(ValueError):
            driver.persist_capture(nominal_client, path, {"nonconforming": float("inf")}, "")
        receipt = json.loads(path.with_suffix(".write-error.json").read_text())
        self.assertTrue(receipt["capture_unscorable"])
        self.assertEqual(receipt["type"], "ValueError")
        self.assertFalse(path.exists())
