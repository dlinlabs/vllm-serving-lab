"""Run only the selected mock controls, retaining subtest failures and JUnit."""

import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import test_native_services_smoke as controls


phase = sys.argv[1]
names = [
    "test_10_nonzero_body_ledger_closure",
    "test_11_invalid_body_ledger_closure",
    "test_12_no_body_nonzero_ledger_still_refused",
    "test_13_telemetry_and_post_gpu_failures_both_retained",
]
if phase == "red01":
    suite = unittest.TestSuite(controls.Controls(name) for name in names)
elif phase == "green01":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(controls.Controls)
else:
    raise ValueError("unplanned phase")
tests = list(suite)
result = unittest.TextTestRunner(verbosity=2).run(suite)
summary = dict(phase=phase, tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
               failure_ids=[test.id() for test, _ in result.failures],
               error_ids=[test.id() for test, _ in result.errors],
               evidence_kind="CPU_MOCK_ONLY_NOT_NATIVE_READINESS_OR_CYCLES", python=sys.version,
               driver_file=controls.driver.__file__,
               driver_sha256=hashlib.sha256(Path(controls.driver.__file__).read_bytes()).hexdigest(),
               test_sha256=hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest(),
               actual_service_HTTP_GPU_model_calls=0)
output = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"])
(output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
xml = ET.Element("testsuite", tests=str(result.testsRun), failures=str(len(result.failures)),
                 errors=str(len(result.errors)), name="B2-" + phase)
for test in tests:
    case = ET.SubElement(xml, "testcase", name=test._testMethodName)
    for kind, entries in (("failure", result.failures), ("error", result.errors)):
        for failed, text in entries:
            if failed.id() == test.id() or failed.id().startswith(test.id() + " "):
                ET.SubElement(case, kind, message=failed.id()).text = text
(output / "junit.xml").write_bytes(ET.tostring(xml, encoding="utf-8", xml_declaration=True))
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
