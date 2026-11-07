"""One fresh worker: actual reverse import order plus the one new drain method."""

import json
import os
from pathlib import Path
import sys
import traceback
import unittest


ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
PUBLIC = ROOT / "experiments/mpc-research/resume-mvp"
ATTEMPT = Path(".runtime/r24-pre-gpu-b3-f1-role6/attempt")
blocked = []


def audit(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyname_ex", "subprocess.Popen", "os.system", "os.exec"):
        blocked.append(event)
        raise RuntimeError("CPU-only worker refused " + event)


def save(name, value):
    with (ATTEMPT / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


assert Path.cwd() == ROOT
sys.addaudithook(audit)
smoke_root = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"])
assert not smoke_root.is_absolute() and not smoke_root.exists() and smoke_root.parent.is_dir()
facts = {"order": [], "smoke_absent_initially": True}
try:
    import test_native_nominal_qualification as qualification
    facts.update(qualification_root=str(qualification.ROOT),
                 qualification_created=qualification.ROOT.is_dir(),
                 smoke_absent_after_qualification=not smoke_root.exists(),
                 qualification_module=str(Path(qualification.__file__).resolve()))
    facts["order"].append("qualification")
    import test_native_services_smoke as smoke
    facts.update(smoke_root=str(smoke.ROOT), smoke_created=smoke.ROOT.is_dir(),
                 smoke_module=str(Path(smoke.__file__).resolve()))
    facts["order"].append("smoke")
except BaseException as error:
    facts["error"] = {"type": type(error).__name__, "message": str(error),
                      "traceback": "".join(traceback.format_exception(error))}
    raise
finally:
    save("IMPORT-ORDER.json", facts)


class ImportOrder(unittest.TestCase):
    def test_qualification_first_preserves_smoke_output_namespace(self):
        self.assertEqual(facts["order"], ["qualification", "smoke"])
        self.assertTrue(facts["qualification_created"])
        self.assertTrue(facts["smoke_absent_after_qualification"])
        self.assertTrue(facts["smoke_created"])
        self.assertEqual(smoke.ROOT, smoke_root)
        self.assertEqual(qualification.ROOT.parent, smoke.ROOT.parent)
        self.assertNotEqual(qualification.ROOT, smoke.ROOT)
        self.assertEqual(Path(qualification.__file__).resolve(), PUBLIC / "test_native_nominal_qualification.py")
        self.assertEqual(Path(smoke.__file__).resolve(), PUBLIC / "test_native_services_smoke.py")


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subcase_results = []

    def addSubTest(self, test, subtest, err):
        self.subcase_results.append({
            "id": subtest.id(), "passed": err is None,
            "error": None if err is None else "".join(traceback.format_exception(*err)),
        })
        super().addSubTest(test, subtest, err)


suite = unittest.TestSuite([
    ImportOrder("test_qualification_first_preserves_smoke_output_namespace"),
    qualification.Controls("test_drain_tolerates_only_idle_cost_residue"),
])
result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
save("CONTROL-RESULTS.json", {
    "methods": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
    "skipped": len(result.skipped), "successful": result.wasSuccessful(),
    "subcases": result.subcase_results, "blocked_events": blocked,
    "scope": "One public drain method/four subcases plus one private actual reverse-import control; no peer methods or historical suites.",
    "python": sys.version, "native_qualification": False,
})
raise SystemExit(0 if result.wasSuccessful() and not blocked else 1)
