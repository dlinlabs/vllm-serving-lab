"""Six CPU-only ownership controls. All dummies belong to this isolated process."""

import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import nominal_services as ns


ROOT = Path(os.environ["NOMINAL_TEST_OUTPUT"])
OUTER_END = float(os.environ["NOMINAL_TEST_DEADLINE"])
assert not ROOT.is_absolute()
ROOT.mkdir()
OWNERS = []
ENV = {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
       "PRIVATE_FAKE_TOKEN": "private-control-not-for-state",
       "KEEP_BACKEND_OPTIMIZATIONS": "unchanged"}

SLEEP = "import time; time.sleep(60)"


def dumped(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def services(code=SLEEP, other=SLEEP):
    return (ns.Service("engine", (sys.executable, "-B", "-c", code), dict(ENV)),
            ns.Service("gateway", (sys.executable, "-B", "-c", other), dict(ENV)))


def owner(path, specs=None, ready=lambda *_: True, seconds=3, **kwargs):
    path.mkdir()
    obj = ns.NominalServices(
        specs or services(), lock_paths=(path.resolve() / "mayor.lock", path.resolve() / "component.lock"),
        run_dir=path / "run", deadline=min(time.monotonic() + seconds, OUTER_END - 5),
        ready=ready, cleanup_reserve=.5, term_grace=.1, **kwargs)
    OWNERS.append(obj)
    return obj


def guardian(obj):
    """Test-only emergency cleanup uses the pre-frozen OUTER budget, never a service grace reset."""
    old_subreaper, pidfds = ns._subreaper(), {}
    ns._subreaper(True)
    try:
        for row in tuple(obj._known.values()):
            if ns._same(row):
                if time.monotonic() >= OUTER_END:
                    raise AssertionError("original test guardian budget exhausted")
                try:
                    fd = os.pidfd_open(row["pid"])
                except ProcessLookupError:
                    continue
                pidfds[row["pid"]] = fd
                if ns._same(row):
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        for process in tuple(obj._roots.values()):
            process.wait(timeout=max(0, OUTER_END - time.monotonic()))
        while time.monotonic() < OUTER_END:
            for fd in pidfds.values():
                try:
                    os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG)
                except ChildProcessError:
                    pass
            if all(not ns._same(row) for row in obj._known.values()):
                break
            time.sleep(min(.01, max(0, OUTER_END - time.monotonic())))
    finally:
        for fd in pidfds.values():
            os.close(fd)
        ns._subreaper(old_subreaper)
    assert all(not ns._same(row) for row in obj._known.values())


class OwnershipControls(unittest.TestCase):
    def tearDown(self):
        for obj in OWNERS:
            if any(ns._same(row) for row in obj._known.values()):
                guardian(obj)

    def record(self, name, obj):
        value = json.loads((obj.run_dir / "ownership.json").read_text())
        dumped(ROOT / (name + ".json"), value)
        self.assertTrue(value["known_identities_gone"])
        self.assertFalse(value["native_eligibility"])
        self.assertFalse(value["universal_OS_proof"])
        self.assertTrue(all(not ns._same(row) for row in value["known"]))
        return value

    def test_01_default_off(self):
        obj = owner(ROOT / "off")
        with patch.object(ns.subprocess, "Popen", side_effect=AssertionError("OFF spawned")) as spawn:
            with self.assertRaisesRegex(ns.OwnershipError, "default OFF"):
                with obj:
                    self.fail("OFF entered")
            self.assertEqual(spawn.call_count, 0)
        self.assertFalse(obj.run_dir.exists())

    def test_02_each_lock_exclusive(self):
        for index in (0, 1):
            obj = owner(ROOT / f"locked-{index}", execute=True)
            fd = os.open(obj.paths[index], os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(ns.subprocess, "Popen", side_effect=AssertionError("lock bypass")) as spawn:
                    with self.assertRaises(BlockingIOError):
                        with obj:
                            self.fail("held exact lock entered")
                    self.assertEqual(spawn.call_count, 0)
                self.assertFalse(obj.run_dir.exists())
            finally:
                os.close(fd)

    def test_03_known_child_startup_failure(self):
        release = ROOT / "startup-release"
        code = ("from pathlib import Path; import time; "
                f"p=Path({str(release)!r})\n"
                "while not p.exists(): time.sleep(.01)\n"
                "raise SystemExit(9)")

        def ready(obj, remaining):
            if not release.exists():
                release.write_text("test-owned release\n")
            return False

        obj = owner(ROOT / "startup", services(code), ready, execute=True)
        with self.assertRaises(ns.OwnershipError):
            with obj:
                self.fail("failed startup entered")
        value = self.record("startup-outcome", obj)
        self.assertEqual(value["status"], "STOPPED")
        self.assertEqual(obj._roots["engine"].returncode, 9)
        self.assertIsNotNone(obj._roots["gateway"].returncode)

    def test_04_finally_descendant_reaping_and_privacy(self):
        leaf_file, ready_file = ROOT / "leaf-pid", ROOT / "engine-ready"
        leaf = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        code = (
            "import os,signal,subprocess,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"p=subprocess.Popen([sys.executable,'-B','-c',{leaf!r}],start_new_session=True); "
            f"Path({str(leaf_file)!r}).write_text(str(p.pid)); "
            f"Path({str(ready_file)!r}).write_text('ready'); "
            "assert os.environ['KEEP_BACKEND_OPTIMIZATIONS']=='unchanged'; time.sleep(60)"
        )

        def ready(obj, remaining):
            return (ready_file.exists() and leaf_file.exists()
                    and int(leaf_file.read_text()) in obj._known)

        obj = owner(ROOT / "finally", services(code), ready, execute=True)
        original_popen, before_spawn = ns.subprocess.Popen, []

        def recorded_popen(argv, **kwargs):
            for path in obj._markers:
                marker = json.loads(path.read_text())
                self.assertEqual(marker["status"], "RUNNING")
                self.assertIsNotNone(marker["pending_spawn"])
                self.assertEqual(marker["owner"]["pid"], os.getpid())
                self.assertTrue(ns._same(marker["owner"]))
            item = next(s for s in obj.services if s.name == marker["pending_spawn"])
            self.assertEqual(argv, list(item.argv))
            self.assertEqual(kwargs["env"], item.env)
            self.assertTrue(kwargs["start_new_session"])
            before_spawn.append(marker["pending_spawn"])
            return original_popen(argv, **kwargs)

        with patch.object(ns.subprocess, "Popen", side_effect=recorded_popen):
            with self.assertRaisesRegex(ValueError, "owned body failure"):
                with obj:
                    self.assertEqual(len(obj._known), 3)
                    for item in obj._metadata:
                        root = item["root_identity"]
                        self.assertEqual(root["pgid"], root["pid"])
                        self.assertEqual(root["sid"], root["pid"])
                    raise ValueError("owned body failure")
        self.assertEqual(before_spawn, ["engine", "gateway"])
        value = self.record("finally-outcome", obj)
        self.assertEqual(value["status"], "STOPPED")
        self.assertEqual(obj._roots["engine"].returncode, -signal.SIGKILL)
        self.assertNotIn(ENV["PRIVATE_FAKE_TOKEN"], json.dumps(value))
        for name in ("engine.log", "gateway.log", "private-errors.log"):
            self.assertEqual((obj.run_dir / name).stat().st_mode & 0o777, 0o600)

    def test_05_original_deadline(self):
        expired = owner(ROOT / "already-expired", seconds=-1, execute=True)
        with patch.object(ns.subprocess, "Popen", side_effect=AssertionError("expired spawned")) as spawn:
            with self.assertRaises(ns.OwnershipError):
                with expired:
                    self.fail("expired entered")
            self.assertEqual(spawn.call_count, 0)
        obj = owner(ROOT / "readiness-expiry", ready=lambda *_: False, seconds=1.1, execute=True)
        with self.assertRaises(ns.OwnershipError):
            with obj:
                self.fail("never-ready entered")
        value = self.record("deadline-outcome", obj)
        self.assertEqual(value["status"], "STOPPED")
        self.assertTrue(value["expired"])
        self.assertLessEqual(time.monotonic(), obj.deadline)

    def test_06_nonjoin_poison_and_refusal(self):
        obj = owner(ROOT / "poison", seconds=1.1, execute=True)
        denied = []
        with self.assertRaisesRegex(ns.OwnershipError, "POISON"):
            with obj:
                original = obj._signal
                engine = obj._roots["engine"].pid

                def failing_signal(row, sig):
                    if row["pid"] == engine:
                        denied.append(int(sig))
                        raise PermissionError("test-only injected signal denial")
                    return original(row, sig)

                obj._signal = failing_signal
        value = json.loads((obj.run_dir / "ownership.json").read_text())
        self.assertEqual(value["status"], "POISON")
        self.assertFalse(value["known_identities_gone"])
        self.assertIn(int(signal.SIGTERM), denied)
        self.assertIn(int(signal.SIGKILL), denied)
        self.assertIsNotNone(obj._roots["gateway"].returncode)
        dumped(ROOT / "poison-before-guardian.json", value)
        guardian(obj)
        future = ns.NominalServices(
            services(), lock_paths=obj.paths, run_dir=obj.run_dir.parent / "must-not-start",
            deadline=min(time.monotonic() + 2, OUTER_END), ready=lambda *_: True, execute=True,
            cleanup_reserve=.5, term_grace=.1)
        with patch.object(ns.subprocess, "Popen", side_effect=AssertionError("poison reused")) as spawn:
            with self.assertRaisesRegex(ns.OwnershipError, "POISON"):
                with future:
                    self.fail("poison entered")
            self.assertEqual(spawn.call_count, 0)
        self.assertFalse(future.run_dir.exists())
        dumped(ROOT / "guardian.json", {
            "scope": "test-owned dummies only; original aggregate test deadline",
            "outer_deadline": OUTER_END, "completed_monotonic": time.monotonic(),
            "known": list(obj._known.values()), "all_gone": obj._gone(),
            "poison_not_cleared": all(json.loads(p.read_text())["status"] == "POISON"
                                     for p in obj._markers)})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OwnershipControls)
    tests = list(suite)
    try:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        for obj in OWNERS:
            if any(ns._same(row) for row in obj._known.values()):
                guardian(obj)
    failures = {test.id(): text for test, text in result.failures}
    errors = {test.id(): text for test, text in result.errors}
    summary = dict(tests=result.testsRun, failures=len(failures), errors=len(errors),
                   actual_model_GPU_network_calls=0, source_sha256=ns.hashlib.sha256(
                       Path(ns.__file__).read_bytes()).hexdigest(),
                   python=sys.version, owner=ns._identity(os.getpid()),
                   owned=[row for obj in OWNERS for row in obj._known.values()],
                   all_owned_gone=all(obj._gone() for obj in OWNERS),
                   all_monitor_threads_joined=all(not obj._thread or not obj._thread.is_alive()
                                                  for obj in OWNERS),
                   outer_deadline=OUTER_END, completed_monotonic=time.monotonic())
    dumped(ROOT / "summary.json", summary)
    xml = ET.Element("testsuite", tests=str(result.testsRun), failures=str(len(failures)),
                     errors=str(len(errors)), name="nominal-services-CPU-only")
    for test in tests:
        case = ET.SubElement(xml, "testcase", name=test._testMethodName)
        for kind, table in (("failure", failures), ("error", errors)):
            if test.id() in table:
                ET.SubElement(case, kind).text = table[test.id()]
    with (ROOT / "junit.xml").open("xb") as stream:
        stream.write(ET.tostring(xml, encoding="utf-8", xml_declaration=True))
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
