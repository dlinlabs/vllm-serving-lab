"""Verify this B2 delivery, not services/models or another test generation."""

import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET


CAP = Path(".runtime/r24-pre-gpu-b2-role3")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
os.umask(0o077)
baseline = json.loads((CAP / "BASELINE.json").read_text())
assert hashlib.sha256((CAP / "PLAN.json").read_bytes()).hexdigest() == baseline["plan_sha256"]
for relative, row in baseline["bindings"].items():
    name = "gateway-lines140-205.txt" if relative == "gateway.py:140-205" else Path(relative).name
    assert hashlib.sha256((CAP / "originals" / name).read_bytes()).hexdigest() == row["sha256"]
red = json.loads((CAP / "attempts/red01/SOURCE-FREEZE.json").read_text())
green = json.loads((CAP / "attempts/green01/SOURCE-FREEZE.json").read_text())
assert red["sources"]["native_services_smoke.py"] == baseline["bindings"][str(PUBLIC / "native_services_smoke.py")]["sha256"]
assert red["sources"]["test_native_services_smoke.py"] == green["sources"]["test_native_services_smoke.py"]
for name, digest in green["sources"].items():
    assert hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() == digest

old_driver = (CAP / "originals/native_services_smoke.py").read_text()
new_driver = (PUBLIC / "native_services_smoke.py").read_text()
opening = "        if closed:\n"
ending = '        report["end"] = stamp()\n'
assert old_driver[:old_driver.index(opening)] == new_driver[:new_driver.index(opening)]
assert old_driver[old_driver.index(ending):] == new_driver[new_driver.index(ending):]

def original_controls(text):
    module = ast.parse(text)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Controls")
    return {node.name: ast.dump(node, include_attributes=False)
            for node in cls.body if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_") and int(node.name.split("_")[1]) <= 9}

old_tests = (CAP / "originals/test_native_services_smoke.py").read_text()
new_tests = (PUBLIC / "test_native_services_smoke.py").read_text()
assert len(original_controls(old_tests)) == 9
assert original_controls(old_tests) == original_controls(new_tests)
diff = ""
for name, old, new in (("native_services_smoke.py", old_driver, new_driver),
                       ("test_native_services_smoke.py", old_tests, new_tests)):
    diff += "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="original/" + name, tofile=str(PUBLIC / name)))
(CAP / "CHANGES.patch").write_text(diff)

summaries, identities = {}, []
for phase, tests, failures, exit_code in (("red01", 4, 6, 1), ("green01", 13, 0, 0)):
    path = CAP / "attempts" / phase
    freeze = json.loads((path / "SOURCE-FREEZE.json").read_text())
    for name, digest in freeze["sources"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    for name, digest in freeze["specifications"].items():
        assert hashlib.sha256((CAP / name).read_bytes()).hexdigest() == digest
    summary = json.loads((path / "evidence/summary.json").read_text())
    assert (summary["tests"], summary["failures"], summary["errors"]) == (tests, failures, 0)
    assert summary["actual_service_HTTP_GPU_model_calls"] == 0
    assert summary["driver_sha256"] == freeze["sources"]["native_services_smoke.py"]
    assert summary["test_sha256"] == freeze["sources"]["test_native_services_smoke.py"]
    assert (path / "exit-code.txt").read_text().strip() == str(exit_code)
    assert (CAP / (phase + "-wrapper.exit-code.txt")).read_text().strip() == str(exit_code)
    junit = ET.parse(path / "evidence/junit.xml").getroot()
    assert junit.attrib["tests"] == str(tests) and junit.attrib["failures"] == str(failures)
    assert junit.attrib["errors"] == "0" and len(junit.findall(".//failure")) == failures
    assert len(junit.findall("testcase")) == tests
    result = json.loads((path / "result.json").read_text())
    assert result["joined"] is True and result["source_bytes_unchanged"] is True
    assert result["started"] <= result["ended"] < result["original_deadline"]
    command = json.loads((path / "command.json").read_text())
    identities.extend([command["runner_identity"], json.loads((path / "process.json").read_text())])
    summaries[phase] = dict(tests=tests, failures=failures, errors=0, exit=exit_code,
                            python=summary["python"], wall_seconds=result["ended"] - result["started"])

def report(phase, case):
    return json.loads((CAP / "attempts" / phase / "evidence" / case / "output/report.json").read_text())

assert report("red01", "b2-body-complete")["lifecycle_passed"] is False
assert report("green01", "b2-body-complete")["lifecycle_passed"] is True
assert report("green01", "b2-no-body-nonzero")["lifecycle_passed"] is False
assert report("green01", "b2-body-failure")["lifecycle_passed"] is False
both = report("green01", "b2-two-closure-failures")
assert both["pre_gpu"]["idle_selected_gpu"] is True and both["post_gpu"]["idle_selected_gpu"] is False
assert {row["message"] for row in both["errors"]} == {
    "post GPU not proven idle", "telemetry closure or no-request ledger mismatch"}
assert both["native_qualification"] is False
fresh = []
for row in identities:
    try:
        text = Path(f"/proc/{row['pid']}/stat").read_text()
        actual = int(text[text.rfind(")") + 2:].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        actual = None
    assert actual != row["start_ticks"], "recorded control process remains"
    fresh.append(dict(row, current_start_ticks=actual, original_identity_gone=True))
result = dict(
    schema="role3-pre-gpu-b2-delivery-v1", model_id="gpt-6-astra", reasoning_effort="max",
    root=str(Path.cwd()), status="B2_CPU_REPAIR_VERIFIED_PENDING_MAYOR_ACCEPTANCE",
    supplied_head="832ab593c8666b8bfa3761b3061355a2c94748c8", git_used=False,
    changed_public_paths={str(PUBLIC / name): green["sources"][name]
                          for name in ("native_services_smoke.py", "test_native_services_smoke.py")},
    baseline_driver_sha256=red["sources"]["native_services_smoke.py"],
    tests_unchanged_between_red_green=True, original_nine_test_methods_unchanged=True,
    driver_edit_limited_to_closed_finalization_branch=True,
    results=summaries, original_failure_preserved=True, attempts=2, retries=0,
    fresh_control_identity_closure=fresh, closure_check_monotonic=time.monotonic(),
    actual_GPU_network_SSH_service_model_execution=False,
    Native_readiness_or_cycles_or_qualification_proven=False,
    unaddressed=["B3 caller/result handling (Role6 scope)", "actual-host preflight/Native verification"],
)
(CAP / "DELIVERY.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
print(json.dumps(result, sort_keys=True))
