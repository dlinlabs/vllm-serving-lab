"""Verify only this frozen source/evidence and the recorded owned PID identities."""

import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET


os.chdir(Path(__file__).resolve().parent)
os.umask(0o077)
freeze = json.loads(Path("CODE-FREEZE.json").read_text())
bindings = []
for name, expected in freeze["sha256"].items():
    for path in (Path("frozen") / name, Path(freeze["source_directory"]) / name):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, str(path)
        bindings.append({"path": str(path), "sha256": digest})
for name, expected in freeze["spec_sha256"].items():
    digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert digest == expected, name
    bindings.append({"path": name, "sha256": digest})
assert Path("generation01.exit-code.txt").read_text().strip() == "0"
attempts = json.loads(Path("control-attempts.json").read_text())
budget = json.loads(Path("outer-budget.json").read_text())
assert len(attempts["attempts"]) == 2 and attempts["no_retries"] is True
elapsed = attempts["finished"] - (budget["outer_deadline"] - 110)
assert 0 <= elapsed <= 110
identities = [budget["owner"]]
results = {}
for entry in attempts["attempts"]:
    assert entry["exit"] == 0 and entry["joined"] is True
    base = Path("attempts") / (entry["interpreter"] + "-01")
    assert (base / "exit-code.txt").read_text().strip() == "0"
    summary = json.loads((base / "evidence/summary.json").read_text())
    assert summary["tests"] == 6 and summary["errors"] == summary["failures"] == 0
    assert summary["all_owned_gone"] is summary["all_monitor_threads_joined"] is True
    assert summary["source_sha256"] == freeze["sha256"]["nominal_services.py"]
    assert summary["outer_deadline"] == budget["test_and_test_guardian_deadline"]
    assert summary["completed_monotonic"] < summary["outer_deadline"]
    junit = ET.parse(base / "evidence/junit.xml").getroot()
    assert junit.attrib["tests"] == "6"
    assert junit.attrib["failures"] == junit.attrib["errors"] == "0"
    assert len(junit.findall("testcase")) == 6 and not junit.findall(".//failure") and not junit.findall(".//error")
    poison = json.loads((base / "evidence/poison-before-guardian.json").read_text())
    guardian = json.loads((base / "evidence/guardian.json").read_text())
    assert poison["status"] == "POISON" and poison["known_identities_gone"] is False
    assert guardian["all_gone"] is guardian["poison_not_cleared"] is True
    assert guardian["completed_monotonic"] < guardian["outer_deadline"]
    for path in (base / "evidence/poison").glob("*.nominal-state.json"):
        assert json.loads(path.read_text())["status"] == "POISON"
    identities.extend([summary["owner"], *summary["owned"]])
    results[entry["interpreter"]] = {
        "python": summary["python"], "tests": 6, "failures": 0, "errors": 0,
        "known_service_identities": len(summary["owned"]),
        "all_monitor_threads_joined": True, "poison_preserved_after_guardian": True,
        "test_names": [case.attrib["name"] for case in junit.findall("testcase")],
    }

fresh = []
for row in identities:
    try:
        text = Path(f"/proc/{row['pid']}/stat").read_text()
        actual = int(text[text.rfind(")") + 2:].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        actual = None
    assert actual != row["start_ticks"], f"recorded owned identity remains: {row}"
    fresh.append({"pid": row["pid"], "start_ticks": row["start_ticks"],
                  "current_start_ticks": actual, "original_identity_gone": True})

result = {
    "schema": "role3-resume-p0-nominal-services-delivery-v1",
    "model_id": "gpt-6-astra", "reasoning_effort": "max",
    "identity_basis": "retained user/Mayor verification; no new model lookup/change",
    "status": "CPU_OWNERSHIP_CONTROLS_PASS_ONLY_PENDING_MAYOR_REVIEW",
    "source_bindings": bindings, "interpreters": results,
    "aggregate_control_wall_seconds": elapsed,
    "known_service_descendant_identities": sum(r["known_service_identities"] for r in results.values()),
    "fresh_owned_identity_checks": fresh, "fresh_checks_monotonic": time.monotonic(),
    "helper_source_lines": len(Path("frozen/nominal_services.py").read_text().splitlines()),
    "test_source_lines": len(Path("frozen/test_nominal_services.py").read_text().splitlines()),
    "attempts_per_interpreter": 1, "generations": 1,
    "intentional_negative_evidence_preserved": True,
    "failed_test_attempts": 0, "new_control_reruns": 0,
    "actual_GPU_service_or_network_execution": False,
    "production_lock_binding_verified": False, "actual_host_cleanup_qualified": False,
    "owner_crash_cleanup_guaranteed": False, "universal_OS_proof": False,
    "native_eligibility": False, "MPC_or_SLO_or_accuracy_claim": False,
    "scientific_gates_changed": False, "actual_execution_authorized": False,
}
with Path("DELIVERY.json").open("x") as stream:
    json.dump(result, stream, sort_keys=True, indent=2)
    stream.write("\n")
print(json.dumps({"status": result["status"], "source_bindings": len(bindings),
                  "fresh_owned_identities_gone": len(fresh),
                  "aggregate_control_wall_seconds": elapsed,
                  "helper_lines": result["helper_source_lines"],
                  "tests": {name: r["tests"] for name, r in results.items()}}, sort_keys=True))
