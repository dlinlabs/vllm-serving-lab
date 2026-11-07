"""Verify saved correction evidence and seal it without rerunning controls."""

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
SCOPE = Path(".runtime/r24-pre-gpu-b3-f1-role6")
REVIEW = Path(".runtime/r24-pre-gpu-final-review-role4")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(name, value):
    with (SCOPE / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}
    except FileNotFoundError:
        return None


assert Path.cwd() == ROOT
plan, baseline, frozen = (load(SCOPE / name) for name in ("PLAN.json", "BASELINE.json", "FREEZE.json"))
assert sha(SCOPE / "PLAN.json") == baseline["plan_sha256"] == frozen["plan_sha256"]
assert sha(SCOPE / "capsule.py") == frozen["capsule_sha256"]
assert sha(SCOPE / "focused_worker.py") == frozen["worker_sha256"]
assert sha(REVIEW / "SHA256SUMS") == sha(SCOPE / "prior-role4/SHA256SUMS") == baseline["role4_seal"]
lines = (SCOPE / "prior-role4/SHA256SUMS").read_text().splitlines()
assert len(lines) == 91
for line in lines:
    expected, name = line.split("  ", 1)
    assert sha(REVIEW / name) == sha(SCOPE / "prior-role4" / name) == expected
for path, expected in frozen["source_pins"].items():
    assert sha(Path(path)) == expected
assert load(SCOPE / "attempt/pins-before.json") == load(SCOPE / "attempt/pins-after.json") == frozen["source_pins"]
run, controls, imports = (load(SCOPE / "attempt" / name)
                         for name in ("result.json", "CONTROL-RESULTS.json", "IMPORT-ORDER.json"))
assert run["returncode"] == 0 and not run["timed_out"] and run["elapsed_s"] <= 60
assert run["joined"] and run["original_identity_gone"]
assert controls["successful"] and controls["methods"] == 2 and controls["failures"] == controls["errors"] == 0
assert len(controls["subcases"]) == 4 and all(case["passed"] for case in controls["subcases"])
assert controls["blocked_events"] == []
assert imports["order"] == ["qualification", "smoke"] and imports["smoke_absent_after_qualification"]
assert Path(imports["qualification_root"]).parent == Path(imports["smoke_root"]).parent
cases = []
for path in sorted(Path(imports["qualification_root"]).glob("drain-residue-*/DRAIN-CHECK.json")):
    value = load(path)
    assert value["client_closed"] and not value["remaining_tasks"]
    assert value["native_qualification"] is False
    reads = len(value["calls"])
    if value["expected_idle"]:
        assert value["error"] is None and reads == 1
        assert value["report"]["caller_drains"]["final"]["readings"] == 1
    else:
        assert reads >= 2 and "caller_drains" not in value["report"]
        assert value["returned"] >= value["deadline"] - value["cleanup_reserve_s"]
        assert value["error"]["type"] == "ProbeError"
        assert value["error"]["message"] == "original work deadline exhausted"
    cases.append({
        "case": value["case"], "metrics": value["metrics"], "reads": reads,
        "metric_evidence_files": len(list(path.parent.glob("final-metrics-*.json"))),
        "elapsed_s": value["returned"] - value["started"], "drained": value["expected_idle"],
        "error": None if value["error"] is None else value["error"]["message"],
        "evidence": str(path.relative_to(SCOPE)), "client_closed": True, "remaining_tasks": [],
    })
assert len(cases) == 4
assert not list(SCOPE.rglob("*.pyc"))
identities = [baseline["baseline_identity"], frozen["freeze_identity"],
              load(SCOPE / "attempt/command.json")["runner_identity"],
              load(SCOPE / "attempt/process-start.json")]
cleanup = []
for original in identities:
    current = identity(original["pid"])
    gone = current is None or current["start_ticks"] != original["start_ticks"]
    assert gone
    cleanup.append({"original": original, "current": current, "original_identity_gone": gone})
save("CLEANUP.json", {
    "owned_helper_identities": cleanup, "all_tracked_helpers_gone": True,
    "real_mock_clients_closed": 4, "remaining_fixture_tasks": [],
    "blocked_network_or_spawn_events": [], "new_bytecode_files": 0,
})
save("SOURCE-POSTCHECK.json", {
    "source_pins": frozen["source_pins"], "stable_during_controls": True,
    "protected_source_files_unchanged": len(frozen["source_pins"]) - 2,
    "source_diff": frozen["source_diff"], "test_diff": frozen["test_diff"],
    "original_role4_and_copied_members_verified": 91, "original_role4_seal": baseline["role4_seal"],
})
save("COMMANDS.json", {
    "cwd": str(ROOT), "python": sys.executable,
    "ordered": ["capsule.py baseline", "capsule.py freeze", "capsule.py run", "finalize.py"],
    "scripts_relative_to": str(SCOPE),
    "wrapper": "env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 nice -n19 <Python3.12> -B",
    "exact_focused_child_command": "attempt/command.json",
    "corrected_generations": 1, "other_test_generations": 0,
    "metadata_only": "Baseline/freeze/finalization and post-test JSON inspection do not execute caller controls.",
    "verification": "sha256sum --check --quiet SHA256SUMS, then match the manifest SHA with SEAL.json.",
})
caller = frozen["source_pins"][str(PUBLIC / "native_nominal_qualification.py")]
test = frozen["source_pins"][str(PUBLIC / "test_native_nominal_qualification.py")]
save("REPORT.json", {
    "status": "F1_F2_FOCUSED_CPU_PASS_PENDING_MAYOR_EXACT_SOURCE_ACCEPTANCE",
    "root": str(ROOT), "caller_sha256": caller, "test_sha256": test,
    "scope": "Only one production drain predicate; test sibling ROOT plus one new drain method.",
    "source_diff_mechanically_verified": True,
    "controls": {"methods": 2, "passed_methods": 2, "drain_subcases": 4, "passed_subcases": 4,
                 "failures": 0, "errors": 0, "corrected_generations": 1,
                 "python": run["python"], "worker_elapsed_s": run["elapsed_s"],
                 "max_seconds": 60, "cases": cases, "reverse_import_order": imports},
    "original_failures_preserved": {
        "role4_members": 91, "role4_seal": baseline["role4_seal"],
        "path": "prior-role4", "F1": "Original629caa23 dual-Python W1 confirmed; epsilon spun to deadline, zero control passed first read.",
        "F2": "Original static finding preserved; actual reverse-order corrected control now PASS.",
        "reviewer_erratum": "E1 acknowledged exact cost-zero checklist premise was incorrect; original report unchanged.",
    },
    "cleanup": {"mock_clients_closed": 4, "tracked_helper_identities_gone": len(cleanup),
                "remaining_tasks": [], "blocked_events": []},
    "stable_source_pins": len(frozen["source_pins"]),
    "peer_sources_unchanged": True, "accepted_core_guards_unchanged": True,
    "not_rerun": "Prior13 caller methods, peer methods and Mayor's521/186 composed controls.",
    "CPU_blockers_reproduced_after_fix": [],
    "remaining": "Mayor exact-source acceptance and the needed integrated generation; actual Native cycles/admission effects/ledger/cleanup/full wall remain unperformed by this task.",
    "native_qualification": False, "GPU_or_real_network": False,
    "no_SSH_Git_installs_models_agents_peer_edits": True,
    "details": ["HANDOFF.json", "LIMITATIONS.json", "FINDINGS.json", "correction.diff",
                "COMMANDS.json", "CLEANUP.json", "SOURCE-POSTCHECK.json"],
    "finalizer_identity": identity(os.getpid()),
})
excluded = {"SHA256SUMS", "SEAL.json"}
members = sorted(path for path in SCOPE.rglob("*")
                 if path.is_file() and str(path.relative_to(SCOPE)) not in excluded)
assert all(not path.is_symlink() for path in members)
with (SCOPE / "SHA256SUMS").open("x") as stream:
    for path in members:
        stream.write(sha(path) + "  " + str(path.relative_to(SCOPE)) + "\n")
for line in (SCOPE / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert sha(SCOPE / name) == expected
seal = {
    "manifest_sha256": sha(SCOPE / "SHA256SUMS"), "verified_members": len(members),
    "excluded_self_referential_members": sorted(excluded),
    "caller_sha256": caller, "test_sha256": test,
    "methods_passed": 2, "drain_subcases_passed": 4, "native_qualification": False,
}
save("SEAL.json", seal)
print(json.dumps(seal, sort_keys=True))
