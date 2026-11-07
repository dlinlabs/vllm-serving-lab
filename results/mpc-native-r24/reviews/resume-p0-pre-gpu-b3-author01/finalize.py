"""Verify existing test evidence and seal it; never import or rerun caller controls."""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
SCOPE = Path(".runtime/r24-pre-gpu-b3-role6")
OWN = ("native_nominal_qualification.py", "test_native_nominal_qualification.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant " + value)
    return json.loads(path.read_text(), parse_constant=reject)


def save(name, value):
    with (SCOPE / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(pid):
    try:
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": values[19], "state": values[0]}
    except FileNotFoundError:
        return None


assert Path.cwd() == ROOT
baseline, frozen = load(SCOPE / "BASELINE.json"), load(SCOPE / "GREEN-FREEZE.json")
combined_pins = frozen["source_pins"] | frozen["data_pins"]
assert all(sha(Path(path)) == value for path, value in combined_pins.items())
assert sha(SCOPE / "PLAN.json") == frozen["plan_sha256"]
assert sha(SCOPE / "control.py") == frozen["control_sha256"]
assert sha(SCOPE / "freeze.py") == frozen["freeze_sha256"]
for name in OWN:
    expected = combined_pins[str(PUBLIC / name)]
    assert sha(SCOPE / "sources/candidate" / name) == expected
    assert sha(SCOPE / "attempts/green/sources" / name) == expected
    assert sha(SCOPE / "sources/before" / name) == baseline["pins"][str(PUBLIC / name)]["sha256"]
red, green = (load(SCOPE / "attempts" / mode / "result.json") for mode in ("red", "green"))
red_log = (SCOPE / "attempts/red/stderr.log").read_text()
green_log = (SCOPE / "attempts/green/stderr.log").read_text()
assert red["returncode"] == 1 and red_log.count("AssertionError: 5 != 0") == 2
assert "Ran 2 tests" in red_log and "FAILED (failures=2)" in red_log
assert green["returncode"] == 0 and "Ran 13 tests" in green_log and green_log.rstrip().endswith("OK")
assert sum(line.endswith(" ... ok") for line in green_log.splitlines()) == 13
for mode, result in (("red", red), ("green", green)):
    before = load(SCOPE / "attempts" / mode / "pins-before.json")
    after = load(SCOPE / "attempts" / mode / "pins-after.json")
    assert before == after
    assert result["joined"] and result["original_identity_gone"] and not result["timed_out"]
    assert result["elapsed_s"] <= 60
assert load(SCOPE / "attempts/green/pins-after.json") == combined_pins
assert load(SCOPE / "attempts/red/pins-before.json")[str(PUBLIC / OWN[0])] == baseline["pins"][str(PUBLIC / OWN[0])]["sha256"]

artifact_root = SCOPE / "attempts/green/artifacts/qualification"
cases, full = [], {}
for path in sorted(artifact_root.glob("*/CPU-CALLER-CHECK.json")):
    record = load(path)
    terminal = record["report"]["qualification_caller"]
    assert all(record["clients_closed"]) and record["remaining_tasks"] == []
    assert record["real_HTTP_or_services"] is False
    assert terminal["native_qualification"] is False and terminal["full_wall"] is None
    for call in record["calls"]:
        assert all(value > 0 for value in call["timeout"].values())
    full[path.parent.name] = record
    cases.append({
        "case": path.parent.name, "evidence": str(path.relative_to(SCOPE)),
        "status": terminal["status"], "stage": terminal["stage"],
        "error": None if record["error"] is None else record["error"]["type"],
        "allocated": terminal["records_allocated"], "sent": terminal["sent"],
        "episode_boundary_calls": record["episode_calls"],
        "mock_GETs": len(record["calls"]), "clients_closed": len(record["clients_closed"]),
        "remaining_tasks": [],
    })
assert len(cases) == 17
complete = full["complete"]["report"]
assert complete["offered_requests_from_driver"] == 5 and complete["episode_sent_requests"] == 4
assert complete["caller_clock_domains"]["SDK_source"] != complete["caller_clock_domains"]["warmup_recorder"]
assert all(value["readings"] == 2 and value["drained"] for value in complete["caller_drains"].values())
binding = load(artifact_root / "complete/runtime-binding.json")
assert binding["caller_sha256"] == combined_pins[str(PUBLIC / OWN[0])]
assert binding["adapter_sha256"] == combined_pins[str(PUBLIC / "nominal_episode.py")]
assert binding["offered_start"] == binding["permit_not_before"] + 1
assert binding["permit_expires_at"] - binding["permit_not_before"] == 20
assert binding["permit_max_commands"] == 8
assert full["episode-cancel"]["report"]["offered_requests_from_driver"] == 2
for name in ("pre-control-error", "prepare-error"):
    receipt = load(artifact_root / name / "episode-not-run.json")
    assert receipt["status"] == "NOT_RUN" and receipt["records_allocated"] == receipt["sent"] == 0
assert not (artifact_root / "terminal-write/qualification-caller.json").exists()
assert full["terminal-write"]["report"]["qualification_caller"]["status"] == "FAILED"
for path in SCOPE.rglob("*.json"):
    load(path)

known = [baseline["identity"], frozen["freezer_identity"]]
for mode in ("red", "green"):
    known.extend((load(SCOPE / "attempts" / mode / "command.json")["runner"],
                  load(SCOPE / "attempts" / mode / "process-start.json")))
cleanup = []
for original in known:
    current = identity(original["pid"])
    gone = current is None or current["start_ticks"] != original["start_ticks"]
    assert gone
    cleanup.append({"original": original, "current": current, "original_identity_gone": gone})
save("CLEANUP.json", {
    "known_processes": cleanup, "all_known_helpers_gone": True,
    "real_mock_clients_closed": sum(row["clients_closed"] for row in cases),
    "detailed_async_fixture_calls": len(cases), "remaining_fixture_tasks": [],
    "simple_fake_client_witnesses": 2,
    "scope": "Only recorded owned helper identities and fixture-owned resources; no global process scan or unrelated process changes.",
})
save("SOURCE-POSTCHECK.json", {
    "source_and_data_pins": combined_pins, "tracked_paths_unchanged_during_green": len(combined_pins),
    "baseline_protected_paths_unchanged": len(baseline["pins"]) - len(OWN),
    "owned_source_copies_match": True, "plan_and_runner_frozen": True,
    "concurrent_smoke_excluded": baseline["concurrent_smoke_not_frozen"],
})
save("CONTROL-RESULTS.json", {
    "red": {"methods": 2, "passed": 0, "failed": 2, "errors": 0, "result": red},
    "green": {"methods": 13, "passed": 13, "failed": 0, "errors": 0, "result": green,
              "unittest_elapsed_s": float(re.search(r"Ran 13 tests in ([0-9.]+)s", green_log)[1])},
    "cases": cases, "scoped_mock_GET_count": sum(row["mock_GETs"] for row in cases),
    "note": "The two simple retained draft witness fixtures are not included in detailed case/GET/client totals.",
})
save("FAILURES.json", {
    "original_red": {
        "caller_sha256": baseline["pins"][str(PUBLIC / OWN[0])]["sha256"],
        "log": "attempts/red/stderr.log", "failures": 2, "message": "AssertionError: 5 != 0",
        "meaning": "Original caller counted planned five as offers before prepare error handling or any episode run.",
        "not_Native": True,
    },
    "green_unexpected_failures": 0,
    "expected_injected_error_evidence": [row for row in cases if row["error"] is not None],
    "additional_codec_error_receipt": "attempts/green/artifacts/qualification/nonfinite-capture.write-error.json",
    "durability_failure_is_not_hidden": "Deliberately failed raw/terminal writes may leave no corresponding capture/receipt; preserved CPU-CALLER-CHECK reports and sidecars expose those failures. No durable-capture guarantee is claimed on a failed filesystem.",
})
save("REPORT.json", {
    "task": "B3 qualification caller and directly coupled obligations",
    "status": "CPU_CONTROLS_PASS_PENDING_MAYOR_INDEPENDENT_ACCEPTANCE",
    "root": str(ROOT), "declared_branch": "snapshot/mpc-p0-evidence-20261105",
    "declared_HEAD": "832ab593c8666b8bfa3761b3061355a2c94748c8",
    "Git": "User-declared branch/HEAD, not inspected or changed with Git.",
    "owned_sources": {str(PUBLIC / name): combined_pins[str(PUBLIC / name)] for name in OWN},
    "accepted_adapter_sha256": combined_pins[str(PUBLIC / "nominal_episode.py")],
    "original_caller_sha256": baseline["pins"][str(PUBLIC / OWN[0])]["sha256"],
    "CPU_results": {
        "original_red": "2 FAIL / 0 ERROR, one old-source attempt",
        "corrected": "13 PASS / 0 FAIL / 0 ERROR, one Python3.12.16 generation",
        "corrected_process_seconds": green["elapsed_s"], "max_seconds": 60,
        "detailed_fixture_calls": len(cases),
        "real_mock_clients_closed": sum(row["clients_closed"] for row in cases),
        "known_helpers_gone": len(cleanup), "tracked_stable_paths": len(combined_pins),
    },
    "implemented": [
        "B3: planned/validated/allocated/sent accounting; error-first/None-safe preparation; durable finally NOT_RUN/capture/terminal evidence.",
        "L2/L3: original BaseException identity/type/traceback survives secondary persistence/cleanup failures; strict-safe report before adapter result insertion.",
        "L4: bounded pre-S and final/failure zero-in-flight/waiting/admitted-cost and zero-fallback drain; all raw readings/errors retained.",
        "L6/B1: protocol re-read against startup pins; accepted adapter/caller source binding before traffic, rechecks and exact caller SHA in runtime-binding.",
        "L7/L9: positive remaining GET bounds and original episode/window fit inside unchanged20s/max8 permit and original D-R; no retiming.",
        "Rejected incorrect L5 advice; distinct real SDK and recorder clock labels preserved.",
    ],
    "remaining_blockers": [
        "No corrected-generation CPU defect reproduced; Mayor exact-source/integrated independent approval remains required.",
        "Role3 native_services_smoke.py is independently owned and not sealed by this capsule.",
        "Actual Native consecutive cycles/admission effects/ledger/request fates/final drain/cleanup/full wall remain unperformed and Mayor-owned.",
    ],
    "external_caller_hash_binding_only": True, "accepted_core_ABIs_unchanged": True,
    "no_real_HTTP_GPU_SSH_services_Git_agents_installs": True,
    "native_qualification": False, "full_wall": None,
    "details": ["INTERFACE.json", "LIMITATIONS.json", "COMMANDS.json", "CONTROL-RESULTS.json",
                "FAILURES.json", "CLEANUP.json", "SOURCE-POSTCHECK.json"],
    "finalized_at_monotonic": time.monotonic(), "finalizer_python": sys.version,
    "finalizer_identity": identity(os.getpid()),
})

excluded = {"SHA256SUMS", "SEAL.json"}
members = sorted(path for path in SCOPE.rglob("*") if path.is_file() and str(path.relative_to(SCOPE)) not in excluded)
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
    "status": "SEALED_CPU_ONLY_PENDING_MAYOR_ACCEPTANCE",
    "caller_sha256": combined_pins[str(PUBLIC / OWN[0])],
    "test_sha256": combined_pins[str(PUBLIC / OWN[1])],
    "native_qualification": False,
}
save("SEAL.json", seal)
print(json.dumps(seal, sort_keys=True))
