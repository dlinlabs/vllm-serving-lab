"""Seal the bounded failed CLI02 attempt; no rerun, repair, forecast or HTTP."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCOPE = Path(".runtime/r24-resume-p0-nominal-runtime02")
ATTEMPT = SCOPE / "attempts/generation01-py312"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (SCOPE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def gone(identity):
    try:
        ticks = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return True
    return ticks != identity["start_ticks"]


assert not (SCOPE / "SHA256SUMS").exists()
freeze = read(SCOPE / "FREEZE.json")
pins = read(SCOPE / "INPUT_PINS.json")
checks = {path: sha(Path(path)) == item["sha256"] for path, item in pins.items()}
assert all(checks.values())
save("SOURCE_POSTCHECK.json", checks)
result = read(ATTEMPT / "result.json")
log = (ATTEMPT / "stderr.log").read_text()
assert result["returncode"] == 1 and result["joined"] and result["original_identity_gone"]
assert not result["timed_out"] and "Ran 4 tests" in log and "FAILED (failures=1, errors=2)" in log
assert all(read(ATTEMPT / "pins-before.json").values())
assert all(read(ATTEMPT / "pins-after.json").values())
identities = [
    {**freeze["preparation_identity"], "owner": "preparation"},
    {**read(ATTEMPT / "runner-identity.json"), "owner": "bounded test supervisor"},
    {**result["original_identity"], "owner": "unittest process"},
]
client_receipts = []
cleanups = sorted((ATTEMPT / "artifacts").glob("*/OWNED_CLEANUP.json"))
assert len(cleanups) == 4
for path in cleanups:
    value = read(path)
    assert not value["runtime_workers"]
    assert all(value["clients_closed"])
    client_receipts.append({"case": path.parent.name, "clients_closed": value["clients_closed"]})
    for child in value["standalone_CLI_children"]:
        assert child["joined"] and child["original_identity_gone"] and child["returncode"] == 0
        identities.append({**child, "owner": "offline standalone CLI child"})
identities = [{**identity, "fresh_original_identity_gone": gone(identity)} for identity in identities]
assert all(item["fresh_original_identity_gone"] for item in identities)
save("FRESH_OWNED_GONE.json", {
    "identities": identities, "clients": client_receipts,
    "all_known_helpers_joined_or_finished": True, "runtime_workers": [],
    "not_universal_engine_or_campaign_cleanup": True,
})
save("FAILURES.json", {
    "unexpected_control_result": "1 PASS, 1 FAIL, 2 ERROR; process exit1",
    "attempts": 1, "source_generations": 1, "retries": 0, "source_repair_after_freeze": False,
    "raw_stdout": str((ATTEMPT / "stdout.log").relative_to(SCOPE)),
    "raw_stderr": str((ATTEMPT / "stderr.log").relative_to(SCOPE)),
    "common_failure": "runtime_configuration_or_parameter_binding_changed",
    "failure_boundary": "wait_for_start->_binding before runtime.run; the transport/replay/pending controls do not establish success.",
    "source_supported_diagnosis": {
        "kind": "inconsistent_root_URL_canonicalization",
        "protocol": "_protocol() uses str(endpoint.copy_with(path='/')).",
        "guard": "_binding() compares str(self.client.base_url) to that canonical string.",
        "HTTPX": "URL.raw_path returns '/' for an absent path, so BaseClient._enforce_trailing_slash may retain the original bare-host URL string. Thus http://cpu.invalid and http://cpu.invalid/ can mismatch.",
        "evidence": "Actual frozen control argv use the bare-host form; offline protocol output includes the slash; both chosen local3.12 HTTPX _client.py234-237 and _urls.py283-297 were read-only inspected.",
        "no_additional_diagnostic_control_run": True,
        "minimal_corrective_direction": "Use the same strict HTTP-root URL canonicalization in _protocol and _binding. Preserve rejection of non-root paths, query/fragment/userinfo and changed origins. Do not mask the bug by only changing test URLs to include a slash.",
        "correction_status": "NOT APPLIED OR TESTED in this once-frozen attempt.",
    },
    "expected_parser_refusals": "Mutually exclusive sources, missing permit, missing forecast purpose: stderr retained; these expected negatives do not explain away the real binding defect.",
})
save("INTERFACE.json", {
    "status": "FAILED_CANDIDATE_DO_NOT_ACTIVATE",
    "entry": "<same-host-python> -B <frozenRoot>/experiments/mpc-research/resume-mvp/nominal_runtime.py",
    "new_arguments": [
        "--train-tape <canonical train.tape.json> (mutually exclusive with --scenario-file)",
        "--forecast-purpose train|evaluation|engineering (required with Train)",
        "--poll-seconds <0..1>",
        "--start-at <absolute same-host monotonic target>",
        "--describe-protocol",
    ],
    "offline_description_verified": "Actual protocol/hash/five-module source bundle, expected identity and module paths; no client/HTTP/approval read/journal write/wait. Stable across new prior availability; standalone from outside Root with PYTHONPATH unset passed.",
    "runtime_not_verified": "The real-loop, full journal replay, scheduled/pending restart and partial-solve controls failed early on endpoint binding. Do not promote source-level wiring to successful execution.",
    "Train_binding_design": "One prior freeze per invocation; static prior/data/tokenizer/producer source/RNG identity excludes available_at. Provider returns make_scenario(...).causal. No Eval tape or workload regeneration API.",
    "replay_design": "ProtocolBound stores parameters; PlanningInput saves bounded full CausalScenario/window before rollouts; Planned repeats full scenario beside its digest. These new paths are NOT validated by this failed attempt.",
    "forecast_receipt_limit": "Only full CausalScenario plus static prior/RNG/source binding is retained, not the complete NominalForecast wrapper/prior.available_at receipt.",
    "shared_start_design": "Wait before run entry; initial target must be nonpast and leave the episode inside min(CLI-entry+60, enabled permit expiry). Existing matching same-target journal resumes old episode. New run uses actual entry, never backdates to target.",
    "host_binding": "Generate description on the actual controller host/interpreter: clock_domain, RNG Python version and absolute journal path are bound.",
    "scope": "Runtime protocol hash is not a substitute for the separate proposed campaign protocol hash or Mayor approval.",
    "remaining_blocker": "Correct endpoint canonicalization under a fresh explicit source freeze/control authorization.",
})
save("LIMITATIONS.json", {
    "current_blocker": "CLI02 endpoint-binding implementation defect; failed candidate, not an accepted executable loop.",
    "validation": [
        "Only one local Python3.12.16 invocation of four new methods. No Python3.14 or host Python3.12.3 execution.",
        "No old12 suite, peer suite, new forecast research, workload regeneration, resampling or retry.",
        "Offline/standalone description passed; remaining controls failed before their intended loop/replay boundaries.",
        "Full realized forecast capture, replay, polling/shared-start and pending restart code must be validated after the surgical correction.",
    ],
    "Native": [
        "No SSH/network/GPU/inference/live approval/Native qualification.",
        "Native14 rejected; coverage1of2 FAIL; G06/G16 pending; legacy D2 startup-SIGALRM R1 remains open and unused.",
        "Real tokenizer-loaded/no-fallback verification, exact cost consistency, source/host/GPU exclusivity and campaign ownership remain Mayor prerequisites.",
        "Correct future interpreter is /workspace/pr32-r23/venv/bin/python3.12, not a nonexistent python3.12.3 filename.",
    ],
    "unchanged_boundaries": [
        "GET-only default; separate opt-in and nominal permit for POST, never ActuationPermit spoofing.",
        "No ledger CAS, no POST retry or fallback on unknown, no invented application timestamps.",
        "No Static Fixed/FIFO implementation, serving owner, replay client or outcome-metrics duplication.",
        "No SLO guarantee, superiority, production or ResumeReady-complete claim.",
        "Known CPU helpers/clients were cleaned; no universal OS/filesystem/engine cleanup claim.",
    ],
    "old01": "Read-only baseline/manifest/REPORT references stayed hash-matched; the old sealed report was not overwritten.",
})
save("COMMANDS.json", {
    "test": read(ATTEMPT / "command.json"),
    "preparation": "standard env/unsetPYTHONPATH/threads1/nobytecode/noautoload/nice19 selected python3.12 -B .runtime/r24-resume-p0-nominal-runtime02/prepare02.py",
    "runner": "same wrapper + selected python3.12 -B .runtime/r24-resume-p0-nominal-runtime02/run02.py",
    "finisher": "same wrapper + selected python3.12 -B .runtime/r24-resume-p0-nominal-runtime02/finish02.py",
    "source_fact_diagnosis": "Read-only source inspection, no extra runtime/control command.",
})
save("REPORT.json", {
    "schema": "resume-p0-nominal-runtime02-report-v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "FAILED_BINDING_BLOCKER_NO_ACTIVATION",
    "model": {"id": "gpt-6-astra", "reasoning_effort": "max", "provenance": "retained verified Role6 assignment"},
    "source_freeze_sha256": sha(SCOPE / "FREEZE.json"),
    "source_ids": freeze["source_ids"], "source_diff": "SOURCE_DIFF.patch",
    "test_result": "1 PASS / 1 FAIL / 2 ERROR", "test_process": result,
    "source_generations": 1, "test_attempts": 1, "retries": 0,
    "selected_new_controls": 4, "old_controls_run": 0,
    "preserved_binding_count": len(pins),
    "known_preparation_supervisor_test_CLI_identities_gone": len(identities),
    "clients_closed": sum(len(row["clients_closed"]) for row in client_receipts),
    "current_source_unchanged_from_failed_freeze": True,
    "minimum_next_step": "Shared strict root-URL canonicalization in protocol and runtime binding, then fresh bounded validation of these same four controls without workload changes.",
    "accepted_by_Mayor": False, "Native_permission": False, "real_approval_created": False,
    "qualification": "Description-only positive proof; not an actuated Native or accepted CLI02 loop.",
    "seal_exclusions": ["SHA256SUMS", "SEAL_RECEIPT.json"],
})
save("HANDOFF.json", {
    "status": "BLOCKED_AND_IDLE", "report": "REPORT.json", "interface": "INTERFACE.json",
    "failure_detail": "FAILURES.json", "limits": "LIMITATIONS.json", "cleanup": "FRESH_OWNED_GONE.json",
    "source_freeze": "FREEZE.json", "source_diff": "SOURCE_DIFF.patch", "commands": "COMMANDS.json",
    "manifest": "SHA256SUMS", "seal_receipt": "SEAL_RECEIPT.json",
})
files = sorted(path for path in SCOPE.rglob("*") if path.is_file())
assert all(not path.is_symlink() for path in files)
lines = [f"{sha(path)}  {path.relative_to(SCOPE).as_posix()}\n" for path in files]
with (SCOPE / "SHA256SUMS").open("x") as stream:
    stream.writelines(lines)
for line in lines:
    expected, relative = line.rstrip("\n").split("  ", 1)
    assert sha(SCOPE / relative) == expected
seal = sha(SCOPE / "SHA256SUMS")
save("SEAL_RECEIPT.json", {
    "manifest_sha256": seal, "members": len(files), "all_members_verified": True,
    "all_source_bindings_unchanged": True, "known_identities_gone": len(identities),
    "status": "FAILED_CANDIDATE_NO_NATIVE_PERMISSION", "receipt_not_self_in_manifest": True,
})
print(json.dumps({
    "seal": seal, "members": len(files), "status": "1 PASS / 1 FAIL / 2 ERROR; blocked, no retry",
    "known_identities_gone": len(identities), "source_generations": 1,
}, sort_keys=True))
