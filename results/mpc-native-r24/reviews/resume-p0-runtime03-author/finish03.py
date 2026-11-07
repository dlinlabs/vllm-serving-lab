"""Verify and seal the one corrected CPU generation; no further controls or HTTP."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCOPE = Path(".runtime/r24-resume-p0-nominal-runtime03")
OLD = Path(".runtime/r24-resume-p0-nominal-runtime02")
ATTEMPT = SCOPE / "attempts/generation01-py312"
ARTIFACTS = ATTEMPT / "artifacts"


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
metadata = read(SCOPE / "SOURCE_METADATA.json")
checks = {path: sha(Path(path)) == item["sha256"] for path, item in pins.items()}
assert all(checks.values())
for name, row in freeze["source_ids"].items():
    assert sha(SCOPE / "sources" / Path(name).name) == row["sha256"]
save("SOURCE_POSTCHECK.json", checks)
result = read(ATTEMPT / "result.json")
log = (ATTEMPT / "stderr.log").read_text()
assert result["returncode"] == 0 and result["joined"] and result["original_identity_gone"]
assert not result["timed_out"] and "Ran 17 tests" in log and log.rstrip().endswith("OK")
assert all(read(ATTEMPT / "pins-before.json").values())
assert all(read(ATTEMPT / "pins-after.json").values())
assert (SCOPE / "outputs/prepare.exit").read_text().strip() == "0"
assert (SCOPE / "outputs/runner.exit").read_text().strip() == "0"
assert not (SCOPE / "outputs/prepare.stderr.log").read_bytes()
assert not (SCOPE / "outputs/runner.stderr.log").read_bytes()
assert sha(OLD / "SHA256SUMS") == freeze["immutable_predecessor_seal"]
old_members = 0
for line in (OLD / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert sha(OLD / name) == expected
    old_members += 1
save("IMMUTABLE02_POSTCHECK.json", {
    "manifest_sha256": sha(OLD / "SHA256SUMS"), "members_verified": old_members,
    "report_sha256": sha(OLD / "REPORT.json"), "failures_sha256": sha(OLD / "FAILURES.json"),
    "old_failure_unchanged": "1 PASS / 1 FAIL / 2 ERROR",
})

identities = [
    {**freeze["preparation_identity"], "owner": "preparation"},
    {**read(ATTEMPT / "runner-identity.json"), "owner": "bounded test supervisor"},
    {**result["original_identity"], "owner": "unittest process"},
]
client_receipts = []
task_count = 0
cleanups = sorted(ARTIFACTS.glob("*/OWNED_CLEANUP.json"))
assert len(cleanups) == 17
for path in cleanups:
    value = read(path)
    assert not value.get("runtime_workers", [])
    assert not value.get("runtime_children_started", [])
    assert not value.get("owned_helper_processes", [])
    assert all(value["clients_closed"])
    assert all(value.get("tasks_joined", []))
    task_count += len(value.get("tasks_joined", []))
    client_receipts.append({
        "case": path.parent.name, "clients_closed": value["clients_closed"],
        "tasks_joined": value.get("tasks_joined", []),
    })
    for child in value.get("standalone_CLI_children", []):
        assert child["joined"] and child["original_identity_gone"] and child["returncode"] == 0
        identities.append({**child, "owner": "offline standalone CLI child"})
identities = [{**identity, "fresh_original_identity_gone": gone(identity)} for identity in identities]
assert all(item["fresh_original_identity_gone"] for item in identities)
save("FRESH_OWNED_GONE.json", {
    "identities": identities, "clients": client_receipts, "owned_async_tasks_joined": task_count,
    "all_known_helpers_joined_or_finished": True, "runtime_workers": [],
    "not_universal_engine_or_campaign_cleanup": True,
})

url_case = ARTIFACTS / "test_root03_canonicalization_preserves_strict_endpoint_binding"
replay_case = ARTIFACTS / "test_cli02_train_loop_records_replayable_forecasts_polling_and_shared_start"
pending_case = ARTIFACTS / "test_cli02_pending_restart_reuses_static_train_and_start_bindings"
describe_case = ARTIFACTS / "test_cli02_describe_is_offline_stable_and_standalone"
partial_case = ARTIFACTS / "test_cli02_input_refusals_and_partial_planning_inputs_are_retained"
canonical = read(url_case / "ROOT_URL_CANONICALIZATION.json")
replay = read(replay_case / "REALIZED_REPLAY.json")
pending = read(pending_case / "PENDING_RESTART.json")
assert canonical["real_MockTransport_GETs"] == 2 and canonical["POSTs"] == 0
assert len(canonical["refused_raw_URLs"]) == 21 and len(canonical["refused_endpoint_changes"]) == 7
assert replay["planner_rollouts"] == 16 and replay["offline_selected_path_replays"] == 2
assert len(replay["plans"]) == len(replay["observed_GET_receipts"]) == 2
assert [item["issued_commands"] for item in pending["results"]] == [1, 1, 1]
evidence_paths = [
    url_case / "ROOT_URL_CANONICALIZATION.json", replay_case / "REALIZED_REPLAY.json",
    pending_case / "PENDING_RESTART.json", describe_case / "DESCRIBE_BINDING.json",
    partial_case / "partial-plan.json", partial_case / "START_REFUSALS.json",
]
assert all(path.is_file() for path in evidence_paths)
save("EVIDENCE_INDEX.json", {
    "files": {str(path.relative_to(SCOPE)): {"sha256": sha(path), "bytes": path.stat().st_size}
              for path in evidence_paths},
    "canonicalization": canonical,
    "realized_CPU_replay": {key: replay[key] for key in
                            ("scope", "plans", "planner_rollouts", "offline_selected_path_replays", "startup")},
    "old_control_bodies_unchanged": 16,
    "qualification": "Actual CPU MockTransport observations and model replay, not Native admission evidence.",
})
save("FAILURES.json", {
    "current_generation": {"unexpected_control_failures": 0, "source_generations": 1,
                           "attempts": 1, "retries": 0, "post_freeze_source_edits": False},
    "immutable_predecessor": {
        "scope": str(OLD), "seal_sha256": sha(OLD / "SHA256SUMS"),
        "result": "1 PASS / 1 FAIL / 2 ERROR",
        "actual_defect": "Protocol root-URL canonicalization disagreed with raw-string runtime endpoint comparison.",
        "preserved_report": str(OLD / "REPORT.json"), "preserved_failures": str(OLD / "FAILURES.json"),
    },
    "current_fix": "One shared strict root canonicalizer is used by protocol creation, current endpoint binding and raw CLI URL validation before HTTPX can normalize a non-root path.",
    "expected_negative_evidence": [
        "Raw invalid URL, changed origin/path/userinfo/query/fragment and redirect refusals.",
        "Parser errors for mixed sources, missing permit and missing forecast purpose; retained in raw stderr.",
        "Intentional second-rollout refusal retains PlanningInput with the realized scenario before aborting.",
        "Transport timeout/malformed ACK, failed reobservation, source/state mutation, partial-fsync and concurrent-owner fixtures remain retained.",
    ],
    "raw_stdout": str((ATTEMPT / "stdout.log").relative_to(SCOPE)),
    "raw_stderr": str((ATTEMPT / "stderr.log").relative_to(SCOPE)),
})
save("INTERFACE.json", {
    "status": "AUTHOR_CPU_PASS_MAYOR_ACCEPTANCE_PENDING",
    "entry": "<same-host-python> -B <frozenRoot>/experiments/mpc-research/resume-mvp/nominal_runtime.py",
    "standalone": "HERE and ROOT bootstrap precede own SDK imports. Actual module paths are verified and recorded; standalone from outside Root with PYTHONPATH unset passed.",
    "constructor": "NominalRuntime(client:httpx.AsyncClient, planner:NominalPlanner, scenario_factory:Callable[[Snapshot],CausalScenario]|None, journal_path:Path, *, admin_key:str, scenario_binding:dict, permit:NominalDemoPermit|None=None, enabled:bool=False, policy='mpc', limits:RuntimeLimits=RuntimeLimits(), parameters:ModelParameters|None=None, clock=time.monotonic, sleeper=asyncio.sleep, start_at:float|None=None)",
    "run": "await runtime.run(); each decision uses an actual GET, fresh typed scenario/plan, only first action, precommit GET equality, durable pending/count before POST, guarded ACK and actual reobservation.",
    "CLI": {
        "arguments": "--url --journal --train-tape|--scenario-file --forecast-purpose train|evaluation|engineering --cost-spec-file --policy mpc|one_move --steps --episode-seconds --poll-seconds --start-at --describe-protocol --experimental-nominal-demo --nominal-permit",
        "default": "GET-only; no experimental POST without the explicit flag AND a separate scoped permit AND a causal input. ADMIN_API_KEY supplies the x-admin-api-key header; redirects and proxy environment disabled.",
        "description": "--describe-protocol computes actual protocol/hash/source bundle/identity/clock/module paths, with no HTTP client, approval read/create, journal write or waiting. --journal and the exact intended configuration remain binding inputs.",
        "root_URL": "HTTP(S) origin root with no trailing slash or one slash canonicalizes identically. Userinfo, query/fragment including empty markers, non-root paths, unsupported/relative forms, whitespace/control bytes and invalid ports refuse. Current origin changes still refuse.",
        "Train": "--train-tape <canonical train.tape.json> requires explicit --forecast-purpose. Freeze Train once with actual clock_domain, then make_scenario for every fresh observation. No Eval path or workload regeneration API.",
        "trusted_scenario_file": "Existing declared-scenario file compatibility retained; its exact file hash is bound. This is not a substitute for causal Train generation.",
        "limits": {"iterations_default": 4, "iterations_max": 32, "episode_s_default": 4.0,
                   "episode_s_max": 60.0, "poll_s_default": 0.0, "poll_s_max": 1.0,
                   "decision_s": 0.5, "solve_s": 0.1, "commands_per_journal_max": 8},
        "shared_start": "Optional absolute same-host monotonic --start-at; wait before run() captures its real entry, never after the first GET. New target must be nonpast and leave the episode inside min(CLI-entry+60s, enabled permit expiry). Offline description does not wait. Matching existing journal resumes its original episode without resetting the start/deadline.",
    },
    "stable_protocol": "Binds actual parameters/policy/limits, canonical endpoint, absolute journal path, start target and static Train prior/data/tokenizer/producer sources/RNG identity. Excludes ephemeral prior.available_at; identical CLI restart binding passed.",
    "source_sha256": metadata["source_digest_after_freeze"],
    "source_closure": "Actual public implementation hash plus permit/planner/runtime/scenario/workload file hashes; never echoed from approval fields.",
    "replay": "ProtocolBound retains parameters/protocol; PlanningInput saves bounded full CausalScenario/window before rollouts; Planned repeats the realized source and digest beside the decision. Two journal/snapshot-derived selected-plan replays matched real CPU predictions.",
    "forecast_receipt_limit": "The complete NominalForecast wrapper is not retained. Full realized CausalScenario plus static prior/RNG/source binding is sufficient for these rollout replays; do not claim independent RNG-wrapper reproduction.",
    "pending": "Persisted command count, run/domain/identity/source/protocol, original episode deadline and pending command survive restart. Unknown is fenced only by same-run observed budget-version advance or observed server expiry; no fallback/retry or another write in that reconciliation iteration.",
    "ACK": "CapApplied is a guarded Gateway admission-budget ACK, never Native token application or admission-behavior proof. ACK still requires real reobservation; hold is not a change. No ledger CAS claim.",
    "timing": "Original decision entry precedes first GET/journal startup. GET, solve, hashing, serialization and pre-POST fsync consume the same deadline. Returned finished_at follows final journal save; on-disk outcome_observed_at precedes that save.",
    "ownership": "No runtime subprocess/worker/executor/Native observer/SIGALRM. Runtime owns its journal flock only; caller closes AsyncClient, CLI does so itself. Client outcome capture, campaign and service/engine cleanup are separate Mayor/Role7 responsibilities.",
    "host_binding": "Describe and execute on the actual controller host/interpreter. Clock namespace, Python RNG runtime and absolute journal path are bound; never compare a GPU server's monotonic stamps to Mayor-local time.",
})
save("LIMITATIONS.json", {
    "validation": [
        "One local Python3.12.16 invocation: 17 PASS (four unchanged CLI02 + one URL regression + twelve existing runtime controls). No runtime03 Python3.14 or host Python3.12.3 run.",
        "All 38 input pins stayed stable before/after controls and at sealing; 16 prior test bodies were AST-identical. No additional generation, retry, resampling, peer suite or historical full suite.",
        "Only CPU MockTransport and one offline standalone CLI child; no real server, network, SSH, GPU inference or approval artifact.",
    ],
    "remaining_integration_blockers": [
        "Mayor independent acceptance of this corrected frozen composition is pending; author CPU PASS is not approval.",
        "Any real activation needs a separate scope-bound Mayor permit based on fresh same-host identity/run/clock and the freshly described actual source/protocol IDs. No such approval exists here.",
        "Actual-host tokenizer-loaded/no-fallback verification, verified local MODEL_NAME/served alias and exact CostSpec/immutable ledger costs remain mandatory preflight. Runtime cost checks alone do not prove tokenizer usage.",
        "Shared client/controller absolute start and actual driver-startup/final-drain wall envelope need external integration; client offered fates/SSE finish_reason/unknown metrics are not duplicated by this runtime.",
        "Strong Fixed/FIFO remains static pre-offer setup using observed GPU Train choice, not this per-tick loop or CPU freeze_fixed ranking.",
    ],
    "B_qualification": [
        "Requires at least two consecutive real timestamped observe/predict/plan/actual-first-action-apply/reobserve cycles plus actual admission behavior changes and admitted/rejected outcomes/overhead.",
        "Neither these CPU controls nor cap ACKs satisfy B. The four spaced low-load Train probe may not cross an admission boundary; no behavior-evidence claim.",
        "Mayor alone preregisters any minimal Train-only successor before Native; no Eval tuning, reseeding or moving criteria after Eval.",
    ],
    "legacy_and_ownership": [
        "Native14 remains rejected/unqualified; no legacy ActuationPermit/native_state/native_actions/calibration/native_http gate changes or waivers.",
        "Legacy D2 startup-SIGALRM R1 remains open and unused. This coordinator does not use the cut-consumer launcher or claim D2 clearance.",
        "Bounded service root/leaf shutdown, GPU locks/poison receipts and engine/campaign exclusivity remain external responsibilities. Only recorded owned CPU helpers/clients/tasks are checked here.",
        "Deadlines and filesystem operations are cooperative, not universal OS hard preemption; finite journal/I/O failures retain evidence and may halt rather than recover speculatively.",
    ],
    "not_claimed": "No Native qualification, model fitness, SLO guarantee, benchmark superiority, production readiness or overall ResumeReady-complete result.",
})
wrapper = [
    "env", "-u", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
    "PYTHONHASHSEED=0", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1", "nice", "-n19",
    metadata["interpreter"], "-B",
]
save("COMMANDS.json", {
    "cwd": str(Path.cwd()), "test": read(ATTEMPT / "command.json"),
    "preparation_argv": wrapper + [str(SCOPE / "prepare03.py")],
    "runner_argv": wrapper + [str(SCOPE / "run03.py")],
    "finisher_argv": wrapper + [str(SCOPE / "finish03.py")],
    "pre_edit": "Baseline INPUT_PINS.json 28 bindings and all 47 runtime02 manifest members were verified before editing; recorded in BASELINE_INPUT_PINS.json.",
    "preparation_output": "outputs/prepare.{stdout.log,stderr.log,exit}",
    "runner_output": "outputs/runner.{stdout.log,stderr.log,exit}",
    "test_output": "attempts/generation01-py312/{command.json,stdout.log,stderr.log,result.json}",
    "additional_controls_after_first_attempt": 0,
})
save("REPORT.json", {
    "schema": "resume-p0-nominal-runtime03-report-v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "AUTHOR_CPU_PASS_MAYOR_ACCEPTANCE_PENDING",
    "model": {"id": "gpt-6-astra", "reasoning_effort": "max", "provenance": "retained verified Role6 assignment"},
    "scope": "Small shared strict root-URL canonicalization correction; existing integration only.",
    "source_freeze_sha256": sha(SCOPE / "FREEZE.json"), "source_ids": freeze["source_ids"],
    "source_bundle_sha256": metadata["source_digest_after_freeze"], "source_diff": "SOURCE_DIFF.patch",
    "test_result": "17 PASS", "test_process": result,
    "selected_controls": freeze["selected_controls"], "unchanged_old_control_bodies": 16,
    "source_generations": 1, "test_attempts": 1, "retries": 0, "post_freeze_source_edits": False,
    "preserved_binding_count": len(pins), "immutable02_members_verified": old_members,
    "known_preparation_supervisor_test_CLI_identities_gone": len(identities),
    "clients_closed": sum(len(row["clients_closed"]) for row in client_receipts),
    "explicit_async_tasks_joined": task_count, "cleanup_receipts": len(cleanups),
    "accepted_by_Mayor": False, "Native_permission": False, "real_approval_created": False,
    "qualification": "CPU integration PASS only; B admission behavior and Native qualification remain unproved.",
    "seal_exclusions": ["SHA256SUMS", "SEAL_RECEIPT.json"],
})
save("HANDOFF.json", {
    "status": "AUTHOR_CPU_PASS_AND_IDLE", "report": "REPORT.json", "interface": "INTERFACE.json",
    "failure_detail": "FAILURES.json", "limits": "LIMITATIONS.json", "cleanup": "FRESH_OWNED_GONE.json",
    "source_freeze": "FREEZE.json", "source_diff": "SOURCE_DIFF.patch", "commands": "COMMANDS.json",
    "evidence_index": "EVIDENCE_INDEX.json", "manifest": "SHA256SUMS", "seal_receipt": "SEAL_RECEIPT.json",
    "remaining_blocker": "Mayor independent acceptance and separately authorized same-host Native integration/preflight; no remaining failure in this selected CPU attempt.",
})
files = sorted(path for path in SCOPE.rglob("*") if path.is_file())
assert all(not path.is_symlink() for path in files)
lines = [f"{sha(path)}  {path.relative_to(SCOPE).as_posix()}\n" for path in files]
with (SCOPE / "SHA256SUMS").open("x") as stream:
    stream.writelines(lines)
for line in lines:
    expected, name = line.rstrip("\n").split("  ", 1)
    assert sha(SCOPE / name) == expected
seal = sha(SCOPE / "SHA256SUMS")
save("SEAL_RECEIPT.json", {
    "manifest_sha256": seal, "members": len(files), "all_members_verified": True,
    "all_source_bindings_unchanged": True, "known_identities_gone": len(identities),
    "status": "AUTHOR_CPU_PASS_NO_NATIVE_PERMISSION", "receipt_not_self_in_manifest": True,
})
print(json.dumps({
    "seal": seal, "members": len(files), "status": "17 PASS; one corrected generation, no retry",
    "known_identities_gone": len(identities), "clients_closed": sum(len(row["clients_closed"]) for row in client_receipts),
    "async_tasks_joined": task_count, "source_generations": 1,
}, sort_keys=True))
