"""Seal the complete CPU result, original failures, exact pins and bounded cleanup."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode04")
OLD = Path(".runtime/r24-resume-p0-nominal-episode03")
EARLIER = Path(".runtime/r24-resume-p0-nominal-episode02")
API = Path(".runtime/r24-resume-p0-nominal-client-api03")
ATTEMPT = SCOPE / "attempts/generation01-py312"
ARTIFACTS = ATTEMPT / "artifacts"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (SCOPE / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_manifest(path, expected):
    assert sha(path / "SHA256SUMS") == expected
    names = []
    for line in (path / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        assert sha(path / name) == checksum, name
        names.append(name)
    return len(names)


def gone(identity):
    try:
        ticks = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return True
    return ticks != identity["start_ticks"]


assert not (SCOPE / "SHA256SUMS").exists()
freeze = read(SCOPE / "FREEZE.json")
pins = read(SCOPE / "INPUT_PINS.json")
checks = {name: sha(Path(name)) == row["sha256"] for name, row in pins.items()}
assert all(checks.values())
for name, row in freeze["source_ids"].items():
    assert sha(SCOPE / "sources" / Path(name).name) == row["sha256"]
save("SOURCE_POSTCHECK.json", checks)
old_members = verify_manifest(OLD, "07da8f077fc42b636c2bc04fa813a7ffa9d5be4de24deab8b43621062b6bced2")
earlier_members = verify_manifest(EARLIER, "0ecc3f73638cc0bcd312962dd077bc987553ee969d40052a5be9ef4df7da6f2e")
api_members = verify_manifest(API, "b3e6174797f8d466181ebf42e87b5c45133341ff7351265d166d0cfc1dfc8a1b")
ancestry = []
for old, label, count in ((EARLIER, "episode02", earlier_members), (OLD, "episode03", old_members)):
    destination = SCOPE / "original-failures" / label
    destination.mkdir(parents=True)
    copies = []
    for name in ("FAILURES.json", "REPORT.json", "attempts/generation01-py312/stderr.log",
                 "attempts/generation01-py312/stdout.log", "attempts/generation01-py312/result.json"):
        source, target = old / name, destination / Path(name).name
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        copies.append({"source": str(source), "copy": str(target.relative_to(SCOPE)), "sha256": sha(target)})
    ancestry.append({"capsule": str(old), "seal": sha(old / "SHA256SUMS"),
                     "all_original_members_verified": count, "failure_copies": copies})
save("ORIGINAL_FAILURES.json", ancestry)
result = read(ATTEMPT / "result.json")
log = (ATTEMPT / "stderr.log").read_text()
assert result["returncode"] == 0 and not result["timed_out"] and result["elapsed_s"] < 60
assert result["joined"] and result["original_identity_gone"]
assert "Ran 4 tests" in log and log.rstrip().endswith("OK") and "\nFAILED " not in log
for name in ("pins-before.json", "pins-after.json"):
    assert all(read(ATTEMPT / name).values())
for name in ("start", "prepare", "runner"):
    assert (SCOPE / f"outputs/{name}.exit").read_text().strip() == "0"
    assert not (SCOPE / f"outputs/{name}.stderr.log").read_bytes()
identities = [
    {**read(SCOPE / "STARTED.json")["identity"], "owner": "initial pinning"},
    {**freeze["preparation_identity"], "owner": "source freezing"},
    {**read(ATTEMPT / "runner-identity.json"), "owner": "bounded test supervisor"},
    {**result["original_identity"], "owner": "unittest process"},
]
identities = [{**item, "fresh_original_identity_gone": gone(item)} for item in identities]
assert all(item["fresh_original_identity_gone"] for item in identities)
cleanup_rows = []
for path in sorted(ARTIFACTS.glob("*/OWNED_CLEANUP.json")):
    value = read(path)
    assert all(value["clients_closed"]) and all(value["transports_closed"])
    assert all(value["test_owned_tasks_joined"]) and value["remaining_loop_tasks"] == []
    assert all(count == 0 for count in value["active_CPU_stream_ledgers"])
    assert value["processes_or_services_started"] == [] and value["signal_handlers_unchanged"] is True
    assert value["signal_checks"] and all(row["same_coroutine"] and row["unchanged"]
                                         for row in value["signal_checks"])
    cleanup_rows.append({"case": path.parent.name, "raw_receipt": str(path.relative_to(SCOPE)), **value})
assert len(cleanup_rows) == 4
cleanup = {
    "identities": identities, "cases": cleanup_rows, "known_helpers_gone": True,
    "clients_closed": sum(len(row["clients_closed"]) for row in cleanup_rows),
    "transports_closed": sum(len(row["transports_closed"]) for row in cleanup_rows),
    "outer_test_tasks_joined": sum(len(row["test_owned_tasks_joined"]) for row in cleanup_rows),
    "same_coroutine_signal_checks": sum(len(row["signal_checks"]) for row in cleanup_rows),
    "scope": "Known CPU processes, async tasks, clients and MockTransports only; not actual services/root-leaf/native cleanup.",
}
save("FRESH_OWNED_GONE.json", cleanup)
episode_rows = []
for path in sorted(ARTIFACTS.rglob("episode.json")):
    value = read(path)
    capture = read(path.parent / "capture.json")
    assert value["all_workers_joined"] and value["persistence_errors"] == []
    assert all(worker["joined"] for worker in value["workers"].values())
    assert sha(path.parent / "capture.json") == value["capture_sha256"]
    assert all(value[key] is None for key in ("admission_effect", "ordered_gateway_ledger", "full_wall", "qualification"))
    episode_rows.append({
        "path": str(path.relative_to(SCOPE)), "status": value["status"],
        "workers": {name: {"joined": worker["joined"],
                           "error_type": worker["error"]["type"] if worker["error"] else None}
                    for name, worker in value["workers"].items()},
        "failures": value["failures"], "capture_status": capture["status"], "capture_error": capture["error"],
        "recorded_fates": ([row["fate"] for row in capture["records"]] if capture["records"] is not None else None),
        "missing_record_ids": value["missing_record_ids"], "capture_sha256": value["capture_sha256"],
        "runtime_journal_present": (path.parent / "runtime-journal.json").is_file(),
    })
assert len(episode_rows) == 16
with_workers = [row for row in episode_rows if row["workers"]]
assert len(with_workers) == 6 and all(set(row["workers"]) == {"controller", "recorder"} for row in with_workers)
shared_path = ARTIFACTS / "test_shared_start_full_outcomes_and_real_controller_feedback"
shared_evidence = read(shared_path / "COMPOSITION_EVIDENCE.json")
shared = read(shared_path / "complete/episode.json")
shared_capture = read(shared_path / "complete/capture.json")
runtime = shared["workers"]["controller"]["result"]
assert shared["status"] == "collected" and shared["failures"] == []
assert shared_capture["status"] == "collected" and shared_capture["summary"]["offered"] == 5
assert shared_capture["summary"]["sent"] == 5 and len(shared_capture["records"]) == 5
assert shared["offered_start"] == shared_capture["offered_start"] == runtime["startup"]["requested_start_at"]
assert shared["deadline"] == shared_capture["config"]["caller_deadline"]
assert shared["cleanup_reserve_s"] == shared_capture["config"]["cleanup_reserve_s"]
assert shared_capture["drain_deadline"] == shared["deadline"] - shared["cleanup_reserve_s"]
statuses = [row["status"] for row in runtime["runtime"]["steps"]]
assert statuses == ["hold"] * 3 and shared_evidence["CPU_budget_commands"] == []
assert any(row["active"] > 0 for row in shared_evidence["runtime_GETs"])
cancel_path = ARTIFACTS / "test_caller_cancellation_joins_workers_and_persists_partial_outcomes"
cancel_evidence = read(cancel_path / "CANCELLATION_EVIDENCE.json")
cancel = read(cancel_path / "cancel/episode.json")
cancel_capture = read(cancel_path / "cancel/capture.json")
assert cancel["status"] == cancel_capture["status"] == "interrupted"
assert cancel["missing_record_ids"] == [] and len(cancel_capture["records"]) == 5
assert any(row["fate"] == "cancelled" for row in cancel_capture["records"])
assert cancel_evidence["runtime_lock_reacquired"] and cancel_evidence["all_workers_joined"]
failure_path = ARTIFACTS / "test_returned_failure_envelopes_cancel_peer_and_preserve_results"
returned_failures = read(failure_path / "RETURNED_FAILURES.json")
assert len(returned_failures) == 4 and all(row["status"] == "failed" for row in returned_failures)
post_worker = read(failure_path / "controller_exception_refused/episode.json")
assert post_worker["status"] == "failed" and set(post_worker["workers"]) == {"controller", "recorder"}
assert post_worker["workers"]["controller"]["error"]["type"] == "RuntimeRefused"
assert post_worker["joined_at"] >= post_worker["offered_start"]
post_capture = read(failure_path / "controller_exception_refused/capture.json")
assert post_capture["status"] == "interrupted"
assert [row["fate"] for row in post_capture["records"]] == ["not_launched"] * 5
no_start_path = ARTIFACTS / "test_default_off_preparation_and_configuration_refusals_do_not_start"
no_start = read(no_start_path / "NO_START_REFUSALS.json")
assert len(no_start) == 10 and all(row["status"] == "refused" for row in no_start)
pre_worker = read(no_start_path / "runtime_endpoint_changed/episode.json")
assert pre_worker["status"] == "refused" and pre_worker["workers"] == {}
assert pre_worker["failures"][0]["error"]["type"] == "RuntimeRefused"
assert pre_worker["failures"][0]["error"]["message"] == "runtime_configuration_or_parameter_binding_changed"
for kind in ("binding_changed", "provider_changed", "policy_changed"):
    value = read(no_start_path / kind / "episode.json")
    assert value["status"] == "refused" and value["workers"] == {}
    assert value["failures"][0]["error"]["message"] == "runtime_Train_provider_or_policy_changed"
save("OBSERVED_EXECUTION.json", {
    "episode_invocations": episode_rows, "invocation_count": len(episode_rows),
    "invocations_with_two_workers": len(with_workers), "worker_task_receipts_joined": 2 * len(with_workers),
    "shared_start_case": shared_evidence, "runtime_step_statuses": statuses,
    "shared_start": shared["offered_start"], "original_deadline": shared["deadline"],
    "original_reserve": shared["cleanup_reserve_s"], "cancellation_fates": cancel_evidence["fates"],
    "cancellation_runtime_lock_reacquired": cancel_evidence["runtime_lock_reacquired"],
    "explicit_failed_envelope_cases": returned_failures, "no_start_refusals": len(no_start),
    "typed_discrimination": {"before_workers": pre_worker, "after_workers": post_worker},
    "Native_actual_admission_effect": None, "Native_ordered_gateway_ledger": None, "full_wall": None,
    "not_actual_apply_evidence": "Success fixture has three real observe/forecast/plan/hold cycles and active mock ledger feedback; zero budget POSTs. It is not Native apply/admission qualification.",
})
save("FAILURES.json", {
    "current_unexpected_failures": [], "current_result": "4/4 methods PASS; zero failures/errors; one attempt",
    "original_failures": "Unmodified episode02 eight failures/one error and episode03 three PASS/one FAIL remain sealed. Exact raw logs/results/reports copied under original-failures/ and indexed in ORIGINAL_FAILURES.json.",
    "resolved_E03-1": {
        "before_workers": {"status": pre_worker["status"], "error": pre_worker["failures"][0]["error"], "workers": {}},
        "after_workers": {"status": post_worker["status"], "error": post_worker["workers"]["controller"]["error"]},
        "change": "Only add runtime_api.RuntimeRefused to the existing typed no-workers refusal branch; no other implementation bytes or guards changed.",
    },
    "explicit_expected_failures": returned_failures,
    "cancellation": "Caller cancellation remains interrupted and is re-raised after persistence and joining; not relabeled refused.",
    "persistence": "All16 final captures/episode receipts were written without errors. Existing fail-closed persistence error code is unchanged; malformed N1 data and filesystem-write failure were not injected.",
    "non_failure_diagnostic_retained": "Unittest asyncio debug logs a 0.101-second synchronous no-start control task. It is retained, not suppressed, and no deadline/test failed.",
    "source_generations": 1, "attempts": 1, "retries": 0, "source_edits_after_freeze": False,
    "Native_failed_experiment": False,
})
interface = read(OLD / "INTERFACE.json")
interface.update({
    "status": "AUTHOR_CPU_PASS_READY_FOR_MAYOR_EXACT_SOURCE_ACCEPTANCE",
    "current_verified_scope": "Four passing focused methods: actual composed MockTransport shared-start five-request replay and three fresh controller holds with active ledger feedback; ten no-start guards; four explicit returned/raised failure branches; caller cancellation, raw capture persistence and async joins.",
    "current_blocker": None,
    "error_classification": "Before workers: EpisodeRefused or runtime_api.RuntimeRefused -> refused. Worker failures/non-collected envelopes -> failed; caller cancellation -> interrupted then re-raised; persistence errors remain fail-closed exceptions.",
    "new_discrimination_fixture": "Only the new post-worker RuntimeRefused branch replaces one test runtime.run with AsyncMock raising the exact class. Shared-start, genuine observer refusal and cancellation controls retain real runtime/planner APIs.",
    "external_authority": "Default OFF, separate valid scoped permit, exact-source acceptance and Mayor-owned services/actual-host preflight remain required. This is a usable CPU-validated library, not Native execution authority.",
})
save("INTERFACE.json", interface)
save("LIMITATIONS.json", {
    "CPU_delivery": "One local Python3.12.16 attempt: four methods PASS. Usable two-phase in-process adapter; parent exact-source acceptance remains separate.",
    "what_ran": [
        "Original five-request engineering probe and shared future S, original D/R unchanged. Success has four complete/one rejected CPU fates, three holds, four active mock ledger entries observed, no budget POST.",
        "Six invocations reached both worker tasks; twelve worker task receipts joined. Cancellation retained partial fates and runtime-lock reacquisition.",
        "Ten no-start guards and four explicit failure cases passed; RuntimeRefused before/after workers is now discriminated.",
        "Sixteen same-coroutine signal checks passed. All test-owned clients/transports/tasks are closed/joined; no test processes remain.",
    ],
    "not_proven": [
        "No actual tokenizer/model/GPU validation, inference, real HTTP, CAS actuation, native admission effect, service lifecycle or full-wall completion by this task.",
        "Three CPU holds do not establish two actual first-action apply/reobserve cycles or an admission change. B qualification remains absent.",
        "No second interpreter, full suite, retry, model/workload/port/timing changes, natural engineering metrics finalization or future Eval stamp.",
        "Write failures/nonconforming N1 payloads and fixed-body N2 concerns were not injected/reviewed. If codec/filesystem writing fails, raw durability is not guaranteed; unchanged code retains failure on receipt/exception and fails closed.",
        "Async joins are cooperative, not a universal hard-preemption or root/leaf shutdown/lock/poison guarantee.",
    ],
    "source_scope": "Only the typed pre-worker classification changed in implementation; test adds one same-class post-worker discrimination. All other test methods, fixtures, clocks, ports, signals and guards are unchanged.",
    "parent_lifecycle_provenance_only": read(SCOPE / "PLAN.json")["parent_provenance_only"],
    "parent_obligations": [
        "Exact-source approval and separate scoped permit; existing resource/lease authority remains Mayor-only.",
        "Actual local model/alias/tokenizer/no-fallback/cost/source/clock/identity/pending/version/age/deadline preflight without guard relaxation.",
        "Actual offered fates, ordered gateway ledger/admission effect and at least two consecutive real action/apply/reobserve cycles for B.",
        "Final service/drain/root-leaf/lock/poison/full-wall evidence after actual Mayor-owned cleanup. Unknown evidence remains null.",
    ],
    "read_budget": read(SCOPE / "SOURCE_READ_BUDGET.json"),
})
metadata = read(SCOPE / "SOURCE_METADATA.json")
wrapper = [
    "env", "-u", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
    "PYTHONHASHSEED=0", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1",
    "nice", "-n19", metadata["interpreter"], "-B",
]
save("COMMANDS.json", {
    "cwd": str(Path.cwd()), "test": read(ATTEMPT / "command.json"),
    "pre_edit_pins": wrapper + [str(SCOPE / "start04.py")],
    "freeze": wrapper + [str(SCOPE / "prepare04.py")], "runner": wrapper + [str(SCOPE / "run04.py")],
    "finisher": wrapper + [str(SCOPE / "finish04.py")], "test_attempts": 1,
    "after_controls": "Read-only artifact checks, predecessor-failure copying and metadata sealing; no rerun.",
})
save("REPORT.json", {
    "schema": "nominal-episode04-report-v1", "status": "AUTHOR_CPU_4PASS_NO_NATIVE_AUTHORITY",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_ids": freeze["source_ids"], "source_freeze_sha256": sha(SCOPE / "FREEZE.json"),
    "source_diff": "SOURCE_DIFF.patch", "source_generations": 1, "test_attempts": 1, "retries": 0,
    "test_result": "4/4 methods PASS; failures=0/errors=0", "test_process": result,
    "selected_controls": freeze["selected_controls"], "new_discrimination_subcases": 1,
    "resolved_defect": "E03-1: typed pre-worker RuntimeRefused classification",
    "same_coroutine_signal_checks": cleanup["same_coroutine_signal_checks"],
    "input_pins_preserved": len(pins), "peer_source_count": 6, "peer_source_bytes": 200013,
    "episode03_members_preserved": old_members, "episode02_members_preserved": earlier_members,
    "API03_members_reverified": api_members, "episode_invocations": len(episode_rows),
    "invocations_reaching_worker_tasks": len(with_workers), "worker_tasks_joined": 2 * len(with_workers),
    "clients_closed": cleanup["clients_closed"], "transports_closed": cleanup["transports_closed"],
    "outer_test_tasks_joined": cleanup["outer_test_tasks_joined"], "known_process_identities_gone": len(identities),
    "source_edits_after_freeze": False, "peer_edits": False,
    "positive_shared_start_result": {"fates": shared_capture["summary"]["fates"], "runtime_steps": statuses,
                                     "CPU_budget_POSTs": 0, "actual_Native_admission_effect": None},
    "remaining_author_CPU_blocker": None, "Mayor_accepted": False, "Native_permission": False,
    "qualification": None, "seal_exclusions": ["SHA256SUMS", "SEAL_RECEIPT.json"],
})
save("HANDOFF.json", {
    "status": "AUTHOR_CPU_PASS_AND_IDLE", "report": "REPORT.json", "failures": "FAILURES.json",
    "original_failures": "ORIGINAL_FAILURES.json", "API": "INTERFACE.json", "limitations": "LIMITATIONS.json",
    "cleanup": "FRESH_OWNED_GONE.json", "source": "FREEZE.json", "diff": "SOURCE_DIFF.patch",
    "commands": "COMMANDS.json", "actual_CPU_execution": "OBSERVED_EXECUTION.json", "manifest": "SHA256SUMS",
    "next_authority": "Mayor exact-source acceptance/host preflight only; no agent execution authority or lease assumption.",
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
    "source_bindings_unchanged": len(pins), "known_identities_gone": len(identities),
    "status": "AUTHOR_CPU_4PASS_NO_NATIVE_PERMISSION", "receipt_not_self_in_manifest": True,
})
print(json.dumps({
    "seal": seal, "members": len(files), "status": "4/4 PASS, one attempt, no Native authority",
    "known_identities_gone": len(identities), "workers_joined": 2 * len(with_workers),
    "clients_closed": cleanup["clients_closed"], "transports_closed": cleanup["transports_closed"],
    "source_bindings_unchanged": len(pins), "signal_checks": cleanup["same_coroutine_signal_checks"],
}, sort_keys=True))
