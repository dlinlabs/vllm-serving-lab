"""Seal the one-shot result and its distinct refusal-classification defect; no rerun."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode03")
OLD = Path(".runtime/r24-resume-p0-nominal-episode02")
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
old_members = verify_manifest(OLD, "0ecc3f73638cc0bcd312962dd077bc987553ee969d40052a5be9ef4df7da6f2e")
api_members = verify_manifest(API, "b3e6174797f8d466181ebf42e87b5c45133341ff7351265d166d0cfc1dfc8a1b")
result = read(ATTEMPT / "result.json")
log = (ATTEMPT / "stderr.log").read_text()
assert result["returncode"] == 1 and not result["timed_out"] and result["elapsed_s"] < 60
assert result["joined"] and result["original_identity_gone"]
assert "Ran 4 tests" in log and "FAILED (failures=1)" in log
assert "kind='runtime_endpoint_changed'" in log
for name in ("pins-before.json", "pins-after.json"):
    assert all(read(ATTEMPT / name).values())
for name in ("start", "prepare"):
    assert (SCOPE / f"outputs/{name}.exit").read_text().strip() == "0"
    assert not (SCOPE / f"outputs/{name}.stderr.log").read_bytes()
assert (SCOPE / "outputs/runner.exit").read_text().strip() == "1"
assert not (SCOPE / "outputs/runner.stderr.log").read_bytes()
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
assert len(episode_rows) == 15
with_workers = [row for row in episode_rows if row["workers"]]
assert len(with_workers) == 5 and all(set(row["workers"]) == {"controller", "recorder"} for row in with_workers)
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
assert len(runtime["runtime"]["steps"]) == 3
assert [row["status"] for row in runtime["runtime"]["steps"]] == ["hold"] * 3
assert any(row["active"] > 0 for row in shared_evidence["runtime_GETs"])
assert shared_evidence["CPU_budget_commands"] == []
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
assert len(returned_failures) == 3 and all(row["status"] == "failed" for row in returned_failures)
no_start_path = ARTIFACTS / "test_default_off_preparation_and_configuration_refusals_do_not_start"
no_start_passes = read(no_start_path / "NO_START_REFUSALS.json")
assert len(no_start_passes) == 9
for kind in ("binding_changed", "provider_changed", "policy_changed"):
    value = read(no_start_path / kind / "episode.json")
    assert value["status"] == "refused" and value["workers"] == {}
    assert value["failures"][0]["error"]["message"] == "runtime_Train_provider_or_policy_changed"
unexpected = read(no_start_path / "runtime_endpoint_changed/episode.json")
assert unexpected["status"] == "failed" and unexpected["workers"] == {}
assert unexpected["failures"][0]["error"]["type"] == "RuntimeRefused"
assert unexpected["failures"][0]["error"]["message"] == "runtime_configuration_or_parameter_binding_changed"
save("OBSERVED_EXECUTION.json", {
    "episode_invocations": episode_rows, "invocation_count": len(episode_rows),
    "invocations_with_two_workers": len(with_workers), "worker_task_receipts_joined": 2 * len(with_workers),
    "shared_start_case": shared_evidence, "runtime_step_statuses": ["hold"] * 3,
    "shared_start": shared["offered_start"], "original_deadline": shared["deadline"],
    "original_reserve": shared["cleanup_reserve_s"], "cancellation_fates": cancel_evidence["fates"],
    "cancellation_runtime_lock_reacquired": cancel_evidence["runtime_lock_reacquired"],
    "returned_failure_envelope_cases": returned_failures,
    "no_start_subcases": {"expected_results": 9, "classification_failure": 1},
    "Native_actual_admission_effect": None, "Native_ordered_gateway_ledger": None, "full_wall": None,
    "not_actual_apply_evidence": "Successful case has three actual observe/forecast/plan/hold cycles, live mock request-ledger feedback and no budget POST. No native actuation/admission qualification.",
})
save("FAILURES.json", {
    "overall": "3 PASS / 1 FAIL method (one failed subcase); unittest failures=1, no errors; no retry.",
    "predecessor": {"seal": "0ecc3f73638cc0bcd312962dd077bc987553ee969d40052a5be9ef4df7da6f2e",
                    "preserved_members": old_members, "result": "8 failures/1 error retained unchanged."},
    "resolved_here": {
        "E02-1": "Canonical digest equality accepts unchanged tuple-containing Train binding; real recorder/controller workers reached. Changed binding content, provider object and policy each refuse with the exact existing reason.",
        "E02-2": "15 actual same-coroutine before/after handler checks are true, including cancellation; original cross-Runner assertions are not suppressed or mislabeled.",
    },
    "distinct_defect": {
        "id": "E03-1", "owner": "nominal_episode.py pre-worker exception-envelope classification",
        "control": "test_default_off_preparation_and_configuration_refusals_do_not_start/runtime_endpoint_changed",
        "observed": "Accepted runtime._binding correctly raises RuntimeRefused before either worker is created. Adapter recognizes only EpisodeRefused as a refused preflight, so it returns status=failed rather than expected refused.",
        "raw": unexpected,
        "safety": "Guard remained closed; no controller/replay workers or runtime journal in this case. This is classification, not action leakage or a peer/runtime defect.",
        "smallest_next_correction_not_applied": "In the existing no-workers exception branch, recognize runtime_api.RuntimeRefused alongside EpisodeRefused. Keep all other exceptions failed and all guards unchanged; do not merely relax the assertion.",
    },
    "expected_injected_failures": "Controller GET refusal, recorder model refusal and recorder transport failure all propagated as failed envelopes and cancelled/joined the other worker with raw captures preserved.",
    "write_failure_scope": "Final capture persistence occurs in the unchanged finally path. Write errors remain recorded and raised fail-closed; malformed/nonconforming N1 payload/write-failure injection was not added or tested here.",
    "all_original_logs": "attempts/generation01-py312/{stdout.log,stderr.log,result.json}; outputs/{start,prepare,runner}.*",
    "source_generations": 1, "attempts": 1, "post_failure_repairs": 0, "blind_retries": 0,
    "Native_failed_experiment": False,
})
interface = read(OLD / "INTERFACE.json")
interface.update({
    "status": "FAILED_ONE_CLASSIFICATION_CONTROL_DO_NOT_ACTIVATE",
    "current_verified_scope": "Real composed MockTransport execution: one full five-request shared-start collection with three fresh controller holds and active mock ledger feedback; three returned-failure branches; caller cancellation with partial fates and worker joins.",
    "current_blocker": "Pre-worker runtime_api.RuntimeRefused is returned as failed, not refused. One existing subcase fails; no in-scope repair or retry.",
    "separate_authority": "Parent reports Role4 RECORDERPASS/METRICSPASS conditional SOURCE gate prefix3dcbc992 at unchanged client933e4161/metrics49c76b8b; this is not Native/actual comparability approval.",
    "caller_persistence": "Inspect status/errors and attempt every raw capture write in finally. This adapter does so; write errors are retained in the receipt/exception and raise fail-closed. If writing itself fails, durable raw evidence is not guaranteed. Caller still owns final cleanup/full-wall completion.",
})
save("INTERFACE.json", interface)
save("LIMITATIONS.json", {
    "activation": "One new envelope-classification control failed. No deployment or Native approval; stop at this distinct defect.",
    "positive_CPU_scope": [
        "Shared absolute S and original D/R actually reached both APIs without retiming. Five original offers were sent; four complete/one rejected in the success fixture.",
        "Controller performed three fresh forecast/plan/hold steps and observed four active mock ledger entries on the third GET. It issued zero budget POSTs; actual apply/reobserve and admission-change qualification are not shown here.",
        "Caller cancellation preserved five partial fates, joined recorder/controller workers and allowed runtime journal-lock reacquisition. Returned non-collected envelopes cancelled the peer worker.",
        "All15 handler checks sampled within the same coroutine/Runner invocation. Test cleanup has no remaining tasks or active mock streams; known process identities are gone.",
    ],
    "unchanged_boundaries": [
        "Only canonical digest expression changed in implementation; exact policy/provider identity and all runtime guards remain. Three original positive test bodies are AST-identical after accounting for the signal-check wrapper; fixture/timings unchanged.",
        "No write-failure/nonconforming N1 input injection, fixed-body N2 review, actual tokenizer validation, full suite, second interpreter, retry, model/workload tuning or natural engineering metrics finalization.",
        "A codec/filesystem failure can prevent raw and/or episode files from being written. The unchanged code retains failure on the returned/raised receipt and fails closed, not a universal durability guarantee.",
        "CPU-only tokenizer interface/fingerprint and in-memory permit/source patches are explicit fixtures, not a live permit or model qualification.",
        "NominalServices remains external. No service spawning, threads, signals, subprocesses by the adapter, GPU, real HTTP, SSH, Git, installs, lease assumption or additional agents.",
    ],
    "parent_gates": [
        "Parent reports conditional Role4 source PASS at unchanged client/metrics pins. Exact adapter acceptance and actual-host preflight remain Mayor-owned.",
        "B requires at least two consecutive real observe/predict/plan/actual-first-action-apply/reobserve cycles plus actual admission behavior/ordered gateway ledger evidence. Three CPU holds and synthetic rejection are not that evidence.",
        "No fallback tokenizer/cost, source, clock, permit, pending, age, version or deadline guards may be weakened to obtain actions.",
        "Full wall, root/leaf shutdown, lock/poison receipts and Native admission evidence require actual Mayor-owned cleanup/execution. Unknown fields remain null.",
        "Joins remain cooperative; these CPU receipts do not establish hard preemption or universal root/service cleanup.",
    ],
    "budget": read(SCOPE / "SOURCE_READ_BUDGET.json"),
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
    "pre_edit_pins": wrapper + [str(SCOPE / "start03.py")],
    "freeze": wrapper + [str(SCOPE / "prepare03.py")], "runner": wrapper + [str(SCOPE / "run03.py")],
    "finisher": wrapper + [str(SCOPE / "finish03.py")], "test_attempts": 1,
    "after_failure": "Read-only artifact/traceback diagnosis and metadata sealing; no source execution or test rerun.",
})
save("REPORT.json", {
    "schema": "nominal-episode03-report-v1", "status": "3PASS_1FAIL_DISTINCT_PREFLIGHT_CLASSIFICATION_STOP",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_ids": freeze["source_ids"], "source_freeze_sha256": sha(SCOPE / "FREEZE.json"),
    "source_diff": "SOURCE_DIFF.patch", "source_generations": 1, "test_attempts": 1, "retries": 0,
    "test_result": "3 PASS / 1 FAIL method, one existing subcase fails; failures=1/errors=0",
    "test_process": result, "selected_controls": freeze["selected_controls"],
    "new_discrimination_subcases": 3, "new_discrimination_result": "all refused before workers with exact expected reason",
    "same_coroutine_signal_checks": cleanup["same_coroutine_signal_checks"],
    "input_pins_preserved": len(pins), "peer_source_count": 6, "peer_source_bytes": 200013,
    "episode02_members_preserved": old_members, "API03_members_reverified": api_members,
    "episode_invocations": len(episode_rows), "invocations_reaching_worker_tasks": len(with_workers),
    "worker_tasks_joined": 2 * len(with_workers), "clients_closed": cleanup["clients_closed"],
    "transports_closed": cleanup["transports_closed"], "outer_test_tasks_joined": cleanup["outer_test_tasks_joined"],
    "known_process_identities_gone": len(identities), "source_edits_after_freeze": False,
    "positive_shared_start_result": {"fates": shared_capture["summary"]["fates"], "runtime_steps": ["hold"] * 3,
                                     "CPU_budget_POSTs": 0, "actual_Native_admission_effect": None},
    "remaining_blocker": "E03-1: pre-worker RuntimeRefused classified failed rather than refused",
    "peers_edited": False, "Mayor_accepted": False, "Native_permission": False, "Native_failed_experiment": False,
    "qualification": None, "seal_exclusions": ["SHA256SUMS", "SEAL_RECEIPT.json"],
})
save("HANDOFF.json", {
    "status": "DISTINCT_FAILURE_PRESERVED_AND_IDLE", "report": "REPORT.json", "failure": "FAILURES.json",
    "API": "INTERFACE.json", "limitations": "LIMITATIONS.json", "cleanup": "FRESH_OWNED_GONE.json",
    "source": "FREEZE.json", "diff": "SOURCE_DIFF.patch", "commands": "COMMANDS.json",
    "actual_CPU_execution": "OBSERVED_EXECUTION.json", "manifest": "SHA256SUMS",
    "minimum_next_change": "Typed pre-worker RuntimeRefused classification only if separately authorized. No guard relaxation, retiming or peer changes.",
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
    "status": "3PASS_1FAIL_NO_NATIVE_PERMISSION", "receipt_not_self_in_manifest": True,
})
print(json.dumps({
    "seal": seal, "members": len(files), "status": "3PASS/1FAIL, no retry",
    "known_identities_gone": len(identities), "workers_joined": 2 * len(with_workers),
    "clients_closed": cleanup["clients_closed"], "transports_closed": cleanup["transports_closed"],
    "source_bindings_unchanged": len(pins), "signal_checks": cleanup["same_coroutine_signal_checks"],
}, sort_keys=True))
