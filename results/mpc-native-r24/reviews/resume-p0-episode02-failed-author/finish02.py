"""Seal the failed one-shot episode attempt; no repair, rerun or client execution."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode02")
ATTEMPT = SCOPE / "attempts/generation01-py312"
ARTIFACTS = ATTEMPT / "artifacts"
API = Path(".runtime/r24-resume-p0-nominal-client-api03")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (SCOPE / name).open("x") as stream:
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
checks = {name: sha(Path(name)) == row["sha256"] for name, row in pins.items()}
assert all(checks.values())
for name, row in freeze["source_ids"].items():
    assert sha(SCOPE / "sources" / Path(name).name) == row["sha256"]
save("SOURCE_POSTCHECK.json", checks)
result = read(ATTEMPT / "result.json")
log = (ATTEMPT / "stderr.log").read_text()
assert result["returncode"] == 1 and not result["timed_out"] and result["elapsed_s"] < 60
assert result["joined"] and result["original_identity_gone"]
assert "Ran 4 tests" in log and "FAILED (failures=8, errors=1)" in log
assert all(read(ATTEMPT / "pins-before.json").values())
assert all(read(ATTEMPT / "pins-after.json").values())
assert (SCOPE / "outputs/prepare.exit").read_text().strip() == "0"
assert (SCOPE / "outputs/runner.exit").read_text().strip() == "1"
assert not (SCOPE / "outputs/prepare.stderr.log").read_bytes()
assert not (SCOPE / "outputs/runner.stderr.log").read_bytes()
api_members = 0
assert sha(API / "SHA256SUMS") == "b3e6174797f8d466181ebf42e87b5c45133341ff7351265d166d0cfc1dfc8a1b"
for line in (API / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert sha(API / name) == expected
    api_members += 1

identities = [
    {**freeze["preparation_identity"], "owner": "preparation"},
    {**read(ATTEMPT / "runner-identity.json"), "owner": "bounded test supervisor"},
    {**result["original_identity"], "owner": "unittest process"},
]
identities = [{**item, "fresh_original_identity_gone": gone(item)} for item in identities]
assert all(item["fresh_original_identity_gone"] for item in identities)
cleanups = sorted(ARTIFACTS.glob("*/OWNED_CLEANUP.json"))
assert len(cleanups) == 4
cleanup_rows = []
for path in cleanups:
    value = read(path)
    assert all(value["clients_closed"]) and all(value["transports_closed"])
    assert all(value["test_owned_tasks_joined"])
    assert value["remaining_loop_tasks"] == []
    assert all(count == 0 for count in value["active_CPU_stream_ledgers"])
    assert value["processes_or_services_started"] == []
    assert value["signal_handlers_unchanged"] is False
    cleanup_rows.append({"case": path.parent.name, "raw_receipt": str(path.relative_to(SCOPE)), **value})
save("FRESH_OWNED_GONE.json", {
    "identities": identities, "cases": cleanup_rows,
    "known_helpers_gone": True, "clients_closed": sum(len(row["clients_closed"]) for row in cleanup_rows),
    "transports_closed": sum(len(row["transports_closed"]) for row in cleanup_rows),
    "test_tasks_joined": sum(len(row["test_owned_tasks_joined"]) for row in cleanup_rows),
    "signal_boolean_not_rewritten": "All four raw false values are retained. The test compares different unittest Runner contexts; it does not demonstrate adapter signal ownership.",
    "scope": "Known CPU test process/clients/transports/outer task only; no episode worker, service or Native cleanup proof.",
})
episodes = sorted(ARTIFACTS.rglob("episode.json"))
assert len(episodes) == 12
episode_rows = []
for path in episodes:
    value = read(path)
    capture = read(path.parent / "capture.json")
    assert value["status"] == "refused" and value["workers"] == {}
    assert value["all_workers_joined"] and value["persistence_errors"] == []
    assert capture["records"] is None
    assert not (path.parent / "runtime-journal.json").exists()
    assert sha(path.parent / "capture.json") == value["capture_sha256"]
    episode_rows.append({
        "path": str(path.relative_to(SCOPE)), "sha256": sha(path),
        "reason": value["failures"][0]["error"]["message"], "workers": value["workers"],
        "capture_status": capture["status"], "capture_error": capture["error"],
        "offered_count": capture["offered_count"], "missing_record_ids": value["missing_record_ids"],
        "capture_sha256": value["capture_sha256"],
    })
assert all("runtime_Train_provider_or_policy_changed" in row["reason"] for row in episode_rows
           if any(part in row["path"] for part in (
               "test_shared_start_full_outcomes", "test_caller_cancellation",
               "test_returned_failure_envelopes")))
save("OBSERVED_FAILURE_BOUNDARY.json", {
    "episodes": episode_rows, "count": len(episode_rows),
    "all_refused_before_workers": True, "runtime_journals_created": 0,
    "client_request_records_allocated": 0,
    "meaning": "Local preparation/refusal persistence ran; intended concurrent runtime/replay paths never started.",
})
save("FAILURES.json", {
    "result": "Four methods, no overall PASS; unittest reports failures=8, errors=1. Counts include subtests and teardown assertions.",
    "source_generations": 1, "attempts": 1, "retries": 0, "source_edits_after_freeze": False,
    "raw_stdout": str((ATTEMPT / "stdout.log").relative_to(SCOPE)),
    "raw_stderr": str((ATTEMPT / "stderr.log").relative_to(SCOPE)),
    "implementation_defect": {
        "id": "E02-1",
        "owner": "new nominal_episode.py, not accepted runtime03 or client API03",
        "observed": "runtime_Train_provider_or_policy_changed before creating workers",
        "source": "nominal_episode.py:182-184 compares runtime.protocol['scenario_binding'] directly with prepared.scenario_binding.",
        "cause": "runtime.load_train_scenario uses asdict(TrainPrior), retaining tuple-containing prior content (including marks). runtime._protocol canonicalizes through JSON, turning tuples into arrays/lists. Raw Python container equality therefore rejects the unchanged Train binding.",
        "source_evidence": [
            "nominal_scenario.py:70-74 frozen TrainPrior.marks is tuple[tuple[int,int,int], ...]",
            "nominal_runtime.py:714-717 preserves asdict(prior) content in the raw factory binding",
            "nominal_runtime.py:176 returns _strict(json.dumps(value, allow_nan=False))",
            "All intended execution-case receipts show the same pre-worker refusal and empty workers.",
        ],
        "minimum_correction_not_applied": "Compare canonical binding digests (runtime_api.digest on both values), retaining policy equality and exact provider identity. Do not remove the binding check.",
    },
    "harness_defect": {
        "id": "E02-2",
        "owner": "new test_nominal_episode.py",
        "observed": "Every asyncTearDown signal assertion compares default_int_handler against functools.partial(Runner._on_sigint, main_task=...).",
        "cause": "Baseline was sampled in synchronous setUp outside asyncio.Runner, then compared during a different running async context. This is a framework-handler comparison, not evidence that the adapter installed a signal handler.",
        "minimum_correction_not_applied": "Sample and compare handlers around the awaited operation within the same async test coroutine/Runner invocation. Moving only to asyncSetUp while comparing in asyncTearDown is not a sufficient invariant.",
        "raw_false_cleanup_values": "Preserved unchanged; not converted to true or suppressed.",
    },
    "derived_error": {
        "kind": "cancellation-control TimeoutError waiting three seconds for a burst that could never start",
        "cause": "E02-1 returned a refusal before recorder/controller workers; no burst was sent.",
        "not_the_runner_timeout": "The whole test process exited normally with code1 after 3.421856s, below its original60s bound.",
    },
    "interpretation": "This is an actual failed CPU integration attempt, unlike episode01's zero-attempt source API no-go. It is not a Native failed experiment.",
    "limits": "Known blockers only. Shared-start replay, runtime planning/application, failure propagation and cancellation joins remain unvalidated on these bytes.",
})
save("INTERFACE.json", {
    "status": "FAILED_CANDIDATE_DO_NOT_ACTIVATE",
    "module": "experiments/mpc-research/resume-mvp/nominal_episode.py",
    "prepare": "prepare_episode(*, output_dir, binding_path, gateway_url, backend_url, policy='mpc', trace_path=PROBE, train_path=TRAIN, request_timeout=120.0, tokenizer_loader=nominal_client.load_tokenizer) -> PreparedEpisode",
    "prepare_design": "Synchronous local peer prepare_capture plus accepted runtime Train-provider preparation; persist prepared-capture.json via peer codec and preparation.json. Choose the shared future start only afterward.",
    "handle": "PreparedEpisode.directory, recorder, policy, entered_at, prepared_at, sources, scenario_factory, scenario_binding, error, input_sha256, used",
    "construct_runtime_design": "Use the returned scenario_factory/scenario_binding, exact expected model parameters and caller-owned HTTP client. Bind runtime.start_at to the same future S, enabled=True and a separate valid scoped NominalDemoPermit; no permit is created here.",
    "run": "await run_episode(prepared, runtime, *, owner, offered_start, enabled=False, transport=None)",
    "owner_interface": "Existing published NominalServices instance: owner.deadline, owner.reserve (constructor cleanup_reserve) and owner.check(). Context remains synchronous and caller-owned; no smoke-driver readiness asyncio.run is used.",
    "scope_design": "Default-OFF opt-in; no subprocess/thread/service/signal ownership. Original D and R go unchanged to capture_replay. A read-only records/preflight barrier precedes runtime wait_for_start and run.",
    "persistence_design": "Final capture.json through unchanged write_capture, episode.json with both returned results/errors, and the unchanged runtime journal when runtime actually starts. Reraise caller cancellation after persistence.",
    "current_verified_scope": "Exact fixed five-request source, local CPU fixture preparation and pre-worker refusal capture persistence; no successful concurrent episode.",
    "current_blocker": "Canonical Train-binding comparison bug plus invalid cross-Runner signal assertion. Neither corrected within this one frozen attempt.",
    "not_implemented": "No CLI, campaign, service launch, observed-Train Fixed/Eval selection, natural metrics finalize or full-wall completion.",
})
save("LIMITATIONS.json", {
    "activation": "FAILED candidate; do not activate or describe as a working controller. Parent must authorize a fresh bounded corrective generation before retesting.",
    "validation": [
        "One local Python3.12.16 run, four methods, failures=8/errors=1, no overall PASS. No Python3.14/actual-host run and no retry.",
        "All42 input pins remained unchanged before/after controls and at sealing; six additional peer inputs total200013bytes (two hash-only), within6/384KiB.",
        "Client API03 still has its own author43PASS evidence, but no success from that peer test suite is transferred to this adapter.",
        "No controller/replay workers started. Cleanup receipts establish only no leftover test tasks/clients and gone known test processes.",
        "Signal test baseline is invalid across Runner contexts; false raw fields remain visible.",
    ],
    "fixtures": [
        "Original five-request admission-boundary probe unchanged (all five Train3072 prompts at1.25s over5s). No old four-request substitution, retiming or resampling.",
        "Tokenizer interface/fingerprint hook and in-memory CPU permit/source identity are literal controls, not model/tokenizer/live authority validation.",
        "No external HTTP, actual CAS/server inference, service/process launch by the adapter, GPU, SSH, Git, installs or additional agents.",
    ],
    "dependency_gates": [
        "Runtime03 remains accepted at Mayor reference5e9696eca54133ee440314b681f11764a0937744 and unchanged.",
        "Client API03/metrics exact sources933e4161/49c76b8b are candidate bytes; parent provenance6113b377846f59f6c519a5b2fc4ba369e7f61fce does not replace independent Role4 acceptance.",
        "NominalServices source5249221b unchanged; native_services_smoke.py was neither read nor used.",
    ],
    "later_host_obligations": [
        "Mayor exact-source approval, independent dependency acceptance, real model/alias/tokenizer/no-fallback/cost/source/clock checks, and existing-resource authorization remain required.",
        "Use actual same-host future monotonic START with original NominalServices deadline/reserve. Preserve pending/runtime guards; no retiming or fallback to obtain a result.",
        "B needs at least two consecutive real observe/predict/plan/actual-first-action-apply/reobserve cycles and actual admission effect/ordered gateway ledger evidence. ACKs and CPU examples are not enough.",
        "Every actual offered fate/SSE finish reason/unknown must remain in raw capture; engineering is not held-out Eval or Train selection. Do not natural-finalize it.",
        "Full wall and root/leaf shutdown/lock/poison receipts end only after Mayor-owned cleanup. Adapter fields remain missing/null, not invented zeros.",
        "Async joins and blocking filesystem operations are cooperative; the current unexecuted join path is not a universal hard-deadline guarantee.",
    ],
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
    "prepare_argv": wrapper + [str(SCOPE / "prepare02.py")],
    "runner_argv": wrapper + [str(SCOPE / "run02.py")],
    "finisher_argv": wrapper + [str(SCOPE / "finish02.py")],
    "source_diagnosis": "Read-only frozen source/tracebacks/receipts, not another diagnostic control.",
    "control_attempts": 1, "additional_runs_after_failure": 0,
})
save("REPORT.json", {
    "schema": "nominal-episode02-failed-report-v1",
    "status": "FAILED_OWN_BINDING_AND_HARNESS_NO_ACTIVATION",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "model": {"id": "gpt-6-astra", "reasoning_effort": "max", "provenance": "retained verified Role6 assignment"},
    "source_ids": freeze["source_ids"], "source_freeze_sha256": sha(SCOPE / "FREEZE.json"),
    "source_diff": "SOURCE_DIFF.patch", "source_generations": 1, "test_attempts": 1,
    "retries": 0, "post_freeze_source_edits": False,
    "test_result": "4 methods; FAILED(failures=8, errors=1), including teardown/subtests; no overall PASS",
    "test_process": result, "selected_controls": freeze["selected_controls"],
    "input_pins_preserved": len(pins), "peer_source_count": 6, "peer_source_bytes": 200013,
    "API03_members_reverified": api_members, "preserved_no_start_episode_receipts": len(episode_rows),
    "controller_or_recorder_workers_started": 0, "runtime_journals_created": 0,
    "clients_closed": sum(len(row["clients_closed"]) for row in cleanup_rows),
    "transports_closed": sum(len(row["transports_closed"]) for row in cleanup_rows),
    "test_tasks_joined": sum(len(row["test_owned_tasks_joined"]) for row in cleanup_rows),
    "known_process_identities_gone": len(identities),
    "remaining_blockers": ["E02-1 canonical Train binding equality", "E02-2 signal baseline inside same Runner"],
    "peers_edited": False, "Mayor_accepted": False, "Native_permission": False, "Native_failed_experiment": False,
    "qualification": "No working composed controller or Native/B qualification demonstrated.",
    "seal_exclusions": ["SHA256SUMS", "SEAL_RECEIPT.json"],
})
save("HANDOFF.json", {
    "status": "FAILED_AND_IDLE", "report": "REPORT.json", "failures": "FAILURES.json",
    "API": "INTERFACE.json", "limits": "LIMITATIONS.json", "cleanup": "FRESH_OWNED_GONE.json",
    "source": "FREEZE.json", "diff": "SOURCE_DIFF.patch", "commands": "COMMANDS.json",
    "observed_boundary": "OBSERVED_FAILURE_BOUNDARY.json",
    "minimum_next_change": "Canonical binding digest comparison and same-coroutine signal invariant, then newly authorized bounded validation. No peer changes or workload/model changes.",
    "manifest": "SHA256SUMS", "seal_receipt": "SEAL_RECEIPT.json",
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
    "status": "FAILED_CANDIDATE_NO_NATIVE_PERMISSION", "receipt_not_self_in_manifest": True,
})
print(json.dumps({
    "seal": seal, "members": len(files), "status": "4 methods; failures=8/errors=1; no retry",
    "known_identities_gone": len(identities), "clients_closed": sum(len(row["clients_closed"]) for row in cleanup_rows),
    "transports_closed": sum(len(row["transports_closed"]) for row in cleanup_rows),
    "source_generations": 1, "peer_source_inputs": 6, "peer_source_bytes": 200013,
}, sort_keys=True))
