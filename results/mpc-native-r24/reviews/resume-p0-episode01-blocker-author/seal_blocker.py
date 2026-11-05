"""Freeze static API evidence and the authorized no-go; no client/model/test execution."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
SCOPE = Path(".runtime/r24-resume-p0-nominal-episode01")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
INDEX = Path("results/mpc-native-r24/reviews/resume-p0-metrics02-parent01.json")
PROBE = Path("results/mpc-native-r24/reviews/resume-p0-probes02/inputs/admission-boundary.json")
PEERS = {
    PUBLIC / "nominal_client.py": "f369e5ce01a54dcfa9086fe8ee4a1528d7e52d19179db78201e06961da4c7bda",
    PUBLIC / "nominal_metrics.py": "49c76b8bcda2560ed9c24770a41c3fa876082cd0e19d171daf025300452a60ec",
}
RETAINED = {
    PUBLIC / "nominal_runtime.py": "2d7a01dbd335297439d31ec00d144db4d4a771b190d8b47a03e1ddd3a5f80959",
    Path(".runtime/r24-resume-p0-nominal-runtime03/SHA256SUMS"):
        "b116dbaa005a27a14fec3ca1cb553cd3b01d3e389ad3adebde5aecb185b9dc1b",
}
REPORTS = [
    INDEX, Path(".runtime/r24-resume-p0-nominal-metrics02/report.json"),
    Path(".runtime/r24-resume-p0-nominal-client01/report.json"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (SCOPE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


assert ROOT == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
assert not (SCOPE / "SHA256SUMS").exists()
assert not (PUBLIC / "nominal_episode.py").exists()
assert not (PUBLIC / "test_nominal_episode.py").exists()
index = json.loads(INDEX.read_bytes())
for path, expected in {**PEERS, **RETAINED}.items():
    assert sha(path) == expected, str(path)
for path, expected in PEERS.items():
    assert index["sources"][path.name] == expected
peer_bytes = sum(path.stat().st_size for path in PEERS)
assert len(PEERS) <= 6 and peer_bytes <= 384 * 1024
assert sha(PROBE) == "6d938ba18240f3e0f8cb677064849a73316ee7a8dc918526c68cba37f02da6be"
probe = json.loads(PROBE.read_bytes())
assert len(probe["requests"]) == 5

client_tree = ast.parse((PUBLIC / "nominal_client.py").read_text())
metrics_tree = ast.parse((PUBLIC / "nominal_metrics.py").read_text())
functions = {node.name: node for node in client_tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
public_api = {
    name: {"async": isinstance(node, ast.AsyncFunctionDef),
           "arguments": ast.unparse(node.args), "lines": [node.lineno, node.end_lineno]}
    for name, node in functions.items() if not name.startswith("_")
}
main_calls = [ast.unparse(node.func) for node in ast.walk(functions["_replay_main"])
              if isinstance(node, ast.Call)]
assert "asyncio.run" in main_calls and "_run_cancellable" in main_calls
assert "_capture" in main_calls and "_write_new" in main_calls
assert "normalized_row" in main_calls and "summarize_records" in main_calls
capture_initializers = []
for name, node in functions.items():
    for statement in ast.walk(node):
        if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Dict)
                and any(isinstance(target, ast.Name) and target.id == "capture" for target in statement.targets)):
            capture_initializers.append({"owner": name, "line": statement.lineno})
assert len(capture_initializers) == 1 and capture_initializers[0]["owner"] == "_replay_main"
signal_calls = [ast.unparse(node.func) for node in ast.walk(functions["_run_cancellable"])
                if isinstance(node, ast.Call)]
assert "loop.add_signal_handler" in signal_calls and "loop.remove_signal_handler" in signal_calls
purposes = next(ast.literal_eval(node.value) for node in metrics_tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PURPOSES" for target in node.targets))
assert purposes == ("train", "eval")

input_paths = sorted(set(list(PEERS) + list(RETAINED) + REPORTS + [
    PROBE, SCOPE / "PLAN.json", SCOPE / "seal_blocker.py",
]))
pins = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in input_paths}
for path in list(PEERS) + [PUBLIC / "nominal_runtime.py"]:
    with (SCOPE / "sources" / path.name).open("xb") as stream:
        stream.write(path.read_bytes())
(SCOPE / "metadata").mkdir()
for path, name in zip(REPORTS, ("parent-index.json", "metrics02-report.json", "superseded-client01-report.json")):
    with (SCOPE / "metadata" / name).open("xb") as stream:
        stream.write(path.read_bytes())
save("INPUT_PINS.json", pins)
save("SOURCE_READ_BUDGET.json", {
    "additional_peer_sources": {str(path): pins[str(path)] for path in PEERS},
    "used_source_files": len(PEERS), "used_source_bytes": peer_bytes,
    "maximum_source_files": 6, "maximum_source_bytes": 384 * 1024,
    "counting": "Full source-file bytes charged even when only API ranges were viewed.",
    "retained_runtime_context": str(PUBLIC / "nominal_runtime.py"),
    "unsealed_Role3_source_read": False, "services_API_not_reached": True,
    "other_source_content_reads": [],
})
save("API_EVIDENCE.json", {
    "source": str(PUBLIC / "nominal_client.py"), "source_sha256": PEERS[PUBLIC / "nominal_client.py"],
    "public_API": public_api,
    "raw_capture_initializer": capture_initializers,
    "CLI_capture_path": {
        "entry": "main -> _replay_main -> asyncio.run(_run_cancellable(_capture(...)))",
        "raw_construction_lines": [690, 710], "input_population_lines": [713, 731],
        "nested_loop_lines": [732, 733], "outcome_mapping_persistence_lines": [734, 747],
        "signal_ownership_lines": [750, 767],
        "observed_calls": sorted(set(main_calls)),
    },
    "usable_library_parts": {
        "binding_preflight": "check_binding / check_tokenizer / check_served_models are reusable.",
        "per_request": "prepare_records creates all-offered placeholders; await replay fills them in place and cancels/gathers its request tasks.",
        "mapping": "normalized_row / summarize_records are reusable; there is no need to duplicate SSE parsing or row mapping.",
    },
    "missing_part": "No public raw-v2 capture preparation/build/finalization or awaitable high-level capture API. The only complete raw-v2 envelope and status/error/persistence path is owned by _replay_main.",
    "private_capture": "_capture is async but requires the caller to supply the client-owned capture/config envelope and leaves outer status/error/rows/summary/fsync to _replay_main.",
    "build_run_not_a_substitute": {
        "lines": [542, 551], "accepted_purposes": purposes,
        "engineering": "Explicit ClientRefusal; build_run constructs natural metrics input, not a raw-v2 engineering capture.",
    },
    "evidence_kind": "Static frozen-source inspection/AST verification, not an executed failing experiment.",
})
save("PROBE_REFERENCE.json", {
    "path": str(PROBE), "sha256": sha(PROBE), "offered_requests": len(probe["requests"]),
    "duration_seconds": probe.get("duration_seconds"), "model": probe.get("model"),
    "provenance": probe.get("provenance"),
    "request_shapes": [{key: request.get(key) for key in
                        ("trace_id", "arrival_seconds", "input_tokens", "max_tokens")}
                       for request in probe["requests"]],
    "execution": "Not executed, retimed, regenerated or substituted with the old four-request probe.",
})
save("BLOCKER.json", {
    "status": "NO_GO_SAFE_RAW_V2_COMPOSITION_API_MISSING",
    "smallest_blocker": "The complete raw-v2 engineering recorder exists only behind synchronous CLI _replay_main. Calling it from the required running async episode reaches asyncio.run; calling only public replay does not provide the required raw-v2 capture envelope/persistence.",
    "why_not_implemented": [
        "Copying the capture template and completion/error writer into nominal_episode.py would duplicate the forbidden capture codec.",
        "Calling private _capture still needs that copied initializer/finalizer; it is not a complete raw-v2 composition entry.",
        "A process/thread detour would violate the in-process async ownership/cancellation objective and is not authorized.",
        "The CLI path also owns signal handlers; borrowing it would interfere with Mayor lifecycle ownership.",
        "build_run/finalize intentionally refuse engineering and must not be used to fake a completed full wall or natural metrics.",
    ],
    "smallest_owner_API_requirement": [
        "Expose the existing raw-capture prepare/build/finalize implementation as public library helpers, or an awaitable high-level raw-capture entry.",
        "Reuse existing replay/SSE/row mapping; no new recorder implementation is needed.",
        "Callable inside an existing event loop without asyncio.run or installing/removing owner signal handlers.",
        "Require the caller's exact future offeredStart, permit original absolute deadline/reserve propagation, and never retime.",
        "Retain the complete/partial raw-v2 capture and every allocated offered record on exception or cancellation, with joined request tasks and caller-visible failure evidence.",
    ],
    "proposed_names_or_signature": "Not invented here; the owning Role7/Mayor must freeze an additive API before this adapter depends on it.",
    "coordination": "Return to Mayor for the smallest peer-owned API extraction after appropriate independent review; no peer edit or broadcast was performed.",
})
save("LIMITATIONS.json", {
    "implemented_adapter": False, "new_public_files": [], "CPU_control_attempts": 0,
    "reason_for_zero_controls": "Explicit go/no-go instruction applies before implementation: no safe nonduplicating raw-v2 composition API.",
    "not_tested": [
        "No concurrent runtime/recorder episode, shared-start composition, cancellation composition, or real-host behavior was executed.",
        "No tokenizer loaded, model binding preflight executed, actual service GET/POST, new permit or live source approval.",
    ],
    "acceptance": {
        "runtime03": "Accepted by Mayor at supplied reference 5e9696eca54133ee440314b681f11764a0937744; exact retained source verified.",
        "client_metrics": "Frozen candidate at supplied reference 7ca6338dee89f2d0749138a215a80a0601d2d4c8; Role4 independent acceptance pending.",
        "episode": "Not implemented/qualified; Mayor exact-source approval does not exist.",
    },
    "actual_host_obligations_after_API_and_adapter": [
        "Use sealed NominalServices inside Mayor ownership; preserve its original absolute deadline/cleanup reserve and actual root/leaf shutdown/lock/poison receipts.",
        "Use one actual host monotonic clock and future offeredStart; freeze/check Train, tokenizer, binding and source identities before start; refuse past start.",
        "Keep all offered fates/raw-v2 capture and runtime journal; missing gateway ledger/admission evidence stays null/missing, never zero or inferred from ACK.",
        "At least two consecutive real ordered observe/predict/plan/actual-first-action-apply/reobserve cycles and actual admission behavior changes remain required for B.",
        "Engineering is not Train selection or held-out Eval; never natural-finalize this probe. Only Mayor may finish full wall after actual cleanup.",
    ],
    "prohibitions_respected": "No peer/product/SDK edits, unsealed smoke-source reads, agents, subprocess/service workers, network, SSH, GPU, Git, installs, Markdown, full-suite testing, modeling or sampling.",
})
save("FAILURES.json", {
    "executed_failures": [], "control_attempts": 0, "source_generations_tested": 0, "retries": 0,
    "blocking_source_fact": "BLOCKER.json and API_EVIDENCE.json",
    "not_an_experiment_failure": "No failing runtime or integration experiment was run. This is the explicitly authorized source-level no-go, not a hidden failed test or a PASS.",
})
save("COMMANDS.json", {
    "cwd": str(ROOT),
    "inspection": [
        "sha256sum accepted runtime/source seal; verify absence of the two proposed new public files",
        "sha256sum and wc -c nominal_client.py nominal_metrics.py; inspect only named frozen source APIs",
        "find constrained paths for report/interface filenames only; never read native_services_smoke.py",
        "sha256sum frozen admission-boundary.json; verify supplied five-request reference",
    ],
    "seal_argv": [
        "env", "-u", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "PYTHONHASHSEED=0", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1", "nice", "-n19",
        sys.executable, "-B", str(SCOPE / "seal_blocker.py"),
    ],
    "test_commands": [], "HTTP_commands": [], "Git_commands": [],
})
save("FREEZE.json", {
    "schema": "nominal-episode01-static-no-go-freeze-v1",
    "frozen_at": datetime.now(timezone.utc).isoformat(),
    "plan_sha256": sha(SCOPE / "PLAN.json"), "input_pins_sha256": sha(SCOPE / "INPUT_PINS.json"),
    "source_read_budget_sha256": sha(SCOPE / "SOURCE_READ_BUDGET.json"),
    "public_code_changes": 0, "source_generations_tested": 0, "control_attempts": 0,
    "interpreter_for_static_evidence_only": sys.version,
})
checks = {path: sha(Path(path)) == item["sha256"] for path, item in pins.items()}
assert all(checks.values())
save("SOURCE_POSTCHECK.json", checks)
save("CLEANUP.json", {
    "services_started": [], "HTTP_clients_created": [], "async_workers_created": [], "helpers_spawned": [],
    "all_owned_runtime_work_joined": True,
    "scope": "Vacuous runtime cleanup: only completed inspection commands and this finite static sealer; no episode was started.",
    "seal_writer_pid": os.getpid(), "not_universal_service_cleanup": True,
})
save("REPORT.json", {
    "schema": "nominal-episode01-no-go-report-v1",
    "status": "BLOCKED_SAFE_RAW_V2_COMPOSITION_API_MISSING",
    "model": {"id": "gpt-6-astra", "reasoning_effort": "max", "provenance": "retained verified Role6 assignment"},
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "accepted_runtime_sha256": RETAINED[PUBLIC / "nominal_runtime.py"],
    "peer_sources": {str(path): pins[str(path)] for path in PEERS},
    "additional_source_inputs": len(PEERS), "additional_source_bytes": peer_bytes,
    "public_files_created_or_edited": [], "control_attempts": 0, "retries": 0,
    "decision": "Honor explicit stop condition rather than duplicate capture codec or invent an unsafe async/ownership bridge.",
    "blocker": "BLOCKER.json", "actual_peer_APIs_and_lines": "API_EVIDENCE.json",
    "source_pins": "INPUT_PINS.json", "read_budget": "SOURCE_READ_BUDGET.json",
    "probe": "PROBE_REFERENCE.json", "limits": "LIMITATIONS.json", "cleanup": "CLEANUP.json",
    "all_input_hashes_unchanged": True, "Native_permission": False, "qualification": False,
    "seal_exclusions": ["SHA256SUMS", "SEAL_RECEIPT.json"],
})
save("HANDOFF.json", {
    "status": "BLOCKED_AND_IDLE", "report": "REPORT.json", "blocker": "BLOCKER.json",
    "API": "API_EVIDENCE.json", "source_budget": "SOURCE_READ_BUDGET.json",
    "next_owner_action": "Mayor arranges the smallest peer-owned raw-v2 awaitable/builder API and its frozen handoff; do not activate or infer an episode adapter exists.",
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
    "all_source_bindings_unchanged": True, "status": "SOURCE_API_NO_GO_NOT_A_TEST_PASS",
    "receipt_not_self_in_manifest": True,
})
print(json.dumps({
    "seal": seal, "members": len(files), "status": "BLOCKED: raw-v2 public composition API missing",
    "peer_source_inputs": len(PEERS), "peer_source_bytes": peer_bytes,
    "CPU_control_attempts": 0, "public_code_changes": 0, "helpers_spawned": 0,
}, sort_keys=True))
