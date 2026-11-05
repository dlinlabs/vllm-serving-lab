"""Freeze the narrow, byte-exact source/plan inputs before CPU attempts."""

import hashlib
import json
from pathlib import Path
import tarfile
import time
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE13 = "experiments/mpc-research/native-r24-prepared-physical13"
ARCHIVE = ROOT / "results/mpc-native-r23/runtime/r24-native-prepared-physical-source13-prospective02.tar.gz"
ARCHIVE_SHA256 = "6f99227cd5f1b266f41015c2f2742e2c44c3f171538aa3d1df8ad1cf7964a509"
SOURCES = (
    "source-closure.json", "pyproject.toml", "research_source_guard.py",
    "phase_observer.py", "input_post_observer.py", "queue_observer.py",
    "process_clock_provenance.py", "clocked_owner_cut.py", "capture_owner_closure.py",
    "live_owner_prefix.py", "flushed_owner_prefix.py", "source_owner_phases.py",
    "gateway_prefix_bridge.py", "launch_gateway_prefix_service.py", "launch_queue_service.py",
    "collect_owner_bootstrap.py", "engine_handoff.py", "run_h16.py",
    "prepared_copy_ready_plugin.py", "prepared_copy_ready_hook.py", "copy_ready_intervals.py",
    "prepared_forward_plugin.py", "prepared_forward_capture.py",
    "prepared_forward_hook.py", "prepared_forward_region.py",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    if path.is_absolute():
        raise ValueError("relative_output_required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def verify(*, originals=False):
    if Path.cwd().resolve() != HERE:
        raise ValueError("exclusive_candidate_cwd_required")
    pins = load("INPUTS.json")
    for row in pins["bindings"]:
        if sha(Path(row["copy"]).read_bytes()) != row["sha256"]:
            raise ValueError("copied_input_changed:" + row["copy"])
        if originals and "::" not in row["source"]:
            if sha(Path(row["source"]).read_bytes()) != row["sha256"]:
                raise ValueError("original_input_changed:" + row["copy"])
    if originals and sha(ARCHIVE.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("source13_archive_changed")
    frozen = load("PRE_SOURCE_FREEZE.json")
    for name, fingerprint in frozen["frozen_sha256"].items():
        if sha(Path(name).read_bytes()) != fingerprint:
            raise ValueError("pretest_frozen_document_changed:" + name)
    return dict(bindings=len(pins["bindings"]), originals=originals,
                input_sha256=sha(Path("INPUTS.json").read_bytes()),
                protocol_sha256=sha(Path("PLAN.json").read_bytes()))


def main():
    if Path.cwd().resolve() != HERE or Path("INPUTS.json").exists():
        raise ValueError("new_exclusive_freeze_required")
    if sha(ARCHIVE.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("source13_archive_changed")
    rows = []

    def retain(raw, destination, origin):
        destination = Path(destination)
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError("relative_copy_required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != raw:
                raise ValueError("partial_freeze_copy_changed")
        else:
            with destination.open("xb") as handle:
                handle.write(raw)
        rows.append(dict(copy=str(destination), source=str(origin),
                         sha256=sha(raw), bytes=len(raw)))

    def copy(source, destination):
        retain(Path(source).read_bytes(), destination, source)

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        closure = json.loads(archive.extractfile("source-closure.json").read())
        for name in SOURCES:
            member = archive.getmember(name)
            if not member.isfile() or member.size > 512000:
                raise ValueError("unexpected_source_member")
            raw = archive.extractfile(member).read()
            if name != "source-closure.json" and sha(raw) != closure["source_sha256"][name]:
                raise ValueError("source13_member_binding")
            retain(raw, f"inputs/source-tree/{SOURCE13}/{name}", str(ARCHIVE) + "::" + name)

    product = [
        "gateway.py", "telemetry.py", "experiments/prefill-vllm-v030-source-map.json",
        "experiments/mpc-research/native_observer/pr33_native_observer.py",
        "experiments/mpc-research/native-v2/batch_capture_bridge.py",
        "experiments/mpc-research/native-v2/launch_batch_service.py",
        "prefill_lab/__init__.py", "prefill_lab/contracts.py", "prefill_lab/runtime.py",
        "prefill_lab/plugin.py", "prefill_lab/live_http.py", "mpc_local/__init__.py",
        "mpc_local/contracts.py",
        "tests/test_r24_capture_owner_closure.py",
        "tests/test_r24_process_clock_provenance.py",
    ]
    product += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "mpc_lab").glob("*.py"))]
    for name in product:
        copy(ROOT / name, "inputs/source-tree/" + name)
    control = ROOT / ".runtime/r24-team16-native-control-probe02"
    for name in ("native_probe02.py", "prelaunch01.py", "freeze_inputs02.py",
                 "INPUTS.json", "EXPERIMENT.json", "AMENDMENT02.json", "SHA256SUMS"):
        copy(control / name, "inputs/control02/" + name)
    launch = ROOT / ".runtime/r24-team16-native-control-launch02"
    for name in ("launch_native.py", "launch_common.py", "AMENDMENT02.json", "REPORT.json",
                 "startup-instrumentation.json", "SHA256SUMS", "inputs/clock02/safe_exec.py"):
        copy(launch / name, "inputs/control-launch02/" + name)
    runtime = ROOT / ".runtime/r24-team16-runtime-integration01"
    for name in ("REPORT.json", "LIMITATIONS.json", "SHA256SUMS"):
        copy(runtime / name, "inputs/runtime16/" + name)
    bridge = ROOT / ".runtime/r24-team16-runtime-native-evidence-bridge01"
    for name in ("REPORT.json", "LIMITATIONS.json", "CONTRACT_FINDINGS.json", "SHA256SUMS"):
        copy(bridge / name, "inputs/bridge01/" + name)
    review = ROOT / "results/mpc-native-r24/reviews"
    copy(review / "native-team16-control-native-result02.json", "inputs/native02/result02.json")
    raw = review / "native-team16-control-native-raw02"
    copy(raw / "r24-team16-native-control-launch02/runtime-identity.json",
         "inputs/native02/runtime-identity.json")
    for name in ("035", "036", "037"):
        copy(raw / f"native-team16-control-probe02-launch02/http/{name}.json",
             f"inputs/native02/http/{name}.json")
    copy(review / "native-team16-verified-model-request01.txt",
         "inputs/reviews/native-team16-verified-model-request01.txt")

    original = load("inputs/control-launch02/startup-instrumentation.json")
    offers = {
        "schema": "team16-owner-acquisition-prospective-offers-v1",
        "status": "PROPOSED_NOT_EXECUTED", "fixed_budget": 32,
        "rows": [
            dict(ordinal=i, request_id=uuid.uuid5(uuid.NAMESPACE_URL, HERE.name + f"/offer{i}").hex,
                 prompt="Answer only with the word ready.", max_tokens=16, fate="not_offered")
            for i in (1, 2)
        ],
        "separate_startup_warmup": dict(offers=1, max_tokens=16, fate="not_offered"),
    }
    write("PROPOSED_OFFERS.json", offers)
    hashes = closure["source_sha256"]
    variant = {
        "schema": "native-v2-batch-capture-plan-v1", "partition": "train",
        "source_commit": original["source_commit"],
        "experiment_id": HERE.name,
        "trace_sha256": sha(Path("PROPOSED_OFFERS.json").read_bytes()),
        "scope": "Proposed fixed32 owner-acquisition component; no model, forecast or comparison.",
        "trials": [dict(trial_id="nv2-team16-owner-acquisition01-fixed32", observer="light",
                        seed=170003, budget=32, budget_steps=[],
                        request_ids=[r["request_id"] for r in offers["rows"]])],
        "queue_observer_extension": original["queue_observer_extension"],
        "prepared_forward_extension": original["prepared_forward_extension"],
        "input_post_extension": dict(source_sha256=hashes["input_post_observer.py"],
                                     vllm_core_sha256="268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5"),
        "phase_frontier_extension": dict(schema="r24-native-live-phase-v1",
                                         source_sha256=hashes["phase_observer.py"]),
        "gateway_prefix_extension": dict(schema="r24-gateway-flushed-prefix-v1",
                                         source_sha256=hashes["gateway_prefix_bridge.py"],
                                         launcher_sha256=hashes["launch_gateway_prefix_service.py"]),
        "owner_bootstrap_extension": dict(scope="InitialCaptureClosure, unchanged",
                                          native_mpc_permitted=False),
        "native_execution_authorized": False, "native_mpc_permitted": False,
    }
    write("PROPOSED_NATIVE_PLAN.json", variant)
    write("INPUTS.json", dict(
        schema="team16-owner-acquisition-input-pins-v1", bindings=rows,
        archive=str(ARCHIVE), archive_sha256=ARCHIVE_SHA256,
        source13_directory=SOURCE13,
        narrow_copies_only=True, tests_started=False, native_execution=False))
    write("DISCOVERY_FAILURES.json", [
        dict(stage="read_only_inspection", reason="Remote Source13 path was initially assumed present locally; it is absent. Located the exact local prospective02 archive instead."),
        dict(stage="read_only_inspection", reason="pyproject.toml absent from narrowed launch02 and runtime16 copies; recovered the exact Source13 manifest from its archive."),
        dict(stage="read_only_inspection", reason="Clock02 INPUTS.json filename absent; used source-pins.json after listing names."),
        dict(stage="read_only_inspection", reason="Launch01 input binding used copy, not copied; KeyError retained as inspection setup failure."),
        dict(stage="read_only_inspection", reason="prefill_lab/runtime.py requested range300 exceeded its217 lines; subsequently read all217."),
        dict(stage="read_only_inspection", reason="Speculative test_mpc_live_gate.py filename absent; actual live-owner fixture already supplies source AST extraction."),
        dict(stage="freeze-py314-01", reason="Launch02 PLAN.json does not exist. Source/plan freeze stopped before tests; retained code/log/all partial byte-exact copies. The actual AMENDMENT02.json replaces this erroneous metadata filename. Resume checks existing copies without overwriting.")
    ])
    names = ("PLAN.json", "PROPOSED_OFFERS.json", "PROPOSED_NATIVE_PLAN.json",
             "INPUTS.json", "DISCOVERY_FAILURES.json", "freeze_inputs01.py")
    write("PRE_SOURCE_FREEZE.json", dict(
        schema="team16-owner-acquisition-pre-source-freeze-v1",
        frozen_at_unix=time.time(), tests_started=False, native_execution=False,
        frozen_sha256={name: sha(Path(name).read_bytes()) for name in names}))
    print(json.dumps(verify(originals=True), sort_keys=True))


if __name__ == "__main__":
    main()
