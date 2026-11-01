"""Post-drain scope witnesses against the frozen bridge; no native execution."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[4]
COMMIT = "1cbe251ced4d54bec8f56cfbce95983081abc16a"
SOURCE = "experiments/mpc-research/native-v2/batch_capture_bridge.py"
SHA256 = "0e24eb2c00d3b3da08f01a0d235e4a397551eee17f6853f5fe23dca295498f4a"
CONTRACTS = "prefill_lab/contracts.py"
CONTRACTS_SHA256 = "4d5cda37f5d5f765f286476da675f4155fffcce5e8b22d35d8623f8d8429a23f"


def load_bridge():
    source = subprocess.check_output(["git", "show", f"{COMMIT}:{SOURCE}"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError("unexpected frozen bridge source")
    module = types.ModuleType("_sol_r3_frozen_bridge")
    module.__file__ = str(ROOT / SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source, f"{COMMIT}:{SOURCE}", "exec"), module.__dict__)
    # Identity plumbing is a fixture, not a test of installation/source guarding.
    module.source_hash = lambda: SHA256
    return module


def load_audit():
    source = subprocess.check_output(["git", "show", f"{COMMIT}:{CONTRACTS}"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != CONTRACTS_SHA256:
        raise ValueError("unexpected frozen audit source")
    # Run the two unchanged stdlib-only functions, not NumPy/wire/native imports.
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in ("finite", "audit_capture")]
    namespace = {"math": math}
    exec(compile(tree, f"{COMMIT}:{CONTRACTS}", "exec"), namespace)
    return namespace["audit_capture"]


def capture():
    return dict(
        schema="prefill-capture-v1", trial_id="nv2-sol-review",
        status="complete", errors=[], overflow=0, pending_step_ids=[],
        requests={"request-A": dict(input_tokens=128, output_tokens=1,
                                   max_tokens=16, first=1.1, finished=2.,
                                   finish_reason="stop")},
        steps=[dict(step_id="step-A", start=1., end=2.,
                    work=[dict(request_id="request-A", position=0,
                               tokens=128, prompt_length=128)],
                    parity=dict(matches=True, actual={"request-A": 128},
                                expected={"request-A": 128}))],
        workers=[dict(pending=0, overflow=0, rows=[
            dict(step_id="step-A", counts={"request-A": 128},
                 executed_counts={"request-A": 128},
                 executed_positions={"request-A": 0}, actual_tokens=128,
                 gpu_forward_s=.01, cuda_scope_qualified=False)])])


def harness(module, raw):
    domain = "fixture-boot:fixture-namespace"
    clocks = [dict(process_id=123, clock_domain=domain, bridge_sha256=SHA256)]
    core = types.SimpleNamespace(model_executor=types.SimpleNamespace(
        collective_rpc=lambda name: clocks))
    calls = []

    def command(core, payload):
        calls.append(payload["operation"])
        if payload["operation"] == "meta":
            return dict(tensor_parallel_size=1, pipeline_parallel_size=1)
        if payload["operation"] == "external_begin":
            return dict(external=True)
        return copy.deepcopy(raw)

    plan = dict(schema="native-v2-batch-capture-plan-v1", partition="train",
                source_commit="a" * 40, trials=[
                    dict(trial_id="nv2-sol-review", seed=170001, observer="light")])
    bridge = module.BatchCapture(core, plan, command, clock=lambda: 3., domain=domain)
    bridge.invoke(dict(operation="metadata"))
    bridge.invoke(dict(operation="begin", trial_id="nv2-sol-review"))
    return bridge, calls


def diagnostic():
    module = load_bridge()
    audit = load_audit()
    cases = {}
    for kind in ("executed_work_mismatch", "restricted_parity_mismatch"):
        raw = capture()
        if kind == "executed_work_mismatch":
            row = raw["workers"][0]["rows"][0]
            row.update(executed_counts={"request-A": 64}, actual_tokens=64)
        else:
            raw["steps"][0]["parity"]["matches"] = False
        bridge, _ = harness(module, raw)
        reply = bridge.invoke(dict(operation="end"))
        cases[kind] = dict(
            bridge_capture_valid=reply["capture_valid"],
            bridge_validation_errors=reply["validation_errors"],
            raw_retained=reply["capture"] == raw,
            offline_audit=audit(raw, raw["workers"][0]["rows"], forced_outputs=False))

    matching = capture()
    bridge, calls = harness(module, matching)
    first = bridge.invoke(dict(operation="end"))
    replay = bridge.invoke(dict(operation="end"))
    ack = dict(operation="ack", trial_id=first["trial_id"],
               capture_sha256=first["capture_sha256"])
    bridge.invoke(ack)
    try:
        bridge.invoke(ack)
        duplicate_ack_rejected = False
    except ValueError:
        duplicate_ack_rejected = True
    status = bridge.invoke(dict(operation="status"))
    return dict(
        schema="sol-round3-bridge-scope-review-v1",
        source_commit=COMMIT, source_sha256=SHA256,
        audit_sha256=CONTRACTS_SHA256,
        native_execution=False, installation_qualification=False,
        matching_capture_audit=audit(matching, matching["workers"][0]["rows"],
                                    forced_outputs=False),
        scope_cases=cases,
        lost_response_workflow=dict(
            end_replays_exact_packet=first == replay,
            external_end_calls=calls.count("external_end"),
            duplicate_ack_rejected=duplicate_ack_rejected,
            same_run=status["run_id"] == first["run_id"],
            unacknowledged_trial=status["unacknowledged_trial"],
            consumed_trial_still_recorded=first["trial_id"] in status["used_trials"]))


if __name__ == "__main__":
    print(json.dumps(diagnostic(), indent=2, allow_nan=False))
