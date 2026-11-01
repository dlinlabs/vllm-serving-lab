"""CPU witnesses for the frozen bridge acceptance boundary; no native execution."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace


FREEZE = "9b02f6ebdbd6681763b562bc3c37aa564b8b330a"
BRIDGE_SHA = "0e24eb2c00d3b3da08f01a0d235e4a397551eee17f6853f5fe23dca295498f4a"
ROOT = Path(__file__).resolve().parents[4]


def load_bridge():
    path = "experiments/mpc-research/native-v2/batch_capture_bridge.py"
    source = subprocess.check_output(["git", "show", f"{FREEZE}:{path}"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != BRIDGE_SHA:
        raise ValueError("bridge source differs")
    module = ModuleType("astra_frozen_bridge")
    module.__file__ = f"{FREEZE}:{path}"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    # The reviewed blob is in Git, not a worktree file; return its verified hash.
    module.source_hash = lambda: BRIDGE_SHA
    return module


def fixture():
    return {
        "schema": "prefill-capture-v1", "trial_id": "nv2-astra-check",
        "status": "complete", "errors": [], "overflow": 0, "pending_step_ids": [],
        "requests": {"r": {"finished": 2, "input_tokens": 128, "output_tokens": 1,
                           "max_tokens": 128, "finish_reason": "stop"}},
        "steps": [{"step_id": "k", "start": 1, "end": 2, "parity": {"matches": True},
                   "work": [{"request_id": "r", "position": 0, "tokens": 128}]}],
        "workers": [{"pending": 0, "overflow": 0, "rows": [{
            "step_id": "k", "counts": {"r": 128}, "executed_counts": {"r": 128},
            "executed_positions": {"r": 0}, "actual_tokens": 128,
            "gpu_forward_s": .01, "cuda_scope_qualified": False,
        }]}],
    }


def bridge_for(capture):
    module = load_bridge()
    calls = []
    clocks = [{"process_id": 7, "clock_domain": "fixture-domain", "bridge_sha256": BRIDGE_SHA}]
    core = SimpleNamespace(model_executor=SimpleNamespace(collective_rpc=lambda name: clocks))
    plan = {"schema": "native-v2-batch-capture-plan-v1", "partition": "train",
            "source_commit": "a" * 40,
            "trials": [{"trial_id": "nv2-astra-check", "observer": "light", "seed": 170001}]}

    def command(core, payload):
        calls.append(payload["operation"])
        if payload["operation"] == "meta":
            return {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}
        if payload["operation"] == "external_begin":
            return {"external": True}
        return copy.deepcopy(capture)

    bridge = module.BatchCapture(core, plan, command, clock=lambda: 3, domain="fixture-domain")
    bridge.invoke({"operation": "metadata"})
    return bridge, calls, clocks


def completed(capture):
    bridge, calls, clocks = bridge_for(capture)
    bridge.invoke({"operation": "begin", "trial_id": "nv2-astra-check"})
    result = bridge.invoke({"operation": "end"})
    return bridge, calls, result


def diagnostic():
    results = {}
    for field, value in (("executed_counts", {"r": 127}),
                         ("executed_positions", {"r": 1})):
        capture = fixture()
        capture["workers"][0]["rows"][0][field] = value
        _, _, result = completed(capture)
        results[field] = {"capture_valid": result["capture_valid"],
                          "validation_errors": result["validation_errors"],
                          "raw_mismatch_retained": result["capture"] == capture}
    return {"freeze": FREEZE, "bridge_sha256": BRIDGE_SHA,
            "native_execution": False, "matching_keys_not_work_parity": results}


if __name__ == "__main__":
    print(json.dumps(diagnostic(), indent=2, sort_keys=True, allow_nan=False))
