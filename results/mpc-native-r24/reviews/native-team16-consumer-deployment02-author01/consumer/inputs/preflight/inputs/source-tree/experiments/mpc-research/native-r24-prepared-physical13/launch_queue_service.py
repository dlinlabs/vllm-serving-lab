"""Explicit queue-observation extension; original qualification launcher is unchanged."""

import argparse
import json
import os
from pathlib import Path
import sys

from queue_observer import source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "plan", "python", "evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    sys.path.insert(0, str(root / "experiments/mpc-research/native-v2"))
    from batch_capture_bridge import validate_plan
    validate_plan(plan)
    if plan.get("queue_observer_extension", {}).get("source_sha256") != source_hash():
        raise ValueError("prospectively_bound_queue_observer_required")
    prepared = plan.get("prepared_forward_extension")
    if prepared is not None:
        from prepared_forward_plugin import validate_plan
        validate_plan(plan)
    prepared_copy = plan.get("prepared_copy_ready_extension")
    if prepared_copy is not None:
        from prepared_copy_ready_plugin import validate_plan as validate_copy_plan
        validate_copy_plan(plan)
    args.evidence.mkdir(parents=True, exist_ok=True)
    with (args.evidence / "vllm.pid").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    os.chdir(root)
    os.environ["PATH"] = str(args.python.parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join((
        str(root), str(root / "experiments/mpc-research/native_observer"),
    ))
    if prepared is not None:
        os.environ["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + os.environ["PYTHONPATH"]
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
        VLLM_PLUGINS="pr33_native_observer,pr33_native_batch,pr33_r23_queue_observer",
        PR33_NATIVE_OBSERVER_ENABLED="1", PR33_NATIVE_OBSERVER_KEY=config["ADMIN_API_KEY"],
        PR33_NATIVE_BATCH_CAPTURE_ENABLED="1", PR33_NATIVE_BATCH_PLAN=str(args.plan.resolve()),
        PR33_R23_QUEUE_OBSERVER_ENABLED="1", PREFILL_LAB_ENABLED="1",
        PR33_R23_INPUT_POST_OBSERVER_ENABLED="0",
        PR33_R24_PHASE_OBSERVER_ENABLED="0",
        PR33_R24_HOST_TRACE_ENABLED="0",
        PR33_R24_COPY_READY_ENABLED="0",
        PR33_R24_COPY_READY_ZERO_TOKENS_ENABLED="0",
        PR33_R24_PREPARED_FORWARD_ENABLED="1" if prepared is not None else "0",
        PR33_R24_PREPARED_COPY_READY_ENABLED="1" if prepared_copy is not None else "0",
    )
    if plan.get("input_post_extension"):
        from input_post_observer import source_hash as input_source_hash
        if plan["input_post_extension"]["source_sha256"] != input_source_hash():
            raise ValueError("prospectively_bound_input_post_observer_required")
        os.environ["VLLM_PLUGINS"] += ",pr33_r23_input_post_observer"
        os.environ["PR33_R23_INPUT_POST_OBSERVER_ENABLED"] = "1"
    if plan.get("phase_frontier_extension"):
        from phase_observer import SCHEMA, source_hash as phase_source_hash
        if (not plan.get("input_post_extension")
                or plan["phase_frontier_extension"]["schema"] != SCHEMA
                or plan["phase_frontier_extension"]["source_sha256"] != phase_source_hash()):
            raise ValueError("prospectively_bound_live_phase_observer_required")
        os.environ["VLLM_PLUGINS"] += ",pr33_r24_phase_observer"
        os.environ["PR33_R24_PHASE_OBSERVER_ENABLED"] = "1"
    if plan.get("host_trace_extension"):
        from run_h16 import digest
        directory = Path(__file__).parent
        extension = plan["host_trace_extension"]
        if (not plan.get("phase_frontier_extension")
                or extension["plugin_sha256"] != digest(directory / "host_trace_plugin.py")
                or extension["helper_sha256"] != digest(directory / "native_host_trace.py")
                or extension["package_manifest_sha256"] != digest(directory / "pyproject.toml")):
            raise ValueError("prospectively_bound_host_trace_and_phase_required")
        os.environ["VLLM_PLUGINS"] += ",pr33_r24_host_trace"
        os.environ["PR33_R24_HOST_TRACE_ENABLED"] = "1"
    if plan.get("copy_ready_extension"):
        from copy_ready_acquisition import validate_extension
        if not plan.get("host_trace_extension"):
            raise ValueError("copy_ready_parent_host_trace_required")
        extension = plan["copy_ready_extension"]
        validate_extension(extension, Path(__file__).parent)
        os.environ["PYTHONPATH"] = extension["overlay"] + os.pathsep + os.environ["PYTHONPATH"]
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["VLLM_PLUGINS"] += ",pr33_r24_copy_ready"
        os.environ["PR33_R24_COPY_READY_ENABLED"] = "1"
        if extension.get("zero_token_routes_extension", {}).get("enabled") is True:
            os.environ["PR33_R24_COPY_READY_ZERO_TOKENS_ENABLED"] = "1"
    for flag in ("STAGES", "LIVE_STATE", "ENVELOPE_SHADOW", "EFFECTIVE_CALIBRATION",
                 "DELAYED_ARRIVALS", "EVENT_CLOCK", "QUEUE_PROBE", "QUEUE_CONFIRMATION", "ENTRY_HTTP"):
        os.environ["PREFILL_LAB_" + flag] = "0"
    command = [
        "serve", config["MODEL_NAME"], "--served-model-name", "Qwen/Qwen3-4B-Instruct-2507",
        "--host", "127.0.0.1", "--port", "8000", "--dtype", "bfloat16",
        "--max-model-len", "8192", "--max-num-seqs", "32", "--max-num-batched-tokens", "2048",
        "--gpu-memory-utilization", "0.8", "--no-enable-prefix-caching",
        "--tensor-parallel-size", "1", "--enable-request-id-headers",
    ]
    with (args.evidence / "vllm-launch.json").open("x") as stream:
        json.dump({
            "source_commit": plan["source_commit"], "arguments": command,
            "queue_observer_extension": plan["queue_observer_extension"],
            "input_post_extension": plan.get("input_post_extension"),
            "phase_frontier_extension": plan.get("phase_frontier_extension"),
            "host_trace_extension": plan.get("host_trace_extension"),
            "copy_ready_extension": plan.get("copy_ready_extension"),
            "prepared_forward_extension": prepared,
            "prepared_copy_ready_extension": prepared_copy,
            "scope": "Development queue observation, loopback and no MPC; not a new H16/E0 qualification",
        }, stream, sort_keys=True, indent=2)
    executable = args.python.with_name("vllm")
    os.execv(str(executable), [str(executable), *command])


if __name__ == "__main__":
    main()
