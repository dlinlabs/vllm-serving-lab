"""Loopback-only qualification service; no resource lifecycle or MPC activation."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("vllm", "gateway"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--identification-protocol", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    from batch_capture_bridge import validate_plan
    validate_plan(plan)
    args.evidence.mkdir(parents=True, exist_ok=True)
    with (args.evidence/f"{args.service}.pid").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    os.chdir(root)
    os.environ["PATH"] = str(args.python.parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_NO_USAGE_STATS="1")
    os.environ["PYTHONPATH"] = str(root) + ":" + str(root/"experiments/mpc-research/native_observer")
    if args.service == "vllm":
        os.environ.update(VLLM_PLUGINS="pr33_native_observer,pr33_native_batch",
            PR33_NATIVE_OBSERVER_ENABLED="1", PR33_NATIVE_OBSERVER_KEY=config["ADMIN_API_KEY"],
            PR33_NATIVE_BATCH_CAPTURE_ENABLED="1", PR33_NATIVE_BATCH_PLAN=str(args.plan.resolve()),
            PREFILL_LAB_ENABLED="1")
        for flag in ("STAGES", "LIVE_STATE", "ENVELOPE_SHADOW", "EFFECTIVE_CALIBRATION",
                     "DELAYED_ARRIVALS", "EVENT_CLOCK", "QUEUE_PROBE", "QUEUE_CONFIRMATION",
                     "ENTRY_HTTP"):
            os.environ["PREFILL_LAB_" + flag] = "0"
        executable = args.python.with_name("vllm")
        command = ["serve", config["MODEL_NAME"],
            "--served-model-name", "Qwen/Qwen3-4B-Instruct-2507", "--host", "127.0.0.1",
            "--port", "8000", "--dtype", "bfloat16", "--max-model-len", "8192",
            "--max-num-seqs", "32", "--max-num-batched-tokens", "2048",
            "--gpu-memory-utilization", "0.8", "--no-enable-prefix-caching",
            "--tensor-parallel-size", "1", "--enable-request-id-headers"]
    else:
        executable = args.python
        allowed = ("MODEL_NAME", "ADMIN_API_KEY", "COST_INTERCEPT", "INPUT_TOKEN_COST",
                   "OUTPUT_TOKEN_COST", "MIN_REQUEST_COST", "DEFAULT_MAX_TOKENS",
                   "MPC_TOKENIZER_IDENTITY")
        for key in allowed:
            if key in config:
                os.environ[key] = str(config[key])
        os.environ.update(VLLM_URL="http://127.0.0.1:8000", ADMISSION_MODE="cost_aware",
            MAX_ADMITTED_COST="32", MPC_LAB_ENABLED="true", BENCHMARK_REQUEST_IDS="1",
            MPC_MODEL_VERSION="native-v2-unfitted", MPC_CALIBRATION_VERSION="unfitted-not-validated",
            MPC_SUPPORT_HASH="qualification-only-no-control")
        if args.identification_protocol:
            import sys
            sys.path.insert(0, str(root))
            from mpc_lab.contracts import CostSpec, ModelParameters
            protocol = json.loads(args.identification_protocol.read_text())
            if (protocol["schema"] != "native-v3-state-response-id-v1"
                    or protocol["source_commit"] != plan["source_commit"]):
                raise ValueError("identification protocol source or schema mismatch")
            parameters = ModelParameters(cost_spec=CostSpec(**protocol["cost_spec"]),
                model_version="native-id-unfitted-v1",
                calibration_version="unfitted-not-validated")
            os.environ.update(MPC_MODEL_VERSION=parameters.model_version,
                MPC_CALIBRATION_VERSION=parameters.calibration_version,
                MPC_SUPPORT_HASH=parameters.identity.support_hash)
        command = ["-m", "uvicorn", "gateway:app", "--host", "127.0.0.1",
                   "--port", "8080", "--workers", "1"]
    with (args.evidence/f"{args.service}-launch.json").open("x") as stream:
        json.dump(dict(source_commit=plan["source_commit"], arguments=command,
            scope=("Prespecified identification only;" if args.identification_protocol
                   else "Qualification only;") + "loopback;no calibrated MPC."
                  " Batch hooks enabled;capture OFF is not pristine disabled."), stream, indent=2)
    os.execv(str(executable), [str(executable), *command])


if __name__ == "__main__":
    main()
