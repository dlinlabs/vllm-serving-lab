"""Default-OFF lifecycle; CLI offers no inference or admin POST."""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time


MODEL = "/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507"
ALIAS = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER = "4daabcf7fab3bce457d30dbd14e7eb9c5e79d77c15916f6abdb85b7b8bf0d2c5"
GPU = "GPU-8610519d-3680-ae5f-cdfc-40037bfb5013"
BACKEND_PORT = 18000
GATEWAY_PORT = 18001
LOCKS = ("/workspace/pr32-r23/r24-team16-mayor-gpu.lock",
         "/workspace/pr32-r23/native-component-GPU-8610519d.lock")
HELPER_SHA = "5249221b9bd795c8265039502746fe30a394e774a52040f9fcf6261146919bed"
RELATIVE = Path("experiments/mpc-research/resume-mvp")
SETTINGS = dict(scheduling_policy="fcfs", enable_prefix_caching=False,
                max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=8192,
                gpu_memory_utilization=0.9, enforce_eager=False, tensor_parallel_size=1)


class ProbeError(RuntimeError):
    pass


def stamp():
    return dict(monotonic=time.monotonic(), unix_ns=time.time_ns())


def write(path, value, secret="", append=False):
    text = json.dumps(value, sort_keys=True, allow_nan=False)
    if secret:
        text = text.replace(secret, "<redacted>")
    with path.open("a" if append else "x") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checked(condition, reason):
    if not condition:
        raise ProbeError(reason)


def configuration(root, output, protocol, identity):
    backend = protocol["backend"]
    checked(tuple(protocol["hardware"]["lock_paths"]) == LOCKS
            and protocol["hardware"]["gpu_uuid"] == GPU, "production lock/GPU mismatch")
    checked(backend["local_assets"] == MODEL and backend["served_alias"] == ALIAS
            and backend["tokenizer_fingerprint"] == TOKENIZER
            and backend["required_common_settings"] == SETTINGS, "backend proposal mismatch")
    checked(protocol["controller"]["model_version"] == "cpu-lifecycle-v1"
            and protocol["controller"]["calibration_version"] == "synthetic-unqualified-v1",
            "nominal labels mismatch")
    checked(identity["model_version"] == protocol["controller"]["model_version"]
            and identity["calibration_version"] == protocol["controller"]["calibration_version"],
            "local ModelParameters labels mismatch")
    env = dict(PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", os.defpath),
               HOME=os.environ.get("HOME", str(output)),
               CUDA_VISIBLE_DEVICES=GPU, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", TMPDIR=str(output / "scratch"),
               XDG_CACHE_HOME=str(output / "cache"),
               VLLM_CACHE_ROOT=str(output / "cache/vllm"),
               TORCHINDUCTOR_CACHE_DIR=str(output / "cache/inductor"),
               TRITON_CACHE_DIR=str(output / "cache/triton"))
    gateway_env = dict(env, VLLM_URL=f"http://127.0.0.1:{BACKEND_PORT}", MODEL_NAME=MODEL,
                       ADMISSION_MODE="cost_aware", MAX_ADMITTED_COST="32",
                       MPC_LAB_ENABLED="true", MPC_OWNER_LEDGER_ENABLED="true",
                       MPC_TOKENIZER_IDENTITY=TOKENIZER, MPC_MODEL_VERSION=identity["model_version"],
                       MPC_CALIBRATION_VERSION=identity["calibration_version"],
                       MPC_SUPPORT_HASH=identity["support_hash"],
                       COST_INTERCEPT="0.259", INPUT_TOKEN_COST="0.000758",
                       OUTPUT_TOKEN_COST="0.0052083333", MIN_REQUEST_COST="0.1",
                       DEFAULT_MAX_TOKENS="128", WEB_CONCURRENCY="1", UVICORN_WORKERS="1",
                       SHARED_UPSTREAM_CLIENT="1", BENCHMARK_REQUEST_IDS="1",
                       TELEMETRY_JSONL=str(output / "telemetry.jsonl"),
                       TELEMETRY_STATUS_JSON=str(output / "telemetry-status.json"))
    return dict(lock_paths=list(LOCKS), safe_environment=dict(engine=env, gateway=gateway_env),
                engine=[str(Path(sys.executable).parent / "vllm"), "serve", MODEL,
                        "--served-model-name", ALIAS, "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
                        "--scheduling-policy", "fcfs", "--no-enable-prefix-caching",
                        "--max-model-len", "8192", "--max-num-seqs", "256",
                        "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.9",
                        "--tensor-parallel-size", "1"],
                gateway=[sys.executable, "-m", "uvicorn", "gateway:app", "--app-dir", str(root),
                         "--host", "127.0.0.1", "--port", str(GATEWAY_PORT), "--workers", "1"])


def source_info(root):
    sys.path[:0] = [str(root), str(root / RELATIVE)]
    for name, module in tuple(sys.modules.items()):
        if name == "mpc_lab" or name.startswith("mpc_lab.") or name == "prefill_lab" or name.startswith("prefill_lab."):
            checked(getattr(module, "__file__", None)
                    and Path(module.__file__).resolve().is_relative_to(root),
                    "stale cached SDK module; use isolated source-root process")
    import mpc_lab.contracts as contracts
    import prefill_lab.live_http as live
    import nominal_services

    checked(Path(contracts.__file__).resolve() == root / "mpc_lab/contracts.py"
            and Path(live.__file__).resolve() == root / "prefill_lab/live_http.py"
            and Path(nominal_services.__file__).resolve() == root / RELATIVE / "nominal_services.py",
            "source bootstrap mismatch")
    paths = [root / "gateway.py", root / RELATIVE / "nominal_protocol.json",
             root / RELATIVE / "nominal_services.py", Path(__file__).resolve(),
             Path(contracts.__file__).resolve(), Path(live.__file__).resolve()]
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    checked(pins[str(root / RELATIVE / "nominal_services.py")] == HELPER_SHA,
            "accepted ownership helper changed")
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "httpx", "uvicorn", "transformers")}
    versions["python"] = platform.python_version()
    checked(versions["python"] == "3.12.3" and versions["vllm"] == "0.30.0",
            "preregistered host version mismatch")
    spec = contracts.CostSpec(
        intercept=.259, input_per_token=.000758, output_per_token=.0052083333, minimum=.1,
        default_max_tokens=128,
        tokenizer_identity=TOKENIZER + f";model={MODEL};available=True;fallback=chars4-chat-v1")
    identity = contracts.ModelParameters(cost_spec=spec).identity
    return dict(identity=asdict(identity), cost_spec=asdict(spec),
                cost_spec_fingerprint=spec.fingerprint, clock_domain=live.clock_domain(),
                versions=versions, source_pins=pins, evidence_kind="ACTUAL_HOST_COMPONENT_ONLY")


def existing_assets(config):
    return Path(config["engine"][0]).is_file() and Path(MODEL).is_dir()


def gpu_receipt(deadline):
    result = dict(start=stamp(), argv=[], commands=[])
    for query in ("--query-gpu=uuid,name", "--query-compute-apps=gpu_uuid,pid,used_memory"):
        remaining = deadline - time.monotonic()
        checked(remaining > 0, "no original GPU-receipt budget")
        argv = ["nvidia-smi", "-i", GPU, query, "--format=csv,noheader,nounits"]
        process = subprocess.run(argv, capture_output=True, text=True, timeout=min(5, remaining))
        result["argv"].append(argv)
        result["commands"].append(dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
    result["end"] = stamp()
    result["idle_selected_gpu"] = (
        all(row["returncode"] == 0 for row in result["commands"])
        and result["commands"][0]["stdout"].split(",")[0].strip() == GPU
        and result["commands"][1]["stdout"].strip() == "")
    return result


def http_client():
    import httpx
    return httpx.AsyncClient(trust_env=False, follow_redirects=False)


async def read_ready(cutoff, key, event):
    values = {}
    async with asyncio.timeout_at(cutoff), http_client() as client:
        for name, url in (
                ("health", f"http://127.0.0.1:{BACKEND_PORT}/health"),
                ("models", f"http://127.0.0.1:{BACKEND_PORT}/v1/models"),
                ("metrics", f"http://127.0.0.1:{GATEWAY_PORT}/metrics"),
                ("state", f"http://127.0.0.1:{GATEWAY_PORT}/admin/mpc/state")):
            remaining = cutoff - time.monotonic()
            checked(remaining > 0, "readiness original deadline")
            started = stamp()
            response = await client.get(url, timeout=min(2, remaining),
                                        headers={"x-admin-api-key": key} if name == "state" else {})
            event(dict(kind="GET", name=name, start=started, end=stamp(), status=response.status_code,
                       body=response.text[:65536], body_truncated=len(response.text) > 65536))
            if response.status_code in (502, 503):
                raise ConnectionError("service not ready")
            checked(response.status_code == 200, "readiness HTTP status mismatch")
            try:
                values[name] = None if name == "health" else response.json()
            except ValueError as error:
                raise ProbeError("non-JSON readiness response") from error
    return values


def validate_ready(values, expected):
    metrics, state = values["metrics"], values["state"]
    checked({row["id"] for row in values["models"]["data"]} == {ALIAS}, "served alias mismatch")
    checked(metrics["tokenizer_available"] is True and metrics["tokenizer_fallback_requests"] == 0
            and metrics["tokenizer_model"] == MODEL, "tokenizer unavailable/fallback/mismatch")
    checked(metrics["admission_mode"] == "cost_aware" and metrics["max_admitted_cost"] == 32
            and metrics["initial_admission_budget"] == 32 and metrics["budget_version"] == 0,
            "fixed startup cap mismatch")
    checked(all(metrics[name] == 0 for name in ("accepted_requests", "rejected_requests",
            "failed_requests", "current_in_flight", "current_waiting", "current_admitted_cost")),
            "unexpected request activity")
    checked(metrics["shared_upstream_client"] is True, "shared transport missing")
    checked(state["identity"] == expected["identity"]
            and state["cost_spec"] == expected["cost_spec"]
            and state["clock_domain"] == expected["clock_domain"], "source/CostSpec/clock identity mismatch")


def owner_scope(config, output, deadline, ready, key, args):
    from nominal_services import NominalServices, Service
    services = tuple(Service(name, tuple(config[name]),
                             dict(config["safe_environment"][name],
                                  **({"ADMIN_API_KEY": key} if name == "gateway" else {})),
                             tuple(config[name])) for name in ("engine", "gateway"))
    return NominalServices(services, lock_paths=LOCKS, run_dir=output / "services",
                           deadline=deadline, ready=ready, execute=True,
                           cleanup_reserve=args.cleanup_reserve, term_grace=args.term_grace)


def run(args, *, body=None):
    if not args.execute:
        return 2
    checked(args.source_root and args.output_root, "explicit source/output roots required")
    root, output = Path(args.source_root), Path(args.output_root)
    checked(root.is_absolute() and output.is_absolute(), "absolute source/output roots required")
    checked(all(math.isfinite(x) for x in (args.seconds, args.cleanup_reserve, args.term_grace))
            and 0 < args.term_grace < args.cleanup_reserve < args.seconds, "invalid original budget")
    root, output = root.resolve(strict=True), output.absolute()
    os.umask(0o077)
    output.mkdir(mode=0o700)
    began = stamp()
    deadline, key, scope = began["monotonic"] + args.seconds, "", None
    report = dict(start=began, deadline_monotonic=deadline, cleanup_reserve=args.cleanup_reserve,
                  term_grace=args.term_grace, lifecycle_passed=False, offered_requests_from_driver=0,
                  admin_POSTs=0, inference_or_MPC_or_benchmark_result=False, native_qualification=False,
                  actual_execution_authority="external Mayor preregistration, not this flag or protocol",
                  ownership_receipt=None, post_gpu=None, errors=[])

    def event(value):
        write(output / "events.jsonl", value, key, append=True)

    try:
        protocol_path = root / RELATIVE / "nominal_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        expected = source_info(root)
        report["source"] = expected
        config = configuration(root, output, protocol, expected["identity"])
        report["configuration"] = config
        for name in ("scratch", "cache"):
            (output / name).mkdir()
        checked(existing_assets(config), "missing existing local assets/entrypoint")
        write(output / "frozen-run-inputs.json", dict(report, frozen_before_service_spawn=True))
        report["pre_gpu"] = gpu_receipt(deadline)
        write(output / "pre-gpu.json", report["pre_gpu"])
        checked(report["pre_gpu"]["idle_selected_gpu"], "selected GPU not proven idle")
        key = secrets.token_hex(32)
        cutoff = deadline - args.cleanup_reserve

        def ready(owner, remaining):
            try:
                values = asyncio.run(read_ready(cutoff, key, event))
            except ProbeError:
                raise
            except Exception as error:
                event(dict(kind="readiness_retry", at=stamp(), error_type=type(error).__name__))
                time.sleep(min(.5, max(0, cutoff - time.monotonic())))
                return False
            event(dict(kind="readiness_values", at=stamp(), values=values))
            validate_ready(values, expected)
            report["readiness"] = dict(at=stamp(), values=values)
            return True

        event(dict(kind="ownership_enter_begin", at=stamp()))
        scope = owner_scope(config, output, deadline, ready, key, args)
        with scope as owner:
            owner.check()
            report["ready_body"] = stamp()
            if body is not None:
                report.update(inference_or_MPC_or_benchmark_result=True,
                              offered_requests_from_driver=None, admin_POSTs=None)
                body(owner=owner, root=root, output=output, key=key, report=report)
        report["ownership_exit_returned"] = stamp()
    except BaseException as error:
        report["errors"].append(dict(type=type(error).__name__, message=str(error)))
        event(dict(kind="failure", at=stamp(), type=type(error).__name__, message=str(error)))
        if not isinstance(error, Exception):
            raise
    finally:
        try:
            receipt_path = output / "services/ownership.json"
            if receipt_path.is_file():
                report["ownership_receipt"] = json.loads(receipt_path.read_text())
        except Exception as error:
            report["errors"].append(dict(type=type(error).__name__, message=str(error)))
        receipt = report["ownership_receipt"]
        closed = False
        try:
            closed = bool(receipt and receipt.get("status") == "STOPPED"
                          and receipt.get("known_identities_gone") is True
                          and receipt.get("unknown") == [] and receipt.get("pending_spawn") is None
                          and receipt.get("lock_paths") == sorted(LOCKS)
                          and scope is not None
                          and (scope._thread is None or not scope._thread.is_alive()) and scope._gone())
        except Exception as error:
            report["errors"].append(dict(type=type(error).__name__, message=str(error)))
        report["known_closure_checked"] = dict(at=stamp(), closed=closed)
        if not closed:
            report["errors"].append(dict(
                type="OwnershipNotClosed",
                message="No joined STOPPED/known-gone exact-lock receipt; post-GPU call skipped"))
        if closed:
            try:
                report["post_gpu"] = gpu_receipt(deadline)
                write(output / "post-gpu.json", report["post_gpu"])
                status = json.loads((output / "telemetry-status.json").read_text())
                report["telemetry_status"] = status
                checked(status["status"] == "closed" and status["mpc_ledger_complete"] is True
                        and all(status[name] == 0 for name in (
                            "dropped", "write_errors", "mpc_remaining_requests",
                            "mpc_unflushed_records", "mpc_final_sequence", "mpc_producer_final_sequence")),
                        "telemetry closure or no-request ledger mismatch")
                checked(report["post_gpu"]["idle_selected_gpu"], "post GPU not proven idle")
            except Exception as error:
                report["errors"].append(dict(type=type(error).__name__, message=str(error)))
        report["end"] = stamp()
        report["lifecycle_passed"] = bool(
            not report["errors"] and closed and "readiness" in report
            and "ownership_exit_returned" in report
            and report["post_gpu"] and report["end"]["monotonic"] <= deadline)
        write(output / "report.json", report, key)
    return 0 if report["lifecycle_passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seconds", type=float, default=480)
    parser.add_argument("--cleanup-reserve", type=float, default=30)
    parser.add_argument("--term-grace", type=float, default=10)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
