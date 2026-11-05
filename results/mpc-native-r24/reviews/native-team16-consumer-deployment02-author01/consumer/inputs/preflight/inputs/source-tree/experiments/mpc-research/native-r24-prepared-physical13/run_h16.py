"""Newly authorized owned engine lifecycle around the immutable H16 collector."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import uuid

from engine_handoff import owned_engine, require_free_ports


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PUBLIC_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL = Path("/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507")
ALIAS = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots") / REVISION
RUNTIME = ROOT / "results/mpc-native-r23/runtime"
BASE_ARCHIVE_SHA256 = "12b846ce97df380ecdbce50f26c25daf29cd6aa57ba3e55e33708d738f79bcbf"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def runtime_identity():
    from mpc_local.control import qualify_source
    from mpc_lab.runtime import local_clock_domain
    from prefill_lab.plugin import installed_sources
    import torch

    expected = {"vllm": "0.30.0", "torch": "2.13.0+cu130", "transformers": "5.22.1"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected or torch.cuda.device_count() != 1:
        raise ValueError("actual_native_versions_or_single_gpu")
    if "4090" not in torch.cuda.get_device_name(0):
        raise ValueError("authorized_hardware_changed")
    manifest = json.loads((MODEL / "verified-manifest.json").read_text())
    if manifest["model"] != PUBLIC_MODEL or manifest["revision"] != REVISION:
        raise ValueError("model_revision")
    lock_path = Path(__file__).with_name("model-content-lock.json")
    lock = json.loads(lock_path.read_text())
    if lock["model"] != PUBLIC_MODEL or lock["revision"] != REVISION:
        raise ValueError("model_content_lock_revision")
    expected = {row["file"]: row for row in lock["files"]}
    if (len(manifest["files"]) != len(expected)
            or {row["file"]: row["sha256"] for row in manifest["files"]}
            != {name: row["sha256"] for name, row in expected.items()}):
        raise ValueError("model_manifest_not_pinned_content")
    for row in expected.values():
        path = MODEL / row["file"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("model_bytes:" + row["file"])
        if not row["lfs"]:
            raw = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if blob != row["upstream_git_blob"]:
                raise ValueError("model_git_blob:" + row["file"])
    if ALIAS.exists() or ALIAS.is_symlink():
        if ALIAS.resolve() != MODEL:
            raise ValueError("preexisting_model_alias_differs")
    else:
        ALIAS.parent.mkdir(parents=True, exist_ok=True)
        ALIAS.symlink_to(MODEL, target_is_directory=True)
    sample = torch.ones((32, 32), device="cuda")
    product = sample @ sample
    torch.cuda.synchronize()
    if product[0, 0].item() != 32:
        raise ValueError("native_cuda_compute_failed")
    del sample, product
    torch.cuda.empty_cache()
    return {
        "packages": actual, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "clock_domain": local_clock_domain(), "hostname": os.uname().nodename,
        "source": qualify_source(), "installed_vllm_sources": installed_sources(),
        "model_manifest_sha256": digest(MODEL / "verified-manifest.json"),
        "model_content_lock_sha256": digest(lock_path),
        "model_alias": str(ALIAS), "actual_model_root": str(MODEL),
        "wrapper_sha256": digest(__file__),
        "engine_handoff_sha256": digest(Path(__file__).with_name("engine_handoff.py")),
        "native_mpc_permitted": False,
    }


def consumer():
    from mpc_local.native_pilot import POST_CONSUMER_SHA
    path = RUNTIME / "native_release_model.py"
    if digest(path) != POST_CONSUMER_SHA:
        raise ValueError("original_h16_consumer_sha256")
    spec = importlib.util.spec_from_file_location("r23_original_h16_consumer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gateway_probe(process, config, protocol, evidence):
    import httpx
    from mpc_lab.actuator import IdentificationPermit
    from mpc_lab.contracts import CostSpec, ModelParameters
    from mpc_lab.runtime import local_clock_domain
    from mpc_local.native_pilot import basic_runtime_probe

    parameters = ModelParameters(
        cost_spec=CostSpec(**protocol["cost_spec"]), model_version="native-id-unfitted-v1",
        calibration_version="unfitted-not-validated",
    )
    with httpx.Client(timeout=5, trust_env=False) as client:
        for _ in range(90):
            if process.poll() is not None:
                raise ValueError("basic_gateway_exited")
            try:
                response = client.get(
                    "http://127.0.0.1:8080/admin/mpc/state",
                    headers={"x-admin-api-key": config["ADMIN_API_KEY"]},
                )
                response.raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            raise ValueError("basic_gateway_not_ready")

    async def probe():
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8080", timeout=5, trust_env=False,
            headers={"x-admin-api-key": config["ADMIN_API_KEY"]},
        ) as client:
            return await basic_runtime_probe(
                client, parameters, pending_path=evidence / "probe-pending.json",
                authorization=IdentificationPermit(
                    parameters.identity, approved=True, run_id=uuid.uuid4().hex,
                    clock_domain=local_clock_domain(), expires_at=time.monotonic() + 180,
                ),
            )
    return asyncio.run(probe())


def native_readiness(process, config, identity, evidence):
    import httpx
    from mpc_local.control import SOURCE_HASHES

    deadline = time.monotonic() + 900
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ValueError("owned_vllm_exited_during_startup")
            try:
                client.get("/health").raise_for_status()
                models = client.get("/v1/models")
                models.raise_for_status()
                if PUBLIC_MODEL not in {row["id"] for row in models.json()["data"]}:
                    raise ValueError("served_model_identity")
                break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            raise ValueError("native_startup_operation_timeout")
        warmup = client.post("/v1/chat/completions", timeout=60, json={
            "model": PUBLIC_MODEL, "messages": [{"role": "user", "content": "Count from one to five."}],
            "temperature": 0, "max_tokens": 16,
        })
        warmup.raise_for_status()
        write(evidence / "warmup-response.json", warmup.json())
        frames = []
        engine_run = None
        deadline = time.monotonic() + 30
        with (evidence / "readiness-native.jsonl").open("x") as observations:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError("owned_vllm_exited_during_drain")
                reply = client.get("/pr33-native/state",
                                   headers={"x-observer-key": config["ADMIN_API_KEY"]})
                observations.write(json.dumps({
                    "received_at": time.monotonic(), "status_code": reply.status_code,
                    "body": reply.text,
                }) + "\n")
                observations.flush()
                os.fsync(observations.fileno())
                reply.raise_for_status()
                frame = reply.json()
                if (frame["clock_domain"] != identity["clock_domain"]
                        or frame["observer_sha256"] != SOURCE_HASHES[
                            "experiments/mpc-research/native_observer/pr33_native_observer.py"]):
                    raise ValueError("native_source_or_clock_changed")
                engine_run = engine_run or frame["run_id"]
                if frame["run_id"] != engine_run:
                    raise ValueError("native_epoch_changed")
                drained = (
                    frame["complete"]
                    and all(frame[key] == 0 for key in
                            ("running", "waiting", "pending_batches", "frontend_unfinished"))
                    and 0 <= time.monotonic() - frame["sampled_at"] <= 0.5
                )
                frames = frames + [frame] if drained else []
                if len(frames) == 2:
                    break
                time.sleep(0.1)
            else:
                raise ValueError("native_initial_drain_not_established")
    return {"frames": frames, "warmup_response": warmup.json(),
            "scope": "Real native HTTP warmup and empty fresh source-bound native state, not H16 outcomes"}


def execute(args):
    from mpc_local.native_pilot import DEPLOYMENT_COMMIT, EXTENSION, H16_PROTOCOL_SHA, SERVING_COMMIT
    from mpc_local.native_pilot import extension
    from mpc_local.control import audit_owner_log
    from mpc_lab.contracts import CostSpec, ModelParameters

    campaign = json.loads((Path(__file__).with_name("campaign.json")).read_text())
    if not campaign["authorization_changes"]["gpu_native_measurement_and_data_collection"]:
        raise ValueError("explicit_r23_campaign_authorization_required")
    original_path = RUNTIME / "h16-original-protocol.json"
    if digest(original_path) != H16_PROTOCOL_SHA:
        raise ValueError("immutable_original_h16_protocol")
    original = json.loads(original_path.read_text())
    proof = consumer()
    binding = proof.h16_source_binding(original_path, ROOT, ROOT / EXTENSION)
    extension().validate_protocol(original)
    require_free_ports()
    if digest(ROOT.parent / "deployment-source.tar.gz") != BASE_ARCHIVE_SHA256:
        raise ValueError("frozen_base_source_archive_changed")
    identity = runtime_identity()
    if args.phase == "h16":
        if args.smoke_receipt is None:
            raise ValueError("actual_successful_smoke_receipt_required")
        smoke = json.loads(args.smoke_receipt.read_text())
        if (smoke.get("status") != "smoke_collected" or smoke.get("phase") != "smoke"
                or smoke.get("runtime_identity") != identity
                or digest(args.smoke_receipt.parent / "collection/index.json")
                != smoke.get("collection_index_sha256")):
            raise ValueError("smoke_runtime_source_or_result_binding")
    run_id = f"{args.phase}-{uuid.uuid4().hex}"
    output = ROOT / "results/mpc-native-r23" / run_id
    output.mkdir(mode=0o700)
    private = ROOT / ".runtime/r23" / run_id
    private.mkdir(parents=True, mode=0o700)
    os.chmod(private, 0o700)
    config = {
        "MODEL_NAME": str(ALIAS), "ADMIN_API_KEY": secrets.token_hex(32),
        "COST_INTERCEPT": original["cost_spec"]["intercept"],
        "INPUT_TOKEN_COST": original["cost_spec"]["input_per_token"],
        "OUTPUT_TOKEN_COST": original["cost_spec"]["output_per_token"],
        "MIN_REQUEST_COST": original["cost_spec"]["minimum"],
        "DEFAULT_MAX_TOKENS": original["cost_spec"]["default_max_tokens"],
        "MPC_TOKENIZER_IDENTITY": "gateway-token-estimate-v1;Qwen3-4B-2507",
        "MPC_OWNER_LEDGER_ENABLED": "true",
    }
    if ((str(config.get("MPC_OWNER_LEDGER_ENABLED", "false")).lower() == "true")
            != original["owner_journal_required"]):
        raise ValueError("owner_journal_config_mismatch_before_startup")
    config_path = private / "control.private.json"
    write(config_path, config)
    os.chmod(config_path, 0o600)
    plan_path = output / "plan.json"
    write(plan_path, {"schema": "native-v2-batch-capture-plan-v1", "partition": "train",
                     "source_commit": SERVING_COMMIT,
                     "trials": [{"trial_id": "nv2-r23-qualification", "observer": "light", "seed": 170001}]})
    protocol = original
    protocol_path = original_path
    if args.phase == "smoke":
        protocol = extension().legacy_collector().protocol_for(
            SERVING_COMMIT, original["cost_spec"], smoke=True, gateway_ledger=True,
        )
        protocol_path = output / "smoke-protocol.json"
        write(protocol_path, protocol)
    env = {**os.environ, "MPC_OWNER_LEDGER_ENABLED": "true", "PYTHONPATH": os.pathsep.join((
        str(ROOT), str(ROOT / "experiments/mpc-research"),
        str(ROOT / "experiments/mpc-research/native_observer"),
    ))}
    launcher = ROOT / "experiments/mpc-research/native-v2/launch_batch_service.py"
    common = ["--config", str(config_path), "--plan", str(plan_path), "--python", sys.executable]
    outcome = {"schema": "r23-native-owned-run-v1", "phase": args.phase, "run_id": run_id,
               "runtime_identity": identity, "status": "starting",
               "protocol_sha256": digest(protocol_path), "h16_binding": binding,
               "native_mpc_permitted": False, "provider_billing_stop_verified": False}
    try:
        basic_dir = output / "basic-gateway"
        command = [sys.executable, str(launcher), "gateway", *common, "--evidence", str(basic_dir),
                   "--identification-protocol", str(original_path)]
        with owned_engine(command, evidence=basic_dir, env=env, cwd=ROOT,
                          ready=lambda p: gateway_probe(p, config, original, basic_dir)) as (_, basic):
            basic_receipt = basic["readiness"]
        parameters = ModelParameters(
            cost_spec=CostSpec(**original["cost_spec"]), model_version="native-id-unfitted-v1",
            calibration_version="unfitted-not-validated",
        )
        audit = audit_owner_log(
            (basic_dir / "gateway-telemetry.jsonl").read_bytes(),
            json.loads((basic_dir / "gateway-telemetry-status.json").read_text()),
            run_id=basic_receipt["gateway_run_id"], clock_domain=identity["clock_domain"],
            identity=parameters.identity, request_costs={},
        )
        write(output / "basic-runtime-probe.json", {"probe": basic_receipt, "closed_audit": audit})
        require_free_ports()
        engine_dir = output / "engine"
        command = [sys.executable, str(launcher), "vllm", *common, "--evidence", str(engine_dir)]
        with owned_engine(
            command, evidence=engine_dir, env=env, cwd=ROOT,
            ready=lambda p: native_readiness(p, config, identity, engine_dir),
        ):
            launch_path = engine_dir / "vllm-launch.json"
            launch = json.loads(launch_path.read_text())
            launch.update(
                serving_base_commit=SERVING_COMMIT, deployment_bundle_commit=DEPLOYMENT_COMMIT,
                r23_actual_source_archive_sha256=digest(ROOT.parent / "deployment-source.tar.gz"),
                r23_actual_source_parent="b3134981f0ee4726b6b06355005311ce63ae89d5",
                r23_new_wrapper_sha256=identity["wrapper_sha256"],
                original_identity_scope="Pinned original serving/deployment components, not entire R23 archive",
            )
            launch_path = engine_dir / "vllm-launch-r23-bound.json"
            write(launch_path, launch)
            entry = ROOT / "experiments/mpc-research/native-v3" / (
                "collect_identification.py" if args.phase == "smoke" else "prospective_identification.py")
            command = [sys.executable, str(entry), "--config", str(config_path),
                       "--protocol", str(protocol_path), "--output", str(output / "collection"),
                       "--vllm-launch", str(launch_path), "--gateway-plan", str(plan_path)]
            if args.phase == "h16":
                extension().execution_preflight(
                    original, config, backend_free=True, epoch_qualified=True,
                )
                command += ["--execute", "--backend-free", "--epoch-qualified"]
            epochs = (len(protocol["episodes"]) + protocol["gateway_epoch_episode_limit"] - 1
                      ) // protocol["gateway_epoch_episode_limit"]
            watchdog_s = (protocol["planned_offered_window_s"]
                          + 120 * len(protocol["episodes"]) + 90 * epochs + 60)
            outcome["collector_watchdog"] = {
                "seconds": watchdog_s,
                "scope": "Per-attempt hang guard; offered window plus120s/episode,90s/epoch and60s shutdown; not a campaign budget or efficacy stopping rule",
            }
            with owned_engine(
                command, evidence=output / "collector-process", env=env, cwd=ROOT,
                ready=lambda p: {"scope": "Collector process started after qualified native readiness"},
            ) as (collector, _):
                if collector.wait(timeout=watchdog_s) != 0:
                    raise ValueError("native_collector_failed_preserved_raw")
            index_path = output / "collection/index.json"
            index = json.loads(index_path.read_text())
            if not index.get("completed") or "error" in index:
                raise ValueError("native_collection_incomplete")
            if any(row["summary"]["other"] for row in index["episodes"]):
                raise ValueError("native_collection_failed_fates")
            outcome["collection_index_sha256"] = digest(index_path)
            outcome["actual_offered_window_s"] = index["actual_offered_window_s"]
            if args.phase == "h16":
                labels = proof.verify_h16_run(binding, output / "collection")
                write(output / "original-q1-q5.json", labels)
                if not labels["ok"]:
                    raise ValueError("original_h16_q1_q5_refused")
                outcome["status"] = "h16_measured_q1_q5"
            else:
                outcome["status"] = "smoke_collected"
                outcome["scope"] = "Reused development smoke seed180001; not pristine data, H16 or efficacy"
    except BaseException as exc:
        outcome.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write(output / "outcome.json", outcome)
        print(json.dumps({"run": str(output), "status": outcome["status"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "h16"), required=True)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute required for an authorized native operation")
    def terminate(signum, frame):
        for interrupted in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(interrupted, signal.SIG_IGN)
        raise SystemExit(f"owned_runner_signal:{signum}")
    for watched in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(watched, terminate)
    execute(args)


if __name__ == "__main__":
    main()
