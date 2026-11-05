"""Cold owned capture, optionally followed by fixed-cap source-time metrology."""

import argparse
import asyncio
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import signal
import sys
import time

from capture_owner_closure import establish_initial_capture_closure
from clocked_owner_cut import collect_clocked_owner_cut
from run_h16 import ROOT, digest, write


async def run(args):
    import httpx
    from mpc_lab.runtime import local_clock_domain

    sys.path.insert(0, str(ROOT / "experiments/mpc-research/native-v2"))
    from collect_batch_qualification import CaptureClient

    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if "owner_bootstrap_extension" not in plan:
        raise ValueError("prospective_owner_bootstrap_required")
    args.output.mkdir(mode=0o700)
    parent = args.output.parent
    native_startup = json.loads((parent / "engine/ready.json").read_text())
    gateway_startup = json.loads((parent / "gateway/ready.json").read_text())
    domain = local_clock_domain()
    trial = plan["trials"][0]["trial_id"]
    sequence = itertools.count(1)
    started = time.monotonic()
    public_memory = {}
    shadow_result = None
    trace = None
    if "owner_source_shadow_extension" in plan:
        from trace_workload import load_trace, verify_prompts
        from transformers import AutoTokenizer
        from owner_source_shadow import shadow_population

        if args.trace is None or args.protocol is None or digest(args.trace) != plan["trace_sha256"]:
            raise ValueError("source_shadow_requires_bound_trace_and_protocol")
        trace = load_trace(args.trace)
        shadow_population(plan, trace)
        verify_prompts(trace, AutoTokenizer.from_pretrained(
            config["MODEL_NAME"], local_files_only=True))

    def record(role, sent, received, response):
        write(args.output / f"reply-{next(sequence):04d}-{role}.json", {
            "sent_at": sent, "received_at": received,
            "status": response.status_code, "body": response.text,
        })

    class SyncClient:
        def __init__(self, client, role):
            self.client, self.role = client, role

        def get(self, path):
            sent = time.monotonic()
            response = self.client.get(path)
            received = time.monotonic()
            record(self.role, sent, received, response)
            response.raise_for_status()
            return response

    native_headers = {"x-observer-key": config["ADMIN_API_KEY"]}
    gateway_headers = {"x-admin-api-key": config["ADMIN_API_KEY"]}
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=5, trust_env=False,
                      headers=native_headers) as native_sync, httpx.Client(
            base_url="http://127.0.0.1:8080", timeout=5, trust_env=False,
            headers=gateway_headers) as gateway_sync:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8000", timeout=5, trust_env=False,
            headers=native_headers,
        ) as native_async, httpx.AsyncClient(
            base_url="http://127.0.0.1:8080", timeout=5, trust_env=False,
            headers=gateway_headers,
        ) as gateway_async:
            capture = CaptureClient(native_async, args.output, plan, domain)
            metadata = await capture.call("metadata")
            await capture.call("begin", trial_id=trial)
            native_client = SyncClient(native_sync, "native-phase")
            first = await asyncio.to_thread(native_client.get, "/pr33-native/phase-state")
            process = first.json()["native_phase"]["process_identity"]

            class BarrierClient:
                async def post(self, path, **kwargs):
                    sent = time.monotonic()
                    response = await gateway_async.post(path, **kwargs)
                    record("flush-barrier", sent, time.monotonic(), response)
                    return response

            sources = {
                "source_sha256": plan["phase_frontier_extension"]["source_sha256"],
                "input_post_source_sha256": plan["input_post_extension"]["source_sha256"],
                "queue_observer_sha256": plan["queue_observer_extension"]["source_sha256"],
                "core_source_sha256": plan["input_post_extension"]["vllm_core_sha256"],
            }
            def cut_options():
                return dict(
                    native_acquire=lambda: native_client.get("/pr33-native/phase-state").json(),
                    native_startup=native_startup, native_process=process, native_sources=sources,
                    expected_capture_id=trial, gateway_sync=SyncClient(gateway_sync, "gateway"),
                    gateway_async=BarrierClient(), gateway_startup=gateway_startup,
                    telemetry_path=parent / "gateway/gateway-telemetry.jsonl",
                    expected_run_id=gateway_startup["readiness"]["gateway_run_id"],
                    expected_clock_domain=domain,
                    expected_identity=gateway_startup["readiness"]["snapshot"]["identity"],
                    initial_budget=gateway_startup["readiness"]["snapshot"]["budget"],
                    public_memory=public_memory, max_age_s=0.5)

            async def acquire(**options):
                return await collect_clocked_owner_cut(**cut_options(), **options)

            cut = await acquire(bootstrap=True)
            closure = establish_initial_capture_closure(
                cut, native_startup=native_startup, gateway_startup=gateway_startup)
            write(args.output / "initial-clocked-cut.json", cut)
            write(args.output / "initial-capture-closure.json", asdict(closure))
            if trace is not None:
                from mpc_lab.contracts import CostSpec, ModelParameters
                from owner_source_shadow import collect_source_shadow

                protocol = json.loads(args.protocol.read_text())
                parameters = ModelParameters(
                    cost_spec=CostSpec(**protocol["cost_spec"]),
                    model_version="native-id-unfitted-v1", calibration_version="unfitted-not-validated")
                native_shadow_factory = None
                if "native_mpc_shadow_extension" in plan:
                    from native_shadow_monitor import NativeShadowMonitor, SOURCE_FILES
                    from prefill_lab.contracts import digest as canonical_digest

                    extension = plan["native_mpc_shadow_extension"]
                    bundle = extension["model_bundle"]
                    if (extension["schema"] != "r24-fixed-cap-native-mpc-shadow-extension-v1"
                            or extension["native_mpc_permitted"] is not False
                            or extension["deployable"] is not False
                            or set(extension["source_sha256"]) != set(SOURCE_FILES)
                            or canonical_digest(bundle) != extension["model_bundle_canonical_sha256"]
                            or any(digest(Path(__file__).with_name(name)) != fingerprint
                                   for name, fingerprint in extension["source_sha256"].items())):
                        raise ValueError("native_shadow_plan_source_or_model_binding")

                    def native_shadow_factory(public_history):
                        return NativeShadowMonitor(
                            output=args.output, bundle=bundle, trace=trace,
                            runtime=metadata["runtime"], cut_options=cut_options(),
                            closure=closure, public_history=public_history,
                            cost_spec=parameters.cost_spec)

                async with httpx.AsyncClient(
                    base_url="http://127.0.0.1:8080", timeout=50, trust_env=False,
                    limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
                ) as traffic:
                    shadow_result = await collect_source_shadow(
                        output=args.output, plan=plan, trace=trace, parameters=parameters,
                        admin=gateway_async, traffic=traffic, domain=domain, acquire=acquire,
                        closure=closure, native_startup=native_startup,
                        public_memory=public_memory, native_shadow_factory=native_shadow_factory)
                await capture.finish(trial)
            else:
                await capture.call("end")
    write(args.output / "result.json", {
        "schema": ("r24-native-owner-source-shadow-result-v1" if shadow_result
                   else "r24-native-owner-bootstrap-result-v1"), "valid": True,
        "original_serving_commit": plan["source_commit"], "collector_source_sha256": digest(__file__),
        "plan_sha256": digest(args.plan), "started_at": started, "finished_at": time.monotonic(),
        "initial_cut_sha256": digest(args.output / "initial-clocked-cut.json"),
        "closure_sha256": digest(args.output / "initial-capture-closure.json"),
        "offered": 0, "policy_commands": 0, "native_mpc_permitted": False,
        "scope": "One empty native/Gateway clock, flush and cold-capture closure. Warmup and readiness CAS only; no tape replay, short-task calibration, prediction, action response or MPC.",
        **(shadow_result or {}),
    })


async def run_with_signals(args):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    watched = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    for signum in watched:
        loop.add_signal_handler(signum, task.cancel)
    try:
        await run(args)
    finally:
        for signum in watched:
            loop.remove_signal_handler(signum)


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--protocol", type=Path)
    asyncio.run(run_with_signals(parser.parse_args()))


if __name__ == "__main__":
    main()
