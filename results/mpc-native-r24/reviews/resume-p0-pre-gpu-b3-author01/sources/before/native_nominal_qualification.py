"""Default-OFF, Train-only engineering collection; qualification requires raw evidence audit."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import native_services_smoke as services


PROBES = Path("results/mpc-native-r24/reviews/resume-p0-probes02/inputs")


def persist_capture(client, path, capture, key):
    try:
        return client.write_capture(path, capture)
    except BaseException as error:
        services.write(path.with_suffix(".write-error.json"), {
            "capture_unscorable": True, "type": type(error).__name__, "message": str(error),
        }, key)
        raise


async def collect(*, owner, root, output, key, report):
    import httpx
    import nominal_client as recorder
    import nominal_episode as episode
    import nominal_runtime as runtime_api
    from mpc_lab.contracts import CostSpec, ModelParameters
    from nominal_permit import NominalDemoPermit
    from nominal_planner import NominalPlanner

    expected, config = report["source"], report["configuration"]
    protocol = json.loads((root / services.RELATIVE / "nominal_protocol.json").read_text())
    gateway = f"http://127.0.0.1:{services.GATEWAY_PORT}"
    backend = f"http://127.0.0.1:{services.BACKEND_PORT}"
    owner.check()
    ownership_path = output / "services/ownership.json"
    ownership = json.loads(ownership_path.read_text())
    actual_engine = next(s["argv"] for s in ownership["services"] if s["name"] == "engine")
    services.checked(actual_engine == config["engine"], "owned backend argv mismatch")
    binding = {
        "schema": recorder.BINDING_SCHEMA, "served_alias": services.ALIAS,
        "local_model_path": services.MODEL, "tokenizer_sha256": services.TOKENIZER,
        "served_model_names": [services.ALIAS], "backend_argv": actual_engine,
        "backend_argv_source": str(ownership_path) + " (owned actual spawn receipt)",
    }
    binding_path = output / "model-binding.json"
    services.write(binding_path, binding)
    warmup = recorder.prepare_capture(
        trace_path=root / PROBES / "warmup.json", binding_path=binding_path,
        arm="fixed32", purpose="engineering", gateway_url=gateway, backend_url=backend,
    )
    try:
        services.checked(warmup.ready, "warmup local preflight failed")
        owner.check()
        await recorder.capture_replay(
            warmup, offered_start=time.monotonic() + 1.0, deadline=owner.deadline,
            cleanup_reserve_s=owner.reserve, clock_domain=warmup.capture["clock_domain"],
        )
    finally:
        report["warmup_capture_sha256"] = persist_capture(
            recorder, output / "warmup-capture.json", warmup.capture, key)
    services.checked(
        warmup.capture["status"] == "collected" and len(warmup.capture["records"]) == 3
        and all(r["fate"] == "complete" and r["stream_done"]
                and r["finish_reason"] in ("stop", "length") for r in warmup.capture["records"])
        and warmup.capture["summary"]["prompt_tokens_mismatch"] == 0,
        "warmup did not complete with valid outcomes",
    )
    owner.check()
    prepared = episode.prepare_episode(
        output_dir=output / "episode", binding_path=binding_path,
        gateway_url=gateway, backend_url=backend, policy="mpc",
        trace_path=root / PROBES / "admission-boundary.json",
    )
    report["offered_requests_from_driver"] = len(prepared.recorder.trace["requests"]) \
        if prepared.recorder.trace is not None else None
    report["warmup_offered_requests"] = 3
    services.checked(prepared.error is None, "episode local preparation failed")
    parameters = ModelParameters(cost_spec=CostSpec(**expected["cost_spec"]))
    planner = NominalPlanner(parameters)
    settings = protocol["controller"]
    limits = runtime_api.RuntimeLimits(
        iterations=settings["iterations"], episode_s=settings["episode_s"],
        decision_s=settings["entry_to_action_deadline_s_max"],
        solve_s=settings["solve_budget_s"], poll_s=settings["poll_s"],
    )
    async with httpx.AsyncClient(base_url=gateway, trust_env=False, follow_redirects=False) as client:
        owner.check()
        response = await client.get("/admin/mpc/state", headers={"x-admin-api-key": key},
                                    timeout=min(2.0, owner.deadline - owner.reserve - time.monotonic()))
        response.raise_for_status()
        state = response.json()
        services.checked(state["identity"] == expected["identity"]
                         and state["clock_domain"] == expected["clock_domain"]
                         and state["cost_spec"] == expected["cost_spec"],
                         "post-warmup source/clock/cost mismatch")
        services.write(output / "pre-control-state.json", state, key)
        entered = time.monotonic()
        start = entered + 1.0
        services.checked(start + limits.episode_s < owner.deadline - owner.reserve,
                         "insufficient original work deadline")
        journal = output / "runtime-journal.json"
        described = runtime_api.describe_protocol(
            planner, journal, endpoint=gateway, scenario_binding=prepared.scenario_binding,
            policy="mpc", limits=limits, start_at=start,
        )
        permit = NominalDemoPermit(
            identity=planner.identity, run_id=state["run_id"],
            clock_domain=expected["clock_domain"], source_sha256=described["source_sha256"],
            protocol_sha256=described["protocol_sha256"],
            not_before=entered, expires_at=entered + 20.0, max_commands=8, approved=True,
        )
        services.write(output / "runtime-binding.json", {
            "description": described, "offered_start": start,
            "permit_not_before": entered, "permit_expires_at": entered + 20.0,
            "permit_max_commands": 8, "authority": "separate Mayor exact-source preregistration",
        }, key)
        runtime = runtime_api.NominalRuntime(
            client, planner, prepared.scenario_factory, journal,
            admin_key=key, scenario_binding=prepared.scenario_binding, permit=permit,
            enabled=True, policy="mpc", limits=limits, parameters=parameters, start_at=start,
        )
        result = await episode.run_episode(
            prepared, runtime, owner=owner, offered_start=start, enabled=True)
        report["engineering_episode"] = result
        services.checked(not result["failures"] and result["all_workers_joined"]
                         and not result["persistence_errors"] and not result["missing_record_ids"],
                         "engineering episode failed or incomplete; see raw episode evidence")
    owner.check()


def body(**context):
    asyncio.run(collect(**context))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seconds", type=float, default=480)
    parser.add_argument("--cleanup-reserve", type=float, default=30)
    parser.add_argument("--term-grace", type=float, default=10)
    args = parser.parse_args(argv)
    return services.run(args, body=body)


if __name__ == "__main__":
    raise SystemExit(main())
