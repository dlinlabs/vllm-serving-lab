"""Owned child: unchanged Source13 services, bounded read-only readiness, sealed client."""

import argparse
import asyncio
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_common as c


class ReadinessRefused(ValueError):
    pass


def ready_require(condition, reason):
    if not condition:
        raise ReadinessRefused(reason)


def queue_observation(record, client, *, epoch=None):
    ready_require(record["method"] == "GET" and record["path"] == "/pr33-native/queue-state"
                  and record["http_status"] == 200, "full_queue_readonly_HTTP_required")
    frame = client.body(record)
    base, state = frame["base"], frame["scheduler_state"]
    expected = c.load(client.HERE / "INPUTS.json")["product_sha256"][
        "experiments/mpc-research/native_observer/pr33_native_observer.py"]
    ready_require(frame["schema"] == "r23-native-queue-read-only-v1"
                  and frame["queue_observer_sha256"] == c.sha(c.HERE / "inputs/source13/queue_observer.py")
                  and base["schema"] == "pr33-native-read-only-v1"
                  and base["observer_sha256"] == expected, "full_queue_source_binding")
    ready_require(frame["complete"] is True and base["complete"] is True
                  and frame["refusal_reasons"] == [], "full_queue_incomplete")
    ready_require(base["clock_domain"] == record["client_clock_domain"]
                  and isinstance(base["run_id"], str) and bool(base["run_id"])
                  and (epoch is None or base["run_id"] == epoch), "full_queue_clock_or_epoch")
    chain = [record["client_send_at"], frame["sample_started_at"], base["sampled_at"],
             frame["sample_finished_at"], base["frontend_sampled_at"], record["client_receive_at"]]
    ready_require(all(client.number(value) for value in chain)
                  and all(left <= right for left, right in zip(chain, chain[1:]))
                  and chain[-1] - chain[1] <= 0.5, "full_queue_source_receipt_interval")
    running, waiting, requests = state["running"], state["waiting"], state["requests"]
    queue, ghosts, skipped = frame["queue_oldest_first"], frame["queued_ghost_request_ids"], frame["skipped_waiting"]
    streaming = frame["streaming_waiting"]
    ready_require(all(isinstance(value, list) for value in (running, waiting, queue, ghosts, skipped))
                  and isinstance(requests, dict) and type(streaming) is int and streaming >= 0,
                  "full_queue_inventory_types")
    represented = running + waiting + skipped
    ready_require(all(type(base[key]) is int and base[key] >= 0 for key in
                      ("running", "waiting", "pending_batches", "frontend_unfinished"))
                  and len(represented) == len(set(represented)) and set(represented) == set(requests)
                  and base["running"] == len(running)
                  and base["waiting"] == len(waiting) + len(skipped) + streaming
                  and base["pending_batches"] == len(queue)
                  and type(base["frontend_unfinished"]) is int and base["frontend_unfinished"] >= 0,
                  "full_queue_inventory_conservation")
    actual_ghosts = sorted({work["request_id"] for row in queue for work in row["work"]} - set(requests))
    ready_require(ghosts == actual_ghosts, "full_queue_ghost_inventory")
    empty = (not represented and not requests and not queue and not ghosts
             and not skipped and streaming == 0 and base["frontend_unfinished"] == 0)
    return frame, empty


async def readiness(process, client, config, deadline, *, native, identity=None,
                    transport=None):
    import httpx
    directory = c.PUBLIC / ("startup-native" if native else "startup-gateway")
    directory.mkdir(mode=0o700)
    settings = dict(client.plan(), maximum_http_exchanges=256)
    capture = client.Capture(directory, settings, config["ADMIN_API_KEY"], client.clock_domain())
    origin = settings["native_origin"] if native else settings["gateway_origin"]
    async with httpx.AsyncClient(base_url=origin, trust_env=False, transport=transport) as http:
        async with asyncio.timeout(max(0, deadline - time.monotonic())):
            while True:
                ready_require(process.poll() is None, "owned_service_exited_during_readiness")
                row = await capture.exchange(http, "GET", "/health" if native else "/admin/mpc/state",
                                             role="read-only-ready", auth=not native)
                ready_require(not row["credential_removed"], "startup_auth_echo_removed_refusal")
                if row["http_status"] == 200 and row["body_complete"]:
                    if not native:
                        state = client.snapshot(row)
                        ready_require(state.budget == 32 and not state.requests,
                                  "gateway_initial_32_and_empty_required")
                        return dict(budget=state.budget, budget_version=state.budget_version,
                                    sampled_at=state.sampled_at, clock_domain=state.clock_domain,
                                    gateway_commands=0)
                    break
                await asyncio.sleep(1)
            models = await capture.exchange(http, "GET", "/v1/models", role="served-model")
            ready_require(settings["model"] in {row["id"] for row in client.body(models)["data"]},
                      "served_model_identity")
            payload = dict(model=settings["model"], messages=[
                dict(role="user", content="Count from one to five.")], temperature=0, max_tokens=16)
            c.write(directory / "warmup-offered.json", dict(
                schema="fixed-control-startup-offer-v1", send_attempted=True,
                client_send_intent_at=time.monotonic(), clock_domain=client.clock_domain(),
                request=payload, accepted_by_server=None, distinct_from_client_eight=True))
            capture.settings = dict(settings, http_timeout_s=60)
            warmup = await capture.exchange(http, "POST", "/v1/chat/completions",
                                            role="single-startup-warmup", payload=payload)
            c.write(directory / "warmup-fate.json", dict(
                record=warmup["sequence"], http_status=warmup["http_status"],
                error_type=warmup["error_type"], body_complete=warmup["body_complete"],
                client_send_at=warmup["client_send_at"], client_receive_at=warmup["client_receive_at"]))
            ready_require(warmup["http_status"] == 200, "startup_warmup_refused_no_retry")
            reply = client.body(warmup)
            ready_require(type(reply.get("usage", {}).get("completion_tokens")) is int
                      and 0 <= reply["usage"]["completion_tokens"] <= 16,
                      "startup_warmup_completion_cap")
            capture.settings = settings
            frames, epoch = [], None
            drain_deadline = min(deadline, time.monotonic() + 30)
            while time.monotonic() < drain_deadline:
                ready_require(process.poll() is None, "owned_native_exited_during_drain")
                row = await capture.exchange(http, "GET", "/pr33-native/queue-state",
                                             role="startup-full-queue", auth=True)
                frame, empty = queue_observation(row, client, epoch=epoch)
                ready_require(frame["base"]["clock_domain"] == identity["clock_domain"], "startup_clock_changed")
                epoch = frame["base"]["run_id"]
                frames = frames + [frame] if empty else []
                if len(frames) == 2:
                    return dict(frames=frames, startup_offers=1, max_tokens=16,
                                scope="Full scheduler/issued queue observed empty at two sampling intervals; not persistent hardware quiescence.")
                await asyncio.sleep(0.1)
            raise ReadinessRefused("native_full_queue_empty_not_established")


def run_stack(client, config, identity, deadline, owned_engine, protocol_path, env):
    common = ["--config", str(c.STATE / "control.private.json"),
              "--plan", str(c.HERE / "startup-instrumentation.json"), "--python", str(c.NATIVE)]
    engine, gateway = c.PUBLIC / "engine", c.PUBLIC / "gateway"
    native_command = c.guarded([c.NATIVE, "-B", c.SOURCE13 / "launch_queue_service.py",
                               *common, "--evidence", engine])
    gateway_command = c.guarded([
        c.NATIVE, "-B", c.ROOT / "experiments/mpc-research/native-v2/launch_batch_service.py",
        "gateway", *common, "--evidence", gateway, "--identification-protocol", protocol_path])
    with owned_engine(native_command, evidence=engine, env=env, cwd=c.ROOT, grace_s=2,
            ready=lambda process: asyncio.run(readiness(
                process, client, config, deadline, native=True, identity=identity))):
        with owned_engine(gateway_command, evidence=gateway, env=env, cwd=c.ROOT, grace_s=2,
                ready=lambda process: asyncio.run(readiness(
                    process, client, config, deadline, native=False))):
            c.require(time.monotonic() < deadline, "startup_deadline_before_attach")
            c.write(c.PUBLIC / "attach-start.json", dict(
                started_at=time.monotonic(), clock_domain=client.clock_domain(),
                client_run_id="launch02", active_s=20, readonly_s=3))
            try:
                return asyncio.run(client.live(SimpleNamespace(
                    mayor_execute=True, source_root=c.ROOT,
                    private_config=c.STATE / "control.private.json",
                    prelaunch_receipt=c.STATE / "tmpdir-preflight.json",
                    output=c.CLIENT_OUTPUT, run_id="launch02")))
            finally:
                c.write(c.PUBLIC / "attach-finished.json", dict(
                    finished_at=time.monotonic(), clock_domain=client.clock_domain()))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--owned", action="store_true", required=True)
    parser.add_argument("--startup-deadline", type=float, required=True)
    args = parser.parse_args()
    c.require_remote()
    c.require(sys.pycache_prefix and (c.STATE / "control.private.json").is_file(),
              "supervisor_owned_private_state_required")
    c.verify_runnable()
    client = c.client_library()
    client.verify_tmpdir_receipt(c.load(c.STATE / "tmpdir-preflight.json"))
    c.write(c.PUBLIC / "native-child-identity.json", dict(
        pid=os.getpid(), start_ticks=c.owners().snapshot(os.getpid())[0]))
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    outcome = dict(status="starting", native_mpc_permitted=False, deployable=False,
                   client_returncode=None, installed_sources_unchanged=False)
    identity = None
    try:
        sys.path[:0] = [str(c.SOURCE13), str(c.ROOT), str(c.ROOT / "experiments/mpc-research"),
                       str(c.ROOT / "experiments/mpc-research/native-v2"),
                       str(c.ROOT / "experiments/mpc-research/native_observer")]
        from engine_handoff import owned_engine, require_free_ports
        from run_h16 import runtime_identity
        from run_batch_qualification import verify_original_sources
        from batch_capture_bridge import validate_plan
        from prepared_copy_ready_plugin import validate_plan as validate_copy_plan
        from mpc_local.native_pilot import H16_PROTOCOL_SHA
        from prefill_lab.plugin import installed_sources
        plan = c.load(c.HERE / "startup-instrumentation.json")
        validate_plan(plan)
        validate_copy_plan(plan)
        require_free_ports()
        c.require(c.ALIAS.is_symlink() and c.ALIAS.resolve() == c.MODEL,
                  "preexisting_exact_alias_no_creation_or_download")
        c.require(c.sha(c.PROTOCOL) == H16_PROTOCOL_SHA, "unchanged_native_protocol")
        outcome["original_source_binding"] = verify_original_sources(
            c.ROOT, c.ROOT.parent / "deployment-source.tar.gz")
        identity = runtime_identity()
        c.write(c.PUBLIC / "runtime-identity.json", identity)
        config = c.load(c.STATE / "control.private.json")
        c.require(config["MODEL_NAME"] == str(c.ALIAS), "exact_client_model_alias")
        env = dict(os.environ, MPC_OWNER_LEDGER_ENABLED="true")
        c.require(env["TMPDIR"] == str(c.STATE / "t"), "same_checked_TMPDIR_required")
        env["PYTHONPATH"] = os.pathsep.join((str(c.ROOT), str(c.ROOT / "experiments/mpc-research"),
                                            str(c.ROOT / "experiments/mpc-research/native_observer")))
        outcome["client_returncode"] = run_stack(
            client, config, identity, args.startup_deadline, owned_engine, c.PROTOCOL, env)
        outcome["status"] = "fixed_client_finished"
    except BaseException as error:
        outcome.update(status="component_refused_raw_preserved", error_type=type(error).__name__)
        if isinstance(error, (ReadinessRefused, client.Refused)):
            outcome["readiness_or_client_refusal_reason"] = str(error)
    finally:
        outcome.update(postflight_status="in_progress", postflight_started_at=time.monotonic())
        try:
            if identity is not None:
                outcome["installed_sources_unchanged"] = identity["installed_vllm_sources"] == installed_sources()
            outcome["source_after"] = c.clock_helpers().source_binding(c.STATE / "python-cache-evidence")
            c.verify_runnable()
            outcome["candidate_sources_unchanged"] = True
            outcome["postflight_status"] = "completed"
        except BaseException as error:
            outcome["postflight_error_type"] = type(error).__name__
            outcome["postflight_status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        outcome["finished_at"] = time.monotonic()
        outcome["post_client_native_quiescence"] = "Only client02 full source-observed sampling scope; no persistent hardware quiescence"
        c.write(c.PUBLIC / "component-outcome.json", outcome)
    return 0 if (outcome["status"] == "fixed_client_finished" and outcome["client_returncode"] == 0
                 and outcome["installed_sources_unchanged"] and outcome.get("candidate_sources_unchanged")
                 and outcome["postflight_status"] == "completed"
                 and "error_type" not in outcome and "postflight_error_type" not in outcome) else 1


if __name__ == "__main__":
    raise SystemExit(main())
