"""Default-OFF, Train-only engineering collection; qualification requires raw evidence audit."""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import native_services_smoke as services


PROBES = Path("results/mpc-native-r24/reviews/resume-p0-probes02/inputs")
ADAPTER_SHA256 = "935c2679c4b9fffdffb23ce88ca5351f91fa84e25c8c1edccedddaea4e4262a8"
DRAIN_FIELDS = ("current_in_flight", "current_waiting", "current_admitted_cost")


def error_record(error):
    return {"type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(error))}


def strict_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def strict_load(raw):
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def pairs(items):
        result = {}
        for key, value in items:
            services.checked(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return strict_copy(json.loads(raw, parse_constant=constant, object_pairs_hook=pairs))


def preserving(action, primary, report, stage):
    try:
        return action()
    except BaseException as error:
        report["caller_errors"].append({"stage": stage, "error": error_record(error)})
        if primary is None:
            raise
        primary.add_note(stage + ": " + type(error).__name__ + ": " + str(error))


def persist_capture(client, path, capture, key):
    try:
        return client.write_capture(path, capture)
    except BaseException as error:
        try:
            services.write(path.with_suffix(".write-error.json"), {
                "capture_unscorable": True, "type": type(error).__name__, "message": str(error),
            }, key)
        except BaseException as write_error:
            error.add_note("write-error receipt also failed: " + type(write_error).__name__
                           + ": " + str(write_error))
        raise


def remaining(owner, deadline, reserve):
    services.checked(owner.deadline == deadline and owner.reserve == reserve,
                     "original owner deadline/reserve changed")
    owner.check()
    value = deadline - reserve - time.monotonic()
    services.checked(math.isfinite(value) and value > 0, "original work deadline exhausted")
    return value


async def observe(client, path, *, owner, deadline, reserve, output, key, report, name):
    evidence = {"path": path, "entered_at": time.monotonic(), "status_code": None, "raw": None}
    primary = None
    try:
        timeout = min(2.0, remaining(owner, deadline, reserve))
        evidence["timeout_s"] = timeout
        async with asyncio.timeout(timeout):
            response = await client.get(path, headers={"x-admin-api-key": key}, timeout=timeout)
        evidence.update(status_code=response.status_code, raw=response.text)
        response.raise_for_status()
        value = strict_load(response.content)
        evidence["value"] = value
        return value
    except BaseException as error:
        primary = error
        evidence["error"] = error_record(error)
        raise
    finally:
        evidence["returned_at"] = time.monotonic()
        report.setdefault("caller_observations", []).append(evidence)
        preserving(lambda: services.write(output / (name + ".json"), evidence, key),
                   primary, report, name + "_persistence")


async def drain(client, *, owner, deadline, reserve, output, key, report, phase):
    attempt = 0
    while True:
        attempt += 1
        metrics = await observe(
            client, "/metrics", owner=owner, deadline=deadline, reserve=reserve,
            output=output, key=key, report=report, name=f"{phase}-metrics-{attempt:04d}")
        values = [metrics[name] for name in (*DRAIN_FIELDS, "tokenizer_fallback_requests")]
        services.checked(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                             for value in values), "invalid drain/fallback metric")
        services.checked(metrics["tokenizer_fallback_requests"] == 0, "tokenizer fallback after traffic")
        if (metrics["current_in_flight"] == 0 and metrics["current_waiting"] == 0
                and metrics["current_admitted_cost"] <= 1e-9):
            report.setdefault("caller_drains", {})[phase] = {
                "drained": True, "observed_at": time.monotonic(), "readings": attempt}
            return
        await asyncio.sleep(min(0.05, remaining(owner, deadline, reserve)))


def capture_of(prepared):
    return getattr(getattr(prepared, "recorder", None), "capture", None)


def counts(capture, entered):
    if not isinstance(capture, dict):
        return {"records_allocated": None if entered else 0, "sent": None if entered else 0}
    records = capture.get("records")
    if records is None:
        return {"records_allocated": 0, "sent": 0}
    services.checked(isinstance(records, list), "capture records must be a list or null")
    known = all(isinstance(row, dict) and "sent_at" in row for row in records)
    return {"records_allocated": len(records),
            "sent": sum(row["sent_at"] is not None for row in records) if known else None}


def source_binding(root, episode, expected):
    protocol_path = root / services.RELATIVE / "nominal_protocol.json"
    raw = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(raw).hexdigest()
    services.checked(expected["source_pins"].get(str(protocol_path)) == protocol_sha,
                     "protocol changed since startup source pins")
    caller_path = root / services.RELATIVE / "native_nominal_qualification.py"
    adapter_path = root / services.RELATIVE / "nominal_episode.py"
    services.checked(Path(__file__).resolve() == caller_path and Path(episode.__file__).resolve() == adapter_path,
                     "caller/adapter source-root mismatch")
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (caller_path, adapter_path)}
    services.checked(pins[str(adapter_path)] == ADAPTER_SHA256, "accepted episode adapter changed")
    pins[str(protocol_path)] = protocol_sha
    return strict_load(raw), {"source_pins": pins, "caller_sha256": pins[str(caller_path)],
                             "adapter_sha256": pins[str(adapter_path)], "protocol_sha256": protocol_sha}


def recheck(binding):
    services.checked(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
                         for path, sha in binding["source_pins"].items()),
                     "caller/adapter/protocol changed during collection")


async def collect(*, owner, root, output, key, report):
    """Caller-owned accounting; collected engineering evidence is never Native qualification."""
    root, output = Path(root).resolve(), Path(output)
    deadline, reserve = owner.deadline, owner.reserve
    primary, primary_tb = None, None
    prepared = warmup = client = recorder = bound = None
    entered_run = False
    stage = "source_preflight"
    report.update(caller_errors=[], episode_planned_requests=5, episode_trace_requests=None,
                  warmup_planned_requests=3, warmup_offered_requests=0,
                  offered_requests_from_driver=0, episode_sent_requests=0)

    def failure(where, error):
        nonlocal primary, primary_tb
        report["caller_errors"].append({"stage": where, "error": error_record(error)})
        if primary is None:
            primary, primary_tb = error, error.__traceback__
        elif error is not primary:
            primary.add_note(where + ": " + type(error).__name__ + ": " + str(error))

    def attempt(where, action):
        try:
            return action()
        except BaseException as error:
            failure(where, error)

    try:
        import httpx
        import nominal_client as recorder
        import nominal_episode as episode
        import nominal_runtime as runtime_api
        from mpc_lab.contracts import CostSpec, ModelParameters
        from nominal_permit import NominalDemoPermit
        from nominal_planner import NominalPlanner

        services.checked(all(type(value) in (int, float) and math.isfinite(value)
                             for value in (deadline, reserve)) and reserve >= 0,
                         "finite original deadline/reserve required")
        expected, config = report["source"], report["configuration"]
        remaining(owner, deadline, reserve)
        protocol, bound = source_binding(root, episode, expected)
        services.write(output / "qualification-source-binding.json", bound, key)
        report["caller_source_binding"] = bound
        gateway = f"http://127.0.0.1:{services.GATEWAY_PORT}"
        backend = f"http://127.0.0.1:{services.BACKEND_PORT}"
        stage = "model_binding"
        ownership_path = output / "services/ownership.json"
        ownership = strict_load(ownership_path.read_bytes())
        actual_engine = next(s["argv"] for s in ownership["services"] if s["name"] == "engine")
        services.checked(actual_engine == config["engine"], "owned backend argv mismatch")
        binding_path = output / "model-binding.json"
        services.write(binding_path, {
            "schema": recorder.BINDING_SCHEMA, "served_alias": services.ALIAS,
            "local_model_path": services.MODEL, "tokenizer_sha256": services.TOKENIZER,
            "served_model_names": [services.ALIAS], "backend_argv": actual_engine,
            "backend_argv_source": str(ownership_path) + " (owned actual spawn receipt)",
        })
        client = httpx.AsyncClient(base_url=gateway, trust_env=False, follow_redirects=False)
        warm_error = None
        try:
            stage = "warmup_prepare"
            warmup = recorder.prepare_capture(
                trace_path=root / PROBES / "warmup.json", binding_path=binding_path,
                arm="fixed32", purpose="engineering", gateway_url=gateway, backend_url=backend)
            services.checked(warmup.ready, "warmup local preflight failed: " + str(warmup.capture.get("error")))
            report["caller_clock_domains"] = {
                "SDK_source": expected["clock_domain"], "warmup_recorder": warmup.capture["clock_domain"]}
            remaining(owner, deadline, reserve)
            recheck(bound)
            stage = "warmup_replay"
            await recorder.capture_replay(
                warmup, offered_start=time.monotonic() + 1.0, deadline=deadline,
                cleanup_reserve_s=reserve, clock_domain=warmup.capture["clock_domain"])
        except BaseException as error:
            warm_error = error
            raise
        finally:
            capture = getattr(warmup, "capture", None)
            if capture is not None:
                report["warmup_capture_sha256"] = preserving(
                    lambda: persist_capture(recorder, output / "warmup-capture.json", capture, key),
                    warm_error, report, "warmup_capture_persistence")
        stage = "warmup_validation"
        services.checked(
            warmup.capture["status"] == "collected" and len(warmup.capture["records"]) == 3
            and all(r["fate"] == "complete" and r["stream_done"]
                    and r["finish_reason"] in ("stop", "length") for r in warmup.capture["records"])
            and warmup.capture["summary"]["prompt_tokens_mismatch"] == 0,
            "warmup did not complete with valid outcomes")
        remaining(owner, deadline, reserve)
        stage = "episode_prepare"
        prepared = episode.prepare_episode(
            output_dir=output / "episode", binding_path=binding_path, gateway_url=gateway,
            backend_url=backend, policy="mpc", trace_path=root / PROBES / "admission-boundary.json")
        prepared_error = getattr(prepared, "error", None)
        services.checked(prepared_error is None,
                         "episode local preparation failed: " + str(prepared_error))
        handle = getattr(prepared, "recorder", None)
        services.checked(handle is not None and isinstance(handle.trace, dict),
                         "episode preparation has no recorder/trace")
        report["episode_trace_requests"] = len(handle.trace["requests"])
        services.checked(report["episode_trace_requests"] == 5, "immutable five-request probe required")
        report["caller_clock_domains"]["episode_recorder"] = handle.capture["clock_domain"]
        parameters = ModelParameters(cost_spec=CostSpec(**expected["cost_spec"]))
        planner = NominalPlanner(parameters)
        settings = protocol["controller"]
        limits = runtime_api.RuntimeLimits(
            iterations=settings["iterations"], episode_s=settings["episode_s"],
            decision_s=settings["entry_to_action_deadline_s_max"],
            solve_s=settings["solve_budget_s"], poll_s=settings["poll_s"])
        stage = "pre_episode_drain"
        await drain(client, owner=owner, deadline=deadline, reserve=reserve,
                    output=output, key=key, report=report, phase="pre-episode")
        stage = "pre_control_state"
        state = await observe(
            client, "/admin/mpc/state", owner=owner, deadline=deadline, reserve=reserve,
            output=output, key=key, report=report, name="pre-control-state-observation")
        services.checked(state["identity"] == expected["identity"]
                         and state["clock_domain"] == expected["clock_domain"]
                         and state["cost_spec"] == expected["cost_spec"],
                         "post-warmup source/clock/cost mismatch")
        services.write(output / "pre-control-state.json", state, key)
        recheck(bound)
        stage = "runtime_binding"
        entered = time.monotonic()
        start = entered + 1.0
        window = handle.trace["duration_seconds"]
        services.checked(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                             for value in (window, limits.episode_s)), "invalid episode/window duration")
        expires = entered + 20.0
        services.checked(start + max(limits.episode_s, window) <= min(deadline - reserve, expires),
                         "episode/window exceeds original deadline or permit expiry")
        journal = output / "runtime-journal.json"
        described = runtime_api.describe_protocol(
            planner, journal, endpoint=gateway, scenario_binding=prepared.scenario_binding,
            policy="mpc", limits=limits, start_at=start)
        permit = NominalDemoPermit(
            identity=planner.identity, run_id=state["run_id"],
            clock_domain=expected["clock_domain"], source_sha256=described["source_sha256"],
            protocol_sha256=described["protocol_sha256"],
            not_before=entered, expires_at=expires, max_commands=8, approved=True)
        services.write(output / "runtime-binding.json", {
            "description": described, "offered_start": start, "caller_source_binding": bound,
            "caller_sha256": bound["caller_sha256"], "adapter_sha256": bound["adapter_sha256"],
            "permit_not_before": entered, "permit_expires_at": expires,
            "permit_max_commands": 8, "authority": "separate Mayor exact-source preregistration",
        }, key)
        runtime = runtime_api.NominalRuntime(
            client, planner, prepared.scenario_factory, journal,
            admin_key=key, scenario_binding=prepared.scenario_binding, permit=permit,
            enabled=True, policy="mpc", limits=limits, parameters=parameters, start_at=start)
        services.checked(runtime.protocol_sha256 == described["protocol_sha256"], "runtime protocol binding changed")
        recheck(bound)
        remaining(owner, deadline, reserve)
        stage = "episode_run"
        entered_run = True
        result = await episode.run_episode(
            prepared, runtime, owner=owner, offered_start=start, enabled=True)
        stage = "episode_result_validation"
        try:
            report["engineering_episode"] = strict_copy(result)
        except (ValueError, TypeError) as error:
            preserving(lambda: services.write(output / "episode-invalid-result.json", {
                "error": error_record(error), "unscorable_result_repr": repr(result)}, key),
                error, report, "invalid_result_persistence")
            raise
        services.checked(isinstance(result, dict) and result.get("status") == "collected"
                         and not result["failures"] and result["all_workers_joined"]
                         and not result["persistence_errors"] and not result["missing_record_ids"],
                         "engineering episode failed or incomplete; see raw episode evidence")
        services.checked(handle.capture["status"] == "collected" and len(handle.capture["records"]) == 5,
                         "episode raw capture not collected or incomplete")
        remaining(owner, deadline, reserve)
        stage = "completed"
    except BaseException as error:
        failure(stage, error)
    finally:
        if client is not None:
            try:
                await drain(client, owner=owner, deadline=deadline, reserve=reserve,
                            output=output, key=key, report=report, phase="final")
            except BaseException as error:
                failure("final_drain", error)
            try:
                await client.aclose()
            except BaseException as error:
                failure("client_close", error)
        if bound is not None:
            attempt("final_source_check", lambda: recheck(bound))
        capture = capture_of(prepared)
        allocated = attempt("episode_counts", lambda: counts(capture, entered_run))
        warm_counts = attempt("warmup_counts", lambda: counts(getattr(warmup, "capture", None), False))
        allocated = allocated or {"records_allocated": None, "sent": None}
        report["offered_requests_from_driver"] = allocated["records_allocated"]
        report["episode_sent_requests"] = allocated["sent"]
        report["warmup_offered_requests"] = warm_counts["records_allocated"] if warm_counts else None
        report["warmup_sent_requests"] = warm_counts["sent"] if warm_counts else None
        path = output / ("caller-episode-capture.json" if entered_run else "episode-not-run-capture.json")
        if capture is not None and recorder is not None:
            report["caller_capture_sha256"] = attempt(
                "episode_capture_persistence", lambda: persist_capture(recorder, path, capture, key))
        else:
            attempt("absent_capture_persistence", lambda: services.write(path, {
                "schema": "qualification-caller-absent-capture-v1", "raw_v2_capture_available": False,
                "status": "NOT_RUN" if not entered_run else "UNKNOWN", **allocated,
            }, key))
        if warmup is None:
            attempt("absent_warmup_persistence", lambda: services.write(output / "warmup-not-run.json", {
                "status": "NOT_RUN", "capture_available": False, "records_allocated": 0, "sent": 0}, key))
        prepared_error = getattr(prepared, "error", None)
        try:
            prepared_error = strict_copy(prepared_error)
        except (ValueError, TypeError) as error:
            failure("prepared_error_serialization", error)
            prepared_error = {"unserializable_error_repr": repr(prepared_error)}
        receipt = {
            "schema": "qualification-caller-terminal-v1", "stage": stage, "entered_run": entered_run,
            "status": "NOT_RUN" if not entered_run else ("FAILED" if primary is not None else "COLLECTED"),
            "prepared_error": prepared_error, "primary_error": error_record(primary) if primary else None,
            "errors": report["caller_errors"], "records_allocated": allocated["records_allocated"],
            "sent": allocated["sent"], "episode_planned_requests": 5,
            "episode_trace_requests": report["episode_trace_requests"],
            "capture_status": capture.get("status") if isinstance(capture, dict) else None,
            "capture_error": capture.get("error") if isinstance(capture, dict) else None,
            "deadline": deadline if type(deadline) in (int, float) and math.isfinite(deadline) else None,
            "cleanup_reserve_s": reserve if type(reserve) in (int, float) and math.isfinite(reserve) else None,
            "native_qualification": False, "full_wall": None,
        }
        report["qualification_caller"] = receipt
        if entered_run and primary is not None and not isinstance(primary, Exception):
            receipt["status"] = "INTERRUPTED"
        if not entered_run:
            attempt("not_run_receipt_persistence",
                    lambda: services.write(output / "episode-not-run.json", receipt, key))
        receipt["primary_error"] = error_record(primary) if primary else None
        attempt("caller_terminal_persistence",
                lambda: services.write(output / "qualification-caller.json", receipt, key))
        if primary is not None:
            receipt["primary_error"] = error_record(primary)
            if entered_run:
                receipt["status"] = "FAILED" if isinstance(primary, Exception) else "INTERRUPTED"
    if primary is not None:
        primary.qualification_caller_report = report["qualification_caller"]
        raise primary.with_traceback(primary_tb)


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
