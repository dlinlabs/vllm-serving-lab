"""One fixed 16/32 CAS/expiry probe; live mode is exclusively for the Mayor."""

import argparse
import asyncio
import base64
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

import httpx


HERE = Path(__file__).resolve().parent
PRODUCT = HERE / "inputs/product"
sys.path[:0] = [str(HERE), str(PRODUCT)]
from mpc_lab.contracts import SCHEMA, implementation_hash
from mpc_lab.runtime import require_guarded_capability, snapshot_from_dict

from freeze_inputs02 import verify as verify_inputs
from prelaunch01 import PrelaunchRefused, verify_receipt as verify_tmpdir_receipt


class Refused(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan():
    verify_inputs(originals=False)
    return json.loads((HERE / "EXPERIMENT.json").read_bytes())


def clock_domain():
    return ("linux-monotonic:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            + ":" + os.readlink("/proc/self/ns/time"))


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def persist(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def body(record):
    require(record.get("error_type") is None and record.get("body_complete") is True,
            "incomplete_HTTP_exchange")
    require(record.get("credential_removed") is False, "credential_echo_in_response")
    raw = base64.b64decode(record["response_body_base64"], validate=True)
    require(sha(raw) == record["response_body_sha256"], "response_body_hash")
    return json.loads(raw)


def snapshot(record, *, binding=None):
    require(record["http_status"] == 200 and record["method"] == "GET"
            and record["path"] == "/admin/mpc/state", "state_HTTP_contract")
    document = body(record)
    require_guarded_capability(document)
    result = snapshot_from_dict(document)
    require(result.complete, "incomplete_public_state")
    require(result.clock_domain == record["client_clock_domain"], "client_server_clock_domain")
    require(record["client_send_at"] <= result.sampled_at <= record["client_receive_at"],
            "state_outside_client_brackets")
    require(result.identity.implementation_hash == implementation_hash(), "endpoint_implementation_identity")
    if binding is not None:
        require((result.run_id, result.clock_domain, result.identity) ==
                (binding.run_id, binding.clock_domain, binding.identity), "endpoint_epoch_changed")
    return result


def native_state(record, settings, *, native_run=None):
    require(record["http_status"] == 200 and record["path"] == settings["native_state_path"],
            "native_context_HTTP")
    frame = body(record)
    pins = json.loads((HERE / "INPUTS.json").read_bytes())["product_sha256"]
    expected = pins["experiments/mpc-research/native_observer/pr33_native_observer.py"]
    require(frame.get("observer_sha256") == expected, "native_observer_source")
    require(frame.get("clock_domain") == record["client_clock_domain"], "native_clock_domain")
    require(frame.get("complete") is True and isinstance(frame.get("run_id"), str), "native_context_incomplete")
    require(number(frame.get("sampled_at"))
            and 0 <= record["client_receive_at"] - frame["sampled_at"] <= 0.5,
            "native_context_stale_or_future")
    if native_run is not None:
        require(frame["run_id"] == native_run, "native_epoch_changed")
    for key in ("running", "waiting", "pending_batches", "frontend_unfinished"):
        require(type(frame.get(key)) is int and frame[key] >= 0, "native_context_population")
    return frame


def full_queue_observation(record, settings, *, native_run=None, deadline=None):
    """Source-observed inventory only: never query a GPU future or infer its completion."""
    try:
        require(record["method"] == "GET" and record["path"] == settings["final_queue_path"]
                and record["http_status"] == 200, "full_queue_HTTP_contract")
        frame = body(record)
        base, state = frame["base"], frame["scheduler_state"]
        pins = json.loads((HERE / "INPUTS.json").read_bytes())
        require(frame["schema"] == "r23-native-queue-read-only-v1"
                and base["schema"] == "pr33-native-read-only-v1"
                and frame["queue_observer_sha256"] == pins["queue_observer_sha256"]
                and base["observer_sha256"] == pins["product_sha256"][
                    "experiments/mpc-research/native_observer/pr33_native_observer.py"],
                "full_queue_source_binding")
        require(frame["complete"] is True and base["complete"] is True
                and frame["refusal_reasons"] == [], "full_queue_incomplete_or_refused")
        require(base["clock_domain"] == record["client_clock_domain"]
                and isinstance(base["run_id"], str) and bool(base["run_id"])
                and (native_run is None or base["run_id"] == native_run), "full_queue_clock_or_epoch")
        chain = [record["client_send_at"], frame["sample_started_at"], base["sampled_at"],
                 frame["sample_finished_at"], base["frontend_sampled_at"], record["client_receive_at"]]
        require(all(number(value) for value in chain)
                and all(left <= right for left, right in zip(chain, chain[1:])),
                "full_queue_source_receipt_chronology")
        require(deadline is None or chain[-1] <= deadline, "full_queue_late_receipt")
        require(chain[-1] - chain[1] <= 0.5, "full_queue_stale_source")
        running, waiting, requests = state["running"], state["waiting"], state["requests"]
        queue, ghosts, skipped = frame["queue_oldest_first"], frame["queued_ghost_request_ids"], frame["skipped_waiting"]
        streaming = frame["streaming_waiting"]
        require(all(isinstance(value, list) for value in (running, waiting, queue, ghosts, skipped))
                and isinstance(requests, dict) and len(requests) <= 512
                and type(streaming) is int and streaming >= 0 and len(queue) <= 2,
                "full_queue_inventory_types")
        represented = running + waiting + skipped
        require(all(isinstance(identifier, str) and identifier for identifier in represented)
                and len(represented) == len(set(represented))
                and set(represented) == set(requests)
                and all(isinstance(value, dict) for value in requests.values()),
                "full_queue_scheduler_inventory")
        require(all(type(base[key]) is int and base[key] >= 0 for key in
                    ("running", "waiting", "pending_batches", "frontend_unfinished"))
                and base["running"] == len(running)
                and base["waiting"] == len(waiting) + len(skipped) + streaming
                and base["pending_batches"] == len(queue), "full_queue_count_conservation")
        require(isinstance(base["requests"], list)
                and [row["request_id"] for row in base["requests"]] == running,
                "full_queue_running_inventory")
        work_ids = set()
        for route in queue:
            require(set(route) == {"step_id", "work"}
                    and isinstance(route["step_id"], str)
                    and re.fullmatch(r"[a-f0-9]{64}", route["step_id"])
                    and isinstance(route["work"], list), "full_queue_route_schema")
            route_ids = []
            for work in route["work"]:
                require(set(work) == {"request_id", "position", "tokens", "prompt_length"}
                        and isinstance(work["request_id"], str) and bool(work["request_id"])
                        and type(work["position"]) is int and work["position"] >= 0
                        and type(work["tokens"]) is int and work["tokens"] > 0
                        and (work["prompt_length"] is None or
                             type(work["prompt_length"]) is int and work["prompt_length"] >= 0),
                        "full_queue_work_schema")
                route_ids.append(work["request_id"])
            require(len(route_ids) == len(set(route_ids)), "full_queue_duplicate_work")
            work_ids.update(route_ids)
        require(ghosts == sorted(work_ids - set(requests)), "full_queue_ghost_inventory")
        empty = (not represented and not requests and not queue and not ghosts
                 and not skipped and streaming == 0 and base["frontend_unfinished"] == 0)
        require(not empty or type(base["orphaned"]) is int and base["orphaned"] == 0,
                "full_queue_empty_count_disagreement")
        return frame, empty
    except Refused:
        raise
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise Refused("unknown_full_queue_output") from None


async def final_observations(index, capture, gateway, native, settings, native_run, clock, sleep):
    started = index["active_finished_at"]
    final = dict(started_at=started, deadline=started + settings["cleanup_observation_cap_s"],
                 finished_at=None, queue_records=[], observations=[], status="incomplete",
                 legacy_scalar_empty=None, legacy_scalar_failure=None,
                 scope="two_complete_source_observed_empty_full_inventories_not_persistent_absence")
    index["final_readonly"] = final
    try:
        async with asyncio.timeout(max(0, final["deadline"] - clock())):
            public = await capture.exchange(gateway, "GET", settings["state_path"], role="final-public", auth=True)
            index["final_public_record"] = public["sequence"]
            require(clock() <= final["deadline"], "final_public_late_receipt")
            legacy = await capture.exchange(native, "GET", settings["native_state_path"], role="native-after", auth=True)
            index["native_after_record"] = legacy["sequence"]
            require(clock() <= final["deadline"], "legacy_final_native_late_receipt")
            frame = native_state(legacy, settings, native_run=native_run)
            final["legacy_scalar_empty"] = all(frame[key] == 0 for key in
                ("running", "waiting", "pending_batches", "frontend_unfinished"))
            if not final["legacy_scalar_empty"]:
                final["legacy_scalar_failure"] = "legacy_final_native_not_drained_at_first_snapshot"
            consecutive = 0
            for position in range(settings["final_queue_max_samples"]):
                if position:
                    await sleep(settings["final_queue_poll_s"])
                require(clock() < final["deadline"], "final_full_queue_deadline_exhausted")
                first = len(capture.records)
                try:
                    record = await capture.exchange(
                        native, "GET", settings["final_queue_path"],
                        role=f"native-final-queue-{position}", auth=True)
                finally:
                    if len(capture.records) > first:
                        final["queue_records"].append(first)
                frame, empty = full_queue_observation(
                    record, settings, native_run=native_run, deadline=final["deadline"])
                require(clock() <= final["deadline"], "final_full_queue_assessment_late")
                final["observations"].append(dict(
                    record=record["sequence"], empty=empty, sampled_at=frame["base"]["sampled_at"]))
                consecutive = consecutive + 1 if empty else 0
                if consecutive == settings["final_queue_required_empty"]:
                    final["status"] = "observed_empty_at_two_intervals"
                    break
            require(final["status"] == "observed_empty_at_two_intervals", "final_full_queue_not_drained")
    except BaseException as error:
        final.update(status="refused",
                     refusal_reason=str(error) if isinstance(error, Refused) else "final_observation_failed",
                     error_type=type(error).__name__)
        raise
    finally:
        final["finished_at"] = clock()


def assess_final_observations(index, records, settings, native_run):
    final = index["final_readonly"]
    require(final["scope"] == "two_complete_source_observed_empty_full_inventories_not_persistent_absence",
            "final_observation_scope_changed")
    require(all(number(final[key]) for key in ("started_at", "deadline", "finished_at"))
            and final["deadline"] == final["started_at"] + settings["cleanup_observation_cap_s"]
            and final["started_at"] == index["active_finished_at"]
            and final["started_at"] <= final["finished_at"] <= final["deadline"],
            "final_absolute_readonly_budget")
    refs = final["queue_records"]
    actual = [row["sequence"] for row in records if row["path"] == settings["final_queue_path"]]
    require(refs == actual and 1 <= len(refs) <= settings["final_queue_max_samples"],
            "final_full_queue_complete_inventory")
    ordered = [index["final_public_record"], index["native_after_record"], *refs]
    require(ordered == list(range(ordered[0], len(records))), "final_readonly_sequence")
    previous_receive = final["started_at"]
    for reference in ordered:
        row = records[reference]
        require(row["method"] == "GET" and previous_receive <= row["client_send_at"]
                <= row["client_receive_at"] <= final["finished_at"],
                "final_readonly_chronology")
        previous_receive = row["client_receive_at"]
    consecutive, observations = 0, []
    for position, reference in enumerate(refs):
        record = records[reference]
        require(record["role"] == f"native-final-queue-{position}", "final_full_queue_role")
        if position:
            require(record["client_send_at"] >= records[refs[position - 1]]["client_receive_at"]
                    + settings["final_queue_poll_s"], "final_full_queue_fixed_cadence")
        frame, empty = full_queue_observation(
            record, settings, native_run=native_run, deadline=final["deadline"])
        observations.append(dict(record=reference, empty=empty, sampled_at=frame["base"]["sampled_at"]))
        consecutive = consecutive + 1 if empty else 0
        require(consecutive < settings["final_queue_required_empty"] or position == len(refs) - 1,
                "unplanned_observation_after_drain")
    require(consecutive == settings["final_queue_required_empty"], "final_full_queue_not_drained")
    require(final["status"] == "observed_empty_at_two_intervals" and final["observations"] == observations,
            "final_full_queue_assessment_binding")
    legacy = native_state(records[index["native_after_record"]], settings, native_run=native_run)
    legacy_empty = all(legacy[key] == 0 for key in ("running", "waiting", "pending_batches", "frontend_unfinished"))
    require(final["legacy_scalar_empty"] is legacy_empty and final["legacy_scalar_failure"] == (
        None if legacy_empty else "legacy_final_native_not_drained_at_first_snapshot"),
        "legacy_scalar_evidence_binding")
    return dict(observed_empty_at_two_intervals=True, observations=observations,
                legacy_scalar_empty_at_first_snapshot=legacy_empty,
                legacy_scalar_failure=final["legacy_scalar_failure"],
                scope=final["scope"], persistent_absence_proved=False, GPU_future_completion_proved=False)


def evaluate_step(step, records, baseline, settings, *, final=True):
    before_record = records[step["before_record"]]
    before = snapshot(before_record, binding=baseline)
    command_record = records[step["command_record"]]
    require(command_record["method"] == "POST"
            and command_record["path"] == settings["command_path"], "command_HTTP_contract")
    request = json.loads(command_record["request_body"])
    require(set(request) == {
        "schema", "run_id", "clock_domain", "identity", "expected_version",
        "not_after", "operation_id", "budget",
    }, "command_wire_fields")
    require(request["schema"] == SCHEMA and request["run_id"] == before.run_id
            and request["clock_domain"] == before.clock_domain
            and request["identity"] == asdict(before.identity), "command_source_binding")
    require(type(request["budget"]) is int and request["budget"] == step["budget"]
            and request["budget"] in (16, 32), "command_fixed_budget")
    require(type(request["expected_version"]) is int and number(request["not_after"])
            and request["operation_id"] == step["operation_id"], "command_values")
    require(before_record["client_receive_at"] <= command_record["client_send_at"]
            <= command_record["client_receive_at"], "command_client_chronology")
    require(command_record["client_clock_domain"] == before.clock_domain, "command_clock_domain")
    expected_version = baseline.budget_version if step["control"] == "stale" else before.budget_version
    expected_deadline = before.sampled_at + (
        -settings["expired_margin_s"] if step["control"] == "expired" else settings["command_ttl_s"])
    require(request["expected_version"] == expected_version
            and request["not_after"] == expected_deadline, "version_deadline_binding")
    reply = body(command_record)
    require(command_record["http_status"] == step["expected_http"], "unexpected_control_status")
    after_records = [records[index] for index in step["after_records"]]
    require(after_records, "missing_next_actual_public_observation")
    after = [snapshot(record, binding=baseline) for record in after_records]
    previous_receive = command_record["client_receive_at"]
    for record in after_records:
        require(previous_receive <= record["client_send_at"], "post_observation_chronology")
        previous_receive = record["client_receive_at"]
    result = dict(
        server_clock_domain=before.clock_domain, server_run_id=before.run_id,
        server_received_at=None, server_received_observability="not_exposed",
        server_received_interval=[command_record["client_send_at"], command_record["client_headers_at"]],
        server_checked_at=None, actual_applied_at=None, acknowledgement_is_application=False,
        before_budget=before.budget, before_version=before.budget_version,
        next_budget=after[0].budget, next_version=after[0].budget_version,
        next_public_sampled_at=after[0].sampled_at,
        client_send_at=command_record["client_send_at"],
        client_receive_at=command_record["client_receive_at"],
    )
    if step["control"] == "fresh":
        require(set(reply) == {
            "schema", "run_id", "clock_domain", "operation_id", "status",
            "previous_budget", "budget", "budget_version", "applied_at",
        }, "success_receipt_fields")
        require(reply["schema"] == SCHEMA and reply["run_id"] == before.run_id
                and reply["clock_domain"] == before.clock_domain
                and reply["operation_id"] == request["operation_id"]
                and reply["status"] == "applied", "success_receipt_binding")
        require(type(reply["budget_version"]) is int
                and reply["budget_version"] == before.budget_version + 1
                and number(reply["budget"]) and reply["budget"] == step["budget"]
                and number(reply["previous_budget"]) and reply["previous_budget"] == before.budget,
                "success_receipt_budget_version")
        checked = reply["applied_at"]
        require(number(checked) and command_record["client_send_at"] <= checked
                <= command_record["client_receive_at"] and checked < request["not_after"],
                "guard_check_outside_send_deadline")
        require(all(item.budget == step["budget"] and item.budget_version == before.budget_version + 1
                    for item in after), "ack_without_observed_application")
        require(checked <= after[0].sampled_at, "application_observation_before_check")
        upper = min(command_record["client_receive_at"], after[0].sampled_at)
        result.update(
            status="observed_budget_transition_interval_censored",
            raw_server_applied_at_claim=checked, server_checked_at=checked,
            server_checked_semantics="source_proven_pre_write_expiry_sample",
            server_received_interval=[command_record["client_send_at"], checked],
            actual_application_interval=[checked, upper],
            interval_justification="Pinned synchronous write before response, corroborated by the next real public state.",
            actual_application_before_deadline=(
                "established_by_interval_upper" if upper < request["not_after"]
                else "not_established_interval_crosses_deadline"),
        )
    else:
        expected_detail = "command expired" if step["control"] == "expired" else "stale budget version"
        require(reply == {"detail": expected_detail}, "wrong_refusal_reason")
        require(all(item.budget == before.budget and item.budget_version == before.budget_version
                    for item in after), "hidden_application_after_rejection")
        if step["control"] == "stale":
            require(expected_version < before.budget_version, "stale_control_not_stale")
        if final:
            require(len(after) >= 2
                    and after[-1].sampled_at - after[0].sampled_at >= settings["rejection_watch_s"],
                    "rejection_watch_incomplete")
            require(after[-1].sampled_at > request["not_after"], "watch_did_not_cross_request_deadline")
        result.update(
            status="refused_no_budget_version_change_within_watch",
            watch_started_at=after[0].sampled_at, watch_ended_at=after[-1].sampled_at,
            watch_samples=len(after), hidden_application_after_watch="unobserved_not_claimed_absent",
        )
    return result


class Capture:
    def __init__(self, output, settings, secret, domain, *, clock=time.monotonic):
        self.output, self.settings, self.secret, self.domain = output, settings, secret, domain
        self.clock, self.records = clock, []
        (output / "http").mkdir()

    def scrub(self, raw):
        return raw.replace(self.secret.encode(), b"[AUTH_VALUE_REMOVED]")

    async def exchange(self, client, method, path, *, role, payload=None, auth=False, identifier=None):
        require(len(self.records) < self.settings["maximum_http_exchanges"], "HTTP_exchange_cap")
        content = canonical(payload) if payload is not None else None
        headers = {"content-type": "application/json"} if content is not None else {}
        if auth:
            headers["x-admin-api-key" if path.startswith("/admin/") else "x-observer-key"] = self.secret
        if identifier is not None:
            headers["x-benchmark-request-id"] = identifier
        row = dict(
            sequence=len(self.records), role=role, method=method, path=path,
            target=dict(scheme=client.base_url.scheme, host=client.base_url.host, port=client.base_url.port),
            request_body=self.scrub(content).decode() if content is not None else None,
            request_id=identifier, client_clock_domain=self.domain,
            client_send_at=self.clock(), client_headers_at=None, client_receive_at=None,
            http_status=None, response_headers={}, error_type=None, body_complete=False,
            auth_values_captured=False, credential_removed=False,
        )
        self.records.append(row)
        raw = bytearray()
        try:
            async with asyncio.timeout(self.settings["http_timeout_s"]):
                async with client.stream(method, path, content=content, headers=headers) as response:
                    row.update(http_status=response.status_code, client_headers_at=self.clock())
                    row["response_headers"] = {
                        key: self.scrub(response.headers[key].encode()).decode(errors="replace")
                        for key in ("content-type", "x-request-id") if key in response.headers}
                    row["credential_removed"] = any(
                        self.secret in response.headers.get(key, "")
                        for key in ("content-type", "x-request-id"))
                    async for chunk in response.aiter_bytes():
                        room = self.settings["body_cap_bytes"] - len(raw)
                        raw.extend(chunk[:room])
                        if len(chunk) > room:
                            raise Refused("HTTP_body_cap")
                    row["body_complete"] = True
        except asyncio.CancelledError:
            row["error_type"] = "CancelledError"
            raise
        except Exception as error:
            row["error_type"] = type(error).__name__
        finally:
            row["client_receive_at"] = self.clock()
            clean = self.scrub(bytes(raw))
            row.update(
                credential_removed=row["credential_removed"] or clean != bytes(raw),
                response_body_base64=base64.b64encode(clean).decode(),
                response_body_sha256=sha(clean), response_bytes_captured=len(clean))
            persist(self.output / "http" / f'{row["sequence"]:03d}.json', row)
        return row


async def execute(output, gateway, native, *, secret, domain, run_id, synthetic,
                  clock=time.monotonic, sleep=asyncio.sleep, source_proof=None):
    """Clients are caller-owned; every offer task created here is drained here."""
    settings = plan()
    require(isinstance(secret, str) and len(secret) >= 16, "private_auth_value_required")
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", run_id), "bounded_run_id")
    output = Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    if source_proof is not None:
        persist(output / "preflight.json", source_proof)
    capture = Capture(output, settings, secret, domain, clock=clock)
    steps = [dict(step, operation_id=f"{run_id}-{step['name']}", status="not_sent",
                  before_record=None, command_record=None, after_records=[])
             for step in settings["steps"]]
    offers = [dict(
        index=index, step=index // settings["offers_per_step"],
        request_id=sha(f"{run_id}:offer:{index}".encode())[:32], status="not_sent",
        record=None) for index in range(settings["total_planned_offers"])]
    index = dict(
        schema="team16-fixed-control-probe-result-v2", run_id=run_id,
        experiment_sha256=sha((HERE / "EXPERIMENT.json").read_bytes()),
        preflight_sha256=sha(canonical(source_proof)) if source_proof is not None else None,
        synthetic_fixture=synthetic, model_control=False, native_mpc_permitted=False,
        deployable=False, live_host_launched_by_probe=False,
        source13_service_ownership="Mayor; this executable attaches and never launches or stops services",
        clock_domain=domain, started_at=clock(), status="incomplete",
        steps=steps, offers=offers, native_before_record=None, native_after_record=None,
        final_public_record=None, failures=[], owned_offer_tasks_remaining=0,
    )
    tasks = set()
    baseline = None
    native_epoch = None

    async def public(role):
        return await capture.exchange(gateway, "GET", settings["state_path"], role=role, auth=True)

    async def offer(row):
        row["status"] = "attempting"
        try:
            record = await capture.exchange(
                gateway, "POST", "/v1/chat/completions", role=f"offer-{row['index']}",
                identifier=row["request_id"],
                payload=dict(model=settings["model"], messages=[dict(role="user", content=settings["prompt"])],
                             temperature=0, max_tokens=settings["offer_max_tokens"], stream=False))
            row["record"] = record["sequence"]
            row["status"] = offer_fate(record, row["request_id"])
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            raise
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__,
                       failure_reason=str(error) if isinstance(error, Refused) else "offer_processing_failed")

    try:
        async with asyncio.timeout(settings["active_duration_cap_s"]):
            native_before = await capture.exchange(
                native, "GET", settings["native_state_path"], role="native-before", auth=True)
            index["native_before_record"] = native_before["sequence"]
            native_frame = native_state(native_before, settings)
            native_epoch = native_frame["run_id"]
            require(all(native_frame[key] == 0 for key in
                        ("running", "waiting", "pending_batches", "frontend_unfinished")),
                    "native_stack_not_initially_drained")
            initial = await public("initial")
            index["initial_public_record"] = initial["sequence"]
            baseline = snapshot(initial)
            require(baseline.budget == settings["initial_budget"] and not baseline.requests,
                    "expected_exclusive_initial_budget_and_empty_gateway")
            if source_proof is not None:
                token_identity = body(initial)["cost_spec"]["tokenizer_identity"]
                suffix = ";model=" + source_proof["model_alias"] + ";available=True;fallback=chars4-chat-v1"
                require(token_identity.endswith(suffix), "gateway_cached_model_tokenizer_identity")
            expected_budget, expected_version = baseline.budget, baseline.budget_version
            for position, step in enumerate(steps):
                before_record = await public(f"before-{step['name']}")
                before = snapshot(before_record, binding=baseline)
                step["before_record"] = before_record["sequence"]
                require(before.budget == expected_budget and before.budget_version == expected_version,
                        "concurrent_budget_writer_or_hidden_change")
                request = dict(
                    schema=SCHEMA, run_id=before.run_id, clock_domain=before.clock_domain,
                    identity=asdict(before.identity), budget=step["budget"],
                    expected_version=(baseline.budget_version if step["control"] == "stale"
                                      else before.budget_version),
                    not_after=before.sampled_at + (
                        -settings["expired_margin_s"] if step["control"] == "expired"
                        else settings["command_ttl_s"]),
                    operation_id=step["operation_id"])
                command_record = await capture.exchange(
                    gateway, "POST", settings["command_path"], role=step["name"], payload=request, auth=True)
                step.update(command_record=command_record["sequence"], status="attempted")
                after_record = await public(f"after-{step['name']}")
                step["after_records"].append(after_record["sequence"])
                evaluate_step(step, capture.records, baseline, settings, final=False)
                if step["control"] != "fresh":
                    start = snapshot(after_record, binding=baseline).sampled_at
                    for _ in range(settings["max_watch_samples_per_rejection"] - 1):
                        await sleep(settings["watch_poll_s"])
                        observed = await public(f"watch-{step['name']}")
                        step["after_records"].append(observed["sequence"])
                        evaluate_step(step, capture.records, baseline, settings, final=False)
                        if snapshot(observed, binding=baseline).sampled_at - start >= settings["rejection_watch_s"]:
                            break
                step["assessment"] = evaluate_step(step, capture.records, baseline, settings)
                step["status"] = "expected_control_observed"
                last = snapshot(capture.records[step["after_records"][-1]], binding=baseline)
                expected_budget, expected_version = last.budget, last.budget_version
                current = [asyncio.create_task(offer(row)) for row in offers if row["step"] == position]
                tasks.update(current)
                await asyncio.gather(*current)
                tasks.difference_update(current)
                for row in offers:
                    if row["step"] == position and row["status"] not in (
                            "completed_stop", "completed_length", "rejected"):
                        raise Refused(row.get("failure_reason", "offer_response_failure"))
            index["status"] = "capture_completed"
    except asyncio.CancelledError:
        index["failures"].append(dict(error_type="CancelledError", reason="caller_cancelled"))
        raise
    except Exception as error:
        index["failures"].append(dict(
            error_type=type(error).__name__,
            reason=str(error) if isinstance(error, Refused) else "capture_error_details_omitted"))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        index["owned_offer_tasks_remaining"] = sum(not task.done() for task in tasks)
        index["active_finished_at"] = clock()
        final_cancelled = None
        try:
            await final_observations(index, capture, gateway, native, settings, native_epoch, clock, sleep)
        except BaseException as error:
            index["failures"].append(dict(
                error_type=type(error).__name__,
                reason=str(error) if isinstance(error, Refused) else "final_observation_failed"))
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                final_cancelled = error
        index.update(
            finished_at=clock(), http_records=len(capture.records),
            control_commands_attempted=sum(row["path"] == settings["command_path"] for row in capture.records),
            offered_attempted=sum(row["path"] == "/v1/chat/completions" for row in capture.records))
        # A cancelled offer still has a durable HTTP receipt; retain its index.
        for row in offers:
            matches = [record["sequence"] for record in capture.records if record["request_id"] == row["request_id"]]
            if matches:
                require(len(matches) == 1, "offer_retry_not_permitted")
                row["record"] = matches[0]
        index["record_sha256"] = {
            f'{row["sequence"]:03d}.json': sha((output / "http" / f'{row["sequence"]:03d}.json').read_bytes())
            for row in capture.records}
        clean_index = json.loads(capture.scrub(canonical(index)))
        persist(output / "index.json", clean_index)
        if final_cancelled is not None:
            raise final_cancelled
    return index


def offer_fate(record, identifier):
    if record.get("error_type"):
        return "cancelled" if record["error_type"] == "CancelledError" else "transport_or_body_failure"
    if record["http_status"] != 200:
        return "rejected" if record["http_status"] in (429, 503) else "HTTP_failure"
    document = body(record)
    require(record["response_headers"].get("x-request-id") == identifier, "offer_response_identity")
    require(isinstance(document, dict) and isinstance(document.get("choices"), list)
            and len(document["choices"]) == 1, "completion_response_schema")
    request = json.loads(record["request_body"])
    require(document.get("model") == request["model"], "completion_model_identity")
    tokens = document.get("usage", {}).get("completion_tokens")
    require(type(tokens) is int and 0 <= tokens <= request["max_tokens"], "completion_token_cap")
    reason = document["choices"][0].get("finish_reason")
    return "completed_stop" if reason == "stop" else "completed_length" if reason == "length" else "completion_failure"


def _replay(directory):
    settings = plan()
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_bytes())
    if index.get("schema") == "team16-fixed-control-probe-preflight-refusal-v1":
        return dict(schema="team16-fixed-control-probe-replay-v1", run_id=index["run_id"],
                    synthetic_fixture=False, component_contract_pass=False,
                    actual_application_deadline_proved=False, application_timing="not_attempted",
                    failures=index["failures"], offered_attempted=0, native_mpc_permitted=False,
                    deployable=False)
    require(index.get("schema") == "team16-fixed-control-probe-result-v2"
            and type(index.get("synthetic_fixture")) is bool, "result_schema")
    require(index["experiment_sha256"] == sha((HERE / "EXPERIMENT.json").read_bytes()), "experiment_binding")
    require({path.name for path in (directory / "http").iterdir()} == set(index["record_sha256"])
            and all(path.is_file() for path in (directory / "http").iterdir())
            and not any(path.is_symlink() for path in (directory / "http").iterdir()),
            "unindexed_or_unknown_HTTP_output")
    records = []
    for name, fingerprint in sorted(index["record_sha256"].items()):
        require(re.fullmatch(r"[0-9]{3}\.json", name), "record_path")
        raw = (directory / "http" / name).read_bytes()
        require(sha(raw) == fingerprint, "record_hash")
        record = json.loads(raw)
        require(record["sequence"] == len(records), "record_sequence")
        require(record["auth_values_captured"] is False, "auth_capture_not_permitted")
        require((record["method"], record["path"]) in {
            ("GET", settings["state_path"]), ("GET", settings["native_state_path"]),
            ("GET", settings["final_queue_path"]),
            ("POST", settings["command_path"]), ("POST", "/v1/chat/completions"),
        }, "unplanned_HTTP_operation")
        if not index["synthetic_fixture"]:
            port = 8000 if record["path"] in (settings["native_state_path"], settings["final_queue_path"]) else 8080
            require(record["target"] == dict(scheme="http", host="127.0.0.1", port=port),
                    "not_the_existing_loopback_stack")
        require(record["client_clock_domain"] == index["clock_domain"]
                and number(record["client_send_at"]) and number(record["client_receive_at"])
                and record["client_send_at"] <= record["client_receive_at"], "HTTP_clock_brackets")
        if record["client_headers_at"] is not None:
            require(number(record["client_headers_at"])
                    and record["client_send_at"] <= record["client_headers_at"] <= record["client_receive_at"],
                    "HTTP_header_clock_brackets")
        records.append(record)
    require(len(records) == index["http_records"], "record_count")
    assessments, failures, terminal = [], list(index["failures"]), None
    try:
        require(index["status"] == "capture_completed", "incomplete_capture")
        if not index["synthetic_fixture"]:
            proof = json.loads((directory / "preflight.json").read_bytes())
            require(sha(canonical(proof)) == index["preflight_sha256"]
                    and proof["source_root"] == settings["source_root"]
                    and proof["client_clock_domain"] == index["clock_domain"]
                    and proof["checked_at"] <= index["started_at"], "live_source_preflight_binding")
        baseline = snapshot(records[index["initial_public_record"]])
        require(baseline.budget == settings["initial_budget"] and not baseline.requests, "initial_state")
        require(len(index["steps"]) == len(settings["steps"]), "step_count")
        expected_budget, expected_version = baseline.budget, baseline.budget_version
        operations = set()
        for step, declaration in zip(index["steps"], settings["steps"]):
            require(all(step[key] == value for key, value in declaration.items()), "fixed_step_changed")
            require(step["operation_id"] not in operations
                    and step["operation_id"] == f"{index['run_id']}-{step['name']}", "operation_identity")
            operations.add(step["operation_id"])
            before = snapshot(records[step["before_record"]], binding=baseline)
            require((before.budget, before.budget_version) == (expected_budget, expected_version),
                    "between_step_hidden_budget_change")
            assessments.append(evaluate_step(step, records, baseline, settings))
            after = snapshot(records[step["after_records"][-1]], binding=baseline)
            expected_budget, expected_version = after.budget, after.budget_version
        require(sum(row["path"] == settings["command_path"] for row in records) == 4
                == index["control_commands_attempted"], "unplanned_or_missing_command")
        require(len(index["offers"]) == settings["total_planned_offers"], "planned_offer_inventory")
        offer_records = [row for row in records if row["path"] == "/v1/chat/completions"]
        require(len(offer_records) == index["offered_attempted"] == settings["total_planned_offers"],
                "all_offered_inventory")
        seen = set()
        for position, offer in enumerate(index["offers"]):
            identifier = sha(f"{index['run_id']}:offer:{position}".encode())[:32]
            require(offer["index"] == position and offer["request_id"] == identifier
                    and offer["record"] not in seen, "offer_identity_or_duplicate")
            seen.add(offer["record"])
            record = records[offer["record"]]
            require(record["request_id"] == identifier and record["method"] == "POST", "offer_record_binding")
            phase = position // settings["offers_per_step"]
            require(offer["step"] == phase, "offer_phase_binding")
            phase_end = index["steps"][phase]["after_records"][-1]
            next_phase = (index["steps"][phase + 1]["before_record"] if phase + 1 < len(index["steps"])
                          else index["final_public_record"])
            require(phase_end < offer["record"] < next_phase
                    and records[phase_end]["client_receive_at"] <= record["client_send_at"]
                    and record["client_receive_at"] <= records[next_phase]["client_send_at"],
                    "offer_outside_declared_budget_phase")
            request = json.loads(record["request_body"])
            require(request == dict(
                model=settings["model"], messages=[dict(role="user", content=settings["prompt"])],
                temperature=0, max_tokens=settings["offer_max_tokens"], stream=False), "offer_cap_or_payload")
            require(offer["status"] == offer_fate(record, identifier), "offer_fate_replay")
        final = snapshot(records[index["final_public_record"]], binding=baseline)
        require((final.budget, final.budget_version) == (expected_budget, expected_version)
                and not final.requests, "final_gateway_not_drained_or_hidden_change")
        initial_native = native_state(records[index["native_before_record"]], settings)
        require(all(initial_native[key] == 0 for key in
                    ("running", "waiting", "pending_batches", "frontend_unfinished")),
                "native_stack_not_initially_drained")
        terminal = assess_final_observations(index, records, settings, initial_native["run_id"])
        require(index["owned_offer_tasks_remaining"] == 0, "owned_offer_tasks_remaining")
    except (ValueError, KeyError, TypeError, IndexError) as error:
        failures.append(dict(error_type=type(error).__name__,
                             reason=str(error) if isinstance(error, Refused) else "invalid_replay_evidence"))
    timing_proved = not failures and all(
        row.get("actual_application_before_deadline") == "established_by_interval_upper"
        for row in assessments if row["status"] == "observed_budget_transition_interval_censored")
    return dict(
        schema="team16-fixed-control-probe-replay-v2", run_id=index["run_id"],
        synthetic_fixture=index["synthetic_fixture"], component_contract_pass=not failures,
        actual_application_deadline_proved=timing_proved,
        exact_application_timestamp_observed=False, application_timing="interval_censored",
        assessments=assessments, failures=failures, planned_offers=len(index["offers"]),
        terminal_observation=terminal, terminal_scope_amended=True,
        native01_disposition="FAIL remains immutable; not replayable as02 evidence",
        offered_attempted=index["offered_attempted"],
        offer_fates={name: sum(row["status"] == name for row in index["offers"])
                     for name in sorted({row["status"] for row in index["offers"]})},
        original_native_batch_capture_not_performed=True,
        GPU_identity_validation_is_Mayor_owned=True,
        installed_source_preflight_bound=index.get("preflight_sha256") is not None,
        Mayor_stack_ownership_and_native_context_required=True,
        native_mpc_permitted=False, deployable=False)


def replay(directory):
    try:
        return _replay(directory)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError):
        return dict(schema="team16-fixed-control-probe-replay-v2", run_id=None,
                    component_contract_pass=False, actual_application_deadline_proved=False,
                    application_timing="unproven", offered_attempted=None, planned_offers=8,
                    failures=[dict(reason="unknown_or_invalid_replay_output")],
                    unknown_outcomes_not_assumed_absent=True, native_mpc_permitted=False, deployable=False)


def verify_live_sources(source_root):
    expected = Path("/workspace/pr32-r23/source")
    require(source_root == expected and source_root.resolve() == expected, "exact_existing_Source13_root")
    pins = json.loads((HERE / "INPUTS.json").read_bytes())
    for relative, fingerprint in pins["product_sha256"].items():
        path = source_root / relative
        require(not path.is_symlink() and sha(path.read_bytes()) == fingerprint, "installed_product_source_changed")
    source13 = source_root / plan()["source13_directory"]
    raw = (source13 / "source-closure.json").read_bytes()
    require(sha(raw) == pins["source13_manifest_sha256"], "installed_Source13_manifest_changed")
    for relative, fingerprint in json.loads(raw)["source_sha256"].items():
        path = source13 / relative
        require(not path.is_symlink() and sha(path.read_bytes()) == fingerprint, "installed_Source13_source_changed")
    reference = json.loads((HERE / "inputs/references/qualification-result.json").read_bytes())["runtime_identity"]
    model = Path("/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507")
    alias = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots") / plan()["model_revision"]
    require(alias.exists() and alias.resolve() == model, "existing_immutable_model_alias_required")
    require(sha((model / "verified-manifest.json").read_bytes()) == reference["model_manifest_sha256"],
            "existing_model_manifest_changed")
    require(sha((source13 / "model-content-lock.json").read_bytes()) == reference["model_content_lock_sha256"],
            "model_content_lock_changed")
    expected_packages = {"vllm": "0.30.0", "torch": "2.13.0+cu130", "transformers": "5.22.1"}
    require({name: importlib.metadata.version(name) for name in expected_packages} == expected_packages,
            "existing_native_interpreter_packages_required")
    return dict(product_files=len(pins["product_sha256"]),
                source13_manifest_sha256=pins["source13_manifest_sha256"],
                source_root=str(source_root), installed_sources_modified=False,
                model_alias=str(alias), model_manifest_sha256=reference["model_manifest_sha256"],
                model_cache_full_byte_revalidation="Mayor-owned run_h16 prerequisite; not repeated here",
                expected_native_packages=expected_packages,
                client_clock_domain=clock_domain(), checked_at=time.monotonic(),
                GPU_identity_validation="Mayor-owned; this probe never calls GPU APIs",
                model_convergence_gate_required=False)


async def live(args):
    require(args.mayor_execute and sys.dont_write_bytecode and sys.flags.optimize == 0,
            "Mayor_explicit_execution_and_no_bytecode_required")
    require(args.source_root == Path("/workspace/pr32-r23/source"), "exact_existing_Source13_root")
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", args.run_id), "bounded_run_id")
    require(args.output == args.source_root / "results/mpc-native-r24" /
            ("native-team16-control-probe02-" + args.run_id), "new_exclusive_result_path_required")
    settings = plan()
    try:
        require(not args.private_config.is_symlink()
                and stat.S_IMODE(args.private_config.stat().st_mode) == 0o600, "private_0600_config_required")
        require(not args.prelaunch_receipt.is_symlink(), "plain_TMPDIR_prelaunch_receipt_required")
        tmpdir_receipt = verify_tmpdir_receipt(json.loads(args.prelaunch_receipt.read_bytes()))
        proof = verify_live_sources(args.source_root)
        proof["TMPDIR_prelaunch"] = tmpdir_receipt
        config = json.loads(args.private_config.read_bytes())
        require(config.get("MODEL_NAME") == proof["model_alias"], "private_config_model_alias")
        secret = config["ADMIN_API_KEY"]
        require(isinstance(secret, str) and len(secret) >= 16, "private_auth_value_required")
    except Exception as error:
        args.output.mkdir(mode=0o700, exist_ok=False)
        persist(args.output / "index.json", dict(
            schema="team16-fixed-control-probe-preflight-refusal-v1", run_id=args.run_id,
            experiment_sha256=sha((HERE / "EXPERIMENT.json").read_bytes()),
            failures=[dict(error_type=type(error).__name__,
                           reason=str(error) if isinstance(error, (Refused, PrelaunchRefused))
                           else "local_preflight_refused")],
            control_commands_attempted=0, offered_attempted=0,
            native_mpc_permitted=False, deployable=False))
        print(json.dumps(dict(run_id=args.run_id, status="preflight_refused", HTTP_operations=0)))
        return 2
    async with httpx.AsyncClient(
        base_url=settings["gateway_origin"], timeout=settings["http_timeout_s"], trust_env=False,
        follow_redirects=False, limits=httpx.Limits(max_connections=3),
    ) as gateway, httpx.AsyncClient(
        base_url=settings["native_origin"], timeout=settings["http_timeout_s"], trust_env=False,
        follow_redirects=False, limits=httpx.Limits(max_connections=1),
    ) as native:
        await execute(args.output, gateway, native, secret=secret, domain=clock_domain(),
                      run_id=args.run_id, synthetic=False, source_proof=proof)
    result = replay(args.output)
    persist(args.output / "replay.json", result)
    print(json.dumps({key: result[key] for key in
                      ("run_id", "component_contract_pass", "application_timing",
                       "offered_attempted", "native_mpc_permitted")}, sort_keys=True))
    return 0 if result["component_contract_pass"] and result["actual_application_deadline_proved"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture = sub.add_parser("capture", allow_abbrev=False)
    capture.add_argument("--mayor-execute", action="store_true")
    capture.add_argument("--source-root", type=Path, required=True)
    capture.add_argument("--private-config", type=Path, required=True)
    capture.add_argument("--prelaunch-receipt", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--run-id", required=True)
    offline = sub.add_parser("replay", allow_abbrev=False)
    offline.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        raise SystemExit(asyncio.run(live(args)))
    result = replay(args.directory)
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["component_contract_pass"] and result["actual_application_deadline_proved"] else 2)


if __name__ == "__main__":
    main()
