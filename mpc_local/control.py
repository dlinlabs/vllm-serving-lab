"""CPU-only ASGI control and strict software-owner receipts, not native permits."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import httpx

from mpc_lab.actuator import GuardedActuator, IdentificationPermit
from mpc_lab.contracts import Decision, Plan, finite
from mpc_lab.controller import validate_snapshot
from mpc_lab.runtime import require_guarded_capability, snapshot_from_dict
from mpc_local.contracts import Choice, PublicRequest, PublicState


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "gateway.py": "0b1a80ade258d8875ba80a072c4f360bfa5d4a7ac312bd222c0571380b341570",
    "telemetry.py": "46d6fb923e558f6435d269dd6fa5cfd5b0d71eb1fd91cbb3aaca8a82385e6370",
    "mpc_lab/actuator.py": "e31dbe92eb3a27be554db15405af0a6046a6820096e84540812d3ee88afc59d5",
    "mpc_lab/runtime.py": "f04c58eb9b5a81e7e1c7a22aaea1cd6b59de88c568ec77d51f1487af4c85b727",
    "mpc_lab/contracts.py": "f39f6b73b41b1433a220f182679852df5c4332d9022ce903dba9863375236e23",
    "mpc_lab/controller.py": "44d705101820d89677035101110c70f890f94f7d82a5a521ec87d5e09dbf08cc",
    "mpc_local/contracts.py": "3fcc41b883cf57a5fa49e9424139a1b949be4db872c0724155710b447c5016ec",
    "experiments/mpc-research/native-v3/r21-local-protocol.json": "929b9fc68a252fe82d01fdea4972fd534012614205b61c9dbbaf6fb1bc20ef5f",
    "experiments/mpc-research/native-v3/r21-candidate-fit-preregistration-v1.json": "c2d62019676eede1d5985243341d68039b7aaeccdd6207a3e32cea320a7805a3",
    "prefill_lab/live_http.py": "cbbd1f338e0d078d5c5eacdfbcec725dcfe2ed9b46c1ee1ed68ad2f26de24e1f",
    "experiments/mpc-research/native_observer/pr33_native_observer.py": "77633dc7652066e45b7680d297d3e36ddb97a3e12222de9f526a76b9a01d66b4",
    "experiments/mpc-research/native-v3/collect_identification.py": "78e2c8b91f6b4a582feb69293adff5aa34d4fbd332dfa93b92735bcd0275a069",
    "experiments/mpc-research/native-v3/prospective_identification.py": "7bedff0aa2097a30938cb4e19744e86da86c7999cb7bab950ba2d93fe2f1a1ac",
}


def qualify_source(root=ROOT):
    observed = {}
    for name, expected in SOURCE_HASHES.items():
        observed[name] = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
        if observed[name] != expected:
            raise ValueError("source_mismatch:" + name)
    return observed


def qualify_snapshot(document, parameters, *, run_id, clock_domain, now):
    require_guarded_capability(document)
    snapshot = snapshot_from_dict(document)
    reason = validate_snapshot(
        snapshot, now=now, clock_domain=clock_domain,
        expected_identity=parameters.identity, max_age_s=0.5,
    )
    if reason:
        raise ValueError(reason)
    if snapshot.run_id != run_id or not run_id:
        raise ValueError("run_mismatch")
    if snapshot.budget not in (16.0, 32.0):
        raise ValueError("unsupported_budget")
    if document.get("cost_spec") != asdict(parameters.cost_spec):
        raise ValueError("cost_spec_mismatch")
    if any(mark.cost != parameters.cost_spec.estimate(mark.prompt_tokens, mark.max_tokens)
           for mark in snapshot.requests):
        raise ValueError("original_cost_mismatch")
    return snapshot


def _received_native_progress(frames, *, cut, now, domain, native_run_id, owners):
    if not isinstance(native_run_id, str) or re.fullmatch("[0-9a-f]{32}", native_run_id) is None:
        raise ValueError("native_run_binding_missing")
    prefix = []
    for frame in frames:
        if not isinstance(frame, dict) or not finite(frame.get("received_at")):
            raise ValueError("native_receipt_clock_missing")
        if frame["received_at"] <= cut:
            prefix.append(frame)
    if not prefix:
        raise ValueError("native_received_prefix_missing")
    progress, engines, engine_owners, latest_ids = {}, {}, {}, set()
    previous_sample = -1.0
    for frame in sorted(prefix, key=lambda f: f["received_at"]):
        sample, frontend, received = (frame.get(k) for k in (
            "sampled_at", "frontend_sampled_at", "received_at",
        ))
        if (
            frame.get("schema") != "pr33-native-read-only-v1"
            or frame.get("run_id") != native_run_id or frame.get("clock_domain") != domain
            or frame.get("observer_sha256") != SOURCE_HASHES[
                "experiments/mpc-research/native_observer/pr33_native_observer.py"]
            or not isinstance(frame.get("vllm_version"), str)
            or frame["vllm_version"].partition("+")[0] != "0.30.0"
            or frame.get("complete") is not True
            or any(not finite(v) for v in (sample, frontend, received))
            or not 0 <= sample <= frontend <= received <= cut
            or sample < previous_sample
            or frontend - sample > .05
            or any(type(frame.get(k)) is not int or frame[k] < 0 for k in (
                "running", "waiting", "pending_batches", "frontend_unfinished",
            ))
            or not finite(frame.get("kv_cache_usage")) or not 0 <= frame["kv_cache_usage"] <= 1
            or not isinstance(frame.get("requests"), list)
            or len(frame["requests"]) != frame["running"] or frame["running"] > 512
        ):
            raise ValueError("native_prefix_schema_clock_source")
        previous_sample = sample
        seen, seen_engine = set(), set()
        for row in frame["requests"]:
            if not isinstance(row, dict):
                raise ValueError("native_progress_schema")
            identifier, engine = row.get("gateway_request_id"), row.get("request_id")
            if (
                not isinstance(identifier, str) or re.fullmatch("[0-9a-f]{32}", identifier) is None
                or not isinstance(engine, str) or not engine or engine in seen_engine
                or identifier in seen or row.get("ignore_eos") is not False
                or any(type(row.get(k)) is not int or row[k] < 0 for k in (
                    "prompt_tokens", "computed_tokens", "output_tokens", "maximum_output_tokens",
                ))
                or row["maximum_output_tokens"] < 1
                or row["output_tokens"] > row["maximum_output_tokens"]
            ):
                raise ValueError("native_uuid_or_output_unknown")
            seen.add(identifier)
            seen_engine.add(engine)
            if engine in engine_owners and engine_owners[engine] != identifier:
                raise ValueError("native_engine_uuid_alias")
            engine_owners[engine] = identifier
            owner = owners.get(identifier)
            if owner is None:
                continue
            if (row["prompt_tokens"] != owner.prompt_tokens
                    or row["maximum_output_tokens"] != owner.max_tokens
                    or sample < owner.admitted_at):
                raise ValueError("native_owner_geometry_or_chronology")
            if identifier in progress:
                previous = progress[identifier]
                if (sample < previous["sampled_at"] or row["output_tokens"] < previous["output_tokens"]
                        or engines[identifier] != engine
                        or sample == previous["sampled_at"]
                        and row["output_tokens"] != previous["output_tokens"]):
                    raise ValueError("native_progress_regression_or_uuid_alias")
            engines[identifier] = engine
            progress[identifier] = {**row, "sampled_at": sample}
        latest_ids = seen
        if not owners and any(frame[k] for k in (
            "running", "waiting", "pending_batches", "frontend_unfinished",
        )):
            raise ValueError("native_nonempty_without_gateway_owners")
    if latest_ids - set(owners):
        raise ValueError("native_uuid_not_in_current_owner_cut")
    if now - previous_sample > .5 or any(now - row["sampled_at"] > .5 for row in progress.values()):
        raise ValueError("native_progress_stale")
    for identifier, row in progress.items():
        row["visible_in_latest_frame"] = identifier in latest_ids
    return progress


def _received_http_endpoints(observations, *, cut, domain, owners):
    endpoints = {}
    prefix = []
    for row in observations:
        if not isinstance(row, dict) or not finite(row.get("published_at")):
            raise ValueError("http_publication_clock_missing")
        if row["published_at"] <= cut:
            prefix.append(row)
    for row in sorted(prefix, key=lambda r: r["published_at"]):
        identifier = row.get("request_id")
        if row.get("clock_domain") != domain or identifier not in owners:
            raise ValueError("http_owner_or_clock_unknown")
        owner = owners[identifier]
        first, terminal = row.get("first_content_at"), row.get("terminal_at")
        if (
            row.get("requested_max_tokens") != owner.max_tokens
            or "observation_error" in row
            or row.get("requested_ignore_eos") is not False
            or row.get("response_request_id") not in (None, identifier)
            or (first is not None or terminal is not None)
            and row.get("response_request_id") != identifier
            or not finite(row.get("scheduled_at"))
            or not 0 <= row["scheduled_at"] <= owner.received_at
            or row["scheduled_at"] > row["published_at"]
            or owner.scheduled_at is not None and row["scheduled_at"] != owner.scheduled_at
            or first is not None and (
                not finite(first) or not owner.admitted_at <= first <= row["published_at"]
                or owner.first_at is None or owner.first_at > first)
            or terminal is not None and (
                first is None or not finite(terminal) or not first <= terminal <= row["published_at"]
                or row.get("status") != "complete" or row.get("http_status") != 200
                or row.get("stream_done") is not True or row.get("stream_error") is not False
                or row.get("finish_reason") not in ("stop", "length")
                or not isinstance(row.get("usage"), dict)
                or type(row["usage"].get("completion_tokens")) is not int
                or not 1 <= row["usage"]["completion_tokens"] <= owner.max_tokens)
            or terminal is None and row.get("finish_reason") is not None
        ):
            raise ValueError("http_first_terminal_visibility")
        previous = endpoints.get(identifier)
        if previous and (
            row["scheduled_at"] != previous["scheduled_at"]
            or previous.get("first_content_at") is not None
            and first != previous["first_content_at"]
            or previous.get("terminal_at") is not None
            and (terminal != previous["terminal_at"] or row.get("usage") != previous.get("usage")
                 or row.get("finish_reason") != previous.get("finish_reason"))
        ):
            raise ValueError("http_endpoint_regression")
        endpoints[identifier] = row
    return endpoints


def public_state(snapshot, *, now, parameters, run_id, clock_domain,
                 native_frames=(), http_observations=(), native_run_id=None):
    from mpc_lab.runtime import snapshot_to_dict

    document = snapshot_to_dict(snapshot)
    document.update(capabilities=["mpc-budget-cas-expiry-v1"],
                    cost_spec=asdict(parameters.cost_spec))
    qualified = qualify_snapshot(document, parameters, run_id=run_id,
                                 clock_domain=clock_domain, now=now)
    owners = {m.request_id: m for m in qualified.requests}
    complete, progress, endpoints = not owners, {}, {}
    if owners or native_run_id is not None or native_frames or http_observations:
        try:
            progress = _received_native_progress(
                native_frames, cut=qualified.sampled_at, now=now, domain=clock_domain,
                native_run_id=native_run_id, owners=owners,
            )
            endpoints = _received_http_endpoints(
                http_observations, cut=qualified.sampled_at, domain=clock_domain, owners=owners,
            )
            complete = set(progress) == set(owners)
        except (ValueError, TypeError, KeyError):
            complete, progress, endpoints = False, {}, {}
    requests = []
    for mark in qualified.requests:
        native, http = progress.get(mark.request_id), endpoints.get(mark.request_id, {})
        output = native["output_tokens"] if native is not None else None
        first, terminal = http.get("first_content_at"), http.get("terminal_at")
        if (native is None or output > 0 and first is None
                or output == 0 and (mark.first_at is not None or first is not None)
                or mark.first_at is not None and first is None
                or native is not None and not native["visible_in_latest_frame"] and terminal is None
                or terminal is not None and (
                    output != http["usage"]["completion_tokens"]
                    or native["sampled_at"] > terminal)):
            complete = False
        # Unknown counters stay None only on incomplete states, never imputed zero.
        requests.append(PublicRequest(
            mark.request_id, mark.prompt_tokens, mark.max_tokens, mark.cost,
            http.get("scheduled_at", mark.received_at if mark.scheduled_at is None
                     else mark.scheduled_at),
            mark.admitted_at, first, output, terminal, http.get("finish_reason"),
            native["sampled_at"] if native is not None else None,
        ))
    return PublicState(
        now, qualified.budget, qualified.budget_version,
        tuple(requests),
        run_id=run_id, clock_domain=clock_domain, sampled_at=qualified.sampled_at,
        complete=complete,
    )


def audit_owner_log(raw, status, *, run_id, clock_domain, identity, request_costs):
    """Replay written original charges, with the unchanged producer tolerance."""
    if not isinstance(raw, bytes) or not isinstance(status, dict):
        raise ValueError("owner_receipt_type")
    if not isinstance(request_costs, dict) or any(
        not isinstance(key, str) or re.fullmatch("[0-9a-f]{32}", key) is None
        or not finite(value, positive=True) for key, value in request_costs.items()
    ):
        raise ValueError("original_uuid_cost_inventory")
    expected_status = {
        "run_id": run_id, "status": "closed", "dropped": 0, "write_errors": 0,
        "mpc_ledger_schema_version": 2, "mpc_remaining_requests": 0,
        "mpc_unflushed_records": 0, "mpc_ledger_complete": True,
        "telemetry_file_sha256": hashlib.sha256(raw).hexdigest(),
        "telemetry_file_bytes": len(raw),
    }
    for key, value in expected_status.items():
        if status.get(key) != value or type(status.get(key)) is not type(value):
            raise ValueError("owner_status:" + key)
    closed_at = status.get("closed_at")
    if not finite(closed_at) or closed_at < 0:
        raise ValueError("owner_closed_clock")
    owners, seen = {}, set()
    sequence, previous_at, version, budget = 0, 0.0, 0, None
    first_content = set()
    for line in raw.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("run_id") != run_id:
            raise ValueError("foreign_written_row")
        if row.get("event") != "mpc_ledger":
            continue
        sequence += 1
        at = row.get("effective_at")
        if (
            type(row.get("sequence")) is not int or row["sequence"] != sequence
            or row.get("ledger_schema_version") != 2 or row.get("complete") is not True
            or row.get("clock_domain") != clock_domain or row.get("identity") != asdict(identity)
            or not finite(at) or not previous_at <= at <= closed_at
        ):
            raise ValueError("owner_row_identity_sequence_clock")
        previous_at = at
        kind = row.get("change_kind")
        identifier, cost = row.get("request_id"), row.get("request_cost")
        if kind == "admission":
            if (
                identifier not in request_costs or identifier in seen
                or not finite(cost, positive=True) or cost != request_costs[identifier]
            ):
                raise ValueError("owner_admission")
            seen.add(identifier)
            owners[identifier] = cost
        elif kind in ("first_content", "terminal"):
            if (
                not isinstance(identifier, str) or identifier not in owners
                or not finite(cost, positive=True) or cost != owners[identifier]
            ):
                raise ValueError("owner_missing_or_repriced")
            if kind == "first_content":
                if identifier in first_content:
                    raise ValueError("duplicate_first_content")
                first_content.add(identifier)
            else:
                del owners[identifier]
        elif kind != "control":
            raise ValueError("owner_change_kind")
        elif identifier is not None or cost is not None:
            raise ValueError("control_row_has_owner")
        expected_version = version + (kind == "control")
        if type(row.get("budget_version")) is not int or row["budget_version"] != expected_version:
            raise ValueError("owner_budget_version")
        version = expected_version
        reserved = row.get("reserved_cost")
        if (
            row.get("budget") not in (16.0, 32.0)
            or type(row.get("running")) is not int or row["running"] != len(owners)
            or not finite(reserved) or reserved < 0
            or not math.isclose(reserved, math.fsum(owners.values()), rel_tol=1e-12, abs_tol=1e-9)
        ):
            raise ValueError("owner_stock")
        if budget is not None and kind != "control" and row["budget"] != budget:
            raise ValueError("owner_budget_changed_without_control")
        budget = row["budget"]
    if owners or seen != set(request_costs):
        raise ValueError("owner_inventory_not_closed")
    for key in ("mpc_producer_final_sequence", "mpc_final_sequence", "mpc_ledger_rows_written"):
        if type(status.get(key)) is not int or status[key] != sequence:
            raise ValueError("owner_suffix:" + key)
    return {
        "closed": True, "owners": len(seen), "ledger_rows": sequence,
        "sha256": expected_status["telemetry_file_sha256"],
        "physical_release": False, "authenticated": False, "crash_durable": False,
    }


class CpuControl:
    """Only in-process ASGI; the existing actuator owns all CAS/journal logic."""

    def __init__(self, app, parameters, pending_path, *, permit=None, headers=None,
                 transport=None):
        qualify_source()
        transport = httpx.ASGITransport(app=app) if transport is None else transport
        if not isinstance(transport, httpx.ASGITransport):
            raise ValueError("cpu_requires_asgi_transport")
        endpoints = {getattr(route, "path", None): getattr(route, "endpoint", None)
                     for route in getattr(app, "routes", ())}
        endpoint = endpoints.get("/admin/mpc/budget")
        if (
            getattr(endpoint, "__name__", None) != "mpc_update_budget"
            or Path(endpoint.__code__.co_filename).resolve() != ROOT / "gateway.py"
            or transport.app is not app
        ):
            raise ValueError("actual_staged_gateway_required")
        self.client = httpx.AsyncClient(transport=transport, base_url="http://asgi.invalid",
                                       headers=headers, follow_redirects=False, trust_env=False)
        self.driver = GuardedActuator(
            self.client, parameters, pending_path, permit=permit,
            enabled=permit is not None, max_age_s=0.5, command_ttl_s=0.1,
        )
        self.parameters = parameters
        self.permit = permit

    async def aclose(self):
        await self.client.aclose()

    async def snapshot(self):
        return await self.driver._state()

    async def assign(self, snapshot, budget):
        if not isinstance(self.permit, IdentificationPermit):
            return {"status": "not_authorized", "reason": "identification_permit_required"}
        decision = Decision(budget, None, "identification_assignment", False,
                            None, None, 0.0, 0, 0, self.parameters.identity)
        return await self.driver.submit(snapshot, decision)

    async def submit(self, snapshot, choice):
        if isinstance(self.permit, IdentificationPermit):
            return {"status": "not_authorized", "reason": "mpc_through_identification_forbidden"}
        if self.parameters.model_version == "native-id-unfitted-v1":
            return {"status": "not_authorized", "reason": "unfitted_model_forbidden"}
        if (
            not isinstance(choice, Choice) or choice.status != "ok"
            or not choice.plan or choice.budget != choice.plan[0]
            or any(action not in (16.0, 32.0) for action in choice.plan)
            or not finite(choice.elapsed_s) or not 0 <= choice.elapsed_s <= 0.05
        ):
            return {"status": "blocked", "reason": "failed_or_unsupported_choice"}
        decision = Decision(choice.budget, Plan(choice.plan), "cpu_optimized", False,
                            None, None, choice.elapsed_s, 0, 0, self.parameters.identity)
        return await self.driver.submit(snapshot, decision)

    async def reconcile(self):
        return await self.driver.reconcile()
