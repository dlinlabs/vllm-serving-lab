"""Bounded Gateway-only nominal coordinator, GET-only unless separately enabled.

CapApplied means a guarded Gateway admission-budget ACK, never Native token
application. The precommit GET is not ledger CAS. Planner read_state returns
the last actually GET-observed version, not a simultaneous remote observation.

Use the same journal across restarts. Its count and episode deadline never
reset; deleting/replacing it is outside this trusted-operator contract.
Clock, planner and HTTP deadlines are cooperative, not OS hard preemption.
Fixed/FIFO benchmark setup is deliberately absent: a strong Fixed baseline
must install its frozen observed-TRAIN cap before offers, not use this loop.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _directory in (ROOT, HERE):
    _path = str(_directory)
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import httpx

from mpc_lab.contracts import (
    CostSpec, Identity, ModelParameters, SCHEMA, Scenario, Snapshot, digest, finite,
    implementation_hash,
)
from mpc_lab.controller import validate_snapshot
from mpc_lab.runtime import require_guarded_capability, snapshot_from_dict
from prefill_lab.live_http import clock_domain
from nominal_permit import NominalDemoPermit, SCOPE, source_digest
from nominal_planner import (
    BUDGETS, CONTRACT_SHA256, MAX_OFFERS, OBJECTIVE_ID, PLANNER_VERSION, CausalScenario,
    DecisionWindow, FirstAction, NominalDecision, NominalPlanner, StateVersion, TimedPlan,
)

JOURNAL_SCHEMA = "resume-p0-nominal-journal-v1"
MAX_BODY = 256 * 1024
MAX_JOURNAL = 8 * 1024 * 1024
MAX_EVENTS = 256


class RuntimeRefused(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise RuntimeRefused(reason)


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_JSON_key")
            result[key] = value
        return result

    def constant(value):
        raise RuntimeRefused("nonfinite_JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def module_paths():
    paths = {"nominal_runtime": str(Path(__file__).resolve())}
    expected = {
        "mpc_lab.contracts": ROOT / "mpc_lab/contracts.py",
        "mpc_lab.controller": ROOT / "mpc_lab/controller.py",
        "mpc_lab.model": ROOT / "mpc_lab/model.py",
        "mpc_lab.runtime": ROOT / "mpc_lab/runtime.py",
        "prefill_lab.live_http": ROOT / "prefill_lab/live_http.py",
        **{name: HERE / (name + ".py") for name in
           ("nominal_permit", "nominal_planner", "nominal_scenario", "nominal_workload")},
    }
    for name, path in expected.items():
        if name not in sys.modules and name in ("nominal_scenario", "nominal_workload"):
            continue
        actual = Path(sys.modules[name].__file__).resolve()
        _require(actual == path, "module_outside_frozen_root:" + name)
        paths[name] = str(actual)
    return paths


@dataclass(frozen=True)
class RuntimeLimits:
    iterations: int = 4
    episode_s: float = 4.0
    decision_s: float = 0.5
    solve_s: float = 0.1
    poll_s: float = 0.0

    def __post_init__(self):
        _require(type(self.iterations) is int and 1 <= self.iterations <= 32, "iteration_limit")
        _require(all(finite(value) for value in (self.episode_s, self.decision_s, self.solve_s, self.poll_s))
                 and 0 < self.episode_s <= 60 and 0 < self.solve_s <= 0.1
                 and self.solve_s < self.decision_s <= 0.5 and 0 <= self.poll_s <= 1,
                 "invalid_runtime_deadlines")


def _protocol(parameters, policy, limits, scenario_binding, endpoint, journal_path, start_at=None):
    module_paths()
    endpoint = httpx.URL(endpoint)
    _require(endpoint.scheme in ("http", "https") and endpoint.host
             and not endpoint.username and not endpoint.password
             and endpoint.path in ("", "/") and not endpoint.query and not endpoint.fragment,
             "explicit_HTTP_root_endpoint_required")
    _require(type(parameters) is ModelParameters and type(limits) is RuntimeLimits
             and policy in ("mpc", "one_move"), "typed_nominal_protocol_required")
    _require(start_at is None or finite(start_at) and start_at >= 0, "invalid_shared_start")
    value = {
        "scope": SCOPE, "runtime": JOURNAL_SCHEMA, "planner_contract": CONTRACT_SHA256,
        "objective": OBJECTIVE_ID, "parameters": asdict(parameters), "policy": policy,
        "limits": asdict(limits), "scenario_binding": scenario_binding,
        "endpoint": str(endpoint.copy_with(path="/")), "journal_path": str(Path(journal_path).absolute()),
        "max_commands": 8, "source_age_s": 0.5, "command_ttl_s": 0.1,
        "precommit_observation_equality": True, "ledger_CAS": False,
        "Fixed_FIFO": "separate_static_preoffer_setup_not_implemented",
    }
    if start_at is not None:
        value["start_at"] = start_at
        value["CLI_start_envelope_s"] = 60.0
    return _strict(json.dumps(value, allow_nan=False))


def describe_protocol(planner, journal_path, *, endpoint, scenario_binding,
                      policy="mpc", limits=RuntimeLimits(), start_at=None):
    """Describe actual frozen bindings without a client, journal write or authority."""
    _require(type(planner) is NominalPlanner, "typed_nominal_planner_required")
    protocol = _protocol(planner.parameters, policy, limits, scenario_binding, endpoint, journal_path, start_at)
    directory = Path(__file__).resolve().parent
    names = ("nominal_permit.py", "nominal_planner.py", "nominal_runtime.py",
             "nominal_scenario.py", "nominal_workload.py")
    bundle = {
        "public_implementation": implementation_hash(),
        "nominal": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names},
    }
    source = source_digest()
    _require(digest(bundle) == source, "source_bundle_description_mismatch")
    return {
        "scope": SCOPE, "mode": "protocol_description_only",
        "protocol": protocol, "protocol_sha256": digest(protocol),
        "source_sha256": source, "source_bundle": bundle,
        "expected_identity": asdict(planner.identity), "clock_domain": clock_domain(),
        "module_paths": module_paths(),
        "not_an_approval": True, "HTTP_performed": False, "Native_qualified": False,
    }


class _Journal:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.partial = self.path.with_name(self.path.name + ".partial")

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        descriptor = os.open(self.path.with_name(self.path.name + ".lock"),
                             os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _require(not self.partial.exists() and not self.partial.is_symlink(),
                     "unclean_journal_commit_preserved")
            yield
        finally:
            os.close(descriptor)

    def read(self):
        if not self.path.exists():
            return None
        _require(not self.path.is_symlink() and self.path.stat().st_size <= MAX_JOURNAL,
                 "journal_file_invalid")
        value = _strict(self.path.read_bytes())
        keys = {"schema", "scope", "controller_id", "identity", "clock_domain", "protocol_sha256",
                "run_id", "episode_started_at", "episode_deadline", "issued_commands",
                "source_sha256", "pending", "events"}
        _require(type(value) is dict and set(value) == keys and value["schema"] == JOURNAL_SCHEMA
                 and value["scope"] == SCOPE and type(value["controller_id"]) is str
                 and len(value["controller_id"]) == 32
                 and type(value["issued_commands"]) is int and 0 <= value["issued_commands"] <= 8
                 and type(value["events"]) is list and len(value["events"]) <= MAX_EVENTS
                 and all(type(event) is dict and type(event.get("event")) is str
                         for event in value["events"])
                 and _sha(value["protocol_sha256"]), "journal_invalid")
        _require(all(finite(value[key]) for key in ("episode_started_at", "episode_deadline"))
                 and 0 <= value["episode_started_at"] < value["episode_deadline"]
                 <= value["episode_started_at"] + 60, "journal_episode_invalid")
        prepared = [event for event in value["events"] if event.get("event") == "PendingPrepared"]
        _require([event.get("ordinal") for event in prepared]
                 == list(range(1, value["issued_commands"] + 1)), "journal_count_invalid")
        pending = value["pending"]
        resolved = [event["pending"]["ordinal"] for event in value["events"]
                    if event["event"] in ("CapReobserved", "UnknownFenced")]
        _require(resolved == list(range(1, value["issued_commands"] + (pending is None))),
                 "journal_unresolved_command_cannot_disappear")
        if pending is not None:
            _require(type(pending) is dict and set(pending) == {"command", "ordinal", "ack"}
                     and pending["ordinal"] == value["issued_commands"] > 0
                     and prepared[-1]["command"] == pending["command"], "journal_pending_invalid")
            command = pending["command"]
            _require(set(command) == {"schema", "run_id", "clock_domain", "identity",
                                     "expected_version", "operation_id", "budget", "not_after"}
                     and command["schema"] == SCHEMA and command["run_id"] == value["run_id"]
                     and command["identity"] == value["identity"]
                     and command["clock_domain"] == value["clock_domain"]
                     and command["operation_id"] == f"nominal-{value['controller_id']}-{pending['ordinal']}"
                     and type(command["expected_version"]) is int and command["expected_version"] >= 0
                     and finite(command["budget"]) and command["budget"] in BUDGETS
                     and finite(command["not_after"])
                     and value["episode_started_at"] < command["not_after"] <= value["episode_deadline"]
                     and _sha(value["source_sha256"]), "journal_pending_binding")
            ack = pending["ack"]
            if ack is not None:
                _require(type(ack) is dict and ack.get("schema") == SCHEMA
                         and ack.get("status") == "applied"
                         and all(ack.get(key) == command[key] for key in
                                 ("run_id", "clock_domain", "operation_id", "budget"))
                         and type(ack.get("budget_version")) is int
                         and ack["budget_version"] == command["expected_version"] + 1
                         and finite(ack.get("applied_at"))
                         and prepared[-1]["not_before"] <= ack["applied_at"] < command["not_after"]
                         and any(event["event"] == "CapApplied" and event.get("acknowledgment") == ack
                                 for event in value["events"]), "journal_ACK_invalid")
        return value

    def save(self, value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        _require(len(raw) <= MAX_JOURNAL and len(value["events"]) <= MAX_EVENTS,
                 "journal_capacity_preserve_and_stop")
        fd = os.open(self.partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.partial, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class NominalRuntime:
    """Caller owns the AsyncClient; run() owns only its bounded journal lock."""

    def __init__(
        self, client: httpx.AsyncClient, planner: NominalPlanner,
        scenario_factory: Callable[[Snapshot], CausalScenario] | None, journal_path: Path, *,
        admin_key: str, scenario_binding: dict, permit: NominalDemoPermit | None = None,
        enabled: bool = False, policy: str = "mpc", limits: RuntimeLimits = RuntimeLimits(),
        parameters: ModelParameters | None = None, clock: Callable[[], float] = time.monotonic,
        sleeper=asyncio.sleep, start_at: float | None = None,
    ):
        _require(type(enabled) is bool and policy in ("mpc", "one_move"),
                 "explicit_mode_required_static_baselines_not_per_tick")
        _require(type(planner) is NominalPlanner and type(limits) is RuntimeLimits, "typed_nominal_planner_required")
        parameters = planner.parameters if parameters is None else parameters
        _require(type(parameters) is ModelParameters and parameters == planner.parameters
                 and parameters.identity == planner.identity, "exact_model_parameter_binding")
        _require(isinstance(admin_key, str) and bool(admin_key), "admin_header_required")
        _require(client.follow_redirects is False and client.base_url.scheme in ("http", "https")
                 and not client.base_url.username and not client.base_url.password
                 and client.base_url.path == "/" and not client.base_url.query and not client.base_url.fragment,
                 "explicit_nonredirecting_HTTP_endpoint_required")
        self.client, self.planner, self.scenario_factory = client, planner, scenario_factory
        self.parameters, self.identity = parameters, planner.identity
        self.permit, self.enabled, self.policy, self.limits = permit, enabled, policy, limits
        self.admin_key, self.clock, self.sleeper = admin_key, clock, sleeper
        self.start_at = start_at
        self.store = _Journal(journal_path)
        self.protocol = _protocol(parameters, policy, limits, scenario_binding, client.base_url,
                                  self.store.path, start_at)
        self.protocol_sha256 = digest(self.protocol)
        self.state = None
        self.observed = None
        self.last_clock = None

    def _binding(self):
        _require(self.planner.parameters == self.parameters and self.planner.identity == self.identity
                 and digest(asdict(self.parameters)) == digest(self.protocol["parameters"])
                 and asdict(self.limits) == self.protocol["limits"]
                 and self.policy == self.protocol["policy"]
                 and self.start_at == self.protocol.get("start_at")
                 and str(self.store.path) == self.protocol["journal_path"]
                 and str(self.client.base_url) == self.protocol["endpoint"]
                 and self.client.follow_redirects is False
                 and digest(self.protocol) == self.protocol_sha256,
                 "runtime_configuration_or_parameter_binding_changed")

    def _now(self):
        now = self.clock()
        _require(finite(now) and now >= 0 and (self.last_clock is None or now >= self.last_clock),
                 "nonmonotonic_local_clock")
        self.last_clock = now
        return now

    def _remaining(self, deadline):
        remaining = deadline - self._now()
        _require(remaining > 0, "absolute_entry_or_episode_deadline")
        return remaining

    def _record(self, event, **values):
        self.state["events"].append({"event": event, "at": self.clock(), **_plain(values)})
        self.store.save(self.state)

    def _version(self):
        _require(self.observed is not None, "no_actual_GET_observation")
        return StateVersion.from_snapshot(self.observed)

    async def _wire(self, method, path, deadline, *, command=None, body=None):
        started = self._now()
        remaining = self._remaining(deadline)
        chunks, size, stored, chunk_count, status, complete, error = [], 0, 0, 0, None, False, None
        arguments = {"headers": {"x-admin-api-key": self.admin_key}, "timeout": min(0.2, remaining)}
        if command is not None:
            _require(type(body) is bytes, "preencoded_durable_command_required")
            arguments["content"] = body
            arguments["headers"]["content-type"] = "application/json"
        try:
            async with asyncio.timeout(min(0.2, remaining)):
                async with self.client.stream(method, path, **arguments) as response:
                    status = response.status_code
                    async for chunk in response.aiter_bytes():
                        chunk_count += 1
                        _require(chunk_count <= 4096, "HTTP_chunk_limit")
                        size += len(chunk)
                        part = chunk[:max(0, MAX_BODY - stored)]
                        chunks.append(part)
                        stored += len(part)
                        _require(size <= MAX_BODY, "HTTP_body_limit")
                        self._remaining(deadline)
                    complete = True
        except BaseException as exc:
            error = exc
        raw = b"".join(chunks)
        saved = raw.replace(self.admin_key.encode(), b"[REDACTED_AUTH]")
        self._record("HTTP", method=method, path=path, status_code=status,
                     started_at=started, received_at=self.clock(), bytes_received=size,
                     response_complete=complete, redacted=saved != raw,
                     received_prefix_sha256=hashlib.sha256(raw).hexdigest(),
                     body_b64=base64.b64encode(saved).decode(),
                     error_type=type(error).__name__ if error else None,
                     operation_id=command["operation_id"] if command else None)
        if error is not None:
            raise error
        self._remaining(deadline)
        return status, raw

    async def _state(self, deadline, *, entered_at):
        get_entered = self._now()
        status, raw = await self._wire("GET", "/admin/mpc/state", deadline)
        _require(status == 200, "state_HTTP_status")
        document = _strict(raw)
        require_guarded_capability(document)
        _require(isinstance(document.get("requests"), (list, tuple))
                 and len(document["requests"]) <= 512, "ledger_capacity")
        snapshot = snapshot_from_dict(document)
        local_domain = clock_domain()
        reason = validate_snapshot(snapshot, now=self._now(), clock_domain=local_domain,
                                   expected_identity=self.identity, max_age_s=0.5)
        _require(reason is None, reason or "invalid_snapshot")
        _require(snapshot.sampled_at >= max(get_entered, entered_at), "snapshot_predates_current_GET_entry")
        _require(snapshot.budget in BUDGETS, "unsupported_observed_budget")
        _require(type(document.get("cost_spec")) is dict, "missing_cost_spec")
        cost = CostSpec(**document["cost_spec"])
        _require(cost.fingerprint == self.parameters.cost_spec.fingerprint == snapshot.identity.cost_hash,
                 "cost_spec_fingerprint_mismatch")
        for mark in snapshot.requests:
            _require(mark.cost == self.parameters.cost_spec.estimate(mark.prompt_tokens, mark.max_tokens),
                     "immutable_ledger_cost_mismatch")
        _require(local_domain == self.state["clock_domain"], "local_clock_domain_changed")
        if self.state["run_id"] is None:
            self.state["run_id"] = snapshot.run_id
            self._record("RunBound", run_id=snapshot.run_id)
        _require(snapshot.run_id == self.state["run_id"], "gateway_run_changed")
        self.observed = snapshot
        self._remaining(deadline)
        return snapshot

    def _reconcile(self, snapshot):
        pending = self.state["pending"]
        if pending is None:
            return None
        command, ack = pending["command"], pending["ack"]
        if (snapshot.run_id != command["run_id"] or snapshot.clock_domain != command["clock_domain"]
                or asdict(snapshot.identity) != command["identity"]):
            return {"status": "pending", "reason": "identity_changed_no_fence"}
        version = snapshot.budget_version
        if version < command["expected_version"]:
            return {"status": "pending", "reason": "version_regression_no_fence"}
        if ack is not None:
            if version < ack["budget_version"] or snapshot.sampled_at < ack["applied_at"]:
                return {"status": "pending", "reason": "ACK_requires_real_reobservation"}
            if version == ack["budget_version"] and snapshot.budget != command["budget"]:
                return {"status": "pending", "reason": "ACK_observation_inconsistent"}
            status, event = "applied_reobserved", "CapReobserved"
        elif version > command["expected_version"]:
            status, event = "unknown_fenced_by_CAS", "UnknownFenced"
        elif snapshot.sampled_at >= command["not_after"]:
            status, event = "unknown_fenced_by_server_expiry", "UnknownFenced"
        else:
            return {"status": "pending", "reason": "unchanged_GET_is_not_a_fence"}
        self.state["pending"] = None
        self._record(event, outcome=status, pending=pending, observed=asdict(snapshot),
                     application_known_from_ACK=ack is not None, native_token_application_at=None)
        return {"status": status, "operation_id": command["operation_id"],
                "cap_application_known": ack is not None, "native_token_application_at": None}

    def _authorize(self, snapshot, budget, entered, deadline, *, issued):
        self._binding()
        if not self.enabled or type(self.permit) is not NominalDemoPermit:
            return "nominal_not_authorized"
        current_source = source_digest()
        reason = self.permit.refusal_reason(
            snapshot, enabled=self.enabled, expected_identity=self.identity,
            current_source_sha256=current_source, protocol_sha256=self.protocol_sha256,
            current_clock_domain=clock_domain(), budget=budget, entered_at=entered,
            deadline=deadline, now=self._now(), issued_commands=issued,
        )
        if reason is None and self.state["source_sha256"] not in (None, current_source):
            return "journal_source_binding_changed"
        if reason is None:
            self.state["source_sha256"] = current_source
        return reason

    def _decision(self, snapshot, window):
        self._binding()
        planning_started = self._now()
        source = self.scenario_factory(snapshot)
        _require(type(source) is CausalScenario, "typed_causal_scenario_required")
        _require(type(source.scenario) is Scenario and type(source.scenario.offers) is tuple
                 and len(source.scenario.offers) <= MAX_OFFERS, "bounded_causal_scenario_required")
        scenario = _plain(source)
        scenario_bytes = json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        _require(len(scenario_bytes) <= MAX_BODY, "causal_receipt_byte_limit")
        scenario_sha256 = hashlib.sha256(scenario_bytes).hexdigest()
        snapshot_sha256 = digest(asdict(snapshot))
        self._record("PlanningInput", planning_started_at=planning_started,
                     snapshot_sha256=snapshot_sha256, scenario=scenario,
                     scenario_sha256=scenario_sha256, window=window)
        decision = self.planner.plan(snapshot, source, window, policy=self.policy,
                                     clock_domain=clock_domain(), read_state=self._version, clock=self._now)
        _require(self._now() < window.solve_deadline, "planner_finished_after_original_solve_deadline")
        _require(type(decision) is NominalDecision and decision.policy == self.policy
                 and decision.contract_sha256 == CONTRACT_SHA256 and decision.objective_id == OBJECTIVE_ID
                 and decision.planner_version == PLANNER_VERSION and decision.evaluated_paths == 8,
                 "typed_nominal_decision_contract")
        action, plan = decision.first_action, decision.plan
        _require(type(action) is FirstAction and type(plan) is TimedPlan
                 and type(action.source) is StateVersion
                 and action.source == StateVersion.from_snapshot(snapshot)
                 and type(plan.actions) is tuple and len(plan.actions) == 3
                 and all(finite(value) and value in BUDGETS for value in plan.actions)
                 and action.budget == plan.actions[0] and finite(action.budget)
                 and plan.received_budget == snapshot.budget and plan.sampled_at == snapshot.sampled_at
                 and plan.first_action_at == action.not_before == window.first_action_at
                 and action.not_after == window.action_deadline, "first_action_source_or_time_mismatch")
        self._record("Planned", planning_started_at=planning_started, selected_at=self._now(),
                     snapshot_sha256=snapshot_sha256, scenario=scenario,
                     scenario_sha256=scenario_sha256, decision=_plain(decision),
                     suffix_execution_permitted=False)
        return decision

    async def _step(self, entered, deadline):
        snapshot = await self._state(deadline, entered_at=entered)
        if self.state["pending"] is not None:
            return self._reconcile(snapshot)
        if self.scenario_factory is None:
            _require(not self.enabled, "causal_scenario_required_for_demo")
            return {"status": "GET_only", "reason": "no_scenario_no_planning", "cap_change_applied": False}
        solve_deadline = entered + self.limits.solve_s
        _require(self._now() < solve_deadline < deadline, "entry_GET_exhausted_solver_window")
        window = DecisionWindow(entered, solve_deadline, solve_deadline,
                                min(deadline, snapshot.sampled_at + 0.5))
        decision = self._decision(snapshot, window)
        self._remaining(deadline)
        action = decision.first_action
        if action.budget == snapshot.budget:
            return {"status": "hold", "cap_change_applied": False, "actual_cap_applied_at": None}
        before = self.state["issued_commands"]
        reason = self._authorize(snapshot, action.budget, entered, deadline, issued=before)
        if reason:
            return {"status": "GET_only" if not self.enabled else "refused", "reason": reason,
                    "cap_change_applied": False}
        delay = action.not_before - self._now()
        if delay > 0:
            _require(delay < self._remaining(deadline), "first_action_not_yet_permitted")
            await self.sleeper(delay)
        _require(self._now() >= action.not_before, "first_action_before_not_before")
        current = await self._state(deadline, entered_at=entered)
        _require(StateVersion.from_snapshot(current) == StateVersion.from_snapshot(snapshot)
                 and current.requests == snapshot.requests, "precommit_observation_changed")
        reason = self._authorize(snapshot, action.budget, entered, deadline, issued=before)
        _require(reason is None, reason or "nominal_authorization")
        now = self._now()
        not_after = min(deadline, self.permit.expires_at, snapshot.sampled_at + 0.5, now + 0.1)
        _require(now < not_after <= action.not_after, "first_action_expired")
        command = {
            "schema": SCHEMA, "run_id": snapshot.run_id, "clock_domain": snapshot.clock_domain,
            "identity": asdict(self.identity), "expected_version": snapshot.budget_version,
            "operation_id": f"nominal-{self.state['controller_id']}-{before + 1}",
            "budget": action.budget, "not_after": not_after,
        }
        body = json.dumps(command, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.state["issued_commands"] = before + 1
        self.state["pending"] = {"command": command, "ordinal": before + 1, "ack": None}
        self._record("PendingPrepared", ordinal=before + 1, command=command,
                     entered_at=entered, deadline=deadline, not_before=action.not_before,
                     exact_POST_body_sha256=hashlib.sha256(body).hexdigest())
        # Count includes this durable reservation. Recheck that same reservation,
        # not authorization for a second command, after fsync consumed real time.
        reason = self._authorize(snapshot, action.budget, entered, deadline, issued=before)
        _require(reason is None, reason or "nominal_authorization_after_persist")
        self._remaining(not_after)
        post_started = self._now()
        status, raw = await self._wire("POST", "/admin/mpc/budget", deadline, command=command, body=body)
        reply = _strict(raw)
        valid = (
            status == 200 and type(reply) is dict and reply.get("schema") == SCHEMA
            and reply.get("status") == "applied" and reply.get("operation_id") == command["operation_id"]
            and reply.get("run_id") == command["run_id"] and reply.get("clock_domain") == command["clock_domain"]
            and type(reply.get("budget_version")) is int
            and reply["budget_version"] == command["expected_version"] + 1
            and finite(reply.get("budget")) and reply["budget"] == command["budget"]
            and finite(reply.get("previous_budget")) and reply["previous_budget"] == snapshot.budget
            and finite(reply.get("applied_at"))
            and post_started <= reply["applied_at"] <= self._now()
            and reply["applied_at"] < not_after
        )
        _require(valid, "unknown_POST_outcome_no_fallback")
        self.state["pending"]["ack"] = reply
        self._record("CapApplied", acknowledgment=reply, actual_cap_change=True,
                     scope="Gateway_admission_budget_only", native_token_application_at=None)
        observed = await self._state(deadline, entered_at=entered)
        return self._reconcile(observed)

    async def run(self):
        call_entered = self._now()
        steps, failure = [], None
        try:
            self._binding()
            with self.store.locked():
                self.state = self.store.read()
                domain = clock_domain()
                if self.state is None:
                    _require(self.start_at is None or call_entered >= self.start_at, "shared_start_not_reached")
                    self.state = {
                        "schema": JOURNAL_SCHEMA, "scope": SCOPE, "controller_id": uuid.uuid4().hex,
                        "identity": asdict(self.identity), "clock_domain": domain,
                        "protocol_sha256": self.protocol_sha256, "run_id": None,
                        "episode_started_at": call_entered, "episode_deadline": call_entered + self.limits.episode_s,
                        "issued_commands": 0, "source_sha256": None, "pending": None, "events": [],
                    }
                    self._record("ProtocolBound", protocol=self.protocol, module_paths=module_paths())
                _require(self.state["identity"] == asdict(self.identity)
                         and self.state["protocol_sha256"] == self.protocol_sha256
                         and self.state["clock_domain"] == domain, "journal_identity_protocol_or_clock_changed")
                for index in range(self.limits.iterations):
                    entered = call_entered if index == 0 else self._now()
                    deadline = min(entered + self.limits.decision_s, self.state["episode_deadline"])
                    if self._now() >= deadline:
                        failure = ("persistent_episode_deadline_exhausted"
                                   if self._now() >= self.state["episode_deadline"]
                                   else "absolute_entry_deadline_exhausted")
                        break
                    try:
                        row = await self._step(entered, deadline)
                    except asyncio.CancelledError:
                        self._record("Cancelled", pending_unknown=self.state["pending"] is not None)
                        raise
                    except OSError:
                        raise
                    except Exception as error:
                        pending = self.state["pending"]
                        ack = pending["ack"] if pending else None
                        row = {"status": "pending" if self.state["pending"] else "refused",
                               "reason": str(error).replace(self.admin_key, "[REDACTED_AUTH]")[:1024],
                               "error_type": type(error).__name__,
                               "cap_change_applied": True if ack else (None if pending else False),
                               "actual_cap_applied_at": ack["applied_at"] if ack else None,
                               "native_token_application_at": None}
                    row.update(entered_at=entered, deadline=deadline, outcome_observed_at=self._now())
                    self._record("StepFinished", **row)
                    row["finished_at"] = self._now()
                    row["elapsed_s"] = row["finished_at"] - entered
                    row["deadline_met"] = row["finished_at"] <= deadline
                    steps.append(row)
                    if index + 1 < self.limits.iterations and self.limits.poll_s:
                        remaining = self.state["episode_deadline"] - self._now()
                        if remaining <= 0:
                            break
                        await self.sleeper(min(self.limits.poll_s, remaining))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = type(error).__name__ + ":" + str(error).replace(self.admin_key, "[REDACTED_AUTH]")[:1024]
        pending = self.state["pending"] if self.state else None
        return {
            "scope": SCOPE, "mode": "experimental_nominal_demo" if self.enabled else "GET_only",
            "steps": steps, "failure": failure, "issued_commands": self.state["issued_commands"] if self.state else None,
            "pending": self.state["pending"] is not None if self.state else None,
            "pending_outcome": ("unknown" if pending["ack"] is None else "ACK_requires_reobservation")
            if pending else None,
            "episode_deadline": self.state["episode_deadline"] if self.state else None,
            "protocol_sha256": self.protocol_sha256, "ledger_CAS": False,
            "Native_qualified": False, "native_token_application_at": None,
            "objective": OBJECTIVE_ID, "static_Fixed_FIFO_benchmark_implemented": False,
        }


def _load_permit(path):
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536,
             "separate_regular_nominal_permit_required")
    document = _strict(path.read_bytes())
    _require(type(document) is dict and set(document) == {field.name for field in fields(NominalDemoPermit)},
             "nominal_permit_fields")
    document["identity"] = Identity(**document["identity"])
    return NominalDemoPermit(**document)


def load_causal_scenario(path):
    """Load one frozen development/TRAIN-derived forecast, never a replay tape.

    The file's actual digest is protocol-bound. Its cutoff/availability must use
    this host's monotonic clock. The producer's Train-only provenance still needs
    an independent audit; this loader does not infer a forecast from Eval offers.
    """
    from mpc_lab.contracts import FutureOffer, Scenario

    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BODY,
             "regular_bounded_causal_scenario_file_required")
    raw = path.read_bytes()
    _require(len(raw) <= MAX_BODY, "causal_scenario_file_limit")
    document = _strict(raw)
    _require(set(document) == {"clock_domain", "information_cutoff_at", "available_at", "origin", "purpose", "scenario"}
             and document["clock_domain"] == clock_domain()
             and document["origin"] in ("development", "past_observations")
             and document["purpose"] in ("train", "evaluation", "engineering"),
             "causal_scenario_file_clock_or_fields")
    scenario = dict(document["scenario"])
    scenario["offers"] = tuple(FutureOffer(**row) for row in scenario["offers"])
    scenario = Scenario(**scenario)
    fingerprint = hashlib.sha256(raw).hexdigest()

    def factory(snapshot):
        _require(snapshot.clock_domain == document["clock_domain"] == clock_domain(),
                 "causal_scenario_clock_changed")
        return CausalScenario(scenario, StateVersion.from_snapshot(snapshot), fingerprint,
                              document["information_cutoff_at"], document["available_at"],
                              document["origin"], document["purpose"])

    return factory, {"file_sha256": fingerprint, "origin": document["origin"], "purpose": document["purpose"]}


def load_train_scenario(path, *, purpose, clock=time.monotonic):
    """Freeze Train once; bind static content, not the new process's availability."""
    import nominal_scenario as producer

    module_paths()
    _require(purpose in ("train", "evaluation", "engineering"), "explicit_forecast_purpose_required")
    prior = producer.freeze_train_prior(path, purpose="train", clock_domain=clock_domain(), clock=clock)
    content = {key: value for key, value in asdict(prior).items()
               if key not in ("clock_domain", "available_at")}
    sources = {
        "nominal_scenario.py": hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
        "nominal_workload.py": hashlib.sha256(Path(producer.workload.__file__).read_bytes()).hexdigest(),
    }
    binding = {
        "provider": "frozen-Train-nominal-scenario-v1", "purpose": purpose,
        "prior_content": content, "prior_content_sha256": prior.content_sha256,
        "clock_domain": prior.clock_domain, "producer_sources": sources,
        "rng": {
            "namespace": producer.RNG_NAMESPACE, "seed": producer.RNG_SEED,
            "runtime": f"Python {sys.version.split()[0]} stdlib random.Random(str, version=2)/expovariate/randrange",
            "horizon_s": producer.HORIZON_S, "max_offers": producer.MAX_OFFERS,
        },
        "assumptions": producer.ASSUMPTIONS,
    }

    def factory(snapshot):
        return producer.make_scenario(prior, snapshot, purpose=purpose, clock=clock).causal

    return factory, binding


async def wait_for_start(runtime, *, cli_entered):
    """Wait before run() entry; an existing validated journal resumes its old episode."""
    now = runtime._now()
    receipt = {"requested_start_at": runtime.start_at, "cli_entered_at": cli_entered,
               "wait_entered_at": now, "resumed_existing_episode": False}
    if runtime.start_at is None:
        return {**receipt, "wait_finished_at": now}
    runtime._binding()
    if runtime.store.path.exists():
        with runtime.store.locked():
            saved = runtime.store.read()
        _require(saved is not None and saved["identity"] == asdict(runtime.identity)
                 and saved["protocol_sha256"] == runtime.protocol_sha256
                 and saved["clock_domain"] == clock_domain()
                 and runtime.start_at <= saved["episode_started_at"] <= now < saved["episode_deadline"],
                 "shared_start_restart_binding_or_episode_invalid")
        return {**receipt, "wait_finished_at": runtime._now(), "resumed_existing_episode": True}
    end = cli_entered + 60.0
    if runtime.enabled:
        _require(type(runtime.permit) is NominalDemoPermit and runtime.permit.approved is True
                 and finite(runtime.permit.not_before) and finite(runtime.permit.expires_at)
                 and 0 <= runtime.permit.not_before <= runtime.start_at < runtime.permit.expires_at
                 and 0 < runtime.permit.expires_at - runtime.permit.not_before <= 60.0,
                 "shared_start_outside_nominal_permit")
        end = min(end, runtime.permit.expires_at)
    _require(finite(cli_entered) and cli_entered <= now <= runtime.start_at
             and runtime.start_at + runtime.limits.episode_s <= end,
             "shared_start_past_or_outside_envelope")
    async with asyncio.timeout(end - now):
        remaining = runtime.start_at - runtime._now()
        if remaining > 0:
            await runtime.sleeper(remaining)
    finished = runtime._now()
    _require(runtime.start_at <= finished and finished + runtime.limits.episode_s <= end,
             "shared_start_wait_exhausted_envelope")
    return {**receipt, "wait_finished_at": finished, "CLI_envelope_deadline": end}


def main(argv=None):
    cli_entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--journal", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scenario-file", type=Path)
    source.add_argument("--train-tape", type=Path)
    parser.add_argument("--forecast-purpose", choices=("train", "evaluation", "engineering"))
    parser.add_argument("--cost-spec-file", type=Path)
    parser.add_argument("--policy", choices=("mpc", "one_move"), default="mpc")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--describe-protocol", action="store_true")
    parser.add_argument("--start-at", type=float)
    parser.add_argument("--experimental-nominal-demo", action="store_true")
    parser.add_argument("--nominal-permit", type=Path)
    args = parser.parse_args(argv)
    if (args.train_tape is not None) != (args.forecast_purpose is not None):
        parser.error("--train-tape requires an explicit --forecast-purpose, used only with Train")
    if not args.describe_protocol and args.experimental_nominal_demo and (
            args.nominal_permit is None or (args.scenario_file is None and args.train_tape is None)):
        parser.error("experimental POST requires a separate nominal permit and causal scenario file or Train tape")
    cost = CostSpec(**_strict(args.cost_spec_file.read_bytes())) if args.cost_spec_file else CostSpec()
    planner = NominalPlanner(ModelParameters(cost_spec=cost))
    if args.train_tape:
        factory, binding = load_train_scenario(args.train_tape, purpose=args.forecast_purpose)
    else:
        factory, binding = load_causal_scenario(args.scenario_file) if args.scenario_file else (None, {"kind": "observe_only"})
    limits = RuntimeLimits(iterations=args.steps, episode_s=args.episode_seconds, poll_s=args.poll_seconds)
    if args.describe_protocol:
        print(json.dumps(describe_protocol(
            planner, args.journal, endpoint=args.url, scenario_binding=binding,
            policy=args.policy, limits=limits, start_at=args.start_at,
        ), sort_keys=True, allow_nan=False))
        return
    permit = _load_permit(args.nominal_permit) if args.nominal_permit else None
    key = os.environ.get("ADMIN_API_KEY", "")
    _require(bool(key), "ADMIN_API_KEY_required")

    async def execute():
        async with httpx.AsyncClient(base_url=args.url, follow_redirects=False, trust_env=False) as client:
            runtime = NominalRuntime(
                client, planner, factory, args.journal, admin_key=key, scenario_binding=binding,
                permit=permit, enabled=args.experimental_nominal_demo, policy=args.policy,
                limits=limits, start_at=args.start_at,
            )
            startup = await wait_for_start(runtime, cli_entered=cli_entered)
            result = await runtime.run()
            result.update(startup=startup, module_paths=module_paths())
            return result

    print(json.dumps(asyncio.run(execute()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
