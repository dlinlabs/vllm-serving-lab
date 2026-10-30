"""Opt-in guarded transport; qualification is an operator attestation, not proof.

The CLI never enables this client. Pending commands survive crashes; a GET only
fences a write if its same-run CAS version advanced or its server-clock deadline
expired. An unchanged fresh GET alone is not a fence.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import httpx

from prefill_lab.live_http import clock_domain

from .contracts import Decision, Identity, ModelParameters, SCHEMA, Snapshot, finite
from .controller import validate_snapshot
from .runtime import require_guarded_capability, snapshot_from_dict


@dataclass(frozen=True)
class ActuationPermit:
    identity: Identity
    evidence: tuple[tuple[str, str], ...] = ()
    approved: bool = False

    def allows(self, expected: Identity) -> bool:
        required = {"native_state", "native_actions", "calibration", "native_http", "fallback"}
        pairs = self.evidence
        return (
            self.approved is True
            and self.identity == expected
            and type(pairs) is tuple
            and len(pairs) == len(required)
            and all(
                type(pair) is tuple and len(pair) == 2 and isinstance(pair[0], str)
                for pair in pairs
            )
            and {pair[0] for pair in pairs} == required
            and all(
                isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
                for _, value in pairs
            )
        )


class GuardedActuator:
    def __init__(
        self, client: httpx.AsyncClient, parameters: ModelParameters,
        pending_path: Path, *, permit: ActuationPermit | None = None,
        enabled: bool = False, max_age_s: float = 2.0, command_ttl_s: float = 0.1,
    ):
        if not finite(max_age_s) or max_age_s < 0 or not finite(command_ttl_s, positive=True):
            raise ValueError("invalid_actuator_limits")
        self.client = client
        self.parameters = parameters
        self.path = Path(pending_path)
        self.permit = permit
        self.enabled = enabled
        self.max_age_s = max_age_s
        self.ttl = command_ttl_s

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _pending(self):
        if not self.path.exists():
            return None
        try:
            pending = json.loads(self.path.read_text())
            keys = {
                "schema", "run_id", "clock_domain", "identity", "expected_version",
                "operation_id", "budget", "not_after",
            }
            if not isinstance(pending, dict) or set(pending) != keys:
                raise ValueError
            if (
                pending["schema"] != SCHEMA
                or any(not isinstance(pending[k], str) or not pending[k] for k in (
                    "run_id", "clock_domain", "operation_id",
                ))
                or type(pending["expected_version"]) is not int
                or pending["expected_version"] < 0
                or not finite(pending["budget"], positive=True)
                or not finite(pending["not_after"])
                or not isinstance(pending["identity"], dict)
            ):
                raise ValueError
            Identity(**pending["identity"])
            return pending
        except (ValueError, TypeError) as exc:
            raise ValueError("pending_state_invalid") from exc

    def _save(self, command):
        descriptor, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(command, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            self._sync_directory()
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _sync_directory(self):
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _clear(self):
        self.path.unlink()
        self._sync_directory()

    async def _state(self) -> Snapshot:
        response = await self.client.get("/admin/mpc/state")
        response.raise_for_status()
        document = response.json()
        require_guarded_capability(document)
        snapshot = snapshot_from_dict(document)
        reason = validate_snapshot(
            snapshot, now=time.monotonic(), clock_domain=clock_domain(),
            expected_identity=self.parameters.identity, max_age_s=self.max_age_s,
        )
        if reason:
            raise ValueError(reason)
        if snapshot.budget not in self.parameters.supported_budgets:
            raise ValueError("unsupported_budget")
        for mark in snapshot.requests:
            if mark.cost != self.parameters.cost_spec.estimate(mark.prompt_tokens, mark.max_tokens):
                raise ValueError("cost_spec_mismatch")
        return snapshot

    async def submit(self, snapshot: Snapshot, decision: Decision) -> dict:
        expected = self.parameters.identity
        if self.enabled is not True or self.permit is None or not self.permit.allows(expected):
            return {"status": "not_authorized"}
        reason = validate_snapshot(
            snapshot, now=time.monotonic(), clock_domain=clock_domain(),
            expected_identity=expected, max_age_s=self.max_age_s,
        )
        if reason:
            return {"status": "blocked", "reason": reason}
        if (
            not isinstance(decision, Decision) or decision.identity != expected
            or not finite(decision.budget, positive=True)
            or decision.budget not in self.parameters.supported_budgets
            or snapshot.budget not in self.parameters.supported_budgets
        ):
            return {"status": "blocked", "reason": "unsupported_decision"}
        with self._locked():
            if self._pending() is not None:
                return {"status": "pending"}
            current = await self._state()
            if current.run_id != snapshot.run_id or current.budget_version != snapshot.budget_version:
                return {"status": "blocked", "reason": "state_changed"}
            now = time.monotonic()
            not_after = min(now + self.ttl, snapshot.sampled_at + self.max_age_s)
            if now >= not_after:
                return {"status": "blocked", "reason": "stale_snapshot"}
            command = {
                "schema": SCHEMA, "run_id": snapshot.run_id, "clock_domain": snapshot.clock_domain,
                "identity": asdict(expected), "expected_version": snapshot.budget_version,
                "operation_id": uuid.uuid4().hex, "budget": decision.budget, "not_after": not_after,
            }
            self._save(command)
            try:
                response = await self.client.post("/admin/mpc/budget", json=command)
                reply = response.json()
            except (httpx.HTTPError, ValueError):
                return {"status": "pending", "operation_id": command["operation_id"]}
            if response.status_code == 200 and (
                isinstance(reply, dict)
                and reply.get("status") == "applied"
                and reply.get("operation_id") == command["operation_id"]
                and reply.get("run_id") == command["run_id"]
                and reply.get("clock_domain") == command["clock_domain"]
                and type(reply.get("budget_version")) is int
                and reply["budget_version"] == command["expected_version"] + 1
                and finite(reply.get("budget"), positive=True)
                and reply["budget"] == command["budget"]
                and finite(reply.get("applied_at"))
                and snapshot.sampled_at <= reply["applied_at"] < not_after
            ):
                self._clear()
                return {"status": "applied", "acknowledgment": reply}
            # Even a server error may follow mutation; resolve through CAS/expiry.
            return {"status": "pending", "operation_id": command["operation_id"]}

    async def reconcile(self) -> dict:
        with self._locked():
            pending = self._pending()
            if pending is None:
                return {"status": "clear"}
            try:
                current = await self._state()
            except (httpx.HTTPError, ValueError, OSError):
                return {"status": "pending"}
            if (
                current.run_id != pending["run_id"]
                or current.clock_domain != pending["clock_domain"]
                or asdict(current.identity) != pending["identity"]
            ):
                return {"status": "pending", "reason": "identity_changed"}
            advanced = current.budget_version > pending["expected_version"]
            expired = time.monotonic() >= pending["not_after"]
            if not advanced and not expired:
                return {"status": "pending"}
            if current.budget_version < pending["expected_version"]:
                return {"status": "pending", "reason": "version_regression"}
            self._clear()
            return {
                "status": "fenced" if advanced else "expired",
                "observed_budget": current.budget, "observed_version": current.budget_version,
            }
