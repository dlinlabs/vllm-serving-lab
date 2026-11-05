"""Separate opt-in for bounded, uncalibrated gateway experiments, not Native qualification."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from mpc_lab.contracts import Identity, Snapshot, digest, finite, implementation_hash
from mpc_lab.controller import validate_snapshot


SCOPE = "gateway-nominal-unqualified-v1"


def source_digest() -> str:
    directory = Path(__file__).resolve().parent
    names = (
        "nominal_permit.py", "nominal_planner.py", "nominal_runtime.py",
        "nominal_scenario.py", "nominal_workload.py",
    )
    return digest({
        "public_implementation": implementation_hash(),
        "nominal": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names
        },
    })


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NominalDemoPermit:
    identity: Identity
    run_id: str
    clock_domain: str
    source_sha256: str
    protocol_sha256: str
    not_before: float
    expires_at: float
    max_commands: int = 8
    approved: bool = False
    scope: str = SCOPE

    def refusal_reason(
        self, snapshot: Snapshot, *, enabled: bool, expected_identity: Identity,
        current_source_sha256: str, protocol_sha256: str, current_clock_domain: str,
        budget: float,
        entered_at: float, deadline: float, now: float, issued_commands: int,
    ) -> str | None:
        if enabled is not True or self.approved is not True or self.scope != SCOPE:
            return "nominal_not_authorized"
        if (
            self.identity != expected_identity
            or expected_identity.model_version != "cpu-lifecycle-v1"
            or expected_identity.calibration_version != "synthetic-unqualified-v1"
        ):
            return "nominal_identity_mismatch"
        if (
            not _sha256(self.source_sha256)
            or not _sha256(self.protocol_sha256)
            or self.source_sha256 != current_source_sha256
            or self.protocol_sha256 != protocol_sha256
        ):
            return "nominal_source_or_protocol_mismatch"
        if (
            not isinstance(self.run_id, str) or not self.run_id
            or not isinstance(self.clock_domain, str) or not self.clock_domain
            or current_clock_domain != self.clock_domain
            or snapshot.run_id != self.run_id
            or snapshot.clock_domain != self.clock_domain
        ):
            return "nominal_run_or_clock_mismatch"
        if (
            not finite(self.not_before) or not finite(self.expires_at)
            or not 0 <= self.not_before < self.expires_at
            or self.expires_at - self.not_before > 60.0
            or type(self.max_commands) is not int or not 1 <= self.max_commands <= 8
            or type(issued_commands) is not int or issued_commands < 0
        ):
            return "nominal_invalid_limits"
        if not all(finite(value) for value in (entered_at, deadline, now)):
            return "nominal_invalid_deadline"
        reason = validate_snapshot(
            snapshot, now=now, clock_domain=current_clock_domain,
            expected_identity=expected_identity, max_age_s=0.5,
        )
        if reason:
            return reason
        if (
            not self.not_before <= entered_at <= snapshot.sampled_at <= now
            or not now < deadline <= entered_at + 0.5
            or not now < self.expires_at
        ):
            return "nominal_expired_or_restarted_deadline"
        if not finite(budget) or budget not in (16.0, 32.0) or snapshot.budget not in (16.0, 32.0):
            return "nominal_unsupported_budget"
        if issued_commands >= self.max_commands:
            return "nominal_command_limit"
        return None
