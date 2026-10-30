"""Shared immutable contracts; cost units are not GPU seconds or KV bytes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "mpc-admission-v1"
GUARDED_CAPABILITY = "mpc-budget-cas-expiry-v1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def finite(value: object, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else True)
    except OverflowError:
        return False


def implementation_hash() -> str:
    root = Path(__file__).resolve().parent
    files = list(root.glob("*.py"))
    files += [
        p for p in (
            root.parent / "gateway.py", root.parent / "telemetry.py",
            root.parent / "prefill_lab/live_http.py",
        ) if p.is_file()
    ]
    return digest({
        str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    })


@dataclass(frozen=True)
class CostSpec:
    intercept: float = 0.259
    input_per_token: float = 0.000758
    output_per_token: float = 0.0052083333
    minimum: float = 0.1
    default_max_tokens: int = 128
    tokenizer_identity: str = "unqualified"
    estimator_version: str = "gateway-token-estimate-v1"

    def __post_init__(self) -> None:
        values = (self.intercept, self.input_per_token, self.output_per_token)
        if any(not finite(v) or v < 0 for v in values) or not finite(self.minimum, positive=True):
            raise ValueError("invalid_cost_spec")
        if type(self.default_max_tokens) is not int or self.default_max_tokens <= 0:
            raise ValueError("invalid_default_max_tokens")
        if not self.tokenizer_identity or not self.estimator_version:
            raise ValueError("missing_cost_identity")

    @property
    def fingerprint(self) -> str:
        return digest({"formula": "max-min-affine-v1", **asdict(self)})

    def estimate(self, prompt_tokens: int, max_tokens: int) -> float:
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("invalid_prompt_tokens")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("invalid_max_tokens")
        try:
            cost = max(
                self.minimum,
                self.intercept + self.input_per_token * prompt_tokens + self.output_per_token * max_tokens,
            )
        except OverflowError as exc:
            raise ValueError("nonfinite_request_cost") from exc
        if not finite(cost, positive=True):
            raise ValueError("nonfinite_request_cost")
        return cost


@dataclass(frozen=True)
class Identity:
    model_version: str
    calibration_version: str
    support_hash: str
    cost_hash: str
    implementation_hash: str

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class RequestMark:
    request_id: str
    cost: float
    prompt_tokens: int
    max_tokens: int
    received_at: float
    admitted_at: float
    first_at: float | None = None
    scheduled_at: float | None = None


@dataclass(frozen=True)
class Snapshot:
    run_id: str
    clock_domain: str
    sampled_at: float
    budget: float
    budget_version: int
    requests: tuple[RequestMark, ...]
    identity: Identity
    sequence: int = 0
    complete: bool = True

    @property
    def reserved(self) -> float:
        return math.fsum(r.cost for r in self.requests)

    @property
    def prompt_backlog(self) -> int:
        return sum(r.prompt_tokens for r in self.requests if r.first_at is None)


@dataclass(frozen=True)
class FutureOffer:
    request_id: str
    offset_s: float
    prompt_tokens: int
    max_tokens: int
    output_tokens: int
    success: bool = True
    ingress_delay_s: float = 0.0
    egress_delay_s: float = 0.0


@dataclass(frozen=True)
class Scenario:
    offers: tuple[FutureOffer, ...]
    service_scale: float = 1.0
    old_output_fraction: float = 1.0
    old_success: bool = True
    ingress_delay_s: float = 0.0
    egress_delay_s: float = 0.0


@dataclass(frozen=True)
class ModelParameters:
    cost_spec: CostSpec = CostSpec()
    prefill_s_per_token: float = 0.00005
    decode_s_per_token: float = 0.004
    first_overhead_s: float = 0.03
    decode_interference: float = 0.02
    model_version: str = "cpu-lifecycle-v1"
    calibration_version: str = "synthetic-unqualified-v1"
    supported_budgets: tuple[float, ...] = (16.0, 32.0)
    max_prompt_tokens: int = 32768
    max_output_tokens: int = 4096

    @property
    def identity(self) -> Identity:
        return Identity(
            self.model_version,
            self.calibration_version,
            digest(asdict(self)),
            self.cost_spec.fingerprint,
            implementation_hash(),
        )


@dataclass(frozen=True)
class Plan:
    actions: tuple[float, ...]
    slot_s: float = 1.0
    continuation_budget: float = 16.0

    def budget_at(self, offset_s: float) -> float:
        index = int(offset_s // self.slot_s)
        return self.actions[index] if 0 <= index < len(self.actions) else self.continuation_budget


@dataclass(frozen=True)
class PlanningConfig:
    actions: tuple[float, ...] = (16.0, 32.0)
    control_steps: int = 3
    slot_s: float = 1.0
    evaluation_window_s: float = 5.0
    tail_s: float = 10.0
    fallback_budget: float = 16.0
    slo_s: float = 0.5
    max_wall_s: float = 0.1
    max_events: int = 20000
    max_active_requests: int = 512
    max_plans: int = 32
    max_scenarios: int = 32
    max_snapshot_age_s: float = 2.0


@dataclass(frozen=True)
class Outcome:
    request_id: str
    scheduled_at: float
    accepted: bool
    first_at: float | None
    terminal_at: float | None
    success: bool | None
    cost: float
    prompt_tokens: int
    max_tokens: int

    def good(self, slo_s: float = 0.5) -> bool:
        return (
            self.accepted
            and self.success is True
            and self.terminal_at is not None
            and self.first_at is not None
            and 0 <= self.first_at - self.scheduled_at <= slo_s
        )


@dataclass(frozen=True)
class RolloutResult:
    outcomes: tuple[Outcome, ...]
    reserved_trace: tuple[tuple[float, float], ...]
    score_lower: float
    score_upper: float
    event_count: int


@dataclass(frozen=True)
class Decision:
    budget: float
    plan: Plan | None
    reason: str
    fallback: bool
    score_lower: float | None
    score_upper: float | None
    elapsed_s: float
    evaluated_plans: int
    scenario_count: int
    identity: Identity


class RolloutError(ValueError):
    """A bounded or unsupported rollout, not a physical safety conclusion."""
