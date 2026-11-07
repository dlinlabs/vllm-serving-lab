"""Causal public inputs shared by the independently implemented R21 components."""

from dataclasses import dataclass
from typing import Protocol

from mpc_lab.contracts import CostSpec


@dataclass(frozen=True)
class Arrival:
    at: float
    prompt_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class PublicRequest:
    request_id: str
    prompt_tokens: int
    max_tokens: int
    cost: float
    scheduled_at: float
    admitted_at: float
    first_at: float | None = None
    observed_output_tokens: int = 0
    terminal_at: float | None = None
    finish_reason: str | None = None
    output_sampled_at: float | None = None


@dataclass(frozen=True)
class PublicState:
    now: float
    budget: float
    version: int
    requests: tuple[PublicRequest, ...]
    arrivals: tuple[Arrival, ...] = ()
    run_id: str = "local"
    clock_domain: str = "simulation"
    complete: bool = True
    sampled_at: float | None = None


@dataclass(frozen=True)
class Forecast:
    rate: float
    marks: tuple[tuple[int, int], ...]
    seed: int = 99173


@dataclass(frozen=True)
class Config:
    actions: tuple[float, ...] = (16.0, 32.0)
    moves: int = 2
    slot_s: float = 1.0
    horizon_s: float = 5.0
    tail_s: float = 10.0
    slo_ttft_s: float = 0.5
    slo_tpot_s: float = 0.05
    http_deadline_s: float = 30.0
    solver_budget_s: float = 0.05
    reactive_threshold: float = 16.0
    max_snapshot_age_s: float = 0.5
    offered_end_s: float | None = None


@dataclass(frozen=True)
class Prediction:
    cohort_good: float
    window_good: float
    admitted: float
    rejected: float
    final_stock: float
    cost_in: float = 0.0
    cost_out: float = 0.0


@dataclass(frozen=True)
class Choice:
    budget: float
    plan: tuple[float, ...]
    scores: tuple[tuple[tuple[float, ...], float], ...]
    status: str = "ok"
    elapsed_s: float = 0.0


class Predictor(Protocol):
    def predict(
        self, state: PublicState, plan: tuple[float, ...],
        forecast: Forecast, config: Config,
    ) -> Prediction: ...
