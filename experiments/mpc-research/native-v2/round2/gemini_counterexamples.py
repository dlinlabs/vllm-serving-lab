"""Gemini Round 2 Counterexamples and Measurement Facade Evaluation.

Implements executable mathematical counterexamples for:
1. Roofline knee vs bi-affine service scaling (challenging linear token models).
2. Whole-reservation release ambiguity from aggregate token backlog (formalizing
   the discrete release obstruction under continuous batching).
3. Mayor's DEFAULT-OFF measurement facade evaluation: forward-scope CUDA event
   timing vs intrusive CUPTI tracing vs blocking synchronization.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple


# ==============================================================================
# 1. Roofline vs Bi-Affine Model Comparison
# ==============================================================================

@dataclass(frozen=True)
class BiAffineModel:
    """Standard linear/bi-affine service law: tau = theta0 + thetaD*D + thetaP*P."""
    theta_0_ms: float = 12.31
    theta_D_ms: float = 0.049
    theta_P_ms_per_token: float = 0.00873  # 8.73 ms / 1000 tokens

    def predict_ms(self, prefill_tokens: int, decode_requests: int) -> float:
        return (
            self.theta_0_ms
            + self.theta_D_ms * decode_requests
            + self.theta_P_ms_per_token * prefill_tokens
        )


@dataclass(frozen=True)
class RooflineHingeModel:
    """Physical roofline model: memory-bandwidth floor + compute slope above knee."""
    floor_ms: float = 8.50       # Memory-bandwidth bound iteration floor (BF16 weights)
    decode_step_ms: float = 0.35 # Per-decoder incremental memory access
    knee_tokens: int = 128       # Tokens required to saturate compute units
    slope_ms_per_token: float = 0.051 # Compute scaling above knee (~51 ms / 1000 tokens)

    def predict_ms(self, prefill_tokens: int, decode_requests: int) -> float:
        base_memory = self.floor_ms + self.decode_step_ms * decode_requests
        total_tokens = prefill_tokens + decode_requests
        compute_excess = max(0, total_tokens - self.knee_tokens)
        compute_time = self.slope_ms_per_token * compute_excess
        return base_memory + compute_time


def compare_batch_scaling() -> Dict[str, object]:
    """Demonstrates why linear models fail on short prefills and small batches."""
    bi_affine = BiAffineModel()
    roofline = RooflineHingeModel()

    # Ground truth profile medians from native-profile-diagnostic.json:
    # 1. Decode 1, prefill 0: kernel busy ~ 9.38 ms
    # 2. Prefill 128, decode 0: kernel busy ~ 10.36 ms
    # 3. Prefill 2048, decode 0: kernel busy ~ 112.46 ms
    measured = {
        "decode_1": {"p": 0, "d": 1, "actual_ms": 9.38},
        "prefill_128": {"p": 128, "d": 0, "actual_ms": 10.36},
        "prefill_2048": {"p": 2048, "d": 0, "actual_ms": 112.46},
    }

    results = {}
    for case, data in measured.items():
        p, d, actual = data["p"], data["d"], data["actual_ms"]
        pred_bi = bi_affine.predict_ms(p, d)
        pred_roof = roofline.predict_ms(p, d)
        results[case] = {
            "actual_ms": actual,
            "bi_affine_pred_ms": round(pred_bi, 2),
            "bi_affine_error_ms": round(abs(pred_bi - actual), 2),
            "roofline_pred_ms": round(pred_roof, 2),
            "roofline_error_ms": round(abs(pred_roof - actual), 2),
        }

    return {
        "comparison": results,
        "conclusion": (
            "Bi-affine model fitted on decode survivors fails on large prefill chunks (predicts 30.17ms vs actual 112.46ms), "
            "while roofline hinge captures both the memory floor (~9-10ms) and the compute slope above knee."
        )
    }


# ==============================================================================
# 2. Whole-Reservation Release Ambiguity (Sol Formalization)
# ==============================================================================

@dataclass(frozen=True)
class ActiveStream:
    request_id: str
    cost: float
    remaining_tokens: int


def evaluate_reservation_release(
    streams_world_a: Sequence[ActiveStream],
    streams_world_b: Sequence[ActiveStream],
    horizon_steps: int,
    tokens_per_step: int = 1,
    budget: float = 16.0,
    incoming_offer_cost: float = 4.0,
) -> Dict[str, object]:
    """Demonstrates that identical total remaining tokens produce opposite admission decisions.

    Formalizes the Sol counterexample: identical aggregate backlog sum(tokens)
    cannot predict discrete whole-cost releases when work is distributed across decoders.
    """
    total_a = sum(s.remaining_tokens for s in streams_world_a)
    total_b = sum(s.remaining_tokens for s in streams_world_b)
    if total_a != total_b:
        raise ValueError("Both worlds must have identical total remaining tokens.")

    cost_a = sum(s.cost for s in streams_world_a)
    cost_b = sum(s.cost for s in streams_world_b)
    if cost_a != cost_b:
        raise ValueError("Both worlds must have identical total active cost.")

    # In continuous batching, all active decoders step in parallel
    def simulate_steps(streams: Sequence[ActiveStream], steps: int) -> Tuple[float, int]:
        current_reserved = sum(s.cost for s in streams)
        released_cost = 0.0
        active_remaining = [s.remaining_tokens for s in streams]
        costs = [s.cost for s in streams]

        for step in range(1, steps + 1):
            for idx in range(len(active_remaining)):
                if active_remaining[idx] > 0:
                    active_remaining[idx] -= tokens_per_step
                    if active_remaining[idx] <= 0:
                        released_cost += costs[idx]

        future_reserved = current_reserved - released_cost
        return future_reserved, released_cost

    res_a, rel_a = simulate_steps(streams_world_a, horizon_steps)
    res_b, rel_b = simulate_steps(streams_world_b, horizon_steps)

    admit_a = (res_a + incoming_offer_cost) <= budget
    admit_b = (res_b + incoming_offer_cost) <= budget

    return {
        "shared_initial_cost": cost_a,
        "shared_total_tokens": total_a,
        "horizon_steps": horizon_steps,
        "budget": budget,
        "incoming_offer_cost": incoming_offer_cost,
        "world_a": {
            "released_cost": rel_a,
            "future_reserved_cost": res_a,
            "can_admit_offer": admit_a,
        },
        "world_b": {
            "released_cost": rel_b,
            "future_reserved_cost": res_b,
            "can_admit_offer": admit_b,
        },
        "decisions_diverge": admit_a != admit_b,
        "cost_ambiguity": abs(res_a - res_b),
    }


# ==============================================================================
# 3. Mayor's Measurement Facade Strategy Evaluation
# ==============================================================================

@dataclass(frozen=True)
class MeasurementMethodEvaluation:
    method_name: str
    is_non_blocking_during_serving: bool
    estimated_host_overhead_us_per_batch: float
    captures_forward_scope: bool
    captures_kernel_busy_union: bool
    requires_post_drain_resolution: bool
    timing_perturbation_risk: str
    verdict: str


def evaluate_measurement_strategies() -> Dict[str, MeasurementMethodEvaluation]:
    """Compares Mayor's DEFAULT-OFF facade with CUPTI and blocking sync."""
    return {
        "mayor_default_off_facade": MeasurementMethodEvaluation(
            method_name="PR32 ExternalCapture + Current-Stream CUDA Event Pair",
            is_non_blocking_during_serving=True,
            estimated_host_overhead_us_per_batch=4.5, # Two cudaEventRecord calls ~4-5 microseconds
            captures_forward_scope=True,
            captures_kernel_busy_union=False,          # Span of forward stream, not individual kernel union
            requires_post_drain_resolution=True,       # cudaEventElapsedTime resolved only after drain
            timing_perturbation_risk="NEGLIGIBLE (< 0.05% of 10ms batch)",
            verdict="RECOMMENDED: Safe, low-overhead, unperturbed serving timing.",
        ),
        "runtime_cupti_tracing": MeasurementMethodEvaluation(
            method_name="CUPTI Activity Tracing during serving",
            is_non_blocking_during_serving=False,
            estimated_host_overhead_us_per_batch=450.0, # Buffer flush, activity callback overhead
            captures_forward_scope=True,
            captures_kernel_busy_union=True,
            requires_post_drain_resolution=False,
            timing_perturbation_risk="CRITICAL (Inflates launch latencies by 5-15%)",
            verdict="REJECTED for serving calibration; diagnostic probe only.",
        ),
        "blocking_cuda_sync": MeasurementMethodEvaluation(
            method_name="cudaDeviceSynchronize per iteration",
            is_non_blocking_during_serving=False,
            estimated_host_overhead_us_per_batch=1200.0, # Complete pipeline stall
            captures_forward_scope=True,
            captures_kernel_busy_union=False,
            requires_post_drain_resolution=False,
            timing_perturbation_risk="FATAL: Destroys CPU-GPU execution pipelining",
            verdict="STRICTLY PROHIBITED.",
        ),
    }
