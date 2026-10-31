"""Gemini Timing Qualification, Probe Overhead Guard, and Fair Protocol Contract.

Part of PR32/PR33 Native MPC Research - Round 1 Independent Research.
Implements timing decomposition, probe overhead budget, serial proxy falsification,
and paired-block statistical testing without requiring GPU or remote database access.
"""

from dataclasses import dataclass, field
import math
import statistics
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ClockDomain:
    """Represents a synchronized monotonic clock domain."""
    domain_id: str
    boot_id: str
    is_monotonic: bool = True

    def validate_compatible(self, other: "ClockDomain") -> None:
        if self.domain_id != other.domain_id:
            raise ValueError(
                f"Incompatible clock domains: '{self.domain_id}' vs '{other.domain_id}'. "
                "Monotonic offsets across distinct clock domains are meaningless."
            )
        if self.boot_id != other.boot_id:
            raise ValueError(
                f"Clock domain boot mismatch: '{self.boot_id}' vs '{other.boot_id}'. "
                "Node reboot or container restart invalidates monotonic continuity."
            )
        if not self.is_monotonic or not other.is_monotonic:
            raise ValueError("Non-monotonic clocks cannot be used for serving timing contracts.")


@dataclass(frozen=True)
class RequestTimingTrace:
    """Exact multi-stage timestamps for an individual request lifecycle.

    All timestamps must be recorded in the specified monotonic clock domain.
    """
    request_id: str
    clock_domain: ClockDomain
    prompt_tokens: int
    output_tokens: int
    scheduled_arrival: float
    gateway_received: float
    gateway_admitted: Optional[float]
    vllm_enqueued: Optional[float]
    first_token_sampled: Optional[float]
    first_content_delivered: Optional[float]
    stream_terminal: Optional[float]
    rejected: bool = False
    cancelled: bool = False

    def validate_monotonic_order(self) -> None:
        """Enforces physical causality across serving pipeline stages."""
        if self.gateway_received < self.scheduled_arrival:
            raise ValueError(
                f"Causality violation: gateway_received ({self.gateway_received}) < "
                f"scheduled_arrival ({self.scheduled_arrival}) for request {self.request_id}."
            )
        if self.rejected:
            if self.gateway_admitted is not None:
                raise ValueError(f"Rejected request {self.request_id} cannot have gateway_admitted.")
            return

        if self.gateway_admitted is None:
            raise ValueError(f"Admitted request {self.request_id} missing gateway_admitted timestamp.")

        if self.gateway_admitted < self.gateway_received:
            raise ValueError(
                f"Causality violation: gateway_admitted < gateway_received for {self.request_id}."
            )

        if self.vllm_enqueued is not None and self.vllm_enqueued < self.gateway_admitted:
            raise ValueError(
                f"Causality violation: vllm_enqueued < gateway_admitted for {self.request_id}."
            )

        if self.first_token_sampled is not None:
            if self.vllm_enqueued is not None and self.first_token_sampled < self.vllm_enqueued:
                raise ValueError(
                    f"Causality violation: first_token_sampled < vllm_enqueued for {self.request_id}."
                )
            if self.first_content_delivered is not None and self.first_content_delivered < self.first_token_sampled:
                raise ValueError(
                    f"Causality violation: first_content_delivered < first_token_sampled for {self.request_id}."
                )

        if self.stream_terminal is not None and self.first_content_delivered is not None:
            if self.stream_terminal < self.first_content_delivered:
                raise ValueError(
                    f"Causality violation: stream_terminal < first_content_delivered for {self.request_id}."
                )

    @property
    def ingress_lag_s(self) -> float:
        return self.gateway_received - self.scheduled_arrival

    @property
    def ttft_s(self) -> Optional[float]:
        """Time-To-First-Token measured from scheduled arrival to client content delivery."""
        if self.first_content_delivered is None or self.rejected:
            return None
        return self.first_content_delivered - self.scheduled_arrival

    @property
    def engine_first_residence_s(self) -> Optional[float]:
        """Engine-internal residence until first token is sampled."""
        if self.first_token_sampled is None or self.vllm_enqueued is None or self.rejected:
            return None
        return self.first_token_sampled - self.vllm_enqueued

    @property
    def egress_transport_s(self) -> Optional[float]:
        """Transport + serialization delay from engine token sample to client delivery."""
        if self.first_content_delivered is None or self.first_token_sampled is None:
            return None
        return self.first_content_delivered - self.first_token_sampled

    @property
    def request_mean_tpot_s(self) -> Optional[float]:
        """Mean Time Per Output Token: (terminal - first) / (output_tokens - 1)."""
        if self.stream_terminal is None or self.first_content_delivered is None or self.rejected:
            return None
        if self.output_tokens <= 1:
            return 0.0
        return (self.stream_terminal - self.first_content_delivered) / (self.output_tokens - 1)

    def is_composite_good(self, max_ttft_s: float = 0.5, max_tpot_s: float = 0.05) -> bool:
        """Evaluates whether the request satisfied the composite SLO:

        TTFT <= 500ms AND request-mean TPOT <= 50ms, without rejection or cancellation.
        """
        if self.rejected or self.cancelled:
            return False
        ttft = self.ttft_s
        tpot = self.request_mean_tpot_s
        if ttft is None or tpot is None:
            return False
        return ttft <= max_ttft_s and tpot <= max_tpot_s


@dataclass(frozen=True)
class BatchExecutionInterval:
    """Execution interval details for an individual iteration/batch on the GPU."""
    batch_id: int
    clock_domain: ClockDomain
    cpu_call_start: float
    cpu_call_end: float
    gpu_span_start: float
    gpu_span_end: float
    kernel_busy_union_s: float
    prefill_requests: int
    prefill_tokens: int
    decode_requests: int
    decode_tokens: int

    def validate(self) -> None:
        if self.cpu_call_end < self.cpu_call_start:
            raise ValueError(f"Batch {self.batch_id}: cpu_call_end < cpu_call_start.")
        if self.gpu_span_end < self.gpu_span_start:
            raise ValueError(f"Batch {self.batch_id}: gpu_span_end < gpu_span_start.")
        gpu_span = self.gpu_span_end - self.gpu_span_start
        if self.kernel_busy_union_s > gpu_span + 1e-6:
            raise ValueError(
                f"Batch {self.batch_id}: kernel_busy_union ({self.kernel_busy_union_s:.6f}s) "
                f"exceeds total GPU span ({gpu_span:.6f}s)."
            )
        if self.kernel_busy_union_s < 0.0:
            raise ValueError(f"Batch {self.batch_id}: negative kernel_busy_union.")

    @property
    def cpu_duration_s(self) -> float:
        return self.cpu_call_end - self.cpu_call_start

    @property
    def gpu_span_s(self) -> float:
        return self.gpu_span_end - self.gpu_span_start

    @property
    def gpu_gap_s(self) -> float:
        """Idle GPU gap/bubbles between kernels within the batch span."""
        return max(0.0, self.gpu_span_s - self.kernel_busy_union_s)

    @property
    def gpu_utilization(self) -> float:
        span = self.gpu_span_s
        if span <= 0:
            return 1.0
        return min(1.0, self.kernel_busy_union_s / span)


@dataclass(frozen=True)
class SerialCapacityScreener:
    """Falsification test for exclusive serial HTTP proxy service models.

    Addresses the empirical falsification where serial HTTP first-residence
    (37.97ms + 0.054ms*prompt) implied an upper bound of 2004 completions in 90s,
    contradicting the observed 2449 actual completions under continuous batching.
    """
    intercept_s: float = 0.03796779
    slope_s_per_token: float = 0.00005423180

    def service_time_s(self, prompt_tokens: int) -> float:
        if prompt_tokens < 0:
            raise ValueError("Prompt tokens cannot be negative.")
        return self.intercept_s + self.slope_s_per_token * prompt_tokens

    def theoretical_serial_capacity(self, window_s: float, prompt_tokens: int,
                                     carry_in_completions: int = 0) -> int:
        """Calculates theoretical max completions under exclusive FIFO serial execution."""
        service_s = self.service_time_s(prompt_tokens)
        if service_s <= 0:
            raise ValueError("Service time must be strictly positive.")
        return int(math.floor(window_s / service_s)) + carry_in_completions

    def evaluate_falsification(self, window_s: float, prompt_tokens: int,
                               actual_completions: int,
                               carry_in_completions: int = 0) -> Dict[str, object]:
        """Screens whether the serial assumption is mathematically falsified by actual count."""
        service_s = self.service_time_s(prompt_tokens)
        serial_limit = self.theoretical_serial_capacity(window_s, prompt_tokens, carry_in_completions)
        is_falsified = actual_completions > serial_limit

        required_serial_time_s = actual_completions * service_s
        time_deficit_s = max(0.0, required_serial_time_s - window_s)
        missing_carry_in = max(0, actual_completions - int(math.floor(window_s / service_s)))

        return {
            "window_s": window_s,
            "prompt_tokens": prompt_tokens,
            "unit_service_s": service_s,
            "actual_completions": actual_completions,
            "serial_capacity_bound": serial_limit,
            "is_falsified": is_falsified,
            "time_deficit_s": time_deficit_s,
            "required_carry_in_to_reconcile": missing_carry_in,
            "conclusion": (
                "FALSIFIED: Serial HTTP proxy model is mathematically impossible. "
                "Continuous batching executes concurrent work on GPU."
                if is_falsified
                else "NOT FALSIFIED: Observed count within serial envelope."
            )
        }


@dataclass(frozen=True)
class ProbeOverheadRecord:
    """Records resource and timing overhead of state probes (e.g. read-only RPC vs CUPTI)."""
    probe_name: str
    clock_domain: ClockDomain
    start_time: float
    end_time: float
    payload_bytes: int
    is_blocking_gpu: bool
    is_tracing_cupti: bool

    @property
    def latency_s(self) -> float:
        return self.end_time - self.start_time


class ProbeOverheadGuard:
    """Enforces non-intrusive probe requirements for online MPC state observation."""

    def __init__(self, max_probe_latency_s: float = 0.015,
                 max_slot_fraction: float = 0.05,
                 disallow_cupti_in_serving: bool = True):
        self.max_probe_latency_s = max_probe_latency_s
        self.max_slot_fraction = max_slot_fraction
        self.disallow_cupti_in_serving = disallow_cupti_in_serving

    def validate_probe(self, probe: ProbeOverheadRecord, control_slot_s: float) -> Tuple[bool, List[str]]:
        violations = []
        if probe.is_tracing_cupti and self.disallow_cupti_in_serving:
            violations.append(
                "CUPTI kernel tracing active during serving. Tracing perturbs kernel launch "
                "latencies, GPU execution bubbles, and cannot be used for unprofiled MPC calibration."
            )
        if probe.is_blocking_gpu:
            violations.append(
                "Probe performs synchronous GPU synchronization (cudaDeviceSynchronize), "
                "stalling the async execution pipeline."
            )
        if probe.latency_s > self.max_probe_latency_s:
            violations.append(
                f"Probe latency ({probe.latency_s*1000:.2f}ms) exceeds max permissible "
                f"({self.max_probe_latency_s*1000:.2f}ms)."
            )
        fraction = probe.latency_s / control_slot_s if control_slot_s > 0 else 1.0
        if fraction > self.max_slot_fraction:
            violations.append(
                f"Probe consumes {fraction*100:.1f}% of control slot ({control_slot_s}s), "
                f"exceeding ceiling of {self.max_slot_fraction*100:.1f}%."
            )
        return len(violations) == 0, violations


@dataclass(frozen=True)
class PairedEvaluationBlock:
    """Represents a matched experimental block comparing two control policies."""
    block_id: str
    seed: int
    offered_requests: int
    duration_s: float
    policy_a_name: str
    policy_a_good_completions: int
    policy_a_rejections: int
    policy_a_solver_overruns: int
    policy_b_name: str
    policy_b_good_completions: int
    policy_b_rejections: int
    policy_b_solver_overruns: int

    @property
    def policy_a_goodput(self) -> float:
        return self.policy_a_good_completions / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def policy_b_goodput(self) -> float:
        return self.policy_b_good_completions / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def paired_difference(self) -> float:
        """Difference in goodput: Policy B - Policy A."""
        return self.policy_b_goodput - self.policy_a_good_completions / self.duration_s


class NativeFairStatisticalProtocol:
    """Implements strict paired-block design, superiority margin, and seed isolation."""

    HELD_OUT_TEST_SEEDS = frozenset({160021, 160022, 160023, 160024})

    def __init__(self, min_superiority_margin_req_s: float = 0.5,
                 max_solver_overrun_rate: float = 0.01):
        self.min_superiority_margin = min_superiority_margin_req_s
        self.max_solver_overrun_rate = max_solver_overrun_rate

    def check_held_out_isolation(self, active_seeds: Sequence[int]) -> None:
        """Verifies that no held-out test seeds were leaked into training or tuning."""
        leaked = set(active_seeds).intersection(self.HELD_OUT_TEST_SEEDS)
        if leaked:
            raise ValueError(
                f"CRITICAL PROTOCOL VIOLATION: Held-out test seeds {sorted(leaked)} "
                "were accessed during development/tuning! Test split integrity compromised."
            )

    def evaluate_paired_blocks(self, blocks: Sequence[PairedEvaluationBlock]) -> Dict[str, object]:
        if not blocks:
            raise ValueError("No blocks provided for paired evaluation.")

        # Ensure no held-out seeds were evaluated prematurely
        self.check_held_out_isolation([b.seed for b in blocks])

        diffs = [b.paired_difference for b in blocks]
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

        # Permutation test for paired differences (exact for small N <= 12)
        n = len(diffs)
        num_perms = 2 ** n
        greater_or_equal = 0
        for i in range(num_perms):
            signs = [(1 if (i >> j) & 1 else -1) for j in range(n)]
            perm_mean = sum(d * s for d, s in zip(diffs, signs)) / n
            if perm_mean >= mean_diff:
                greater_or_equal += 1
        p_value = greater_or_equal / num_perms

        # Guardrail checks
        total_b_overruns = sum(b.policy_b_solver_overruns for b in blocks)
        total_b_offers = sum(b.offered_requests for b in blocks)
        overrun_rate = total_b_overruns / total_b_offers if total_b_offers > 0 else 0.0

        margin_satisfied = mean_diff >= self.min_superiority_margin
        statistically_significant = p_value < 0.05
        guards_passed = overrun_rate <= self.max_solver_overrun_rate

        is_superior = margin_satisfied and statistically_significant and guards_passed

        return {
            "n_blocks": n,
            "mean_goodput_difference_req_s": mean_diff,
            "std_goodput_difference": std_diff,
            "pre_registered_margin_req_s": self.min_superiority_margin,
            "margin_satisfied": margin_satisfied,
            "permutation_p_value": p_value,
            "statistically_significant": statistically_significant,
            "solver_overrun_rate": overrun_rate,
            "guards_passed": guards_passed,
            "verdict": (
                "CONFIRMED_SUPERIOR" if is_superior
                else "EQUIVALENT_OR_INSUFFICIENT"
            ),
            "details": (
                f"Mean delta: {mean_diff:+.3f} req/s (req >= {self.min_superiority_margin:.3f}), "
                f"p-val: {p_value:.4f}, overrun rate: {overrun_rate:.4f}"
            )
        }
