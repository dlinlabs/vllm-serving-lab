"""Small exact witnesses against frozen Round1 claims, not a serving benchmark.

Reads two hash-verified, reviewed Python modules from the local Git object store.
No peer files are written, no native/raw workloads are loaded, and no network or
GPU is used. Failures below are the expected scientific output, not fixed peers.
"""

from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import types


FREEZE = "bc217acf9db3f6222026c0dd345db77a379d6151"
ROOT = Path(__file__).resolve().parents[4]
SOURCES = {
    "gemini_timing_contract": "8c9fbf413ed6e4baadf71ffe17d4eebfa18cdc1de844473c2bfa6edc77b2ee6f",
    "opus_batch_models": "0e6cc9787191875b6049f23dc4e4495ddda787a341010aec8708e78d05397502",
}


def frozen_module(name):
    path = f"experiments/mpc-research/native-v2/round1/{name}.py"
    payload = subprocess.check_output(["git", "show", f"{FREEZE}:{path}"], cwd=ROOT)
    if hashlib.sha256(payload).hexdigest() != SOURCES[name]:
        raise ValueError("frozen source hash differs")
    module_name = f"astra_review_{name}"
    module = types.ModuleType(module_name)
    module.__file__ = f"{FREEZE}:{path}"
    sys.modules[module_name] = module
    exec(compile(payload, module.__file__, "exec"), module.__dict__)
    return module


def exact_signflip_p(differences, margin=Fraction(0), *, two_sided=False):
    """Exact finite reference; a statistical interpretation needs a valid null."""
    centered = [Fraction(d) - margin for d in differences]
    observed = sum(centered)
    extreme = 0
    for signs in product((-1, 1), repeat=len(centered)):
        value = sum(d * sign for d, sign in zip(centered, signs))
        extreme += abs(value) >= abs(observed) if two_sided else value >= observed
    return float(Fraction(extreme, 2 ** len(centered)))


def protocol_witness(gemini):
    differences = [Fraction(1, 10)] * 7 + [Fraction(4)]
    blocks = [
        gemini.PairedEvaluationBlock(
            block_id=f"analytic-{i}", seed=160101 + i, offered_requests=10000,
            duration_s=10, policy_a_name="A", policy_a_good_completions=100,
            policy_a_rejections=1000, policy_a_solver_overruns=0,
            policy_b_name="B", policy_b_good_completions=100 + int(10 * difference),
            policy_b_rejections=8000, policy_b_solver_overruns=1,
        )
        for i, difference in enumerate(differences)
    ]
    protocol = gemini.NativeFairStatisticalProtocol()
    result = protocol.evaluate_paired_blocks(blocks)
    duplicates = protocol.evaluate_paired_blocks([blocks[-1]] * 8)
    return {
        "peer_verdict": result["verdict"],
        "mean_difference": result["mean_goodput_difference_req_s"],
        "peer_p": result["permutation_p_value"],
        "reference_two_sided_zero_null_p": exact_signflip_p(differences, two_sided=True),
        "reference_one_sided_margin_null_p": exact_signflip_p(differences, Fraction(1, 2)),
        "peer_solver_overrun_rate": result["solver_overrun_rate"],
        "overrun_rate_if_ten_decisions_per_block": 8 / 80,
        "rejection_fraction_A": 0.1,
        "rejection_fraction_B": 0.8,
        "eight_copies_of_one_block_verdict": duplicates["verdict"],
    }


def timing_witness(gemini):
    domain = gemini.ClockDomain("analytic", "one-boot")
    trace = gemini.RequestTimingTrace(
        request_id="negative-ttft", clock_domain=domain, prompt_tokens=128,
        output_tokens=1, scheduled_arrival=1, gateway_received=1.01,
        gateway_admitted=1.02, vllm_enqueued=None, first_token_sampled=None,
        first_content_delivered=0.9, stream_terminal=1.1,
    )
    trace.validate_monotonic_order()
    probe = gemini.ProbeOverheadRecord(
        probe_name="invalid-duration", clock_domain=domain, start_time=0,
        end_time=float("nan"), payload_bytes=1, is_blocking_gpu=False,
        is_tracing_cupti=False,
    )
    return {
        "invalid_trace_validated": True,
        "negative_ttft_s": trace.ttft_s,
        "peer_composite_good": trace.is_composite_good(),
        "nan_probe_accepted": gemini.ProbeOverheadGuard().validate_probe(probe, 1)[0],
    }


def warmup_witness(opus):
    rows = [
        dict(status="complete", sent_at=0.9, first_content_at=1.4),
        dict(status="complete", sent_at=1.1, first_content_at=1.5),
    ]
    episode = dict(prompt_tokens=4096, start=0, rows=rows)
    result = opus.ttft_by_prefill_ahead([episode], 4096)
    actual_ahead = sum(r["sent_at"] < 1.1 < r["first_content_at"] for r in rows)
    return {"peer_target_ahead": result[0]["ahead"], "actual_target_ahead": actual_ahead}


def randomized_assignment_confounding():
    """U independent Z, composition C=U+Z, latency Y=Z; C has zero causal effect."""
    worlds = [(u + z, z) for u, z in product((0, 1), repeat=2)]
    mean_c = Fraction(sum(c for c, _ in worlds), 4)
    mean_y = Fraction(sum(y for _, y in worlds), 4)
    slope = sum((c - mean_c) * (y - mean_y) for c, y in worlds) / sum(
        (c - mean_c) ** 2 for c, _ in worlds
    )
    return {"composition_regression_slope": float(slope),
            "true_do_composition_effect": 0, "assignment_independent_of_hidden_state": True}


def accounting_and_horizon_witnesses():
    initial, reserved, cap = 0, 0, 32
    for _ in range(4):
        assert reserved + 8 <= cap
        reserved += 8
    cap = 16
    # No release or eviction accompanies a decrease.
    return {
        "reservation_after_cap_reduction": reserved,
        "peer_current_cap_bound": max(initial, cap),
        "correct_historical_cap_bound": max(initial, 32),
        "missing_request_timing": {
            "timeline_ms": {
                "scheduled": 0, "gateway_received": 10, "admitted": 20,
                "engine_enqueued": 40, "chunk1_start": 50, "chunk1_end": 60,
                "chunk2_start": 160, "chunk2_end": 180, "sampled": 190, "first": 200,
            },
            "actual_ttft_ms": 200,
            "peer_named_terms_sum_ms": 10 + 10 + 10 + (10 + 20) + 10,
            "missing_bridge_wait_sample_ms": 20 + 100 + 10,
        },
        "long_memory_no_value": {
            "request_residence_s": 20, "control_slot_s": 1,
            "offered_costs": [4, 4], "peak_reserved_cost": 8,
            "all_budget_plans_accept_both": all(8 <= cap for cap in (16, 32)),
            "extra_good_from_more_moves": 0,
        },
        "rare_disagreement": {
            "control_slots": 100, "disagreeing_slots": 4,
            "disagreement_fraction": 0.04,
            "extra_good_per_disagreement": 16,
            "extra_good_per_second": 4 * 16 / 100,
            "assumption": "each selected slot begins R=16; 16 cost-1 offers; all complete before next selected slot",
        },
    }


def diagnostic():
    gemini, opus = (frozen_module(name) for name in SOURCES)
    return {
        "freeze": FREEZE, "peer_source_sha256": SOURCES,
        "native_data_used": False, "native_performance_claim": False,
        "statistical_contract": protocol_witness(gemini),
        "timing_contract": timing_witness(gemini),
        "warmup_history": warmup_witness(opus),
        "composition_confounding": randomized_assignment_confounding(),
        "mathematical_claims": accounting_and_horizon_witnesses(),
    }


if __name__ == "__main__":
    print(json.dumps(diagnostic(), indent=2, sort_keys=True, allow_nan=False))
