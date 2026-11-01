"""Small Round2 frozen-peer witnesses, not a simulator or controller benchmark."""

import functools
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "bc217acf9db3f6222026c0dd345db77a379d6151"
PREFIX = "experiments/mpc-research/native-v2/round1/"
SOURCES = {
    "gemini_timing_contract.py":
        "8c9fbf413ed6e4baadf71ffe17d4eebfa18cdc1de844473c2bfa6edc77b2ee6f",
    "opus_batch_models.py":
        "0e6cc9787191875b6049f23dc4e4495ddda787a341010aec8708e78d05397502",
    "astra_horizon_diagnostic.py":
        "a0230eb3f2f959a9a58978cd6434e3a221fecc67147b910df5dc2ca9f9d432be",
}


@functools.lru_cache(maxsize=3)
def frozen_module(filename):
    """Load only hash-pinned local Git blobs; do not extract or modify peers."""
    expected = SOURCES[filename]
    source = subprocess.check_output(
        ["git", "show", f"{FREEZE}:{PREFIX}{filename}"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("frozen peer source hash mismatch")
    name = "_sol_round2_" + filename.removesuffix(".py")
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / PREFIX / filename)
    sys.modules[name] = module
    try:
        exec(compile(source, f"{FREEZE}:{PREFIX}{filename}", "exec"), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return module


def statistical_witness():
    peer = frozen_module("gemini_timing_contract.py")

    def block(i, sign=1, *, rejection=1, overruns=0):
        return peer.PairedEvaluationBlock(
            block_id=f"constructed-{i}", seed=160001 + i,
            offered_requests=1000, duration_s=10,
            policy_a_name="A", policy_a_good_completions=500,
            policy_a_rejections=1, policy_a_solver_overruns=0,
            policy_b_name="B", policy_b_good_completions=500 + sign * 10,
            policy_b_rejections=rejection, policy_b_solver_overruns=overruns)

    protocol = peer.NativeFairStatisticalProtocol()
    blocks = [block(i, 1 if i < 7 else -1) for i in range(8)]
    diffs = [b.paired_difference for b in blocks]
    threshold = abs(sum(diffs))
    two_sided = sum(
        abs(sum(d * s for d, s in zip(diffs, signs))) >= threshold
        for signs in itertools.product((-1, 1), repeat=8)
    ) / 256
    duplicate = block(0)
    bad_guards = [block(i, rejection=100, overruns=2) for i in range(8)]
    return {
        "constructed_differences_req_s": diffs,
        "peer_result": protocol.evaluate_paired_blocks(blocks),
        "prespecified_two_sided_sign_flip_p": two_sided,
        "duplicate_same_block_result":
            protocol.evaluate_paired_blocks([duplicate] * 8),
        "bad_guard_result": protocol.evaluate_paired_blocks(bad_guards),
        "bad_guard_rejection_rate_ratio_B_over_A": 100,
        "bad_guard_true_overrun_fraction_if_10_solves_per_block": 2 / 10,
    }


def reservation_witness():
    initial, high, low, cost = 0, 32, 16, 8
    reserved = initial
    for _ in range(4):
        assert reserved + cost <= high
        reserved += cost
    return {
        "initial_reserved": initial, "budget_history": [high, low],
        "reserved_after_non_evicting_reduction": reserved,
        "gemini_current_budget_bound": max(initial, low),
        "correct_history_budget_bound": max(initial, high, low),
    }


def decomposition_witness():
    # One common clock, sequential chunks: missing terms cannot be blamed on skew.
    scheduled, received, admitted, enqueued = 0, .01, .02, .05
    first_batch, first_end, second_batch, second_end = .10, .20, .30, .40
    sampled, first_content = .42, .45
    claimed_sum = (
        received - scheduled + admitted - received + first_batch - enqueued
        + first_end - first_batch + second_end - second_batch
        + first_content - sampled)
    omitted = (
        enqueued - admitted + second_batch - first_end + sampled - second_end)
    return {"actual_ttft_s": first_content - scheduled,
            "claimed_decomposition_sum_s": claimed_sum,
            "omitted_forwarding_interchunk_sampling_s": omitted}


def persistent_but_no_multistep_gain():
    def evaluate(plan):
        active, admissions = [], []
        for slot, budget in enumerate(plan):
            active = [release for release in active if release > slot]
            accepted = 0
            for _ in range(20):
                if len(active) + 1 <= budget:
                    active.append(slot + 2)
                    accepted += 1
            admissions.append(accepted)
        return sum(admissions), admissions

    plans = tuple(itertools.product((16, 32), repeat=3))
    best = max(evaluate(plan)[0] for plan in plans)
    one_move_with_high_backup = max(
        evaluate((first, 32, 32))[0] for first in (16, 32))
    return {
        "reservation_holding_slots": 2,
        "high_plan_admissions_per_slot": evaluate((32, 32, 32))[1],
        "best_three_move_good_count": best,
        "one_move_with_common_high_backup_good_count": one_move_with_high_backup,
        "assumptions": "unit costs, 20 offers/slot, two-slot holding, "
                       "all admitted requests eventually meet SLO, no evictions",
        "scope": "finite exact-ledger counterexample to sufficiency; not native data",
    }


def opus_metric_witness():
    peer = frozen_module("opus_batch_models.py")
    row = dict(status="complete", scheduled_at=0., sent_at=.4, first_content_at=.6)
    result = peer.ttft_by_prefill_ahead(
        [dict(prompt_tokens=128, start=0., rows=[row])], 128, warmup_s=0)
    return {"peer_send_origin_slo_fraction": result[0]["within_slo_fraction"],
            "scheduled_origin_slo_fraction": int(.6 - 0 <= .5),
            "shadow_disagreement_fraction": 1 / 100,
            "possible_good_counts_one_move_and_multistep": [99, 119],
            "possible_relative_gain_despite_one_percent_disagreement": 20 / 99,
            "scope": "metric mismatch and a finite reward table, not a native effect"}


def tail_witness():
    peer = frozen_module("astra_horizon_diagnostic.py")

    def row(terminal, tokens):
        return dict(scheduled_at=0., first_content_at=.1, terminal_at=terminal,
                    status="complete", usage=dict(completion_tokens=tokens))

    a = [row(3, 100) for _ in range(3)]
    b = [row(.4, 21) for _ in range(2)]
    b.append(dict(scheduled_at=0., first_content_at=None, terminal_at=.4,
                  status="rejected", usage=None))
    early_a, early_b = peer.score_bounds(a, 1), peer.score_bounds(b, 1)
    return {
        "early_A": early_a, "early_B": early_b,
        "early_B_minus_A": peer.contrast(early_a, early_b),
        "eventual_A": peer.score_bounds(a, 3)["lower"],
        "eventual_B": peer.score_bounds(b, 3)["lower"],
        "scope": "Astra bounds work; selecting largest lower bound can still lose"}


def diagnostic():
    return {
        "schema": "sol-round2-frozen-peer-counterexamples-v1",
        "freeze_commit": FREEZE, "source_sha256": SOURCES,
        "native_data_used": False, "native_performance_claim": False,
        "statistics": statistical_witness(),
        "cap_reduction": reservation_witness(),
        "timing_decomposition": decomposition_witness(),
        "persistent_state": persistent_but_no_multistep_gain(),
        "opus_metric_and_disagreement": opus_metric_witness(),
        "tail_selection": tail_witness(),
    }


if __name__ == "__main__":
    print(json.dumps(diagnostic(), indent=2, allow_nan=False))
