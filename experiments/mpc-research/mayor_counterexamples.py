"""Deterministic research counterexamples, not a serving benchmark."""
import json


def checks():
    results = {}

    reserved, fallback, mean_cost = 60.0, 32.0, 2.0
    terminal_limit = fallback + mean_cost
    assert reserved > terminal_limit
    results["hold_does_not_imply_terminal_feasibility"] = {
        "initial_reserved_cost": reserved,
        "releases_before_horizon": 0,
        "reserved_after_holding_fallback": reserved,
        "claimed_terminal_limit": terminal_limit,
        "feasible": False,
    }

    fixed_component = 0.259 + 0.0052083333 * 128
    token_coefficient = 0.000758
    reserved, slack, released, arriving_cost = 20.0, 1.0, 6.0, 6.5
    budget = reserved + slack
    prompt = (arriving_cost - fixed_component) / token_coefficient
    nominal_prompt_slack = (slack - fixed_component) / token_coefficient
    expected_release_tokens = 0.3 * 1.8 / token_coefficient
    assert reserved - released + arriving_cost <= budget
    assert 0 < prompt < 8192 - 128
    assert prompt > nominal_prompt_slack + expected_release_tokens
    results["expected_release_is_not_a_pathwise_bound"] = {
        "valid_arriving_prompt_tokens": prompt,
        "nominal_prompt_slack_tokens": nominal_prompt_slack,
        "mean_release_allowance_tokens": expected_release_tokens,
        "actual_release_cost": released,
        "new_request_fits": True,
    }

    legacy = {"version": 0, "budget": 32}
    legacy.update(version=1, budget=32)
    legacy.update(version=2, budget=16)
    guarded = {"version": 1, "budget": 32}
    old_command_version = 0
    old_command_accepted = old_command_version == guarded["version"]
    assert legacy["budget"] == 16 and not old_command_accepted
    results["timeout_then_old_state_read_does_not_cancel_command"] = {
        "timeline": [
            "10.1: A(cap16) sent",
            "10.2: A times out at client",
            "10.3: metrics still returns version0/cap32",
            "10.5: B(cap32) applies, version1",
            "10.9: delayed A applies, version2/cap16",
        ],
        "legacy_final": legacy,
        "cas_rejects_late_version0_command": not old_command_accepted,
        "server_expiry_would_also_be_required_without_a_later_write": True,
    }

    old_only = 2
    new_only_model_gain = 1
    actual_with_admission = 1
    assert new_only_model_gain > 0 and actual_with_admission - old_only < 0
    results["ignoring_existing_cohorts_reverses_action_preference"] = {
        "without_new_admission": {"old_hits": 2, "new_hits": 0},
        "with_new_admission": {"old_hits": 0, "new_hits": 1},
        "new_only_gain": new_only_model_gain,
        "actual_total_gain": actual_with_admission - old_only,
    }

    hit_streams, later_failed = 10, 2
    results["first_token_hit_is_not_final_success"] = {
        "first_token_hits": hit_streams,
        "eventual_successful_hits": hit_streams - later_failed,
    }

    early, late = [5, 0, 0, 0, 0], [0, 0, 0, 0, 5]
    assert sum(early) == sum(late)
    next_early, next_late = sum(early[1:]), sum(late[1:])
    assert next_early != next_late
    results["rolling_sum_without_expiration_memory_is_not_closed"] = {
        "equal_current_totals": sum(early),
        "next_totals_with_no_new_events": [next_early, next_late],
    }

    positive = lambda z, u: 1 + u
    negative = lambda z, u: 1 + 2 * z - u
    observations = []
    for z in (0, 1):
        observed_action = z
        assert positive(z, observed_action) == negative(z, observed_action)
        assert positive(z, 1) - positive(z, 0) == 1
        assert negative(z, 1) - negative(z, 0) == -1
        observations.append({"state": z, "action": z, "reward": 1 + z})
    results["perfect_factual_prediction_does_not_identify_action_effect"] = {
        "identical_observed_data": observations,
        "perfect_factual_predictor": "1 + state",
        "action_effect_in_world_one": 1,
        "action_effect_in_world_two": -1,
        "missing_condition": "intervention support or valid structural assumptions",
    }

    unobserved_violation_interval = (0.0, 1.0)
    claimed_limit = 0.001
    assert unobserved_violation_interval[1] > claimed_limit
    results["unobserved_kv_risk_is_not_a_calibrated_chance_constraint"] = {
        "risk_range_without_identifying_information": unobserved_violation_interval,
        "proposed_limit": claimed_limit,
        "certified": False,
    }
    return {
        "schema": "mpc-research-counterexamples-v1",
        "role": "executed arithmetic and logical falsification, not GPU evidence",
        "checks": results,
    }


if __name__ == "__main__":
    print(json.dumps(checks(), indent=2, allow_nan=False))
