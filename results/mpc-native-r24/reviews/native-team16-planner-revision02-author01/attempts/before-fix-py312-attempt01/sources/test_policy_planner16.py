"""Synthetic algorithm/guard controls, not posterior or native-win evidence."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import product
import math
from types import SimpleNamespace

import pytest

import native_response_planner as original
import policy_planner16 as api
import test_policy_planner15 as old_tests
from test_policy_planner15 import (
    test_absent_fixed_action_refuses,
    test_all_work_charged_and_partial_panel_never_selected,
    test_bad_provenance_refuses_whole_panel,
    test_common_types_are_original_identities,
    test_configuration_refuses_unfrozen_or_ambiguous_choice,
    test_exact_ties_keep_original_transition_sum_lexicographic_order,
    test_full_weighted_panel_unchanged,
    test_real_rollouts_policies_same_panel_different_paths,
    test_stale_first_action_refuses_before_selection,
    test_stale_reuse_consumed_even_after_refusal,
)
from test_predecessor_witnesses import (
    bounded_materialization, external_world_mutation, frozen_clock,
    internal_world_mutation, positive_weight_scale_invariance,
)
from test_r24_historical_scenario import history_case, planning_case


@pytest.fixture(autouse=True)
def inherited_controls_use_derived_class(monkeypatch):
    monkeypatch.setattr(old_tests, "api", api)
    monkeypatch.setattr(api, "PolicyPlanner15", api.PolicyPlanner16)


def planner(case, policy="mpc", **changes):
    return api.PolicyPlanner16(**{
        "cost_spec": case.planner().cost_spec, "budgets": (16, 32),
        "step_offsets_s": (0.05, 0.25), "solver_budget_s": 0.04,
        "actuation_budget_s": 0.005, "policy": policy, **changes,
    })


def no_ranked_output(result):
    assert result["status"] == "refused" and result["first_action"] is None
    assert all(key not in result for key in api.RANKED_FIELDS)
    assert not result["native_mpc_permitted"] and not result["deployable"]


def test_fixed_bounded_iterable_regression(planning_case, monkeypatch):
    bounded_materialization(planning_case, monkeypatch, api.PolicyPlanner16)


def test_fixed_external_world_regression(planning_case, monkeypatch):
    external_world_mutation(planning_case, monkeypatch, api.PolicyPlanner16)


def test_fixed_internal_world_regression(planning_case, monkeypatch):
    internal_world_mutation(planning_case, monkeypatch, api.PolicyPlanner16)


def test_fixed_positive_weight_regression(planning_case, monkeypatch):
    positive_weight_scale_invariance(planning_case, monkeypatch, api.PolicyPlanner16)


def install_tape(monkeypatch, offsets, scores, walls):
    """Explicit score-table double: never described as a native rollout."""
    calls = []

    class Actor:
        def __init__(self, _offers, *, frontier, budget_steps, **_kwargs):
            current = frontier.source_cut["owner_prefix"]["snapshot"]["budget"]
            entry_offset = frontier.decision_at - frontier.source.source_at
            self.path = []
            for offset in offsets:
                budget = current
                for step in budget_steps:
                    if step["at_s"] <= entry_offset + offset:
                        budget = step["budget"]
                self.path.append(budget)

    def simulate(requests, _eos, model, _dispatch, *, admission, **_kwargs):
        world, path = model["oracle_world"], tuple(admission.path)
        calls.append((world, path))
        return {
            "all_offered": len(requests),
            "natural_slo_good": scores[world][path],
            "cohort_wall_s": walls[world][path],
        }

    monkeypatch.setattr(original, "HistoricalAdmissionScenario", Actor)
    monkeypatch.setattr(original, "simulate", simulate)
    monkeypatch.setattr(original, "score_cohort", lambda outcome, _scenario: outcome)
    return calls


def exact_oracle(paths, received, weights, scores, walls):
    denominator = sum(map(Fraction, weights))
    ranks = {}
    for path in paths:
        good = sum(Fraction(w) * scores[index][path] for index, w in enumerate(weights))
        wall = sum(Fraction(w) * Fraction(walls[index][path])
                   for index, w in enumerate(weights))
        transitions = sum(a != b for a, b in zip((received,) + path, path))
        ranks[path] = (-good / denominator, wall / denominator, transitions, sum(path), path)
    return sorted(paths, key=ranks.__getitem__), ranks


def one_move_paths(budgets, horizon, received):
    return {
        (received,) * pivot + (target,) * (horizon - pivot)
        for pivot in range(horizon + 1) for target in budgets
    }


def policies():
    return (
        ("mpc", {}), ("strong_one_move", {}), ("fixed16", {}),
        ("fixed32", {}), ("train_selected_fixed", {"fixed_budget": 16}),
    )


def check_tape_arms(case, monkeypatch, budgets, offsets, scores, walls):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    calls = install_tape(monkeypatch, offsets, scores, walls)
    weights = (3, 1)
    panel = tuple(replace(
        case.scenario, weight=weight,
        model={**case.scenario.model, "oracle_world": index})
        for index, weight in enumerate(weights))
    paths = list(product(budgets, repeat=len(offsets)))
    order, ranks = exact_oracle(paths, 16, weights, scores, walls)
    allowed_one_move = one_move_paths(budgets, len(offsets), 16)
    reference = None
    for policy, extra in policies():
        result = planner(
            case, policy, budgets=budgets, step_offsets_s=offsets, **extra).plan(
                panel, clock_now=case.clock_now)
        assert result["status"] == "model_plan_only"
        assert result["evaluations_attempted"] == len(paths) * len(panel)
        assert [tuple(row["path"]) for row in result["rankings"]] == order
        assert {tuple(row["path"]) for row in result["rankings"]
                if row["transitions"] <= 1} == allowed_one_move
        for row in result["rankings"]:
            rank = ranks[tuple(row["path"])]
            assert Fraction(row["expected_natural_slo_good"]) == -rank[0]
            assert Fraction(row["expected_cohort_wall_s"]) == rank[1]
        selected = {
            "mpc": order[0],
            "strong_one_move": min(allowed_one_move, key=ranks.__getitem__),
            "fixed16": (16,) * len(offsets),
            "fixed32": (32,) * len(offsets),
            "train_selected_fixed": (16,) * len(offsets),
        }[policy]
        assert tuple(result["selected_plan"]["path"]) == selected
        assert not result["first_action"]["execute_permitted"]
        common = (result["rankings"], result["common_panel_sha256"],
                  result["planner_contract_sha256"])
        if reference is not None:
            assert common == reference
        reference = common
    expected_calls = [(world, path) for world in range(2) for path in paths]
    assert calls == expected_calls * 5


@pytest.mark.parametrize("bits", range(16))
def test_exact_all_binary_payoff_tables_h2(planning_case, monkeypatch, bits):
    budgets, offsets = (16, 32), (0.05, 0.25)
    paths = list(product(budgets, repeat=2))
    scores = [
        {path: (bits >> index) & 1 for index, path in enumerate(paths)},
        {path: 1 - ((bits >> index) & 1) for index, path in enumerate(paths)},
    ]
    walls = [{path: Fraction((index + world) % 4, 8)
              for index, path in enumerate(paths)} for world in range(2)]
    walls = [{path: float(value) for path, value in world.items()} for world in walls]
    check_tape_arms(planning_case, monkeypatch, budgets, offsets, scores, walls)


@pytest.mark.parametrize("horizon", (2, 3, 4))
def test_exact_three_budget_delayed_strong_one_move(planning_case, monkeypatch, horizon):
    budgets = (32, 24, 16)
    offsets = (0.05, 0.15, 0.25, 0.35)[:horizon]
    paths = list(product(budgets, repeat=horizon))
    delayed = (16,) * (horizon - 1) + (32,)
    scores = [{path: 5 if path == delayed else 1 for path in paths} for _ in range(2)]
    walls = [{path: 0.25 for path in paths} for _ in range(2)]
    check_tape_arms(planning_case, monkeypatch, budgets, offsets, scores, walls)


def independent_cohort_score(outcome, scenario):
    scheduled = scenario.scheduled_offsets
    if not scheduled:
        return 0, 0.0
    good, terminals = 0, []
    rejected_at = {row["request_id"]: row["at_s"] for row in outcome["admission"]["decisions"]}
    for rid, start in scheduled.items():
        row = outcome["requests"][rid]
        if row["status"] == "rejected":
            terminals.append(rejected_at[rid] + scenario.rejection_http_lag_s)
            continue
        first, end, tokens = (
            row["predicted_http_first_s"], row["predicted_http_terminal_s"], row["output_tokens"])
        terminals.append(end)
        if row["finish_reason"] == "stop" and first - start <= 0.5:
            if tokens == 1:
                good += end - start <= 0.5
            elif tokens > 1:
                good += (end - first) / (tokens - 1) <= 0.05
    return good, max(terminals) - min(scheduled.values())


@pytest.mark.parametrize("horizon", (2, 3, 4))
def test_generated_native_rollouts_match_independent_exhaustion(
        planning_case, monkeypatch, horizon):
    case = planning_case
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    offsets = (0.05, 0.15, 0.25, 0.35)[:horizon]
    owner = replace(
        case.scenario, requests=case.scenario.requests[:1], offers=(),
        eos_lengths={"r": 2}, lags={"r": case.scenario.lags["r"]},
        scheduled_offsets={"r": -0.2}, latent_offer_clocks=(), weight=1)
    panel = (replace(case.scenario, weight=3), owner)
    expected = {}
    with original.OwnerSourceAudit(reuse_received_prefix=True) as audit:
        for path in product((16, 32), repeat=horizon):
            values = []
            previous, steps = 16, []
            for offset, budget in zip(offsets, path):
                if budget != previous:
                    steps.append({
                        "at_s": case.scenario.frontier.decision_offset_s + offset,
                        "budget": budget})
                previous = budget
            for scenario in panel:
                actor = original.HistoricalAdmissionScenario(
                    scenario.offers, frontier=scenario.frontier,
                    cost_spec=case.planner().cost_spec, initial_budget=16,
                    budget_steps=steps, lags=scenario.lags,
                    http_law=scenario.http_law, source_audit=audit)
                outcome = original.simulate(
                    scenario.requests, scenario.eos_lengths, scenario.model,
                    scenario.dispatch_table, scope=scenario.scope,
                    warm_start=scenario.warm_start, admission=actor)
                values.append(independent_cohort_score(outcome, scenario))
            expected[path] = (
                Fraction(3 * values[0][0] + values[1][0], 4),
                (3 * Fraction(values[0][1]) + Fraction(values[1][1])) / 4)
    result = planner(case, step_offsets_s=offsets).plan(panel, clock_now=case.clock_now)
    assert result["status"] == "model_plan_only"
    assert result["evaluations_attempted"] == 2 * 2 ** horizon
    for row in result["rankings"]:
        good, wall = expected[tuple(row["path"])]
        assert row["expected_natural_slo_good"] == float(good)
        assert row["expected_cohort_wall_s"] == pytest.approx(float(wall), abs=1e-14)
    oracle_rank = lambda path: (
        -expected[path][0], float(expected[path][1]),
        sum(a != b for a, b in zip((16,) + path, path)), sum(path), path)
    assert tuple(result["best_multistep"]["path"]) == min(expected, key=oracle_rank)
    assert tuple(result["best_one_move"]["path"]) == min(
        one_move_paths((16, 32), horizon, 16), key=oracle_rank)


def test_maximum_complete_panel_is_1296_not_pruned(history_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    frontier, _ = history_case.build(owned=False)
    empty = api.NativeScenario(
        (), (), {}, history_case.model, history_case.dispatcher, frontier, None,
        {}, {}, "native-endpoint-offsets-v1", 0)
    panel = tuple(replace(empty, weight=index + 1) for index in range(16))
    result = api.PolicyPlanner16(
        cost_spec=history_case.cost, budgets=(32, 24, 16),
        step_offsets_s=(0.05, 0.15, 0.25, 0.35),
        solver_budget_s=0.04, actuation_budget_s=0.005, policy="mpc").plan(
            panel, clock_now=lambda: api.PlanningEntry("clock", 10.1))
    assert result["status"] == "model_plan_only"
    assert result["evaluations_attempted"] == result["evaluations_expected"] == 1296
    assert len(result["rankings"]) == 81 and result["scenario_count"] == 16
    assert all(len(row["per_scenario"]) == 16 for row in result["rankings"])
    paths = list(product((32, 24, 16), repeat=4))
    expected = sorted(paths, key=lambda path: (
        sum(a != b for a, b in zip((16,) + path, path)), sum(path), path))
    assert [tuple(row["path"]) for row in result["rankings"]] == expected


def test_extreme_equal_weights_do_not_overflow_objective(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    expected = planner(case).plan([case.scenario] * 2, clock_now=case.clock_now)
    actual = planner(case).plan(
        [replace(case.scenario, weight=1e308)] * 2, clock_now=case.clock_now)
    assert actual["status"] == "model_plan_only"
    assert [row["path"] for row in actual["rankings"]] == [
        row["path"] for row in expected["rankings"]]
    for left, right in zip(actual["rankings"], expected["rankings"]):
        assert left["expected_natural_slo_good"] == right["expected_natural_slo_good"]
        assert left["expected_cohort_wall_s"] == pytest.approx(
            right["expected_cohort_wall_s"], abs=1e-14)


def test_unrepresentable_positive_weight_never_silently_dropped(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    panel = [replace(case.scenario, weight=math.ulp(0.0)),
             replace(case.scenario, weight=1e308)]
    with pytest.raises(ValueError, match="positive_weight_lost"):
        planner(case).plan(panel, clock_now=case.clock_now)


def test_partially_underflowed_weight_ratio_also_refuses(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    panel = [replace(case.scenario, weight=1e-14),
             replace(case.scenario, weight=1e308)]
    with pytest.raises(ValueError, match="weight_ratio_not_exactly_representable"):
        planner(case).plan(panel, clock_now=case.clock_now)


def test_empty_draw_retains_its_weight_in_nonempty_panel(history_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = history_case
    frontier, _ = case.build(owned=False)
    empty = api.NativeScenario(
        (), (), {}, case.model, case.dispatcher, frontier, None, {}, {},
        "native-endpoint-offsets-v1", 0, weight=3)
    request = case.Request("new", 128, 4, 0.2)
    occupied = replace(
        empty, requests=(request,), offers=(request,), eos_lengths={request.id: 1},
        lags={request.id: {key: 0.0 for key in case.lag_fields}},
        scheduled_offsets={request.id: 0.2}, weight=1,
        latent_offer_clocks=(original.LatentPublicOfferClock(request.id, 0.2, 0.2),))
    result = api.PolicyPlanner16(
        cost_spec=case.cost, budgets=(16, 32), step_offsets_s=(0.05, 0.25),
        solver_budget_s=0.04, actuation_budget_s=0.005, policy="mpc").plan(
            [empty, occupied], clock_now=lambda: api.PlanningEntry("clock", 10.1))
    assert result["status"] == "model_plan_only" and result["evaluations_attempted"] == 8
    for row in result["rankings"]:
        zero, nonempty = row["per_scenario"]
        assert zero == {"all_offered": 0, "natural_slo_good": 0, "cohort_wall_s": 0.0}
        assert row["expected_natural_slo_good"] == nonempty["natural_slo_good"] / 4
        assert row["expected_cohort_wall_s"] == nonempty["cohort_wall_s"] / 4


@pytest.mark.parametrize("stage", (
    "entry_clock", "materialize", "copy", "snapshot_hash", "audit_entry",
    "rollout", "selection", "audit_exit", "output", "version_return",
))
def test_every_added_stage_charged_before_release(planning_case, monkeypatch, stage):
    case = planning_case
    now, audited, hashed, calls = [0.0], [], [], []
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(api, "time", clock)
    monkeypatch.setattr(original, "time", clock)
    import live_owner_prefix as prefix
    real_copy, real_hash, real_audit = (
        api.deepcopy, api._fingerprint, prefix._verified_source_implementation)
    real_simulate, real_select, real_dump = (
        original.simulate, api.PolicyPlanner16._select_policy, api.json.dumps)

    def delayed_clock():
        if stage == "entry_clock":
            now[0] = 0.3
        return case.clock_now()

    def panel():
        if stage == "materialize":
            now[0] = 0.3
        yield case.scenario
        yield case.scenario

    def copy(value):
        result = real_copy(value)
        if stage == "copy":
            now[0] = 0.3
        return result

    def fingerprint(value):
        result = real_hash(value)
        if type(value) is tuple:
            hashed.append(True)
            if stage == "snapshot_hash":
                now[0] = 0.3
        return result

    def audit():
        result = real_audit()
        audited.append(True)
        if stage == "audit_entry" and len(audited) == 1:
            now[0] = 0.3
        if stage == "audit_exit" and len(audited) == 2:
            now[0] = 0.3
        return result

    def simulate(*args, **kwargs):
        result = real_simulate(*args, **kwargs)
        calls.append(True)
        if stage == "rollout":
            now[0] = 0.3
        return result

    def select(self, result):
        real_select(self, result)
        if stage == "selection":
            now[0] = 0.3

    def dump(*args, **kwargs):
        result = real_dump(*args, **kwargs)
        if stage == "output" and type(args[0]) is dict and "policy" in args[0]:
            now[0] = 0.3
        return result

    versions = []

    def version_now():
        versions.append(True)
        if stage == "version_return" and len(versions) == 3:
            now[0] = 0.3
        return 1

    monkeypatch.setattr(api, "deepcopy", copy)
    monkeypatch.setattr(api, "_fingerprint", fingerprint)
    monkeypatch.setattr(prefix, "_verified_source_implementation", audit)
    monkeypatch.setattr(original, "simulate", simulate)
    monkeypatch.setattr(api.PolicyPlanner16, "_select_policy", select)
    monkeypatch.setattr(api.json, "dumps", dump)
    p = planner(case, step_offsets_s=(0.2, 0.3), solver_budget_s=0.18)
    result = p.plan_acquisition(
        api.NativePanelAcquisition(panel(), case.clock_now(), 1),
        version_now=version_now, clock_now=delayed_clock)
    no_ranked_output(result)
    assert result["timing_refused"] and result["solver_elapsed_s"] == 0.3
    expected = 0 if stage in (
        "entry_clock", "materialize", "copy", "snapshot_hash", "audit_entry") else (
            1 if stage == "rollout" else 8)
    assert result["evaluations_attempted"] == expected


@pytest.mark.parametrize("entry_at,accepted", ((10.25, True), (10.25000001, False)))
def test_source_age_half_second_boundary_unchanged(
        planning_case, monkeypatch, entry_at, accepted):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    result = planner(
        planning_case, step_offsets_s=(0.25, 0.35), solver_budget_s=0.2).plan(
            [planning_case.scenario], clock_now=lambda: api.PlanningEntry("clock", entry_at))
    assert (result["status"] == "model_plan_only") is accepted
    if not accepted:
        no_ranked_output(result)
        assert result["evaluations_attempted"] == 0 and result["timing_refused"]


@pytest.mark.parametrize("when", ("rollout", "output"))
def test_version_change_refuses_whole_panel_and_consumes_version(
        planning_case, monkeypatch, when):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case, version = planning_case, [1]
    simulate, dump = original.simulate, api.json.dumps

    def changed(*args, **kwargs):
        result = simulate(*args, **kwargs)
        if when == "rollout":
            version[0] = 2
        return result

    def serialized(*args, **kwargs):
        result = dump(*args, **kwargs)
        if when == "output" and type(args[0]) is dict and "policy" in args[0]:
            version[0] = 2
        return result

    monkeypatch.setattr(original, "simulate", changed)
    monkeypatch.setattr(api.json, "dumps", serialized)
    p = planner(case)
    acquired = api.NativePanelAcquisition((case.scenario,) * 2, case.clock_now(), 1)
    result = p.plan_acquisition(acquired, version_now=lambda: version[0])
    no_ranked_output(result)
    assert result["evaluations_attempted"] == 8
    assert any("observation_changed" in reason for reason in result["policy_guard_refusals"])
    with pytest.raises(ValueError, match="newer_version"):
        p.plan_acquisition(acquired, version_now=lambda: 1)


@pytest.mark.parametrize("version", (True, -1, 1.0, None))
def test_untyped_acquisition_version_refuses(planning_case, version):
    case = planning_case
    with pytest.raises(ValueError, match="typed_versioned"):
        planner(case).plan_acquisition(
            api.NativePanelAcquisition((case.scenario,), case.clock_now(), version),
            version_now=lambda: version)


def test_legacy_and_versioned_entrypoints_keep_same_plan(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    direct = planner(case).plan([case.scenario], clock_now=case.clock_now)
    bound = planner(case).plan_acquisition(
        api.NativePanelAcquisition((case.scenario,), case.clock_now(), 7),
        version_now=lambda: 7)
    assert direct["status"] == bound["status"] == "model_plan_only"
    assert direct["rankings"] == bound["rankings"]
    assert direct["common_panel_sha256"] == bound["common_panel_sha256"]
    assert direct["observation_version"] is None and bound["observation_version"] == 7


def test_clock_domain_change_is_not_a_newer_source(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    p = planner(case)
    assert p.plan([case.scenario], clock_now=case.clock_now)["status"] == "model_plan_only"
    frontier = replace(
        case.scenario.frontier, decision_at=11.1,
        source=replace(case.scenario.frontier.source, source_at=11, clock_domain="other"))
    with pytest.raises(ValueError, match="reobserve_new_source"):
        p.plan([replace(case.scenario, frontier=frontier)],
               clock_now=lambda: api.PlanningEntry("other", 11.1))


def test_conflicting_actual_clocks_are_not_future_world_variation(planning_case):
    case = planning_case
    old = case.scenario.source_owner_clocks[0]
    changed = replace(
        case.scenario, source_owner_clocks=(replace(old, scheduled_at=9.7),),
        scheduled_offsets={**case.scenario.scheduled_offsets, "r": -0.3})
    with pytest.raises(ValueError, match="shared_actual_public_clocks"):
        planner(case).plan([case.scenario, changed], clock_now=case.clock_now)


@pytest.mark.parametrize("bad_clock", (float("nan"), float("inf"), -1.0))
def test_invalid_or_backward_return_clock_never_releases_plan(
        planning_case, monkeypatch, bad_clock):
    case, now = planning_case, [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(api, "time", clock)
    monkeypatch.setattr(original, "time", clock)
    simulate = original.simulate

    def invalidating_clock(*args, **kwargs):
        result = simulate(*args, **kwargs)
        now[0] = bad_clock
        return result

    monkeypatch.setattr(original, "simulate", invalidating_clock)
    result = planner(case).plan([case.scenario], clock_now=case.clock_now)
    no_ranked_output(result)
    assert result["solver_elapsed_s"] is None and result["timing_refused"]
    api.json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changes", (
    {"step_offsets_s": (0.05,)},
    {"step_offsets_s": (0.05, 0.15, 0.25, 0.35, 0.45)},
    {"budgets": (16, 24, 32, 48)},
    {"budgets": (16, 16)},
))
def test_original_action_and_horizon_bounds_unchanged(planning_case, changes):
    with pytest.raises(ValueError, match="bounded_multistep_delay_budget"):
        planner(planning_case, **changes)


def test_settings_changed_inside_solver_refuse_not_retarget(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    p = planner(case)
    simulate = original.simulate

    def changing(*args, **kwargs):
        result = simulate(*args, **kwargs)
        p._fixed15 = 32
        return result

    monkeypatch.setattr(original, "simulate", changing)
    result = p.plan([case.scenario], clock_now=case.clock_now)
    no_ranked_output(result)
    assert any("configuration_changed" in reason for reason in result["policy_guard_refusals"])


def test_runtime_solver_tightening_supported_but_enlargement_refused(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    p = planner(case)
    p.solver_budget_s = 0.02
    result = p.plan([case.scenario], clock_now=case.clock_now)
    assert result["status"] == "model_plan_only" and result["solver_budget_s"] == 0.02
    p = planner(case)
    p.solver_budget_s = 0.041
    with pytest.raises(ValueError, match="frozen_configuration"):
        p.plan([case.scenario], clock_now=case.clock_now)


def training_panel(case, monkeypatch, fixed16_good, fixed32_good):
    offsets = (0.05, 0.25)
    paths = list(product((16, 32), repeat=2))
    scores = [{path: fixed16_good if path == (16, 16)
               else fixed32_good if path == (32, 32) else 0 for path in paths}]
    walls = [{path: 0.25 for path in paths}]
    install_tape(monkeypatch, offsets, scores, walls)
    scenario = replace(case.scenario, model={**case.scenario.model, "oracle_world": 0})
    return planner(case).plan([scenario], clock_now=case.clock_now), scenario


def test_train_selected_fixed_is_global_frozen_before_evaluation(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    train_a, _ = training_panel(case, monkeypatch, 5, 0)
    train_b, _ = training_panel(case, monkeypatch, 0, 4)
    choice = api.freeze_fixed_choice(
        [train_a, train_b], split="train", panel_weights=(1, 2))
    assert choice.budget == 32
    selected = planner(case, "train_selected_fixed", training_choice=choice)
    with pytest.raises(FrozenInstanceError):
        choice.budget = 16
    train_a["rankings"].clear()
    evaluation, scenario = training_panel(case, monkeypatch, 5, 0)
    assert evaluation["best_fixed"]["path"] == [16, 16]
    result = selected.plan([scenario], clock_now=case.clock_now)
    assert result["status"] == "model_plan_only"
    assert result["selected_plan"]["path"] == [32, 32]
    assert result["training_fixed_choice"]["training_panel_sha256"] == choice.training_panel_sha256
    assert result["rankings"] == evaluation["rankings"]


def test_training_token_cannot_smuggle_mutable_evidence(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    result, _ = training_panel(planning_case, monkeypatch, 5, 0)
    choice = api.freeze_fixed_choice([result], split="train")
    with pytest.raises(ValueError, match="immutable_train_choice"):
        replace(choice, training_panel_sha256=list(choice.training_panel_sha256))


def test_training_weights_callback_cannot_rewrite_captured_training_panel(
        planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    result, _ = training_panel(planning_case, monkeypatch, 5, 0)

    def weights():
        for row in result["rankings"]:
            value = 5 if row["path"] == [32, 32] else 0
            row["expected_natural_slo_good"] = value
            row["per_scenario"][0]["natural_slo_good"] = value
        yield 1

    choice = api.freeze_fixed_choice([result], split="train", panel_weights=weights())
    assert choice.budget == 16


def test_training_weight_evidence_preserves_exact_integer_preference(
        planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    train_a, _ = training_panel(planning_case, monkeypatch, 5, 0)
    train_b, _ = training_panel(planning_case, monkeypatch, 0, 5)
    weights = (2 ** 53, 2 ** 53 + 1)
    choice = api.freeze_fixed_choice([train_a, train_b], split="train", panel_weights=weights)
    assert choice.budget == 32
    assert choice.training_weights == weights
    assert all(type(weight) is int for weight in choice.training_weights)


@pytest.mark.parametrize("defect", ("heldout", "refused", "partial", "tampered_score", "configuration"))
def test_training_selection_refuses_unfair_or_partial_inputs(
        planning_case, monkeypatch, defect):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    case = planning_case
    result, _ = training_panel(case, monkeypatch, 5, 0)
    split = "train"
    if defect == "heldout":
        split = "heldout"
    elif defect == "refused":
        result["status"] = "refused"
    elif defect == "partial":
        result["rankings"].pop()
    elif defect == "tampered_score":
        result["rankings"][0]["expected_natural_slo_good"] = 0
    else:
        choice = api.freeze_fixed_choice([result], split=split)
        with pytest.raises(ValueError, match="training_configuration"):
            planner(case, "train_selected_fixed", training_choice=choice,
                    step_offsets_s=(0.05, 0.3))
        return
    with pytest.raises(ValueError, match="policy16_"):
        api.freeze_fixed_choice([result], split=split)


def test_sequential_world_inputs_stay_supported_and_unchanged(planning_case, monkeypatch):
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    from test_r24_historical_scenario import (
        test_native_generated_plan_records_sequential_world_without_mutating_input as control)

    case = planning_case
    adapted = SimpleNamespace(
        scenario=case.scenario, clock_now=case.clock_now,
        planner=lambda **changes: planner(case, **changes))
    control(adapted)
