from dataclasses import replace
import math

from test_policy_planner16 import (
    api, exact_oracle, frozen_clock, history_case, install_tape, no_ranked_output, planner,
    planning_case,
)


def test_nonzero_subnormal_products_cannot_silently_change_the_best_path(
        planning_case, monkeypatch):
    case = planning_case
    frozen_clock(monkeypatch, api.PolicyPlanner16)
    paths = [(16, 16), (16, 32), (32, 16), (32, 32)]
    weights = (1.0, 32 * math.ulp(0.0))
    scores = [{path: 0 for path in paths} for _ in range(2)]
    walls = [
        {path: 0.0 for path in paths},
        {path: 0.9375 if path == (32, 32) else 1.0 for path in paths},
    ]
    install_tape(monkeypatch, (0.05, 0.25), scores, walls)
    panel = tuple(replace(
        case.scenario, weight=weight,
        model={**case.scenario.model, "oracle_world": index},
    ) for index, weight in enumerate(weights))
    expected, _ = exact_oracle(paths, 16, weights, scores, walls)
    assert expected[0] == (32, 32)
    result = planner(case).plan(panel, clock_now=case.clock_now)
    if result["status"] == "refused":
        no_ranked_output(result)
    else:
        assert tuple(result["selected_plan"]["path"]) == expected[0], (
            "Common scaling rounded unequal positive products to a tie; "
            "the plan must refuse or preserve the exact objective ordering."
        )
