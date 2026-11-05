"""Reproduce the Fixed timing defect using the archived, unmodified candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.frozen_source_root.resolve()
    sys.path[:0] = [str(root / "experiments/mpc-research/resume-mvp"), str(root)]
    import nominal_planner as planner
    from test_nominal_planner import NominalPlannerTests

    source = Path(planner.__file__).resolve()
    expected = "746dc6b0e0174a5cb84617ad0e1c582df439d5a9e1cfd74cf8d520c05f5e12f8"
    if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("wrong_frozen_candidate")
    fixture = NominalPlannerTests()
    fixture.planner = planner.NominalPlanner()
    snapshot, scenario, window, _, kwargs = fixture.case(
        offers=(planner.FutureOffer("good", 0.2, 0, 3300, 1),), purpose="train",
    )
    panel = fixture.planner.evaluate(snapshot, scenario, window, **kwargs)
    choice = fixture.planner.freeze_fixed((panel,))
    decision = fixture.planner.fixed_action(snapshot, choice, window, **kwargs)
    delay = decision.first_action.not_before - time.monotonic()
    if delay <= 0 or decision.first_action.budget == snapshot.budget:
        raise RuntimeError("expected_artificial_delay_not_reproduced")
    now = time.monotonic()
    delayed_get = planner.DecisionWindow(now - 0.15, now - 0.06, now + 0.05, now + 0.2)
    try:
        fixture.planner.fixed_action(snapshot, choice, delayed_get, **kwargs)
    except planner.PlanningRefused as exc:
        reason = str(exc)
        if reason != "solve_deadline_exhausted":
            raise
    else:
        raise RuntimeError("expected_solver_only_refusal_not_reproduced")
    print(json.dumps({
        "status": "FIXED_BASELINE_ARTIFICIAL_DELAY_REPRODUCED",
        "source_sha256": expected,
        "first_action_delay_after_return_s": delay,
        "fixed_refused_with_live_action_deadline": reason,
        "native_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
