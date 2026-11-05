"""Literal CPU integration/refusal witnesses, not Native performance evidence."""

from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import nominal_planner as n
from mpc_lab.contracts import FutureOffer, RequestMark, RolloutError, Scenario, Snapshot, digest
from mpc_lab.model import rollout as real_rollout


class NominalPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = n.NominalPlanner()
        modules = {}
        for name in ("nominal_planner", "mpc_lab", "mpc_lab.contracts", "mpc_lab.controller", "mpc_lab.model"):
            path = Path(sys.modules[name].__file__).resolve()
            modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        cls.receipt("identity", identity=asdict(cls.planner.identity), modules=modules)

    @staticmethod
    def receipt(kind, **values):
        print("NOMINAL_RECEIPT " + json.dumps(
            {"kind": kind, "scope": "literal-CPU-only-not-Native", **values},
            sort_keys=True, allow_nan=False,
        ), flush=True)

    def case(self, *, budget=16.0, offers=None, purpose="engineering"):
        if offers is None:
            # At cap32 the slow, large-cost first offer blocks the later good
            # offer. Cap16 rejects both; a delayed increase admits only the good.
            offers = (
                FutureOffer("slow", 0.2, 32768, 1, 1),
                FutureOffer("good", 1.2, 0, 3300, 1),
            )
        now = time.monotonic()
        snapshot = Snapshot("cpu-test", "test-local-monotonic", now, budget, 7, (), self.planner.identity, 11)
        state = [n.StateVersion.from_snapshot(snapshot)]
        source = n.CausalScenario(
            Scenario(tuple(offers)), state[0], digest({"literal": "development-control"}),
            now, now, "development", purpose,
        )
        window = n.DecisionWindow(now, now + 0.09, now + 0.1, now + 0.2)
        kwargs = {"clock_domain": snapshot.clock_domain, "read_state": lambda: state[0]}
        return snapshot, source, window, state, kwargs

    def test_real_full_panel_distinguishes_mpc_from_strong_one_move(self):
        snapshot, source, window, _, kwargs = self.case(budget=32.0)
        with patch.object(n, "rollout", wraps=real_rollout) as forecast:
            panel = self.planner.evaluate(snapshot, source, window, **kwargs)
        self.assertEqual(forecast.call_count, 8)
        self.assertEqual({row.plan.actions for row in panel.candidates}, set(n.PATHS))
        mpc = self.planner.select(panel, policy="mpc", **kwargs)
        one = self.planner.select(panel, policy="one_move", **kwargs)
        self.receipt(
            "mpc-vs-one-move", rollout_calls=forecast.call_count,
            mpc_path=mpc.plan.actions, mpc_lower=str(mpc.predicted_lower),
            one_move_path=one.plan.actions, one_move_lower=str(one.predicted_lower),
        )
        self.assertEqual(mpc.plan.actions, (16.0, 32.0, 32.0))
        self.assertEqual(mpc.plan.transitions, 2)
        self.assertEqual(mpc.predicted_lower, Fraction(1))
        self.assertEqual(one.predicted_lower, Fraction(0))
        self.assertLessEqual(one.plan.transitions, 1)
        fixed_rows = tuple(row for row in panel.candidates if len(set(row.plan.actions)) == 1)
        self.assertEqual(max(row.forecast.score_lower for row in fixed_rows), 0)
        self.assertEqual(
            {row.plan.actions for row in panel.candidates if len(set(row.plan.actions)) == 1},
            {(16.0, 16.0, 16.0), (32.0, 32.0, 32.0)},
        )

    def test_real_one_move_can_delay_its_only_transition(self):
        snapshot, source, window, _, kwargs = self.case()
        decision = self.planner.plan(snapshot, source, window, policy="one_move", **kwargs)
        self.receipt(
            "delayed-one-move", path=decision.plan.actions,
            transitions=decision.plan.transitions, lower=str(decision.predicted_lower),
        )
        self.assertEqual(decision.plan.actions, (16.0, 32.0, 32.0))
        self.assertEqual(decision.plan.transitions, 1)
        self.assertEqual(decision.first_action.budget, 16.0)
        self.assertEqual(decision.predicted_lower, Fraction(1))

    def test_real_forecast_holds_received_budget_until_applicable_time(self):
        offers = (
            FutureOffer("during-solve", 0.01, 0, 3300, 1),
            FutureOffer("after-action", 0.2, 0, 3300, 1),
        )
        snapshot, source, window, _, kwargs = self.case(offers=offers)
        panel = self.planner.evaluate(snapshot, source, window, **kwargs)
        row = next(r for r in panel.candidates if r.plan.actions == (32.0,) * 3)
        outcomes = {o.request_id: o for o in row.forecast.outcomes}
        self.receipt(
            "causal-prefix", during_solve_accepted=outcomes["during-solve"].accepted,
            after_action_good=outcomes["after-action"].good(),
        )
        self.assertFalse(outcomes["during-solve"].accepted)
        self.assertTrue(outcomes["after-action"].good())
        first = window.first_action_at - snapshot.sampled_at
        self.assertEqual(row.plan.budget_at(first - 0.001), 16.0)
        self.assertEqual(row.plan.budget_at(first), 32.0)
        self.assertEqual(row.plan.budget_at(14.0), 32.0)
        decision = self.planner.select(panel, policy="mpc", **kwargs)
        self.assertEqual(decision.first_action.not_before, window.first_action_at)
        self.assertEqual(decision.first_action.source, n.StateVersion.from_snapshot(snapshot))

    def test_incomplete_duplicate_and_wrong_objective_panels_refuse(self):
        snapshot, source, window, _, kwargs = self.case()
        panel = self.planner.evaluate(snapshot, source, window, **kwargs)
        mutations = (
            (replace(panel, candidates=panel.candidates[:-1]), "incomplete_candidate_panel"),
            (replace(panel, candidates=panel.candidates[:-1] + panel.candidates[:1]), "incomplete_candidate_panel"),
            (replace(panel, objective_id="natural_slo_good"), "objective_or_contract_mismatch"),
        )
        for changed, reason in mutations:
            with self.subTest(reason=reason), self.assertRaisesRegex(n.PlanningRefused, reason):
                self.planner.select(changed, policy="mpc", **kwargs)

    def test_partial_forecast_and_solver_failure_never_return_a_winner(self):
        for mode, reason in (
            ("missing-outcome", "incomplete_forecast_cohort"),
            ("solver-refusal", "forecast_refused:event_limit"),
        ):
            with self.subTest(mode=mode):
                snapshot, source, window, _, kwargs = self.case()
                calls = []

                def fail_after_some_paths(*args, **kw):
                    result = real_rollout(*args, **kw)
                    calls.append(args[1].actions)
                    if len(calls) == 4:
                        if mode == "solver-refusal":
                            raise RolloutError("event_limit")
                        return replace(result, outcomes=result.outcomes[:-1])
                    return result

                with patch.object(n, "rollout", side_effect=fail_after_some_paths):
                    with self.assertRaisesRegex(n.PlanningRefused, reason):
                        self.planner.plan(snapshot, source, window, policy="mpc", **kwargs)
                self.assertEqual(len(calls), 4)

    def test_expired_backdated_and_incomplete_entry_refuses_before_forecast(self):
        for mode, reason in (
            ("stale", "stale_snapshot"),
            ("expired", "solve_deadline_exhausted"),
            ("backdated", "invalid_decision_window"),
            ("incomplete", "incomplete_snapshot"),
        ):
            with self.subTest(mode=mode):
                snapshot, source, window, _, kwargs = self.case()
                if mode == "stale":
                    snapshot = replace(snapshot, sampled_at=snapshot.sampled_at - 0.6)
                elif mode == "expired":
                    window = replace(window, entered_at=window.entered_at - 0.1, solve_deadline=window.entered_at)
                elif mode == "backdated":
                    window = replace(window, first_action_at=window.entered_at - 0.01)
                else:
                    snapshot = replace(snapshot, complete=False)
                with patch.object(n, "rollout") as forecast:
                    with self.assertRaisesRegex(n.PlanningRefused, reason):
                        self.planner.evaluate(snapshot, source, window, **kwargs)
                forecast.assert_not_called()

    def test_post_forecast_deadline_and_clock_regression_refuse(self):
        for mode, reason in (
            ("expired", "solve_deadline_exhausted"),
            ("regressed", "nonmonotonic_clock"),
        ):
            with self.subTest(mode=mode):
                snapshot, source, window, _, kwargs = self.case()
                now = [window.entered_at]

                def change_clock(*args, **kw):
                    result = real_rollout(*args, **kw)
                    now[0] = window.solve_deadline if mode == "expired" else window.entered_at - 0.001
                    return result

                with patch.object(n, "rollout", side_effect=change_clock):
                    with self.assertRaisesRegex(n.PlanningRefused, reason):
                        self.planner.evaluate(snapshot, source, window, clock=lambda: now[0], **kwargs)

    def test_clock_identity_and_mid_solve_versions_refuse(self):
        changes = (
            ({"clock_domain": "another-boot"}, "state_clock_changed"),
            ({"identity": replace(self.planner.identity, model_version="other")}, "state_identity_changed"),
            ({"budget_version": 8}, "state_version_changed"),
            ({"sequence": 12}, "state_version_changed"),
        )
        for changed, reason in changes:
            with self.subTest(changed=changed):
                snapshot, source, window, state, kwargs = self.case()

                def change_state(*args, **kw):
                    result = real_rollout(*args, **kw)
                    state[0] = replace(state[0], **changed)
                    return result

                with patch.object(n, "rollout", side_effect=change_state) as forecast:
                    with self.assertRaisesRegex(n.PlanningRefused, reason):
                        self.planner.evaluate(snapshot, source, window, **kwargs)
                self.assertEqual(forecast.call_count, 1)
        snapshot, source, window, _, kwargs = self.case()
        with self.assertRaisesRegex(n.PlanningRefused, "clock_domain_mismatch"):
            self.planner.evaluate(snapshot, source, window, **dict(kwargs, clock_domain="wrong"))
        with self.assertRaisesRegex(n.PlanningRefused, "identity_mismatch"):
            self.planner.evaluate(
                replace(snapshot, identity=replace(snapshot.identity, calibration_version="qualified")),
                source, window, **kwargs,
            )

    def test_future_or_unbound_scenario_and_mislabeled_objective_refuse(self):
        snapshot, source, window, _, kwargs = self.case()
        changes = (
            ({"origin": "evaluation_future"}, "unapproved_scenario_provenance"),
            ({"available_at": window.action_deadline}, "noncausal_scenario_information"),
            ({"information_cutoff_at": snapshot.sampled_at + 1}, "noncausal_scenario_information"),
            ({"conditioned_on": replace(source.conditioned_on, sequence=99)}, "scenario_state_mismatch"),
            ({"objective_id": "natural_slo_good"}, "objective_mismatch"),
        )
        for changed, reason in changes:
            with self.subTest(changed=changed), patch.object(n, "rollout") as forecast:
                with self.assertRaisesRegex(n.PlanningRefused, reason):
                    self.planner.evaluate(snapshot, replace(source, **changed), window, **kwargs)
                forecast.assert_not_called()

    def test_full_action_train_fixed_selection_has_no_runtime_solver_tax(self):
        offers = (FutureOffer("good", 0.2, 0, 3300, 1),)
        snapshot, source, window, _, kwargs = self.case(offers=offers, purpose="train")
        panel = self.planner.evaluate(snapshot, source, window, **kwargs)
        choice = self.planner.freeze_fixed((panel,))
        self.assertEqual(choice.candidate_totals, ((16.0, Fraction(0)), (32.0, Fraction(1))))
        self.assertEqual(choice.budget, 32.0)
        with patch.object(n, "rollout", side_effect=AssertionError("Fixed must not solve")) as forecast:
            decision = self.planner.fixed_action(snapshot, choice, window, **kwargs)
        self.receipt(
            "train-fixed", budget=choice.budget, runtime_rollout_calls=forecast.call_count,
            totals=[(budget, str(score)) for budget, score in choice.candidate_totals],
        )
        forecast.assert_not_called()
        self.assertEqual(decision.evaluated_paths, 0)
        self.assertEqual(decision.plan.actions, (32.0,) * 3)
        self.assertIsNone(decision.predicted_lower)
        with self.assertRaisesRegex(n.PlanningRefused, "fixed_selection_requires_train"):
            self.planner.freeze_fixed((replace(panel, scenario=replace(source, purpose="evaluation")),))
        with self.assertRaisesRegex(n.PlanningRefused, "incomplete_fixed_training_reference"):
            self.planner.fixed_action(
                snapshot, replace(choice, candidate_totals=choice.candidate_totals[:1]), window, **kwargs,
            )
        with self.assertRaisesRegex(n.PlanningRefused, "use_frozen_choice_for_fixed_execution"):
            self.planner.select(panel, policy="fixed_reference", **kwargs)
        invalid_ledger = replace(
            snapshot,
            requests=(RequestMark("bad-cost", 100.0, 0, 1, snapshot.sampled_at, snapshot.sampled_at),),
        )
        with self.assertRaisesRegex(n.PlanningRefused, "ledger_cost_mismatch"):
            self.planner.fixed_action(invalid_ledger, choice, window, **kwargs)


if __name__ == "__main__":
    unittest.main()
