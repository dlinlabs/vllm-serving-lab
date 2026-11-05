"""Bounded Train-only CPU controls; no Eval tape, tokenizer run or Native claim."""

from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import nominal_scenario as ns
import nominal_workload as workload
import nominal_planner as planner
from mpc_lab.contracts import Snapshot
from mpc_lab.model import rollout as real_rollout


TRAIN_PATH = (
    Path(__file__).resolve().parents[3]
    / "results/mpc-native-r24/reviews/resume-p0-workload-author01/tapes-v1/train.tape.json"
)


class NominalScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = planner.NominalPlanner()
        with patch.object(workload, "generate_split", side_effect=AssertionError("No workload regeneration")):
            with patch.object(workload, "load_tape", side_effect=AssertionError("load_tape regenerates")):
                cls.prior = ns.freeze_train_prior(
                    TRAIN_PATH, purpose="train", clock_domain="literal-cpu-monotonic",
                )
        print("SCENARIO_RECEIPT " + json.dumps({
            "kind": "frozen-train-prior", "prior": asdict(cls.prior),
            "prior_content_sha256": cls.prior.content_sha256,
            "scope": "Train-only nominal CPU; tokenizer proof supplied by Mayor, not rerun",
        }, sort_keys=True), flush=True)

    def snapshot(self):
        return Snapshot(
            "literal-forecast-run", self.prior.clock_domain, self.prior.available_at + 0.01,
            16.0, 3, (), self.planner.identity, 7,
        )

    def test_repeatable_decision_separate_rng_domain_and_truthful_availability(self):
        snapshot = self.snapshot()
        first_clock = iter((snapshot.sampled_at + 0.01, snapshot.sampled_at + 0.03))
        second_clock = iter((snapshot.sampled_at + 0.04, snapshot.sampled_at + 0.06))
        original_open = Path.open
        allowed = {Path(ns.__file__).resolve(), Path(workload.__file__).resolve()}

        def source_only_open(path, *args, **kwargs):
            self.assertIn(path.resolve(), allowed, "Per-decision tape/data read")
            return original_open(path, *args, **kwargs)

        with patch.object(workload, "generate_split", side_effect=AssertionError("No generator calls")):
            with patch.object(Path, "open", new=source_only_open):
                first = ns.make_scenario(self.prior, snapshot, purpose="evaluation", clock=lambda: next(first_clock))
                second = ns.make_scenario(self.prior, snapshot, purpose="evaluation", clock=lambda: next(second_clock))
        self.assertEqual(first.causal.scenario, second.causal.scenario)
        self.assertEqual(first.rng_seed_material, second.rng_seed_material)
        self.assertEqual(first.causal.available_at, snapshot.sampled_at + 0.03)
        self.assertEqual(second.causal.available_at, snapshot.sampled_at + 0.06)
        self.assertEqual(first.causal.conditioned_on, planner.StateVersion.from_snapshot(snapshot))
        material = json.loads(first.rng_seed_material)
        self.assertNotEqual(material[0], workload.GENERATOR_ID)
        self.assertEqual(material[:2], [ns.RNG_NAMESPACE, ns.RNG_SEED])
        self.assertTrue(all(o.request_id.startswith(ns.RNG_NAMESPACE + ":")
                            and o.output_tokens == o.max_tokens == 128
                            and 0 <= o.offset_s < 15 for o in first.causal.scenario.offers))
        changed = ns.make_scenario(
            self.prior, replace(snapshot, sequence=8), purpose="evaluation",
            clock=lambda: snapshot.sampled_at + 0.07,
        )
        self.assertNotEqual(first.rng_seed_material, changed.rng_seed_material)
        self.assertNotEqual(first.causal.scenario, changed.causal.scenario)
        with self.assertRaises(FrozenInstanceError):
            self.prior.count = 1
        print("SCENARIO_RECEIPT " + json.dumps({
            "kind": "repeatability", "offers": len(first.causal.scenario.offers),
            "seed_material": first.rng_seed_material,
            "source_sha256": first.causal.source_sha256,
            "producer_sources": first.producer_sources,
            "scenario_sha256": planner.digest(asdict(first.causal.scenario)),
        }, sort_keys=True), flush=True)

    def test_prior_and_purpose_refuse_before_unapproved_file_access(self):
        with patch.object(Path, "open", side_effect=AssertionError("Must refuse before I/O")):
            with self.assertRaisesRegex(ns.ScenarioRefused, "train_only_prior"):
                ns.freeze_train_prior("not-opened", purpose="evaluation", clock_domain="unit")
            with self.assertRaisesRegex(ns.ScenarioRefused, "train_filename_required"):
                ns.freeze_train_prior("not-a-train-file", purpose="train", clock_domain="unit")
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ns.ScenarioRefused, "invalid_forecast_purpose"):
            ns.make_scenario(self.prior, snapshot, purpose="eval_future")
        with self.assertRaisesRegex(ns.ScenarioRefused, "unfrozen_train_prior"):
            ns.make_scenario(
                replace(self.prior, count=100), snapshot, purpose="evaluation",
                clock=lambda: snapshot.sampled_at,
            )

    def test_clock_causality_and_freshness_refuse(self):
        snapshot = self.snapshot()
        cases = (
            (replace(snapshot, clock_domain="other-boot"), self.prior, (snapshot.sampled_at,) * 2, "clock_domain_mismatch"),
            (snapshot, replace(self.prior, available_at=snapshot.sampled_at + 1),
             (snapshot.sampled_at,) * 2, "prior_not_available_at_source_time"),
            (snapshot, self.prior, (snapshot.sampled_at - 0.001,) * 2, "future_snapshot"),
            (snapshot, self.prior, (snapshot.sampled_at + 0.01, snapshot.sampled_at + 0.6), "stale_snapshot"),
            (snapshot, self.prior, (snapshot.sampled_at + 0.02, snapshot.sampled_at + 0.01), "nonmonotonic_clock"),
        )
        for state, prior, values, reason in cases:
            with self.subTest(reason=reason):
                clock = iter(values)
                with self.assertRaisesRegex(ns.ScenarioRefused, reason):
                    ns.make_scenario(prior, state, purpose="engineering", clock=lambda: next(clock))

    def test_overcap_refuses_without_truncation_or_resampling(self):
        snapshot = self.snapshot()
        with patch.object(ns, "Random") as random_type:
            rng = random_type.return_value
            rng.expovariate.return_value = 0.001
            rng.randrange.return_value = 0
            with self.assertRaisesRegex(ns.ScenarioRefused, "future_offer_cap_exceeded"):
                ns.make_scenario(
                    self.prior, snapshot, purpose="evaluation", clock=lambda: snapshot.sampled_at,
                )
            random_type.assert_called_once()
            self.assertEqual(rng.expovariate.call_count, 513)
            self.assertEqual(rng.randrange.call_count, 512)
        with patch.object(ns, "Random") as random_type:
            random_type.return_value.expovariate.return_value = 0.0
            with self.assertRaisesRegex(ns.ScenarioRefused, "nonadvancing_or_invalid_arrival"):
                ns.make_scenario(
                    self.prior, snapshot, purpose="engineering", clock=lambda: snapshot.sampled_at,
                )
            random_type.return_value.expovariate.assert_called_once()

    def test_real_cpu_panel_reuses_one_factory_scenario_for_both_policies(self):
        now = time.monotonic()
        snapshot = replace(self.snapshot(), sampled_at=now)
        forecast = ns.make_scenario(self.prior, snapshot, purpose="engineering")
        window = planner.DecisionWindow(now, now + 0.09, now + 0.1, now + 0.2)
        kwargs = {
            "clock_domain": snapshot.clock_domain,
            "read_state": lambda: planner.StateVersion.from_snapshot(snapshot),
        }
        with patch.object(planner, "rollout", wraps=real_rollout) as simulate:
            panel = self.planner.evaluate(snapshot, forecast.causal, window, **kwargs)
            mpc = self.planner.select(panel, policy="mpc", **kwargs)
            one_move = self.planner.select(panel, policy="one_move", **kwargs)
        self.assertEqual(simulate.call_count, 8)
        self.assertTrue(all(call.args[2] is forecast.causal.scenario for call in simulate.call_args_list))
        self.assertIs(panel.scenario, forecast.causal)
        self.assertLessEqual(one_move.plan.transitions, 1)
        self.assertGreaterEqual(mpc.predicted_lower, one_move.predicted_lower)
        print("SCENARIO_RECEIPT " + json.dumps({
            "kind": "real-cpu-panel", "scope": "nominal forecast integration only; not Native performance",
            "offers": len(forecast.causal.scenario.offers), "rollout_calls": simulate.call_count,
            "seed_material": forecast.rng_seed_material, "source_sha256": forecast.causal.source_sha256,
            "availability_after_sample_s": forecast.causal.available_at - snapshot.sampled_at,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
