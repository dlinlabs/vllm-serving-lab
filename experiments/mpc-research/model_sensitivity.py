"""Development-only horizon, censoring and parameter sensitivity, not policy gain."""

from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mpc_lab.contracts import ModelParameters, Plan, PlanningConfig, RequestMark, Snapshot
from mpc_lab.model import generate_scenarios, rollout


def run():
    base = ModelParameters()
    long = RequestMark(
        "old-hit", base.cost_spec.estimate(128, 4096), 128, 4096,
        99.7, 99.7, 99.9, 99.7,
    )
    pending = tuple(RequestMark(
        f"old-{i}", base.cost_spec.estimate(8192, 2048), 8192, 2048, 99, 99,
    ) for i in range(2))
    states = {"empty": (), "long-hit": (long,), "over-cap-awaiting": pending}
    settings = (
        ("control-1", base, 1, 10), ("reference", base, 3, 10),
        ("control-5", base, 5, 10), ("tail-4", base, 3, 4),
        ("tail-16", base, 3, 16),
        ("prefill-slower", replace(base, prefill_s_per_token=base.prefill_s_per_token * 1.2), 3, 10),
        ("decode-slower", replace(base, decode_s_per_token=base.decode_s_per_token * 1.2), 3, 10),
    )
    rows = []
    for state, marks in states.items():
        for label, params, steps, tail in settings:
            snapshot = Snapshot("development-sensitivity", "cpu-clock", 100, 32, 0, marks, params.identity)
            scenarios = generate_scenarios(
                snapshot, arrival_rate=14, prompt_tokens=(128, 512, 2048),
                max_tokens=(64, 128, 256), count=4, seed=2025, duration_s=21,
                service_scales=(0.8, 1.0, 1.3, 1.5),
            )
            config = PlanningConfig(control_steps=steps, tail_s=tail)
            best, best_key = None, None
            for actions in itertools.product((16.0, 32.0), repeat=steps):
                plan = Plan(actions)
                results = [rollout(snapshot, plan, scenario, params, config) for scenario in scenarios]
                for result in results:
                    assert all(
                        -1e-8 <= reserved <= max(snapshot.reserved, 32) + 1e-8
                        for _, reserved in result.reserved_trace
                    )
                lower = statistics.mean(r.score_lower for r in results)
                upper = statistics.mean(r.score_upper for r in results)
                changes = sum(a != b for a, b in zip((snapshot.budget,) + actions, actions))
                key = (lower, actions[0] == snapshot.budget, -changes)
                if best_key is None or key > best_key:
                    best_key = key
                    best = dict(
                        state=state, setting=label, actions=actions,
                        mean_lower=lower, mean_upper=upper,
                        unresolved_reward_width=upper - lower,
                        parameters=asdict(params), control_steps=steps, tail_s=tail,
                    )
            rows.append(best)
    return {
        "scope": "constructed CPU development sensitivity; no fitting, native uncertainty or policy-performance claim",
        "model_sha256": hashlib.sha256((ROOT / "mpc_lab/model.py").read_bytes()).hexdigest(),
        "contracts_sha256": hashlib.sha256((ROOT / "mpc_lab/contracts.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixed_evaluation_window_s": 5, "scenarios_per_plan": 4,
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
