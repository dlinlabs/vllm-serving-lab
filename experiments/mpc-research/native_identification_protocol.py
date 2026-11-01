"""Generate an unapproved, offline-only protocol; never generate held-out arrivals."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mpc_lab.contracts import digest, implementation_hash


def build_protocol():
    cells = [
        {"id": f"p{p}-m{m}", "prompt_tokens": p, "max_tokens": m}
        for p in (128, 4096) for m in (128, 256)
    ]
    episodes, groups = [], {}
    for index, seed in enumerate(range(160001, 160025)):
        split = ("train" if index < 12 else "calibration" if index < 16
                 else "validation" if index < 20 else "test")
        cell = cells[index % 4]
        groups[seed] = {"split": split, "cell": cell["id"]}
        rng = random.Random(f"id-v2-actions:{seed}")
        at, step, schedule = 0, 0, []
        while at < 90:
            schedule.append({"at_s": at, "budget": rng.choice((16, 32))})
            at += (2, 5, 10)[step % 3]
            step += 1
        arms = [0, 1]
        rng.shuffle(arms)
        for arm in arms:
            episodes.append({
                "episode_id": f"{split}-{seed}-{arm}", "seed_block": seed, "split": split,
                "cell": cell["id"], "prompt_tokens": cell["prompt_tokens"],
                "max_tokens": cell["max_tokens"],
                "arm": arm, "arrival_tape_key": f"id-v2-arrivals:{seed}",
                "prompt_tape_key": f"id-v2-prompts:{seed}",
                "request_id_rule": "unique 32-hex physical ID per arm/offer; preserve common logical offer ID",
                "natural_eos": True, "offers_s": 90, "maximum_drain_s": 20,
                "schedule": [{"at_s": s["at_s"],
                              "budget": s["budget"] if arm == 0 else 48 - s["budget"]}
                             for s in schedule],
            })
    result = {
        "schema": "native-identification-v2",
        "approved": False, "execution_enabled": False,
        "implementation_hash": implementation_hash(),
        "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "One exact RTX4090 deployment; exploratory identification, not controller acceptance",
        "independent_unit": "arrival/prompt seed block; both arms share the same offered tape",
        "groups": groups, "episodes": episodes,
        "pilot_capacity_reference": {cell["id"]: None for cell in cells},
        "rate_phases": [
            {"start_s": 0, "capacity_multiple": 0.6},
            {"start_s": 30, "capacity_multiple": 1.1},
            {"start_s": 60, "capacity_multiple": 0.6},
        ],
        "capacity_selection": {
            "scope": "TRAIN-only per-cell Fixed16 scout",
            "rates_req_s": [0.5, 1, 2, 4, 8, 16],
            "offers_s": 10, "maximum_drain_s": 10,
            "reference": "lowest rate maximizing observed successful goodput; abort if all rates have zero goodput",
            "freeze": "algorithm now; measured per-cell references before calibration/validation/test",
        },
        "primary": {
            "target": "reserved_cost_1s", "normalizer": 32,
            "forecast_origins_s": list(range(90)),
            "truth": "contiguous complete gateway ledger and final watermark; no polling interpolation",
            "aggregation": "mean decisions per episode, mean paired arms per seed, mean seeds",
            "baseline_selection": "complete leave-one-arrival-seed-out TRAIN folds; freeze hash before validation",
            "validation_screen": "10% normalized MAE improvement is exploratory only; no statistical pass",
        },
        "secondary": [
            "reserved_cost_3s", "reserved_cost_5s", "first_residence", "terminal_residence",
            "all-offered successful-goodput Brier bounds", "prediction availability",
        ],
        "brier": {
            "event": "complete matching HTTP response AND scheduled TTFT <= 0.5s",
            "cohort": "all scheduled offers, including admission rejects and errors",
            "prediction_time": "no later than client POST; no actual future arrivals/EOS features",
            "censoring": "unknown timely incomplete outcomes retain both possible losses",
        },
        "causal_estimand": "whole-sequence randomized assignment ITT; later divergent states are not matched",
        "action_gate": {
            "allowed_cost_budgets": [16, 32], "maximum_snapshot_age_s": 0.5,
            "maximum_command_ttl_s": 0.1,
            "model_version": "native-id-unfitted-v1",
            "calibration_version": "unfitted-not-validated",
            "mpc_authorized": False, "durable_pending_and_fencing_required": True,
        },
        "qualification": [
            "standard Chat framing/DONE/finish/request-ID and full offered cohort",
            "gateway ledger plus engine running/waiting/orphaned drain, fresh same clock",
            "exact software/model/tokenizer/cost/hardware/run identities",
            "assignment schedule versus actual CAS application and posting/state-read lag",
            "zero lost trace records/write errors; preserve all failed trials",
            "usage is not token timestamps; native token-event evidence required for TPOT tails",
        ],
        "stop_conditions": [
            "unsupported identity/clock/tokenizer or unresolved pending action",
            "stale state, failed drain, loss of trace completeness or resource authorization",
            "scout shows insufficient delivered action variation/state support; return unresolved",
        ],
        "budget": {
            "paid_resource_authorized": False, "maximum_billable_hours_proposal": 2.5,
            "maximum_all_in_usd_proposal": 5,
            "active_and_drain_seconds": 48 * 110,
            "setup_scout_export_contingency_seconds": 3720,
            "full_completion_guaranteed": False,
        },
        "confirmation": {
            "authorized": False, "new_independent_data_required": True,
            "comparators": ["Fixed16 cost budget", "TRAIN-tuned best fixed cost budget",
                            "simple feedback", "one-move", "MPC"],
            "sample_size": "choose once from development/pilot paired variance and power; freeze before Test",
            "superiority": "positive multiplicity-adjusted paired confidence bound on successful goodput",
            "equivalence_proposal": "TOST 90% CI inside +/-2% relative goodput, plus frozen safety/rejection bounds",
            "not_acceptance": [
                "CI contains zero", "all fallback/unused actions", "retuning on confirmation data",
                "ignoring solver/GET/POST wall time", "single-run or cross-GPU claims",
            ],
        },
    }
    return {**result, "protocol_hash": digest(result)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_protocol(), indent=2, allow_nan=False) + "\n")
