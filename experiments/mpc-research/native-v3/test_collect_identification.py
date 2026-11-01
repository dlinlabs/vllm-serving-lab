import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("native_identification",
    Path(__file__).with_name("collect_identification.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complementary_assignments_share_independent_tape_not_future_state():
    protocol = module.protocol_for("a"*40, {}, smoke=True)
    first, second = protocol["episodes"][:2]
    assert first["seed_block"] == second["seed_block"]
    assert all(a["budget"]+b["budget"] == 48
               for a, b in zip(first["schedule"], second["schedule"]))
    assert module.tape_for(first["seed_block"], first["cell"], 12) == module.tape_for(
        second["seed_block"], second["cell"], 12)
    assert "not same-state" in protocol["contrasts"]


def test_actual_ten_hours_not_maximum_drain_or_setup_is_preregistered():
    protocol = module.protocol_for("a"*40, {})
    assert protocol["planned_offered_window_s"] == 48000
    assert protocol["minimum_requested_experiment_s"] == 36000
    blocks = {s:{e["seed_block"] for e in protocol["episodes"] if e["split"] == s}
              for s in ("train", "calibration", "validation")}
    assert tuple(map(len, blocks.values())) == (16, 20, 4)
    assert all(not blocks[a] & blocks[b] for a, b in (("train", "calibration"),
        ("train", "validation"), ("calibration", "validation")))
    assert protocol["final_test_used"] is False
    assert {protocol["episodes"][i]["arm"]
            for i in range(0, len(protocol["episodes"]), 2)} == {0, 1}
    assert protocol == module.protocol_for("a"*40, {})
    limit = protocol["gateway_epoch_episode_limit"]
    assert limit == 2
    for position in range(0, len(protocol["episodes"]), limit):
        pair = protocol["episodes"][position:position+limit]
        assert len({(e["seed_block"], e["cell"]) for e in pair}) == 1
        assert sum(len(e["schedule"]) for e in pair) < 1024


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_measurement_window_rejected(duration):
    with pytest.raises(ValueError):
        module.tape_for(1, "mixed", duration)


def test_native_wire_usage_and_capped_outcomes_do_not_invent_goodput():
    base = dict(status="complete", finish_reason="stop",
        usage={"completion_tokens":32}, first_content_at=10.1,
        scheduled_at=10.0, terminal_at=10.5)
    rows = [base, {**base, "finish_reason":"length"},
            {**base, "usage":{"completion_tokens":1}},
            {"status":"rejected"}]
    result = module.summary(rows, 10, .4)
    assert result["offered"] == 4
    assert result["natural_cohort_good"] == 1
    assert result["natural_in_window_good"] == 0
    assert result["capped"] == result["rejected"] == 1


def test_continuation_skips_only_verified_complete_train_blocks():
    full = module.protocol_for("a"*40, {})
    resumed = module.protocol_for("b"*40, {}, completed_blocks=3)
    assert resumed["episodes"] == full["episodes"][30:]
    assert resumed["planned_offered_window_s"] == 44400
    previous = dict(schema=module.SCHEMA, source_commit="a"*40,
        actual_offered_window_s=3600, episodes=full["episodes"][:30])
    assert module.continuation_evidence(resumed, previous)["complete_seed_blocks"] == 3
    for invalid in ({**previous, "actual_offered_window_s":3599},
                    {**previous, "episodes":previous["episodes"][:-1]}):
        with pytest.raises(ValueError):
            module.continuation_evidence(resumed, invalid)
    for count in (True, -1, 16):
        with pytest.raises(ValueError):
            module.protocol_for("a"*40, {}, completed_blocks=count)


def test_confirmed_hold_requires_fresh_unchanged_applied_provenance():
    from mpc_lab.contracts import ModelParameters, Snapshot

    parameters = ModelParameters()
    snapshot = Snapshot("run", "clock", 10.0, 32, 7, (), parameters.identity)
    acknowledgment = dict(run_id="run", clock_domain="clock", budget=32,
        budget_version=7, applied_at=9.9, operation_id="old-operation")
    result = module.held_budget_evidence(snapshot, acknowledgment, parameters, "clock", 10.1)
    assert result["status"] == "confirmed_hold"
    assert result["reference_operation_id"] == "old-operation"
    assert "command" not in result and "acknowledgment" not in result
    for invalid in (replace(snapshot, budget=16), replace(snapshot, budget_version=8),
                    replace(snapshot, run_id="other"), replace(snapshot, complete=False)):
        with pytest.raises(ValueError):
            module.held_budget_evidence(invalid, acknowledgment, parameters, "clock", 10.1)
    with pytest.raises(ValueError, match="stale"):
        module.held_budget_evidence(snapshot, acknowledgment, parameters, "clock", 10.6)
    with pytest.raises(ValueError, match="clock_domain"):
        module.held_budget_evidence(replace(snapshot, clock_domain="foreign", sampled_at=float("nan")),
            acknowledgment, parameters, "clock", 10.1)
