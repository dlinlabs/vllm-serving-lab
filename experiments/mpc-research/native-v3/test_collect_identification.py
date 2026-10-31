import importlib.util
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
