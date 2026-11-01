from dataclasses import asdict, replace

import pytest

from mpc_lab.contracts import ModelParameters, Snapshot
from mpc_lab.identification import (
    brier_bounds, cohort_brier, freeze_baseline, primary_forecast_mae, reserved_cost_at,
)


def test_complete_ledger_reconstructs_exact_horizon_and_rejects_gaps_or_loss():
    params = ModelParameters()
    initial = Snapshot("run", "clock", 0, 16, 0, (), params.identity)
    final = replace(initial, sampled_at=2, sequence=2)
    frames = [
        dict(event="mpc_ledger", run_id="run", clock_domain="clock", identity=asdict(params.identity),
             complete=True, sequence=i, effective_at=t, reserved_cost=r)
        for i, t, r in ((1, 0.5, 2), (2, 1.5, 0))
    ]
    kwargs = dict(dropped_records=0, write_errors=0)
    assert reserved_cost_at(initial, frames, final, 1, **kwargs) == 2
    with pytest.raises(ValueError, match="gap"):
        reserved_cost_at(initial, frames[1:], final, 1, **kwargs)
    with pytest.raises(ValueError, match="trace_loss"):
        reserved_cost_at(initial, frames, final, 1, **{**kwargs, "dropped_records": 1})
    with pytest.raises(ValueError, match="clock_domain"):
        reserved_cost_at(initial, frames, replace(final, clock_domain="other", sampled_at="bad"),
                         1, **kwargs)


def test_forecast_scoring_weights_episodes_and_seeds_not_decision_count():
    def row(seed, episode, decision, actual):
        return dict(seed_block=seed, episode_id=episode, decision_id=decision,
                    split="train", cell="cell-1", target="reserved_cost_1s", truth_complete=True,
                    predicted_reserved=0, actual_reserved=actual,
                    origin_at=0, target_at=1, prediction_issued_at=0.1)
    rows = [row(1, "a", i, 32) for i in range(10)]
    rows += [row(1, "b", 0, 0), row(2, "a", 0, 0)]
    expected = [(r["seed_block"], r["episode_id"], r["decision_id"]) for r in rows]
    result = primary_forecast_mae(rows, expected_decisions=expected)
    assert result["normalized_mae"] == 0.25
    assert result["independent_seed_blocks"] == 2
    with pytest.raises(ValueError, match="seed_leakage"):
        primary_forecast_mae(rows + [{**row(1, "c", 0, 0), "split": "test"}],
                             expected_decisions=expected)
    with pytest.raises(ValueError, match="duplicate"):
        primary_forecast_mae(rows + [rows[0]], expected_decisions=expected)
    with pytest.raises(ValueError, match="unqualified"):
        primary_forecast_mae([{**rows[0], "truth_complete": False}], expected_decisions=expected)
    with pytest.raises(ValueError, match="cohort"):
        primary_forecast_mae(rows[:-1], expected_decisions=expected)
    with pytest.raises(ValueError, match="horizon"):
        primary_forecast_mae([{**rows[0], "target_at": 3}], expected_decisions=expected)


def test_right_censored_timely_content_is_unknown_not_a_success_or_silent_negative():
    row = dict(clock_domain="clock", scheduled_at=1, sent_at=1, first_content_at=1.1,
               terminal_at=2, status="timeout")
    result = brier_bounds(0.8, row, domain="clock", predicted_at=1)
    assert result["label"] is None
    assert result["lower"] == pytest.approx(0.04)
    assert result["upper"] == pytest.approx(0.64)
    assert brier_bounds(0.8, {**row, "first_content_at": 1.7},
                        domain="clock", predicted_at=1)["label"] == 0
    with pytest.raises(ValueError, match="noncausal"):
        brier_bounds(0.8, row, domain="clock", predicted_at=1.1)
    with pytest.raises(ValueError, match="clock_domain"):
        brier_bounds(0.8, {**row, "clock_domain": "foreign", "sent_at": "bad"},
                     domain="clock", predicted_at=1)


def test_brier_cannot_discard_rejections_or_missing_offered_requests():
    rejected = dict(request_id="a", clock_domain="clock", scheduled_at=1, sent_at=1,
                    first_content_at=None, terminal_at=2, status="rejected",
                    http_status=503, admission_rejection=True)
    prediction = dict(request_id="a", clock_domain="clock", predicted_at=1, probability=0.8)
    assert cohort_brier([prediction], [rejected], ["a"], domain="clock")["lower"] == pytest.approx(0.64)
    with pytest.raises(ValueError, match="cohort"):
        cohort_brier([prediction], [], ["a"], domain="clock")


def test_baseline_freeze_uses_only_complete_grouped_training_folds():
    scores = [
        dict(candidate=candidate, validation_seed_block=seed, fit_seed_blocks=[3 - seed],
             split="train", normalized_mae=error)
        for candidate, error in (("persistence", 0.3), ("queue", 0.2)) for seed in (1, 2)
    ]
    assert freeze_baseline(scores, train_seeds=[1, 2])["selected"] == "queue"
    with pytest.raises(ValueError, match="leakage"):
        freeze_baseline([{**scores[0], "fit_seed_blocks": [1, 2]}], train_seeds=[1, 2])
    with pytest.raises(ValueError, match="leakage"):
        freeze_baseline([{**scores[0], "split": "test"}], train_seeds=[1, 2])
    with pytest.raises(ValueError, match="incomplete"):
        freeze_baseline(scores[:-1], train_seeds=[1, 2])
