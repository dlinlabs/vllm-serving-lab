import importlib.util
from pathlib import Path

import pytest

from mpc_lab.contracts import ModelParameters, PlanningConfig, RolloutResult, Snapshot


path = Path(__file__).resolve().parents[1] / "experiments/mpc-research/native_id_preparation.py"
spec = importlib.util.spec_from_file_location("native_id_preparation", path)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_v2_protocol_uses_independent_seed_blocks_and_no_heldout_arrival_generation():
    from collections import Counter, defaultdict
    source = path.with_name("native_identification_protocol.py")
    module_spec = importlib.util.spec_from_file_location("native_identification_protocol", source)
    protocol_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(protocol_module)
    protocol = protocol_module.build_protocol()
    assert protocol == protocol_module.build_protocol()
    pairs = defaultdict(list)
    for episode in protocol["episodes"]:
        pairs[episode["seed_block"]].append(episode)
    assert len(pairs) == 24
    assert Counter(v[0]["split"] for v in pairs.values()) == {
        "train": 12, "calibration": 4, "validation": 4, "test": 4,
    }
    for episodes in pairs.values():
        a, b = episodes
        assert a["arrival_tape_key"] == b["arrival_tape_key"]
        assert a["prompt_tape_key"] == b["prompt_tape_key"]
        assert a["split"] == b["split"] and a["arm"] != b["arm"]
        assert all(x["at_s"] == y["at_s"] and x["budget"] + y["budget"] == 48
                   for x, y in zip(a["schedule"], b["schedule"]))
        assert "arrivals" not in a and "arrivals" not in b


def test_fresh_source_train_start_preserves_failed_history_without_claiming_completion():
    source = path.parent/"native-v3"/"collect_identification.py"
    module_spec = importlib.util.spec_from_file_location("fresh_native_identification", source)
    collector = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(collector)
    cost = ModelParameters().cost_spec.__dict__
    legacy = collector.protocol_for("a" * 40, cost)
    fresh = collector.protocol_for("b" * 40, cost, first_train_block=14, gateway_ledger=True)
    assert fresh["completed_seed_blocks"] == 0
    assert fresh["first_train_block"] == 14
    assert fresh["episodes"] == collector.protocol_for("b" * 40, cost)["episodes"][140:]
    assert fresh["episodes"][0]["seed_block"] == 181015
    assert {e["seed_block"] for e in fresh["episodes"] if e["split"] == "train"} == {
        181015, 181016,
    }
    assert fresh["planned_offered_window_s"] == 31200
    assert fresh["gateway_ledger_capture"] is True and not fresh["final_test_used"]
    assert "first_train_block" not in legacy and "gateway_ledger_capture" not in legacy
    assert legacy["planned_offered_window_s"] == 48000
    for invalid in ({"first_train_block": 16}, {"first_train_block": True},
                    {"first_train_block": 14, "completed_blocks": 13},
                    {"first_train_block": 14, "smoke": True},
                    {"gateway_ledger": 1}):
        with pytest.raises(ValueError):
            collector.protocol_for("b" * 40, cost, **invalid)


def row(rid="r", **changes):
    return dict(
        request_id=rid, response_request_id=rid, clock_domain="boot",
        scheduled_at=1.0, sent_at=1.01, first_content_at=1.2,
        last_content_at=1.8, terminal_at=1.9, http_status=200,
        status="complete", stream_done=True, stream_error=False,
        finish_reason="stop", **changes,
    )


def test_http_late_failure_and_rejection_stay_in_offered_denominator():
    good = row()
    failed = {**row("f"), "status": "failed", "stream_done": False}
    rejected = {**row("x"), "status": "rejected", "http_status": 503,
                "first_content_at": None, "last_content_at": None}
    result = prep.audit_http([good, failed, rejected], ["r", "f", "x"], "boot")
    assert result["valid"] and result["good"] == 1
    assert result["failed"] == result["rejected"] == 1
    assert result["all_offered_nonattainment"] == pytest.approx(2 / 3)


def test_clock_guard_precedes_foreign_time_comparisons():
    result = prep.audit_http([{**row(), "clock_domain": "other", "sent_at": {}}],
                             ["r"], "boot")
    assert result["errors"] == [{"request_id": "r", "reason": "clock_domain"}]
    assert not result["valid"]


@pytest.mark.parametrize("changes", [
    {"stream_done": False}, {"response_request_id": "other"},
    {"last_content_at": 1.1}, {"first_content_at": True},
    {"terminal_at": float("nan")}, {"stream_error": True},
    {"finish_reason": "error"}, {"terminal_at": 10**400},
])
def test_http_invalid_completion_cannot_earn_goodput(changes):
    result = prep.audit_http([{**row(), **changes}], ["r"], "boot")
    assert not result["valid"] and result["good"] == 0


def test_http_missing_and_duplicate_offers_fail_closed():
    result = prep.audit_http([row(), row()], ["r", "missing"], "boot")
    assert {e["reason"] for e in result["errors"]} == {
        "cohort_identity", "missing_offered_request",
    }
    assert not result["valid"]


def test_http_right_censoring_is_not_a_fabricated_completion():
    pending = {**row(), "status": "unresolved", "stream_done": False,
               "terminal_at": None, "observed_until_at": 2.0}
    result = prep.audit_http([pending], ["r"], "boot")
    assert result["valid"] and result["unresolved"] == 1
    assert result["good"] == 0 and result["all_offered_nonattainment"] == 1


def test_higher_open_loop_value_need_not_change_first_action(monkeypatch):
    params = ModelParameters()
    snapshot = Snapshot("test", "cpu", 0.0, 16.0, 0, (), params.identity)
    def predictor(state, plan, scenario, parameters, config):
        score = float(plan.actions[1] == 32)
        return RolloutResult((), (), score, score, 1)

    monkeypatch.setattr(prep.model, "rollout", predictor)
    result = prep.exact_plan_comparison(snapshot, (object(),), params, PlanningConfig())
    assert result["open_loop_gain"] == 1
    assert not result["first_action_differs"]
    assert result["one_move_equivalent"]["actions"][1:] == [16.0, 16.0]


def test_identification_pairs_share_tape_but_have_complementary_legal_actions():
    manifest = prep.identification_manifest()
    pairs = {}
    for episode in manifest["episodes"]:
        assert episode["prompt_tokens"] + episode["max_tokens"] <= manifest["context_limit"]
        pairs.setdefault(episode["offered_tape_key"], []).append(episode)
    assert len(pairs) == 24
    for pair in pairs.values():
        a, b = pair
        assert a["split"] == b["split"]
        assert a["prompt_tokens"] == b["prompt_tokens"]
        assert [x["at_s"] for x in a["schedule"]] == [x["at_s"] for x in b["schedule"]]
        assert all({x["budget"], y["budget"]} == {16, 32}
                   for x, y in zip(a["schedule"], b["schedule"]))
    assert {e["prompt_tokens"] for e in manifest["episodes"] if e["split"] == "train"} == {128, 4096}
    assert 7168 in {e["prompt_tokens"] for e in manifest["episodes"] if e["split"] == "test"}
