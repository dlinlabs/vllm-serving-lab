from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def bootstrap(monkeypatch):
    monkeypatch.syspath_prepend(str(
        Path(__file__).resolve().parents[1] / "experiments/mpc-research/native-r23"))
    import capture_owner_closure as api

    snapshot = {
        "run_id": "gateway", "clock_domain": "clock", "sequence": 0,
        "budget_version": 0, "budget": 32, "requests": [],
    }
    command = {
        "run_id": "gateway", "clock_domain": "clock",
        "operation_id": "operation", "expected_version": 0,
    }
    ack = {
        **command, "status": "applied", "budget_version": 1,
        "budget": 32, "previous_budget": 32, "applied_at": 2,
    }
    gateway = {"readiness": {
        "snapshot": snapshot,
        "same_budget_ack": {"status": "applied", "command": command, "acknowledgment": ack},
    }}
    warm_id = "chatcmpl-warm-12345678"
    phase = {
        **{key: "c" * 64 for key in api.SOURCE_FIELDS},
        "run_id": "native", "clock_domain": "clock", "source_at": 4,
        "complete": True, "lifecycle": {},
        "capture": {"trial_id": "trial", "status": "running", "started_at": 3},
        "publications": {warm_id: {"put_started_at": 0.8, "put_finished_at": 0.9}},
    }
    frame = {
        "run_id": "native", "clock_domain": "clock", "sampled_at": 1.5, "complete": True,
        "running": 0, "waiting": 0, "pending_batches": 0, "frontend_unfinished": 0,
    }
    native = {"readiness": {
        "frames": [dict(frame), dict(frame)],
        "warmup_response": {
            "id": "chatcmpl-warm", "choices": [{"finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        },
    }}
    cut = {
        "schema": "r24-clocked-owner-cut-v1", "cut_at": 4,
        "common_clock_proof": {
            "clock_process_binding_only": True, "clock_domain": "clock",
            "native_run_id": "native", "gateway_run_id": "gateway",
        },
        "native_observation": {
            "complete": True, "native_phase": phase,
            "scheduler_state": {"requests": {}}, "queue_oldest_first": [],
        },
        "owner_prefix": {
            "run_id": "gateway", "clock_domain": "clock", "cut_at": 4,
            "gateway_prefix_closed": True, "owners": [], "closed_through_sequence": 1,
            "snapshot": {**snapshot, "sequence": 1, "budget_version": 1},
        },
        "baseline_native_ids": [warm_id],
    }
    return api, cut, native, gateway


def test_known_completed_warm_marker_is_not_classified_as_pending(bootstrap):
    api, cut, native, gateway = bootstrap
    result = api.establish_initial_capture_closure(
        cut, native_startup=native, gateway_startup=gateway)
    assert result.capture_id == "trial"
    assert set(result.baseline_publications) == {"chatcmpl-warm-12345678"}
    cut["native_observation"]["native_phase"]["publications"].clear()
    assert result.baseline_publications


@pytest.mark.parametrize("change", ["admission", "finished_before_snapshot", "inflight", "wrong_warm", "undrained", "epoch"])
def test_empty_snapshot_alone_does_not_close_initial_native_history(bootstrap, change):
    api, original_cut, original_native, original_gateway = bootstrap
    cut, native, gateway = deepcopy((original_cut, original_native, original_gateway))
    if change == "admission":
        cut["owner_prefix"]["snapshot"]["sequence"] = 2
    elif change == "finished_before_snapshot":
        cut["owner_prefix"]["snapshot"]["sequence"] = 3
    elif change == "inflight":
        cut["native_observation"]["queue_oldest_first"] = [{"work": []}]
    elif change == "wrong_warm":
        native["readiness"]["warmup_response"]["id"] = "foreign"
    elif change == "undrained":
        native["readiness"]["frames"][0]["pending_batches"] = 1
    else:
        cut["common_clock_proof"]["native_run_id"] = "other"
    with pytest.raises(ValueError, match="capture_closure"):
        api.establish_initial_capture_closure(
            cut, native_startup=native, gateway_startup=gateway)


@pytest.mark.parametrize("change", ["live_baseline", "changed_marker", "capture", "epoch", "source"])
def test_later_baseline_exemption_is_bound_to_established_closure(bootstrap, change):
    api, cut, native, gateway = bootstrap
    closure = api.establish_initial_capture_closure(
        cut, native_startup=native, gateway_startup=gateway)
    phase = deepcopy(cut["native_observation"]["native_phase"])
    baseline = next(iter(closure.baseline_publications))
    if change == "live_baseline":
        phase["lifecycle"][baseline] = {"finished": None}
    elif change == "changed_marker":
        phase["publications"][baseline]["put_finished_at"] += 0.1
    elif change == "capture":
        phase["capture"]["trial_id"] = "other"
    elif change == "epoch":
        phase["run_id"] = "other"
    else:
        phase["source_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="current_baseline_binding"):
        api.bound_baseline_ids(
            closure, phase, native, expected_gateway_run_id=closure.gateway_run_id)


@pytest.fixture
def phase_cut(bootstrap):
    api, cut, native, gateway = bootstrap
    closure = api.establish_initial_capture_closure(
        cut, native_startup=native, gateway_startup=gateway)
    from live_owner_prefix import PublicRequest
    from source_owner_phases import classify_source_owners

    rid = "a" * 32
    nid = "chatcmpl-" + rid + "-11111111"
    cut["cut_at"], cut["validated_at"] = 5, 5.1
    phase = cut["native_observation"]["native_phase"]
    phase["source_at"] = 5
    body = cut["native_observation"]
    body["scheduler_state"].update(running=[], waiting=[])
    body["queued_ghost_request_ids"] = []
    prefix = cut["owner_prefix"]
    prefix.update(cut_at=5, snapshot_at=5.05, native_run_id="native", budget=16,
                  budget_version=2, closed_through_sequence=3,
                  released_native_ids=[])
    prefix["owners"] = [{
        "request_id": rid, "native_id": None, "prompt_tokens": 128, "max_tokens": 128,
        "cost": 2, "admission_effective_at": 4.1, "budget_at_admission": 32,
    }]
    memory = {rid: PublicRequest(128, 128)}
    kwargs = dict(initial_closure=closure, native_startup=native,
                  public_memory=memory, now=5.2, expected_clock_domain="clock")
    return classify_source_owners, cut, kwargs, nid


@pytest.mark.parametrize("kind", [
    "pre_publication", "published", "active", "waiting", "native_finished", "issued_ghost",
])
def test_source_phase_join_keeps_pending_terminal_and_ghost_distinct(phase_cut, kind):
    classify, cut, kwargs, nid = phase_cut
    body = cut["native_observation"]
    phase = body["native_phase"]
    if kind != "pre_publication":
        cut["owner_prefix"]["owners"][0]["native_id"] = nid
        phase["publications"][nid] = {"put_started_at": 4.2, "put_finished_at": 4.25}
    if kind not in ("pre_publication", "published"):
        phase["lifecycle"][nid] = {
            "input_tokens": 128, "max_tokens": 128, "enqueue": 4.3, "first": 4.4,
            "finished": None if kind in ("active", "waiting") else 4.8,
        }
    if kind in ("active", "waiting"):
        body["scheduler_state"]["running" if kind == "active" else "waiting"] = [nid]
        body["scheduler_state"]["requests"][nid] = {}
    if kind == "issued_ghost":
        body["queue_oldest_first"] = [{"work": [{"request_id": nid}]}]
        body["queued_ghost_request_ids"] = [nid]
    result = classify(cut, **kwargs)
    assert result.owners[0].phase == kind
    assert result.owners[0].historical_admission_budget == 32
    assert result.source_at == 5 and result.available_at == 5.1
    assert result.source_budget == 16 and result.source_budget_version == 2
    assert result.capture_id == "trial"
    assert result.received_prefix_through_sequence == 3
    evidence = result.evidence()
    assert evidence["requires_source_age_propagation"]
    assert not evidence["decision_time_state_verified"]
    assert not evidence["native_mpc_permitted"]


def test_released_physical_ghost_has_no_owner_charge_but_live_release_refuses(phase_cut):
    classify, cut, kwargs, nid = phase_cut
    body = cut["native_observation"]
    body["native_phase"]["publications"][nid] = {
        "put_started_at": 4.2, "put_finished_at": 4.25}
    life = {"input_tokens": 128, "max_tokens": 128, "enqueue": 4.3,
            "first": 4.4, "finished": 4.8}
    body["native_phase"]["lifecycle"][nid] = life
    cut["owner_prefix"].update(owners=[], released_native_ids=[nid])
    body["queue_oldest_first"] = [{"work": [{"request_id": nid}]}]
    body["queued_ghost_request_ids"] = [nid]
    result = classify(cut, **kwargs)
    assert result.owners == ()
    assert result.released_ghost_ids == (nid,)
    life["finished"] = None
    body["scheduler_state"].update(running=[nid], requests={nid: {}})
    body["queued_ghost_request_ids"] = []
    with pytest.raises(ValueError, match="released_native_not_finished"):
        classify(cut, **kwargs)


@pytest.mark.parametrize("change", ["baseline", "source", "stale", "loose_age", "epoch", "gap"])
def test_source_phase_join_refuses_unbound_or_mixed_evidence(phase_cut, change):
    classify, cut, kwargs, nid = phase_cut
    if change == "baseline":
        cut["baseline_native_ids"].append(nid)
    elif change == "source":
        cut["native_observation"]["native_phase"]["source_sha256"] = "d" * 64
    elif change == "stale":
        kwargs["now"] = 6
    elif change == "loose_age":
        kwargs.update(now=1000, max_age_s=1e6)
    elif change == "epoch":
        cut["owner_prefix"]["native_run_id"] = "other"
    else:
        cut["native_observation"]["native_phase"]["publications"][nid] = {
            "put_started_at": 4.2, "put_finished_at": 4.25}
    with pytest.raises(ValueError):
        classify(cut, **kwargs)
