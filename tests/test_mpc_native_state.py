from dataclasses import replace

import pytest

from mpc_lab.contracts import CostSpec, Identity, RequestMark, Snapshot
from mpc_lab import native_state
from mpc_lab.runtime import snapshot_to_dict


@pytest.fixture
def frame_and_binding():
    identity = Identity("unfitted", "unqualified", "support", CostSpec().fingerprint, "core")
    binding = native_state.NativeBinding("clock", "gateway", "engine", "observer", identity)
    snapshot = Snapshot("gateway", "clock", 10, 16, 0, (
        RequestMark("gateway-id", 2, 100, 128, 9, 9.1),
    ), identity)
    frame = dict(
        schema="pr33-native-read-only-v1", clock_domain="clock", run_id="engine",
        vllm_version="0.30.0",
        observer_sha256="observer", gateway_snapshot=snapshot_to_dict(snapshot),
        sampled_at=10.01, frontend_sampled_at=10.02, received_at=10.03,
        gateway_received_at=10.005, complete=True, running=1, waiting=2,
        pending_batches=1, frontend_unfinished=4, kv_cache_usage=0.4,
        requests=[dict(request_id="engine-id", prompt_tokens=100, computed_tokens=80,
                       output_tokens=0, maximum_output_tokens=128)],
    )
    return frame, binding


def test_preserves_original_cost_unknown_waiting_and_separate_namespaces(frame_and_binding):
    frame, binding = frame_and_binding
    result = native_state.estimate_native_state(frame, binding, now=10.04)
    assert result.reason is None
    state = result.state
    assert state.gateway.reserved == 2
    assert state.waiting == 2 and state.visible_prefill_remaining == 20
    assert state.visible_output_headroom == 128
    assert state.progress[0].engine_request_id != state.gateway.requests[0].request_id
    frame["requests"][0]["computed_tokens"] = 101
    later = native_state.estimate_native_state(frame, binding, now=10.04).state
    assert later.visible_prefill_remaining == 0 and later.gateway.reserved == 2
    assert state.progress[0].computed_tokens == 80


def test_wrong_clock_refuses_before_parser_or_time_arithmetic(frame_and_binding, monkeypatch):
    frame, binding = frame_and_binding
    frame["clock_domain"] = "other-boot"
    frame["sampled_at"] = "not-a-timestamp"
    monkeypatch.setattr(native_state, "snapshot_from_dict",
                        lambda _: pytest.fail("cross-boot snapshot must not be parsed"))
    result = native_state.estimate_native_state(frame, binding, now=10.04)
    assert result.state is None and result.reason == "clock_domain_mismatch"


@pytest.mark.parametrize("change,reason", [
    ({"run_id": "other-run"}, "native_source_mismatch"),
    ({"vllm_version": "0.31.0"}, "native_source_mismatch"),
    ({"complete": False}, "incomplete_native_state"),
    ({"sampled_at": 9.0}, "native_state_timing"),
    ({"received_at": 10.1}, "native_state_timing"),
    ({"frontend_sampled_at": 10.08}, "native_state_timing"),
    ({"running": 2}, "native_running_inventory"),
    ({"waiting": True}, "native_state_values"),
    ({"waiting": 10 ** 400}, "native_state_values"),
    ({"kv_cache_usage": float("nan")}, "native_state_values"),
])
def test_refuses_unusable_native_observations(frame_and_binding, change, reason):
    frame, binding = frame_and_binding
    frame.update(change)
    result = native_state.estimate_native_state(frame, binding, now=10.04)
    assert result.state is None and result.reason == reason


def test_identity_and_bad_progress_never_produce_state(frame_and_binding):
    frame, binding = frame_and_binding
    wrong = replace(binding, identity=replace(binding.identity, cost_hash="other-cost"))
    assert native_state.estimate_native_state(frame, wrong, now=10.04).reason == \
        "native_identity_mismatch"
    frame["requests"][0]["computed_tokens"] = True
    assert native_state.estimate_native_state(frame, binding, now=10.04).state is None
    frame["requests"][0]["computed_tokens"] = 10 ** 400
    assert native_state.estimate_native_state(frame, binding, now=10.04).state is None
