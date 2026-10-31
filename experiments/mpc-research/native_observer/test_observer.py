import importlib.metadata
from types import SimpleNamespace

import pytest

import pr33_native_observer as observer


def core(requests=()):
    return SimpleNamespace(
        scheduler=SimpleNamespace(running=list(requests), waiting=[], skipped_waiting=[],
                                  num_waiting_for_streaming_input=0,
                                  kv_cache_manager=SimpleNamespace(usage=0.0)),
        batch_queue=None,
    )


def test_idle_state_is_fresh_and_run_identity_is_stable(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")
    monkeypatch.setattr(observer, "clock_domain", lambda: "test-clock")
    c = core()
    first, second = observer.observe(c), observer.observe(c)
    assert first["run_id"] == second["run_id"]
    assert first["sampled_at"] <= second["sampled_at"]
    assert first["running"] == first["waiting"] == first["pending_batches"] == 0


def test_active_work_and_pending_batches_are_not_reported_as_idle(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")
    request = SimpleNamespace(request_id="r", num_prompt_tokens=128,
                              num_computed_tokens=130, num_output_tokens=2, max_tokens=256)
    c = core((request,))
    c.batch_queue = [object()]
    c.scheduler.skipped_waiting = [object()]
    result = observer.observe(c)
    assert result["running"] == result["waiting"] == result["pending_batches"] == 1
    assert result["requests"][0]["output_tokens"] == 2


def test_invalid_kv_measurement_fails_closed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")
    c = core()
    c.scheduler.kv_cache_manager.usage = float("nan")
    with pytest.raises(ValueError, match="kv_usage"):
        observer.observe(c)


@pytest.mark.parametrize("sampling, expected", [
    (None, None),
    (SimpleNamespace(ignore_eos=True), True),
    (SimpleNamespace(ignore_eos=False), False),
    (SimpleNamespace(ignore_eos=1), None),
])
def test_running_request_sampling_intent_is_observed_without_inventing_missing_values(
        monkeypatch, sampling, expected):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")
    request = SimpleNamespace(request_id="r", num_prompt_tokens=128,
        num_computed_tokens=130, num_output_tokens=2, max_tokens=256,
        sampling_params=sampling)
    result = observer.observe(core((request,)))
    assert result["requests"][0]["ignore_eos"] is expected
    assert request.sampling_params is sampling


@pytest.mark.parametrize("extra, expected", [
    (None, None),
    ({}, None),
    ({"pr33_cohort_id": "a" * 32}, "a" * 32),
    ({"pr33_cohort_id": "not-an-opaque-cohort-id"}, None),
    ({"pr33_cohort_id": True}, None),
])
def test_cohort_membership_is_read_from_sampling_metadata_not_native_id_suffix(
        monkeypatch, extra, expected):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")
    request = SimpleNamespace(request_id="unrelated-native-id", num_prompt_tokens=128,
        num_computed_tokens=130, num_output_tokens=2, max_tokens=512,
        sampling_params=SimpleNamespace(extra_args=extra))
    result = observer.observe(core((request,)))
    assert result["requests"][0]["cohort_id"] == expected
    assert request.sampling_params.extra_args is extra


def test_gateway_lineage_is_an_explicit_sampling_tag_not_an_opaque_native_id(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")
    request = SimpleNamespace(request_id="opaque-native-identity", num_prompt_tokens=128,
        num_computed_tokens=130, num_output_tokens=2, max_tokens=512,
        sampling_params=SimpleNamespace(
            extra_args={"pr33_gateway_request_id": "d" * 32, "pr33_cohort_id": "e" * 32}))
    result = observer.observe(core((request,)))["requests"][0]
    assert result["request_id"] == "opaque-native-identity"
    assert result["gateway_request_id"] == "d" * 32
    assert result["cohort_id"] == "e" * 32


def test_disabled_registration_does_not_require_or_modify_native_runtime(monkeypatch):
    monkeypatch.delenv("PR33_NATIVE_OBSERVER_ENABLED", raising=False)
    assert observer.register() is None


def test_disabled_endpoint_is_inert_when_vllm_discovers_installed_plugins(monkeypatch):
    import asyncio
    from fastapi import FastAPI
    monkeypatch.delenv("PR33_NATIVE_OBSERVER_ENABLED", raising=False)
    endpoint, app = observer.Endpoint(), FastAPI()
    original = len(app.routes)
    endpoint.attach_router(app)
    asyncio.run(endpoint.init_state(None, None, None))
    assert len(app.routes) == original
    assert not hasattr(endpoint, "engine")


def test_endpoint_auth_and_frontend_work_prevent_false_drain(monkeypatch):
    import asyncio
    from fastapi import FastAPI
    import httpx
    monkeypatch.setenv("PR33_NATIVE_OBSERVER_ENABLED", "1")
    monkeypatch.setenv("PR33_NATIVE_OBSERVER_KEY", "x" * 32)
    frontend = [0]
    async def rpc(name):
        assert name == "pr33_native_observe"
        return dict(running=0, waiting=0, pending_batches=0)
    engine = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=rpc),
                             get_num_unfinished_requests=lambda: frontend[0])
    endpoint, app = observer.Endpoint(), FastAPI()
    endpoint.attach_router(app)
    async def run():
        await endpoint.init_state(engine, None, None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            assert (await client.get("/pr33-native/state")).status_code == 401
            headers = {"x-observer-key": "x" * 32}
            assert (await client.get("/pr33-native/state", headers=headers)).json()["orphaned"] == 0
            frontend[0] = 1
            assert (await client.get("/pr33-native/state", headers=headers)).json()["orphaned"] is None
    asyncio.run(run())
