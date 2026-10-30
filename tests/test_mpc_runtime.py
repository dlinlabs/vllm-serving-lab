import asyncio
from dataclasses import asdict, replace
import json
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from mpc_lab.contracts import (
    SCHEMA, GUARDED_CAPABILITY, CostSpec, Decision, Identity, ModelParameters,
    RequestMark, Snapshot, digest,
)
from mpc_lab import runtime


@pytest.fixture
def snapshot():
    return Snapshot(
        "run", "boot-clock", 10.0, 16.0, 3,
        (
            RequestMark("awaiting", 2.0, 100, 128, 8.0, 8.2),
            RequestMark("streaming", 3.0, 500, 256, 7.0, 7.5, 9.0),
        ),
        Identity("model", "calibration", "support", CostSpec().fingerprint, "implementation"),
        sequence=7,
    )


@pytest.fixture
def synthetic_snapshot():
    return Snapshot(
        "synthetic-run", runtime.local_clock_domain(), time.monotonic(),
        16.0, 0, (), ModelParameters().identity, sequence=4,
    )


def document(snapshot):
    return {
        **runtime.snapshot_to_dict(snapshot),
        "capabilities": [GUARDED_CAPABILITY],
    }


def test_snapshot_roundtrip_and_reserved_over_budget(snapshot):
    restored = runtime.snapshot_from_dict(json.loads(json.dumps(document(snapshot))))
    assert restored == snapshot and restored.reserved == 5 and restored.prompt_backlog == 100
    reduced = replace(snapshot, budget=1)
    assert runtime.snapshot_from_dict(runtime.snapshot_to_dict(reduced)) == reduced


@pytest.mark.parametrize("key,value", [
    ("schema", "other"), ("clock_domain", ""), ("run_id", True),
    ("sampled_at", float("nan")), ("budget", True), ("budget", 10 ** 400),
    ("budget_version", True), ("sequence", -1), ("complete", 1),
    ("identity", {"cost_hash": "only"}), ("requests", {}),
])
def test_malformed_snapshot(snapshot, key, value):
    raw = document(snapshot)
    raw[key] = value
    with pytest.raises(ValueError):
        runtime.snapshot_from_dict(raw)


@pytest.mark.parametrize("key,value", [
    ("cost", -1), ("cost", float("inf")), ("prompt_tokens", True),
    ("max_tokens", 0), ("received_at", 9.0), ("admitted_at", 11.0),
    ("first_at", 7.0), ("scheduled_at", 9.0),
])
def test_malformed_marks(snapshot, key, value):
    raw = document(snapshot)
    raw["requests"][0][key] = value
    with pytest.raises(ValueError):
        runtime.snapshot_from_dict(raw)


def test_duplicate_marks_and_wrong_cost_identity(snapshot):
    raw = document(snapshot)
    raw["requests"].append(raw["requests"][0])
    with pytest.raises(ValueError):
        runtime.snapshot_from_dict(raw)
    raw = document(snapshot)
    raw["cost_spec"] = {"intercept": 1.0}
    with pytest.raises(ValueError, match="cost_identity"):
        runtime.snapshot_from_dict(raw)


@pytest.mark.parametrize("legacy", [
    {"budget": 16}, {"schema": SCHEMA, "capabilities": []},
    {"schema": SCHEMA, "capabilities": [GUARDED_CAPABILITY + "-almost"]},
    {"schema": SCHEMA, "capabilities": GUARDED_CAPABILITY},
])
def test_legacy_is_not_a_guarded_capability(legacy):
    with pytest.raises(ValueError, match="capability"):
        runtime.require_guarded_capability(legacy)


def test_shadow_rejects_legacy_without_any_post():
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={"budget": 32})
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test",
        ) as client:
            with pytest.raises(ValueError, match="capability"):
                await runtime.shadow_once(client)
        assert calls == ["GET"]
    asyncio.run(scenario())


def test_shadow_does_not_post_and_keeps_injectable_planner(synthetic_snapshot, monkeypatch):
    snapshot = synthetic_snapshot
    calls = []

    def planner(state):
        assert state == snapshot
        return Decision(32, None, "test-forecast", False, 2, 3, 0.01, 8, 16, state.identity)

    monkeypatch.setitem(sys.modules, "mpc_lab.controller", SimpleNamespace(
        validate_snapshot=lambda state, **kwargs: None,
    ))

    def handler(request):
        calls.append((request.method, request.url.path))
        assert request.method == "GET" and request.url.path == "/admin/mpc/state"
        return httpx.Response(200, json=document(snapshot))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test",
        ) as client:
            result = await runtime.shadow_once(client, planner=planner)
        assert result["mode"] == "shadow" and result["applied"] is False
        assert result["decision"]["budget"] == 32
        assert result["decision"]["reason"] == "test-forecast"
        assert result["budget_version"] == 0 and result["sequence"] == 4
        assert result["reproducibility"]["planner"] == "injected"
        assert result["reproducibility"]["scenario_fingerprint"] is None
        assert calls == [("GET", "/admin/mpc/state")]

    asyncio.run(scenario())


def test_shadow_incomplete_state_never_calls_planner(synthetic_snapshot):
    snapshot = synthetic_snapshot
    async def scenario():
        def forbidden(state):
            pytest.fail("incomplete ledger reached planner")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(
                200, json=document(replace(snapshot, complete=False)),
            )), base_url="http://test",
        ) as client:
            result = await runtime.shadow_once(client, planner=forbidden)
        assert result["decision"]["reason"] == "incomplete_snapshot"
        assert result["applied"] is False
    asyncio.run(scenario())


def test_missing_controller_is_explicit_shadow_fallback(synthetic_snapshot, monkeypatch):
    snapshot = synthetic_snapshot
    monkeypatch.setitem(sys.modules, "mpc_lab.controller", None)
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=document(snapshot))),
            base_url="http://test",
        ) as client:
            result = await runtime.shadow_once(client)
        assert result["decision"]["reason"] == "controller_unavailable"
        assert result["applied"] is False
    asyncio.run(scenario())


def test_cli_is_get_only(synthetic_snapshot, monkeypatch, capsys):
    snapshot = synthetic_snapshot
    real_client = httpx.AsyncClient
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        assert request.headers["x-admin-api-key"] == "test"
        return httpx.Response(200, json=document(snapshot))
    monkeypatch.setenv("ADMIN_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "mpc_lab.controller", None)
    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handler),
    ))
    runtime.main(["--url", "http://test"])
    assert calls == [("GET", "/admin/mpc/state")]
    assert json.loads(capsys.readouterr().out)["applied"] is False


@pytest.mark.parametrize("fault,reason", [
    (None, "selected"),
    ("stale", "stale_snapshot"),
    ("clock", "clock_domain_mismatch"),
    ("support", "identity_mismatch"),
    ("cost", "identity_mismatch"),
    ("calibration", "identity_mismatch"),
    ("source", "identity_mismatch"),
])
def test_real_integrated_cli_validates_before_generation(
    synthetic_snapshot, monkeypatch, capsys, fault, reason,
):
    pytest.importorskip("mpc_lab.controller", reason="canonical controller not yet integrated")
    model = pytest.importorskip("mpc_lab.model", reason="canonical model not yet integrated")
    snapshot = synthetic_snapshot
    if fault == "stale":
        snapshot = replace(snapshot, sampled_at=snapshot.sampled_at - 100)
    elif fault == "clock":
        snapshot = replace(snapshot, clock_domain="other-host", sampled_at=1e300)
    elif fault is not None:
        field = {
            "support": "support_hash", "cost": "cost_hash",
            "calibration": "calibration_version", "source": "implementation_hash",
        }[fault]
        snapshot = replace(snapshot, identity=replace(snapshot.identity, **{field: "different"}))
    generated, predicted, methods = [], [], []
    generate, rollout = model.generate_scenarios, model.rollout
    def tracked_generate(*args, **kwargs):
        result = generate(*args, **kwargs)
        generated.append(result)
        return result
    def tracked_rollout(*args, **kwargs):
        predicted.append(True)
        return rollout(*args, **kwargs)
    monkeypatch.setattr(model, "generate_scenarios", tracked_generate)
    monkeypatch.setattr(model, "rollout", tracked_rollout)
    real_client = httpx.AsyncClient
    def handler(request):
        methods.append((request.method, request.url.path))
        assert request.method == "GET" and request.url.path == "/admin/mpc/state"
        return httpx.Response(200, json=document(snapshot))
    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handler),
    ))
    runtime.main(["--url", "http://test", "--seed", "17", "--max-wall-s", "2"])
    record = json.loads(capsys.readouterr().out)
    assert methods == [("GET", "/admin/mpc/state")] and record["applied"] is False
    assert record["decision"]["reason"] == reason
    metadata = record["reproducibility"]
    assert metadata["seed"] == 17
    assert metadata["snapshot_fingerprint"] == digest(runtime.snapshot_to_dict(snapshot))
    assert metadata["parameters_fingerprint"] == digest(metadata["parameters"])
    assert metadata["config_fingerprint"] == digest(metadata["config"])
    assert metadata["prior_fingerprint"] == digest(metadata["prior"])
    assert metadata["expected_identity"] == asdict(ModelParameters().identity)
    assert metadata["observed_identity"] == asdict(snapshot.identity)
    assert metadata["local_clock_domain"] == runtime.local_clock_domain()
    if fault is None:
        assert not record["decision"]["fallback"]
        assert record["decision"]["evaluated_plans"] == 8
        assert generated and predicted and metadata["scenario_count"] == 16
        assert metadata["scenario_fingerprint"] == digest(tuple(asdict(s) for s in generated[0]))
        replay = generate(
            snapshot, **metadata["prior"], seed=metadata["seed"],
            duration_s=metadata["scenario_duration_s"],
        )
        assert replay == generated[0]
    else:
        assert record["decision"]["fallback"]
        assert generated == [] and predicted == []
        assert metadata["scenario_fingerprint"] is None and metadata["scenario_count"] == 0


def test_clock_lookup_failure_is_observable(synthetic_snapshot, monkeypatch):
    def unavailable():
        raise OSError("no local clock identity")
    monkeypatch.setattr(runtime, "local_clock_domain", unavailable)
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(
                200, json=document(synthetic_snapshot),
            )), base_url="http://test",
        ) as client:
            result = await runtime.shadow_once(client)
        assert result["decision"]["reason"] == "clock_domain_unavailable"
        assert result["reproducibility"]["local_clock_domain"] is None
    asyncio.run(scenario())
