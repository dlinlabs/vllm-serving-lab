import asyncio
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from mpc_lab.contracts import (
    SCHEMA, GUARDED_CAPABILITY, CostSpec, Decision, Identity, RequestMark, Snapshot,
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


def test_shadow_does_not_post_and_lazily_calls_controller(snapshot, monkeypatch):
    calls = []

    def decide(state):
        assert state == snapshot
        return Decision(32, None, "test-forecast", False, 2, 3, 0.01, 8, 16, state.identity)

    monkeypatch.setitem(sys.modules, "mpc_lab.controller", SimpleNamespace(decide=decide))

    def handler(request):
        calls.append((request.method, request.url.path))
        assert request.method == "GET" and request.url.path == "/admin/mpc/state"
        return httpx.Response(200, json=document(snapshot))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test",
        ) as client:
            result = await runtime.shadow_once(client)
        assert result["mode"] == "shadow" and result["applied"] is False
        assert result["decision"]["budget"] == 32
        assert result["decision"]["reason"] == "test-forecast"
        assert result["budget_version"] == 3 and result["sequence"] == 7
        assert calls == [("GET", "/admin/mpc/state")]

    asyncio.run(scenario())


def test_shadow_incomplete_state_never_calls_planner(snapshot):
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


def test_missing_controller_is_explicit_shadow_fallback(snapshot, monkeypatch):
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


def test_cli_is_get_only(snapshot, monkeypatch, capsys):
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
