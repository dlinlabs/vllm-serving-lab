import asyncio
from dataclasses import asdict, replace
import json

import httpx
import pytest

from mpc_lab import actuator
from mpc_lab.actuator import ActuationPermit, GuardedActuator
from mpc_lab.contracts import (
    Decision, GUARDED_CAPABILITY, ModelParameters, Snapshot,
)
from mpc_lab.runtime import snapshot_to_dict


def fixture(monkeypatch, tmp_path, handler):
    monkeypatch.setattr(actuator, "clock_domain", lambda: "test-clock")
    now = [100.0]
    monkeypatch.setattr(actuator.time, "monotonic", lambda: now[0])
    params = ModelParameters()
    snapshot = Snapshot("run", "test-clock", 100, 16, 0, (), params.identity)
    evidence = tuple((name, "a" * 64) for name in (
        "native_state", "native_actions", "calibration", "native_http", "fallback",
    ))
    permit = ActuationPermit(params.identity, evidence, True)
    decision = Decision(32, None, "test-fixture", False, 1, 1, 0, 1, 1, params.identity)
    document = {**snapshot_to_dict(snapshot), "capabilities": [GUARDED_CAPABILITY]}
    calls = []

    def transport(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=document)
        return handler(request, now, document)

    client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(transport))
    controller = GuardedActuator(
        client, params, tmp_path / "pending.json", permit=permit, enabled=True,
    )
    return controller, client, snapshot, decision, now, document, calls


def test_unqualified_and_disabled_actuation_never_even_contacts_server(monkeypatch, tmp_path):
    def no_post(*args):
        raise AssertionError("must not contact server")
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, no_post)

    async def run():
        controller.enabled = False
        assert await controller.submit(state, decision) == {"status": "not_authorized"}
        controller.enabled = True
        controller.permit = replace(controller.permit, approved=False)
        assert await controller.submit(state, decision) == {"status": "not_authorized"}
        controller.permit = ActuationPermit(state.identity, (), True)
        assert await controller.submit(state, decision) == {"status": "not_authorized"}
        await client.aclose()
    asyncio.run(run())
    assert calls == []


def test_acknowledged_command_is_persisted_before_post_and_cleared(monkeypatch, tmp_path):
    def apply(request, now, document):
        command = json.loads(request.content)
        assert (tmp_path / "pending.json").exists()
        assert command == json.loads((tmp_path / "pending.json").read_text())
        assert request.url.path == "/admin/mpc/budget"
        return httpx.Response(200, json={
            "status": "applied", "operation_id": command["operation_id"],
            "run_id": command["run_id"], "clock_domain": command["clock_domain"],
            "budget": command["budget"], "budget_version": 1, "applied_at": now[0],
        })
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, apply)
    async def run():
        assert (await controller.submit(state, decision))["status"] == "applied"
        await client.aclose()
    asyncio.run(run())
    assert calls == ["GET", "POST"]
    assert not controller.path.exists()


@pytest.mark.parametrize("advance", [False, True])
def test_timeout_survives_restart_and_fresh_get_alone_does_not_fence(
    monkeypatch, tmp_path, advance,
):
    def timeout(*args):
        raise httpx.ReadTimeout("lost reply")
    controller, client, state, decision, now, document, calls = fixture(monkeypatch, tmp_path, timeout)
    async def run():
        assert (await controller.submit(state, decision))["status"] == "pending"
        restarted = GuardedActuator(
            client, controller.parameters, controller.path,
            permit=controller.permit, enabled=True,
        )
        assert await restarted.submit(state, decision) == {"status": "pending"}
        assert await restarted.reconcile() == {"status": "pending"}
        if advance:
            document["budget_version"] = 1
            document["budget"] = 32
        else:
            now[0] = 100.2
        result = await restarted.reconcile()
        assert result["status"] == ("fenced" if advance else "expired")
        assert not controller.path.exists()
        await client.aclose()
    asyncio.run(run())
    assert calls.count("POST") == 1


def test_wrong_boot_precedes_incomparable_time_and_no_post(monkeypatch, tmp_path):
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, None)
    async def run():
        foreign = replace(state, clock_domain="other-boot", sampled_at=1e10)
        result = await controller.submit(foreign, decision)
        assert result == {"status": "blocked", "reason": "clock_domain_mismatch"}
        await client.aclose()
    asyncio.run(run())
    assert calls == []


def test_legacy_endpoint_capability_cannot_receive_guarded_post(monkeypatch, tmp_path):
    controller, client, state, decision, _, document, calls = fixture(monkeypatch, tmp_path, None)
    document.pop("capabilities")
    async def run():
        with pytest.raises(ValueError, match="guarded_capability_missing"):
            await controller.submit(state, decision)
        await client.aclose()
    asyncio.run(run())
    assert calls == ["GET"]


@pytest.mark.parametrize("status", [200, 409, 500])
def test_malformed_ack_or_server_error_is_unknown_not_a_new_write(
    monkeypatch, tmp_path, status,
):
    def broken(*args):
        return httpx.Response(status, json={"unexpected": True})
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, broken)
    async def run():
        assert (await controller.submit(state, decision))["status"] == "pending"
        assert await controller.submit(state, decision) == {"status": "pending"}
        await client.aclose()
    asyncio.run(run())
    assert calls.count("POST") == 1


def test_corrupt_pending_state_is_not_discarded_to_allow_new_action(monkeypatch, tmp_path):
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, None)
    controller.path.write_text("invalid json")
    async def run():
        with pytest.raises(ValueError, match="pending_state_invalid"):
            await controller.submit(state, decision)
        await client.aclose()
    asyncio.run(run())
    assert calls == []
    assert controller.path.read_text() == "invalid json"


@pytest.mark.parametrize("now_value", [1.0, 100.0, 1000.0])
def test_changed_domain_cannot_expire_an_unknown_write(monkeypatch, tmp_path, now_value):
    def timeout(*args):
        raise httpx.ReadTimeout("unknown application")
    controller, client, state, decision, now, document, calls = fixture(monkeypatch, tmp_path, timeout)
    async def run():
        assert (await controller.submit(state, decision))["status"] == "pending"
        document["clock_domain"] = "other-boot"
        now[0] = now_value
        assert (await controller.reconcile())["status"] == "pending"
        assert controller.path.exists()
        await client.aclose()
    asyncio.run(run())
    assert calls.count("POST") == 1


def test_second_writer_cannot_overlap_the_same_durable_command_file(monkeypatch, tmp_path):
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, None)
    other = GuardedActuator(
        client, controller.parameters, controller.path,
        permit=controller.permit, enabled=True,
    )
    async def run():
        with controller._locked():
            with pytest.raises(BlockingIOError):
                await other.submit(state, decision)
        await client.aclose()
    asyncio.run(run())
    assert calls == []


def test_unsupported_action_never_reaches_transport(monkeypatch, tmp_path):
    controller, client, state, decision, _, _, calls = fixture(monkeypatch, tmp_path, None)
    async def run():
        result = await controller.submit(state, replace(decision, budget=24))
        assert result == {"status": "blocked", "reason": "unsupported_decision"}
        await client.aclose()
    asyncio.run(run())
    assert calls == []
