import asyncio
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from mpc_lab.contracts import GUARDED_CAPABILITY, SCHEMA
from mpc_lab.runtime import snapshot_from_dict, snapshot_to_dict


HEADERS = {"x-admin-api-key": "test-key"}
CONTENT = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
DONE = b'data: [DONE]\n\n'


@pytest.fixture
def load_gateway(monkeypatch):
    def load(enabled="true", settings=None):
        monkeypatch.setenv("MPC_LAB_ENABLED", enabled)
        monkeypatch.setenv("ADMISSION_MODE", "cost_aware")
        monkeypatch.setenv("MAX_ADMITTED_COST", "80")
        monkeypatch.setenv("ADMIN_API_KEY", "test-key")
        monkeypatch.setenv("WEB_CONCURRENCY", "1")
        monkeypatch.setenv("UVICORN_WORKERS", "1")
        for key in ("MPC_MODEL_VERSION", "MPC_CALIBRATION_VERSION", "MPC_SUPPORT_HASH",
                    "MPC_TOKENIZER_IDENTITY"):
            monkeypatch.delenv(key, raising=False)
        for key, value in (settings or {}).items():
            monkeypatch.setenv(key, value)
        def offline(*args, **kwargs):
            raise OSError("test has no tokenizer")
        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=offline),
        ))
        spec = importlib.util.spec_from_file_location(
            "mpc_test_gateway", Path(__file__).parents[1] / "gateway.py",
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module
    return load


@pytest.fixture
def gateway(load_gateway):
    return load_gateway()


def client_for(gateway):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway.app), base_url="http://test",
        headers=HEADERS,
    )


def command(gateway, **changes):
    payload = {
        "schema": SCHEMA, "run_id": gateway.telemetry.run_id,
        "clock_domain": gateway.mpc_clock_domain, "identity": asdict(gateway.mpc_identity),
        "expected_version": gateway.state["budget_version"],
        "operation_id": "operation-1", "budget": 16.0, "not_after": gateway.mpc_now() + 10,
    }
    return {**payload, **changes}


def control_state(gateway):
    return dict(gateway.state), dict(gateway.mpc_operations), gateway.mpc_sequence


def test_default_off_preserves_legacy(load_gateway):
    gateway = load_gateway("false")
    async def scenario():
        async with client_for(gateway) as client:
            assert (await client.get("/admin/mpc/state")).status_code == 404
            assert (await client.post("/admin/mpc/budget", json={})).status_code == 404
            assert (await client.post("/admin/budget", json={"budget": 32})).status_code == 200
            metrics = (await client.get("/metrics")).json()
            assert metrics["admission_budget"] == 32 and metrics["budget_version"] == 1
            assert (await gateway.begin_request(30, False)).accepted
            assert (await gateway.begin_request(3, False)).accepted is False
            await gateway.finish_request(30)
            assert gateway.mpc_requests == {} and gateway.mpc_sequence == 0
    asyncio.run(scenario())


def test_configured_identity_seals_actual_cost_parameters(load_gateway):
    gateway = load_gateway(settings={
        "MPC_MODEL_VERSION": "test-model", "MPC_CALIBRATION_VERSION": "test-cal",
        "MPC_SUPPORT_HASH": "test-support", "MPC_TOKENIZER_IDENTITY": "test-tokenizer",
        "COST_INTERCEPT": "0.5", "INPUT_TOKEN_COST": "0.002",
        "OUTPUT_TOKEN_COST": "0.01", "MIN_REQUEST_COST": "0.2",
        "DEFAULT_MAX_TOKENS": "64",
    })
    cost, prompt, maximum, fallback = gateway.estimate_request_cost({})
    assert maximum == 64 and fallback
    assert cost == gateway.mpc_cost_spec.estimate(prompt, maximum)
    assert gateway.mpc_identity.model_version == "test-model"
    assert gateway.mpc_identity.calibration_version == "test-cal"
    assert gateway.mpc_identity.support_hash == "test-support"
    assert "test-tokenizer" in gateway.mpc_cost_spec.tokenizer_identity
    assert gateway.mpc_identity.cost_hash == gateway.mpc_cost_spec.fingerprint


def test_auth_mode_and_single_worker_restrictions(gateway, monkeypatch):
    async def scenario():
        async with client_for(gateway) as client:
            for path, method in (("/admin/mpc/state", client.get),
                                 ("/admin/mpc/budget", client.post)):
                assert (await method(path, headers={"x-admin-api-key": "bad"})).status_code == 401
            gateway.ADMISSION_MODE = "baseline"
            assert (await client.get("/admin/mpc/state")).status_code == 409
            gateway.ADMISSION_MODE = "cost_aware"
            monkeypatch.setenv("WEB_CONCURRENCY", "2")
            assert (await client.get("/admin/mpc/state")).status_code == 409
            gateway.ADMIN_API_KEY = ""
            assert (await client.get("/admin/mpc/state")).status_code == 503
    asyncio.run(scenario())


def test_snapshot_identity_cost_and_truthful_incomplete_state(gateway):
    async def scenario():
        async with client_for(gateway) as client:
            raw = (await client.get("/admin/mpc/state")).json()
            snap = snapshot_from_dict(raw)
            assert raw["capabilities"] == [GUARDED_CAPABILITY]
            assert snap.identity.cost_hash == gateway.mpc_cost_spec.fingerprint
            assert raw["cost_spec"]["default_max_tokens"] == gateway.DEFAULT_MAX_TOKENS
            assert raw["cost_spec"]["input_per_token"] == gateway.INPUT_TOKEN_COST
            assert raw["cost_spec"]["estimator_version"] == "gateway-token-estimate-v1"
            assert "unqualified" in raw["cost_spec"]["tokenizer_identity"]
            assert snap.identity.model_version == "unqualified"
            assert snap.identity.implementation_hash == gateway.implementation_hash()
            assert snap.complete and snap.requests == ()
            # Legacy helper use without marks cannot masquerade as a complete sensor.
            await gateway.begin_request(1, False)
            assert not snapshot_from_dict((await client.get("/admin/mpc/state")).json()).complete
            await gateway.finish_request(1)
            assert snapshot_from_dict((await client.get("/admin/mpc/state")).json()).complete
    asyncio.run(scenario())


@pytest.mark.parametrize("field", [
    "schema", "run_id", "clock_domain", "model_version", "calibration_version",
    "support_hash", "cost_hash", "implementation_hash",
])
def test_wrong_identity_never_mutates(gateway, field):
    async def scenario():
        payload = command(gateway)
        if field in payload:
            payload[field] = "wrong"
        else:
            payload["identity"][field] = "wrong"
        before = control_state(gateway)
        async with client_for(gateway) as client:
            response = await client.post("/admin/mpc/budget", json=payload)
        assert response.status_code == 409 and "identity" in response.json()["detail"]
        assert control_state(gateway) == before
    asyncio.run(scenario())


@pytest.mark.parametrize("uptime", [0.0, 50.0, 1000.0])
def test_cross_boot_clock_rejected_before_uptime(gateway, monkeypatch, uptime):
    monkeypatch.setattr(gateway, "mpc_now", lambda: uptime)
    async def scenario():
        before = control_state(gateway)
        async with client_for(gateway) as client:
            response = await client.post("/admin/mpc/budget", json=command(
                gateway, clock_domain="other-boot", not_after=50.0,
            ))
        assert response.status_code == 409 and response.json()["detail"] == "command identity mismatch"
        assert control_state(gateway) == before
    asyncio.run(scenario())


@pytest.mark.parametrize("key,value", [
    ("budget", True), ("budget", "32"), ("budget", 0), ("budget", -1),
    ("budget", float("nan")), ("budget", float("inf")), ("budget", 10 ** 400),
    ("not_after", True), ("not_after", float("nan")), ("not_after", float("-inf")),
    ("not_after", 10 ** 400), ("expected_version", True), ("expected_version", -1),
    ("expected_version", 0.0), ("operation_id", ""), ("operation_id", []),
    ("identity", []),
])
def test_malformed_guarded_input_no_mutation(gateway, key, value):
    async def scenario():
        before = control_state(gateway)
        async with client_for(gateway) as client:
            response = await client.post(
                "/admin/mpc/budget", content=json.dumps(command(gateway, **{key: value})),
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 422
        assert control_state(gateway) == before
    asyncio.run(scenario())


@pytest.mark.parametrize("raw", ["{", "[]", "null", "{}", '{"budget":16}', '{"extra":true}'])
def test_malformed_command_shape(gateway, raw):
    async def scenario():
        before = control_state(gateway)
        async with client_for(gateway) as client:
            response = await client.post("/admin/mpc/budget", content=raw)
        assert response.status_code == 422 and control_state(gateway) == before
    asyncio.run(scenario())


def test_concurrent_commands_have_one_cas_winner(gateway):
    async def scenario():
        async with client_for(gateway) as client:
            first = command(gateway)
            second = {**first, "operation_id": "second", "budget": 32}
            replies = await asyncio.gather(*(
                client.post("/admin/mpc/budget", json=payload) for payload in (first, second)
            ))
            assert sorted(reply.status_code for reply in replies) == [200, 409]
            assert gateway.state["budget_version"] == 1
            assert len(gateway.mpc_operations) == 1
    asyncio.run(scenario())


def test_legacy_update_invalidates_guarded_cas_but_keeps_legacy_response(gateway):
    async def scenario():
        async with client_for(gateway) as client:
            payload = command(gateway)
            legacy = await client.post("/admin/budget", json={"budget": 32})
            assert legacy.status_code == 200 and set(legacy.json()) == {
                "previous_budget", "budget", "current_admitted_cost", "budget_version", "applied_at",
            }
            before = control_state(gateway)
            response = await client.post("/admin/mpc/budget", json=payload)
            assert response.status_code == 409 and control_state(gateway) == before
    asyncio.run(scenario())


def test_stale_cas_idempotency_expiry_and_payload_reuse(gateway, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(gateway, "mpc_now", lambda: now[0])
    async def scenario():
        async with client_for(gateway) as client:
            payload = command(gateway, not_after=20)
            result = await client.post("/admin/mpc/budget", json=payload)
            assert result.status_code == 200
            assert result.json()["budget_version"] == 1 and result.json()["applied_at"] == 10
            # Same payload acknowledges the original application, even after expiry.
            now[0] = 30.0
            before = control_state(gateway)
            replay = await client.post("/admin/mpc/budget", json=payload)
            assert replay.json() == result.json() and control_state(gateway) == before
            changed = await client.post("/admin/mpc/budget", json={**payload, "budget": 32})
            assert changed.status_code == 409 and "reused" in changed.json()["detail"]
            stale = await client.post("/admin/mpc/budget", json=command(
                gateway, operation_id="stale", expected_version=0,
            ))
            assert stale.status_code == 409 and "stale" in stale.json()["detail"]
            expired = await client.post("/admin/mpc/budget", json=command(
                gateway, operation_id="expired", not_after=now[0],
            ))
            assert expired.status_code == 409 and "expired" in expired.json()["detail"]
            assert control_state(gateway) == before
    asyncio.run(scenario())


@pytest.mark.parametrize("invalidate", ["expiry", "version"])
def test_rechecks_after_queued_lock(gateway, monkeypatch, invalidate):
    now = [10.0]
    monkeypatch.setattr(gateway, "mpc_now", lambda: now[0])
    async def scenario():
        async with client_for(gateway) as client:
            payload = command(gateway, not_after=11)
            await gateway.state_lock.acquire()
            task = asyncio.create_task(client.post("/admin/mpc/budget", json=payload))
            await asyncio.sleep(0)
            assert not task.done()
            if invalidate == "expiry":
                now[0] = 12.0
            else:
                gateway.state["budget_version"] += 1
            before = control_state(gateway)
            gateway.state_lock.release()
            result = await task
            assert result.status_code == 409
            assert ("expired" if invalidate == "expiry" else "stale") in result.json()["detail"]
            assert control_state(gateway) == before
    asyncio.run(scenario())


def test_bounded_outcomes_refuse_new_without_forgetting_old(gateway, monkeypatch):
    monkeypatch.setattr(gateway, "MPC_OPERATION_LIMIT", 1)
    async def scenario():
        async with client_for(gateway) as client:
            payload = command(gateway)
            first = await client.post("/admin/mpc/budget", json=payload)
            before = control_state(gateway)
            assert first.status_code == 200
            assert (await client.post("/admin/mpc/budget", json=command(
                gateway, operation_id="second",
            ))).status_code == 503
            assert (await client.post("/admin/mpc/budget", json=payload)).json() == first.json()
            assert control_state(gateway) == before
    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "connect_error"])
def test_marked_concurrent_lifecycle_and_lower_cap(gateway, monkeypatch, outcome):
    async def scenario():
        real_client = httpx.AsyncClient
        async def upstream(sent):
            if outcome == "connect_error":
                raise httpx.ConnectError("offline test")
            class Stream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield CONTENT
                    if outcome == "failed":
                        raise httpx.ReadError("offline test")
                    yield DONE
            return httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})
        admin = client_for(gateway)
        monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda: real_client(
            transport=httpx.MockTransport(upstream),
        ))
        monkeypatch.setattr(gateway, "estimate_request_cost", lambda p: (20.0, 98, 128, True))
        def request():
            async def receive():
                return {"type": "http.request", "body": b'{"stream":true}', "more_body": False}
            return Request({
                "type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [],
            }, receive)
        async with admin as client:
            responses = await asyncio.gather(*(gateway.proxy_chat_completions(request()) for _ in range(3)))
            if outcome == "connect_error":
                assert all(response.status_code == 502 for response in responses)
            else:
                raw = (await client.get("/admin/mpc/state")).json()
                before = snapshot_from_dict(raw)
                assert before.complete and before.reserved == 60 and len(before.requests) == 3
                assert all(mark.prompt_tokens == 98 and mark.max_tokens == 128
                           and mark.cost == 20 and mark.first_at is None for mark in before.requests)
                assert snapshot_from_dict(snapshot_to_dict(before)) == before
                for response in responses:
                    assert await anext(response.body_iterator) == CONTENT
                after = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                assert all(mark.first_at is not None for mark in after.requests)
                assert after.prompt_backlog == 0 and after.reserved == 60
                assert after.sequence > before.sequence
                reduction = await client.post("/admin/mpc/budget", json=command(gateway, budget=16))
                assert reduction.status_code == 200
                reduced = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                assert reduced.requests == after.requests and reduced.reserved == 60 > reduced.budget
                assert (await gateway.proxy_chat_completions(request())).status_code == 503
                for response in responses:
                    if outcome == "cancelled":
                        await response.body_iterator.aclose()
                    elif outcome == "failed":
                        with pytest.raises(httpx.ReadError):
                            await anext(response.body_iterator)
                    else:
                        assert await anext(response.body_iterator) == DONE
                        with pytest.raises(StopAsyncIteration):
                            await anext(response.body_iterator)
                    await response.background()
            final = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
            assert final.complete and not final.requests and final.reserved == 0
            assert gateway.state["current_admitted_cost"] == gateway.state["current_in_flight"] == 0
    asyncio.run(scenario())
