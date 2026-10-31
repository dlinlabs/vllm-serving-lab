import asyncio
from dataclasses import replace
import json

import httpx
import pytest

from mpc_lab.contracts import ModelParameters, Snapshot
from mpc_lab.http_capture import ChatTranscript, capture_chat, drain_evidence


RID = "a" * 32


def event(content=None, finish=None, usage=None):
    return ("data: " + json.dumps({
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
        "usage": usage,
    }, ensure_ascii=False) + "\n\n").encode()


def test_fragmented_utf8_events_preserve_first_last_and_usage_not_chunk_tokens():
    transcript = ChatTranscript()
    first = event("中")
    for byte in first:
        transcript.feed(bytes([byte]), 1)
    transcript.feed(event("文") + event(finish="stop") + event(
        usage={"prompt_tokens": 7, "completion_tokens": 3},
    ) + b"data: [DONE]\n\n", 2)
    assert transcript.complete()
    assert (transcript.first_at, transcript.last_at, transcript.content_events) == (1, 2, 2)
    assert transcript.usage["completion_tokens"] == 3


@pytest.mark.parametrize("suffix", [
    b"data: [DONE]\n\n",
    event(finish="stop") + b"data: [DONE]",
    event(finish="stop") + event("late") + b"data: [DONE]\n\n",
    event(finish="stop") + b"data: [DONE]\n\ndata: {}\n\n",
    b"data: \xff\n\n",
    b"data: {broken}\n\n",
    event(usage={"prompt_tokens": True, "completion_tokens": 1}),
])
def test_invalid_or_unterminated_stream_cannot_earn_completion(suffix):
    transcript = ChatTranscript()
    transcript.feed(event("first") + suffix, 1)
    assert not transcript.complete()


@pytest.mark.parametrize("mode", ["complete", "upstream503", "rejected", "wrong-id", "partial"])
def test_capture_standard_chat_labels_errors_without_confusing_upstream_rejection(mode):
    def serve(request):
        body = json.loads(request.content)
        assert body["messages"] and body["stream_options"] == {"include_usage": True}
        assert request.headers["x-benchmark-request-id"] == RID
        headers = {"content-type": "text/event-stream", "x-request-id": RID}
        if mode in ("rejected", "upstream503"):
            data = {"detail": "upstream overloaded"}
            if mode == "rejected":
                data = dict(detail="request rejected: admission capacity exceeded",
                            budget_version=1, max_admitted_cost=16)
            return httpx.Response(503, json=data, headers={"x-request-id": RID})
        if mode == "wrong-id":
            headers["x-request-id"] = "b" * 32
        content = event("answer") + event(finish="stop")
        if mode != "partial":
            content += b"data: [DONE]\n\n"
        return httpx.Response(200, content=content, headers=headers)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            captured = []
            row = await capture_chat(client, "http://test/v1/chat/completions",
                                     {"messages": [{"role": "user", "content": "hello"}]},
                                     RID, 0, enabled=True, clock=lambda: 1, domain="test",
                                     record=captured.append)
            assert captured == [row]
            assert row["token_timestamps_qualified"] is False
            assert row["status"] == {"complete": "complete", "rejected": "rejected"}.get(mode, "failed")
    asyncio.run(run())


def test_disabled_capture_and_cancelled_capture_do_not_lose_offered_identity():
    calls = []
    async def serve(request):
        calls.append(request)
        raise asyncio.CancelledError
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            payload = {"messages": [{"role": "user", "content": "hello"}]}
            assert (await capture_chat(client, "http://test", payload, RID, 0))["status"] == "disabled"
            assert not calls
            captured = []
            with pytest.raises(asyncio.CancelledError):
                await capture_chat(client, "http://test", payload, RID, 0, enabled=True,
                                   clock=lambda: 1, record=captured.append)
            assert captured[0]["request_id"] == RID
            assert captured[0]["status"] == "cancelled"
    asyncio.run(run())


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_capture_preserves_ignore_eos_intent_in_body_and_online_receipts(ignore_eos):
    def serve(request):
        assert json.loads(request.content)["ignore_eos"] is ignore_eos
        return httpx.Response(200, content=event("answer") + event(finish="stop")
            + b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream", "x-request-id": RID})

    async def run():
        receipts = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            row = await capture_chat(client, "http://test",
                {"messages": [], "ignore_eos": ignore_eos}, RID, 0, enabled=True,
                clock=lambda: 1, observe=lambda kind, receipt: receipts.append(receipt))
            assert row["status"] == "complete"
            assert row["requested_ignore_eos"] is ignore_eos
            assert all(r["requested_ignore_eos"] is ignore_eos for r in receipts)
    asyncio.run(run())


@pytest.mark.parametrize("ignore_eos", [None, 1, "true"])
def test_capture_rejects_ambiguous_ignore_eos_before_offering(ignore_eos):
    def serve(request):
        pytest.fail("Invalid sampling intent must not become an offered request")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            with pytest.raises(ValueError, match="invalid_ignore_eos"):
                await capture_chat(client, "http://test",
                    {"messages": [], "ignore_eos": ignore_eos}, RID, 0, enabled=True)
    asyncio.run(run())


def test_transport_timeout_is_censored_and_receive_clock_regression_is_invalid():
    transcript = ChatTranscript()
    transcript.feed(event("first"), 2)
    transcript.feed(event("last") + event(finish="stop") + b"data: [DONE]\n\n", 1)
    assert not transcript.complete()
    def timeout(request):
        raise httpx.ReadTimeout("unknown terminal")
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
            row = await capture_chat(client, "http://test", {"messages": []}, RID, 0,
                                     enabled=True, clock=lambda: 1)
            assert row["status"] == "timeout" and row["stream_done"] is False
            assert row["capture_error_type"] == "ReadTimeout"
            assert row["capture_error_stage"] == "response_headers"
    asyncio.run(run())


def test_preheader_transport_failure_retains_class_without_retry_or_sensitive_message():
    calls = []

    def fail(request):
        calls.append(request)
        raise httpx.RemoteProtocolError("sensitive connection and credential context")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            row = await capture_chat(client, "http://test", {"messages": []}, RID, 0,
                                     enabled=True, clock=lambda: 1)
            assert row["status"] == "failed" and row["http_status"] is None
            assert row["sent_at"] is not None
            assert row["capture_error_type"] == "RemoteProtocolError"
            assert row["capture_error_stage"] == "response_headers"
            assert len(calls) == 1
            assert "sensitive" not in json.dumps(row)
    asyncio.run(run())


def test_online_receipts_are_published_at_send_first_content_and_terminal():
    receipts = []
    ticks = iter(range(1, 30))

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            assert [kind for kind, _ in receipts] == ["offered"]
            yield event("answer")
            assert [kind for kind, _ in receipts] == ["offered", "first_content"]
            assert receipts[-1][1]["terminal_at"] is None
            assert receipts[-1][1]["usage"] is None
            assert receipts[-1][1]["content_events"] == 1
            yield event(finish="stop") + event(
                usage={"prompt_tokens": 7, "completion_tokens": 2},
            ) + b"data: [DONE]\n\n"

    def serve(request):
        assert [kind for kind, _ in receipts] == ["offered"]
        assert receipts[0][1]["sent_at"] < receipts[0][1]["published_at"]
        return httpx.Response(200, stream=Stream(),
            headers={"content-type": "text/event-stream", "x-request-id": RID})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            row = await capture_chat(client, "http://test", {"messages": []}, RID, 0,
                enabled=True, domain="test", clock=lambda: next(ticks),
                observe=lambda kind, receipt: receipts.append((kind, receipt)))
            assert row["status"] == "complete"
            assert [kind for kind, _ in receipts] == ["offered", "first_content", "terminal"]
            assert receipts[0][1]["first_content_at"] is None
            assert receipts[-1][1]["usage"] == row["usage"]
            receipts[-1][1]["usage"]["completion_tokens"] = 999
            assert row["usage"]["completion_tokens"] == 2
            assert all(r["clock_domain"] == "test" and r["request_id"] == RID for _, r in receipts)
            assert all(r["sent_at"] <= r["published_at"] for _, r in receipts)
            assert receipts[0][1]["published_at"] < receipts[1][1]["published_at"] < receipts[2][1]["published_at"]
    asyncio.run(run())


@pytest.mark.parametrize("cancel_before_send", [False, True])
def test_online_cancellation_distinguishes_sent_offer_from_unsent_plan(cancel_before_send):
    receipts = []

    async def serve(request):
        raise asyncio.CancelledError

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            async def capture():
                return await capture_chat(client, "http://test", {"messages": []}, RID,
                    10**12 if cancel_before_send else 0, enabled=True,
                    observe=lambda kind, receipt: receipts.append((kind, receipt)))
            task = asyncio.create_task(capture())
            if cancel_before_send:
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert [kind for kind, _ in receipts] == (
                ["terminal"] if cancel_before_send else ["offered", "terminal"])
            assert receipts[-1][1]["status"] == "cancelled"
            assert (receipts[-1][1]["sent_at"] is None) == cancel_before_send
    asyncio.run(run())


def test_online_observation_failure_does_not_change_http_completion_or_echo_error_message():
    def fail(kind, receipt):
        raise RuntimeError("sensitive callback context")

    def serve(request):
        return httpx.Response(200, content=event("answer") + event(finish="stop")
            + b"data: [DONE]\n\n", headers={"content-type": "text/event-stream", "x-request-id": RID})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            captured = []
            row = await capture_chat(client, "http://test", {"messages": []}, RID, 0,
                enabled=True, clock=lambda: 1, record=captured.append, observe=fail)
            assert row["status"] == "complete" and captured == [row]
            assert row["observation_error"] == "RuntimeError"
            assert "sensitive callback context" not in json.dumps(row)
    asyncio.run(run())


def test_async_publication_callback_is_not_silently_treated_as_receipt_delivery():
    async def observe(kind, receipt):
        pytest.fail("An asynchronous callback must not execute as a synchronous receipt")

    def serve(request):
        return httpx.Response(503, json={"detail": "upstream overloaded"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            row = await capture_chat(client, "http://test", {"messages": []}, RID, 0,
                enabled=True, clock=lambda: 1, observe=observe)
            assert row["status"] == "failed"
            assert row["observation_error"] == "TypeError"
    asyncio.run(run())


def test_gateway_empty_is_not_drain_and_foreign_clock_precedes_time_comparison():
    params = ModelParameters()
    snapshot = Snapshot("gateway", "test", 100, 16, 0, (), params.identity)
    engine = dict(run_id="engine", clock_domain="test", sampled_at=100,
                  running=0, waiting=0, orphaned=0)
    kwargs = dict(identity=params.identity, gateway_run="gateway", engine_run="engine",
                  domain="test", now=100, pending_operation=False)
    assert drain_evidence(snapshot, engine, **kwargs)["drained"]
    assert not drain_evidence(snapshot, {**engine, "orphaned": 1}, **kwargs)["drained"]
    assert not drain_evidence(snapshot, {**engine, "sampled_at": 99}, **kwargs)["drained"]
    assert not drain_evidence(snapshot, engine, **{**kwargs, "pending_operation": True})["drained"]
    assert drain_evidence(snapshot, {**engine, "clock_domain": "foreign", "sampled_at": "bad"},
                          **kwargs)["reason"] == "engine_clock_domain"
    assert not drain_evidence(replace(snapshot, run_id="other"), engine, **kwargs)["drained"]
