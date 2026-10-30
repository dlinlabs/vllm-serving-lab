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
