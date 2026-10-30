"""Standard Chat capture and strict drain evidence; no launch or CLI activation."""

import asyncio
import codecs
import json
import re
import time

import httpx

from telemetry import has_sse_content
from .contracts import finite
from .controller import validate_snapshot
from .runtime import local_clock_domain


class ChatTranscript:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self.buffer = ""
        self.data = []
        self.done = False
        self.error = False
        self.first_at = self.last_at = None
        self.finish_reason = None
        self.usage = None
        self.content_events = 0
        self.last_received_at = None

    def _event(self, payload, at):
        if self.done:
            raise ValueError("event_after_done")
        if payload == "[DONE]":
            if self.finish_reason not in ("stop", "length"):
                raise ValueError("done_without_finish")
            self.done = True
            return
        event = json.loads(payload)
        content = has_sse_content(event)
        if content and self.finish_reason is not None:
            raise ValueError("content_after_finish")
        choices = event.get("choices", [])
        if len(choices) > 1:
            raise ValueError("multiple_choices")
        for choice in choices:
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected_choice")
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "length") or self.finish_reason is not None:
                    raise ValueError("invalid_finish")
                self.finish_reason = reason
        if content:
            if self.first_at is None:
                self.first_at = at
            self.last_at = at
            self.content_events += 1
        usage = event.get("usage")
        if usage is not None:
            if any(type(usage.get(k)) is not int or usage[k] < 0
                   for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("invalid_usage")
            if self.usage is not None and self.usage != usage:
                raise ValueError("changed_usage")
            self.usage = usage

    def feed(self, chunk, at):
        try:
            if (not finite(at) or at < 0
                    or (self.last_received_at is not None and at < self.last_received_at)):
                raise ValueError("invalid_receive_time")
            self.last_received_at = at
            self.buffer += self.decoder.decode(chunk)
            if len(self.buffer) + sum(map(len, self.data)) > 1024 * 1024:
                raise ValueError("sse_event_limit")
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("data:"):
                    self.data.append(line[5:].lstrip(" "))
                elif not line and self.data:
                    data, self.data = "\n".join(self.data), []
                    self._event(data, at)
        except (ValueError, UnicodeError, TypeError, AttributeError):
            self.error = True

    def complete(self):
        try:
            self.buffer += self.decoder.decode(b"", final=True)
        except UnicodeError:
            self.error = True
        return (self.done and not self.error and not self.buffer.strip()
                and not self.data and self.first_at is not None)


async def capture_chat(client, url, payload, request_id, scheduled_at, *,
                       enabled=False, domain=None, clock=time.monotonic, timeout_s=45,
                       record=None):
    if enabled is not True:
        return {"request_id": request_id, "status": "disabled"}
    if not isinstance(request_id, str) or not re.fullmatch("[0-9a-f]{32}", request_id):
        raise ValueError("invalid_request_id")
    if not finite(scheduled_at) or scheduled_at < 0 or not finite(timeout_s, positive=True):
        raise ValueError("invalid_capture_time")
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError("standard_chat_required")
    if type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1 or payload.get("tools"):
        raise ValueError("unsupported_chat_mode")
    body = json.dumps({**payload, "n": 1, "stream": True,
                       "stream_options": {"include_usage": True}}, allow_nan=False).encode()
    row = dict(request_id=request_id, clock_domain=domain or local_clock_domain(),
               request_model=payload.get("model"), requested_max_tokens=payload.get("max_tokens"),
               scheduled_at=scheduled_at, sent_at=None, first_content_at=None,
               last_content_at=None, terminal_at=None, status="attempted",
               http_status=None, stream_done=False, stream_error=False,
               finish_reason=None, response_request_id=None, usage=None,
               content_events=0, token_timestamps_qualified=False, admission_rejection=False)
    observer = ChatTranscript()
    try:
        await asyncio.sleep(max(0, scheduled_at - clock()))
        row["sent_at"] = clock()
        async with asyncio.timeout(timeout_s):
            async with client.stream("POST", url, content=body, headers={
                "content-type": "application/json", "x-benchmark-request-id": request_id,
            }) as response:
                row["http_status"] = response.status_code
                row["response_request_id"] = response.headers.get("x-request-id")
                if response.status_code != 200:
                    row["status"] = "failed"
                    if response.status_code == 503:
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > 65536:
                                raise ValueError("error_body_limit")
                        rejection = json.loads(data)
                        if (isinstance(rejection, dict)
                                and rejection.get("detail") == "request rejected: admission capacity exceeded"
                                and type(rejection.get("budget_version")) is int
                                and rejection["budget_version"] >= 0
                                and finite(rejection.get("max_admitted_cost"), positive=True)
                                and row["response_request_id"] == request_id):
                            row.update(status="rejected", admission_rejection=True,
                                       admission=rejection)
                elif response.headers.get("content-type", "").split(";")[0].strip().lower() != "text/event-stream":
                    row["status"] = "failed"
                else:
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 8 * 1024 * 1024:
                            raise ValueError("body_limit")
                        observer.feed(chunk, clock())
                    row["status"] = ("complete" if observer.complete()
                                     and row["response_request_id"] == request_id else "failed")
    except asyncio.CancelledError:
        row["status"] = "cancelled"
        raise
    except (TimeoutError, httpx.TimeoutException):
        row["status"] = "timeout"
    except (httpx.HTTPError, ValueError):
        row["status"] = "failed"
    finally:
        row.update(first_content_at=observer.first_at, last_content_at=observer.last_at,
                   terminal_at=clock(), stream_done=observer.done, stream_error=observer.error,
                   finish_reason=observer.finish_reason, usage=observer.usage,
                   content_events=observer.content_events)
        if record is not None:
            record(dict(row))
    return row


def drain_evidence(snapshot, engine, *, identity, gateway_run, engine_run,
                   domain, now, pending_operation=True, max_age_s=0.5):
    reason = validate_snapshot(snapshot, now=now, clock_domain=domain,
                               expected_identity=identity, max_age_s=max_age_s)
    if reason:
        return {"drained": False, "reason": reason}
    if not isinstance(engine, dict) or engine.get("clock_domain") != domain:
        return {"drained": False, "reason": "engine_clock_domain"}
    if snapshot.run_id != gateway_run or engine.get("run_id") != engine_run:
        return {"drained": False, "reason": "run_identity"}
    at = engine.get("sampled_at")
    if not finite(at) or not 0 <= now - at <= max_age_s:
        return {"drained": False, "reason": "engine_stale"}
    if any(type(engine.get(k)) is not int or engine[k] < 0
           for k in ("running", "waiting", "orphaned")):
        return {"drained": False, "reason": "engine_unavailable"}
    if pending_operation is not False:
        return {"drained": False, "reason": "pending_operation"}
    clear = not snapshot.requests and snapshot.reserved == 0 and not any(
        engine[k] for k in ("running", "waiting", "orphaned")
    )
    return {"drained": clear, "reason": "clear" if clear else "active_backend"}
