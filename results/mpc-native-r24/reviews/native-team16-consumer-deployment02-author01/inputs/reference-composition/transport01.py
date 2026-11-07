"""Private concurrent receipt boundary derived from the unchanged control02 Capture."""

import asyncio
import base64
from copy import deepcopy
import math
from pathlib import Path
import re
import threading
from types import MappingProxyType

import composition_common01 as c


class Capture:
    def __init__(self, output, settings, secret, domain, *, clock, api, publisher=None):
        c.require(isinstance(secret, str) and len(secret) >= 16, "private_auth_value_required")
        self.output, self.secret, self.domain, self.clock, self.api = Path(output), secret, domain, clock, api
        self.settings = MappingProxyType(dict(settings))
        c.require(type(self.settings["maximum_http_exchanges"]) is int
                  and 0 < self.settings["maximum_http_exchanges"] <= 1900, "global_HTTP_cap")
        self.records, self.refusals, self.used_ids, self.active = [], [], set(), set()
        self.loop, self.thread = asyncio.get_running_loop(), threading.get_ident()
        self.closing = False
        self.publisher = publisher
        self.invalidator = getattr(getattr(publisher, "__self__", None), "invalidate_publication", None)
        c.require(publisher is None or callable(self.invalidator), "owned_publication_invalidator_required")
        for name in ("http", "http-reservations", "http-refusals"):
            (self.output / name).mkdir(mode=0o700)

    def scrub(self, raw):
        return raw.replace(self.secret.encode(), b"[AUTH_VALUE_REMOVED]")

    def observed(self, row, phase):
        if self.publisher is None:
            return True
        offer = row["request_id"] is not None or (
            row["role"] == "one-warmup" and row["method"] == "POST" and row["path"] == "/v1/chat/completions")
        control = row["path"] in ("/admin/mpc/budget", "/pr33-native/batch") and row["method"] == "POST"
        if not offer and not control:
            return True
        kind = ("offer_intent" if phase == "reservation" else
                "offer_dispatched" if phase == "dispatch_attempt" else "offer_outcome") \
            if offer else "control_history_published"
        evidence = dict(sequence=row["sequence"], request_id=row["request_id"],
                role=row["role"], method=row["method"], path=row["path"], phase=phase,
                scope="Caller-observed transport history,not proof of native receipt/application.")
        try:
            receipt = self.publisher(kind, evidence)
        except Exception as error:
            receipt = self.invalidator(kind, evidence, error)
        if isinstance(receipt, dict) and receipt.get("status") == "consumer_refused_producer_continues":
            c.require(receipt["authority_poisoned"] is True, "publication_refusal_requires_sticky_poison")
            row.setdefault("consumer_publication_refusals", []).append(dict(phase=phase, receipt=receipt))
        return True

    def refuse(self, role, method, path, identifier, reason):
        row = dict(sequence=len(self.refusals), role=role, method=method, path=path,
                   request_id=identifier, reason=reason, refused_at=self.clock(),
                   client_clock_domain=self.domain, HTTP_reserved=False, dispatch_attempted=False)
        self.observed(row, "reservation_refused")
        self.refusals.append(row)
        self.api.persist(self.output / "http-refusals" / f'{row["sequence"]:03d}.json', row)
        raise ValueError(reason)

    async def exchange(self, client, method, path, *, role, payload=None, auth=False, identifier=None,
                       deadline, limit, started=None, before_send=None):
        c.require(asyncio.get_running_loop() is self.loop and threading.get_ident() == self.thread,
                  "single_receipt_reservation_loop")
        if self.closing:
            self.refuse(role, method, path, identifier, "transport_closing")
        if len(self.records) >= self.settings["maximum_http_exchanges"]:
            self.refuse(role, method, path, identifier, "HTTP_exchange_cap")
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (deadline, limit)):
            self.refuse(role, method, path, identifier, "unknown_HTTP_time_bound")
        if identifier is not None and (not isinstance(identifier, str)
                or not re.fullmatch("[0-9a-f]{32}", identifier) or identifier in self.used_ids):
            self.refuse(role, method, path, identifier, "duplicate_or_invalid_managed_identifier")
        content = self.api.canonical(payload) if payload is not None else None
        headers = {"content-type": "application/json"} if content is not None else {}
        if auth:
            headers["x-admin-api-key" if path.startswith("/admin/") else "x-observer-key"] = self.secret
        if identifier is not None:
            headers["x-benchmark-request-id"] = identifier
            self.used_ids.add(identifier)
        row = dict(sequence=len(self.records), role=role, method=method, path=path,
            target=dict(scheme=client.base_url.scheme, host=client.base_url.host, port=client.base_url.port),
            request_body=self.scrub(content).decode() if content is not None else None,
            request_id=identifier, client_clock_domain=self.domain, reserved_at=self.clock(),
            client_send_at=None, client_headers_at=None, client_receive_at=None,
            http_status=None, response_headers={}, error_type=None, body_complete=False,
            auth_values_captured=False, credential_removed=False, dispatch_attempted=False)
        # No await precedes this append: reservation/cap/ID admission is one loop operation.
        self.observed(row, "reservation")
        self.records.append(row)
        task = asyncio.current_task()
        self.active.add(task)
        raw, late = bytearray(), False
        try:
            reservation = dict(deepcopy(row), scope="RESERVATION_NOT_PROOF_OF_SOCKET_SEND_OR_NATIVE_RECEIPT")
            self.api.persist(self.output / "http-reservations" / f'{row["sequence"]:03d}.json', reservation)
            if before_send is not None:
                before_send()
            timeout = min(limit, deadline - self.clock())
            c.require(timeout > 0, "HTTP_absolute_deadline")
            settings = MappingProxyType(dict(self.settings, http_timeout_s=timeout, absolute_deadline=deadline))
            row["call_settings"] = dict(settings)
            self.observed(row, "dispatch_attempt")
            row["client_send_at"], row["dispatch_attempted"] = self.clock(), True
            if started is not None and not started.done():
                started.set_result(dict(sequence=row["sequence"], client_send_at=row["client_send_at"]))
            async with asyncio.timeout(timeout):
                async with client.stream(method, path, content=content, headers=headers,
                                         timeout=settings["http_timeout_s"]) as response:
                    self.observed(row, "response_headers")
                    row.update(http_status=response.status_code, client_headers_at=self.clock())
                    row["response_headers"] = {
                        key: self.scrub(response.headers[key].encode()).decode(errors="replace")
                        for key in ("content-type", "x-request-id") if key in response.headers}
                    row["credential_removed"] = any(
                        self.secret in response.headers.get(key, "") for key in ("content-type", "x-request-id"))
                    async for chunk in response.aiter_bytes():
                        room = settings["body_cap_bytes"] - len(raw)
                        raw.extend(chunk[:room])
                        c.require(len(chunk) <= room, "HTTP_body_cap")
                    self.observed(row, "body_complete")
                    row["body_complete"] = True
        except (asyncio.CancelledError, TimeoutError, ValueError, KeyError) as error:
            self.observed(row, "transport_exception")
            row["error_type"] = type(error).__name__
            raise
        except Exception as error:
            self.observed(row, "transport_exception")
            row["error_type"] = type(error).__name__
        finally:
            self.observed(row, "terminal_receipt")
            row["client_receive_at"] = self.clock()
            late = row["dispatch_attempted"] and (
                row["client_receive_at"] > deadline or
                row["client_receive_at"] - row["client_send_at"] > row["call_settings"]["http_timeout_s"])
            if late and row["error_type"] is None:
                row["error_type"] = "LateReceipt"
            clean = self.scrub(bytes(raw))
            row.update(credential_removed=row["credential_removed"] or clean != bytes(raw),
                       response_body_base64=base64.b64encode(clean).decode(),
                       response_body_sha256=self.api.sha(clean), response_bytes_captured=len(clean))
            try:
                self.api.persist(self.output / "http" / f'{row["sequence"]:03d}.json', row)
            finally:
                self.active.discard(task)
        c.require(not late, "HTTP_late_receipt_retained")
        c.require(row["credential_removed"] is False, "HTTP_credential_echo_retained_redacted")
        return row

    async def close(self):
        self.closing = True
        tasks = [task for task in self.active if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=0.2)
            c.require(not pending, "concurrent_HTTP_tasks_not_joined")
        c.require(not self.active, "transport_still_active")
