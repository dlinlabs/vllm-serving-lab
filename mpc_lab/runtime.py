"""Strict snapshot adapter and GET-only shadow runner, not a live controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import importlib
import json
import math
import os
from typing import Callable

import httpx

from .contracts import (
    SCHEMA,
    GUARDED_CAPABILITY,
    CostSpec,
    Decision,
    Identity,
    RequestMark,
    Snapshot,
    finite,
)


def _number(value: object, *, positive: bool = False) -> bool:
    try:
        return finite(value, positive=positive)
    except (OverflowError, TypeError):
        return False


def _text(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 1024


def snapshot_to_dict(snapshot: Snapshot) -> dict:
    document = {"schema": SCHEMA, **asdict(snapshot)}
    document["requests"] = list(document["requests"])
    snapshot_from_dict(document)
    return document


def snapshot_from_dict(document: object) -> Snapshot:
    """Validate server-clock marks without comparing them to a client clock."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("snapshot_schema")
    keys = {field.name for field in fields(Snapshot)}
    if not keys <= document.keys() or document.keys() - keys - {
        "schema", "capabilities", "cost_spec",
    }:
        raise ValueError("snapshot_fields")
    if not _text(document["run_id"]) or not _text(document["clock_domain"]):
        raise ValueError("snapshot_clock_identity")
    if (
        not _number(document["sampled_at"])
        or document["sampled_at"] < 0
        or not _number(document["budget"], positive=True)
        or type(document["budget_version"]) is not int
        or document["budget_version"] < 0
        or type(document["sequence"]) is not int
        or document["sequence"] < 0
        or type(document["complete"]) is not bool
    ):
        raise ValueError("snapshot_values")
    raw_identity = document["identity"]
    if (
        not isinstance(raw_identity, dict)
        or set(raw_identity) != {field.name for field in fields(Identity)}
        or not all(_text(value) for value in raw_identity.values())
    ):
        raise ValueError("snapshot_identity")
    identity = Identity(**raw_identity)
    if "cost_spec" in document:
        try:
            cost_spec = CostSpec(**document["cost_spec"])
            if cost_spec.fingerprint != identity.cost_hash:
                raise ValueError("snapshot_cost_identity")
        except (TypeError, OverflowError) as exc:
            raise ValueError("snapshot_cost_spec") from exc
    raw_requests = document["requests"]
    if not isinstance(raw_requests, (list, tuple)):
        raise ValueError("snapshot_requests")
    marks = []
    seen = set()
    for raw in raw_requests:
        if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(RequestMark)}:
            raise ValueError("snapshot_mark_fields")
        mark = RequestMark(**raw)
        if (
            not _text(mark.request_id)
            or mark.request_id in seen
            or not _number(mark.cost, positive=True)
            or type(mark.prompt_tokens) is not int
            or mark.prompt_tokens < 0
            or type(mark.max_tokens) is not int
            or mark.max_tokens <= 0
            or not _number(mark.received_at)
            or not _number(mark.admitted_at)
            or not 0 <= mark.received_at <= mark.admitted_at <= document["sampled_at"]
            or (mark.first_at is not None and (
                not _number(mark.first_at)
                or not mark.admitted_at <= mark.first_at <= document["sampled_at"]
            ))
            or (mark.scheduled_at is not None and (
                not _number(mark.scheduled_at)
                or not 0 <= mark.scheduled_at <= mark.received_at
            ))
        ):
            raise ValueError("snapshot_mark_values")
        seen.add(mark.request_id)
        marks.append(mark)
    try:
        if not math.isfinite(math.fsum(mark.cost for mark in marks)):
            raise ValueError("snapshot_reserved")
    except OverflowError as exc:
        raise ValueError("snapshot_reserved") from exc
    return Snapshot(
        **{key: document[key] for key in keys - {"identity", "requests"}},
        identity=identity,
        requests=tuple(marks),
    )


def require_guarded_capability(document: object) -> None:
    """Reject legacy metrics/admin documents; this never grants live authority."""
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or not isinstance(document.get("capabilities"), list)
        or GUARDED_CAPABILITY not in document["capabilities"]
    ):
        raise ValueError("guarded_capability_missing")


async def shadow_once(
    client: httpx.AsyncClient,
    *,
    planner: Callable[[Snapshot], Decision] | None = None,
) -> dict:
    response = await client.get("/admin/mpc/state")
    response.raise_for_status()
    document = response.json()
    require_guarded_capability(document)
    snapshot = snapshot_from_dict(document)
    reason = None
    if not snapshot.complete:
        reason = "incomplete_snapshot"
    elif planner is None:
        try:
            controller = importlib.import_module("mpc_lab.controller")
        except ModuleNotFoundError as exc:
            if exc.name != "mpc_lab.controller":
                raise
            reason = "controller_unavailable"
        else:
            planner = controller.decide
    if reason is not None:
        decision = Decision(
            snapshot.budget, None, reason, True, None, None, 0.0, 0, 0, snapshot.identity,
        )
    else:
        decision = planner(snapshot)
        if not isinstance(decision, Decision) or decision.identity != snapshot.identity:
            raise ValueError("shadow_decision_identity")
    record = {
        "schema": SCHEMA,
        "mode": "shadow",
        "applied": False,
        "run_id": snapshot.run_id,
        "clock_domain": snapshot.clock_domain,
        "sampled_at": snapshot.sampled_at,
        "sequence": snapshot.sequence,
        "budget_version": snapshot.budget_version,
        "decision": asdict(decision),
    }
    json.dumps(record, allow_nan=False)
    return record


def main(argv: list[str] | None = None) -> None:
    import asyncio

    parser = argparse.ArgumentParser(description="One GET-only MPC shadow decision; never actuates.")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args(argv)

    async def run():
        async with httpx.AsyncClient(
            base_url=args.url,
            headers={"x-admin-api-key": os.getenv("ADMIN_API_KEY", "")},
            timeout=5.0,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            print(json.dumps(await shadow_once(client), sort_keys=True, allow_nan=False))

    asyncio.run(run())


if __name__ == "__main__":
    main()
