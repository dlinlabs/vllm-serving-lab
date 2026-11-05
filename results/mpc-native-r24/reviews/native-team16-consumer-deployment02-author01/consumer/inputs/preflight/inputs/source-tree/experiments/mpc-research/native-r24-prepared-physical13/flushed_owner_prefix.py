"""Explicit GET -> serialized flush RPC -> bounded received-prefix acquisition."""

import asyncio
from pathlib import Path
import time

from gateway_prefix_bridge import ROUTE, SCHEMA, source_sha256
from live_owner_prefix import (
    MAX_PREFIX_BYTES, Receipt, number, sha256, strict_json, validate_live_owner_prefix,
)


async def collect_flushed_owner_prefix(http, telemetry_path, *, clock=time.monotonic, **kwargs):
    sent = clock()
    response = await http.get("/admin/mpc/state")
    response.raise_for_status()
    snapshot_raw, received = response.content, clock()
    if len(snapshot_raw) > MAX_PREFIX_BYTES:
        raise ValueError("owner_prefix_capacity")
    snapshot = strict_json(snapshot_raw)
    sequence = snapshot["sequence"]
    if type(sequence) is not int or sequence < 0:
        raise ValueError("owner_prefix_invalid_sequence")
    flush_sent = clock()
    barrier = await http.post(ROUTE, json={
        "expected_run_id": kwargs["expected_run_id"], "expected_sequence": sequence,
    })
    barrier.raise_for_status()
    barrier_raw, flush_received = barrier.content, clock()
    if len(barrier_raw) > 16384:
        raise ValueError("owner_prefix_barrier_capacity")
    reply = strict_json(barrier_raw)
    if (reply["schema"] != SCHEMA or reply["source_sha256"] != source_sha256()
            or reply["run_id"] != kwargs["expected_run_id"]
            or reply["clock_domain"] != kwargs["expected_clock_domain"]
            or type(reply["flushed_through_sequence"]) is not int
            or reply["flushed_through_sequence"] < sequence
            or reply["dropped"] != 0 or reply["write_errors"] != 0
            or reply["native_mpc_permitted"] is not False):
        raise ValueError("owner_prefix_barrier_binding")
    times = [number(reply[key]) for key in
             ("source_at", "flush_started_at", "flush_finished_at", "sample_finished_at")]
    if times != sorted(times) or times[0] < number(snapshot["sampled_at"]):
        raise ValueError("owner_prefix_barrier_chronology")
    if not sent <= received <= flush_sent <= flush_received:
        raise ValueError("owner_prefix_receipt_chronology")

    def read_prefix():
        with Path(telemetry_path).open("rb") as handle:
            return handle.read(MAX_PREFIX_BYTES + 1)

    started = clock()
    if started < flush_received:
        raise ValueError("owner_prefix_receipt_chronology")
    raw = await asyncio.to_thread(read_prefix)
    ended = clock()
    receipt = Receipt(sent, received, started, ended, sha256(snapshot_raw), sha256(raw))
    checking = clock()
    result = validate_live_owner_prefix(
        snapshot_raw, raw, receipt=receipt, now=checking, **kwargs)
    finished = number(clock())
    if finished < checking:
        raise ValueError("owner_prefix_receipt_chronology")
    result["validation_elapsed_s"] = finished - checking
    result["source_age_upper_s"] += finished - checking
    result["age_as_of_receiver_clock"] = finished
    if result["source_age_upper_s"] > kwargs["max_age_s"]:
        raise ValueError("owner_prefix_stale_after_validation")
    result.update({
        "acquisition": "explicit_serialized_flush",
        "prefix_bytes": len(raw), "snapshot": snapshot,
        "barrier": reply, "barrier_sha256": sha256(barrier_raw),
        "barrier_sent_at": flush_sent, "barrier_received_at": flush_received,
        "barrier_roundtrip_s": flush_received - flush_sent,
    })
    return result
