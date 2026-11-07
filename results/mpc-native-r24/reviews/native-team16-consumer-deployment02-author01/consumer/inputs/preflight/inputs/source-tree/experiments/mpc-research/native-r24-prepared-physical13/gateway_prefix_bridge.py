"""Opt-in experiment bridge; original product files and no-flush API stay intact."""

import argparse
import asyncio
import hashlib
from pathlib import Path
import runpy
import sys
import time

from live_owner_prefix import PRODUCT_SHA256, strict_json
from process_clock_provenance import self_process_metadata


ROUTE = "/admin/mpc/flush-prefix"
SCHEMA = "r24-gateway-flushed-prefix-v1"


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def application():
    import gateway

    install_gateway_prefix_bridge(gateway.app, enabled=True)
    return gateway.app


def install_gateway_prefix_bridge(app, *, enabled=False):
    if not enabled:
        return None
    if getattr(app.state, "r24_prefix_bridge", None) is not None:
        raise ValueError("gateway_prefix_bridge_already_installed")
    endpoints = [route.endpoint for route in app.routes
                 if getattr(route, "path", None) == "/admin/mpc/state"]
    if len(endpoints) != 1 or any(getattr(route, "path", None) == ROUTE for route in app.routes):
        raise ValueError("gateway_prefix_bridge_route_binding")
    snapshot = endpoints[0]
    namespace = snapshot.__globals__
    root = Path(namespace["__file__"]).resolve().parent
    for name, expected in PRODUCT_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("gateway_prefix_bridge_unreviewed_product")
    if namespace.get("MPC_OWNER_LEDGER_ENABLED") is not True or not namespace.get("TELEMETRY_JSONL"):
        raise ValueError("gateway_prefix_bridge_owner_ledger_required")
    telemetry = namespace["telemetry"]
    module = sys.modules.get(type(telemetry).__module__)
    if (module is None or getattr(module, "Telemetry", None) is not type(telemetry)
            or Path(module.__file__).resolve() != (root / "telemetry.py").resolve()):
        raise ValueError("gateway_prefix_bridge_telemetry_module_binding")
    original_flush = telemetry.flush
    lock = asyncio.Lock()
    bridge_sha = source_sha256()
    poisoned = False

    async def serialized_flush(path):
        nonlocal poisoned
        async with lock:
            if poisoned:
                raise RuntimeError("gateway_prefix_bridge_flush_poisoned")
            pending = asyncio.create_task(original_flush(path))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(pending)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if pending.cancelled():
                        poisoned = True
                        telemetry.write_errors += 1
                        raise
                except Exception:
                    telemetry.write_errors += 1
                    raise
            if cancelled:
                raise asyncio.CancelledError
            return result

    from fastapi import Request
    from fastapi.responses import JSONResponse

    async def flush_prefix(request: Request):
        # This is the original authenticated, single-worker, stock-locked endpoint.
        state = await snapshot(request)
        if not isinstance(state, dict):
            return state
        try:
            body = await request.body()
            payload = strict_json(body) if len(body) <= 1024 else None
            if (not isinstance(payload, dict)
                    or set(payload) != {"expected_run_id", "expected_sequence"}
                    or not isinstance(payload["expected_run_id"], str)
                    or not payload["expected_run_id"]
                    or type(payload["expected_sequence"]) is not int
                    or payload["expected_sequence"] < 0):
                raise ValueError
        except (ValueError, TypeError, UnicodeDecodeError):
            return JSONResponse({"detail": "invalid prefix barrier"}, status_code=400)
        if (state["run_id"] != payload["expected_run_id"]
                or state["sequence"] < payload["expected_sequence"]):
            return JSONResponse({"detail": "prefix barrier epoch or sequence conflict"}, status_code=409)
        if state["complete"] is not True or telemetry.enabled is not True:
            return JSONResponse({"detail": "prefix producer unavailable"}, status_code=503)
        started = time.monotonic()
        try:
            await telemetry.flush(namespace["TELEMETRY_JSONL"])
        except Exception:
            return JSONResponse({"detail": "prefix flush failed"}, status_code=503)
        finished = time.monotonic()
        if telemetry.dropped or telemetry.write_errors:
            return JSONResponse({"detail": "prefix producer lost records"}, status_code=503)
        process = self_process_metadata()
        return {
            "schema": SCHEMA, "source_sha256": bridge_sha,
            "run_id": state["run_id"], "clock_domain": state["clock_domain"],
            "flushed_through_sequence": state["sequence"],
            "source_at": state["sampled_at"],
            "flush_started_at": started, "flush_finished_at": finished,
            "process_identity": process, "sample_finished_at": time.monotonic(),
            "write_errors": telemetry.write_errors, "dropped": telemetry.dropped,
            "scope": "Serialized append completion, not fsync crash durability or native common-cut proof.",
            "native_mpc_permitted": False,
        }

    app.add_api_route(ROUTE, flush_prefix, methods=["POST"])
    telemetry.flush = serialized_flush
    app.state.r24_prefix_bridge = {"source_sha256": bridge_sha, "lock": lock}
    return app.state.r24_prefix_bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", type=Path, required=True)
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args()
    if not args.enabled:
        runpy.run_path(str(args.gateway), run_name="__main__")
        return
    import uvicorn
    original_run = uvicorn.run

    def run(app, *positional, **kwargs):
        install_gateway_prefix_bridge(app, enabled=True)
        uvicorn.run = original_run
        return original_run(app, *positional, **kwargs)

    uvicorn.run = run
    try:
        runpy.run_path(str(args.gateway), run_name="__main__")
    finally:
        uvicorn.run = original_run


if __name__ == "__main__":
    main()
