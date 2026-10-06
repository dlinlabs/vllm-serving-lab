import asyncio
import json
import math
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from telemetry import Telemetry, SSEObserver

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
ADMISSION_MODE = os.getenv("ADMISSION_MODE", "baseline").strip().lower()

# Static request-count admission settings.
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "4"))
MAX_WAITING = int(os.getenv("MAX_WAITING", "2"))
MAX_OUTSTANDING = MAX_IN_FLIGHT + MAX_WAITING

# V1 token-cost model, fitted from the RTX 3090 workload-shape saturation
# anchors: baseline ~= 1x, prefill-heavy ~= 2x, decode-heavy ~= 3x.
COST_INTERCEPT = float(os.getenv("COST_INTERCEPT", "0.259"))
INPUT_TOKEN_COST = float(os.getenv("INPUT_TOKEN_COST", "0.000758"))
OUTPUT_TOKEN_COST = float(os.getenv("OUTPUT_TOKEN_COST", "0.0052083333"))
MAX_ADMITTED_COST = float(os.getenv("MAX_ADMITTED_COST", "64.0"))
MIN_REQUEST_COST = float(os.getenv("MIN_REQUEST_COST", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "128"))

if ADMISSION_MODE not in {"baseline", "protected", "cost_aware"}:
    raise RuntimeError(
        "ADMISSION_MODE must be 'baseline', 'protected', or 'cost_aware'."
    )

if MAX_IN_FLIGHT <= 0:
    raise RuntimeError("MAX_IN_FLIGHT must be greater than zero.")
if MAX_WAITING < 0:
    raise RuntimeError("MAX_WAITING must be non-negative.")
if not math.isfinite(MAX_ADMITTED_COST) or MAX_ADMITTED_COST <= 0:
    raise RuntimeError("MAX_ADMITTED_COST must be finite and greater than zero.")

# The admission model was calibrated using vLLM-reported prompt token counts.
# Reusing the model tokenizer at the gateway keeps the runtime estimate aligned
# with those measurements. A deterministic character heuristic is retained as a
# fallback so the gateway can still start if the tokenizer is unavailable.
tokenizer = None
tokenizer_load_error: Optional[str] = None
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )
except Exception as exc:  # pragma: no cover - environment-dependent fallback
    tokenizer_load_error = f"{type(exc).__name__}: {exc}"

telemetry = Telemetry()
TELEMETRY_JSONL = os.getenv("TELEMETRY_JSONL", "")


@asynccontextmanager
async def lifespan(app):
    stop = asyncio.Event()
    task = None
    if TELEMETRY_JSONL:
        path = Path(TELEMETRY_JSONL)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fail startup if the experiment log cannot be opened.
        with path.open("a", encoding="utf-8"):
            pass
        telemetry.enabled = True
        task = asyncio.create_task(telemetry.run(path, snapshot_metrics, stop))
    try:
        yield
    finally:
        stop.set()
        if task is not None:
            await task
        telemetry.enabled = False


app = FastAPI(title="vLLM Admission Gateway", lifespan=lifespan)
state_lock = asyncio.Lock()
sem = asyncio.Semaphore(MAX_IN_FLIGHT) if ADMISSION_MODE == "protected" else None

state: Dict[str, Any] = {
    "admission_budget": MAX_ADMITTED_COST,
    "budget_version": 0,
    "current_in_flight": 0,
    "current_waiting": 0,
    "accepted_requests": 0,
    "rejected_requests": 0,
    "failed_requests": 0,
    "max_waiting": 0,
    "current_admitted_cost": 0.0,
    "peak_admitted_cost": 0.0,
    "accepted_cost": 0.0,
    "rejected_cost": 0.0,
    "tokenizer_fallback_requests": 0,
}


def _message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            pieces.append(str(message))
            continue

        content = message.get("content", "")
        if isinstance(content, str):
            pieces.append(content)
        else:
            # Multimodal or structured message content is uncommon in this
            # benchmark. JSON serialization gives a deterministic fallback.
            pieces.append(json.dumps(content, sort_keys=True))

    return "\n".join(pieces)


def estimate_input_tokens(payload: Dict[str, Any]) -> tuple[int, bool]:
    messages = payload.get("messages", [])

    if tokenizer is not None and isinstance(messages, list):
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )

            input_ids = tokenized["input_ids"]

            # Transformers may return a flat list, a batch-shaped nested list,
            # or a tensor depending on tokenizer/version/options.
            if hasattr(input_ids, "shape"):
                token_count = int(input_ids.shape[-1])
            elif input_ids and isinstance(input_ids[0], (list, tuple)):
                token_count = len(input_ids[0])
            else:
                token_count = len(input_ids)

            return max(1, token_count), False
        except Exception:
            # Fall through to the deterministic approximation below. The
            # metrics endpoint records how often this path is used.
            pass

    text = _message_text(messages)
    # Conservative generic approximation for English-heavy prompts plus a
    # small chat-template overhead. It is a fallback, not a calibrated model.
    estimated = math.ceil(len(text) / 4) + max(8, 4 * len(messages))
    return max(1, estimated), True


def estimate_request_cost(payload: Dict[str, Any]) -> tuple[float, int, int, bool]:
    input_tokens, used_fallback = estimate_input_tokens(payload)

    max_tokens_value = payload.get(
        "max_tokens",
        payload.get("max_completion_tokens", DEFAULT_MAX_TOKENS),
    )
    try:
        max_output_tokens = max(1, int(max_tokens_value))
    except (TypeError, ValueError):
        max_output_tokens = DEFAULT_MAX_TOKENS

    cost = (
        COST_INTERCEPT
        + INPUT_TOKEN_COST * input_tokens
        + OUTPUT_TOKEN_COST * max_output_tokens
    )
    return max(MIN_REQUEST_COST, cost), input_tokens, max_output_tokens, used_fallback


async def snapshot_metrics() -> Dict[str, Any]:
    async with state_lock:
        metrics = dict(state)

    metrics.update(
        {
            "telemetry_run_id": telemetry.run_id,
            "telemetry_dropped_records": telemetry.dropped,
            "telemetry_write_errors": telemetry.write_errors,
            "admission_mode": ADMISSION_MODE,
            "max_in_flight": MAX_IN_FLIGHT,
            "max_waiting_capacity": MAX_WAITING,
            "max_admitted_cost": metrics["admission_budget"],
            "initial_admission_budget": MAX_ADMITTED_COST,
            "cost_intercept": COST_INTERCEPT,
            "input_token_cost": INPUT_TOKEN_COST,
            "output_token_cost": OUTPUT_TOKEN_COST,
            "tokenizer_model": MODEL_NAME,
            "tokenizer_available": tokenizer is not None,
            "tokenizer_load_error": tokenizer_load_error,
        }
    )
    return metrics


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    budget: Optional[float] = None
    budget_version: Optional[int] = None


async def begin_request(
    request_cost: float, used_tokenizer_fallback: bool
) -> AdmissionDecision:
    if ADMISSION_MODE == "baseline":
        async with state_lock:
            state["accepted_requests"] += 1
        return AdmissionDecision(True)

    if ADMISSION_MODE == "cost_aware":
        async with state_lock:
            if used_tokenizer_fallback:
                state["tokenizer_fallback_requests"] += 1

            projected_cost = state["current_admitted_cost"] + request_cost
            budget = state["admission_budget"]
            version = state["budget_version"]
            if projected_cost > budget:
                state["rejected_requests"] += 1
                state["rejected_cost"] += request_cost
                return AdmissionDecision(False, budget, version)

            state["accepted_requests"] += 1
            state["accepted_cost"] += request_cost
            state["current_in_flight"] += 1
            state["current_admitted_cost"] = projected_cost
            state["peak_admitted_cost"] = max(
                state["peak_admitted_cost"],
                projected_cost,
            )
        return AdmissionDecision(True, budget, version)

    async with state_lock:
        outstanding = state["current_in_flight"] + state["current_waiting"]
        if outstanding >= MAX_OUTSTANDING:
            state["rejected_requests"] += 1
            return AdmissionDecision(False)

        state["accepted_requests"] += 1
        state["current_waiting"] += 1
        state["max_waiting"] = max(state["max_waiting"], state["current_waiting"])

    semaphore_acquired = False
    try:
        await sem.acquire()
        semaphore_acquired = True
        async with state_lock:
            state["current_waiting"] -= 1
            state["current_in_flight"] += 1
        return AdmissionDecision(True)
    except BaseException:
        if semaphore_acquired:
            sem.release()
        async with state_lock:
            state["current_waiting"] = max(0, state["current_waiting"] - 1)
        raise


async def finish_request(request_cost: float) -> None:
    if ADMISSION_MODE == "baseline":
        return

    if ADMISSION_MODE == "cost_aware":
        async with state_lock:
            state["current_in_flight"] = max(0, state["current_in_flight"] - 1)
            state["current_admitted_cost"] = max(
                0.0,
                state["current_admitted_cost"] - request_cost,
            )
        return

    async with state_lock:
        state["current_in_flight"] = max(0, state["current_in_flight"] - 1)

    sem.release()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    request_id = telemetry.received()
    reserved = False
    cleaned = False
    request_cost = 0.0
    client = None
    upstream = None

    async def cleanup(outcome):
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        try:
            if upstream is not None:
                await upstream.aclose()
        finally:
            try:
                if client is not None:
                    await client.aclose()
            finally:
                if reserved:
                    await finish_request(request_cost)
                telemetry.terminal(request_id, outcome)
                if outcome == "failed":
                    async with state_lock:
                        state["failed_requests"] += 1

    async def safe_cleanup(outcome):
        # Cleanup survives cancellation of the downstream response task.
        task = asyncio.create_task(cleanup(outcome))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    try:
        body = await request.body()
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "x-admin-api-key"}
        }
        headers["x-request-id"] = request_id
        headers["accept-encoding"] = "identity"
        try:
            parsed = json.loads(body)
            payload = parsed if isinstance(parsed, dict) else {}
        except (ValueError, UnicodeDecodeError):
            payload = {}

        estimated_input_tokens = estimated_output_tokens = 0
        used_tokenizer_fallback = False
        if ADMISSION_MODE == "cost_aware":
            (request_cost, estimated_input_tokens, estimated_output_tokens,
             used_tokenizer_fallback) = estimate_request_cost(payload)

        decision = await begin_request(request_cost, used_tokenizer_fallback)
        if not decision.accepted:
            telemetry.terminal(request_id, "rejected", budget=decision.budget,
                               budget_version=decision.budget_version)
            return JSONResponse(
                {"detail": "request rejected: admission capacity exceeded",
                 "estimated_request_cost": request_cost,
                 "estimated_input_tokens": estimated_input_tokens,
                 "estimated_output_tokens": estimated_output_tokens,
                 "max_admitted_cost": decision.budget,
                 "budget_version": decision.budget_version},
                status_code=503, headers={"x-request-id": request_id},
            )
        reserved = True
        telemetry.admitted(request_id, request_cost=request_cost,
                           estimated_input_tokens=estimated_input_tokens,
                           max_output_tokens=estimated_output_tokens,
                           budget=decision.budget, budget_version=decision.budget_version)
        client = httpx.AsyncClient()
        upstream_request = client.build_request(
            method=request.method, url=f"{VLLM_URL}/v1/chat/completions",
            content=body, headers=headers, timeout=60.0,
        )
        telemetry.emit("backend_sent", request_id)
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        await safe_cleanup("failed")
        return JSONResponse({"detail": "upstream request failed"}, status_code=502,
                            headers={"x-request-id": request_id})
    except BaseException as exc:
        await safe_cleanup("cancelled" if isinstance(exc, asyncio.CancelledError) else "failed")
        raise

    async def stream_upstream():
        observer = SSEObserver()
        is_sse = "text/event-stream" in upstream.headers.get("content-type", "")
        streaming_expected = bool(payload.get("stream")) or is_sse
        outcome = "cancelled"
        try:
            async for chunk in upstream.aiter_bytes():
                if is_sse and upstream.is_success:
                    observer.feed(chunk)
                    if observer.has_content:
                        telemetry.first_content(request_id)
                yield chunk
            complete = upstream.is_success and (
                not streaming_expected or (observer.done and not observer.invalid)
            )
            outcome = "completed" if complete else "failed"
        except Exception:
            outcome = "failed"
            raise
        finally:
            await safe_cleanup(outcome)

    return StreamingResponse(
        stream_upstream(), status_code=upstream.status_code,
        headers={**{key: value for key, value in upstream.headers.items()
                    if key.lower() not in {"content-length", "transfer-encoding", "content-encoding", "x-request-id"}},
                 "x-request-id": request_id},
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(safe_cleanup, "cancelled"),
    )


@app.post("/admin/budget")
async def update_budget(request: Request):
    # This state belongs to one process. Run the experiment with one worker.
    if not ADMIN_API_KEY:
        return JSONResponse({"detail": "budget updates disabled"}, status_code=503)
    supplied_key = request.headers.get("x-admin-api-key", "")
    if not secrets.compare_digest(supplied_key.encode(), ADMIN_API_KEY.encode()):
        return JSONResponse({"detail": "invalid admin API key"}, status_code=401)
    if ADMISSION_MODE != "cost_aware":
        return JSONResponse(
            {"detail": "budget updates require cost_aware mode"}, status_code=409
        )

    try:
        payload = await request.json()
        value = payload.get("budget") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("budget must be a number")
        budget = float(value)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("budget must be finite and greater than zero")
    except (ValueError, OverflowError, UnicodeDecodeError):
        return JSONResponse(
            {"detail": "budget must be a finite number greater than zero"},
            status_code=422,
        )

    async with state_lock:
        previous_budget = state["admission_budget"]
        state["admission_budget"] = budget
        state["budget_version"] += 1
        result = {
            "previous_budget": previous_budget,
            "budget": budget,
            "current_admitted_cost": state["current_admitted_cost"],
            "budget_version": state["budget_version"],
        }
    telemetry.emit("budget_updated", **result)
    # Existing requests retain their reservations, even after a budget reduction.
    return result


@app.get("/metrics")
async def metrics():
    return await snapshot_metrics()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
