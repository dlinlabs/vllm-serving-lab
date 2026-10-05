import asyncio
import json
import math
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
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
if MAX_ADMITTED_COST <= 0:
    raise RuntimeError("MAX_ADMITTED_COST must be greater than zero.")

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

app = FastAPI(title="vLLM Admission Gateway")
state_lock = asyncio.Lock()
sem = asyncio.Semaphore(MAX_IN_FLIGHT) if ADMISSION_MODE == "protected" else None

state: Dict[str, Any] = {
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
            "admission_mode": ADMISSION_MODE,
            "max_in_flight": MAX_IN_FLIGHT,
            "max_waiting_capacity": MAX_WAITING,
            "max_admitted_cost": MAX_ADMITTED_COST,
            "cost_intercept": COST_INTERCEPT,
            "input_token_cost": INPUT_TOKEN_COST,
            "output_token_cost": OUTPUT_TOKEN_COST,
            "tokenizer_model": MODEL_NAME,
            "tokenizer_available": tokenizer is not None,
            "tokenizer_load_error": tokenizer_load_error,
        }
    )
    return metrics


async def begin_request(request_cost: float, used_tokenizer_fallback: bool) -> bool:
    if ADMISSION_MODE == "baseline":
        async with state_lock:
            state["accepted_requests"] += 1
        return True

    if ADMISSION_MODE == "cost_aware":
        async with state_lock:
            if used_tokenizer_fallback:
                state["tokenizer_fallback_requests"] += 1

            projected_cost = state["current_admitted_cost"] + request_cost
            if projected_cost > MAX_ADMITTED_COST:
                state["rejected_requests"] += 1
                state["rejected_cost"] += request_cost
                return False

            state["accepted_requests"] += 1
            state["accepted_cost"] += request_cost
            state["current_in_flight"] += 1
            state["current_admitted_cost"] = projected_cost
            state["peak_admitted_cost"] = max(
                state["peak_admitted_cost"],
                projected_cost,
            )
        return True

    async with state_lock:
        outstanding = state["current_in_flight"] + state["current_waiting"]
        if outstanding >= MAX_OUTSTANDING:
            state["rejected_requests"] += 1
            return False

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
        return True
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
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    payload: Dict[str, Any] = {}
    if ADMISSION_MODE == "cost_aware":
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                payload = parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}

    request_cost = 0.0
    estimated_input_tokens = 0
    estimated_output_tokens = 0
    used_tokenizer_fallback = False
    if ADMISSION_MODE == "cost_aware":
        (
            request_cost,
            estimated_input_tokens,
            estimated_output_tokens,
            used_tokenizer_fallback,
        ) = estimate_request_cost(payload)

    if not await begin_request(request_cost, used_tokenizer_fallback):
        return JSONResponse(
            {
                "detail": "request rejected: admission capacity exceeded",
                "estimated_request_cost": request_cost,
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "max_admitted_cost": MAX_ADMITTED_COST,
            },
            status_code=503,
        )

    client = httpx.AsyncClient()
    upstream = None
    try:
        upstream_request = client.build_request(
            method=request.method,
            url=f"{VLLM_URL}/v1/chat/completions",
            content=body,
            headers=headers,
            timeout=60.0,
        )
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        async with state_lock:
            state["failed_requests"] += 1
        return JSONResponse({"detail": "upstream request failed"}, status_code=502)
    finally:
        if upstream is None:
            try:
                await client.aclose()
            finally:
                await finish_request(request_cost)

    async def stream_upstream():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            async with state_lock:
                state["failed_requests"] += 1
            raise
        finally:
            try:
                await upstream.aclose()
            finally:
                try:
                    await client.aclose()
                finally:
                    await finish_request(request_cost)

    return StreamingResponse(
        stream_upstream(),
        status_code=upstream.status_code,
        headers={
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding"}
        },
        media_type=upstream.headers.get("content-type"),
    )


@app.get("/metrics")
async def metrics():
    return await snapshot_metrics()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
