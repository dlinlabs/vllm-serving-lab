import asyncio
import os
from typing import Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_URL = "http://localhost:8000"
ADMISSION_MODE = os.getenv("ADMISSION_MODE", "baseline").strip().lower()
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "4"))
MAX_WAITING = int(os.getenv("MAX_WAITING", "2"))
MAX_OUTSTANDING = MAX_IN_FLIGHT + MAX_WAITING

if ADMISSION_MODE not in {"baseline", "protected"}:
    raise RuntimeError("ADMISSION_MODE must be 'baseline' or 'protected'.")

app = FastAPI(title="vLLM Admission Gateway")
state_lock = asyncio.Lock()
sem = asyncio.Semaphore(MAX_IN_FLIGHT) if ADMISSION_MODE == "protected" else None

state: Dict[str, int] = {
    "current_in_flight": 0,
    "current_waiting": 0,
    "accepted_requests": 0,
    "rejected_requests": 0,
    "failed_requests": 0,
    "max_waiting": 0,
}


async def snapshot_metrics() -> Dict[str, int]:
    await state_lock.acquire()
    try:
        return dict(state)
    finally:
        state_lock.release()


async def begin_request() -> bool:
    if ADMISSION_MODE == "baseline":
        await state_lock.acquire()
        try:
            state["accepted_requests"] += 1
        finally:
            state_lock.release()
        return True

    await state_lock.acquire()
    try:
        outstanding = state["current_in_flight"] + state["current_waiting"]
        if outstanding >= MAX_OUTSTANDING:
            state["rejected_requests"] += 1
            return False

        state["accepted_requests"] += 1
        state["current_waiting"] += 1
        state["max_waiting"] = max(state["max_waiting"], state["current_waiting"])
    finally:
        state_lock.release()

    semaphore_acquired = False
    try:
        await sem.acquire()
        semaphore_acquired = True
        await state_lock.acquire()
        try:
            state["current_waiting"] -= 1
            state["current_in_flight"] += 1
        finally:
            state_lock.release()
        return True
    except BaseException:
        if semaphore_acquired:
            sem.release()
        await state_lock.acquire()
        try:
            state["current_waiting"] = max(0, state["current_waiting"] - 1)
        finally:
            state_lock.release()
        raise


async def finish_request() -> None:
    if ADMISSION_MODE != "protected":
        return

    await state_lock.acquire()
    try:
        state["current_in_flight"] = max(0, state["current_in_flight"] - 1)
    finally:
        state_lock.release()

    sem.release()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    if not await begin_request():
        return JSONResponse(
            {"detail": "request rejected: admission capacity exceeded"},
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
        await state_lock.acquire()
        try:
            state["failed_requests"] += 1
        finally:
            state_lock.release()
        return JSONResponse({"detail": "upstream request failed"}, status_code=502)
    finally:
        if upstream is None:
            try:
                await client.aclose()
            finally:
                await finish_request()

    async def stream_upstream():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            await state_lock.acquire()
            try:
                state["failed_requests"] += 1
            finally:
                state_lock.release()
            raise
        finally:
            try:
                await upstream.aclose()
            finally:
                try:
                    await client.aclose()
                finally:
                    await finish_request()

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
