import asyncio
import json
import math
import os
import secrets
import time
import re
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from http.cookiejar import CookieJar
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from telemetry import Telemetry, SSEObserver, GatewayDiagnostics, DIAGNOSTICS_SCHEMA

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
MPC_LAB_ENABLED = os.getenv("MPC_LAB_ENABLED", "false").strip().lower() in {"true", "1"}

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
BENCHMARK_REQUEST_IDS = os.getenv("BENCHMARK_REQUEST_IDS", "0") == "1"
LOOP_LAG_TELEMETRY = os.getenv("LOOP_LAG_TELEMETRY", "0") == "1"
TELEMETRY_STATUS_JSON = os.getenv('TELEMETRY_STATUS_JSON', '')
GATEWAY_DIAGNOSTICS = os.getenv('GATEWAY_DIAGNOSTICS', '0') == '1'
SHARED_UPSTREAM_CLIENT = os.getenv('SHARED_UPSTREAM_CLIENT', '0') == '1'
PREFILL_RAW_COMPLETIONS = os.getenv('PREFILL_LAB_RAW_COMPLETIONS', '0') == '1'
if PREFILL_RAW_COMPLETIONS and (not BENCHMARK_REQUEST_IDS or not SHARED_UPSTREAM_CLIENT or ADMISSION_MODE!='protected'):
    raise RuntimeError('Raw prefill bridge requires benchmark IDs, shared client and fixed protected admission')
NO_PHASE = nullcontext()
benchmark_ids = set()


class NoUpstreamCookies(CookieJar):
    """A shared proxy transport must not carry response cookies between users."""
    def set_cookie(self, cookie):
        pass


def create_upstream_client():
    # Admission remains the concurrency bound; do not add a hidden connection-pool queue.
    return httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                             cookies=NoUpstreamCookies(), trust_env=False)


@asynccontextmanager
async def lifespan(app):
    stop = asyncio.Event()
    task = None
    lag_task = None
    started = False
    shared_client = None
    if GATEWAY_DIAGNOSTICS and not (TELEMETRY_JSONL and TELEMETRY_STATUS_JSON):
        raise RuntimeError('Gateway diagnostics require telemetry log and shutdown status paths')
    try:
        if SHARED_UPSTREAM_CLIENT:
            shared_client = create_upstream_client()
            app.state.upstream_client = shared_client
        if TELEMETRY_JSONL:
            path = Path(TELEMETRY_JSONL)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Fail startup if the experiment log cannot be opened.
            with path.open("a", encoding="utf-8"):
                pass
            telemetry.enabled = True
            if GATEWAY_DIAGNOSTICS:
                telemetry.diagnostics = GatewayDiagnostics(telemetry.emit)
                telemetry.diagnostics.start()
            task = asyncio.create_task(telemetry.run(path, snapshot_metrics, stop))
            if LOOP_LAG_TELEMETRY:
                lag_task = asyncio.create_task(telemetry.monitor_loop(stop))
        started = True
        yield
    finally:
        stop.set()
        results = await asyncio.gather(*(t for t in (lag_task, task) if t is not None),
                                       *([shared_client.aclose()] if shared_client is not None else []),
                                       return_exceptions=True)
        if shared_client is not None:
            app.state.upstream_client = None
        if telemetry.diagnostics is not None:
            async def final_drain():
                await telemetry.diagnostics.stop()
                telemetry.sample(await snapshot_metrics())
                await telemetry.flush(TELEMETRY_JSONL)
            results.extend(await asyncio.gather(final_drain(), return_exceptions=True))
        errors = [result for result in results if isinstance(result, BaseException)]
        telemetry.enabled = False
        if TELEMETRY_STATUS_JSON:
            failed = (not started or errors or telemetry.dropped or telemetry.write_errors or
                      (telemetry.diagnostics is not None and not telemetry.diagnostics.health()['complete']))
            status = dict(run_id=telemetry.run_id, status='failed' if failed else 'closed', closed_at=time.monotonic(),
                          dropped=telemetry.dropped, write_errors=telemetry.write_errors)
            if errors:
                status['error_types'] = [type(error).__name__ for error in errors]
            if telemetry.diagnostics is not None:
                status.update(diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                              diagnostics=telemetry.diagnostics.health())
            def persist_status():
                path = Path(TELEMETRY_STATUS_JSON)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(status, allow_nan=False) + '\n')
                temporary.replace(path)
            await asyncio.to_thread(persist_status)
        if errors:
            raise errors[0]
        if TELEMETRY_JSONL and (telemetry.dropped or telemetry.write_errors):
            raise RuntimeError('Experiment telemetry was lost or could not be persisted')
        if telemetry.diagnostics is not None and not telemetry.diagnostics.health()['complete']:
            raise RuntimeError('Gateway diagnostic callback coverage is incomplete')


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

mpc_requests = {}
mpc_operations = {}
mpc_sequence = 0
MPC_OPERATION_LIMIT = 1024
if MPC_LAB_ENABLED:
    from mpc_lab.contracts import (
        SCHEMA, GUARDED_CAPABILITY, CostSpec, Identity, RequestMark, Snapshot,
        digest, implementation_hash,
    )
    from mpc_lab.runtime import snapshot_to_dict

    from prefill_lab.live_http import clock_domain as mpc_local_clock_domain

    mpc_clock_domain = mpc_local_clock_domain()
    mpc_cost_spec = CostSpec(
        intercept=COST_INTERCEPT,
        input_per_token=INPUT_TOKEN_COST,
        output_per_token=OUTPUT_TOKEN_COST,
        minimum=MIN_REQUEST_COST,
        default_max_tokens=DEFAULT_MAX_TOKENS,
        tokenizer_identity=(
            os.getenv("MPC_TOKENIZER_IDENTITY", "unqualified")
            + f";model={MODEL_NAME};available={tokenizer is not None};fallback=chars4-chat-v1"
        ),
    )
    mpc_identity = Identity(
        os.getenv("MPC_MODEL_VERSION", "unqualified"),
        os.getenv("MPC_CALIBRATION_VERSION", "unqualified"),
        os.getenv("MPC_SUPPORT_HASH", "unqualified"),
        mpc_cost_spec.fingerprint,
        implementation_hash(),
    )


def mpc_now():
    return time.monotonic()


def mpc_changed():
    global mpc_sequence
    mpc_sequence += 1
    if MPC_LAB_ENABLED and telemetry.enabled:
        marks = tuple(mpc_requests.values())
        reserved = math.fsum(mark.cost for mark in marks)
        telemetry.emit(
            "mpc_ledger", clock_domain=mpc_clock_domain, identity=asdict(mpc_identity),
            sequence=mpc_sequence, effective_at=mpc_now(), reserved_cost=reserved,
            running=len(marks),
            complete=(
                len(marks) == state["current_in_flight"]
                and math.isclose(reserved, state["current_admitted_cost"],
                                 rel_tol=1e-12, abs_tol=1e-9)
            ),
            budget=state["admission_budget"], budget_version=state["budget_version"],
        )


async def mpc_first_content(request_id):
    observed_at = mpc_now()
    async with state_lock:
        mark = mpc_requests.get(request_id)
        if mark is not None and mark.first_at is None:
            mpc_requests[request_id] = replace(mark, first_at=observed_at)
            mpc_changed()


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
        metrics['state_sampled_at'] = time.monotonic()
        metrics.update(telemetry.observed_state(metrics['state_sampled_at']))

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
            "gateway_diagnostics": telemetry.diagnostics is not None,
        }
    )
    if telemetry.diagnostics is not None:
        metrics['diagnostics'] = dict(gc=telemetry.diagnostics.gc_state(),
                                     health=telemetry.diagnostics.health(), sampled_at=time.monotonic())
    if PREFILL_RAW_COMPLETIONS:metrics['prefill_raw_completions']=True
    if SHARED_UPSTREAM_CLIENT:
        metrics['shared_upstream_client'] = True
    return metrics


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    budget: Optional[float] = None
    budget_version: Optional[int] = None


async def begin_request(
    request_cost: float, used_tokenizer_fallback: bool, phases=None, *, mpc_mark=None
) -> AdmissionDecision:
    if phases is not None:
        phases.begin('admission_lock_wait')
    if ADMISSION_MODE == "baseline":
        async with state_lock:
            if phases is not None:
                phases.end('admission_lock_wait')
            state["accepted_requests"] += 1
        return AdmissionDecision(True)

    if ADMISSION_MODE == "cost_aware":
        async with state_lock:
            if phases is not None:
                phases.end('admission_lock_wait')
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
            if MPC_LAB_ENABLED:
                if mpc_mark is not None:
                    identifier, received_at, prompt_tokens, max_tokens = mpc_mark
                    mpc_requests[identifier] = RequestMark(
                        identifier, request_cost, prompt_tokens, max_tokens,
                        received_at, mpc_now(),
                    )
                mpc_changed()
        return AdmissionDecision(True, budget, version)

    async with state_lock:
        if phases is not None:
            phases.end('admission_lock_wait')
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


async def finish_request(request_cost: float, *, request_id=None) -> None:
    if ADMISSION_MODE == "baseline":
        return

    if ADMISSION_MODE == "cost_aware":
        async with state_lock:
            state["current_in_flight"] = max(0, state["current_in_flight"] - 1)
            state["current_admitted_cost"] = max(
                0.0,
                state["current_admitted_cost"] - request_cost,
            )
            if MPC_LAB_ENABLED:
                mpc_requests.pop(request_id, None)
                mpc_changed()
        return

    async with state_lock:
        state["current_in_flight"] = max(0, state["current_in_flight"] - 1)

    sem.release()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    return await _proxy_completions(request, '/v1/chat/completions')


async def proxy_raw_completions(request: Request):
    if not PREFILL_RAW_COMPLETIONS:return JSONResponse({'detail':'not found'},status_code=404)
    try:
        payload=json.loads(await request.body());ids=payload['prompt'];extra=payload['vllm_xargs']
        identifier=request.headers.get('x-benchmark-request-id','')
        if (set(payload)-{'model','prompt','max_tokens','temperature','ignore_eos','stream','stream_options','request_id','vllm_xargs'}
                or not re.fullmatch('[0-9a-f]{32}',identifier) or payload['request_id']!=identifier
                or not isinstance(ids,list) or not ids or any(type(x) is not int or x<0 for x in ids)
                or payload['max_tokens'] not in (128,256) or len(ids)+payload['max_tokens']>8192
                or payload['ignore_eos'] is not True or payload['stream'] is not True
                or type(payload['temperature']) not in (int,float) or payload['temperature']!=0
                or extra['prefill_lab_trace_id']!=identifier):raise ValueError('Invalid frozen raw request')
    except (ValueError,KeyError,TypeError):return JSONResponse({'detail':'invalid benchmark raw request'},status_code=422)
    return await _proxy_completions(request, '/v1/completions')


if PREFILL_RAW_COMPLETIONS:app.post('/v1/completions')(proxy_raw_completions)


async def _proxy_completions(request, upstream_path):
    request_id = None
    if BENCHMARK_REQUEST_IDS:
        request_id = request.headers.get('x-benchmark-request-id')
        if request_id is not None:
            if not re.fullmatch(r'[0-9a-f]{32}', request_id):
                return JSONResponse({'detail': 'invalid benchmark request ID'}, status_code=400)
            if request_id in benchmark_ids:
                return JSONResponse({'detail': 'duplicate benchmark request ID'}, status_code=409)
            # Finite experiment-only ID registry; refuse instead of evicting IDs.
            if len(benchmark_ids) >= 100000:
                return JSONResponse({'detail': 'benchmark ID capacity exceeded'}, status_code=503)
            benchmark_ids.add(request_id)
    request_id = telemetry.received(request_id)
    mpc_received_at = mpc_now() if MPC_LAB_ENABLED else None
    if upstream_path=='/v1/completions':
        from prefill_lab.live_http import clock_domain
        telemetry.emit('bridge_clock',request_id,clock_domain=clock_domain())
    phases = telemetry.diagnostics.request(request_id) if telemetry.diagnostics is not None else None
    reserved = False
    cleaned = False
    request_cost = 0.0
    client = None
    upstream = None
    cleanup_task = None

    async def cleanup(outcome):
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        terminal_recorded = False
        try:
            try:
                if upstream is not None:
                    with phases.phase('upstream_close') if phases else NO_PHASE:
                        await upstream.aclose()
            finally:
                try:
                    if client is not None and not SHARED_UPSTREAM_CLIENT:
                        with phases.phase('client_close') if phases else NO_PHASE:
                            await client.aclose()
                finally:
                    if reserved:
                        with phases.phase('reservation_release') if phases else NO_PHASE:
                            if MPC_LAB_ENABLED:
                                await finish_request(request_cost, request_id=request_id)
                            else:
                                await finish_request(request_cost)
                    telemetry.terminal(request_id, outcome)
                    terminal_recorded = True
                    if outcome == "failed":
                        async with state_lock:
                            state["failed_requests"] += 1
        finally:
            if phases is not None:
                phases.finish(outcome, cleanup_complete=terminal_recorded and
                              all(phases.phases.get(name, {}).get('status', 'not_reached') in
                                  {'completed', 'not_reached'} for name in
                                  ('upstream_close', 'client_close', 'reservation_release')))

    async def safe_cleanup(outcome):
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(cleanup(outcome))
        cancellation = None
        # Repeated downstream cancellation must not cancel the one cleanup task.
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                cancellation = exc
        cleanup_task.result()
        if cancellation is not None:
            raise cancellation

    try:
        with phases.phase('body_read') if phases else NO_PHASE:
            body = await request.body()
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "x-admin-api-key", "x-benchmark-request-id"}
        }
        headers["x-request-id"] = request_id
        headers["accept-encoding"] = "identity"
        try:
            with phases.phase('json_parse') if phases else NO_PHASE:
                parsed = json.loads(body)
                payload = parsed if isinstance(parsed, dict) else {}
        except (ValueError, UnicodeDecodeError):
            payload = {}

        estimated_input_tokens = estimated_output_tokens = 0
        used_tokenizer_fallback = False
        if ADMISSION_MODE == "cost_aware":
            with phases.phase('cost_estimation') if phases else NO_PHASE:
                (request_cost, estimated_input_tokens, estimated_output_tokens,
                 used_tokenizer_fallback) = estimate_request_cost(payload)

        if MPC_LAB_ENABLED:
            decision = await begin_request(
                request_cost, used_tokenizer_fallback, phases,
                mpc_mark=(request_id, mpc_received_at, estimated_input_tokens, estimated_output_tokens),
            )
        else:
            decision = await begin_request(request_cost, used_tokenizer_fallback, phases)
        if not decision.accepted:
            telemetry.terminal(request_id, "rejected", budget=decision.budget,
                               budget_version=decision.budget_version,
                               estimated_input_tokens=estimated_input_tokens,
                               max_output_tokens=estimated_output_tokens,
                               request_cost=request_cost)
            if phases is not None:
                phases.finish("rejected", cleanup_complete=True)
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
        if SHARED_UPSTREAM_CLIENT:
            client = getattr(app.state, 'upstream_client', None)
            if client is None or client.is_closed:
                raise RuntimeError('Shared upstream client requires an active application lifespan')
        else:
            with phases.phase('client_construction') if phases else NO_PHASE:
                client = httpx.AsyncClient()
        upstream_request = client.build_request(
            method=request.method, url=f"{VLLM_URL}{upstream_path}",
            content=body, headers=headers, timeout=60.0,
        )
        with phases.phase('http_send_headers') if phases else NO_PHASE:
            telemetry.emit("backend_sent", request_id)
            upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        if phases is not None:
            phases.abort(exc)
        await safe_cleanup("failed")
        return JSONResponse({"detail": "upstream request failed"}, status_code=502,
                            headers={"x-request-id": request_id})
    except BaseException as exc:
        if phases is not None:
            phases.abort(exc)
        await safe_cleanup("cancelled" if isinstance(exc, asyncio.CancelledError) else "failed")
        raise

    async def stream_upstream():
        observer = SSEObserver(completions=upstream_path=='/v1/completions')
        mpc_first_marked = False
        is_sse = "text/event-stream" in upstream.headers.get("content-type", "")
        streaming_expected = bool(payload.get("stream")) or is_sse
        outcome = "cancelled"
        try:
            async for chunk in upstream.aiter_bytes():
                if phases is not None and chunk:
                    phases.mark('first_body')
                if is_sse and upstream.is_success:
                    observer.feed(chunk)
                    if observer.has_content:
                        if MPC_LAB_ENABLED and not mpc_first_marked:
                            await mpc_first_content(request_id)
                            mpc_first_marked = True
                        if phases is not None:
                            phases.mark('first_content')
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
        if MPC_LAB_ENABLED:
            mpc_changed()
        applied_at = time.monotonic()
        result = {
            "previous_budget": previous_budget,
            "budget": budget,
            "current_admitted_cost": state["current_admitted_cost"],
            "budget_version": state["budget_version"],
            "applied_at": applied_at,
        }
    telemetry.emit("budget_updated", **result)
    # Existing requests retain their reservations, even after a budget reduction.
    return result


def mpc_access(request):
    if not MPC_LAB_ENABLED:
        return JSONResponse({"detail": "not found"}, status_code=404)
    if not ADMIN_API_KEY:
        return JSONResponse({"detail": "MPC admin disabled"}, status_code=503)
    supplied = request.headers.get("x-admin-api-key", "")
    if not secrets.compare_digest(supplied.encode(), ADMIN_API_KEY.encode()):
        return JSONResponse({"detail": "invalid admin API key"}, status_code=401)
    if ADMISSION_MODE != "cost_aware":
        return JSONResponse({"detail": "MPC requires cost_aware mode"}, status_code=409)
    if any(os.getenv(name, "1") != "1" for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS")):
        return JSONResponse({"detail": "MPC requires a single worker"}, status_code=409)
    return None


@app.get("/admin/mpc/state")
async def mpc_snapshot(request: Request):
    if (failure := mpc_access(request)) is not None:
        return failure
    async with state_lock:
        marks = tuple(mpc_requests.values())
        complete = (
            len(marks) == state["current_in_flight"]
            and math.isclose(
                math.fsum(mark.cost for mark in marks),
                state["current_admitted_cost"], rel_tol=1e-12, abs_tol=1e-9,
            )
        )
        snapshot = Snapshot(
            telemetry.run_id, mpc_clock_domain, mpc_now(),
            state["admission_budget"], state["budget_version"],
            marks, mpc_identity, mpc_sequence, complete,
        )
        result = snapshot_to_dict(snapshot)
        result["capabilities"] = [GUARDED_CAPABILITY]
        result["cost_spec"] = asdict(mpc_cost_spec)
        return result


@app.post("/admin/mpc/budget")
async def mpc_update_budget(request: Request):
    if (failure := mpc_access(request)) is not None:
        return failure
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or set(payload) != {
            "schema", "run_id", "clock_domain", "identity", "expected_version",
            "not_after", "operation_id", "budget",
        }:
            raise ValueError("command_fields")
        if (
            any(not isinstance(payload[key], str) or not 0 < len(payload[key]) <= 1024
                for key in ("schema", "run_id", "clock_domain", "operation_id"))
            or type(payload["expected_version"]) is not int
            or payload["expected_version"] < 0
            or type(payload["budget"]) not in (int, float)
            or not math.isfinite(payload["budget"])
            or payload["budget"] <= 0
            or type(payload["not_after"]) not in (int, float)
            or not math.isfinite(payload["not_after"])
            or not isinstance(payload["identity"], dict)
        ):
            raise ValueError("command_values")
        command_hash = digest(payload)
    except (ValueError, TypeError, OverflowError, UnicodeDecodeError):
        return JSONResponse({"detail": "invalid guarded command"}, status_code=422)

    async with state_lock:
        # Identity precedes any comparison of server-clock uptime.
        if (
            payload["schema"] != SCHEMA
            or payload["run_id"] != telemetry.run_id
            or payload["clock_domain"] != mpc_clock_domain
            or payload["identity"] != asdict(mpc_identity)
        ):
            return JSONResponse({"detail": "command identity mismatch"}, status_code=409)
        previous = mpc_operations.get(payload["operation_id"])
        if previous is not None:
            if previous[0] != command_hash:
                return JSONResponse({"detail": "operation ID reused"}, status_code=409)
            return dict(previous[1])
        if payload["expected_version"] != state["budget_version"]:
            return JSONResponse({"detail": "stale budget version"}, status_code=409)
        # Refuse new operations when full; do not silently forget idempotency.
        if len(mpc_operations) >= MPC_OPERATION_LIMIT:
            return JSONResponse({"detail": "operation outcome capacity"}, status_code=503)
        applied_at = mpc_now()
        if applied_at >= payload["not_after"]:
            return JSONResponse({"detail": "command expired"}, status_code=409)
        result = {
            "schema": SCHEMA, "run_id": telemetry.run_id, "clock_domain": mpc_clock_domain,
            "operation_id": payload["operation_id"], "status": "applied",
            "previous_budget": state["admission_budget"], "budget": payload["budget"],
            "budget_version": state["budget_version"] + 1, "applied_at": applied_at,
        }
        state["admission_budget"] = payload["budget"]
        state["budget_version"] += 1
        mpc_changed()
        mpc_operations[payload["operation_id"]] = (command_hash, dict(result))
    telemetry.emit(
        "budget_updated", operation_id=result["operation_id"],
        previous_budget=result["previous_budget"], budget=result["budget"],
        budget_version=result["budget_version"], applied_at=result["applied_at"],
    )
    return result


@app.get("/metrics")
async def metrics():
    return await snapshot_metrics()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
