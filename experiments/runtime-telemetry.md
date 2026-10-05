# Runtime telemetry for budget step experiments

The gateway can observe how requests respond to live budget changes. This adds
instrumentation, not an MPC controller or an automatic step experiment runner.

## Start and collect

Use one gateway worker and start vLLM as usual. From the repository root:

```bash
export ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
ADMISSION_MODE=cost_aware MAX_ADMITTED_COST=80 \
  TELEMETRY_JSONL=results/budget-pilot/gateway.jsonl python gateway.py
```

In another terminal, run the existing client:

```bash
python sustained_load.py --rps 30 --duration 150 \
  --json-out results/budget-pilot/client.json
```

In a terminal with the same `ADMIN_API_KEY`, change the budget:

```bash
curl -X POST http://localhost:8080/admin/budget \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' -d '{"budget":96}'
```

The 150-second command is a collection example, not an automated or warmed-up
80/96/112/96/80 experiment. Warmup, step scheduling, and draining are the next step.
Without `TELEMETRY_JSONL`, request state is still tracked but no file is written.
The file is append-only; use a new path per experiment or group by `run_id`.

## Event records

Every row has `event`, `run_id`, wall-clock `timestamp` (Unix seconds), and
`monotonic` (seconds on this gateway host). Use monotonic differences for durations;
do not compare monotonic clocks across machines. Request events have `request_id`:

- `received`: gateway handler entry, before reading the request body.
- `accepted`: admission granted, with request cost, estimated input tokens,
  requested output cap, and the decision's budget/version. Token estimates are
  available in cost-aware mode; other modes currently report zero.
- `rejected`: terminal admission rejection, with decision budget/version.
- `backend_sent`: just before the HTTP client sends to vLLM; not proof that vLLM
  has begun GPU work.
- `first_content`: first nonempty SSE delta content observed by the gateway.
  `ttft` is measured from gateway receipt, so it differs from client TTFT.
- `completed`, `failed`, `cancelled`: terminal outcome and `end_to_end` duration.
- `budget_updated`: successful control update, with old/new budget and version.

The gateway generates unique IDs, sends them to the upstream as `X-Request-ID`,
and returns them to the client in that header, including rejection/error responses.
`sustained_load.py` saves this header as `request_id` in its request results.
Prompts, generated text, and API keys are not written to telemetry.

For streaming responses, completion requires successful HTTP status, a `[DONE]`
SSE event, and no observed malformed SSE JSON. A truncated 200 stream counts as
failed. Non-streaming successful HTTP responses count as completed and do not emit
`first_content`. The harness now also requires `[DONE]` for streaming success.
Historical benchmark files used the previous HTTP-status-only success definition.

## Snapshot records

A background task samples approximately every second and on graceful shutdown:

- Current admission budget/version/cost and existing gateway metrics.
- `counts`: arrivals, accepted, rejected, completed, failed and cancelled **during
  the actual interval**. Divide by `interval_seconds` for rates.
- `totals`: lifecycle counters since process startup.
- `awaiting_first_content`: admitted requests with no content yet, including any
  gateway waiting time and upstream queueing.
- `oldest_awaiting_first_seconds`: maximum age since receipt among those requests.
- `streaming_requests`: requests with observed content that have not terminated.
- `active_requests`: all received requests without a terminal outcome, including
  any requests still being read or waiting for admission.

These are gateway observations, not vLLM's internal prefill/decode counts. Completed
requests in a snapshot may have arrived in an earlier interval. Use terminal
request events for latency analysis; do not average per-second P99 values.

All mutable telemetry state is owned by the event loop. File serialization/writes
run in a background thread, outside the admission lock. Slow disk writes can delay
the next sample; the actual interval is recorded. The event queue is bounded at
100,000 records. Check `telemetry_dropped_records` and `telemetry_write_errors` in
`/metrics` (also recorded in snapshots); discard runs with losses. Startup fails
if the log cannot be opened. Runtime write failures count lost records and allow
serving to continue. Graceful shutdown flushes queued records; abrupt process
termination can lose unflushed records. Run one worker only.

## Validation

```bash
python -m pip install fastapi httpx pytest
python -m pytest -q
```

Tests use mock upstream streaming responses; no GPU is needed. They cover budget
updates, streaming cleanup, fragmented UTF-8/SSE, terminal counters, periodic file
output, incomplete streams, and client/gateway request ID correlation. Real GPU
smoke testing and measurement of instrumentation overhead remain necessary before
using the data for system identification.
