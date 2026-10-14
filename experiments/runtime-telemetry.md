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

## Opt-in periodic-peak diagnostics

The trace runner's `--gateway-diagnostics` flag enables gateway timing and the
external engine/resource sampler. Omission explicitly sets `GATEWAY_DIAGNOSTICS=0`,
even if the parent environment has enabled it. `--engine-observations` enables
only the external sampler, for a future control with common external monitoring.
Neither option changes admission policy, cost coefficients, per-request HTTP
clients, forwarding bytes/headers, output lengths, or the 60-second upstream
timeout. The original lifecycle and 100 ms loop monitor remain enabled.

The gateway mode requires CPython with its GIL enabled and both persistent
telemetry paths; unsupported runtimes or missing paths fail startup explicitly.
Disabled mode creates no diagnostic collector, request phase objects, GC
callbacks, or diagnostic tasks. This is not a measured zero-overhead claim.

Diagnostic rows use `diagnostics_schema_version=1`, the existing `run_id`, and
normal log-emission `monotonic`. Actual operation timestamps are separate:

| Event | Meaning |
|---|---|
| `diagnostics_started` | Coverage start, process and loop-thread identities, Python/GIL and GC state. |
| `diagnostic_cpu` | Actual sample time, cumulative process CPU and loop-thread CPU, in seconds; GC state and collector health. |
| `gc_span` | Matched callback boundaries: actual `span_start`/`span_end`, generation, Python/native thread identity, callback-thread CPU duration, collected/uncollectable counts. |
| `gc_unmatched` | Unmatched start/stop boundary; incomplete coverage, not zero GC. |
| `request_diagnostics` | Exactly one terminal summary after cleanup, with outcome, phases and first-body/first-content milestones. |
| `diagnostics_stopped` | Coverage end and final producer/buffer/loss health. |

Request phases are `body_read`, `json_parse`, `cost_estimation`,
`admission_lock_wait`, `client_construction`, `http_send_headers`,
`upstream_close`, `client_close`, and `reservation_release`. Each phase has actual
start/end and completed/failed/cancelled/not-reached status. Unreached milestones
are null. A streaming response's construction is not its completion: the summary
follows body termination, closes and reservation release. Repeated downstream
cancellation shields the same cleanup task, not multiple cleanup attempts.

The callback writes primitive tuples into a bounded 4,096-edge handoff, without
logging, JSON, locks, heap walks, or waiting for the loop. The loop owns emission
and matching; pending starts are bounded too. Health records include produced/
received/buffered/in-flight callbacks, errors, unmatched boundaries, lost and
undelivered callbacks, quiescence, registration and closure. Counter probes can
leave sequence-number gaps; gaps alone are **not** callback loss. Exact loss is
reconciled after quiescence. A quiescence timeout leaves loss unknown and coverage
incomplete. Shutdown removes the callback, awaits in-flight producers, drains,
flushes and writes final status; incomplete coverage or failed persistence cannot
produce a successful diagnostic verdict. No collection is forced and GC policy
and thresholds are unchanged.

Join GC and phase spans by their actual timestamps, never their later emission
time. CPU observations at activation and shutdown bracket warmup, measurement
and drain; a missing boundary sample fails qualification. The monitor interval
`scheduled_at -> monotonic` means an overdue callback,
not an uninterrupted CPU stall. A long HTTP/lock await is elapsed time, not its
request's exclusive CPU time. Process, loop-thread and GC CPU evidence overlaps;
do not add them or assign all CPU between awaits to one request.

### External engine/resource observations

`diagnostic_sampling.py` runs synchronous HTTP, OS reads, parsing and file writes
outside both arrival and gateway loops. The default `--sampler-mode thread`
uses a dedicated runner thread; opt-in `--sampler-mode process` uses a separate
Python interpreter with an explicit multiprocessing `spawn` context. Process
mode requires `--gateway-diagnostics` or `--engine-observations`; it does not
implicitly enable monitoring. The gateway remains a separate subprocess in
either mode. Default cadence is
one second; each observation records actual request/header/response/resource
times and collection-duration overrun. The retained `late_by_seconds` field is
`max(finished_at - sent_at - interval_seconds, 0)`, not worker start lateness;
use actual timestamps to reconstruct inter-sample gaps.
HTTP uses two-second operation timeouts; body reads check a
five-second deadline between chunks and stop at four MiB. These limits do not
turn a partial body into a valid sample. A fresh final sample follows stop,
including when stop arrives during an earlier HTTP request, before gateway
termination.

Both modes wait for a flushed first observation and atomically written startup
health before warmup. The sampler writes JSONL directly, without moving metrics
payloads back through the runner's event loop. The owner reads the small health
file only at readiness and shutdown. Readiness proves persistence, not that
every required metric/resource is available; the diagnostic audit still checks
the full capture. Separate one-way, single-message pipes carry readiness and stop
signals, without a cross-process event mutex. Readiness is signalled only after
the atomic status-file replacement returns, not merely when the file becomes
visible. Losing the owner channel is a worker failure, not a successful stop.

The manifest records `sampler.mode` and `sampler.worker` lifecycle metadata:
worker PID, start method, readiness/stop times, stopped state, exit code,
unexpected exit and forced termination. Health and every observation record the
collector's `sampler_process_id`; health also records the original
`runner_process_id`. The child receives the original runner PID explicitly, so
`resources.runner` never silently switches to the collector's own CPU/RSS.
Thread-mode collector PID must equal the runner PID; process-mode PID must differ
from both runner and gateway. Old captures without execution metadata remain
implicit-thread captures, not process-mode evidence.

Cooperative cleanup requests stop, awaits a fresh final sample and atomic final
health, then terminates the gateway. Repeated asynchronous cancellation cannot
skip this owned cleanup sequence. Worker failures, invalid health and unexpected
exits surface at readiness or shutdown; there is no continuous mid-run crash
supervisor. A process exceeding `--drain-timeout` is terminated, then killed if
needed, with two seconds allowed for each escalation. Forced termination, a
nonzero exit or incomplete cleanup cannot qualify a capture. A stuck thread
instead reports a timeout; Python threads are not forcibly killed. The default
thread is a daemon, not an asyncio-executor job that holds interpreter shutdown
indefinitely. Abrupt OS termination and unresponsive OS operations cannot
guarantee final persistence or cleanup.

This is an isolation mechanism, not evidence of lower latency or lower total
CPU. It separates sampler interpreter/GIL/heap work from the load generator, not
host cores, CPU quota, server-side `/metrics` work, state polling or gateway GC.
Offline analysis already runs after collection. See the
[separately approved A/B protocol](trace-replay.md#sampler-isolation-ab-after-native-validation).

The live Prometheus inventory checks exported family names, types, finite
nonnegative values and histogram bucket/count/sum presence. Required observations
are waiting/running requests; queue/prefill/decode duration histograms; KV-cache
usage (including the legacy GPU-cache alias); and preemption, prompt-token and
generation-token counters. Raw exposition, unavailable reasons, errors, exporter
identity where available, counter resets/disappearing series and counter epochs
are retained. Missing data is not a measured zero. Epoch boundaries must not be
crossed when deriving counter changes.

Process observations identify gateway and runner PID/creation time/host and sample
**current RSS** and cumulative user/system CPU. Before each process read on Linux,
the collector checks `self/status` under the configured `psutil.PROCFS_PATH`.
Its `NStgid` field must contain exactly one PID, equal to the collector's
`os.getpid()`, so the collector and procfs view agree on the PID namespace.
Matching a numeric PID or stable creation time alone is insufficient: another
namespace can contain an unrelated process with that same PID.
Nested, missing, malformed, duplicate or unreadable namespace metadata produces
`available=false` with `error_reason="procfs_pid_namespace_unverified"`, without
reading CPU/RSS for the requested PID. OS read failures also retain `error_type`.
These samples count as collector errors and fail diagnostic qualification; no
PID remapping or fallback to unverified resources is attempted. Compatible
procfs views retain existing missing-process errors, and non-Linux process
collection is unchanged.

Host observations are cumulative
CPU counters and logical CPU count, not an inferred GPU utilization percentage.
Exporter PID/start-time metrics are not proof of the vLLM engine process identity.
Raw metric labels are retained; review their contents before publishing a capture.
The runner omits backend URL credentials/query/fragment from its manifest and
omits HTTP error URLs; it never archives the full environment or admin key.

These are aggregate engine observations, not request-linked prefill timing.
Post-dispatch delay still includes transport, frontend, engine queue/compute and
gateway observation. Even aligned GC spans identify candidates for a new run,
not proof of what caused the historical R1/R2 peaks.

### Offline CPU interval alignment

Run `analyze_trace_dynamics.py` against a complete saved bundle, with a separate
`--output-dir`. No new capture, process isolation or runtime probes are needed.
The investigation's additive `cpu_analysis` section has CPU schema version 1,
and `cpu_scheduling.png` compares saved CPU intervals with client launch delays.
Original audit reports, fidelity limits and raw files are not changed.

- `sources.runner` and `sources.gateway` difference cumulative user/system CPU.
  `cpu_cores = delta(user + system) / elapsed_seconds`: one means one logical
  CPU's worth of execution averaged over that interval. Values may exceed one;
  they are not divided by host CPU count. `cpu_analysis.sampler_mode` identifies
  scope: runner CPU includes sampler work in thread mode, but excludes the
  separate collector in process mode. The plot labels follow this distinction.
  Separate sampler-process CPU is not measured, so a lower runner value across
  modes is not proof of lower total CPU work or negligible observer overhead.
- External interval timestamps are the midpoints of each resource's actual
  `sample_start`/`sample_end`, not HTTP send/receive or log-emission times.
  Both original read bounds remain in `sample_bounds_s`. These are estimates of
  the counter-read times, not a claim of instantaneous sampling. Counter-time
  granularity can make estimates over very short intervals especially noisy.
- `sources.gateway_loop` differences loop-thread CPU using `sampled_at`, not
  later emission time. Gateway process, loop-thread and GC CPU overlap; do not
  add them. The existing diagnostic process-CPU deltas remain available too.
  Sources have different sampling windows; do not subtract their rates to infer
  CPU used by other threads.
- `sources.host` preserves per-mode CPU deltas and reports a non-idle counter
  fraction: exclude `idle`/`iowait` from the numerator and avoid counting Linux
  `guest`/`guest_nice` or Windows interrupt/DPC time twice. Steal time remains
  part of the non-idle fraction. This is host-level context, not container CPU
  capacity, quota, per-core load or proof of spare CPU available to the runner.
- Adjacent samples are never reordered or joined across an unavailable sample.
  Invalid read times, identity changes, decreasing/nonfinite counters, changed
  host CPU count/fields and zero host counter progress retain interval issues
  and null derived values. Zero process CPU progress is a valid measured zero.
  Engine counter epochs do not reset independent process CPU counters.
- Disabled, legacy and diagnostic-integrity-failing sources have null intervals,
  not zero CPU. Enabled sources with fewer than two samples have no intervals
  and an explicit reason. Valid intervals in partially usable sources remain
  individually identifiable; inspect each interval's `issues`.

`late_launches` preserves each qualifying ledger row, its outcome and its original
`scheduled_s -> started_s` interval. CPU references index the source's full
interval list, including invalid intervals with unknown values. Possible overlap
uses the outer resource-read bounds, so overlap at a read boundary is uncertain.
An empty reference list means no overlapping saved interval, not idle CPU.
Never multiply a full CPU delta by the overlap fraction to invent per-request
CPU or assume that all work occurred during the late launch. Long sampling gaps
remain visibly long intervals; interval averages are not instantaneous peaks.

These views can locate CPU activity around delayed launches. They cannot separate
sampler-thread CPU from other runner work, identify runner GC, measure GIL wait,
or establish CPU quota/throttling or causality. Those observations were not
collected. A fidelity-failing run remains exploratory even when its CPU
observations qualify. No process split, admission change or GPU experiment is
performed by this analysis.

## CPU validation and handoff

**Deferred acceptance:** merging sampler isolation and the manual Linux CPU
workflow does not establish a Linux pass. The identification-tooling follow-up
also needs acceptance of its exact revision; added regression source and static
checks are not executed tests. No test or GPU run is scheduled by these changes;
the next native execution is reserved for an available, compatible host.

### Run later on the rental machine

Use a clean checkout of the PR revision on Linux x86_64 with CPython 3.12 and
its matching procfs view. Run this **before** starting vLLM or the gateway:
ports 8000 and 8080 must be free. No GPU, vLLM/Torch installation or model
download is needed for these tests. Use a fresh, dedicated CPU-test environment,
not the GPU server's environment:

```bash
python3.12 -m venv .venv-cpu-tests
. .venv-cpu-tests/bin/activate
python -m pip install -r requirements-cpu-tests.txt
python scripts/run_linux_cpu_tests.py
```

The shared entry point checks platform, revision cleanliness, the existing
PID/procfs guard and ports without terminating any service or remapping PIDs.
It runs the complete `tests` directory with `python -m pytest -q -ra`, clears
configuration `addopts`, uses the checked-out frozen fixtures, disables unrelated
third-party pytest plugin autoloading and refuses inherited
`PYTEST_ADDOPTS`/`PYTEST_PLUGINS`. Do not use `-k`, deselections or a guard bypass
as native acceptance.

Each invocation creates a fresh ignored `cpu-ci-artifacts/<UTC-timestamp>/`
directory. Retain `acceptance.json`, `packages.json`, `pytest.log` and
`junit.xml` as evidence; they record the exact commit, Python/platform, resource
preflight, installed package versions, command, exit code and test coverage.
Early failures may leave only partial evidence. `valid` remains false unless
pytest exits zero, **no case fails, errors or skips**, all five real trace-runner
configurations appear exactly once and the checkout remains unchanged.
A missing, stale or incomplete JUnit result cannot supply a pass. The five
configurations each exercise success, rejection, upstream failure and a scheduled
down/up-budget run with a joint TTFT/E2E objective. Their five JUnit identities
remain unchanged.

### Manual GitHub Actions job

[`.github/workflows/linux-cpu.yml`](../.github/workflows/linux-cpu.yml) uses a
GitHub-hosted Ubuntu 24.04 runner, CPython 3.12, CPU-only dependencies and the
same entry point. It has **only `workflow_dispatch`**: no push, pull-request or
scheduled trigger. Adding or updating this PR does not start it. A future
manual dispatch uses GitHub's Linux host; it does not need access to the rental
machine or its credentials.

The workflow definition is already on `main`. A future, explicitly authorized
manual run can select the revision's branch under **Actions > Linux CPU
acceptance (manual) > Run workflow**. Alternatively use the rental-host command
above. Neither path is started automatically by publishing a PR.

The job has a 20-minute limit, read-only repository permissions and immutable
action pins. It attempts to upload the evidence even after failure, with a
30-day retention period; download needed evidence before it expires.
No workflow has been dispatched as part of adding this configuration.

Tests use mock upstream streaming responses; no GPU is needed. Coverage
includes deterministic off/on byte and lifecycle parity, disabled allocation,
failed/cancelled phases, repeated cancellation, close/flush/status-write failure,
bounded/reentrant/cross-thread GC delivery, late delivery and shutdown, sampler
errors/counter epochs, parent-environment override, and collection-to-audit-to-
offline-analysis integration with damaged-artifact cases. Process identity
regressions cover existing-PID collisions, unverifiable procfs metadata and
persisted unavailable resources failing diagnostic qualification. Sampler-worker
regressions use a real spawned process and an ephemeral local HTTP server, plus
thread/process collection-to-audit cases. They exercise parent/collector PID
separation, startup/final persistence, in-flight stop, timeout, unexpected exit,
cancellation, execution-metadata corruption and runner cleanup ordering.

The original diagnostic implementation's archived Linux results do not validate
later sampler-isolation changes. Native acceptance uses CPython 3.12 matching
the recorded baseline, with a compatible PID/procfs view and no PID-remapping bridge.
Failing closed on an incompatible host is not a successful diagnostic capture.
The real subprocess replay test covers disabled sampling and thread/process
sampling with gateway diagnostics on or external-only, each across
success/rejection/failure and scheduled-budget/joint-SLO collection. Control
regressions also cover malformed/missing plans, command/version/timing damage,
traffic/control cancellation, deadline boundaries and legacy semantics.
It uses ports 8000/8080 and may skip if occupied;
a skip is not coverage and the acceptance entry point rejects it. Existing
process-termination, Linux setup, POSIX ninja and socket-reuse tests have known
Windows incompatibilities. Local Windows checks are not native acceptance.

See the PR handoff for exact executed checks and remaining validation. A CPU mock
pass does not establish GPU-host load fidelity, observer overhead or policy
benefit. Any GPU capture and stronger comparisons remain separately approved.
Proceed through the [small staged identification protocol](trace-identification.md),
not an automatic GPU matrix.
