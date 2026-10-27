# Busy-state and HTTP qualification v1

Implementation and CPU verification precede the first native attempt. No native
busy-state/HTTP result, production model, adaptive-cap or MPC acceptance exists.
The previously passing fixed-B2048 event-clock evidence stays unchanged.

## Prediction point and state contract

The new predictor starts at an EngineCore input boundary immediately BEFORE the
new request's native ADD. It observes current running/waiting FCFS order,
computed tokens, committed outputs, placeholders, and the still-uncommitted
batch. It then generates future work itself. Already submitted work is legitimate
past state; future native schedules, first-token times, service durations and
output lengths observed after the cut are forbidden inputs.

Snapshots preserve the actual pending FIFO, including empty futures. At the
reviewed native two-slot input boundary, at most one old batch remains queued;
a full two-entry backlog is not such a boundary and fails. Pending counts,
positions, placeholder/sample counts and committed progress must agree. Unknown
requests/history, duplicate membership, speculative work, preemption, impossible
progress or context/capacity overflow fail. No synthetic extension of background
lifetimes is made. Known forced output lengths are required in this pilot.

Continuation reuses the original running-first reference and dispatcher with the
six-term effective lane. The old from-empty predictor and fitted basis are not
modified. Reserve ceil((prompt+forced output)/block-size) for each active/new
request with10% capacity headroom. This is a conservative KV upper bound, NOT
measured allocator utilization or a preemption model.

Pending readiness is read nonblockingly from the existing V2 AsyncOutput's
copy_event.query(). It creates no CUDA event and never calls synchronize(),
Future.result(), get_output(), or materializes future token IDs. UniProc's
AsyncOutputFuture.done() stays false until result() runs and is NOT GPU readiness.
The source-pinned empty execute Future may use done(), because it carries no
sampling work. The reviewed sources already occur in the existing38-file map:
EngineCore, Scheduler/AsyncScheduler, Request, UniProcExecutor and AsyncOutput.

An unready old batch uses effective residual =
max(submission-schedule time,last observed output-commit time)+modeled service-cut.
Copy-ready means the D2H payload is ready; it does not prove the main stream is
idle or that post-update work has finished. This rolling commit anchor is an
explicit approximation. It can overestimate when
GPU work overlaps CPU commit; it does not identify exclusive physical stage time.
If an unready batch is already overdue under that approximation, the forecast is
UNAVAILABLE, never silently zero. Ready old output still occupies a slot: new
submission retains priority over FIFO reap. Native conditional1ms idle yields
and output placeholders/termination are retained. There is no cap decision.

The engine stores the state/forecast before ADD. Snapshot width and cut-to-ADD
age include the observer/predictor cost. Errors record an unavailable forecast
and preserve ordinary request execution. Native errors still propagate. Only
explicit opt-in enables these extra callbacks; normal modes reset the flag.

## HTTP path and clocks

A benchmark-only /v1/completions gateway route forwards exact token IDs through
the existing shared-client, protected admission and SSE cleanup path. It exists
only with PREFILL_LAB_RAW_COMPLETIONS=1 plus benchmark IDs/shared client/protected
mode. The dedicated localhost launcher fixes32 in-flight/0 waiting. Chat routing
and default admission semantics stay unchanged; this is not a dynamic cap policy.
Raw payloads require a matching32-hex correlation ID, token list, fixed128/256
outputs, temperature0, ignore_eos and streaming. Completions text and chat delta
content have separate SSE observation modes; role-only events do not start TTFT.

The isolated per-trial generator warms its HTTP transport with GET /metrics
before offer timestamps, then sends backgrounds, waits for their first CONTENT,
and posts the four foregrounds/probe at declared offsets. A fresh subprocess
avoids the controller's retained calibration heap/GC; automatic GC is otherwise
unchanged. Owned children terminate/join on cancellation and save partial rows.
There is no retry or stitched best run.

Client/gateway/engine records bind correlation ID, Linux boot ID and time-namespace
identity to CLOCK_MONOTONIC. Different domains, missing/duplicate milestones,
wrong run IDs, invalid chronology, non-content/truncated streams or incomplete
native outputs invalidate the join. Epoch wall-clock fields are never subtracted.
The decomposition is client-to-gateway + gateway-before-backend + backend-to-engine
+ engine queue/compute + engine-to-client first content. Components telescope;
overlapping worker wall spans are not added.

Prediction occurs AFTER the request has reached EngineCore. Client-send-to-cut
elapsed time is already known. The future engine interval is forecast online;
future egress uses a scalar frozen from TRAIN OFF only. HTTP TTFT is recomposed
from these causal components, without using held egress or native first-output
labels. This does NOT qualify a pre-admission gateway decision or unknown future
network/admission latency. Native snapshot/observer overhead needs qualification.

## Frozen experiment and gates

One loaded4090/vLLM0.30/Torch2.13+cu130/Qwen revision, B2048/seq32/context8192:

1. Run the unchanged136-trial clock pilot with fresh same-host fitting/freezing.
   Require its full candidate gates before configuring the live predictor.
2. Run98 HTTP trials from prefill-live-http-v1.json:10 LIGHT warmups,16 training
   trials,4 anchor-before,64 held and4 anchor-after. Every paired OFF/LIGHT trial
   shares exact token content; before/after anchors share their content too.
3. The two transport-training cases are mixed [1024,1024,7168,7168], sentinel512
   at200ms, D0/D16. Four OFF repeats each provide8 egress labels. Freeze their
   single nonnegative median before any anchor/held trial. Spread must be<=10ms.
4. Eight held cases use50/400/800ms mixed sentinels, plus an alternate
   [2048,6144,2048,6144] composition at200ms, all D0/D16, four OFF/LIGHT pairs each.

All10 warmup snapshots must be available and structurally match before the full
HTTP phase proceeds. A failure preserves the diagnostics and stops; no model
selection/refitting occurs from these warmups. Warmup latency gates are not scored.
For every scored LIGHT probe (8 train/32 held/4 anchor), require availability,
exact own future-work prefix to first output, snapshot width<=1ms, cut-to-ADD age
<=5ms. No later unmodeled arrivals may occur after its snapshot. Each per-trial
engine error must be<=max(10ms,20% measured interval), HTTP error<=max(15ms,20%).
These are new preregistered bridge budgets, not changes to older clock gates.
No sign cancellation or medians across changing states can hide individual errors.
Unavailable predictions fail and stay in the denominator. Separate before/after
anchors are scored individually. Paired HTTP observer delta must be<=max(2ms,10%)
for train/held/anchors. Posting lag stays<=5ms, including rejected/failed offers;
all planned requests must finish with exact inputs/forced counts and backgrounds
must outlive foreground completion. Native worker/dispatcher audits still apply.

The gateway must start empty with the declared fixed configuration, keep one
run identity, accept exactly the1274 planned HTTP requests, finish with no active,
waiting, rejected or failed requests, and close its complete, lossless journal.
Warmup/observer/state failures cannot be excluded, thresholds cannot change after
seeing results, and no production/p99 guarantee follows from passing this pilot.

## Native host commands, only after CPU implementation verification

Fresh native Linux/Python3.12 RTX4090 host, clean PR32 source:

```bash
nohup python3 scripts/setup_prefill_calibration.py --next live-http > ~/pr32-prepare.log 2>&1 < /dev/null &
tail -f ~/pr32-prepare.log
```

After PREPARED (exit tail with Ctrl+C):

```bash
nohup .venv/bin/python scripts/prefill_host.py live-http > ~/pr32-live-http.log 2>&1 < /dev/null &
tail -f ~/pr32-live-http.log
```

Preparation verifies the pinned runtime/model and complete exact-source native CPU
acceptance; it does not collect. Dedicated live-clock and live-http token manifests
keep model calibration and HTTP content separate. An inherited live flag cannot
activate normal collection modes. No archived coefficient vector is imported.

The combined234 trials/fitting/scoring have a shared1800s active limit INCLUDING
gateway startup. Initial engine/model startup has its separate900s limit; install,
download and native CPU tests are separate. This is a bound, not a measured runtime
estimate for this unrun protocol. Each generator has a bounded50s watchdog,45s
request window and owned cancellation cleanup. Captures/partial client/native
records, logs, original native CPU JUnit and both phase reports are archived at
~/pr32-live-http/<timestamp>.tar.gz, including on failure.

After successful collection and graceful shutdown, the host command automatically
runs the independent audit (separate180s CPU watchdog) before archiving. It can
also be rerun on the exact captured checkout:

```bash
.venv/bin/python scripts/check_prefill_live_http.py /root/pr32-live-http/TIMESTAMP
```

It audits the parent clock phase under the explicitly identified bridge launch,
re-extracts native inputs/clocks/snapshots, regenerates forecasts, checks the full
gateway journal/closure and recomputes the frozen transport and verdict. A valid
rejected candidate exits2. No native qualification is claimed from CPU fixtures.
Full model/HTTP production/cap/MPC flags remain false. Next after this qualification
is the separately designed fixed-versus-state-aware cap experiment.
