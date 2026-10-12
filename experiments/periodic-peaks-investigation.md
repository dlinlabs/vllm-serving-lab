# Periodic pauses and TTFT waves: reviewed experiment plan

Status: proposed, plan-only. No diagnostic code, policy changes, GPU runs, or
performance improvements are delivered by this document.

Source reviewed: `68fdfa5073b278fa50e408f165facc14a869b21b`, which merged
PR #11 on October 11, 2026 at 18:22:46 PDT (October 12 at 01:22:46 UTC).
The existing diagnosis and analyzer are now on main; do not create a competing
analyzer or treat PR #11 as an unmerged dependency.

## In plain language

First analyze the existing logs, then add timing probes as a separate change.
Neither change introduces a new controller or changes admission policy:

- Count every request, including rejections and failures, rather than studying
  only the tallest peaks.
- Time gateway work and Python garbage collection, and collect available vLLM
  metrics. If the evidence cannot distinguish causes, report "unknown."
- Start without a GPU campaign. If real evidence is needed, separately approve
  one existing five-minute replay, not six runs.
- Use the findings to choose one follow-up experiment. Judge a later fix by
  requests completed within their deadlines, not just smoother graphs.
- Qualified studies of how the system responds to budget changes can proceed
  alongside deeper pause diagnosis; they need not wait for every cause to be known.

## 1. Are we investigating the right thing?

**The eventual objective is to serve more useful requests while meeting each
admitted request's TTFT and E2E deadlines.** A smooth graph, a high GPU utilization
percentage, or a lower successful-only P99 is not that objective.

The immediate investigation has three distinct questions:

| Question | Required result | What is insufficient |
|---|---|---|
| Q1. What produces recurring gateway callback delays? | Measured operation/thread timing and a targeted intervention consistent with that mechanism. | A roughly 500-admission cadence, an adjacent log gap, or assuming GC. |
| Q2. Where do TTFT deadline misses accumulate, and how are they associated with admission waves? | All-request deadline accounting, same-request timing, and engine observations where available. | Only examining the slowest 1%, calling all post-dispatch time prefill, or correlating plot peaks. |
| Q3. Can an admission change improve the stated SLO metric? | A qualified policy comparison with unchanged offered work, deadlines for the claimed metric, explicit guardrails, and per-input-size outcomes. TTFT-only results are allowed; joint TTFT/E2E compliance needs both deadlines. | Removing pauses alone, admitting less without improving goodput, or presenting TTFT-only improvement as joint compliance. |

**No 30-minute campaign is required now.** Deliver A, offline accounting from
existing logs, first; deliver B, minimum opt-in diagnostics with collection,
auditing, and CPU-only regression coverage, separately. Neither requires GPU results.
When a real capture is approved, start with at most one existing five-minute
replay. Dedicated overhead/repeatability comparisons are deferred; a single
capture can reveal a candidate mechanism, but cannot establish negligible
overhead, historical causation, or policy benefit.

### Metrics and claim-specific decisions

Preserve the client fields from `sustained_load.py::consume_request` and the
scheduled-arrival TTFT goodput definition from `analyze_trace_pilot.py::stats`.
Add scheduled-arrival E2E only as a separately named offline investigation field:

```text
Existing client fields (unchanged):
  ttft_i       = first_content_at_i - started_at_i
  end_to_end_i = finished_at_i - started_at_i

Existing TTFT goodput =
  count(successful complete streams with
        first_content_at_i - scheduled_at_i <= 0.5 seconds)
  / arrival-window seconds

New field in separate offline investigation output:
  scheduled_e2e_i = finished_at_i - scheduled_at_i

Future joint goodput =
  count(successful complete streams with
        first_content_at_i - scheduled_at_i <= TTFT_budget_i and
        scheduled_e2e_i <= E2E_budget_i)
  / arrival-window seconds
```

Keep `client_p99_e2e` based on successful requests' existing `end_to_end`,
not `scheduled_e2e`. Preserve the existing client/scheduled TTFT report fields
and cohorts too. Missing timestamps yield unknown values, not zero latency.
Report successful-stream E2E distributions separately from rejection/failure
termination times; fast rejection is not successful completion.

`scheduled_at` is the benchmark's original arrival, not gateway admission.
Client-start TTFT/E2E remain separately labeled diagnostic metrics. Gateway
waiting, if introduced later, must remain inside the original deadline.
Goodput is an arrival-cohort metric, including its drain outcomes, not a
steady-state completion rate.

The baseline's 0.5-second TTFT target is an experimental setting, not an approved
product requirement. There is **no existing E2E SLO**. Do not invent one or claim
joint TTFT/E2E success. A qualified, explicitly **TTFT-only comparison** may use
the existing 0.5-second target without a new E2E deadline. It must report both
E2E timebases, failures, rejections, admitted-request TTFT misses, and input-stratum
effects, with predeclared comparison criteria and guardrails. Its conclusion is
about TTFT goodput, not joint compliance or a production SLO commitment.

Before claiming **joint TTFT/E2E compliance**, agree on both budgets, any request
classes, the tolerated admitted-request violation rate, and rejection/fairness
guardrails. An E2E contract is not a prerequisite for every TTFT-only experiment.
If the requirement is zero observed admitted-request violations, evaluate every
admitted request; P99 cannot substitute for that requirement, and a finite passing
run cannot guarantee future zero violations.

Always report all offered outcomes, admission/rejection counts, admitted-request
deadline misses and failures, actual known token work, and the existing input
strata: `<=128`, `129..512`, and `>512` input tokens. Unknown work on failed
streams is unknown, not zero. Pair policies by run; do not restrict their primary
comparison to requests that happened to succeed under both policies.

## 2. Evidence and boundaries

The raw baseline is `1cafbda60a9d140e1f023c7e42a58bf00a2d10bc`.
Use `results\trace-budget96-analysis\report.md` and its joined records as existing
evidence, not newly collected results:

- Both long runs pass integrity but fail the original 50 ms maximum client
  scheduling-lag rule. Preserve both verdicts and every late request.
- The slowest 23 successful requests per run spend approximately 98% of their
  combined TTFT after `backend_sent`. Their selected post-dispatch intervals do
  not overlap the selected large monitor-overdue intervals. This does not rule
  out indirect queue effects or smaller delays.
- The raw `report.json` totals contain 1,518 successful TTFT misses in R1
  (`2239 - 721`) and 1,561 in R2 (`2220 - 659`). The 23 tail requests per run are
  1.52% / 1.47% of those successful misses, approximately 1.5%. This motivates
  all-request accounting first; it is cohort coverage, not an estimate of the
  misses that burst smoothing or another intervention would prevent.
- `backend_sent` precedes HTTP send. Its interval includes transport, frontend,
  engine queue/compute, and gateway observation. It is not GPU-only time.
- Four large monitor delays appear per long run, with approximately 65-73 seconds
  between recurring events. Work-count cadence is a clue, not a proven trigger.
- Equal reserved cost can conceal different awaiting-first-content and streaming
  populations. These are gateway observations, not vLLM prefill/decode gauges.
- The trace is one source window with synthetic prompts. All successful baseline
  requests generated 128 tokens. Neither arbitrary traffic nor variable decode
  demand nor an MPC response to budget changes has been evaluated.

The recorded runtime was dirty; the five recorded source hashes establish only
those files' identity. The uploaded bundles lack gateway process and vLLM launch
logs, and disabled prefix caching was self-reported. A new run must close these
provenance gaps, not infer settings from the old upload.

## 3. Exactly what code would change

This is a specification for subsequent implementation, not changes made by this
plan. Define two independently reviewable initial deliverables:

- **A: offline accounting.** Use existing artifacts for the all-request outcome/
  deadline ledger, available same-request latency stages, `scheduled_e2e`, and
  admission/completion/input-token windows. Include offline fixtures and preserve
  frozen reports. A can merge and be used without B or any new runtime events.
- **B: minimum opt-in diagnostics.** Deliver collection, diagnostic span joins,
  auditing, and the required correctness tests together. All gateway, telemetry,
  runner, and diagnostic-auditor changes below belong to B.

Request-linked engine tracing, detailed transport profiling, confirmatory
overhead studies, and policy interventions remain conditional, separately scoped
follow-ups. Splitting A and B does not make B's measurement invariants optional.

| File and existing entry points | Required change | Invariant / acceptance |
|---|---|---|
| `analyze_trace_dynamics.py::analyze_run`, `figures`, `main` | A: extend the existing join with all-request deadline/outcome accounting, available latency stages, `scheduled_e2e`, and admission/completion/input-token windows. B: add diagnostic span joins. Write investigation outputs separately from frozen reports. | Reconcile stage sums where the necessary events exist; retain missing stages and all outcomes explicitly; overlapping evidence categories are not additive causal attribution. |
| `telemetry.py::Telemetry`, `run`, `flush`, `monitor_loop` | Add an opt-in GC callback collector, gateway process/loop-thread CPU observations, bounded callback handoff, and explicit diagnostic health. Drain actual spans through the loop-owned emitter. | No cross-thread mutation of `records`/`active`; no lost records reported as zero activity; keep the 100 ms monitor semantics. |
| `gateway.py::lifespan`, `proxy_chat_completions`, `begin_request`, `finish_request` | Register/remove diagnostics with lifespan; collect per-request phase boundaries, including admission-lock wait and actual reservation release. Use optional timing context where helper signatures need it. | Keep cost formula, budget logic, HTTP-client lifetime, headers, byte forwarding, timeouts, and cancellation-safe exactly-once cleanup unchanged. |
| `run_trace_pilot.py::main`, `run`, `poll_state` | Add a default-off diagnostic option, protocol/arm/pair metadata, complete code/package fingerprints, explicit health/artifact collection, and controlled engine/resource sampling. Reuse `finalize` rather than forking the runner. | Preserve `prepare_replay` preallocation and `replay` scheduling; do not serialize JSON, tokenize, or synchronously sample resources on the arrival path. |
| `analyze_trace_pilot.py::_analyze` and runner exit handling | Validate diagnostic schemas, span timing, final persistence, and mode agreement when the new protocol requests them. Keep a separate diagnostic-integrity verdict and require it for diagnostic qualification. | Existing schema-1 bundles must reproduce the same original report; absent legacy diagnostics are not zero GC. Existing integrity/fidelity fields and definitions remain unchanged. |
| Existing telemetry, replay, budget-lifecycle, and client-error tests; new `tests\test_trace_dynamics.py` | A: add offline accounting/analysis fixtures. B: add diagnostic fixtures and extend existing subprocess/streaming tests. | A covers all outcomes, missing stages, zero-success cohorts, and frozen-report compatibility. B retains bounded-buffer, cross-thread/reentrant callback, cancellation, exactly-once cleanup, shutdown, and legacy-compatibility requirements. |
| `experiments\runtime-telemetry.md`, `experiments\trace-replay.md`, and this plan | Document opt-in schema, commands after implementation, qualification, and the new protocol identifier. | Do not silently change frozen presets, thresholds, source window, or historical conclusions. |

Paired-run statistical reporting is deferred with the paired studies; it is not
a requirement for the first instrumentation change.

Proposed interface, not implemented: `--gateway-diagnostics` enables
`GATEWAY_DIAGNOSTICS=1`; omission disables it. The runner must explicitly set
`GATEWAY_DIAGNOSTICS` to `1` or `0` for the chosen mode, rather than inherit it
from the parent environment. Record a new
`investigation_protocol`, `diagnostics_schema_version`, requested and observed
diagnostic modes, and sampler configuration in the manifest. Keep the existing
run identifier; pair/arm identifiers are needed only for a paired study.
Mode disagreement or missing required observations must fail
diagnostic qualification; explicitly disabled observations are not missing data.
Fingerprint changed runtime files, the analyzer, and relevant packages including
`httpcore`. Archive allowlisted launch/configuration details, not the entire
environment or unredacted credentials.

`sustained_load.py::consume_request` already records scheduled/start/first-content/
finish timestamps and failed/cancelled outcomes. It needs no change for A or the
initial gateway diagnostics. Calculate the new `scheduled_e2e` offline without
renaming or redefining the existing client-start `end_to_end` or `client_p99_e2e`.

Do not initially change `experiments\trace-presets.json`, `setup_and_start.sh`,
cost coefficients, GC policy, HTTP-client reuse, output lengths, or admission
pacing. Do not introduce MPC or a second experiment runner.

## 4. Measurement contract and implementation hazards

### Per-request phases

Capture body-read, JSON-parse, token/cost-estimation, admission-lock wait,
client-construction, HTTP-send/headers, first body bytes, first nonempty text
content, upstream/client close, and reservation-release boundaries.

Collect one compact terminal phase summary per observed request, including
rejected and failed requests. Record first-byte/content timestamps once, not on
every token. Preserve partial phases as failed/cancelled/not reached, not
successful zero-duration operations. Do not finalize a streaming request's
summary when the handler merely returns `StreamingResponse`; its body and cleanup
have not finished. Handle early rejection separately, without duplicating terminal
events or changing cleanup behavior.

A long asynchronous send/lock wait is elapsed time, not evidence that its caller
blocked the CPU. Use synchronous phase boundaries and thread/stack evidence for
that claim. Reserve per-chunk profiling and detailed HTTP transport tracing for
a separately measured, targeted diagnostic mode.

### GC, CPU, ordering, and shutdown

- Use `gc.callbacks` for collection start/stop, generation, thread identity,
  collection sequence, actual monotonic timestamps, and callback-thread CPU duration.
  Record GC enabled state, thresholds, and counts at diagnostic startup and around
  warmup. Do not force collection or change thresholds in this change. These
  callbacks observe cyclic GC in this gateway process, not ordinary reference-
  count cleanup or collection in a separate vLLM process. [R1]
- Callbacks must be minimal and nonblocking: no logging, JSON, heap walks, or
  waiting for the event loop. An ordinary thread-safe queue is not automatically
  reentrancy-safe if GC interrupts code while that queue's lock is held.
  Require a design review and isolated-subprocess tests of callback reentrancy,
  cross-thread delivery, bounded memory, and loss accounting before accepting the
  handoff implementation. An unbounded queue is not a bounded-memory solution. [R2]
- Emit span records on the loop with normal emission-time `monotonic`, storing
  `span_start`/`span_end` separately. Never backdate emitted records and thereby
  violate the existing ordered-log audit. Span fields use seconds on the shared
  monotonic clock; label CPU units explicitly and do not join intervals by UTC.
- Sample loop-thread CPU on the loop thread and process CPU separately. Measure
  GC CPU on the callback's thread. Do not add overlapping CPU measurements.
  A GC wall interval can include descheduling; an async request's thread-CPU
  delta across awaits includes other tasks and is not its exclusive CPU cost. [R3]
- RSS and host contention sampling must identify the process/host and its actual
  timing. Do not substitute peak RSS for current RSS. Keep slow OS/GPU collection
  out of the gateway loop and the client arrival loop.
- Stop producers and safely quiesce callbacks before the last diagnostic drain,
  snapshot, flush, and closed-status record. The current stop event alone is not
  proof that late callback records persisted. Unmatched spans, buffer loss, write
  failure, and abrupt termination must remain visible.
- Disabled diagnostics must not register callbacks, start diagnostic tasks, or
  allocate per-request diagnostic state. Preserve existing lifecycle sequences.
  This is a functional requirement, not a claim of literally zero timing cost.
- Log identifiers, times, and counts only; no prompts, generated content, keys,
  headers, or full HTTP trace objects.

### Engine observations

Use a modest fixed initial cadence, proposed at one second, to archive raw
`/metrics` responses with request/response times, process identity, and errors.
Verify live metric names and types on the pinned server; counter exposition names
can differ from documentation family names. Candidate families include waiting/
running requests, queue/prefill/decode duration, KV-cache usage, preemptions, and
prompt/generation token counters. Record unavailable metrics explicitly. [R4]

Histograms are aggregate, often completion-reported observations: they cannot
assign a particular request's queue or prefill time. Never subtract their P99s.
Handle counter resets and warmup boundaries. If this is insufficient, select a
bounded engine trace plus a quiet control window and verify request-ID mapping;
do not promise that coarse metrics will settle the question.

Keep client and gateway on the same monotonic clock initially. Cross-host
collection needs clock mapping with an error bound and revised auditor checks.
Loop `scheduled_at -> monotonic` remains an overdue-callback interval, not an
exact continuous CPU-pause interval.

### How the new observations distinguish candidates

The documentation-only change cannot recover events absent from the old R1/R2
logs. The proposed instrumentation makes a future capture more informative; it
must not output a binary "GC or prefill" verdict when neither or both may apply.

| Observation in a new capture | Justified interpretation | Remaining limit |
|---|---|---|
| A gateway GC start/stop span covers much of an overdue-loop interval, with substantial callback-thread CPU consumption during the span. | Stronger evidence of a GC contribution than cadence or collection counts alone. Repeated alignment strengthens the candidate. | Wall overlap alone can reflect descheduling; P2 provides a targeted intervention before a causal claim. This does not automatically explain separate TTFT waves. |
| Complete callback coverage shows no GC spanning the observed gateway delay. | Direct gateway cyclic-GC blocking during that interval is not supported. | This does not exclude earlier indirect effects, reference-count cleanup, another process's GC, or other synchronous work. |
| Gateway timing is healthy while engine waiting/queue metrics rise around the delayed requests. | Evidence consistent with engine-side backlog, provided the engine observations are valid. | Aggregate histograms do not identify an individual request's phase, and backlog is not proof of prefill contention. |
| Request-linked engine events locate excess delay in queue/prefill-stage intervals during a prompt-work burst, and a targeted matched-work comparison reduces it. | Evidence supporting prefill-related contention, after considering preemption, KV pressure, decode work, and input-size feasibility. | A prefill-stage elapsed interval is not exclusive GPU-kernel execution time. A GPU-compute attribution may need separately bounded scheduler/GPU profiling. |

For request-level engine attribution, require verified request-ID mapping and
available engine events such as `QUEUED`, `SCHEDULED`, `PREEMPTED`, and first
`NEW_TOKENS`, with their deployed-version semantics. Use durations from a known
clock domain; verify compatibility before aligning them with gateway timestamps.
The documented prefill interval is a scheduling-to-first-output stage, which
can include interruptions, not a direct GPU timer. [R4]

`run_trace_pilot.py` should initially archive exposed engine metrics, and
`analyze_trace_dynamics.py` should report their limitations. If the deployed
server does not export the required request-linked events, mark precise prefill
attribution unavailable and scope a separate trace adapter or targeted capture.
More gateway timestamps alone do not close that gap. Do not subtract queue P99
from TTFT P99 or treat missing engine evidence as confirmation of prefill.

## 5. Staged protocol

The stages below organize scopes, not a requirement to finish every diagnosis
before model work. A comes first for existing-data prioritization. Separately
approved, qualified budget-response identification may proceed alongside B or
deeper pause diagnosis, as specified in P3; it does not require a P2 root-cause verdict.

### P0-A. Offline accounting and prioritization

1. Reuse the existing auditor and analyzer against the frozen artifacts without
   overwriting them. Export all requests, not only tail/late rows. Do not require
   new diagnostic events to produce A's output.
2. Build an all-offered outcome ledger: rejected, failed/cancelled, successful
   within TTFT deadline, and successful but late. Add `scheduled_e2e` distributions
   without changing existing client E2E fields. Join available same-request stages
   and admission/completion/input-token windows. Show unavailable gateway,
   transport, or engine evidence as unknown; an adjacent log gap is not that
   request's phase duration or a counterfactual explanation of its miss.
3. Determine whether the visually large peaks account for an important share of
   misses, rather than selecting work solely by maximum TTFT. If a prompt class
   remains slow under low contention, plan a separately labeled low-load
   feasibility reference before promising admission can meet its deadline.

Use offline fixtures for missing timestamps/lifecycle events, never-launched
requests, zero-success cohorts, incomplete streams, and deadlines crossed before
admission. A fast first token followed by stream failure is not goodput. Check
cohort/count reconciliation and preserve the original report exactly; a
failed-fidelity baseline remains
failed-fidelity even when its ledger is useful.

Exit A: reproducible all-request accounting and evidence priorities from existing
logs, with unknowns preserved. No runtime instrumentation or GPU capture is needed.

### P0-B. Minimum opt-in diagnostics and correctness

1. Test mock replay with ties, slow streams, pre-header errors, cancellation,
   pending-limit overflow, and shutdown failures. Add synthetic GC, non-GC CPU,
   overdue-callback, async-wait, and delayed-emission fixtures so the analyzer can
   distinguish them.
2. Implement and test the minimal diagnostics before the P1 instrumented capture.
   Use existing test fixtures and the project's pytest runner. A mock run qualifies
   code behavior, not the actual shared GPU host's load fidelity.

Mandatory regression cases include a fast first token followed by stream failure
(not goodput); a deadline crossed by client/gateway waiting; overlapping diagnostic
spans counted by interval union rather than summation; and disabled, missing, and
zero-valued observations kept distinct. Cover partial streams, rejections,
cancelled lock/send waits, exactly-once cleanup, late callback delivery, handoff
overflow, flush failure, and unmodified legacy report output. Compare forwarded
bytes and lifecycle events with diagnostics off/on under deterministic mocks.
Include a parent environment that enables diagnostics while the runner flag is
omitted; the runner must still launch diagnostics off.
Port-bound integration tests skipped because another service is running do not
count as exercised coverage; use an isolated test allocation, not process killing.

Exit B: diagnostic schema and failures are covered, legacy audits reproduce, and
there is a concrete Q1/Q2 evidence table to collect. No runtime changes are part
of this plan-only document.

### P1. Optional single-run diagnostic capture

No GPU run is required to deliver the instrumentation change. When ready to
collect real evidence, separately approve at most one instrumented replay using
the existing 300-second trace: five minutes of measurement plus startup, warmup,
and drain. Reusing that trace avoids adding a new workload just to shorten the
first attempt, and includes the portion containing the historical late TTFT wave.
Do not require six runs or automatically repeat a capture.

Use **`ADMISSION_MODE=cost_aware` and budget 96**, with diagnostics on. Hold
constant the 7,643-row trace and its hash, input sequence, model/tokenizer
revisions, output cap 128, natural EOS, temperature, three fully drained warmup
requests, timeouts, pending limit, and single-worker configuration. This is an
observational diagnostic, not a policy or overhead comparison against the old
runs.

An approved GPU allocation and total wall-clock/rental budget are launch gates.
The trace runner starts only the gateway; declare a separate, approved backend-
reset procedure rather than assuming gateway restart resets vLLM. Record launch
flags, package/source fingerprints, warmup, process/GC state, and host conditions.
Use the existing prefix-cache-disabled launch protocol, archive backend launch/
runtime records, and verify the observed setting rather than trusting the runner's
`--cache-policy disabled` label. This establishes the new run's configuration,
not the missing historical setting.
Do not alter unrelated processes or run competing experiments on the allocation.

Inspect each observed large loop delay and TTFT-miss group using the evidence
table above. Preserve all offered outcomes, all diagnostic spans, and smaller
monitor delays; the over-300-ms selection is only for detailed rows, not a new
validity threshold. Report candidates and missing evidence, not a forced root
cause. If the symptoms do not reproduce, report that limited result and stop.

**Stopping/qualification rules:**

- Keep the existing integrity checks and 50 ms maximum client scheduling-lag
  limit. A failed-fidelity capture can inform diagnosis but cannot qualify a
  performance comparison; do not relax the limit or delete late rows.
- Diagnostic loss, incomplete callback spans, or missing required observations
  block negative conclusions such as "no GC occurred." Preserve their health
  verdict separately from client/outcome evidence.
- One attempt is the initial cap, including a failed attempt. Additional runs,
  deeper profiling, or interventions require a new decision, not an automatic
  repeat-until-success loop.

#### Deferred controls and overhead assessment

The first capture does not measure observer overhead. Only when a comparison or
performance claim is needed, separately approve paired runs, practical margins,
sample size, ordering, and a total resource cap. Both arms below remain cost-aware
at budget 96; the control is not the repository's admit-all `baseline` mode.
Match each pair's trace and declared process/warmup/reset conditions; randomize or
counterbalance C/I order within comparison blocks.

| Arm | Gateway | Other observations |
|---|---|---|
| C: control | Same candidate build, new gateway diagnostics off; original lifecycle/loop telemetry remains. | Same predeclared external engine/resource sampling as I. |
| I: instrumented | Same build and admission behavior, new gateway diagnostics on. | Identical external sampling, versions, and cadence. |

This contrast estimates the incremental effect of enabling in-process diagnostics
under common external monitoring. It does **not** estimate the common sampler's
cost or prove that the patched-but-disabled build is identical in timing to the
old commit. Historical attribution or total-instrumentation-cost claims also
require a separately specified original-code/common-sampler comparison. Do not
assume a mechanism observed after instrumentation caused the old R1/R2 pauses.

Analyze paired run-level differences, not individual requests as independent
replicates. A nonsignificant difference does not establish negligible overhead.
Confirmation needs predeclared practical margins and uncertainty bounds inside
those margins, plus tolerances for admitted work/mix, failures, and recurrence
timing. No confirmatory run count or numerical margins are approved here. [R5]

Keep failed runs in the comparison. If I alone fails fidelity, retain it as
potential observer harm. If both arms are unqualified, revise the collection
protocol openly and compare both under the revision; do not erase the failures.

### P2. One cause-discriminating experiment

Choose a branch using P0-A/P0-B/P1 evidence and its relevance to SLO misses. Keep the
diagnostic configuration fixed between its arms; predeclare the predicted
signature, run count, and resource limits. Do not run the whole matrix.

| Candidate mechanism | Next contrast | Evidence needed / limits |
|---|---|---|
| Gateway cyclic GC | Normal collection versus a separately approved bounded GC-policy diagnostic, only after actual spans repeatedly align. | The GC and overdue-callback signatures must change consistently. GC-disabled runs need fresh processes, an external watchdog, explicit RSS/duration ceilings, and restoration afterward; they are not a production fix. |
| Per-request HTTP-client lifecycle | Existing per-request clients versus a lifespan-scoped client, without changing admission. | Preserve headers, cookies/isolation, response closure, cancellation and timeouts; record pool limits and avoid a new pool bottleneck. Reuse changes connections as well as allocation, so improvement alone does not prove GC. |
| Synchronous application work or host contention | Bounded profiling of the implicated phase/thread; then one targeted code/resource change. | Long await time alone is insufficient. Keep detailed profiling out of primary performance runs; alter only owned/approved resources. |
| Admission/refill waves and post-dispatch TTFT | Budget 96 versus budget 96 plus a calibrated short-window admitted-input-token limit. | First obtain transport/engine evidence. Keep immediate rejection; do not add queueing or change GC/client lifetime in this comparison. |
| Work-count versus timer recurrence | Separately versioned arrival-rate protocols with the same request-shape sequence. | Compare actual admitted/received/completed work and tokens as well as time. Offered rate alone does not identify the trigger; insufficient recurrences remain inconclusive. |

For the token-window experiment, choose the window and cap on separate calibration
data and freeze them before comparison. Define oversized-prompt behavior so an
eligible request cannot be permanently rejected accidentally by a cap smaller
than its prompt. Preserve the total-cost reservation invariant.

A policy may legitimately reject more and increase all-offered SLO goodput:
equal admission counts are not a universal success requirement. Report class
effects and admitted-request reliability. However, attributing improvement to
burst smoothing rather than less work requires an additional matched-request/work
mechanism control. Forced acceptance or retimed dispatch is not the original
offered-arrival policy benchmark. Include any real pacing wait in end-to-end SLOs.

Label mechanisms supported, not supported in the measured runs, or unresolved.
Support requires repeated measured timing plus the predicted targeted-intervention
effect, not overlap alone. A partial or unresolved diagnosis is a valid result;
do not require eliminating every natural pause before later control research.

### P3. Parallel identification and claim-specific policy evaluation

Budget-response identification may proceed alongside deeper pause diagnosis when
the input/state/outcome observations and load fidelity qualify for its declared
protocol. It does not require every pause to be explained, a P2 causal result,
or completion of B when existing observations suffice. Preserve real pauses and
uncertainty in the data and validation; do not remove disturbances to make the
model look better. Opening this path does not change the R1/R2 fidelity failures
or qualify those runs for policy comparisons.

Identification needs a separately scoped and approved protocol with recorded
applied controls, bounded excitation, resource limits, and sufficient observations.
Do not insert budget changes into the frozen budget-96 diagnostic capture or treat
this plan as approval to launch additional runs.

For an explicitly TTFT-only policy result, use the existing experimental target
and the reporting/guardrails in section 1; no new E2E deadline is required.
Joint TTFT/E2E compliance requires the agreed joint contract. Predeclare meaningful
improvement in the claimed metric, guardrails, independent repeats, and held-out
source windows. Use qualified all-offered comparisons, not only shared successes.

A fixed-budget diagnosis alone does not establish the effect of varying the
budget. Subsequent MPC work needs deliberate budget/admission excitation, state
and uncertainty estimation, held-out prediction checks, variable output-demand
coverage, and comparisons with tuned fixed and simple reactive policies.
Forecasts of arrivals or output demand computed from information available at
decision time are allowed. Do not expose realized future trace arrivals or
realized future output lengths to the controller. Keep offline fitting and model
selection separate from held-out evaluation data. Predeclared online updates may
use only observations already available at the decision, never future evaluation
data. GPU utilization remains supporting evidence of useful capacity use, not
the objective by itself.

## 6. Self-review and disposition

Review standard: falsifiable questions, defined measurement boundaries, valid
comparison units, preserved workload/outcomes, explicit alternative explanations,
failure-path correctness, and no claim stronger than the evidence.

| Finding in the earlier proposal | Disposition in this plan |
|---|---|
| Measuring overhead could become the experiment's goal. | P1 now starts with an optional single diagnostic capture. Dedicated overhead assessment is deferred. |
| "Baseline" could be confused with the admit-all mode. | The diagnostic capture and later diagnostics-off/on controls explicitly remain cost-aware at budget 96; identification uses its separately declared excitation protocol. |
| E2E and per-request reliability were not acceptance criteria, and the proposed gate could block TTFT-only work. | Permit qualified TTFT-only comparisons with E2E/outcome reporting and guardrails; require agreed E2E budgets and the joint violation policy before joint-compliance claims. |
| The opening could redefine existing client E2E as scheduled-arrival E2E. | Preserve `end_to_end` and `client_p99_e2e`; add a separately named offline `scheduled_e2e`. Existing TTFT goodput remains unchanged. |
| Focusing on 23 tail requests could miss most SLO failures. | Require the all-offered deadline/outcome ledger before prioritizing mechanisms. |
| Bundling offline analysis with runtime diagnostics could delay the inexpensive first result. | Define independently reviewable A/B deliverables. A uses existing logs; B keeps all collection, audit, bounded-buffer, cancellation, cleanup, shutdown, and compatibility requirements. |
| Serial phase wording could make complete root-cause diagnosis a gate on MPC model work. | Permit separately approved, qualified budget-response identification in parallel; do not waive fidelity, observation, or held-out validation requirements. |
| "No future arrivals" could incorrectly exclude forecasting. | Allow forecasts computed from decision-time information; prohibit realized future trace arrivals/output lengths and held-out forecast leakage. |
| A mandatory six-run block was premature for the first diagnosis. | Removed. The instrumentation change needs CPU-only coverage; any first real capture is separately approved and capped at one existing five-minute replay. |
| Paired-study reporting remained in the first implementation's scope. | Deferred with paired studies; A serves existing-data accounting and B adds single-capture diagnostic joins. |
| An omitted diagnostics flag could inherit an enabled parent setting. | Require explicit runner environment values and an inherited-setting regression case. |
| Warmup/cache state and run order could confound comparisons. | Pin three drained warmups, verify the new launch's cache setting, and require matched conditions with randomized or counterbalanced order for future pairs. |
| One capture could be misread as proof of repeatability or negligible overhead. | Restrict it to observed candidates; require separately approved run-level comparisons for stronger claims. |
| Off/on testing could be called total observer overhead. | State its incremental estimand and the additional bridge needed for original-code/common-sampler claims. |
| Failed instrumented runs could be filtered out as "bad load." | Preserve them as possible observer harm and count them against the pilot cap. |
| A callback-safe buffer and final flush were underspecified. | Require reentrancy/cross-thread tests, actual-span versus emission ordering, bounded memory, producer quiescence, and explicit loss. |
| Long phases or aggregate histograms could be mistaken for CPU/prefill attribution. | Add an explicit evidence table and request-linked engine-event requirement; an elapsed prefill stage is not a GPU-only timer. |
| Rejecting more was treated as automatically invalid. | Judge policy utility by all-offered goodput and guardrails; reserve matched-work controls for mechanism claims. |
| PR #11 was described as an unmerged dependency. | Pin the merged source and extend its existing analyzer. |
| A single fixed-budget, fixed-output window could be generalized to MPC. | Restrict conclusions to diagnosis; keep identification and held-out controller evaluation separate. |

**Review verdict:** suitable for implementing observability first, with no
mandatory long GPU campaign: A first, B separately, and qualified identification
alongside deeper diagnosis when its observations suffice. It is not an approved
production SLO contract or a completed causal/performance result. TTFT-only and
joint-compliance claims have different gates. Stronger overhead, historical
causation, and policy claims still need their separately approved comparisons.
GC, prefill contention, and a 500-request trigger remain unproven in the old runs.

## References

Repository references are to the reviewed commit unless explicitly pinned above:
`gateway.py:76,244,301,321`; `telemetry.py:70`; `run_trace_pilot.py:35,42,109,203`;
`analyze_trace_pilot.py:20,34`; `analyze_trace_dynamics.py:46,165,236`;
`sustained_load.py:89`; `experiments\trace-replay.md`;
`results\trace-budget96-analysis\report.md`.
The successful-miss counts use `performance.success - performance.slo_successes`
in `results\trace-experiment-96-r1\report.json` and
`results\trace-experiment-96-r2\report.json` at the raw-baseline commit.

- [R1] Python 3.12 GC interface and callback semantics:
  `https://docs.python.org/3.12/library/gc.html`
- [R2] Python 3.12 queue locking/reentrancy and capacity semantics:
  `https://docs.python.org/3.12/library/queue.html`
- [R3] Python 3.12 wall, process, and thread clocks:
  `https://docs.python.org/3.12/library/time.html`
  (pinned documentation source:
  `https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/time.rst`).
- [R4] vLLM production metrics and engine-event interval definitions; verify
  availability and semantics on the measured version:
  `https://docs.vllm.ai/en/stable/usage/metrics/`
  `https://docs.vllm.ai/en/stable/design/metrics/`
- [R5] NIST randomized-block and paired-observation guidance; the initial capture
  cap and deferred-confirmation requirements above are this plan's decisions:
  `https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm`
  `https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm`

If detailed HTTPX transport tracing is needed, use an async trace handler with
pinned `httpcore` event semantics; never log full callback payloads:
`https://www.python-httpx.org/advanced/extensions/`.
