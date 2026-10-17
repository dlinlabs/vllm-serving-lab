# Trace identification: small, gated steps before MPC

This is an experiment protocol and collection interface, not a fitted model or
controller. Test execution and GPU experiments are deferred until an explicitly
approved run on an available host. There is no automatic sweep, CI dispatch, model
fit, shadow controller or closed-loop experiment in this change.

## Declare the input and objectives before collection

Keep one prepared `trace.json` unchanged across compared runs. Budget controls live
in a separate JSON array, supplied with `--budget-schedule`. For example:

```json
[
  {"at_seconds": 0, "budget": 96},
  {"at_seconds": 100, "budget": 64},
  {"at_seconds": 200, "budget": 96}
]
```

This is an **interface example**, not evidence that 64 is optimal or that
100-second holds identify the dynamics. Select a safe range and hold lengths from
the earlier stages below, then freeze the actual file. Budget units are the
gateway's **admitted-work cost cap**, not request count, concurrency or requests
per second. A reduction does not evict existing work; reserved cost can temporarily
exceed the new cap while requests drain.

Each entry has exactly `at_seconds` and `budget`. Offsets must start at zero,
strictly increase, and remain inside the trace's arrival window. Budgets must be
finite and positive. Repeated budget values are allowed, but a no-op is not useful
excitation. The first budget is also the gateway's startup/warmup cap. It must
permit warmup; do not infer that an arbitrarily small positive cap is operationally
usable. `--budget` and `--budget-schedule` are mutually exclusive. With neither,
the fixed budget remains 96.

Predeclare the TTFT SLO and, for the joint objective, an explicit E2E SLO. There is
**no default E2E threshold**. Both deadlines start at the scheduled arrival:

```text
TTFT = first_content_at - scheduled_at
E2E  = finished_at - scheduled_at
joint success = complete successful stream AND TTFT <= TTFT_SLO AND E2E <= E2E_SLO
goodput = joint successes / arrival-window seconds
success fraction = joint successes / all offered requests
```

Equality meets a deadline. Rejected requests, incomplete streams and client
failures are not SLO successes, even if they terminate quickly. E2E is client
completion, not engine last-token time or gateway cleanup. Without `--e2e-slo`,
the existing successful-completion-plus-TTFT definition remains unchanged.
Historical captures are not retroactively joint-SLO experiments.

For a **later authorized** replay against an already running backend, first set
the product deadlines and approved file/directory paths:

```bash
: "${TTFT_SLO:?Predeclare TTFT seconds}"
: "${E2E_SLO:?Predeclare E2E seconds}"
: "${TRACE:?Prepared trace.json}"
: "${BUDGET_SCHEDULE:?Approved schedule JSON}"
: "${OUTPUT_DIR:?Fresh output directory}"
python run_trace_pilot.py \
  --trace "$TRACE" --output-dir "$OUTPUT_DIR" \
  --budget-schedule "$BUDGET_SCHEDULE" \
  --ttft-slo "$TTFT_SLO" --e2e-slo "$E2E_SLO" \
  --max-scheduling-lag 0.05 --max-budget-lag 0.05 \
  --engine-observations --sampler-mode process --cache-policy disabled
```

Process sampling is opt-in; use only a natively accepted configuration and keep
observation mode identical across comparisons. The command does not establish
backend cache configuration; retain actual backend launch evidence. Do not run it
as part of preparing or reviewing this PR.

## Timing, failure and audit contract

Before warmup, each new CLI run writes `replay_plan.json`, binding the normalized
budget schedule, trace SHA-256, duration, SLOs and both lag limits. The manifest
records `control_protocol="trace-identification-v1"` and the plan's SHA-256.
Keep the plan with the raw capture. These are consistency/provenance checks, not a
digital signature proving that nobody edited and rehashed the files.

The initial command is acknowledged before the shared monotonic measurement
origin. Every subsequent target is `measurement_start + at_seconds`, not a sleep
relative to the preceding acknowledgement. Arrival scheduling and prepared
request IDs are unchanged. The last budget is held through drain.

The existing budget endpoint is reused. Commands are serial, with no retries.
`budget_changes.json` records each attempted/waiting command's index, requested
value, target, send time, acknowledgement and observed application/version.
Unattempted steps remain in the declared plan; they are not fabricated as sends.
Failures/cancellation retain their status and error type. A request cancelled
while awaiting HTTP may already have applied at the gateway: missing
acknowledgement means **unknown/unqualified**, never "not applied."

A missed send/application window, malformed acknowledgement, HTTP failure, or a
previous acknowledgement crossing the next target stops control and cancels
replay. Replay failure also cancels control. Overdue commands are not replayed as
a catch-up burst. Cancellation after measurement starts retains every planned
request outcome, including unlaunched/cancelled rows, before gateway termination.

The separate `control_fidelity` verdict reports send/application lag, acknowledgement
round trips and planned/recorded/applied counts. `--max-budget-lag` defaults to
50 ms and limits **send and application lateness for noninitial commands**;
it is independent of the client's `--max-scheduling-lag`. Initial command lag is
not measured against a future replay start. Acknowledgement RTT is reported
separately, and acknowledgements must precede the next control target.

Core integrity checks bind exact command count/order/values/targets to the saved
plan, consecutive gateway versions from a fresh gateway, previous budgets,
application events, causal send/application/acknowledgement ordering, snapshots,
admission decisions and final budget. Partial/missing declarations cannot silently use
the legacy audit. CLI success requires integrity, load fidelity, control fidelity
and every enabled diagnostic gate; offline comparison qualification uses the same
gates. A gate failure does not establish model or policy performance.

Both budget plots use **actual application times**, not an assumed constant cap.
An unavailable or malformed control curve is not replaced with an invented
fixed cap; both primary and offline plots label it unavailable without replacing
the audit report.
With an E2E deadline, performance and input-size reports use joint goodput and
also expose marginal TTFT/E2E successes and scheduled E2E P99. The offline ledger
adds `e2e_slo_met`, `joint_slo_met` and E2E slack at admission. Recorded failures
cannot meet the SLO; missing/inconsistent outcomes or necessary timing remain
unknown unless a known deadline miss already rules out joint success. Existing
TTFT outcome labels remain intact. One-second `cohort_slo_*` counts are attributed
to **scheduled arrival cohorts**, including rejection/failure/unknown outcomes,
not to completion-time throughput. The plan is included in immutable-input
hashing and temporary audit copies.

## Staged GPU protocol: do not start with a matrix

| Stage | Smallest next experiment | Gate before proceeding |
|---|---|---|
| 0: native CPU | Run the shared [Linux acceptance entry point](runtime-telemetry.md#cpu-validation-and-handoff) on the exact clean revision, before starting services. | Complete suite, all five runner configurations, no failures/errors/skips; retain commit-bound evidence. Added test source and compilation are not this gate. |
| 1: measurement | One fresh fixed-budget smoke, then one full target-trace baseline at the established starting cap (currently 96). | Integrity, load/control fidelity and enabled diagnostics pass; verified backend/model/cache/host provenance. Preserve the 50 ms client gate. Stop on failure. |
| 2: operating range | One lower candidate on the same trace, for example 64 only if safe. | Inspect joint all-offered goodput, rejection/failure, both deadlines, per-input-size outcomes, actual output lengths and engine state. A lower cap is not presumed better. |
| 3: excitation | One predeclared down/up schedule within the qualified range, with holds informed by observed response/drain. | Both directions are observed; commands/versions qualify; enough transitions/holds expose a repeatable response rather than only a change in arrivals. Otherwise authorize a targeted follow-up, not an automatic grid. |
| 4: model acceptance | Fit offline; validate on untouched chronological windows and held-out control sequences. | The model gates below pass before any shadow or closed-loop claim. |

Each GPU stage needs separate approval and a fresh output directory. Freeze the
commit, trace bytes, model/launch flags, cache reset policy, host allocation,
instrumentation, SLOs, lag limits and run order. Record warmup and drain separately.
Retain failing runs rather than changing thresholds or removing late arrivals.
If settings change, start a new protocol series.

An isolation A/B is optional evidence about observer overhead, **not an obligatory
large campaign before identification**. If the chosen configuration cannot produce
qualified offered load, investigate that blocker. A separate sampler process
does not create CPU capacity or prove negligible observer overhead.

One budget pair or one down/up run is exploratory. Repeat/counterbalance a small
number of runs if variation or correlation between arrival bursts and command
times prevents attribution. Use run-level variation, not thousands of dependent
requests as independent repetitions. Do not claim an optimal cap or a transferable
model from one trace, one GPU or the current fixed output cap.

## What "ready for MPC" must mean

Collecting a qualified response is the start of identification, not the last MPC
acceptance step. Require multi-step forecasts under held-out inputs, not only
one-step fit, before treating a model as suitable for control.

Before touching held-out evaluation, freeze a model-validation protocol containing:

| Required declaration | Acceptance evidence |
|---|---|
| State/features, control input, disturbances and prediction targets | Use observed budget **application** history, request mix and causal state. Predict admission/completion and joint deadline behavior, not latency after filtering away rejected requests. Missing engine/CPU evidence remains unknown. |
| Sampling/control period and prediction horizon in seconds and steps | Compatible with observation age, actuator RTT and measured response/drain times. One-second aggregate samples cannot be claimed as per-request GPU timing or finer-grained observations. |
| Operating envelope and chronological train/validation/final-test split | Include held-out load windows and unseen control sequences; no shuffled neighboring seconds. Existing smoke/experiment presets are not already held-out validation. |
| Numeric prediction-error and optimistic-tail limits at the chosen horizon | Report errors in physical units, especially underprediction near TTFT/E2E deadlines, both step directions and relevant input-size cohorts. Compare with simple persistence/static baselines. Select limits before evaluation; none is established by this PR. |
| Uncertainty/calibration and adequacy criteria | Enough repeats/coverage to support the intended safety margins; show the error distribution and calibration relevant to the chosen output model, not only an aggregate training score. |
| Stale/missing-state, out-of-range and deadline-failure behavior | Specify a qualified fixed-budget fallback, bounded budget/rate changes, and decision/actuation deadlines before online work. |

Future arrivals, historical `GeneratedTokens`, eventual output lengths and
post-decision observations are not online features. Training labels may use
eventual outcomes, but validation features must respect when observations were
actually available, including collection/delivery delays. The current offline
sampler files do not implement an online feature-delivery channel.

If numeric limits, a held-out set or operating-range coverage are missing, model
readiness is **not established**. The [offline feature and forecast-baseline
tooling](forecast-baselines.md) builds the causal table and persistence/static
comparisons; development captures alone cannot establish readiness.
[Excitation v1](excitation-v1.md) declares the stage 3 schedules, a held-out
window and schedule, and forecast limits frozen on 2026-10-17 before collection.
Model fitting, an instrumented shadow-mode
feed/solver and a later protected closed-loop comparison with a qualified fixed
policy are separate work. Shadow mode must not actuate; closed-loop experiments
require their own approval and safeguards. This PR supplies neither controller
code nor evidence of MPC benefit.
