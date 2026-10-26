# Async pipeline model: proposed next protocol

Status: unfitted CPU predictive prototype now implemented separately in
prefill_lab/async_predictor.py; see prefill-async-predictor-v1.md for runnable
commands, approximations and explicit limits. It is not an accepted replacement
for the current serial fitter. The uploaded 6edb3e3 foreground diagnostic motivates this
proposal. Keep the serial overlap guard, current statistical thresholds and all
GPU/model acceptance flags unchanged. Review and freeze a new collection/analysis
protocol before collecting fresh confirmatory data.

Implementation update: the separate `prefill_host.py stages` development command
now implements stage instrumentation and integrity checks under the frozen
prefill-stage-diagnostic-v1.json plan. The uploaded ab24416 run passed native stage
integrity for four trials/1056 steps after recorded927-test CPU acceptance; its
two traces attribute all4421 GPU activities to forward/logits/copy/update leaves
and include fifth-call token/output work. See
../results/prefill-stages-ab24416-analysis/report.md. This is a narrow measurement
milestone, not full CUDA-scope acceptance. The separate CPU prototype implements
the predictive resource graph and two-slot scheduler/placeholder rollout. Stage
coefficient fitting and physical-model acceptance remain unimplemented. The
legacy diagnose/full-suite collector does not enable stage instrumentation.

## Problem and evidence

The b04477e complete suite observed AsyncScheduler with two batches in flight.
Its schedule-to-output wall intervals overlap and cannot become serial additive
costs. The subsequent 6edb3e3 diagnostic verifies foreground worker steps11–15 in
two cases. Launch correlations associate GPU work with these exact steps and
also show logits/sampling/update kernels and DtoH copies outside the forward
hook. A GPU forward duration therefore does not predict first-token completion
by itself. See ../results/prefill-foreground-6edb3e3-analysis/report.md.

These two development traces include a held composition and are mechanism
diagnostics, not calibration rows, noise estimates, a family-selection dataset or
fresh held validation. Do not reuse their timings to tune an acceptance rule.

## State and resource graph

Track the FCFS waiting queue; each request's prompt length, scheduled/computed
positions, native async placeholders, output count and limit; block-rounded KV
occupancy; and each occupied batch slot with its stable step ID and outstanding
dependencies. Keep scheduled progress separate from output committed to the
scheduler. A request may have speculative *scheduled* progress under the ordinary
async placeholder mechanism even though speculative decoding is disabled.

Represent CPU submission, GPU input preparation/forward, logits/sampling, async
output copy, GPU state update and CPU output commit as distinct graph nodes.
Preserve the native V2 ordering: `sample_tokens` starts async output copy before
`postprocess_sampled`. Copy completion and GPU state-update completion are
different dependencies. DtoH can overlap later main-stream work. Preserve stream
FIFO order and native event waits; do not put all work on an invented serial lane.

For a node v with a frozen, nonnegative duration prediction d(v), use:

```
start(v) = max(ready(v), finish(each dependency of v), available(resource(v)))
finish(v) = start(v) + d(v)
```

This event-graph rule is implemented by the separate CPU prototype, without a claim that the
entire engine follows a simple two-stage queue. CPU enqueue spans can overlap the
GPU work they launch. In particular, do not use worker CPU-return time as a
blanket prerequisite for GPU forward start; the new traces contradict that
assumption. Split enqueue work where required to represent those dependencies.
The installed dispatcher must still choose mode/padding from predicted state;
observed held mode, progress and completion times must not enter prediction.

The two native batch slots constrain schedule/submit/reap behavior. Mirror the
reviewed EngineCore queue and placeholder rules, rather than assuming that every
slot is always busy or that GPU kernels from both slots overlap. Queue occupancy,
CPU/GPU concurrency and measured simultaneous GPU activity are distinct facts.

Predict engine TTFT at the first output commit for the request, relative to its
recorded arrival. HTTP/gateway TTFT additionally needs transport/admission stages
and its own qualification. A forward finish is not an output commit.

## Measurements required before fitting

1. Preserve existing work/position/dispatcher/source guards and schedule-to-commit
   intervals. Add stable step-bound sample/postprocess/copy/commit identities;
   fail on missing, duplicate, reordered or mismatched stages.
2. Record CPU enqueue boundaries and delayed, stream-local CUDA events around
   forward and token/output work. Cross-stream copy completion needs its own
   event and explicit dependencies. Do not synchronize the device per step.
   Elapsed event pairs alone have no common-clock start/end timestamps, so retain
   deep CUDA launch correlations to qualify dependency and event coverage.
3. Extend deep capture through the fifth call's token/output work. The legacy
   diagnose profiler stops at that call's forward return; opt-in stages now stops
   after its sample_tokens return and retains stop/export timestamps. Keep stopping/export
   outside performance-scored windows and document its synchronization cost.
4. Cover every runtime mode/padding and pipeline stage used in a claim. The earlier
   6edb3e3 upload lacked the fifth sampling tail; the ab24416 stage upload includes
   it. Both profile samples contain nine NONE calls and one FULL call, with no
   profiled PIECEWISE call. Neither qualifies other modes or all possible streams.

The ab24416 trace makes a resource-model distinction concrete: the copy_enqueue
event starts before the native copy-stream wait. Its15.140–154.043ms recorded
spans include waiting, whereas the two small DtoH transfers per profiled call
occupy3.648–7.072us envelopes. A predictor must not use that wait-inclusive span
as a copy-service node *after* waiting on logits readiness: it would charge the
same dependency twice. It must also exclude the nested sample_tokens parent when
separately costing logits/sample and post-update. CPU submission, GPU execution
and output-reap times remain distinct. These are label-scope findings, not fitted
costs; do not choose coefficients from the held-composition development trace.

Forward feature candidates can retain M0/M1/M0Z/M2's actual prefill, decode,
decode-context and attention-pair features, plus supported mode/padding terms.
Token/output stages require their actual logits rows, sampled requests, copy
bytes and update work; the number of foreground requests alone is insufficient.
Inspect rank/condition and support separately for each stage. CUDA event spans
can include device idle time while the CPU submits work; do not call them pure
kernel compute costs or add CPU and CUDA spans with overlapping scope.

## Calibration and validation

Use only declared calibration cells/repeats to derive stage features, scales,
coefficients, uncertainty and family selection. Freeze the new protocol and
all policies before its collection. The old b04477e archive lacks complete
step-bound stage labels and must not be relabeled into a passing pipeline fit.
Separate whole-case/repeat validation from training; preserve raw failures and
report unsupported modes and block dependence. Do not infer repeatability from
the two deep/light development comparisons.

Require a causal rollout to reproduce arrivals, scheduling work, mode/padding,
slot occupancy, per-request first outputs and later output progression using its
own state. A replay that consumes observed held completion timestamps is a trace
reconstruction, not a predictor. Validate both per-stage errors and end-to-end
TTFT under predefined engineering/statistical criteria. Review those criteria
with the new protocol; do not relax them to pass the existing measurements.

Only after this qualification should cap policies be compared. Keep gateway
admission cap, vLLM batch token budget2048, and output-token limit separate.
Compare a frozen fixed-cap policy against a causal state-dependent policy on the
same offered workload; count rejections and failed offers in the objective and
retain TTFT SLO, timing fidelity, source identity and repeated-run checks. Pipeline
simulation, physical-model acceptance, adaptive-cap benefit and MPC benefit are
four separate milestones; none is established by this diagnostic.

## Source boundary

Reviewed vLLM tag: v0.30.0. The captured installed35-file source map is verified
against that capture's Git identity. Relevant upstream code:

- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/model_runner.py
  (`execute_model`, `sample`, `sample_tokens`, `postprocess_sampled`).
- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py
  (native async batch submission/reaping).
- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/async_scheduler.py
  (scheduled placeholders and output accounting).

Any additional hook or async-output file must be inspected and source-pinned
before its collector implementation is allowed to run. The separate stage
diagnostic adds reviewed source pins for AsyncOutput, InputBatch and SamplerOutput.
It changes no engine setting, fits no stage costs and grants no model acceptance.
