# Prefill service identification v1 — revised after PR32 review

**Draft design, revision2. Collector/hooks are not implemented; this is not an executable
GPU protocol.** Parent evidence: PR26–31. The cases, order, predictions and budget are
reproducible CPU artifacts. Freeze accepted collector, installed-source mapping,
analysis and token manifests before GPU collection. C remains unused.

## Review decisions and CPU evidence

| Review item | Revision |
|---|---|
| 1: test2048 before GPU | Completed the six-capture,288-window frozen-parameter sensitivity. Capture gates re-audit;8192 predictions reproduce the archive.2048 does not repair the pooled error. Keep both as preregistered scenarios. |
| 1: establish compatibility | Preserve OpenAI-server usage context. Pin2048 for the new experiment, archive resolved config/source/launch. Compare short replay ITL tails descriptively; zero gaps above0.2s is not a validity gate or proof of old config. |
| 2: calibration support | Train on single1024/7168 and uniform4x4096, at both background levels. Hold2048/4096, mixtures/orders and all sentinels out of fitting. Check actual chunk-position coverage/rank. |
| 3: realistic decode pressure | Replace D8 with D16:16 backgrounds,input1024/output512. Record actual counts/context; target about16–24k background context. D9 is deferred to keep the budget fixed. |
| 4: sentinel discrimination | Replace staggered arrival cell with a four4096 burst and sentinel offsets50/400/800ms; add one4096 plus sentinel200ms. |
| 5: prefill-sensitive targets | Primary: prefill-bearing step timing AND per-request cumulative release-to-last-prefill time. Decode-only steps are a separate no-regression check. |
| 6: measurable rollout gate | Declare latency-error, tie-aware ordering and6-of8 improvement checks below. No required preferred cap or named segment. |
| 7: preregister predictions/power | Commit8192/2048 frozen-barrier predictions, nonlinear sensitivity and round-robin alternative. Compare contrast sizes with10/25/50ms planning bands; actual noise is unknown until GPU calibration. |
| 8: normal endpoint bridge | One block of all eight sentinel cases through the shared-client gateway and normal OpenAI server, with per-request cross-process joins. |
| 9: transfer slices | Two fixed32,60s development slices are scheduled, not adaptively added. One warmup pass and two deep traces keep the recomputed proposal within1800s. |
| Minor: overhead/drift | Mean-log-ratio t interval;20 cheap and10 demanding pairs. Fixed reference before/after each main block. Actual step work comes from scheduler/execution records, not iteration-token histograms. |

The verified vLLM v0.30.0 `EngineArgs.get_batch_defaults` source sets API-server2048
versus LLM-class8192 in its smaller-GPU branch. This is an upstream default, not
proof of the historical wheel/config or its overrides. Bypassing HTTP must not
instantiate an offline LLM with different defaults. Use one OpenAI-server engine
with an engine-side enqueue barrier instead.

### Completed development sensitivity

| Unchanged service/overhead parameters;288 common windows | Goodput MAE | Cap32 bias |
|---|---:|---:|
| B8192 |0.7290|+0.3163|
| B2048 |0.7423|+0.3816|

See [CPU sensitivity](../results/prefill-budget-sensitivity/summary.json). Only B
changes; the original causal bias rule is recomputed from closed history per B.
The old model restarts pending prompts from full input and uses approximate decode
progress. A worse2048 score cannot reject2048 as the real historical setting.
The additional short-miss quartile diagnostic forces observed admissions from the
new-host fixed32 capture and uses raw service only; it is not an online prediction
or an isolated causal test of the scheduler. Its scope/limitations are in the JSON.
The forced-admission quartile diagnostic already shows increasing short-miss rates
at8192 (0.4/6.3/17.5/52.5%);2048 gives0/0/12.5/59.2%, versus actual0.8/3.8/15.4/63.3%.
Thus the association is not unique evidence for2048. An ITL histogram is not a per-step work trace, and frontend/async effects can alter
which gaps are observed. We do not infer a unique scheduler budget from a0.2s bound.

## Questions and environment

Separate (H1) work/counter timing, (H2) chunk/context-dependent service cost, and
(H3) scheduler backlog ahead of short requests. A null nonlinear result is useful.
Record per-step evidence before deciding which simulator assumptions to change.

One RTX4090, one loaded OpenAI-server EngineCore, Qwen/Qwen3-4B-Instruct-2507,
Python3.12, vLLM0.30.0, Torch2.13.0+cu130. Pin model/tokenizer/package/source revisions.
New launch explicitly uses max_model_len8192 and max_num_batched_tokens2048;
record max_num_seqs (must support at least21 requests), chunked/partial-prefill
settings, thresholds, dtype/attention backend, graph/compile/async settings, KV
capacity, prefix cache, speculative decoding and TP. Require prefix cache off,
speculation off, TP1. Keep recorded normal graph/async behavior; no silent eager
or serialization workaround. Archive literal launch argv and resolved config.

Before first trial, bind actual host/GPU UUID, driver/CUDA, CPU quota/affinity,
clock/power/temperature, config and installed-source hashes to the manifest.
Fail unexpected runtime/config rather than choosing settings from observed latency.
Calculate peak block-rounded KV demand from inputs plus full output caps and
require at least10% capacity headroom. The old86k KV capacity is not assumed on a
new rental. Backgrounds plus foreground/sentinel can approach42k tokens before
block rounding. A failed capacity check stops collection; do not call it safe by assumption.

## Workload and splits

Actual engine input token IDs define length, including templates/special tokens.
Materialize/hash all trial content before freeze, content seed20261024. Foreground
and sentinel outputs are forced128 tokens, background outputs512, temperature0,
ignore EOS. This is a mechanistic workload; the later Azure slices retain natural
EOS and the historical128 output cap. All requests must fit context8192.

D0 means no background. D16 means16 requests with1024 input tokens, released first;
wait until all are decoding before the foreground barrier. Record their actual
context and remaining output at release and every subsequent step. D16 is an
initial target, not an assertion that the decode population remains16. Drain all
work before each trial;250ms idle gap; external traffic prohibited.

| Family | Conditions at each of D0/D16 | Cells | Split |
|---|---|---:|---|
| Single |1024/2048/4096/7168 input |8|1024/7168 calibration;2048/4096 held |
| Composition |4x4096;1024/1024/7168/7168; reverse order |6|Uniform calibration; mixed/reverse held |
| Sentinel |4x4096 burst,512 sentinel at50/400/800ms;1x4096 with sentinel200ms |8|All held |

Composition cases have four requests and16384 total prompt tokens. This controls
offered work/count, not actual step composition. Preserve common content pools
within a block, including request IDs/arrival order mapping. Insert simultaneous
requests at one engine-side barrier; let the real scheduler choose chunks.
Sentinel target and actual enqueue times are logged; maximum offset error5ms.
The case/order files contain22 cells, five randomized complete main blocks, seed20261022.

Calibration: six cells x blocks1–3 =18 fitting trials. The same cells in blocks4–5
are12 repeat checks. Other16 cells x five blocks =80 held-case trials. Warmup/bridge
and held timings must not be used to choose model features or coefficients.
These controlled-case checks are not unseen-production or MPC acceptance.

## Execution and time budget

| Stage | Count | Role |
|---|---:|---|
| One warmup pass |22|Cover each shape/background; unscored |
| Observer off/light |60|20 pairs single1024-D0,10 pairs uniform-D16 |
| Main |110|Five22-cell blocks |
| Drift references |10|Single1024-D16 before/after each main block |
| HTTP sentinel bridge |8|One randomized block; diagnostic |
| Deep traces |2|Single7168-D16 and mixed-D16; five steps from foreground release |
| **Total trials** |**212**|Plus the two60s transfer slices |

Order: warmup, observer check, main+references, HTTP bridge, transfer slices, deep
traces. Main blocks are not reordered based on outcomes. Deep traces are last so
profiler flush/clock effects cannot contaminate later performance stages.

[Budget calculation](prefill-service-v1.budget.json) sums the frozen linear scenario
wall times, multiplies by1.35, then adds idle gaps,120s of replay and120s frontend/
profiler reserve. It estimates about1777s; this is a planning estimate, not a
measured completion-time promise. Hard active budget1800s excludes installation
and initial model load, includes every stage/flush. Per-trial timeout45s. No automatic
retry, shape removal, opportunistic extra level or resizing a partial suite into a
successful one. If on-host preparation predicts insufficient margin, stop and
version a smaller plan before scoring. An overrun preserves an incomplete attempt.

## Observer, drift and timing contracts

Observer modes retain identical minimal request/release logging. Light additionally
records bounded per-step data and CUDA events; deep adds short profiling. Flush
buffers after drain, fail overflow/drop/reconciliation errors. Off does not pretend
to have step evidence. All modes retain request outcomes and release background state.

For each observer cell and each of trial wall time/foreground TTFT, form paired
log(light/off) ratios. The one-sided95% Student-t upper limit on their mean,
converted with expm1, must be<=5%. Use20 cheap/10 demanding pairs as declared,
balanced off-first/light-first and identical paired content. Record raw ratios,
variance and assumptions; this is a geometric-mean overhead allowance, not proof
of no effect. Failed/inconclusive qualification stops before main trials.

Drift references are calibration cells only. Report before/after wall/TTFT ratios
and clocks in each block. A>10% change flags the batch as drift-affected; preserve
all trials and do not make an unqualified pooled service-parameter claim. Do not
remove the worst block or refit an unregistered clock feature after viewing held data.

Map request IDs through scheduler, worker and output with stable step/iteration IDs.
Capture pre-step prompt length/computed position, scheduled/executed/committed
prefill tokens, decode count/context, queue order, waiting/running status, preemption,
mode and timestamps. Reconcile actual input/output progress at drain. The exporter
prompt and iteration histograms are time-correlated observations, not ground truth
for executed per-step work. Trace/source-map where counters are updated.

CPU monotonic iteration wall time and stream-correct CUDA spans are separate
labels. Use delayed CUDA-event resolution, no per-step global synchronize. Confirm
multi-stream/graph scope with the two short traces; asynchronous overlapping steps
cannot simply be added into a serial service model. Mark unavailable scopes; never
substitute Python return duration for GPU time. Exact-revision CPU/mock acceptance
and adapter tests must cover partial chunks, async IDs, cancellations, duplicates,
missing events, bounded buffers and future-observation isolation before collection.

## Predictive models, primary targets and gates

Fit fixed nonnegative models, with training-only scaling:

* M0: intercept + executed decode-token count + executed prefill-token count.
* M1: M0 + summed decode context lengths.
* M2: M1 + Z=sum(q*l+q*(q+1)/2), actual prompt chunk q and pre-chunk position l.

Fit GPU-span and valid iteration-wall targets separately. Primary service scores
are (a) prefill-bearing steps and (b) cumulative per-request release-to-last-prefill
wall time. Neither is diluted by hundreds of decode-only steps. Report decode-only
error separately, with a no-regression allowance of max(10% M0 MAE,1ms).
These service scores condition on observed executed work, including held-case chunks;
they are not causal workload forecasts. Cumulative targets include all intervening
work from release until that request's final prefill chunk, not just its own chunks.

Select the family using leave-one-calibration-cell-out errors on fitting trials,
with equal case weighting and equal weight to normalized primary targets (each
scaled by that target's M0 calibration MAE, floor1ms); simpler wins ties. Freeze
candidate/source/coefficient hashes before opening held cases. Do not select a
winner from the held report. Actual q/l coverage, scaled design rank/condition and
held feature ranges must be reported; numerical interpolation is not proof of
causal support. Report unsupported regions rather than retune after the fact.

Service screen: selected context/nonlinear model must improve each primary
held-case target's case-weighted MAE by>=10% versus refitted M0, without per-case
regression above max(10% case M0 MAE,1ms). M0 may remain the appropriate model if
accounting/state changes suffice; do not force M2 to pass. These are engineering
screening margins, not physical laws or MPC guarantees.

**Separate causal sentinel rollout gate, eight cells, five blocks each:**

1. Per cell, abs(median(predicted-measured engine TTFT)) <=max(10% measured median,25ms).
2. Compare50/400/800ms offsets within each D level. Measured pairs with a block-bootstrap
   95% interval excluding0 and median difference>25ms are distinguishable; predicted
   ordering must agree for all such pairs. Others are unresolved/ties, not automatic
   ranking successes. No discernible ordering means policy identification is unresolved.
3. Paired absolute-error MAE must beat the preregistered frozen-B2048 **raw engine**
   baseline in at least6/8 cells; report B8192 too. No cell may regress by more than
   max(10% baseline MAE,25ms).

The causal rollout must use only release-time state, injected arrivals and its own
predicted scheduler progress. Measured future chunks/first tokens are oracle-only.
If no causal rollout exists, this gate is **not evaluated**, not passed by step-fit
accuracy. Old fitted TTFT overhead is archived separately and is not pure engine
cost. Passing microbenchmark gates nominates a simulator candidate, not an MPC policy.

## Preregistered predictions and discrimination

[Predictions](prefill-service-v1.predictions.json) include the unchanged PR28 simulator
with an explicit barrier-state adapter at B8192 and B2048: D16 backgrounds start
already decoding with511 remaining outputs; raw engine and old empirical correction
are separate. The model cannot infer their actual release state before measurement.
A variable-output CPU scenario additionally models background preparation for budgeting.
Neither adapter is represented as a new fitted physical model.

Further scenarios use an unvalidated nonlinear sensitivity (3.15e-9 per full L²,
charged incrementally as2*k*Z) and a round-robin512 alternative, **not a claim about
vLLM's actual interleaving policy**. [Discrimination](prefill-service-v1.discrimination.json)
records expected contrasts and planning noise bands10/25/50ms. These are not measured
noise or power estimates. Before main, compute1.4826*MAD of off-mode foreground
TTFT in each calibration observer cell; use the larger value as a provisional jitter
scale, not a universal bound. Mark a contrast unresolved if its expected gap is less
than three times that scale. If all intended primary contrasts are unresolved, stop.
This calibration-only check does not inspect held timing or justify precision on
other states. Stop if the
intended distinction cannot be measured; do not swap held cells after inspecting them.

## Endpoint bridge and real-composition transfer

For all eight sentinel cases, use normal shared-client gateway -> OpenAI API ->
EngineCore. Gateway baseline mode admits all (backgrounds included); this stage
measures transport, not cap policy. Record scheduled/actual send, API/gateway receipt,
EngineCore enqueue, first engine output, gateway first content, SSE send and client
first content, with request IDs and host monotonic clocks. If a scope is not observable,
mark it unavailable rather than merge spans. HTTP cannot guarantee simultaneous
engine enqueue; retain actual arrivals and state. Do not subtract unrelated percentiles
or assume a matched barrier/HTTP state. No held bridge delay becomes a fitted predictor.

Restart only the gateway between transport baseline and cost-aware transfer; keep
EngineCore/config unchanged and bind each gateway identity separately. Then run two
fixed32,60s slices [60,120),[180,240) from the historical development trace
SHA25662f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e,
rebased locally with original trace IDs/provenance. Natural EOS, output cap128,
original shared-client/process sampler/runner-GC settings and all enabled replay
integrity/load/control/diagnostic gates, including50ms limits, remain required.
Freeze derived trace hashes before collection. No C. ITL-tail/CDF comparisons and
actual chunk distributions connect to old captures, but are not old-default or
same-host policy equivalence tests. These slices are development observation;
no new model is fitted to their outcomes before reporting its frozen predictions.

## Reproduce and freeze checklist

```
python scripts/build_prefill_design.py --output-dir /tmp/fresh-prefill-design
python scripts/check_prefill_budget.py --suite-dir NEW/results/low-cap-switch-v1 --old-suite-dir OLD/results/cap-range-v1 --output-dir /tmp/fresh-budget-check
```

The plan generator reproduces cases/order/predictions/discrimination/budget. The
budget sensitivity re-audits the archived suite and checks8192 parity. Raw bundles
are the public PR30/31 evidence; see their extraction instructions. Published
CPU results are development sensitivity, not a native collector/GPU acceptance.

Next implement/review the version-specific server adapter, collector, trace slicing,
metric update-site mapping, exact token manifests and causal rollout. Run CPU
acceptance at that exact revision, bind host config, verify budget/observer and
freeze an executable protocol. Afterwards compare revised simulator on Azure dev
captures; independent repeated fixed/reactive/MPC evaluation remains separate.

Sources: [v0.30.0 defaults](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py),
[scheduler output](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/output.py),
[profiling](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/),
[CUDA async timing](https://docs.pytorch.org/docs/stable/notes/cuda.html).
