# Prefill service identification v1 — revised after PR32 review

**Draft design, revision3. Collector/hooks are not implemented; this is not an executable
GPU protocol.** Parent evidence: PR26–31. The cases, order, predictions and budget are
reproducible CPU artifacts. Freeze accepted collector, installed-source mapping,
analysis and token manifests before GPU collection. C remains unused.

## Review decisions and CPU evidence

Revision3 responds to the [round-2 review](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6330431494).
It keeps round1's source mapping, matched D16, extreme-length calibration, observer
qualification and separation of conditional fits from causal forecasts.

| Round2 item | Revision / qualification |
|---|---|
| 1: ITL mechanism check | Use2048 as the historical working assumption,8192 as sensitivity. Add reproducible accepted-path ITL counts and conditional transfer predictions. Pooled goodput MAE is a different test. Tail agreement does not establish old config. |
| 2: nonlinear sensitivity | Include k=1e-9/1.5e-9 in historical ITL checks and1.5e-9 alongside3.15e-9 in cell planning. Neither is a fitted physical bound. Single2048 is low-signal validation, not an M2 identification cell. |
| 3: wasted background drain | Reduce background output512 to256. All five declared toy scenarios preserve foreground execution through completion; record real release state and require backgrounds to outlive foreground in measured trials. Bank savings. |
| 4: weak sentinel cell | Replace one4096@200 with one7168@200. Four4096@800 is a FIFO/RR tie control, not a required policy-discrimination win. |
| 5: wrong contrast / lucky wins | Publish sentinel-only, per-request and within-D offset contrasts. Freeze contrast eligibility using calibration-only jitter and the frozen candidate; do not count unresolved comparisons in a6/8 tally. |
| 6: M1 selection | Evaluate decode-only improvement for M1 separately; M2 targets prefill-bearing and cumulative timing. A small share of total runtime does NOT imply a small relative MAE improvement. |
| 7: empty transfer boundary | Replay[40,120) and[160,240), score[60,120) and[180,240). Log boundary running/queue/KV state;20s is a declared lead-in, not demonstrated historical-state reconstruction. |
| 8: stage completion | Separate main/bridge/transfer/deep completion and gates; full suite still requires all stages. Later failures invalidate prior results only if shared integrity/observer assumptions are compromised. |

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
Thus the association is not unique evidence for2048. ITL tails test a more direct
consequence of the joint running-first/step-time hypothesis than pooled goodput MAE.
With every live decoder producing one token each step, a long prefill step should
leave long token gaps. [Reproducible ITL scenarios](../results/prefill-design-round2/itl-scenarios.json)
use actual admitted paths/output lengths and compare them with observed exporter tails.
They support2048 as a working assumption, not a verified old config. Async/counter
semantics and output observation still need source/step reconciliation. No preferred
tail is forced as a capture-validity gate; matching it does not identify a unique model.

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
new rental. Backgrounds plus foreground/sentinel can approach38k tokens before
block rounding. A failed capacity check stops collection; do not call it safe by assumption.

## Workload and splits

Actual engine input token IDs define length, including templates/special tokens.
Materialize/hash all trial content before freeze, content seed20261024. Foreground
and sentinel outputs are forced128 tokens, background outputs256, temperature0,
ignore EOS. This is a mechanistic workload; the later Azure slices retain natural
EOS and the historical128 output cap. All requests must fit context8192.

D0 means no background. D16 means16 requests with1024 input tokens, released first;
wait until all are decoding before the foreground barrier. Record their actual
context and remaining output at release and every subsequent step. D16 is an
initial target. In microbenchmarks, require all16 backgrounds to remain decoding
until all foreground/sentinel requests complete; otherwise mark that cell's intended
matched-load test invalid, preserve it and do not silently extend the output length. Drain all
work before each trial;250ms idle gap; external traffic prohibited.

| Family | Conditions at each of D0/D16 | Cells | Split |
|---|---|---:|---|
| Single |1024/2048/4096/7168 input |8|1024/7168 calibration;2048/4096 held |
| Composition |4x4096;1024/1024/7168/7168; reverse order |6|Uniform calibration; mixed/reverse held |
| Sentinel |4x4096 burst,512 sentinel at50/400/800ms;1x7168 with sentinel200ms |8|All held |

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
wall times, multiplies by1.35, then adds idle gaps,160s of replay including lead-ins,
and120s frontend/profiler reserve. It estimates about1318s; this is a planning estimate, not a
measured completion-time promise. Hard active budget1800s excludes installation
and initial model load, includes every stage/flush. Per-trial timeout45s. No automatic
retry, shape removal, opportunistic extra level or resizing a partial suite into a
successful one. If on-host preparation predicts insufficient margin, stop and
version a smaller plan before scoring. An overrun preserves an incomplete attempt. Each stage records planned/attempted/
completed trial IDs, integrity status, deadline and reason for any skipped/failed work.
An independently complete main stage can retain its scoped findings if transfer/deep
runs out of time. It cannot be called a complete suite; shared observer/config/data
failures invalidate every affected stage. No failed main cell is dropped from a pass.

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

Fit GPU-span and valid iteration-wall targets separately. Report three target classes:
(a) decode-only step MAE, (b) prefill-bearing step MAE, and (c) per-request cumulative
release-to-last-prefill wall MAE. The last includes all intervening steps, not just
that request's chunks. Case-weighted MAE prevents hundreds of decode-only steps
from diluting prefill evidence. Conditional scores use observed executed chunks;
they are not causal workload forecasts.

Use training-only scaling and leave-one-calibration-cell-out (LOCO) errors on the
18 fitting trials. Two nested decisions, with simpler family winning ties:

1. M0 versus M1: retain M1 only if decode-only LOCO case-weighted MAE improves>=10%,
   with neither prefill target regressing by more than max(10% M0 MAE,1ms) per case.
2. Compare M2 with that retained base on both prefill targets. Retain M2 only if
   each LOCO case-weighted MAE improves>=10%, with no target/case regression beyond
   max(10% base MAE,1ms), including decode-only timing. M2 is still fit as M1+Z;
   the comparison base can be M0 if M1 alone did not justify its complexity.

Freeze the selected family, coefficients, scaling, LOCO report, jitter, causal
rollout and eligibility hashes before opening held timing. The12 repeat-check
trials diagnose calibration repeatability; they cannot select another family.
Actual q/l coverage, scaled rank/condition and held feature ranges must be reported.
Numerical interpolation is not proof of causal support. Do not retune after held failure.

Held screens mirror the frozen decision: an M1 decode-benefit claim requires>=10%
decode-only case-weighted MAE improvement versus M0 and the declared prefill
no-regression limits. An M2 prefill-benefit claim requires>=10% improvement on BOTH
prefill targets versus the retained base and no-regression on all targets/cases.
Report M0/M1/M2 ablations regardless; do not replace the selected candidate from
held results. Failed/inconclusive is not passed. M0 can remain appropriate.
All margins refer to relative prediction ERROR, not percentage of total runtime;
a3ms context effect can reduce a4ms baseline MAE by much more than10%.

**Separate causal sentinel rollout gate, eight cells, five blocks each:**

1. Per cell, abs(median(predicted-measured engine TTFT)) <=max(10% measured median,25ms).
2. Compare50/400/800ms offsets within each D level. Measured pairs with a block-bootstrap
   95% interval excluding0 and median difference>25ms are distinguishable; predicted
   ordering must agree for all such pairs. Others are unresolved/ties, not automatic
   ranking successes. No discernible ordering means policy identification is unresolved.
3. Before held scoring, roll out the calibrated frozen candidate and raw-B2048
   baseline on the predeclared eight cells. A cell is eligible only if the predicted
   sentinel-TTFT difference exceeds max(3*calibration-only jitter,25ms).
   Hash the eligibility list before reading held timing (`contrast_eligibility`).
   Require at least4 eligible cells; otherwise improvement is inconclusive. On
   eligible cells require strictly lower paired absolute-error MAE in at least
   ceil(0.75*n) cells. Report paired block errors and bootstrap intervals; this is
   an engineering screen, not a binomial significance test of independent cells.
   Every one of the eight cells, eligible or not, retains the no-regression limit
   max(10% baseline MAE,25ms). Report B8192 sensitivity too. No ties count as wins.

The causal rollout must use only release-time state, injected arrivals and its own
predicted scheduler progress. Measured future chunks/first tokens are oracle-only.
If no causal rollout exists, this gate is **not evaluated**, not passed by step-fit
accuracy. Old fitted TTFT overhead is archived separately and is not pure engine
cost. Passing microbenchmark gates nominates a simulator candidate, not an MPC policy.

## Preregistered predictions and discrimination

[Predictions](prefill-service-v1.predictions.json) include the unchanged PR28 simulator
with an explicit barrier-state adapter at B8192 and B2048: D16 backgrounds start
already decoding with255 remaining outputs; raw engine and old empirical correction
are separate. The model cannot infer their actual release state before measurement.
A variable-output CPU scenario additionally models background preparation for budgeting.
Neither adapter is represented as a new fitted physical model.

Further scenarios use unvalidated nonlinear sensitivities (1.5e-9 and3.15e-9 per full L²,
charged incrementally as2*k*Z) and a round-robin512 alternative, **not a claim about
vLLM's actual interleaving policy**. [Discrimination](prefill-service-v1.discrimination.json)
records each sentinel-only or individual-request contrast versus raw-B2048 and
each within-D offset-pair difference, for every scenario. Planning jitter bands
are10/25/50ms and resolution means a gap>3*band, not just>band. These are not measured
noise or power estimates. Before main, compute1.4826*MAD of off-mode foreground
TTFT in each calibration observer cell; use the larger value as a provisional jitter
scale, not a universal bound. Mark a contrast unresolved if its expected absolute gap is<=three times that scale. The declared identification contrasts are (H2) M2-versus-M0 request TTFT for
single7168 and mixed/reverse composition at each D, and (H3) FIFO-versus-RR
sentinel TTFT for four@50/four@400/one7168@200 at each D. Single2048 remains
validation-only; four@800 is a policy tie control (not necessarily an M2 tie).
Within-D offset ordering tests policy, not the nonlinear coefficient. Publish
resolvability for every contrast/scenario. If all identification contrasts for an
H2 or H3 claim are unresolved, that claim is inconclusive; stop main only if BOTH
families are unresolved. Do not replace cells. Candidate-versus-baseline eligibility
is a separate frozen rule after calibration fitting and before opening held timing.
This calibration-only check does not inspect held timing or justify precision on
other states. Do not swap held cells after inspecting them.

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
fixed32,80s replays [40,120),[160,240), scoring only[60,120),[180,240), from the historical development trace
SHA25662f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e,
rebased locally with original trace IDs/provenance. Natural EOS, output cap128,
original shared-client/process sampler/runner-GC settings and all enabled replay
integrity/load/control/diagnostic gates, including50ms limits, remain required.
Freeze derived trace hashes before collection. Preserve original trace IDs,
original scheduled seconds and rebased local times. Lead-in requests participate
in full lifecycle/drain audits but not scored offered counts or arrival-window
denominators. Boundary running/waiting/KV/context and actual surviving lead-in
requests are reported beside historical state; do not assume20s guarantees settling.
No C. ITL-tail/CDF comparisons and
actual chunk distributions connect to old captures, but are not old-default or
same-host policy equivalence tests. These slices are development observation;
no new model is fitted to their outcomes before reporting its frozen predictions.
The committed ITL scenario counts use historical fixed32 accepted-arrival/output
paths, selected over each80s replay, starting empty at lead-in. They are conditional
preregistration sensitivities, not predictions of fresh admission/EOS. A second
forced128-output variant shows that dependence. After capture, also run the same
frozen algorithm on its actual accepted/output path, labelled retrospective.
Score gaps by right token timestamp within(20,80] local seconds; compare exporter
scrape endpoints explicitly. A tail mismatch challenges the joint scheduling/service
scenario; matching does not establish uniqueness or pass causal prediction gates.

## Reproduce and freeze checklist

```
python scripts/build_prefill_design.py --output-dir /tmp/fresh-prefill-design
python scripts/check_prefill_itl.py --old-evidence OLD --new-evidence NEW --output /tmp/fresh-itl.json
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
