# Prefill service identification v1 — revised after PR32 review

**Draft design, revision9. Collector/hooks are not implemented; this is not an executable
GPU protocol.** Parent evidence: PR26–31. The cases, order, predictions and budget are
reproducible CPU artifacts. Freeze accepted collector, installed-source mapping,
analysis and token manifests before GPU collection. C remains unused.

## Noise-aware structural correction (revision9)

[Noise report](../results/prefill-design-round7/report.md) and
[seed-level results](../results/prefill-design-round7/noise-screen.json) address
[Round5](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6351586487).
The [follow-up review](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6352203816)
identified that pure paired-t admits deterministic1us gains when SD=0. Revision9
adds a frozen5ms minimum mean improvement, preserving both structural groups and
all dependence/regression checks. Revision8 workloads, counts and budget remain
unchanged; old frozen plans without the floor are rejected. The floor is an
engineering choice, not a statistically calibrated minimum benefit.

Revision7's four eligible cells represented only one composition effect, and its
noise-free feasibility check was not a power analysis. Reusing the eligibility
margin as the required MAE gain also discarded useful paired evidence.

Use512/512/7680/7680 at50/400ms and one7680 at200ms. Keep uniform4x4096 at800ms
as a control. Mixed bursts still total16384 input tokens. There are still22 cells,
212 trials and six calibration cells; the updated planning estimate is1518.29s.
The7680 tests extend beyond the7168 calibration maximum, so this is bounded length
extrapolation. Actual mode/feature support and exact token manifests remain gates.

Keep per-cell eligibility >max(3*jitter,25ms), but require BOTH fixed structural
groups: mixed_burst and single_long. Average paired error gains across group members
within each block FIRST, then use five block means. For BOTH groups require
mean gain >max(t(.95,4)*sample_SD/sqrt(5),5ms) against every required comparator;
all eight cells retain regression guards. Offsets and D levels are not independent
structural contrasts. See the exact component rule below.

The synthetic k=1.5e-9 gaps are about57.2ms and30.2ms. At10ms iid held Gaussian
noise with noise-free calibration, the new screen passes98.83% of20000 draws.
This is conditional synthetic sensitivity, not measured GPU power. The single-long
gap barely clears30ms; at12–15ms the design remains INCONCLUSIVE. With1–5% training
label noise, refitting and LOCO can change selection and eligibility substantially.
The report archives480 training fits, including linear-truth negative controls.

A correlated-block midpoint null exposes about14% nominal false passes. The public
gate therefore defaults block dependence to unresolved; unresolved or violated
assumptions cannot pass. A calibration-only assessment must be specified and reviewed
before executable freeze. Five held blocks cannot establish independence, and a
non-significant autocorrelation test is insufficient. Synthetic independent blocks
are qualified by construction only. No full-experiment false-positive guarantee
or automatic model acceptance is claimed.

After the first three calibration blocks, before model and held-score freeze,
`calibration_noise_readiness` checks three repeat durations per preregistered matched
work/position/mode/padding stratum. Missing strata are inconclusive; CV>5% stops
structural claims, CV>2% is flagged. Three repeats give an imprecise engineering
screen, not a population-CV bound. Never pool different work or token gaps as
independent repeats, and never tune this screen using held timings.

`prefill_planning.py` performs training-only nonnegative fitting, LOCO family
selection and causal serial FIFO rollout for these synthetic checks. All models
use the same training rows and weighting. M0Z (M0 plus Z, without decode context)
is a mandatory reporting ablation; it is not a new selectable family. If M2 wins
against M0 while M1 was rejected, report joint-feature predictive benefit, not
unique Z attribution. Actual mode-aware feature support/rollout is still pending.

The raw contrast table remains a policy/sensitivity diagnostic. It cannot certify
structural feasibility. Before executable freeze, review calibrated-comparator
sensitivity as well; after calibration fitting, insufficient actual eligibility
means structural INCONCLUSIVE, never a threshold adjustment or held-cell swap.

Azure development transfer remains mandatory for transfer claims. The current
`prepare_trace.py` preserves retained arrivals/input lengths but synthesizes
prompts and uses output cap128 with natural EOS; source GeneratedTokens are
audit-only. Synthetic text is normal for this dataset, which does not release
prompt content. Its documented benchmark approach forces recorded output sizes;
our fixed128 policy instead changes the decode workload.
[Azure source](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md).
Realistic arrivals do not guarantee independent feature variation or faithful output work. Assess actual training support before asserting that a trace
identifies or cannot identify a mechanism. Controlled cells separate explanations;
Azure checks usefulness under mixed load. C remains untouched.

Before executable freeze, add an exact-installed-revision scheduler parity gate:
compare request IDs, chunk sizes, computed positions and completion/admission
events under matched state/config, including partial chunks and async behavior.
CPU toy-to-toy parity does not satisfy it. Collector, native adapter, mode-aware
causal rollout, exact token manifests and GPU acceptance are still pending.
Later action validation must compare caps/transitions on matched offered traces,
measure action ranking/regret and draining transients, and avoid oracle output
lengths. D16 is a decoder population, not admission cap16.

Earlier round reports below are historical. Revision6 and revision7 generator/case snapshots are archived
beside round4/round5 so historical source hashes remain verifiable. Their old
cell-count/margin gates are superseded; revision8's pure-t gate is also historical.
Round6 source snapshots preserve the pre-floor noise screen. Reproduce it at
commit4a33e830816cd98ea4c571263f442cb92a1cb554. Current scoring is revision9.

## Round4 changes: support, resolution and repeat references

Revision6 addresses [round4](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6331855173).
[Historical CPU report](../results/prefill-design-round4/report.md) supersedes round3's
floor5 headline table. Old reports remain historical snapshots.

| Review point | Change |
|---|---|
| Floor-driven ranking | Report zero-bin support separately; headline magnitude uses5 equivalent toy steps, with a window-specific frozen linear-B2048 gaps/step reference shared by every scenario. Common-positive bins and floors1/5 are sensitivity only. |
| Clustered gaps | Archive toy decode-bearing steps, per-bin step counts and gaps/step. These are not observed real steps or independent sample sizes. Future uncertainty uses step/time-block resampling, never independent gaps. |
| Repeat noise | Show one same-host pair and two cross-host descriptive references beside every model window. One pair cannot calibrate a statistical envelope or factor2. Plan two attempts for each transfer window, frozen ABBA order, full drain/boundary logs and no retries. |
| Meaningful wins (historical; superseded above) | Require comparator MAE minus candidate MAE strictly greater than the SAME per-cell calibration-only eligibility threshold. Equality/near-ties are non-wins; insufficient resolved gains are inconclusive; any material regression on all8 fails. |

Adding repeat replays changes the budget:320s replay plus40s explicit terminal
reserve, rather than160s replay. Total estimated1517.78s (25.3min), including the
existing120s frontend/flush reserve, under the unchanged1800s hard limit.
The review's30.9% simulation concerns only the old win-count component, not the
full experiment false-positive rate. D0-to-D16 latency-reference mapping is a proxy:
a longer median alone does not prove conservative variance or statistical power.

## Round3 changes: full ITL distribution and fair structural comparators

Revision5 addresses [round3](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6331161574).
Historical development checks are committed before GPU collection; no new GPU run
or new fitted physical model is represented here.

| Review point | Change |
|---|---|
| Full ITL distribution | Archive EVERY exporter bucket, including the fast bins and+Inf, and score every disjoint interval starting at25ms. Use matched first/last scrape sent-time endpoints; retain nominal-window legacy counts separately. Compare both60–300s historical and the two transfer windows now. |
| No k support from one bin | Withdraw selected-tail support for1.5e-9 or any bound. The positive-k cases remain sensitivity/possible counterexamples; neighboring bins and joint scores may contradict them. |
| Recalibration versus structure | Freeze same-training-data calibrated M0, retained M1 and candidate rollouts. Structural eligibility/wins compare against calibrated M0 AND retained M1. Raw-B2048 is a transfer/staleness baseline only. Selected M0 => structural N/A. |
| Position-specific jitter | MAD across off trials at EACH request position, not across positions. Preregister cell/request-to-reference mapping from raw-B2048 latency ranges before collection; use each cell's assigned jitter in its threshold. |
| Graph/padding confounding | Per step log runtime mode, actual/padded tokens and requests, capture/replay state and execution IDs. Match mode/padding within D16 before attributing a context coefficient to KV cost; identical D level does not prove identical padding. |
| Output lengths | All five checked historical admitted paths emit128 tokens. The forced128 variant is explicitly duplicate evidence here, not an independent sensitivity. Preserve the planned natural-EOS transfer; record fresh realized lengths instead of silently changing ignore-EOS. |

[Round3 CPU report](../results/prefill-design-round3/report.md) and the complete
vectors quantify the mismatches. A joint discrepancy challenges a joint scheduler/
service/observation scenario; it does not by itself attribute the error to GPU kernels.

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

### Additional self-review (revision4)

No new reviewer comment was present after the round2 response. Two remaining
specification gaps are corrected here, without expanding the GPU trial matrix:

* Score service steps only from the foreground release barrier through the last
  foreground/sentinel completion, inclusive of its final executed step. Exclude
  background preparation and background-only drain from fitting and every service
  MAE. Those phases remain in observer whole-trial timing, full lifecycle/ITL audits
  and the collection budget. Real CPU/GPU labels use the same executed step IDs
  for membership; never compare a GPU-clock timestamp with a host-clock barrier.
  This makes the claim that background256 preserves
  the scored workload precise. Cumulative prefill targets still stop at each
  request's last prefill; request TTFT/rollout targets retain their own definitions.
* The earlier q/Z rank2 check is only a partial diagnostic. Check the full M0/M1/M2
  design matrices, including intercept, decode count, prompt count, decode context
  and Z, on each training partition. A q/Z matrix can be rank2 while the full
  model is rank-deficient. Use equal total squared row weight per calibration
  cell (weight the response too), normalize columns from that fold's training rows
  only, and report rank,
  zero columns and normalized condition number. Predeclared condition limit1e4
  is an engineering guard. Unsupported families cannot earn a fitted-parameter
  or held-benefit claim; do not drop columns or choose a new threshold from held
  outcomes. Repeat checks on valid real training observations for each wall/GPU
  label mask; toy support is not measured support or parameter certainty.

[Self-review evidence](../results/prefill-design-self-review/report.md) records
score-window invariance and full-feature LOCO checks. Earlier round2 results and
source hashes remain archived at their original revision.

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
leave long token gaps. [Reproducible ITL scenarios](../results/prefill-design-round4/itl-scenarios.json)
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
| Sentinel |Mixed512/512/7680/7680 at50/400ms; uniform4x4096 at800ms;1x7680 at200ms;512-token sentinel |8|All held |

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
| **Total trials** |**212**|Plus four80s transfer attempts (two windows, two replicates each) |

Order: warmup, observer check, main+references, HTTP bridge, transfer slices, deep
traces. Main blocks are not reordered based on outcomes. Deep traces are last so
profiler flush/clock effects cannot contaminate later performance stages.

[Budget calculation](prefill-service-v1.budget.json) sums the frozen linear scenario
wall times, multiplies by1.35, then adds idle gaps,320s of replay including lead-ins,
40s transfer drain/gateway/flush reserve (10s per attempt), and120s general
frontend/profiler reserve. It estimates1518.29s; this is a planning estimate, not a
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

## Graph/padding accounting and context attribution

The v0.30.0 [dispatcher](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/cudagraph_dispatcher.py)
selects NONE/PIECEWISE/FULL and a padded BatchDescriptor. This supports recording
these fields; it does not establish the historical mode or its performance effect.
For every observed step, collect actual runtime mode, capture/replay/ordinary
execution, actual and padded token counts, actual/padded request counts where
available, query uniformity, capture key/size and attention backend. Pair step wall
and valid CUDA spans with decode count/context and exporter ITL using stable step
IDs. Missing fields remain unavailable, never fabricated from batch count.

Predeclare a mode-aware M0-level accounting variant with runtime-mode-specific
intercepts (one-hot modes, replacing the global intercept) and padding-excess work
as a covariate when training-supported. The encoding/capture grid, training-only
constant-column rule and comparator accounting basis must be frozen before held
scoring. Every M0/M1/M2 structural comparison uses that SAME accounting basis;
its full rank/condition checks must be repeated on the actual encoded rows. The
current five-column toy support check does not validate this mode-aware variant.
Implementation/source mapping is a pre-freeze blocker, not permission to add
features after a held failure. Plain-M0 remains an archived ablation.

A mechanistic M1 context claim additionally needs improvement within D16 and the
SAME observed runtime mode/padded-size strata, with varying context and adequate
training support. Merely grouping D16 is insufficient. No within-stratum support
means context attribution is unresolved. Do not force eager execution to simplify
measurement. A causal rollout must predict future mode/padding from its own work
state and frozen dispatcher/config rules; reading measured future mode tags is
oracle-only. Without that implementation, the mode-aware causal gate is unevaluated.

## Predictive models, primary targets and gates

Fit fixed nonnegative models, with training-only scaling and equal total squared
row weight per calibration cell, within the release-to-last-foreground-completion
step window defined above:

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

Freeze the selected family AND calibrated comparators, coefficients, scaling,
LOCO reports, per-position jitter/mapping, causal rollouts and eligibility hashes before opening held timing. The12 repeat-check
trials diagnose calibration repeatability; they cannot select another family.
Actual q/l coverage, full-family scaled rank/condition on all fitting rows AND
every LOCO training partition, and held feature ranges must be reported. Exclude
unsupported families before held scoring; stop if even M0 lacks support.
Numerical interpolation is not proof of causal support. Do not retune after held failure.

Held screens mirror the frozen decision: an M1 decode-benefit claim requires>=10%
decode-only case-weighted MAE improvement versus M0 and the declared prefill
no-regression limits. An M2 prefill-benefit claim requires>=10% improvement on BOTH
prefill targets versus the retained base and no-regression on all targets/cases.
Report M0/M1/M0Z/M2 ablations regardless; do not replace the selected candidate from
held results. Failed/inconclusive is not passed. M0 can remain appropriate.
All margins refer to relative prediction ERROR, not percentage of total runtime;
a3ms context effect can reduce a4ms baseline MAE by much more than10%.

**Separate causal sentinel rollout gate, eight cells, five blocks each:**

1. Per cell, abs(median(predicted-measured engine TTFT)) <=max(10% measured median,25ms).
2. Compare50/400ms offsets within each D level using the same mixed foreground.
   The uniform800ms cell is a separate tie control; do not include it in that ordering. Measured pairs with a block-bootstrap
   95% interval excluding0 and median difference>25ms are distinguishable; predicted
   ordering must agree for all such pairs. Others are unresolved/ties, not automatic
   ranking successes. No discernible ordering means policy identification is unresolved.
3. Before held scoring, freeze candidate, same-host calibrated M0, retained
   calibrated M1 (when applicable), their causal rollout implementations and
   per-cell jitter assignments. Use the SAME fitting trials, weights, label masks,
   work/config assumptions and accounting basis for all calibrated comparators.
   Structural comparisons are candidate-versus-calibrated-M0 and, for M2 after
   retaining M1, candidate-versus-calibrated-M1. Both must satisfy the component
   rule. Selected M0 has structural status **not applicable**, never "passed".
   Raw historical B2048 remains a separately reported staleness/recalibration
   comparison; beating it cannot establish improved model structure.
4. For EACH required comparator, a cell is eligible only if its predicted sentinel
   TTFT gap exceeds max(3*jitter[cell],25ms). Freeze/hash those lists before held
   timing. Require both fixed groups: mixed_burst (50/400ms x D0/D16) and
   single_long (200ms x D0/D16), with EVERY member eligible. Uniform800ms cells
   are controls, not a third group. Within each group/block, average comparator
   absolute error minus candidate absolute error across its member cells. Across
   the FIVE paired block means, require mean >max(t(.95,4)*sample_SD/sqrt(5),5ms)
   for BOTH groups, with1e-12s numerical tolerance. The5ms absolute mean-gain floor
   is fixed before held outcomes; equality and epsilon gains are non-wins. The
   requirement is a maximum, not the sum of the uncertainty margin and5ms. Never count20 cell observations as20 independent blocks.
   Fewer than two eligible groups or insufficient gains is inconclusive. Any cell
   regression >max(10% comparator MAE,25ms) is failed, including controls.
   All8 cells x5 paired block errors and canonical block IDs are required.
   `grouped_comparison_plan`, `grouped_error_check`, and `grouped_structural_check`
   in `prefill_grouped_validation.py` implement the current CPU component contracts.
   The old `prefill_validation.py` gates remain for historical reproductions only.
   Independent, approximately normal block differences are assumptions of this
   paired-t screen. `block_dependence_status` defaults to unresolved; unresolved
   or violated status prevents passing and preserves nominal diagnostics. Actual
   qualification requires the preregistered calibration-only assessment, not a
   caller's assertion inferred from five held outcomes. Components passing does
   NOT pass capture integrity, bias/order, future-isolation or complete experiment
   gates; the readiness CV check is a separate required pre-freeze condition.



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
are10/25/50ms and resolution means a gap>3*band, not just>band. These are not measured noise or power estimates. Before main, use off-mode
trials from observer qualification:20 single1024-D0 trials x1 position and10
uniform-D16 trials x4 positions. For each position separately compute
j=1.4826*median(abs(TTFT-median(TTFT))) across trials. Never pool positions.
`position_jitter` requires complete declared trial matrices; missing/nonfinite
observations do not yield zero jitter. `jitter_references` in the discrimination
artifact freezes the mapping: choose the reference position with smallest absolute
log-latency distance to the raw-B2048 predicted target (stable reference-ID tie
break), before collection. `assign_jitter` applies that mapping to calibration-only
measurements. Four@50 maps to uniform-D16's last position; four@400 to its second
position. Other cells retain their explicitly listed range match.
These are noise proxies, not cell-specific measured variance. Report out-of-range
latencies and different execution modes; the25ms floor remains an engineering
resolution margin, not a power guarantee. Mark a planned contrast unresolved when
abs(expected gap)<=3*its mapped jitter; candidate/comparator eligibility additionally
retains the25ms floor.
The declared identification contrasts are (H2) M2-versus-M0 request TTFT for
single7168 and mixed/reverse composition at each D, and (H3) FIFO-versus-RR
sentinel TTFT for four@50/four@400/one7680@200 at each D. Single2048 remains
validation-only; four@800 is a policy tie control (not necessarily an M2 tie).
Within-D offset ordering tests policy, not the nonlinear coefficient. Publish
resolvability for every contrast/scenario. If all identification contrasts for an
H2 or H3 claim are unresolved, that claim is inconclusive; stop main only if BOTH
families are unresolved. Do not replace cells. Candidate-versus-EACH-calibrated-comparator eligibility
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
EngineCore/config unchanged and bind each gateway identity separately. Run four
fixed32,80s replay attempts in frozen ABBA order: A-r1, B-r1, B-r2, A-r2.
A replays[40,120), scoring[60,120); B replays[160,240), scoring[180,240).
Use the historical development trace
SHA25662f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e,
rebased locally with original trace IDs/provenance. Natural EOS, output cap128,
original shared-client/process sampler/runner-GC settings and all enabled replay
integrity/load/control/diagnostic gates, including50ms limits, remain required.
Freeze derived trace hashes before collection. Each transfer attempt has a unique
ID and one allowed attempt. Drain all requests between attempts and record boundary
state/engine/gateway identities. Preserve failures, no replacement or outcome-based
reordering. Two repeats per window provide a descriptive repeat reference, not a
well-calibrated statistical envelope; missing replicates cannot qualify transfer. Preserve original trace IDs,
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
preregistration sensitivities, not predictions of fresh admission/EOS. The forced128-output copy is identical on these historical paths and is marked
duplicate, not independent evidence. After capture, also run the same
frozen algorithm on its actual accepted/output path, labelled retrospective.
Score gaps by right token timestamp within(20,80] local seconds; compare exporter
scrape endpoints explicitly. A tail mismatch challenges the joint scheduling/service
scenario; matching does not establish uniqueness or pass causal prediction gates.

## Full ITL vector contract

Commit all cumulative-derived disjoint exporter buckets, including under25ms and
+Inf. Source snapshots require stable series/bucket schemas, monotone counters,
identity/epoch continuity and valid samples; a failed internal sample or reset is
not silently bridged. Preserve requested AND actual first/last scrape sent-time
endpoints. Compare toy gaps using right-token endpoints at those same approximate
boundaries (subtract the transfer lead-in origin where applicable). Sent timestamps
are not exact token observation times; no interpolation or false exact alignment.
Legacy nominal-window tail counts remain separately labelled.

Primary evidence has TWO parts, both retained for every disjoint interval starting
at25ms, including+Inf. First record every observed-only or predicted-only support
bin, with counts/direction and a material flag when its nonzero count is at least
5 equivalent steps. Second compute magnitude using the SHARED window reference:
f=5*(total toy gaps / toy decode-bearing steps), from frozen linear-B2048 only.
For every candidate use L_b=log[(max(pred_b,f)/pred_total)/(max(obs_b,f)/obs_total)],
archive all values, max|L_b|, mean|L_b| and exp(max|L_b|). This is a toy-equivalent
step normalization, not real step counting, effective sample size, or a calibrated
noise threshold. Its floor remains an engineering choice. Candidate-specific floors
are forbidden. Report common-positive bins (shared observed/linear/k1.5 mask) and
legacy count floors1/5 as sensitivity; neither replaces primary support evidence.
Common-positive bins cannot excuse missing support. Factor2 remains archived as
an UNCALIBRATED engineering flag, never an acceptance threshold.

Archive toy decode-bearing steps, gap weights and per-bin step counts. Many gaps
come from one step, so uncertainty must use measured step clusters or declared time
blocks, not independent token gaps. No confidence claim is based on gap totals.
Report same-host old32r1/r2 and cross-host32/16 reference factors beside each model
window, explicitly labelling the240s reference windows as not matched to the60s
transfer windows. One same-host pair is not a calibrated noise envelope. Future
paired transfer repeats must be reported before qualifying its discrepancy flag;
if a usable measured envelope is unavailable the conclusion remains descriptive.
No below-factor2 result accepts a model or identifies k. Full fast bins are also
archived. >200ms can distinguish the SPECIFIC frozen8192/2048 scenarios under the
toy all-running-decode assumption, not uniquely identify the historical config.

The same historical-window comparison is already run. It is development diagnosis,
not an independent acceptance test or a fit of k. Transfer must log step duration,
decode population, graph/padding and exporter timestamps so discrepancies can be
separated from work-count and output-observation effects. All five historical paths
have exactly128 outputs, so the forced128 copy is explicitly marked duplicate;
future natural-EOS outcomes are still measured and reported without pretending the
old output vector forecasts them. Admission timing and release state remain conditional.

## Reproduce and freeze checklist

```
python scripts/build_prefill_design.py --output-dir /tmp/fresh-prefill-design
python scripts/check_prefill_noise.py --output-dir /tmp/fresh-prefill-noise
python scripts/check_prefill_itl.py --old-evidence OLD --new-evidence NEW --output /tmp/fresh-itl.json
python scripts/check_prefill_comparators.py --output /tmp/fresh-comparators.json
python scripts/check_prefill_win_margin.py --output /tmp/fresh-win-margin.json
python -m pytest -q tests/test_prefill_design.py tests/test_prefill_identifiability.py tests/test_prefill_review_gates.py tests/test_prefill_planning.py tests/test_prefill_noise.py
python scripts/check_prefill_budget.py --suite-dir NEW/results/low-cap-switch-v1 --old-suite-dir OLD/results/cap-range-v1 --output-dir /tmp/fresh-budget-check
```

Historical round5 calibrated checks must run at commit c726a423426aaa3ece2334ac0d1f0d9de245153a;
their generator/case hashes resolve through round5/source-snapshots.json.

The plan generator reproduces cases/order/predictions/discrimination/budget,
including complete-feature training-fold support diagnostics. The
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
