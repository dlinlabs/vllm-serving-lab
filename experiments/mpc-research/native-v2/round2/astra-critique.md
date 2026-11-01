# Astra Round2: adversarial review of the frozen native-v2 proposals

Assignment `pr-nv2-astra`; reviewer GPT-6 Astra (`gpt-6-astra`).
Reviewed integration freeze `bc217acf9db3f6222026c0dd345db77a379d6151` using
`git show`, without changing branches. Read the OTHER THREE complete reports,
prototypes and tests, plus the freeze manifest. No additional agents used.
References to `opus.*`, `gemini.*`, `sol.*` and their Python files below mean
`experiments/mpc-research/native-v2/round1/` **at that freeze**, not later edits.
The source commits and hashes are in `results/mpc-native-v2/round1-freeze.json`.

## 1. Decision

**Do not deploy any proposal as a qualified native MPC model.** Preserve the
useful ingredients: exact marked release accounting, causal observation cuts,
batch/context features, probabilistic EOS, and small same-window comparisons
against reactive/One-move. Reject the universal horizon theorem, current-cap
reservation bound, and statistical certification supplied by Gemini. Opus's
new descriptive fits are informative but overstate causal identification and
some measurement targets. Sol's finite-witness theorem is sound within its
carefully declared scope, not a proof of opposite optimal actions or MPC value.

The proposed default-off measurement facade is a sensible **qualification
candidate**, not yet an accepted instrument. Reusing PR32 capture machinery is
preferable to building a second pipeline. Forward-scope CUDA event elapsed time
plus host schedule/commit labels can answer useful questions without claiming
whole-pipeline GPU time, kernel busy, or a synchronized GPU/host clock.
**Yes: a supported whole-policy/reduced outcome model can be attempted before
perfect GPU decomposition.** It still requires prospective assignment/receipt
records, complete all-offered outcomes, causal features and independent blocks.

Evidence labels:

* **EXEC**: reproduced by the committed small CPU diagnostic against frozen
  peer code, or by the explicitly recorded TRAIN reanalysis.
* **PROOF**: mathematical defect or correct result under stated assumptions.
* **UNSUPPORTED**: asserted mechanism/guarantee is not established; not a claim
  that the proposed mechanism is impossible.
* **SOUND**: useful, correctly bounded idea worth retaining.

| Peer | Strongest contribution | Weakest claim / blocking gate |
|---|---|---|
| Opus | New composition-conditioned timing/EOS descriptions; explicit knee uncertainty; strong prefills-ahead reactive comparator | Randomized assignment allegedly removes composition confounding; exact iteration timing inferred from survivor tokens; unprofiled use of fitted busy-time terms |
| Gemini | Explicit actuator boundary and insistence on clock/CPU/GPU distinctions | False horizon iff/optimality and reservation formulas; wrong statistical guardrails; timing contract accepts invalid good outcomes |
| Sol | Exact hidden-release projection witness and distinction between whole-policy effects and physical decomposition | Proposed large latent state and transition law still lack identifiable, timely inputs and a measured compute budget; not an executable identified controller |

## 2. Opus: retain the diagnostic, narrow its conclusions

### O1. Randomized budgets do not randomize batch composition [PROOF, EXEC]

`opus.md` section 5, lines 194-199 says structural pieces are identifiable from
observed composition and, because actions were randomized, the remaining risk
for the law is support rather than confounding. This does not follow.
Composition is a post-assignment variable, affected by backlog, service state,
hidden context and selection. Randomization of an upstream action does not
randomize its mediator.

Executed finite witness: independent equiprobable `U,Z in {0,1}`, composition
`C=U+Z`, latency `Y=Z`. Regression of Y on C has slope **0.5**, although
`do(C)` has **zero** effect on Y. All four assignment/state combinations have
support. Random U therefore cannot justify the observational composition slope.
This is an abstract causal witness, not a claim these are the native equations.
It refutes the inference, not the possibility of identifying a useful batch law
with measured confounders or a stronger experiment.

**Gate:** define structural versus predictive estimands; collect pre-batch
context/mode/hardware/pending state; justify residual exogeneity or keep the fit
descriptive. Validate supported randomized *sequence* effects separately.
Do not instrument composition with budget without an exclusion argument:
budget also changes waiting, admission and later work.

### O2. LOCO knee selection sees the held cell [EXEC]

`opus_batch_models.py:370-386` selects the knee on all batches, then passes it
to `leave_one_condition_out()` (`141-156`). Coefficients are refitted in each
fold, but the hyperparameter has seen the held target. Consequently the quoted
1.36ms MAE is not a wholly held-composition estimate of model selection.

I reran the same frozen functions and 84 existing profiled TRAIN batches:

| Selection | MAE (ms) | Max absolute error (ms) |
|---|---:|---:|
| Published, global knee before LOCO | 1.361494 | 3.352886 |
| Knee selected inside each training fold | 1.706663 | 3.690203 |

Fold knees include 0,32,64. This modest degradation does **not** reverse the
large descriptive advantage over the serial baseline; it does invalidate
calling the original score independent of knee selection. Composition folds
also remain correlated within a single profiler experiment, not independent
GPU runs. `knee_profile()`'s 10% SSE band and support-gap heuristic
(`118-134`) are diagnostic heuristics, not statistical identification intervals.

**Gate:** nested selection with block/run uncertainty; call the knee
unidentified until the new design has independent nearby support. Prefer
QR/SVD or a verified tiny NNLS implementation with scaled inputs and KKT/rank
diagnostics for later fitting: the current normal equations plus a `1e-12`
ridge (`53-56`) are not evidence of coefficient identifiability. I found no
numerical failure in the executed existing-data fit and do not invent one.

### O3. Warmup drops live history, not merely scoring targets [EXEC]

`ttft_by_prefill_ahead()` (`292-307`) filters all rows by
`sent_at >= start+warmup` before counting preceding unfinished requests.
A request sent at 0.9s with first content at 1.4s is ahead of a request sent at
1.1s. With the default 1s warmup the prototype reports **ahead=0**, not **1**.
Dropping a warmup target does not erase its remaining work.

Actual TRAIN recheck: **1 of 932** scored p4096 targets and **0 of 10041**
p128 targets have this boundary defect. Thus it is a real executable defect,
but not evidence that the whole F4 descriptive pattern disappears. Keep raw
results, correct feature history in a future revision, and rescore all bins.

The future `status=="complete"` filter is another online-transfer limitation:
with cancellations/failures it would select history using later outcomes.
The existing audited intervention data have no such failed accepted outcomes;
I am not claiming that hypothetical failure contamination occurred here.
F4 is client-send-to-first on complete requests, not all-offered scheduled-time
goodput. `opus.md:83-94,304` acknowledges important parts of this distinction.

### O4. Busy, span, commit and token-progress targets are not interchangeable

**SOUND:** F1 fits a batch rather than serial-request unit and explicitly leaves
the knee unidentified. **UNSUPPORTED:** `opus.md:77-80` concludes that roughly
3.6-4.1ms is "not hidden by async" from unprofiled token-progress versus
profiled busy measurements. Its next sentence says this is not decomposition,
but the preceding causal conclusion still requires a joint same-run clock and
completion measurement. Cross-run subtraction cannot reveal hidden overlap.

`opus.md:209-210` calls a continuing decoder's output increment an **exact
iteration counter**. At most this counts committed output advances for that
decoder under a qualified one-output-per-scheduled-step regime. It need not
count steps in which it was not scheduled, pending iterations, different
requests' steps, or the fractional first/last interval at an RPC sample.
For D=0, E1 cannot even use its proposed persistent decoder without changing
the D=0 condition. `delta_t/delta_output` is a sampled effective rate, not the
duration of a particular batch or physical GPU iteration.

Likewise `prefill_backlog_s()` (`220-228`) serializes one prompt per chunk.
It openly warns about short prompts, but "for long prompts packing does not
save GPU time" is not general even in its hinge/intercept model: packing
residual chunks can save a batch floor and change padding. The method is a
restricted approximation, not an identified scheduler transition.
`opus.md:173`'s decode-first rule needs parity testing against the actual
running-first scheduler and pending-placeholder behavior; knowing source is
not proof that a different simplified rule has identical schedules.

`batch_time_lower_bound_s()` (`167-184`) correctly uses
`sum max(0,s_j-k) >= max(0,sum s_j-Nk)` **conditional on** a nonnegative
per-iteration lower law and appropriate nonoverlap/resource accounting.
Its docstring states that condition. A mean fitted busy law is not a lower
envelope of native wall durations, however. F2 consistency therefore means
"not excluded under this assumed bound," not that 5640 is a qualified native
capacity ceiling. Passing a loose necessary screen never establishes
schedulability or throughput.

**Gate:** use directly joined host commits and scoped forward durations;
do not fit whole wall service by subtracting separately measured quantities.
Qualify membership/commit parity and effective rate targets, then compare
predictions of releases/TTFT/TPOT at fixed future horizons.

### O5. EOS, KV and horizon claims need weaker language

F5's paired length disagreements are useful [SOUND]; they reject a deterministic
per-prompt-length assumption on those paired completions. They do not prove
EOS is non-exogenous or batch-caused (`opus.md:96,106`); same-action repetitions
and admitted-in-both selection matter. The report itself acknowledges this at
107. Retain that caveat in the heading and downstream scenario generator.

`age_value_held_out()` (`274-289`) uses **generated-token age**, not wall-clock
age. `age_samples()` enumerates every age to EOS, weighting long outputs more
often. The result is a token-age-weighted retrospective remaining-token score,
not a wall-time sampled residual-service estimate or calibrated EOS hazard.
Preserve paired blocks, censored/failed populations and request-weighting
alternatives when promoting it. Shared quantiles are not calibrated joint
uncertainty, especially if batching changes EOS.

F8's maximum observed KV 0.37 supports an **observed low-pressure domain**,
not its approximately 70-cost-unit threshold for chance-constraint activity.
`kv_fraction_per_token()` uses prompt+output on running marks, not a full
allocated-block/pending/recompute inventory. Never turn the regression slope
into an allocator safety proof. A 0.9 observed-usage trigger cannot guarantee
future usage stays below 0.9; a cap decrease cannot reclaim existing KV.

`opus.md:233`'s "effects last at most" 0.24+1.4s is not a proven bound under
interference, transport delay, or failures. `279`'s stop rule, first-action
disagreement below 5% implies no material goodput change, is **false without a
bound on impact per disagreement**. Executed arithmetic witness: 4/100 slots
differ, each with R=16 and 16 new cost-1 offers. A change from16 to32 admits
16 additional good requests per selected slot in an otherwise identical
feasible toy lifecycle. Over100s, that is **0.64 good/s** at **4%** disagreement.
This is a stipulated mathematical case, not native gain or a new benchmark.
Use offered-load/impact-weighted disagreements and downstream burden instead.

A forecastable exogenous shift is a useful hypothesis, not a universal
necessity for planning value: endogenous release/membership changes can make a
later optimal action different even with stationary or absent disturbances.
Failure of N3 to beat N1 even with an oracle also does not uniquely diagnose
short memory: nonbinding actions, an adequate backup, poor solver/model, or no
action gap can all explain it (`opus.md:259-279`).

## 3. Gemini: reject certification, salvage measurement discipline

### G1. Reservation bound is false after a legal cap reduction [PROOF, EXEC]

`gemini.md:154-155` claims
`R(t) <= max(R(0), u_applied(t))`. Start R(0)=0, admit four cost-8 requests
under32, then decrease to16 without any terminal. R is still32 and the claimed
right side is16. The code-free mathematical witness is executed in the
diagnostic. Correct invariant:

    R(t) <= max(R(0), sup_{s <= t} u_applied(s))

or the fixed maximum supported budget. This is only reservation accounting,
not physical KV/SLO safety. The report's non-eviction explanation in section2
is sound but contradicts its own constraint.

### G2. The horizon iff/optimality theorem is false [PROOF, UNSUPPORTED]

`gemini.md:171-180` supplies no assumptions/proof for
"value iff settling time exceeds slot" or mathematical optimality of N1.
Long physical memory is not sufficient. Executed finite witness: two cost-4
requests persist20s; peak R=8 and both caps admit both. Every plan yields the
same outcomes despite memory exceeding a1s slot. The admission mechanism can
couple future membership, but it is nonbinding on this workload.

For a valid sufficient N1 statement one needs separability of action-dependent
reward/state across slots, no relevant information/constraints coupling them,
and correct continuation value. Generic settling time is not that condition;
the report even describes 1-1.5s clearance inside a supposedly <=1s regime.
The claimed 10-20s long-request KV residence and impending thrashing are not
established from the cited TRAIN data. Low-pressure F8 evidence from Opus is
not a proof they can never happen, but it makes assuming them as this
deployment's decisive MPC mechanism especially unjustified.

**Gate:** replace theorem with a falsifiable conditional mechanism; compare
executed first actions of same-model, same-cohort, equally replanning N1/N3
with actual binding decisions. Report indistinguishable outcomes as such.

### G3. Wrong coefficient target and ambiguous 1000x scaling [PROOF of mismatch]

`gemini.md:108-110` promotes the decode-survivor denominator
12.31 + 0.049 D + 8.73 Pk to a batch iteration service law. Those coefficients
were fitted to **output increments of surviving decoders over realized
snapshot gaps** (`native_observer/native_token_model.py:1-17,69-115`).
They are not identified prefill/forward durations, and pure-prefill D=0 is
not that target's support. The state definition at `gemini.md:123` uses prompt
**tokens**, while the coefficient is ms/**kilo**-prompt-token: the formula
needs an explicit P/1000. At p4096,D1 the two readings give 35770.439 versus
48.11708ms. Fixing the units does not fix the target substitution.

Also, `gemini.md:27,98` overgeneralizes the serial-proxy falsification to all
serial/fluid models and continuous batching. A bulk-service/marked fluid model
can be a valid reduced predictor; it is this **specific per-request exclusive
HTTP-demand interpretation** that fails the 2004-versus2449 screen.
Do not call the proposed hybrid model "optimal identifiable" before comparison.

### G4. Claimed exact timing partition omits real intervals [PROOF, EXEC]

`gemini.md:63-76`'s terms are not an exact nonoverlapping TTFT decomposition:
admit->engine-enqueue is missing; waiting between prefill chunks and
post-forward sampling/copy are not accounted by summing GPU batch spans.
Executed same-clock timeline, in milliseconds:

    scheduled0, Gateway10, admit20, enqueue40,
    chunk1[50,60], chunk2[160,180], sampled190, first200.

TTFT=200. The named ingress/gateway/initial-engine-wait/prefill-spans/egress
terms sum to **70**, missing20+100+10=130ms.
A legitimate same-clock telescoping partition is ingress + Gateway processing
+ admit-to-enqueue + enqueue-to-first-sampled **residence** + sample-to-client.
The engine residence may then be annotated with batch scopes, not decomposed
into exclusive per-request compute without additional qualification.

`RequestTimingTrace` stores one user-supplied domain for all raw numbers and
does not perform cross-domain mapping or attach per-event availability.
That can be a schema for **already qualified normalized** times, not a clock
qualification procedure. `ClockDomain.validate_compatible()` is useful, but
calling an unrelated helper on synthetic examples does not validate provenance
of every subtracted timestamp.

### G5. Executable timing/overhead validators fail closedness [EXEC]

`RequestTimingTrace.validate_monotonic_order()` (`56-95`) accepts
scheduled1, received1.01, admitted1.02, missing engine/sample times,
first-content0.9, terminal1.1. `is_composite_good()` returns **True** for its
negative TTFT. Even explicit validation does not save it. Success fields also
do not distinguish a non-cancelled failed stream from complete success.

`ProbeOverheadGuard.validate_probe()` (`275-298`) accepts NaN end time:
comparisons against NaN produce no violations. Input finiteness, positive slot
duration, complete-success status, chronology even with missing intermediate
stages, and actual domain/receipt checks are prerequisites for a reusable guard.
An overhead flag supplied by a caller cannot establish absence of hidden
synchronization. The implementation defaults to5% slot overhead versus the
report's2%; neither is measured qualification. RPC latency divided by a slot
also differs from **causal serving slowdown**, and repeated probes accumulate.

**Gate:** test malformed/missing/nonfinite events and failures; specify
measurement versus admission acceptance independently. A timing record's
incompleteness should not fabricate a complete success or a zero-duration GPU.

### G6. Statistical evaluator can falsely certify superiority [EXEC]

`NativeFairStatisticalProtocol.evaluate_paired_blocks()` (`350-401`) has
material departures from `gemini.md` section8:

* It divides solver overruns by **offered requests**, not solver invocations.
* It never checks the proposed relative rejection-rate guard at all.
* It accepts eight copies of the **same block/seed** as eight independent
  observations and labels them `CONFIRMED_SUPERIOR`.
* It implements a **one-sided zero-null** sign-flip test, not the claimed
  two-sided test. A significant zero-null test plus sample mean above0.5
  does not establish an effect of at least0.5.
* `2**n` enumeration has no n<=12 bound despite its comment; larger designs
  have unbounded exponential runtime. No minimum independent-block check
  enforces the stated K>=8.

Executed witness with eight distinct analytic block IDs:
differences `[0.1]*7+[4]` good/s, each10s, 10000 offers, A/B rejection
fractions0.1/0.8, B one solver overrun per block. The frozen evaluator returns:

| Quantity | Executed result |
|---|---:|
| Mean goodput difference | 0.5875 |
| Reported p (one-sided zero null) | 0.00390625 |
| Two-sided zero-null reference | 0.0078125 |
| One-sided constant-margin0.5 sign-flip reference | 0.5 |
| Reported overrun fraction (per offer) | 0.0001 |
| Overrun fraction if10 solves/block | 0.1 |
| Verdict | `CONFIRMED_SUPERIOR` |

The margin reference is an exact finite test under its **stated sharp
constant-shift/sign-exchangeability assumptions**, not a universally valid
replacement for heterogeneous native effects. It demonstrates the tested null
is different. Rejection and invocation-denominator failures alone already
invalidate certification against the declared guardrails. Repeating one of
these blocks eight times also returns `CONFIRMED_SUPERIOR`.

Alternating policy order is not itself random assignment or guaranteed drift
cancellation. Sign-flip inference needs the actual randomized-block design
or justified symmetry. Failed significance is not equivalence; the verdict
`EQUIVALENT_OR_INSUFFICIENT` should not suggest otherwise. The0.5 margin and
eight-block size are proposals, not established power/precision choices.

**Gate:** unique block/seed/policy/tape identities, valid count/duration bounds,
actual solver attempt counts, rejection guard including a zero-baseline
definition, predeclared randomization unit/null/margin, bounded computation
and a confidence/decision rule matching the claimed estimand. Do not integrate
this evaluator as an acceptance authority merely because its18 tests pass.

### G7. Tail and fallback language is stronger than its guarantees

The shared cohort and all-offered composite metric are SOUND. A 2s arrival
continuation followed by a drain (`gemini.md:163`) changes sustained arrival
interference; even complete target resolution in that artificial future does
not eliminate every horizon-edge incentive. Report the continuation policy,
traffic process and residual reward interval explicitly.

An actuator cannot automatically deliver Fixed16 on every solver exception
or network failure (`158`). Holding a known applied cap is possible without
new communication; changing it is not. A lost acknowledgment remains unknown
until fenced/reconciled. Neither16 nor a KV threshold is proven SLO-safe.

## 4. Sol: sound finite obstruction, not yet a fitted design

### S1. The minimax projection certificate is correct [SOUND, PROOF]

`sol_counterexamples.py:21-92` fixes per-request observations plus total
remaining work. Worlds(1,9)/(5,5), equal costs8, unit independent decode rate
give R(2)=8/16. For any scalar estimate r,
`max(|r-8|,|r-16|) >= 4`; r=12 attains4. Exact release at the boundary is
handled consistently. The source explicitly disclaims probabilities,
opposite optimal actions and full-history equivalence. I found no
high-confidence mathematical defect within this declared scope.

The request costs are abstract, as stated; the example is not evidence those
physical native states have equal probability. Under budget32 both worlds
accept the cost4 offer, so the certificate does **not** show that uncertainty
necessarily changes the optimal budget. Its constructive lesson is marked
release beliefs, not that fluid models or all reduced states are impossible.

`available_at_decision()` (`95-104`) correctly distinguishes source from
receipt time in one qualified clock domain. It is not a complete freshness/
identity/atomic-snapshot validator; the report does not claim it is.

### S2. Strong experimental distinction; remaining implementation gaps

`sol.md:224-237` properly separates randomized assignment from load-dependent
application lag and proposes pre-action randomization with a common
continuation. `337-342` correctly warns that hand-paired quantiles and marginal
coverage do not establish joint trajectory/selection coverage. Preserve these.

The weakest design claim is feasibility of the proposed high-dimensional
belief/state: full waiting marks, in-flight dependencies, KV allocation modes,
hardware regime and context are not jointly observed by today's adapter.
The model functions and posterior update are not specified sufficiently to
count states/events or show identification from eight intervention blocks.
This is an **unverified proposal**, not an implementation error in the tiny
witness script. The declared three-move/5s design is a development starting
point, not a supported timescale.

**Gates:** ablate small candidate states/history features against a supported
whole-policy predictor; fit only a minimal low-rank batch law; label log-link
predictions as median versus mean and simulate calibrated residuals when
optimizing expected counts; measure belief update plus full control path.
Include a continued-traffic terminal experiment, not just "common drain"
(`350-355`). Keep offline labels unavailable at a live decision cut even
when later collection provides perfectly aligned source timestamps.

No finding against Sol is manufactured for symmetry with the other peers.
Its proposed conditions and tests are mostly sound; they remain conditions
to satisfy, not evidence that the proposed controller is ready.

## 5. Corrections and limits of my own Round1

Sol's witness strengthens, rather than refutes, my insistence on per-request
residual/release beliefs. Its stronger warning about **intention-to-treat
versus actual application** should be explicit in my next design: recording
the applied budget is necessary for dynamics, but conditioning treatment
inference on load-dependent application can defeat randomization.

My `astra.md` section3 log-duration model must specify retransformation:
`exp(E[log tau])` is generally not `E[tau]`. Sample calibrated joint residuals
or estimate the relevant distribution; do not optimize mean goodput using an
unlabeled median-duration trajectory. Batch mode/context features must earn
inclusion before a large belief implementation.

My Round1 tail diagnostic correctly labels client-prefix reconstruction as
offline and does not award unknown terminal successes. Its loader accepts only
the audited clean complete/rejected TRAIN collections; it deliberately fails
on failed/incomplete collections. It is **not yet a generic production outcome
evaluator**, where qualified failed/cancelled outcomes must remain in the cohort
with zero credit. Its TTFT-only upper pruning is conservative for unresolved
TPOT/failure uncertainty, not a tight action-value bound.

The six-of-eight resolved90s rankings are contrasts of already assigned whole
sequences, not nearby MPC first-action choices. They neither justify a planning
horizon nor demonstrate a model-confidence interval. My conditional
`V_open <= V_tree <= V_perfect` ordering requires nested feasible policies and
the same objective/model; it cannot be used to rank implementations with
different compute/fallback costs. No peer evidence here establishes a native
MPC gain or refutes my report's explicit non-acceptance.

## 6. Minimal usable telemetry contract and facade assessment

### Contract: enough to join work and outcomes, not perfect decomposition

| Record | Required contents and causal meaning |
|---|---|
| Run/measurement manifest | Experiment/episode/TRAIN split, pre-frozen logical offer IDs and prompt recipe hash, model revision, cost/tokenizer identity, actual runtime flags, 38-file source guard, facade/source hash, process/boot/clock domain, observer mode |
| Offered/Gateway lifecycle | Scheduled and sent client times in their domain; receipt/admission/first/release/terminal times and availability; charged immutable cost and marks; rejection/failure/cancel/success, usage and finish reason; complete expected offered cohort |
| Assignment | Intended randomization time/probability and treatment, actual request/ack/expiry/version/application times, pending/reconcile status; no intended==applied substitution |
| Schedule | Epoch+batch key+sequence, host before/after schedule, request IDs with pre-execution positions and scheduled token counts, derived prompt/decode/context features, waiting/running/pending summaries, preemption/speculation flags |
| Forward measurement | Matching step key and worker/device/stream identity; CPU enter/return; current-stream CUDA-event elapsed forward-scope duration, query-resolution time, missing/error/overflow flags, graph/padding/executed-count metadata |
| Output commit | Same step key, host commit/after-output time, per-request emitted-token increments and terminal/release linkage; distinguish scheduler-computed from committed outputs |
| Availability/drain | Source and collector-receipt times; collection watermark; engine waiting/running/batch queue, frontend/Gateway reservations, worker unresolved events and overflow; exactly-once joins and explicit unknowns |

GPU **elapsed duration** can be used without subtracting a GPU timestamp from
host monotonic time. Absolute cross-clock placement requires an independent
mapping with uncertainty. Batch elapsed spans cannot be assigned as exclusive
per-request demands. Event timing collected only after drain is a **training
label**, not a feature available to a decision during the episode. A qualified
online state subset can remain coarser and carry an explicit unknown-work belief.

Every admitted engine-visible request must reconcile; rejected offers should
remain Gateway/client outcomes, not be demanded in the engine's ID set.
`num_computed_tokens` is bookkeeping, not proof the associated GPU work
completed. Pending request counts without IDs cannot define an exact Markov
physical state. Partial joins must not be silently filled with zero.

### What PR32 reuse already offers [SOUND]

Read current source, unchanged from the research base:
`prefill_lab/runtime.py:162-216` (`ExternalCapture`) observes normal external
traffic without filtering away frontend outputs.
`prefill_lab/contracts.py:74-100` (`scheduled_work`, `step_key`) tags the
pre-execution positions and token counts.
`prefill_lab/plugin.py:251-353` (`WorkerRecorder`) records CUDA events on the
captured current stream, stores explicit scope text and leaves
`cuda_scope_qualified=False`. Resolution queries events later.
`plugin.py:238-245` refuses external-end before scheduler/batch drain and
retains pending worker events.

That is a useful base for the Mayor's proposed facade. It avoids a second
request/batch identity pipeline and does not require a new controller or
synthetic/private requests. It is still **not** whole-GPU forward+sampling+
copy service, CUPTI kernel busy, or per-request exclusive prefill duration.
The wrapper includes execute-model preparation/forward scope; retain the
actual scope label instead of shortening it to "pure kernel execution."

### Specific qualification risks to check, not findings against unseen code

1. **Default-off versus instrumented OFF:** `plugin.py:38-43,147-148`
   calls `sys.setswitchinterval(5e-4)` when installing core runtime, even via
   a metadata command. Capture-OFF inside an already hooked process is not
   necessarily untouched baseline. The facade must avoid/import-guard unwanted
   activation and measure literal-disabled versus enabled behavior, including
   GIL/dispatch hooks. Do not change global sandbox settings.
2. **No hidden live predictor:** `plugin.py:218-235` can create a LiveTracker
   or EnvelopeTracker depending on environment. Measurement-only begin must
   explicitly reject/disable that path without faking a calibrated model.
   `request_factory()` uses forced EOS for private trials; the facade must
   stay on `ExternalCapture`, not accidentally reuse that generator.
3. **Counts are not an ID manifest:** `ExternalCapture` tracks
   `expected_foreground` count but does not by itself prove pre-frozen TRAIN
   IDs/marks or account for Gateway rejections. Compare immutable expected
   client offers with actual admitted/native IDs; capture unexpected work as
   invalid evidence rather than quietly assigning it to a target.
4. **Stream and completion scope:** prove event placement around the intended
   native graph/async path. Same-stream elapsed time may contain gaps and
   excludes independent unjoined streams. No device-wide synchronize in the
   hot path. Later sampling/copy/output commits require separate host/identity
   records; CPU return does not stand in for completion.
5. **Capacity/errors/drain:** Buffer capacity is8192; a long decode-heavy
   episode can exceed it. Bound capture duration/records or mark overflow
   invalid and preserve it. `external_end` plus worker pending/error/overflow
   checks and Gateway/frontend drain all matter. A partial timing capture
   cannot be rescued by good HTTP summary counts.
6. **Unsupported scheduler events:** `Trial.after_schedule()`'s parity model
   rejects preemption/speculation; that is a scope detector, not evidence the
   native engine cannot do them. Recompute may repeat token positions;
   identity must include epoch/order, not assume content hashes prove unique
   physical execution forever. Preserve the request stream on instrumentation
   errors where possible; do not alter admission or EOS to complete a capture.

Acceptance requires CPU adapter tests, source/default-off checks, a small native
join/drain/error smoke and a measured observer-effect comparison with the actual
facade. No such facade code/native execution was supplied for this review.

## 7. Whole-policy reduced model: proceed in parallel, within scope

A blocked randomized comparison of two supported cap sequences can estimate
their all-offered cohort difference without knowing how many milliseconds
belong to GPU versus transport. A small predictive model can condition on
available R, marked phase/age/context summaries, arrival/mix history and
waiting/pending/unknown-work flags, and predict releases/SLO outcomes at fixed
future offsets under a declared sequence/common continuation.

Do not equate this with identifying arbitrary pointwise `do(u)` transitions
or a complete scheduler simulator. Whole-sequence assignment is the primary
causal estimand; application lag is an outcome/compliance record. Dynamic local
response requires separate pre-action randomization and overlap. Under hidden
state aliasing, predict a distribution/history-dependent reduced response, not
an exact deterministic state transition. Failed cancellation/orphan accounting
still invalidates scope.

A sensible first gate is better *held-block action ranking and count/release
calibration* than persistence/history/reactive baselines, with causal input
availability and simultaneous/model uncertainty sufficient for choosing a cap.
Observational token MAE alone is not enough. Admit that eight existing blocks
support exploratory diagnosis, not a fitted universal response surface. Keep
budget16/32 in cost units and internal scheduler tokens out of the actuator.

## 8. SINGLE next most informative native experiment

**One blocked, randomized mixed-burst admission/release and observer-effect
pilot**, not a controller tournament or a knee-only profiling sweep.
It tests whether measurable batch/commit history explains action-relevant
first/release outcomes on the very mixed-workload support currently missing.

Before execution, freeze a new TRAIN-only protocol, offer/ID manifests, source
hashes, qualification gates and bounded capture/drain durations. Suggested
pilot: four independent blocks; each replays the same12s standard Chat offered
tape in four randomized, position-balanced arms:
`budget16/32 x literal observer OFF/LIGHT`. Keep model/runtime flags, warmup,
cost estimator and natural EOS unchanged. Each tape contains a short-request
prelude, a modest long-prompt microburst followed by short probes, and a
reverse-order pulse; jitter/recipe vary only across blocks and are shared
within a block. These are planned ordinary HTTP offers, not engine-injected
requests. Budget is fixed per arm, normally qualified before its offers.

OFF must be a genuine disabled-measurement baseline. If the installed hooks
or GIL changes cannot be undone safely, use clean same-configuration service
starts with matched warmup rather than calling an instrumented OFF pristine.
Run sequentially on the already authorized single native resource; this
report neither starts it nor authorizes resource expansion.

Record the minimal contract above and hardware timeline. In LIGHT retain
actual chunk membership, current-stream forward spans, schedule/commit lag,
pending depth and HTTP outcomes; resolve CUDA labels only after drain.
At predeclared *client/observer* decision cuts, seal simple past-only
release/first-content forecasts if the online record subset is qualified.
Otherwise label prediction evaluation retrospective, not issued online.

Primary **pilot** questions, jointly from this one design:

* Does LIGHT change offered-cohort goodput/latencies, batch distribution or
  host commit delays versus OFF beyond a predeclared measurement-tolerance
  interval? If so, do not transfer LIGHT timing coefficients as normal service.
* Are all offered/accepted/native/batch/commit/terminal identities complete,
  including rejected requests that never enter the engine? If not, stop
  physical fitting and preserve the failure.
* Does added pending/batch/context history improve fixed-origin first/terminal
  and whole-cap contrast predictions over scalar backlog/persistence, and are
  short-behind-long effects explained by actual batch/commit exposure?

The outcomes can reject a reduced state, a batch closure, or the instrument
itself. They cannot by themselves prove EOS action causation (same-action
replication is not separately powered), MPC superiority, a roofline knee, or
statistical equivalence. Four blocks are a bounded exploratory pilot for
measurement failure and variance/support discovery; allocate subsequent
calibration/validation and choose final margins/power before final Test.
Held-out160021-160024 stays unopened, unused and ungenerated.

## 9. Executed receipt, reproduction and scope

Owned new paths only:
`experiments/mpc-research/native-v2/round2/astra-critique.md`,
`astra_counterexamples.py`, `test_astra_counterexamples.py` (same directory).
Frozen Round1 files and peers were not changed.

The diagnostic loads only two reviewed Python modules via `git show` and checks
their freeze SHA256 before executing selected functions in memory:

| Module | SHA256 |
|---|---|
| `opus_batch_models.py` | `0e6cc9787191875b6049f23dc4e4495ddda787a341010aec8708e78d05397502` |
| `gemini_timing_contract.py` | `8c9fbf413ed6e4baadf71ffe17d4eebfa18cdc1de844473c2bfa6edc77b2ee6f` |

It writes no peer source copies, uses no network/GPU and loads no workload
archive. The analytic statistical blocks are tiny invented count vectors,
not native workload tapes or evidence of a winning controller.
The separate Opus audit reads the existing profile JSON and the two explicit
TRAIN archive members through its frozen loader; no held-out file is involved.
Raw inputs/failures were preserved.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/mpc-research/native-v2/round2 \
  -p test_astra_counterexamples.py -v
PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/mpc-research/native-v2/round2/astra_counterexamples.py
```

The tests deliberately assert the frozen peer's bad behavior as a
counterexample; a passing test is **not** certification of that behavior.
The original seven witnesses/tests passed; an eighth test was then added for
the missing request-timing partition. **Final run: 8 tests passed in0.021s.**
The CLI reproduced the expected invalid peer verdicts and verified both
frozen source hashes. New diagnostic SHA256:
`f2bb19d23be4d3951b6c2a77bdaf1b1ab01fe781ca9f2040b23fb4a2857edecd`.
New test SHA256:
`00efd938fdfc4ef4d5f5f75ed69af14474dc7fcda8f41b9f45e34e9a454f76f8`.
Environment: Python3.14.4, standard library only, no dependency installation.
No native/GPU test or full repository suite was rerun. Mayor's independent
52-test receipt is not a mathematical or physical certification.

The numerical Opus recheck is reproducible with the same frozen functions:
for every `condition_key` in `native-profile-diagnostic.json["batches"]`,
select `knee_profile(train)["best_knee"]` on the other cells, `fit_law(train,
knee)`, then aggregate `abs(predict_ms(law,b)-b["kernel_busy_union_us"]/1000)`
over the held cell. This gives the1.706663/3.690203ms row above. For warmup,
keep all complete predecessors but score only targets sent after1s; count
targets having a still-awaiting predecessor sent before1s. It gives the
1/932 long and0/10041 short differences above.

Final commit and artifact hashes are supplied to Mayor separately to avoid
self-referential hashes. Bead remains in progress for Round3. No push, merge,
rebase, remote change, credentials, provider, SSH, GPU or completion operation
was performed in this Round2.
