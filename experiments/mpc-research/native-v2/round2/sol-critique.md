# Sol Round2: adversarial review of the frozen independent proposals

Integration-only namespace note: diagnostic/test copies are named
`sol_round2_counterexamples.py` / `test_sol_round2_counterexamples.py` to avoid
colliding with Round1 during repository-wide discovery. Logic is unchanged;
original names/bytes remain at `974923bde924b524ccdc37196f87d22d2169993a`.

Assignment `pr-nv2-sol`; review target commit
`bc217acf9db3f6222026c0dd345db77a379d6151`.
In citations, `R1/` means the exact repository prefix
`experiments/mpc-research/native-v2/round1/`, at that commit, not a moving branch.
Read all three other complete reports, prototypes and test files using
`git show`, plus the freeze manifest. No peer or Round1 file was modified.
No agents, GPU/SSH/provider/network calls, push, merge, rebase or completion
operation were used. Final held-out 160021--160024 was not opened or generated.

## 1. Decisions, not a vote

* **Keep Astra's finite-tail accounting and distinction between prediction
  window, optimized moves and tail resolution.** Its prototype is an accounting
  diagnostic, not a native predictor. Its uncertainty warnings are substantive.
* **Keep Opus's batch-sharing hypothesis, explicit knee nonidentification,
  output-progress-conditioned residual work and strong workload-aware baseline.**
  Do not promote its profiled kernel-busy fit into iteration wall time, a causal
  service law, or a certified capacity lower envelope.
* **Do not adopt Gemini's horizon iff theorem, reservation invariant,
  coefficient interpretation, exact TTFT decomposition or confirmatory evaluator
  as written.** Several have executable mathematical/code counterexamples.
  Keep the separation of CPU return/span/busy and compatible-clock checks as
  engineering intentions, not as evidence of completed qualification.
* **Attempt a whole-policy reduced model before demanding a perfect GPU twin.**
  Parallel batch-telemetry qualification is valuable, but its incomplete physical
  scope need not block randomized policy-response experiments. It must block
  claims that unidentified internals have become measured exclusive service.

These are source-specific conclusions. The fact that several reports recommend
a hybrid model neither identifies its parameters nor establishes its superiority.

### Classification used below

**Executed defect**: the supplied code or a stated equality/theorem has a
specific counterexample executed in the owned CPU diagnostic.
**Proved limitation**: follows from the mathematical estimand/information
structure or directly visible control flow.
**Unsupported assertion**: plausible or implausible, but current evidence does
not identify it. It is not necessarily false.
**Sound idea**: valid under explicitly stated scope, still subject to deployment
qualification. No CPU pass proves native performance.

## 2. Opus: strongest mechanisms, but several overextensions

Sources: `R1/opus.md`, `R1/opus_batch_models.py`,
`R1/test_opus_batch_models.py`.

### Strongest claims worth retaining

The batch law uses scheduled tokens, decode count/context and prefill attention
work with coherent feature units (`batch_features`, lines87--90; `fit_law`,
105--107). Milliseconds per kilo-token and per mega query-key product are not
Gateway cost units. Shared memory-bound overhead versus growing large-chunk
cost is a useful candidate mechanism, unlike the already falsified exclusive
HTTP service reading. Nonidentification of the knee is explicitly reported
(`opus.md:60-70`; `knee_profile`, 118--134).

I reproduced the code on the saved 84 profiled batches: global best grid knee32,
`knee_identified=false`; LOCO MAE 1.36149ms for the hinge, 3.02489ms affine and
26.68850ms per-request serial. These are reproducible **profiled-busy regression
diagnostics**, not new unprofiled service measurements.

`mean_residual_life` and `age_value_held_out` (264--289) explicitly use generated
output count as age and keep seed blocks out of fit. That is a useful
conditional-survival starting point. The class-aware reactive baseline and
separate authorization for budgets near8 are sensible (`opus.md:247-255`).
Exact whole-cost release and non-evicting cap reductions are correctly retained
in the proposed lifecycle (`opus.md:156-175,239`).

### O1. Estimand changes: busy time is neither wall time nor measured host gap

**Unsupported assertion**, `opus.md:77-81,172-175,209-210`.
The 12.31ms coefficient is an intercept in a reciprocal expected
decode-survivor increment model with a realized snapshot gap. The 8--9ms
quantity is profiled batch kernel-busy union. Subtracting these different
run/population/measurement estimands does not identify a 3.6--4.1ms
"not hidden" host component, even though both have units milliseconds.
The cross-run caveat at line80 should replace, not merely accompany, the
physical conclusion at line79.

Continuing decoder output increments count that decoder's output-generating
participations, not every scheduler iteration in the interval. The "exact
iteration counter" claim additionally requires verified one-token participation
in every relevant iteration, no skips/preemption, and qualified commit/sample
boundaries. A scheduler-running request is not automatically scheduled in every
batch. The current snapshots do not establish this parity. Even if parity
holds, boundary and asynchronous commit delays remain in a finite-gap average.
Use matched scheduled-work/commit records, not that assertion, for E1/E5.

`prefill_backlog_s` (220--228) honestly says it overstates short-prompt backlogs.
It serializes one prompt at a time; it cannot itself demonstrate the
short-prompt batching mechanism. `interference_tpot_ms` (231--239) assumes fixed
membership/context and that each prefill chunk replaces one decode iteration.
Those are useful conditional approximations, not identified native
prefill-induced TPOT coefficients. Check partial remaining lifetimes and commit
ordering before using its insertion counts in SLO probabilities.

### O2. Randomized budget is not randomized composition

**Proved identification limitation**, `opus.md:194-199`.
The line199 claim that randomization leaves "support, not confounding" for the
batch law is too strong. Realized composition is a post-action mediator and
is also affected by residual backlog, service fluctuations and observation
selection. Randomizing budget does not randomize every conditional batch
feature, nor guarantee an iteration-local independent residual.

A measured conditional busy-time predictor can be useful without a causal
coefficient interpretation. For transport/service separation or intervention
transfer, record actual composition and drift, test conditional residuals and
collect crossed conditions. Use randomized assignment for the whole-sequence
effect; do not claim mediator causal effects merely from that assignment.

### O3. Knee selection leaks into the stated held-cell diagnostic

**Proved validation limitation**, `run` lines375--386:
`knee_profile(batches)` selects the knee on all target values, then passes that
global knee to `leave_one_condition_out`. Held-cell coefficients are fitted
without that cell, but the selected hyperparameter already saw its labels.
Also the LOCO score pools batch rows rather than equally weighting independent
blocks; it is not a seed-level generalization estimate.

The numerical comparison remains descriptive and reproducible. It is not a
fully out-of-sample model-selection estimate. Freeze the knee independently or
select it within each outer training fold, keep whole episode/block repetitions
together, and report errors near graph/composition boundaries.

The custom NNLS solves tiny normal equations with a small diagonal perturbation
(`_lstsq`, 53--56). Its positive coefficients do not establish rank or parameter
uncertainty. Before using coefficients, require scaled-design condition/rank
checks, solver convergence/KKT residuals and stability across grouped folds.
I did not establish a numerical solver failure on the existing data.

### O4. Capacity and KV screens have conditional, not universal, force

**Sound conditional math; unsupported native envelope**, `batch_time_lower_bound_s`
167--184 and `opus.md:72-75,132-136`.
The hinge convexity argument is a legitimate lower-bound construction **if**
each modeled core term lower-bounds the relevant serialized iteration duration
and the token/sequence assumptions hold. An NNLS mean fit is not such a
uniform lower envelope. A batch count inside that optimistic screen merely
fails to contradict the assumed model; it does not prove physical feasibility.
The test comment "observed2449 is feasible" (`test_opus_batch_models.py:81`)
should be "not excluded by this necessary-condition relaxation."

The through-origin KV fit (`kv_fraction_per_token`, 310--320) uses full prompt
marks of running requests, not verified per-block allocated/completed contexts.
It lacks a deterministic allocator/pending-work/orphan bound. Observed maximum
0.37 is useful descriptive evidence of low pressure on these runs. Neither
"chance constraint inactive below budget70" nor hard future `KV<=0.9` follows
from it. A measured-usage guard can stop new actions; lowering the Gateway cap
cannot enforce a bound against already admitted work.

### O5. EOS, clock origin and progress-age need narrower wording

**Unsupported causal assertion**, `opus.md:96-108`.
Paired EOS disagreement shows that fixed per-logical-offer lengths cannot be
assumed equal across runs. It does not show the noise is endogenous to action,
or isolate batch numerics as the cause. The report itself correctly asks for
same-action repeats. Keep that caveat decisive: an action-independent random
EOS law is still compatible with these observations. Pair selection requires
completion in both arms and hence describes that selected population.

`age_samples` creates one row per possible generated-token age of each completed
length, not one row per live wall-time observation or one equally weighted
request. Thus F6 demonstrates residual-token prediction under that token-age
weighting. It does not establish a wall-time release model, conditional
calibration for surviving congested requests, or censorship handling.
Completed natural-EOS-only inputs do not justify ignoring truncated/failed
requests in a future collection.

**Executed metric mismatch**, `ttft_by_prefill_ahead`, 292--307:
the label is explicitly client-send-to-first, not scheduled-offer-to-first.
In my witness, scheduled0, sent0.4, first0.6 yields function SLO share1,
but research TTFT SLO share0. Its source caveat is sound; calling those shares
"TTFT-good" or the mechanism of composite-good ranking needs qualification.
Earlier requests are selected by eventual completion and client receipt, so
the offline "ahead" count is not automatically an available Gateway feature.

### O6. Horizon and disagreement claims are not stopping theorems

**Unsupported necessity statements**, `opus.md:27-29,259-277`.
Persistence plus useful forecasts suggests where gains may occur. It is not
a proof that an external disturbance must be forecastable for multistep plans
to help: endogenous admission/release cycles and an imperfect continuation can
matter under stationary demand. Conversely even an oracle failing to improve
N3 does not prove memory is short; a dominating cap or nonbinding resource can
make future control freedom valueless despite long memory.

**Executed counterexample to a universal stopping rule**, `opus.md:279`.
One first-action disagreement among100 decisions can concern20 additional
good completions; let the other99 each have one good completion under either
policy. Totals99 versus119 differ by20.2%, with only1% disagreement.
This finite reward table is not a native gain, but disproves the claimed
frequency-only materiality bound. Disagreement and **binding**, affected
offered/cohort mass, persistence and per-disagreement opportunity cost must
jointly inform a stopping rule. A shadow state distribution also changes after
deployment, so even mass-weighting needs caution.

**Opus acceptance gates:** nested/grouped validation; unprofiled matched
batch/commit targets; explicit parity for decoder participation; same-action
EOS repeats; scheduled-origin/available-state labels; empirical versus hard
capacity/KV distinction; materiality based on affected reward mass.

## 3. Astra: sound accounting, not an accepted native control model

Sources: `R1/astra.md`, `R1/astra_horizon_diagnostic.py`,
`R1/test_astra_horizon_diagnostic.py`.

### Strongest claims

`astra.md:276-303` and `score_bounds`/`contrast` (27--74) correctly separate
settled good outcomes from unresolved possible credit and propagate difference
intervals. A timely first-content event is not complete success; missed TTFT
is irreversible, but old failing work still consumes resources. Future
status/output length do not enter a prefix before its terminal event.

The bounds are explicitly trajectory/scenario accounting, not statistical
native uncertainty. This distinction prevents a particularly serious
overclaim. The fixed-cohort and continued-arrival treatment
(`astra.md:255-274,526-550`) is also preferable to changing the scored population
with N or silently draining future contention at every horizon.

I ran the frozen loader/diagnostic on its two named TRAIN members:16 episodes,
13792 rows,8 pairs and10885 eventual composite-good completions. Member hashes
match those in the report. That checks the replay result, not causal inference.

Its nested-class inequality `V_open<=V_tree<=V_perfect`
(`astra.md:326-339`) is sound under the explicitly stated common
objective/model/nested feasible classes. It is not a native optimality theorem.
The warning that a replanning one-move policy already uses later observations
is essential. Command uncertainty, total path latency and no guaranteed fallback
delivery are handled correctly in sections7--8.

### Weakest / still-unearned claims

I found **no proved core accounting defect** in the stated valid-input,
client-prefix scope. Do not invent a failure just to make all three reviews
symmetrical.

* `validate_episode`/`read_train` (77--129) intentionally accept complete
  natural-EOS/rejection cohorts, rejecting collections with failures. This is
  defensible for these already complete evidence files. It is not a general
  evaluator that may discard failed native episodes and keep only clean ones.
  For new data, retain every outcome/failure and separate measurement-invalid
  scope from controller performance. The source correctly disclaims a full
  ledger/actuator/drain audit.
* Bounds are client-prefix retrospective. Source/availability timestamps for
  an actual Gateway consumer and calibrated future arrivals remain required.
  They cannot authorize actions from raw future final rows merely because
  accounting itself masks those labels.
* A finite set of scenario intervals only bounds scores in those scenarios.
  Maximizing over plans selects model error; nesting and common random numbers
  do not provide simultaneous coverage under action selection.
* The recommended hybrid belief and max-plus event graph have no identified
  duration distribution, concrete fitted belief update or measured rollout
  compute in this prototype. Repeated source-level state conservation can
  coexist with wrong native action ranks.
* "One-switch" pruning may remove the useful sequence. The report properly
  asks for exhaustive-small-H comparison. Keep that as an actual gate, not a
  complexity claim substituted for measurement.
* The TRAIN tail result does not calibrate comparable-plan tail uncertainty:
  a whole-sequence contrast with a large margin can resolve despite a tail
  that dominates two close MPC candidates. This caveat is already explicit
  at `astra.md:543-550` and should survive integration.

My executed `tail_witness` uses the same three-offer cohort in both alternatives:
A has bounds[0,3] at1s and eventual3; B has bounds[2,2] and eventual2 (one
rejection). Choosing largest lower bound picks the eventual loser; Astra's
contrast correctly remains unresolved[-1,2]. This **supports**, not refutes,
Astra's warning against a resolution-speed bias.

**Astra acceptance gates:** retain failed/censored outcome accounting for new
collections; qualify consumer availability; distinguish finite-tail uncertainty
from joint model uncertainty; verify native action ranking; measure bounded
event/particle/plan cost; demonstrate any pruning against exhaustive small-H.

## 4. Gemini: useful measurement intentions, unacceptable theorem/contracts

Sources: `R1/gemini.md`, `R1/gemini_timing_contract.py`,
`R1/test_gemini_timing_contract.py`.

### Strongest sound ideas

The actuator is correctly described as cost-budget admission
(`gemini.md:33-40`). CPU return, GPU span and kernel-busy union must differ
(`gemini.md:42-56,71-75`), and the code's span/busy arithmetic is useful **when
the caller supplies valid same-scope finite measurements**. Clock-domain and
boot identity compatibility checking is a useful prerequisite.
Protecting the held-out split and insisting on paired blocks are also correct
intentions. Reproducing the serial capacity contradiction is sound for the
specified empty-initial exclusive-HTTP interpretation.

### G1. The horizon iff/optimality claim is false

**Executed counterexample**, `gemini.md:171-180`.
Two-slot irreversible whole-cost reservations,20 unit-cost offers/slot,
caps16/32, all admitted requests eventually SLO-good: enumerate the eight
three-move plans from empty. High/high/high admits `[20,12,20]` for52 good;
the best three-move score and best one-move-with-common-high-backup score both
equal52. There is real cross-slot coupling, binding admission and no evictions.
Long carry-over is therefore **not sufficient** for a multistep advantage,
even with perfect fixed-demand knowledge. This refutes the iff; it does not
claim a native benchmark result or prove a converse theorem.

The report's 1--1.5s residence example is also not a `<=1s` premise, and the
10--20s pinned-KV / inevitable thrashing account is unsupported for the current
TRAIN workload and permitted action range. Neither empirical settling time
nor average residence proves one-move mathematical optimality.

### G2. The reservation invariant fails under the actuator's own semantics

**Executed defect**, `gemini.md:153-158`.
Start with R0=0, admit four cost8 requests under32, reduce cap to16 without
terminals. R remains32, contradicting
`R(t)<=max(R0,u_applied(t))=16`.
The correct elementary bound is
`R(t)<=max(R0,sup_{s<=t}u_applied(s))`, or the supported global maximum.
A correct proposal must not simultaneously forbid evictions and require
instantaneous feasibility against the reduced cap.

### G3. Decode-survivor coefficients are not an iteration GPU service law

**Unsupported/wrong target identification**, `gemini.md:98-110`.
The existing theta values parameterize milliseconds per expected decode token
in a reciprocal survivor-increment fit. Its Pk is observed native
prefill-mark kilo-tokens, not scheduled query work or all Gateway-admitted
prompt work. It does not identify whole iteration duration, prefill service,
current-stream forward span or a causal interference law.
Calling it the "optimal identifiable model" and reusing these coefficients
as iteration service is not justified by either matching units or source data.

The blanket rejection of all queueing/fluid models at lines27/98 is also too
broad. The capacity contradiction rejects the particular exclusive serial
HTTP interpretation. Marked batch queues, processor-sharing reductions and
empirical controlled-response models are not all falsified.

### G4. The "exact non-overlapping" TTFT decomposition omits intervals

**Executed equality counterexample**, `gemini.md:62-76`.
On one common clock: scheduled0, Gateway receive.01, admit.02, engine
enqueue.05, two GPU chunks[.10,.20] and[.30,.40], sampling.42 and client
first.45. Even granting GPU-prefill as the sum of those two disjoint spans,
the stated terms sum to.30s, not TTFT.45s. Missing intervals are
admit-to-enqueue.03, interchunk wait.10, and sampling/post-forward.02.

The formula also lacks a defined request-level ownership of mixed batches.
Its `RequestTimingTrace` does not contain these batch/chunk boundaries and
therefore cannot enforce that proposed decomposition. One shared
`ClockDomain` label is an assertion by the caller, not a cross-clock mapping;
the trace methods do not validate device/host conversion or receipt availability.
`BatchExecutionInterval.validate` checks interval order/busy length, not whether
its input spans were actually qualified or all finite. These are fixture
containers, not completed timing qualification.

### G5. The confirmatory evaluator contradicts its stated protocol

**Executed defects**, `gemini.md:192,234-244` versus
`NativeFairStatisticalProtocol.evaluate_paired_blocks`, code350--401:

| Witness | Actual frozen-code result | Why unacceptable |
|---|---|---|
| Eight distinct constructed blocks, differences `[1,1,1,1,1,1,1,-1]` req/s | Mean.75, one-sided p9/256=.03515625, `CONFIRMED_SUPERIOR` | The report specifies two-sided; that same sign-flip tail is18/256=.0703125 |
| Same single block repeated eight times | p1/256=.00390625, `CONFIRMED_SUPERIOR` | No unique seed/block or independent-replicate enforcement; pseudoreplication is accepted |
| B rejection count100 versus A1 per1000 offers; B2 overruns per10 actual solves | `guards_passed=true`; reported overrun fraction.002 | Rejection guard is ignored; solver overruns are divided by offers, not solves (true fraction.2) |

These are deliberately constructed data, not allegations about recorded native
runs. They demonstrate that the code cannot enforce the report's claimed
acceptance gate. The original positive tests explicitly use the offer
denominator (`test_gemini_timing_contract.py:267-284`).

Additional **proved limitations**: the code does not enforce the claimed
minimum eight blocks, matching policy names/workload identities, count validity
or randomized assignment/exchangeability. `2**n` enumeration is unbounded despite
the "small N<=12" comment. A zero-effect p-value plus a point estimate>=0.5
does not establish a confidence bound exceeding a practical margin. A
nonsignificant result is not equivalence, despite the ambiguous
`EQUIVALENT_OR_INSUFFICIENT` label. The .5 margin and fixed eight-block plan
have no development precision/power justification in the report.

The result "exact permutation" also needs the actual randomization design
(or a justified symmetry model); merely assigning paired seed numbers does
not make arbitrary sign flips a valid native randomization distribution.
Alternating order can balance a simple trend but cannot automatically cancel
thermal drift, fragmentation or carry-over; use actual randomized/block-balanced
order, drain and hardware records.

### G6. Overhead, failures and constraints are not qualified by object fields

`ProbeOverheadGuard` (265--298) considers a single call, self-reported blocking/
CUPTI booleans and latency/slot fraction. The report demands2% while the default
fraction is5%. Neither threshold measures repeated-probe cumulative cost,
event allocation overhead, observer-induced scheduling changes or goodput loss.
Off/on blocks are needed. "No synchronize" is necessary but not sufficient
for negligible intrusion.

`RequestTimingTrace.is_composite_good` (131--142) has rejection/cancellation
flags but no general complete-success/failure status. Its input validation
does not establish positive output count, terminal stream success or model/
clock provenance, and scoring does not invoke validation. Do not use it as
the all-offered native outcome auditor without a verified-success precondition
and explicit failure/censoring handling.

The 100ms solver condition and "automatically safe fallback" are intentions,
not implemented actuation/fencing. A cap cannot undo old work, a timeout does
not cancel an applied command, and Fixed16 is not certified SLO/KV safe.
A 35s forced drain after stopping scenario arrivals does not eliminate
steady-load horizon bias or solve unresolved rewards; use Astra's accounting
and declared continuing traffic before the real episode drain.

**Gemini acceptance gates:** replace the iff claim with testable hypotheses;
correct the invariant; identify the coefficient target; use a complete
event/clock/availability contract; repair and independently validate the
statistical protocol/denominators; qualify full-path overhead and failures.
None of these is repaired by18 passing fixture tests.

## 5. Changes required to my own Round1 stance

My finite release-alias proof survives: it asserts a finite projection
obstruction, not native state equivalence or unavoidable error for a
distributional predictor. Peers provide no counterevidence to that limited
claim. However, that proof alone does not show a marked batch twin is needed:
a reduced history-to-release distribution can represent both futures.

I would tighten `R1/sol.md` as follows in later integration, without changing
the frozen report:

1. Make Astra's **uncertain comparison** outcome explicit. Do not select the
   largest settled/lower score merely because long-lived good outcomes are
   unresolved. My executed equal-cohort tail witness makes the failure concrete.
2. Do not let my proposed batch "completion interval" inherit a whole-pipeline
   interpretation from current-stream forward events. Fit the actual measured
   scope, and separately predict output-commit/HTTP outcomes. A complete
   decomposition is optional for whole-policy prediction, not a prerequisite
   to any useful experiment.
3. Keep Opus's distinction between output-token progress age and wall age.
   My survival proposal needs qualified committed progress and censoring;
   merely adding another observed aggregate is not an identification strategy.
4. Explicitly allow policy-dependent EOS/success distributions until same-action
   repetitions resolve nondeterminism versus action dependence. Common latent
   prompt draws are variance reduction, not proof EOS is unchanged by policy.
5. Neither proposed1s cadence nor a3-move/5s window is earned. Enumerating eight
   plans is cheap only if belief update, joint scenarios, lifecycle events,
   tails, persistence and transport fit the measured full path. No report
   supplies that deployment result.

## 6. Minimal usable batch-telemetry contract

This is the minimum for a **scoped batch/commit diagnostic**, not a demand to
recover every hidden GPU operation before studying policy response.

| Contract element | Required content / acceptance check |
|---|---|
| Identity | Run/epoch, boot/clock IDs, installed source38 seal, model/runtime flags, cost/observer hash, schema version; new run invalidates old joins |
| Offered/accepted mapping | Pre-frozen TRAIN logical IDs and unique physical IDs; prompt/output caps and hashes; accepted/rejected/failed all preserved; admitted Gateway cost distinct from physical tokens |
| Scheduler work | Monotonic schedule sequence, `step_key`, per-request pre-execution positions and scheduled counts, prompt-length registry; fixed token/sequence limits and unsupported-event flags |
| Worker parity | Same key, actual worker positions/counts, actual/padded shape and graph mode where available; missing/duplicate/repeated work invalidates this scope, never zero-fill |
| Host event scope | Before/after schedule, execute entry/return, matched output-commit and per-request emitted count/finish; source time versus collection-available time kept separate |
| CUDA label scope | Paired current-stream events, stream/device identity, event readiness, elapsed seconds; explicitly "forward-scope", not absolute host time, kernel busy, token sampling or total pipeline |
| Pending/async | Submitted versus committed work, pending keys/attempts and output association; allow overlapping host schedule-to-commit intervals; never add them as serial service |
| Complete capture | Drop/overflow/error counters, required key-set reconciliation, exceptions/cancellations retained, native/Gateway drain and zero unresolved event records before acceptance |
| Outcome scope | Scheduled-client TTFT, terminal complete-success and mean TPOT from verified usage; natural EOS versus cap stop/failure separated; all-offered denominator fixed |
| Perturbation | OFF/ON capture mode and event count/bytes/latency; record hardware timeline; qualify overhead by blocked paired outcomes, not a single self-reported probe latency |

`step_key` hashes request/position/count, not a globally unique epoch or retry.
Pair it with run and sequence/attempt identity. Do not mistake CUDA
`elapsed_time` for an absolute timestamp that can be subtracted from host
monotonic values. Within a verified same host/boot, compatible monotonic clocks
can share a domain; process names alone neither establish nor refute that.
For distinct clocks, retain a measured conversion and uncertainty.

### Independent assessment of Mayor's proposed measurement facade

**Sound reuse strategy, conditionally acceptable for qualification, not yet
qualified code.** Existing components already carry the right narrow scope:

* `prefill_lab/contracts.py:74-99`: `scheduled_work` uses PRE-execution positions
  and `step_key` binds request/count/position.
* `prefill_lab/runtime.py:162-235`: `ExternalCapture` observes external HTTP
  without stripping frontend outputs, unlike the controlled injection trial.
* `prefill_lab/plugin.py:251-290,325-345`: `WorkerRecorder` records current-stream
  events, resolves only ready ends, and explicitly declares other streams/
  sampling unqualified. `gpu_forward_s` is not kernel-busy union.
* `prefill_lab/contracts.py:151-214`: `audit_capture` checks work parity,
  unmatched keys, final progress, pending/overflow and explicitly labels
  overlapping wall spans as nonadditive. Its default `forced_outputs=True`
  must not be inherited for natural-EOS standard Chat; use the supported
  natural-output mode with preserved finish reasons and failures.

Reusing these avoids a second incompatible identity/clock pipeline. Keep
DEFAULT-OFF, no live predictor, no private synthetic request injection, no
deep profiler and no output mutation. Bind the allowed pre-frozen offer IDs
but reconcile **accepted** IDs with the engine: Gateway-rejected offers must
remain in outcomes without being required to appear as engine executions.
Turn capture mismatches/unsupported events into explicit qualification failures;
do not erase the raw run or count its offered failures as nonexistent.

Drain-only event resolution intentionally yields **offline labels**. A value
read after drain cannot become a predictor feature claimed available earlier.
Recording current-stream events can include stream idle/launch gaps and does
not fence unrelated streams; compare only that declared interval. Sampling,
copy, output commit and transport remain separate or partially unidentified.
These limitations do not defeat the facade's measurement purpose.

I have reviewed the existing reuse points and the proposed strategy, **not the
forthcoming facade implementation**. Later source/deployment qualification is
still necessary, including default-off behavior and graph/async parity.

## 7. A whole-policy reduced model can start without perfect decomposition

**Yes**, conditional on valid end-to-end outcomes, available pre-action
histories, applied-command receipts, randomized supported assignments and
contained episode interference. Start with distributional predictions of
release counts and all-offered composite-good under fixed16/fixed32 or a small
predeclared continuation family, conditioned on past mix/progress/pending
observations. Treat internal duration and transport as joint effective
uncertainty rather than falsely separated physics.

This can identify whole-policy intention-to-treat contrasts on the randomized
workload distribution. It does not identify arbitrary same-state action
derivatives, all unexecuted MPC sequences, or a counterfactual GPU scheduler.
Conditional response additionally needs overlap at pre-action states and
enough independent blocks; action-dependent observation selection is retained
in the belief, not cured by conditioning on successful survivors.
Rejected jobs have no observed native completion label, so do not impute their
counterfactual length as factual data.

Attempt simple stratified/persistence/queue-history models first. Freeze
features and evaluate issued fixed-offset release/goodput probabilities,
grouped calibration and policy/action ranking. Use probability/trajectory
uncertainty, not independent point heads for Q,D,R. Accept a richer batch law
only if its extra state improves these action-relevant targets after overhead.
Reduced models remain workload/policy-support specific and require OOD fallback.

## 8. SINGLE next most informative native experiment

**One replicated, randomized fixed-cap x measurement-mode mixed-traffic pilot.**
Not an MPC trial and not a new physics benchmark suite.

Before any GPU use, Mayor freezes a new TRAIN-only protocol/tape/ID manifest
and the facade source/runtime/cost identity, with held-out untouched.
Suggested bounded first stage: two independent new TRAIN blocks, each with
two fixed budgets16/32 x captureOFF/lightON x two same-condition repetitions:
16 short episodes total. Each arm uses the block's same30s offered tape:
short-prompt phase, a predefined mixed short/long microburst phase, recovery.
All are ordinary Chat requests with natural EOS; no injected engine requests,
forced decoder output length, schedule-knob changes or nominal-burst-equals-batch
assumption. Repeated arms get unique physical IDs and shared logical offer IDs.

Randomize/block-balance arm order, qualify initial application, drain before
every arm and resolve the light recorder only after drain. Keep runtime graphs/
async unchanged and record hardware continuously. Impose a predeclared wall/
drain/event capacity bound; stop and preserve failure evidence rather than
relaxing TTL, buffer integrity or measurement scope.

The single pilot jointly asks:

1. Do normal-traffic request/step/worker/commit joins and natural-EOS accounting
   work without unqualified scope changes?
2. Does light capture materially perturb the end-to-end outcome/timing
   distribution? OFF/ON comparisons bound what transferred coefficients mean.
3. Does actual batch/context/pending composition explain effective forward/
   commit outcomes beyond a simple history baseline, and do whole-policy
   fixed-cap rankings change with mix?
4. Are repeated same-cap EOS disagreements comparable to cross-cap
   disagreements? This separates an unsupported action-EOS story from ordinary
   run variability without assuming their equivalence.

**Predeclared progression gates:** exact identities/key reconciliation and
complete outcomes; no pending/overflow or unsupported-event ambiguity in
accepted scope; no synchronizing/deep instrumentation; report native forward
scope accurately; show OFF/ON differences and uncertainty against a
development-chosen perturbation tolerance; inspect design rank/support before
fitting any batch coefficients. Two blocks can reject gross problems and
estimate development variability, **not establish equivalence or controller
superiority**. If perturbation precision is inadequate, retain that uncertainty
and plan additional development blocks before calibration.

This is more informative now than an N1/N3 contest built on an unqualified
service model. It can produce scoped unprofiled labels and a first reduced
policy-response dataset even if sampling/multi-stream physical decomposition
remains unresolved. It does not authorize or execute GPU use in this round.

## 9. Executed receipt and boundaries

Owned files only:

```text
experiments/mpc-research/native-v2/round2/sol-critique.md
experiments/mpc-research/native-v2/round2/sol_counterexamples.py
experiments/mpc-research/native-v2/round2/test_sol_counterexamples.py
```

The diagnostic loads exactly three hash-pinned prototype blobs from the frozen
local Git commit into memory; it never extracts/overwrites peer files or
accesses a remote. The eight focused tests cover the executed witnesses above.
All numeric counterexamples are constructed and explicitly not native data.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/mpc-research/native-v2/round2 \
  -p test_sol_round2_counterexamples.py -v
PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/mpc-research/native-v2/round2/sol_round2_counterexamples.py
git diff --check
```

Local runtime Python3.14.4, stdlib only. **8 focused tests passed.**
Separately executed the frozen Astra TRAIN replay and Opus saved-profile fit,
with results in sections2--3. No new parameters were fitted for deployment.
The Mayor-reported52 original tests were read in source, not rerun here; neither
those tests nor my eight certify any mathematical assertion they do not check.
No installed-vLLM/GPU/performance suite ran, and no new native evidence,
calibration acceptance or performance acceptance is claimed.

Round1 files, peer files, raw archives and held-out remain unchanged. Preserve
the bead in progress for Round3; host metadata stays with Mayor.
