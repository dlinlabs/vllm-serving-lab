# Sol Round1: identify action-relevant distributions, not fictitious GPU seconds

Assignment: `pr-nv2-sol`. Base: `e75dbdd5a436543f2416a9fc007d805af680ad14`.
Independent Round1; no new peer reports read or conclusions exchanged.
All existing data below are TRAIN/exploratory. No held-out 160021--160024
contents were opened, generated or used. No native service, GPU, SSH, provider
API, push, merge, remote change or completion operation was used for this work.

## 1. Answer and decision boundary

**No currently identified model establishes a native MPC goodput advantage.**
The promising candidate is a **partially observed, batch-aware semi-Markov
lifecycle model**, with a small piecewise nonlinear batch-completion law,
conditional EOS survival, explicit asynchronous pending work, and exact
Gateway reservation accounting. Its learnable object is a conditional
distribution of future releases and SLO outcomes under supported budget
interventions, not a universal per-request GPU demand coefficient.

This is a falsifiable proposal, not a calibrated replacement for the rejected
HTTP service model. Begin with shadow predictions and a strong same-state
reactive/one-move baseline. Promote multistep control only if it improves
action-relevant prediction and then randomized native goodput. If hidden-state
uncertainty overwhelms the action gap, the correct result is abstention or
one-move/reactive control, not a longer planning horizon.

The two useful levels of identification must not be conflated:

1. Conditional **whole-policy/sequence effects** can be identified on the
   randomized TRAIN workload with correct assignment, overlap and outcome
   accounting, even if GPU internals remain partly latent.
2. A transport/service decomposition and a scheduler transition model require
   additional qualified batch measurements. Existing HTTP observations alone
   do not identify them. A fitted observational transition is not thereby an
   intervention-valid transition, especially outside its action/state support.

## 2. Evidence checked and what it does not prove

Paths in this report are repository-relative. These are historical sources
available at the base commit, not new peer Round1 deliverables.

| Source | Fact used | Limitation |
|---|---|---|
| `experiments/mpc-research/system-model.md:1-74` | Ledger plus latent-work belief; cost admission changes future membership | No qualified physical service/SLO theorem |
| `experiments/mpc-research/mpc-mathematical-specification.md:8-148` | Exact lifecycle, causal information set, fixed cohort/tail scoring | CPU engineering specification, not native calibration |
| `experiments/mpc-research/native-identification-and-model-falsification.zh-TW.md`, sections 1--5 | 13792 intervention offers, 32 isolated natural-EOS requests; serial proxy falsified | Eight intervention seed blocks, not thousands of independent trials |
| `experiments/mpc-research/native-gpu-qualification-and-scout.zh-TW.md`, paired-scout table | Short/long prompt workloads can reverse fixed-cap ranking | One seed per cell; not adaptive-controller superiority |
| `results/mpc-native-4090-scout/native-prefill-combined.json:1-33` | HTTP proxy 44.90946ms for p128; bound 2004 versus 2449 | Rejects this exclusive serial interpretation, not all queueing models |
| `experiments/mpc-research/native_observer/native_token_model.py:1-17,69-100,167-192` | Decode-survivor target; realized next snapshot gap; persistence comparator | Disappearing requests excluded; no fixed-future-horizon issued forecast |
| `results/mpc-native-4090-scout/native-token-diagnostic.json:1-34` | Declares offline, no causal/global-throughput claim | The feature phrase "dt ... known at forecast" must not imply it was known at an earlier online origin |
| `experiments/mpc-research/native_observer/native_prediction.py:1-13,88-141` | Effective-time Gateway reconstruction and delayed HTTP labels | Source-time legality does not establish consumer availability |
| `experiments/mpc-research/native_observer/pr33_native_observer.py:15-44,84-94` | Running marks, waiting count, pending count, KV usage; nonempty orphan state unknown | Computed tokens are scheduler bookkeeping, not GPU completion |
| `experiments/mpc-research/native_observer/state_sample.py:36-98` | Receipt/source/identity checks; unmatched Gateway marks retain unknown physical work | Sequential samples are not one atomic plant state |
| `results/mpc-native-4090-scout/native-profile-diagnostic.json:1-139` | 37112 kernels, 84 batches, 840 outside annotations; CPU/span/busy differ | Profiled smoke, not unprofiled service calibration |
| `experiments/mpc-research/native_observer/native_mpc.py:1-48,239-257` | Existing challenger explicitly lacks batching/transport/KV model | Keep unaccepted; do not install HTTP proxy coefficients into it |
| `gateway.py:305-323,366-408,439-452`; `mpc_lab/contracts.py:56-96` | Weighted affine cost, admission check and terminal whole-cost release | Budget 16/32 is COST, not concurrency, GPU seconds or KV bytes |
| `experiments/mpc-research/native-v2/master-research-report.zh-TW.md`, ground-truth section | 1865 accepted/not-native work joins remain unknown; original Test sealed | Missing work is not zero; qualification is not model acceptance |

### Independent raw check

Read only the explicit member
`native-id-train/interventions.json` of
`results/mpc-native-4090-scout/native-identification-train-evidence.tar.gz`.
For `train-160001-1`, initial requests are empty and both initial drain
receipts are true. Direct row counting gives **2449** complete terminals by
`start+90`, versus **2459** including drain. Thus this report did not confuse
cohort completions with in-window completions. With the saved HTTP coefficients:

```text
s = 0.037967790465172464 + 128 * 0.00005423179911128862
  = 0.044909460751417406 seconds
floor(90 / s) = 2004 < 2449.
```

No positive decode cost can rescue the exclusive serial interpretation.
Removing the intercept post hoc would merely replace one unsupported physical
model with another. Existing `prefill_identification.py:109-141` implements
this screen; I did not duplicate or change it.

Raw archive SHA256 fingerprints, computed locally without unpacking:

```text
native-evidence.tar.gz
400e96a79c17ad7a127d4b4a5afb8db24cb0a917cde717c5ba2b23d551aa50b5
native-identification-train-evidence.tar.gz
b86094f1e7d14343efd89741a21430fdd623c0f11381b2e674e4a2353501129f
native-profiler-evidence.tar.gz
13a2e00907c666a692faaee957eb6688990d62d16de45fb688a675612cbff9a2
```

Hashing authenticates these local inputs, not every scientific interpretation.
Only the named TRAIN JSON member was parsed for the independent raw count.
The large profiler trace was not reanalyzed in this round.

## 3. Complete plant, information and clock definitions

At wall/monotonic plant time t define physical state:

```text
X(t) = (L, G, E, B, K, Z, V).
L: exact admitted Gateway request ledger and immutable charged cost c_i.
G: accepted/not-engine-visible work, transport queues and pending cleanup.
E: engine membership/order; per-request completed physical prompt progress,
   emitted output n_i, context length, prefill/decode phase and latent EOS L_i.
B: submitted/in-flight batches, dependency/order, composition and completion.
K: actual KV block allocation, reusable/free blocks, preemption/recompute mode.
Z: service regime (graphs/batch shapes, GPU clock/power and host contention).
V: applied budget/version and unresolved control command.
```

`L_i` is a hidden output length, not Gateway ledger L. Naming collision is
avoided in implementation. No current sensor reveals all of X.

**Action:** set Gateway budget `u in {16,32}` through the already qualified
guarded channel. Actual `u_applied(t)`, not intended command time, determines
admission. Each offer with mark `(p_est,p_phys?,m,c,receipt,logical_id)` gets
`accept = 1[R+c <= u_applied]`; `R=sum_active c`. Raising u does not admit a
request without a new offer. Lowering u does not evict, reprice, or release old
work; R may remain above u. First content does not release c.
Scheduler max sequences and batch-token limit 2048 are fixed deployment
parameters in this design, **not Gateway actuators**. Engine waiting is not a
new Gateway queue.

**Disturbances:** marked offered arrivals (including rejected marks), prompt
semantics, hidden natural EOS, backend failure/cancellation, transport and
cleanup delay, hardware/service drift, prefix/recompute/preemption state.
Future arrivals must come from past-only estimates or an explicitly qualified
external signal, never the evaluator's realized future tape.

**Observations:** Gateway accept/first/terminal ledger and budget receipts;
native running marks, generated counts, waiting/pending counts and KV usage;
offered arrival/mark history; available hardware timeline; HTTP terminal usage
and SLO labels only after their receipt. Engine waiting count without its marks
cannot identify queued token work. Backend cancellation is not proven by a
Gateway cleanup.

For every observation retain source time s, consumer-available time a, clock
domain, epoch, sequence and source hash. At decision t:

```text
I_t = sigma{records with a <= t, prior own commands/receipts}.
b_t = Law(X(t), model parameters | I_t).
```

Core source samples, frontend samples, GPU events and client scheduled time
are different clocks/locations. A clock conversion requires a measured mapping
and uncertainty, not equality of timestamp magnitudes. Freshness 0.5s and
sample width 0.05s are existing qualification ceilings, not accuracy guarantees
or justification for a 20Hz feedback loop.

Continuous batching uses scheduler/submit/GPU-complete/token-send events,
not the controller clock. Admissions during a committed batch affect later
batches only. Maintain a dependency event graph for pending async work;
do not sum overlapping GPU spans as exclusive work or mark submitted work
completed. Sampling can itself depend on utility-RPC scheduling. A model fitted
with realized next sample gap answers a different question from "what will
happen over the next fixed 500ms?"

## 4. Proposed dynamics and identifiable quantities

At each offer/terminal, update R and ledger exactly. At scheduler event j,
predict batch composition using a passive scheduler approximation:

```text
C_j = S(E_j, K_j, pending_j; fixed runtime configuration).
C_j contains per-request prefill query chunks q_ij, contexts k_ij,
decode membership, graph/padding shape and dependency information.
```

S is not a controller. Do not equate offered burst size with actual batch size.
For qualified, completion-aligned batches, start with a small positive
piecewise model for the **identified execution interval**:

```text
log tau_j = beta_mode
          + f_mode(sum q_ij, sum q_ij*k_ij, sum decode_context_ij)
          + interaction_mode(prefill_query, decode_count) + epsilon_j.
```

Use few fixed knots/regimes, positivity and a residual distribution; test
interaction and context terms rather than insisting they exist. Log-link
positivity does not require every coefficient to be positive: graph padding,
occupancy and batching can create locally nonmonotone effects. Fit CPU
dispatch gaps/transport separately only if measured; otherwise retain a joint
completion distribution and admit that the decomposition is unidentifiable.
Elapsed pending duration conditions the residual completion law.

At qualified batch completion update each physical prompt chunk and generated
token count coherently. Do not decrement all queued prompts at a universal
per-token rate. Completion of prompt compute, token sampling, first-content
send and terminal cleanup are distinct transitions. Preemption/recompute is
an explicit mode; unknown mode triggers uncertainty or abstention, not negative
remaining work.

Use conditional survival:

```text
P(L_i > n+r | L_i > n, request marks/semantics, available history).
remaining length = L_i - n, bounded by m_i-n, not equal to that bound.
```

Cap-terminated outputs are censored for natural-EOS estimation; HTTP failures
are not EOS. Completed-at-origin labels only; fit at seed/request level, not
repeated polling rows. Temperature zero does not make different prompts'
remaining lengths known. Preserve prompt-recipe correlation and joint
arrival/mix regimes in scenarios rather than independent marginal sampling.

KV state needs allocation/block-size/context mapping and preemption events.
Current usage is an observation of occupancy, not a calibrated future OOM
probability. Pending/not-visible work makes a zero-demand assignment unsafe.
Fit or bound future block increments including fragmentation and recompute
before introducing a KV chance constraint. Until then, memory safety relies
on the engine's existing enforcement plus operational monitoring; it is not
a theorem of the Gateway MPC.

### Identifiability and observability

Batch features must vary independently enough for the chosen law: varying only
prompt length does not separate dispatch overhead from per-batch service;
varying nominal burst without actual composition does not identify batching.
Use rank/condition diagnostics and profile-likelihood/confidence sets. Reject
an overparameterized interaction when the design has no independent support.
The 84 profiled batches are not 84 independent randomized episodes.

Randomized budget sequences identify intention-to-treat sequence outcomes at
the randomized block level. Actual-applied action is not automatically
randomized if lag depends on load; retain assignment and receipt, report lag,
and define the estimand before conditioning on application. Treating assignment
as an instrument would require separately justified exclusion/monotonicity.
Within-pair states diverge after actions; matching their later wall times does
not restore a same-state contrast.

For local state/action response, independently randomize a decision among both
supported budgets at a pre-action eligibility state, with known probabilities,
then run a declared common continuation. Repeat independent blocks/resets.
State-bin overlap, no interference across drained episodes, consistency of
runtime and complete all-offered outcomes are required. Observational
high-budget/low-latency correlation can otherwise just encode "short prompt."

Not every hidden component needs reconstruction. The practical requirement is
conditional sufficiency for **action ranking and outcome distributions** over
the decision horizon. Test whether omitted history/membership/context improves
held-block prediction conditional on the candidate state. Failure rejects a
Markov closure, not necessarily belief-state control. Passing a finite test
does not prove observability of the scheduler's physical state.

## 5. Counterexamples that change the design

The one CPU deliverable is `sol_counterexamples.py`, with focused tests in
`test_sol_counterexamples.py`, both in this directory. It is an exact
finite-witness diagnostic, **not another synthetic MPC benchmark**.

Two active decoders have identical current costs `(8,8)`, outputs `(1,1)`,
caps `(16,16)` and ages `(1,1)`. Grant an oracle their total remaining output:
10 tokens. Possible hidden residual allocations are `(1,9)` and `(5,5)`.
In the explicitly assumed parallel unit-rate/no-delay toy mechanism, after 2s
their reserved costs are respectively 8 and 16. A new cost-4 offer under
budget16 is accepted in the first world and rejected in the second.

For any deterministic scalar predictor of R using only that common projection,
one of these worlds has absolute error at least `(16-8)/2 = 4` cost units.
The midpoint attains that finite-witness bound. Even **exact total** remaining
work is insufficient: release timing depends on the distribution of work
across request identities and their whole costs. A single fluid workload state
cannot replace a marked release belief.

This proves neither that the toy rate is native-valid nor that these worlds
have equal probabilities, require opposite optimal actions, or are
indistinguishable from their entire histories. Costs are abstract. At horizon0
or9 both projections have equal R again; tests cover this and event ordering.
No contradiction with a calibrated distributional predictor follows: it may
correctly retain both possible futures. Claims of a deterministic exact native
state need much stronger evidence than this diagnostic.

The same program shows a source sample at1.0, available at1.2, cannot be used
for an online decision at1.1 despite its old source time. This is precisely
why offline effective-time joins cannot be called issued online forecasts.

Other action-relevant counterexamples, proposed tests rather than executed
native findings:

| Assumption | Falsifier / experiment | Consequence if it fails |
|---|---|---|
| Full HTTP first residence is exclusive GPU service | Empty-initial 90s capacity inequality, already observed 2449 > 2004 | Rejected; never reuse that proxy as exclusive service |
| Aggregate R or total work determines release | Executed two-world diagnostic above | Keep marked EOS/release distribution |
| Computed tokens represent completed compute | Match submit/completion events while async batches pending; compare bookkeeping | Separate submitted and completed progress |
| A fixed batch law transfers from profiler to normal execution | Randomize instrumentation on/off by block at identical runtime/loads; compare predictions/outcomes | Quantify observer effect or abandon transferred calibration |
| Batch-feature coefficients are identifiable | Cross prompt chunks, actual batch membership and background decode; inspect design rank | Reduce law or acquire new independent conditions |
| Per-request EOS samples are independent/calibrated | Leave prompt recipe/block out; survival calibration including censored caps and failures | Condition on mix/recipe, widen joint uncertainty or abstain |
| The reduced state is action-sufficient | Conditional history/context ablation and randomized actions in overlapping pre-action states | Add a belief/history feature only if predictive value replicates |
| Future arrival pressure is forecastable at useful lead time | Issued forecast vs seasonal/persistence baselines on new TRAIN blocks and regime shifts | Reactive/one-move may be sufficient |
| Gateway release implies backend drain | Induce bounded cancellation in a separately authorized qualification run; track backend work | Model orphan risk, invalidate state rather than assume zero |
| KV usage plus cost predicts memory headroom | Context/mix/preemption sweep with true block counts and pending state | No claimed memory chance constraint until coverage qualifies |
| More lookahead is worth its computation | Matched N=1/N=3 ablation in section7, charged full path overhead | Prefer simpler controller if no practical gain |

## 6. Model alternatives: what earns its complexity

These are structural alternatives to compare, not names for the same fitted
curve. Keep observation/action/outcome definitions fixed.

| Candidate | Identifiable target / strength | Failure mode and discriminating check |
|---|---|---|
| Queueing/fluid | Marked arrivals, admitted work and effective completion distributions; useful load/recovery baseline | Constant service/exclusive serial assumptions fail under batching; release-alias test and burst/recovery residuals |
| Batch state-space | Pending composition, membership/context and piecewise completion dynamics | Hidden composition/async state aliases transitions; compare matched R,D but different contexts/pending work |
| Hybrid semi-Markov | Batch completion, EOS, preemption and cleanup events with elapsed-time conditioning | Mode count explodes; unseen transitions need abstention; mode-stratified calibration |
| Learned transition | Small history encoder or boosted distributional regressor, trained only on supported histories/actions | Leakage, policy confounding and rollout drift; fixed-origin issued forecasts plus randomized action ranking |
| Piecewise nonlinear | Graph/mixed-batch/context-dependent local service law with low parameter count | Sparse regime boundaries/rank deficiency; leave-composition/regime out, report support holes |
| Probabilistic service/survival | Joint duration, EOS, failure/transport distributions; expresses irreducible hidden-state risk | Marginal coverage does not give trajectory or action-selection coverage; block/trajectory calibration and tail sensitivity |

Recommended combination is hybrid bookkeeping + small piecewise batch law +
survival belief, not six competing full simulators. A learned residual is
allowed only after beating persistence/queue/history baselines on action-relevant
labels; retain lifecycle conservation. Do not independently regress Q,D and R
in ways that create phantom releases. More accurate MAE on survivors can
coexist with worse near-EOS release prediction and worse control.

## 7. Control design and the actual multistep question

Use the common research SLO, not a newly invented easier metric. For a fixed
offered cohort C over duration T, and unresolved old admitted requests O:

```text
g_i = complete_success_i
      * 1[client_first_i - scheduled_offer_i <= 0.5s]
      * 1[request_mean_TPOT_i <= 0.05s].
J(U) = E[sum_{i in C union O} g_i | I_t, do(U)].
evaluation goodput = sum_{i in C} g_i / T.
```

Mean TPOT is `(terminal-first)/(output_tokens-1)`, with the existing single-token
zero convention. It is not token-tail P99 or an SSE chunk-gap guarantee.
Rejects/failures score zero; unresolved streams have no realized positive credit.
Old streams already meeting TTFT may still fail mean TPOT/terminal success.
Include those action-sensitive outcomes once in optimization; do not add them
repeatedly to measured episode goodput. Live Gateway cannot assume scheduled
client time is known: qualify an ingress-delay model or use a labeled interval.

The estimator supplies scenarios from b_t, including correlated arrivals/mix,
EOS and service/transport uncertainty. Separate parameter uncertainty across
blocks from process randomness. Three hand-paired quantiles are not calibrated
joint tail coverage. Report score intervals and action gaps across plausible
models. A worst-case set too broad to distinguish plans is evidence for
abstention, not for pretending worst-case reject-all has useful goodput.

**Hard constraints:** supported finite cost budgets, exact ledger releases,
bounded plans/events/compute, valid source/cost/runtime identity, freshness,
CAS/version/expiry and no unresolved command. R<=new_u is not a post-reduction
constraint. SLO/KV chance constraints require joint calibrated coverage under
the deployed policy; no such current qualification is asserted.

**Initial development design, not an optimal setting:** 1s control slots, up to
three optimized moves (eight binary plans), 5s fixed offered prediction cohort
and common continuation/drain. Determine useful horizon from the measured
release/residence and disturbance lead-time distribution, not convention.
Vary horizon only on development partitions; test tail/drain sensitivity and
unresolved reward bounds. First test whether the signal-to-noise and command
latency permit even this cadence. The existing 0.5s observation ceiling may
still be too stale near the 0.5s TTFT boundary.

Run prediction off handler locks with an explicit bounded worker/deadline.
The existing 100ms command TTL is not 100ms of free solver time: its origin
and application deadline must cover the relevant GET/solve/persist/POST/fence
path. Measure p50/p95/p99 and budget each component; do not widen TTL to hide
overhead. Recheck expiry/state version at actuation. Solver exhaustion or OOD
causes observable fallback; timeout leaves an unknown command unresolved until
reconciled. A previously qualified fixed budget is operational fallback, not
a proved latency-safe controller. Before qualification: shadow-only. If writes
cannot be delivered, the last applied budget persists.

### When multiple optimized moves can matter

An admitted long-lived cohort survives the next decision and has nonlinear
interference with a predicted later prompt burst. Current admission has an
irreversible option cost; a deliberate high-now/low-before-burst sequence may
outperform either constant continuation. Conversely, preparing low-now then
high-after-clustered-release may require multiple planned moves. These are
hypotheses contingent on forecastable lead time and membership memory.

A one-move planner with the **same long prediction window** can already price
future congestion from its first action. Therefore "memory exists" alone does
not establish multistep advantage. Additional optimized continuation must
change the preferred first action, or improve later receding actions, compared
with that strong baseline. If all plans have the same first action and later
replanning repairs continuation differences, multistep adds no demonstrated
executed-policy value. An open-loop scenario plan also does not compute the
value of future information; branching may use only observations available
at that future branch, never latent EOS/future tape.

### Discriminating ablation, not horizon inflation

Freeze one state estimator/model/scenario generator, action set, 1s cadence,
5s scoring cohort, old-stream scoring, common continuation and drain.
Compare best development fixed cap, same-state hysteretic/reactive controller,
one optimized move followed by a declared fixed/hold continuation, and three
optimized moves. Give all predictive arms the same 5s prediction horizon.
Test both fixed and hold one-move continuations on development data to avoid
a weak one-move straw man.

On NEW TRAIN seeds randomize prompt-mix/burst transitions with enough
pre-action lead signal to make anticipation possible; pair with a null
unpredictable-onset regime and a stationary regime. Never provide future tape
to a controller. Require matched source availability, randomized policy order,
drained episodes, actual command receipts and full offered outcomes.
Use independent blocks as the uncertainty unit.

Log N=1/N=3 first-action disagreement, binding admissions, projected action
gap, its uncertainty, and actual future cohort burden. A causal randomized
continuation study at eligible disagreement states tests whether the forecast
ranking is useful, not just different. At equal model and scoring window,
N=3 gain specifically in predictable persistent regimes, but not when lead
information or persistence is removed, is evidence for the proposed mechanism.
No disagreements/no binding decisions/all fallback is not successful MPC.
Report both equal artificial compute-delay comparison (isolates planning
freedom) and actual full-overhead deployment comparison (decides usefulness).

## 8. Prediction, OOD and final evidence plan

First issue timestamped, sealed predictions at fixed origins and fixed horizons
(development candidates 0.1/0.5/1/3/5s); qualify target timing resolution.
Measure R releases, first-content/terminal survival, count of all-offered SLO
successes, and batch-completion increments. Resolve vanished request identities
using terminal records; never silently drop them into a survivor-only score.

Compare constant, persistence, queue/fluid and state/history baselines.
Score block-weighted bias/MAE for counts/releases, Brier/log score for
completion/SLO probabilities, survival calibration and simultaneous trajectory
coverage. Include rollout without replacing intermediate states by later
observations; teacher-forced one-step accuracy is not multistep qualification.
Audit calibration separately by mode, request age, prompt mix and action.

Leave paired seed blocks intact. Freeze fit/model choices on TRAIN, calibrate
on separately allocated development blocks, then validation before final Test.
Leave prompt recipe, contexts, burst intensity and composition/regime out for
OOD tests within new development allocations. D1--31/Pk0--8.192k is observed
token-model support, not a universal physical validity box; marginal ranges do
not guarantee joint support. Unsupported states request fallback; report how
often that happens and its effect on goodput.

Action relevance is primary: on randomized supported policy/continuation
comparisons, test sign/ranking of predicted goodput differences and cost of
mistakes, with confidence intervals clustered by independent block. Existing
trajectories do not reveal all unexecuted plans' counterfactual outcomes.
Do not score plans against an oracle simulator trained on the same traces.

Only after these gates, preregister practical superiority/equivalence margins
from development variance and resource/overhead costs. Run fixed/reactive/
one-move/MPC native comparisons with common offered tapes and full drain.
Freeze controller, primary metric, seed assignment and stopping rule before
opening/generating final held-out. Native noninferiority, superiority and
mechanism claims are distinct. Failure to reject a difference is not equivalence.

## 9. Executed receipt and limitations

Executed locally on Python **3.14.4**, stdlib only:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/mpc-research/native-v2/round1 \
  -p test_sol_counterexamples.py -v
PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/mpc-research/native-v2/round1/sol_counterexamples.py
git diff --check
```

**8 tests passed.** Default diagnostic returns R `[8,16]`, minimax point error
4 cost units, differing cost-4 admission at budget16, and rejects the
source-only availability example. Tests cover cap validity, differing
projections, EOS boundary, changed horizon, nonfinite inputs and clock mismatch.
The focused tests exercise the new diagnostic only; no claim that repository
native suites, installed vLLM tests, performance trials or GPU tests ran.
No repository lint/format configuration applies to these stdlib-only research
files in the inspected root manifest; whitespace is checked by Git.

The shell has `python3`, not `python`; the initial raw-inspection command with
`python` failed before execution and was rerun successfully as `python3`.
No dependencies were installed. Archive/raw count checks are evidence audits,
not additional model fitting. No new native parameters, controller, synthetic
performance leaderboard or final-held-out artifact was produced.

The two-world diagnostic establishes a finite projection obstruction only.
The proposed batch law, conditional EOS calibration, useful horizon, compute
budget, OOD handling and native policy gains remain **unverified hypotheses**.
Round1's deliverable is a narrower identifiable target, an executable
counterexample, and experiments capable of rejecting the design.
