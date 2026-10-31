# Astra Round1: identify decision-relevant batch dynamics before adding moves

Bead: `pr-nv2-astra`. Independent Round1, CPU-only, TRAIN/exploratory.
Author runtime: GPT-6 Astra (`gpt-6-astra`); host verification of backend identity
was supplied by Mayor, not independently inspected here.
Research base: `e75dbdd`; working parent:
`a02fbaac749002b960800203844ab10bf3c7eac8` (an automatic checkpoint adding
workflow instructions, not a change to the research model).
No new peer Round1 report was read. No SSH, GPU, provider, API, remote, or
held-out workload operation was performed for this research.

## 1. Answer and present evidence status

**Candidate, not demonstrated winner:** a constrained, probabilistic
batch-event state-space model with an explicit asynchronous pending-work state,
wrapped in a small receding-horizon budget planner. Retain exact admission and
whole-reservation release accounting; learn conditional batch completion and EOS
distributions, not an exclusive per-request HTTP service demand. Compete against
a reduced queue/fluid model and same-state workload-aware feedback before paying
for additional moves. None is currently identified well enough to claim native
MPC goodput improvement.

Multi-step control is worth testing where an admission now commits scarce
capacity past a later decision, and advance knowledge of workload composition or
release timing makes withholding/releasing capacity valuable. Information is
valuable only if a later observation can change a feasible action before its
consequence becomes irreversible. A larger horizon by itself provides neither.
In many supported regimes, a one-move policy with the same forecast and terminal
accounting may already choose the correct first cap. This is an explicit null,
not a failed implementation to hide with more horizon or weights.

**Facts from existing evidence, not findings of a new GPU run:**

* Gateway actions 16/32 are **reservation COST**, not concurrency or scheduler
  tokens. With the recorded estimator, p128/m128 costs about 1.02269 and
  p4096/m128 about 4.03043. Even homogeneous active-count implications differ.
  Cost does not measure GPU seconds or KV bytes [S1,S2].
* The fitted p128 HTTP proxy is 44.90946ms/request. Exclusive serial service would
  allow at most 2004 first services in an initially drained 90s window; actual
  native completions were 2449 in `train-160001-1`. The proxy's **exclusive
  interpretation is falsified**. Subtracting its intercept, rescaling its slope
  to match throughput, or calling it a batch coefficient is not identification
  [S3; `native-prefill-combined.json:1-29`].
* Native TRAIN intervention evidence totals 13792 offers, 11076 natural-EOS
  completions and 2716 rejections over eight independent paired seed blocks.
  Complementary assignment histories diverge after different actions; these are
  not same-state instantaneous causal comparisons [S3].
* The decode-progress candidate's mean grouped-fold token MAE was 1.043 versus
  persistence 1.029; it beat the best MAE baseline in only 1/8 folds. Its
  realized next-snapshot time gap and survivor conditioning prevent calling it
  a qualified fixed-future-horizon online prediction [S3,S6].
* Receipt-qualified joins in the lower-load batch include 1865 Gateway-active
  marks absent from native running records. Unknown physical work is not zero.
  Scheduler `computed_tokens` can include pending asynchronous work [S3,S5].
* Profiling found 84 batch annotations, 37112 GPU kernels, 840 outside the
  annotations. For query128/prefill1/decode0, median span was 67.66ms but busy
  union 10.36ms. Profiled max8/length outputs are not natural-EOS training data.
  These are useful measurement counterexamples, not unprofiled batch calibration
  [S3,S7].

## 2. Plant, state, actions, observations and clocks

### Exact ledger and partially observed physical state

At decision time `t_k`, use `b_k = P(X(t_k), theta | I_k)` plus the exact local
ledger, where `I_k` contains only records actually received by `t_k`.
The proposed state has the following parts:

| Part | Definition and unit | Status at current interface |
|---|---|---|
| Ledger `L` | Per-request ID, immutable cost `c_i`, estimated prompt, output cap `m_i`, receipt/admission/first-content times, terminal/release state; `R=sum c_i` | Gateway-local accounting, not engine occupancy |
| Engine membership `M` | Waiting/running/transport-pending membership, order, prompt progress, committed output count and context per request | Running marks and aggregate waiting observable; full waiting/transport identity is not supplied by S5 |
| Pending batches `B` | Batch identity, scheduled token positions, stream/event dependencies, submit times, readiness, committed-versus-in-flight work | Only pending count currently supplied by S5; richer join needs qualification |
| Residual work `W` | Unexecuted prompt tokens; latent remaining natural-EOS output; remaining time of submitted work | Belief/interval, not equal to prompt sum or output maximum |
| Resource/mode `K,z` | KV blocks/pressure, preemption/cache state, graph/padding regime, hardware drift | Aggregate KV usage observed; allocation transitions and pressure-to-stall map not identified |
| Forecast `F` | Past-only marked-arrival intensity, burst/mix regime belief, EOS/transport/model uncertainty | Must be estimated using available data; no future offered tape |
| Command `C` | Applied cap/version, outstanding operation, same-domain expiry, identity/support seals | Required to distinguish requested, applied and unknown actions |

This is a **structured state representation**, not a proposal to regress on an
unbounded vector of all request IDs. Keep the exact marked ledger for conservation
and reward attribution; use a small batch feature vector and age/context bins for
learned dynamics. A particle may retain individual residual work without fitting
one parameter per request. Treat any missing joining information as latent or
abstain; never silently switch from native to Gateway phase counts.

### Action and disturbance

The only present action is a timed choice `u_k in {16,32}` at the Gateway. At an
offer, `accept_i = 1[R(t-)+c_i <= u_applied(t)]`. Admission adds the whole `c_i`;
exactly-once terminal cleanup releases it. A cap decrease does not evict or
reprice old requests and need not satisfy `R <= new_cap`. A cap increase does
nothing until an offer arrives [S1,S2].

The vLLM configuration's 2048-token scheduler limit and max-sequences32 belong
inside the transition/support definition, **not the action vector**. Changing
them, adding a Gateway waiting queue, or scheduling individual prefill/decode
tokens would require new actuators and a separate causal experiment.

Disturbances are marked offered arrivals, prompt/content mix, natural EOS,
transport and output-delivery delay, cancellation/failure/orphan work, hardware
drift, and internal scheduler/resource events. Sample joint arrival/mix/EOS
scenarios; independent marginals can erase the burst/long-output correlation
that determines goodput. A controlled periodic workload may have a known
announcement, but the ordinary predictor must not see its future realized tape.

### Continuous batching and asynchronous clocks

Requests enter and leave between scheduler iterations. An admitted arrival can
change the *next eligible* batch, not retroactively alter an in-flight batch.
Scheduling, CPU submit return, GPU completion, copy readiness, engine output
commit, Gateway first-content/release, and client receipt are different events.
The total model must advance to these events, not serialize a whole HTTP
residence for every request.

Use a causal resource graph. For event `v`,

    start(v) = max(release(v), end(predecessors(v)), resource_free(v))
    end(v)   = start(v) + duration(v; batch_features, mode, theta)

Serialize only work that actually shares an exclusive resource. Allow CPU
submission, GPU execution and copy/output processing to overlap where qualified.
CPU function return is not a GPU completion fence. Do not insert synchronizing
instrumentation into the performance path to manufacture clean timings.
`prefill_lab/async_predictor.py:103-130` provides a useful existing max-plus
representation, but its unfitted stage costs are not native calibration [S8].

Maintain source-time, receive-time, clock/boot domain and sequence identity for
every record. Advance a belief from its source time to `t_k` through elapsed
unobserved events; do not regard a 0.49s-old valid snapshot as a current state.
The currently qualified 0.5s age / 0.05s sample-width checks are measurement
guards, not evidence supporting 20Hz actuation [S5]. HTTP scheduled-arrival TTFT
also includes client dispatch lag, unavailable to a normal Gateway unless an
explicitly qualified signal/model supplies it.

## 3. Recommended identifiable model family and competitors

Start with a **restricted hybrid semi-Markov batch model**. Conservation and
scheduler eligibility are structural; duration/EOS/arrival uncertainty is
probabilistic. A scheduler transition proposes actual prompt chunks and decoder
steps from a state, subject to the frozen runtime's supported rules. A
duration/commit transition advances them and emits outcomes. No head predicts
admissions, first hits and releases independently.

For batch `j`, candidate features are:

    phi_j = [1, q_j, d_j, sum_decode_context_j,
             sum_prompt_chunks(q_ij * position_ij + q_ij*(q_ij+1)/2),
             padding_j, pending_depth_j]

`q_j` is scheduled prompt tokens, `d_j` scheduled decode tokens, not Gateway
cost. Different prompt chunks at different positions have different attention
work. The existing feature extractor encodes this distinction [S8]. A
low-parameter positive model such as
`log(duration_j) = beta_mode dot scaled(phi_j) + residual_j` is a hypothesis.
Use graph/dispatch modes or justified piecewise regions only when data support
them. There is no assertion that every coefficient is identifiable, every
monotonicity constraint is valid across graph transitions, or all seven features
are needed. Fit a smaller model first; nonnegative mean durations alone do not
make a learned transition physically correct.

With only current snapshots, fit a reduced **distribution of committed progress
and first/terminal events at predeclared future times**, conditional on the
belief and applied action history. Do not claim to have identified each
GPU/transport subcomponent. Richer request-to-batch/event measurements can then
separate selected duration stages. Use conditional effective commit latency if
physical decomposition remains ambiguous; give that model its own explicit
scope rather than relabeling it GPU service.

| Candidate | What it can identify cheaply | Failure risk and discriminating test |
|---|---|---|
| Queueing/fluid | Backlog drift and effective batch throughput by mix/load; useful strong baseline | A scalar rate loses age/context/pending state. Matched backlog but different decode ages or prompt chunk positions should expose errors. Do not require one-server per-request service. |
| Batch state-space | Progress/release distributions from batch features plus pending state | Missing pending-work identity causes state aliasing. Compare identical aggregate states with different known pending batches. |
| Hybrid event model | Conservation, batch membership and non-eviction exactly within supported scheduler semantics | Wrong parity/preemption assumptions. Compare predicted next-batch membership and commits against request-ID-tagged traces before fitting time. |
| Learned transition | Small residual model over constrained physical transitions, possibly better nonlinear effects | Easy interpolation/leakage, weak action support. Block/mix/clock OOD and multi-step conservation tests; unconstrained sequence model is not the default. |
| Piecewise nonlinear | Graph/padding/load regimes with interpretable local durations | Boundary discontinuity and sparse cells. Hold out near-boundary batch shapes, not random rows of the same episode. |
| Probabilistic service/belief | EOS, partial observations and risk/ranking intervals | Miscalibrated tails/joint dependence. Check conditional coverage and goodput-rank uncertainty, not only point MAE. It complements, rather than replaces, the structural families above. |

### EOS, remaining work and KV

For committed output `n`, represent conditional survival
`P(N_i > n+r | N_i > n, prompt/content class, m_i, I_k)`, capped by `m_i`.
Sample `N_i` before simulating future release; do not read that request's later
completion. Age is informative jointly with progress and slowing service; a
survivor-only regression is selected by both workload and congestion.
Treat known stops, truncation, failures and right-censoring as distinct outcomes.
The 22 forced max8 profiler probes must not enter the natural-EOS fit.
Temperature0 does not make the future output length known to the controller.

Prompt residuals require subtracting *committed* work, not all scheduled
`computed_tokens`. If qualified per-batch in-flight counts exist, bound the
committed quantity using them; absent that join, retain multiple compatible
progress states. Full original prompt work may be a conservative scenario, but
cannot be declared a measured residual.

KV risk comes from live contexts, block rounding, pending allocations, reuse,
preemption and orphans, not simply `R`. Aggregate measured occupancy may support
a drift/risk feature. A hard block bound requires verified allocator geometry,
full membership and supported cache/preemption semantics. Without them, unknown
KV or pressure outside training support triggers abstention/qualified fallback,
not a fabricated chance constraint. Even a worst-case output-max block bound
would certify only that supported allocation model, not TTFT or goodput.

## 4. Causal identification and observability: what must be earned

The full hidden state is not observable from `R,Q,D` or even running count plus
KV usage. For example, two states with equal total prompt marks can have
different already-computed chunks and pending batches; two equal decode counts
can have different imminent releases. The useful criterion is weaker than
recovering every latent variable: **do these ambiguities alter the first-action
ranking?** If yes, represent the ambiguity in `b_k`; if no across calibrated
plausible states, a reduced model is adequate.

Randomized 16/32 sequences identify whole-policy effects under their assignment
protocol, not partial derivatives of GPU time with respect to cap. Cap affects
acceptance and subsequent composition. Regressing service on cap after selecting
admitted/surviving requests introduces selection and mediator issues. Likewise,
randomized microburst size changes queue/transport as well as realized batch:
it is not automatically a valid instrument for exclusive batch duration.

Identification pilot proposal (not executed): retain native model/flags, use new
TRAIN blocks; randomize prompt size/chunk composition, offered microburst size,
background decode load/context and budget sequences. Include repeated null/hold
actions and matched drained starts. Record actual assignments and actual batch
exposure; nominal burst4 need not produce prefill batch4. Collect enough crossed
variation that normalized feature design has useful singular values and that
mixed-prefill/decode terms are not aliases for elapsed time or workload cell.
If the design is rank deficient, report identifiable predictions/subspaces or
merge terms; do not publish arbitrary individual coefficients.

Joint qualification must precede fitting: request/batch join, asynchronous
completion ordering, clock conversion, unprofiled overhead, dropped frames,
cancel/orphan paths, and backend drain. Profiled timing is a different environment
until observer-on/off equivalence is measured. Support violations cannot be
repaired by adding them as zero-work observations.

For action-relevant validation, prospectively randomized decisions conditional
on **pre-action** state strata can estimate local treatment contrasts if both
actions have sufficient support and interference is contained within an
episode. Never pair later diverged states as if they were counterfactual twins.
Use seed/episode blocks as inference units; thousands of correlated token rows
do not provide thousands of independent treatment replicates.

## 5. Predictive control: reward population, tail and feasibility

Use the research composite SLO:

    g_i = successful_complete_i
          * 1[client_first_i - scheduled_offer_i <= 0.5s]
          * 1[request_mean_TPOT_i <= 0.05s]

where TPOT is `(terminal-first)/(N-1)` for `N>1`, zero for `N=1`.
It is **not** token-level P99. All offers count in the denominator; rejection,
failure and unresolved completion earn no positive observed reward.

For each decision, fix the target cohort `C`: all already-admitted unresolved
requests whose success is still actionable, plus all scenario offers in a
fixed lookahead window `D`. This population and the scenario draws must be
identical across compared numbers of optimized moves `H`. Count target rewards
once. Old first-hit requests matter: admitting future long prefill can still
spoil their mean TPOT or completion. Previously inevitable misses have zero
possible good credit but still consume capacity.

Optimize `E[sum_{i in C} g_i]` first. Report rejection, TTFT and TPOT miss counts
separately; do not hide them behind an arbitrary mixed-unit penalty. Any switch
cost has units successful requests per switch and needs measured justification.
No chance/CVaR constraint is advertised before joint predictive-tail calibration.

After the optimized moves, continue with the **same declared backup policy**
and continuing scenario traffic, then evaluate the same cohort. Future
non-cohort traffic creates contention but no additional reward in this score.
Stopping all arrivals at the horizon gives a favorable artificial drain; it
does not approximate sustained load unless that is the actual experiment.
A pure drain after a real episode's offer window is valid *cohort evaluation*,
not a steady-state control theorem.

### Finite-tail intervals and a stopping rule

For a rollout prefix, let `K(U)` be already settled good outcomes and `V(U)` the
number of unresolved cohort outcomes still able to meet the SLO. Then

    L(U) = K(U),  B(U) = K(U) + V(U)
    J(U) in [L(U), B(U)]
    J(B)-J(A) in [L(B)-B(A), B(B)-L(A)].

These are accounting bounds *inside the specified trajectory/scenarios*.
They are not native prediction confidence intervals. Ensemble/model uncertainty
must widen the decision comparison separately. Useful policies:

1. Extend only unresolved, action-discriminating tails, with a shared maximum
   computation budget. Stop when one first-action class dominates all others.
2. If dominance cannot be resolved before the deadline, keep the qualified
   incumbent/backup, label the comparison uncertain, and log why. Maximizing a
   lower bound alone may reward fast resolution instead of higher eventual
   goodput.
3. If a learned terminal value replaces the tail, train it on separate
   development rollouts with an explicit unresolved-cohort mask and backup
   policy. Never add terminal value to the same completions already counted.

When aggregating several plans with the same first action, compare the best
lower bound in that class with the best upper bounds of competing classes.
Eliminating one plan does not eliminate its first action if another continuation
remains competitive. Shared scenarios improve variance of differences but do
not establish uncertainty coverage.

This does not prove infinite-horizon optimality, terminal-set reachability,
physical KV safety or recursive SLO feasibility. Bounded positive budgets and
exact releases imply only the reservation invariant
`R <= max(R_initial, max_supported_budget)` (and a request-count bound if all
costs have a positive lower bound). Reject-all would be bounded and useless.

## 6. When additional moves and information can matter

Distinguish three independent lengths: physical prediction window `D`,
number/timing of optimized moves `H`, and tail resolution length `T_tail`.
Increasing all three simultaneously cannot identify the cause of a gain.

**Delay/composition hypothesis:** a high budget admits work that persists across
several control periods. If a predictable short-request burst arrives before
those reservations release, lowering the cap in advance may preserve goodput.
Conversely, expected clustered releases may make a temporarily low budget
unnecessarily reject useful work. A multi-move plan can coordinate these phases.
But a one-move-with-backup evaluated on the same `D,T_tail` may make the same
first decision. Only different, better *executed first actions* count as
evidence for additional moves.

**Recourse is not clairvoyance:** an open-loop multi-move plan chooses one future
sequence across all unrevealed scenarios. A feedback tree can choose different
later actions only after observations distinguish their nodes. Compare:

    V_open = max_U E[J(U)]
    V_tree = max_nonanticipative_policies E[J(policy)]
    V_perfect = E[max_U J(U | full_future)].

For nested feasible classes under the same model/objective,
`V_open <= V_tree <= V_perfect`. The perfect-information gap is an upper bound
on the value of all future information, not an achievable score. It also
includes information no sensor can ever provide. Ordinary receding-horizon
one-move control already reacts to the next observation; it must not be compared
to an artificially open-loop one-move baseline at episode level.

A tiny illustrative decision table (invented counts, **not native evidence**):
after an unknown equal-probability short/long regime is revealed, high versus
low budget gives `(8,3)` versus `(4,6)` good counts. Best open-loop continuation
gets 5.5, perfect timely observation gets 7. Without an observation until after
admission, the achievable information gain is zero. If an earlier irrevocable
admission destroys the later high-budget opportunity, preserving that option
can change today's decision. This is the causal path to test, rather than
assuming entropy reduction is itself useful.

**Dual-control hypothesis:** an action may change how informative future
observations are (admitted progress is missing for rejected work). Exploration
has value only if it improves later *feasible* reward more than its current
opportunity/risk cost. It can also bias the observer. Do not add an uncalibrated
information bonus to online MPC. First evaluate passive recourse and directed
TRAIN identification. Near SLO/KV limits, online probing is not justified by
a speculative future learning benefit.

### Discriminating ablation, rather than horizon inflation

Freeze one observer, identified model, action alphabet, scenario distribution,
backup, `D=5s`, tail method and total wall-clock allowance for a development
pilot. These numbers are proposals, not tuned/accepted settings.
Start at 1s control periods with `H=1,2,3`; all controllers replan every period.
Hold/reject actual stale decisions identically. Include:

| Arm | What differs | Interpretation |
|---|---|---|
| Fixed16, Fixed32 | No prediction/control adaptation | Deployment-supported fixed baselines |
| Workload-aware reactive | Same available state; hysteresis/risk thresholds fitted only on development blocks | Separates state-feature gain from planning gain |
| One-move | Optimize first cap, same backup/forecast/cohort/tail thereafter | Strong myopic-in-control-freedom baseline, not short-sighted scoring |
| Open-loop multi-move | Optimize additional future slots; execute only first | Incremental value of planned control freedom |
| Small feedback tree | Same moves; future branches use only timely sensor observations | Incremental value of anticipating recourse |
| Oracle future, offline only | Unavailable future information | Diagnostic upper bound; never a performance competitor |

Cross the first-action comparisons with forecast lead times shorter/equal/longer
than measured reservation persistence, and with early versus delayed reveal of
workload mix. Delay/drop observations deliberately in replay; a gain surviving
after its necessary information arrives too late is evidence of leakage or
another mechanism. Keep the actual offered tape paired and immutable; predictors
receive only permitted past signals. Use fresh TRAIN/development seeds, not
160021-160024.

Also compare equal **compute**, not merely equal scenarios: more plans with fewer
scenarios may increase selection error. Separate tail-only changes and
state-feature-only changes. Record how often caps differ from One-move,
how often the different caps actually bind at an offer, downstream
cohort-goodput differences, action delays, and fallback time. Identical first
actions, unbinding cap changes, or all-fallback execution cannot demonstrate
multi-step value.

## 7. Compute, deployment boundary and safety

Enumeration is discontinuous but easy to audit for two budget values:
`2^H` plans times `M` scenarios times `E` events, roughly
`O(2^H M E)` transition calls (plus event-queue/membership costs).
`H=3` is eight plans; `H=10` is 1024. A sampled feedback tree adds observation
branches exponentially unless compressed. Do not solve a large mixed-integer
program or train a generic policy before identifying useful state and timing.

Use common random draws; reuse identical plan prefixes; group by first action;
prune on valid reward intervals; cache invariant batch features. First test a
one-switch move-blocked family, at most `2(H+1)` generated sequences before
deduplication, against exhaustive small-H enumeration to quantify missed
opportunities. If learned stage costs cannot support reliable bounds, label
pruning heuristic rather than certified. Maintain a feasible incumbent before
search and finite limits on events, particles, states and tail extensions.

A 100ms proposed solver allowance is separate from the existing 100ms command
TTL and separate again from end-to-end state-to-application latency. Prior
intervention maximum application lag 78.76ms is not solver time. Measure GET,
observer age, estimation, planning, persistence, POST, acknowledgment/fence and
final freshness separately. Begin with 1s slots only if this total measured
pipeline leaves useful decision lead time; otherwise reduce work/cadence, not
inflate command expiry or silently accept stale state.

Run CPU planning outside Gateway handlers/locks, with a separately supervised
worker and bounded data/result messages. Cooperative timeout checks do not
guarantee timely return of a blocked synchronous solve. Revalidate identity,
version, clock and expiry at the existing guarded actuator [S1]. A lost POST
reply is unknown application, not a cancelled command. Reconcile/fence it
before sending another action; when communication fails, the last applied cap
persists and a fallback cannot be promised delivered.

No deployment is authorized by this report/code. Before model/observer/normal
actuation qualification: shadow only. Afterwards use the best empirically
qualified fixed cap on stale input, model/OOD disagreement, unknown work,
deadline exhaustion or uncertainty that makes the action comparison unreliable.
That fallback is operational, not a latency/KV safety theorem.

## 8. Validation and falsification ladder

| Assumption | Experiment that can falsify it | Rejection/next step |
|---|---|---|
| Measurement represents normal execution | Randomized unprofiled observer OFF/ON blocks; join request/batch/commit/readiness and compare overhead, ordering and missingness | If timings change or joins fail, do not fit a native service law |
| Aggregate state is predictively sufficient | Matched aggregate load with different pending depth, context or output-age distributions | Conditional residual/rank differences require richer belief or abstention |
| Batch law generalizes | Cross prompt/chunk positions, decode background/context, graph/padding boundaries; withheld condition blocks | OOD undercoverage or worse than simple baseline rejects detailed law |
| Model is action-relevant | Randomized supported action sequences and pre-action state-stratified decisions | Good token MAE with wrong goodput/release/action ranking is insufficient |
| EOS survival is valid | Natural-EOS prospective fixed-time release forecasts; retain disappearances/censoring and content-class shifts | Survivor-only fit or underestimated long tails rejects release model |
| KV risk is bounded | Supported controlled pressure variation with measured allocation/preemption/orphan accounting | Unexplained stalls invalidate claimed resource constraints |
| Tail treatment preserves decisions | Fixed-cohort tail extension and explicit unresolved intervals; old-cohort and continued-arrival ablations | Rank reversal/double counting rejects score, not proof against MPC |
| Multiple moves help | Section 6 same-model/common-cohort replanning comparison with actual binding-action differences | If improvement vanishes versus strong One-move, retain One-move |
| Timely information has value | Reveal-delay/forecast-lead-time factorial; nonanticipativity tests | No sensitivity where theory requires it suggests leakage or no information value |
| Compute fits control timescale | Full CPU plus observer/write/fence latency and stale-result fault injection under contention | Deadline/freshness failures select fallback; do not report simulated controller as deployed |

**One-step:** issue predictions at predeclared future offsets (e.g.
0.25/0.5/1s in development), log the prediction before its outcome, and score
progress, release probability, first-content probability, composite-good count,
and request censoring. Persistence and empirical workload-stratified rates are
mandatory. Using the realized next-RPC gap as a feature is a separate descriptive
task.

**Multi-step:** recursively predict 1/2/3/5s without replacing hidden future
states with captured states. Score bias, interval coverage, conservation,
distributional metrics and downstream action ranking. Evaluate both corrected
belief updates (when actually available) and no-update rollouts so error
correction is not confused with intrinsic model skill.

**OOD:** leave entire prompt/mix/load/graph/context regimes out, then test
intermediate and boundary conditions; hardware changes are a new domain, not
random-row validation. Measure abstention and fallback, not only retained
in-support errors. Source/cost/clock violations are hard-invalid inputs, not
statistical OOD examples to interpolate.

All existing raw data are TRAIN. Use grouped TRAIN fits now, then newly allocated
Calibration/Validation blocks for uncertainty and design choices. Choose a
minimum practically meaningful paired-goodput improvement and power/precision
plan from development variance before any final test. The final paired metric
is successful composite good completions divided by the common offered duration;
report guard metrics and full drain. Do not equate non-significance with
equivalence or claim eight exploratory blocks establish stable gains.
Held-out 160021-160024 remains unopened, unused and ungenerated here.

## 9. Delivered CPU diagnostic and actual TRAIN result

Files owned by this round:

* `experiments/mpc-research/native-v2/round1/astra.md`
* `experiments/mpc-research/native-v2/round1/astra_horizon_diagnostic.py`
* `experiments/mpc-research/native-v2/round1/test_astra_horizon_diagnostic.py`

The diagnostic reads **only** `native-id-train/interventions.json` and
`native-id-low-train/interventions.json` inside the existing TRAIN archive,
without extracting it. It checks TRAIN paired seeds, IDs, offered tapes, clock
domains and complete natural-EOS/rejection outcomes. It is not a replacement
for `audit_interventions.py`'s full ledger/actuator/drain audit.

For a fixed half-open offered cohort and multiple tail cutoffs it reconstructs
client event prefixes and computes the bounds in section 5. Future terminal
status/output count and future first-content time cannot change a prefix score
before their event. This is retrospective **client** availability, not evidence
the Gateway received these events online. Final truth appears in separately
labeled output for checking eventual containment, never in prefix selection.

The upper bound rules out irreversible TTFT misses, but deliberately does not
use future EOS or unqualified chunk counts to rule out pending TPOT success.
Timely first-content is not terminal success. Bounds can be wide; that is
information about unresolved credit rather than permission to invent a value.
The utility does not fit a model, simulate a plant, choose budgets or issue
requests. Its possible use in a future solver is the accounting/stopping rule,
not using recorded future outcomes to choose a real action.

### Whole 90s offered cohort

Executed on the existing archive; eight paired assigned sequences, **not eight
MPC-vs-baseline comparisons**. Difference below is arm1 minus arm0 good COUNT.

| TRAIN seed | Eventual difference | Difference interval at 90s | Pending terminals arm0/arm1 | First resolving tail on grid (s) |
|---|---:|---|---|---:|
| 160001 | 393 | [383,403] | 10/10 | 0 |
| 160002 | 83 | [71,91] | 8/12 | 0 |
| 160003 | 0 | [-1,1] | 1/1 | 1 |
| 160004 | -2 | [-5,1] | 5/3 | 1 |
| 160005 | 167 | [159,175] | 8/8 | 0 |
| 160006 | -4 | [-7,-1] | 3/3 | 0 |
| 160007 | 3 | [1,8] | 8/4 | 0 |
| 160008 | 3 | [3,3] | 0/0 | 0 |

Eventual composite good total: **10885**, matching 6364+4521 in S3.
At tail lengths `0,0.5,1,2,5,10s`, pending terminals across both batches are
`84,41,6,0,0,0`; still-possible good credit is `77,36,6,0,0,0`.
Seven pending terminals at 90s already cannot meet TTFT; ignoring them for
reward does **not** mean they consume no work. Six of eight pair rankings
are determined without waiting for all terminals; the other two resolve by
1s on this grid. This is deterministic accounting, not statistical significance.

### Fixed five-second cohort [30s,35s)

This second diagnostic keeps that cohort fixed across tail cutoffs while the
recorded episode continues admitting/interfering until 90s. It is not an
artificial no-arrival drain at 35s.

| TRAIN seed | Eventual difference | Difference interval at 35s | First resolving tail on grid (s) |
|---|---:|---|---:|
| 160001 | 89 | [65,103] | 0 |
| 160002 | 12 | [5,23] | 0 |
| 160003 | -1 | [-2,5] | 1 |
| 160004 | 2 | [-1,5] | 5 |
| 160005 | 36 | [26,43] | 0 |
| 160006 | -8 | [-19,0] | 0.5 |
| 160007 | 2 | [-3,5] | 1 |
| 160008 | 1 | [0,3] | 2 |

Only three of eight are resolved at the cohort boundary. The 5s grid result
does not mean exactly 5s is necessary; resolution was not located between 2
and 5s. These chosen TRAIN windows are exploratory, not confirmation that any
particular controller horizon is sufficient. In particular, small tail mass
relative to a large **whole-sequence** margin does not imply small uncertainty
between similar MPC plans. No native rank reversal is claimed from these tables;
the unit test supplies an explicit accounting counterexample where truncation
would rank the lower eventual score first.

### Reproduction and receipt

Environment actually used: **Python 3.14.4**, standard library only.
`python` was unavailable; the successful commands use `python3`.
No packages installed, no native vLLM runtime imported, no GPU tests run.
Nine focused unittest cases passed, including future-label/first-time masking,
strict SLO boundaries, one-token TPOT, rank reversal, monotone containment,
fixed-cohort boundaries, invalid windows and reserved-seed rejection.
An additional executed native replay check verified monotone narrowing and
containment of the eventual arm difference at every cutoff, for all eight pairs
in both reported cohorts (16 pair/cohort checks, six cutoffs each).
It read 13792 rows across 16 episodes. One local timing sample was 0.690s for
archive load and 0.016s for both diagnostics; these are offline CPU timings,
not an MPC solver benchmark or controller-latency qualification.

Code SHA256:
`a0230eb3f2f959a9a58978cd6434e3a221fecc67147b910df5dc2ca9f9d432be`.
Test SHA256:
`d38ae4569e9fc3a7435cd6b1301c8ee8270fdf4a1bd7b0271d0116ce1b823d9c`.
The runtime automatically committed the initial three-file snapshot as
`55bde39858af8856f2e876ed995075da9f1bdb08` before explicit staging.
That checkpoint is preserved, not amended. The final local research receipt
commit carries the requested Copilot coauthor; its hash and this report's hash
are reported outside the report to avoid self-referential hashes.

Run from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/mpc-research/native-v2/round1 \
  -p test_astra_horizon_diagnostic.py -v

PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/mpc-research/native-v2/round1/astra_horizon_diagnostic.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/mpc-research/native-v2/round1/astra_horizon_diagnostic.py \
  --cohort-start-s 30 --cohort-duration-s 5
```

The two raw JSON members actually read have SHA256:

| Member | SHA256 |
|---|---|
| `native-id-train/interventions.json` | `58ca7e53b3a663926f326a7dc752d6a50ebed26f9e357575a2cb4d3977ce56a9` |
| `native-id-low-train/interventions.json` | `cdc7c392437064dfff95f6a98c8fbedea252371bcaf8feed4d78347f60f409cd` |

The legacy GPU experiments, complete native test suite, model calibration,
controller latency and new controller comparisons were **not rerun** in this
Round1. Only the diagnostic/tests and this report are delivered.
No native performance acceptance or actuation authority follows.

## 10. Source ledger

All paths are relative to the repository root. References distinguish source
facts from the proposals and newly executed diagnostic above.

| Ref | Exact source/artifact | Use |
|---|---|---|
| S1 | `gateway.py:305-323,366-460,764-830` | Cost estimator, immediate admission, whole release, guarded version/expiry application |
| S2 | `mpc_lab/contracts.py:47-83`; `experiments/mpc-research/system-model.md`; `experiments/mpc-research/mpc-mathematical-specification.md` sections 1-7 | Cost units, original ledger/belief and control/tail contract; proposal here does not silently inherit unqualified service law |
| S3 | `experiments/mpc-research/native-identification-and-model-falsification.zh-TW.md` sections 1-5; `results/mpc-native-4090-scout/native-prefill-combined.json` | Native counts, causal limits, serial-capacity falsification, missing joins, qualification gaps |
| S4 | `experiments/mpc-research/native-gpu-qualification-and-scout.zh-TW.md`; `results/mpc-native-4090-scout/native-evidence.tar.gz` | Historical workload-dependent ranking and archive provenance; scout raw payload was not reanalyzed in this Round1 |
| S5 | `experiments/mpc-research/native_observer/state_sample.py:14-116`; `experiments/mpc-research/native_observer/pr33_native_observer.py:15-47` | Actual sensor fields, receipt/clock checks, unknown physical marks and pending counts |
| S6 | `experiments/mpc-research/native_observer/native_token_model.py:1-115`; `experiments/mpc-research/native_observer/native_mpc.py:1-50,203-280` | Survivor/gap limits and existing unaccepted serial-prefill challenger; not reused as the new plant |
| S7 | `results/mpc-native-4090-scout/native-profile-diagnostic.json:1-128`; `results/mpc-native-4090-scout/native-profiler-evidence.tar.gz` | Parsed batch/kernel counts and duration distinctions; raw profiler trace not reopened for new fitting |
| S8 | `prefill_lab/async_predictor.py:1-15,103-130`; `prefill_lab/contracts.py:105-154`; `prefill_lab/live_predictor.py:37-89,114-170` | Existing causal event graph, chunk/context features, pending-state checks; restricted prototypes, not new native identification |
| S9 | `experiments/mpc-research/native_observer/audit_interventions.py:72-224`; TRAIN archive member paths/hashes above | Cohort/SLO definitions, expected paired sequence semantics; new diagnostic reads actual rows |
| S10 | `experiments/mpc-research/native-v2/master-research-report.zh-TW.md` | Current acceptance status, independent rounds, resource and held-out boundaries |

**Round1 conclusion:** pursue the smallest identifiable batch/commit belief that
improves action-relevant predictions. Resolve reward-tail uncertainty separately
from increasing control freedom, and test timely recourse against a properly
replanning One-move baseline. Native superiority remains an open experimental
question; these CPU results neither establish it nor rule it out.
