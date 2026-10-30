# Astra Round 1: admission MPC as a falsifiable, partially observed control problem

Assignment: `pr-mpc-astra`. Independent Round 1; stop at the Round 1 barrier.
Source freeze: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Authoring model: GPT-6 Astra, model ID `gpt-6-astra`.

## 1. Decision and evidence status

**Recommendation [proposal]:** investigate a gateway-budget, finite-action,
scenario MPC with exact request reservation accounting, an evolving
age/phase/cohort belief state, and a fixed-policy fallback. Start with
gateway-observable state; add engine progress/pending observations only if their
causal availability and incremental decision value are demonstrated. Do not
deploy MPC yet. A well-tuned fixed cap or hysteretic controller remains the
deployment recommendation until independent policy evidence favors MPC.

The useful distinction is not "MPC versus no model." It is whether a model
predicts **differences between feasible admission policies** accurately enough
to overcome forecast, actuation, observation and switching costs. Lower factual
MAE alone does not answer this. An elaborate scheduler replica is not a
prerequisite, and a generic Q/D/R regression is not sufficient evidence either.

This document contains source inspection, mathematical deductions and proposed
experiments, not a new benchmark. No GPU, HTTP, simulator, test suite, model
fitting or final-trace experiment was executed for this report. No peer
proposal/conclusion was read. The shared audit and protocol were read at their
specified Mayor paths. Product sources inspected here match the freeze:
the local parent is `8e92371`; local HEAD `1841418` additionally contains four
workflow configuration files, not product changes. Existing local modifications
are not findings or implementation from this report.

Evidence labels used throughout:

- **[source]** behavior verified by inspecting the frozen implementation.
- **[historical]** a result reported in a primary repository report, not rerun here.
- **[deduction]** a conclusion under assumptions stated here.
- **[proposal]** an unimplemented design or an experiment to distinguish hypotheses.

### Evidence ledger

| Evidence | Legitimate inference | Explicit exclusion |
|---|---|---|
| [source] `gateway.py:301-330,366-376,588-630` | Weighted reservation admission, release on cleanup, versioned budget updates, nonpreemptive reductions | No dynamic gateway waiting queue or scheduler action |
| [historical] `results/low-cap-frozen-20261021/report.md:46-101` | Frozen forecast MAE 0.7290 versus 0.9676 on 288 development windows; cap32 bias +0.3163 req/s | No same-state intervention labels, policy gain or untouched final test |
| [historical] `results/state-model-20261021/report.md:10-37,134-159` | Nested family selection gives 0.7135 MAE; segment bias/action readiness still fail | Small factual gains do not qualify cap ranking |
| [historical] `results/flow-diagnosis-20261020/report.md:34-124` | Endogenous-history refresh helps some oracle forecasts; cost-in/out errors cancel | Oracle refresh is not available online or a causal decomposition |
| [historical] `results/prefill-queue-confirmation-2ebb185/report.md:36-109` | 144 held queue samples pass their controlled gate; F0 and entry forecasts identical | No correction advantage, HTTP coverage or MPC acceptance |
| [historical] `results/prefill-entry-http-train-rejection-efca3e4/report.md:34-74` | 21/24 readiness checks pass; same recorded case has disjoint acceptable point intervals | Recorded scheduler counts/order do not determine first-commit time |
| [source] `prefill_lab/state_envelope.py:127-166` | Conditional fixed-graph timing envelopes and explicit abstention | TRAIN extrema are neither probability calibration nor universal service bounds |
| [historical, excerpts only] `experiments/prefill-envelope-shadow-v1.md:139-165` | Missing LIGHT predictions and posting-lag failures remain rejected evidence | Cannot reconstruct missing native forecasts or call the corrected sender qualified |

The shared audit's 289-pass/one-skip CPU result is contextual historical
evidence here, not a new execution. Acceptance remains:
`native_state_action_qualified=false`, `http_qualified=false`,
`gpu_performance_qualified=false`, `mpc_policy_accepted=false`.

## 2. Plant, clocks, actions and objective estimand

Let request i have scheduled offer time s_i, gateway receipt r_i, admission
a_i, engine enqueue e_i, client first-content f_i, terminal time z_i,
and success flag S_i. Units are seconds in explicitly identified clock domains.
Not all these times exist for a rejected or failed request.

The primary measured objective for a full offered interval [0,T) remains

    G_T = (1/T) sum_{i: 0 <= s_i < T} 1{S_i=1 and f_i-s_i <= 0.5 s}.

Rejections, connection failures, incomplete streams and late successful streams
contribute zero; the denominator is T, not admitted time or completion/drain
time. Eventual success requires follow-up after the offered interval. An
unresolved observation is censored, not silently successful. Preserve the
existing run's declared timeout/drain rules and report unresolved outcomes.
Do not introduce an E2E SLO that the existing experiment did not declare.

[source] Gateway `first_content` uses gateway observation, not client receipt
(`telemetry.py:332-336`); `end_to_end` uses gateway receipt
(`telemetry.py:338-343`). Thus gateway TTFT is a proxy. When clocks and event
joins are qualified,

    f_i-s_i = (r_i-s_i) + (a_i-r_i) + (e_i-a_i)
              + (engine_first_i-e_i) + (f_i-engine_first_i).

This is an identity between joined timestamps, not permission to add exporter
histogram means from different cohorts. A production controller may not know
s_i at all; it must model pre-gateway delay or explicitly optimize a proxy
while evaluation retains scheduled-arrival TTFT. No future arrival tape is a
controller input, even if the experiment runner already owns that tape.

### Existing actuator

The sole initial control is requested cost budget u_k, applied as B(t) by one
gateway process. Cost units are abstract reservation units, not tokens,
requests, joules or GPU seconds:

    c_i = max(c_min, b0 + b_in * estimated_prompt_i
                        + b_out * requested_max_output_i).

[source] The estimator and fallback are `gateway.py:199-246`. Existing default
coefficients originated in RTX3090 saturation anchors (`gateway.py:29-37`);
they must not be reinterpreted as identified RTX4090 physical service times.
Natural EOS changes service/reservation lifetime, not the reserved c_i.

At each atomic admission event, in actual gateway lock order,

    accept_i = 1{R(a_i-) + c_i <= B(a_i)}.

Accepted requests reserve their full cost until release. Simultaneous arrivals,
budget applications and releases must use a declared event ordering consistent
with the observed gateway serialization; the client schedule alone does not
identify that order. For uncertain order, simulate it as disturbance.

The API admits any finite positive budget; zero is invalid
(`gateway.py:603-616`). Experimentally supported controls are a much smaller
set. Initially consider C16 and C32 only as **candidate** cost budgets requiring
new action qualification on the pinned plant, not certified safe values.
`protected32`, cost C32, B2048 engine tokens per step, and D16 background
decoders have different units and meanings.

Budget reduction example [deduction]: R=30, B_old=32, B_new=16 leaves R=30.
A new cost-1 request is rejected. After cost 15 is released, R=15 and a
cost-1 request can enter. Neither `R := min(R,B)` nor removing 14 cost units
from a modeled queue is legitimate. Raising a budget admits nothing by itself;
it only changes subsequent arrival decisions.

[source] `gateway.py:75-77` restricts the raw prefill HTTP bridge to protected
admission. It cannot simply be combined with the cost-aware budget endpoint.
Natural-EOS cost-aware chat collection and controlled forced-output prefill
collection must be separately qualified and linked by evidence, not renamed.

### Proposed actuators explicitly outside the first scope

Changing chunk size/B2048, per-step token allocation, sequence priority,
prefill/decode shares, preemption, cancellation, request-specific rejection or
gateway queueing would change the plant. They require separate scheduler or
gateway implementation and randomized qualification. The first MPC cannot
pretend to choose which prompt to accept when the actual API chooses only B.

## 3. Minimal state is an observable belief, not three physical scalars

Define the available information filtration I_k as all messages **received by
the controller** by decision t_k, with source timestamps, sequence numbers,
clock domains, version identities and completeness flags. An event timestamp
before t_k in a file delivered afterward is not in I_k.

A useful conceptual plant state is

    x = (gateway lifecycle ledger, engine progress/order/pending work,
         unfinished offered cohorts, service regime, ingress regime).

Only a projection is currently observed. Use

    y_j = h_j(x(t_j)) + measurement_error_j,
    received_j = t_j + transport_delay_j,
    b_k = P(x(t_k), theta | I_k).

Counts can be exact at their sample boundary yet stale at receipt; this is not
well described by independent Gaussian noise. Request token estimates can be
biased, observation loss is discrete, and service regime is latent. A bounded
particle ensemble or interval-plus-scenario belief is appropriate; do not
claim the posterior is identified merely because a filter outputs numbers.

**Minimum model candidate [proposal]:** a request ledger of unfinished
reservations with receipt/admission ages, prompt/max-output/cost marks and
gateway first-content phase; a terminal-outcome ledger for recent cohorts;
the actual budget/version and outstanding command; and a causal arrival/service
scenario context. Q, D and R are derived, not separately fitted independent
states. At present `/metrics` alone cannot supply this complete ledger.

### Inclusion and rejection tests for each candidate state

Each inclusion must improve frozen-origin multi-step prediction **and/or**
decision quality enough to justify collection cost, under whole-episode
holdout. A feature that improves same-origin fit but harms rollout is rejected.

| Candidate | Necessity and incremental prediction hypothesis | Observability, delay/noise, identifiability | Ablation that can reject inclusion |
|---|---|---|---|
| Applied B, version, pending command | Necessary to simulate admission and cap transitions | B/version are current metrics/API acknowledgements; application delay and lost response are uncertain; ownership must be single-writer | No scientific removal: omit pending-command state in delayed-ack falsification and show mispredicted admission |
| Reserved request identities/c_i and R | Exact admission/release composition; equal R need not imply equal next release | Admission/terminal records carry identity and cost; `/metrics` exposes only aggregate R; token fallback is flagged, not assumed accurate | Compare exact ledger to R-only at matched R with heterogeneous costs; retain detailed composition only if release/admission forecast gains justify overhead |
| Awaiting-first Q and streaming D | Captures phase-dependent contention/lifetimes without calling Q an engine queue | Existing `observed_state`; gateway first event can lag engine and does not expose progress; Q+D is admitted active population at consistent boundaries | Compare to in-flight count alone, stratifying prompt mix; drop separate phase feature if no held predictive gain |
| Age distribution, especially near 0.5 s | Required to distinguish identical counts with different residual TTFT slack; oldest alone loses cohort mass | Gateway receipt ages known, scheduled ages generally unavailable; first-content missingness and clock error matter | Compare per-request age, fixed bins with edges 0.1/0.25/0.5 s, oldest-only and no age; tune bins on development only |
| Prompt/max-output composition and queue-order belief | Long prompts and reserved decode maxima affect service and head-of-line risk | Estimated prompt/max-output exist at admission, not exact executed chunks; gateway order is not proof of engine order | Match Q/D/R/ages but vary prompt length dispersion; compare aggregate tokens to per-request marks; remove unjustified order assumptions |
| Remaining decode distribution | Affects future release and prefill/decode interference | Actual EOS unknown; SSE chunks are not token counts; engine emitted tokens need a new qualified feed | Fixed-128, elapsed-time survival and progress-conditioned survival compete; keep only calibrated improvements under natural EOS |
| Pending batches, ready bits, progress/order | May resolve phase aliasing exposed by failed constant-case forecasts | Available only at current pre-ADD engine probe in restricted scope; observer cost, sample age, placeholders and identities essential; unready does not identify residual duration | Gateway-only versus identical causal model plus these observations; if benefit vanishes after age/overhead accounting, do not build continuous engine feed |
| Recent arrivals/marks and burst state | Forecasts exogenous demand, not realized future arrivals | Gateway sees received offers, not offers lost upstream or scheduled future arrivals; marks for rejected requests need event feed | Constant rate, empirical causal block bootstrap and adaptive burst model on chronological shocks; retain complexity only if calibrated reward/action improvements |
| Recent first/terminal outcomes and delayed cohort flags | Corrects effective service/success regime; must evolve during rollout | Success known at terminal; recent unresolved cohort outcomes are censored; sparse outcomes give weak regime identification | No history, frozen history, causally evolved history; future-oracle history stays diagnostic only |
| Service/regime latent state | Allows throughput/transport drift without inventing CPU/GPU causes | Estimate from lagged innovations; not separately physically identifiable when CPU/GPU/transport covary | Static residual distribution versus bounded regime adaptation on later episodes; reject adaptation if confidence becomes overoptimistic |
| KV usage/engine gauges | Potential overload/OOD detector under memory pressure | Scrape age/identity/reset handling required; gauge not per-request progress; physical meaning depends on configuration | Add gauges after other state fixed; reject as predictive state if no gain; retain independently justified invalid-scope alarm |
| GPU utilization, GC, temperatures and large histories | Possible diagnostics, not default controller state | Different clocks/lag, confounding, observer overhead; no demonstrated independent action-conditioned value | Grouped removal and time-shift negative controls; initially exclude from predictor |

For the "necessary" accounting/clock fields, ablation can reject a cheaper
representation, not remove the underlying safety contract. For predictive
features, compare uncertainty calibration, cap-gap error and runtime cost,
not just global MAE. Match effective model capacity where possible to avoid
calling extra parameters evidence for a specific physical state.

[source] `telemetry.py:320-352` currently retains only receipt/admitted/first in
its in-memory active map, although emitted admission records contain marks.
`telemetry.py:377-387` writes on an approximately one-second loop. A proposed
loss-detecting live event feed or coherent marked snapshot is therefore a
real new observation interface, not an already available 50 ms state channel.
Until that feed qualifies, use coarse metrics only for baselines/shadow work.

## 4. Dynamics: exact accounting coupled to uncertain service

### Conservation at a gateway-observation boundary

For a sampling interval, let A be admissions, F first-content observations,
E_Q exits before observed first content and E_D exits after it. Let C_in and
C_out be the corresponding sums of actual reserved costs, not separate
unconstrained predictions:

    Q_{k+1} = Q_k + A_k - F_k - E_Q,k
    D_{k+1} = D_k + F_k - E_D,k
    R_{k+1} = R_k + C_in,k - C_out,k.

Counts have units requests; R/C have cost units. With losses absent,
Q+D equals admitted nonterminal requests. Rejected offers never enter Q/D/R.
An admitted failure releases its reservation but earns zero reward.

For request i use phase Z_i in {awaiting, streaming, terminal}, first-content
and terminal event times and an alive indicator n_i. Then R=sum_i c_i n_i.
Generate C_out from the very requests that become terminal. Do not fit six
independent heads then rely on large feasibility projection to hide conflict.
[source] Existing conservation is in `transition_features.py:64-68`;
`transition_baselines.py:32-56` explains its aggregate ordering approximation.

Reservation release and telemetry terminal are not literally one atomic
operation: cleanup releases then emits terminal (`gateway.py:438-476`).
The estimator needs a coherent snapshot/sequence boundary or a short
release-pending observation state. An apparent transient Q+D discrepancy must
not be "corrected" by deleting a request. Falsify coherence under concurrent
cleanup and metric sampling before assuming sample-wise identities.

### State evolution below those identities

Two service realizations are competing implementations of the same interface:

1. **Gateway semi-Markov surrogate [preferred first candidate].** For each
   active marked/aged cohort, predict joint first-content and terminal event
   hazards conditional on the evolving aggregate state, workload and service
   regime. Draw a shared service-regime scenario to retain correlated stalls.
   A terminal transition from awaiting can represent pre-token failure; after
   first content it includes success/failure. Hazards have units 1/s, with
   interval event probability `1-exp(-integral hazard dt)`. Do not independently
   generate first and terminal events in impossible order. Condition hazards on
   forecast state and action-mediated admissions; no arbitrary direct cap bonus.

2. **Scheduler-informed hybrid [challenger].** Maintain estimated ordered
   prefill residuals, decoder progress and pending effective service debt.
   At event n, choose scheduler work W_n with the frozen scheduler rule;
   consume prompt tokens and emit decode tokens at completion/reap boundaries.
   Model service `g_n = theta' phi(W_n) + epsilon_n` in seconds with nonnegative
   supported costs and correlated residual scenarios. Respect token/sequence
   limits, context, async slots and placeholder updates. CPU return and GPU
   service overlap; they are not both simply added to each request's TTFT.

Both are models, not exact physics. The surrogate risks poor extrapolation;
the hybrid risks a wrong dispatcher/timing approximation. Require identical
state/action/arrival information when comparing them. Prefer the surrogate if
the hybrid's extra information and timing budget bring no decision gain.

For a simple serial GPU lane and overlapping CPU submission, useful structural
relations are

    GPU_finish_n = max(submit_n, GPU_finish_{n-1}) + g_n
    CPU_available_n = submit_n + cpu_return_n
    reap_n >= max(CPU_available_relevant, GPU_finish_n),

with FIFO two-slot reaping and scheduler eligibility specified separately.
These inequalities alone do not define the actual async engine. The current
restricted continuation implements a specific event graph in
`prefill_lab/live_predictor.py:89-174`; a multi-arrival model must reproduce
that graph in its restricted limit and separately qualify all extensions.

### Unknown decode length and request progress

Let L_i be eventual generated length bounded by a validated max_tokens M_i.
For known emitted count l, sample residual length from

    P(L_i-l > q | L_i>l, prompt_class_i, M_i, regime).

If l is not observed, integrate over its belief rather than substitute the
future final length. Cancellations, censoring and errors are distinct outcomes,
not EOS samples. Survival estimates require adequate conditional support;
independent censoring given conditioning variables is an assumption to test,
not guaranteed by timeout logs. Fit a joint dropout model or report sensitivity
when censoring is informative. Forced-output traces identify only the forced
case, not the natural-EOS survival law.

Prompt progress and output progress must distinguish scheduled, in-flight,
committed and gateway-observed work. At an engine cut, pending tokens cannot
be charged again as unscheduled prefill. Ready=true gives zero pending
copy debt in the restricted envelope, not zero remaining request service.

### Histories and rewards must also advance

For any rolling M-bin predictor, retain its ring buffer, not only the sum:

    h_{k+1} = h_k + new_generated_flow_k - expired_flow_{k-M}.

Future arrivals come from scenarios, and future activity/outcome histories
come from the same simulated request events. Terminal success for a cohort
is not observable at its first token. Keep each cohort's first-token SLO flag
until terminal resolution. This avoids appending predicted arrival-cohort
goodput as an immediate completion.

[source] `transition_baselines.py:105-117` advances only Q/D/R while its vector
also uses rolling history. [historical] Truth Q/D/R alone did not fix goodput
in the flow diagnostic. This motivates coherent latent lifecycle propagation,
not copying future truth into the new model or assuming history is causal.

## 5. MPC specification and limits of safety/stability claims

### Optimization variables, sampling and finite search

Initial research configuration [proposal, not measured capacity]:

| Quantity | Initial candidate | Reason and falsifier |
|---|---|---|
| Control cadence Delta | 250 ms | Faster than historical 1 s prediction bins; not a per-request 0.5 s guarantee; compare 100/250/500 ms on development |
| Reward horizon H | 5 s (20 control slots) | Matches historical forecast target and gives time to see reservation drainage; compare 1/2/5 s and longer only if residual burden matters |
| Action alphabet | {16,32}, subject to qualification | Existing development region, not proof of current-plant support |
| Plan family | Constant or at most one switch | At most 40 sequences for 20 slots/two values; contains fixed and delayed-switch comparators |
| Scenario count | 32 initial draws | Computational starting point; test doubling/order invariance of chosen action and Monte Carlo error |
| Event resolution | Actual/scenario event times within slots | Avoid treating all arrivals as simultaneous at the slot boundary |
| Observation target | source age <=50 ms on decision start | New feed qualification needed; one-second logs do not pass |
| Compute budget | <=20 ms solver/model wall time | Must measure p99 and maximum under worst supported state; timeout never extends admission lock |
| Actuation target | <=20 ms to acknowledged application | Measured separately from compute; no zero-delay assumption |
| End-to-end age target | <=100 ms at application | Sum source age, observation handling, compute and command delay; recheck at send/ack |

These are pre-implementation engineering targets, to be revised only on
development feasibility evidence before freeze. They are not native measured
latencies or the existing engine envelope's 5 ms gate. A live 5 ms envelope
cannot be made usable by adopting the looser gateway-controller age policy.
Do not run 1,280 potentially large event rollouts on the engine thread.
Use a bounded external controller; if the budget is unattainable, reduce the
plan/scenario family or retain a simpler baseline instead of hiding latency.

Use common exogenous random draws across plans; preserve correlations between
arrivals, marks, EOS and service where identified. Candidate paths share one
first action and are open-loop afterward; only the first action is executed.
Open-loop plans are nonanticipative. A scenario-specific future action is an
oracle unless a scenario-tree information structure explicitly permits it.
Recede after the next qualified observation, including while draining.

### Reward and risk

For scenario m and candidate plan U, let G_H^m(U) count eventual successful
TTFT hits for offers scheduled in [t,t+H), and let O_H^m be their offered
count. Define loss L_H^m=O_H^m-G_H^m. This includes rejections and failures.
Existing unresolved requests also have action-sensitive future outcomes;
let Delta V_old^m(U) be their expected successful-hit value relative to
continuing the fallback, computed with the same scenarios.

A risk-aware objective in **request equivalents** is

    minimize_U  mean_m[L_H^m(U) - Delta V_old^m(U)]
                + lambda_tail * CVaR_beta(L_H^m(U))
                + lambda_switch * number_of_switches(U)
                + lambda_terminal * terminal_burden(U).

CVaR can be computed with auxiliary eta:

    eta + mean_m[max(L_H^m-eta,0)]/(1-beta).

The primary expected term is equivalent to maximizing expected new-cohort
goodput because offered demand is action-independent in this experiment.
If clients retry in response to rejection, that assumption fails: retries must
be modeled and the offered-request identity/denominator defined explicitly.
Do not introduce a retry-generating feedback model without new evidence.

`Delta V_old` prevents sacrificing already-admitted eligible requests at the
horizon boundary. It does not double count new arrivals. Evaluate eventual
success by continuation under a declared backup policy; continue background
arrivals in the tail. Artificially stopping arrivals at H would give optimistic
completion and release estimates. Tail truncation leaves a value/residual
interval rather than labeling every unresolved first-token hit successful.

Terminal burden is an explicitly approximate cost of residual future
contention, e.g. estimated additional lost hits under backup over a fixed
tail, excluding old-request and new-cohort rewards already counted.
Start with a lexicographic lower-burden tie-break to avoid an arbitrary
penalty. If a learned terminal value is not stable on development, extend
the rollout or compare against the no-terminal model; do not claim infinite
horizon optimality from a convenient terminal term.

Weights are not picked to manufacture an MPC win. With beta initially 0.9
as a research tail diagnostic, report beta 0.95 sensitivity; 32 scenarios are
too few for a serious extreme-tail claim. Set lambda_tail=0 for the
risk-neutral challenger. Choose nonzero weights only on a frozen development
Pareto frontier of mean all-offered goodput, worst-episode loss, switching,
failure rate and timing overhead. Lambda_switch has units requests/switch;
derive it from measured transition overhead or use a tie-break. A dollar or
energy term requires measured pricing/energy and a declared objective change.
Raising/lowering abstract cost B is not itself a monetary cost.

Operational selection rule: compare the candidate to fallback and hold using
a calibrated uncertainty interval on their value difference. Switch only if
the lower bound exceeds a preregistered practical gain margin; otherwise hold
or use fallback. Scenario standard error alone is insufficient because it
omits structural model error and unsupported actions.

### Hard, soft and terminal constraints

Hard enforceable contracts:

- B is finite, positive, in the qualified alphabet/range; one command is in
  flight; no actuation from malformed, expired, mixed-domain or incomplete state.
- Per-event admission/reservation conservation and valid lifecycle ordering.
  `R<=new_B` is explicitly **not** a hard constraint after a reduction.
- Single owner/process, observed application versions, bounded compute/memory,
  supported request/configuration scope and no implicit new actuator.
- A plan begins with a known applied action and respects the controller's
  minimum dwell except a declared emergency/fallback reduction.

Candidate soft constraints are predicted existing-request failure increases,
TTFT miss probability, queue/age burden and switch frequency. Report them as
statistical/model-relative constraints. A per-request p99 TTFT guarantee is
not enforceable by this scalar budget under arbitrary bursts and service
stalls. If all plans violate a predicted risk threshold, report infeasibility,
choose the defined fallback and retain the violation.

Terminal requirement initially is valid nonnegative lifecycle state plus
continued evaluation under backup, not `Q_H=0`. With sustained offers,
emptying every horizon is usually infeasible and unnecessarily rejects work.
A robust invariant terminal set is optional future work, requiring bounded
arrivals/service, a compatible backup and a proved reachable set.

### What can actually be proved

[deduction] Suppose admitted costs are finite and at least c_min>0, B never
exceeds B_max, releases are nonnegative and the gateway serializes checks.
Then

    R(t) <= max(R(0), B_max),
    admitted_nonterminal_count <= max(R(0), B_max)/c_min.

This follows by induction over admission/release events. It is a reservation
and concurrency bound, not a service-time bound; a hung admitted request may
remain forever. If release reflects gateway cleanup before the backend has
stopped processing, the bound is not automatically an engine-work bound.
Backend cancellation/release coupling needs observation and qualification.

Under bounded residual work and a strictly positive minimum effective service
rate, plus reliable cancellation/completion, a no-new-admission policy drains.
The current API lacks B=0. A positive B_stop<c_min can emulate rejection only
if the cost lower bound is validated and that fallback action is authorized.
This is an optional separately tested emergency action, not an assumption
that the current supported C16 fallback drains under continuing load.

Recursive feasibility of the **optimization's accounting constraints** follows
if holding a supported budget always remains an allowed candidate and latency
constraints are soft. It does not imply recursive feasibility of a hard SLO
chance constraint. Standard terminal-Lyapunov MPC arguments require model,
invariance and disturbance conditions not established here; no asymptotic
latency stability or deterministic SLO claim is made.

### Fallback and unknown actuation

Before native authorization: shadow only, no admin commands.
After qualification: use one frozen empirically selected fixed cap on
startup/OOD/stale state/solver failure. It is a conservative operational
comparator, not a mathematically certified latency-safe cap.

A timed-out write may have applied. The API increments versions and does not
provide request-id idempotency or compare-and-set (`gateway.py:618-630`).
Do not blindly retry or issue a competing command. Enter action-uncertain
state; reconcile `/metrics` and the observed version/application ledger,
model both possible budgets until reconciliation, and suspend optimization.
Loss of all connectivity means the plant retains its last applied budget:
the controller cannot promise that a fallback reduction reached it.

Hold after a bad observation; require a preregistered consecutive-valid
recovery period and minimum dwell before resuming optimization. Freeze those
settings on development; log every fallback and its offered traffic, never
score only healthy-controller windows.

## 6. Controller comparison and discriminating predictions

| Policy family | Strength | Failure mode here | Required comparison |
|---|---|---|---|
| Fixed protected concurrency | Simple, bounded request count/waiting | Costs/lengths heterogeneous; different plant mode | Separate fixed-mode baseline; no mislabeled protected32=C32 |
| Fixed cost cap | Exact current actuator; low overhead | Cannot anticipate burst/service shifts | Best fixed cap selected on development, then frozen; indispensable primary baseline |
| AIMD/threshold heuristic | Cheap response to Q/age/failure signals | Delayed feedback, oscillation, cannot immediately shed reservations | Identical observations/action support and tuned development budget |
| Hysteretic cap switching | Noise resistance and dwell without a model | Delayed reaction or missed short opportunities | Strong likely winner if action gaps are uncertain |
| PI/backlog or virtual-queue control | Transparent target tracking | Q is not engine waiting; age mix matters; integral windup on draining | Anti-windup, saturation and nonpreemptive reduction tests |
| One-step/myopic model control | Uses predicted admission/SLO effect cheaply | Ignores reservation lifetime/tail | Same model/scenarios, H one action; determines whether horizon adds value |
| Linear MPC on Q/D/R | Fast convex optimization, interpretable | Hybrid threshold admission, cost composition, age and EOS not captured | Local identification with evolving histories; assess multi-step residuals near cap changes |
| Nonlinear/hybrid MPC | Captures threshold admission and lifecycle | Nonconvex model errors; potentially excessive solve time | Finite candidate search before continuous nonlinear optimization |
| Robust MPC | Conditional worst-case handling if disturbance set valid | Unbounded stalls/arrivals; overly broad sets choose rejection; fixed-graph envelope does not cover branching arrivals | Box-violation and usefulness rates plus supported bounded synthetic proof cases |
| Stochastic MPC | Balances expected goodput and modeled tail loss | Miscalibrated distribution, action support/selection effects | Recommended research candidate; decision-gap calibration required |
| Learning MPC | Can adapt unknown EOS/regime | Online policy changes break calibration; confounding/exploration risk | Initially freeze transition family; bounded residual adaptation only, compared to no adaptation |
| Hierarchical MPC | Slow admission plus fast scheduler decisions | Adds a new actuator and independently interacting loops | Defer; only pursue if scalar admission is shown insufficient and scheduler access authorized |

A queue controller is not automatically worse than MPC. If future regimes
are unforecastable or optimal caps hardly change, it may achieve the same
goodput at lower cost. If a multistep planner never differs from myopic choice,
its computational complexity is not justified by factual horizon accuracy.

## 7. Uncertainty, tails and out-of-distribution behavior

Separate uncertainty into unknown future offers/EOS, latent current progress,
service/transport variability, model parameters/misspecification, and
observation/action delay. Scenario sampling must not collapse these into
independent white noise. Host stalls affect many requests together.

The fixed-membership envelope uses monotone additions and maxima on one
time-independent event graph. Its proof does not extend to budget control
with future arrivals: slower service changes reservation releases, which
changes admissions and graph membership. Corner rollouts can bound the
restricted graph only. Equality of two corner work lists is a useful guard,
not a general theorem for every interior branch of a new arrival-dependent
model. Preserve the existing scope and abstentions.

### Calibration protocol and exact statistical assumptions

1. Fit the transition/survival models on training episodes only. Freeze them
   before calibration. Calibrate residuals by horizon, action/transition
   stratum and workload class where sample size permits; pool with explicit
   caveats otherwise. Do not silently borrow confidence from unmatched hosts.
2. For an upper prediction band, split-conformal calibration uses the
   `ceil((n+1)(1-alpha))` order statistic of n exchangeable calibration scores;
   if that rank exceeds n the valid bound is infinite. The finite-sample
   marginal statement needs exchangeability of calibration and next score.
   Overlapping one-second windows are not n independent episodes.
3. Use episode-level maximum violation scores when claiming simultaneous
   within-episode coverage. That gives marginal coverage for a new
   exchangeable episode, not guaranteed coverage conditional on every cap,
   state or rare burst. A selected action needs simultaneous coverage over
   the evaluated candidate set or a separate frozen-policy calibration
   collection; selecting the narrowest/action-best pointwise interval
   invalidates naive pointwise coverage.
4. Calibration of factual outcomes does not calibrate an unobserved
   counterfactual action gap. Direct randomized intervention comparisons,
   known action assignment, sufficient state/action overlap and repeated
   episodes are necessary to justify that decision interval.
5. For independent episode-level Bernoulli violations, use exact binomial
   confidence limits. With zero failures in n independent episodes, the
   one-sided 95% upper violation rate is `1-0.05^(1/n)`; roughly 299 episodes
   are needed to put it below 1%. Neither 32 scenarios nor 144 correlated
   controlled samples establish a deployed p99 guarantee.

CVaR_beta is a **modeled** tail diagnostic until independent outcomes verify
it. Simulated Monte Carlo precision is conditional on the simulator. Parameter
ensembles do not automatically contain structural error. Empirical TRAIN
minima/maxima can be reported as observed support only, never physical bounds.

### Drift/OOD and admission selection

Monitor source/runtime/tokenizer identities, clock age, request-size/output
support, action transitions, observation completeness, recent arrival-mark
distribution and delayed residual calibration. Define numerical thresholds
using training/calibration plus development false-alarm analysis, then freeze.
State-action support is joint; being between min/max observed caps is not
sufficient. Missing or ambiguous scope triggers fallback, not an extrapolated
optimistic action.

Arrival forecasting should retain bursts and mark correlation via causal
block scenarios, with change-point/stress alternatives. Their weights are
estimated from past data or explicitly assumptions. Under arbitrary
nonstationarity no distribution-free conditional coverage is promised.
Delayed successes make drift detection slow; missing labels are not zeros.
During adaptation, do not claim the original frozen-model conformal theorem
still applies. Recalibrate in a new epoch or state a weaker empirical claim.

Action selection censors outcomes: rejected requests have unknown hypothetical
latency. Report all-offered prediction/coverage availability, conditional
coverage among available predictions and stratified errors separately.
Never score only jointly admitted simulator/native survivors as an all-offered
accuracy result. The historical matched-request TTFT result explicitly has
that survivor limitation.

## 8. Identification and experiment plan

### Stage A: inexpensive model rejection, no new compute claims

Use existing development data with its original chronology and availability
purges. Compare frozen Q/D/R, evolved-history/cohort, age-marked surrogate and
restricted hybrid variants. Replay only information available at each origin;
perturb future events to detect leakage. Reproduce historical coefficients
only under their recorded numerical environment and provenance.

Report at 1/2/5 s: joint flow calibration, state error, successful cohort
goodput error, optimistic cap-conditioned bias, near-transition errors,
cost-in/out error covariance, uncertainty width/coverage, model availability
and CPU latency. Net R error alone cannot establish accurate release flows.
Do not continue a feature search against known cap32 segment errors after
predeclared candidates fail; retain the negative result.

### Stage B: observer/plant qualification before policy evaluation

Require an immutable manifest with source/config/model/cost identities,
one worker, clock-domain/sequence evidence, all offers and outcomes, token
fallbacks, actual applied action times and observer overhead. Demonstrate
marked-ledger availability at the decision boundary and repair neither missing
state nor failed samples with future offline captures.

For engine augmentation, measure gateway-only versus engine-feed observer
overhead with randomized observer-on/off episodes. Pre-ADD snapshots are
post-gateway-admission observations: they can inform a **later** budget action,
not retrospectively authorize the admission that caused the snapshot.
Test state change between snapshot and application and the effect of the
forecast itself on serving latency. Native shadow success remains separate
from native actuation success.

Paid/GPU collection requires a separately approved plan, time/cost ceiling
and human authorization. This report authorizes none.

### Stage C: identify action-conditioned dynamics

Randomize supported budget interventions independently of unobserved service
innovations, stratified by observed backlog/age/workload region. Use balanced
up/down transitions, multiple dwell times long enough to observe release
lags, and fixed reference episodes. Record assignment probabilities and
actual application. Include high R after reductions rather than discard
"transient" samples. Randomize episode order, record host/runtime blocks and
allow predeclared washout or explicitly carry state between treatments.

Whole-episode policy randomization from comparable initial distributions
identifies policy value most directly. Randomized cap blocks identify their
assigned interventions including carryover, not instantaneous same-state
effects after ignoring treatment history. State-conditioned transition
identification additionally assumes adequate state sufficiency and positivity;
nonrandom schedule phase correlated with workload is confounded.

Vary prompt composition at matched aggregate load and vary natural output
length distribution within support. These experiments discriminate simple
counts from marks, age state from average age, and service coefficient
identifiability from mere regression fit. Check design rank/condition number,
parameter uncertainty and held prediction; constrained nonnegative fits do
not cure a rank-deficient experiment. The historical negative decode
coefficients (`results/state-model-20261021/report.md:80-89`) warn against
interpreting short-prefix fits physically.

### Data split and freezing order

Use entire chronological episode blocks, never shuffled overlapping windows:

    TRAIN -> DEVELOPMENT/model selection -> CALIBRATION
          -> VALIDATION/frozen policy readiness -> FINAL reserved trace C.

Training features/labels must be available by its boundary; purge by the full
forecast horizon **and actual label resolution time**, plus declared history
overlap. Earlier unresolved labels cannot cross the fitting cutoff merely
because their arrivals occurred earlier. If any parameter/weight/family is
changed after calibration, allocate fresh calibration. A change based on
validation consumes it as development and requires fresh validation.

The existing six low-cap captures are development evidence, already inspected,
on shared trace/host support. Do not relabel them calibration/final evidence.
Reserve later independent trace/host/episode groups for claims that generalize
to those axes. C stays unopened until policy, fallback, metrics, significance
rule, action support and test count are frozen. If authorized C is used to
revise the system, it ceases to be final and must not be reused as untouched.

### Counterfactual limits

Same offered trace does not imply same state: admissions and completions
depend on prior actions. The historical A/B sign agreement at different states
is not action-choice accuracy. Simulation with oracle arrivals is not an
attainable gain. Importance weighting/doubly robust estimators require logged
propensities, overlap and appropriate sequential assumptions; deterministic
historical schedules and long-horizon feedback can make them unusable.
Do not publish offline policy regret against unobserved true actions.

### Final repeated-policy comparison and inference

Compare the selected MPC against the frozen best fixed cost cap, strongest
development-selected hysteresis/queue baseline and same-model myopic policy.
Include protected concurrency separately if mode changes are within approved
scope. Give all adaptive competitors equal permitted observations, action
support, tuning opportunities and runtime accounting.

Randomize whole policy episodes in host/time blocks; pair trace seeds across
policies where appropriate but do not call requests within an episode
independent. Primary unit is the episode, with paired difference in G_T.
Choose repeat count from development episode variance and a declared minimum
useful effect delta (req/s), not observed final significance. For approximate
paired normal planning, n is proportional to
`(z_(1-alpha)+z_(power))^2 * variance(diff)/delta^2`; validate assumptions or
use prospective resampling/power analysis on independent development blocks.
There is no defensible numerical n or delta without those inputs and an
operational value decision.

Use randomization inference consistent with actual treatment assignment or
episode/block bootstrap under independent-block assumptions. Report all
episode scores and interval estimates, paired and absolute differences,
rejections, failures, TPOT/ITL/E2E distributions, queue/age burden, transitions,
fallback frequency and overhead. Define multiplicity/hierarchical comparison
before final testing; do not declare success against whichever baseline loses.
No repeated peeking or optional stopping without a prespecified sequential
method.

### Negative-result and stop criteria

Reject MPC deployment if any native state/action gate fails, if the primary
gain's lower confidence bound does not exceed the preregistered useful margin
against the required baselines, or if declared failure/latency guardrails
regress. "Inconclusive with wide intervals" is not equivalence or a win.
Reject extra engine state if gains vanish after overhead/age accounting.
Reject long-horizon planning if myopic or hysteresis matches performance
within the prespecified practical margin at lower complexity.
Reject current model family if cap-gap uncertainty overwhelms prospective
gain, or supported perturbations systematically reverse action ranking.
After the finite candidate/ablation budget, stop and preserve all failures.

## 9. Implementable boundaries and targeted falsification

All names below are proposed interfaces, not files implemented in Round 1.
Module ownership should be assigned only after architecture review.

| Module | Input/output contract | Invariant |
|---|---|---|
| Observation adapter | Metrics/events -> `ObservationBatch(source_id, clock_domain, seq_range, sampled_at, received_at, completeness, budget_version, marked_events)` | Causal availability, no silent loss/reset, bounded collection time |
| Ledger/estimator | Previous belief + batch + application history -> immutable `BeliefSnapshot` | Reconciled identities/reservations, phase/cohort state, age and uncertainty explicit |
| Scenario provider | Belief + model seal + RNG seed -> correlated exogenous paths | No future trace or action-dependent resampling; EOS/censoring scope explicit |
| Transition model | Belief/particle + plan + path -> lifecycle rollout | Exact accounting and generated histories; service approximation labeled |
| Risk evaluator | Rollouts + frozen objective/calibration -> value/risk intervals and support flags | No empirical interval promoted to hard guarantee; no reward double counting |
| Planner | Candidate generator + evaluator + deadline -> `Decision` | Includes hold/backup; bounded search; no per-scenario clairvoyance |
| Actuator/shield | Decision + fresh identity/version checks -> command/ack/reconciliation state | Disabled by default, one writer, no blind retries, nonpreemptive semantics |
| Evidence/scorer | Immutable offers/events/decisions -> episode metrics/gates | Full offered denominator and delayed complete-stream outcomes |

A decision record should include observation cutoff and hash, model/support
hashes, candidate plans, scenario seed, all risk/value summaries, reason for
selection/abstention, elapsed solve time, intended action and expiry.
Application records separately bind send, actual applied time, ack receipt,
budget/version and uncertainty. Do not expose credentials in either artifact.
An imported statistical model must not itself receive admin authority.

### Falsification matrix

| Test | Counterexample or perturbation | Required outcome |
|---|---|---|
| Cap reduction | R=30; reduce 32->16; later releases | No clamp/cancellation; admission follows actual remaining reservation |
| Event ordering | Release and cost-1 arrival at same timestamp | Deterministic declared order or uncertainty, never a free release assumed for all ties |
| Heterogeneous reservation release | Equal R with one large versus many small reservations | Correct event-linked release; distinguish when supported, otherwise wide uncertainty |
| Age aliasing | Same Q/D/R; all requests age .49 s versus .01 s | Different on-time-hit predictions without future labels |
| Pending aliasing | Same scheduler counts/progress but ready versus unready batch | Different residual distributions or abstention; no fixed-case certainty |
| EOS versus cap cost | Early EOS under large max_tokens | Release at actual terminal, not after max_tokens; no changing reserved c before release |
| First then failure | On-time first token followed by malformed/incomplete stream | Zero successful-SLO reward; no early success history |
| Frozen-history trap | No new completions in a rollout versus many completions | History ring buffers diverge from generated events, not fixed origin features |
| Endogenous future leakage | Alter every later completion/scrape/arrival | Earlier observation, belief and candidate evaluation unchanged for a fixed scenario seed |
| Lost/delayed observation | Drop event, reorder batch, replay old run/domain | Flag inconsistency, reconcile or abstain, never infer exact missing state |
| Post-admission engine cut | Snapshot arrives after triggering request accepted | It cannot change the recorded earlier admission decision |
| Ambiguous command | Apply budget but lose response; duplicate send attempt | Reconciliation, no unlogged version jump/retry |
| Release/backend divergence | Gateway cleanup while backend termination delayed | Expose hidden engine work or invalidate scope; gateway R is not declared engine occupancy |
| Horizon-edge gaming | Long decoder admitted just before H | Tail/terminal estimate retains burden and outcome uncertainty |
| Nonanticipativity | Different future scenarios but identical prefix | Same initial action; no scenario-specific first control |
| Monotone-envelope overreach | Arrival at a release threshold changes admission between service paths | Restricted envelope rejected for global MPC guarantee; no two-corner proof |
| Solver overload | Max supported population, burst and scenarios | Deadline fallback outside engine loop; misses included in score |
| Cost/tokenizer change | Same apparent workload but altered coefficients/fallback | New identity/support check, not reused calibration |
| Calibration-selection trap | Pick best of many optimistic action intervals | Simultaneous/selected-policy calibration requirement exposed |
| Denominator trap | Reject nearly everything and retain one fast success | High success fraction cannot masquerade as high all-offered goodput |

Existing tests can anchor later implementations:
`tests/test_dynamic_budget.py:25-139` covers API validation, concurrent
admission, retained reservations and release lifecycle; the existing envelope,
live-wire and transition tests supply restricted-scope fixtures. These are not
tests of the proposed MPC. New tests should target the counterexamples above,
not merely assert a chosen architecture's constants.

## 10. ADR, unresolved hypotheses and Round 1 exit

**ADR [proposal]:** select exact gateway accounting plus an age/mark/cohort
belief representation as the common research contract. Select finite-action
scenario MPC as the initial control challenger, not as an accepted product.
Compare a semi-Markov gateway surrogate to a scheduler-informed hybrid under
the same interfaces. Start with cost-budget actuation only. Keep the fixed and
hysteretic policies viable outcomes, not straw baselines.

The most important unresolved hypotheses are:

1. Can causal gateway state distinguish actionable cap effects without a
   high-frequency engine feed? Test augmentation at equal observation budget.
2. Does pending/readiness information explain enough residual variability to
   beat probabilistic gateway state after its overhead? Controlled failed
   points motivate this question but do not identify its answer.
3. Are cap changes beneficial often enough, with enough advance warning, to
   outweigh uncertainty and reservation-drain lag? Only repeated randomized
   policy evidence resolves this.
4. Does coherent age/cohort propagation fix the old flow model's failure,
   or is service/transport misspecification dominant? Future-history oracles
   cannot distinguish those explanations alone.
5. Does natural EOS make reservation lifetime predictable enough for a
   useful five-second plan? Forced-output success is not evidence for it.
6. Are scalar budgets fundamentally too coarse for long-prompt interference?
   If so, first report the negative result; scheduler control is a new project.

Alternative outcomes remain legitimate: fixed C16/C32 selected on fresh
development; hysteresis with the same live state; short-horizon model control;
or deferred MPC pending qualified observations. Robust/tube MPC needs a
defensible disturbance set and invariant backup; hierarchical scheduler MPC
needs new actuators. Neither follows merely from the word "MPC."

Round 1 supplies a source-grounded model, explicit assumptions, control
specification, identification plan, falsifiers and an evidence-based stopping
rule. It does not supply implementation, native performance acceptance or
consensus. The report is available for the Mayor's Round 1 barrier; peer
critique and revisions belong in separate Round 2/3 artifacts after release.
