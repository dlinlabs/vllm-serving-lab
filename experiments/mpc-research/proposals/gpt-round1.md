# Round 1: action identification before model-predictive admission

Researcher: onyx / gpt; model: GPT-6.1 Sol (`gpt-6.1-sol`).
Assignment: pr-mpc, model child pr-mpc-gpt.
Source freeze: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Round: independent proposal; no peer proposals read. Final trace C was not read,
executed, summarized, or used for selection. This is an offline research report,
not an implementation, native qualification, or performance acceptance.

## 1. Recommendation and epistemic status

**Recommend a gated, belief-state, discrete-action scenario MPC experiment,
not immediate deployment and not a new scheduler.** Its exact component is the
gateway reservation/admission ledger; its uncertain component is a coherent
request-lifecycle/service model. First determine whether its action ranking and
longer horizon add value beyond a one-step controller and hysteresis. Keep a
development-selected fixed cost budget as the operational fallback. It is an
acceptable research outcome to reject MPC.

The central unresolved hypothesis is not whether another predictor lowers
factual MAE. It is whether information available before a budget update predicts
the *difference* in all-offered successful SLO goodput between supported actions,
including effects on requests already admitted. The policy must outperform
strong non-MPC baselines in fresh randomized repeated episodes. Good forecasting
under an observed schedule is neither necessary nor sufficient evidence of that
claim.

Evidence labels used below:

- **M:** historical measurement reported in an inspected primary report; not a
  GPU experiment or raw-archive audit rerun in this round.
- **S:** inspected source behavior at the freeze.
- **D:** mathematical deduction, with its assumptions.
- **P:** proposed model, experiment, interface, threshold, or decision rule; not
  an existing capability or qualified numerical setting.

Local HEAD is `0183ac9022bca7d19f9a0e23d57e8c093d6851ae`, a startup safety-net
commit above `8e92371`. A read-only diff against the freeze found no changes in
`gateway.py`, `prefill_lab/`, `telemetry.py`, `service_forecasts.py`,
`greybox_simulator.py`, or `forecast_baselines.py`. I did not rewrite history or
make another commit. This report does not treat the safety-net commit as a new
scientific source version.

### Evidence that changes the design

| Evidence | Status | Consequence |
|---|---|---|
| Weighted admission checks `reserved + request_cost <= applied_budget`; reservations survive budget reduction (`gateway.py:301-330,365-376,588-630`). | S | Model a reject-on-arrival actuator, not a controllable gateway waiting queue; never clamp reservation to a reduced budget. |
| Frozen forecast MAE 0.7290 versus trailing baseline 0.9676 on 288 development windows; cap32 bias +0.3163 req/s (`results/low-cap-frozen-20261021/report.md:46-101`). | M | Useful factual forecast evidence, but specifically dangerous optimistic action bias at 32. |
| The 71.1% factual gap-sign result compares different starting states; same-origin cap probes have no measured counterfactual (`results/low-cap-frozen-20261021/report.md:114-139`). | M | Do not call either action-choice accuracy or policy value. |
| Original flow goodput MAE 1.3962; truth Q/D/R resets do not fix it; activity plus outcome history refresh helps; cost-in/out residual correlation 0.9117 (`results/flow-diagnosis-20261020/report.md:34-107`). | M | Evolve history/cohort information and couple flows; net-state accuracy alone is an inadequate model gate. Oracle refresh is not a deployable feature. |
| Nested schedule-group model selection yields only a small development improvement and three regressing captures; segment biases remain (`results/state-model-20261021/report.md:10-37,134-159`). | M | Stop unbounded feature search on these same captures. Do not select by average bias alone. |
| 144/144 held controlled queue samples pass, but F0 and entry predictions are identical on that host (`results/prefill-queue-confirmation-2ebb185/report.md:36-109`). | M | Controlled timing qualification does not establish correction benefit, HTTP validity, or admission action response. |
| Same planned TRAIN case admits no single point satisfying all observed first-commit budgets (`results/prefill-entry-http-train-rejection-efca3e4/report.md:34-74`). | M, D | Recorded scheduler progress is not a complete timing state; use pending-state evidence or uncertainty, not another blind constant. |
| Conditional envelopes exclude later arrivals, force lengths, and keep acceptance/actuation flags false (`prefill_lab/state_envelope.py:1-8,140-180`). | S | They are a conditional timing diagnostic, not an admission safety oracle or a complete MPC transition. |
| First shadow attempt had absent LIGHT forecasts and posting-lag scope failures; the current sender/marker fix has no new native result (`experiments/prefill-envelope-shadow-v1.md:140-163`). | M, S | Unavailable predictions must remain in denominators. Do not claim native usefulness from source repair. |

## 2. System boundary, units, and available control

The controlled plant is one gateway worker plus one identified vLLM engine,
with fixed model/runtime/scheduler configuration. The desired output is complete
successful streams with scheduled-arrival TTFT at most 0.5 seconds, divided by
the entire offered-arrival window. Engine B2048 (tokens/step), gateway C16/C32
(reservation cost units), protected32 (request count), and D16 (background
decoders) are not interchangeable. `greybox_simulator.py:23` still has B8192 as
an old analysis assumption; transplanting that constant into the current native
B2048 model would be an error.

**Existing control (S):**

`u_k = gateway reservation budget`, in V1 dimensionless weighted cost units,
held until a subsequent authenticated `/admin/budget` update applies. Budget
updates are allowed only in cost-aware mode and return applied time and version.
The actuator accepts finite positive budgets; it has no built-in C16/C32
restriction, slew limit, compare-and-set, deadline expiry, or scheduler priority
control. Those are controller/adapter policies if later implemented.

The fixed protected request-count admission limits and semaphore are startup
configuration, not a second online actuator. Raw-token prefill completions
currently require protected mode (`gateway.py:75-77`); a cost-budget controller
cannot simply be attached to that same raw bridge. A future cost-aware HTTP
qualification path must be separately designed and authorized.

**Not existing actuators:** engine token budget, prefill/decode partition,
preemption choice, request cancellation for cap reduction, per-class routing,
dynamic protected concurrency, or a cost-aware gateway queue. These require
different product/runtime work and a different action-identification study.

Reservation cost is the configured gateway estimate:

```text
c_i = max(c_min, c0 + c_p * estimated_prompt_i + c_o * requested_output_max_i).
```

In the default code, `c0=0.259`, `c_p=0.000758 cost/token`,
`c_o=0.0052083333 cost/token`, and `c_min=0.1`. These are a historical V1
reservation convention, not a newly identified RTX4090 service law or physical
GPU-time conversion (`gateway.py:31-39,233-255`). Keep them fixed during a
controller comparison; changing costs simultaneously changes the meaning of
the action. The tokenizer/fallback flag is part of the observed operating mode.

## 3. Partial observation and a minimal *candidate* information state

### 3.1 Physical state is not the observed aggregate

Let physical state `X(t)` contain active request identities, scheduler order,
committed and in-flight progress, pending CPU/GPU work, output/EOS state,
transport buffers, and resource state. The online controller instead maintains
an information set:

```text
I_k = {observations whose available_at <= decision_at_k,
       acknowledged past actions, runtime/clock identities}.
b_k = Law(X(t_k), theta, remaining_lengths | I_k).
```

`b_k` is an approximate belief, not an assertion that the state has been
identified. A practical implementation may use particles or an ensemble of
request ledgers plus latent service regimes. Unknown output lengths and pending
service debt must not be filled with eventual observed values.

There is no established finite-dimensional minimal Markov state in the current
data. The following is the smallest *auditable candidate representation* I would
compare, not an assumption that all its fields deserve permanent inclusion.
Q and D below mean gateway awaiting-first and streaming phases, not exact engine
prefill/decode occupancy.

| Candidate state or memory | Necessity / incremental prediction hypothesis | Online observability, delay, noise | Identification and rejection ablation |
|---|---|---|---|
| Applied budget, version, runtime/run/clock identity | Defines the action that actually governs new admission; disambiguates stale replies and restarts. | Budget/version/sample time exist in `/metrics`; application time/version in update reply. Network receipt is later than sampling. | Exact contract state, not a statistical feature to drop. Inject reorder/restart and require refusal, not an MAE tradeoff. |
| Per-admitted-request reserved `c_i`; total `R=sum c_i` | Exact future admission depends on reservation and release composition, not just active count. | Acceptance/terminal events carry lifecycle/cost; `/metrics` supplies total, not a full request ledger. A live causal ledger feed is proposed, not currently exported by `/metrics`. Loss or duplicate records invalidate exact reconstruction. | Reconcile ledger with sampled total; remove request cost composition but retain R, then compare admission/release and action-gap errors. Keep composition only if it changes decisions usefully. |
| Active phase, prompt estimate, requested output max, admission/first-content ages per request | Distinguishes nearly finished decoder from new long stream; estimates upcoming releases, prefill interference and TTFT margins. | Events and payload-time estimates exist; prompt estimate is not residual prefill. Gateway first-content is delayed relative to engine commit. Current telemetry `active` lacks several payload fields, so do not equate it with this proposed complete ledger. | Compare Q/D/R only, phase-age histograms, and full ledger. Hold whole runs out; test conditional residual autocorrelation and action-response calibration. Reject extra detail if no held improvement exceeds its collection/compute cost. |
| Scheduled-age / ingress-delay belief for pending TTFT cohorts | The primary deadline starts before gateway receipt; receipt-age alone is insufficient. | Benchmark client knows schedule; live gateway generally does not. Do not use future joined client sidecars as live inputs. An authenticated schedule timestamp in a shared qualified clock would be a new input contract; otherwise infer ingress distribution and widen uncertainty. | Remove ingress channel, stratify errors by measured ingress delay on validation, and perturb pre-handler delay. If no causal ingress signal exists, restrict claims to uncertainty-aware scheduled TTFT, not exact per-request margins. |
| Remaining prompt work and decode/EOS residual belief | Total admitted prompt and max output do not reveal current work; residual affects releases beyond next sample. | Latent under gateway-only telemetry. Approximate from observed ages, phase and available engine counters; token-by-token client observations would need a new qualified feed. | Calibrate completion/survival by emitted length where causally available, requested max, prompt mix and service regime. Compare fixed128, conditional survival, and worst-max scenarios. Reject point residuals if interval calibration fails. |
| Explicit ring buffers for recent arrivals and lifecycle flows, plus unresolved cohort labels | Makes a model using rolling predictors dynamically closed; captures short burst/mix/service information not explained by current ledger. | Gateway-received/offered-at-decision history is available only after the handler/estimation executes. Completion-resolved outcomes arrive late. Existing histories are reconstructed offline; live transport must be qualified. | Compare ledger-only against ledger plus arrival history, activity history, and delayed outcome history, with nested whole-run validation. Remove redundant history groups if no incremental gain. A sum without expiration memory is not a sufficient state. |
| Pending engine work, copy-ready bits, schedule/commit anchors, scheduler order | May distinguish equal-progress states with different short-term availability, especially near TTFT threshold. | Available in pre-ADD probes for an already admitted request; no generally qualified periodic pre-admission engine-state RPC exists. Bits are observations, not physical durations. | Optional enhanced observer only. Compare gateway-only and enhanced forecasts using actual availability timestamps and measured overhead; reject if information arrives too late or does not improve action margins. Never use a probe to justify its own earlier admission. |
| Engine running/waiting/KV gauges; prompt/gen rates; preemptions; service-regime belief | May expose saturation, memory pressure, and timing drift. | Scrape-delayed, coarse, resettable; `service_features.py:44-98` gates rates by availability, continuity and age. Original prompt totals and Q are not exact remaining work. | Add gauge/rate groups separately; leave host/runtime groups out. A negative/unstable service coefficient or no incremental held gain rejects a physical interpretation, not necessarily every predictive use. Treat reset/identity changes as unavailable, not zero. |

For each retained group the decision record must include an ablation result,
availability fraction, age distribution, missingness by action/load, and compute
overhead. A state can be physically relevant yet useless to this controller
because it is inaccessible at the decision time. Conversely a predictive
history can be useful without identifying a unique physical cause.

### 3.2 Observation equations and clock discipline

For sensor j:

```text
y^j_k = h_j(X(s^j_k), theta) + e^j_k,
available_at^j_k = s^j_k + transport^j_k + processing^j_k,
age^j_k = decision_at_k - s^j_k.
```

Count/cost ledgers may be exact conditional on complete ordered events;
`e=0` is not justified for prompt estimates, exported gauges, physical residual
service, or engine-to-client TTFT conversion. Availability includes parsing and
validation. Every sample carries clock domain, process identity and sample time.
Do not compare uptime numbers until domains match; do not subtract observation
age from an old snapshot and call that the new engine state. Propagate a belief
through the age interval using possible arrivals, releases and applied actions.

Gateway `/metrics` currently returns Q, D, oldest awaiting age, total reservation,
budget and sample time (`gateway.py:258-286`, `telemetry.py:345-352`). That is a
feasible gateway-only pilot input, not the complete ledger proposed above.
Completed offline capture reconstruction is not proof that the same information
was retrievable online during a gateway stall.

### 3.3 Observable-state aliasing is a falsifiable problem

**D:** `(Q,D,R)` is not Markov even with exact measurement. Two states can have
`(0,1,8)`, with respectively one and 128 remaining decode steps. At 10 ms/step
the first releases eight cost units in 10 ms and the second releases none.
Identical next arrivals can therefore be admitted in one and rejected in the
other. Likewise Q=1 can conceal prompt lengths 512 versus 7168, or different
residual chunks, under the same reservation by compensating requested max.

**D:** Equal recorded progress/order can still alias ready versus unready
pending work. The TRAIN failure is evidence for incomplete observed timing state,
not proof that adding one particular pending variable resolves it. Evaluate
conditional outcome dispersion after adding each sensor; if substantial
dispersion remains, represent it rather than invent deterministic debt.

## 4. Dynamics: exact ledger plus uncertain coherent lifecycle

### 4.1 Admission, action delay and conservation

Sort gateway decisions, terminals and budget applications by actual causal
order, retaining a deterministic sequence number for equal timestamps. A
controller command at `t_k` applies at `a_k=t_k+delta_k`; until then the previous
acknowledged budget governs admission.

For a received request at its locked decision time d:

```text
accept_i = 1{R(d-) + c_i <= u_applied(d)}.
R(d+) = R(d-) + accept_i*c_i.
```

At exactly-once terminal cleanup of admitted request i:

```text
R(t+) = R(t-) - c_i.
```

Failures and cancellations release reservation too; first token does not.
The release is the whole original reservation, not the remaining estimated
compute cost. Floating tolerances in an offline simulator must not silently
change the actual strict admission boundary.

At sample interval k, with admitted count A, first-content count F, terminal
exits before/after first content E_Q/E_D, and accepted/released costs C_in/C_out:

```text
Q_(k+1) = Q_k + A_k - F_k - E_Q,k       [requests]
D_(k+1) = D_k + F_k - E_D,k            [requests]
R_(k+1) = R_k + C_in,k - C_out,k       [cost units]
```

Flows are jointly generated by request transitions, so all are nonnegative and
no identity exits twice. Same-interval admission, first content and terminal
are possible and must be ordered. There is no deterministic aggregation
`A = f(mean offered cost, R, cap)` without request cost/order information.

**Cap-reduction example (D):** R=24, cap changes 32 to 16. R remains 24.
A new cost1 request is rejected. After a cost12 terminal release, R=12 and
cost4 fits exactly. No existing request is cancelled; the time of regained
admission authority depends on residual lifetime. A cap reduction cannot
rescue an already committed GPU batch by undoing its work.

### 4.2 Service and lifecycle transition

For the gateway-only model propose an event-driven semi-Markov request model:

```text
z_i: awaiting -> streaming -> successful terminal / failed / cancelled
     awaiting -> failed / cancelled
hazard_i(event,dt | ages, prompt/max, belief state, engine regime, action history)
```

Fit competing-risk transitions coherently, allowing multiple events within a
sample. Do not fit six unrelated flow heads then rely on large projections to
repair them. Admission itself is mechanistic; the model learns service,
first-content/transport, terminal release, and natural EOS/failure behavior.
Current cap may affect future service through admission composition; include
direct cap terms only if randomized validation supports effects beyond that
mediator, rather than treating cap as a physical service-rate knob.

An engine-enhanced candidate instead schedules token work from observed
committed/in-flight state using the pinned dispatcher. A simple diagnostic law is:

```text
step_wall = positive_overhead(regime)
          + beta_d * decode_tokens
          + beta_p * prompt_tokens
          + supported_context/chunk terms
          + correlated residual.
```

Each beta carries seconds/token or its explicitly scaled equivalent. Attention
context, padding, graph mode and CPU/GPU overlap may require the existing
six-term effective lane instead of this simple law. Estimate only supported
terms; nonnegative coefficients prevent physically negative work but do not
identify them when regressors are collinear. Counter deltas cannot reconstruct
unrecorded per-step chunk composition.

Enforce actual B2048 and scheduler sequence/context limits if invoking the
native dispatcher model. In asynchronous execution:

```text
gpu_finish_j = max(gpu_finish_(j-1), submit_j) + gpu_service_j
commit_j >= max(gpu_finish_j, required_CPU_ready_j, prior_commit_dependencies).
```

CPU and GPU overlap is not a sum of independent measured wall intervals.
`prefill_lab/live_predictor.py:89-175` provides a useful fixed-membership
continuation and readiness/debt checks, but adding arbitrary arrivals, natural
EOS, failures, changing scheduler membership and HTTP delay is a proposed
extension. Endpoint monotonicity of its fixed event graph does not establish
monotonicity once admission and future membership depend on timing.

### 4.3 Histories and delayed reward state

If a learned head consumes an L-step rolling count, retain its ring buffer:

```text
h_(k+1) = h_k + new_events_k - events_(k-L).
```

For ages, advance all surviving request ages by elapsed time and initialize new
phase age at the actual transition. For unresolved outcomes retain cohort ID,
scheduled time or ingress belief, first-content status/margin, and terminal
status. A first-content hit enters an *unresolved-success* bucket, not completed
goodput. Only a complete successful stream finalizes a positive reward.

The code currently freezes history covariates in `FlowModel.forecast`
(`transition_baselines.py:88-126`) and explicitly declares that approximation in
`transition_features.py:20-26`. Repairing it means evolving all endogenous
predictors consumed by the new model, not refreshing them with measured future
values. If the new ledger makes a history unnecessary, deleting that predictor
is simpler than constructing an elaborate redundant dynamics model.

## 5. Disturbances, uncertainty, tails and identification limits

### 5.1 What is exogenous, latent, or endogenous?

Exogenous disturbances: offered gateway receipt times and request attributes,
client/network ingress delay, external CPU contention, thermal/host regime
changes, backend failures, cancellations where externally caused, and update
transport delay. Admitted workload, internal queue, decode population, history
of accepted tokens and completion outcomes are endogenous. Natural output
length is latent conditional on prompt/generation settings and may share
request-level dependence with runtime. Endogenous service variability must not
be sampled independently of batch composition merely for convenience.

If clients retry on rejection, future offered load is policy-dependent. The
current open-loop replay assumption excludes that feedback; either explicitly
restrict the claim to no-retry replay or identify a retry model and include
it. Never claim an exogenous-arrival guarantee for a closed-loop retry workload.

### 5.2 Scenario construction

P: At each origin draw shared scenarios for all action sequences. Preserve joint
arrival burst, prompt/max mix, ingress, residual lifetime and service-regime
dependence. Candidate scenarios:

1. Past-only block resampling of received request attribute sequences, with
   explicit treatment of events not yet cost-estimated.
2. A small fitted arrival-intensity/regime model with burst/mix change scenarios.
3. Stress scenarios for long output, slow service, delayed observations/actions,
   or high-cost requests beyond the typical calibration region.

Do not resample only accepted traffic: action changes which offered requests
would be admitted. Rejected traffic must contribute its available prompt/max
attributes after its decision; before-handler missing traffic remains unknown.
Historical code and audits do not guarantee every capture has every offered
attribute, so version-specific completeness is a data gate.

For remaining natural decode length L given generated length n, use a
conditional survival law:

```text
P(L > n+r | L>n, prompt_class, requested_max, settings, regime).
```

Apply the requested-max bound only where enforced by the backend contract;
forced128/256 microbenchmark outputs are not a population law for natural EOS.
Right-censored requests are not short successful outputs. Fit with censoring
methods only when censoring assumptions are defensible; an observation timeout
that preferentially removes slow streams violates naive independent censoring.
Otherwise retain interval/bound or explicit unresolved mass. Failure and
cancellation are competing outcomes, not successful EOS.

### 5.3 Calibration is not a physical guarantee

Separate aleatoric scenario variability, parameter/representation uncertainty,
and observation staleness. Use run-level model ensembles or parameter sets plus
joint residual blocks, not independent Gaussian errors for Q, D, R and reward.
Report conditional coverage/error by action, load, transition direction, prompt
class, age and availability.

A possible P calibration procedure uses held *episodes*, scores the maximum
normalized optimistic reward/reservation error over the prespecified action and
horizon family, and sets a finite-sample quantile. Split-conformal marginal
coverage requires exchangeable calibration and test score units and a fixed
score/model before calibration. Independent exchangeable episodes, not overlapping
one-second origins, would be those units. This would give marginal coverage for
the chosen score, not arbitrary conditional coverage or a deterministic bound.
Calibration on factual actions alone cannot establish unobserved action coverage.

Closed-loop action selection changes the distribution. A marginal interval for
one fixed factual forecast is not simultaneous protection against searching
32 plans. Require randomized action-support evidence and either a calibrated
simultaneous score for a justified candidate family or treat the score as a
heuristic uncertainty penalty and leave safety qualification false. A small
number of runs cannot support credible extreme-tail (e.g. p99.9) statements.

P: Report 90/95% intervals where sample size permits, availability-inclusive
coverage, width, proper scores, optimistic-tail error and action-gap calibration.
Choose nominal levels and error normalization on training/development data before
validation; 95% is not derived from the TTFT SLO and is not an implicit allowed
5% SLO-miss rate.

### 5.4 OOD behavior

Hard OOD: source/runtime/model/tokenizer identity mismatch, clock mismatch,
sequence gaps, invalid budget acknowledgement, missing required ledger,
unqualified mode, or action outside supported set. Refuse MPC selection.

Soft OOD: state/mix outside training neighborhoods, drift in service residuals,
elevated preemptions, increasing interval width, or persistent one-sided
optimistic errors. Restrict to qualified actions, increase uncertainty or fall
back. Predeclare drift thresholds and hysteresis on development data; do not
silently refit to held outcomes. Refit/recalibration is a new version.

Unbounded arrival/service disturbances rule out deterministic TTFT guarantees.
TRAIN extrema are empirical support, not hard physical bounds; current envelope
flags correctly do not promote them to native probability coverage.

## 6. Objective and control optimization

### 6.1 Preserve the actual metric

For offered request i:

```text
g_i = 1{successful complete stream}
      * 1{client_first_content_i - scheduled_arrival_i <= 0.5 s}.
G_episode = sum_i g_i / T_offered.               [requests/second]
```

Rejected, failed and incomplete offered requests earn zero. No E2E deadline is
invented. Report rejection/failure, scheduled and receipt TTFT, ITL/TPOT, E2E,
drain and transition behavior separately, including prompt/output strata.
Gateway first-content timing is not interchangeable with client first content.
An HTTP/transport observation model or a qualified client feed is necessary
for predicting this metric rather than an engine-only surrogate.

P: For short-horizon optimization evaluate expected *counts* of successes among
all currently unresolved admitted requests and requests scheduled to arrive
during the planning horizon. Baseline comparisons use the same cohort and
scenario, including already admitted jobs. A request previously rejected or
already terminal is immutable and can be removed as an action-independent
constant. A first-token-hit stream is not immutable: future congestion/failure
can still affect eventual successful completion.

Ignoring existing cohorts causes externality errors. Example: without a new
admission an existing request first-tokens at .49 s; admitting it shifts both
first tokens to .51 s. A new-request-only reward reports zero gain, while the
true total SLO count change is minus one.

### 6.2 Candidate optimization and dimensional weights

Let `U=(u_0,...,u_(N-1))`, scenario m, and `G_m(U)` be the rollout success count,
including a declared continuation/terminal approximation. The base problem is:

```text
maximize_U  mean_m G_m(U)                         [successful requests]
subject to supported action sequences and qualified guardrails.
```

P: Compare risk-neutral optimization to a risk-sensitive variant:

```text
J(U) = mean_m G_m(U)
       - lambda * CVaR_alpha(L_m(U))
       - kappa * number_of_budget_changes(U).

L_m(U) = G_m(U_baseline) - G_m(U).
CVaR_alpha(L) = min_eta [eta + E(max(L-eta,0))/(1-alpha)].
```

L may be negative; CVaR measures the bad upper tail of loss relative to the
same-origin baseline continuation, not total failure count under some other
offered workload. Shared scenarios reduce comparison noise; they do not prove
the model's counterfactual validity.

Lambda is dimensionless; kappa is successful-request equivalents per change.
Do not combine cost units, seconds and request counts without explicit
conversion. Default lambda=kappa=0 for the simplest mean-reward candidate,
with an empirical switch-margin guard. Select risk/switch settings on a small
predeclared development Pareto grid of goodput, bad-tail loss, switches and
forecast error. If validation does not improve, retain the simpler candidate.
A larger rejection penalty is not automatically needed: rejected traffic
already earns zero. An added admission reward can conflict with the metric.

P: Only deviate from the fallback action when the lower calibrated/empirical
action-gap margin exceeds the development-defined minimum useful improvement
and estimated switching error. If calibration is not qualified, label this a
conservative selection heuristic, not a confidence guarantee. Near ties select
hold/fallback deterministically, never the action with the largest noisy mean.
Baseline policy value from a reset episode and baseline continuation from this
controller's current state are different quantities.

### 6.3 Hard, soft and terminal constraints

Hard implementable constraints:

- One authenticated writer, cost-aware mode, finite positive budgets, approved
  runtime and a predeclared action set. Initially study `{16,32}`; historical
  use establishes numerical familiarity, not current-host causal qualification.
- Exact per-request reservation accounting and nonnegative lifecycle state.
  Do not impose `R<=u_k` immediately after a cap reduction.
- Commands must meet freshness/deadline and acknowledgement/version rules;
  no speculative second outstanding update and no unsupported action sequence.
- A minimum hold time or maximum number of switches can be a hard controller
  policy; it is not a mathematical property of the gateway.

Soft/statistical guardrails:

- Predicted bad-tail loss, admitted-late counts and terminal burden. No feasible
  action may satisfy a tight TTFT target under an overload burst; surface
  infeasibility and use fallback rather than pretending latency constraints
  became hard.
- Any ITL/TPOT or failure noninferiority margin must be specified by the owner
  and frozen before policy trials. Report the full curve if no product threshold
  exists. A made-up E2E deadline is not a substitute.
- Optional conservative KV/context checks are relevant only when matching
  observer/model support exists. Reservation cost is not a hard KV allocation
  measurement. The native engine remains responsible for its own resource
  limits; do not claim a gateway hard KV safety certificate.

Terminal treatment is essential. Finite TTFT horizon does not bound completion
or time-to-reservation-release. After the last planned action, simulate the
fallback policy with continued arrival/service scenarios for a fixed declared
tail, then apply `V_fallback(b_tail)` to unresolved cohorts and congestion.
Include those tail arrivals' interference even if their own rewards are not
scored; no-later-arrivals drain is an optimistic sensitivity, not the nominal
serving continuation.

Train V only from completed development continuations under supported policies.
Track unresolved-tail probability and upper/lower reward bounds. If the
unresolved mass can reverse action ordering, abstain or lengthen the model
horizon in development; do not assume immediate successful drain. With K
unresolved relevant requests, their reward alone has interval [0,K], before
considering effects on additional cohorts. For differential action scoring,
bound each plan's uncertain reward and compare worst/best intervals rather than
assuming correlated tail errors cancel.

Terminal congestion proxy:

```text
W = sum(expected remaining prefill tokens * beta_p
        + expected remaining decode work * beta_d)     [seconds-equivalent]
```

can be reported/penalized with a dimensioned conversion, but is not physical
clearance time under continuous batching. Compare a learned continuation value
against a simple burden penalty and explicit longer continuation. Reject any
terminal approximation that manufactures gains by pushing releases/failures
just outside the horizon.

### 6.4 Feasibility and stability: limited claims

D: If individual reservations have a true positive lower bound c_min, caps are
bounded by U_max, the initial reservation is finite, and accounting is correct,
then `R(t) <= max(R(0),U_max)` and admitted request count is at most that bound
divided by c_min. Thus reject-on-arrival admission bounds *admitted population*
even under excess offered load. It does not bound pre-admission HTTP backlog,
guarantee finite service/stream completion, or make offered arrivals stable.

D: Under finite enforced output maxima, a positive lower service capacity,
bounded stalls and no new admissions, the admitted population eventually drains.
Those are assumptions, not facts proven by these captures. CPU/transport stalls,
failures, natural output tails and future arrivals invalidate an unconditional
drain guarantee. Lowering a cap cannot force immediate negative drift.

No conventional recursive-feasibility or Lyapunov theorem is currently
established. It would require bounded disturbances, a terminal invariant set,
admissible fallback and an adequately accurate transition. Always-feasible
actuator choice is weaker than feasible latency constraints. A practical
research controller can be stable in bounded occupancy yet have terrible
goodput; do not conflate the two.

## 7. Sampling, horizon, computational budget and fallback

P initial design, subject to development falsification:

| Quantity | Initial proposal | Reason and rejection test |
|---|---|---|
| Action/decision interval | 1 s | Existing transition labels support 1/2/5 s; avoid claiming ms-level reaction from scrape history. Compare .25/.5/1/2 s only in an explicitly bounded dev study if live sensors support them. Reject 1 s if transition latency destroys useful action opportunities. |
| Planned action horizon | 5 s, N=5 | Existing factual horizon is 5 s, not proof of optimality. Compare N=1,2,5 and a longer continuation using identical models, state and baselines. Reject MPC if N>1 adds no policy value. |
| Action set / plan search | `{16,32}`, all 32 five-step sequences | Small exact enumeration avoids nonlinear solver/local-optimum claims. Enforce support and dwell constraints; full search is an upper computational bound, not permission to execute all plans. |
| Scenario count | Start 16 common scenarios, compare with 64 on development | Matches an existing historical analysis scale but does not estimate tails reliably. Require action ranking stable against scenario seed/count and model ensemble. 32*16=512 rollouts per cycle; benchmark actual CPU cost before adopting it. |
| Proposed observation budget | 50 ms retrieval/validation; gateway sample age <=100 ms at planning start | Engineering targets, not measured capability. Age includes sample timing, not just response RTT. A full live ledger must meet these targets before use. |
| Proposed planning budget | 100 ms wall time, single bounded worker | Enumerate with hard elapsed cutoff, not just mean runtime. If exhaustive rollout cannot fit, use constants plus supported one-switch plans or return fallback. Measure p50/p95/p99/max and impact on the gateway. |
| Proposed actuation budget | 50 ms request/ack target; total cut-to-confirmed-apply <=250 ms | Targets, not guarantees. Compare clocks in the same qualified domain and reject plans that expire. The existing endpoint lacks server-side expiry; an in-flight timed-out update may still apply. |

The above age targets are for the proposed slow gateway controller. They do
not relax the envelope diagnostic's existing 5 ms age/posting thresholds.
Action latency uncertainty is part of scenarios; actual applications, including
late ones, are logged for evaluation. At 1 s sampling the controller cannot
retroactively save a request with an imminent .5 s deadline. It acts mainly
through future admission and workload carryover.

Fallback state machine:

1. Before actuation qualification, all plans are shadow-only; current operator
   policy remains unchanged.
2. After qualification, invalid observation, OOD, unsupported plan, uncertain
   action gap, solver timeout or infeasible soft targets selects the predeclared
   fixed-budget policy, subject to the same actuation rules.
3. Keep at most one update in flight. Timeout means *unknown application*, not
   proof of no change. Stop issuing changes and reconcile version/budget.
   The existing gateway has neither idempotency key nor compare-and-set;
   a single writer plus reconciliation limits but does not erase delayed
   application ambiguity.
4. If control transport fails, the last actually applied cap persists. State
   this degraded behavior plainly: the controller cannot guarantee a lower
   fallback cap when it cannot communicate. Automatic reset on loss would be
   a separate server-side lease/watchdog feature, not present now.
5. After restart or lost event history, rebuild and qualify the ledger before
   resuming; do not continue a stale belief across clock/engine identities.

Emergency lowering may reduce later admissions but does not cancel reservations
or repair active overload instantly. Human shutdown remains outside an MPC
performance claim.

## 8. Competing controllers and discriminating comparisons

| Family | What must be compared | Why it may win / where it fails |
|---|---|---|
| Fixed request concurrency | Proper protected-mode count caps, tuned on development | Simple and robust, but mixes request sizes; not numerically equivalent to cost caps. Same offered streams/model settings; mode differences must be disclosed. |
| Fixed weighted budget | Every supported fixed action and the development-selected best fixed baseline | Direct comparator using the same actuator. If its repeated-run value matches MPC, dynamic sophistication is unjustified. |
| Open-loop schedule | Predeclared historical A/B-style schedule, if included | Measures whether workload phase knowledge rather than state feedback explains gains. It is not adaptive MPC. |
| Pressure threshold / hysteresis | Lower cap on high awaiting-age/cost pressure; raise only below a second threshold, with dwell | Minimal implementable adaptive baseline. Thresholds must be developed without final labels. Can react late, but MPC must beat it rather than only a weak trailing nowcast. |
| AIMD / PI queue control | Adjust budget using gateway awaiting/oldest-age error, bounded integral and anti-windup | Cheap feedback handles drift; gateway Q is not engine queue and control gain is delayed/nonlinear. Compare tuned gains and saturations, not a straw man. |
| One-step model controller | Same belief, scenarios, uncertainty gate and actuator; N=1 | Isolates value of receding multi-step planning from new features/model quality. Most important ablation. |
| Linear MPC | Local coherent linearized flows, constrained positive state, supported region | Fast; possibly sufficient around a narrow operating region. Stepwise integer admission, SLO threshold and cap-reduction drain limit linear validity. Require local residual/action tests. |
| Nonlinear deterministic MPC | Mean lifecycle or native-event surrogate | Handles saturation and carryover, but certainty equivalence can misrank tails; nominal model optimum may exploit error. Test against the scenario version. |
| Robust/minimax MPC | Optimize over explicitly bounded service/arrival sets | Meaningful only with defensible bounded sets and uncertainty-dependent action authority. Unbounded bursts make deterministic latency targets impossible; broad boxes often collapse to fixed conservative cap. |
| Stochastic/scenario MPC | Joint workload/lifecycle scenarios plus calibrated action-gap uncertainty | Recommended research candidate because randomness and latent state are central. Needs more action support, simulation budget, and distributional validation than nominal MPC. |
| Learning MPC / bandit / RL | Offline supervised dynamics first; narrowly authorized randomized identification later | Adaptation might capture drift, but overlapping delayed success and irreversible early rejection complicate credit assignment. No unrestricted online exploration or learned safety claim. A contextual one-step bandit is a relevant simpler alternative if memory adds little. |
| Hierarchical control | Slow budget controller plus fixed native scheduler | Already a useful conceptual decomposition. A second dynamic prefill/decode controller changes the actuator and must be a separate study, not silently included in this MPC's gains. |

Queue control baseline example, with units explicit:

```text
e_k = target_awaiting_age_seconds - observed_oldest_age_seconds
u_(k+1) = project_supported(u_k + Kp*(e_k-e_(k-1)) + Ki*Delta*e_k).
```

Kp has cost/second and Ki cost/second^2 in this incremental formulation. Reset or
freeze the integral on stale data, saturation or unsupported action; use the
same update delay and dwell limits as MPC. A hysteresis threshold derived from
the .5 s client SLO must subtract estimated ingress/transport/actuation and
existing backlog uncertainty, not treat gateway age .5 s as a safe threshold.

## 9. Identification and experimental design

### 9.1 What existing data can establish

Use existing A/B and fixed captures for development-only comparison of coherent
state propagation, history closure, uncertainty, lifecycle labels and factual
1/2/5 s forecasts. Reconstruct only attributes available at origin, with
availability-time rather than event-time joins. Keep future application timing
as an outcome; simulate delay using a model known at origin.

Do not infer a cap intervention from an association between cap and goodput
when host, workload phase and state history differ. Nearest-state matching,
good factual MAE, a coverage-distance index, or a range check is not sufficient
identification. `service_forecasts.py:137-175` usefully refuses unsupported action
patterns and explicitly labels support indices descriptive; extend rather than
bypass that discipline.

A scalar counter regression with negative decode coefficient is not physical
service identification. Examine the design matrix rank/condition, coefficient
stability, residual-vs-context structure and out-of-run predictive performance.
Fitting nonnegative coefficients alone does not solve limited excitation.
Use real per-step chunk/context/graph/pending labels only if captured and
available for the relevant modeling purpose.

### 9.2 Proposed action-identification experiments, not authorized execution

Stage A: qualify the current shadow transport/state/timing path first under its
unchanged scope and gates. It answers a different question from policy value;
do not block a gateway-only predictor on an unnecessary engine sensor, but do
not claim an enhanced observer before it works.

Stage B: on an explicitly authorized host, independently qualify budget
application, strict boundary admission, reservation carryover, lifecycle stream,
and decision-time sensor availability under cost-aware HTTP workloads. Record
all offered requests and replay starts/delays, not just admitted engine work.

Stage C: randomized *supported* budget pulses with varied dwell and prehistory,
balanced across workload class and host-time blocks. Include up/down transitions,
initial R above new cap, short/long prompts, natural EOS variation, bursts and
near-tie states. Randomize at the decision using only allowed information,
record propensity and action selection rule, and allow service washout/reset
between independent episodes. Dwell must include plausible reservation-release
times; a pulse too short to change admission cannot identify steady action
effect. Do not randomize repeatedly on a 5 ms engine diagnostic just to create
action data.

Randomized sequential assignments identify conditional action effects in the
visited histories under consistency, known assignment probabilities/positivity,
correct temporal ordering and no unrecorded cross-episode interference. They
do not identify all state/action pairs outside support. Residual state can carry
across pulses within an episode; either model it or treat the whole episode as
the randomized unit. Counterfactual potential outcomes for two exact physical
states are not both observable; replication supports a distributional effect.

Stage D: freeze candidate model/policy, baseline tuning, action-support rules,
objective, horizon, calibration and tail procedure. Run policies as randomized
whole episodes with matched offered workload seeds and balanced order within
host-time blocks. Start each measured episode from a defined reset/lead-in and
verify drain, stable identity and anchors. Same trace seeds reduce workload
variance but do not make GPU stochastic trajectories identical.

Policy comparison estimates the causal value of complete policies without
requiring every model state to have paired action labels. Use separate pulse
data to test action ranking/model bias; policy trials are not reused to tune
the winning model. No experiment here is run or funded by this report.

### 9.3 Chronological split and freeze

Existing inspected results are development data forever; calling a later slice
"held out" now does not erase prior feature/segment inspection.

For newly authorized collection, preassign entire episodes in chronology:

```text
train -> calibration -> validation -> sealed final policy episodes
```

All fitting, feature-family and baseline tuning uses train/development with
inner whole-run splits; calibration chooses interval parameters only; validation
chooses among a small preregistered set once, then freezes. If validation is
used to select the candidate, it is not the final performance test. Final
collection is later, with episode IDs/seeds sealed before outcomes. Reserved
trace C stays entirely with the owner and is not needed for this round.

Purge a training row unless its full reward/terminal label was available before
the training cutoff. Purge overlapping outcome windows across splits; add an
embargo covering relevant history and outstanding carryover, or use separately
reset episodes to remove the ambiguity. Natural EOS can outlast an arbitrary
5 s embargo, so use actual resolution or explicit censoring, not a fixed
assumption that 5 s drains everything. Freeze scaling/imputation, action support,
scenario generator, model ensemble, uncertainty quantile and fallback together.

Chronological later episodes evaluate temporal transfer. If host transfer is
claimed, hold hosts out as groups and report their identity; one GPU host with
many origins cannot establish cross-host robustness. Multiple runs of one trace
support repeatability conditional on that trace, not a population over workloads.

### 9.4 Repeated-run inference and negative outcomes

Primary estimate: episode-level all-offered goodput difference between the
frozen candidate and each strong frozen baseline, paired within randomized
workload/host-time blocks. Report absolute req/s and relative differences with
uncertainty. All planned episodes, failures, interruptions, fallback fractions,
late applications and runtime overhead stay in the accounting. A collection
integrity failure can invalidate a block under predeclared rules; report it and
its reason rather than deleting an unfavorable episode.

Use independent episode/block units for t intervals or randomization inference
consistent with the actual assignment design. Cluster/block bootstrap only if
enough independent blocks exist; it does not manufacture independence from
overlapping windows. Student-t intervals assume approximately independent block
differences and suitable mean behavior; randomization inference needs the actual
randomization and a clearly stated null. Report sensitivity to block/time drift.
For several baselines/guardrails, use simultaneous intervals or a prespecified
testing hierarchy instead of selecting the most flattering comparison.

P: Predeclare minimum practically useful gain delta in req/s, based on serving
value and complexity/overhead, not an invented number in this report. Use pilot
paired-block variance sigma_d^2 to budget sample size approximately:

```text
n_blocks ~= ((z_(1-alpha_adjusted) + z_(power)) * sigma_d / delta)^2.
```

This is a planning approximation, not achieved power; include uncertainty in
pilot variance, finite-n and multiple-comparison inflation. Obtain resource/time
authorization before collecting. If the allowed budget cannot distinguish
delta, report inconclusive rather than lower delta retrospectively.

Predeclared rejection/stop criteria:

- Model never shows supported randomized action-response skill beyond a
  constant/action-agnostic baseline, or optimistic error is large enough to
  reverse planned action gaps.
- N=5 adds no practically useful independent value over N=1 with the same model,
  or either loses to tuned hysteresis/fixed cost cap.
- Policy benefit disappears when all offered failures, natural output,
  terminal continuation, measured action/observation delay or controller CPU
  interference are included.
- Tail/availability/support/timing guardrails fail; coverage succeeds only by
  producing unusably wide or mostly unavailable predictions.
- Estimated lower confidence bound fails the frozen useful-gain criterion:
  no superiority claim. If the upper bound also excludes delta, evidence
  argues against useful gain in this scope; if intervals straddle delta, the
  result is inconclusive, not proof of equality.
- Increasing feature/solver complexity on the same inspected captures adds
  no new discriminating evidence. Stop that iteration rather than mine final C.

Off-policy importance sampling is not a shortcut for the existing deterministic
schedules. It requires known nonzero sequential propensities for the target
policy histories; variance grows with horizon, and delayed unresolved reward
adds difficulty. Doubly robust evaluation still needs overlap and at least one
valid nuisance model under its assumptions. Simulator rollouts are model-based
predictions, not off-policy ground truth or measured regret.

## 10. Proposed module boundaries and immutable records

These are implementation specifications for later disjoint work, not new files
or product edits in Round 1. Keep the existing gateway and prefill diagnostic
acceptance paths unchanged until interfaces and authorization are frozen.

| Module | Inputs -> outputs | Required invariant |
|---|---|---|
| Causal observation adapter | Gateway samples/events and optional qualified engine observations -> `ObservationBatch` | Run/clock/engine identity, source event sequence, sampled/available times, missingness, loss/reset status. No future joins. |
| Ledger/belief estimator | Previous belief, observations, acknowledged actions -> `BeliefSnapshot` | Per-request reservation/phase, age/length distributions and history expiration memory; reconcile exact totals. Q/D are explicitly gateway phases. |
| Exogenous scenario generator | Belief plus past-only history, frozen model/version, seed -> joint scenarios | Offered not survivor-only workload; correlated burst/mix/service/length/delay; scenario identity shared across actions. |
| Coherent transition model | Belief, timed action plan, scenario -> lifecycle trajectory and terminal uncertainty | Strict admission, whole-reservation release, exactly-once terminal, workload/token constraints if supported; no frozen consumed endogenous covariates. |
| Cohort scorer/continuation | Trajectory, scored cohort set, unresolved-tail belief -> success counts/bounds and diagnostics | Client scheduled TTFT plus eventual complete success; existing cohort externalities; no duplicate reward. |
| Support and uncertainty gate | State/plan provenance, calibration artifact, rollout diagnostics -> eligible plans/reasons | Identity/range/pattern/support/OOD separated; no acceptance promotion from descriptive distance. |
| Bounded planner | Eligible plans, shared scenarios, budget -> selected first action or fallback | Deterministic tie rule, wall deadline, per-plan evidence; N=1 and fixed controllers use the same observer/model where appropriate. |
| Actuation adapter | Chosen budget plus expected state/version and expiry -> attempt/ack record | One writer, at most one outstanding update, application uncertainty explicit. Server expiry/CAS would require a separately reviewed gateway extension. |
| Episode evaluator | Immutable raw offered/client/gateway/engine/action logs and frozen protocol -> audited results | Independent metric reconstruction, full denominator, complete/invalid/unavailable/fallback separation; no rewriting raw evidence. |

Essential `BeliefSnapshot` fields: schema/version; model and source hashes;
clock/run/engine identity; observation cut and latest availability; applied
budget/version; ledger sequence and reconciliation result; per-phase/cohort
state; missingness/ages; uncertainty provenance; action-support status.

Essential `DecisionRecord`: observation/model/scenario/plan hashes; decision
start/end/deadline; candidate actions and exclusions; predicted mean/tails and
unresolved mass; baseline gap; chosen action/fallback reason; command send,
acknowledged version/applied time or unknown status; actual next-state
evaluation linkage. Exclude credentials from every artifact.

The planner should be pure and replayable, with transport outside it. A small
constant/one-switch plan set is preferable to a general nonlinear solver if it
meets the evidence need. No RL framework, distributed optimizer, scheduler fork
or online retuning infrastructure is warranted before the simpler candidate
passes the action and policy gates.

## 11. Targeted falsification tests

### 11.1 Checks actually executed in this round

An inline standard-library Python 3 script was executed locally; no backend,
network, dependencies, product tests, or paid compute were used. `python` was
absent; the same check ran successfully with `python3`. These are arithmetic
counterexample checks, not native/service-model validation.

| Counterexample / calculation | Executed result | Claim it rejects |
|---|---|---|
| R24, lower to C16, cost12 terminal then cost4 arrival | R stays24, then12; cost4 fits at16 | Reservation clipping or immediate cancellation on lower cap. |
| Equal `(Q,D,R)=(0,1,8)`, residual decode1 vs128 at10ms/step | Released cost after10ms: 8 vs0 | Q/D/R as a complete Markov transition state. |
| Five-bin activity buffers `[5,0,0,0,0]` and `[0,0,0,0,5]` | Equal total5; next totals0 vs5 with zero new activity | Updating a rolling sum without retaining expiration information. |
| TRAIN repeated observations 51.639,50.539,25.161,49.974ms, budget max(10ms,20%) | Required intersection lower41.3112ms, upper35.161ms; gap6.1502ms | Any single planned-case constant passes all four original budgets. Rounded published observations explain sub-microsecond differences. |
| Existing first token .49s becomes .51s after new admission; new first token .51s | New-only gain0, complete cohort gain-1 | Ignoring existing-cohort congestion externalities. |
| Ten first-token SLO hits, two later failed streams | Final positive rewards8, not10 | Treating first-token hits as complete-success goodput. |
| Two caps, five decisions, sixteen scenarios | 32 plans,512 rollouts | Any unmeasured assertion that this is computationally free or a tail-calibrated sample. |

### 11.2 Required CPU/model tests before shadow control

1. **Event/accounting parity:** admit exactly at the boundary, reject just over,
   admit after whole-cost terminal release, process two equal-time arrivals
   in recorded order, and include failed/cancelled/reentrant cleanup paths.
   Reconstruct captured Q/D/R without clipping to cap.
2. **History closure:** the equal-sum/different-expiration example above must
   diverge correctly; activity and outcome ring buffers expire independently.
   Same-interval admit/first/terminal events preserve phase and reward counts.
3. **Causal isolation:** alter all future arrivals, commits, application times,
   client outcomes and unavailable scrapes after origin; belief and scenarios
   already frozen at origin must not change. Explicit oracle diagnostics may
   change and must be labeled separately.
4. **Clock/staleness:** cross-boot smaller/equal/larger timestamps all refuse
   before age comparisons; delayed parsing, missing samples, counter resets,
   sequence gaps and reused IDs produce explicit unavailable/fallback states.
5. **Latent state:** identical Q/D/R but different residual lengths, pending
   readiness, prompt composition and ingress beliefs produce different
   transitions or appropriately wider uncertainty. Reject a model that
   confidently collapses these tested differences.
6. **Terminal/outcome validity:** first-token-hit then failure earns zero; late
   terminal labels cannot train earlier cutoffs; right-censored streams do not
   become successful short EOS. Existing cohort effects enter action scoring.
7. **Horizon gaming:** construct a plan that looks best at5s by deferring
   congestion/failure to5.1s. Explicit continuation must remove that spurious
   gain or mark ordering unresolved. Continue tail arrivals; do not use empty
   drain as the only terminal model.
8. **Support/selection:** a fitted cap range without the planned transition
   pattern cannot authorize it; action-family search widens required
   uncertainty. Include N=1, constants, ties, high uncertainty and no-eligible-
   action cases with deterministic fallback.
9. **Timeout ambiguity:** acknowledge after client timeout, duplicate/reordered
   reply, external writer and process restart. No next speculative write;
   reconcile actual state and log unknown application.
10. **Resource budget:** worst-load observation/rollout logs meet measured
    budgets or cut off to fallback, including GC/OS scheduling interference.
    No blocking prediction in an admission lock or EngineCore hot path.
11. **Metrics:** all-offered count remains fixed when requests reject/fail;
    client schedule versus receipt clock and complete-stream checks are
    distinct. Prompt/output-stratified results expose composition selection.

CPU synthetic cases can falsify bookkeeping and modeling logic. They cannot
establish real delay distributions, action ranking or policy benefit. Any
future implementation must use the repository's required native acceptance
harness as well as targeted tests; this documentation-only round does not
claim that harness was run.

### 11.3 Required native discriminating tests, authorization still absent

- Current shadow probe transport/timing and observer overhead under unchanged
  gates, retaining every unavailable/invalid prediction.
- Cost-aware mode actuator/state retrieval under load, with actual application
  delay, single-worker identity and reservations above lowered caps.
- Natural EOS/long-stream and burst stress, with causal ingress/client timing,
  to distinguish latent-length, service-regime and HTTP-model errors.
- Randomized supported cap pulses with sufficiently varied histories to test
  action-response calibration and distinguish monotone-pressure heuristics
  from genuine nonmyopic release planning.
- Randomized repeated frozen policies: fixed, hysteresis/PI, N=1 and N=5,
  balanced order, full offered window and drain accounting.

No passing mock or attractive synthetic win substitutes for these measurements.

## 12. Architecture decision, alternatives and unresolved hypotheses

**Decision proposed:** pursue a gateway-first, coherent lifecycle belief model
and a small discrete scenario planner in shadow mode only after interface
review. Preserve the native fixed-membership envelope as a separate optional
sensor/model qualification experiment. Use fixed/hysteresis/one-step controls
as genuine first-class candidates, not stepping stones that MPC must beat by
definition. Do not require a rich engine observer if a gateway-only model can
win; do not invent that observer if gateway-only uncertainty is inadequate.

Prefer the coherent lifecycle/hazard candidate initially because it can model
natural EOS and delayed successful outcomes while enforcing exact gateway
admission. Prefer the detailed native event simulator instead if action-pulse
errors show reproducible dependence on chunk/async/pending state that the
gateway model cannot represent and a fresh observer is actually available.
Retain a local linear model if it is equally accurate and faster. Reject all
three in favor of fixed/hysteresis if policy evidence does not justify them.

Unresolved, independently rejectable hypotheses:

| Hypothesis | Discriminating evidence |
|---|---|
| Gateway-observable ages/composition suffice for useful cap ranking without exact EngineCore state. | Gateway-only versus enhanced observer on matched causal origins, then independent policy comparison; include availability and overhead. |
| Delayed cohort/history closure improves deployable multi-step predictions. | Origin-only coherent augmentation versus frozen-history and no-history heads on grouped development holdouts; no future refresh. |
| Natural decode residuals materially determine cap-reduction benefit. | Fixed/max-length versus calibrated survival model, stratified release/action-gap tests and natural-EOS policy trials. |
| A five-second horizon captures useful nonmyopic reservation carryover. | Same observer/model/gates with N=1/2/5 and terminal alternatives; randomized policy value, not just lower multi-step state MAE. |
| Scenario risk control improves outcomes rather than just suppressing admissions. | Risk-neutral versus risk-sensitive Pareto comparison with all-offered goodput, rejection and admitted-late tails; freeze useful-gain/guardrail criteria. |
| Native pending state resolves the failed constant's ambiguity. | Condition on causally available pending anchors/readiness and test held dispersion; if unresolved, retain mixture uncertainty and refuse deterministic claims. |
| Supported actions can outperform best fixed policy on fresh workloads. | Full randomized repeated policy experiment with uncertainty and all strong frozen comparators; existing A/B descriptive differences do not answer it. |

**Round 1 barrier:** report complete; scientific and implementation work remain
unqualified. Model/HTTP/adaptive-cap/MPC/performance acceptance remains false.
No new native run, raw historical archive audit, final trace C access, peer
proposal review, product edit, commit, push, merge, or resource request was
performed for this research round. Await Round 2 release with this context
retained.
