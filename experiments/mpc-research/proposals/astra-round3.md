# Astra Round 3: corrected gateway-control design and evidence adjudication

Assignment: `pr-mpc-astra`. Author: GPT-6 Astra (`gpt-6-astra`).
Source freeze: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Round 1 and Round 2 are preserved unchanged.

## 1. Decision, readiness and evidence boundary

**Recommend default-off CPU/shadow engineering of one shared gateway ledger,
a backlog-feedback baseline, a compact lifecycle model, and a small discrete
planner. Do not recommend or authorize live MPC actuation.** Whether feedback,
myopic model control or multistep planning wins remains an empirical question.
The new factual backlog evidence changes the order of experiments, not that
conclusion.

I read all three complete peer Round 2 reports, the Mayor's
`experiments/mpc-research/mayor_counterexamples.py`, and
`results/mpc-research-cpu/mayor-counterexamples.json` in the supplied Mayor
worktree. I reproduced its eight checks by importing the script with
`python3 -B` and comparing JSON-normalized output to the stored artifact.
Additional executed arithmetic checks below cover whole-cohort tail risk and
server-side expiry/CAS. None is a native handler, CPU performance, GPU, HTTP,
causal policy or final-trace experiment.

The strongest current numerical predictor evidence remains the Mayor-verified
development reanalysis: W_tok plus own prompt has held-run AUC
0.959016-0.994851 and TTFT MAE0.033374-0.093152s across12 captures. My Round 2
recomputed summaries from that artifact; I did not re-extract its archives.
The measured target was client scheduled-arrival TTFT, not gateway receipt
TTFT. These inspected captures are permanently development evidence.

Readiness is split explicitly:

| Gate | Round 3 status |
|---|---|
| Coherent specification for a later default-off CPU implementation | Ready for owner review and scoped implementation assignment |
| Implemented observer/model/planner or timing-qualified worker | Not delivered by these research reports |
| Qualified live marked observation/label feed | Not established |
| Safe stale-command rejection on current `/admin/budget` | Absent; new server contract required |
| Current-host action support, HTTP/GPU acceptance, policy improvement | Not established |
| Permission for live writes, paid/GPU collection or final trace C | Not granted by this report |

My earlier20ms/250ms configuration was an **unmeasured target**, not a runtime
measurement. It is superseded below, not claimed to have passed. I made no
native-acceptance or benchmark claim to retract. Any reading of the reports
as certifying such performance is explicitly withdrawn. Gemini's unexecuted
runtime/native-acceptance statements are not adopted into this design.

## 2. Disposition of substantive critiques

References below use the named peer Round 2 sections/IDs, not a claim that the
peers endorse this revision. "Accept" changes the design; "partly accept"
distinguishes a real engineering concern from an unsupported theorem.

### Critiques directed at Astra

| Critique | Disposition and evidence | Round 3 change |
|---|---|---|
| Opus A-1; Gemini2.2/Defect2.3; GPT A5:1280 rollouts/20ms | **Accept the missing benchmark and excessive initial scope; reject a universal infeasibility proof.** Opus's57.1ms toy timing measures one implementation/model/CPU. Vectorization, prefix reuse and different representations can change it; Gemini's100-300ms estimates are not measured limits. | Reduce initial plan count/cadence ambition; benchmark the actual retained model, tails and logging. No performance acceptance. |
| Opus A-2:250ms switch-time distinctions may be unidentifiable | **Accept as a testable concern, not proved equivalence.** Empirical drain extrema and Monte Carlo errors do not prove all such plans identical. | Start1s; keep finer cadence as a later bounded experiment only if current-state information and decision sensitivity justify it. |
| Opus A-3; Gemini4.2:state is excessive given W_tok accuracy | **Partly accept.** Prioritize W_tok predictive features. Reject the claim that high AUC eliminates reservation identity/cohort bookkeeping or proves all additional state valueless. | Separate the exact accounting ledger from the small learned feature vector; no engine feed or expansive history model by default. |
| Opus A-4/G-2:linear old-cohort correction replaces full continuation and avoids doubling | **Reject asserted equivalence; accept it as a challenger.** b_after is an outcome-window regression with no identified bias sign. Its threshold flips are not causal upper bounds. A shared baseline rollout can be cached rather than duplicated per plan. | Score all cohorts in one rollout; compare a cheap correction only as development approximation after causal tests. |
| GPT A1:new-only CVaR omits old-cohort risk | **Accept.** Executed ten-scenario example reverses choice when old loss is included: new-only A/B1/4 versus full-cohort11/4. | Default risk-neutral; any risk-sensitive challenger uses the same complete action-sensitive reward scope in mean and tail. |
| GPT A2:release-pending state inferred from adjacent source operations | **Accept the narrower source correction.** Inspected `finish_request` and cleanup contain no suspension after cost mutation before `telemetry.terminal` in the ordinary cost-aware same-loop path. | Do not add a release-pending state for that supposed asyncio interleaving. Keep sequence/reconciliation handling for transported events and backend-work divergence separately. |
| Opus A-5/X8; GPT A4:one fresh GET is not a fence | **Accept.** Mayor's delayed-write example and current endpoint prove old requests can apply later. | Define pending until definitive resolution, not client timeout. Require server expiry plus CAS/operation identity before authorizing adaptive live writes. |
| Gemini2.2/Defect2.2:suspension is unacceptable during saturation | **Reject continuing with assumed-unapplied state as a safe repair.** An active probe may improve information but cannot cancel a queued write. A controller cannot guarantee communication while the serving loop is stalled. | Monitoring continues; new writes stop during ambiguity. Admit that degraded operation may persist indefinitely with the current API. |
| GPT A3:maximum candidate counterfactual score is unobservable | **Accept.** One executed history does not reveal all candidate outcomes; Mayor's two-world example proves factual equality with opposite action effects. | Calibrate measurable frozen-policy or randomized-intervention targets only; no fabricated per-origin maximum score. |
| Opus A-6:CVaR over32 scenarios is unstable | **Accept the need to validate it; reject a quantified variance conclusion without experiments.** Only about3.2 draws determine an empirical90% tail. | Remove CVaR from initial selection. Log stress/tail diagnostics and test seed/count stability before adding a risk term. |
| Opus A-7/G-3/G-4:per-decision confidence gate makes MPC inert | **Partly accept.** It can abstain frequently, but historical average cap32 bias is neither an action-gap error bound nor proof of zero switching. Unresolved [0,K] bounds can overlap without proving all learned bounds will. | No unsupported significance gate per decision. Shadow selection is model-based and explicitly unqualified; actual gain is tested at policy level. Safety/support rejection is not relaxed. |
| Gemini2.2/Defect2.1:299 episodes is an impossible conformal requirement | **Reject.**299 is a valid illustration for a1%/95% zero-violation binomial bound on independent units, not a requirement to implement CPU code or obtain any conformal interval. Cost does not make correlated windows independent. | State the requested probability claim and its sample requirements; if unaffordable, weaken/omit the claim rather than change the unit silently. |
| Opus A-8/G-8:no numerical power plan | **Partly accept.**49 pairs is a useful sensitivity calculation for assumed independent paired SD1 and delta.4, not achieved power from correlated development windows or three sessions. | Include a numerical planning illustration with assumptions; leave authorization/sample size contingent on the actual estimand, pilot variance and approved resource ceiling. |
| Opus A-9:occupancy/conformal/binomial arithmetic correct | **Retain.** Mayor source and executed arithmetic agree. | Keep the conditional occupancy proof and exact statistical assumptions; do not upgrade them to latency safety. |

### Other critiques that affect the shared design

| Issue | Evidence-based disposition |
|---|---|
| Opus empirical7.4req/s ceiling and mean-release700token bound | Retracted by Opus and rejected by independent arithmetic. Retain observed rates as descriptions, actual-release algebra as conditional accounting, and no policy-uniform physical ceiling. |
| Opus hard terminal inequality plus always-feasible hold | Reject the general feasibility claim. No-release examples suffice; a residence median alone does **not** prove how much cost remains. Under caps<=fallback the proposed terminal inequality may instead be vacuous. |
| Gemini17.88% deadband incidence | Reject as measured incidence. It is a Poisson probability of at least two **offers** under assumed rate/latency. Joint marks, acceptance under old cap, release events, state and latency determine deadband incidence. The constructed race is valid; its claimed frequency and goodput harm are not established. |
| Gemini6,000 writes imply massive contention/disk writes |6,000 updates is arithmetic at20Hz for300s. It does not measure contention; telemetry emission and batched flush are not one synchronous disk write per update. Measure overhead; do not infer universal overload from count alone. |
| Opus zero failures among13,962 correlated first-token hits | Keep as peer-reported development description, not an iid rule-of-three certificate. It motivates a simple failure model, not deleting eventual-success semantics or claiming a population upper bound. |
| Opus b_after upward bias and "old effects bounded by.17req/s" | Reject the directional/causal bound. Outcome-dependent windows, admission selection and shared capacity can bias associations in different directions. Preserve diagnostic associations without causal calibration. |
| Opus reservation-to-token bound as deterministic KV substitute | Reject native safety inference. Bounds on estimated/requested tokens need qualified token estimation, backend ownership, allocation/block overhead, prefix/configuration and orphan-work assumptions before bounding actual KV occupancy. |
| Gemini omitted/excluded KV but imposed.001 chance constraint | Remove it. A latent resource bound is possible only with a valid identified conservative model; none is supplied here. Native engine allocation safety remains separate. |
| Gemini drain `min(tokens/step,tokens/s)` | Reject; cost release occurs at terminal identities, not a token processing rate. Adopt coherent lifecycle events. |
| Gemini qualified/safe C24 and C16/C32 labels | C24 has no established evidence here. C16/C32 have historical development exposure, not blanket current-host qualification or latency safety. No selected fallback is called "proven safe." |
| Gemini terminal triple-counting critique of GPT | Accept a **definition risk**, reject automatic triple-counting as a proven error. Explicit tail, value of only unresolved cohorts, and an optional alternative burden proxy need not overlap. Specify the partition below and do not sum alternatives. |
| Gemini GPT "demands retries" | Reject that characterization. GPT explicitly restricted the benchmark to no-retry replay or required a new model only for a broader claim. We retain no retries in scope without implementing a retry model. |
| Gemini empirical quantile/.88 coverage/Mahalanobis claims | Reject the coverage/geometric equivalences. Use corrected ranks and declared score units; an ellipsoid is not a convex hull or action-support certificate. |
| Gemini runtime/native289 acceptance claims, and Round2<15ms claim | Unsupported; not retained. A toy script timing is not the specified model benchmark.289 selected local passes with a skip are not full native acceptance. |
| Shared terminal rewards, old-cohort effects and delayed success | Accept the accounting correction, independent of whether their observed effect is large. Mayor examples demonstrate preference reversals. |
| Shared same-model myopic comparison | Accept GPT's refinement: distinguish control freedom from scoring horizon; changing both does not isolate planning value. |
| Shared sequential testing/carryover critiques | Accept. ARL from a fitted CUSUM is not automatically an anytime error guarantee under arbitrary drift; randomization must match the episode/history estimand. |

## 3. Smallest coherent CPU implementation target

This is a proposed interface freeze, not new code. Implement only the common
ledger, a compact gateway-observable model, planner/scorer and disabled
transport adapter. Full engine simulation, rich particles, online RLS, RL,
KV chance constraints and scheduler changes are not initial requirements.

### 3.1 Plant, state and units

Single gateway worker and identified backend; cost-aware chat admission;
fixed tokenizer/cost semantics; no retry feedback; fixed scheduler settings.
The protected forced-token prefill bridge remains a separate experiment.

For each admitted, not-yet-terminal request store:

```text
id                         unique within gateway epoch
c                          original reserved cost units
p_hat, M                   estimated prompt tokens, requested output maximum
received_at, admitted_at   gateway monotonic seconds
first_gateway_at           optional gateway monotonic seconds
phase                      awaiting_first | streaming
gateway_terminal           outcome/time when received; then leave active ledger
scheduled/ingress_belief    optional qualified schedule or explicit delay model
client_outcome             asynchronous label, with available_at and clock identity
```

The last field may remain in a separate label/cohort table after reservation
release; gateway cleanup is not necessarily the instant a client label arrives.
Store terminal/failure/censoring distinctly. Never infer exact emitted token
count from SSE chunk count.

Exact gateway-derived summaries:

```text
R = sum_active c_i                              [cost]
Q = count awaiting_first                       [requests]
D = count streaming                            [requests]
W_tok = sum_awaiting_first p_hat_i              [estimated tokens]
phase_age_i = now - relevant observed phase time [seconds]
```

W_tok is original admitted prompt size awaiting gateway first content, **not**
remaining engine prefill work. Preserve mark/identity information for exact
reservation release even if only W_tok/D/age are learned features. That is
bookkeeping, not evidence that a large predictive state has won.

Control state additionally contains actual/possible applied budget B, version,
controller lease/epoch if supported, pending operation and latest coherent
ledger sequence. Exogenous scenarios contain future offered times/marks,
ingress/client delay, effective service regime and latent lifetime/outcomes.
No actual future trace is an input.

The cost convention remains

    c_i = max(c_min, c0 + c_p*p_hat_i + c_o*M_i),

with coefficients bound into an immutable cost identity. It is not GPU time
or physical memory. Do not change that convention simultaneously with the
controller comparison. Enforced request/context bounds and tokenizer-fallback
status are part of support, not guessed physical guarantees.

### 3.2 Observation contract and exact dynamics

Input batch:

```text
ObservationBatch(
  schema, source_commit, runtime_hash, cost_hash, tokenizer_hash,
  gateway_epoch, engine_epoch, clock_domain,
  source_seq_start, source_seq_end, sampled_at, received_at, validated_at,
  completeness, events, coherent_summary, applied_budget, budget_version
)
```

Only data with `validated_at<=decision_at` are usable. Same-domain source age,
receipt age and validation age are recorded separately. Clock domains are
checked before comparing monotonic timestamps. A backend clock is not assumed
comparable to the gateway merely because both report seconds.

Current `/metrics` lacks the complete marked ledger and domain/sequence
contract. Offline reconstructed fields do not prove timely live access.
Build the adapter from fixtures first; live feed work is separate and must
measure loss, reordering, coherent sampling and observer interference.

At the actual admission serialization point:

    accept_i = 1{R^- + c_i <= B_applied};
    R^+ = R^- + accept_i*c_i.

At terminal release of accepted request i:

    R^+ = R^- - c_i.

The sampled conservation identities are

    Q' = Q + A - F - E_Q
    D' = D + F - E_D
    R' = R + sum_admitted c_i - sum_terminal c_i.

F is observed first content; E_Q/E_D are terminal exits by gateway phase.
Generate all these flows from the same request events, with recorded or
declared tie order. No independent cost-out head or large projection repair.
Reducing B leaves R unchanged; existing reservations cannot be evicted.

Correcting my Round 1: ordinary same-loop cleanup does not yield between cost
mutation and terminal bookkeeping, so a fictional release-pending state is
not required for that snapshot. A transported event feed can still have
incomplete prefixes; reconcile by sequence/coherent summary, not by deleting
work. Physical backend work persisting after gateway release is a **different**
uncertainty. Qualify cancellation propagation or track residual/orphan-work
uncertainty and fail scope after unexplained release/backend disagreement.

If a rolling feature is retained, retain expiration memory:
`h'=h+new-expired`. Initially only the causal offered-arrival/mark history
needed for scenarios is required; additional activity/outcome history must
earn inclusion. Never append a predicted first hit as an observed successful
completion.

### 3.3 Minimal model and the backlog challenger

Two models share one ledger and scorer:

**M0, factual/backlog challenger.** Freeze a small calibrated development
relationship such as

    predicted scheduled TTFT =
      theta0 + thetaW*(W_tok_at_admission+p_hat)/1000 + thetaD*D.

ThetaW is seconds/kTok; thetaD is seconds/request. This uses the strongest
available factual evidence. Its threshold behavior and live scheduled/client
label adapter still need qualification. AUC alone is not calibration. M0
supports backlog-feedback thresholds and a prediction benchmark; by itself
it is not a complete action-conditioned lifetime simulator.

**M1, smallest coherent rollout candidate.** A gateway-phase semi-Markov
kernel produces first-content and terminal events, conditioned on marked
phase ages, W_tok, D and a shared effective-service scenario. For example,

    h_e(a,x) = h0_e(age_bin) * exp(beta_e' scaled_features(x)),

where e is first content, successful terminal, failed terminal or cancellation
when valid for the current phase; h has units1/s. A request awaiting first
content can fail/cancel but cannot successfully finish a required complete
text stream without its modeled first event. A streaming request can succeed
or fail. For constant hazards over a small event interval d,

    P(no event) = exp(-sum_e h_e*d),
    P(type=e | an event) = h_e/sum_e h_e.

Age-bin boundaries and arrivals are events at which hazards are reevaluated.
Within a sampled interval allow successive first/terminal events in causal
order. First/terminal times cannot precede admission. Costs release only at
terminal, including failure/cancellation. A shared regime variable preserves
some correlated stalls; independent request hazards are conditional modeling
assumptions, not a complete physical noise description.

This equation is an implementable candidate family, **not a fitted or
identified model in this report**. Begin with frozen small feature groups and
whole-run development fitting; do not introduce online RLS or engine progress
feeds to fill missing evidence. Natural EOS, failure and transport changes
outside support return unavailable/explicit stress scenarios. Forced128 data
do not identify a general survival law. If emitted length is later qualified,
use a conditional residual distribution respecting backend-enforced M, not
future final output labels.

Old-cohort effects are generated by the evolving common W_tok/D/service
state, not inserted as a causally certified b_after correction. Compare
frozen-at-admission M0 plus a proposed cheap correction to M1, but label any
correction coefficient's origin and intervention support. Lack of sufficient
data may leave both unqualified; CPU fixture execution is still possible.

## 4. Exact reward partition and terminal treatment

The measured episode metric is unchanged:

    G_episode = sum_{i scheduled in offered episode}
      1{client first_content_i - scheduled_i <= .5s
        AND eventual complete success_i} / offered_duration.

Failures, rejections and incomplete outcomes earn no successful reward; a
censored unknown is reported as such, not silently treated as a modeled
success. An evaluation drain does not extend the offered denominator.

### 4.1 Finite planner estimand

Separate **control horizon** H_c=3s from a proposed **scoring window**
H_s=5s. The latter includes a2s opportunity tail so reservations near H_c
are charged for some later admission opportunities. Both are development
targets; neither is a physical drain bound.

At origin t partition scored requests into disjoint sets:

- O: already admitted requests whose final scored reward is not yet known.
  A first-token-miss request has zero possible hit reward but still contributes
  resource interference until terminal.
- N: future offered cohorts with scheduled time in[t,t+H_c).
- F: future offered cohorts in[t+H_c,t+H_s).

Previously rejected or fully resolved requests are action-independent
constants and omitted. All candidate plans use the same exogenous offered
identities/marks and initially the same O. Which future requests are accepted
is endogenous. After H_c every plan uses the same declared fallback **policy**
from its own resulting state, not a reset baseline state.

For scenario m,

    V_m(U) = sum_{i in O union N union F} g_i(U then fallback, scenario_m).

Unions here are disjoint and each identity earns at most one unit. This
incorporates old outcomes, new offers and finite future opportunity cost
without adding a separate arbitrary backlog penalty. Compare to fallback
continuation from the same origin; a reset fixed-cap episode is a different
policy-value comparison.

### 4.2 Event horizon is not completion horizon

Simulate through t+H_s and, within an explicit CPU event/time budget, extend
to a declared completion cutoff t+T, initially10s as a **computational**
target. Offers after H_s continue as interference scenarios but receive no
reward in this finite planner estimand. Artificially emptying future arrivals
would change terminal success and release predictions.

Partition K=O union N union F at the simulation cutoff into:

    K_resolved and K_unresolved.
    V_m = sum_{K_resolved} g_i + V_unresolved(K_unresolved, state_at_cutoff).

V_unresolved concerns only the unresolved identities in K. It cannot include
already resolved K or add another unrestricted sum of future-cohort rewards.
Initial implementation returns a modeled expectation where supported **and**
an explicit residual reward interval. For known first-token misses the
interval is[0,0]; for remaining potentially successful hits it is[0,1].
If continuation is unsupported, return a partial forecast/unknown flag,
not assumed successful completion.

A learned continuation head may later replace further event rollout only
after calibrated completed continuations exist. A seconds-equivalent burden
proxy is an **alternative development model**, not an extra term silently
stacked on the same counted interference. Do not add an infinite
undiscounted future sum; it generally diverges under continuing offered load.
Residual opportunity beyond H_s is a finite-horizon approximation, not
solved by the label "terminal value." Compare H_s/T expansions for action
stability and retain the unresolved limitation if rankings change.

### 4.3 Selection and risk

Initial selection maximizes mean modeled V across common scenarios. Use
deterministic tie preference for hold/fallback; a practical switching margin
can be frozen from development, stated in request equivalents. There is no
claimed statistically certified per-origin gain threshold.

An optional later risk-sensitive comparison defines

    L_m(U) = V_m(U_fallback) - V_m(U)
    J(U) = mean_m V_m(U) - lambda*CVaR_beta(L_m(U)).

The same O/N/F scope and shared randomness enter both mean and CVaR. L can be
negative. This corrects my new-cohort-only tail objective. Default lambda=0;
no CVaR-driven action until scenario-size/seed/model sensitivity justifies it.
Uncertainty widths, unsupported tails and forecast availability remain logged
even if the shadow recommendation uses a model mean. Removing an unqualified
confidence gate is not permission to call the mean safe or accurate.

## 5. Plans, baselines, budgets and constraints

### 5.1 Initial executable configuration targets

| Item | Proposed initial setting | Meaning/qualification |
|---|---|---|
| Recommendation cadence |1s | Supersedes initial250ms ambition; not a guarantee that stale data become useful |
| Future control slots |3x1s | H_c=3s; adopts the smaller Mayor reference as the initial engineering configuration |
| Candidate actions |{16,32} cost units | Historical development exposure only; no current-host safety qualification |
| Plan family |All2^3=8 sequences before support filtering | At most two internal switches; constants and same-model myopic included |
| Prior applied action -> first plan action |Explicitly included in change count/dwell rules | Additional to the plan's internal switches; declare minimum dwell before collection |
| Scenario count |16 common scenarios |128 plan rollouts before reuse; not an extreme-tail calibration sample |
| Scoring/initial maximum continuation |H_s=5s; T=10s | Finite computational horizons, not guaranteed lifetime bounds |
| Source age at planning start |<=100ms target | Requires a qualified feed; existing1s log flush does not satisfy this |
| Planning target / watchdog cutoff |50ms target /100ms cutoff | Neither measured; benchmark actual kernel, tails, maximal supported state and logging |
| Command acknowledgement target |<=50ms | Diagnostic target, not proof of no later application |
| Maximum observation cut-to-application |<=250ms candidate | Must be enforced by qualified server-domain expiry, not client stopwatch alone |

If a planned transition pattern lacks support, filter it out rather than claiming
the numerical alphabet provides transition support. Include hold only if
the actual applied state and configuration are supported. Unknown applied
state blocks live selection. During CPU work a fake transport is mandatory.

No timing claim follows from128<1280 rollouts. Use shared prefix/baseline
computation where it preserves identical simulation semantics; partial work
must never compare a favorable subset of one candidate's scenarios with all
of another's. At cutoff either use a matched complete common subset under a
prespecified rule or return no new recommendation. Stale worker results are
discarded by decision ID and expiry.

### 5.2 Strong baselines with the same information

All comparisons use the same observed ledger, marked features, supported
actions, actual transport delay and accounting:

1. Development-selected best fixed cost cap; protected count caps, if
   studied, are a separately labeled plant mode.
2. Backlog pressure/hysteresis using W_tok plus available own-prompt
   distribution, D and a qualified scheduled/client-delay margin. Thresholds
   and dwell are developed then frozen, not tuned on final labels.
3. Optional absolute-slack feedback via current scalar budget, **only as a
   separately supported action/cadence regime**. Model actual within-tick
   releases/delay; a relative update does not remove subsequent releases.
   A true per-request gate is a new actuator, not this baseline.
4. Same-model myopic control: choose the first slot's action from{16,32},
   then use fallback for the remaining control slots. Score the exact same
   O/N/F cohorts and continuation as MPC.
5. Same-model constants and full three-slot planner. Separately vary
   scoring/control horizon in a bounded ablation, not conflate them with
   different state/model information.

This prevents crediting MPC for a new W_tok feature that a cheap feedback
policy could use equally well. Stationary arrivals do not prove a myopic
Bellman value; unknown future demand does not prove long-horizon value.

### 5.3 Constraint and boundedness claims

Hard local contracts: valid identity/clock/sequence; finite positive
supported budgets; lifecycle conservation; explicit dwell; bounded work;
disabled-by-default transport; no unsupported scenario made factual.
Hard future transport contracts are specified in section6.

No hard `R<=new_cap` after reduction; no hard terminal empty-queue or
`R_H<=fallback+mean_cost`; no KV chance bound from unobserved gauges.
Latency, success risk and terminal burden are model diagnostics or
predeclared empirical guardrails, not deterministic constraints.

If positive admitted costs satisfy c_i>=c_min, budgets never exceed B_max,
initial reservation is finite, and gateway accounting/serialization is
correct, induction gives

    R(t)<=max(R(0),B_max),
    admitted_nonterminal_count<=max(R(0),B_max)/c_min.

This bounds gateway reservations, not pre-handler HTTP backlog, backend
orphan work, latency or completion time. If service can stall unboundedly,
there is no deterministic drain/SLO guarantee. Holding a legal budget
proves only input admissibility; the optimizer may reject every modeled
plan for support or quality reasons and must return that status.

For nonnegative configured cost coefficients one can bound **estimated**
token totals algebraically from cost. Actual physical KV safety additionally
requires true-token, allocator/capacity, configuration, sharing and orphan-work
qualification. No such certificate is claimed here.

## 6. Fallback and transport: the required new server contract

### 6.1 Current endpoint limitations and default behavior

`gateway.py:588-630` overwrites an absolute budget and increments a version
under its lock. It has no server expiry, CAS or idempotent operation identity.
At least one delayed request can survive client timeout. A later GET of
the old state cannot prove it has terminated.

During this engineering phase the transport adapter has **no live write
authority**. It emits shadow decisions and fake acknowledgements for tests.
If a later explicitly authorized experiment uses the old endpoint, fallback
is best-effort and action ambiguity is a real failure mode; it must not
inherit the stronger guarantees below.

Required client states:

```text
DISABLED -> SHADOW
authorized + qualified server -> READY
READY -> SENT(operation_id)
SENT -> CONFIRMED or REJECTED or UNCERTAIN
UNCERTAIN -> READY only on definitive operation resolution plus state reconcile
identity loss -> REBUILD / DISABLED
```

Timeout changes SENT to UNCERTAIN, not READY. Monitoring/read-only probes can
continue, but no assumption that the requested action was unapplied is made.
Without a bound on old-command lifetime, finite recovery may be impossible.
Last **actually applied** budget persists, not necessarily last acknowledged
budget. No communications algorithm can promise an emergency reduction while
the server cannot process messages.

### 6.2 Proposed guarded request and atomic processing

The following is a separately reviewed product/API change, **not implemented
or authorized by Round 3**:

```text
BudgetCommand(
  schema, operation_id, payload_hash,
  gateway_epoch, clock_domain, controller_lease_epoch,
  expected_budget_version, not_after_server_monotonic,
  observation_cut, model_hash, decision_id, budget
)
```

Authentication precedes any write. Parse and validate finite values, identity,
authorized budget policy and lease ownership. Inside the same state lock,
immediately before mutation:

1. Match gateway epoch and server clock domain. No cross-boot TTL reuse.
2. If this operation ID was applied already, return its original recorded
   result for the same payload, **without reapplying**. A conflicting payload
   under the same ID is rejected. Retained result/query semantics must remain
   unambiguous through the allowed retry window.
3. Require the current writer lease/fencing epoch and expected budget version.
4. Read server monotonic time at this lock boundary and require
   `now < not_after`; check after body/queue/lock delays, not only at receipt.
5. Apply the budget, increment version and atomically record operation result,
   applied time, previous/current budget and observed reservation.

Lease/epoch and result retention must either persist appropriately or be
invalidated by a new gateway epoch on restart. A bounded result table cannot
silently evict a still-retryable operation and permit reapplication; after
the operation's expiry it can reject a new attempt even if the old result
record was retired.

The acknowledgement/query response carries operation status, operation ID,
gateway/domain/lease epoch, budget/version, server receive/apply timestamps
and the applied expiry/expected-version data. A mismatch is not accepted as
a successful fallback. Read-only operation status is more useful than a
generic metrics snapshot for resolving a timed-out command.

**CAS and expiry solve different races.** CAS rejects an old expected-version
write after a newer operation has changed the version. Without a newer write,
the version may still match indefinitely; expiry is needed to reject an old
but otherwise valid command. Expiry alone does not fence two valid unexpired
commands racing on the same state; CAS/lease ordering does. Idempotency avoids
duplicate execution; it does not itself make an obsolete command fresh.

The controller must derive `not_after` in the server's qualified clock domain
with bounded mapping error; same-host monotonic comparability is a contract
to qualify, not an assumption. If the requirement is cut-to-apply<=250ms,
the expiry derives from the **observation cut**, accounting for mapping error,
not simply250ms after server receipt. A disconnected client cannot renew
authority indefinitely. Server lease expiration/watchdog action, if desired,
is another explicit policy and never retroactively cancels existing work.

### 6.3 Operational fallback under the guarded API

Select a fixed cap from development before validation; do not hard-code a
"safe"16/24/32 based on a favorable single capture. On stale observations,
unsupported state, expired computation or model failure, request that fixed
cap only through the same guarded transport. On uncertainty, first resolve
or expire the outstanding operation; a conflicting version requires a
fresh qualified decision, not blind overwrite.

Even a successful fallback acknowledgement guarantees only that a particular
budget applied at that time under the contract. It does not guarantee
latency, drain, service availability or that a later authorized action never
changes it. The exact desired "no obsolete overwrite" guarantee is provided
by version/lease/expiry checks, not by calling a cap conservative.

## 7. Support, calibration and identification contracts

### 7.1 Artifact identities and observable scores

A sealed model bundle binds source/runtime/tokenizer/cost identities, feature
schema and units, fitting data/chronology, imputation, lifecycle/kernel
parameters, ingress/success adapter, scenario generator, support rules,
planner family/horizons, fallback, calibration unit and score definition.
Changing a controller/model/observer setting changes the relevant version.

Support is joint over marked workload, observed state, action histories,
delay and runtime. Being between observed cap values or within a Mahalanobis
ellipsoid is not intervention support. Missing engine information cannot
be filled from future offline records. Prefix64/96 historical observations
do not qualify every corresponding current-host transition.

Initial shadow intervals/ensembles are descriptive, not simultaneous
counterfactual confidence bounds. Choose among these measurable future
calibration targets explicitly:

- Episode-level forecast scores for a **frozen full policy** on fresh
  exchangeable episodes.
- Prespecified randomized-intervention outcome scores for their defined
  visitation law, with overlap and temporal assumptions.
- Simultaneous confidence intervals for **population mean policy contrasts**
  from randomized independent episode blocks.

None exposes all candidate potential outcomes from one exact physical
origin. Do not construct missing labels by comparing simulator outputs to
themselves. The Mayor's two-world example has identical factual observations
and opposite intervention signs; no factual predictor score resolves that.

### 7.2 Exact statistical limits

For fixed score/model and n exchangeable calibration units, split-conformal
uses rank`ceil((n+1)*(1-alpha))`; when the rank exceeds n, the required bound
is infinite. Marginal coverage is not state-conditional or automatically
simultaneous under policy search/adaptation.

For n independent identically distributed Bernoulli episode violations and
zero observed violations, the one-sided95% upper probability is
`1-.05**(1/n)`. Executed values:

    n=299  -> .00996915
    n=2995 -> .00099974.

These valid bounds concern the specified independent units. They are not
conformal sample minima in general. Correlated requests/windows cannot
replace the independent episodes merely because GPU time is costly.
Zero failures among13,962 correlated requests is a descriptive count unless
an appropriate dependence/design model justifies more.

Repeated fixed-level tests need a defined sequential procedure or declared
finite monitoring horizon with error control. CUSUM thresholds/ARL measured
on development are empirical operating characteristics, not guarantees under
arbitrary nonstationarity. Online adaptation is initially excluded; a later
RLS/dither version needs a new calibration/selection protocol.

### 7.3 Experiment order and numerical planning illustration

1. CPU: exact ledger, cohort partition, guarded fake transport and bounded
   worker tests. Fit/compare W_tok and compact kernels on existing development
   only, respecting label availability. Stop an unproductive finite model
   search without inspecting C.
2. Separately authorized native observation/action qualification: coherent
   marked feed, scheduled/client label availability, cancellation behavior,
   observer overhead, server guard timing, request modes and identities.
3. Authorized randomized cap/prompt/history interventions with known
   assignment probabilities and enough excitation. Test old-cohort effects
   without outcome-window regression as a causal substitute.
4. Freeze fixed, same-state feedback, same-model myopic and planner policies;
   perform randomized whole episodes within host/time blocks with matched
   offered seeds and complete denominator accounting. A switchback may be a
   separate design with treatment-history estimand and justified carryover.

Chronology: development/model selection -> fresh calibration of frozen
candidate -> readiness validation -> sealed final policy experiment.
If validation selects among candidates, it is not the final superiority
test; changed models/policies may need fresh calibration. Purge by actual
label resolution and history carryover, not a guessed5s universal drain.
Natural-EOS excitation and fresh-host/workload transfer remain distinct gates.

Illustration only: for independent paired block differences with SD1req/s,
delta=.4req/s, two-sidedalpha=.05 and power.8, normal planning gives
`((1.96+.84)/.4)^2=49` pairs. This is not measured achieved power, not
49 necessarily independent windows in one run, and not authorization for
GPU time. Pilot variance uncertainty, clustered hosts, multiple baselines
and guardrails may require more. The owner must set practical delta and
resource limits before collection. If precision is unaffordable, report
inconclusive/limited scope rather than manufacture an independence claim.

A superiority claim requires the predeclared interval criterion against the
required baselines and guardrails. A lower bound<=delta means no demonstrated
useful superiority; an upper bound<delta supports a scoped negative result.
No significance does not prove equivalence. Include fallback/invalid episodes
and their reasons under frozen integrity rules; no retrospective favorable
subset.

## 8. Interfaces, decision record and falsification gates

### Five proposed module boundaries

| Module | Contract | Hard boundary |
|---|---|---|
| Observer/ledger |`ingest(batch) -> LedgerSnapshot or Unavailable` |Validated causal prefixes, marked reservations and label/cohort tables; no control writes |
| Frozen model/scenarios |`prepare(snapshot, bundle, seed) -> Context`; `rollout(context, plan, work_budget) -> Trajectory/Partial` |Exact admission/release; small learned feature set; supported stochastic assumptions; no future trace |
| Scorer/planner |`score(trajectory, cohort_partition) -> Value/Bounds`; `choose(context, plans, deadline) -> Decision` |Same cohorts/scenarios for all plans; complete-success accounting; bounded matched work |
| Disabled/guarded transport |`submit(command) -> OperationStatus`; `resolve(id)` |Fake-only initially; explicit authorization and server expiry/CAS/lease required for guarded writes |
| Evidence evaluator |`audit(offers, outcomes, observations, decisions, operations, protocol)` |Independent full-offered metric reconstruction and immutable acceptance statuses |

A lifecycle kernel can be replaced without changing ledger or reward
semantics. A feedback policy bypasses the model planner but consumes the
same observation context and transport. An external worker receives immutable
snapshots; it cannot import another process's mutable `gateway.state` and
pretend it is a live feed.

Decision record:

```text
decision_id, schema, created_at, expires_at, clock_domain
source/runtime/model/cost/tokenizer/calibration/support hashes
gateway/engine/controller epochs; observation seq/hash and all ages
applied budget/version or explicit possible-set; pending operation identity
cohort partition O/N/F; outcome/ingress assumptions; scenario seed and count
candidate plans, support exclusions, initial/dwell/switch definitions
mean complete reward, baseline gap, residual reward bounds, tail diagnostics
completed work units, cutoff reason, processing start/end, stale-result flag
chosen action/hold/unavailable and reason; shadow/authorized status
transport send/status/apply/ack timestamps, expiry/CAS data, resulting version
```

No credentials enter artifacts. Recommendation, command acceptance, actual
application, collection integrity and performance acceptance remain separate
fields.

### Required falsification tests before even qualified shadow claims

| Test | Expected property |
|---|---|
| Strict boundary and cap reduction |No eviction/clamp; exact-cost fit admitted; over-budget rejected; recorded event order |
| Clustered release/slack |Actual release opens headroom; expected release is not used as a deterministic bound |
| Same-loop snapshot versus transported events |Do not invent asyncio interleavings; lost/reordered external prefixes refuse or reconcile |
| Orphan work after gateway release |R=0 is not automatically engine idle; qualification/uncertainty survives cancellation |
| Old-cohort effect and first-hit-then-failure |One total reward per eligible identity; new-only score cannot substitute |
| Whole-cohort CVaR |Ten-scenario example uses old+new loss, not just new arrivals |
| Terminal boundary shift |Moving an event between explicit rollout and continuation cannot duplicate/drop its reward |
| Opportunity tail |Late long reservation cost appears in F or remains a disclosed beyond-H_s limitation |
| History expiry |Equal rolling totals with different buffers can diverge correctly |
| Future-label perturbation |Changing unavailable future outcomes/arrivals cannot alter an earlier context |
| Timeout, GET and delayed old write |Unchanged GET never resolves a pending write; old result cannot overwrite a newer decision |
| CAS with no later write |Matching version does not bypass expired deadline |
| Clock/epoch/lease mismatch and duplicate IDs |Reject before mutation; exact duplicate returns old result without reapplication |
| Event-loop blocked solver |Same-loop timeout is not treated as preemption; independent worker/cutoff enforced |
| Scenario-prefix completion |No cherry-picked partial candidate/scenario set after cutoff |
| Unobservable calibration maximum |Reject absent counterfactual labels; do not synthesize certification |
| Same-state feedback/myopic fairness |Same observation/model/scoring/transport when isolating planning value |
| Full-offered denominator |Rejection/failure/fallback cannot inflate success rate by changing the denominator |

These tests falsify code/specification mistakes. Passing them does not estimate
real service distributions, server timing, KV safety or policy gain.

## 9. Unresolved disagreements and decisive evidence

| Open question | Current position | Evidence that could change it |
|---|---|---|
| W_tok feedback versus richer phase-age prediction |W_tok first; exact ledger retained for accounting |Fresh causal marked-state ablations, threshold calibration and independent policy value, including overhead |
| Cheap b_after correction versus common lifecycle rollout |Association-only challenger, not a causal replacement |Randomized later-work interventions and old-cohort outcomes; not outcome-window sign arguments |
| Myopic versus multistep |No presumed winner |Same model/cohorts/transport, different future control freedom, then full-policy randomized outcomes |
| Finer than1s cadence |Deferred, not disproved |Actual worker/feed timings and action stability showing useful switch-time distinctions |
| Per-decision uncertainty abstention |Keep hard support gates; model confidence initially heuristic |Identified action contrasts/calibration plus measured abstention/usefulness tradeoff |
| Natural EOS/failure complexity |Preserve semantics; fit only supported small model |New realistic-output/censoring evidence; correlated zero-failure counts do not settle it |
| Engine observer necessity |Not initial dependency |Availability-adjusted incremental decision value greater than its measured overhead |
| Need for live transport changes |Required for the stronger no-stale-write contract |Source-bound server expiry/CAS/lease/idempotency tests and native timing qualification |

This is an evidence-based common engineering core, not a majority-selected
architecture or a unanimous claim of performance. Opus's assertion that
confidence-gated MPC is necessarily inert, Gemini's asserted deadband incidence
and universal compute infeasibility, and any deterministic KV substitution
remain rejected as established facts.

### Supplemental review of the Mayor's provisional mathematical specification

I accept `experiments/mpc-research/mpc-mathematical-specification.md` as a
**CPU-only reference specification**, not a native scheduler or actuation
certificate. Its exact ledger/latent-work separation, whole-cohort success
reward, continuing-arrival tail, nonpreemptive reductions and explicitly
capability-checked guarded transport address the major counterexamples.
I adopt its smaller N=3/1s initial search above because it reduces engineering
scope, not because it is proved optimal or faster than a deadline. The
kernel-inspired transition is a legitimate challenger alongside the measured
backlog law and compact phase kernel; its nonnegative interaction coefficient
is a modeling restriction, not an identified physical interference law.

Four clarifications are necessary before its interfaces are frozen.
**First,** specify the first-output accounting: does completing prefill emit
the first generated token, leaving L-1 decode iterations, or does another
decode step generate it? For p prompt tokens and forced L=1, those conventions
produce different TTFT/work. Pick one, record scheduled/committed/observed
events distinctly, and test it against the restricted reference; also bind B,
sequence/context limits, per-request eligibility and EOS to the model seal.
This is an underspecified surrogate event boundary, not evidence that its
equation currently violates native timing. **Second,** comparing N=1/3/5 while
scoring only offers in each varying prediction window changes both reward
cohorts and control freedom. Hold H_s=5s, O/N/F and continuation fixed for the
primary myopic test; vary control slots separately. A separate scoring-horizon
ablation can then assess truncation. **Third,** distinguish observed unresolved
evaluation outcomes, which cannot receive verified positive credit, from
forecast unresolved mass, which needs a supported conditional expectation or
interval rather than an automatic point zero. **Fourth,** capability checking
must verify an actually enforced API schema and configured identity/seal
registry, not an old handler's acceptance of ignored extra JSON fields.
Require operation identity, server-side expiry checked after lock acquisition,
CAS and restart/writer fencing as detailed in section6; CAS alone leaves
late writes valid when no later version exists.

No concrete dimensional contradiction was found in the Mayor's
`theta_interference*q*d` law: its stated seconds/(prefill-token*decode-token)
units are consistent. Whether that term improves predictions, whether the
kernel surrogate matches current async scheduling, and whether a feedback
baseline makes either simulator unnecessary remain unresolved experiments.
The simpler correction is explicit event/reward/transport semantics, not
another layer of physical-state claims or a vote to approve the model.

## 10. Executed Round 3 verification

The successful command below reproduced the Mayor artifact and checked the
specific corrections. A preceding direct Python-object comparison failed
because the script returns a tuple for the KV risk interval whereas parsed
JSON returns a list. JSON normalization resolved that representation
difference; no numerical result or source was changed.

```bash
python3 -B - <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path
p=Path('/home/daniel_lin/gt/pr32/mayor/mpc-8e92371')
spec=importlib.util.spec_from_file_location('mayor_checks',p/'experiments/mpc-research/mayor_counterexamples.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
expected=json.loads((p/'results/mpc-research-cpu/mayor-counterexamples.json').read_text())
assert json.loads(json.dumps(m.checks(),allow_nan=False))==expected
print('Mayor counterexamples reproduced exactly after JSON normalization:',len(expected['checks']))
print('Independent zero-failure upper95: n299=%.8f; n2995=%.8f' % (1-.05**(1/299),1-.05**(1/2995)))
newA,oldA,newB,oldB=[0]*10,[10]+[0]*9,[2]*10,[0]*10
for full in (False,True):
    scores=[]
    for new,old in ((newA,oldA),(newB,oldB)):
        total=[a+b for a,b in zip(new,old)]
        scores.append(sum(total)/10+max(total if full else new))
    print('full_cohort_tail=%s; A_B=%s'%(full,scores))

def guarded_apply(state,cmd,now,domain):
    if cmd['domain']!=domain:return 'domain_mismatch'
    if now>=cmd['not_after']:return 'expired'
    if cmd['expected_version']!=state['version']:return 'version_conflict'
    state.update(version=state['version']+1,budget=cmd['budget'])
    return 'applied'
old=dict(domain='bootA',not_after=50,expected_version=0,budget=32)
state=dict(version=0,budget=16)
assert guarded_apply(state,old,100,'bootA')=='expired'
print('CAS without newer write is insufficient: expiry rejects late version0 write')
state=dict(version=1,budget=16)
unexpired=dict(old,not_after=200)
assert guarded_apply(state,unexpired,100,'bootA')=='version_conflict'
assert guarded_apply(state,unexpired,100,'bootB')=='domain_mismatch'
print('Guard checks: expiry, CAS and domain all rejected without mutation:',state)
for name,digest in [('astra-round1.md','dbc98841590575cf7d2bbeceb7aa04a09cf79871b309580b0930503bf30754d0'),('astra-round2.md','af7814d3cb372b9c2c9baa897380ed45c1bf3de649a794438e717eec6918e939')]:
    assert hashlib.sha256((Path('experiments/mpc-research/proposals')/name).read_bytes()).hexdigest()==digest
print('Frozen Astra Round1/2 hashes unchanged')
PY
```

Exact output:

```text
Mayor counterexamples reproduced exactly after JSON normalization: 8
Independent zero-failure upper95: n299=0.00996915; n2995=0.00099974
full_cohort_tail=False; A_B=[1.0, 4.0]
full_cohort_tail=True; A_B=[11.0, 4.0]
CAS without newer write is insufficient: expiry rejects late version0 write
Guard checks: expiry, CAS and domain all rejected without mutation: {'version': 1, 'budget': 16}
Frozen Astra Round1/2 hashes unchanged
```

The guard function is arithmetic on dictionaries, not an implementation or
test of the native gateway contract. It omits lease/idempotency behavior,
which remains in the required future test matrix. The Mayor artifact is also
a logical/arithmetical falsification suite, not a serving benchmark.

**Round 3 barrier:** this revised report is the only new deliverable.
No product changes, commits, pushes, live writes, paid resources, final C,
sandbox changes or extra agents were used. All native/HTTP/GPU/action/policy
performance acceptance remains false.
