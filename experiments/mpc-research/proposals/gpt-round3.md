# Round 3: revised minimal design and evidence-based adjudication

Researcher: onyx / GPT-6.1 Sol (`gpt-6.1-sol`).
Product source freeze: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Scope: independent revision after reading all four complete Round2 reports,
the Mayor's counterexample script, and its saved JSON. Prior reports remain
frozen. This document proposes default-off CPU/replay/shadow engineering;
it does not authorize live budget updates or claim native performance.

## 1. Decision and corrected evidence standard

**Recommendation:** implement the smallest common observation/accounting,
cohort-scoring and bounded-planning contracts first. Compare fixed control,
strong same-state backlog feedback, a same-model one-decision policy and a
small multi-step planner. Start with a reduced gateway model; do not make a
rich engine observer, learned terminal value, online RLS, CVaR optimizer or
per-decision counterfactual confidence certificate prerequisites.

The defensible convergence is on contracts and falsifiers, not on the winning
controller. Opus's reproduced W_tok predictor is a strong reason to test
backlog feedback early. It is not evidence that feedback beats MPC, or that
either beats the best fixed policy. Longer planning remains a challenger
that may legitimately lose.

### What is evidence, and what is retracted or explicitly not claimed?

- The Mayor-reproduced development AUC/MAE remains useful factual evidence:
  backlog-plus-own-prompt AUC about0.959-0.995 and MAE0.033-0.093s across12
  held-run development scores. Those are not fresh chronological acceptance,
  conditional calibration or intervention results.
- I executed the Mayor's eight deterministic checks and matched the saved
  artifact after JSON normalization. The initial raw Python-object comparison
  failed only because a Python tuple serializes as a JSON list; normalizing
  through JSON resolved it. No model/numerical discrepancy was concealed.
- I did not execute a serving-model/runtime benchmark in any round. Every
  proposed100ms/20ms/other timing number is a target until source-bound
  measurement exists. I retract any interpretation of my operation-count
  estimates as measured runtime or universal compute lower bounds.
- A toy implementation taking57ms cannot prove every implementation needs
  at least57ms. Compilation, prefix sharing, aggregation, data layout and a
  different correct algorithm can change runtime. Nor does a toy fitting
  a target qualify the specified observer/model/tail pipeline.
- Selected289-pass/one-skip local checks are not full native acceptance.
  The shared audit separately records a full Python3.12 CPU/mock baseline;
  that still does not qualify installed-vLLM schema, GPU/HTTP timing or MPC.
  No new native acceptance was executed here.
- No empirical maximum residence, observed throughput, per-tick expected
  release, or diagnostic b_after flip count is a physical/causal upper bound.
  The sign of b_after endogeneity bias is **not established**.
- Zero failures among13,962 correlated development hits is descriptive. It
  is not automatically a rule-of-three certificate for a deployment population.
- The299-independent-zero-failure-episode result for a one-sided95% upper
  violation probability below1% is mathematically valid. It is not a generic
  minimum for every conformal interval. Cost cannot justify replacing its
  independent trials with correlated windows while retaining its theorem.
- Reservation-derived token inequalities, even if arithmetically correct,
  do not certify physical KV safety without actual tokenizer/context,
  allocation/block/padding/capacity and backend-orphan-work qualification.
- The Poisson probability of at least two offers during an assumed30ms
  interval is not a measured deadband incidence. Marks, old headroom,
  releases, actual delay and subsequent arrivals also matter.

These corrections apply regardless of which author made the original claim.
All runtime, acceptance and deployment statements below use this standard.

## 2. Response to every substantive critique of the gpt design

References are to frozen peer Round2 IDs/headings. "Accept" changes this
revision; "reject" rejects the stated inference, not the underlying concern.

| Critique | Adjudication | Evidence and concrete disposition |
|---|---|---|
| opus G1: preserve old cohorts/success; empirical failures negligible | **Accept accounting; reject probability certificate.** | Keep complete-stream reward exactly. A simple success model may be an explicit development assumption; correlated zero failures do not establish deployment risk. |
| opus G2: replace old-cohort continuation with a closed-form b_after term | **Accept as a challenger; reject equivalence/causal-bound claim.** | Its coefficient and window are endogenous. Test the approximation against observed supported interventions; do not call it the same causal quantity by construction. |
| opus G3: per-decision confidence/margin gate makes MPC provably inert | **Accept deployability concern; reject proof.** | Factual action bias need not equal action-gap error: common bias can cancel. Remove a mandatory policy-level useful-gain threshold from each decision. Keep support/identity gates; evaluate nominal policy value independently. |
| opus G4: [0,K] terminal bounds dwarf useful gaps | **Accept conservative-mode risk; reject universal degeneracy.** | Broad worst-case intervals can cause abstention; K alone does not determine differential uncertainty. Return bounds as diagnostics, not mandatory certification of every nominal action. Freeze/evaluate any interval-abstention policy separately. |
| opus G5, astra S1: pending writes and acknowledged-versus-applied state | **Accept.** | Acknowledgement loss can leave actual budget different from last acknowledged. Introduce persistent command-uncertain state and require server fencing for a finite-recovery live design. A GET is not a fence. |
| opus G6: weak oldest-age-only baselines | **Accept.** | Compare tuned W_tok/R/D backlog feedback with exactly the same observation access as MPC. An information win is not a horizon win. |
| opus G7: toy runtime supports100ms, richer model3-10x slower | **Accept need to measure; reject universal multiplier.** | Neither toy timing nor an asserted3-10x factor bounds the retained implementation. Benchmark it at its actual supported event/state limits. |
| opus G8, astra A8: compute a numerical sample-size plan now | **Partially accept.** | Include illustrative planning arithmetic, but independent episode variance, practical margin and multiplicity remain unresolved. Correlated30s-window variance does not identify reset-episode variance. |
| opus G9: forced128 phase age nearly suffices; no engine feed needed | **Accept first experiment; reject established sufficiency.** | Age/phase is a cheap residual-life proxy worth testing. Shared service variation and pending work can still alias it. Engine augmentation must earn inclusion; it is not assumed necessary. |
| astra S2: terminal value/cohort double-counting specification gap | **Accept.** | Replace unspecified learned V plus burden with the finite, disjoint cohort/value equation in Section4. No additive burden proxy in the initial objective. |
| astra S3: full counterfactual-family calibration score is unavailable | **Accept.** | Calibrate observable frozen-policy scores only; randomized average contrasts are distinct. Model action-gap spreads remain heuristics, not conformal guarantees. |
| astra S4: backend work can outlive reservation; worker cutoff underdefined | **Accept.** | Exact ledger bounds only gateway reservations. Retain orphan-work uncertainty/invalid-scope flag; qualify cancellation. Bound worker events, elapsed time and result expiry outside serving locks. |
| gemini Defect3.1: terminal formulation necessarily triple-penalizes | **Accept ambiguity; reject inevitability.** | Round1 proposed alternatives, not an instruction to sum all three. Nevertheless it lacked a unique partition. Section4 now has no learned V or separate W penalty, so the alleged triple charge cannot arise. |
| gemini Defect3.2: gpt demands retry feedback in no-retry replay | **Reject.** | Round1 explicitly restricted current experiments to open-loop no-retry replay and made retry modeling conditional on expanding scope. This revision keeps no-retry replay; no retry subsystem is requested. |
| gemini proposal comparison: gpt/ astra require engine probes | **Reject.** | Both gateway-first proposals made engine augmentation conditional on causal availability and incremental value. W_tok feedback remains a first-class gateway-only candidate. |
| gemini:299 episodes are impossible, use correlated windows instead | **Reject retained statistical claim; accept resource limitation.** |299 independent zero-failure episodes give the stated bound. With insufficient authorization/data, weaken/defer the claim; do not silently change its unit of independence. |
| gemini: assumed arrival probability proves17.9% deadbands;20Hz necessarily ruins goodput | **Reject measured/causal conclusion.** | The arithmetic is a toy arrival probability, not event incidence or policy value. The stale-slack race is real, its frequency and effect unmeasured. Apply the same standard to1s and250ms controllers. |
| gemini: active probing/continued writes solves timeout paralysis | **Reject as a general repair.** | Probing helps observe state but cannot prove a pending old write has terminated. Additional writes can be overtaken. Finite safe recovery requires a specified fence/expiry contract. |
| own Round2: Astra's new-only CVaR omits old-tail risk | **Retain.** | Executed scores A/B change from1/4 to11/4 when old loss is included. Initial objective is risk-neutral; any later tail diagnostic uses complete old+new+opportunity reward. |
| own Round2: release/terminal Python statements imply no same-loop race by themselves | **Retain with scope.** | The inspected non-suspending cleanup segment and asyncio test support same-loop coherence, not arbitrary event-feed delivery or physical backend completion. Observation-feed atomicity must be separately tested. |
| own Round1: rich lifecycle/ledger state as initial predictor requirement | **Narrow.** | Keep a minimal identity/cost/phase/age ledger for exact accounting/scoring. Derive W_tok/R/D for the first predictor; richer predictive features must win an ablation. Storage for correctness is not a mandate for a large learned state. |

### Peer corrections that remain necessary

Opus's own retractions of the7.4/s ceiling and hard terminal claim are accepted.
Its new b_after "upper estimate," shared-request rule-of-three certificate,
"cheapest possible implementation" lower bound and physical KV substitute are
not accepted. The proof of terminal infeasibility should use a possible
no-release residual trajectory, not infer cost-weighted residual work from a
median total residence. Its proposed cheap-job Little's-law calculation is a
sensitivity calculation, not measured policy capacity.

Gemini's retraction of its absent-KV chance constraint and C24 qualification
claim is accepted. Its Round2 "robust, unassailable" design still claims an
unexecuted<15ms runtime, a proven safe C16 and immediate successful fallback on
timeout. Those claims must be retracted. Historical C16/C32 evidence establishes
development use, not current-host native policy qualification.

Astra's corrected calibration scope, explicit old-value term and caution about
unresolved writes remain useful. Its original new-only tail risk and an
unobservable all-plan error maximum are not adopted. No proposal wins by how
many authors endorse it.

## 3. Smallest coherent observer and model contract

### 3.1 Boundary and units

One cost-aware gateway worker, one identified backend, fixed tokenizer/cost
definition, fixed scheduler/runtime configuration, open-loop no-retry offers.
The only candidate online action is positive weighted reservation budget B,
not protected concurrency, engine B2048 token work, priority, cancellation or
per-request deadline rejection.

Default reservation convention, held fixed:

```text
c_i = max(c_min, c0 + c_p*p_hat_i + c_o*M_i)              [cost units]
c0=.259; c_p=.000758 cost/token; c_o=.0052083333 cost/token.
```

M is requested maximum output tokens, not eventual generated length. Prefix,
tokenizer fallback, cost coefficient changes and engine restarts alter support
identity. The constants above are source defaults, not a physical service law.

### 3.2 Minimal observable record

Use an immutable snapshot plus sequenced updates with:

| Field | Exact meaning / units |
|---|---|
| Identity | Schema/source/model/cost/tokenizer/runtime hashes; gateway run/boot; backend identity; clock domain; writer epoch |
| Cut/availability | Source sample/cut time and end of coherent snapshot; observer receipt/validation time, in seconds in identified domains |
| Budget state | Latest **observed** budget/version, its sample time, and any unresolved command's possible application state |
| Ledger per admitted request | ID, reserved c_i, estimated prompt p_hat_i, requested max M_i, receipt/admission timestamps, gateway first-content timestamp if observed, terminal status if resolved |
| Scoring metadata | Client schedule/first-content/complete-success labels only if already causally delivered; otherwise explicit unknown/ingress belief |
| Feed health | Sequence range/gaps, completeness, reset, tokenizer fallback and delayed/unknown release/backend termination flags |

Derived predictor inputs:

```text
Q = count(admitted, no gateway first content, nonterminal)       [requests]
D = count(gateway first content, nonterminal)                    [requests]
R = sum(c_i for gateway-nonterminal reservations)                 [cost]
W_tok = sum(p_hat_i for requests in Q)                            [tokens]
age_i = cut - phase_entry_i                                     [seconds].
```

W_tok is **not residual physical prefill tokens**: a request can have partially
computed its prompt without observed first content. Current metrics expose R,
Q/D proxies and receipt-based oldest-awaiting age, but not this entire marked,
admission-aged ledger. A coherent marked snapshot/feed is future observation
engineering. CPU replay may reconstruct it from already available events;
offline reconstruction alone is not live availability qualification.

Old offers still in transit before the gateway are not magically in this
ledger. If needed, they are a scenario belief conditional on available ingress
observations; their scheduled-age law is an assumption to validate. No future
client sidecar or prerecorded future offer tape is an online model input.

### 3.3 Exact accounting, uncertain service

In the gateway's actual serialized event order:

```text
accept_i = 1{R_before + c_i <= B_actually_applied}.
R_after_admit = R_before + accept_i*c_i.
R_after_terminal_release = R_before - c_i.
```

Failures/cancellations release the original c_i; first content does not.
Reducing B never clips R. A sample with unknown pending application simulates
possible applied budgets, not the last acknowledged budget as unquestioned fact.
Equal timestamps require sequence order or explicit order uncertainty.

If A,F,E_Q,E_D are admitted, first-content and phase-terminal counts:

```text
Q_next = Q + A - F - E_Q
D_next = D + F - E_D
R_next = R + sum(accepted c_i) - sum(released c_i).
```

Generate all flows from the same request transitions. No independent cost-out
head, smooth cost release per token, or projected R<=new_B.

**First predictive challenger:** the reduced backlog-plus-own-prompt TTFT law,
with source-bound frozen coefficients and episode/regime residuals. Coefficients
in s/kTok require explicit p/1000 scaling. Target is client-scheduled TTFT
only when fitted/calibrated against that target; a live gateway label is not
silently substituted. R and D enter competing ablations, not a mandatory large
parameter model.

**Minimal multi-step transition adapter:** an event-driven request lifecycle
model exposing a conditional joint distribution of next first-content,
terminal time and success/failure, given phase, age, marks and the evolving
W_tok/D regime. A discrete table/empirical residual resampler is sufficient
for an initial CPU implementation; a neural model is unnecessary.

- Use observed paired first/terminal spans or competing-risk transitions,
  enforcing first-before-success and valid phase ordering.
- Residual-life sampling conditions on survival to current phase age; it
  must not redraw an unconditional fresh lifetime at every tick.
- Retain shared episode/service-regime shocks across requests. A common
  exogenous scenario is shared across plans, while admissions and lifecycle
  transitions respond to each plan's evolving state.
- Do not freeze already-admitted first-token times and then claim old-cohort
  interference has been modeled. A frozen-at-admission version is a labeled
  reduced-model challenger; a state-responsive transition challenger can
  revise residual hazards under later admissions. Their discrepancy is a
  model-risk diagnostic until randomized interventions distinguish them.
- Existing all128-success data support a forced/effective128 development
  model, not an identified natural-EOS law. Natural length, cancellation and
  failures remain explicit scope/support gaps; observed failures are still
  scored correctly even when rare.
- For unknown or empty strata, return unsupported/missing or a declared
  pooled prior plus provenance, not a falsely precise physical prediction.
- No rolling history is required by the first predictor. If later retained,
  add only its required expiration buffer and evolve it from generated events.

This adapter is a statistical model, not an identified physical simulator.
Its full conditional form cannot be certified from the current factual AUC.
A richer age/phase hazard or qualified engine model is an alternative only
after a finite ablation shows a relevant decision/prediction improvement.

Physical backend work can outlive gateway reservation on uncertain cancellation.
R therefore bounds gateway reservations, not GPU work. Until cancellation
propagation is qualified, such an event sets an orphan-work uncertainty or
invalid-scope flag; do not reset the simulated engine to empty merely because
the gateway ledger reaches zero.

## 4. Corrected cohort reward and terminal treatment

### 4.1 Evaluation metric remains unchanged

```text
g_i = 1{successful complete client stream}
      * 1{client_first_content_i - scheduled_offer_i <= .5 seconds}.
G_episode = sum_{offered during full offered interval} g_i / T_offered.
```

Rejections, failures, malformed/incomplete streams and late streams earn zero.
No E2E deadline is added. Terminal/drain follow-up is required for measurement
but does not shorten/change T_offered. ITL/TPOT, E2E, rejection, failures,
transition behavior and size strata remain separate reported metrics.

### 4.2 Disjoint finite prediction objective

At origin t define fixed **scenario cohort membership** from scheduled time,
not from action-dependent admission/completion:

```text
O = offered before t with action-sensitive unresolved outcomes at t.
N = offers scheduled in [t,t+H).
F = offers scheduled in [t+H,t+D_eval), D_eval > H.
K = O disjoint-union N disjoint-union F.
```

O includes known admitted survivors and, where represented, uncertain
pre-gateway in-transit old offers. Known irreversibly late/rejected/terminal
reward can be removed as an action-independent constant, but their remaining
resource work must not disappear. No unseen old request is supplied by future
capture joins.

Execute candidate U only over H, then fixed development-selected backup pi_B.
For scenario m the model value is the finite cohort total:

```text
Z_m(U) = sum_{i in K_m} g_i under (U until H, then pi_B).
Delta_m(U) = Z_m(U) - Z_m(U_backup).
J(U) = mean_m Delta_m(U).                          [successful requests]
```

O,N,F partition rewards; they do not partition away resource interference.
F makes post-H opportunity cost explicit. Arrivals after D_eval continue as
unscored interference through a declared resolution cutoff T_resolve.
This is a finite-window control surrogate, not an infinite undiscounted value.
Its suitability requires sensitivity to D_eval and T_resolve.

At cutoff, split K into outcome-resolved C and unresolved U_K:

```text
Z_hat = sum_{i in C} g_i
        + sum_{i in U_K} P_model(eventual g_i=1 | trajectory to cutoff).
Z_lower = sum_{i in C} g_i
Z_upper = Z_lower + count(unresolved members whose g_i can still be1).
```

An already known late first token contributes zero, not [0,1]. An on-time first
token with unresolved terminal outcome is not automatically a success.
The remainder law must be frozen with the model and labeled assumption/model,
not calibrated merely by zero observed correlated failures. If unavailable,
return an interval/no nominal value or use a separately declared development
assumption; do not invent a finite learned guarantee.

**Initial design has no learned terminal V and no additive burden W penalty.**
The finite F cohort already accounts for the specified future opportunity
window. The unresolved remainder concerns only unresolved members of K.
Thus no reward appears in both tail simulation and another terminal term.
Moving an event across the explicit/remainder boundary must preserve value
when the remainder model is exact. Extending D_eval changes the declared
objective window; report that sensitivity rather than pretend invariance.

Worst/best intervals are honest diagnostics and an optional separately frozen
robust-abstention policy. The nominal shadow planner need not prove a
counterfactual confidence bound at every origin. If unknown tail mass dominates
the nominal choice, log it and reject readiness of that policy/model if it
persists; do not relabel ignorance as confidence.

If a later CVaR variant is justified, apply it to complete-population loss or
regret from the same Z_m, including old cohorts and correlation. Initial
lambda_tail=0; switching is a tie-break/recorded diagnostic, not an invented
dimensionless penalty mixed with seconds or cost.

## 5. Candidate controls, planner and strong baselines

All numbers in this section are **engineering targets/proposed development
settings**, not measurements or approved live settings.

| Item | Minimal initial configuration |
|---|---|
| Candidate budgets | {16,32} cost units, conditional on source/runtime/state/action support; neither is called universally safe |
| Decision cadence |1s target, with measured observation/compute/application ages; no claim it can rescue an already-imminent .5s deadline |
| Planned controls | H=3s, three1s slots; N=1/3/5 are development comparisons with fixed scored cohort window |
| Plan family | All binary three-slot paths:2^3=8 before support/dwell pruning; one-switch and five-slot families are later ablations |
| Prior applied action | Known/observed before planning; every transition, including from it to u0, is counted and dwell-checked |
| Scenarios |16 common paths initially; seed/count sensitivity against64 is a development test, not tail calibration |
| Opportunity/resolution tail | Initially D_eval=10s and T_resolve=20s after origin, held fixed across N=1/3/5; finite model cutoffs, not drain bounds; test longer tails before freeze |
| Compute target | <=100ms outside serving locks/threads; max events, particles and tail bounds also enforced |
| Observation target | Source age<=100ms at planning start, with full cut-to-use age recorded |
| Application target | <=50ms response target; total cut-to-confirmed-apply<=250ms target in a future qualified transport |

Eight paths times16scenarios is128rollouts before tails. No timing is inferred
from that count. Compare only matched completed scenario sets; a partially
evaluated optimistic path cannot defeat a fully evaluated baseline.
Candidate paths are open-loop after the first action; no scenario-specific
clairvoyant future controls. Retain constants/backup/hold when supported.

Selection in nominal shadow mode: maximize mean modeled Delta; numerical ties
prefer hold, then backup, then a deterministic order. No mandatory
"independently significant useful policy gain" test is performed per origin.
Hard source/support/clock/health gates still refuse malformed or unsupported
plans. Optional uncertainty-margin and robust-abstention policies are separate
frozen competitors whose firing/fallback rates must be reported.

### Same-state baselines, not weak information-limited straw men

1. **Fixed:** all qualified fixed actions, with one best fixed selected on
   development and then frozen.
2. **Backlog feedback:** tuned monotone/hysteretic mapping from W_tok,R,D and
   available ages/marks to {16,32}, with the same observer, application-delay
   model and dwell rules as MPC. Tune thresholds/margins on development only.
   The useful W_tok signal is not reserved exclusively for the planner.
3. **One-decision, same-model:** choose u0 in{16,32}, then pi_B from slot1 onward;
   score the same H/D_eval cohorts and resolution procedure as the multi-step
   policy. Both resulting paths are in the eight-path family. This isolates future
   control freedom while retaining identical lookahead information/model.
4. **Short-reward-horizon model policy:** genuinely one-slot arrival reward
   plus old-cohort outcomes, with their eventual-success resolution. Label
   the different scored opportunity window; do not confuse this with item3.
5. **Optional PI/AIMD:** tuned backlog/age control with anti-windup, matched
   supported actions and observations, only if not redundant with item2.

Opus continuous `R+slack` is an additional important challenger, not silently
equivalent to item2: it uses different continuous/high-frequency action
support. Test it in CPU replay with actual releases/delay scenarios first;
live comparison requires separately qualified support. An atomic relative
update removes read-to-apply R mismatch, not releases later in the tick.
A true per-request deadline gate is a new actuator/project, not an existing
budget policy.

No hard terminal-clearance, per-request latency, absent-KV or OOM constraint
is imposed. Hard constraints are valid positive supported budgets, exact
accounting, lifecycle/event validity, identity/age limits and bounded work.
Latency/tail/model-risk thresholds are soft/model-relative. If unmet, surface
infeasibility/abstention, not a deterministic SLO guarantee.

Under validated c_min>0 and bounded budgets, exact gateway accounting gives
`R<=max(R_initial,B_max)` and the corresponding reservation-count bound.
It gives neither a bound on pre-handler HTTP backlog nor on orphaned engine
work nor finite stream lifetime. No Lyapunov/recursive-latency theorem is
claimed.

## 6. Transport, fallback and decision records

### Existing API: honest degraded behavior

Before separate authorization, **transport is a recording stub: no POSTs**.
A planner result cannot itself acquire admin authority.

For any legacy-API experiment that is separately authorized, one timed-out
write remains unresolved until positive evidence establishes application or
definitive termination. Observing the prior version is insufficient; loss of
connectivity means the last actually applied cap persists. A fixed backup is
a desired policy, not a guaranteed transmitted reduction. It is acceptable
to remain command-uncertain indefinitely with this API.

### Proposed requirements for a finite-recovery live transport

These are a separate future gateway change, **not implemented or authorized**:

```text
BudgetCommand {
  operation_id, writer_epoch, gateway_boot_and_clock_domain,
  expected_budget_version, requested_budget,
  not_after_in_gateway_clock,
  decision_id, source_observation_cut, model_and_support_hashes
}
```

At the atomic application boundary, after acquiring state_lock:

1. Validate owner/epoch/boot, mode, action authorization and budget.
2. Check expiry against gateway monotonic time **inside** that boundary.
3. Check expected budget version (CAS).
4. Apply exactly once, increment version and durably retain/query operation
   outcome; return applied time, operation ID and version.

Repeated operation IDs return the same retained outcome, not another write.
Cross-machine clocks require a qualified mapping/server-issued lease, not
comparison of unrelated monotonic numbers. Writer ownership/epoch prevents
restart-created split writers; boot identity prevents version0 reuse after
restart. CAS does not freeze R or W_tok between sample and application.

**Both CAS and expiry are required for the proposed contract.** CAS rejects
an older command overtaken by a newer applied command. Without a newer command,
a long-delayed old command can still pass CAS; expiry rejects that case.
Expiry alone need not reject an older command that arrives before its expiry
but after a newer action. Idempotency/queryable outcome resolves retry
ambiguity; operation retention and epoch reset semantics must be specified.

A local decision deadline rejects late worker results before they are sent;
server expiry rejects late already-sent requests. A watchdog on a blocked
event loop is not a preemption mechanism. None of these guarantees an update
arrives during complete network/server failure. Automatic fallback on lease
loss would require a separately specified server-side watchdog/lease policy;
it is not supplied by an expiring command.

### Minimal state machine

```text
SHADOW -> never sends
READY -> plan from qualified immutable snapshot
PENDING(op) -> at most one unresolved operation
APPLIED/REJECTED(op) -> reconcile positive retained outcome/version
UNCERTAIN(op) -> timeout/lost response; no new speculative legacy writes
DEGRADED -> stale/OOD/model/worker failure or unresolved transport
```

Model/worker failure with no pending command selects the frozen backup policy
as an intention. Only a confirmed application establishes its actual budget.
Recovery requires a fresh coherent snapshot and known operation/epoch state;
a stale worker output is discarded by decision ID and expiry.

### Required immutable DecisionRecord

Snapshot/schema/source/run/clock identities and hash; cut/receipt/validation
times; observed budget/version plus uncertainty; ledger sequence/health;
model/fit/calibration/support IDs; scenario IDs/seeds; exact cohort partition
and cutoffs; candidate paths and exclusion reasons; completed-scenario counts;
nominal old/new/opportunity rewards, unresolved bounds and diagnostics;
chosen path/first action/backup reason; tie rule; elapsed and event budgets;
intended expiry; and explicit `shadow/live_authorized` status.

ApplicationOutcome is a distinct record: operation ID, send/timeout/query
times, server boot/domain, accepted/rejected/unknown, applied budget/version
and time. Join to outcomes by decision/operation identity, not nearest time.
Do not include credentials. Expired/unavailable/fallback decisions remain in
evidence denominators.

## 7. Minimal CPU interfaces and falsification gates

The following can be implemented with pure functions plus one bounded worker;
they are contracts, not a request for a new framework or many files:

| Boundary | Input -> output | Required property |
|---|---|---|
| Observer/reducer | coherent snapshot + available sequenced events -> immutable ledger/health | Exact identities/costs, no future joins, gap/restart detection; derive W_tok consistently |
| Scenario/model | ledger + frozen fit + plan + common seed + event/time limits -> lifecycle rollout or explicit unavailable | Exact admission/release, valid correlated phases/residual life; no hidden actual future tape |
| Scorer | rollout + fixed O/N/F membership + cutoffs/remainder law -> nominal value, partition totals, unresolved bounds | Eventual complete success and scheduled client TTFT; no duplicate reward |
| Policy selector | same state/model/scenarios + supported path family -> shadow decision | Includes fixed/backlog/one-decision comparators; bounded matched evaluation; deterministic ties |
| Transport stub/state machine | decision + authorization/operation state -> recorded intention/outcome | Default-off; planner cannot send; pending/expiry/CAS contract explicit |
| Independent evaluator | immutable offers/client/gateway/actions + frozen protocol -> episode metrics/gates | Full offered denominator, follow-up, invalid/unavailable/interrupted accounting |

Initialize coefficients/models once outside timed observation. Worker receives
only immutable snapshots and no admin credentials. Bound per-rollout events,
population/particles and tail length; check elapsed time at bounded event
intervals; stop/ignore results after the decision expires. If a worker cannot
be cooperatively bounded, isolate it in a killable external process and keep
gateway/engine serving loops independent. Test that a timed-out worker neither
accumulates indefinitely nor later sends a budget.

Essential executable tests before even shadow-readiness claims:

- Strict cost boundary, same-time lock order, whole-cost terminal release,
  duplicate cleanup, rejected requests absent from reservations, and reductions
  with R above new cap.
- W_tok parity at sampled cuts; receipt/admission ages distinct; token fallback
  and cost definition sealed; future event/label perturbations cannot change
  an earlier observation/decision with a fixed scenario seed.
- Same R/Q/D with different remaining lifetimes; equal rolling sums with
  different expiration; correlated releases exceeding mean700token allowance.
- On-time first token then incomplete stream earns zero; old-cohort
  externality reversal; all O/N/F sets disjoint; known late reward zero;
  exact remainder-model boundary invariance and finite opportunity tail
  sensitivity.
- Frozen versus state-responsive old-cohort transition; no causal gain claimed
  from diagnostic coefficient subtraction; unknown natural EOS stays unsupported.
- Orphaned backend work after cancellation, engine restart and missing feed:
  no inference of physical empty state from gateway R=0.
- Client timeout then old-state GET then late write; CAS rejects overtaken
  command; expiry rejects a late command even with no later write; duplicate
  operation, wrong boot/epoch and restart fence correctly.
- Source-to-application age, compute cutoff, max events, stale worker result,
  serving-loop noninterference and complete evidence log under overload.
- All eight paths counted before support/dwell pruning; prior-to-u0
  changes recorded; one-decision paths contained; no scenario-specific
  clairvoyant action; matched scenario prefixes across candidates.
- Calibration refuses nonexistent counterfactual labels and declares sample
  units/population; unsupported claims cannot be promoted by a passing mock.

CPU replay can establish these bookkeeping/algorithm properties and measure
that CPU implementation. It cannot certify native observation freshness,
natural-EOS survival, action-effect accuracy or serving-policy benefit.

## 8. Calibration, experiment and evidence gates

### Observable targets only

Freeze model, policy family/selection, observer, remainder law, scenario
generator, support gates and fallback before calibration/evaluation.
For a selected frozen policy, an observable episode score can be maximum
normalized optimistic forecast error over its **actually executed** decisions.
It can include multiple horizons for those executed trajectories. It cannot
include errors of31 unexecuted same-state action plans.

Split-conformal rank is `ceil((n+1)*(1-alpha))`; if beyond n, the corresponding
distribution-free finite upper score is unavailable/infinite. Its marginal
statement needs exchangeability of calibration and next score units. Using
episode maxima can support an episode-level marginal statement under those
assumptions, not arbitrary conditional state/action guarantees.

Per-origin ensemble/scenario spreads are model diagnostics unless separately
justified. Randomized action experiments can estimate supported population
contrasts; randomized complete policies estimate average policy-value
differences. Neither reveals every individual's potential-outcome vector.

For independent episode violation indicators with zero failures:

```text
p_upper95 = 1 - .05^(1/n).
n=299  -> .00996915
n=2995 -> .00099974.
```

If that experiment is unaffordable, do not claim that confidence target.
Correlated request/windows are not replacement independent episodes. Valid
regenerative/mixing/block assumptions would need their own argument and
changed theorem. Do not equate this binomial calculation with conformal
calibration sample requirements.

Monitoring starts with fixed-version descriptive episode diagnostics.
Any future sequential alarm must declare its horizon/error spending or a
valid conditional/sequential assumption and measure false alarms/detection
delay. A CUSUM label or selected TRAIN ARL does not prove a guarantee under
unmodeled correlation, online RLS or drift.

### Identification and policy experiment

Existing inspected captures remain development data. Use chronological entire
episodes/groups, purge labels by actual resolution/availability, and freeze
calibration/support artifacts with their cost/runtime identities. Changes
selected after validation consume that validation and require fresh evidence.
Final trace C remains unopened and owner-controlled.

Before live comparative trials, separately authorize and qualify the marked
observer, client/ingress target, cost-aware HTTP actuator including fencing,
native timing/overhead and cancellation semantics. Current post-admission
pre-ADD probes cannot justify their own earlier admission and are not assumed
required for the gateway-first test.

Then identify supported up/down actions with recorded randomized assignment,
adequate history support and dwell/release variation; natural output and
light-load transfer need distinct data. An offer-count process with Fano.94
does not identify all burst/marked-token regimes. Stress arrivals are proposals,
not observations.

Primary policy comparison: randomized complete episodes within host/time
blocks, matched offered seeds where useful, against fixed, same-state
backlog and same-model one-decision policies. Reset/verify initial state or
define a treatment-history estimand; median/p90 residence is not a washout
bound. Switchbacks remain possible with an explicit carryover design, not
automatic attribution by current policy-at-arrival.

Give all policies equal observation/action support and tuning budget. Report
all-offered successful goodput, delays, rejection/failure/size strata, late
applications, fallback/unavailable fractions and CPU overhead. Inference uses
independent assignment units and a prespecified paired/randomization method,
effect margin and multiple-comparison hierarchy. No repeated peeking.

For resource planning only, if independent paired differences really had
sigma_d=1req/s and an owner-approved delta=.4req/s, the idealized normal
two-sided.05/80% calculation gives:

```text
n ~= (1.96+.84)^2 * 1^2 / .4^2 = 49 independent pairs.
```

This is an illustrative lower-detail planning scenario, not achieved power
or a permission request. Opus's correlated30s-window dispersion and three
sessions do not establish that sigma_d for independently reset300s episodes.
Actual unit variance, practical serving value, multiplicity and budget must
be frozen from suitable development evidence and owner decisions.

Default-off engineering may proceed without proving a99% safety statistic.
Live actuation and performance claims may not proceed by weakening their
evidence standard to fit a resource ceiling.

## 9. Remaining disagreements and decisive experiments

| Unresolved question | What would decide it |
|---|---|
| W_tok feedback versus richer state | Same-state/availability-controlled reduced model, age/phase and engine augmentation ablations; near-threshold calibration and policy choices, not AUC alone |
| Continuous slack versus discrete budgets | Exact event replay plus later separately supported randomized policies; include clustered releases and actual command delay. No assumed Poisson incidence promoted to measured loss |
| Closed-form old-cohort correction versus coherent lifecycle | Supported randomized later-work pulses; compare old and total eventual reward, not endogenous diagnostic flips as an upper bound |
| One decision versus multiple future controls | Identical10s scored-cohort window/tail/model/scenarios with2 versus8candidate paths initially, then frozen independent policy value |
| Nominal versus uncertainty-abstention policy | Firing/support/tail-unknown rates, action stability and repeated policy value. Broad intervals may correctly reveal insufficient evidence rather than prove all MPC useless |
| Tail length/remainder complexity | Fixed-cohort boundary tests and D_eval/T_resolve sensitivity; longer reservation opportunity cost and late failures retained |
| CPU feasibility | Source-bound retained-model benchmarks with max supported state/events, full cut-to-use age, prefix sharing and serving interference; toy measurements neither lower bounds nor acceptance |
| KV/resource safety | Actual conservative tokenizer/allocator/block/capacity/cancellation qualification or an honest lack of certificate; reservation arithmetic alone is insufficient |

A negative result against backlog feedback or fixed control is a legitimate
successful research outcome. No new GPU experiment is authorized here.

### Supplemental review of the Mayor's mathematical specification

I read the complete provisional `mpc-mathematical-specification.md`. I accept
its separation of exact reservation accounting from latent service, its
kernel-inspired model as a **challenger rather than native physics**, and its
smaller N=3/1s search as the first CPU configuration; Section5 is aligned
to that smaller search. The concrete correction is to separate the **scored
cohort window** from the **planned-control horizon**: scoring only old plus
H-window cohorts can miss a newly admitted long resident's opportunity cost
on later offers, and changing H changes the target population. Use the fixed
D_eval cohort window of Section4 for N=1/3/5 comparisons, or explicitly label
the different objectives; continue unscored interference after that window.
Also define whether unresolved-reward intervals select by lower bound,
nominal value or abstention, rather than calling unresolved both zero credit
and an unspecified expected value. For the kernel reference, specify eligible
decoder selection/sequence limits, guard total scheduled tokens<=B, skip
zero-work iterations to the next event, and expose unknown initial progress;
nonnegative coefficients alone do not close that event model. Finally the
guarded-path requirement is correct but needs the Section6 boot/epoch,
operation-ID/outcome and **both CAS and expiry** semantics, with negative
capability tests proving stale/expired/mismatched commands leave budget and
version unchanged. Merely sending extra fields to the legacy endpoint is not
capability checking. These are focused CPU-specification corrections, not
objections to default-off engineering or authorization for native actuation.

## 10. Verification and Round 3 barrier

Executed here: the Mayor's `checks()` from
`experiments/mpc-research/mayor_counterexamples.py` compared with
`results/mpc-research-cpu/mayor-counterexamples.json` after JSON normalization;
all8checks matched. Additional standard-library arithmetic produced:

```text
independent zero-failure upper95 n299=0.00996915; n2995=0.00099974
shared_bias=1.6; true_gap=2.0; predicted_gap=2.0
late command: CAS_only=True; CAS_and_expiry=False
cohort bounds: resolved=3; unresolved=2; lower=2; upper=4
```

The common-bias example rejects a proof of inevitable gate paralysis from
factual bias magnitude alone; it does not show real action-gap errors cancel.
The CAS example establishes the missing-expiry case, not an implementation.
No serving runtime/native benchmark or policy experiment was run.

Frozen own-report hashes recorded before this revision:

```text
gpt-round1.md 84a0eb0ad207ae4cb3d42133c54786f784f9da846c6d3a00bf1bf281d959be97
gpt-round2.md 092f7441fe0fae4fd5e2191ac62a23af7f515d36cb52f4355ebf57015a4c3bf4
```

**Readiness:** coherent specification for default-off CPU/replay/shadow
engineering, subject to executable falsification and measured budgets.
**Not ready/authorized:** live actuation, native HTTP/state/action qualification,
physical KV/latency safety, calibrated counterfactual decision guarantees,
cross-host generalization or policy-performance superiority.

**Round 3 barrier reached.** Only this new report was written for Round3;
Round1/2 remain unchanged. No product edits, commits, pushes, merges, live
commands, network/infrastructure changes, extra agents, paid resources or
final trace C access were performed. Mayor owns integration, tracking and
any subsequent authorization.
