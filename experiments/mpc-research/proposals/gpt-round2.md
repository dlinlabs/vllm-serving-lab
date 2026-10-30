# Round 2 adversarial review: gpt

Researcher: onyx / GPT-6.1 Sol (`gpt-6.1-sol`).
Source under discussion: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Scope: complete frozen Round 1 reports by opus, astra, gemini and gpt were
read. Round 1 files were not edited. This is a separate critique, not a vote,
implementation, native qualification or policy-performance result.

## 1. Findings and evidence boundary

**Main result:** retain Opus's backlog predictor as a serious low-complexity
challenger, but reject its deterministic slack bound, universal empirical
throughput ceiling and recursive-feasibility claim as written. Astra's cautious
plant/observation contract is useful, but its tail objective does not protect
old cohorts in the same way as new ones, its calibration procedure still needs
an observable score, and reconciliation needs a stronger termination condition.
Gemini's proposal needs mathematical and evidence corrections before its
rollout or safety claims can be implemented.

The Mayor's hash-verified reanalysis JSON was inspected, not independently
re-extracted from archives in this round. Its 12-run ranges agree with Opus:

| Model | AUC range | TTFT MAE range, seconds |
|---|---|---|
| R plus own prompt | 0.585827-0.774981 | 0.097756-0.358656 |
| W_tok plus own prompt | 0.959016-0.994851 | 0.033374-0.093152 |
| queue_causal | 0.966418-0.998080 | 0.028416-0.089091 |

This is strong **factual development** evidence for a useful representation,
not independent fresh-session qualification, calibrated threshold decisions,
or policy gain. I inspected the adjacent `opus-round1-analysis.py` in full and
executed its pure `drain` helper on a synthetic input. No peer Round 2 file was
read. Source reads were restricted to the already identified gateway/telemetry
paths and local research artifacts.

In the findings table, **B** means a mathematical/specification contradiction
that blocks the stated claim; **Q** means a substantial unresolved
qualification/identification question, not a demonstrated runtime failure.
These are scientific/design priorities, not exploit-severity labels.

| ID | Peer and location in frozen Round 1 | Finding | Status | Discriminating correction/test |
|---|---|---|---|---|
| O1 | opus Sections 2.5, 6.2 | Maximum observed fixed-policy throughput is not a universal admission-policy ceiling; mean completions/tick is not a hard release bound. | B | Delete physical-bound language; measure supported release tails and derive only bounds from validated physical constraints. |
| O2 | opus Section 6.2 | Absolute `C=R+s` admits far more than 700 excess prompt tokens after a clustered release; synchronous/atomic slack also cannot prevent releases later in the tick. | B | Exact event-order counterexample below; distinguish a statistical error budget from a bound and distinguish atomic relative update from an actual per-request predicate. |
| O3 | opus Sections 3.3-3.4 | `R_H<=C*+mean_cost` may be infeasible under every action, including hold, and does not guarantee a bumpless fallback. | B | No-release horizon with R64 and terminal limit34; soften/drop the terminal condition or prove reachability under declared bounded service. |
| O4 | opus Section 3.2 | The rollout reward omits old unresolved cohorts and eventual stream success, although the whole-run metric states both correctly. | B | First token then failure, and old-cohort externality tests; score complete-success probability jointly with on-time first content. |
| O5 | opus Sections 2.3-2.4, 4, 5.3 | Future-after-arrival regression and stationary arrivals do not identify weak intervention effects or establish myopic optimality; RLS exogeneity is assumed, not shown. | Q | Randomized later-work pulses; Bellman/opportunity-cost counterexample; sequential innovation tests and support checks. |
| O6 | opus Sections 5.1, 5.4, 8 F4 | Per-200-request binomial monitoring ignores shared service regimes and repeated testing; fixed-bin conformal coverage does not fix that. | B/Q | Cluster-level or valid sequential monitor; measure false alarms under the actual arrival/controller process. |
| O7 | opus Sections 2.2, 7.2; adjacent script `drain` | Helper misses already-fitting reservation at update time; switchback carryover is not bounded by a median/p90 plus a five-second exclusion. | B/Q | Fix diagnostic definition in a new artifact; retain frozen original. Use reset episodes or an explicit carryover estimand/design. |
| O8 | opus Sections 1.2, 1.5, 5.3 | Claimed observation/label contracts differ from the actual metrics and fitted client-scheduled TTFT target. | B/Q | Receipt-age versus admission-age parity; qualify causal client-label delivery or model gateway proxy plus ingress. |
| A1 | astra Section 5, reward and risk | Expected value includes old cohorts, but CVaR excludes old-cohort loss and their correlation with new outcomes. | Q, concrete consequence | Ten-scenario reversal below; either state this restricted risk preference or apply tail risk to the complete action-sensitive reward. |
| A2 | astra Section 4, release-pending observation state | Separate source operations do not by themselves imply an observable race in this single cooperative event loop. | Q, source correction | Check suspension points; same-loop lock/cleanup test below. Add pending-release state only for a genuinely different observer boundary. |
| A3 | astra Section 7, calibration steps 3-4 | A maximum over all candidate counterfactual errors is not observable in a single executed episode. | Q | Specify the actual measured score, randomization unit and target law; do not instantiate an unobservable maximum as if labels existed. |
| A4 | astra Section 5, unknown actuation | Reading the old budget/version after timeout is not proof that the old command cannot apply later. The same ambiguity remains in gpt Round 1. | Q, operational blocker | Retain pending-write uncertainty until positive application/termination evidence; a current-state read alone is not a write fence. |
| A5 | astra Section 5, cadence/search | 1,280 event rollouts in 20 ms requires about 15.6 microseconds/rollout before filtering/serialization; no performance measurement exists. | Q | Benchmark at actual maximum active population, tails and event counts; measure full end-to-end age and serving interference, not a vectorized toy. |
| G1 | gemini Section 7.2 | Proposed drain law compares tokens/step to tokens/second and does not produce reservation-cost release. Mean cost/count/oldest age do not close this model. | B | Unit audit and equal-aggregate/different-residual counterexamples; generate cost release from terminal request identities. |
| G2 | gemini Sections 1.2, 3.2, 5.1 | KV state is excluded, yet a 0.001 KV chance constraint is imposed without a qualified KV predictor; max_tokens reservation does not prevent OOM. | B | Remove qualification/safety language; obtain a valid resource observation/model or use a scoped diagnostic only. |
| G3 | gemini Sections 3.3, 7.3, 8 | C24 is not a qualified safe fallback; the conditional fallback can retain C48, and timeout does not cancel the original update. | B | Explicit action-uncertain state and frozen empirically qualified fallback; no promise of successful reduction without transport evidence. |
| G4 | gemini Sections 7.3, 9.1, 10 | Claimed <4 ms benchmark, native 289-test acceptance, and satisfied <20 ms deadline lack the stated evidence. | B, evidence correction | Supply source-bound executable measurement or relabel targets; preserve the audit's local/mock versus native distinction. |
| G5 | gemini Section 5.2, test_conformal_coverage | Chronological residual quantile lacks a coverage theorem under policy shift; Mahalanobis ellipsoid is not a convex hull; a constant 0.88 test is not a 90% guarantee. | B | Exact conformal rank/assumptions, selected-policy calibration, proper geometric support definition and sample-size-aware inference. |
| G6 | gemini Sections 3, 6, 7 | Actions/horizon change between sections; interpolation to C24 is not support; `2 sigma` is not a paired superiority/equivalence procedure. | B/Q | Freeze one executable specification and a paired episode-level estimand with effect margin and multiplicity. |
| G7 | gemini Sections 2.2, 3.1, 3.3 | Median drain is not a complete-tail horizon; zero new-arrival-cohort reward during rejection is not zero ongoing completed goodput; old-cohort reward/tail missing. | B | Long-lived survivor crossing H, continuing old hits, first-token failure and terminal-continuation cases. |

## 2. Opus: useful factual result, overextended control conclusions

### O1-O2: two empirical quantities have been promoted to upper bounds

Little's-law identity `E[R]=lambda_admitted*E[c*S]` can hold for a stable marked
admission process. It is not a policy-independent capacity estimate: both
`E[c*S]` and the admitted mix change with the policy. From

```text
goodput(policy) <= admission_rate(policy)
```

one cannot deduce

```text
goodput(any policy) <= max_observed admission_rate(fixed policies).
```

A simple countermodel is a workload with ample short jobs that sustain
10 successful requests/s if admitted alone, while a first-fit fixed-cap policy
mix admits enough long residents to reduce its throughput to 7.4/s. This does
not prove 10/s is attainable on the current GPU/trace. It demonstrates that the
observed 7.4/s is not the claimed mathematical upper bound. An offered-rate bound
or a service-work bound with validated per-request costs would be different.

The more immediate deployment hazard is the 700-token slack bound. With
fixed m=128, define

```text
b = 0.259 + 0.0052083333*128 = 0.9256666624 cost
c(p) = b + 0.000758*p.
```

At a tick with `W_tok=K`, the Opus rule leaves slack b. Three decoders with
prompt estimate1153 each terminate within the next50ms. They release
`3*c(1153)=5.3989219872 cost`. The stale absolute cap now leaves
`b+5.3989219872` headroom. A new7000-token prompt costs `6.2316666624`,
fits, and can add7000 awaiting tokens even though the tick's remaining
prefill allowance was zero. Prompt7000 plus output128 is within context8192.
Assume the pending prefill has not advanced before the new arrival; the release
can be a gateway-observed clustered completion of already-decoding requests.

Opus's inequality retaining **actual** released cost is a useful accounting
relation under its fixed-m assumptions. Replacing released cost with
`mean completions/tick * mean cost` does not yield a deterministic upper bound.
Even stationary Poisson arrivals have unbounded counts over a nonzero interval;
correlated terminal observation makes clustered releases especially natural.
The admissible-population/cost bound limits the absolute worst case, but is
much larger than the mean-based700-token estimate.

Two further qualifications:

- If `K-W<0`, use `(K-W)_+` consistently; the unclipped bound can become
  negative even when no request is admitted.
- A new atomic gateway-relative-slack endpoint would remove the read-to-apply
  R race, but would **not** prevent subsequent within-tick release from
  reopening headroom. A true per-request backlog/deadline predicate is a
  different actuator and can address that issue. Those two changes should
  not be described as equivalent repairs.

**Experiment:** shared-release stress over observed active populations, with
state-feedback tick, observation lag and command lag varied independently.
Measure excess admitted prompt tokens and action-margin errors, including
tail/maximum and all failed updates. Compare existing absolute budget,
atomic-relative update and actual per-request gate as separate contracts.
No inference from their shared endpoint family should imply identical support.

### O3: hold is an admissible input, not a feasible terminal trajectory

Take R0=64, H=3s, fixed fallback C*=32 and mean request cost2. Suppose all
currently reserved requests have residual lifetimes greater than3s, and no
new request is accepted. For every budget plan:

```text
R_H = 64 > C* + mean_cost = 34.
```

No action can release those reservations early. Holding C* is legal at the
actuator but violates the terminal condition. Hence the proposed optimization
is not always feasible as written. This is compatible with finite completion
later; no infinite stall is needed. A state reached by the slack controller at
C64 is inside the proposed hierarchical operating range.

Moreover `R_H<=C*+mean_cost` does not even imply `R_H<=C*`, let alone admission
of the next actual request. R_H=33 and C*=32 still blocks every positive-cost
offer until sufficient release. There is no general bumplessness theorem here.

Repair: make terminal burden a soft objective/value interval with explicit
fallback, or restrict initial/reachable states and prove a backup-controlled
terminal set. A hard terminal condition needs assumptions on service and
residual work; an empirical mean/p90 is not a reachability proof.

### O4-O5: predictive sufficiency does not give the right decision objective

Opus correctly defines whole-run eventual complete-success goodput. Its
Section3.2 optimization then counts only future arrival cohorts times
on-time-hit probability. Counting expected reward at arrival is not itself
wrong: it is valid if the conditional expected reward is the full
`P(on-time first content AND eventual successful complete stream | plan)`.
The written expression omits the second part and omits old unresolved cohorts.
The correction is a joint outcome expectation, not simply delaying every reward
numerically until completion.

The stationarity argument also needs the average-reward Bellman equation:

```text
g + h(x) = max_a { immediate_reward(x,a) + E[h(X_next)|x,a] }.
```

Stationarity can make the optimal policy stationary in state. It does not
eliminate the differential value h or make greedy feasibility optimal.
Example: low/high caps16/32, empty initial reservation. A cost20 long resident
arrives now and can make an on-time first token but holds its reservation
through three later cost13 short arrivals. High cap admits the long resident
and cannot fit any cost13 request even after later reductions; low cap rejects
it and can admit/finish the three short requests sequentially. Every admitted
request's TTFT can be a fixed .1s, with zero decode-interference coefficient.
Reward is1 versus3. Reservation opportunity cost alone creates nonmyopic value.
Opus already identifies residence as a coupling channel; the counterexample
shows why the small fitted TTFT decode coefficient does not bound that value.

F1's "tokens admitted before observed first content" regressor uses a
future-outcome-defined window. Its coefficient is not a randomized treatment
effect. Future admissions depend on shared capacity, prior control and how long
the target remains unresolved; later prompt count also omits possible effects
of future decoders, output mix and changed service regime. Low incremental MAE
under existing policies cannot rule out large intervention effects outside
their visited states. The development relationship remains useful, but the
claimed direction of endogeneity bias is not identified without a structural
argument.

**Experiment:** at matched decision-history strata, randomly add allowed later
work pulses and observe effects on an already-admitted target plus subsequent
cohorts. Separately test reservation opportunity cost by changing output maxima
and remaining residence with similar first-token workload. Use recorded
propensities and whole-episode outcome accounting; do not use F1 as action
support for a new continuous20Hz policy.

High AUC does not validate the .5s decision boundary: adding .2s to every
prediction preserves rankings/AUC but can reverse many accept-feasible labels.
Require near-boundary calibration, optimistic-error quantiles, error under the
new admitted mix, and eventual-success prediction, not only AUC>=.95.

RLS consistency needs `E[epsilon_i | regressors_i, admitted_i]=0` or another
valid identification argument, sufficient excitation, and appropriately
handled dependent noise. Admission not explicitly reading the current residual
does not prove this: hidden service state can influence both the measured
backlog/history and the residual. With policy-restricted W_tok and missing
prompt progress, this assumption is substantial. Dither is not a general cure;
test the information matrix and conditional innovations, and distinguish
unbiased reduced-form prediction from physical parameter identification.

### O6-O8: sequential monitoring, carryover and live-label contracts

If a shared regime Z makes all200 requests fail together with probability.05,
each request has marginal violation probability.05, yet an iid binomial test
assigns an all200 failure batch probability `.05**200 = 6.223e-261`.
The actual all-fail probability is .05. This is a direct counterexample to
calibrating that request-level monitor as an independent binomial test.

Even genuinely independent level.01 nonoverlapping tests have
`1-.99**360 = .973167` probability of at least one false alarm in360 tests.
The report does not specify enough to determine the exact deployed false-alarm
rate, but fixed-window nominal coverage is not an anytime theorem. Use a
prespecified cluster-aware sequential procedure or explicitly simulate and
report false alarms under stated dependence/drift assumptions. RLS adaptation
changes residuals again and requires its own monitored epoch/calibration rule.

The adjacent script has a separate diagnostic defect:

```text
rows: accept at -1s, cost8, terminal at10s
budgets: C32 at -2s; C16 at0s
```

Actual R at reduction is8, so it already fits: first-fit-under-cap delay0.
Calling the inspected `drain` function returns10s because it only tests
`t>update_time` after future ledger events. This does not invalidate the
reproduced AUC/MAE. It means its drain statistic needs an at-update state check
and explicit tie/order definition before being used as an actuation-tail
estimate. "R<=cap" and "R+actual_next_cost<=cap" are also distinct estimands.
Preserve the frozen original and publish a corrected diagnostic separately.

For switchbacks, residual lifetime p90 is not a carryover bound. A long request
admitted at second29 can reserve through second60 or later; dropping the first
five seconds of a30s following block does not remove that interference.
Randomized switchbacks can validly estimate effects of the randomized treatment
sequence or support a Fisher sharp-global-null test. They do not automatically
estimate the value difference of two policies run continuously from reset.
Specify treatment-history potential outcomes, washout assumptions, inference
unit and estimand. Reset episodes within host/time blocks also control session
effects; whole-episode randomization is not inherently confounded.

Finally the live observer needs correction. `telemetry.py:345-351` exposes
oldest-awaiting **receipt** age, not time since acceptance. The current active
map does not retain admission timestamps or marked token totals. Opus's
offline reconstruction uses admission age; extending metrics with W_tok alone
does not make the entire fitted state identical to live metrics.
The analysis fits client-scheduled TTFT, whereas Section1.5's gateway
first-content/receipt label is a different target and arrives at a different
time. Either qualify asynchronous client labels and their clock/join latency,
or explicitly model a gateway proxy plus ingress/client transport. Accurate
offline causal features do not imply the desired live label is already present.

## 3. Astra: substantive gaps inside an otherwise cautious specification

### A1: old-cohort protection disappears from the tail term

Astra's expected loss correctly subtracts old-cohort value, but its CVaR term
contains only new-cohort loss. Consider10 equally weighted scenarios,
beta=.9 and lambda_tail=1, with other penalties equal:

```text
plan A: new loss0 in every scenario; old loss10 in one scenario,0 otherwise
plan B: new loss2 in every scenario; old loss0 always
```

Relative old loss is `-Delta V_old`, so the written expected terms are1 and2.
The written new-only CVaRs are0 and2. Total objectives are A=1,B=4:
choose A. Tail risk over complete action-sensitive cohort loss gives CVaRs10
and2, yielding A=11,B=4: choose B.

This is not an algebra error in CVaR. It is a consequential risk-preference
scope choice, inconsistent with interpreting the tail term as protection of
all successful-SLO reward. If old requests are intentionally lower priority
in the risk term, state and authorize that change. Otherwise use total
old-plus-new loss or same-scenario total regret, preserving correlation.
Adding a separate old-failure soft constraint only resolves this if it is
specified and actually enforced in the relevant scenarios.

**Experiment:** construct rare shared failure/stall regimes with many old
on-time streams. Compare new-only CVaR, total-reward CVaR and risk-neutral
control, with all-offered and already-admitted failure diagnostics. A tail term
should be checked on the actual protected population, not just mean reward.

### A2: do not add a release-pending state without an observable interleaving

The source does call `finish_request` before `telemetry.terminal`
(`gateway.py:438-463`), but in cost-aware mode there is no suspension *after*
reservation mutation/release and *before* terminal bookkeeping. Awaiting a
coroutine that runs to completion without a suspension does not yield control.
`snapshot_metrics` samples gateway state and `telemetry.observed_state` under
the same asyncio lock on the same event loop.

I executed a deterministic asyncio test with lock contention forcing cleanup
to suspend before the release. Its trace was:

```text
release
terminal
snapshot(R=0, active=0)
```

No competing same-loop metrics task can observe the intermediate boundary in
that execution pattern. This is a model of the inspected suspension structure,
not a product regression test or proof for arbitrary observer threads.

A release-pending state could be needed for separately delivered event streams,
an observer in another thread/process, changed awaits/hooks, or physical
backend work persisting after gateway cleanup. Those are different issues.
The physical backend/gateway divergence warning is valid and must stay.
The narrower source claim should be corrected so unnecessary state/complexity
is not introduced on the assumption that two Python source statements always
permit an asyncio task switch. This also tightens gpt's exact-snapshot wording.

### A3: name the observable calibration score

Astra correctly warns that pointwise intervals do not survive best-action
selection and that factual calibration cannot label unseen counterfactuals.
However an episode-level maximum error over every candidate action is not a
directly measurable score when that episode executes only one treatment path.
Randomizing one action does not reveal the others' same-state outcomes.

A later implementer therefore cannot simply compute the proposed action-family
maximum. Specify one of:

- A frozen-policy performance/trajectory score calibrated on independent
  episodes generated by that complete policy.
- Prespecified action-specific marginal scores under a randomized exploration
  distribution, with sufficient support and a justified simultaneous coverage
  correction. This is not state-conditional counterfactual coverage.
- Confidence bounds for *average policy-value differences* from randomized
  episodes, explicitly different from per-origin decision-gap intervals.
- A declared model-based heuristic uncertainty score with no distribution-free
  counterfactual guarantee.

An unobserved maximum can be a theoretical target, but not a logged calibration
artifact. This same limitation applies to gpt Round1 Section5.3; its conditional
phrasing does not supply a realizable calibration dataset.

**Experiment:** give the calibration module a dataset with one randomly selected
action per origin and no paired counterfactual labels. Require it to state
which score/coverage population it can estimate and to reject the unavailable
family maximum rather than fabricate it from simulator predictions.

### A4: stale-version reads are not fences for a pending write

The proposal rightly prohibits blind retries. The remaining ambiguity is its
exit criterion from reconciliation:

```text
t=0ms:    send original C32; network/request handling delays it
t=20ms:   client times out
t=30ms:   fresh metrics read returns old C16/version0
t=40ms:   fallback C16 applies/version1
t=100ms:  original C32 finally applies/version2
```

A fresh read at30ms does not prove that the original command has terminated.
The fallback can be overwritten later. This does not rely on two authorized
writers; two still-live requests from one controller suffice.

Astra and gpt are safer than the automatic fallback recipes because they say
one outstanding command and suspension during uncertainty. Round3 must define
"outstanding" until positive application or terminal failure evidence, not
until client timeout or one reassuring metrics read. A changed version/budget
can positively establish the known sole pending application under a qualified
single-writer model; an unchanged version cannot establish nonapplication
forever. Without bounded delivery/server expiry or definitive cancellation,
finite-time recovery may be impossible. Remaining degraded is an honest
outcome; claiming an always-available transmitted fallback is not.

Server request IDs, expiry and compare-and-set would be new gateway contracts,
requiring separate review. This critique does not authorize those product edits.

### A5: the compute target needs a maximum-state benchmark, not only a cutoff

40plans times32scenarios is1280 rollouts.20ms permits15.625 microseconds per
rollout before observation, scoring, tail continuation and actuation. For
100active requests and20control slots, even a naive per-slot hazard evaluation
is2.56million request-slot evaluations/cycle, before events/tails.

This is not a proof of infeasibility: prefix reuse, compiled/vectorized kernels,
aggregation and smaller active populations can change it. Astra explicitly
labels the numbers targets, which is correct. The missing experiment must
benchmark the **retained state representation**, competing risks, rare tail
scenarios and evidence logging together. Otherwise it is possible to achieve
the target only by silently replacing the proposed model with a different one.
Compare fewer plans, fewer scenarios and simplified states on both action
stability and latency. Any-time search must never select on only the early
fortunate scenarios of a plan; compare plans on matched completed scenario sets
or fall back.

## 4. Gemini: dimensional, statistical and evidence blockers

### G1: token capacity is not reservation-cost release

The proposed law

```text
mu_drain = min(B, N_flight / mean_token_time)
```

compares B in tokens/step with approximately tokens/second. Even after adding
a steps/second conversion, tokens processed do not release cost linearly.
Whole reservation is released at request terminal. One long decoder can consume
many tokens this second with zero released cost; many nearly complete decoders
can release most reservation with little remaining token work.

The proposed aggregate state omits phase/remaining-work composition needed to
distinguish those cases. Equal R, count and oldest age can hide different
residual distributions. A successful old age can mean nearly complete or a
long-lived stalled straggler; a memoryless model is a testable alternative,
not a justification for deterministic drain.

Repair units first:

```text
cost_out(interval) = sum(c_i for terminal request identities in interval).
```

Generate those terminals using a qualified lifecycle/survival or token-progress
model. Keep token work and reservation cost as distinct state/dynamics layers.
The current formula cannot be accepted as "analytical batch-token drain curves"
fitted from service evidence without a defined outcome, units and artifact.

### G2-G3: missing observations and an unqualified fallback cannot confer safety

Gemini excludes causal KV state and provides no KV transition/measurement model,
then constrains `P(KV>=.90)<=.001`. A probability over an unmodeled, unobserved
quantity is not an enforceable constraint. Observability is not strictly
necessary if a valid conservative latent-resource model exists, but none is
specified or qualified here. The cost estimator does not measure allocated
KV blocks; reserving requested max_tokens in *gateway cost units* does not
prevent OOM.

With zero violations in8 independent trials, the one-sided95% binomial upper
violation probability is `.312344`, not .001. Even32 gives `.089368`.
To bound it below .001 with zero failures takes2995 independent units under
that Bernoulli model. Simulated scenarios are not real independent deployment
trials anyway; their precision is conditional on the model. The number is a
useful scale check, not a proposed native resource request.

C24 has no demonstrated safe/native-qualified status in the cited evidence;
interpolating between C16 and C32 is not an action-support or latency proof.
The conditional fallback also returns the previous budget if R>24: starting
at C48,R30 leaves C48, which violates the proposed high-risk tightening
`u<=24`. It avoids no actuator illegality: a lower cap with R>cap is explicitly
legal. The rule cannot be described as a generally safe reduction.

Finally watchdog timeout does not cancel a running CPU worker, cure a blocked
event loop, or cancel an already-sent budget request. If the evaluator is
synchronous on the gateway loop, its own watchdog may not run until the
computation returns. Use an independently bounded/off-loop computation and
explicit write-uncertainty state; report last-applied-budget persistence.

### G4: replace unsupported measurements with targets

The report says the proposed NumPy evaluator is "<4.0ms (benchmarked on
Python3.12/3.14)" and concludes its search satisfies20ms. It supplies no
executable implementation, benchmark inputs, command, repetitions, timing
distribution or source-bound artifact. The three named proposed
`prefill_lab/mpc/` module paths do not exist in Jasper's worktree at inspection.
That absence does not prove no private toy timing ever occurred; it does mean
this is not reproducible evidence for the specified model/controller.

The shared audit identifies289passed/1skipped as **selected local CPU checks**,
not native acceptance. Later shared-audit entries separately report the full
Python3.12 CPU/mock baseline with the installed-vLLM schema still skipped.
None qualifies GPU/HTTP/MPC performance. Correct Fact4 rather than upgrading
local selected tests to a native pass.

Likewise <.5ms state-read delay,50-200ms engine scrape delay and10-30completions/s
need artifacts or proposal labels. The reproduced development admission rates
are about4-7/s in the cited fixed-cap region; offered25.5/s does not justify
using10-30 completion labels per second for inference planning.

### G5: three different statistical/geometric statements are conflated

1. A chronological calibration split is a leakage control, not exchangeability.
   The displayed empirical quantile needs the finite-sample corrected rank,
   an explicit score/coverage unit, and exchangeability or another justified
   theorem. State/action selection and adaptation change the target law.
2. With small n, passing `coverage>=.88` on a validation set does not establish
   a90% guarantee. Even95% *observed* coverage is not a population lower bound
   without sample size, dependence and confidence calculation. One unit test
   can test score arithmetic, not establish real-world uncertainty validity.
3. Convex-hull membership is not a Mahalanobis-distance threshold. Calibration
   points `(-1,-1),(-1,1),(1,-1),(1,1)` have population covariance I. Query(2,0)
   is outside their square hull but squared Mahalanobis distance4 is below
   the chi-square(2)99% threshold9.21034. The proposed detector accepts it.
   Even for Gaussian data the chi-square comparison is with squared distance;
   estimated mean/covariance and non-Gaussian serving states add assumptions.

These are fixable definitions, but they must be fixed before naming the result
a conformal state envelope or calibrated OOD shield.

### G6-G7: freeze a coherent problem before optimizing it

The proposal changes `{16,24,32,48}`, horizon5, into `{16,24,32}`, horizon4,
while also specifying bounds through64 and a different cap alphabet in its
diagram. For the final three-action/four-step family, its15plan count is in
fact correct **if** at most one adjacent switch counts only changes within
the proposed plan:

```text
3 constant plans + 4 directed adjacent pairs * 3 switch positions = 15.
```

If changing from the actually applied prior cap to the first planned action
also consumes the one-switch allowance, the set differs. Freeze that definition,
fallback membership and support before benchmarking. No full tree or
scenario-specific future controls should silently replace this open-loop family.

`goodput_MPC-goodput_fixed <=2*sqrt(replicate variance)` is not a superiority
or equivalence test. It ignores paired covariance, n, independent units and
the practical effect margin; unlike a standard error, it does not shrink with
more independent replicates. The frozen decision needs confidence intervals or
randomization inference matching the assignment, effect delta and multiple
comparators. "Matched offered prefixes" do not create identical physical
states or paired counterfactual labels.

The model also lacks a specified old-cohort/terminal-success continuation.
Scheduled TTFT needs queue/scheduler and client-observation components; writing
`prefill_exec+transport` must not silently omit engine waiting.
Rejections lower the numerator under a fixed offered-time denominator; there
is no special admitted-only denominator to adjust.

A five-second horizon cannot "capture the complete drain transient" from a
median2-3.5s. Construct a surviving reservation lasting6s: at5s it still
exists. Unbounded stalls make a universal bound impossible. Offers rejected
through a drain earn zero for their **arrival cohorts**, but already-admitted
requests can complete successful SLO streams throughout that interval.
Do not treat their ongoing completed goodput as zero or as an action-independent
constant without checking future-success and contention effects.

Generic Fano>>1 burstiness also does not describe the inspected development
trace, whose Opus reanalysis finds about.94 in1s bins. A modulated model may be
a useful stress/transfer hypothesis, but it is not a measurement of this trace.
Similarly fixed C16 need not "starve throughput during light traffic" if every
offered request fits; low offered load alone cannot establish that comparison.

## 5. Exact executed arithmetic command and result

The following exact command ran successfully in this worktree using only
Python's standard library. It is a deterministic model/specification
counterexample, not a native serving experiment:

```bash
python3 - <<'PY'
import math
b = .259 + .0052083333 * 128
beta = .000758
released = 3 * (b + beta * 1153)
print('slack_release_cost=%.6f; prompt7000_fits=%s' % (released, b + beta*7000 <= b + released))
print('terminal: R_H=64 > Cstar+mean_cost=34; hold_is_not_feasible')
A_new, A_old = [0]*10, [10]+[0]*9
B_new, B_old = [2]*10, [0]*10
for full in (False, True):
    scores = []
    for new, old in ((A_new,A_old),(B_new,B_old)):
        total = [a+b for a,b in zip(new,old)]
        scores.append(sum(total)/10 + max(total if full else new))
    print('full_cohort_tail=%s; scores_A_B=%s' % (full, scores))
print('zero_of_8_upper95=%.6f; n_for_0.001=%d' % (1-.05**(1/8), math.ceil(math.log(.05)/math.log(.999))))
events = [(100,32,'late_original'),(40,16,'fallback')]
cap = 16
for t,cap,label in sorted(events):
    print('apply_at_ms=%d; cap=%d; command=%s' % (t,cap,label))
print('after_timeout_and_fallback_final_cap=%d' % cap)
PY
```

Exact stdout:

```text
slack_release_cost=5.398922; prompt7000_fits=True
terminal: R_H=64 > Cstar+mean_cost=34; hold_is_not_feasible
full_cohort_tail=False; scores_A_B=[1.0, 4.0]
full_cohort_tail=True; scores_A_B=[11.0, 4.0]
zero_of_8_upper95=0.312344; n_for_0.001=2995
apply_at_ms=40; cap=16; command=fallback
apply_at_ms=100; cap=32; command=late_original
after_timeout_and_fallback_final_cap=32
```

Additional executed checks, described above, returned:

| Check | Result |
|---|---|
| Peer `drain` helper on the already-fitting synthetic reduction | `reserved_fits_after_s=10.0`; true already-fitting delay0 |
| Cooperative same-loop cleanup under lock contention | `['release','terminal',['snapshot',0,0]]` |
| Shared-regime versus iid binomial all-fail probability | .05 versus6.223015277861211e-261 |
|360 independent nominal.01 alarms | Probability of any alarm .973167 |
| Zero failures in32 independent trials | One-sided95% upper violation rate .089368 |
| Four-step adjacent at-most-one-switch plans |15, excluding transition from the prior applied cap |
| Mahalanobis versus convex-hull counterexample | Query outside hull; squared distance4<threshold9.21034 |
| Conditional Gemini fallback at R30,previous C48 | Returns48 despite high-risk budget limit24 |

## 6. Changes I would require of my own Round 3 position

These are revisions to pursue in a new Round3 artifact, not edits to frozen
gpt Round1:

1. Put Opus's W_tok/own-prompt reduced form ahead of a rich hazard model as an
   explicit low-cost challenger. Its factual gain deserves a fresh-session
   causal-observation test, even though its policy claims are unproved.
2. Separate scoring horizon from future control freedom in the myopic
   comparison: same five-second scoring/continuation with one current action
   then fallback versus five planned actions, plus a genuinely one-second
   reward-horizon arm. Otherwise horizon/control/terminal changes are confounded.
3. Define command-uncertain recovery by positive application/termination
   evidence; do not let an unchanged fresh metrics version silently fence a
   pending write. Accept indefinitely degraded last-applied-cap operation if
   no termination bound exists.
4. Replace any implied observable "maximum candidate counterfactual error"
   calibration with a measurable frozen-policy or randomized-action score and
   its exact population. Keep per-origin action-gap intervals explicitly
   heuristic until justified by a valid identification/calibration design.
5. Clarify that snapshot-ledger coherence is a qualified same-loop/event-feed
   contract; separate Python statement order from true suspension points and
   from backend work that can outlive gateway reservation.
6. Require shared old-plus-new reward scope for tail terms, with joint eventual
   success and no cutoff-based release optimism. Compare all risk choices to
   risk-neutral control before accepting their complexity.

No majority view settles whether backlog state feedback, a one-step controller
or longer MPC wins. The discriminating sequence is: qualify causal marked
gateway state and labels; benchmark bounded planners on real representation
sizes; identify supported action effects; freeze strong simple baselines;
then compare full policies in independently assigned episodes with valid
carryover accounting and uncertainty. All native/GPU/HTTP/action/policy
acceptance claims remain false for this research round.

**Round 2 barrier reached.** Only this new report was written for Round2.
No product edits, Round1 edits, final C access, GPU/paid compute, commits,
pushes, merges, infrastructure repair or network operations were performed.
Context is retained for Round3.
