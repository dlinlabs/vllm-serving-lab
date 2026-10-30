# Astra Round 2: adversarial review of Opus, Gemini and GPT

Assignment `pr-mpc-astra`; reviewer GPT-6 Astra (`gpt-6-astra`).
Product source freeze: `8e92371f342be98a50682bf4e2ddbb014665498b`.
All three complete peer Round 1 reports were read after the Mayor released the
barrier. Round 1 files were not edited. This is an offline review, not a vote,
new policy experiment, performance acceptance or authorization to actuate.

## 1. Findings and evidence standard

**Main result:** Opus contributes substantial reproducible factual evidence for
gateway backlog features, but its slack safety bound, optimality/ceiling
deductions and terminal-feasibility claim do not follow. Gemini's proposed
transition equation has incompatible units, its KV constraint has no
identified observation/model basis, and its claimed safe fallback and measured
runtime are unsupported. GPT preserves the key scientific distinctions, but
needs implementable contracts for unresolved writes, terminal value,
counterfactual calibration and backend work after reservation release.

These are not equally severe or equally certain findings. In the table:
**error** means an equation/claim is contradicted by source or a counterexample;
**unsupported** means evidence does not establish it; **specification gap**
means a necessary implementation condition is acknowledged or implied but not
specified sufficiently. A gap is not presented as proof the architecture fails.

References O, G and S denote `opus-round1.md`, `gemini-round1.md` and
`gpt-round1.md`, respectively, at the peer paths supplied by the Mayor.
Line numbers refer to those frozen files. Source references are relative to
this worktree unless otherwise stated.

| ID | Peer / location | Classification and consequence | Required repair or discriminating experiment |
|---|---|---|---|
| O1 | O:490-515, slack bound | **Error, safety-relevant:** expected releases per tick are called an upper bound; correlated releases permit much more than 700 excess prompt tokens | Keep a conditional bound using actual/bounded released cost; stress clustered releases, stale R and maximum marks |
| O2 | O:281-315, terminal/fallback | **Error:** `R_H<=C*+k_bar` is incompatible with the claim that holding C* is always feasible; it also does not imply immediately open admission | Treat terminal burden as soft/uncertain, or prove a reachable invariant set under explicit release bounds |
| O3 | O:223-251,363-390 | **Unsupported:** observed 7.4 req/s is not an admission-policy ceiling; stationary policy does not imply myopic feasibility is optimal or that only decode coupling gives horizon value | Separate descriptive throughput maximum from a policy-uniform bound; test opportunity cost even under strict FCFS |
| O4 | O:270-279,149-159,428-443 | **Error/gap:** stage reward omits eventual success and old cohorts; proposed online TTFT label differs from the reanalysis target | Add complete-success/cohort continuation and a qualified client/ingress model; replay first-hit-then-failure |
| O5 | O:396-413,428-443 | **Unsupported:** repeated binomial checks lack sequential/dependence control; accepted-state sufficiency and dither do not establish unbiased RLS | Define score units, monitoring horizon/error spending, state sufficiency and excitation rank; validate adaptation separately |
| O6 | O:537-616 and analysis script:179-198 | **Error/gap:** switchback estimand/power and drain statistic need qualification; already-fit drain state is assigned a positive time by the script | Specify treatment-history estimand/washout, independent blocks and a common gain criterion; correct statistic in a later unfrozen artifact |
| G1 | G:235-255 | **Error, model-blocking:** `min(B,N/t_token)` combines tokens/step and tokens/s and is then used for reservation drainage | Generate request terminal events and release their whole costs; independently model token work and wall time |
| G2 | G:22-33,97-123,144-151 | **Unsupported, safety-relevant:** excluded KV state supports no claimed 0.001 chance bound; max_tokens-based weighted reservation does not prevent OOM | Remove the chance claim or identify a qualified resource model with uncertainty; native allocator remains separate |
| G3 | G:108-123,257-274 | **Error/unsupported:** C24 is not shown qualified/safe; conditional hold can remain at the overloaded cap forever; timeout does not ensure fallback applies | Qualify the baseline and define degraded/unknown-action states; do not call hold a drain policy |
| G4 | G:84-95,235-255 | **Error/gap:** switching penalty is repeated inside the outer sum; TTFT decomposition omits engine queue explicitly; scoring loses old cohorts/success in the rollout | One dimensioned objective and complete cohort scorer shared by all variants |
| G5 | G:153-185,187-198 | **Error/gap:** ordinary empirical quantiles, Mahalanobis/convex-hull equivalence, fixed-cap calibration under switch policy, and the `2 sigma` decision rule do not establish claimed validity | Correct finite-sample quantile and OOD geometry; freeze/calibrate the selected policy and use episode-level effect intervals |
| G6 | G:13-33,108-140,257-274,331-358 | **Unsupported evidence:** <0.5 ms state delay, benchmarked <4 ms rollout, qualified C24, 10-30 completions/s and full native CPU acceptance are not established | Relabel targets as proposals and supply exact artifacts for any measured claim; reconcile H/action/interface versions |
| S1 | S:186-200,475-486,555-597 | **Specification gap with local inconsistency:** last acknowledged budget is not always actual budget; one GET cannot fence a still-pending timed-out write | Distinguish observed action from unresolved possibility; specify resumption only after definitive completion/fencing |
| S2 | S:402-530 | **Specification gap:** terminal V is not defined enough to distinguish already-counted outcomes from future opportunity cost | Publish an explicit cohort partition and relative continuation-value equation; test horizon expansion invariance |
| S3 | S:354-382,653-724 | **Specification gap:** per-episode maximum scores over all plans require unavailable counterfactual outcomes | Choose a genuinely observable calibration target/design, or retain uncertainty as a heuristic with no simultaneous action guarantee |
| S4 | S:186-229,533-551,784-817 | **Specification gap:** exact gateway release can coexist with ongoing backend work; "single bounded worker" is not a cancellation/overhead contract | Add orphaned-backend-work/invalid-scope handling and a killable or cooperatively bounded planner outside serving locks |

GPT explicitly disclaims deterministic SLO guarantees, unqualified KV safety,
extreme-tail calibration and guaranteed fallback on transport failure. Those
are strengths of its specification, not claims this review attributes to it.
Likewise, Opus explicitly marks its performance analysis as development-only;
the verified AUC should not be dismissed merely because its subsequent
control deductions are too strong.

## 2. What the new Opus evidence does establish

I read the complete adjacent `opus-round1-analysis.py` and summarized the
Mayor's existing
`/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/results/mpc-research-cpu/development-backlog-reanalysis.json`.
The Mayor reports hash-verified archive reconstruction. I did **not** rerun
that archive verification or the full fitting procedure; my command below
independently reads the resulting numerical artifact and computes summaries.

| Variant | Held-run AUC min / median / max | TTFT MAE min / max, seconds |
|---|---|---|
| Cost + own prompt | 0.585827 / 0.685161 / 0.774981 | 0.097756 / 0.358656 |
| Backlog tokens + own prompt | 0.959016 / 0.976295 / 0.994851 | 0.033374 / 0.093152 |
| Additional causal gateway state | 0.966418 / 0.981449 / 0.998080 | 0.028416 / 0.089091 |

Each row contains 12 held-run scores. This supports testing W_tok as a strong,
cheap feature before adding detailed engine state. It does not establish
calibration at 0.5 s under intervention, causal transfer, or policy gain.

Important target/availability details from the script:

- `load_run:63` defines TTFT as **client first content minus scheduled time**;
  `lorocv:149-176` fits that target where first-content time exists.
- AUC uses successful-on-time labels for admitted requests, including misses;
  latency MAE excludes requests without observed first content. Thus latency
  regression and successful-SLO classification have different missing-label
  semantics. Neither is an all-offered counterfactual prediction.
- `gateway_state:72-100` uses events before each acceptance. This is a
  retrospective causal reconstruction, not a measurement of live feed latency.
- `fcfs_after_test:223-240` uses future admissions until the observed first
  content and fits/scores within the run. It is a descriptive diagnostic,
  not a randomized causal estimate of interference from future admissions.
- Leave-one-run-out is useful development transfer analysis, but can train
  on later sessions when predicting earlier ones. It is not evidence that
  the proposed chronological online RLS procedure was evaluated.

The report's measured successful output mean 128 and fraction >=128 equal to
1 support the claim that these successful outputs were all 128 in this
artifact. That is evidence of a **lack of natural-length excitation**, not
evidence that a survival model for realistic EOS has been identified.

## 3. Opus: counterexamples and needed revisions

### 3.1 Slack emulation: valid conditional algebra, invalid numerical bound

Write b=alpha+gamma*m and beta=0.000758 cost/token. With positive backlog slack
L=K-W and s=b+beta*L, let n requests be admitted and C_rel released between
sampling and the next tick. Exact cost accounting gives

    n*b + beta*sum(p_i) <= s + C_rel.

For n>=1 this implies

    sum(p_i) <= L + C_rel/beta - (n-1)*b/beta.

This is a useful **conditional** bound. Replacing C_rel with its expectation
does not make a hard bound. The 0.3 expected terminations per tick in O:501-505
can coexist with occasional synchronized batches of terminal releases.

Executed example: L=1000 tokens, C_rel=8 cost units and 1000-token arriving
requests permit five admissions in one tick, i.e. 5000 prompt tokens and
4000 tokens beyond the initial slack. That is not bounded by 700. Existing
cost8 may be held by streaming requests; let it release just before five
arrivals while existing awaiting work does not produce first content within
that tick. This is a feasible gateway event ordering, not a claim about its
frequency on the measured host.

The relevant interval is sample-to-next-effective-command, not nominal 50 ms.
R can decrease before the command applies; stale C=R_sample+s then has more
headroom than intended. The endpoint's returned R detects this only after
some decisions may have occurred. A margin must use a justified release/arrival
tail bound over the entire interval or be explicitly probabilistic.

Additional contract issues: the action bound C<=64 can bind before R+s and
change the proposed emulation; K units must be consistent with coefficients
reported in s/kTok; varying max_tokens invalidates the common-b algebra unless
request-dependent base costs are handled. Keep the strict gateway boundary,
including tokenizer-fallback identities, in every emulation test.

### 3.2 Terminal feasibility and "bumpless" handoff

Executed arithmetic: R0=32, C*=16, k_bar=2, H=3 s, no terminals before H.
All plans preserve R_H=32 because the actuator cannot evict. Consequently
R_H<=18 is infeasible even under hold C*. This scenario requires no infinite
stall: all current streams can finish at 3.1 s.

Moreover R_H<=C*+k_bar allows R_H>C*. Even at R_H=C*, a positive-cost next
request cannot fit. "Immediately consistent with fallback" could mean the
API accepts the budget; that is always true for a positive budget and has
nothing to do with the terminal constraint. It cannot mean no admission
drain transient.

The minimum repair is to keep budget feasibility separate from terminal
performance. Use soft burden/unresolved-tail bounds; or require a genuinely
reachable terminal set under proved service and arrival bounds. Do not
claim recursive feasibility from an input-only hold action when a terminal
constraint remains hard.

For O's currently suggested C*=32, the same failure occurs at R0=48,
k_bar=2 and no release before H: 48>34. R48 is reachable under its proposed
48/64 or continuous-slack action regime. If the entire plant history is
restricted to caps<=32 and C*=32, the terminal inequality is instead
automatically satisfied by correct accounting; in that narrow case it supplies
no additional bumpless-handoff guarantee. The finding is not that every
possible fixed action configuration makes the terminal problem infeasible.

### 3.3 Throughput maximum is not a policy-uniform ceiling

Little's law with marked jobs gives R_mean=lambda_adm*E_adm[c*S] under the
usual steady-state/finite-mean assumptions. The marked distribution and
service residence depend on policy. The maximum observed lambda_adm across
fixed caps therefore does not bound another admitted mix.

Executed existence example: a 10,000-token/s service plant admits 5 req/s
when mean admitted size is 2000 tokens, and 20 req/s when it is 500. This
does **not** predict 20 req/s for this vLLM engine; it disproves the deduction
that an empirical 7.4 req/s maximum is a physical upper bound. A valid bound
would need unavoidable minimum resource work per successful request and a
policy-uniform service capacity, or a rigorously bounded policy class.

Nor does stationary Poisson overload imply myopic feasibility is optimal.
A stationary average-reward optimal policy can depend on a nontrivial
differential future value. Consider a strict FCFS, one-reservation plant:
a long job at t=0 would take .4 s and meet the .5 s SLO. Three short jobs at
.05/.15/.25 s each take .1 s. Accepting the long job rejects the three under
the count1 gate; rejecting it serves three. This horizon benefit exists with
no separate decode phase and no later interference on the accepted long
job's first token. Reservation opportunity cost is enough.

Thus F1's small future-token coefficient does not prove "only decode
coupling matters." Its integration window is outcome-dependent, and fixed-cap
admission suppresses some future interventions. A discriminating experiment
randomizes supported later admissions/pulses or prompt mixes while holding
the information at the focal admission comparable; estimates must include
all ensuing cohorts and explicit state-history conditioning.

### 3.4 Target, old cohorts and eventual success

O:270-279 sums `accept * P(T<=tau)` over new arrivals only. Compare:

    A: 2 new eventual hits, 0 surviving old hits -> total2
    B: 1 new eventual hit, 10 surviving old hits -> total11.

The new-only score chooses A. Similarly ten on-time first tokens with only
four complete successes lose to eight on-time complete streams, despite a
first-token-only score preferring ten. These arithmetic counterexamples were
executed; they identify missing terms, not measured frequencies.

Arrival-time reward attribution is legitimate when the **eventual** reward
is modeled correctly. It does not by itself fix delayed outcome feedback.
Previously on-time streaming requests can still fail and remain consequential.

For online adaptation, O:149-156 uses gateway `f-r`, whereas the fitted
script uses client `f_client-scheduled`. A hidden 100 ms ingress delay shifts
the target by 100 ms even if every gateway feature is unchanged. Either
obtain qualified client/ingress observations, freeze a justified adapter, or
state that online updates fit a different target. Also the existing
`oldest_awaiting_first_seconds` uses **receipt**, not accept time
(`telemetry.py:345-351`); it is not the A_W defined in O's table.

### 3.5 Sequential monitoring and selected data

Under ideal independent tests of size .01, 100 checks have false-alarm
probability `1-.99^100 = .633968`. Rolling overlapping 200-request tests
are not independent; their exact rate differs and a binomial null is already
suspect under common service stalls. Valid marginal conformal coverage does
not make all request errors independent Bernoulli trials.

Use independent/predefined episode scores, a declared monitoring horizon with
error spending, or a sequential method whose conditional assumptions match
the data. State detection delay and false alarms as performance measures.
RLS parameter changes also change the conformal score law: a fixed-calibration
theorem does not automatically survive online forgetting and dither.

The "selection on x is harmless" argument requires
E[epsilon | x, accepted]=0, correct regression form and sufficient state.
That is not implied by F1. Hidden congestion can correlate with observed
admission state and epsilon; lack of action support and first-content
censoring further matter. A +/-5% scalar dither is not proof that the
design matrix for intercept/backlog/decoder effects has full rank.

### 3.6 Experimental and script-specific issues

Switchback randomization is potentially useful, but policy-at-arrival
outcomes with retained carryover estimate a history-mixture intervention,
not automatically the value of sustained policies. Toy example:

    Y_t = 10*A_(t-1) + A_t.

Under alternation, A=1 blocks score1 and B=0 blocks score10, although sustained
A scores11 and sustained B scores0. This is an existence counterexample,
not a claim that 30 s blocks necessarily fail. Thirty seconds may suffice
empirically, but a 3 s p90 residence and observed 2.5 s drain maximum do not
prove finite carryover <=5 s under stalls or new workloads. Define a
treatment-history estimand, predeclared washout or whole-episode comparison.
Randomized whole episodes **within** host/time blocks are not intrinsically
confounded by session effects; their variance can be higher.

O's 49-pair normal approximation can be a rough budget input, not achieved
power with correlated blocks and only three sessions. O:605-616 and F5 use
different success thresholds: lower confidence bound>0 plus point estimate
>=delta is weaker than a lower bound>delta. Choose one before collection.
Do not treat "not significant" as no practical benefit. Per-stratum fairness
noninferiority also needs multiplicity/sample-size planning.

The script's `drain` loops only over events **after** the update and checks
R<=budget, not whether the first actual request can fit. On an executed
synthetic case with R=1 already below a new16 budget at t=1 and the next
terminal at t=10, it returns `reserved_fits_after_s=9.0`; the true first-fit
time for R<=16 is zero. This does not invalidate the verified AUC/MAE.
It requires rechecking the drain summary before treating its extrema as
even descriptive transition timings, and never as a universal upper bound.

## 4. Gemini: model, constraint and evidence defects

### 4.1 Dimensionally invalid service-to-reservation transition

G:247-250 defines

    mu_drain = min(B, N_flight / t_token).

B2048 has units tokens/step. If t_token is seconds/token, N/t_token has
tokens/s (with N a count). Their minimum is dimensionally undefined. Neither
is reservation cost/s. For N=16 and t_token=.02, the computed numerical
candidate800 is tokens/s, not 800 cost units/s.

Even a corrected token-throughput rate cannot release reservations smoothly:
each request keeps its entire cost until terminal cleanup. Two equal-cost
states with different residual output lengths have different next releases.
The proposed state `(R,N,lambda)` and a mean size do not resolve this. A
terminal hazard with per-request cost/age/phase marks or a coherent event
simulator is required before an action-conditioned drain curve is meaningful.

The generic conservation equation in section2 is correct. The error is the
claim that the later greybox equation instantiates it. CPU mocks that only
reproduce cap reduction do not validate that transition model.

### 4.2 Absent KV state cannot certify the resource chance constraint

G excludes KV as unobservable, yet imposes
`P(KV>=.90)<=.001`. Without a validated resource observation/model, two
physical states compatible with the same gateway summary can have .20 and
.99 occupancy, for example through different hidden committed contexts or
backend state. Their mixture probability is not identified by R/N/oldest age.
No probability .001 follows from excluding the sensor.

Even exact prompt/max lengths would need validated KV block size, allocation,
shared/prefix behavior, scheduler settings and conservative capacity semantics.
Weighted cost is a reservation convention, not a physical KV model.
"Using max_tokens prevents OOM" (G:144-149) must be removed as a claim about
this gateway. The native engine's allocation/preemption rules are separate.
For illustration, a zero-violation independent-episode binomial argument
needs about2995 episodes for a 95% upper violation rate below .001; eight
model scenarios cannot supply that guarantee.

### 4.3 Fallback can permanently preserve the high cap

G's fallback is24 only when R<=24; otherwise it holds the prior cap.
Executed event example: R=C=32. Each tick one cost1 request releases and
one new cost1 request enters at C32. At every fallback decision R is again32,
so five tested ticks remain `(R,C)=(32,32)` and the construction can continue.
No rule initiates drainage toward24.

Lowering directly to24 while R>24 is a legal API action. Whether24 is a
good fallback is another question: the historical facts cited do not qualify
it, and numerical interpolation between16 and32 is not action support.
G also uses48/64 elsewhere without one coherent approved action set.

The unconditional "watchdog issues24" later in the report conflicts with the
conditional fallback above. Worse, it does not resolve a previously timed-out
write, which may apply later and supersede24 (section5.1). Both policy and
transport state machines must be specified before calling this safe.

### 4.4 Reward, clock and penalty consistency

G:87 places a sum of switching penalties inside another sum over H stages:
as written the same total penalty is charged H times. The later decision
rule penalizes only the first change. Those are different optimizations.
Choose one objective; when stage reward is requests/s, either multiply by
Delta to obtain counts or give rho the corresponding rate units. Delta=1
hides but does not fix the inconsistency when cadence changes.

The TTFT expression lists scheduled-to-admit, prefill execution and transport,
but no explicit engine queue/pending delay. If "prefill_exec" is intended to
include that, rename and define it. FIFO queue delay is precisely the risk
the controller claims to manage.

Section3's success indicator is correct; section7's Gaussian TTFT probability
alone does not supply eventual success or old-cohort effects. Use the complete
scorer in the transition implementation, not just in the report's top-level
definition. A 5 s horizon does not capture the "complete drain" solely
because median lifetime is2-3.5 s.

### 4.5 Calibration, OOD geometry and experiment validity

For n=10 calibration residuals and target95%, the split-conformal rank is
`ceil(11*.95)=11`, beyond the sample: the distribution-free upper bound is
infinite. The largest finite residual covers only10/11=.909091 of a new
exchangeable continuous score. An ordinary empirical95th percentile is
not a finite-sample95% guarantee. Chronology alone does not establish
exchangeability; action search additionally needs simultaneous/selected-policy
calibration or an explicitly heuristic interpretation.

Mahalanobis distance is not a convex-hull test. A chi-square cutoff, when
justified by a multivariate Gaussian model and squared distance, defines an
ellipsoid; states outside the finite sample hull can lie inside it. Writing
`d_M > chi_square` also needs clarity on whether d_M denotes squared distance.
Neither geometric test establishes action-pattern support.

Fixed-cap calibration followed by switching validation changes the residual
distribution that matters. Tuning H/rho/fallback after calibration changes
selection; freeze the complete candidate then allocate fresh calibration
appropriate to its action regime. PRBS excitation helps, but does not remove
unmodeled history confounding without recording assignment and causal state.
Same offered prefixes do not give identical physical starting states.

The proposed `gain<=2*SD(replicates)` criterion ignores standard error,
pairing and sample count. With independent paired differences, uncertainty
in the mean scales as SD(diff)/sqrt(n), not SD(outcome). Use a preregistered
minimum useful gain and a design-appropriate effect interval; broad noise
does not prove equivalence. Ten repeated episodes require power justification,
not a fixed assertion that the counterfactual problem is resolved.

### 4.6 Measured-claim and implementation audit

No artifact or command accompanies G:265's "runtime <4.0 ms (benchmarked on
Python3.12/3.14)." The report also concludes its search satisfies <20 ms,
although its model/interface is not implemented here. Relabel as a budget
until measured; this review cannot verify an absent benchmark.

Likewise <.5 ms in-process state access, 50-200 ms engine scrape delay,
10-30 request completions/s and Fano>>1 are not established for the studied
plant by the cited material. The Mayor-verified backlog reanalysis instead
contains approximately4-7.4 admissions/s in these fixed-cap
development runs and Opus reports an approximately.94 one-second Fano
statistic on this offered trace. Other real workloads may be burstier, but
that is a stress hypothesis, not this trace's measurement.

The shared audit's 289 passing tests and one skip were a **selected local CPU
suite**, not full native acceptance. G:335 must not upgrade them.
G's phase2 prefill-backlog feature should be tested early given Opus' new
factual evidence; this is a ranking of experiments, not proof the feature
identifies policy effects.

There are legitimate count details: with3 actions,4 slots, at most one
adjacent switch, there are3 constant plans plus4 directed adjacent pairs
times3 switch times =15. That count is correct under those restrictions.
But it excludes direct16->32 switches and uses H4, whereas earlier sections
use H5 and a different action alphabet. Freeze one specification before
testing the runtime claim. `/admin/budget` is the real route, not `/budget`.

An in-process state filter plus synchronous NumPy rollout in an async
gateway task is not automatically outside the serving hot path. A watchdog
on the same blocked event loop cannot preempt its own computation. An
external process cannot simply import the live process's `gateway.state`.
Specify a coherent IPC snapshot or a measured, bounded same-process design
with actual event-loop interference tests.

## 5. GPT: substantive remaining specification challenges

### 5.1 Timeout recognition is correct; safe resumption is not yet specified

S:580-593 explicitly recognizes late application and states that
reconciliation does not erase ambiguity. That is correct. Nevertheless
S:188-190 says the previous **acknowledged** budget governs until application.
If a prior command applied but its acknowledgement is lost, the current
budget is not the acknowledged one. Use a set/belief over possible applied
states whenever a prior write is unresolved.

The following executed schedule is compatible with the current endpoint:

| Time | Event |
|---|---|
| 0 ms | Send A requesting32; server processing is delayed before lock acquisition |
| 50 ms | Client times out; A may still be pending |
| 60 ms | GET observes16/version0 |
| 70 ms | A mistaken "reconciled" controller sends fallback B=16; B applies/version1 |
| 100 ms | A finally applies32/version2 |

Reading the old version is not evidence A cannot apply. A single-writer
policy and "one in flight" help only if timeout does **not** clear the
outstanding slot. The safe existing-API behavior may be to stay
action-uncertain indefinitely until the original operation definitively
settles, rather than resume after one poll. A successful application with
the expected version can be evidence of settlement under one-writer
assumptions; an unchanged observation cannot prove nonapplication.

No server-side expiry, operation ID or CAS exists in
`gateway.py:603-630`. A future fenced/leased endpoint is a separate product
change. This limitation also applies to my Astra Round 1 reconciliation
language and is more severe for Opus/Gemini's immediate fixed fallback claims.
The Round 3 contract should explicitly say which finite observation sequence,
if any, permits resuming with the existing API.

### 5.2 Terminal V needs a non-overlapping reward definition

GPT correctly includes old cohorts, complete success, future tail arrivals
and uncertainty bounds. Its remaining gap is mathematical specification of
`V_fallback(b_tail)` for both unresolved reward and congestion.

Partition requests into K, the old+planning-horizon scored cohorts, and F,
offers after the planning horizon. A consistent construction can be:

    value(U) = E[sum_{i in K} eventual_g_i under U then fallback]
               + differential_future_opportunity_value(U, reference).

The first term may be simulated through tail time and then completed with a
value on **only unresolved members of K**. The second term concerns future
cohorts F and must exclude all K rewards already counted. For sustained
arrivals, an undiscounted infinite sum of future rewards is generally infinite;
use an explicit finite evaluation tail, discount with changed objective
disclosed, or a justified average-reward bias/reference-value formulation.

Without that definition, two implementations can both claim conformance:
one adds completed K rewards twice through V; the other scores K correctly
but omits the long reservation's opportunity cost on F. For example, a job
that earns one immediate hit but occupies all admission capacity until H+10
can dominate a K-only reward while preventing many post-H hits. Calling a
burden proxy "congestion" does not specify its request-equivalent value.

Test continuity when moving the planning/tail boundary by one event: rewards
must not jump merely because the same request moves from explicit simulation
to V. Separately compare action ordering under longer tails and bound the
unmodeled opportunity term. If the sign remains unresolved, abstention is
honest but its frequency must be part of the deployability gate.

### 5.3 Counterfactual maximum calibration is not currently observable

GPT already warns that factual actions do not identify simultaneous action
coverage. The proposal to score a maximum over the prespecified action/horizon
family still needs an executable observation design. One episode realizes
one treatment history, not all32 candidate trajectories from the same latent
state. A `max_U residual(U)` score is missing31 counterfactual labels.

Randomized pulses identify distributional action effects under support and
sequential assumptions. They do not expose each episode's full potential
outcome vector. Repeated matched prefixes likewise sample different latent
states. Feasible alternatives include:

- Calibrate observable outcomes of a frozen selected policy on fresh,
  exchangeable episodes; this evaluates policy predictions, not every
  unchosen counterfactual.
- Use a finite, predeclared randomized intervention family to estimate
  simultaneous **population mean contrasts** with multiplicity control.
  Distinguish those confidence intervals from individual-state prediction bands.
- Keep action-gap ensembles as selection heuristics until such assumptions
  and data exist, and rely on randomized complete-policy evaluation for gain.

The design should choose one, not leave "simultaneous score" as if additional
labels appear once the planner searches all plans. This same concern applies
to my Round 1 simultaneous-action option.

### 5.4 Gateway accounting versus backend work, and compute bounds

S's gateway conservation and population bound are correct conditional on
complete ordered events and positive costs. They do not identify physical
engine load after cancellation. Source cleanup calls upstream close, releases
cost, then records terminal (`gateway.py:438-476`). The source does not prove
that every backend has stopped work synchronously at that release.

Two observations R=0 can correspond to an idle backend or residual orphaned
work still consuming service. A gateway-only hazard model may absorb this
statistically, but cannot assert an empty physical queue. Qualify cancellation
propagation, retain a residual-work belief, or invalidate the model scope
after uncertain termination. This matters especially when failures cause
rapid reservation reuse under overload.

Finally,512 rollouts of variable-length continuations do not have a bounded
runtime merely because a wall-clock deadline exists. Define where budget
checks occur, maximum per-rollout events/particles/tail, cancellation of the
worker, immutable snapshot handoff and isolation from the gateway/engine.
If a worker times out, its eventual output must be ignored by decision ID
and expiry; it must not enqueue an old budget after a newer decision.
GPT correctly calls its100 ms a proposal; what remains is an executable
deadline contract and worst-supported-state benchmark, not proof it is slow.

## 6. Cross-proposal discriminating experiments

These are proposed later tests, not experiments executed in this review.

| Question | Smallest discriminating experiment | Decision affected |
|---|---|---|
| Does W_tok replace richer state for control? | Same causal origin information; backlog-only versus age/marked ledger versus qualified engine augmentation; compare calibrated action contrasts and observer cost, not only AUC | Order of observer/model implementation |
| Is slack tracking a useful existing-actuator approximation? | Exact event-ledger replay with clustered terminals, random update delay and marks; later randomized fresh-policy episodes with full denominators | Slack baseline inclusion; probabilistic margin versus hard claim |
| Does lookahead matter without decode effects? | Strict-FCFS synthetic opportunity-cost case, then supported randomized prompt/arrival mixes | Reject the deduction that F1 establishes myopia |
| Can fallback recover after uncertain writes? | Deferred server handler, lost response, overtaking GET, writer restart and stale worker result | Required actuator state machine; whether server fencing is necessary |
| Is a terminal value well defined? | Move every event across the explicit/tail boundary; vary tail length and late failures with constant cohort set | Eliminate double counts and horizon gaming |
| Does calibration survive adaptation/selection? | Frozen versus RLS-updated models; fixed versus selected policies; episode-correlated stall errors and repeated monitoring | Scope of probability claims and monitoring design |
| Does a switchback estimate the desired policy value? | Simulated finite/long carryover; compare exposure-history estimator and blocked whole-episode trial | Block length, washout and primary estimand |
| Can KV/resource claims be supported? | Paired aliased gateway summaries with different qualified engine state or conservative allocation models | Remove or qualify the KV constraint |
| Do declared time budgets hold? | Worst supported active population and continuation length with serving interference measured separately | Plan family/scenario count/process boundaries |

Priority is not "build all four architectures." First fix exact accounting,
cohort reward and unresolved-action contracts shared by every adaptive
baseline. Then use the now-verified backlog signal to test the cheapest
credible controller. Preserve a negative MPC result if its horizon does not
improve independent policy outcomes.

## 7. Exact offline commands and outputs executed for this review

No files outside this report were created by these commands. `python3 -B`
prevents bytecode writes when importing the peer analysis script. Only the
Mayor's existing development JSON was read; no raw archive extraction,
network, GPU, final trace C, product tests or model fitting was performed.
Some printed lines are explicitly toy constructions; the import executes the
actual peer `drain` function on synthetic inputs.

### Command A

```bash
python3 -B - <<'PY'
import importlib.util
import json
import math
import statistics
from pathlib import Path

report = Path('/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/results/mpc-research-cpu/development-backlog-reanalysis.json')
data = json.loads(report.read_text())
for key in ('cv_cap_only', 'cv_backlog_only', 'cv_queue_causal'):
    auc = [v['auc'] for v in data[key].values()]
    mae = [v['mae'] for v in data[key].values()]
    print(f'{key}: n={len(auc)}, AUC min/median/max={min(auc):.6f}/{statistics.median(auc):.6f}/{max(auc):.6f}, MAE min/max={min(mae):.6f}/{max(mae):.6f}')

base = 0.259 + 0.0052083333 * 128
beta, slack_tokens, released = 0.000758, 1000, 8
slack = base + beta * slack_tokens
request_cost = base + beta * 1000
admitted = math.floor((slack + released) / request_cost)
print(f'slack burst: released={released} cu, admitted={admitted} x1000-token requests, excess={admitted*1000-slack_tokens} tokens (>700)')
print(f'terminal hold: R0=32, no releases before H=3, terminal limit=16+2=18, feasible={32 <= 18}')
print('late write schedule: timeout(A=32) at50ms; GET at60ms sees16/v0; B=16 applies at70ms/v1; A applies at100ms/v2; final=32 (not fallback16)')
print(f'100 independent size-.01 checks: false-alarm probability={1-(1-.01)**100:.6f}')
print(f'conformal n=10 alpha=.05: corrected rank={math.ceil((10+1)*.95)}, largest-finite-score coverage={10/11:.6f}')
print('old-cohort reward: A(new2,old0)=2; B(new1,old10)=11; new-only chooses A incorrectly')
print('eventual success: A(10 on-time first,4 successful)=4; B(8 on-time first,8 successful)=8; first-only chooses A incorrectly')
print('KV alias: same gateway R/count/ages, occupancy .20 versus .99; unobserved-mixture violation probability can range 0..1')
print('drain units: B=2048 tokens/step; N=16 and t_token=.02 s/token imply800 tokens/s; min(2048,800) mixes units, not cost/s')
print(f'Gemini one-switch adjacent plans: {3+4*3}; unrestricted one-switch plans: {3+6*3}')
print(f'Gemini repeated switch penalty: H=5 weights internal penalty by {5}, rather than1')

path = '/home/daniel_lin/gt/pr32/polecats/obsidian/pr32/experiments/mpc-research/proposals/opus-round1-analysis.py'
spec = importlib.util.spec_from_file_location('opus_analysis', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('Opus drain already-fit toy:', mod.drain([{'acc':0.,'cost':1.,'end':10.}], [(-1.,32.),(1.,16.)], 0.))
assert admitted*1000-slack_tokens > 700
assert 32 > 18
assert mod.drain([{'acc':0.,'cost':1.,'end':10.}], [(-1.,32.),(1.,16.)], 0.)[0]['reserved_fits_after_s'] == 9.
print('All deterministic assertions passed; no plant/model performance was measured.')
PY
```

Exact stdout:

```text
cv_cap_only: n=12, AUC min/median/max=0.585827/0.685161/0.774981, MAE min/max=0.097756/0.358656
cv_backlog_only: n=12, AUC min/median/max=0.959016/0.976295/0.994851, MAE min/max=0.033374/0.093152
cv_queue_causal: n=12, AUC min/median/max=0.966418/0.981449/0.998080, MAE min/max=0.028416/0.089091
slack burst: released=8 cu, admitted=5 x1000-token requests, excess=4000 tokens (>700)
terminal hold: R0=32, no releases before H=3, terminal limit=16+2=18, feasible=False
late write schedule: timeout(A=32) at50ms; GET at60ms sees16/v0; B=16 applies at70ms/v1; A applies at100ms/v2; final=32 (not fallback16)
100 independent size-.01 checks: false-alarm probability=0.633968
conformal n=10 alpha=.05: corrected rank=11, largest-finite-score coverage=0.909091
old-cohort reward: A(new2,old0)=2; B(new1,old10)=11; new-only chooses A incorrectly
eventual success: A(10 on-time first,4 successful)=4; B(8 on-time first,8 successful)=8; first-only chooses A incorrectly
KV alias: same gateway R/count/ages, occupancy .20 versus .99; unobserved-mixture violation probability can range 0..1
drain units: B=2048 tokens/step; N=16 and t_token=.02 s/token imply800 tokens/s; min(2048,800) mixes units, not cost/s
Gemini one-switch adjacent plans: 15; unrestricted one-switch plans: 21
Gemini repeated switch penalty: H=5 weights internal penalty by 5, rather than1
Opus drain already-fit toy: [{'at': 1.0, 'from_': 32.0, 'to': 16.0, 'reserved_fits_after_s': 9.0, 'first_admission_after_s': None}]
All deterministic assertions passed; no plant/model performance was measured.
```

### Command B

```bash
python3 -B - <<'PY'
import math
# Each tuple is (prediction, observed scheduled TTFT); lower predicts hit.
rows=[(.49,.48),(.495,.49),(.51,.52),(.515,.53)]
auc=sum(p_hit < p_miss for p_hit,y in rows if y<=.5 for p_miss,z in rows if z>.5)/4
print(f'AUC={auc:.1f}; adding .10s to all predictions preserves AUC but predicted hits change {sum(p<=.5 for p,_ in rows)} -> {sum(p+.1<=.5 for p,_ in rows)}')
# Stylized first-token-service-only plant, an arithmetic existence counterexample.
capacity=10000
print(f'composition capacity: service={capacity} tokens/s, mean admitted prompt2000 ->{capacity/2000:.1f} req/s; mean admitted prompt500 ->{capacity/500:.1f} req/s (>7.4)')
# A fallback that refuses to lower until reservations already fall is not a drain policy.
r,cap=32,32
states=[]
for _ in range(5):
    fallback=24 if r<=24 else cap
    r-=1
    if r+1<=fallback: r+=1
    cap=fallback
    states.append((r,cap))
print('Gemini fallback with cost1 release/refill each tick:', states)
# Exposure to preceding treatment can invert policy-at-arrival interpretation.
print('switchback carryover: Y_current = 10*previous_treatment + current_treatment; alternating A=1/B=0 gives observed A=1, B=10; sustained A=11, B=0')
# Time-bounded client reads do not fence pending writes.
events=[(50,'timeout A=32'),(60,'GET sees16/v0'),(70,'apply B=16/v1'),(100,'apply A=32/v2')]
state=(16,0)
for t,kind in events:
    if kind.startswith('apply B'):state=(16,1)
    if kind.startswith('apply A'):state=(32,2)
print('delayed-write arithmetic final state:',state)
print('TTL-free reconciliation remains ambiguous after GET; no additional actual commands were sent.')
PY
```

Exact stdout:

```text
AUC=1.0; adding .10s to all predictions preserves AUC but predicted hits change 2 -> 0
composition capacity: service=10000 tokens/s, mean admitted prompt2000 ->5.0 req/s; mean admitted prompt500 ->20.0 req/s (>7.4)
Gemini fallback with cost1 release/refill each tick: [(32, 32), (32, 32), (32, 32), (32, 32), (32, 32)]
switchback carryover: Y_current = 10*previous_treatment + current_treatment; alternating A=1/B=0 gives observed A=1, B=10; sustained A=11, B=0
delayed-write arithmetic final state: (32, 2)
TTL-free reconciliation remains ambiguous after GET; no additional actual commands were sent.
```

Both commands exited0. AUC invariance in Command B is an arithmetic check of
ranking versus calibration, not a challenge to the reproduced Opus AUC value.
The terminal, units, cohort and KV lines printed in Command A are transparent
constructed examples; only their stated elementary arithmetic is asserted.
The observed development metrics and actual `drain` function output are
distinguished from those constructions above.

## 8. Round 2 exit

Round 2 artifact complete; stop at the barrier and retain context for Round 3.
No conclusion here promotes GPU/HTTP/native state-action/performance/MPC
acceptance. No peer report was modified, and no product code, commits,
pushes, infrastructure changes or live actuation were performed.
