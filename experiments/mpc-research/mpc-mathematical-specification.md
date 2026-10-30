# Hybrid admission MPC specification

Status: engineering specification under Round3 review, not native qualification.
Source baseline: `8e92371`. Every numerical controller setting below is a
development configuration, not an experimentally optimal setting.

## 1. Plant, information and actuator

The actual control is the single gateway's positive weighted reservation budget
`u(t)`. It neither schedules engine tokens nor evicts an admitted request.
The experimental action alphabet starts at `{16,32}`; even these historical
values require qualification on the current native plant. Intermediate caps and
high-frequency relative-slack control are separate action-support experiments.

For admitted request i retain its immutable reserved cost c_i, prompt estimate
p_i, declared output maximum m_i, receipt/admission/first-content timestamps,
phase and identity. The exact accounting ledger is L. It is not a high-dimensional
learned regression vector. Derived low-dimensional observer candidates are

    z = (R, Q, D, W_prompt, oldest_awaiting_age, arrival_history),
    R = sum_{i active} c_i,
    W_prompt = sum_{i awaiting observed first content} p_i.

W_prompt is an original-prompt backlog proxy, not residual GPU work. Q/D mean
gateway phases, not true engine occupancy. Request ages/marks are retained for
residual-lifetime ambiguity and outcome accounting; their predictive complexity
must earn inclusion through ablation.

The physical state includes remaining prompt/decode work, scheduler/pending
state and transport. These are latent. The estimator maintains an approximate
belief b over them, conditioned only on information received by the decision:

    b_k = Law(X(t_k), theta | I_k),
    y_j = h_j(X(s_j)) + e_j, available_j >= s_j.

The observation adapter must supply source time, availability, run/boot/clock
identity, cost fingerprint, sequence/version and coherent marked request data.
It must not silently equate delayed offline reconstruction with a live sensor.
Check clock identity before comparing uptime values.

## 2. Exact admission and release dynamics

At each ordered gateway offer:

    a_i = 1{R(t-) + c_i <= u_applied(t)},
    R(t+) = R(t-) + a_i*c_i.

At the exactly-once terminal cleanup of admitted i:

    R(t+) = R(t-) - c_i.

No cost is released at first token or because u decreases. Lowering u below R
is valid and can block subsequent offers. Raising it admits no request by itself.
Strict boundary comparison, event order and whole-cost release are shared by all
controllers and the CPU reference plant.

With A admissions, F first-content observations and E_Q/E_D terminal exits:

    Q_next = Q + A - F - E_Q,
    D_next = D + F - E_D,
    R_next = R + C_in - C_out.

Generate all flows from the same request lifecycle. Independent unconstrained
flow heads are not an acceptable implementation of these identities.
Gateway release is not proof that cancelled backend work has stopped; suspected
orphan work invalidates a native model's scope.

## 3. Executable predictive model boundary

Implement a pure event-driven lifecycle reference model, explicitly labeled a
CPU development surrogate rather than a vLLM scheduler replica.
Its state contains a request ledger, sampled remaining work, pending observation/
cleanup events, a scheduler clock, and the applied budget. A scenario supplies
causally sampled future offers/marks, output lengths, service variation and
transport delay. The generator receives past-only state and a frozen seed, never
the benchmark's actual future arrival tape or realized EOS.

A kernel-inspired candidate transition packs at most B tokens/iteration:
one next token for each eligible decoder, then FIFO prefill chunks in the
remaining budget. For scheduled prefill tokens q and decode tokens d, use

    tau = theta_base + theta_prefill*q + theta_decode*d
          + theta_interference*q*d,

where tau is seconds, theta_base is seconds/iteration,
theta_prefill/theta_decode are seconds/token and theta_interference is
seconds/(prefill-token*decode-token). Nonnegative, finite coefficients and an
explicit positive-duration rule are mandatory.
This formula is a candidate surrogate, not an identified native service law.
Changing it changes the model version and support seal.

Actions/arrivals during an in-flight iteration change admission and pending work,
not work already committed to that iteration. First-content/terminal observations
occur only after their modeled compute/transport events. Output samples cannot
exceed a request's declared maximum; first-token hit and eventual complete
success remain separate events. The initial remaining-work belief must represent
the ambiguity of identical observed aggregates.

Alternatives to test are the measured W_prompt TTFT law plus conditional residence
model and more detailed native scheduling models. Do not select the detailed
surrogate merely because its equations look physical. The model interface accepts
state, a timed budget plan and a shared scenario, and returns coherent outcome/
reservation trajectories, unresolved reward mass and explicit invalid-scope reasons.

## 4. Cohort objective and terminal treatment

Let C_k include every currently unresolved admitted request whose outcome can
change and every offered request scheduled in the prediction window. Let

    g_i = 1{eventual complete success}
          * 1{client_first_content_i - scheduled_offer_i <= 0.5s}.

Optimize expected successful-request COUNT, not accepted-only latency:

    maximize_U J(U) = (1/M) sum_m sum_{i in C_k} g_i^m(U).

The full episode evaluator divides by the fixed offered-arrival duration.
Rejections, failures, cancellations and unresolved streams receive no positive
success credit. Previously resolved outcomes are action-independent constants.
Current first-token-hit streams remain unresolved until terminal success.

Scheduled client time is not generally known to the live gateway. The predictor
needs a qualified ingress/egress-delay model or an explicit uncertainty interval.
Receipt-age prediction must not be relabeled scheduled-arrival prediction.

After the planned horizon, continue under the declared backup with continuing
scenario arrivals. Score each target cohort once; tail arrivals affect contention
but are not also rewarded unless they belong to the declared cohort set.
Unresolved target rewards retain a lower/upper interval. A learned residual value
may replace explicit tail evaluation only with a disjoint, documented partition;
it is not added on top of already counted outcomes or congestion penalties.

Initial switch/risk weights are zero; deterministic ties prefer hold, fewer
changes, then a fixed action order. This avoids arbitrary mixed-unit weights.
Any nonzero switch penalty has units successful requests/change and needs a
measured transition-cost or development Pareto rationale.
CVaR/chance variants remain separate challengers requiring calibrated,
observable scores and declared distribution assumptions.

## 5. Constraints, horizon and compute

Hard contracts: finite supported budgets; coherent accounting and nonnegative
work; matching model/calibration/support/cost/run/clock identities; causal
observations; one known command state; bounded planner search; no action from
expired or invalid input. R<=new_budget is not a hard post-reduction constraint.
No unobserved KV chance constraint or deterministic SLO guarantee is introduced.

The initial small development search uses 1s control slots and a 3s horizon,
enumerating at most eight `{16,32}` sequences. Compare N=1/3/5 with the same model,
state, scenarios and action set. Shorter/longer cadence and one-switch pruning
are development ablations, not settled performance facts.
Use shared scenario draws across plans; later actions do not depend on unrevealed
scenario outcomes. Execute only the first action and replan.

A configurable wall-time deadline and finite state/event/plan limits return an
observable fallback on exhaustion. Run prediction outside gateway handlers/locks.
Cooperative checks are not a hard real-time OS theorem; recheck freshness at the
actuator. A synchronous solver in an async task is not made preemptible by
`asyncio.wait_for`.

Latency/queue/resource risk is a reported soft/model-relative quantity unless
native evidence supports a stronger contract. Terminal burden is not a hard
drain constraint. Holding a legal action does not prove a terminal set reachable.

## 6. Transport, fallback and what can be proved

Before native state/action/calibration authorization: shadow only; no admin
budget writes. Unknown qualification cannot authorize a "conservative" cap.
After qualification, an empirically selected fixed budget is the operational
fallback, not a certified latency-safe policy.

A guarded server capability must be confirmed before a controller writes:
match runtime and cost identity, expected budget version, model/support seal,
clock domain and expiry checked under the same application lock.
An old endpoint ignoring extra fields is not a guarded endpoint. A timeout is
unknown application, not cancellation; suspend new speculative writes until
the command is definitively applied or fenced/expired.
If transport is unavailable the last applied budget persists. No promise of a
delivered fallback is possible without communication.

Under exact serial accounting, true c_i>=c_min>0, bounded budgets u<=u_max,
finite initial R and nonnegative releases:

    R(t) <= max(R(0), u_max),
    admitted_count(t) <= max(R(0), u_max)/c_min.

Induction over admission/release events proves these reservation bounds.
They do not prove physical KV safety, backend-work boundedness after orphaned
cancellation, finite latency, queueing stability before admission, or Lyapunov
stability. No unconditional recursive-feasibility/SLO theorem is claimed.

## 7. Required discrimination

Compare best development fixed cap, same-state W_prompt hysteresis/feedback,
same-model N=1 and the MPC candidate. Separate feature gain from lookahead gain.
Use model ablations, horizon-edge/old-cohort/failure tests, clustered release,
cross-boot timestamp, missing metadata, timeout/late-command and state-aliasing
counterexamples before native trials. All performance claims remain blocked on
the preregistered repeated native HTTP experiment.
