# PR32 MPC research — Round 1 independent proposal (opus)

Researcher: polecat `obsidian`, Copilot CLI session reporting model `claude-opus-5.5`.
I cannot independently verify the Gas Town profile text or process arguments from
inside the sandbox; this line records the runtime's self-reported model only.
Source: worktree at `8e92371f342be98a50682bf4e2ddbb014665498b` (no product edits).
Inputs read: the Mayor's `current-state-audit.md` and `research-protocol.md`, current
source, and primary reports. No peer proposal was read. Reserved final trace C was
not inspected, and no GPU, network or paid resource was used.

Evidence labels used throughout:

- **[M]** measured in this round by offline re-analysis of archived captures
  (development reuse of already-inspected data; reproducible with
  `opus-round1-analysis.py`, see §11);
- **[H]** historical claim from a committed report (cited, not re-run);
- **[D]** deduction from source code or mathematics;
- **[P]** proposal, hypothesis or experiment not yet performed.

No statement below is a held-out, counterfactual, native-qualified or policy-gain
result. GPU/HTTP/performance acceptance remains **false**.

---

## 0. Executive summary

1. **The binding problem is not forecasting; it is which requests to admit.** The
   offered development trace is ≈25.5 req/s, Poisson-like (1-s Fano factor 0.94)
   and stationary at 30-s scale (CV 3.5%) [M]. Every fixed cap rejects 71–83% of
   offers [H, cap-range report]. The system is in permanent 4–5× overload, so
   arrival-rate forecasting has little decision value in this regime. The
   controllable quantity is the admitted set.
2. **A zero-age, gateway-internal state predicts each admitted request's SLO
   outcome almost perfectly, and the current cost reservation does not.**
   Leave-one-run-out across 12 qualified development captures [M]:

   | Predictor at the gateway's own accept instant | held-run AUC (hit vs miss) | held-run TTFT MAE |
   |---|---|---|
   | reserved cost R + own prompt (what C "sees") | 0.586–0.775 (median 0.685) | 0.098–0.359 s |
   | **prompt tokens admitted but without first content (W_tok) + own prompt** | **0.959–0.995 (median 0.976)** | **0.033–0.093 s** |
   | + decoding count, oldest-waiting age, gateway receive→accept lag | 0.966–0.998 (median 0.981) | 0.028–0.089 s |

   The two-parameter law is TTFT ≈ θ0 + θ1·(W_tok + p). It fits with θ0 ≈ 0.046–0.095 s
   and θ1 ≈ 0.045–0.064 s/kTok, i.e. 16–22 kTok/s of effective prefill drain [M].
3. **TTFT is approximately determined at admission.** Prompt tokens admitted *after* a
   request add only 0.001–0.016 s/kTok to its TTFT, versus 0.048–0.060 s/kTok for
   tokens *ahead* of it. Adding them improves MAE by ≤4.2 ms [M]. This is consistent
   with FCFS waiting order in vLLM V1. Consequence [D]: the dominant SLO decision is
   myopic per-request feasibility, and multi-step lookahead matters mainly through
   decode-residence coupling (2–7 s) and parameter drift.
4. **Current evidence never exercised unknown decode lengths.** Every successful
   stream in all 13 captures emitted exactly 128 tokens [M]. The synthetic
   random-letter prompts never hit EOS, so "natural EOS" behaved as forced-128.
5. **Session effects are as large as cap effects.** Fixed C32 whole-run goodput
   ranges 3.733 / 4.590 / 5.080 req/s across sessions [H/M]. The C32−C16 difference
   is +0.46 to +0.95 req/s within the cap-range session. Per-30-s-window replicate
   noise SD is ≈0.57–0.76 req/s [M]. Any policy comparison must block on session and
   use a switchback design (§7).
6. **Recommendation.** I recommend a hierarchical controller, staged:
   - **Inner (existing gateway rule):** first-fit cost admission, unchanged.
   - **Middle, state feedback (20 Hz):** a slack-tracking budget controller,
     C(t) = R(t) + s(x(t)), sent through the *existing* `/admin/budget` endpoint. It
     implements a deadline-derived backlog cap on W_tok with no new actuator. It
     does extend the *action support*: continuous C at high rate, which needs its
     own qualification.
   - **Outer, receding horizon (1–5 s):** a small scenario MPC only if the
     falsification experiments show residual value beyond the state feedback.
   - **Baseline arm:** the roadmap's {16,32} discrete MPC.

   A per-request deadline gate inside the gateway is the cleanest form of the same
   idea. It is a **new actuator** and is proposed only as a separately approved
   alternative.
7. The prefill-lab engine-level envelope (ms precision, post-admission, pre-ADD
   state) is not on the critical path for a 500-ms admission SLO. A gateway-only
   model already reaches ≈50 ms MAE. Its inclusion in the controller should be
   decided by an incremental-value ablation (F3), not assumed.

---

## 1. Plant, variables and units

### 1.1 Requests and the existing cost

For offered request i [D, `gateway.py:206-226`]:

- a_i: scheduled arrival (s); r_i: gateway receive time (s);
- p̂_i: gateway token estimate (tokens, via the chat template; falls back to a
  character heuristic, counted in `tokenizer_fallback_requests`);
- m_i: max output tokens (128 in all evidence);
- o_i ≤ m_i: realized output tokens (unknown at admission; =128 in all evidence [M]);
- the reservation cost, in cost units (cu):

  k_i = max(0.1, α + β p̂_i + γ m_i), with α = 0.259 cu, β = 7.58·10⁻⁴ cu/token,
  γ = 5.2083·10⁻³ cu/token.

For m = 128, k_i = 0.926 + 0.758·(p̂_i/1000) cu. A 4000-token prompt costs ≈3.96 cu,
only ≈4× a 30-token prompt, although its prefill time is ≈100× larger [D from the
θ1 fit]. **The cost is dominated by the decode reservation, not by
TTFT-relevant prefill work.** This explains why the R-only predictor is weak [M].

### 1.2 Gateway state (exactly observable, zero age at the gateway)

The gateway's lifecycle events (`accepted`, `first_content`,
`completed|failed|cancelled`; `gateway.py:301-330, 430-561`) define two disjoint
sets:

- 𝒲(t): admitted, first content not yet observed (backlog: transport + engine
  waiting + prefilling);
- 𝒟(t): first content observed, not terminal (decoding/streaming).

From these sets:

| Symbol | Definition | Unit | Available today? |
|---|---|---|---|
| R(t) | Σ_{j∈𝒲∪𝒟} k_j | cu | yes (`current_admitted_cost`, `/metrics`) |
| W_n(t) | \|𝒲\| | req | yes (`awaiting_first_content` in telemetry snapshot) |
| **W_tok(t)** | Σ_{j∈𝒲} p̂_j | tokens | **not exposed**; reconstructible from the event log; proposed read-only metric |
| A_W(t) | max_{j∈𝒲}(t − accept_j) | s | yes (`oldest_awaiting_first_seconds`) |
| D_n(t) | \|𝒟\| | req | derivable (active − awaiting) |
| C(t) | applied budget | cu | yes, with version and applied time |
| θ(t) | (θ0, θ1, θ2) slow service parameters | s, s/token, s/req | latent; estimated online (§5.3) |

Conservation invariant [D]: R(t) = Σ_{j∈𝒲∪𝒟} k_j at all times. The code floors at 0
(`gateway.py:338`), so floating-point drift must be checked (test T1).

### 1.3 Controls

| Control | Exists? | Semantics |
|---|---|---|
| u_C(t) = C(t) > 0, via authenticated `POST /admin/budget` | **yes** (`gateway.py:588-630`) | Absolute budget. Takes effect for the next admission check. Existing reservations survive reductions. |
| Startup MAX_IN_FLIGHT / MAX_WAITING (protected mode) | startup only | Not a dynamic action; protected32 ≠ C32. |
| Per-request deadline gate g_i ∈ {0,1} | **no — new actuator** | Would reject i if the predicted TTFT exceeds τ − margin. Requires gateway code. |
| Engine priority, preemption, batch budget B, cancellation, gateway waiting queue | **no** | Need vLLM or gateway work. Not claimed. |

### 1.4 Disturbances

- w_A: the arrival point process with marks (p_i, m_i). Measured on the development
  trace [M]: mean 25.48 req/s; 1-s rate SD 4.89 (Fano 0.94); 5-s SD 1.90; 30-s SD
  0.88. Input-token arrival rate per 5 s: 29–52 kTok/s. Prompt quantiles 10/50/90/99%:
  328 / 907 / 3713 / 6674 tokens.
- w_o: realized output length o_i. Degenerate (=128) in all evidence [M], so it is
  out of distribution (OOD) for any natural-EOS workload.
- w_θ: service-rate drift across sessions/hosts. Per-run θ1 is 0.0455–0.0635 s/kTok
  and θ0 is 0.046–0.095 s [M].
- w_P: host pauses. The cap16-r2 run had a 5.855-s gateway loop stall and a 431.7-ms
  client lag [H, cap-range report]. These are unbounded in principle.
- w_L: client scheduling lag (gated ≤50 ms) and budget application lag (gated ≤50 ms).

### 1.5 Observation equations

- y_G(t) = x_G(t): gateway state, exact at the gateway. A controller outside the
  gateway sees it with age Δ_obs plus the HTTP round trip. The `/metrics` snapshot
  takes `state_lock`, so it is internally consistent.
- y_F,i = f_i − r_i, observed at f_i: per-request TTFT labels. The feedback delay
  equals the TTFT itself, ≤ ~1–2.5 s [M], which is **fast** for online
  identification.
- y_E(t): vLLM Prometheus gauges (waiting/running, histograms), scraped ≈1 Hz.
  These are aggregate, aged and not request-aligned [H, state-model report].
- y_pre-ADD: engine snapshot inside EngineCore. It exists only after admission, so it
  cannot justify that admission [H, audit].

---

## 2. Serving dynamics

### 2.1 Admission (existing, exact) [D]

At receive time r_i:

  accept_i = 1[ R(r_i⁻) + k_i ≤ C(r_i) ]  (× g_i if the proposed gate exists)

If accepted: 𝒲 ← 𝒲 ∪ {i}, R ← R + k_i. On first content: move i from 𝒲 to 𝒟.
On termination: remove i, R ← R − k_i.

This is **first-fit with size bias**. When R is near C, only requests with
k_i ≤ C − R fit. Measured acceptance by cost quartile (Q1→Q4) is 0.25→0.07 at C16,
0.31→0.14 at C32 and 0.34→0.20 at C96 [M]. Small requests are preferentially
served under every cap. This matters for metric gaming (§3.5).

### 2.2 Cap-reduction semantics [D + M]

After a reduction C⁺ → C⁻ at t0, nothing is evicted. Admissions are blocked until

  T_drain = inf{ t > t0 : R(t) + k_next ≤ C⁻ },  with R(t) = R(t0) − Σ_{j: e_j∈(t0,t]} k_j.

R(t) decreases only through terminations, so the drain time is governed by
residual residence. Measured over 8 C32→C16 reductions: time until reserved cost
first fits is 0.0006–2.41 s, and the first admission follows 0.04–2.46 s after the
change [M]. Increases act on the next arrival (mean gap 39 ms).

**The actuator is asymmetric: about 40 ms upward, up to ≈2.5 s downward at C≤32.**
Any model or MPC must encode this. Instant population shrink is invalid.

### 2.3 First-token dynamics (structural assumption + measured fit)

Engine assumptions [D, structural, partly unverified for these captures]: vLLM 0.30
V1 scheduler, FCFS waiting queue, chunked prefill. Running requests are scheduled
before waiting prefill.

Caveat: B (max batched tokens) is **not recorded** for the admission captures.
`backend_launch_provenance` says "Not observed by this runner" [M]. The
prefill-service protocol pins B2048. This provenance gap must be closed before any
claim that transfers θ1 between configurations.

The measured reduced form [M]:

  T_i ≡ f_i − a_i ≈ θ0 + θ1·(W_tok(r_i⁻) + p̂_i) + θ2·D_n(r_i⁻) + ε_i

Pooled leave-one-run-out fit: θ0 ≈ 0.027 s, θ1 ≈ 0.0553 s/kTok, θ2 ≈ 0.0018 s/req.
Backlog-only per run: θ1 = 0.0455–0.0635 s/kTok. θ1 is higher at C96
(0.057–0.059), consistent with decode interference slowing prefill drain.

**FCFS falsification test F1 [M].** I added the prompt tokens admitted during
(accept_i, f_i), i.e. *after* i, as a regressor. Their coefficient is 0.0014–0.0155
s/kTok, versus 0.048–0.060 for tokens ahead. MAE improves by ≤4.2 ms. This regressor
is endogenous: a longer T_i opens a longer window, so the true effect is likely
smaller still.

Deduction: T_i is ≈ fixed at the admission instant up to decode interference and
noise. Admitting j after i affects i mainly through θ2, and affects *later*
requests through W_tok.

### 2.4 Decode residence and coupling

Residence S_i = e_i − accept_i has p50 2.00 / 2.82 / 4.84 / 6.72 s at C16/32/64/96
[M]. With o = 128, the implied mean inter-token latency is ≈15.6 ms at D_n≈6 and
≈52 ms at D_n≈41. Admitted requests therefore hold budget for 2–7 s. They slow the
steps that also carry prefill chunks (θ1 rises with D_n). This is the only channel
through which an admission now affects first tokens seconds later, and hence the
only place where a horizon beyond one TTFT matters [D].

### 2.5 Fluid steady state and goodput ceiling [D + M]

Under overload, R ≈ C − κ, where κ is fragmentation from the first-fit rule. Little's
law gives the admission rate

  λ_a(C) ≈ (C − κ) / E_adm[k·S],

and goodput is g(C) = λ_a(C)·P(T ≤ τ | C).

Measured admission throughputs are 4.2–4.3 / 5.3–5.8 / 6.5 / 7.3–7.4 req/s at
C16/32/64/96 [M], with P(hit | admitted) 0.98 / 0.71–0.87 / 0.35 / 0.36–0.46.
Hits are bounded by the admitted throughput, so **the ceiling on goodput for any
admission-only policy is ≈7.4 req/s on this host and trace**. That is +46% over the
best single fixed run (5.08) and +66% over the C32 mean (4.47). The ceiling is not
attainable, because selecting for feasibility also changes throughput. Any claimed
gain must be well below it.

Backlog grows far faster than admissions: from C32 to C64, admissions rise by +12%
while mean W_tok rises from 2.8k to 10.1k tokens [M]. The SLO collapse at high C is
a backlog phenomenon, not a throughput one.

---

## 3. Objective, constraints, horizon, budgets

### 3.1 Primary objective (unchanged)

  J = (1/T_off) · Σ_{i offered in [0,T_off)} 1[accepted_i ∧ success_i ∧ f_i^client − a_i ≤ τ],
  τ = 0.5 s, T_off = 300 s.

Rejections, failures and transition time stay in the denominator. Cohorts are
assigned by scheduled arrival. There is no E2E SLO (`e2e_slo: null` in
`experiments/low-cap-switch-v1.json`), and none is invented.

Secondary metrics, reported separately: rejection fraction; failure count; TTFT
p50/p90/p99 of admitted requests; ITL/TPOT; E2E; per-input-size-stratum goodput and
acceptance; number of budget updates; time-in-fallback.

### 3.2 Stage reward for a receding horizon [P]

At decision time t with candidate action sequence u_{t:t+H}:

  Ĵ(u) = E_ω[ Σ_{i: a_i∈[t,t+H)} accept_i(u, x, ω) · P̂(T_i ≤ τ | x(r_i), p_i, θ̂) ]

ω ranges over arrival/mark scenarios bootstrapped from TRAIN windows, and output
lengths drawn from the TRAIN distribution (degenerate at 128 for now). Cohort hits
are counted at arrival, which avoids the completion-delay mistake identified in
`results/flow-diagnosis-20261020/report.md`.

### 3.3 Constraints

- **Hard (actuator):**
  - C ∈ 𝒰. Initial discrete MPC: 𝒰 = {16, 32} (roadmap scope). Slack tracking:
    C ∈ [C_lo, C_hi] = [R + 0, 64].
  - Update period ≥ 50 ms.
  - Measured application lag ≤ 50 ms (existing gate). Otherwise the update counts
    as failed and triggers fallback.
- **Hard (model validity):**
  - Observation age ≤ 100 ms.
  - θ̂ inside the validity box spanned by TRAIN sessions; otherwise fallback.
  - Output-length distribution not OOD (two-sample KS against TRAIN at α = 0.01 over
    a rolling 200 completions); otherwise fallback.
- **Soft:**
  - Per-size-stratum SLO goodput ≥ (1 − δ) × the fixed C* baseline's stratum
    goodput, with δ pre-registered (proposed δ = 0.1). This is a constraint, not a
    weight, to block a pure short-job selection win.
  - Switching rate below a TRAIN-chosen limit.
- **Terminal (MPC only):** R(t+H) ≤ C* + k̄. The plan must end where the fallback
  fixed policy C* is immediately consistent, so a fallback handoff has no extra
  drain transient (bumpless).

### 3.4 Feasibility and stability

- **Recursive feasibility [D].** All hard constraints restrict inputs or are
  supervisory. "Hold C*" is always admissible, so the optimization is always
  feasible and the fallback is always available.
- **Boundedness [D].** Rejection is immediate and there is no gateway queue, so
  R(t) ≤ C_max + max_i k_i for all t. In addition, W_n ≤ C_max/k_min and
  W_tok ≤ C_max/β ≈ 84k tokens at C_max = 64. The closed loop cannot diverge at the
  gateway regardless of the controller. "Stability" here is a performance question
  (backlog relative to τ), not a Lyapunov one.
- **No deterministic latency guarantee.** w_P (host stalls of several seconds) and
  θ drift are unbounded. All SLO statements are statistical, conditional on the
  stated assumptions.

### 3.5 Weights

The objective is a count, so no weights are needed. A switching penalty λ_sw is
included only to suppress chattering due to model noise. Its value is chosen on
TRAIN as the knee of the Pareto curve (goodput vs switches/min). Fairness is a
constraint, not a weight. The opportunity cost of admission is endogenous to the
rollout, not a tuned constant.

### 3.6 Time scales and horizon [D + M]

| Time scale | Value | Source |
|---|---|---|
| Mean inter-arrival time | 39 ms | [M] |
| Budget application lag | ≈3 ms (≤50-ms gate) | [M] (`budget_changes.json`) |
| Backlog drain θ1·W_tok | 0.1–0.5 s at C≤32 | [M] |
| TTFT label feedback | 0.1–1 s | [M] |
| Downward actuation (drain) | 0–2.5 s at C≤32 | [M] |
| Decode residence p90 | 2.2–3.1 s at C≤32; up to 7.4 s at C96 | [M] |
| Session drift in θ | minutes–hours | [M]/[H] |

Choices [P]:

- State-feedback tick Δ_f = 50 ms. This matches the existing lag gate and is ≈1.3
  arrivals.
- MPC tick Δ_m = 1 s, horizon H = 3 s (≈ p90 residence at C32), with switching
  allowed at 1-s boundaries. That is 2³ = 8 sequences over {16,32}, or 3³ = 27 with
  {16,32,48} — exhaustive enumeration, no solver risk.
- Compute budget: ≤10 ms per MPC tick for 8–27 sequences × 32 scenarios with an
  event-driven rollout of ≈120 arrivals each. Must be measured; this is a budget,
  not a claim.
- Observation budget: gateway state read ≤10 ms after the decision instant.
- Actuation budget: ≤50 ms acknowledgement.

### 3.7 Fallback ladder

1. Any budget update fails or exceeds 50 ms, or observation age exceeds 100 ms → hold
   C* (the fixed TRAIN-best cap; currently 32 by [H]) and stop adapting for 30 s.
2. The TTFT residual CUSUM alarms (one-sided, on (T − T̂)/σ̂, threshold h chosen for
   ARL₀ ≥ 1 h on TRAIN), or output lengths are OOD → hold C* for the rest of the
   episode, and record it.
3. Telemetry loss → episode invalid (existing integrity gates).

Every fallback interval stays in the denominator.

---

## 4. Controller comparison

Columns: what the controller needs, what it can exploit in this plant, and the
failure mode or prediction [D/P].

| Controller | Needs | Exploits | Failure mode / prediction |
|---|---|---|---|
| Fixed C (16/32/…) | nothing | static operating point | Cannot react to backlog bursts or session drift. Best fixed C is session-dependent (C32: 3.73–5.08). |
| Fixed C* tuned on TRAIN | TRAIN sweep | same | Strong baseline under a stationary overload regime. |
| Hysteresis on W_tok (C∈{16,32}) | W_tok | backlog bursts, coarsely | Quantization; up to 2.5-s downward lag. |
| AIMD on the observed hit rate | TTFT labels | drift | Label delay ≈ TTFT, plus noisy rate estimates; oscillation. |
| **Slack-tracking state feedback** C = R + s(x) | W_tok, R, θ̂ | per-arrival backlog feasibility, with no new actuator | Between-tick over-admission is bounded (§6.2). Size bias is strengthened, so fairness needs monitoring. |
| Deadline gate in the gateway (new actuator) | T̂_i | exact per-request feasibility | Same as above, without tick error. Needs gateway change and approval. |
| Linear MPC | linear plant | — | The plant is hybrid/event-driven (first-fit, set membership, asymmetric drain). A linearization around a cap misrepresents rejection. Not recommended. |
| Nonlinear scenario MPC over {16,32} (roadmap) | rollout model + scenarios | horizon coupling via residence | Value is limited if T_i is fixed at admission (F1). Predicted to land between fixed and slack tracking. |
| Robust/tube MPC | disturbance bounds | worst case | w_P is unbounded, so the bounds are either arbitrary or infinitely conservative. Not recommended as primary. |
| Stochastic (chance-constrained) MPC | calibrated P̂ | margin | Reasonable outer loop. Calibration under policy shift is the risk (§5). |
| Learning (RL, contextual bandit) | off-policy data with action support | could learn the index | Evidence has only two caps, no randomized per-request actions and no counterfactual labels. Off-policy RL is unidentified. Bandits over the margin with switchback randomization are feasible later. |
| **Hierarchical (recommended)** | the above | inner feasibility + outer drift/horizon | Interface complexity, mitigated by module boundaries (§9). |

**Why MPC on C alone is likely dominated [D, falsifiable].** Under Poisson
overload, the average-reward optimal admission policy is stationary in the state.
If F1 holds (T_i fixed at admission), most of the value of that policy is
captured by per-arrival feasibility on W_tok. A horizon only adds value through θ2
coupling, which is small (+18 ms per 10 decoders [M]), and through nonstationarity,
which is absent in the evidence. **Prediction P1:** a scenario MPC over {16,32}
will not beat a tuned slack-tracking controller by more than the replicate noise.
P1 is falsified if, in the §7 switchback, the MPC beats it by ≥Δ_min.

---

## 5. Uncertainty, calibration, OOD, nonstationarity

### 5.1 TTFT predictive distribution [P]

Model T_i = μ_θ(x, p̂) + ε_i. The admission or slack rule uses a quantile:
admit-feasible iff μ̂ + q̂_{1−α}(x) ≤ τ.

- **Calibration.** Use split-conformal residual quantiles from a CAL set disjoint
  from TRAIN. Use a Mondrian partition by W_tok tercile, because heteroscedasticity
  is visible: MAE is 0.03 s at C16 and 0.09 s at C96 [M].
- **Validity assumption.** Marginal coverage ≥ 1−α requires exchangeability between
  CAL and deployment residuals given the bin. A new policy shifts the distribution
  of W_tok (covariate shift). Mondrian bins on W_tok guard only the binned
  covariate. A shift in ε given the bin, e.g. from θ drift, violates the
  assumption.
- **Monitoring.** Therefore monitor online coverage with a binomial test per 200
  admitted requests, and fall back on violation. Do not claim coverage for
  policies never run during CAL.
- **α is an economic margin, not a guarantee.** Choose it on TRAIN by a goodput
  Pareto sweep. Then freeze it.

### 5.2 Decode-length uncertainty

All evidence has o ≡ 128 [M], so for natural-EOS workloads:

1. R overstates the residual decode load.
2. Residence shortens and D_n dynamics change.
3. θ1 transfer is untested.

The TTFT law (W_tok-based) is structurally independent of o and is the most likely
component to transfer. The residence model is not. Treat any non-degenerate o
distribution as OOD until a TRAIN session with natural EOS (realistic prompts) is
collected. Collecting it is part of E1 (§7).

### 5.3 Session drift and online identification [P]

The per-session spread is θ1 ∈ [0.0455, 0.0635] s/kTok. At W_tok + p = 8 kTok, that
moves the TTFT prediction by 0.14 s — 29% of τ. Fixed coefficients are inadequate.

Use recursive least squares (RLS) with forgetting factor λ_f on (θ0, θ1, θ2). Labels
arrive at f_i, so information arrives at ≈5 labels/s.

- **Excitation.** Under a feasibility controller, W_tok is restricted to [0, K],
  which reduces excitation. Inject a small randomized dither on the slack (±5%,
  logged, seeded) so θ remains identified.
- **Selection bias.** RLS sees only admitted requests. Under FCFS, T_i depends on
  the admitted state, which is fully observed, so selection on x does not bias θ.
  Selection on ε would, but admission does not condition on ε. This is a [D]
  argument contingent on F1.
- **Validity box.** If θ̂ leaves the TRAIN box, apply fallback rule 1.

### 5.4 Nonstationary arrivals and regime OOD

The minimum 5-s offered rate in evidence is 20.4 req/s, far above the ≈5–7 req/s
admission capacity [M]. **Light-load and load-crossing regimes have never been
observed.** In light load, the feasibility controller rarely binds and the R ≤ C
rule may bind instead. Behavior there is extrapolation.

Required: a TRAIN session using time-scaled trace windows (e.g. 0.2×, 0.5×, 1×
arrival rate, constructed from non-C Azure windows) before claiming generality. The
arrival forecast only enters the outer MPC, and only matters near the crossing.

### 5.5 Statistical claims: stated assumptions

- AUC/MAE in §0 are leave-one-run-out across 12 captures of **one trace on two
  sessions/hosts**. Windows and runs are correlated, and all outcomes were already
  inspected. This is development evidence, not a population estimate.
- Per-window noise SD is computed from same-cap, same-window paired differences
  (2–3 replicate runs), divided by √2. With 10 windows per pair it is a rough
  estimate, not a noise floor.

---

## 6. State justification and ablations

For each candidate state: necessity / observability / delay & noise /
identifiability / incremental value / rejecting ablation.

| State | Necessity | Observability | Delay / noise | Identifiability | Incremental value | Ablation that rejects it |
|---|---|---|---|---|---|---|
| **W_tok** (+ own p̂) | Determines T_i under FCFS [D] | exact at the gateway; needs a read-only metric outside | 0 at the gateway; p̂ tokenizer error | θ1 well excited (W_tok 0→10k) [M] | AUC 0.685→0.976 over R-only [M] | Reject if a held-session AUC gain < 0.05 or MAE gain < 20 ms |
| R | Actuator constraint (exact) | exact | 0 | n/a (accounting) | Needed for C = R + s; weak predictor of T [M] | Never removed (constraint state) |
| D_n | Decode interference (θ2) and residence | exact | 0 | Confounded with C/session; needs multi-cap TRAIN | MAE −2 to −13 ms [M] | Drop if held MAE gain < 5 ms and MPC ranking unchanged |
| A_W, receive→accept lag | Captures stalls | exact | 0 | Weak | Small AUC gain (0.976→0.981) [M] | Drop unless it improves held stall-window prediction |
| θ̂ (online) | Session drift 0.0455–0.0635 [M] | latent; RLS | ~1 s label lag | Needs dither (§5.3) | **Unmeasured** [P] | Compare RLS vs frozen θ on fresh sessions; drop if MAE gain < 10 ms |
| Engine gauges (waiting/running, 1 Hz) | Possibly partial-prefill progress | aged 0–1 s, aggregate | high | Collinear with W_n | **Unmeasured** [P] | F3: drop if held MAE gain < 10 ms |
| Engine pre-ADD snapshot / state envelope | ms-level engine timing | post-admission only | ≤5 ms age gate | Native coverage unqualified [H] | **Unmeasured** for admission decisions | F3: same threshold; it must be computable *before* the admission decision |
| Rolling activity/outcome histories (flow model) | Flow-model heads used them [H] | causal, but endogenous | window lag | Flow diagnosis shows stale-history error [H] | Superseded if W_tok explains T [P] | Drop if no held gain over W_tok-only |
| Arrival forecast | Only near the load crossing | from history | — | Poisson-like evidence → weak | ≈0 in overload [D] | Drop unless the time-scaled TRAIN shows gain |
| Decode remaining Σ(m − ô) | Residence under natural EOS | needs SSE token counting | 0 | Untested (o ≡ 128) | n/a now | Required only once o is non-degenerate |

### 6.1 Minimal state

x = (W_tok, W_n, D_n, R, C, θ̂) with exogenous marks (p̂_i, m_i) per arrival. Every
other candidate must earn inclusion by the listed ablation on fresh sessions.

### 6.2 Slack-tracking controller with the existing actuator [D]

Let K̂ = (τ − m_α − θ̂0 − θ̂2·D_n)/θ̂1 (tokens), and set

  s(t) = α + γ·m + β·max(0, K̂ − W_tok(t)),  C(t) = R(t) + s(t).

Between ticks, admitted cost accumulates: Σ_{admitted since tick} k_j ≤ s + Σ_{released since tick} k_j. Hence

  admitted prompt tokens since the tick ≤ (s − n·(α + γm) + released)/β ≤ K̂ − W_tok(t) + released/β.

The emulation of "admit iff W_tok + p̂ ≤ K̂" is conservative except for cost
released by terminations within one tick. At 50 ms that is ≈0.3 terminations ×
≈1.8 cu ≈ 0.5 cu ≈ 700 tokens of over-admission, an upper bound [M rates].

Two race conditions:

1. R moves between the controller's read and the application. The endpoint returns
   `current_admitted_cost` and the version, so the controller can detect error; it
   cannot prevent it.
2. Prefill drain between ticks makes the bound conservative, not unsafe.

A gateway-side "slack" endpoint, or the gate itself, removes both. That is a new
actuator, listed as an alternative.

**Scope flag.** This uses the existing endpoint at ≈20 Hz with continuous C. That is
outside the studied {16,32} action support. Qualifying it requires the update-lag,
lock-contention and telemetry-volume checks in E1.

---

## 7. Identification and experimental design

### 7.1 Data roles and chronological splits [P]

- **TRAIN:** the existing development trace window (2024-05-12 00:01, 300 s) and its
  13 captures, already inspected. New sessions may reuse it, plus additional
  non-C Azure windows strictly *earlier* than any VALID window.
- **CAL:** one fresh non-C window and fresh sessions, used only for conformal
  quantiles, the CUSUM threshold and fixing α / λ_sw from the TRAIN-chosen
  candidate set.
- **VALID:** later non-C windows and fresh sessions, used for the switchback policy
  comparison. Freeze everything before collection.
- **FINAL:** reserved trace C, used once with the frozen policy, never for tuning.

Sessions are also chronological: no parameter may be fit on a session that started
after the session it is evaluated on.

### 7.2 Why switchback [D]

Admission decisions interfere: the backlog couples requests, and residence couples
seconds. Per-request randomization of a policy is therefore invalid for policy
evaluation. Whole-episode comparisons are confounded by large session effects:
C32 whole-run goodput spans 3.73–5.08 [H/M].

A **randomized switchback** within each session has these properties:

- Policy blocks of length L, with the order randomized and balanced (ABBA or
  permuted).
- Cohorts are assigned by scheduled arrival.
- It blocks the session effect and handles interference when L ≫ the carryover
  (≈3 s residence plus 2.5 s drain).

Proposed L = 30 s. The first 5 s are analysed both included (primary, policy-at-arrival)
and excluded (sensitivity). Inference is design-based: a randomization test over the
assignment, plus session-level random effects for between-session generalization.

### 7.3 Power [P, from M noise]

Measured per-30-s-window replicate SD is σ_b ≈ 0.57–0.76 req/s, so the paired block
difference SD is σ_d ≈ 1.0. To detect Δ = 0.4 req/s (≈8% of 5.0) with α = 0.05
(two-sided) and power 0.8:

  n ≈ (1.96 + 0.84)²·σ_d²/Δ² ≈ 49 block pairs ≈ 49 min of GPU time per policy pair
  per session.

Run three sessions (to bound host/session generalization) × {fixed C*, slack
tracking, discrete MPC{16,32}}. That is ≈2.5 h per session. Δ_min = 0.25 req/s
would need ≈125 pairs, which is likely unaffordable. **Pre-register Δ_min = 0.4
req/s**, and state that smaller gains are undetectable by this design.

### 7.4 Experiments

- **E0 (CPU, now, no GPU).**
  - E0.1: reproduce §0 [M] numbers from the archives (done here).
  - E0.2: incremental value of 1-Hz engine gauges over W_tok, using
    `engine_observations.jsonl` with as-of joins at each accept (F3, gauge part).
  - E0.3: build an event-driven simulator (first-fit + FCFS TTFT law + residence
    model) fitted on cap-range sessions only. Validate it on low-cap sessions: it
    must reproduce the per-run goodput ordering and level within ±15% before use for
    controller design. A simulator result is never acceptance evidence.
  - E0.4: shadow replay — compute controller decisions on logged event streams to
    test the interfaces and the no-future-leakage property.
- **E1 (GPU, authorization required).** One session, TRAIN-role:
  - Gateway-read latency at 20 Hz, budget update lag and lock contention.
  - Engine B recorded and pinned.
  - Natural-EOS realistic prompts (one block) to break o ≡ 128.
  - Time-scaled arrival blocks (0.2×, 0.5×, 1×).
  - Small randomized slack dither for identification.

  Stop if update lag > 50 ms at p99, or if TTFT-law AUC < 0.95 on the fresh session.
- **E2 (GPU).** CAL session(s): freeze θ priors, α, λ_sw, CUSUM h and the fairness δ.
- **E3 (GPU).** VALID switchback, 3 sessions × 3 arms, as in §7.3.
- **E4.** Final trace C, single frozen run set, only after E3 passes.

### 7.5 Counterfactual limits

- Factual prediction accuracy (§0) does not identify the effect of a different
  policy on goodput.
- F1 makes the TTFT law *structurally* plausible under a policy change: it depends
  only on the admitted state. But θ2 and residence under a new admitted mix are
  extrapolations.
- The only valid policy comparisons are randomized (E3). Offline "flagged misses"
  (e.g. 197/263 flagged requests at C32-r1 were true misses [M]) are **not** gains:
  rejecting them changes everyone else's state.

### 7.6 Negative-result criteria (pre-registered)

- If the 95% CI for (slack tracking − C*) lies below +0.4 req/s (or includes 0 with
  an upper bound < 0.4): declare no practical benefit, publish, and stop. Do not
  add features.
- If the MPC does not beat slack tracking by ≥ 0.4: keep the simpler controller.
- If any per-stratum goodput falls below (1 − δ)× baseline: the win is
  disqualified as selection.
- If E1 shows AUC < 0.95 on a fresh session: the gateway-state premise is falsified.
  Return to state selection before any controller GPU time.

---

## 8. Falsification tests (targeted)

### 8.1 Model and data (F)

| ID | Test | Pass condition | Status |
|---|---|---|---|
| F1 | Effect of tokens admitted after i on T_i | coefficient ≪ ahead coefficient; MAE gain < 5 ms | [M] 0.0014–0.0155 vs 0.048–0.060; MAE gain ≤ 4.2 ms (pass, dev) |
| F2 | TTFT-law held-session discrimination | AUC ≥ 0.95, MAE ≤ 0.10 s on each fresh session | [M] dev 0.959–0.995 / 0.033–0.093 s; fresh pending |
| F3 | Incremental value of engine gauges / pre-ADD envelope at decision time | MAE gain ≥ 10 ms or Brier gain ≥ 0.005, *computable before admission* | [P] |
| F4 | Calibration under the new policy | online conformal coverage within the binomial 99% band | [P] (E3) |
| F5 | Policy gain | switchback CI lower bound > 0, point estimate ≥ Δ_min, fairness constraint met | [P] (E3) |
| F6 | Decode-length transfer | TTFT law AUC ≥ 0.95 on the natural-EOS block | [P] (E1) |
| F7 | Light-load transfer | slack tracking ≥ fixed C* in 0.2× / 0.5× blocks (no harm) | [P] (E1/E3) |

### 8.2 Implementation (T, CPU)

- **T1** Tracker R equals the gateway `current_admitted_cost` at every logged snapshot,
  within 1e-6 cu.
- **T2** Slack emulation: on synthetic arrivals, cumulative admitted tokens between
  ticks ≤ K̂ − W_tok + released/β.
- **T3** No future leakage: perturbing any event after t leaves the decision at t
  unchanged.
- **T4** Fallback triggers: stale state, update lag, CUSUM, θ out of box, OOD
  lengths → fixed C*, logged.
- **T5** Simulator cap reduction never evicts; drain equals the gateway rule exactly
  on recorded A/B schedules (compare to the [M] drain times).
- **T6** Monotonicity: T̂ is nondecreasing in W_tok, p̂ and D_n.
- **T7** Terminal constraint: every MPC plan satisfies R(t+H) ≤ C* + k̄ in the rollout.

---

## 9. Module boundaries and interfaces [P]

There are no product edits in this round. The following is the proposal for Mayor
bead slicing.

1. **`mpc/state.py` — `GatewayStateTracker`**
   - `apply(event)` consumes accepted, first_content and terminal events.
   - `snapshot(t) -> GatewayState` returns W_tok, W_n, D_n, R, C, version and age.
   - Invariant: R equals Σk.
2. **`mpc/ttft_model.py` — `TTFTModel`**
   - `predict(state, p_hat) -> (mean, q_alpha)`.
   - `update(label)` performs RLS, with a frozen mode.
   - `validity() -> bool` checks the θ box and the coverage monitor.
3. **`mpc/policy.py` — `Policy.decide(state, t) -> BudgetCommand | Hold`**
   - Implementations: `Fixed`, `Hysteresis`, `SlackTracking`,
     `ScenarioMPC(actions, H, Δ, scenarios)`.
4. **`mpc/actuator.py` — `BudgetActuator.apply(C) -> Ack(version, applied_at, R_at_apply, lag)`**
   - Wraps `POST /admin/budget` with timeout and lag accounting. Live actuation
     stays disabled by default.
5. **`mpc/supervisor.py` — `Supervisor`**
   - Health monitors (age, lag, CUSUM, OOD) and the fallback ladder.
   - Writes a decision log of inputs, outputs, hashes and timing.
6. **`mpc/simulator.py`** — development-only event simulator implementing §2
   exactly (first-fit, no eviction, FCFS law, residence model).
7. **`mpc/switchback.py`** — seeded, pre-registered block assignment, and an
   evaluator for all-offered goodput by arrival cohort and policy-at-arrival.
8. **Gateway (separate approved bead)**
   - Read-only `/metrics` fields: `awaiting_first_tokens` (W_tok) and
     `decoding_requests`. This extends the observation channel, not the actuator.
   - Optional, flagged and default-off: a deadline-gate admission predicate (new
     actuator).

---

## 10. Recommended architecture, alternatives, open hypotheses

**Recommended architecture: H-SF+MPC.** Inner, existing first-fit rule → middle,
slack-tracking state feedback on W_tok with online θ̂ (existing actuator, new action
support) → outer, optional scenario MPC over a margin or C ceiling with H = 3 s →
supervisor with fixed-C* fallback.

Evaluate in this order: fixed C* → slack tracking → (only if F5 passes and residual
headroom remains) outer MPC.

**Alternatives:**

- (a) The roadmap's discrete {16,32} MPC alone. This is the baseline arm. P1
  predicts it is dominated.
- (b) A deadline gate in the gateway (new actuator). This is the cleanest
  implementation of the same physics.
- (c) A flow/history model per the earlier flow diagnosis. Its stale-history problem
  [H] is sidestepped if the W_tok law holds.
- (d) Bandit tuning of margin α via switchback. Feasible after E3.

**Open hypotheses (each with a discriminating test):**

- **H1:** FCFS-at-admission holds on fresh sessions (F1/F2).
- **H2:** Slack tracking beats fixed C* by ≥ 0.4 req/s on VALID without stratum
  harm (F5). The fluid ceiling (≈7.4 req/s admitted throughput) bounds any claim.
- **H3:** The session effect (C32: 3.73 vs 5.08) is largely backlog-mediated, so
  feedback on W_tok reduces between-session goodput variance relative to fixed C*.
  Test: compare between-session SD of arms in E3. Note lc32 had mean W_tok of 4.6k
  vs 2.8k at the same C [M]; the cause is unexplained.
- **H4:** Engine-level state adds < 10 ms MAE at the admission decision (F3).
- **H5:** The TTFT law transfers to natural EOS and lighter load (F6/F7). Unknown
  today.

Unresolved and outside my evidence: the engine B used in the admission captures; the
cause of the 5.9-s stall; native qualification of the pre-ADD channel; and whether
real (non-synthetic) prompts change θ1 via tokenization or attention cost.

---

## 11. Reproduction of [M] numbers

```bash
python results/cap-range-20261020/extract_bundle.py --output-dir /tmp/obs-mpc/cap-range
python results/state-model-20261021/extract_evidence.py --output-dir /tmp/obs-mpc/lowcap
python experiments/mpc-research/proposals/opus-round1-analysis.py \
  /tmp/obs-mpc/cap-range /tmp/obs-mpc/lowcap > /tmp/obs-mpc/analysis.json
```

The script is pure Python (no NumPy) and reads only the archived captures. It
computes per-run goodput, admitted misses, acceptance by cost quartile, residence and
TTFT quantiles, drain after reductions, 30-s window goodput, and leave-one-run-out
TTFT models (`cap_only`, `backlog_only`, `backlog_decode`, `queue_causal`, `queue`).

`queue` uses the client schedule lag, which is not gateway-causal. It is reported
only to show that it adds nothing over `queue_causal`.

The cap16-r2 capture (failed load fidelity [H]) is summarized but excluded from
cross-validation. The output also contains `fcfs_after_test` (F1) and `per_run_backlog_fit`
(per-run θ0, θ1 and output-token check).

Selected [M] values (all development data):

- **Per-run goodput:** cap-range 16r1 4.130; 32r1 5.080; 32r2 4.590; 64r1 2.267;
  96r1 3.330; 96r2 2.667. Low-cap: F32 3.733; F16 4.240; A1 4.200; A2 3.530;
  B1 4.290; B2 4.483. These match the committed reports exactly.
- **Admitted-miss fraction:** 0.021–0.023 at C16; 0.130–0.293 at C32; 0.65 at C64;
  0.54–0.64 at C96.
- **Mean W_tok at accept:** ≈1.2k at C16; 2.8k–4.6k at C32; 7.8k–10.1k at C64–96.
- **Successful output tokens:** 128 for 100% of successes in every capture.
