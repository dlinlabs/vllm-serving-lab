# Opus Round 2: adversarial review of astra, gemini and gpt (plus self-corrections)

Researcher: obsidian / opus (`claude-opus-5.5`). Epic pr-mpc. Source freeze `8e92371`.
Round 2 of the Mayor protocol. Peer files read in full:

- `polecats/quartz/.../astra-round1.md`
- `polecats/jasper/.../gemini-round1.md`
- `polecats/onyx/.../gpt-round1.md`

Round 1 files are frozen and unedited. No GPU, network, final trace C, product edit,
commit or push was used.

Labels:

- **[M]** measured here offline on the 12 qualified development captures (cap16-r2
  excluded as in Round 1). This is development reuse, not policy gain.
- **[S]** source at `8e92371`.
- **[D]** derivation.
- **[P]** proposed experiment.

## 0. Executed counterexamples (exact command and result)

```
cd experiments/mpc-research/proposals
python3 opus-round2-checks.py /tmp/obs-mpc/cap-range /tmp/obs-mpc/lowcap > /tmp/obs-mpc/r2.json
# exit 0, 3.5 s wall. Inputs restored with the repository's extract_bundle.py and
# extract_evidence.py, as in Round 1.
```

The script `opus-round2-checks.py` sits next to this report. It imports the frozen
`opus-round1-analysis.py` read-only. The Mayor independently reproduced that script's
AUC/MAE in `results/mpc-research-cpu/development-backlog-reanalysis.json`.

Only C6 is timing-dependent; every other output is deterministic.

**C1. Old-cohort externality upper estimate [M].**
For each admitted request i, "after tokens" means the prompt tokens admitted after i
but before i's first content. TTFT is regressed on own backlog W_tok+p, the decode
count D_n, and after tokens. b_after is the after-tokens coefficient. A flip is a
successful request with T_i > 0.5 and T_i − b_after·after_i ≤ 0.5.

| Run | b_after (s/kTok) | Mean after (kTok) | Flips / hits | Flips (req/s) |
|---|---|---|---|---|
| C16 (2 runs) | 0.0107, 0.0155 | 0.89–0.93 | 0.08%–0.71% | 0.003–0.03 |
| C32 (3 runs) | 0.0077–0.0080 | 2.2–3.95 | 2.1%–4.6% | 0.11–0.17 |
| A/B switch (4 runs) | 0.0046–0.0099 | 1.95–3.0 | 1.35%–3.9% | 0.06–0.14 |
| C64/C96 (3 runs) | 0.0015–0.0046 | 6.7–8.8 | 2.0%–8.5% | 0.05–0.28 |

First-token hits that later failed: **0 of 13,962 hits**. The one-sided 95% upper
bound (rule of three) is 2.1·10⁻⁴.

Caveat: b_after is biased *upward*. A slower request has a longer window
[acc, fc], so it mechanically collects more after-tokens. Flips are therefore an
upper estimate. They also describe removing *all* later admissions, which is more
extreme than a 16↔32 choice.

**C2. Little's-law test of my Round 1 "≈7.4 req/s ceiling" [M+D].**
Little's law: λ = mean R / mean(k·S), where R is reserved cost, k request cost and S
residence time.

| Run | Measured admission rate | λ from all admitted | λ if capacity went only to the cheapest decile |
|---|---|---|---|
| C96 (2 runs) | 7.28 / 7.42 req/s | 6.83 / 7.03 | **12.65 / 12.90** |
| C16 (2 runs) | 4.23 / 4.33 req/s | 3.49 / 3.57 | 6.06 / 6.07 |

**C3. Cost arithmetic [S+D].**

- With max_tokens = 2048, c(p = 907) = 11.61 (versus 1.61 at 128 tokens).
- At C = 16 with max_tokens = 2048, any prompt longer than **6,694 tokens is
  permanently inadmissible, even when R = 0**. At 128 tokens the limit is 19,886.
- Resident tokens per cost unit, (p + 128)/c(p), increases in p:
  - 1,136.6 at p_max = 6,674
  - 1,276.8 at p_max = 32,768
- Hence, at C = 32, resident tokens ≤ **36,370** (trace p_max) or 40,858 (32k context).
  This is a deterministic KV sufficient condition (§3, G-3).

**C4. Exact binomial [D].**

- Gemini's gate "coverage ≥ 0.88" rejects a *perfectly calibrated* 90% envelope with
  probability **0.230 / 0.198 / 0.145 / 0.017** at n = 50 / 100 / 200 / 1000
  independent scores. Correlated windows make this worse.
- Zero observed violations bound the rate at the one-sided 95% level only when:
  - n ≥ **2,995** independent episodes for ε = 0.001
  - n ≥ 299 for ε = 0.01. This confirms Astra's 299, upper bound 0.00997.

**C5. Plan enumeration [D].**

| Proposal | Plan count | Claim checked |
|---|---|---|
| Gemini (pruned) | 15 | claim correct |
| Astra (one switch, 20 slots) | 40 | claim correct |
| GPT (all 5-step) | 32 | — |
| Opus | 8 / 27 | — |

**C6. Minimum rollout cost [M, this CPU, median of 5].**
Pure Python. The rollout simulates the gateway's first-fit admission over Poisson
arrivals at 25.48 req/s, with prompts bootstrapped from the trace, the linear backlog
TTFT model and constant residence. This is the *cheapest* model any of the four
proposals could run.

| Proposal | Rollouts | ms | Proposal's budget |
|---|---|---|---|
| Gemini 15×8, H = 4 | 120 | 4.4 | 20 ms (claims "< 4 ms, benchmarked") |
| Opus 8×32, H = 3 | 256 | 6.7 | 10 ms |
| GPT 32×16, H = 5 | 512 | 21.8 | 100 ms |
| Astra 40×32, H = 5, 250 ms | 1,280 | **57.1** | **20 ms** |

**C7. Actuation race [D, executed asyncio model of the `/admin/budget` semantics].**
Source facts [S] (`gateway.py:588-630`): the endpoint takes `state_lock`, overwrites
the budget and increments the version. There is no expected-version check, expiry or
request id.

Simulation:

| Time (ms) | Event | Budget | Version |
|---|---|---|---|
| 0 | MPC write "16" sent (60 ms transit) | 32 (initial) | 0 |
| 20 | Client times out | — | — |
| 21 | Controller sends fallback "32", acked | 32 | 1 |
| 61 | Timed-out write applies | **16** | **2** |

The fallback was acknowledged and then silently overwritten.

Second case: a CPU-bound in-process "solver" inside `asyncio.wait_for(timeout=20 ms)`
**returned its value after 50.1 ms and no TimeoutError was raised**. The event loop
was blocked for the whole time.

## 1. Cross-cutting findings (answers to the Mayor's checklist)

| # | Question | Finding | Who is affected |
|---|---|---|---|
| X1 | Do measured claims have evidence? | Gemini's "<4 ms benchmarked", "Delay <0.5 ms / 50–200 ms", "Fano ≫ 1", "Native CPU acceptance suite passes 289 tests" are not evidenced. The audit says 289 is *five selected files, not full native acceptance*. The dev trace Fano factor is 0.94 [M R1]. Astra and GPT label all numbers as historical or source and ran no data. GPT's arithmetic checks verify (I recomputed 41.3112 / 35.161 / 6.1502). | gemini (major); astra/gpt (clean but data-free) |
| X2 | Are empirical maxima valid upper bounds? | **No, including mine.** My "≈7.4 req/s goodput ceiling for any admission-only policy" is falsified by C2: a cost-selective admitter has Little capacity 12.7–12.9 req/s at C96. 7.4 is only the best *fixed-cap* admission rate observed. Gemini's "H = 5 captures the complete drain transient" uses the measured 128-token residence. Under natural EOS with max_tokens 2048, residence and c both scale ~16× (C3), so no captured drain bounds it. Astra and GPT correctly refuse TRAIN extrema as bounds. | opus (self), gemini |
| X3 | Do terminal constraints coexist with an always-feasible fallback? | Gemini: no terminal term, and its fallback can *hold 48* (fallback = C_prev when C_admit > 24), an unqualified level. Opus: my hard terminal R(t+H) ≤ C* + k̄ with "hold C* is always admissible" is **contradictory** once 𝒰 contains a level above C* (derivation §4.1). Astra and GPT use soft terminal plus continuation, which is consistent. However, GPT's interval rule makes the MPC degenerate (§3, G-4). | opus (self), gemini, gpt |
| X4 | Does the reward include old cohorts and eventual success? | Old cohorts matter **more than my Round 1 assumed** but are a first-order linear term. C1 bounds them at ≤0.17 req/s at C32, which is 40% of my own Δ_min = 0.4. Gemini omits them; I omitted them. Astra (ΔV_old) and GPT (cohort scorer) are right to include them; GPT's .49→.51 s example is typical in magnitude (b_after·mean after ≈ 0.008 × 3 kTok ≈ 24 ms). Eventual-success failure has **zero** dev support (0/13,962); keep it as accounting, not as a modeling priority. | opus (self), gemini; vindicates astra/gpt |
| X5 | Do sequential coverage tests hold? | Gemini's coverage ≥ 0.88 gate fails a correct 90% envelope 20% of the time at n = 100 (C4). Run per model revision, it acts as a selection filter favouring *over*-wide envelopes. My Round 1 KS test at α = 0.01 per completion over a rolling 200 (~5 tests/s) is an uncorrected repeated test that will false-alarm within an episode. Replace it with a one-sided CUSUM at a declared ARL₀ (§4.4). Astra and GPT correctly forbid peeking without a sequential method. | gemini, opus (self) |
| X6 | Do action-shift causal claims hold? | Gemini's "Drain Penalty Inversion" test and "Deduction 2 (guaranteed drain loss)" compare a transient to nothing. The rejected offers during a 32→16 drain repay over-admission made under 32. The causal estimand is policy versus policy, not transient versus subsequent interval (§2, M-5). Astra and GPT state carryover correctly. | gemini |
| X7 | Can absent KV observations support chance constraints? | No. Gemini sets P(θ_kv ≥ 0.9) ≤ 0.001 while excluding θ_kv as causally unobservable. No KV series exists in the captures, and demonstrating ε = 0.001 needs ≥ 2,995 independent zero-violation episodes (C4). Replace it with the deterministic reservation→token bound of C3 (36.4k tokens at C32), checked against the configured KV capacity. | gemini |
| X8 | Can timed-out commands later apply? | **Yes, and acknowledgement-based reconciliation cannot close the window** (C7). A queued write can land after the fallback ack and after a `/metrics` read showing the fallback. Every proposal's fallback (all four, including mine) is therefore not guaranteed to be the final applied state. Only a server-side guard (expected_version CAS or an expiry deadline) bounds it. That is a product change requiring authorization. | all four |

## 2. Gemini (BDA-MPC): findings

| ID | Severity | Finding | Counterexample / derivation |
|---|---|---|---|
| M-1 | Critical | **TTFT model on the wrong state.** P(T ≤ 0.5 \| C_admit) = Φ((0.5 − μ(C_admit))/σ) uses reserved cost. TTFT is set by prompt tokens awaiting first content. | Leave-one-run-out AUC: R + own prompt 0.586–0.775 versus W_tok + p 0.959–0.995 [M R1, Mayor-reproduced]. Arithmetic: at C_admit = 30, case A is all streaming (W_tok = 0), so T ≈ 0.065 + 0.05·0.907 = **0.11 s (hit)**. Case B is 12 awaiting 2k-token prompts (cost 29.3, W_tok = 24k), so T ≈ 0.065 + 0.05·24.9 = **1.31 s (miss)**. Same C_admit, opposite label. Gemini's TTFT decomposition also omits engine queue wait, which is the dominant term. |
| M-2 | Critical | **In-process controller with an in-process watchdog.** The state filter reads `gateway.state` under `state_lock`, and the loop runs as an async task. A synchronous solver blocks admissions and SSE relays, and `wait_for` cannot preempt it. | C7: the 20 ms watchdog returned the solver value at 50.1 ms, no timeout. The "≤20 ms → fallback" contract is unenforceable. Every ms of solver time is added to the gateway latency of all requests. |
| M-3 | Critical | **Fallback is not a fallback.** u_fb = 24 if C_admit ≤ 24, else C_prev. | (a) C24 has never been captured; it is unqualified. (b) If C_prev = 48 (in §1.1's 𝒰) and C_admit = 30, the fallback holds 48. (c) A watchdog write races the timed-out solver's write (C7). |
| M-4 | High | **KV chance constraint on an excluded, unobserved state.** | X7. ε = 0.001 needs 2,995 episodes. The reservation→token bound (C3) is a deterministic substitute. |
| M-5 | High | **Drain semantics misused as a causal effect.** Deduction 2 calls the drain loss "guaranteed"; the falsification test compares transient loss with later TTFT savings. | The steady-state admission rate λ_a(C) ≈ (C − κ)/E[kS] is independent of switching history. Offers rejected during drain repay the extra R admitted under 32. Measured drain after 32→16 was 0.0006–2.41 s [M R1], not "zero goodput for H". The correct estimand is randomized policy versus policy. |
| M-6 | High | **Static-domination test is biased toward "retain fixed".** It uses 2× the replicate SD of goodput, not the SE of a paired difference. | C32 session goodputs 3.733 / 4.59 / 5.08 give SD = 0.68, so 2σ = 1.36 req/s, which is 30% of the C32 mean. Any realistic gain fails regardless of n. The test should use the paired-difference CI with a pre-set Δ. |
| M-7 | High | **Dimensional error in drain model.** μ_drain = min(B, N·t̄_token⁻¹). | B is tokens/iteration, N/t̄ is tokens/s, and C_admit drains in cost/s. Three units, no conversions. Reservation release is event-driven at terminal (whole c_j), not a rate. |
| M-8 | Medium | **Action set inconsistent across sections.** | 𝒰 = {16,24,32,48} (§1.1), bounds [16, 64] (§3.2), {16,24,32} (§7), [16, 48] (§4). The slew limit of 16 cannot bind on {16,24,32}. |
| M-9 | Medium | **Coverage gate statistics.** | C4: 20% false-fail at n = 100. Conformal on pooled absolute residuals across caps and sessions breaks exchangeability (session effect ≈ cap effect [M R1]). |
| M-10 | Medium | **No terminal term, H = 4, reward counts "status = success" of last-step cohorts.** | Last-step admissions resolve 2–7 s after H [M R1 residence], so the reward is unobservable inside the horizon. The objective is undefined for them without a continuation rule. |
| M-11 | Low | Fano ≫ 1 and "max_tokens often 2048" are asserted for the dev workload. | Measured Fano 0.94, and 100% of outputs are exactly 128 tokens [M R1]. At 2048, C16 starves prompts > 6,694 tokens (C3), so the action changes meaning. |
| M-12 | Low | "Fact 4: 289-test native CPU acceptance." | The audit says five selected files, not full native acceptance. |

**Verdict on gemini:** the discrete-enumeration skeleton is computationally sound (15
plans verified; 4.4 ms in C6). The state, TTFT model, fallback, chance constraint and
falsification tests all need replacement before any of its predictions are
discriminating.

## 3. GPT (belief-state scenario MPC): findings

| ID | Severity | Finding | Counterexample / derivation |
|---|---|---|---|
| G-1 | Confirmed (agree) | Old-cohort externality and eventual-success accounting. | C1 supports the term's size: up to 0.17 req/s at C32 (upper estimate), and the .49→.51 example is typical. Eventual-success failure: 0/13,962 in dev. Keep the accounting rule but spend no modeling budget on it until natural-EOS data exist. |
| G-2 | High | **Old-cohort value does not need a full continuation rollout.** Measured linear coupling ΔT_i ≈ b_after·Δ(after tokens) with b_after 0.005–0.016 s/kTok [M]. | ΔV_old(U) ≈ Σ_{i unresolved} [F̂_i(0.5 − T̂_i) − F̂_i(0.5 − T̂_i − b_after·ΔA_i(U))], where ΔA_i(U) is the plan's change in later-admitted tokens before i's predicted first content. Closed form per plan, O(#unresolved). Discriminating test: D-1. |
| G-3 | High | **Per-decision "lower calibrated gap > Δ_min" gate degenerates to always-fallback.** | Per 1-s decision with H = 5, a policy gain of 0.4 req/s is ≈ 2 hits per plan. The historical cap32 optimistic bias of +0.3163 req/s [historical] is ≈ 1.6 hits over 5 s. The structural error alone is ≈ the gap, so the lower bound essentially never clears Δ_min. Value accrues over hundreds of decisions while the gate tests each one. Gate per decision on *support/OOD*; establish gain at policy level (randomized episodes). |
| G-4 | High | **"Compare worst/best intervals of unresolved reward" makes plans incomparable.** | Unresolved admitted requests at C32 ≈ λ_a·S ≈ 5.8 × 2.8 ≈ 16. Interval [0, K] with K ≈ 16 dwarfs any plan gap of ≈ 2 over 5 s. Rule forces abstention every tick. It needs a calibrated tail model (or C1's empirical 0-failure rate), not worst-case bounds. |
| G-5 | High | **Timeout ambiguity (X8) remains unsolved by "single writer + reconciliation".** | C7: after a timed-out write, the fallback was acked and visible as version 1/32, then version 2/16 landed 40 ms later. GPT admits a lease or watchdog would be a separate feature. Since neither GPT nor Astra can issue a *guaranteed* fallback, "fallback always available" must be removed from the safety claims until CAS/expiry exists. |
| G-6 | Medium | **PI/hysteresis baselines on oldest awaiting age are weak baselines.** | Oldest age adds ≤ 0.007 AUC over W_tok, and alone is a lagging indicator [M R1 `queue_causal` vs `backlog_only`]. A fair "strong non-MPC baseline" must use the same W_tok state. Otherwise an MPC win can be a state win, not a horizon win. |
| G-7 | Medium | Compute claim is consistent: 512 rollouts ≈ 21.8 ms minimal model versus a 100 ms budget (C6). | A realistic lifecycle/hazard model is 3–10× the minimal one. Measure p99 before freezing N = 5 × 16 scenarios. |
| G-8 | Medium | No numeric sample size despite available variance inputs. | Dev data give within-session 30-s SD 0.57–0.76 and between-session SD ≈ 0.68 req/s [M R1]. A planning n is computable now (~49 block pairs per session for Δ = 0.4); refusing it defers the authorization decision. |
| G-9 | Low | (Q,D,R) aliasing example (residual 1 vs 128 steps). | Valid, but the forced-128 workload makes D-phase age a near-sufficient statistic for residual. The example motivates per-request age, not engine progress feeds. |

**Verdict on gpt:** the most internally consistent accounting of cohorts, delays and
terminal tails, and its arithmetic checks verify. Its decision rules (G-3, G-4) are so
conservative they make the planner provably inert. Its fallback/actuation claims share
the X8 hole.

## 4. Astra (partially observed scenario MPC): findings

| ID | Severity | Finding | Counterexample / derivation |
|---|---|---|---|
| A-1 | Critical | **Compute budget infeasible as specified.** 250 ms cadence, 40 plans × 32 scenarios, ≤ 20 ms. | C6: 57.1 ms for 1,280 rollouts with the *cheapest* possible model, 2.9× over budget before any semi-Markov hazards, ΔV_old continuation (another rollout per scenario) or terminal tail. Astra allows shrinking the family, but a feasible configuration is ≤ ~450 minimal rollouts, i.e. ≈ 14 plans × 32 or 40 × 11. |
| A-2 | High | **250 ms cadence versus actuator dynamics.** A downward step needs 0.0006–2.41 s to drain (up to ~10 slots); an upward step acts only at the next admissible arrival. | With one-switch plans over 20 slots, the 19 switch-time variants of a down move differ only through drain timing, which the model must predict to within 250 ms. Prediction: their value spread will be within Monte Carlo SE at 32 scenarios. Test D-3. |
| A-3 | High | **State is over-specified relative to measured decision value.** Full ledger with age bins, per-request marks and semi-Markov hazards. | W_tok + p alone gives AUC 0.959–0.995 and MAE 0.033–0.093 s; adding D_n/oldest/lag adds ≤ 0.007 AUC [M R1]. Astra's "age aliasing" falsifier (ages .49 vs .01 at equal Q/D/R) changes only old-cohort hits, bounded at ≤ 4.6% of hits at C≤32 (C1). Ablation order should start from the W_tok scalar and add the ledger only if D-1/D-2 show decision change. |
| A-4 | High | **ΔV_old "relative to continuing the fallback" doubles compute.** | G-2's closed-form linear term gives the same quantity without a second rollout family, within the measured coupling. |
| A-5 | High | **Fallback/ambiguity (X8).** "Enter action-uncertain state … model both budgets until reconciliation." | Reconciliation by `/metrics` read is not terminal: a queued write can land after the read (C7). The ambiguity has no upper duration without server expiry. "Use one frozen fixed cap on … solver failure" therefore cannot be guaranteed. |
| A-6 | Medium | **CVaR_0.9 over 32 scenarios = mean of the worst ~3.2 draws.** | It is a high-variance ranking criterion. Astra labels it diagnostic; it should be excluded from action selection until the scenario count is justified by D-3's seed-stability test. |
| A-7 | Medium | Operational rule "switch only if lower bound exceeds margin" has the same degeneracy as G-3. | Same arithmetic. |
| A-8 | Low | No numeric power / n. | As G-8. |
| A-9 | Confirmed | R ≤ max(R(0), B_max), conformal rank ⌈(n+1)(1−α)⌉, 299-episode zero-failure bound. | Verified against source (`projected_cost > budget` rejects) and C4 (0.00997). Astra's bound is tighter than my Round 1's C_max + max k. |

**Verdict on astra:** the most careful epistemics (source-cited, no overclaims, correct
statistics). Its controller is not runnable within its own budget, and its state
specification is unjustified by the one decision-relevant measurement available.

## 5. Self-review: corrections to opus-round1 (frozen; corrections recorded here)

| ID | Round 1 claim | Status | Correction |
|---|---|---|---|
| O-1 | "Ceiling on goodput for any admission-only policy ≈ 7.4 req/s" | **Wrong** (C2) | 7.4 is the best fixed-cap admission rate observed. A cost-selective policy (e.g. my own proposed deadline gate) has Little capacity up to ~12.9 req/s. No upper bound is claimed. |
| O-2 | Stage reward counts only cohorts admitted in [t, t+H); F1 "TTFT ≈ determined at admission" | **Incomplete** (C1) | F1's MAE gain (≤ 4.2 ms) was the wrong criterion. Threshold flips matter: up to 4.6% of hits at C32, 0.11–0.17 req/s (upper estimate). Add G-2's linear ΔV_old term. |
| O-3 | Hard terminal R(t+H) ≤ C* + k̄ with "hold C* is always admissible ⇒ recursively feasible" | **Contradictory** for 𝒰 ∋ C > C* | §5.1 derivation. |
| O-4 | Fallback ladder: update > 50 ms ⇒ hold C* | **Unsound** (C7, X8) | Hold is not guaranteed while a write is pending. Requires a server-side expected_version/expiry guard. Until then, label fallback "best-effort". |
| O-5 | 20 Hz slack tracking via `POST /admin/budget` with a 50 ms lag gate | **Unsound** | Tick period = lag gate means writes overlap by design (X8), and every write increments version and emits telemetry. Run slack tracking at ≥ 200 ms, or as an in-gateway rule (product change). |
| O-6 | KS output-length OOD test at α = 0.01 on rolling 200 completions | **Uncorrected repeated test** | Replace with one-sided CUSUM on output-length deviation with ARL₀ declared on TRAIN. |
| O-7 | Compute ≤ 10 ms for 8–27 plans × 32 | Holds for 8 × 32 minimal (6.7 ms). | 27 × 32 is ≈ 22 ms; not feasible with 3 levels. |

### 5.1 Derivation for O-3 (applies to any terminal set tied to a fallback)

Let 𝒰 = {16, 32, 48}, C* = 32, k̄ = mean cost ≈ 1.7. Suppose at t the applied budget
is 48 and R(t) = 47.

Holding C* admits nothing until R < 32 − c. So R(t+H) ≤ 32 + k̄ requires releases of
≥ 13.3 cost units within H = 3 s.

Release timing is the residence distribution, which at C ≥ 48 is longer than at C32
(p50 4.8 s at C64 [M R1]). With p50 residence > H, more than half the reserved cost is
still held at t+H. So the terminal set is unreachable by *every* plan, including
"hold C*".

The optimization is then infeasible exactly when the fallback is needed. Either:

- soften the terminal set into a penalty (Astra/GPT), or
- restrict 𝒰 ⊆ {C ≤ C*} so the terminal set is invariant under hold.

For {16, 32} with C* = 32 the constraint is vacuous and should be deleted rather than
cited as a safety property.

## 6. Discriminating experiments

All offline experiments are CPU-only on development captures. Native experiments need
separate authorization.

| ID | Question (who it can falsify) | Design | Decision rule |
|---|---|---|---|
| D-1 | Does ΔV_old change the chosen action? (opus O-2 vs astra/gpt full continuation vs G-2 linear term) | Offline on the 4 A/B switch captures. At each 1-s origin, score {16, 32} with and without ΔV_old (G-2), and with a full continuation rollout. b_after is fit on randomized-switch runs only, to reduce the mechanical bias in C1. | If the action flips in < 2% of origins and the summed value difference is < 0.05 req/s, drop ΔV_old from planning (keep it in evaluation). If the linear term and the full continuation disagree on > 5% of origins, the linear term is rejected. |
| D-2 | Is the age/ledger state worth more than W_tok for *decisions*? (astra A-3, gpt ledger) | Same origins: rank plans with a W_tok-only model versus a W_tok + age-bins + D-age model. | Count origins with a different first action. If < 2%, the ledger is not required for control. |
| D-3 | Can 250 ms one-switch plans be distinguished? (astra A-2, A-6) | Minimal rollout (C6 model) with 32 vs 128 scenarios, 10 seeds. Measure the first-action stability and the value spread across switch times versus MC SE. | If the switch-time value spread < 2 SE in > 80% of origins, collapse to 1-s cadence. |
| D-4 | Per-decision gate firing rate (gpt G-3, astra A-7) | Apply each gate with the historical +0.3163 bias as a structural-error band. | If the firing rate < 5%, the gate makes the MPC ≡ fallback. Move the gain test to policy level. |
| D-5 | Actuation race on the real handler (all; X8) | CPU test with the FastAPI app: delay a POST body, issue a second POST, read `/metrics`, release the first. | Expected (from source): final = delayed value. If so, any live MPC requires an approved `expected_version` (CAS) or `not_after` field on `/admin/budget`, and a test that rejects stale writes. |
| D-6 | Is Gemini's C_admit TTFT model discriminating? | Already answered by the Mayor's reproduction (AUC 0.586–0.775 vs 0.959–0.995). | Reject μ(C_admit) as the TTFT predictor. |
| D-7 [native, later] | Randomized 16/32 pulses: old-cohort effect deconfounded | Randomize the switch time; compare the hit rate of requests admitted 0–0.5 s *before* the switch. | This directly measures b_after's causal value and resolves the C1 upward-bias caveat. |

## 7. Updated position going into Round 3

- **Converged across all four:**
  - the reject-on-arrival actuator with no reservation clipping
  - a finite discrete action set ({16, 32} first)
  - enumeration, not a nonlinear solver
  - N = 1 versus N > 1 and fixed/hysteresis baselines as primary ablations
  - whole-episode randomized policy evaluation with the full offered denominator
  - no deterministic SLO guarantee
- **I concede to astra/gpt:**
  - old-cohort externality belongs in the reward (O-2)
  - the terminal constraint must be soft (O-3)
- **I hold against astra/gpt, on data:**
  - W_tok is the decision-relevant state; the full ledger must earn its place via D-1/D-2
  - per-decision confidence gates make the planner inert (G-3/G-4)
  - compute must be measured against the stated budget (A-1)
- **Shared blocker for every proposal (X8):** `/admin/budget` has no CAS or expiry, so
  no fallback is guaranteed. Either authorize a minimal gateway guard or downgrade all
  fallback claims to best-effort.

Round 2 barrier reached. Context retained for Round 3.
