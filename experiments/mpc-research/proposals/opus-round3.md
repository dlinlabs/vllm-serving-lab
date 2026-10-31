# Opus Round 3: revised design and evidence-based consensus

Researcher: obsidian / Claude Opus 5.5 (`claude-opus-5.5`), alias **opus**.
Product source: `8e92371f342be98a50682bf4e2ddbb014665498b` (unchanged).

Inputs read in full:

- Peer Round 2 reports: `astra-round2.md`, `gemini-round2.md` and
  `gpt-round2.md`, at the peer paths given by the Mayor.
- Mayor's `experiments/mpc-research/mayor_counterexamples.py` and
  `results/mpc-research-cpu/mayor-counterexamples.json`.
- My own frozen Round 1 and Round 2 artifacts.

`opus-round1.md`, `opus-round1-analysis.py`, `opus-round2.md` and
`opus-round2-checks.py` are unchanged. The only new artifacts are this report
and `opus-round3-checks.py`. That script is pure stdlib Python. It reads the
same restored development captures as Round 1 and writes
`/tmp/obs-mpc/r3.json`. See §11 for the exact command and output.

**Scope.** This report contains no GPU, network, final trace C, product edit,
commit or live actuation. No result here is a performance, native-acceptance
or policy-gain claim.

---

## 1. Position in one paragraph

**What the evidence supports.** A gateway-observable backlog ledger
(W_tok + own prompt) is a strong, cheap TTFT feature on development data. The
Mayor reproduced held-run AUC of 0.959–0.995 and MAE of 0.033–0.093 s. That
justifies building a default-off, CPU-only **shadow** controller stack around
that state.

**What it does not support.** It supports no claim that receding-horizon
planning beats either of two simpler controllers:

- a same-state backlog-feedback controller; or
- a same-model one-step (N = 1) controller.

It also supports no claim that any controller is safe to actuate live.

**What blocks live actuation.** The current `/admin/budget` contract (no CAS,
no expiry, no idempotency) makes finite-time recovery from a timed-out write
impossible to prove (§6.6, K4). No live budget writes by any adaptive policy
should be authorized until all of the following exist and are qualified:

- a server-side fencing, expiry and lease contract;
- randomized action-support data;
- a qualified fixed fallback.

My Round 1 safety and optimality deductions are retracted or downgraded as
listed in §3.

---

## 2. Decision record: critiques accepted or rejected

Status key:

- **A** = accept.
- **A-q** = accept with a stated qualification.
- **R** = reject.
- **R-s** = reject the strength of the claim but keep the mechanism.

"Evidence" names an executed artifact or a source line. "M-x" are the Mayor's
counterexamples; "AS", "GP" and "GM" are Astra, GPT and Gemini Round 2.

### 2.1 Critiques of Opus

| # | Critique (source) | Status | Evidence / reason | Consequence in this design |
|---|---|---|---|---|
| 1 | Expected release per tick is not a pathwise bound on excess prompt admission (M "expected release", AS O1, GP O2) | **A** | Mayor: a 7,354-token prompt fits after a 6-cu release, against a nominal 98 + 712 allowance. AS: 4,000 excess tokens. GP: a 7,000-token prompt after 5.40 cu released. My K2: 7,000 excess tokens under both the absolute and the atomic-relative contract | Slack emulation keeps only the conditional identity Σp ≤ L + C_rel/β − (n−1)b/β with **actual** C_rel. There is no numeric bound (§6.4) |
| 2 | 20 Hz absolute C = R + s is unsafe under stale R; 6,000 writes per run flood the lock and telemetry (GM) | **A** (mechanism), **R** (17.88% incidence) | A stale R can raise headroom (K2). The Poisson P(≥2 offers) is a model probability, not a measured deadband incidence. That follows the Mayor's standard; I did not measure incidence either | 20 Hz absolute slack tracking is withdrawn. Cadence target is 1 s |
| 3 | Atomic relative slack does not fix post-apply releases (GP O2) | **A** | K2: atomic relative slack still admits 7,000 tokens when releases occur after apply. Only a per-request predicate W_tok + p ≤ K admitted 0 | A per-request backlog gate is a distinct **product** actuator and baseline (B1′), not an equivalent repair |
| 4 | Hard terminal R_H ≤ C* + k̄ is infeasible under every plan, including hold (M "hold ≠ terminal", AS 3.2, GP O3) | **A** | Mayor R = 60 against limit 34. AS R0 = 32 against 18 at C* = 16, and R0 = 48 at C* = 32. GP R0 = 64 against 34 | The hard terminal constraint is removed. The terminal is now a finite evaluation tail under C* plus valuation of unresolved requests (§6.3) |
| 5 | GM quotes my C* as 16 | **R** (factual) | Round 1 set C* = 32; GM's counterexample still applies with C* = 32 (see row 4) | None beyond row 4 |
| 6 | The 7.4 req/s maximum is not a policy-uniform ceiling (AS 3.3, GP O1) | **A** | Already self-falsified in Round 2 C2: the cheapest decile at C96 gives 12.65–12.9 req/s. AS composition example: 5 vs 20 req/s | Retracted. No ceiling claim |
| 7 | Stationarity does not make myopic feasibility optimal; reservation opportunity cost alone creates horizon value (AS 3.3, GP O4–O5) | **A** | K5: same model, zero decode interference. N = 1 admits the long job (1 hit); N = 4 picks (16,16,16,16) (3 hits) | "Only decode coupling gives horizon value" is retracted. The N = 1 baseline is mandatory, with the same model and scoring window (§7) |
| 8 | Stage reward omits old cohorts and eventual success (M, AS 3.4, GP O4) | **A** | Mayor new-only +1 vs actual −1; first-token 10 vs final 8. AS 2 vs 11; GP CVaR reversal | Single cohort-partitioned reward with eventual success (§6.3). Boundary invariance is verified in K3 |
| 9 | F1 / b_after "biased upward ⇒ upper estimate" (Mayor standard) | **A** — retracted | Endogeneity sign is not proven. The future window is outcome-defined (GP O5, AS 3.3) | F1 is descriptive only. b_after is excluded from the planner model until randomized pulses identify it (§6.2) |
| 10 | Zero failures among 13,962 correlated hits is not a rule-of-three certificate (Mayor standard, GP O6 shared-regime example) | **A** — retracted | GP: P(all 200 fail) is 0.05 under a shared regime vs 6.2e-261 under iid | "0 / 13,962 hits failed after first token" is descriptive. The 2.1e-4 bound is retracted |
| 11 | C3's reservation→token bound (36,370 at C32) is not native KV safety (Mayor standard) | **A** | No tokenizer, allocator, prefix-sharing or orphan-work qualification exists | C3 is gateway accounting arithmetic only. No KV constraint appears in the design |
| 12 | My C6 timings are not compute lower bounds (Mayor standard) | **A** | C6 timed my own toy encodings of four models | Restated as "minimal-model pure-Python timing on this CPU". No feasibility inference about peers' designs |
| 13 | Sequential binomial monitoring ignores dependence and multiplicity (AS 3.5, GP O6) | **A** | GP: 360 tests give 0.973 false-alarm probability. AS: 100 tests give 0.634 | Monitoring now uses episode-level scores with a declared horizon and error spending (§8) |
| 14 | Online label (gateway f − r) ≠ fitted label (client f − scheduled); oldest age is receipt-based (AS 3.4, GP O8) | **A** | Source check: `telemetry.py:350` computes `now - x['received']` | The label identity and a new accept-age field are specified (§6.1, §8) |
| 15 | The frozen `drain` helper ignores already-fitting state (AS 3.6, GP O7) | **A** | K1 on 12 development runs: 8 decreases, 2 already fit at update. The frozen helper gave 0.0006 s and 0.024 s; corrected is 0. The other 6 agree. Range 0–2.408 s, median 0.739 s | Corrected diagnostic in K1. It is descriptive only, not a bound |
| 16 | Switchback estimand, carryover and power are unqualified; the success criterion is inconsistent (AS 3.6, GP O7) | **A** | AS Y = 10A₋₁ + A carryover example | The primary design is randomized whole episodes within host/time blocks, with one criterion: LCB(Δ) > δ (§8). The 49-pair figure is withdrawn as a power claim |
| 17 | Selection on x is harmless only with E[ε \| x, accepted] = 0; ±5% dither does not prove rank (AS 3.5, GP O5) | **A** | No identification argument was given | Online RLS is dropped from the first build. Models are frozen per epoch; adaptation is a later, separately validated feature |

### 2.2 Adjudication of critiques I raised, or that concern peers

These are recorded so that Round 3 does not silently drop them.

| # | Topic | Status now | Basis |
|---|---|---|---|
| P1 | Gemini drain law mixes tokens/step and tokens/s, and releases cost continuously | Maintained (AS G1, GP G1 concur) | Cost is released only at terminal events: cost_out = Σ_{terminals} c_i |
| P2 | Gemini KV chance constraint without a KV observation or model | Gemini self-retracted; closed | Mayor: an unobserved KV risk is not a calibrated chance constraint |
| P3 | Gemini conditional fallback can hold the high cap forever | Maintained (AS 4.3 trace (32,32)×5; GP at R30/C48) | Fallback must be an unconditional command, best-effort under the transport contract (§6.6) |
| P4 | Gemini "<4 ms benchmarked" and "289-test native acceptance" claims | Maintained as unsupported | No artifact. The audit labels the 289 tests a selected local CPU suite |
| P5 | Astra 299 episodes "infeasible" (GM) | **Gemini's objection rejected** | 1 − 0.99^299 = 0.9505. The mathematics holds. If 299 independent episodes are unaffordable, the 1%/95% claim is **not made**; correlated windows are not substituted |
| P6 | Astra CVaR applied to the new cohort only (GP A1) | Accept GP | GP 10-scenario reversal (A = 1 vs 11). In this design any risk term applies to the whole cohort-partitioned reward |
| P7 | Astra release-pending state (GP A2) | Accept GP, with qualification | Same-loop asyncio cleanup has no suspension point between release and terminal (GP trace). A pending state is needed only for an off-loop observer, which this design avoids by snapshotting under the gateway's own lock (§6.1) |
| P8 | Astra/GPT family-maximum counterfactual calibration score (AS 5.3, GP A3) | Accept; also applies to my own Round 1 | One episode realizes one treatment history. Calibration targets only the frozen selected policy or randomized-action marginals (§8) |
| P9 | GPT terminal triple-counting (GM) and V_fallback definition (AS 5.2) | Accept as a hazard; not proven to be an error in GPT's text | K3 shows the naive "explicit + V over all K" total moves from 5 to 7 across cuts; the partition stays at 4. Adopt one mechanism plus a boundary-invariance unit test |
| P10 | One GET does not fence a timed-out write (AS 5.1, GP A4, Mayor) | Accept, including against my own Round 1 fallback language | K4: no CAS and no expiry gives a final cap of 32 after fallback 16 |
| P11 | Gemini "suspend optimization under ambiguity freezes during stalls" (against Astra) | A-q | A freeze is correct behaviour if the frozen state is the qualified fallback. The cap16-r2 5.855 s event-loop stall shows the transport must be off-loop and the stall itself must be a monitored health flag |
| P12 | Open-loop benchmark client does not retry (GM) | Accept as a scope fact | Rejections lose reward in the offered-time denominator. Retry/closed-loop behaviour is a separate workload stress test |

---

## 3. Retractions (complete list)

| Round / item | Retracted claim | Replacement |
|---|---|---|
| R1 §6.2, slack | "≤ ~700 excess prompt tokens per tick" (expected release used as a bound) | Conditional identity with actual C_rel only. The design drops 20 Hz absolute slack |
| R1 §3.3–3.4 | Hard terminal R_H ≤ C* + k̄; recursive feasibility; "bumpless" handoff | Soft finite-tail continuation (§6.3) |
| R1 §2.5 | 7.4 req/s as a policy ceiling | None; descriptive fixed-cap maximum only |
| R1 §2.3–2.4, §4 | Myopic optimality under stationarity; horizon value only through decode coupling | N = 1 is a baseline to be beaten, not a presumption |
| R1 §3.2 | New-arrival × first-token-hit stage reward | Cohort-partitioned eventual-success reward |
| R2 C1 | Old-cohort flip rates as "upper estimates" of b_after effect | Descriptive diagnostic; sign unidentified |
| R2 O-x / C1 | 2.1e-4 rule-of-three bound on post-first-token failure | Descriptive 0 / 13,962, correlated |
| R2 C3 | 36,370 tokens as a resource-safety bound | Gateway reservation arithmetic only |
| R2 C6 | Peer rollout timings as feasibility evidence | Minimal-model pure-Python timing on this CPU |
| R1 §5.1, F4 | Rolling 200-request binomial monitor | Episode-level monitor (§8) |
| R1 §7.2 | 49-pair power; mixed success criteria | One preregistered criterion; power to be computed from pilot variance |
| R1 / R2 | Any implication that an immediate fixed fallback is guaranteed on the current API | Fallback is best-effort until server CAS/expiry/lease exists |
| All rounds | Any runtime, native or performance statement not backed by an executed artifact | None claimed. All horizon/cadence/scenario numbers below are **targets** |

**Retained facts** (development, descriptive, reproduced by the Mayor):

- W_tok + p held-run AUC 0.959–0.995 and MAE 0.033–0.093 s, versus R + p AUC
  0.586–0.775.
- θ1 ≈ 0.046–0.064 s/kTok; θ0 ≈ 0.046–0.095 s.
- Successful outputs were all 128 tokens (no EOS-length excitation).
- Offered rate 25.48 req/s; Fano factor 0.94 at 1 s.
- Corrected 32→16 fit delay 0–2.41 s over 8 events (K1).

---

## 4. Corrected design: summary of choices

All values below are **targets for shadow engineering**. None is a measured
optimum.

| Element | Choice | Rationale / status |
|---|---|---|
| Plant boundary | Gateway admission only; engine treated as black box | KV/allocator excluded; no resource claim |
| Actuator | Existing budget C ∈ 𝒰 = {16, 32} cu | Both values have development switch support (low-cap-switch captures). 24/48/64 lack switch support |
| Cadence Δ | 1.0 s target | Telemetry ≈ 1 s; residence p50 2.0–2.8 s at C16/C32 |
| Control horizon H | 4 s target (4 slots) | Must be compared with N = 1 (§7) |
| Evaluation tail T | 8 s target; sensitivity 4 / 8 / 16 s | Plan ordering must be stable across T (F-4) |
| Plan family | Constant plus at most one switch: 2 + 2·3 = **8 plans** | Small, enumerable, cheap |
| Scenarios | S = 32 common-random-number scenarios per plan (target) | Plans compared only on matched, completed scenario sets |
| Fallback C* | Fixed; 32 proposed, **16 vs 32 unresolved** (§9) | Must be qualified by fixed-cap episodes before use |
| Terminal | No hard constraint; finite tail under C* plus valuation of unresolved requests | §6.3 |
| Hard constraints | Only actuator legality: C ∈ 𝒰; one outstanding command; support check passes, else C* | §6.5 |
| Transport | Existing API: shadow only. Live needs CAS + expiry + lease | §6.6 |

---

## 5. Units and notation

| Symbol | Meaning | Unit |
|---|---|---|
| t | Gateway `time.monotonic()` | s |
| c_i = 0.259 + 0.000758·p̂_i + 0.0052083·m_i | Reservation cost (`gateway.py` cost model) | cu |
| p̂_i | Gateway input-token estimate; `fb_i` flag if tokenizer fallback used | tokens |
| m_i | Requested max_tokens | tokens |
| C(t), v(t) | Applied budget and version | cu, integer |
| R(t) = Σ_{i∈A(t)} c_i | Reserved cost | cu |
| W(t) = Σ_{i∈A_aw(t)} p̂_i | Awaiting-first-content prompt backlog (W_tok) | tokens |
| D(t) = \|A_st(t)\| | Streaming requests | req |
| Q(t) = \|A_aw(t)\| | Awaiting requests | req |
| a_old(t) = max_{i∈A_aw}(t − t_acc,i) | Oldest awaiting age since **acceptance** (new field) | s |
| τ = 0.5 | TTFT SLO | s |
| g_i | Eventual success indicator (§6.3) | req |
| λ̂ | Offer-rate estimate | req/s |
| θ = (θ0, θ1, θ2) | TTFT model parameters | s, s/kTok, s/req |

---

## 6. Module contracts and corrected equations

### 6.1 Observer (in-gateway, read-only)

**State snapshot.** One immutable record, produced under the same asyncio
lock that admission and cleanup use. GP A2 shows that same-loop cleanup has no
suspension between release and terminal, so the snapshot is coherent without
a release-pending state.

```
Snapshot {
  seq: int,  t: float [s, gateway monotonic],
  C: float [cu], v: int, R: float [cu],
  ledger: [ {id, c[cu], p_hat[tok], fb: bool, m[tok],
             t_recv[s], t_acc[s], phase: awaiting|streaming, t_first[s]|None} ],
  W: float [tok], Q: int, D: int, a_old: float [s],
  offers_since_last: [ {t_recv, p_hat, c, accepted: bool} ],
  health: {loop_lag_s, telemetry_age_s, orphan_suspect: bool}
}
```

**Identities.** These are checked on every snapshot in shadow, and violation
invalidates the snapshot:

- O-I1: R = Σ_ledger c to within 1e-6 cu, and R equals `/metrics` reserved
  cost at the same seq.
- O-I2: W = Σ_{awaiting} p̂ and Q = |awaiting|.
- O-I3: a_old uses t_acc. The existing `oldest_awaiting_first_seconds` uses
  t_recv (`telemetry.py:350`) and is retained as a separate field. The two
  differ by queueing before admission.
- O-I4: v is non-decreasing; C equals the last applied write at version v.

**Required product change (future, separately reviewed; not made here):**

- record t_acc and the marked ledger in the active map;
- expose W, a_old and seq.

The observer is the minimal state supporting the reproduced AUC; it adds no
engine scrape.

**Orphan work.** `gateway.py:438-476` closes upstream, releases cost and
records the terminal. It does not prove the backend stopped. After any
failed/cancelled terminal, set `orphan_suspect` for T_orph (target 2 s,
to be measured). While it is set, model predictions are flagged out of scope
and the planner returns C* (§6.5).

### 6.2 Model (frozen per epoch)

**TTFT for an awaiting request i**, with FCFS backlog ahead of i:

```
T_i = θ0 + θ1·(W_ahead,i + p̂_i)/1000 + θ2·D + ε_i        [s]
```

- θ2 is kept because the "queue_causal" variant adds D and improved MAE
  (0.028–0.089 s). It must be re-validated on fresh sessions.
- b_after (the effect of later admissions) is **set to 0** in the planner
  model. It is not identified, and its sign is unknown (§2 row 9).
- The model's blindness to b_after is a declared limitation. The randomized
  pulse experiment F-6 tests it.

**Label identity.** The fitted target is client TTFT:

```
TTFT_client,i = (t_first,i − t_recv,i)_gateway + δ_in,i + δ_out,i
```

δ_in and δ_out are ingress/egress transport delays. Online, only the gateway
term is observed. Two options:

- (a) fit θ on the gateway target and estimate δ = δ_in + δ_out as a separate
  offset with a qualified client join; or
- (b) declare that online updates fit the gateway target, and convert the SLO
  to τ_g = τ − q̂_δ, with q̂_δ a measured upper quantile.

The first build uses (b) with q̂_δ estimated offline from paired
client/gateway logs.

**Streaming lifetime and release.** Each streaming request terminates after:

- duration L_i = m_i·κ(D) + η_i on the m_i = 128 development regime; or
- an empirical survival curve by phase age.

At termination it releases c_i in full. Release is an event, never a rate
(P1). Eventual success follows from first-token hit and a stream-success
probability π_s(·), estimated per epoch. Development shows 0/13,962
post-first-token failures (descriptive), so π_s is not assumed to be 1. It is
a monitored parameter, and failures enter g_i.

**Arrivals.**

- Scenario offers are resampled from the last 60 s of observed offers:
  inter-arrival gaps and (p̂, m) marks, block-bootstrapped in 1 s blocks to
  keep the Fano structure.
- Admission inside the rollout uses the exact gateway rule
  `accept ⇔ R + c ≤ C(t)` (`gateway.py:~321`), including the tokenizer
  fallback cost.

### 6.3 Reward: cohort partition and terminal treatment

The decision is at t0, with evaluation window E = [t0, t0 + H + T].

Partition every request relevant to the decision into exactly one set:

- **K_old**: admitted before t0 and unresolved at t0. This includes awaiting
  requests, and hit-but-streaming requests whose final success is unresolved.
- **K_new**: offers in E. All arrivals in E are explicitly simulated under the
  plan for [t0, t0 + H), then under C* for [t0 + H, t0 + H + T).
- Everything outside these sets (already resolved, or offered after
  t0 + H + T) is plan-invariant within the window and is excluded.

The eventual reward of request i is:

```
g_i = 1{admitted_i} · 1{TTFT_i ≤ τ} · 1{stream_i completes successfully}
```

The score of plan U under scenario s is:

```
J_s(U) =  Σ_{i∈K_old ∪ K_new, resolved ≤ t0+H+T}  g_i
        + Σ_{i∈K_old ∪ K_new, unresolved at t0+H+T} ĝ_i(x_end)
        − ρ·n_sw(U, C_applied)                                       [requests]
```

- ĝ_i(x_end) = P(g_i = 1 | model state at window end). This is each
  request's own model probability, counted **once**.
- ρ is in requests per switch. n_sw counts a change from the currently
  applied cap as a switch.
- Plan value is J(U) = (1/S)·Σ_s J_s(U) on common random numbers.

This construction rules out three failure modes:

- **Double counting.** No request appears both as explicitly resolved and in
  the continuation term.
- **Boundary jumps.** Moving the cut T moves a request from ĝ to g. If ĝ is
  calibrated, the expected total does not change. K3 is the unit test: the
  partition total stays at 4 for T ∈ {3, 3.5, 4, 7}, while "explicit + V over
  all K" gives 5 / 6 / 6 / 7.
- **Unpriced opportunity cost.** A long reservation made inside H prices its
  effect on K_new arrivals during the tail (K5 mechanism).

Opportunity cost beyond t0 + H + T is **not** valued. That is the declared
truncation. F-4 tests the sensitivity of plan ordering to T.

**Risk term (optional).** If used, CVaR_β is taken over the **total**
J_s(U), never over the new cohort alone (P6). The default is risk-neutral;
any risk term must beat the risk-neutral version in F-8 to be kept.

### 6.4 Slack emulation: kept only as a conditional identity

For completeness, with fixed m, slack s granted at a tick, n admissions and
actual released cost C_rel before the next effective command:

```
Σ_{admitted} p̂_i ≤ (K − W)_+ + C_rel/β − (n − 1)·b/β
```

Here β = 0.000758 cu/token and b = c(0). This is **not** used to claim a
bound, because C_rel has no validated pathwise bound (M, K2). The
per-request gate W + p̂ ≤ K (product actuator B1′) is the only contract
examined here that bounds admitted backlog per request (K2: 0 excess). It is
recorded as a candidate product change, not part of the MPC.

### 6.5 Planner, constraints and support

```
decide(snap: Snapshot, deadline_s: float) -> Decision {
  id: uuid, snap_seq, plans_evaluated, J_hat[U], gap = J(1st) − J(2nd),
  support_ok: bool, chosen_first_action ∈ 𝒰 ∪ {C*}, reason
}
```

**Procedure.**

1. Pre-check scope. If any of the following holds, return C* with no planning:
   - support fails (below);
   - `orphan_suspect` is set;
   - `health.loop_lag_s > 0.25` or `telemetry_age_s > 2Δ`;
   - transport is not IDLE.
2. Draw S scenarios with a fixed seed derived from snap.seq, and roll out all
   8 plans on the same scenarios.
3. If the deadline arrives before every plan completes every scenario, return
   C*. Plans are never compared on partial scenario sets.
4. Choose argmax J only if the gap exceeds a calibrated indifference margin;
   otherwise hold the current applied cap.

**Support.** Allowed only where all of the following hold:

- (W, D, Q, λ̂) falls in development bins with at least n_min sessions × n_min
  episodes;
- each plan's action sequence is in the randomized-switch support set;
- the current applied cap is in 𝒰.

Support is defined by axis-aligned bins with counts, not Mahalanobis distance
(GP G5's hull counterexample).

**Isolation.** The planner runs in a separate OS process. It receives an
immutable snapshot through IPC and returns a Decision. The gateway side
discards any Decision whose snap_seq is older than the latest snapshot by
more than one cadence, or that arrives after the deadline. A timed-out worker
is killed; its result can never enqueue a write. No compute timing is claimed.
F-7 measures the full end-to-end cycle at the largest observed active
population.

### 6.6 Fallback and transport

**Fallback.** C* is a fixed member of 𝒰, chosen from qualified fixed-cap
episodes. "Fallback" means the planner *requests* C*. Whether C* is
*applied* depends on the transport state.

**Transport states (client side):**

```
IDLE(C, v)
  → PENDING(cmd_id, C_new, v_expected, sent_at)
  → IDLE(C_new, v')            on 2xx response with applied version v'
  → UNCERTAIN(set={C_old, C_new}, cmd_id)
                               on timeout / transport error / lost response
UNCERTAIN → IDLE only on positive evidence (below)
```

Shadow mode never leaves IDLE, because it never writes.

**Existing API (`gateway.py:588-630`: no CAS, no expiry, no id).** A GET
showing the old (C, v) is **not** evidence that the pending write cannot
apply later (K4; AS 5.1; GP A4). The only positive evidence is a GET showing
v = v_expected + 1 with C = C_new under a qualified single-writer assumption.
Without that, UNCERTAIN may last indefinitely. No other write may be sent
while UNCERTAIN, because a fallback write can be overwritten by the late one
(K4: final 32). **Therefore the existing API does not support live adaptive
actuation.**

**Required server contract** (a future product change; this report does not
authorize or make it):

| Field / behavior | Requirement | Why (K4 / Mayor) |
|---|---|---|
| `expected_version` | Apply only if the current v equals it; else 409 with current (C, v) | Rejects late A after any newer write (K4: CAS + fallback → 16) |
| `not_after` | Gateway-monotonic deadline evaluated **at lock acquisition**; else 410. Valid only if the controller shares the gateway host's `CLOCK_MONOTONIC`; otherwise use a server-issued lease epoch | Kills late A even when no newer write exists (K4: expiry, no fallback → 16) |
| `command_id` | Idempotent: a duplicate id returns the original result without reapplying | Safe retry after a lost response |
| Lease | Adaptive budgets carry `lease_s`. If the lease is not renewed, the **server** reverts to C* and increments v | Controller death or partition still returns to C* without any client write |
| Response | Applied (C, v, server_t, command_id) | Positive evidence for leaving UNCERTAIN |

With CAS, expiry and lease in place, UNCERTAIN resolves in bounded time: by
`not_after`, the command is either applied (visible as version
v_expected + 1 with its command_id) or can never apply. K4 executes all eight
CAS × expiry × fallback combinations.

---

## 7. Baselines the MPC must beat

All baselines share the same observer, cadence, transport, fallback and
support gate. They differ only in decision logic.

| ID | Baseline | Exact rule | Purpose |
|---|---|---|---|
| B0 | Fixed C* | C = C* always | Qualified reference; also the fallback |
| B1 | Same-state backlog feedback (hysteresis) | Set C = 16 if W > K_hi, C = 32 if W < K_lo, else hold. K_hi and K_lo are derived from θ: K* = 1000·(τ_g − θ0 − θ2·D̄)/θ1 − p̄, with K_hi = K* and K_lo = K* − h. h and the offset are tuned on development only | Strong same-state feedback. Development values give K* of a few thousand to ~8,000 tokens (target, not a measurement) |
| B1′ | Per-request backlog gate (product actuator) | Accept iff R + c ≤ C and W + p̂ ≤ K* | Different actuator. Tested only if a product change is approved |
| B2 | Same-model N = 1 | Same model, scenarios and window E. Plans are restricted to "u for [t0, t0 + Δ), then C*" | Isolates the value of planned future control freedom from horizon scoring |
| B2s | Same-model short score | Same model; window E = [t0, t0 + Δ + T_res] with T_res = residence p90 | Isolates the value of the long scoring window (GP §6.2) |
| MPC | 8-plan receding horizon | §6 | Candidate |

**Adoption rule** (preregistered). MPC is adopted for further engineering only
if both of the following hold on randomized whole episodes:

- LCB₉₅(J_MPC − J_B1) > δ; and
- LCB₉₅(J_MPC − J_B2) > δ.

If it fails either, the simpler baseline is carried forward and a negative
MPC result is recorded. δ is a pre-registered minimum useful gain in successful
requests per offered request, set before data collection. A K5-type opportunity
cost gives MPC a mechanism to beat B2 in principle. Whether it occurs on the
real workload is unmeasured.

---

## 8. Calibration, support and statistics

- **Calibration unit.** Use the episode (an independent run within a
  host/time block), not the request.
  - Conformal residual bands use the corrected rank ⌈(n + 1)(1 − α)⌉ on a
    frozen model, with episode-cluster scores (each episode's max or 90th
    percentile residual near τ).
  - With n < 19 episodes, no 95% band exists (AS: n = 10 gives rank 11).
- **What calibration covers.** Only the frozen selected policy's factual
  predictions on fresh episodes, plus randomized-action marginal contrasts
  under the recorded propensities. There is no family-maximum counterfactual
  score (P8). The planner's gap is a heuristic selection statistic, not a
  guarantee.
- **Near-boundary calibration.** Report reliability for predicted T ∈ τ_g ±
  0.15 s, and optimistic-error quantiles. AUC alone is not accepted because it
  is invariant to the offset (AS Command B).
- **Monitoring.**
  - Declare in advance the number of looks per epoch (e.g. 10) and use
    Bonferroni / α-spending across those looks.
  - The score is the per-episode mean of 1{T_obs > T̂ + q̂}.
  - Report false-alarm rates measured under replayed fixed-cap episodes.
- **Zero-failure safety claims.** A 1%/95% claim requires ≥ 299 independent
  episodes with zero events (1 − 0.99^299 = 0.9505), and 0.1%/95% requires
  2,995. If those episodes are unaffordable, **the claim is not made**.
  Correlated windows or requests are reported as descriptive counts only.
- **Clock identities.**
  - Control and observer use gateway `time.monotonic()`.
  - Labels use the client clock.
  - The offset is estimated from paired events and checked for drift each
    epoch (|drift| < 5 ms per 300 s target).
  - Transport `not_after` uses gateway monotonic, which requires the same host.
- **Support data.** Randomized 16↔32 switch pulses at decision points, with
  recorded propensity (e.g. 0.2 per decision), in development sessions only.
  Pulses are part of the experimental design, not of live control.

---

## 9. Preserved disagreements (not settled by vote)

| # | Question | Positions | Decisive experiment |
|---|---|---|---|
| U1 | C* = 16 or 32 | Gemini 16 (safety margin); Opus 32 (development goodput). Neither is qualified | Fixed-cap randomized whole episodes at 16 vs 32. C* is the cap with higher LCB of goodput subject to the pre-registered failure-rate criterion |
| U2 | Does planned control freedom beat N = 1? | GPT and Astra: plausibly. Opus now: unknown (Round 1 claim retracted). K5 shows the mechanism exists | MPC vs B2 vs B2s (§7) |
| U3 | Does MPC beat same-state backlog feedback? | Unknown; development AUC suggests W is the dominant state | MPC vs B1; also B1′ if a product change is approved |
| U4 | Is b_after nonzero and planner-relevant? | Sign unknown (Mayor) | F-6 randomized later-work pulses at matched strata |
| U5 | Can the existing API ever exit UNCERTAIN in finite time? | Opus/GPT/Astra: not provably; only positive evidence | Fault injection (F-5). The decision is whether the server contract is mandatory before live use (Opus: yes) |
| U6 | Is the cadence 1 s or faster? | Gemini 1 s; Astra/GPT sub-second; Opus now 1 s | Shadow age measurement plus decision-stability vs cadence on replay |
| U7 | Is a risk term (CVaR) needed? | Astra yes (with corrected scope); others risk-neutral | F-8 |
| U8 | Is 299-episode qualification affordable? | Gemini no; mathematics unchanged | Budget decision. If no, ship no rate claim |

---

## 10. Falsification tests and decisive experiments

The CPU tests (F-1 to F-5 and F-7) can run in default-off shadow engineering.
The experiment rows (F-6, F-8 to F-10) need development sessions on the
serving host, approved separately.

| ID | Test | Falsifies if | Tier |
|---|---|---|---|
| F-1 | Observer identities O-I1..O-I4 on mock and replay | Any identity is violated beyond tolerance | CPU |
| F-2 | Label identity: gateway TTFT + δ vs client TTFT on paired logs | Residual drift beyond tolerance, or δ quantile unstable across sessions | Replay |
| F-3 | Forward-chaining fresh-session calibration of θ (train on earlier sessions, test on later) | Near-boundary reliability error > 0.05, or MAE worse than the R + p model | Replay |
| F-4 | Cohort-partition unit test (K3), plus plan ordering under T ∈ {4, 8, 16} s | Total changes with the cut; argmax flips for > 10% of origins between T = 8 and 16 | CPU |
| F-5 | Transport fault injection: delayed handler, lost response, overtaking GET, writer restart, stale worker result | On the existing API, the client leaves UNCERTAIN without positive evidence. On the mock CAS/expiry/lease server, any late write applies | CPU |
| F-6 | Randomized later-work pulses at matched strata | b_after differs from 0 beyond θ1 / 10 → model invalid; add the term | Development serving |
| F-7 | End-to-end cycle at the largest observed active population and S = 32: snapshot → IPC → plan → decision; gateway loop lag measured with and without planner | The cycle misses the Δ deadline at the p99 target, or the planner raises gateway loop lag | CPU |
| F-8 | Randomized whole-episode comparison B0 / B1 / B2 / B2s / MPC within host/time blocks | MPC fails LCB > δ against B1 or B2 → MPC not adopted | Development serving (separately authorized) |
| F-9 | Corrected drain (K1) re-run on new sessions | Already-fit handling disagrees with the ledger | Replay |
| F-10 | Zero-failure safety claim | Fewer than 299 independent clean episodes → no 1% claim | Development serving |

---

## 11. Readiness

**Ready for default-off CPU / shadow engineering** (subject to normal product
review of each change):

- Observer ledger fields (t_acc, marked ledger, W, a_old, seq) behind a flag,
  with the identity checks.
- Model fitting and calibration on replay.
- Planner process with the 8-plan family, deadline and stale-decision
  discard.
- Baselines B0, B1, B2 and B2s as interchangeable decision modules.
- Transport state machine against a **mock** gateway implementing CAS,
  expiry, lease and id.
- Unit tests F-1, F-4, F-5 and F-7.

In shadow, decisions are logged and never written.

**Not authorized: live actuation by any adaptive policy (including B1)** until
every item below is satisfied:

1. The server contract in §6.6 is implemented, reviewed and passes F-5 against
   the real gateway.
2. Observer identities pass on the native serving host (F-1 native).
3. Randomized action-support data exists for 𝒰 (F-6 design).
4. C* is qualified by fixed-cap episodes (U1).
5. The comparison F-8 is preregistered.
6. A human/Mayor authorization decision is recorded.

None of this report's evidence is native, GPU or policy-gain evidence.

---

## 12. Limitations

- **Development data scope.** All quantitative facts come from 12
  development captures: one host, one workload family, outputs fixed at
  128 tokens and Fano factor 0.94. Transfer to EOS-length outputs, bursty
  traffic or other hardware is untested.
- **Causal identification.** W_tok's predictive strength is factual under
  historical fixed-cap and 16/32 switch policies. It does not identify
  intervention effects.
- **Truncated reward.** The reward truncates opportunity cost after
  t0 + H + T, so ordering may change with longer tails.
- **Orphan work.** Orphan handling is a conservative timer, not a
  measurement.
- **Product changes.** The server contract and ledger fields are product
  changes that need their own review. Nothing here authorizes them.
- **Constructed checks.** K2–K5 are constructed counterexamples: they show
  existence, not frequency. K1 is a descriptive re-analysis of 8 switch
  events, not a drain bound.

---

## 13. Executed commands and outputs

The script was written fresh for this round; Round 1 and Round 2 artifacts were
not modified. K1 imports `load_run` and the frozen `drain` from
`opus-round1-analysis.py` to compare against it. Data is
`/tmp/obs-mpc/{cap-range,lowcap}`, restored previously by the repository
extract scripts.

Source check:

```bash
sed -n 345,352p telemetry.py
# -> oldest_awaiting_first_seconds = max((now - x['received'] for x in awaiting), default=0)
```

Main command (exit 0):

```bash
cd experiments/mpc-research/proposals
python3 -B opus-round3-checks.py /tmp/obs-mpc/cap-range /tmp/obs-mpc/lowcap
```

Exact stdout:

```text
K1 decreases 8 fit_observed 8 already_fit 2 min/max [0.0, 2.4084] median 0.73925 disagree_frozen 2
K2 {'absolute_C': 7000, 'atomic_relative_slack': 7000, 'per_request_backlog_gate': 0}
K3 {'T': 3.0, 'partition_total': 4.0, 'naive_total': 5.0}
K3 {'T': 3.5, 'partition_total': 4.0, 'naive_total': 6.0}
K3 {'T': 4.0, 'partition_total': 4.0, 'naive_total': 6.0}
K3 {'T': 7.0, 'partition_total': 4.0, 'naive_total': 7.0}
K4 cas=False,expiry=False,fallback=True 32 [(70, 'B', 'applied'), (100, 'A', 'applied')]
K4 cas=False,expiry=False,fallback=False 32 [(100, 'A', 'applied')]
K4 cas=False,expiry=True,fallback=True 16 [(70, 'B', 'applied'), (100, 'A', 'expired')]
K4 cas=False,expiry=True,fallback=False 16 [(100, 'A', 'expired')]
K4 cas=True,expiry=False,fallback=True 16 [(70, 'B', 'applied'), (100, 'A', 'cas_reject')]
K4 cas=True,expiry=False,fallback=False 32 [(100, 'A', 'applied')]
K4 cas=True,expiry=True,fallback=True 16 [(70, 'B', 'applied'), (100, 'A', 'expired')]
K4 cas=True,expiry=True,fallback=False 16 [(100, 'A', 'expired')]
K5 {'myopic_first_action': 32, 'myopic_hits': 1, 'N4_best_plan': (16, 16, 16, 16), 'N4_hits': 3}
assertions passed; JSON at /tmp/obs-mpc/r3.json
```

**K1, per-event detail.** These come from `/tmp/obs-mpc/r3.json`; every event
is a 32→16 decrease. R at update is in cu; delays are in seconds.

| Run | R at update | Corrected | Frozen helper | Next offer |
|---|---|---|---|---|
| a-r1 | 29.44 | 0.849 | 0.849 | rejected |
| a-r1 | 31.99 | 2.408 | 2.408 | rejected |
| a-r2 | 30.96 | 1.989 | 1.989 | rejected |
| a-r2 | 31.32 | 1.972 | 1.972 | rejected |
| b-r1 | 30.68 | 0.186 | 0.186 | rejected |
| b-r1 | 11.26 | **0** | 0.0006 | accepted at 0.050 |
| b-r2 | 14.41 | **0** | 0.024 | accepted at 0.038 |
| b-r2 | 26.03 | 0.630 | 0.630 | rejected |

**K1 tie convention.** Events at t == applied_at count as before the update.

**Interpretation limits.**

- K2–K5 are constructed existence examples, not measurements.
- K4 is a deterministic model of server semantics, not a test of
  `gateway.py`.
- K5 uses the same zero-interference model for both N = 1 and N = 4, so the
  difference is due to reservation opportunity cost alone.

---

## 14. Supplement: review of the Mayor's provisional `mpc-mathematical-specification.md`

I read the complete specification:
`/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/experiments/mpc-research/mpc-mathematical-specification.md`
(199 lines). This section is a technical acceptance/correction, not a vote.
Supporting script: `opus-round3-supplement-checks.py` (stdlib only, constructed
examples; command and output at the end of this section).

### 14.1 Accepted as written

- **§2, exact accounting.** The strict-boundary admission rule, whole-cost
  release only at terminal, and lifecycle-generated Q/D/R flows match
  `gateway.py` and my §6.2–6.3.
- **§6, reservation bound.** R(t) ≤ max(R(0), u_max) and
  count ≤ max(R(0), u_max)/c_min are correct by induction. Admission requires
  R + c ≤ u ≤ u_max, and releases only decrease R. S3: 2,000 randomized
  event sequences gave 0 violations, with max R/bound = 1.0, so the bound is
  attained. The spec's disclaimers (no KV, orphan-work, latency or Lyapunov
  claim) are correct.
- **§1 and §3, honesty about surrogates.** The spec treats W_prompt as a
  prompt-backlog proxy, not residual work; treats the kernel transition as an
  unidentified surrogate; and requires belief over aliased residual work. I
  agree with all three.
- **§4, objective details.**
  - g_i requires eventual complete success × a scheduled-time hit.
  - The objective is a count objective over a fixed offered duration.
  - Current hit-streams stay unresolved.
  - Switch and risk weights start at zero, with deterministic ties.
  - Scheduled client time is acknowledged as unavailable live.
- **§5–6, transport and planning.**
  - Shadow-only mode before qualification.
  - The guarded capability is checked under the application lock (cost, run,
    clock, version, expiry and seal). An old endpoint that ignores fields is
    not guarded.
  - Timeout is unknown application, not cancellation.
  - Prediction runs outside handlers and locks, and freshness is rechecked at
    the actuator.

  These agree with my §6.5–6.6 and K4.
- **Planner search.** Full enumeration of N = 3 over {16, 32} gives 8
  sequences with no one-switch pruning. I **adopt this in place of my §4
  "H = 4, constant plus one switch"**. It has the same count and makes no
  structural pruning assumption; one-switch pruning becomes an ablation.

### 14.2 Concrete flaw: §4 does not reward tail arrivals, so it prices no opportunity cost beyond N

**What the spec does.** It scores only C_k (unresolved admitted requests plus
offers inside the prediction window). Tail arrivals "affect contention but are
not also rewarded". A reservation that blocks post-window offers therefore
costs nothing. This reintroduces, at the window edge, the myopic error the
Mayor's own counterexamples and my K5 refute.

**Counterexample S1.** The executed first action is at t = 0, with N = 3 and
1 s slots, and backup 32 after the window.

- Offers: one long request at t = 0 (cost 20, holds 6 s), and three short
  requests at t = 3, 4, 5 (cost 13, hold 0.5 s each). All admitted requests
  are on time.
- **Cohort-only reward** chooses first action **32**: it admits the long
  request and scores 1. The true outcome over [0, 6) is **1 success**, because
  the long request blocks all three shorts.
- **Fixed-window reward** (tail offers rewarded) chooses first action **16**,
  with true outcome **3**.
- Receding-horizon replanning cannot repair this: the long request's
  admission is irreversible at t = 0, and no eviction exists.

**Why the spec's concern does not require the exclusion.** Rewarding tail
arrivals counts each request once and does not reward them twice: they belong
to no other term. The evaluation window [t0, t0 + N·Δ + T) is identical
across plans, so it cannot be gamed by the plan choice.

**Simpler correct alternative.** My §6.3 partition, which redefines the cohort
as all offers in a plan-invariant window:

- **Cohort.** C_k = unresolved admitted requests ∪ **all** offers in
  [t0, t0 + N·Δ + T). Offers in the tail run under the backup.
- **Resolved requests.** Each scores g_i.
- **Unresolved at window end.** Each scores its own ĝ_i, counted once.

Truncation then moves to T and is measured, not removed. Test F-4 (plan
ordering stable across T ∈ {4, 8, 16} s) and the K3 boundary-invariance unit
test become required. With this change, "bounded continuation" means
continuation under the backup **with reward**, not continuation for contention
only.

### 14.3 Corrections that matter for implementation

1. **Kernel surrogate is not identifiable from saturated packing (§3).**
   - **The confound.** Under the spec's own packing rule, a saturated
     iteration has q = B − d. τ = θ_base + θ_prefill·q + θ_decode·d +
     θ_int·q·d then reduces to a quadratic in d with only 3 estimable
     combinations of 4 coefficients:
     - θ_base + θ_prefill·B;
     - θ_decode − θ_prefill + θ_int·B;
     - −θ_int.
   - **Evidence.** S2: the design matrix has rank 3 on saturated iterations,
     and rank 4 only after unsaturated iterations (q < B − d) are added.
   - **Observability.** Iteration timing is not observed at the gateway at
     all; only first-content and terminal events are.
   - **Required change.**
     - State the identifiability condition: unsaturated iterations, or a
       native per-iteration trace.
     - Until it holds, fit the saturated reduced form, or fix a
       reparametrization. Do not report four "physical" coefficients.
   - **Discriminating test against the W_prompt law.** The kernel model
     predicts a TTFT slope in W of roughly τ̄(D)/(B − D); in effect, θ1
     varies with D. The backlog law has a constant θ1 plus an additive θ2·D.
     Fit θ1 within D-bins on the development captures. A significant
     D-dependence favours the kernel form; otherwise the simpler backlog law
     stands.
2. **Plan-count inconsistency (§5).** The spec says "at most eight {16, 32}
   sequences" but also asks to compare N = 1/3/5. Full enumeration at N = 5 is
   32 sequences (S4). Either state "≤ 2^N, with 8 at N = 3", or name the
   pruning used for N = 5. Otherwise the N = 5 arm is a different search
   family, and the N comparison is confounded with pruning.
3. **Unresolved-reward interval (§4).** The spec keeps a lower/upper interval
   for unresolved target rewards, but no selection rule is given.
   - Selecting on the lower end (unresolved = 0) biases toward plans whose
     cohorts resolve sooner. That means rejecting long requests: the same
     horizon gaming, reversed.
   - **Simpler rule.** Select on the point value ĝ under common scenarios.
     Report the interval and use it only for an indifference / abstain test
     (hold when the plans' intervals overlap by more than the calibrated
     margin).
4. **Budget exhaustion (§5).** "Return an observable fallback on exhaustion"
   should mean **emit no new command; hold the last applied qualified cap**.
   It should not mean writing the backup cap. Otherwise compute overload
   generates budget writes and switches exactly when the system is stressed.
   Under the spec's §6, both members of {16, 32} must be qualified fixed
   budgets before any write, so holding either is an acceptable degraded
   state.
5. **Persistence and lease (§6).** The spec says "last applied budget
   persists" when transport is unavailable. This is acceptable **only**
   because every applied value must be a qualified fixed cap.
   - It should be stated as a precondition: no adaptive write may ever apply a
     value that is not itself an acceptable indefinite fixed budget.
   - A server-side lease that reverts to the backup (my §6.6) is then
     optional hardening rather than mandatory. I withdraw "mandatory lease"
     under that precondition. CAS, expiry and command-id remain mandatory.
6. **Age observable (§1).** `oldest_awaiting_age` must name its origin. The
   existing metric uses receipt time (`telemetry.py:350`); the ledger can
   supply admission age. They are different observer candidates and must be
   ablated separately.

### 14.4 Net position

I accept the specification as the shadow-engineering baseline, subject to
14.2 (reward tail offers in a plan-invariant window; this is mandatory) and
corrections 14.3(1)–(6).

My §4 table is superseded where it conflicts:

- **Search:** N = 3 full enumeration replaces H = 4 one-switch.
- **Lease:** optional under the precondition in 14.3(5).

Native qualification remains false. Neither the Mayor's specification nor this
report authorizes live budget writes.

### 14.5 Command and output

```bash
cd experiments/mpc-research/proposals
python3 -B opus-round3-supplement-checks.py
```

Exact stdout (exit 0):

```text
S1 {'scores': {'(16, 16, 16)': (0, 3), '(16, 16, 32)': (0, 3), '(16, 32, 16)': (0, 3), '(16, 32, 32)': (0, 3), '(32, 16, 16)': (1, 1), '(32, 16, 32)': (1, 1), '(32, 32, 16)': (1, 1), '(32, 32, 32)': (1, 1)}, 'cohort_only_first_actions': [32], 'cohort_only_true_value': 1, 'fixed_window_first_actions': [16], 'fixed_window_value': 3}
S2 {'params': 4, 'rank_saturated': 3, 'rank_with_unsaturated_iterations': 4}
S3 {'trials': 2000, 'max_R_over_bound': 1.0, 'violations': 0}
S4 {'N=1': 2, 'N=3': 8, 'N=5': 32}
assertions passed
```

How to read S1 scores: each tuple is (cohort-only score, true successes over
[0, 6)). An earlier draft placed the long offer at t = 2. Replanning at t = 2
then brings the shorts into the window and corrects the choice, so that draft
was not decisive and was replaced before recording. S1–S4 are constructed
examples, not plant measurements.

---

**Round 3 barrier reached.** New files:

- `opus-round3.md`
- `opus-round3-checks.py`
- `opus-round3-supplement-checks.py`
- scratch output `/tmp/obs-mpc/r3.json`

Round 1 and Round 2 files are unchanged. No product edits, commits, pushes,
merges, `gt done`, GPU or paid compute, final-C access, sandbox changes or
network operations were performed by me in this round.
