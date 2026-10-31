# Round1 (Opus): physical batch service, remaining work and causal identification

Author: Polecat `opal`, Claude Opus 5.5 (model ID `claude-opus-5.5`). Round1 independent research.
I did not read any peer Round1 reports.
Branch `research/native-v2-opus-local`; research base `e75dbdd` (from `research/pr33-batch-dynamics-v2`).
All data used is **Train/exploratory** (seed blocks 159901–159904, 160001–160010).
Held-out 160021–160024 was not opened or generated, and the loader refuses it (`HELDOUT_SEEDS`).
No GPU, SSH, provider or API lifecycle calls were made. No push, merge, rebase, PR or remote edit, and no `gt done`.

## 0. Answer to the core question

**There is no evidence yet that MPC improves native goodput over Fixed/One-move/reactive control, and this round does not claim it.**
The question is what *identifiable* model could. The evidence points to a **hybrid, iteration-level batch model with a roofline service law and age-conditioned remaining work**, in which:

1. **The service unit is a scheduler iteration, not a request.**
   Iteration time is roughly flat in batch size while the iteration is memory-bound (decode and short prefill), and roughly linear in tokens once it is compute-bound (long prefill chunks).
   This dissolves the falsified serial-HTTP contradiction instead of tuning around it (F1, F2).
2. **TTFT for long prompts is controlled by *how many long prefills are already ahead*.**
   For short prompts it is controlled by almost nothing (F4).
   Long-prompt TTFT-good falls from 99.7% to 67% to 13% to 0% at 0, 1, 2 and 3 prefills ahead.
   Short prompts stay at about 66 ms median for 0–16 ahead.
   This is the physical mechanism behind the scout finding that budget 32 wins for p128 and loses for p4096.
3. **The budget actuates concurrency per cost class.**
   Under the current cost spec, budget 16/32 means at most 15/31 concurrent p128-m128 requests but only 3/7 concurrent p4096-m128 requests (F7).
   Most of the attainable gain is therefore expected to come from **a state-feedback, class-aware use of the existing actuator**, and possibly from an action value outside {16, 32}.
   Horizon length is not expected to be the source.
4. **Multi-step adds value only if action effects outlive one control period *and* the disturbance is forecastable within the horizon.**
   Effects here last about 0.3–2 s against a 1 s control period, and arrivals are Poisson in the Train data.
   **My prior is that N=3 ≈ N=1 on every Train workload so far.**
   §8 gives a discriminating ablation that can show that prior is wrong; it does not inflate the horizon.

A controller is worth testing only after the falsification experiments in §9 E1–E4 pass.
Even then, the strongest baseline to beat is the reactive "prefills-ahead" rule (§7.5), not Fixed16.

## 1. Sources read, scope and honesty notes

Read: `native-v2/master-research-report.zh-TW.md`, `native-identification-and-model-falsification.zh-TW.md`,
`native-gpu-qualification-and-scout.zh-TW.md`, `system-model.md`, `mpc-mathematical-specification.md`,
and native_observer sources `prefill_identification.py`, `analyze_profile.py`, plus the `native_mpc.py`/`native_token_model.py` interfaces.

Artifacts used (all under `results/mpc-native-4090-scout/`):
- `native-profile-diagnostic.json`: 84 profiled GPU batches. SHA256 `348e66c1…48a3`.
- `native-identification-train-evidence.tar.gz`: SHA256 `b86094f1…129f`, as sealed. Only the two `interventions.json` members (16 Train episodes) are parsed.
- `native-prefill-combined.json`: the falsified HTTP proxy coefficients, used only to reproduce the serial screen.
- `native-token-diagnostic.json` and the identification report §2: θ0 ≈ 12.31 ms, θD ≈ 0.049 ms/request, θP ≈ 8.73 ms per kilo-prefill-token.

Not used: `native-profiler-evidence.tar.gz` raw trace (its derived JSON was sufficient), and isolated/scout archives beyond listing.

Environment limits, stated plainly:
- The host has no numpy, scipy or pytest, and the sandbox blocks `pip` network access.
- The prototype is therefore **stdlib-only**, and its tests are `unittest`, which pytest can also collect.
- **The existing `native_observer` pytest suite (50 tests) was not run here.** It needs numpy/scipy. I did not modify it.

**Fact vs hypothesis.** "F" items are computed from Train artifacts by the committed prototype; each carries its caveats. "H" items are hypotheses awaiting the §9 experiments.

## 2. New facts from Train artifacts (prototype output)

Reproduce with the command in §11. Numbers below are from that run.

**F1: Batch iteration law, profiled GPU kernel busy.**
- Fitted law: τ ≈ 7.76 ms + 0.38 ms·D + 0.27 ms/kKV + 39.9 ms·max(0, s − knee)/1000 + 4.0 ms·QK/1e6.
- Here s is scheduled tokens and D is decode requests. The fit uses nonnegative least squares, with knee 32 chosen by profile.
- Leave-one-composition-cell-out (13 cells, 84 batches), MAE / max error:
  - **hinge batch law: 1.36 / 3.35 ms**
  - affine-in-tokens: 3.02 / 14.1 ms
  - **per-request-serial: 26.7 / 191 ms**
- Raw cells agree with this. A 128-token prefill is 10.4 ms busy and a 1-decode step is 8.7–9.4 ms, so they are nearly equal. A 2048-token chunk is 103 ms with 2048 context and 121 ms with 4096 context.
- Caveat: the knee is **not identified**. Support has no batch between 13 and 128 tokens; the SSE-flat set is [32, 64], and `knee_identified=false`.
  Physically (4B BF16 on a 4090) the roofline knee is plausibly about 100–200 tokens. That is a hypothesis (E1).
- Caveat: profiler data shifts timing; the GPU-span law has a 13.4 ms intercept versus 7.8 ms for kernel busy. Only two repeats. Max decode background is 4.

**F2: Serial versus batch capacity screen** (16 Train episodes, 90 s windows from empty).
- The exclusive-serial HTTP reading is again rejected: 2449 and 2056 completions against a bound of 2004.
- Under the batch law's *lower-bound* terms, every episode is consistent. For example, train-160001-1 has 2449 completions against a batch bound of 5640.
- The batch bound is a necessary-condition screen, not a capacity estimate. It is loose because it ignores host time.

**F3: Unprofiled wall time exceeds profiled GPU busy per decode iteration.**
- Token-progress θ0 = 12.31 ms (unprofiled) versus 8.7 ms decode-1 kernel busy (profiled). The law's D=1 prediction gives a gap of 4.1 ms.
- So roughly 3.6–4.1 ms per iteration is **not hidden by async scheduling**. The model needs τ_wall = τ_gpu + h, or max(·) — distinguishing these is E5.
- Cross-run comparison: different runs, profiler on versus off, so this is not a decomposition.
- Under load, train-160001-1 produced 1223 output tok/s at a mean running count of 17.6, which implies about 14.4 ms per iteration. That is consistent with θ0 + θD·D plus prefill share.

**F4: TTFT versus prefills already ahead.**
Observational, so a correlation, not causal proof. "Ahead" means earlier-sent complete requests still awaiting first content at send time. Values are send→first-content.

| prompt | ahead 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 4096: median s / ≤0.5 s share (n) | 0.310 / 0.997 (581) | 0.454 / 0.67 (246) | 0.624 / 0.13 (78) | 0.869 / 0.00 (19) | 1.126 / 0.00 (6) |
| 128: median s / ≤0.5 s share (n) | 0.066 / 1.00 (3742) | 0.071 / 1.00 (2560) | 0.066 / 1.00 (1692) | 0.066 / 1.00 (1008) | 0.064 / 1.00 (546) |

- The law predicts 0.239 s of exclusive GPU prefill per p4096. The measured increment per additional long prefill ahead is about 0.15–0.25 s, because the head is partially done.
- For p128, "ahead" requests share iterations, as F1 predicts.
- This is the first state variable with a **mechanistic** explanation of the workload-dependent action ranking.
- Caveats: single-class cells only, and no mixed short-behind-long data, which is OOD. Gateway "awaiting first" includes transit time.

**F5: EOS is not an exogenous mark.**
Temperature is 0, and each logical offer is paired across the two arms of a block.

| cell | pairs | identical | mean abs diff | diff over 10% |
|---|---|---|---|---|
| p128-m128 | 3175 | 0.74 | 1.41 tok | 0.11 |
| p128-m256 | 989 | 0.64 | 1.81 | 0.03 |
| p4096-m128 | 224 | 0.82 | 1.90 | 0.15 |
| p4096-m256 | 146 | 0.76 | 1.93 | 0.05 |

- Output length is mostly determined by the prompt, but batch-dependent numerics change it in 18–36% of cases.
- **Different actions mean different batches, which can mean different EOS.** I have not shown the action causes this (same-action replicates are missing; E4).
- Remaining work must be modeled as a latent prompt mark plus a stochastic perturbation. A fixed per-prompt length would be wrong.

**F6: Request age is a strong remaining-work state for this workload.**
Leave-one-seed-block-out MAE of remaining tokens, unconditional mean residual versus age-conditional mean residual life:
- p128-m128: 12.0 → 6.1
- p128-m256: 25.1 → 9.8 / 25.3 → 9.6
- p4096-m128: 14.7 → 9.7 / 14.7 → 8.7
- p4096-m256: 30.5 → 17.0 / 29.6 → 13.2

Length support is narrow, for example [31, 61] and [71, 166]. This comes from the synthetic 4-topic × 3-length prompt recipe.
**This is workload-specific.** Heavy-tailed production lengths can make mean residual life *increase* with age. Age informativeness must be re-estimated, which is an OOD gate.

**F7: The budget-to-concurrency map is set by the cost spec, not by GPU work.**
Cost = 0.259 + 0.000758·p + 0.005208·m gives 1.02, 1.69, 4.03 and 4.70 for the four cells.

| budget | p128-m128 | p128-m256 | p4096-m128 | p4096-m256 |
|---|---|---|---|---|
| 16 | 15 | 9 | 3 | 3 |
| 32 | 31 | 18 | 7 | 6 |

These match the observed maximum running counts exactly.
At p128, budget 32 is effectively the engine limit (`max_num_seqs` = 32), so vLLM waiting reached 5–6.
At p4096, budget 16 already allows two long prefills ahead, which is where TTFT-good drops to 13% (F4).

**F8: KV is not binding inside the current action support.**
- kv_usage ≈ 1.263e-5 per resident token, about 79.2k tokens at full. Maximum residual is 0.026 (block granularity).
- Maximum observed usage is 0.37.
- At most 7 concurrent long requests (budget 32) gives about 0.4.
- A KV chance constraint is inactive unless action support exceeds about 70 budget units for p4096. It is not worth modeling now; keep it as a guard (§7.3).

## 3. Plant definition for a batch-aware controller

There are three clocks:
- the Gateway monotonic clock, used for the ledger and admissions;
- the EngineCore monotonic clock, used for native snapshots (same boot domain, verified per frame with age ≤ 0.5 s and width ≤ 0.05 s);
- the client clock, used for scheduled arrival and first/terminal receipt.

The engine runs iterations of variable length τ_j, which is roughly 9–14 ms for decode and 100–125 ms for long chunks. The controller runs at a fixed period T_c = 1 s.
Native snapshots arrive about 14 Hz, but irregularly (maximum gap 0.2 s). All three are **asynchronous**: the decision at t_k uses only data with availability ≤ t_k.

**State** x(t), split between exact and latent parts:
- Exact Gateway ledger L: for each admitted request, cost c_i, class (p_i, m_i), timestamps admitted/first/terminal. Also R = Σc_i, applied budget u, version, and pending command.
- Per active request, a phase:
  - φ_i ∈ {transit (Gateway-admitted, not yet native: the 1865 "unknown, not 0" marks), waiting, prefilling(progress π_i), decoding(age a_i)};
  - the latent remaining output ℓ_i, as a belief over the age-conditional residual distribution from F5/F6.
- Engine aggregates: running, waiting, pending_batches, kv_usage.
- Slow parameters: host gap h (F3), service drift ρ (observed iteration rate ÷ law), and class arrival intensities λ_c.

**Action**: u_k ∈ U = {16, 32}, the current permit. It is applied with a lag of 35–80 ms and acts only on *future* admissions through a_i = 1{R + c_i ≤ u}. It does not evict.
16/32 are **cost** budgets, not concurrency (F7).
vLLM token budgets (2048), `max_num_seqs` (32) and chunking are fixed plant parameters, **not** Gateway actuators.

**Disturbances**:
- marked arrivals (class, rate, burstiness);
- EOS mark plus batch perturbation (F5);
- service drift (thermal/clock, NVML only partially recorded);
- host jitter, HTTP transport, failures and cancellations (orphan work invalidates scope).

**Observations**:
- Exact ledger events.
- Native snapshots giving running, waiting, per-request `output_tokens`/`computed_tokens` (scheduler bookkeeping can lead GPU completion by about one async iteration) and kv.
- Delayed HTTP first/terminal labels.
- Phase transit → native is only partially observable (frames sometimes miss it).

**Dynamics (iteration level).**
1. Each iteration schedules decode-first: every decoding request gets one token, if seqs ≤ 32. FIFO prefill chunks fill the remaining 2048 − D tokens.
2. τ_j = τ_gpu(s_j, D_j, K_j, QK_j; θ) + h + noise, with τ_gpu from F1. Decoders' ages increase by 1. A decoder terminates when age reaches ℓ_i. A prefill finishing its last chunk emits its first token.
3. The Gateway releases c_i at terminal cleanup.
4. Aggregated over T_c this is a **hybrid automaton**: continuous token progress plus discrete membership, with a piecewise-linear iteration-time law (memory-bound floor, then a compute-bound hinge).

## 4. Model-class comparison

| class | captures batch sharing | captures long-prefill head-of-line | remaining work | identifiable from current data | verdict |
|---|---|---|---|---|---|
| M/G/1 or serial queue on HTTP proxy | no | yes (wrongly for short) | no | coefficients yes, interpretation **falsified** (F2) | reject as a service law |
| Fluid (rates by class, processor sharing) | approximately | no (PS smooths head-of-line away) | mean only | partly | baseline for long horizons only; fails F4's step-like TTFT |
| Linear state-space in (R, Q, D, W_prompt) | no | via W_prompt only linearly | no | yes (regression) | fails at the knee; useful as a persistence baseline |
| **Hybrid iteration-level, roofline law, this proposal** | yes | yes | age-conditional | law: mostly, knee open (E1); host gap needs E5; EOS needs E4 | **primary candidate** |
| Piecewise (memory-bound/compute-bound regimes) | yes | yes | needs add-on | yes once knee is bracketed | equivalent to the hybrid aggregated per regime; good for fast rollout |
| Learned transition (GBM/NN on snapshots) | implicit | implicit | implicit | 8 seed blocks, single-class cells | **not identifiable for OOD mixtures**; a residual corrector at most |
| Probabilistic (scenario/Bayesian over θ, ℓ, λ) | wrapper | wrapper | yes | needs a calibration split | required wrapper around the hybrid for chance and lower-bound scoring |

Recommendation: hybrid core, a piecewise fast surrogate for rollout, and probabilistic scenarios over (λ_c, ℓ_i, ρ, h). A learned residual is allowed only after E1–E5, and only on a disjoint calibration split.

## 5. Identifiability, observability and causality

- **Mechanism versus reduced form.**
  - The reduced form "budget → goodput" needs many randomized blocks. Today there are 8, and paired arms diverge after the first different action.
  - The structural pieces are identifiable from *observed variation in composition*, whatever the action that produced it: the iteration law (s, D, K → τ), the scheduler rule (deterministic code) and EOS (prompt → ℓ).
  - This requires the noise to be iteration-local and not caused by composition itself.
  - Thermal drift is a common cause of both load and slowdown. NVML coverage is partial, so this is an open confounder; record it in E1.
- **Exogeneity of composition.** Composition is caused by arrivals and actions. Actions were randomized in complementary sequences, so for the *law* the remaining risk is support, not confounding.
- **Not identifiable today:**
  1. The knee in 13–128 tokens (F1).
  2. Whether the host gap h is additive or max-overlapped (F3).
  3. Whether EOS differences are action-caused or pure nondeterminism (F5).
  4. Mixed-class head-of-line, short behind long; no data exists.
  5. Transport versus engine split of first-residence. The HTTP proxy's 38 ms intercept is not separable without per-request engine first-token timestamps.
- **Observability.**
  - Phase and age are observable from native snapshots. Remaining ℓ_i is latent; the belief comes from the class mark plus age.
  - Transit work (1865 marks) is observable only as "admitted, not yet native". Carry it explicitly; never zero-fill it.
  - Iteration count per interval is **not** directly observed. It is inferred from Δoutput_tokens of continuing decoders: one token per iteration per decoder, so Δoutput is an exact iteration counter while a decoder persists.
  - That makes τ_wall observable at snapshot resolution without the profiler. This is the main unprofiled identification lever (E5).

## 6. Evaluation: one-step, multi-step, OOD

- **One-step (1 s):** predict R, running, waiting, the transit count and per-class first/terminal events. Score with grouped leave-one-seed-block-out against persistence and fluid baselines.
- **Multi-step (3 s and 10 s):** open-loop under the *recorded* action sequence. Score the TTFT-good indicator for every offered request using the AUC/Brier score of P(TTFT ≤ 0.5 | state at offer), computed from the predicted prefill backlog.
  This is the **action-relevant** target. Accuracy on R is not, because R persistence reaches 0 MAE while TTFT P95 is 718 ms (scout §"新的警訊").
- **Action-relevant check:** on paired arms, compare predicted against observed *whole-sequence* contrasts of composite-good per phase. Do not compare pointwise.
- **OOD gates; abstain if any is outside support:** batch size between 13 and 128 tokens (until E1), mixed classes, prompt lengths outside {128, 4096}, D > 31, kv > 0.4, age above the training maximum per class (counted in `outside_support`), or a θ-drift alarm (ρ outside a calibrated band).

## 7. Controller design

### 7.1 Objective
Maximize the expected count of composite-good completions (TTFT ≤ 0.5 s from scheduled arrival, request-mean TPOT ≤ 50 ms, natural completion) over a fixed cohort.
The cohort is the active requests plus offers in [t, t + D_eval]. Rejections, misses and unresolved requests score 0. Use the existing spec §4 lower-bound tail scoring.

The physical sub-models enter as follows:
- P(TTFT good) comes from the predicted FIFO prefill backlog (F1 + F4, `prefill_backlog_s`) plus a transport intercept.
- P(TPOT good) comes from the long-prefill insertion count during each decoder's remaining life (`interference_tpot_ms`).
  With θ0 = 12.31 ms, a p4096-m128 decoder tolerates about 9 insertions and an m256 decoder about 20 (prototype output).

### 7.2 Horizon and rate
- T_c = 1 s, N ∈ {1, 3} control moves, D_eval = 5 s, plus a 10 s tail under the backup policy.
- Rationale: action effects last at most the prefill (0.24 s) plus decode (up to 1.4 s) of the admitted cohort.

### 7.3 Constraints
- u ∈ U and data freshness (existing guards).
- KV ≤ 0.9 as a hard guard, not a chance constraint, since it is non-binding (F8).
- Abstain outside support (§6).
- No R ≤ u post-reduction constraint, as in the existing spec.

### 7.4 Solver and overhead
- Enumerate at most 2^N = 8 plans × M = 32 shared scenarios.
- Iteration-level rollout costs about 250 iterations × 32 requests per scenario per 3 s, which is too slow in Python.
- Use the **piecewise event surrogate** instead. Events are chunk completions (deterministic from the law), EOS (sampled ℓ), arrivals and admissions; decode progress between events is closed-form at rate 1/τ(D). That is about 10² events per rollout.
- The 100 ms deadline must be *measured* on target hardware; it has not been measured.

### 7.5 Fallback and the baseline it must beat
- The **reactive prefills-ahead rule**:
  - u = 16 when the Gateway long-prompt awaiting-first count is ≥ 1 or the recent long-cost share is above x;
  - otherwise u = 32;
  - x is fixed on Train before any comparison.
- This uses the same F4 state with no lookahead. If MPC ties it, lookahead added nothing.
- Fallbacks in order: the reactive rule, then the last applied budget (if transport is down), then Fixed16 only if the reactive rule's state is stale.

**H-act (hypothesis).** For pure p4096, a budget allowing at most one long prefill ahead (about 8 cost units) may beat both 16 and 32. That is an action-support change: it needs a new permit and is out of scope today. Do not let a horizon ablation stand in for it.

## 8. When multi-step beats one-move, and an ablation that discriminates

Multi-step can only help when:
- (a) the first action's consequence on the scored cohort depends on later actions, meaning the effect persists past T_c; **and**
- (b) the optimal later action differs from the current one in a forecastable way, for example a class or rate shift that is predictable from past-only information; **and**
- (c) a one-move plan with the correct terminal continuation cannot represent it.

In Train data, (b) is absent: offered phases follow a hidden 30/30/30 s schedule that a past-only forecaster cannot anticipate.

**Ablation (2 × 4 × 2).** Same model, same scenarios, same terminal backup (the reactive rule, not Fixed16), same D_eval.
- Factor A: N ∈ {1, 3}.
- Factor B: workload
  - (i) stationary single-class;
  - (ii) stationary mixed p128/p4096;
  - (iii) unannounced step change of mix;
  - (iv) step change with an observable precursor ramp.
- Factor C: forecast — past-only, or a Train-only oracle of the phase schedule (an upper bound on the value of information, never deployable).

Predictions that would falsify my prior:
- N3 > N1 in (i)–(iii) with a past-only forecast. Diagnose this as a terminal-value or model artifact before believing it: rerun N1 with N3's tail value.
- N3 ≈ N1 even under the oracle in (iv). That shows the plant's memory is too short for lookahead to pay.

**Cheap pre-check (shadow mode, no actuation):** measure the first-action disagreement rate between N1 and N3 on live state. If it is below 5%, multi-step cannot change goodput materially, so stop.

## 9. Experiments that can falsify each major assumption

All use new protocol versions and new Train seeds; never 160021–160024.

| # | Assumption | Experiment | Falsified if |
|---|---|---|---|
| E1 | Roofline hinge law; knee 100–200 tokens | Unprofiled q ∈ {32, 64, 96, 128, 192, 256, 384, 512, 1024} × D ∈ {0, 8, 24}. Measure τ from Δoutput of a persistent decoder (§5); add one profiled replicate. | The LOCO MAE of the hinge is not below the affine law, or the curvature is not hinge-like |
| E2 | Prefill-ahead *causes* the TTFT steps | Randomized microbursts of k ∈ {0, 1, 2, 3} p4096 prompts, then a probe p4096 **and a probe p128** at +δ | Probe TTFT is not ≈ intercept + backlog(k) ± 25%. The **short probe behind long** prediction (+~110 ms per chunk ahead) fails |
| E3 | TPOT interference law | Background D = 8 decoders of known length; inject k long prompts | Request-mean TPOT is outside θ_w + k·extra/(ℓ−1) ± 20% |
| E4 | EOS perturbation is action-independent noise | Replay the same tape twice under the same fixed u, and once under the other u | Same-action agreement ≫ cross-action agreement, which would mean the action causes EOS change and the model must couple them |
| E5 | Host gap h is additive and roughly constant | Δoutput-implied τ_wall at D ∈ {1, 8, 16, 31}, compared with profiled busy | The gap scales with D, or is not additive (max-overlap) |
| E6 | Transit state is short and bounded | Gateway admitted → native-visible delay distribution | Its tail is above T_c, which would require transit dynamics in the state |
| E7 | Single-class physics composes for mixtures | 2-class mixtures at fixed u ∈ {16, 32} | Predicted per-class TTFT-good is outside paired-block intervals |
| E8 | KV non-binding | Log kv in all of the above | kv > 0.6 or any preemption |
| E9 | Lookahead value | §8 shadow disagreement, then the factorial | See §8 |

Order: E1, E2 and E5 run together in one pilot (they share the setup). E4 is cheap: add a replicate arm. E7 comes before any controller comparison.
Power: use E2 and E4 variances to size the paired-block confirmatory test before any held-out data is generated.

## 10. Limitations

- Profiler-perturbed batch timings, with two repeats. Unprofiled law coefficients are not identified.
- The batch bound is a necessary-condition screen and does not estimate capacity.
- F4 is observational. "Ahead" uses client send and first receipt, so it includes transport. An online version must use the Gateway awaiting-first count, whose equivalence is not yet checked.
- EOS and age results come from a synthetic 12-variant prompt recipe at temperature 0. They do not transfer to production.
- Single-class cells only. Mixtures are untested.
- No controller was run, no solver timing was measured, and no performance claim is made.
- This repo's numpy-based tests were not run here (environment).

## 11. Reproduction

Files (owned):
- `experiments/mpc-research/native-v2/round1/opus.md`
- `experiments/mpc-research/native-v2/round1/opus_batch_models.py`
- `experiments/mpc-research/native-v2/round1/test_opus_batch_models.py`

```bash
python3 -m unittest discover -s experiments/mpc-research/native-v2/round1 -p 'test_opus_*.py'
cd experiments/mpc-research/native-v2/round1
R=../../../../results/mpc-native-4090-scout
python3 opus_batch_models.py --profile $R/native-profile-diagnostic.json \
  --train-archive $R/native-identification-train-evidence.tar.gz \
  --prefill-proxy $R/native-prefill-combined.json --output /tmp/opus-diag.json
```

Python 3.14.4, stdlib only. The run takes about 1.2 s and reads the archive in memory without extracting it to the repo.
The output JSON is not committed, because it lies outside my owned paths. It is regenerated deterministically.
