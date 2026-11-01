# Opus Round2: adversarial review of Astra, Gemini, Sol (and of Opus Round1)

Status: exploratory TRAIN-only review. Sources are the frozen Round1 files at
integration commit `bc217acf9db3f6222026c0dd345db77a379d6151`
(`experiments/mpc-research/native-v2/round1/{astra,gemini,sol}.*`). Held-out
seeds 160021-160024 were not opened or used. No GPU, SSH, provider or network
call was made. All proposals, including this one, remain unaccepted hypotheses.

Labels used below:

* **[EXEC]** – executable counterexample in `round2/opus_counterexamples.py`
  (focused test in `round2/test_opus_counterexamples.py`), or a direct call to
  the peer's frozen code.
* **[PROOF]** – short proof of a defect.
* **[UNSUP]** – assertion without evidence; not shown false.
* **[SOUND]** – idea I checked and would keep.

## 0. Bottom line

1. **No peer, and not my Round1, has identified a model that shows native MPC
   goodput gains.** The strongest shared and sound result is negative and
   structural. Gateway budget is cost, not concurrency. Releases need marked
   request identity (Sol). One-move with the same window and tail accounting is
   the correct baseline (Astra, Sol).
2. **Decode iteration wall time is already identifiable from unprofiled
   snapshots. This refutes part of my own Round1.** Each output-progressing
   decoder advances by one token per engine iteration, so Δ`output_tokens`
   counts iterations. On 8,276 clean pure-decode TRAIN intervals:
   * Wall time ≈ 9.67 ms + 0.035 ms·D + 0.17 ms per thousand decode-KV tokens.
   * Leave-one-seed-block-out MAE is **0.77 ms per interval**.
   * Competing predictors do much worse: the profiled kernel-busy law gives
     6.11 ms, the profiled law plus my Round1 F3 "host gap" gives 25.6 ms, and
     the θ0/θD token model gives 10.6 ms.

   My Round1 F3 (a ~3–4 ms host gap) and the profiled per-decoder slope
   0.38 ms/D are therefore **wrong outside D ≤ 4**. Details are in §4.
3. **Prefill duration is *not* identified from existing snapshots.** NNLS of
   the `computed_tokens` delta gives 82 ms/k (p128) vs 30 ms/k (p4096). The
   latter is incompatible with the observed long-prompt TTFT and is close to
   the FLOP floor. Both Astra and Sol warned that `computed_tokens` includes
   scheduled, in-flight work; this is consistent with that warning. Prefill
   needs forward spans.
4. **Single next experiment (§7):** a pre-frozen TRAIN mixed-class tape.
   * p128 Poisson background plus randomized-onset p4096 bursts with
     k ∈ {0,1,2,3}, run under paired Fixed16/Fixed32 blocks.
   * The Mayor facade is on in most blocks; one replicate block has it off.
   * This tests short-behind-long TTFT, the chunk-duration law at unprofiled
     shapes, observer effect and the whole-policy budget effect on mixtures,
     all at once.

## 1. Gemini (`round1/gemini.md`, `gemini_timing_contract.py`)

### Strongest claims

* [SOUND] §2.3/`ClockDomain` + `RequestTimingTrace`: never subtract across
  clock domains, and reject non-monotone traces. This is a useful guard,
  although §3.1 violates it (below).
* [SOUND] The statement that profiler traces must not be used as serving
  calibration (§3.2) is correct in direction. My §4 quantifies it: the profiled
  per-decoder slope is roughly 10× too steep at D ≈ 17.
* [SOUND] `ProbeOverheadGuard` as a concept: observation age and skew limits
  checked *before* use.

### Weakest claims and defects

1. **[PROOF][EXEC] §5.3(1) "hard reservation invariant"
   `R(t) ≤ max(R(0), u_applied(t))` is false.**
   * Counterexample: admit 30 cost units under budget 32, then cut the budget
     to 16. Releases are not forced, so R = 30 > max(0, 16).
   * The correct invariant is Astra §5:
     `R ≤ max(R_0, max_{s≤t} u(s))`, i.e. the history maximum, plus the cost of
     one admitted offer if admission is check-then-add.
   * Witness: `reservation_invariant_witness()`.
   * Consequence: a solver that hard-constrains the false form becomes
     infeasible immediately after any cap cut, and the declared fallback then
     fires spuriously.
2. **[PROOF][EXEC] §6.1 "iff" horizon theorem is false.** The claim is
   "multi-step adds value **iff** T_settle > T_slot".
   * Only one direction holds. Without carry-over, every first action is
     independent of later moves, so N = 1 equals N = N.
   * The "if" direction fails. With carry-over and an **optimal** backup
     (cost-to-go), N = 1 is Bellman-equivalent to any N. Extra moves add value
     only to the extent that the declared backup or continuation is
     suboptimal.
   * `horizon_witness()` (arrivals (2,2,2,1,1), two-slot memory, actions {1,3}):
     * With carry-over: 0 first-action disagreements between N = 1 and N = 3
       under the optimal DP backup, but 2 (fixed-low) and 4 (fixed-high)
       disagreements under fixed-cap backups.
     * Memoryless: 0 disagreements under every backup.
   * Design consequence: the multi-step-vs-one-move ablation is a test of the
     **backup policy**. A one-move arm with a good (development-fitted) backup
     is the required baseline. Astra and Sol both say this; Gemini does not.
3. **[EXEC] The confirmatory protocol cannot confirm anything.** I ran
   Gemini's own `NativeFairStatisticalProtocol.evaluate_paired_blocks` against
   the frozen code.
   * §8.1 makes the four held-out seeds the confirmatory set, but also
     requires K ≥ 8 blocks. The code raises
     `CRITICAL PROTOCOL VIOLATION` on any held-out seed, so the declared
     confirmatory evaluation is unreachable by construction.
   * With four blocks, the exact sign-flip p has a floor of 1/16 = 0.0625
     (one-sided) or 0.125 (two-sided), even if every difference favours B.
     Running the frozen code on four TRAIN blocks, each +3.3 good/s, returns
     p = 0.0625 and `EQUIVALENT_OR_INSUFFICIENT`. The witness is
     `sign_flip_p_floor`.
   * The code is one-sided (it counts `>= observed`, not `|.|`). The report
     (§8.1) says two-sided.
   * The solver overrun rate is divided by **offered requests**, not by
     **control decisions**. Eight blocks with 9 overruns out of 90 decisions
     each (10%) yield `solver_overrun_rate = 0.009`, guards pass, and the
     verdict is `CONFIRMED_SUPERIOR`. A 10× under-estimate makes the 1%
     guard ineffective.
   * The §8.1 "rejection-rate ≤ baseline + 5%" guard is not implemented.
   * `ProbeOverheadGuard().max_slot_fraction` defaults to 0.05, but the
     report says 2%.
4. **[EXEC] `RequestTimingTrace.is_composite_good()` has no
   success/finish-reason field.** A trace for a failed or truncated stream with
   good timing is counted as good (executed: `True`). HTTP failures are not
   EOS and must never score as goodput.
5. **[PROOF] The §3.1 TTFT "exact non-overlapping decomposition" does not
   telescope.**
   * It omits `t_vllm_enq − t_admit` and `t_first_token_sampled − t_batch_end`.
   * It sums a *per-batch* GPU span into a *per-request* identity.
   * It subtracts Gateway, engine and GPU clocks that §2.3 itself declares
     non-comparable.
   * The CPU class (`RequestTimingTrace`) quietly uses one clock domain, so
     the code does not implement §3.1.
6. **[EXEC] Units/physics misuse of θP.** §4.1 uses
   `τ_iter = θ0 + θD·D + θP·P_k` with θP = 8.73 ms per thousand
   prompt tokens, applied to `P_k` = all prompt tokens of running
   requests with no output yet (`native_observer/native_token_model.py:60`).
   * The source report defines θP as a **decode-progress slowdown in the
     presence of prefill-phase requests**
     (`native-identification-and-model-falsification.zh-TW.md:105-109`), not a
     prefill duration coefficient.
   * Read as prefill duration, the marginal 8.73 ms per thousand tokens needs
     **≈ 866 TFLOPS**, and a 2048-token chunk iteration at 48.1 ms needs
     ≈ 335 TFLOPS.
   * For comparison, the RTX 4090 dense BF16 peak is ≈ 165 TFLOPS with FP32
     accumulate (≈ 330 with FP16 accumulate). Qwen3-4B has ≈ 3.633e9
     non-embedding matmul parameters, which is about 7.27 GFLOP per token.
   * The profiled 2048 chunk (103 ms) needs ≈ 157 TFLOPS, which is tight but
     feasible.
   * Witness: `compute_floor`, `required_tflops`, `gemini_affine_iteration_ms`.
7. **[EXEC on TRAIN] §6.1 physics is contradicted by the TRAIN data.**
   * The claim is "KV pinned 10–20 s" for p4096. Observed decode residence
     (`terminal − first_content`) has a maximum of 2.8 s (p4096-m128) and
     3.3 s (p4096-m256), with KV usage ≤ 0.37.
   * The §8.1 superiority margin of 0.5 good/s exceeds the whole p4096
     per-episode composite goodput (0.76–1.21 good/s; arm differences
     ≤ 0.03 good/s). So the margin is unattainable in exactly the regime where
     Gemini claims multi-step is "decisive".
8. **[UNSUP]**
   * The 15 ms probe latency and 2% slot limits have no measurement.
   * "Fixed16 then complete drain" is an artificial continuation that changes
     the scored cohort composition.
   * N = 5 in the ablation is horizon inflation with no stated mechanism.

**Gemini acceptance gates.**

* Replace §5.3(1) with the history-max form.
* Delete the iff theorem, or restate it as "necessary, not sufficient".
* Rewrite the statistics:
  * The confirmatory unit is the number of fresh TRAIN-protocol seed blocks
    with K ≥ 5 for a one-sided exact test (floor 1/32), or ≥ 6 two-sided (floor 1/32); or use a
    pre-registered effect-size interval instead of a p-gate.
  * Overrun rate counts decisions.
  * Implement the rejection guard.
  * `is_composite_good` requires success and a finish reason in {stop, length}.
* Drop θP as a prefill duration.

## 2. Astra (`round1/astra.md`, `astra_horizon_diagnostic.py`)

### Strongest claims

* [SOUND] §5 finite-tail accounting (`score_bounds`, `contrast`).
  * `L = K` (settled good) and `B = K + V`.
  * `J(B) − J(A) ∈ [L_B − B_A, B_B − L_A]`.
  * It never credits an unresolved request, and it marks TTFT-expired
    requests irreversible using only information at the cutoff. I checked the
    code and it does not read a future `first_content_at`.
  * This is the right scoring discipline for any horizon or tail comparison.
* [SOUND] §6 separates the prediction window D, optimized moves H and tail
  length; recourse is not clairvoyance (`V_open ≤ V_tree ≤ V_perfect` for
  nested feasible classes under one model). The ablation with an
  equal-compute arm and "identical first actions/unbinding caps are not
  evidence" is the best discriminating design among the four reports.
* [SOUND] §5 reservation invariant uses the history maximum (correct, unlike
  Gemini).

### Weakest claims

1. **[EXEC] §3 literal duration family
   `log τ = β·scaled(φ)` (log-linear in token counts) fails physically.**
   * The fit is on the profiled batches with the same knee-free features.
     Leave-one-condition-out MAE is 2.90 ms (max 18.6), vs 1.36 ms (max 3.35)
     for the Round1 hinge.
   * Extrapolated along a pure-prefill chunk q ∈ {128, …, 4096}, it gives
     10.5 / 16.5 / 30.0 / 100.9 / 1206 ms.
     * The 1024-token point is **below the FLOP floor** (46.9 ms at
       165 TFLOPS).
     * The 4096-token point is about 6× the floor, because an exponential in
       token count cannot represent a compute-bound linear regime.
   * Witness: `loglinear_vs_hinge_loco`.
   * Astra hedged ("hypothesis", "fit a smaller model first"), so this is a
     defect of the default family, not of the method. A log *link* on a
     **piecewise linear** compute law is acceptable; log-linear features are not.
2. [UNSUP] No quantitative physics anchors the 5 s window or 1 s cadence. TRAIN
   residence (p128-m128 ≤ 0.92 s) means most p128 reservations release
   inside one control period. For that cell, H > 1 has nothing to coordinate.
3. Deployment complexity: tree search with tail extension and an
   equal-compute arm is a large pilot before any action-relevant prediction is
   qualified. I would gate it behind §6 gate G3 below.
4. A possible tightening of `score_bounds` (proposed, not executed): if a
   pending request already shows ≥ 2 generated tokens and
   `(cutoff − first)/(m_i − 1) > 0.05`, its TPOT is already impossible and it
   can move from `possible` to `expired`. With n = 1 unknown this is unsafe, so
   apply it only with progress evidence.

## 3. Sol (`round1/sol.md`, `sol_counterexamples.py`)

### Strongest claims

* [SOUND][EXEC by Sol] §5 finite alias witness (`release_certificate`).
  * Two worlds with identical aggregate projection and even identical total
    remaining work give residual splits (1,9) vs (5,5). Their reserved cost at
    2 s differs (8 vs 16).
  * Every deterministic scalar predictor errs by ≥ 4 cost units in one world.
  * The proof is correct within the stated toy. Its conclusion — a release
    belief must be marked by request identity and cost, not by a fluid scalar —
    is the most durable structural result of Round1.
* [SOUND] §1 two-level identification: whole-policy effects are identifiable
  from randomized TRAIN assignments even if GPU internals stay latent. I agree
  and build on it in §6.
* [SOUND] §7 "a one-move planner with the same long prediction window already
  prices future congestion"; randomized local action at pre-action eligible
  states with a declared common continuation. This is the cleanest *causal*
  test of first-action value.
* [SOUND] `available_at_decision`: source time ≠ availability time. Offline
  effective-time joins are not online forecasts.

### Weakest claims

1. [PROOF, scope] The alias witness relies on a unit-token-rate,
   parallel, no-delay toy.
   * On TRAIN p128-m128, every decode residence ends within 0.92 s. At the
     proposed ≥ 1 s control period both worlds have released fully, which
     reproduces Sol's own "equal again at horizon 9" case.
   * So the witness is decision-relevant only for sub-second horizons or the
     long-output/long-prompt cells (residence up to 1.7–3.3 s). The structural
     point stands; its native magnitude is unmeasured.
2. [UNSUP] §4 `log τ_j = β_mode + f_mode(...)` leaves `f_mode`
   unspecified. Without a physical floor (§2.1 above) it inherits Astra's risk.
3. Experimental validity: randomized local actions at eligible states need
   enough eligible states with *binding* caps. TRAIN p128 under budget 32
   rarely binds (Round1 table), so the eligible-state rate must be measured
   before power is claimed.

## 4. Challenge to my own Round1 (`round1/opus.md`)

* **F3 is refuted ([EXEC], `decode_wall_*`).** I claimed the unprofiled decode
  wall exceeds profiled kernel busy by ~3.6 ms (θ0 = 12.31 vs 8.75 ms). By
  decode-bin, unprofiled wall minus profiled busy is +1.18, −0.10, −1.34,
  −3.09 and −4.34 ms at D ≈ 2.3, 5.7, 9.0, 13.5 and 17.1 (p128). For p4096 it
  is +0.43 ms (D ≈ 1.8) and −2.11 ms (D ≈ 5.0).
  * The real host/launch gap is ≤ ~1 ms at small D.
  * θ0 = 12.31 is inflated by intervals containing prefill.
  * The profiled D-slope (0.38 ms/request, fitted on D ≤ 4 under the
    profiler) does not transfer; the unprofiled slope is ≈ 0.035–0.05.
  * Most likely cause: CUDA-graph replay makes small-batch decode nearly flat
    in D, and the profiler distorts it.
* **F2 survives.** The batch capacity *lower bound* uses only the 7.755 ms
  intercept (below the ~10 ms wall) and the 39.85 ms per thousand above-knee
  slope (below the ≈ 44 ms-per-thousand FLOP floor). Both are conservative
  under the corrected wall, so 2449 > 2004 remains explained by batching.
* **TPOT insertion numbers barely move.** Using the 10.0 ms unprofiled wall
  instead of 12.31 ms gives the same ≤ 9 long-prompt insertions at L = 50 and
  21 instead of 20 at L = 112 (`tpot_max_long_insertions_round1_vs_corrected`).
* **My prefill numbers remain profiled-only.** The snapshot regression (§0.3)
  cannot confirm or refute them.

## 5. Mayor measurement facade (independent assessment)

The plan reuses PR32 `ExternalCapture`, `scheduled_work`, `step_key` and
`WorkerRecorder`. It records current-stream CUDA forward spans plus host
schedule and commit times, resolves only after drain, and makes no live
predictor or synthetic requests. This is the right direction: it measures the
missing prefill duration (§0.3) on standard traffic without the profiler.
Conditions:

1. **Scope.** A current-stream event pair brackets the work enqueued on that
   stream between the two records. It:
   * includes dependency waits on other streams;
   * excludes async D2H/sampler/output copies on other streams;
   * excludes host scheduling;
   * excludes CUDA-graph capture differences unless the mode is logged.

   Call it `forward_stream_interval`, never GPU service or busy time.
2. **Timeline anchor.** `elapsed_time` gives only relative durations. Record
   one reference event per episode (synchronised once at start, outside the
   scored window). Then `elapsed_time(ref, start_j)` places each step on the
   GPU timeline. The inter-step GPU idle (`start_{j+1} − end_j`) is the
   host-gap measurement that refutes or confirms §4 directly.
3. **Observer effect.** Event-record cost is small, but resolution after drain
   needs a bounded per-step buffer. Overflow must mark the episode unqualified,
   not drop silently. Include facade-off replicate blocks and compare goodput,
   TTFT/TPOT quantiles and the §4 Δoutput decode-wall law. Gate: |Δ decode
   wall| ≤ 0.3 ms and goodput difference inside the block-to-block spread.
4. **Async scheduling.** With async scheduling, step j+1 may be scheduled
   before j commits. Key everything by `step_key` and never infer duration from
   adjacent host commit times.
5. **Source guard 38 / pre-frozen TRAIN IDs.** This is sound. The facade must
   refuse 160021-160024 the way the Round1 loaders do.

## 6. Usable minimal batch-telemetry contract

Per engine step (`step_key`):

* Host times: `schedule_start`, `schedule_end`, `submit`, `commit` (one
  monotonic host clock + boot id).
* Forward stream interval: `fwd_start_offset_ms` and `fwd_end_offset_ms`
  relative to the episode reference event; `cuda_graph_mode` (and padded
  batch size).
* Composition aggregates:
  * scheduled tokens `s`;
  * decode requests `D`;
  * Σ decode context;
  * prefill requests, Σ prefill q and Σ prefill q·k (or the per-request
    `(scheduled_tokens, computed_before, phase)` from which these follow).
* KV blocks used/free; preemption count; running/waiting depth.

Per request:

* engine add time;
* first scheduled step;
* first-token commit step;
* finish step;
* finish reason;
* output token count;
* `max_tokens`.

Per episode:

* clock domain ids;
* facade flag;
* vLLM/model/config hashes;
* seed block, arm and tape hash.

Anything beyond this (kernel union, per-stream timing) is a mechanism check,
not a control input.

## 7. Can a reduced whole-policy model be attempted now?

**Yes, in shadow, without waiting for perfect GPU decomposition.** The
decision-relevant quantities are:

* (a) decode iteration wall, already identified (§4);
* (b) TTFT-good probability as a function of long-prefill work ahead, measured
  at the request level from Gateway timestamps (Round1 `ttft_by_prefill_ahead`);
* (c) TPOT-good as a function of long-prompt insertions during residence;
* (d) marked residual output length (EOS survival, cap-censored).

These are whole-policy conditional distributions identifiable from randomized
TRAIN assignment (Sol §1). GPU decomposition is needed only to *explain* and
extrapolate (b), not to score policies inside support.

Limits: mixed short/long traffic is **out of support** today, because every
TRAIN cell is single-class. That is why §8 is the next experiment.

## 8. Single next most informative native experiment

**Mixed-class burst tape, TRAIN-only, standard Chat, facade on, plus one
facade-off replicate.**

* Background: p128-m128 Poisson at the existing TRAIN rate.
* Overlay: p4096-m128 bursts of k ∈ {0,1,2,3} requests at onsets randomized
  per block (pre-frozen tape and hash).
* Arms: Fixed16 vs Fixed32, paired on the identical tape, order alternated.
* Blocks: ≥ 6 fresh TRAIN seed blocks (one-sided exact floor 1/64), plus one
  facade-off replicate of one arm.
* No controller, no predictor, no synthetic probe.

What it can falsify, in one run:

1. **TTFT law.** Short-behind-long TTFT is predicted from Σ queued prefill
   work using a forward-span chunk law (k = 0..3 gives a dose–response).
2. **Chunk law at unprofiled shapes.** Final chunks of size about 2D fill the
   13–128 token knee gap naturally. The forward-span law must agree with the
   profiled hinge within 10% above the knee, or the profiled law is retired.
3. **Host gap.** The inter-step GPU idle must be ≤ 1 ms median at small D
   (§4).
4. **Observer effect** (§5 gate).
5. **Whole-policy budget effect on mixtures.** Is Fixed32 − Fixed16
   composite goodput nonzero once long and short requests coexist? If it is
   zero within block spread, **no** budget controller (one-move or multi-step)
   has anything to optimize in this regime. That ends the multi-step question
   for these cells before any MPC is built.

## 9. Acceptance gates (proposed, all TRAIN)

| Gate | Test | Pass |
|---|---|---|
| G0 ledger | Reservation history-max invariant; release exactly on terminal | 0 violations over all events |
| G1 decode wall | §4 Δoutput law, leave-one-block-out | MAE ≤ 1 ms; slope ≤ 0.1 ms/D; facade-on vs off ≤ 0.3 ms |
| G2 chunk law | Forward-span duration vs hinge on unprofiled chunks | ≥ FLOP floor for every chunk; LOCO MAE ≤ 10% above knee |
| G3 action relevance | Fixed32 − Fixed16 composite goodput on §8 tape | Paired effect exceeds block spread in ≥ 1 cell; otherwise stop control work for that cell |
| G4 one-move first | One-move with development-fitted backup vs fixed/reactive | Beats both on new TRAIN blocks before any H > 1 arm is run |
| G5 multi-step | Astra/Sol ablation with equal window, tail accounting and binding disagreements | Gain appears only in predictable-lead regimes, not in a null onset regime |
| G6 statistics | Exact paired sign-flip, decisions-denominated overrun, rejection guard, success-required scoring | Implemented and unit-tested before any confirmatory run |

## 10. Reproduction and receipt

Run from the repository root (Python ≥ 3.10, stdlib only):

```bash
python3 experiments/mpc-research/native-v2/round2/opus_counterexamples.py --output /tmp/opus-r2.json
python3 -m unittest discover -s experiments/mpc-research/native-v2/round2 -p 'test_opus_*.py' -v
```

Inputs:

* `results/mpc-native-4090-scout/native-profile-diagnostic.json`
* `results/mpc-native-4090-scout/native-identification-train-evidence.tar.gz`
  (16 TRAIN episodes; the loader refuses held-out seeds)

Result: 11 tests, OK (Python 3.14.4, ~0.9 s).

The Gemini protocol witnesses in §1 were executed directly against the frozen
`gemini_timing_contract.py` extracted from `bc217ac`. They are not
re-implemented in this module, to avoid a duplicate protocol.

Not run: any GPU or native experiment; no new data; held-out untouched.
