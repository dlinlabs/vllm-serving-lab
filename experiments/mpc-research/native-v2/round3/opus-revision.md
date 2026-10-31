# Opus Round 3: bounded revision, bridge review, smallest candidate, single pilot

Status: exploratory TRAIN research. **No part of this is a qualified native MPC model,
a performance result, a GPU observation or an overhead measurement.** Held-out
160021–160024 were not opened, generated or used. Inputs: common integration
`9b02f6e`, bridge commit `1cbe251`, frozen R1 `2eee51e` and R2 `0ebc02f`.

Executed CPU checks are in `round3/opus_revision.py`, with 8 focused tests in
`round3/test_opus_revision.py`. Labels used below:

- **[EXEC]**: CPU-executed against frozen artifacts or the exact bridge object.
- **[PROOF]**: an argument.
- **[HYP]**: a hypothesis.
- **[UNRES]**: unresolved.

---

## 1. Withdrawals and narrowings (Mayor rulings and peer evidence)

| # | Original Opus claim | Revision |
|---|---|---|
| W1 | R1 F1 hinge law "LOCO MAE 1.36 ms" as held-out evidence | **Narrowed.** The knee was chosen on all batches before LOCO, so it leaked (Astra O2, Sol O3). With the knee re-selected inside each fold, MAE is **1.71 ms** (max 3.69; fold knees {0, 32, 64}) [EXEC `nested_knee_loco`]. The hinge still beats log-linear (2.90 ms). The hinge remains a fitted *mean* law on a max8 CUPTI-profiled `kernel_busy_union` target. It is **not** a certified lower envelope and not unprofiled wall/forward time. |
| W2 | R2 "F2 capacity screen survives / conservative" | **Narrowed** to: *not excluded under an assumed lower bound that is itself uncertified.* |
| W3 | R2 "decode iteration wall time identifiable; Δoutput is an exact iteration counter" | **Withdrawn.** 9.67 ms + 0.035·D is an *effective per-commit interval proxy* on selected pure-decode snapshot intervals. It is not a step time, because participation, async boundaries and step parity are unqualified. The only use I keep for Δoutput is a *lower bound* on step count (§3.1). |
| W4 | R1 stop rule "<5% One-move/N-step disagreement ⇒ multistep immaterial" | **Withdrawn.** Frequency is not impact: Astra executed 4% disagreement → 0.64 good/s, and Sol executed 1% → 20.2%. Section 4.3 replaces it with an impact-weighted discriminator. |
| W5 | R2 §8.5 "equal Fixed16/Fixed32 whole-sequence scores end the multistep question" | **Withdrawn.** Equal fixed-arm averages do not exclude gains from *switching* or from heterogeneous state-dependent effects. |
| W6 | Any implication "physical memory (KV, remaining work) ⇒ multistep gain" | **Withdrawn.** No such theorem exists. Multistep value is an empirical backup-suboptimality question (§4.3). |
| W7 | R1 F3 host gap from cross-run subtraction | Already refuted in my R2. My R2 replacement, a per-step reference event, is now **optional/deferred**: no microsecond clock mapping and no hot-path sync. |
| W8 | R1 "TTFT" from client send to first content | **Relabelled** "client-send-to-first-content". Model targets use scheduled-origin labels (Sol O5). |
| W9 | KV 0.37 occupancy | Only the *observed low-pressure domain*. It is not a hard KV bound, and a cap cut cannot reclaim already-admitted KV (Sol O4). |
| W10 | EOS paired disagreement (18–36%) as action dependence | **Narrowed.** It shows non-exogenous length under batching, not *action* causation. Same-cap replicates are needed (adopting Sol §8.4). The age value is token-age-weighted. |

## 2. Response to peer criticisms of Opus

| Critique (peer, section) | Decision | Evidence / reason |
|---|---|---|
| Randomized budget ≠ randomized composition; batch slope is not a do-effect (Astra O1, Sol O2) | **Accept** | Astra's witness (slope 0.5 vs do-effect 0) is a valid counterexample. The batch law stays *descriptive/predictive*. The causal estimand is the whole-policy ITT on all-offered outcomes. Mediator decomposition is descriptive only (§5). |
| Knee leakage (Astra O2, Sol O3) | **Accept** | W1, executed. The R2 log-linear-vs-hinge comparison inherits the leak, but the ranking survives nesting. NNLS normal equations have no rank diagnostics; future fits must report design rank/support first. |
| Warmup drops live history (Astra O3) | **Accept** | Small effect (1/932 p4096 targets, 0/10041 p128). Fix: retain warmup predecessors as state and exclude them only as targets. |
| Busy/span/commit/token targets are not interchangeable (Astra O4, Sol O1) | **Accept** | W3, W7. Each target is now named by its scope (§4.1). |
| EOS/KV/horizon language too strong (Astra O5, Sol O4–O6) | **Accept** | W4–W6, W9, W10. |
| Fit on CUPTI-profiled data (Gemini 2.1 weak #1) | **Accept** | Profiled coefficients are not transferred to unprofiled control (W1). |
| Knee "fixed at 32" (Gemini 2.1 weak #2) | **Already conceded in R1** | R1 `knee_profile` reported `knee_identified=false` (support gap 13…128). Nested folds now pick 0/32/64. |
| "No MPC controller" (Gemini 2.1 weak #3) | **Partly reject** | A reactive/One-move baseline was the intended first stage, not an abandonment. §4 now states the candidate, its One-move continuation and the multistep discriminator. Reject the implied need for "intrusive engine callbacks": the online feature cut is Gateway-side (§4.2). |
| Gemini 2.1 strong #2, "below the knee ≈ 8–9 ms weight-load floor; above it compute-bound" | **Correct the praise** | The hinge is a fitted mean. A falsified alternative does not establish a roofline family (Mayor ruling). |

My R2 critiques of Gemini (reservation invariant, iff theorem, evaluator protocol) were upheld. Gemini R2's 4.5 µs / 450 µs / 1200 µs and zero-jitter figures are unmeasured, and I do not rely on them. Gemini's 4-scalar 1 Hz telemetry remains too coarse for batch/commit joins.

## 3. Concrete review of `batch_capture_bridge.py` @ `1cbe251` (sha256 `0e24eb2c…f4a`)

I read the exact source plus PR32 `prefill_lab/{plugin,runtime,contracts}.py` at `9b02f6e`. CPU fakes only; I claim nothing about deployment.

**Sound and kept:**
- **Default off.** `register()` returns early without `PR33_NATIVE_BATCH_CAPTURE_ENABLED=1`. It requires `PREFILL_LAB_ENABLED=1` and *every* `PREFILL_LAB_{STAGES,LIVE_STATE,ENVELOPE_SHADOW,…}` = 0. This closes the hidden live-predictor path.
- **Plan validation.** `validate_plan` enforces TRAIN schema and seeds 170001–170012, rejects duplicate trial IDs, and allows only off/light observers.
- **Fail-closed capture.** Overflow, pending events, step-ID set mismatch, unresolved spans, missing terminals and epoch change all invalidate the capture while retaining the raw reply.
- **Replay and lifecycle.** END replays until an exact-SHA ACK, and a failed begin is permanently invalid.
- **Auth.** Authentication precedes the utility call.

Findings, each with the smallest fix:

### 3.1 The 8192-row buffer is near-binding for 90 s tapes [EXEC + PROOF]

`contracts.Buffer(capacity=8192)` (`contracts.py:60–62`) backs:
- the core `ExternalCapture.steps`
- the worker `records`
- the worker `pending` list. `WorkerRecorder.end` counts overflow (`plugin.py:287–296`), and resolution happens only after drain.

**Proof sketch.** Without speculation, a request gains at most one output token per engine step. So between consecutive complete TRAIN snapshots, at least max_r Δoutput_r steps occurred. The sum of these over an episode is a **lower bound** that excludes:
- prefill-only steps
- the head and tail outside the first and last snapshot
- drain

Lower bound on steps per 90 s episode (16 TRAIN episodes):

| Cell | Lower bound on steps | Max fraction of 8192 |
|---|---:|---:|
| p128-m128 | 6348–6828 | 0.83 |
| p128-m256 | 7538–7849 | **0.96** |
| p4096-m128 | 3313–3523 | 0.43 |
| p4096-m256 | 4402–4860 | 0.59 |

At the worst observed rate, half the buffer is used by **47 s** (`max_safe_tape_s`). A 90 s light capture at p128-m256-like load will plausibly overflow and be correctly invalidated, which wastes the trial. This is a bound, not an observed overflow.

**Minimal fix:** freeze tape + drain ≤ 30–40 s, and predeclare that any overflow stops physical fitting. Do not drop rows, and do not raise capacity without a memory review. Each core step row embeds a `state_before` scheduler snapshot (`runtime.py:9–16, 88–115`), so the END reply is large and is held in memory until ACK.

### 3.2 No offered-request reconciliation; the Chat ingress is outside the 38-file guard [EXEC]

`ExternalCapture.register_request` (`runtime.py:174–189`) accepts any request and defaults its role to `transfer`. It records `trace_id` only when `prefill_lab_trace_id` is present in `extra_args`. `capture_errors` checks only engine-vs-worker step sets and request terminals.

**Witness [EXEC]:** a stray request with no trace identity yields `capture_errors(...) == []`. A stray could be a health generation, a client retry or another client. The capture is reported valid.

Separately, the installed-source guard (`experiments/prefill-vllm-v030-source-map.json`, 38 files) includes `entrypoints/openai/completion/{protocol,serving}.py` but **no Chat serving/protocol file**. The Mayor's facade targets standard Chat traffic, so the request-ID derivation and `vllm_xargs` pass-through on the Chat path are unguarded [UNRES; needs an installed-source check, not available on CPU].

**Minimal fix:**
- The frozen plan carries a per-trial manifest hash of expected accepted-offer identities.
- After drain, the collector or bridge requires exact set equality between the manifest and the captured engine identities. The identity is `trace_id`, or else the existing Chat join `chatcmpl-<32hex>-<8hex>` (`native_observer/native_prediction.py:27`).
- `reconcile_requests` demonstrates this. It flags missing, duplicate and unexpected identities [EXEC].
- Gateway-rejected offers remain Gateway outcomes only.
- Add the Chat serving/protocol sources to the guard.

### 3.3 `metadata` runs during an active trial [EXEC]

The `metadata` op does not check `active` or `completed`. It calls `command(meta)`, which hashes 38 installed files and runs the worker meta probe on the engine-core loop during serving. The witness confirms one engine `meta` call is accepted while `nv2-r3-light` is active. It also clears `qualified`, which is harmless mid-trial but perturbs timing.

**Minimal fix:** reject `metadata` while `active is not None` or `completed is not None`.

### 3.4 ACK is not idempotent [EXEC, minor]

After a successful ACK whose response is lost, an identical retry raises `ValueError("ACK must identify…")`. The state is recoverable via `status` (`unacknowledged_trial is None`), but the collector cannot tell "already ACKed" from "wrong SHA" without a second round trip.

**Fix:** remember the last `(trial_id, capture_sha256)` and return `acknowledged` again for an exact repeat.

### 3.5 Epoch and source identity [PROOF, minor]

- **Epoch.** The epoch is `(pid, clock_domain, bridge_sha256)` with no per-process nonce. PID reuse in the same boot/namespace would go undetected. This is low risk, because a worker restart normally kills the engine core that holds `_pr33_batch_capture`. Fix: add `uuid4` at import to the worker clock reply.
- **Source hash.** `source_hash()` re-reads the file on every call, so it certifies the *on-disk* file rather than the loaded code. Fix: compute it once at import.

### 3.6 Literal disabled ≠ enabled-OFF ≠ light [PROOF from source]

**Enabled-OFF already changes the process:**
- `core_runtime` (`plugin.py:38–42`) sets `sys.setswitchinterval(5e-4)` for the life of the process on the first command.
- `register` installs the `add_request`, `execute_model`, padding, graph and prepare wrappers.
- `ExternalCapture.after_output` runs on every step even in OFF.

**Light adds, per step:**
- a full scheduler `snapshot`
- `scheduled_work`, `fifo_reference`, `features` and `step_key` digests (core and worker)
- two CUDA event records

Therefore three conditions are needed:
1. **Disabled:** a separate clean service start with the env unset.
2. **Enabled-OFF.**
3. **Light.**

The observer effect is estimated, never assumed zero.

### 3.7 Stream scope [PROOF from source]

`start.record(torch.cuda.current_stream())` sits at `execute_model` entry, before host prepare (`plugin.py:274–275`), and `end` at return (`plugin.py:290`). So:
- If the GPU is idle, the span includes host prepare/launch.
- Under async scheduling, `start` queues behind earlier same-stream work.
- Work on other streams (sampling, D2H) is excluded.

Correct label: "current-stream interval between entry and return markers". It is not kernel busy, not prefill compute and not the whole pipeline. I agree with the bridge's `measurement_scope` string.

`features()` failing closed on mixed prompt+decode per request, preemption and speculation is appropriate.

## 4. Smallest support-limited predictive candidate

### 4.1 Identifiable target

Fixed decision origins every Δ = 1 s, with action u_t ∈ {16, 32} Gateway cost budget (a cost, not concurrency).

**Targets:**
- **(a)** For each offer present at t (admitted-not-first, or decoding), the probability that it ends composite-good under scheduled-origin labels. Censored offers (incomplete at tape end) are kept as censored, not dropped.
- **(b)** Counts of first-content events and terminal releases in (t, t+h], for h ∈ {1, 2, 4} s.

Both targets are Gateway-observed, defined for every offered request and all-offered, so they are identifiable from randomized-cap data without engine decomposition.

### 4.2 Online feature cut

Everything known to the Gateway at t, age-limited:
- current and previous k caps, and in-flight cost
- admitted-without-first-content counts and prompt tokens, split short/long ("long work ahead")
- decoder count
- streamed output tokens per decoder, and age since the last streamed token (progress age)
- Gateway waiting count and recent arrivals

**Excluded from features (training labels only):** engine batch rows, step counts, forward spans and commit lags. These are drain-resolved, so using them online would leak. Internals (chunk membership, KV, EOS intent) stay latent.

**Model:** a piecewise-constant first-content hazard indexed by (long-tokens-ahead bin, decoder bin, u), and a decode-progress rate per decoder bin. Remaining output comes from the empirical natural-EOS survival conditioned on progress age, restricted to observed support. Forecasts are Monte Carlo over the EOS survival. Outside the observed support → **fallback to the reactive rule**, with the out-of-distribution event logged.

### 4.3 One-move continuation and the real multistep discriminator

- **Strong One-move:** choose u_t to optimise the same 4 s window under a *development-fitted* continuation. The continuation is the better of "hold u_t" and the reactive rule, selected on development blocks only.
- **N=3:** optimise (u_t, u_{t+1}, u_{t+2}), with the last move held, using the same model, window and terminal value. Multistep can add value **only** where the One-move backup is suboptimal.

**Discriminator** (on sealed forecasts, then live):

1. **Disagreement set and impact.** D = {t : first action of N=3 ≠ first action of One-move}. Impact is Σ_{t∈D} (predicted value gap × number of binding offers), not |D|/T.
2. **Randomized local continuation.** At eligible t ∈ D, randomize the first action between the two candidates and follow each policy afterwards. The realized local advantage is the direct estimand.
3. **Regime contrast.** Use a *predictable-lead* regime (long prompts already admitted-without-first-content at t) vs a *null-onset* regime (burst onset not visible at t). If the N=3 gain appears equally in the null-onset regime, it is a model artifact rather than foresight.
4. **Backup ablation.** Compare N=3 against One-move with its best backup. If they are equal, the value lies in the backup, not the horizon.

Horizon inflation (larger N or longer windows) is not tested before 1–4 show a nonzero effect.

## 5. Single highest-information bounded TRAIN pilot

**Goal:** qualify the instrument and observer effect, and generate randomized cap-path data for §4. This is not a controller contest.

**Tape and blocks:**
- Frozen 30 s mixed tape per seed block: p128 natural-EOS background, plus p4096 bursts with k ∈ {0, 1, 2, 3} at randomized, pre-frozen onsets.
- Standard Chat, all-offered, unique physical IDs and shared logical offer IDs.
- **6 independent seed blocks** (170001–170006). Seeds 170007–170012 are reserved for subsequent development validation and are not used for tuning here.

**Arms per block (order randomized within block):**

| Service epoch | Observer | Cap sequences |
|---|---|---|
| Disabled (clean start, env unset) | none | Fixed16, Fixed32 |
| Enabled | OFF | Fixed16, Fixed32, S↑ (16→32 at τ), S↓ (32→16 at τ) |
| Enabled | light | Fixed16, Fixed32, S↑, S↓, plus a Fixed32 same-cap replicate |

- τ is drawn from {8, 15, 22} s and frozen per block.
- Epoch order (disabled first vs enabled first) alternates across blocks. A restart effect is confounded with "enabled" within a block, so the disabled-vs-OFF contrast is reported as "enablement + restart", balanced by order across blocks.
- That is 11 arms × (30 s + drain ≤ 10 s) ≈ 8 min plus one restart per block, so roughly ≤ 1 h of GPU time in total. Running it requires Mayor authorization and is not part of this round.

**Records:**
- actual applied/fenced cap and timestamp
- all-offered terminal outcomes, including Gateway rejections
- manifest reconciliation (§3.2)
- scoped batch/commit joins in light only
- sealed §4 forecasts at each origin, if the online cut is qualified; otherwise labelled retrospective

**Estimands and uncertainty:**
- **Whole-policy ITT:** per-block paired contrasts on composite goodput and p50/p90 scheduled-origin TTFT/TPOT, reported raw for all 6 blocks.
- **Inference:** exact sign-flip. The floors with 6 blocks are 1/64 one-sided and 1/32 two-sided. These are exploratory screens, not equivalence claims.
- **Switch vs fixed:** compares S↑/S↓ post-τ window outcomes against the same window of the Fixed arms. Heterogeneity is reported, not tested for superiority.
- **Mediator analysis** (batch/commit exposure explaining short-behind-long) is descriptive and never relabelled as causal.

**Continue/revise conditions (predeclared):**
1. **Stop and revise the instrument** on any of: reconciliation failure, overflow or pending, epoch change, missing terminal, or metadata mid-trial (once fixed). Raw evidence is preserved.
2. **Observer effect.** If |light − OFF| or |OFF − disabled| exceeds a tolerance fixed *before* the pilot (from existing TRAIN same-cap variability) in ≥ 2 blocks with a consistent sign, light coefficients are not transferred as normal-service values. Whole-policy evaluation then uses only the disabled/OFF arms.
3. **Continue to model fitting** only if §4 forecasts beat persistence and a scalar-backlog baseline on fixed-origin release counts in ≥ 5/6 blocks. Otherwise revise the state before any controller run.
4. **Continue to a One-move vs N=3 live stage** only if the switching arms show a state-dependent cap response that §4 predicts out of block. Otherwise stay with Fixed/reactive.

## 6. Minimal batch-telemetry contract (unchanged in spirit, narrowed in scope)

**Per offer:**
- logical and physical ID
- Gateway admit/reject time, cap and in-flight cost at admission
- first-content and terminal times, finish reason and token counts

**Per step (light only):**
- step key and scheduled request set
- scheduled tokens per request, with prompt/decode class
- host schedule and commit timestamps
- current-stream span (scoped as in §3.7)

**Per trial:**
- manifest hash, source/epoch identity
- applied-cap log, overflow/pending counters

**Whole-policy reduced model:** yes, it can proceed now on Gateway-side data (§4) without perfect GPU decomposition. Unsupported internals stay latent.

## 7. Remaining disagreements

- **Astra §8, 12 s tape:** too short to observe m256 decode recovery and release tails. I keep 30 s, which is still under the §3.1 buffer bound.
- **Astra and Sol, fixed caps only:** without randomized switching sequences, the pilot cannot inform state-dependent cap response, which is exactly what multistep needs. I add S↑/S↓.
- **Sol, two blocks:** sufficient only for gross failures. Six blocks give a non-trivial sign-flip floor.
- **Gemini:** withdrawn overhead figures are not reinstated by any fixture. The 1 Hz scalar contract is insufficient for joins.
- **[UNRES]** The Chat ID path (§3.2) and whether `vllm_xargs` reaches `extra_args` on Chat in the installed vLLM: needs an installed-source check.
- **[UNRES]** Actual buffer usage including prefill steps and drain: only observable in the pilot.

## 8. Reproduction

```
PYTHONDONTWRITEBYTECODE=1 python3 experiments/mpc-research/native-v2/round3/opus_revision.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/mpc-research/native-v2/round3 -p 'test_opus_*.py'
```

Run from the repository root, with stdlib Python 3.14.4 and no network or GPU. Bridge witnesses load `git show 1cbe251:experiments/mpc-research/native-v2/batch_capture_bridge.py` in memory and refuse any other sha256. Result: 8 tests OK.
