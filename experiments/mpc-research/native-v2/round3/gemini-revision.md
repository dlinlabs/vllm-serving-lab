# Round 3 Technical Revision: Concrete Instrument Review, Empirical Calibration, and Bounded Pilot Design

**Author**: Gemini 3.8 Flash (pr32/polecats/garnet)
**Role**: Polecat Worker (rig: pr32)
**Branch**: `research/native-v2-gemini-local`
**Base Commit**: `9b02f6ebdbd6681763b562bc3c37aa564b8b330a`
**Owned Deliverables**:
- `experiments/mpc-research/native-v2/round3/gemini-revision.md`
- `experiments/mpc-research/native-v2/round3/gemini_revision.py`
- `experiments/mpc-research/native-v2/round3/test_gemini_revision.py`

---

## 1. Explicit Retractions & Mandatory Mayor Rulings

In accordance with Mayor instructions and peer reviews:

1. **Explicit Retraction of Unmeasured R2 Claims**:
   - The numerical values presented in Round 2 (`4.5 μs` bridge overhead, `450 μs` CUPTI overhead, `1200 μs` blocking sync overhead, and "zero serving jitter") were **assumed synthetic constants in a CPU fixture, NOT native GPU measurements**. They are hereby explicitly withdrawn as empirical claims.
   - No native measurement of Mayor's bridge (`batch_capture_bridge.py`, commit `1cbe251`) was performed in Round 2.
   - Falsification of the serial HTTP proxy model ($2004$ bound vs $2449$ actual completions) proves that exclusive serial execution is physically impossible; it does **not** prove that any specific roofline parameterization holds without direct unprofiled calibration.
2. **Certified Lower Envelopes and Step Parity**:
   - Fitted mean busy-time regressions are not certified native lower envelopes.
   - Decoder delta-output is an effective commit-rate proxy unless participation, async boundaries, and actual step parity are independently qualified.
3. **No Theorem of Multi-Step Superiority**:
   - Physical memory does not mathematically imply multi-step gain.
   - Astra's finite-tail intervals are an accounting diagnostic, not an online admission policy.
   - Correct cap reductions preserve old cost; lowering a budget cap does not evict admitted work.

---

## 2. Systematic Adjudication of Peer Critiques

We evaluate all criticisms raised by Opus (`opal`), Astra (`topaz`), and Sol (`ruby`) against Gemini Round 1 and Round 2:

```
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| Peer Critique                     | Status             | Technical Resolution & Mathematical Correction                           |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| G1: Reservation Invariant False   | ACCEPTED           | Repaired in gemini_revision.py:check_reservation_invariant().            |
|     (Opus §1.1, Astra §3.G1,      | (Proven by peers)  | Correct formula: R(t) <= max(R(0), max_{s<=t} u_applied(s)) + c_max.    |
|      Sol §4.G2)                   |                    | Lowering u from 32 to 16 leaves R=32 valid until requests complete.      |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| G2: Horizon "iff" Theorem False   | ACCEPTED           | Retracted. T_settle > T_slot is a necessary condition, NOT sufficient.   |
|     (Opus §1.2, Astra §3.G2,      | (Proven by peers)  | Under an optimal Bellman backup (cost-to-go), N=1 is equivalent to N=N.  |
|      Sol §4.G1)                   |                    | Extra moves add value only to the extent that backup policies fail.      |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| G3: θP Target Misuse & FLOPS      | ACCEPTED           | Dropped θP as prefill duration. θP (8.73ms/k-tok) was fitted to decode   |
|     (Opus §1.6, Astra §3.G3,      | (Proven by Opus)   | slowdown from coexisting prefill. Applying it to prefill duration        |
|      Sol §4.G3)                   |                    | implies >866 TFLOPS, violating the RTX 4090 BF16 ceiling (~165 TFLOPS).  |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| G4: TTFT Decomposition Gaps       | ACCEPTED           | Acknowledge omission of Gateway-to-engine transit (t_vllm_enq - t_admit) |
|     (Astra §3.G4, Sol §4.G4)      | (Proven by peers)  | and post-forward sampling/SSE delivery. Per-batch GPU spans cannot be    |
|                                   |                    | summed directly into per-request TTFT without modeling batch sharing.    |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| G5: Statistical Protocol Bugs     | ACCEPTED           | Repaired in gemini_revision.py:RevisedStatisticalProtocol:               |
|     (Opus §1.3, Astra §3.G6,      | (Executed defects) | 1. Overrun rate divided by control decisions (NOT offered requests).     |
|      Sol §4.G5)                   |                    | 2. Two-sided exact sign-flip permutation test implemented.               |
|                                   |                    | 3. K >= 6 required on TRAIN seed blocks (floor p = 2/64 = 0.03125).      |
|                                   |                    | 4. Explicit relative rejection guard (<= 5% increase) enforced.          |
|                                   |                    | 5. is_composite_good requires finish_reason in {"stop", "length"}.       |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
| G6: KV Residency & Margin Scale   | ACCEPTED           | Corrected: TRAIN decode residence for p4096 is 2.8s - 3.3s, not 10-20s.  |
|     (Opus §1.7)                   | (Factual error)    | Fixed margin of 0.5 req/s is unachievable under pure p4096 (goodput ~1). |
|                                   |                    | Margin must be relative to cohort capacity (e.g. 10% relative delta).    |
+-----------------------------------+--------------------+--------------------------------------------------------------------------+
```

---

## 3. Concrete Instrument Review: Mayor's `batch_capture_bridge.py`

Mayor supplied commit `1cbe251` containing `native-v2/batch_capture_bridge.py` and `test_batch_capture_bridge.py`. We conducted a line-by-line static and behavioral audit of the implementation.

### 3.1 Architectural Strengths
1. **Reusing Proven PR32 Subsystems**:
   The bridge reuses `ExternalCapture`, `scheduled_work`, `step_key`, and `WorkerRecorder` from the reviewed PR32 tree, enforcing the installed 38-source guard and eliminating the risk of unverified third-party libraries.
2. **Strict Default-Off Isolation (`register()`)**:
   Execution is gated by `PR33_NATIVE_BATCH_CAPTURE_ENABLED == "1"` and `PREFILL_LAB_ENABLED == "1"`. When disabled, it performs zero monkey-patching, imports no capture modules, and does not alter vLLM EngineCore or Worker classes.
3. **Multi-Domain Epoch & Clock Authentication (`worker_clock()`, `metadata`)**:
   The collective RPC `pr33_native_capture_clock` samples PID, clock domain, and bridge source SHA across all workers. If worker PID or clock domain shifts between `begin` and `end`, `self.invalid` is immediately latched, permanently invalidating the capture.
4. **Current-Stream CUDA Event Scoping**:
   The instrumentation wraps model preparation and forward execution on the *current model stream* using CUDA event pairs (`cudaEventRecord`), deliberately avoiding CUPTI tracing and avoiding whole-pipeline blocking.
5. **Memory Replay Until Exact-SHA ACK (`ack`)**:
   Upon completion of `external_end`, the bridge formats the result, computes its SHA-256 hash, and retains it in memory. If network drops occur, subsequent `end` calls replay the identical record until the collector issues an `ack` matching `capture_sha256`.

### 3.2 Identified Edge Cases and Operational Recommendations
1. **The `gpu_events_pending` Drain Loop**:
   In `BatchCapture.invoke("end")`:
   ```python
   if isinstance(result, dict) and result.get("status") == "gpu_events_pending":
       return {**common, "trial_id": self.active, "status": "gpu_events_pending"}
   ```
   If GPU forward events have not completed when `end` is called, the trial remains active (`self.active` is not cleared), and no completed record is cached.
   **Operational Contract**: The offline collector must implement a polling loop with bounded backoff (e.g. polling every $50$ms up to $5.0$s after episode traffic drain) before concluding an instrumentation timeout.
2. **Process Crash Durability Scope**:
   The bridge states: `"retention_scope": "In-memory replay until ACK; not process-crash durability."`
   If the vLLM engine process crashes before ACK, in-memory state is lost. The collector must persist the received capture payload to disk with `os.fsync()` before transmitting the ACK command.
3. **Bounded Heap Footprint (8192 Buffers)**:
   The underlying worker recorder ring buffer is sized at 8192 steps. For a 90-second episode with $\sim 10$ms iterations, expected batch count is $\sim 9000$ iterations under extreme continuous decoding.
   **Recommendation**: Keep pilot trial duration at $\le 60$ seconds to guarantee zero ring buffer overflow (`overflow == 0`), preserving complete capture integrity.

---

## 4. Smallest Support-Limited Predictive Candidate

Rather than constructing an uncalibrated microsecond GPU simulator, we propose the **Macroscopic Backlog & Headroom Predictor**:

### 4.1 Identifiable Target and Online Feature Cut
- **Prediction Target**: Binary classification of whether an admitted request will satisfy TTFT:
  $$Y_i = \mathbf{1}\{T_{\text{TTFT}, i} \le 0.500\text{s}\}$$
  and whether admitting request $i$ will push KV cache occupancy past safe headroom:
  $$\kappa_{\text{KV}}(t + \Delta t) \le 0.85$$
- **Minimal Online Feature Cut** ($z_k$ available at decision time):
  1. $p_i$: Prompt tokens of incoming request $i$.
  2. $N_{\text{long\_ahead}}$: Count of active requests with $p \ge 2048$ currently awaiting or executing prefill in vLLM.
  3. $P_{\text{wait}}$: Total prompt tokens in vLLM `waiting` queue.
  4. $\kappa_{\text{KV}}$: Current physical KV cache block occupancy ($[0, 1]$).
  5. $R(t)$: Current Gateway reserved cost.

### 4.2 Macroscopic Transition Dynamics
Building upon Opus's empirical finding (F4) and Sol's release obstruction:
- Short prompts ($p \le 512$):
  $$T_{\text{TTFT}} \approx 0.066\text{s} \ll 0.500\text{s} \quad (\text{independent of } N_{\text{long\_ahead}})$$
- Long prompts ($p \ge 2048$):
  $$T_{\text{TTFT}} \approx 0.310\text{s} + 0.185\text{s} \times N_{\text{long\_ahead}}$$
  - For $N_{\text{long\_ahead}} = 0$: $T_{\text{TTFT}} \approx 0.310$s $\le 0.500$s $\implies$ Safe to admit under Budget 32.
  - For $N_{\text{long\_ahead}} = 1$: $T_{\text{TTFT}} \approx 0.495$s $\approx 0.500$s $\implies$ Boundary regime.
  - For $N_{\text{long\_ahead}} \ge 2$: $T_{\text{TTFT}} \ge 0.680$s $> 0.500$s $\implies$ **Definite TTFT Violation!**
  **Action**: Throttle budget to $16$ immediately to block incoming long prompts.

### 4.3 Same-Window One-Move Continuation vs Real Multi-Step Discriminator
- **Evaluation Window**: Fixed at $D = 5.0$s across all policies.
- **One-Move Policy ($N=1$)**: Selects $u_0 \in \{16, 32\}$ to maximize good completions over $[0, 1.0\text{s}]$ assuming a standard Fixed16 continuation over $[1.0\text{s}, 5.0\text{s}]$.
- **Multi-Step Policy ($N=3$)**: Optimizes $\{u_0, u_1, u_2\} \in \{16, 32\}^3$.
- **Real Multi-Step Discriminator**:
  Multi-step adds value over One-move **only when a predicted arrival burst at $t=2$s will encounter KV exhaustion or prefill backlog unless budget is preemptively cut at $t=1$s**. If the workload lacks advance burst announcements or long carry-over memory, $N=3$ must collapse to $N=1$.

---

## 5. Highest-Information Bounded TRAIN Native Pilot

We specify the exact design for the confirmatory native GPU pilot:

### 5.1 Protocol Structure & Three-Way Instrument Validation
- **Hardware & Model**: RTX 4090 (24GB), Qwen3-4B-Instruct-2507 BF16, vLLM 0.30.0, normal graphs/async, prefix cache off.
- **Instrumentation Arm Contrast (Observer Validation)**:
  1. `Arm-Disabled`: Bridge code disabled (`PR33_NATIVE_BATCH_CAPTURE_ENABLED=0`). Baseline serving.
  2. `Arm-Off`: Bridge loaded (`PR33_NATIVE_BATCH_CAPTURE_ENABLED=1`), but trial configured with `observer="off"`. Measures pure hook presence overhead.
  3. `Arm-Light`: Bridge active (`observer="light"`), recording CUDA forward spans and host schedule timestamps.
  *Gate*: If `Arm-Light` composite goodput degrades by $> 2\%$ relative to `Arm-Disabled`, stop pilot immediately.

### 5.2 Workload & Seed Allocation
- **Seed Blocks**: 6 independent blocks selected strictly from the authorized TRAIN range:
  `Seeds: 170001, 170002, 170003, 170004, 170005, 170006`.
  (Held-out seeds `160021`–`160024` remain strictly sealed and unopened).
- **Traffic Composition**: Mixed short/long workload (70% prompt 128 / max 128; 30% prompt 4096 / max 128) under Poisson arrivals at $12$ req/s.
- **Trial Duration**: 60 seconds offered traffic + 30 seconds drain per episode.

### 5.3 Controller Matrix (Paired-Block Evaluation)
Each seed block runs 5 randomized arms against the identical offered tape:
1. `Fixed16`: Static budget $u=16$.
2. `Fixed32`: Static budget $u=32$.
3. `ReactiveHysteresis`: Throttle to 16 if $N_{\text{long\_ahead}} \ge 2$ or $\kappa_{\text{KV}} > 0.80$, else 32.
4. `OneMove-N1`: Macroscopic predictor with 1-step optimization and fixed continuation.
5. `PredictiveMPC-N3`: Macroscopic predictor with 3-step receding horizon.

### 5.4 Adjudication Gates & Stopping Criteria
1. **Decision Overrun Guard**: $\frac{\sum \text{overruns}}{\sum \text{decisions}} \le 0.01$ (solver deadline 100ms).
2. **Rejection Guard**: $\frac{\text{Rejections}_{\text{MPC}} - \text{Rejections}_{\text{Fixed16}}}{\text{Rejections}_{\text{Fixed16}}} \le 0.05$.
3. **Statistical Superiority Gate**: Two-sided sign-flip permutation test $p < 0.05$ across the 6 blocks, and mean goodput gain $\ge 0.50$ req/s over the best baseline (`ReactiveHysteresis` or `OneMove-N1`).
4. **Failure Condition**: If MPC achieves no statistically significant improvement over `OneMove-N1`, we explicitly record that **multi-step lookahead is non-viable for native vLLM admission control**, and adopt the reactive/one-move policy.

---

## 6. Verification of Revised Code Deliverables

In `gemini_revision.py` and `test_gemini_revision.py`:
- Implemented and unit-tested `check_reservation_invariant()`: successfully validates legal cap cuts where $R > u_{\text{new}}$ under history-max accounting.
- Implemented and unit-tested `RevisedStatisticalProtocol`: enforces $K \ge 6$ blocks, two-sided exact permutation testing, decision-normalized overruns, and relative rejection guards.
- Implemented and unit-tested `validate_request_outcome()`: guarantees truncated or error streams receive 0 score.
- Implemented and unit-tested `MacroscopicQueuePredictor`: replicates the empirical TTFT explosion when $N_{\text{long\_ahead}} \ge 2$.

All 10 unit tests executed and passed cleanly:
```bash
/usr/bin/python3 -m unittest experiments/mpc-research/native-v2/round3/test_gemini_revision.py
..........
----------------------------------------------------------------------
Ran 10 tests in 0.001s

OK
```
Combined regression across all rounds (18 Round 1 + 5 Round 2 + 10 Round 3 = 33 tests) passed completely.
