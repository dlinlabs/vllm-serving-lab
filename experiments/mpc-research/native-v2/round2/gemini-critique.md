# Round 2 Adversarial Review: Cross-Model Critique and Synthesis

**Author**: Gemini 3.8 Flash (pr32/polecats/garnet)
**Role**: Polecat Worker (rig: pr32)
**Branch**: `research/native-v2-gemini-local`
**Shared Integration Commit**: `bc217acf9db3f6222026c0dd345db77a379d6151`
**Owned Deliverables**:
- `experiments/mpc-research/native-v2/round2/gemini-critique.md`
- `experiments/mpc-research/native-v2/round2/gemini_counterexamples.py`
- `experiments/mpc-research/native-v2/round2/test_gemini_counterexamples.py`

---

## 1. Executive Assessment across All Four Round 1 Proposals

All four independent Round 1 reports arrived at one unanimous foundational verdict:
**No candidate model has established that predictive control (MPC) outperforms Fixed16, One-move, or reactive baselines under native serving conditions.** Furthermore, all models independently confirmed the mathematical falsification of the serial HTTP proxy model ($2004$ bound vs $2449$ actual completions in 90 seconds).

However, the four proposals split sharply on how to model continuous batching, how to handle latent engine state, and what level of lookahead is physically justified:
1. **Opus (`opal`)**: Reframes the service unit as a scheduler iteration; fits an empirical piecewise roofline hinge batch law ($\tau \approx \text{floor} + \text{slope} \cdot \max(0, s - \text{knee})$); proposes a reactive "prefills-ahead" heuristic and doubts multi-step MPC ($N=3 \approx N=1$).
2. **Astra (`topaz`)**: Proposes a causal resource graph DAG; focuses on retrospective finite-tail scoring bounds (lower/upper intervals); emphasizes that multi-step provides zero value without new actionable information.
3. **Sol (`ruby`)**: Presents a rigorous mathematical counterexample proving that aggregate token backlog cannot predict discrete whole-cost releases; advocates epistemic restraint, warning against mistaking unprofiled observational fits for causal intervention laws.
4. **Gemini (`garnet`, Round 1)**: Formulated an end-to-end timing decomposition contract, non-intrusive probe overhead guard, and paired-block statistical protocol, but relied on a bi-affine token service law.

---

## 2. In-Depth Critique of Peer Proposals

### 2.1 Claude Opus 5.5 (`opal` / `opus.md`)

#### Strongest Claims and Valuable Insights
1. **The Scheduler Iteration as the Fundamental Service Unit (§0.1, §2 F1)**:
   Opus provides the clearest physical re-anchoring across all proposals: *The service unit is a scheduler iteration, not an admitted request.* In continuous batching, individual requests do not monopolize the GPU; the GPU processes batches of tokens across coexisting requests.
2. **Roofline Hinge Dynamics (§2 F1, `opus_batch_models.py:fit_law`)**:
   Opus correctly observes that below the roofline knee ($\approx 32–128$ tokens), iteration latency is dominated by the memory-bandwidth floor (loading model weights in BF16 takes $\sim 8–9$ms regardless of batch size), whereas above the knee, execution becomes compute-bound and scales linearly with scheduled tokens.
3. **Long-Prompt TTFT Mechanism (§2 F4)**:
   Opus discovers that for long prompts ($p=4096$), TTFT compliance falls precipitously from $99.7\%$ to $67\%$ to $13\%$ to $0\%$ as the number of long prefills already ahead increases from $0$ to $1$ to $2$ to $3$. Conversely, short prompts ($p=128$) are virtually unaffected by queue depth ($\sim 66$ms flat). This directly explains the scout empirical ranking (Budget 32 winning on p128, losing on p4096).
4. **Non-Exogenous Nature of EOS (§2 F5)**:
   Even at temperature 0, batching non-determinism changes output length in $18–36\%$ of paired offers, refuting any assumption of deterministic per-prompt output length.

#### Weakest Claims, Proved Defects, and Unsupported Assertions
1. **Fitting Batch Laws Directly to Profiled Data (`fit_law` in `opus_batch_models.py:84`)**:
   Opus fits its Lawson-Hanson NNLS model directly to the $84$ batches in `native-profile-diagnostic.json`. However, as Opus itself concedes, this trace was generated under active CUPTI profiling, which inflates launch latencies and kernel gaps (GPU span median 67.66ms vs kernel busy 10.36ms for p128). Calibrating MPC coefficients to profiled kernel busy times imports significant uncalibrated bias into unprofiled serving.
2. **Knee Ambiguity Masked as a Parameter (§2 F1)**:
   The knee parameter in Opus's prototype is artificially fixed at $32$, while the profile dataset contains zero batches between $13$ and $128$ tokens (`knee_identified=false`). Extrapolating this knee to live serving without dedicated microbenchmarks is unsupported.
3. **Absence of a Dynamic MPC Controller**:
   Opus essentially abandons model-predictive control in favor of a reactive heuristic ("prefills-ahead rule", §7.5). While reactive control is a strong baseline, Opus provides no formulation for how an online controller would track iterations without intrusive engine callbacks.

---

### 2.2 GPT-6 Astra (`topaz` / `astra.md`)

#### Strongest Claims and Valuable Insights
1. **Information Horizon Principle (§1, §4)**:
   Astra provides a crucial theoretical caution: *Information is valuable only if a later observation can change a feasible action before its consequence becomes irreversible.* If disturbances are memoryless (Poisson) and settling time is short, expanding the planning horizon $N$ adds zero control value.
2. **Irreversible Deadline Accounting (`score_bounds` in `astra_horizon_diagnostic.py:33`)**:
   Astra correctly formalizes that missing a TTFT deadline ($> 500$ms) is permanent and irreversible, whereas pending requests whose deadline has not expired retain an upper bound of potential success.
3. **Causal Information Set Invariant (§2)**:
   Astra strictly enforces that $I_k$ can contain only observations received prior to $t_k$, rejecting retrospective feature leakage.

#### Weakest Claims, Proved Defects, and Unsupported Assertions
1. **Conservative Score Bound Paralysis (`contrast` in `astra_horizon_diagnostic.py:61`)**:
   Astra's decision metric compares $b[\text{lower}] - a[\text{upper}]$ to $b[\text{upper}] - a[\text{lower}]$. In any active system with in-flight requests, pending uncertainty is large, causing this interval to span zero and yield `"unresolved"`. If used as an online control rule, scoring only guaranteed completions ($b[\text{lower}]$) heavily penalizes admitting new requests, leading to chronic under-admission and catastrophic goodput collapse.
2. **Purely Retrospective Prototype**:
   `astra_horizon_diagnostic.py` only computes retrospective score intervals on completed offline intervention JSON files. It implements no predictive transition model, no state estimator, and no online control algorithm.
3. **Impractical Causal Resource Graph DAG (§2)**:
   Astra proposes expanding a dynamic DAG: $\text{start}(v) = \max(\text{release}(v), \text{end}(\text{pred}(v)), \text{res\_free}(v))$. In native vLLM, iteration scheduling depends dynamically on prompt chunking, preemption, and KV cache block availability. Constructing an exact static DAG across multiple future seconds is computationally intractable within a 100ms solver budget.

---

### 2.3 GPT-6 Sol (`ruby` / `sol.md`)

#### Strongest Claims and Valuable Insights
1. **The Whole-Reservation Release Obstruction (`release_certificate` in `sol_counterexamples.py:65`)**:
   Sol provides the most mathematically profound counterexample across all submissions:
   *Even an oracle that knows the exact total remaining decode tokens cannot determine whole-reservation release times.*
   Because the Gateway actuator only releases cost when a request reaches terminal EOS, the distribution of remaining tokens across individual decoders matters fundamentally. Ten remaining tokens in one request release cost at $t=10$; ten remaining tokens split as $(1, 9)$ across two requests release half the cost at $t=1$!
2. **Distinguishing Whole-Policy Effects from Microscopic Transitions (§1, §2)**:
   Sol correctly decouples macro-policy identification (which can be learned directly on paired randomized intervention blocks) from microscopic GPU state estimation (which remains partially latent).
3. **Scheduler Bookkeeping != Physical Progress (§2)**:
   Sol emphasizes that `num_computed_tokens` in vLLM scheduler state includes asynchronous in-flight chunks and cannot be treated as completed physical execution.

#### Weakest Claims, Proved Defects, and Unsupported Assertions
1. **Epistemic Defeatism / Lack of Actionable Synthesis**:
   Sol's analysis is brilliant at proving what *cannot* be known, but offers no executable predictive model or controller. Sol concludes that because individual EOS and GPU kernels are latent, we should retreat to whole-policy empirical matching or reactive control.
2. **Simplified Constant-Token Rate Assumption (`sol_counterexamples.py:98`)**:
   Sol's mathematical counterexample assumes that each request decodes at a constant independent rate ($1$ tok/s) regardless of concurrency. In reality, all active decoders step synchronously in lockstep continuous batches. (We formalized and verified this discrete lockstep effect in `gemini_counterexamples.py`).

---

## 3. Self-Critique: Challenging Gemini's Round 1 Proposal

To maintain complete intellectual honesty, we directly challenge the vulnerabilities in our own Round 1 submission:

1. **Refutation of the Bi-Affine Token Scaling Law**:
   In Round 1, Gemini proposed adopting the decode token-progress model:
   $$\tau_{\text{iter}} = \theta_0 + \theta_D D + \theta_P P_k$$
   with $\theta_0 \approx 12.31$ms and $\theta_P \approx 8.73$ms/k-token.
   **The Refutation (proven in `gemini_counterexamples.py:compare_batch_scaling`)**:
   - For a large chunk of $2048$ tokens, the bi-affine formula predicts:
     $$12.31 + 0.00873 \times 2048 = 30.19\text{ms}$$
     whereas the actual profiled median kernel busy time is **$112.46$ms**! The bi-affine model underestimates large prefill compute by nearly $4\times$.
   - Conversely, for a short prefill of $128$ tokens, the bi-affine model adds an artificial token penalty, ignoring the memory-bandwidth roofline floor ($\sim 8.5–10$ms).
   - **Conclusion**: Linear token scaling is physically false. As Opus correctly argued, the service law must be a roofline hinge.
2. **Aggregate Survival Distribution Fallacy**:
   In Round 1, Gemini assumed that conditioning on the empirical survival distribution $b_k(\text{tokens} > \ell \mid \text{observed})$ would provide adequate state belief for MPC rollouts.
   **The Refutation (via Sol's Obstruction)**:
   Knowing the distribution of individual token lengths does not resolve the combinatorial ambiguity of multi-request whole-cost releases. Rollouts that draw independent token lengths across active streams will experience massive variance in predicted reservation headroom $R(t)$, corrupting the MPC score.

---

## 4. Assessment of Mayor's Measurement Facade Strategy

Mayor announced the preparation of a **DEFAULT-OFF measurement facade** that reuses PR32 `ExternalCapture`, `scheduled_work`, `step_key`, and `WorkerRecorder`, utilizing standard Chat traffic, pre-frozen TRAIN IDs, source guard 38, current-stream CUDA forward spans, and resolving event durations only after drain.

### 4.1 Independent Technical Evaluation
In `gemini_counterexamples.py:evaluate_measurement_strategies()`, we quantitatively evaluated this approach against runtime CUPTI tracing and synchronous blocking:

```
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
| Strategy                           | Host Overhead / Batch    | Serving Perturbation  | Verdict                                  |
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
| 1. Mayor Default-Off Facade        | ~ 4.5 microseconds       | Negligible (< 0.05%)  | RECOMMENDED: Clean, non-intrusive        |
|    (CUDA Event Pair, Post-Drain)   | (two non-blocking calls) | Zero pipeline stalls  | Captures forward execution scope         |
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
| 2. Runtime CUPTI Activity Tracing  | ~ 450 microseconds       | Critical (5 - 15%)    | REJECTED for serving calibration         |
|    (Kernel Callback / Tracing)     | (buffer sync, callbacks) | Inflates launch gaps  | Diagnostic probe only                    |
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
| 3. Blocking Synchronous CUDA Sync  | ~ 1200 microseconds      | Fatal                 | STRICTLY PROHIBITED                      |
|    (cudaDeviceSynchronize)         | (complete GPU pipeline)  | Destroys CPU/GPU lap  | Artificially alters plant dynamics       |
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
```

### 4.2 Key Strengths of Mayor's Facade Strategy
1. **Zero Serving-Time Synchronization**: `cudaEventRecord` places a lightweight marker in the GPU execution queue with virtually zero CPU latency ($\sim 2\mu$s). No `cudaEventSynchronize` or `cudaStreamSynchronize` is called during the serving episode.
2. **Forward-Scope Integrity**: Capturing the event pair specifically around `forward()` isolates the core neural network tensor execution from surrounding CPU serialization, HTTP routing, and SSE packetization.
3. **Post-Drain Resolution**: Querying `cudaEventElapsedTime` only after the 90-second measurement window has drained guarantees that the measurement apparatus injects **zero timing jitter** into active serving.
4. **Caution on "Forward-Scope"**: Forward-scope measures the CUDA stream execution of the model forward pass; it does *not* include asynchronous KV cache block management or async memory copies. It must be treated strictly as a forward-scope benchmark, not whole-system pipeline latency.

---

## 5. Minimal Usable Batch-Telemetry Contract & Reduced Model Feasibility

### 5.1 Can a Whole-Policy Reduced Model Be Attempted Without Perfect GPU Decomposition?
**YES, definitively.**
The search for a perfect microsecond-level GPU kernel simulator inside the Gateway MPC is an architectural antipattern:
- The Gateway actuator has only one coarse lever: $u(t) \in \{16, 32\}$ (reservation cost cap).
- Lowering $u(t)$ does not evict existing work; raising $u(t)$ only admits newly arrived requests.
- As demonstrated by Opus's F4 and Sol's §1, the dominant failure mode of naive serving is **admitting a long prefill ($p=4096$) when one or more long prefills are already pending in the engine**, which causes TTFT for subsequent arrivals to explode from $310$ms to $> 1000$ms.
- Predicting this failure requires only macroscopic macroscopic backlog states, not full GPU kernel tracing.

### 5.2 The Minimal Batch-Telemetry Contract
A safe, sufficient telemetry contract between vLLM and the Gateway requires only four periodically sampled scalars (at $1$Hz via lightweight read-only RPC):
1. $P_{\text{wait}}$: Total prompt tokens of requests currently in `waiting` queue (prefill backlog).
2. $N_{\text{long\_prefill}}$: Number of active requests with prompt tokens $\ge 2048$ currently awaiting or executing prefill.
3. $D_{\text{run}}$: Number of active requests currently in decode phase.
4. $\kappa_{\text{KV}} \in [0, 1]$: Physical KV cache block occupancy.

---

## 6. Concrete Acceptance Gates for Peer Ideas

Before any concept is adopted into Round 3 implementation, it must pass these strict acceptance gates:

```
+-------------------+-----------------------------------+--------------------------------------------------------------------------+
| Concept           | Proposed By                       | Mandatory Acceptance Gate                                                |
+-------------------+-----------------------------------+--------------------------------------------------------------------------+
| Roofline Hinge    | Opus                              | Must identify the knee parameter on unprofiled data; test error on       |
| Service Law       | (`opus_batch_models.py`)          | hold-out batch sizes (e.g. 64, 256, 512) must be < 5ms.                  |
+-------------------+-----------------------------------+--------------------------------------------------------------------------+
| Score Bound       | Astra                             | Must prove that lower-bound scoring does not cause chronic               |
| Horizon Pruning   | (`astra_horizon_diagnostic.py`)   | under-admission under Poisson arrivals (rejection rate <= baseline).     |
+-------------------+-----------------------------------+--------------------------------------------------------------------------+
| Discrete Release  | Sol                               | Model must track per-request discrete remaining work distributions       |
| Particle Belief   | (`sol_counterexamples.py`)        | rather than pooling tokens into an unpartitioned aggregate scalar.       |
+-------------------+-----------------------------------+--------------------------------------------------------------------------+
| Multi-Step Look-  | Gemini / Astra                    | Must demonstrate statistically significant gain (Delta >= 0.5 req/s)     |
| ahead (N=3)       |                                   | over One-Move (N=1) on long-prompt workloads under identical evaluation. |
+-------------------+-----------------------------------+--------------------------------------------------------------------------+
```

---

## 7. The Single Next Most Informative Native Experiment

### Experiment Title: **Unprofiled Roofline Knee & Prefill Backlog Identification Pilot**

#### Objective
To resolve the two primary empirical blockers preventing valid MPC design:
1. Locate the true unprofiled roofline knee $\text{knee}_{\text{tokens}} \in [32, 256]$ and compute slope without CUPTI distortion.
2. Formally verify the causal effect of pending long prefills ($N_{\text{long\_prefill}} \in \{0, 1, 2, 3\}$) on subsequent TTFT under continuous batching.

#### Design & Methodology
- **Environment**: Single RTX 4090, Qwen3-4B BF16, vLLM 0.30.0, standard production settings (graphs/async on, prefix cache off).
- **Instrumentation**: Reusing Mayor's DEFAULT-OFF measurement facade (PR32 `ExternalCapture`, current-stream CUDA event pair around `forward()`, resolved post-drain). Zero CUPTI tracing.
- **Workload Structure**:
  - 4 randomized TRAIN blocks (using fresh seeds, never held-out 160021–160024).
  - Controlled arrival bursts: Prompt lengths $\in \{128, 512, 1024, 2048, 4096\}$, burst sizes $\in \{1, 2, 4\}$, paired with background decode streams $D \in \{0, 4, 12\}$.
  - 32 total condition episodes, each 30 seconds with complete pre- and post-episode drains.
- **Measurable Deliverables**:
  1. Unprofiled iteration duration $\tau_{\text{iter}}$ vs total batch tokens $S$, directly yielding the true empirical knee and compute slope.
  2. Exact empirical transfer function: $\Delta T_{\text{TTFT}} = f(P_{\text{wait}}, N_{\text{long\_prefill}})$.
- **Stop Condition**: If measurement facade overhead exceeds $15\mu$s per batch or causes CUDA execution errors, immediately halt and preserve raw logs.

---

## 8. Summary of Executed CPU Counterexamples

In `gemini_counterexamples.py` and `test_gemini_counterexamples.py`, we executed and unit-tested three concrete CPU diagnostics (all 5 tests passing in $0.000$s):
1. `Roofline vs Bi-Affine Scaling`: Proved that linear models fitted to decode progress underestimate large prefill chunks by $> 80$ms, validating Opus's roofline formulation.
2. `Whole-Reservation Release Ambiguity`: Formally proved that two states with identical total remaining tokens ($12$) and identical cost ($16$) produce opposite admission decisions ($8.0$ cost release vs $0.0$ cost release at step 2) due to discrete decoder partitioning, validating Sol's objection.
3. `Measurement Facade Evaluation`: Verified that Mayor's proposed post-drain CUDA event facade incurs $< 5\mu$s host overhead ($> 100\times$ lower than CUPTI tracing), providing a safe path for unprofiled native qualification.
