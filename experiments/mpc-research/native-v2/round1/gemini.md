# Round 1 Independent Research Report: Identifiable Batch-Aware Dynamical Modeling and Predictive Control for Native vLLM Serving

**Author**: Gemini 3.8 Flash (pr32/polecats/garnet)
**Role**: Polecat Worker (rig: pr32)
**Branch**: `research/native-v2-gemini-local`
**Base Commit**: `e75dbdd`
**Focus Area**: Exact request/batch/GPU completion timing qualification, probe overhead, and native fair statistical protocol
**Artifacts Owned**:
- `experiments/mpc-research/native-v2/round1/gemini.md`
- `experiments/mpc-research/native-v2/round1/gemini_timing_contract.py`
- `experiments/mpc-research/native-v2/round1/test_gemini_timing_contract.py`

---

## 1. Executive Summary & Ground Truth Rulings

### 1.1 Current Scientific Standing
1. **MPC has not demonstrated superiority over Fixed16, One-move, or reactive baselines.**
   Historical synthetic CPU benchmarks and exploratory GPU scouts showed workload-dependent rankings (e.g. Budget32 achieving higher completions on short prompts, Budget16 performing better on saturated long prompts), but no adaptive Model Predictive Controller (MPC) has demonstrated statistically rigorous, reproducible outperformance over simple baselines under a fair native evaluation protocol.
2. **Empirical Falsification of the Serial HTTP Proxy Model**:
   The candidate surrogate model that treated HTTP first-residence ($T_{\text{first\_residence}} \approx 0.037968 + 0.00005423 \times p$ seconds) as an exclusive, serial GPU prefill service has been **mathematically and empirically falsified**:
   - For prompt $p = 128$, the unit serial service time is $0.044909$ seconds.
   - Over a 90-second measurement window from a drained zero initial state, the maximum theoretical completions under serial FIFO execution is:
     $$\left\lfloor \frac{90.0}{0.044909} \right\rfloor = 2004 \text{ completions}$$
   - However, in native episode `train-160001-1`, **2449 natural EOS completions** were empirically observed within the 90-second window (and 2056 in `train-160001-0`).
   - Reconciling 2449 completions under a serial model requires $2449 \times 0.044909 = 109.98$ seconds of serial work—a deficit of $19.98$ seconds—requiring at least 445 carry-in requests at $t=0$, directly contradicting the verified initial drain.
   - **Conclusion**: Continuous batching, GPU tensor core parallelism, and coexisting prefill/decode execution violate the exclusive serial hypothesis. Any MPC formulation that relies on serial prefill service assumptions will produce systematically biased, invalid control actions.

---

## 2. Plant, Clocks, and Information Structure

### 2.1 The Native Plant and Actuator Boundaries
A critical architectural principle must be maintained:
- **Gateway Reservation Budget ($u(t) \in \{16, 32\}$) is a COST budget, NOT a request concurrency cap.**
  Each request $i$ consumes a weighted cost $c_i = \text{estimate}(p_i, m_i)$ based on prompt tokens $p_i$ and declared maximum output tokens $m_i$. The gateway admits request $i$ if and only if:
  $$R(t-) + c_i \le u_{\text{applied}}(t)$$
  where $R(t) = \sum_{i \in \text{active}} c_i$. Lowering $u(t)$ halts future admissions; it never preempts, evicts, or throttles in-flight requests. Raising $u(t)$ creates admission headroom but does not inject synthetic load.
- **Engine-Level Token Budgets are NOT Gateway Actuators.**
  Internal vLLM scheduler knobs (such as `max_num_batched_tokens = 2048`, `max_num_seqs = 32`, chunked prefill chunk sizes, and KV block allocation policies) operate inside the vLLM EngineCore process. They are static configuration parameters for the native serving instance, not dynamic control inputs exposed to the Gateway HTTP actuator.

### 2.2 Continuous Batching and Asynchronous Execution Architecture
vLLM 0.30.0 operates an asynchronous iteration loop:
1. **Scheduler Step**: The scheduler selects active requests from the `running` and `waiting` sets, packing up to `max_num_batched_tokens` across prefill chunks and decode tokens.
2. **Model Forward Launch**: `GPUModelRunner.execute_model` is invoked on the CPU, enqueuing CUDA kernels across multiple streams (compute, memory transfer, KV cache management).
3. **CPU Return $\ne$ GPU Completion**: The CPU function `execute_model` returns asynchronously once kernels are enqueued. The CPU scheduler begins preparing metadata for the *next* iteration while the GPU is executing the *current* iteration.
4. **Token Sampling & Transport**: Sampled token IDs are transferred via asynchronous host-to-device/device-to-host copies, processed by the frontend, and streamed back to the Gateway via Server-Sent Events (SSE).

### 2.3 Clock Domains and Monotonicity Guarantees
The distributed serving pipeline spans multiple distinct clock domains:
- $\mathcal{C}_{\text{client}}$: Client workload generator monotonic clock (measures scheduled arrival $t_{\text{sched}}$ and received stream chunks).
- $\mathcal{C}_{\text{gw}}$: Gateway process monotonic clock (measures arrival $t_{\text{gw\_recv}}$, admission decision $t_{\text{admit}}$, and SSE forward).
- $\mathcal{C}_{\text{engine}}$: vLLM EngineCore monotonic clock (measures scheduler enqueue, utility observe time, and sample callbacks).
- $\mathcal{C}_{\text{gpu}}$: GPU device hardware timer (CUPTI kernel timestamps, nanosecond hardware counters).

**Timing Contract Invariant**: Timestamps cannot be subtracted across clock domains without an explicit, validated synchronization protocol. All online serving decisions must operate strictly on observations within certified domain-compatible timestamps. Monotonic continuity is invalidated if a container or node reboots (boot ID mismatch).

---

## 3. Observability, Probe Overhead, and Exact Timing Qualification

### 3.1 Request Lifecycle Timing Decomposition
To eliminate the conflation of network transport with GPU execution, we define the exact non-overlapping timing decomposition for each admitted request $i$:

$$T_{\text{TTFT}, i} = \Delta t_{\text{ingress}, i} + \Delta t_{\text{gw\_queue}, i} + \Delta t_{\text{engine\_wait}, i} + \Delta t_{\text{gpu\_prefill}, i} + \Delta t_{\text{egress}, i}$$

Where:
1. **Ingress Transport Lag**: $\Delta t_{\text{ingress}, i} = t_{\text{gw\_recv}, i} - t_{\text{sched}, i}$ (client network delay, scheduler dispatch jitter).
2. **Gateway Processing/Reservation Gating**: $\Delta t_{\text{gw\_queue}, i} = t_{\text{admit}, i} - t_{\text{gw\_recv}, i}$ (admission check, CAS serialization).
3. **Engine Waiting / Backlog**: $\Delta t_{\text{engine\_wait}, i} = t_{\text{batch\_start}, i} - t_{\text{vllm\_enq}, i}$ (time spent waiting in scheduler queues before the first chunk is scheduled).
4. **GPU Execution Span**: $\Delta t_{\text{gpu\_span}, b} = t_{\text{gpu\_end}, b} - t_{\text{gpu\_start}, b}$ for each iteration batch $b$.
5. **Kernel Busy Union vs GPU Gaps**:
   $$\Delta t_{\text{kernel\_busy}, b} = \mu\left(\bigcup_{k \in \text{kernels}(b)} [t_{\text{start}, k}, t_{\text{end}, k}]\right)$$
   $$\Delta t_{\text{gpu\_gap}, b} = \Delta t_{\text{gpu\_span}, b} - \Delta t_{\text{kernel\_busy}, b}$$
   The GPU gap represents pipeline bubbles, CUDA launch latency, and synchronization overhead.
6. **Egress Transport & SSE Streaming**: $\Delta t_{\text{egress}, i} = t_{\text{client\_first}, i} - t_{\text{first\_token\_sampled}, i}$.

### 3.2 Profiling Probe Distortion vs Low-Overhead RPC Observation
Empirical evidence from `analyze_profile.py` and CUPTI traces demonstrates:
- **CUPTI Profiler Intrusion**: Enabling the built-in PyTorch/CUPTI profiler injects non-negligible CPU and GPU kernel launch overhead. CPU annotations for batch execution report median durations (e.g. 68.39ms for Prefill1, 128 tokens) that deviate significantly from kernel busy times (10.36ms). **Profiler traces cannot be used for serving calibration.**
- **Read-Only Utility RPC Qualification (`pr33_native_observer`)**:
  - The utility endpoint `/pr33-native/state` calls `EngineCore.pr33_native_observe` directly via async utility RPC.
  - It reports scheduler queue depths (`running`, `waiting`, `pending_batches`), KV cache utilization (`kv_cache_usage`), and per-request token progress.
  - **Probe Overhead Budget**: A compliant probe must satisfy:
    1. Zero GPU execution stalling (strictly no `cudaDeviceSynchronize` or blocking kernel inspections).
    2. Execution latency $\le 15$ms.
    3. Fractional overhead $\le 2\%$ of the control time slot $T_{\text{slot}} = 1.0$s.
    4. Observation age $\le 0.5$s and sample skew $|t_{\text{frontend\_sampled}} - t_{\text{sampled}}| \le 0.05$s.

---

## 4. Comparative Dynamical Model Formulations

We evaluate six candidate dynamical model classes for capturing vLLM serving behavior:

| Model Class | Mathematical Formulation | Strengths | Fatal Failure Modes in Native vLLM |
|---|---|---|---|
| **1. Classical Queueing / Fluid Model** | $M/G/1/\text{PS}$ or fluid differential equations: $\dot{Q}(t) = \lambda(t) - \mu(R(t))$ | Closed-form analytical tractability, simple parameter identification. | **Fails completely under continuous batching.** Assumes homogeneous processor sharing or single-server FIFO; ignores token-level chunking, prefill-decode asymmetry, and non-Markovian EOS. |
| **2. Batch-State-Space Model** | Discrete-time linear/bilinear state update: $\begin{bmatrix} P_{k+1} \\ D_{k+1} \end{bmatrix} = A \begin{bmatrix} P_k \\ D_k \end{bmatrix} + B u_k + \Gamma w_k$ | Directly tracks prefill query tokens $P_k$ and active decoders $D_k$. Fast matrix QP/LP solvers. | **Nonlinear interference**: The true service rate is non-affine: $E[\Delta \text{out}] = \frac{1000 \Delta t}{\theta_0 + \theta_D D + \theta_P P_k}$. Fails when batch composition transitions between compute-bound and memory-bound regimes. |
| **3. Hybrid Discrete-Continuous Model** | Hybrid Automata with discrete modes $m \in \{\text{Idle}, \text{PrefillDominated}, \text{DecodeSaturated}, \text{KVPressed}\}$ and continuous remaining work dynamics. | Explicitly captures discrete event admissions, chunk transitions, and continuous token progression. Respects whole-cost release invariants. | High combinatorial search complexity if solving online MILP. Requires approximation to fit within the 100ms solver deadline. |
| **4. Learned Neural Transition Model** | Neural Network (MLP/LSTM/Transformer): $x_{k+1} = \text{NN}(x_k, u_k)$ | High expressive capacity, can fit complex nonlinear empirical profiles. | **Severe OOD vulnerability**: Extrapolates catastrophically outside training load support; lacks physical mass/token conservation; non-convex optimization prevents certified real-time fallback. |
| **5. Piecewise Nonlinear Model** | Bilinear service law with regime switching: $\tau_{\text{iter}} = \max(\tau_{\text{prefill}}(P), \tau_{\text{decode}}(D)) + \tau_{\text{interf}}(P, D)$ | Captures physical GPU roofline: compute-bound prefill regime vs memory-bandwidth-bound decode regime. | Requires identifying phase boundary parameters; requires careful regularization during calibration. |
| **6. Probabilistic / Survival Model** | Hazard function for token completion: $h(t \mid p, \text{age}) = \frac{f(t)}{1 - F(t)}$ with stochastic remaining work belief $b_k$. | Directly models unknown natural EOS distributions and KV cache exhaustion risk without unverified point estimates. | Requires integration over scenario realizations (Monte Carlo scenario sampling); variance can be high if sample count $M$ is small. |

### 4.1 Recommended Formulation: Hybrid Probabilistic Batch-State Model
The optimal identifiable model is a **Hybrid Probabilistic Batch-State Model**:
- **Discrete Transitions**: Exact gateway ledger updates upon admission ($R \leftarrow R + c_i$) and completion ($R \leftarrow R - c_i$).
- **Continuous Batch Execution Dynamics**: Nonnegative bi-affine iteration service law:
  $$\tau_{\text{iter}}(P_k, D_k) = \theta_0 + \theta_D D_k + \theta_P P_k$$
  with $\theta_0 \approx 12.31$ms, $\theta_D \approx 0.049$ms/seq, and $\theta_P \approx 8.73$ms/kilo-prompt-token (identified from native scout data).
- **Stochastic Remaining Work Belief ($b_k$)**: Nonparametric empirical conditional survival distribution conditioned on already observed decode tokens:
  $$\mathbb{P}(\text{Remaining} > \ell \mid \text{Observed} = n_{\text{obs}}) = \frac{\sum_{j} \mathbf{1}\{Y_j > n_{\text{obs}} + \ell\}}{\sum_{j} \mathbf{1}\{Y_j > n_{\text{obs}}\}}$$
  evaluated strictly on TRAIN split records prior to the declared cutoff.

---

## 5. Complete Control Problem Formulation

### 5.1 State, Action, Disturbance, and Observation Vectors
- **Physical State** $x_k \in \mathcal{X}$:
  $$x_k = \left( L_k, P_k, D_k, \kappa_k, u_{\text{applied}, k}, v_k, \text{pending\_cmd}_k \right)$$
  - $L_k$: Exact Gateway admission ledger containing active request records $(i, c_i, p_i, m_i, t_{\text{admit}, i}, n_{\text{out}, i})$.
  - $P_k$: Total prompt tokens of admitted requests awaiting or undergoing prefill.
  - $D_k$: Count of requests currently undergoing decode.
  - $\kappa_k \in [0, 1]$: Physical KV cache block occupancy in the engine.
  - $u_{\text{applied}, k} \in \{16, 32\}$: Currently active gateway reservation budget.
  - $v_k$: Monotonic sequence/version of the applied command.
  - $\text{pending\_cmd}_k$: Unacknowledged in-flight budget update command (or null).

- **Control Action** $u_k \in \mathcal{U}$:
  $$u_k \in \{16, 32\}$$
  Discrete reservation-cost budget assigned to the Gateway for the interval $[t_k, t_{k+1})$.

- **Disturbance Vector** $w_k \in \mathcal{W}$:
  $$w_k = \left( \{ (t_{\text{arr}, j}, p_j, m_j) \}, \text{EOS}_j, \Delta t_{\text{transport}, j}, \delta_{\text{engine}} \right)$$
  Exogenous arrival events, stochastic EOS realizations, transport latency jitter, and engine execution variance.

- **Observation Vector** $y_k \in \mathcal{Y}$:
  $$y_k = \left( L_{\text{gw}}(t_k), \text{Obs}_{\text{native}}(t_k) \right)$$
  Joined observation record containing Gateway ledger snapshot and vLLM EngineCore utility sample ($N_{\text{running}}, N_{\text{waiting}}, N_{\text{pending\_batches}}, \kappa_{\text{kv}}, \{ (r_j, p_j, n_{\text{computed}, j}, n_{\text{out}, j}) \}$).

### 5.2 Objective Function
The controller maximizes the **Expected All-Offered Composite Good Completions**:

$$\max_{u_{k:k+N-1}} J = \frac{1}{M} \sum_{m=1}^M \sum_{i \in \mathcal{C}_{\text{eval}}} g_i^{(m)}$$

where:
$$g_i = \mathbf{1}\{\text{Complete Success}\} \cdot \mathbf{1}\{T_{\text{TTFT}, i} \le 0.500\text{s}\} \cdot \mathbf{1}\{\text{TPOT}_{\text{mean}, i} \le 0.050\text{s}\}$$
- Rejections, dropouts, and cancellations score $0$.
- Admitted requests whose latency exceeds either the 500ms TTFT or 50ms TPOT threshold score $0$.
- $\mathcal{C}_{\text{eval}}$ comprises all unresolved requests active at decision time plus all requests offered during the evaluation horizon $D = 5.0$s.

### 5.3 Constraints, Horizon, and Real-Time Solver Budget
1. **Hard Reservation Invariant**:
   $$R(t) \le \max(R(0), u_{\text{applied}}(t)), \quad \forall t$$
2. **Real-Time Solver Deadline**:
   $$t_{\text{solve}} \le 100\text{ms}$$
   If solver computation exceeds 100ms or encounters an unhandled exception, the actuator automatically executes the **Safe Fallback**: maintain the current budget $u_{\text{applied}}$ or revert to Fixed16.
3. **Control Cadence and Planning Horizon**:
   - Control Slot: $T_{\text{slot}} = 1.0$s.
   - Horizon Length: $N = 3$ steps ($3 \times 1.0$s plans).
   - Candidate Sequences: $2^3 = 8$ candidate action sequences: $\{16, 32\}^3$.
   - Common Continuation: All 8 plans append a fixed 2.0-second continuation at Fixed16 followed by complete system drain (bounded at 35.0s) to eliminate horizon-edge truncation gaming.

---

## 6. Discrimination Analysis: When Does Multi-Step MPC Add Value vs One-Move / Reactive?

A central question in predictive control design is whether multi-step lookahead ($N > 1$) adds any genuine value over a 1-step lookahead ("One-move") or a well-tuned reactive feedback controller.

### 6.1 The Horizon Value Theorem for Continuous Batching
Multi-step lookahead provides value **if and only if the plant state possesses coupling dynamics whose settling time $T_{\text{settle}}$ exceeds the control slot $T_{\text{slot}}$**:
1. **Degenerate Regime ($T_{\text{settle}} \le T_{\text{slot}}$)**:
   Under short prompt workloads (e.g. prompt 128, output 36–116 tokens), requests clear within 1.0 to 1.5 seconds. Admitting a burst at step $k$ has negligible carry-over congestion into step $k+2$. In this regime, **One-move ($N=1$) is mathematically optimal**, and multi-step MPC ($N=3$) merely inflates computational overhead without improving goodput.
2. **Coupled Carry-Over Regime ($T_{\text{settle}} \gg T_{\text{slot}}$)**:
   Under long prompt workloads (e.g. prompt 4096), a request admitted at step $k$:
   - Consumes prefill execution capacity across multiple iterations.
   - Allocates 4096 tokens of KV cache blocks that remain pinned in GPU HBM for 10–20 seconds during subsequent decoding.
   - Admitting an additional request at step $k$ creates a delayed "congestion wave" at step $k+2$ and $k+3$, causing subsequent arrivals to suffer severe TTFT degradation and potentially inducing KV cache thrashing.
   - **Here, Multi-step MPC adds decisive value**: It can anticipate the future KV allocation peak and proactively lower the admission budget at step $k+1$ before the physical threshold is breached.

### 6.2 Discriminating Ablation Design (Preventing Horizon Inflation)
To scientifically establish whether multi-step MPC outperforms One-move:
- **Controlled Evaluation Window**: Fix the evaluation scoring window $D = 5.0$s and continuation tail identical across all policies.
- **Factorial Comparison Matrix**:
  1. **Fixed16**: Constant $u = 16$.
  2. **Fixed32**: Constant $u = 32$.
  3. **Reactive Hysteresis**: $u(t) = 16$ if $W_{\text{prompt}} > \theta_{\text{thresh}}$ or $\kappa_{\text{KV}} > 0.75$, else $32$.
  4. **One-Move ($N=1$)**: Optimize $u_k \in \{16, 32\}$ over 1 step, holding continuation fixed.
  5. **Multi-Step MPC ($N=3$)**: Optimize $\{u_k, u_{k+1}, u_{k+2}\} \in \{16, 32\}^3$.
  6. **Multi-Step MPC ($N=5$)**: Optimize over 5 steps.
- **Success Criterion**: MPC ($N=3$) is justified if and only if it achieves a statistically significant improvement over One-move ($N=1$) and Reactive Hysteresis by at least the pre-registered margin $\Delta \ge 0.5$ good completions/s under long prompt or mixed shift workloads.

---

## 7. Falsification Protocol for Major Assumptions

Every candidate model assumption must be subjected to a concrete, repeatable falsification experiment:

```
+-----------------------------------+---------------------------------------------+----------------------------------------------+
| Assumption                        | Proposed Falsification Experiment           | Falsification Condition                      |
+-----------------------------------+---------------------------------------------+----------------------------------------------+
| 1. Batch-Additive Service Time:   | Measure GPU span across batch sizes         | Falsified if:                                |
|    Tau_iter = theta_0 + theta_D*D | B in {1, 2, 4, 8, 16, 32} with fixed prompt | Residuals exhibit quadratic scaling or       |
|    + theta_P*P                    | tokens. Compare linear fit against actual.  | sub-linear saturation (|R^2_lin| < 0.90).    |
+-----------------------------------+---------------------------------------------+----------------------------------------------+
| 2. Stationary Transport Latency:  | Inject HTTP requests under zero load vs     | Falsified if:                                |
|    Delta_t_transport is constant  | 90% network saturation. Compare client-to-  | Egress transport delay shifts by > 20ms      |
|    and load-independent.          | gateway vs engine-to-client latencies.      | under load (p < 0.01).                       |
+-----------------------------------+---------------------------------------------+----------------------------------------------+
| 3. Unbiased Conditional EOS:      | Compare conditional empirical quantile      | Falsified if:                                |
|    Remaining decode work matches  | predictions against realized output tokens  | Realized output tokens violate the 95%       |
|    TRAIN survival distribution.   | on held-out validation sequences.           | confidence interval of the survival model.   |
+-----------------------------------+---------------------------------------------+----------------------------------------------+
| 4. KV Memory Independence:        | Run constant arrival rate at 20% KV cache   | Falsified if:                                |
|    Decode TPOT is independent of  | occupancy vs 85% KV cache occupancy.        | TPOT increases by > 15% at 85% KV cache      |
|    KV allocation level.           |                                             | due to memory fragmentation or paging.       |
+-----------------------------------+---------------------------------------------+----------------------------------------------+
| 5. Multi-Step Superiority:        | Run paired-block comparison of N=3 MPC      | Falsified if:                                |
|    N=3 MPC outperforms N=1       | vs N=1 One-Move across 8 matched random     | Mean goodput delta (N=3 - N=1) < 0.5 req/s   |
|    One-Move on native plant.      | arrival seeds under long-prompt workload.   | or permutation test p-value >= 0.05.         |
+-----------------------------------+---------------------------------------------+----------------------------------------------+
```

---

## 8. Native Fair Statistical Protocol

### 8.1 Protocol Integrity & Held-Out Partition Protection
1. **Unopened Held-Out Test Set**:
   - Seeds `160021`, `160022`, `160023`, `160024` are strictly designated as the **confirmatory held-out Test set**.
   - **Zero-Access Rule**: These seeds must not be opened, inspected, executed, or tuned against during Round 1, Round 2, or model development.
2. **Paired-Block Design**:
   - All controller evaluations must execute in matched pairs against identical pre-generated Poisson offered arrival tapes (same arrival timestamps, same prompt contents, same max output limits).
   - Order of controller execution within each block must alternate (e.g. Block 1: A then B; Block 2: B then A) to cancel linear thermal drift or memory fragmentation effects.
3. **Statistical Power & Superiority Margin**:
   - Pre-registered minimum meaningful superiority margin:
     $$\Delta_{\text{margin}} = 0.50 \text{ good completions / second}$$
   - Exact permutation test on paired differences: requires two-sided $p < 0.05$ across $K \ge 8$ independent seed blocks.
4. **Primary Metric and Hard Guardrails**:
   - Primary Metric: All-offered composite goodput (TTFT $\le 500$ms, TPOT $\le 50$ms).
   - Guardrail 1: Rejection rate must not exceed baseline by more than $5\%$ relative.
   - Guardrail 2: Solver real-time deadline violation rate must satisfy $P(t_{\text{solve}} > 100\text{ms}) \le 0.01$.

---

## 9. CPU Diagnostic Prototype & Verification

### 9.1 Prototype Implementation (`gemini_timing_contract.py`)
In accordance with the bounded execution mandate, we implemented a self-contained, reproducible CPU prototype (`gemini_timing_contract.py`) providing:
1. `ClockDomain`: Monotonic clock domain validation preventing cross-domain arithmetic and detecting container/node reboot drift.
2. `RequestTimingTrace`: Physical causality enforcement across the serving pipeline, isolating transport lag from GPU execution, and computing exact composite goodput metrics.
3. `BatchExecutionInterval`: Microscopic execution tracking separating CPU scheduling time, GPU batch span, active kernel busy union, and GPU idle gaps.
4. `SerialCapacityScreener`: Mathematical falsification engine replicating the 2004 vs 2449 capacity contradiction.
5. `ProbeOverheadGuard`: Online overhead monitoring rejecting intrusive profilers (CUPTI) and synchronous GPU blocks during serving.
6. `NativeFairStatisticalProtocol`: Paired-block statistical evaluator enforcing pre-registered margins, permutation testing, and held-out test seed isolation.

### 9.2 Test Suite Execution & Verification (`test_gemini_timing_contract.py`)
The diagnostic suite contains 18 comprehensive unit and integration tests.
All tests were executed on the local environment and passed cleanly:
```bash
/usr/bin/python3 -m unittest experiments/mpc-research/native-v2/round1/test_gemini_timing_contract.py
..................
----------------------------------------------------------------------
Ran 18 tests in 0.001s

OK
```
All 18 tests also executed and passed under pytest (`18 passed in 0.02s`).

---

## 10. Conclusion and Next Steps

1. **Round 1 Completion**: We have established the theoretical foundations, dynamical model formulations, timing decomposition contracts, and falsification criteria required for native vLLM predictive control.
2. **Safe Stance**: No remote push, merge, rebase, or `gt done` operations were invoked. The code and report are committed locally to `research/native-v2-gemini-local`.
3. **Ready for Mayor Review**: Awaiting the release of peer reports for Round 2 cross-review.
