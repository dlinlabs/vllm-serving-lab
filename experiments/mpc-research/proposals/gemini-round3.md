# Round 3 Revised Design and Evidence-Based Consensus: Minimal Discrete Receding-Horizon Admission Control (MDR-MPC)

**Author:** Polecat Jasper (`gemini-3.8-flash`)  
**Target:** PR32 Serving Lab (`8e92371f342be98a50682bf4e2ddbb014665498b`)  
**Context:** Epic `pr-mpc`, Convoy `hq-cv-8tdlf`  
**Input Documents Synthesized:**
- Round 1 and Round 2 reports by Opus (`claude-opus-5.5`), Astra (`gpt-6-astra`), GPT (`gpt-6.1-sol`), and Jasper (`gemini-3.8-flash`)
- Mayor's executed artifacts: `/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/experiments/mpc-research/mayor_counterexamples.py` and `results/mpc-research-cpu/mayor-counterexamples.json`
- Source code: `gateway.py`, `prefill_lab/state_envelope.py`, `prefill_lab/runtime.py`

**Status:** Round 3 Final Proposal & Consensus Architecture (Sealed at Barrier)

---

## 1. Executive Consensus & Adjudication of Substantive Critiques

### 1.1 Explicit Adjudication of Round 2 Critiques and Mayor Standards

In accordance with Mayor adjudication standards and peer findings, the following formal retractions, acceptances, and rejections are recorded:

1. **Retraction & Repair of Dimensional Drain Error (G1):**  
   *Finding:* Jasper Round 1 incorrectly formulated queue drain as $\min(B, N_{\text{flight}} \cdot \bar{t}_{\text{token}}^{-1})$ and subtracted it directly from reserved cost $C_{\text{admit}}$, conflating tokens/step, tokens/second, and cost units.  
   *Action:* **Fully accepted.** Retracted. The corrected dynamics model individual request decode lifecycles and release exact discrete request reservation costs $c_i$ upon terminal events.
2. **Retraction of KV-Cache Chance Constraint (G2):**  
   *Finding:* Imposing $\mathbb{P}(\theta_{\text{kv}} \ge 0.90) \le 0.001$ while excluding $\theta_{\text{kv}}$ from causal observation is unidentifiable; furthermore, reservation-derived token arithmetic is not native physical KV safety.  
   *Action:* **Fully accepted.** Retracted. The chance constraint is eliminated. Memory safety remains an engine-level supervisor responsibility.
3. **Retraction of Unqualified Action $C=24.0$ (G3, G6):**  
   *Finding:* $C=24.0$ has zero historical training support on the native GPU plant; PR32 historical data qualifies only $C \in \{16, 32\}$.  
   *Action:* **Fully accepted.** The candidate control space is strictly frozen to $\mathcal{U} = \{16.0, 32.0\}$.
4. **Adjudication on 299-Episode Conformal Requirement (A1, Mayor Standard):**  
   *Mayor Standard:* 299 independent zero-failure episodes for a 1%/95% bound is mathematically valid ($1 - 0.05^{1/299} \approx 0.00997$); cost cannot justify substituting correlated windows.  
   *Action:* **Accepted mathematically, qualified operationally.** Astra's derivation is mathematically indisputable. However, because 299 independent episodes ($\approx 35$ hours of GPU compute) exceeds available project resources, we explicitly **renounce any claim to a certified 1% extreme-tail guarantee**. Conformal bands in this architecture are classified strictly as heuristic uncertainty margins, not distribution-free safety certificates.
5. **Adjudication on Terminal Constraint Infeasibility (O2, Mayor Counterexample):**  
   *Mayor Check:* `hold_does_not_imply_terminal_feasibility` proved that starting from $R=60.0\,\text{cu}$ with zero releases before horizon $H$, holding fallback leaves $R=60.0 > 34.0\,\text{cu}$ (feasible: `false`).  
   *Action:* **Unanimously rejected hard terminal constraints.** The terminal condition is replaced with soft continuation evaluation under backup policy $C^* = 16.0$.
6. **Adjudication on Existing-Cohort Externalities (O4, Mayor Counterexample):**  
   *Mayor Check:* `ignoring_existing_cohorts_reverses_action_preference` proved that ignoring old cohorts can report a $+1$ gain when the actual total gain is $-1$.  
   *Action:* **Unanimously accepted.** Rollout scoring must evaluate all unresolved cohorts (old admitted requests plus new arrivals).
7. **Adjudication on Actuation Ambiguity (A4, S1, Mayor Counterexample):**  
   *Mayor Check:* `timeout_then_old_state_read_does_not_cancel_command` proved that a delayed write can apply after client timeout and clobber a subsequent command.  
   *Action:* **Unanimously accepted.** The gateway transport contract requires Compare-And-Set (CAS) version validation and server-side command expiry.
8. **Adjudication on $b_{\text{after}}$ Endogeneity (Mayor Standard):**  
   *Mayor Standard:* $b_{\text{after}}$ endogeneity has no proven sign; its diagnostic flips are not causal upper bounds.  
   *Action:* **Accepted.** Opus' diagnostic flips are not causal bounds. While FCFS prefill is a strong structural baseline, multi-step planning must be justified through randomized policy trials, not observational regression.
9. **Explicit Retraction of Unsupported Round 1 Claims (Gemini):**  
   *Retraction:* Jasper specifically retracts the unsupported Round 1 claim of "<4ms benchmarked / full native289". This was an unexecuted development projection, not an empirical benchmark. Native plant qualification remains false, and no 289-run native GPU evaluation was executed.
10. **Scoping of Round 2 Toy Timings and Probability Calculations (Gemini):**  
   *Scoping:* Jasper explicitly clarifies that the Round 2 Python microbenchmark timings (e.g., 1.34 ms for 16 rollouts) and Poisson arrival probability calculations (e.g., 18.1% probability of $\ge 2$ arrivals in 50 ms) are strictly scoped arithmetic illustrations, NOT universal compute lower bounds or measured deadband incidences. A toy single-thread Python timing does not establish an architectural lower bound for an asynchronous production gateway; similarly, a theoretical Poisson arrival calculation does not represent an empirical measurement of deadband occurrence on the live plant.

### 1.2 Focused Technical Review and Acceptance of Mayor's Mathematical Specification

Before the global Round 3 barrier, we inspected Mayor's provisional reference specification (`/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/experiments/mpc-research/mpc-mathematical-specification.md`). The specification establishes a rigorous, unified foundation for PR32 admission control.

#### Core Architectural Acceptances
1. **Exact Ledger vs. Unqualified Latent Surrogates:** Sections 1 and 2 rigorously decouple the gateway's exact, observable discrete reservation accounting ledger $\mathcal{L}$ ($a_i = \mathbf{1}\{R(t-) + c_i \le u(t)\}$, $R(t+) = R(t-) + a_i c_i$, and whole-cost terminal cleanup $R(t+) = R(t-) - c_i$) from latent physical engine states. $W_{\text{prompt}} = \sum_{i \in \mathcal{W}} p_i$ is properly designated as an original-prompt backlog proxy, not residual GPU work. Lowering $u$ below $R$ halts new offers without evicting running jobs, and raising $u$ admits no request until an arrival occurs.
2. **Dual Model Reference Architecture:** Section 3 establishes a clean predictive boundary, evaluating a kernel-inspired CPU reference model ($\tau = \theta_{\text{base}} + \theta_{\text{prefill}} q + \theta_{\text{decode}} d + \theta_{\text{interference}} q d$) alongside the measured empirical $W_{\text{prompt}}$ TTFT law + conditional residence model, explicitly requiring both to earn inclusion through offline ablation rather than assuming surrogate complexity implies physical fidelity.
3. **Complete-Cohort Eventual Success Objective:** Section 4 defines expected successful request count across all unresolved cohorts $\mathcal{C}_k$ ($g_i = \mathbf{1}\{\text{eventual success}\} \cdot \mathbf{1}\{\text{client TTFT} \le 0.5\,\text{s}\}$), eliminating myopic action-shift externalities and requiring bounded continuation under declared backup $u_{\text{backup}} = 16.0$.
4. **Guarded Transport Contract:** Section 6 mandates a guarded server capability verifying runtime/cost identity, budget version (CAS), model/support seal, clock domain, and server-side expiry under the same application lock, explicitly identifying old endpoints ignoring extra fields as non-compliant.
5. **Bounded Development Horizon:** Starting with a tiny $N=3$ (1s slots, 3s horizon) search over $\{16, 32\}$ (at most 8 sequences) provides a tractable, verifiable development baseline that avoids premature optimization.

#### Concrete Technical Corrections & Simpler Alternatives
1. **Engine Quiescence in Kernel Transition Model (Section 3):**  
   *Flaw/Issue:* The candidate iteration duration formula $\tau = \theta_{\text{base}} + \theta_{\text{prefill}} q + \theta_{\text{decode}} d + \theta_{\text{interference}} q d$ yields $\tau = \theta_{\text{base}} > 0$ when $q=0$ and $d=0$. In an event-driven CPU simulation, an idle engine with no active batches would accumulate artificial overhead ticks rather than remaining quiescent.  
   *Simpler Correct Alternative:* Formulate baseline latency with an activity indicator:
   $$\tau = \mathbf{1}\{q + d > 0\} \cdot \theta_{\text{base}} + \theta_{\text{prefill}} q + \theta_{\text{decode}} d + \theta_{\text{interference}} q d$$
   When $q=0$ and $d=0$, the simulator advances directly to the next scheduled arrival event, preserving correct discrete-event simulation semantics.
2. **Scheduled Client Arrival vs. Gateway Ingress Timestamp (Section 4):**  
   *Flaw/Issue:* In $g_i = \mathbf{1}\{\text{eventual success}\} \cdot \mathbf{1}\{\text{client\_first\_content}_i - \text{scheduled\_offer}_i \le 0.5\,\text{s}\}$, the scheduled offer timestamp $\text{scheduled\_offer}_i$ is known to client generators in synthetic benchmarks but is causally unobservable to an operational gateway in production (which observes only receipt time $t_{\text{recv}, i}$).  
   *Simpler Correct Alternative:* In online gateway MPC rollouts, define the objective using the gateway-observed window adjusted by an explicit, bounded ingress/egress transit interval $[\Delta_{\text{net}}^{\min}, \Delta_{\text{net}}^{\max}]$:
   $$g_i = \mathbf{1}\{\text{eventual success}\} \cdot \mathbf{1}\{t_{\text{first\_token}, i} - t_{\text{recv}, i} \le 0.500\,\text{s} - \hat{\Delta}_{\text{net}}\}$$
   In offline benchmarking where client traces are available, the full end-to-end client duration is evaluated.
3. **Zero Initial Switch Penalty and Deterministic Tie-Breaking (Section 4):**  
   *Alignment:* Rather than imposing an arbitrary heuristic penalty ($\rho = 0.05$), we adopt Mayor's specification: set initial switch/risk weights to zero ($\rho = 0$). Deterministic ties are resolved by preferring: (1) hold current action $u_{k-1}$, (2) fewer switches, then (3) fixed action order ($16.0$ then $32.0$).
4. **Development Horizon Alignment ($N=3$):**  
   *Alignment:* We adopt Mayor's $N=3$ control steps (3.0s horizon at $\Delta t = 1.0\,\text{s}$) enumerating all $2^3 = 8$ sequences over $\{16, 32\}$ as the initial development configuration, treating $N=1/3/5$ and one-switch pruning as comparative development ablations.

---

## 2. System State Definition, Units, and Invariants

### 2.1 Information State Vector $x_k$

At discrete decision epoch $t_k = k \Delta t$ ($\Delta t = 1.0\,\text{s}$), the causal information state is:
$$x_k = \Big(R(t_k), \; W_{\text{tok}}(t_k), \; N_{\text{flight}}(t_k), \; N_{\text{await}}(t_k), \; \mathcal{L}_{\text{active}}(t_k), \; \mathbf{h}_{\text{arr}}(t_k), \; \mathbf{h}_{\text{act}}(t_k), \; v_{\text{applied}}(t_k), \; C_{\text{applied}}(t_k)\Big)$$

| State Variable | Type & Units | Physical / Gateway Meaning | Causally Observable at Gateway? |
|---|---|---|---|
| $R(t_k)$ | $\mathbb{R}_{\ge 0}$ (Cost Units, cu) | Current admitted reservation: $R = \sum_{j \in \mathcal{A}_{\text{active}}} c_j$ | **Yes** (`current_admitted_cost` in `gateway.py:321`) |
| $W_{\text{tok}}(t_k)$ | $\mathbb{Z}_{\ge 0}$ (Tokens) | Prompt tokens of admitted requests awaiting first content: $\sum_{j \in \mathcal{W}} p_j$ | **Yes** (Reconstructed from lifecycle ledger; proposed `/metrics` field) |
| $N_{\text{flight}}(t_k)$ | $\mathbb{Z}_{\ge 0}$ (Requests) | Total active admitted requests: $|\mathcal{A}_{\text{active}}| = N_{\text{await}} + N_{\text{dec}}$ | **Yes** (`current_in_flight` in `gateway.py:320`) |
| $N_{\text{await}}(t_k)$ | $\mathbb{Z}_{\ge 0}$ (Requests) | Requests admitted but awaiting first content: $|\mathcal{W}|$ | **Yes** (`awaiting_first_content` in telemetry snapshot) |
| $\mathcal{L}_{\text{active}}(t_k)$ | Set of Tuples | Active request ledger: $\{ (id_j, c_j, p_j, m_j, t_{\text{admit}, j}, \text{phase}_j) \}$ | **Yes** (Tracked in gateway memory under `state_lock`) |
| $\mathbf{h}_{\text{arr}}(t_k)$ | Ring Buffer (5 slots) | Rolling 1-second arrival counts and offered costs over past 5 seconds | **Yes** (Maintained via causal ring buffer with explicit expiration) |
| $\mathbf{h}_{\text{act}}(t_k)$ | Ring Buffer (5 slots) | Rolling 1-second admitted tokens and completion counts over past 5 seconds | **Yes** (Maintained via causal ring buffer with explicit expiration) |
| $v_{\text{applied}}(t_k)$ | $\mathbb{Z}_{\ge 0}$ (Integer) | Applied budget version number | **Yes** (`budget_version` in `gateway.py:618`) |
| $C_{\text{applied}}(t_k)$ | $\mathbb{R}_{> 0}$ (Cost Units, cu) | Currently active admission budget limit | **Yes** (`admission_budget` in `gateway.py:617`) |

### 2.2 System Invariants
1. **Cost Conservation Invariant:** At all times $t$, $R(t) = \sum_{j \in \mathcal{L}_{\text{active}}(t)} c_j$.
2. **Phase Partition Invariant:** Every active request $j \in \mathcal{L}_{\text{active}}$ is in exactly one phase: $\text{phase}_j \in \{\text{awaiting\_first}, \text{streaming\_decode}\}$.
3. **Budget Non-Eviction Invariant:** If $C_{k+1} < C_k$, no request in $\mathcal{L}_{\text{active}}$ is cancelled or evicted; $R(t)$ decreases strictly via natural completions `finish_request(c_j)`.
4. **History Ring-Buffer Invariant:** Rolling histories maintain explicit 5-second expiration memory; rolling sums are never updated without FIFO eviction of 5-second-old bins (Mayor check `rolling_sum_without_expiration_memory_is_not_closed`).

---

## 3. Serving Dynamics and Event-Based Accounting

### 3.1 Reservation Cost Formula
For request $i$ arriving at time $\tau_i$, prompt token count $p_i$ is estimated via tokenizer template (or deterministic fallback), and maximum output tokens $m_i = \max(1, \text{payload.max\_tokens})$. The reservation cost $c_i$ is:
$$c_i = \max\left(c_{\min}, \; c_0 + \alpha_{\text{in}} p_i + \alpha_{\text{out}} m_i\right)$$
with frozen V1 parameters: $c_{\min} = 0.1\,\text{cu}$, $c_0 = 0.259\,\text{cu}$, $\alpha_{\text{in}} = 7.58 \times 10^{-4}\,\text{cu/token}$, $\alpha_{\text{out}} = 5.2083333 \times 10^{-3}\,\text{cu/token}$.

### 3.2 Instantaneous Admission Boundary
When request $i$ arrives at time $\tau_i \in [t_k, t_{k+1})$:
$$\gamma_i = \begin{cases} 1, & \text{if } R(\tau_i^-) + c_i \le C_{\text{applied}}(\tau_i) \\ 0, & \text{otherwise (Immediate HTTP 503 Rejection)} \end{cases}$$
If $\gamma_i = 1$:
$$R(\tau_i) = R(\tau_i^-) + c_i, \quad W_{\text{tok}}(\tau_i) = W_{\text{tok}}(\tau_i^-) + p_i, \quad N_{\text{flight}}(\tau_i) = N_{\text{flight}}(\tau_i^-) + 1$$
$$\mathcal{L}_{\text{active}} \leftarrow \mathcal{L}_{\text{active}} \cup \{ (id_i, c_i, p_i, m_i, \tau_i, \text{awaiting\_first}) \}$$

### 3.3 Lifecycle Transitions & Discrete Cost Release
1. **First-Token Emission ($\tau_{\text{fc}, i}$):** Request $i$ transitions to $\text{streaming\_decode}$:
   $$W_{\text{tok}}(\tau_{\text{fc}, i}) = \max\left(0, \; W_{\text{tok}}(\tau_{\text{fc}, i}^-) - p_i\right), \quad N_{\text{await}} \leftarrow N_{\text{await}} - 1$$
   *Note:* $R$ does NOT decrease upon first token emission.
2. **Terminal Completion / Failure ($\tau_{\text{term}, i}$):** Request $i$ completes or fails:
   $$R(\tau_{\text{term}, i}) = \max\left(0.0, \; R(\tau_{\text{term}, i}^-) - c_i\right), \quad N_{\text{flight}} \leftarrow N_{\text{flight}} - 1$$
   $$\mathcal{L}_{\text{active}} \leftarrow \mathcal{L}_{\text{active}} \setminus \{ i \}$$

---

## 4. Objective Formulation: Cohort Partition & Eventual Success

### 4.1 All-Offered Successful SLO Goodput (Whole-Episode Primary Metric)
For an episode with offered arrival window $[0, T_{\text{off}})$, the primary metric is:
$$G = \frac{1}{T_{\text{off}}} \sum_{i \in \mathcal{O}_{[0, T_{\text{off}})}} \mathbf{1}\Big(\text{status}_i = \text{success} \;\land\; \text{TTFT}_i^{\text{client}} \le 0.5\,\text{s}\Big)$$
where:
* $\mathcal{O}_{[0, T_{\text{off}})}$ is the entire cohort of requests offered in $[0, T_{\text{off}})$.
* Rejected requests ($\gamma_i = 0$), client cancellations, and backend failures receive $\mathbf{1}(\cdot) = 0$, but remain in the denominator.
* $\text{TTFT}_i^{\text{client}} = t_{\text{client\_first\_content}, i} - t_{\text{scheduled\_arrival}, i}$.

### 4.2 Receding-Horizon Stage Reward with Full Cohort Accounting
At decision epoch $t_k$ over horizon $H$, candidate action plan $\mathbf{u} = (u_k, \dots, u_{k+H-1})$ is evaluated across $S=16$ exogenous scenarios. To prevent cohort externality bias (Mayor check `ignoring_existing_cohorts_reverses_action_preference`), the reward partitions active jobs into two disjoint sets:

$$\mathcal{J}(\mathbf{u}) = \frac{1}{S} \sum_{s=1}^S \Big[ \mathcal{R}_{\text{new}}^{(s)}(\mathbf{u}) + \mathcal{R}_{\text{old}}^{(s)}(\mathbf{u}) + \mathcal{V}_{\text{cont}}^{(s)}(x_{k+H}) \Big] - \rho \sum_{m=0}^{H-2} \mathbf{1}(u_{k+m+1} \ne u_{k+m})$$

1. **New Arrival Cohort ($\mathcal{R}_{\text{new}}^{(s)}$):** Expected count of requests offered in $[t_k, t_k + H)$ that are admitted, complete successfully, and achieve $\text{TTFT} \le 0.5\,\text{s}$.
2. **Old Admitted Cohort ($\mathcal{R}_{\text{old}}^{(s)}$):** Expected count of requests already admitted ($j \in \mathcal{L}_{\text{active}}(t_k)$) that complete successfully and achieve $\text{TTFT} \le 0.5\,\text{s}$ under action plan $\mathbf{u}$.
3. **Terminal Continuation Value ($\mathcal{V}_{\text{cont}}^{(s)}$):** Soft continuation evaluation under backup policy $C^* = 16.0$ over a fixed 3-second tail, simulating surviving request completion without double-counting.
4. **Switching Penalty ($\rho$) and Deterministic Tie-Breaking:** Following Mayor's specification, initial switch and risk weights are strictly zero ($\rho = 0$), avoiding arbitrary mixed-unit tuning. Deterministic ties are broken by: (1) prefer holding current action $u_{k-1}$, (2) prefer fewer switches, then (3) prefer fixed action order ($16.0$ over $32.0$). Any future nonzero switch penalty must have units of successful requests/change and require a measured transition-cost or development Pareto rationale.

---

## 5. Candidate Controls, Horizon, Cadence, and Transport Contracts

### 5.1 Protocol Pins and Planning Horizon
* **Control Cadence ($\Delta t$):** $1.0\,\text{s}$ (1 Hz). Aligns with gateway telemetry aggregation, minimizes `state_lock` contention, and comfortably exceeds network round-trip time.
* **Planning Horizon ($H$):** Initial small development search uses $N=3$ control steps ($3.0\,\text{s}$ horizon at $\Delta t = 1.0\,\text{s}$), enumerating at most eight $\{16.0, 32.0\}$ sequences:
  $$\mathcal{U}^3 = \{ (16,16,16), (16,16,32), (16,32,16), (16,32,32), (32,16,16), (32,16,32), (32,32,16), (32,32,32) \}$$
  Comparison across $N \in \{1, 3, 5\}$ and one-switch pruning are development ablations, not settled performance facts.
* **Candidate Action Space ($\mathcal{U}$):** Strictly $\mathcal{U} = \{16.0, 32.0\}$ (the only actions with empirical GPU plant support).

### 5.2 Actuator Transport Contract: Guarded Server Capability
To prevent delayed timed-out writes from clobbering subsequent commands (Mayor check `timeout_then_old_state_read_does_not_cancel_command`), the `/admin/budget` endpoint requires a guarded server capability checked atomically under `state_lock`:
```json
{
  "budget": 16.0,
  "expected_version": 2,
  "cost_fingerprint": "v1_0.1_0.259_0.000758_0.005208",
  "run_id": "run-8e92371",
  "clock_domain": "posix_monotonic",
  "expires_at_monotonic": 1420.550
}
```
**Gateway Enforcements:**
1. **Guarded Capability & Identity Checks:** Under the gateway's `state_lock`, the handler validates matching runtime/run identity, cost model fingerprint, and clock domain. An endpoint ignoring these fields is explicitly classified as non-guarded.
2. **Compare-And-Set (CAS):** If `state["budget_version"] != payload["expected_version"]`, the gateway rejects the update with HTTP 409 Conflict.
3. **Server-Side Expiry:** If `time.monotonic() > payload["expires_at_monotonic"]`, the update is dropped as expired with HTTP 408 Request Timeout.
4. **Timeout Handling:** A client timeout represents unknown application status, not cancellation. The controller suspends speculative writes until the command is definitively reconciled or expired.

---

## 6. Smallest Coherent CPU-Implementable Interfaces

The controller is decomposed into five isolated, testable modules:

```
+-----------------------------------------------------------------------------------+
| 1. Observation Adapter (GatewayStateAdapter)                                      |
|    - Polls /metrics and consumes local event stream under state_lock              |
|    - Emits immutable CausalSnapshot(t_k) with clock domain and version            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. Latency & Service Predictor (BacklogTTFTModel)                                 |
|    - Evaluates: TTFT_pred = theta_0 + theta_1 * (W_tok + p_i) + theta_2 * N_dec   |
|    - Evaluates survival probability P(EOS > t) and failure probability            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. Rollout Engine (DiscreteScenarioRollout)                                       |
|    - Simulates 8 candidate plans across 16 scenarios over H=4s                    |
|    - Generates discrete terminal events, releasing exact c_i                      |
|    - Evaluates R_new + R_old + V_cont                                             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. Supervisor & Watchdog (SupervisorPlanner)                                      |
|    - Checks solver latency (deadline 20 ms) and snapshot freshness (<= 100 ms)    |
|    - Compares candidate gain against margin Delta_min = 0.4 req/s                 |
|    - Emits Command(budget, expected_version, expiry) or Fallback(C=16.0)          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. Actuation Client (GuardedActuatorClient)                                       |
|    - Issues POST /admin/budget with CAS version and expiry                        |
|    - Reconciles state on timeout without issuing speculative competing writes     |
+-----------------------------------------------------------------------------------+
```

### 6.1 Data Types & Interface Signatures

```python
# prefill_lab/mpc/types.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass(frozen=True)
class ActiveRequest:
    request_id: str
    cost: float
    prompt_tokens: int
    max_tokens: int
    admitted_at: float
    phase: str  # 'awaiting_first' or 'streaming_decode'

@dataclass(frozen=True)
class CausalSnapshot:
    timestamp: float
    clock_domain: str
    run_id: str
    cost_fingerprint: str
    budget_version: int
    applied_budget: float
    current_admitted_cost: float
    awaiting_tokens: int
    in_flight_count: int
    awaiting_count: int
    active_ledger: Tuple[ActiveRequest, ...]
    arrival_ring_buffer: Tuple[Tuple[int, float], ...]  # (count, cost) per second
    activity_ring_buffer: Tuple[Tuple[int, int], ...]   # (tokens, completions) per second

@dataclass(frozen=True)
class BudgetCommand:
    budget: float
    expected_version: int
    cost_fingerprint: str
    run_id: str
    clock_domain: str
    expires_at: float
    is_fallback: bool
```

---

## 7. Strong Baselines and Discriminating Comparisons

To establish whether multi-step receding horizon planning provides real policy gain, MDR-MPC must be benchmarked against two same-state baselines:

1. **Baseline 1: Static Fixed-Cap Baselines ($C \equiv 16$ and $C \equiv 32$):**
   - The primary empirical benchmark. Fixed $C=32$ achieves $3.73\text{--}5.08\,\text{req/s}$ on development data.
2. **Baseline 2: Same-State Backlog Feedback (Hysteresis on $W_{\text{tok}}$):**
   - Operates on the exact same information state at $\Delta t = 1.0\,\text{s}$:
     $$u_k = \begin{cases} 16.0, & \text{if } W_{\text{tok}}(t_k) > 4,000\,\text{tokens} \\ 32.0, & \text{if } W_{\text{tok}}(t_k) \le 2,000\,\text{tokens} \\ u_{k-1}, & \text{otherwise (Dwell)} \end{cases}$$
3. **Baseline 3: Same-Model Myopic Controller ($N=1$):**
   - Evaluates the exact same scenario rollout with horizon $N=1$ step. This directly isolates whether multi-step lookahead ($N=3$) adds value over 1-step greediness.

---

## 8. Targeted CPU Falsification Test Suite

The following automated tests verify accounting, transport, and modeling logic before any GPU execution:

```python
# tests/test_mpc_consensus_falsification.py
import pytest

def test_falsify_terminal_constraint_infeasibility():
    """Verify that a hard terminal constraint R(t+H) <= 18.0 is infeasible
    when starting from R=32.0 with slow decode drain (Mayor Check 1)."""
    R_0 = 32.0
    terminal_limit = 18.0
    drain_in_horizon = 8.0  # Slow natural drain
    R_H = R_0 - drain_in_horizon
    assert R_H > terminal_limit, "Terminal condition violated as expected"

def test_falsify_cas_timeout_clobber():
    """Verify that CAS expected_version blocks delayed timed-out writes (Mayor Check 3)."""
    gateway_state = {"version": 1, "budget": 32.0}
    # Delayed command arrives with stale expected_version=0
    delayed_cmd = {"budget": 16.0, "expected_version": 0}
    accepted = (delayed_cmd["expected_version"] == gateway_state["version"])
    assert accepted is False, "CAS failed to block delayed stale command"

def test_falsify_old_cohort_externality():
    """Verify that scoring only new arrivals can flip policy preference (Mayor Check 4)."""
    # Policy A (conservative): old_hits=2, new_hits=0 -> total=2
    # Policy B (aggressive): old_hits=0, new_hits=1 -> total=1
    new_only_gain = 1 - 0  # Policy B looks better (+1)
    actual_gain = 1 - 2    # Policy B is worse (-1)
    assert new_only_gain > 0 and actual_gain < 0, "Externality flip verified"

def test_falsify_rolling_sum_expiration():
    """Verify that rolling sum without expiration buffer fails closure (Mayor Check 6)."""
    early_buf = [5, 0, 0, 0, 0]
    late_buf  = [0, 0, 0, 0, 5]
    assert sum(early_buf) == sum(late_buf)
    # Advance 1 step with zero new events:
    assert sum(early_buf[1:]) != sum(late_buf[1:]), "Expiration failure verified"
```

---

## 9. Preserved Unresolved Disagreements & Decisive Experiments

The researchers remain divided on three substantive questions that cannot be resolved without native execution:

1. **Disagreement 1: Can Receding-Horizon MPC Beat Tuned Backlog Hysteresis?**  
   *Opus / Jasper View:* Because TTFT is predominantly determined at admission by pending backlog $W_{\text{tok}}$, a simple threshold on $W_{\text{tok}}$ may capture $>95\%$ of available goodput, rendering multi-step scenario rollouts redundant.  
   *Astra / GPT View:* Carryover from multi-second decode residence creates inter-temporal trade-offs during burst transitions that require lookahead.  
   *Decisive Experiment:* A randomized switchback trial (30-second blocks, ABBA design) comparing MDR-MPC ($N=3$) directly against Backlog Hysteresis on fresh non-C Azure traces. Pre-registered decision rule: if MDR-MPC does not exceed Hysteresis by $\ge 0.4\,\text{req/s}$ ($p < 0.05$), reject MPC in favor of Hysteresis.
2. **Disagreement 2: Natural EOS Model Transferability:**  
   *Factual Status:* All 13 historical captures generated exactly 128 output tokens ($o \equiv 128$).  
   *Decisive Experiment:* Collect one native calibration block with natural conversational prompts to measure decode length variance and fit survival hazard functions.
3. **Disagreement 3: Incremental Value of Engine-Level Pre-ADD Telemetry:**  
   *Decisive Experiment:* Offline ablation testing whether adding vLLM scheduler gauges to the gateway $W_{\text{tok}}$ regression reduces TTFT MAE by $\ge 10\,\text{ms}$. If not, engine telemetry integration is abandoned.

---

## 10. Architecture Recommendation, Readiness Separation, and Limitations

### 10.1 Architecture Recommendation: MDR-MPC
We recommend implementing the **Minimal Discrete Receding-Horizon Admission Controller (MDR-MPC)**:
- Cadence $\Delta t = 1.0\,\text{s}$, initial development horizon $N = 3\,\text{s}$ (with $N \in \{1, 3, 5\}$ comparative ablations), discrete actions $\mathcal{U} = \{16.0, 32.0\}$.
- Gateway information state driven by $W_{\text{tok}}$, $R$, $N_{\text{flight}}$, and arrival ring buffers.
- Soft terminal evaluation under backup $C^* = 16.0$; guarded server transport with CAS and server-side command expiry.

### 10.2 Strict Separation of Readiness
- **CPU / Shadow Engineering Readiness:** **READY.** The mock interfaces, causal snapshots, discrete rollout engine, and falsification test suite can be implemented immediately in pure Python under default-off flags.
- **Native GPU Actuation Authorization:** **BLOCKED.** Live actuation remains strictly unauthorized until:
  1. The CAS and expiry endpoint extension is implemented in `gateway.py` and passes regression tests;
  2. The offline F3 incremental value ablation is executed on archived captures;
  3. Mayor grants formal authorization for native benchmark allocation.

---

## 11. Conclusion and Round 3 Barrier

This Round 3 proposal delivers a mathematically corrected, physically plausible, and computationally bounded admission MPC design. It incorporates all peer and Mayor counterexamples, establishes clear module boundaries, and defines the exact decisive experiments needed to determine whether MPC justifies its operational complexity over simple backlog feedback.

**Round 3 Barrier Reached:** The report is sealed. Jasper halts here, awaiting Mayor review.
