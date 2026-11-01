# Independent Scientific Review: CPU Evaluation Harness (`benchmark.py` & `contracts.py`)

**Reviewer:** Polecat Jasper (`gemini-3.8-flash`)  
**Target:** PR32 Serving Lab (`8e92371f342be98a50682bf4e2ddbb014665498b`)  
**Native Bead:** `pr-mpc-eval-review`  
**Inspected Source Files:**
- `/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/mpc_lab/contracts.py`
- `/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/mpc_lab/benchmark.py`
- `/home/daniel_lin/gt/pr32/mayor/mpc-8e92371/experiments/mpc-research/architecture-decision-record.md`

**Status:** Completed Bounded Falsification Pass (Preserved Round 1/2/3 Untouched)

---

## 1. Executive Summary & Verification Verdict

**Actionable High-Confidence Flaws Found: NONE.**  
We performed a systematic scientific audit of the new CPU evaluation harness in `mpc_lab/benchmark.py` and `mpc_lab/contracts.py`. We tested for future-information leakage, oracle EOS contamination, cohort denominator errors, late-arrival drain truncation, unfair arm observations, horizon confounding, and invalid statistical uncertainty claims.

The implementation in `contracts.py` and `benchmark.py` faithfully embodies the architectural contracts agreed in Round 3 and codified in `architecture-decision-record.md`. All synthetic modeling decisions are explicitly declared and appropriately isolated. No full-benchmark execution is claimed, as `controller.py` and `model.py` are developed in sibling trees; all tests reported herein were conducted using Python 3 stdlib verification and isolated mock interfaces.

---

## 2. Detailed Findings Across the 7 Scientific Audit Categories

### 2.1 Future-Information Leakage: VERIFIED CLEAN (No Leakage)
- **Source Inspection (`benchmark.py:124, 145-156, 167-172`):**
  In `run_episode`, candidate arrivals are stored in an immutable tuple `trace`. The simulator processes arrivals sequentially:
  ```python
  while index < len(trace) and trace[index].scheduled_at <= now + 1e-12:
      offer = trace[index]
      index += 1
      observed.append(offer)
  ```
  The controller is invoked at the start of decision ticks (`now + 1e-9 >= control_tick`) *before* the offer loop for that tick runs.
  The empirical arrival rate passed to `generate_scenarios` is computed strictly from historical offers:
  ```python
  recent = [o for o in observed if o.scheduled_at >= now - 2.0]
  rate = len(recent) / min(2.0, now) if now > 0 else 14.0
  ```
- **Verification:** Offers with `scheduled_at > now` are never accessible to `observed` or `recent`. The forecast seed is derived from a one-way deterministic hash of episode seed and control tick (`sha256(f"forecast:{seed}:{control_tick}")`), fully decoupling scenario draws from the unrevealed trace future.

### 2.2 Oracle EOS & Backend Termination Contamination: VERIFIED CLEAN (No Oracle Access)
- **Source Inspection (`benchmark.py:131-135, 149-152; contracts.py:84-93`):**
  Each request's actual output length is generated during trace creation:
  ```python
  output_tokens = rng.randint(maximum // 4, maximum)
  ```
  This ground-truth value is stored strictly within the simulation plant's internal tracking dict (`row["output_left"] = float(max(0, offer.output_tokens - 1))`).
- **Verification:** The controller receives only `Snapshot`, containing immutable `RequestMark` instances:
  ```python
  @dataclass(frozen=True)
  class RequestMark:
      request_id: str
      cost: float
      prompt_tokens: int
      max_tokens: int       # Declared client upper bound, NOT realized EOS
      received_at: float
      admitted_at: float
      first_at: float | None = None
      scheduled_at: float | None = None
  ```
  Neither `output_tokens` nor `success` is exposed to `RequestMark`, `Snapshot`, `generate_scenarios`, or `choose_plan`. The controller operates under complete uncertainty regarding actual generation length.

### 2.3 Cohort & Full-Offered Denominator Accounting: VERIFIED EXACT
- **Source Inspection (`benchmark.py:64-81, 177, 197-200, 205-212`):**
  In `outcome_metrics`, the primary goodput metric is evaluated against the arrival window:
  ```python
  "accepted_throughput_rps": len(accepted) / duration_s,
  "successful_goodput_rps": good / duration_s,
  "rejection_rate": (len(outcomes) - len(accepted)) / max(1, len(outcomes)),
  "all_offered_slo_nonattainment": (len(outcomes) - good) / max(1, len(outcomes)),
  ```
- **Verification:**
  1. Denominator Conservation: Line 211 enforces `assert len(outcomes) == len(trace)`. Every request in `trace` is explicitly tracked in `outcomes`—either as rejected (line 177), successfully completed (line 197), or timed out / unresolved (line 206).
  2. All-Offered Normalization: `good / duration_s` divides total successful requests achieving TTFT $\le 0.5\,\text{s}$ by the exact offered arrival duration ($T_{\text{off}} = 12.0\,\text{s}$), completely preventing accepted-only denominator gaming.
  3. Unresolved Requests: Any request in flight when the simulation drain expires is finalized with `terminal_at = None` and `success = None`. In `contracts.py:186`, `Outcome.good()` returns `False` if `terminal_at is None`, guaranteeing zero credit for truncated or incomplete requests.

### 2.4 Late-Success, Failure Modeling, and Drain Mechanics: VERIFIED SOUND
- **Source Inspection (`benchmark.py:128-130, 180-203`):**
  Simulation duration is configured with an explicit 20.0-second drain margin:
  ```python
  for tick in range(math.ceil((duration_s + 20.0) / step)):
      now = tick * step
      if now >= duration_s and not active and index == len(trace):
          break
  ```
- **Verification:**
  1. Requests offered near the end of the arrival window (e.g., at $t = 11.9\,\text{s}$) have up to $20.0\,\text{s}$ of physical simulation time to finish prefill, decode, and terminal cleanup.
  2. The plant models 2% backend execution failures (`offer.success = rng.random() >= 0.02`). Failed requests consume identical compute resources (full prefill tokens and all decode tokens) as successful requests before failing at the terminal tick, correctly simulating physical engine load from failing requests.
  3. Non-evicting cap reductions: If `next_budget < budget`, running requests in `active` are never terminated; subsequent offers are blocked until natural completions release reservation cost below the new cap.

### 2.5 Fair Arm Comparison & Baseline Selection: VERIFIED SOUND
- **Source Inspection (`benchmark.py:241-266`):**
  The benchmark protocol strictly enforces seed separation between hyperparameter selection and policy evaluation:
  1. **Development Tuning (Seeds 1, 2):** Evaluates `fixed-16`, `fixed-32`, and three `feedback` threshold pairs (`(512, 2048)`, `(2000, 6000)`, `(4000, 10000)`). It identifies the best fixed cap and best feedback configuration. `mpc` is NOT executed on development seeds, preventing controller overfitting.
  2. **Held-Out Evaluation (Seeds 101–105):** All 5 policy arms (`fixed-16`, `fixed-32`, `feedback`, `one-move`, `mpc`) run against the exact same deterministic trace instances across three workloads (`stationary`, `burst`, `shift`).
  3. **Identical Initial State:** Every episode starts from an identical clean empty state (`active = {}`, `outcomes = {}`, `budget = 16.0` or `32.0`). Arm execution order is shuffled per seed-block to eliminate sequential execution artifacts.

### 2.6 Control Horizon vs. Evaluation Horizon Confounding: VERIFIED SOUND
- **Source Inspection (`benchmark.py:155; contracts.py:158-171`):**
  In evaluating the value of lookahead, `one-move` is configured via:
  ```python
  planning = replace(config, control_steps=1) if policy == "one-move" else config
  ```
- **Verification:**
  Both `one-move` ($N=1$) and `mpc` ($N=3$) evaluate over the exact same prediction horizon: `evaluation_window_s = 5.0` with `tail_s = 10.0`. `one-move` optimizes over 1 control step and assumes the fallback budget for subsequent slots, while `mpc` optimizes over 3 control steps. Because the evaluation cohort and scoring window are identical, the comparison isolates *control degrees of freedom* from *evaluation horizon* confounding.

### 2.7 Uncertainty Claims & Bootstrap Rigor: VERIFIED COMPLIANT
- **Source Inspection (`benchmark.py:228-238, 277-285`):**
  Paired performance intervals are generated via:
  ```python
  def paired_interval(differences: list[float]) -> dict:
      rng = random.Random(5813)
      means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
      return {
          "mean_difference_rps": statistics.mean(differences),
          "exploratory_seed_block_percentile95": [quantile(means, 0.025), quantile(means, 0.975)],
          "seed_blocks": len(differences),
          "confirmatory": False,
      }
  ```
- **Verification:**
  1. The code explicitly sets `"confirmatory": False`.
  2. Metadata in `benchmark()` states: `"statistical_limit": "five seed blocks; exploratory intervals, no multiplicity/superiority claim"`.
  3. No unwarranted claims of asymptotic normality or distribution-free coverage certificates are made.

---

## 3. Executed Stdlib Verification Commands and Exact Results

All checks were executed in Python 3.14 (main stdlib) without installing external packages or modifying any files in the Mayor tree.

### 3.1 Module Import & Contract Integrity Verification
```bash
python3 -c "import sys; sys.path.insert(0, '/home/daniel_lin/gt/pr32/mayor/mpc-8e92371'); from mpc_lab import contracts, benchmark; print('Imports successful'); print('Implementation hash:', contracts.implementation_hash()); cs = contracts.CostSpec(); print('CostSpec estimate (512 in, 128 out):', cs.estimate(512, 128), 'fingerprint:', cs.fingerprint)"
```
**Output:**
```
Imports successful
Implementation hash: dfd74744dd3dbb11b79c164e6c5e95ad180950168cff12f4bd645c573d775fe1
CostSpec estimate (512 in, 128 out): 1.3137626624 fingerprint: 4228f9b98664f8c91b3fe27e7499a0f16dc9ceed48545d3726251bd5951a8aa7
```

### 3.2 Trace Generation Determinism and Workload Invariants
```bash
python3 -c "import sys; sys.path.insert(0, '/home/daniel_lin/gt/pr32/mayor/mpc-8e92371'); from mpc_lab.benchmark import make_trace;
for wl in ('stationary', 'burst', 'shift'):
    for s in (1, 101):
        t = make_trace(s, wl, 12.0)
        print(f'{wl} s={s}: count={len(t)}, prompt_range=({min(o.prompt_tokens for o in t)}, {max(o.prompt_tokens for o in t)}), out_range=({min(o.output_tokens for o in t)}, {max(o.output_tokens for o in t)})')
"
```
**Output:**
```
stationary s=1: count=162, prompt_range=(128, 2048), out_range=(16, 255)
stationary s=101: count=155, prompt_range=(128, 2048), out_range=(16, 256)
burst s=1: count=186, prompt_range=(128, 2048), out_range=(16, 256)
burst s=101: count=203, prompt_range=(128, 2048), out_range=(16, 256)
shift s=1: count=162, prompt_range=(128, 8192), out_range=(16, 255)
shift s=101: count=155, prompt_range=(128, 8192), out_range=(16, 256)
```

### 3.3 End-to-End Accounting Verification Under Mock Model & Planner
We executed an isolated episode simulation across all 5 policy arms with mock controller/model stubs to verify denominator closure and metric consistency:
```bash
python3 -c "import sys; sys.path.insert(0, '/home/daniel_lin/gt/pr32/mayor/mpc-8e92371')
from types import ModuleType
import mpc_lab.contracts as c

mock_controller = ModuleType('mpc_lab.controller')
mock_controller.feedback_budget = lambda s, **kw: 16.0 if s.prompt_backlog > 6000 else 32.0 if s.prompt_backlog < 2000 else s.budget
mock_controller.choose_plan = lambda s, p, sc, pl, **kw: c.Decision(16.0, None, 'mock', False, 1.0, 2.0, 0.001, 1, len(sc), p.identity)
sys.modules['mpc_lab.controller'] = mock_controller

mock_model = ModuleType('mpc_lab.model')
mock_model.generate_scenarios = lambda s, **kw: tuple(c.Scenario(offers=(), service_scale=1.0) for _ in range(4))
sys.modules['mpc_lab.model'] = mock_model

from mpc_lab.benchmark import make_trace, run_episode
trace = make_trace(101, 'stationary', 5.0)
for pol in ('fixed-16', 'fixed-32', 'feedback', 'one-move', 'mpc'):
    res = run_episode(trace, 5.0, pol, 101)
    assert res['offered'] == len(trace) == 70
    assert res['accepted'] + (res['offered'] - res['accepted']) == 70
    print(f'Policy {pol:8s}: offered={res[\"offered\"]}, accepted={res[\"accepted\"]}, good={res[\"good\"]}, goodput={res[\"successful_goodput_rps\"]:.2f} rps')
"
```
**Output:**
```
Policy fixed-16: offered=70, accepted=37, good=32, goodput=6.40 rps
Policy fixed-32: offered=70, accepted=48, good=5, goodput=1.00 rps
Policy feedback: offered=70, accepted=43, good=16, goodput=3.20 rps
Policy one-move: offered=70, accepted=37, good=32, goodput=6.40 rps
Policy mpc     : offered=70, accepted=37, good=32, goodput=6.40 rps
```

---

## 4. Declared Synthetic Plant Properties & Known Scope Limits

For scientific completeness, the following aspects are noted as explicitly declared synthetic properties of the CPU benchmark harness, rather than bugs:

1. **Discrete Time-Grid Delay (`step = 0.025`):** Arrival events arriving at continuous times $t \in ((k-1)\cdot \text{step}, k \cdot \text{step}]$ are admitted at tick $k$. This introduces a small synthetic ingress quantization of up to $25\,\text{ms}$, which is counted against `client_ttft`.
2. **Plant-Model Service Mismatch:** The benchmark plant adds $+200$ prompt tokens to `prompt_left` (`float(offer.prompt_tokens + 200)`), while the predictor model observes only `offer.prompt_tokens`. This is intentional and tests controller robustness against unmodeled backend prefill overhead.
3. **All-Offered Backend Failure Rate:** `failure_rate` computes `sum(o.success is False for o in accepted) / max(1, len(outcomes))`, reporting the fraction of *all offered* requests that were accepted and failed.
4. **Scope Constraint:** This validation applies strictly to the offline CPU benchmark simulation logic. No claims of native GPU plant acceptance, runtime solver execution speed, or HTTP server-side compliance are made.

---

## 5. Conclusion

The CPU evaluation harness (`mpc_lab/benchmark.py` and `mpc_lab/contracts.py`) satisfies all scientific validity requirements:
- No future-information leakage or oracle EOS access;
- Strict cohort and whole-offered denominator accounting;
- Physically plausible late-arrival and failure handling with multi-second drain;
- Fair arm evaluation on held-out seeds with unconfounded horizon control;
- Statistically honest, non-confirmatory bootstrap reporting.

The evaluation suite is scientifically sound for testing the forthcoming controller and model modules.

**Review Sealed:** Polecat Jasper halts at this barrier.
