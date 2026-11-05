# resume-p0-contract01: P0 MVP metric/baseline contract (proposal)

Role7 (fair existing benchmark metrics/negative cases), claude-opus-5.5 / max.
CONTRACT PROPOSAL ONLY: no implementation, tests, runtime, Native, GPU, SSH,
network, git or measurement. It produces no P0 values and closes no gate
(G06/G16/Native14/D2-F1 unchanged). No superiority, production or SLO-guarantee
claim. Machine-readable twin: `p0-contract.json`.

## 0. Pinned inputs (8 files, 158,206 B; full hashes in PINS-*.txt)

| Pin | File | Role |
|---|---|---|
| a3e4a33d | results/.../native-team16-send-order-author01/comparison16.py | current corrected scorer (Mayor: limited-source+CPU accepted fb884d3e) |
| 9a173ab9 | experiments/mpc-research/native-r23/score_http.py | SLO-good rule + all-offered score imported by a3e4 |
| 164f3947 | results/.../native-team16-planner-revision02-author01/policy_planner16.py | Fixed/OneMove/MPC path definitions (E2, 2b4ed5bc) |
| 5f889665 | results/.../native-team16-comparison-author01/native-gpu-protocol16.json | preregistered protocol, eligible/launch false |
| 415737cd | docs/benchmark_results.md | P0 required table + metric requirements (b1b61d93) |
| be89d12b | docs/architecture.md | admission/actuator semantics |
| c84858d2 | mpc_lab/benchmark.py | public runnable CPU baseline interface |
| eadbfe17 | mpc_lab/model.py | public CPU plant queue semantics |

Selection discovery: names/sizes and keyword-count grep only. Tracked status
and commits come from Mayor messages; Role7 did not verify them with git. The
a3e4 docstring still says "not accepted", which is stale relative to Mayor
fb884d3e. Not pinned and not read: gateway.py, mpc_lab/contracts.py
(`Outcome.good`) and mpc_lab/controller.py.

## 1. Lineages (never pool or co-tabulate)

- **L-T16:** comparison16 a3e4 + score_http + planner rev02. This is a paired,
  all-offered validator/scorer of caller-produced episode journals.
  - Fixtures only today.
  - On a non-synthetic world it refuses model arms with status ok (a3e4:730).
  - Native rows can therefore only ever be non-model arms, and only after gates.
- **L-PUB:** mpc_lab/benchmark.py `run_episode`. This is a runnable synthetic CPU
  plant (exploratory; plant differs from the forecast model). It uses a
  different estimator (linear-interpolation quantile, benchmark.py:56-62) and an
  unpinned `good()` rule.

Every P0 row must carry `{lineage, basis}`, where basis is `synthetic_fixture`,
`observed_research` or `cpu_synthetic`.

## 2. Clock origins

All values below are in one `endpoint_clock_domain`.

| Clock | Definition |
|---|---|
| `scheduled_at` | Equals offered_start + frozen offset_s (a3e4:679). |
| `sent_at` | Actual client send, or null meaning `not_launched`. Must satisfy scheduled ≤ sent ≤ observed (a3e4:683). |
| `first_content_at`, `terminal_at` | Client-observed. Must be ≥ sent_at, otherwise the case is refused (a3e4:755). Terminal must satisfy scheduled ≤ terminal ≤ fate_observed_at (a3e4:713). |
| Windows | Offered window [offered_start, +duration_s); cut = offered_start + cut_offset_s; accounting envelope [started_at, finished_at] ⊇ offered window (score_http:41-48). |
| Costs | Native costs must use the endpoint domain inside the envelope (a3e4:748). Fixture costs are not convertible. |

## 3. Metrics

Percentiles use nearest-rank `ordered[ceil(q*n/100)-1]` (a3e4:775). Every
percentile reports n. Flags: `p95_is_sample_max` when n ≤ 19, and
`p99_is_sample_max` when n ≤ 99. Arm-level values pool request samples across
cases; percentiles are never averaged across cases.

### M1 TTFT

| ID | Definition | Status |
|---|---|---|
| M1a `ttft_sched_cond` | first_content_at − scheduled_at over status==complete rows with finite first. | Existing (a3e4:782-803); selection-biased; not SLO credit. |
| M1b `ttft_send_cond` | Same rows, first_content_at − sent_at, joined to offers by position and id (make_case enforces identical order). | New |
| M1c `ttft_sched_all_offered_censored` | Every offered row: first − scheduled when a finite first was observed (any fate), else +∞. If the rank lands on +∞, value = null and `censored=true`. | New; selection-robust companion. |

E2E (`e2e_from_scheduled_s`) stays as an existing descriptive.

### M2 All-offered SLO goodput (unchanged)

The score_http rule `r23-natural-endpoint-slo-v1` (score_http:7-38) requires all of:
- TTFT ≤ 0.5 s from scheduled_at;
- multi-token mean output gap ≤ 0.05 s;
- for one-token outputs, e2e ≤ 0.5 s;
- complete, stop, ignore_eos=false, stream_done, no stream_error.

Reported values:
- **Primary:** `natural_slo_good / all_offered`. This is null when all_offered = 0.
- `cohort_slo_goodput_rps` = good / offered_window_s.
- Existing in-window and policy-and-tail variants.

The E1 historical "SLO goodput" was TTFT-only and is not comparable, so do not
mix it in.

### M3 Throughput (new)

- `delivered` = status ∈ {complete, length}. Status is authoritative and rows
  are never relabelled.
- Request throughput = delivered / D, for each of three windows (mirroring
  score_http):
  - **Primary D:** offered_window_s. All cohort completions, including those in the tail.
  - In-window: terminal ≤ window_end. Missing terminals are counted, not dropped.
  - Policy-and-tail envelope: includes setup, warmup and drain.
- Output-token throughput = Σ usage.completion_tokens (int ≥ 0) over delivered
  rows / the same D.
  - Missing usage is excluded, counted, and flags the value as `lower_bound`.
  - Never substitute max_tokens, HTTP-record counts or SSE chunk counts.
  - Tokens reported on undelivered rows are listed separately.
- Context value: `offered_rps` = all_offered / offered_window_s.

### M4 Fates (counts exist; add rates)

Report all 9 FATES (a3e4:50) as counts and as rates over all_offered (null if 0).

| Rate | Definition |
|---|---|
| `rejection_rate` | rejected / all_offered (gateway admission refusals only). |
| `failure_rate` | (timeout + error + failed + cancelled) / all_offered, plus each component. |
| `not_launched` | Infrastructure non-send. Never merged into rejection, never replaced (G12 pending). |
| `pending` | Right-censored unknown, zero credit. |
| `length` | Capped: delivered, but no SLO credit. |
| `complete_not_slo_good` | Count of complete rows minus natural_slo_good. |

Policy-episode statuses and unknown application/cleanup records are retained.

### M5 Group fairness (new)

**Group key.** A pure function of frozen request fields; no workload schema
change (the a3e4:153 schema has no group field).
- Default (Mayor choice): prompt_tokens classes S ≤ 512 < M ≤ 2048 < L.
- Frozen as a code constant before outcomes and recorded in the identity bundle.
- Rationale: admission R + c ≤ u prices requests by size.

**Per group:** n_offered, good, attainment, rejection rate, failure rate,
not_launched, pending, and M1a/M1c percentiles with n.

**Summary over nonempty groups** (k ≥ 2, otherwise null with reason):
- `worst_group_attainment` = min.
- `gap` = max − min.
- `min_over_max`: null if max = 0.
- Unweighted Jain index = (Σa)² / (k·Σa²): null when all a = 0 (never 1).
- `worst_group_rejection`.
- `worst_group_ttft_p95`, reported with its n.

Paired per-case contrasts vs train_selected_fixed are descriptive only. There is
no starvation guarantee, and absolute group latency is confounded by size.

### M6 End-to-end controller overhead

Already available:
- From account_costs (a3e4:379-406): envelope, covered union, and nonadditive
  category unions (setup/acquisition/export/ipc/solver/control/journal/cleanup).
- Category mapping: observation → acquisition/export; model and optimization → solver.

New:
- `union_fraction_of_endpoint_envelope`, only when the costs domain equals the
  endpoint domain; otherwise null.
- Per-decision values, each with n, p50/95/99 and max:
  - model ok tick: decision_at − plan.source_at;
  - model ok tick: solver_elapsed_s;
  - reactive tick: decision_at − observation.observed_at;
  - per applied change: `actuation_lag_s` (existing, a3e4:522) and applied_at − source_at.
- Tick counts by exact status and reason. Missed deadlines appear only as
  refused/failed ticks, with no inferred classification.

Rules:
- Never add, subtract or convert overhead into TTFT, E2E or goodput.
- The frozen schedule already stops a policy from slowing the workload.
- Fixed arms have no ticks: delay n = 0 (null), and costs remain explicit.
- L-PUB `controller_wall_s` (benchmark.py:162,235) is decision compute only, so
  it is insufficient as end-to-end overhead.

### M7 Counts and design (unchanged)

- n_offered ≤ 1024 per case.
- Complete arms × cases, paired exposure and Williams order (a3e4:854-896).
- Protocol16 minimums: S1 ≥ 16 Train episodes and S2 ≥ 36 evaluation episodes,
  totalling 52 > 48 pilot cap. This remains unresolved.
- No inference: decision.winner is None (a3e4:994); G07/G08 are absent.

### M8 Missing, zero and censored rules

| Situation | Rule |
|---|---|
| 0 samples | Percentile null. |
| all_offered = 0 | Rates null, never 0 or 1. |
| Zero completions | Throughput 0.0 (duration > 0 is enforced). |
| Complete row without finite first | Excluded from M1a/M1b and counted; +∞ in M1c. |
| `not_launched` | Absent from M1b; +∞ in M1c. |
| `pending` | Zero credit; +∞ in M1c unless first content was observed. |
| Empty group | Listed with n = 0 and excluded from summaries. |
| Inputs | Validated via a3e4 `validate_case`/`compare_campaign` first. A refused campaign yields no metrics, only the retained refusal. |

## 4. Configuration fidelity

**Identity bundle on every row:**
- commit, argv, config sha, Python version;
- scorer/score_http/planner/P0-module shas;
- panel, Train-choice, world (21 keys) and workload shas;
- seed, horizon, objective, scenario shas;
- model sha and qualification;
- initial, fixed and reactive parameters;
- budget support {16, 32}, SOURCE_AGE_LIMIT 0.5, SLO rule schema, group edges;
- order and actual sequence.

**Arm equality is already enforced:**
- paired_exposure/validate_runs;
- the model-arm panel binding;
- fixed-budget binding;
- the Train-only frozen choice on a disjoint evaluation set.

**Fidelity gap (low severity).** comparison16 takes solver_budget_s and
actuation_budget_s from each plan (a3e4:465-468) and never binds them across
ticks or arms. Complete enumeration keeps path selection unaffected, but refusal
incidence could differ between arms. The P0 module must require one identical
planner contract across strong_one_move and mpc (budgets, offsets, solver and
actuation budgets, cost spec), and refuse otherwise.

**Strong baselines stay unchanged:**

| Arm | Definition |
|---|---|
| Fixed | Best constant path. |
| OneMove | Best path with ≤ 1 transition from the received budget, delayed transitions included. |
| MPC | Best multistep path, first action only. |

Supporting sources: shared exact ranking (planner:111-117,203-214; a3e4:453);
Train-only fixed selection (planner:268-324; a3e4:913-934). L-PUB one-move uses
`control_steps=1` (benchmark.py:155); its delayed-transition strength must be
verified in the unpinned controller.py before it is called strong.

## 5. FIFO decision: ALIAS of fixed32, not an independent algorithm

**Evidence:**
- L-T16 arms differ only by an applied budget timeline in {16, 32} (a3e4:476-491, 617-642).
- No ordering parameter exists.
- L-PUB admits on arrival if R + c ≤ u and otherwise rejects immediately
  (benchmark.py:175-182). There is no gateway wait queue.
- Admitted prefill is served in admission order (benchmark.py:188-196; model.py:7,22).
- A lower budget never evicts (architecture.md:50-53).

**Conclusion.** FCFS is the shared service discipline of every arm. A
non-predictive FIFO under the same hard resource, queue and workload limits
holds the largest supported budget with FCFS. That is exactly fixed32 (L-T16)
or fixed-32 (L-PUB).

**How the FIFO row is reported:**
- Same values, with `alias_of: fixed32`, `independent_algorithm: false`, and no
  extra episodes, Williams slot or comparison-family member.
- If Train selects 32, train_selected_fixed is also the same algorithm (its
  separately run episodes are already preregistered); disclose this.

**Rejected alternatives:**
- Unthrottled FCFS breaks the shared {16, 32} support. Budget units are a
  weighted reservation cost (cost_semantics world key not frozen), so 32 must
  not be described as "32 sequences".
- A gateway FIFO wait queue is a new admission mechanism with unequal queue
  limits, so it is not P0.

## 6. Minimum code changes (later, own frozen P0 scope; none made now)

- **C1:** a new additive `p0_metrics16.py` that runs post-validation on a3e4
  outputs.
  - Reuse nearest_rank, conditional_latency, account_costs and score_http unchanged.
  - Add M1b, M1c, M3, M4 rates, M5, the M6 additions, the FIFO alias row, the
    identity bundle, refusal passthrough and the planner-contract equality check.
- **C2:** only if a CPU table is chosen. Additive extensions to L-PUB
  outcome_metrics:
  - delivered req/s;
  - tokens/s from the plant's output_tokens, labelled synthetic;
  - fairness using the same edges;
  - nearest-rank (or a disclosed interpolation);
  - the fixed-32 alias;
  - never present `accepted_throughput_rps` (admissions/s) as throughput;
  - verify one-move strength.
- **C3:** a table renderer with lineage, basis, alias and refusal flags.
  - Native model rows read "prohibited (Native14 FAIL)".
  - No winner field.
- **Unchanged:** a3e4, score_http, planner rev02, protocol16, ARMS, Williams
  order, budgets, thresholds, denominators, Train-only selection, send-order
  refusal, native model prohibition, and 52 > 48.
- **Future focused cases:** censored-∞ rank, zero denominators, Jain all-zero,
  alias identity, missing usage, and clock-domain mismatch giving a null fraction.

## 7. Mayor choices required

1. Lineage for the P0 numbers: L-PUB CPU exploratory now, and/or L-T16 native
   non-model arms after gates.
2. Group edges (default 512/2048).
3. Primary throughput window (default offered_window_s).
4. FIFO = fixed32 alias.
5. Optional TTFT-only secondary count (default: no).

## 8. Status and smallest blocker

- No measured P0 value exists (benchmark_results.md:72-83), and none was produced here.
- Native14 FAIL: model arms are prohibited on Native.
- D2-F1 is an accepted blocker (26139be5); Role6's repair is pending.
- G06 and G16 are pending. Protocol16 still binds b4b83d (2f336bc9 is not adopted).
- E4 coverage is 1 of 2 FAIL, and it is not a throughput result.

**Smallest resume blocker:** the Mayor's choice in §7, then a frozen C1 scope
(plus C2 if a CPU table is chosen). Any native number additionally needs the
D2-F1 repair accepted, G06/G16, and a fresh host plus preregistration.
