# Opus implementation note: `mpc_lab/model.py` (pr-mpc-model)

Implementer: obsidian / Claude Opus 5.5, alias **opus**.

## What was built

- **Status.** This is a compact event/lifecycle **CPU development
  surrogate**. It is not vLLM parity, not an identified service law and not
  native qualification. Its service coefficients are the synthetic
  `ModelParameters` defaults (`calibration_version="synthetic-unqualified-v1"`).
- **Specifications followed:** the Mayor's ADR and
  `mpc-mathematical-specification.md` §2–§5.
- **Shared contracts unchanged.** `mpc_lab/contracts.py` and `__init__.py`
  were read but not edited. No contract change is requested.

### Files

| Path | Contents |
|---|---|
| `/home/daniel_lin/gt/pr32/polecats/obsidian/pr32/mpc_lab/model.py` | Public `rollout(...)` and `generate_scenarios(...)`, with the exact signatures in the assignment |
| `/home/daniel_lin/gt/pr32/polecats/obsidian/pr32/tests/test_mpc_model.py` | 33 tests: 21 functions, one of them parametrized with 13 invalid cases |
| `/home/daniel_lin/gt/pr32/polecats/obsidian/pr32/experiments/mpc-research/proposals/opus-implementation.md` | This note |

### Model (the simplest coherent phase/work surrogate)

**Lifecycle.** Each request moves through these phases:

- pending offer;
- gateway admission;
- FIFO prefill (head-of-line only);
- first content;
- parallel decode;
- terminal (success or failure).

Every flow (Q, D, R) is generated from this one lifecycle. There are no
independent cost/queue regression heads.

**Latent remaining work.** It is explicit per request, in nominal seconds:

```
prefill work = first_overhead_s + prefill_s_per_token * prompt
decode work  = decode_s_per_token * (output_tokens - 1)
slowdown(t)  = service_scale * (1 + decode_interference * D(t))
```

All active work advances at 1/slowdown between events. Because slowdown is
piecewise constant, event times are exact. More decoders slow both prefill
(TTFT) and decode (lifetime and therefore release), so service, phase and
lifetime are coupled.

**Positive work.** `first_overhead_s > 0` and `decode_s_per_token > 0` are
validated, so there is no zero-work busy loop. A zero prompt costs
`first_overhead_s`. A request with zero remaining decode is released at its
first-content event in a single step.

**Exact accounting.** This mirrors `gateway.py` `begin_request` and
`finish_request`:

- An offer is rejected iff `R + c > budget` (equality admits). R is updated
  incrementally on admission and set to `max(0, R − c)` on release.
- The whole cost is held until the terminal event and released exactly once.
  Double release raises `double_release`.
- A budget decrease never evicts or clips a reservation.
- The budget applying to an offer is `plan.budget_at(receipt offset)`,
  evaluated at gateway receipt. An action change therefore affects only later
  receipts.
- Tie order is fixed: service completions and releases are processed before
  arrivals at the same instant.

**Cohort and reward** (spec §4 as frozen by the Mayor):

- **Scored cohort.** Every snapshot request (old, unresolved), plus every
  offer with `0 <= offset_s < evaluation_window_s`.
- **Tail offers.** Offers in `[evaluation_window_s, evaluation_window_s +
  tail_s)` are simulated and interfere (reservation and service), but are not
  rewarded and not returned. The rollout stops at `evaluation_window_s +
  tail_s`.
- **Outcomes returned.** `outcomes` contains exactly the scored cohort, so
  `score_lower == sum(o.good(slo))` holds identically (asserted in tests).
- **`score_lower`.** Resolved good outcomes. Eventual success is required,
  so a first-token hit followed by failure is not counted.
- **`score_upper`.** Additionally counts unresolved cohort members still
  possibly good under the scenario: the scenario's success flag is true, and
  the client first content is on time or could still be.

**Labels and clocks.**

- `Outcome.first_at` is the **client-observed** first content (gateway first
  content plus egress). `scheduled_at` is the client-scheduled time.
- Known old `first_at` values are respected; they are not re-simulated.
- An old request with `scheduled_at=None` uses the explicit scenario
  assumption `received_at − scenario.ingress_delay_s`.
- Future offers use their own `ingress_delay_s` and `egress_delay_s`.
- All times are absolute values in `snapshot.clock_domain`.

**Declared latent-work belief for old requests** (an assumption, not a
measurement):

- Awaiting old requests are assumed to have made no prefill progress; this is
  the pessimistic member of the aliased set.
- Old streaming requests have total output `ceil(old_output_fraction *
  max_tokens)`. Emitted tokens are estimated from the time since `first_at` at
  the nominal rate, and at least one token always remains.
- All old requests share `scenario.old_success`.

**Failure semantics for offers.** `success=False` with `output_tokens=k`
fails after `k` tokens. With `k=0` the request fails at prefill completion
without first content.

**Validation and bounds.** Inputs that are invalid, unsupported or
non-causal raise `RolloutError(<stable_reason>)`. Covered checks:

- parameters, config and plan budgets (must be in `supported_budgets`);
- snapshot completeness, cost-identity match and causal timestamps;
- duplicate ids;
- prompt/max-token support;
- `output_tokens <= max_tokens`;
- `success` requiring at least 1 output token;
- sorted, finite offers;
- transport delays.

Exceeding limits also raises, never returning a truncated result:

- `max_events` → `event_limit`;
- `max_active_requests` → `active_limit` (both the snapshot size and during
  the run);
- an absolute `time.monotonic()` deadline → `deadline_exceeded` (checked
  before the run and every 64 events).

**Scenarios.** `generate_scenarios` uses only caller-supplied past summaries:
the rate and the empirical prompt/max_tokens marks.

- Arrivals are Poisson on `[0, duration_s)`, with marks drawn independently
  with replacement.
- `output_tokens = max_tokens`, because no oracle EOS is available.
  `success=True`.
- Service scales are assigned round-robin.
- The RNG is seeded by `sha256(seed|run_id|sequence|index)`. The output is
  deterministic, and every plan shares the same scenarios.
- The function has no parameter that could carry a future tape or realized
  outputs (asserted in tests).

## Test evidence

Command (from the worktree root):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/obs-pt:. python3 -m pytest -q -p no:cacheprovider tests/test_mpc_model.py
```

Result (initial delivery): `27 passed in 0.07s`. After the integration-review fixes below: `33 passed in 0.14s` (Python 3.14.4).

**Toolchain caveat.** The supplied Python 3.12 venv's base interpreter
(`~/.local/share/uv/python/cpython-3.12...`) is not visible inside this worker
sandbox; executing the venv's `bin/python` fails with "No such file". I did
not try to repair that.

- Tests ran on system **Python 3.14.4**.
- pytest came from that venv's pure-Python `site-packages` (`_pytest`,
  `pytest`, `pluggy`, `iniconfig`, `packaging`), symlinked into `/tmp/obs-pt`.
- The model is stdlib-only and uses no 3.13+ features. A Python 3.12 run is
  still owed before merge.

**Coverage map** (required cases):

| Required case | Tests |
|---|---|
| Exact boundary | `test_exact_boundary_admits_and_one_more_rejects` |
| No eviction; exactly-once release | `test_cap_decrease_never_evicts_and_release_exactly_once` |
| Decreases block new admissions | `test_decrease_blocks_new_admission_until_release` |
| Causal action change | `test_causal_action_change_affects_only_later_arrivals` |
| Old cohort, late failure, known `first_at` respected | `test_old_cohort_late_failure_is_not_success_and_known_first_respected` |
| Explicit ingress/egress assumption | `test_old_scheduled_unknown_uses_explicit_ingress_and_egress_assumption` |
| First hit then failure; zero-output failure | `test_new_offer_first_hit_then_failure_and_zero_output_failure` |
| Zero prompt | `test_zero_prompt_has_positive_work_and_completes` |
| Tail scoring and interference | `test_only_window_offers_scored_but_tail_offers_interfere` |
| Unresolved lower/upper bounds | `test_unresolved_bounds_at_tail_end` |
| Score identity and nonnegative R | `test_rejected_offers_and_scores_match_good_outcomes` |
| Deterministic, past-only seeds | `test_generate_scenarios_deterministic_and_past_only` |
| Common scenarios across plans | `test_rollout_deterministic_and_common_scenarios_across_plans` |
| Invalid inputs (12 parametrized reasons) | `test_invalid_inputs_raise_stable_reasons` |
| Limits and deadline | `test_limits_and_deadline_raise_instead_of_truncating` |
| Generator validation | `test_generate_scenarios_rejects_invalid_inputs` |

## Integration-review fixes (Mayor same-task review)

Three bounded correctness fixes, confined to `model.py`, `test_mpc_model.py` and this note.
Shared contracts were not edited.

1. **Bounded scenario generation.**
   - `generate_scenarios` now raises `ValueError("scenario_count_limit")` when
     `count > MAX_GENERATED_SCENARIOS`. That cap equals `PlanningConfig().max_scenarios`, which is 32.
   - It raises `ValueError("offer_limit")` once the total number of offers
     across all scenarios exceeds `MAX_GENERATED_OFFERS`. That cap equals `PlanningConfig().max_events`,
     which is 20000.
   - It raises `ValueError("nonadvancing_arrivals")` when a positive
     inter-arrival time no longer advances the float clock.
   - Every loop iteration either appends a bounded offer or exits, so at most 20001 draws are made.
   - Huge finite inputs now fail fast: rate 1e300, duration 1e300,
     count 1e12, and 32 scenarios at 1000/s (tested under 2 s total).
   - The 100 ms planning budget is unchanged.
2. **Client terminal consistency.**
   - `Outcome.terminal_at` is now the client terminal: the gateway terminal plus the request's egress delay. Previously it was the gateway time, so a one-token request with egress
     had `terminal_at < first_at` and was credited before the client completed.
   - The reservation release in `reserved_trace` still happens at the gateway terminal.
   - Client events after the rollout cut (`sampled_at + evaluation_window_s + tail_s`) are censored:
     `first_at` / `terminal_at` are reported as `None` and `success` stays `None`.
   - `score_upper` uses the known gateway outcome where one exists. So a gateway
     failure whose client terminal falls past the cut adds 0, and a gateway success past the cut with on-time first content adds 1.
   - Cohort membership and counts are unchanged.
3. **Sealed-ledger cost identity.**
   - Every old `RequestMark.cost` must equal
     `cost_spec.estimate(prompt_tokens, max_tokens)`, with relative tolerance 1e-12 and no absolute slack.
   - A mismatch raises `RolloutError("ledger_cost_mismatch")`. Before this fix, `cost_hash` alone could claim the formula while the marks violated it.
   - The test fixtures now use a synthetic dyadic `CostSpec` in which cost is exactly
     `prompt_tokens / 1024`. The check itself was not weakened.

**Mutation evidence.** I ran the new tests against the pre-fix model (commit `c7922c6`):
- The five new non-generation tests fail:
  - the `ledger_cost_mismatch` parametrized case;
  - the sealed-ledger test;
  - the egress-terminal test;
  - the transport-tail test;
  - the late-failure test.
- The result was `5 failed, 27 passed, 1 deselected`.
- The huge-input generation test did not terminate on the pre-fix model; it was killed by a 120 s `timeout`.

Final run on this machine: `33 passed in 0.14s` on **Python 3.14.4** with
the venv's pure-Python pytest. A Python 3.12 run is left to the Mayor.

## Descriptive local timing (not a compute claim)

One run on this worker CPU, Python 3.14:

- Setup: an empty snapshot, 25.48 req/s synthetic offers, and 8 `{16,32}`³
  plans × 32 scenarios with the default 5 s window and 10 s tail.
- Result: **0.331 s wall for 256 rollouts, 171,815 events.**

This **exceeds** the default `PlanningConfig.max_wall_s = 0.1`. With that
deadline, a full 8×32 search would raise `deadline_exceeded` and the planner
must fall back observably. Options are fewer scenarios, a shorter tail, an
off-process worker with a longer cadence, or optimization. That is a decision
for the planner/runtime owners, not something to hide inside the model.

The plan scores from that run favoured constant 16. They come from
**synthetic** coefficients and carry no performance meaning.

## Limitations and open items

- **Unidentified coefficients.** The service law is a reduced surrogate with
  unidentified coefficients. No GPU, KV, scheduler or native claim is made.
- **Awaiting old requests.** The no-progress prefill assumption is
  pessimistic. The contract has no field for sampling that ambiguity across
  scenarios; only `old_output_fraction` is varied for streams. If the Mayor
  wants prefill-progress ambiguity, it needs a contract field. That is a
  suggestion only; nothing was changed.
- **Actuation delay.** The plan's first action applies from offset 0. The
  model has no actuation delay; the transport/runtime layer must account for
  it, or the plan must encode it.
- **Mark sampling.** Prompt and max_tokens marks are sampled independently,
  not as pairs, and there is no burstiness model beyond Poisson.
- **Scenario success.** Generated scenarios always set `success=True`.
  Failure-regime scenarios must be built explicitly by the caller.
- **Exceptions.** `generate_scenarios` raises plain `ValueError` with stable
  reasons for invalid inputs. `rollout` raises `RolloutError`, a
  `ValueError` subclass.
