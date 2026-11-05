# Role7 frozen cross-critique: team16 planner, original and revision01 (read-only)

Status: **sealed read-only critique**. This is not code acceptance, native evidence,
qualification, a speed result or a win claim. It is separate from the main comparison
seal `bb646f03df21e95ca4a22ffd607a324dc77ecb2e1fb73e1550b272f6c87e808a`, which is
unchanged. Remedies below are suggestions and were not implemented. No peer file was edited.

## Inputs and method

- `PLAN.json` (`fa9a22de…`) pinned every peer file before I read any peer source.
- Original package seal `cd10bed4…` (parent did not accept it). Revision01 seal `2051bc18…`.
  Both pass `sha256sum --check --strict` before and after every attempt. Peer tree
  listings (path, type, size, mtime) are unchanged, and no bytecode or pytest cache was written.
- `controls/test_role7_planner_controls.py` (10 cases) imports the frozen peer module and
  its unchanged helpers through `run_controls.py`. `run_attempt.sh` runs it on
  {original, revision01} × {Python 3.14.4, 3.12.16}, each in a new attempt directory.
- Every attempt exits 1, and the failures are preserved: 7 defect controls FAIL, and the A0
  positive control (×2) plus the C characterization PASS. All 4 attempts print
  byte-identical `ROLE7_OBSERVATION` output (sha `b863c715…`).
- `make_critique.py` re-verifies all records with 9 checks. Its `critique.json` output is
  byte-identical on both Pythons (sha `602d1baa…`). `check_sensitivity.py` confirms that
  7 different evidence mutations are each rejected, on both Pythons.

## I1 — a tiny positive draw is absorbed into an artificial tie (HIGH confidence)

The panel uses weights `(1.0, tiny)`, and the tiny draw makes `(32,32)` strictly better.
Synthetic score tables and a frozen clock are used.

| control | tiny weight | large draw's common term | result (both packages, both Pythons) |
|---|---|---|---|
| A0 positive control | 32·ulp(0) (scaled to an exact subnormal) | wall 0.0 | selects (32,32) = oracle, PASS |
| A1 | 32·ulp(0), exact subnormal products | wall 1.0 | **accepted**, selects (16,16) |
| A2 | 2^-60 (normal) | wall 1.0 | **accepted**, selects (16,16) |
| A3 | 2^-60, primary natural-good key | good 1, wall 1.0 | **accepted**, selects (16,16) |

In A1–A3, all four paths get identical float `(good, wall)` values and there are no refusals.
Both `mpc` and `strong_one_move` select `(16,16)`, while the exact rational oracle selects `(32,32)`.
A0 differs from A1 only in the common term, which isolates the cause. The cause is
`math.fsum(terms)` absorbing the tiny draw's exact term into the large term (revision01
`policy_planner16.py:155-165`). The `_rank` tie-break (`:97-101`, by transitions, then path
sum, then lex) then resolves the artificial tie toward holding the received budget. The
original package (`:96-100,136,154-161`) behaves identically.

This contradicts three documented claims:
- revision01 `README.md:31-32`: "Exact representable subnormal terms are still accepted.
  No positive draw is dropped".
- The revision test `test_representable_tiny_draws_are_retained_and_keep_exact_oracle_order`
  (`:65-76`). Its fixtures use an all-zero common term (`:19-23`), so absorption is never
  exercised.
- original `README.md:59-61`: "refuses instead of dropping a draw".

`README.md:33-35` disclaims universal exact-real ordering for normal arithmetic. That covers
the *ordering* in A2/A3, but those cases still drop a whole positive draw. A1 falls inside the
revision's own subnormal domain. A1 also reproduces the parent's accepted-(16,16)-vs-oracle-(32,32)
pattern using exact terms.

Fairness impact:
- Rankings are common to all arms, so no arm gains a specific advantage.
- The bias is deterministic: absorbed differences turn into hold/lower-budget choices, so
  MPC and strongOneMove collapse together.
- Train-selected fixed inherits the absorbed per-panel floats.
- The model-expected loss is below one ulp of the dominant term. It is practically relevant
  only for weight ratios near or below about 2^-52 relative to the term scale. With
  equal-weight draws, only differences below about 2^-49 of the largest term (far below
  measurement resolution) can be absorbed.
- This is never evidence of an MPC advantage.

## I2 — train-selected fixed ignores an accepted third budget (HIGH confidence, conditional)

Control B uses budgets `(16,24,32)`. The planner accepts this contract, and its own peer tests
use `(32,24,16)` for all arms (`test_policy_planner16.py:184-191`). On the training panel,
`best_fixed == best_multistep == (24,24)`. Even so:
- `freeze_fixed_choice` accepted the panel and froze 16, because it iterates only
  `for budget in (16, 32)` (revision01 `:244`). `FrozenFixedChoice` (`:201,214`) and the
  constructor (`:284`) also hard-code {16,32}.
- The held-out `train_selected_fixed` then executed `(16,16)` as `model_plan_only`, without refusing.

The peer training tests (`:593-695`) use only `(16,32)`. Consequence: the "strongest fixed"
baseline can be weaker than the strongest constant in the shared action set, which would
manufacture an MPC advantage.

Severity is high for any 3-budget comparison and nil when every arm's action set is locked to
{16,32}. `comparison16.py:404` already refuses any path element outside {16,32}.

## I3 — planner age and elapsed time exclude acquisition and queueing (documented scope)

Control C is a characterization and passes. With a 10 s acquisition-to-dequeue delay:
- The default adapter clock (`:341-353`, `lambda: acquisition.entry`) accepts the plan with
  `first_action_source_age_s = 0.15 ≤ 0.5`.
- The same acquisition, evaluated at the true dequeue time, refuses with
  `policy16_full_entry_solver_actuation_deadline`.

`solver_elapsed_s` starts at the adapter entry (`:485-489`). The README (`:41-42`) states that
Role6 owns acquisition/IPC deadlines and final freshness. These fields are therefore lower
bounds and must never be used as model-arm freshness or as total controller cost. The runtime
must either pass the true common-domain dequeue clock or enforce TTL itself. The accounting
must charge acquisition, queueing, planning and actuation to the model arms, in the endpoint
domain; comparison16 requires measured costs inside the endpoint envelope.

## I4 — a whole-panel refusal also refuses the fixed arms (documented scope)

`_refuse` (`:325-334`) removes the ranked output for every policy. A single numerical refusal
(`:444-450`) therefore also covers fixed16, fixed32 and train-selected fixed (README `:26-29`;
already tested by revision01, so not duplicated here).

If a runtime routed fixed arms through PolicyPlanner16, a refused model panel would make
fixed32 hold the received 16 or drop the decision point. It would also charge the fixed arms
for the full panel search. The fixed arms must instead actuate their frozen constant
independently. Model-arm refusals should hold and remain in the all-offered denominators.
comparison16 `check_policy`/`model_decisions` already bind each fixed arm to one applied budget,
so such routing would be refused rather than silently scored.

## Fairness assessment of the revision01 patch

- It correctly refuses the targeted case: inexact positive products at or below min-normal.
  The parent and Mayor controls cover this; it is not duplicated here.
- The refusal is common to the whole panel and identical for every arm. It changes no threshold
  or baseline, and the guard runs inside the timed solver window.
- It is incomplete: summation absorption (I1) reproduces the same wrong-plan pattern with exact
  terms. Narrow the "no positive draw is dropped" claim, or fix I1, before relying on it.

## Checked, no issue found

- strongOneMove is the best row with at most one transition from the received budget,
  including delayed single switches, taken over the same full ranking.
- All arms share the rankings, `common_panel_sha256`, the contract sha and the objective
  (all-offered good, then wall, transitions, path sum, lex).
- The fixed choice uses only the training split, is frozen and bound to the contract, and is
  aggregated with exact Fractions across panels (subject to I1 within a panel, and to I2).
- Binary weight rescaling either preserves exact ratios or refuses.

## Remedy options (not implemented)

- I1: rank on exact Fraction keys (at most 81 paths × 16 draws), keeping floats for display
  only. Alternatively, refuse via an explicit weight-ratio contract bound, never silently.
  Either way, add a regression test with a nonzero common term.
- I2: refuse unless the budgets are exactly {16,32}, or select over every contract budget.
  Preregister {16,32} as the action set for all arms.
- I3/I4: runtime and accounting requirements as stated above.
