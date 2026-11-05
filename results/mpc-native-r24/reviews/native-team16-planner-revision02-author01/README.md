# Planner16 revision02 — shadow-only numerical correction

This continues revision01, not a new planner design. It changes only the derived
`policy_planner16.py`; prior packages, parent witnesses, the authorized Role7
critique and native/product/runtime sources remain unchanged. Parent acceptance
and integration are still required. No native advantage is claimed.

## Evidence and correction

The frozen Role7 controls reproduced **7 failures / 3 passes on each Python**
against byte-identical revision01:

* **I1, six failures:** exact positive weighted terms disappear when a float sum
  absorbs them into a nonzero common term. Both MPC and strong-OneMove accepted
  `(16,16)` instead of the rational oracle's `(32,32)`. This includes exact
  subnormal products and normal `2**-60` weights, in wall and good-count keys.
  Revision01's product-rounding refusal remains correct but was insufficient.
* **I2, one failure:** the full panel allowed budgets `(16,24,32)`, yet training
  considered only fixed16/fixed32 and could omit the best constant24. The actual
  comparison described by the critique locked all arms to 16/32, so this is a
  conditional fairness defect, not evidence of an inflated native win.
* The zero-common-term positive controls and I3 scope characterization passed.

Accepted panels now use exact `Fraction` means of every frozen finite per-world
metric and positive weight. Common binary weight scaling is still proved exact.
The reduced means are serialized on each ranking row as:

* `exact_expected_natural_slo_good`
* `exact_expected_cohort_wall_s`

Each is `[numerator_string, denominator_string]`. Ordering uses these exact
values, then transitions from the received budget, exact path sum, and path
lexicographic order. All cells must pass validation before rankings and all three
best-row references are replaced. The original complete evaluator is unchanged.

The old `expected_*` floats and frozen-choice `candidate_objectives` floats
are **display summaries only**. They may tie even when the exact keys differ;
consumers must not re-rank them. Training recomputes and validates the exact keys,
then aggregates exact per-panel means with the declared training weights.
`ranking_arithmetic=exact-rational-frozen-binary-inputs-v1` is bound into the
planner contract, so old absorbed-float training artifacts cannot silently bind
to this revision.

`freeze_fixed_choice` considers every constant in the shared action set, up to
three budgets. `FrozenFixedChoice.candidate_budgets` records its canonical sorted
set; integral floating budgets become integers. The constructor requires this
set and the full contract hash to match. A typed frozen third-budget choice uses
the unchanged base selector after initializing the base's legacy two-point
constructor through its MPC branch. Named fixed16/fixed32 and legacy explicit
caller-declared 16/32 choices retain their existing interface. Legacy explicit
choices are not proof of strongest-fixed training.

This is exact ordering of the **supplied frozen binary numeric values**, not an
exact physical model or a claim that arbitrary numerical inputs are accepted.
Revision01's zero/nonfinite/inexact-subnormal product refusals, including the
normal/subnormal boundary, remain. No positive world is removed or renormalized
away to resolve ambiguity. Preserving entries and weight ratios alone did not
justify the older report's broad “no positive world dropped” implication: float
aggregation could still erase its decision influence. That claim is corrected
here without rewriting old evidence.

## Validation and refusals

**130 passed on Python 3.14 and 130 passed on Python 3.12**:

* 102 unchanged previous controls, including the parent's product-rounding oracle.
* 10 unchanged independent Role7 controls.
* 18 new focused cases: six accepted exact-order absorption cases, three exact
  training/frozen-heldout choices, three full-action-set/canonical choices, one
  partial-candidate rejection, one tampered-key rejection, two exact-work deadline
  refusals, one version-change refusal during exact work, and one strict primary
  objective versus secondary-wall tradeoff.

Both red attempts, both green attempts, full output, exact child arguments and
environment, and top-level source snapshots are retained in `attempts/`. The
green attempts contain the final planner/test bytes. The earlier parent attempts
01 (import) and 02 (fixture setup) remain harness errors, not algorithm defects;
parent attempt03 is the valid product-rounding counterexample.

Invalid, partial, source-mutated, version-stale or late panels refuse atomically,
with no ranked result or first action. Exact arithmetic, sorting and output
preparation are inside the existing cooperative solver/deadline charge. This is
not preemption or a hard OS/transport latency guarantee.

Unchanged bounds: at most 16 worlds, 2–4 steps, at most 3 budgets (81 paths and
1296 cells), source age at first action at most 0.5 seconds, and all-offered
natural-SLO-good then complete HTTP cohort wall. Natural SLO thresholds remain
0.5 seconds first/single-token and 0.05 seconds TPOT. Strong-OneMove still permits
at most one planned transition from the received budget, including delayed
switches. Every shadow policy sees the same complete world/path/objective panel.
The future path suffix is a forecast, never a live execution script.

## Integration requirements, not additional planner defects

* **I3:** the optional adapter's default entry is acquisition time. The runtime
  must supply a true same-domain dequeue clock or separately enforce queue/IPC/
  delivery TTL and endpoint accounting. A ten-second queue-delay characterization
  remains unchanged; the true explicit dequeue clock refuses.
* **I4:** actual fixed benchmark arms must apply their frozen constants
  independently of model-panel success or computational cost. Their observations
  cannot disappear when a shadow panel refuses. Model refusal holds, timing and
  all offered requests must remain in comparison denominators.

No runtime/comparison peer implementation was changed. Source/version callbacks
and declared train splits are trusted caller contracts, not authentication.
Training choices must actually be frozen before untouched heldout evaluation.

## Reproducibility and qualification

`PLAN.json` and 532 inputs were frozen before the red replay, including the
predecessor seals, 79 critique files and 80 planner-only parent proof copies.
The parent-reported provenance commit is
`a5f676aa1fcc035eb6225c4a7c218722471c81c8`; no git verification or git operation was
performed. Unrelated role2 source/proofs were not read.

`report.json` records pins, all attempts, exact changes and blockers.
`source_changes.patch` is relative to revision01. `commands.txt` records the
executed launch commands and the read-only post-seal verification commands.
`SHA256SUMS` covers every delivery file other than itself, including the copied
critique's nested seal. `verify_delivery.py` verifies all delivery bytes, tested
source bytes and unchanged input pins without writing files.

The sealed attempt labels are once-only. Do not add attempts to this sealed
directory; use a new exclusive parent replay directory, or run `run_tests.py
derived` read-only under the recorded environment. Both prescribed interpreters
use `-B`, unset `PYTHONPATH`, one-thread numerical settings, disabled pytest
plugins/cache and `nice -n 19`. All owned test children were synchronously joined.
There were no agents, broadcasts, network/GPU runs, git operations or live CAS/
HTTP actuation.

Native14 remains rejected: Rhat `3.6750452719276576`, minimum bulk ESS
`4.336133031401349`, minimum tail ESS `10.94451798214706`, against unchanged
`1.01/400/400` gates. No accepted native posterior/model, shared normalized
native/client law, qualified public-only filtering, untouched-heldout
calibration/PIT or native action-response qualification is supplied.
Synthetic controls do not establish native MPC performance, wins or an
unpredictability cause. Native/deployable/action-permission flags remain false.
