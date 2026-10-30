# Receipt-state acquisition and next confirmation design

PR32 now contains an executable calibration diagnostic and a separate design
for independent confirmation. It does not claim that the three residual queue
misses have been solved without fresh native data. The prior native rejection
and phase-margin development rejection remain unchanged. No new GPU or HTTP
trial was performed for this change.

## Implemented

- Short `prefill_host.py queue-probe` command, dedicated exact-token manifest,
  one loaded backend, no gateway/old clock-gate prerequisite, post-shutdown audit
  and archive even on failure. Setup supports `--next queue-probe`.
- 232 frozen trials: 16 original LIGHT warmups, 48 zero-offset TRAIN references,
  then 24 probe warmups and 144 scored probe trials. Original hard offsets remain.
- Baseline OFF, receipt-only OFF and LIGHT-plus-receipt arms; exact paired tokens,
  disjoint block content, all six arm permutations and alternating case order.
- Opt-in EngineCore CPU receipt evidence on existing hooks: matched overlapping
  schedule/commit objects, release/progress, pending batches, pre-ADD scheduler
  state, original queue-post/enqueue times, snapshot span and post-event bracket.
- Separate observer controls, state comparability, queue distributions/counts,
  raw rows and explicit unobserved OFF state. No measured receipt becomes a
  pre-arrival forecast input; no midpoint median hides modes.
- Exact-source offline raw reconstruction with installed source/runtime pins,
  original complete native CPU/JUnit evidence, manifest pairing, TRAIN events,
  chronology, counters, report and summary checks.
- Offline six-term fitter uses only 48 TRAIN rows and 24 TRAIN LIGHT events;
  delayed probe labels are filtered out before fitting. Raw capture is immutable;
  fit failures are saved outside it. A fit is not held qualification.

## Design and self-review

The independent confirmation design is `experiments/prefill-queue-confirmation-v2.md`.
It proposes 240 fresh-host trials with all warmup/TRAIN data before the forecast
seal and held trials. Receipt measurement, queue accuracy, endpoints/anchors and
integrity have separate verdicts. Each cell needs comparable evidence, and all
old numerical budgets remain. The 240-trial collector is **not enabled**.

F0 retains the six-term law and TRAIN-only analytic phase eligibility. An extra
candidate F1 requires a committed equation, causal inputs, estimator, bounds,
readiness/failure rules and development selection metric before confirmation.
It is not yet specified: no convenient bias/branch is added from old held misses.
Calibration-probe correction selection must group all paired arms/case/block
inside leave-one-block-out folds. Those folds are development, not confirmation.
Different-host coefficients require fresh TRAIN recalibration under an already
frozen estimator. Cap/MPC and HTTP still have their own later acceptance scopes.

Self-review covered: observer perturbation; unknown baseline phase; leakage of
post-arrival state; overlapping batch identity and FIFO commits; CPU versus GPU
label semantics; buffer overflow; sticky invalid captures; no missing/duplicate
trial deletion; different-state/unassessable denominators (including a final guard that blocks
the whole state-control verdict when any individual cell has zero comparable
pairs); exact-token pairing;
JSON ordering on offline replay; source/JUnit/runtime provenance; interrupted
collection, preserved partial evidence and no automatic retries; same-host
training availability before releasing a rental; and immutable offline fit output.

Observer errors are retained without swallowing native scheduler errors. LIGHT
receipt events are independently aligned with existing step records. The native
vLLM hooks and original upstream source map are unchanged in scope. Reports do
not interpret observed-state equality as proof of physical phase or causation.

## Evidence

Validation: **578 broad CPU regressions passed** (294.75 s) on the initial
implementation, followed by **48 final queue-probe/host cases passed** (10.87 s)
including the per-cell coverage guard. Counts overlap and are not summed. The
broad run loaded the pre-guard implementation; it is not an exact-source native
acceptance claim.

See `cpu-validation.json` for exact source fingerprints, final test stages,
JUnit/stdout hashes and scope. Counts from overlapping runs must not be added.
These are local CPU regressions with mocked hardware/transport boundaries,
not complete native CPU acceptance or measured GPU accuracy.

`synthetic-clock-fit.json` records an additional real execution of the fitter on
synthetic own-rollout labels under the new plan: 48 TRAIN endpoints, 24 TRAIN
LIGHT event rows, rank six and forecasts for 16 cases. The recovered parameter
maximum absolute error is about 6.33e-17 s. This proves synthetic algebra and
plan compatibility, not generalization to native timing.

Next required evidence: a fresh 232-trial queue-probe run. After its audit, choose
an admissible calibration-only correction or report no adequate candidate; freeze
it before new held confirmation. Existing live-http is not an acquisition
substitute, and no current model/HTTP/cap/MPC approval flag is promoted.
