# Queue-model selection and independent confirmation: design

Status: **the native 240-trial controlled confirmation at 2ebb185 passes and is independently replayed; global model/HTTP/cap/MPC approval remains false**.
See [the implementation addendum](prefill-entry-clock-v1.md) for the exact estimator, numeric plan, sealing and independent audit.
The queue-probe collector is implemented separately. This document governs the
next decision after fresh calibration probes; it cannot infer missing evidence
from the old failed run. The current six-term model, phase uncertainty and
measurement controls have separate verdicts.

## Decision point and permitted inputs

For controlled queue prediction, freeze forecasts at foreground release before
the delayed sentinel is posted. Permitted inputs are the frozen workload,
planned arrival offset, configured B2048/seq32, reviewed runtime/dispatch rules,
state actually available at release and TRAIN-derived costs/uncertainty.
This is controlled-arrival prediction, not prediction of unknown HTTP ingress.

The eventual HTTP continuation model has its own pre-ADD decision point and
cannot claim it predicted the preceding input-queue wait. An adaptive cap would
require forecasts before admission, qualified across several caps; neither is
accepted by this confirmation.

Forbidden forecast inputs include actual receipt-state snapshots, native future
commit times, measured queue residence, actual held phase, future work signatures
or later outputs. Receipt labels may diagnose calibration failures and test an
observed-state explanation, but cannot become an after-the-fact shortcut.

## Acquisition and development decision

Use the new 232-trial calibration archive. Audit it first. Refit the six original
service-law coefficients using only the 48 TRAIN endpoints and 24 TRAIN LIGHT
event rows. Keep delayed probes out of this fit. Preserve the six-term model as
candidate F0 and the uncorrected point forecast as an ablation.

Use receipt probes to answer the following, without promoting the result to
independent validation:

| Evidence | Permitted conclusion | Next action |
|---|---|---|
| Invalid chronology, runtime/source mismatch or missing samples | Evidence invalid | Repair collection; no model selection |
| Baseline/receipt endpoint controls fail | Recorder may perturb the target | Reduce/qualify observer; do not fit away its bias |
| Receipt/LIGHT have no comparable states for a case | State-specific observer effect unresolved | Mark that case unassessable; no whole-domain qualification |
| Matching observed states exceed existing queue budget | Measurement/state explanation remains incomplete | Preserve failures; investigate same-state timing |
| Valid controls and residual aligns with an own predicted boundary change | Phase-margin hypothesis supported for development | Retain TRAIN-only analytic phase eligibility |
| Valid controls, but residual misses every predeclared branch | Service/phase-origin model incomplete | Specify an additional model before confirmation |

A different receipt state is not proof that baseline OFF switched. Do not turn
nearest-branch assignments into native phase labels or branch probabilities.
All eight original offsets/background cells remain in development and held
reports. No special exemption for the 50 ms/D16 or 400 ms/D16 failures.

## Candidate admission and selection

F0 uses the original six-term law. Its common-scale half-width comes only from
TRAIN OFF endpoint residuals; branch eligibility follows the analytic phase
algorithm. Existing limits and readiness failures remain: unsupported source,
arrival after completion, coincident anchors, excessive complexity or more than
two branches cannot silently broaden the prediction.

An additional candidate F1 is conditional on new calibration evidence. Before
any confirmation collection, it needs a committed addendum containing:

1. A complete timing equation or state transition algorithm, parameter count,
   causal input schema and implementation, with source hashes.
2. An estimator with identified calibration roles, fixed bounds/regularization,
   deterministic tie-breaking and explicit nonconvergence/readiness failures.
3. Own-rollout dependence and scheduling tests showing nonnegative durations,
   causal output order, barriers and outstanding-batch progress. A correction
   of reported queue values alone cannot replace this requirement.
4. A bounded uncertainty construction that cannot add branch points from held
   misses, impute OFF phase, or claim confidence/tail coverage from six repeats.

Do not preselect a convenient +12 ms bias or alter the width using old held
errors. Clock-origin, per-state interference and observation effects are competing
explanations until the new probes distinguish them. A TRAIN-only CPU-return hypothesis is now specified and prototyped in
[prefill-submit-clock-development-v1.md](prefill-submit-clock-development-v1.md).
It remains an ablation: nominal first-commit timing and unobserved branch controls
are unresolved. It has **not been admitted or selected** for confirmation.
If only F0 is admissible, retain it or report no adequate candidate.

For any additional queue correction estimated from probe outcomes, use grouped
leave-one-scored-block-out development across all six blocks. Keep each
case/block's three observer arms together; never split individual requests from
the same trial across folds. Every fold refits all probe-derived correction
parameters and uncertainty using its other five blocks. TRAIN-only six-term
cost fitting remains separate. Held-out fold receipt states cannot be forecast
inputs. Report per-cell and per-observer failures and signed errors, plus
endpoint/work/lifetime results. These folds are model development, not the final
confirmation or statistical tail evidence.

Prefer F0 whenever no admissible correction improves validation without new
per-cell failures. If multiple admissible corrections survive, their selection
metric, minimum gain and complexity tie-break must be frozen in the addendum
before calculating selection scores. Do not choose them after seeing results.
No admitted candidate means stop and report model inadequacy.

## Freeze and fresh confirmation

Freeze model family, estimator, feature schema, calibration roles, coefficient
bounds, uncertainty rule, all branch forecasts/eligibility, error budgets, input
IDs/content namespaces, trial order, runtime/source hashes and report schema.
Seal forecasts before the first held trial. All files must be independently
reproducible from permitted calibration evidence.

On the same unchanged host, the calibrated freeze can be reused only if runtime
identity and anchors agree. On a different rental host, collect fresh TRAIN
references and run the already frozen estimation algorithm before held scoring;
old absolute millisecond coefficients do not automatically transfer. Model
family and selection rules may not change based on the new held outcomes.

The proposed fresh-host confirmation has **240 trials**:

- 16 original LIGHT warmups, 24 three-arm probe warmups, and 48 TRAIN references;
- fit and seal forecasts; four before-anchor trials;
- 144 held trials: eight original delayed cells × six balanced blocks × three arms;
- four after-anchor trials.

This order keeps all 88 warmup/TRAIN trials before held data. The 48 TRAIN rows
are the only same-host cost fitting inputs. Warmup labels cannot select the model.
Use new qc- identities and content disjoint from qr-/ct-/qp-; observer arms share
content within a held case/block. Every arm's six permutations are retained.
The numeric case/order manifest and executable collector are now committed.
`queue-confirmation` implements this order with the first-entry family specified
in the implementation addendum. The older submit ablation remains unselected;
the entry family now has a passing native confirmation on one host. F0 and entry
forecasts are identical there, and all cells have one eligible branch. This does
not confirm the previous host's phase switch or a correction advantage. Missing
branch controls remain inconclusive.

## Independent verdicts

1. **Integrity:** exact-source CPU/native preflight, token identity, chronology,
   no missing/duplicate outputs, background lifetime, own/native work and
   dispatcher parity. An invalid sample invalidates the complete confirmation;
   it is never dropped from a denominator.
2. **Measurement qualification:** preserve baseline/receipt and baseline/LIGHT
   endpoint checks and observed-state receipt/LIGHT checks. Report comparable
   denominators for every cell; zero coverage is inconclusive. Global success
   cannot hide an unassessable cell. A predicted branch with no matching observed
   control does not become qualified because another branch passed; report its
   coverage gap and retain an inconclusive whole-domain verdict. OFF phase remains unknown.
3. **Queue prediction:** each held baseline OFF sample must satisfy the original
   max(10 ms,20% observed queue residence) budget against the nearest frozen
   eligible point. Stable cells keep their point condition. Score receipt and
   LIGHT separately as diagnostics; they do not replace OFF accuracy. Exact
   ties are unresolved; no midpoint median or per-case average masks a miss.
4. **Endpoints:** preserve original TTFT/completion absolute-relative rules,
   separate before/after anchors and structural/lifetime gates. A queue repair
   cannot regress endpoints and still be accepted.

A failed measurement gate is distinct from a failed model gate, but either
blocks overall qualification. No passing diagnostic, TRAIN fit, development
fold or limited queue confirmation sets HTTP/cap/MPC acceptance true. Keep raw
point scores, phase scores, all sample IDs, signed errors, branch counts,
unobserved branches and failures in the independently rebuilt report.

## After this confirmation

Only a passing controlled predictor permits a new same-host HTTP qualification
with its existing ingress/egress/state-bridge checks. Cap experiments must then
show predictive validity and safety across candidate caps before a controller
can compare actions. MPC acceptance requires its own closed-loop comparison
against fixed-cap baselines. These are subsequent scopes, not implied successes.
