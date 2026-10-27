# First-entry submit clock and independent controlled confirmation

Status: CPU implementation and retrospective calibration development. The
single entry-clock family is specified for a fresh confirmation attempt;
no model, HTTP, cap or MPC accuracy has been accepted.

## Estimator refinement

The previous `submit_clock` family and its saved evidence remain unchanged.
Its median across all prefill host spans conflates entry and subsequent work.
This family changes just one estimator: P is the median of the first foreground
prefill execute_model host span in each of the 24 TRAIN LIGHT trials. Scheduled
prompt work and planned foreground request identities identify that entry, not
probe outcomes or future receipt states. Nonfinite release, out-of-order steps,
missing entry work or invalid worker identity fails readiness.

Keep D at the original median of per-trial TRAIN decode medians. Keep the six
GPU coefficients fitted by the original bounded multistart algorithm on the
48 TRAIN endpoint rows and 24 TRAIN LIGHT event records. There are still six
fitted GPU coefficients and two directly estimated effective return constants.
There is no joint CPU/GPU physical identification and no per-cell fitted bias.

The causal equation is unchanged: each own prefill forward.cpu_after takes P,
each own pure-decode forward.cpu_after takes D; GPU and CPU may overlap through
the existing two-slot max-plus graph. The first-entry statistic is an effective
representative for this family, not a claim every later launch takes that time.

Keep the original full-prefix TRAIN prefill min/max sensitivity domain, rather
than narrowing it to the entry samples after inspecting probe results. One P is
shared across the complete rollout. This is a bounded hypothesis, not a coverage
interval, per-step noise law or confidence statement. The GPU-scale family is
not unioned with it. All original at-most-two-branch, own-release, independent
point-rollout and exact piecewise-affine readiness conditions remain.

Add a TRAIN readiness check on each first post-release commit, matched by the
independently generated work prefix, under max(10ms,20% observed). This is a
predeclared sanity check for this family, not held evidence. Entry clock must
also retain TRAIN endpoints and own/native work. An unready new-host fit stops
before any held anchor or queue trial; there is no held-dependent F0 fallback.

On archive 1793126846787142213, P=44.914831ms, D=1.685664ms and the P domain stays
14.984692–57.353666ms. All 24 TRAIN first-commit checks pass. The 12 D16 errors
are -7.252 to +6.943ms, versus -17.011 to -2.816ms for the predecessor. These
are calibration results. Point queue forecasts still have three baseline OFF
and one LIGHT miss at 50ms/D16. The phase score covers 144/144 calibration
samples, with the same unobserved early branches. No physical cause, OFF phase,
probability or independent accuracy is inferred from these results.

## Executable confirmation contract

The policy and all 240 numeric trial identities/order/content namespaces are
committed in `prefill-queue-confirmation-v2.json` and `.plan.json`. Regeneration
must match the committed numeric plan. `queue_confirmation.protocol` retains
all eight original offset/background cells and every observer-arm permutation.

| Trial range | Role | Count |
|---|---|---:|
| qc-001–qc-040 | 16 clock warmups and 24 three-arm warmups | 40 |
| qc-041–qc-088 | Same-host TRAIN OFF/LIGHT references | 48 |
| seal | Fit, readiness and seal before any held trial | 0 |
| qc-089–qc-092 | D0/D16 uniform OFF/LIGHT before anchors | 4 |
| qc-093–qc-236 | Eight cells × six blocks × three arms | 144 |
| qc-237–qc-240 | D0/D16 uniform OFF/LIGHT after anchors | 4 |

The fresh qc content keys are disjoint from qr/ct/qp; paired arms share exact
engine tokens, each block has a separate namespace. The original Azure-derived
main replay is not being rerun in this controlled experiment.

Freeze the predeclared algorithm before rental. On a fresh host estimate only
from its TRAIN labels, then seal the model, both F0 and entry predictions,
branch eligibility, source, manifest and completed TRAIN IDs. A monotonic seal
must lie after the last TRAIN capture and before the first held capture. Source
or freeze changes fail collection. Held outcomes never enter an estimator.

The report keeps every one of the 144 held queue samples and separate F0 point,
entry point and nearest-frozen-branch errors. Original max(10ms,20%) queue and
endpoint budgets remain; exact nearest-branch ties fail. A passing branch score
cannot hide a raw point miss. All arms are retained and scored separately.
Endpoint observer controls use original gates. Same-state receipt/LIGHT controls
retain different-state pairs as unassessable, with per-cell denominators.

Every eligible branch must additionally have at least one passing, comparable
receipt/LIGHT pair whose two samples are assigned to that branch. Assignments
are still not native OFF phase labels. A branch absent from controls makes an
otherwise passing result **inconclusive**, not passed. Before/after anchor
endpoint scores, incremental observer checks, independent comparison to TRAIN
and own/native LIGHT work checks all remain separate. Existing extraction and
native audit enforce token identities, chronology, output counts, lifetimes,
dispatcher/source/runtime and CPU acceptance. No invalid sample is discarded.

Canonicalize nested row key ordering before fitting/reporting so saved JSON can
be independently refitted without insertion-order changes in quality arrays.
The post-shutdown checker rebuilds all raw rows, refits TRAIN, rebuilds the entry
freeze, verifies seal chronology and compares the complete report. A fabricated
self hash does not satisfy that audit. Hardware boundaries are mocked only in
unit tests, never in the native checker.

A complete run may pass, fail or be inconclusive. All outcomes are archived;
there is no automatic retry, offset removal, tolerance widening or family
selection after seeing held data. The family was developed using calibration
evidence, so only fresh held outcomes can confirm it. Six repeats need not cover
all predicted branches. No result here promotes production model, HTTP, adaptive
cap or MPC flags. Those remain separate later scopes.

## Rental commands, after CPU review

From the project directory on a fresh RTX4090/Linux/Python3.12 host:

```bash
python3 scripts/setup_prefill_calibration.py --next queue-confirmation
nohup .venv/bin/python scripts/prefill_host.py queue-confirmation > ~/pr32-queue-confirmation.log 2>&1 < /dev/null &
```

Preparation runs full exact-checkout native CPU acceptance. The 2400s collection
limit excludes model startup, dependency installation and the final independent
CPU audit. A previous diagnostic took about 16 minutes; no fixed runtime or
successful verdict is promised for the new host. The helper audits complete
pass/fail/inconclusive reports, then archives evidence even when collection or
qualification fails. Keep the archive before returning the machine.
