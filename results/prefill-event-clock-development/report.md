# Batch-clock calibration: retained-data development result

The original delayed pilot remains **complete, valid and rejected**. The new
model satisfies all old numeric rules on the now-retrospective delayed data.
This is development evidence, not fresh independent validation. No GPU was run
locally and model/physical-cost/full-suite/cap/MPC acceptance remains false.

## Inputs and fitting

Archive1793057749935383344.tar.gz SHA256
43594328c4d355406ca97a11a55d60f53bc262b39203dc65ba24e50e7f76382b,
9,089,762 bytes. A clean copy exactly matches every archived byte. The original
39d0c7d audit reproduces its rejected verdict from148 verified captured source
files and original native JUnit1078 passing cases. The uploaded archive, original
freeze and thresholds are unmodified. An initial working-copy inventory mismatch
was rejected; analysis used a fresh exact archive extraction, preserving the
strict inventory check.

Fit six positive effective coefficients using only the six original TRAIN cells:
24 OFF endpoint trials and24 LIGHT event trajectories. Check native integrity,
full own-work signatures and the foreground completion prefix before using event
labels. Fit absolute schedule/commit points relative to each barrier; prediction
uses its own release. Do not add overlapping wall labels. Give prefill and decode
event phases equal total weight, retaining existing OFF endpoint weights. The
new terms are decode count and simultaneous prefill/decode interference, not
exclusive hardware costs. The new cost schema explicitly serializes the mixed
term; generic stage-cost loading rejects it.

Frozen parameters in [bias,prefill,decode_context,attention_pairs,decode,mixed]
normalized seconds:

`[0.009465982483362589, 0.09198428776091734, 0.005465854090231256, 0.010425437177291807, 0.00108264619394694, 0.004771251503057694]`

Both fixed optimizer starts converge and agree; normalized local sensitivity has
rank6 and condition10.012911. This is local sensitivity, not global identification.
Training LIGHT trial/phase-balanced event MAE improves from19.025ms to10.177ms.
Four training-only leave-one-block-out fits all pass original endpoint gates;
left-out event MAEs are10.316, 9.858, 9.976, 10.644ms. Coefficient full ranges/original are0.029, 0.214, 0.462, 0.470, 1.877, 0.738%.
These repeat-block diagnostics are not confidence intervals or workload/host
independence. LIGHT event overhead has not been separately qualified as physical
or uninstrumented production service.

## Retrospective delayed scoring

| Case | Observed queue ms | Original prediction ms | New prediction ms | New signed error ms | Unchanged limit ms |
|---|---:|---:|---:|---:|---:|
| at200-D0 | 25.804 | 30.121 | 19.511 | -6.293 | 10.000 |
| at200-D16 | 50.885 | 48.386 | 47.686 | -3.199 | 10.177 |
| at400-D0 | 107.055 | 114.313 | 107.630 | 0.575 | 21.411 |
| at400-D16 | 148.476 | 137.034 | 149.888 | 1.413 | 29.695 |
| at50-D0 | 62.713 | 62.821 | 56.668 | -6.045 | 12.543 |
| at50-D16 | 78.079 | 78.590 | 77.724 | -0.355 | 15.616 |
| at800-D0 | 116.731 | 123.023 | 116.850 | 0.119 | 23.346 |
| at800-D16 | 29.092 | 8.328 | 29.817 | 0.725 | 10.000 |

All eight queue cells, primary endpoints, separate before/after anchors, work and
background lifetime, and existing observer/repeat rules pass this development
check. The formerly failed800ms/D16 case improves from absolute20.765ms error to
0.725ms. Across40 request positions (median of four OFF repeats), target-to-first
MAE is2.818ms and maximum6.414ms. Target-to-final MAE remains50.145ms; maximum
72.693ms. These are request-position medians, not per-request tails or p99.
The completion error has not materially improved and is reported explicitly.

The model design was informed by the prior delayed failure. Forecasts/parameters
are saved before this run opens the delayed scoring rows, but this procedural
separation cannot make previously inspected data fresh. Retrospective agreement
cannot promote the historical rejected result or grant cap/MPC acceptance.

## Exploratory attempts retained

1. Four original terms with an unbalanced event objective: converged training
   fit, but retrospective800ms/D16 queue error about110.0ms. Other queue failures
   also appeared. Replacing endpoint loss alone is insufficient.
2. Add decode count with that objective: training SSE falls21.4332 to12.6064;
   old800ms/D16 queue error remains17.325ms, over the unchanged10ms limit.
3. Equal prefill/decode phase weighting with five terms: training SSE19.3728
   under the revised objective; no confirmatory claim. SSEs across different
   weight definitions are not directly comparable.
4. Add the nonnegative mixed-work term with the phase-balanced objective:
   training SSE16.1810 in exploratory fitting; the final audited extraction uses
   the exact prefix through foreground final commit and independently repeats
   fitting/freezing/cross-validation. See optimizer_runs in frozen.json.

Exploratory parameter vectors and objectives are retained in exploratory-attempts.json.
No delayed labels enter numerical coefficient fitting. Later model design did
use earlier development findings; no independent held-selection claim is made.
The final model is a new candidate, not a repaired historical measurement.

## Implementation and self-review

The dedicated future `prefill_host.py clock` operation has a frozen136-trial plan
with new ct- trial IDs/content keys, fresh same-host training, pre-confirmation
freeze, independent raw-event audit and failed-run archives. All old endpoint,
queue, posting, observer, work and anchor rules remain. Every launch resets the
new flag; endpoint conflicts and stale source/native CPU acceptance fail.
Legacy v1/v2 models, native worker hooks and the serial-overlap guard are unchanged.

**445 related local CPU tests pass in203.29s, zero failures/errors/skips**, including30 new event-clock tests. Compilation and registered source fingerprints pass. Exact validation/source hashes are in validation.json and the JUnit.
Synthetic recovery, held poison isolation, corrupt event/clock/freeze rejection,
optimizer/rank/bound failures, mixed-term serialization, fresh manifest separation,
endpoint/host mode isolation, partial archives and freeze-before-confirmation are
tested. An initial integration fixture counted only48 training rows at the fit
call; the collector also retains16 warmup rows, while the fitter selects TRAIN
only. The fixture was corrected to64 stored rows/24 event trajectories. No
selection rule or model gate changed for that correction.

No local Torch/vLLM/GPU or verified native procfs acceptance is available; no
native resource gate was bypassed. The next GPU step is fresh confirmation via
[the documented clock operation](../../experiments/prefill-event-clock-v1.md),
not a rerun of the rejected four-term model. HTTP/state observability, state-aware
cap comparisons and MPC remain downstream.
