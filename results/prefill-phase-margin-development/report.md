# Review integration: TRAIN-bound phase margins

The CPU proposal implements the useful parts of
[the latest PR32 review](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6437204879)
and adds strict calibration binding and exact continuous phase analysis. It does
not change the historical rejection, native `clock`/`live-http` gates, coefficients
or numeric error budgets. **Both the historical attempt and this development
proposal remain rejected.** No new native collection or model/HTTP/cap/MPC
acceptance is claimed.

## Review response and implementation choices

| Review suggestion | Integrated approach |
| --- | --- |
| The late OFF samples are near another model boundary. | Preserve both nominal boundary points when the TRAIN-derived scale envelope predicts an input-iteration flip. The late samples fit that branch within the existing budget. |
| Estimate spread from calibration only. | Use the maximum absolute relative TRAIN OFF endpoint residual across all cases/repeats/positions/metrics: s=0.04626523891083423. The held-derived ±2–3% observation is not a calibration input. |
| Freeze phase ambiguity before held scoring. | Freeze policy, training, model, point-forecast, plan and runtime hashes, along with the entire branch set. Evaluation regenerates it from TRAIN; rehashed wider points/eligibility are rejected. |
| Score two branches for ambiguous cells, keep point rules otherwise. | Freeze at most two nominal poll-residence points; score every OFF/LIGHT sample at the original max(10ms,20%) budget. Stable cells also retain their point-median rule. No held sample is excluded. |
| Separate expected phase changes, same-phase shifts and model error. | Separate prediction-budget and measurement-qualification flags; retain expected phase differences as diagnostics. A residual miss is not labeled proof of a specific physical cause. |
| Alternatively move arrivals to mid-interval. | Keep the hard offsets and original cases. Moving them would avoid the boundary behavior this proposal needs to expose. |

An additional improvement over a scaling table is exact piecewise-affine
max-plus propagation through the model's own no-probe graph. It keeps fixed idle
yields, barrier conditioning, every linear intersection and tie. Six source pins
and a readiness check constrain this algebra to the reviewed effective lane.
Unsupported timing assumptions, excessive complexity, drained work or more than
two branches fail closed. A finite grid cannot silently certify no flip.

## Development result

The unchanged archive is bound by SHA256
`2e9ef84b537857652251cb17e89b485a5d6835a333a630ee2a96dd78241649fd`.
Its original 4c89188 source audit still verifies all 167 source files and original
1180-case native CPU acceptance with candidate_passed=false. No six-term
coefficient was refitted. The current proposal's extra quantities come only from
its original TRAIN OFF rows and model-generated graphs.

| Cell | Frozen classification | Branch residence points |
| --- | --- | --- |
| 400ms/D16 | phase_ambiguous | 6.698ms / 175.183ms |
| 800ms/D0 | phase_ambiguous | 0ms / 143.297ms |
| Other six delayed cells | point | Original point prediction |

All 64 held queue samples and 32 observer pairs remain in the report. Three
pairs are expected phase differences under the frozen branch set. Three pairs
still fail model-assigned same-phase observation qualification, and three queue
samples miss their prediction budget:

| OFF trial | Remaining error | Why it remains a failure |
| --- | ---: | --- |
| ct-086, 400ms/D16 | 13.700ms | Early branch miss exceeds the 10ms floor. Its paired difference is 11.773ms. |
| ct-118, 400ms/D16 | 16.312ms | Early branch miss exceeds the 10ms floor. Its paired difference is 12.717ms. |
| ct-120, 50ms/D16 | 83.010ms | The TRAIN-derived common-scale envelope predicts no flip here; this sample is unexplained. |

There is also a within-model-phase OFF repeat failure at 50ms/D16. The phase
model cannot excuse an outlier in a cell it classified as stable. The 400ms/D16
OFF counts are 2 early / 2 late; LIGHT counts are 4 early / 0 late. Assignments
mean nearest model point, not observed native phase. Zero-count branches are
explicitly unobserved, and one-sample repeat groups are underpowered. No branch
probability, unseen-branch accuracy or tail coverage follows from these counts.

Endpoint scores and separate anchor scores match the original report exactly.
The new report independently exposes endpoint/structure, queue-residual and
measurement flags. The raw strict queue diagnostics and point verdict remain
available. Expected multimodality alone is no longer mislabeled as a large point
error in this separate proposal, but residual/measurement failures still block it.

## What remains before another GPU run

The source-derived no-probe timing algebra is exact within its declared shared
scale family; the family itself is a hypothesis. Calibration endpoint residuals
do not establish independent per-stage errors, input-queue phase-origin shifts,
mode-specific timing corrections or probabilistic coverage. OFF internal input
poll boundaries were not measured in this capture. In particular, the 50ms/D16
outlier shows that a shared scaling family does not explain every retained case.

Further protocol design should add calibration-only queue probes or receipt-state
observations that can identify those missing quantities. Do not enlarge `s`, add
a third convenient branch, shift early points or move hard arrivals using held
labels. The proposed qp- IDs/content keys are distinct, but the native collection
entry is intentionally not enabled by this CPU change. The unchanged old run is
not the requested next rental experiment.

## Reproduction and verification

See [the full policy and assumptions](../../experiments/prefill-phase-margin-v1.md).
The CPU command audits the raw archive and original source before development:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/develop_prefill_phase_margin.py \
  --capture /path/to/1793086097723591624 \
  --archive /path/to/1793086097723591624.tar.gz \
  --output /new/output/directory
```

`phase-freeze.json`, `phase-report.json`, `policy.json`, `proposed-protocol.json`
and `summary.json` preserve the current development result. `input-files.json`
binds every raw file. The supplied archive remains available through the
[prior native diagnosis](../prefill-clock-4c89188-queue-phase/report.md).
This policy was developed after seeing the old rejection; replaying its held
measurements is retrospective development, not fresh independent confirmation.

`cpu-validation.json` and the included JUnit reports record local regression
verification. Tests cover held-label poisoning, rehashed point/eligibility
tampering, narrow interior affine crossings, comparison against independent
existing rollouts, expected switches passing without wider budgets, remaining
native failures, unresolved ties and unsupported source/branch guards. Local CPU
tests establish implementation behavior, not native timing accuracy.

The broader local run passed 581 cases (one warning, 198.78 seconds). A final
40-case phase/queue/host run passed after self-review added explicit rejection of
coincident commit anchors; it includes all 18 phase-margin cases. The broad run
preceded that final defensive guard, and the final focused run includes it.
Neither run has failures, errors or skips; their counts overlap. The new policy
JSON is also included in exact-source acceptance fingerprints, so changing it
cannot reuse a previous native CPU acceptance record.
