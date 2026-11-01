# Revision9: paired structural groups and explicit noise

CPU-only synthetic results. No GPU collection, C use or MPC validation.

## Protocol change

Mixed bursts at50/400ms now use [512,512,7680,7680]; single-at200 uses7680.
Uniform4x4096-at800 remains a control. Counts remain22 cells/212 trials.
The frozen linear budget is1518.29s versus1517.78s previously, below1800s.
7680+128 leaves384 tokens below8192; exact input token manifests are still required.
The single7680 test extends beyond the7168 calibration maximum: bounded length
extrapolation, not pure interpolation. Mode-aware actual feature support remains a gate.

Eligibility is unchanged per cell: prediction gap >max(3*jitter,25ms).
Count TWO fixed groups, mixed_burst and single_long. Every constituent cell must
be eligible; the uniform control is never another structural contrast.
Within each group/block average paired absolute-error improvement across member
cells, then compute the mean and sample SD over FIVE blocks. Both groups need
mean gain >max(t(.95,4)*SD/sqrt(5),5ms), with1e-12s numerical tolerance.
The5ms engineering floor is frozen before this rerun, not fitted to its outcomes.
Equality and deterministic1us gains are non-wins. This is NOT lower-bound>5ms.
All-eight-cell regression guards remain. Workload revision8 is unchanged;
table revision numbers identify workloads, all scored with the revision9 gate.
Calibrated M0 and retained M1 must each satisfy this rule. M0 is structural N/A.
The .75 cell-win tally and reuse of3*jitter as a required MAE gain are superseded.

This is a one-sided paired-t engineering screen. Independent approximately normal
block differences are assumptions; cells inside a group are not independent n.
Cross-block correlation is explicitly stressed below, not fixed by grouping.
The public gate defaults block_dependence_status to unresolved and cannot pass.
Qualified status needs a preregistered calibration-only dependence assessment;
unresolved/violated status blocks acceptance while preserving nominal t diagnostics.
The synthetic AR1 profile is known violated. Iid/shared-within-block profiles
are independent across blocks by construction, not evidence about actual GPU noise.
No full-experiment5% false-positive claim is made. Source for paired differences:
https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm

## Held measurement noise (no training noise)

| Revision | Sigma | Noise | Eligible groups vs M0 | Nominal t pass | Qualified pass |
|---|---:|---|---|---:|---:|
| 7 | 8ms | iid | mixed_burst | 0.00% | 0.00% |
| 7 | 8ms | shared_block | mixed_burst | 0.00% | 0.00% |
| 7 | 8ms | shared_block_ar1 | mixed_burst | 0.00% | 0.00% |
| 7 | 10ms | iid | mixed_burst | 0.00% | 0.00% |
| 7 | 10ms | shared_block | mixed_burst | 0.00% | 0.00% |
| 7 | 10ms | shared_block_ar1 | mixed_burst | 0.00% | 0.00% |
| 7 | 12ms | iid | mixed_burst | 0.00% | 0.00% |
| 7 | 12ms | shared_block | mixed_burst | 0.00% | 0.00% |
| 7 | 12ms | shared_block_ar1 | mixed_burst | 0.00% | 0.00% |
| 7 | 15ms | iid | none | 0.00% | 0.00% |
| 7 | 15ms | shared_block | none | 0.00% | 0.00% |
| 7 | 15ms | shared_block_ar1 | none | 0.00% | 0.00% |
| 8 | 8ms | iid | mixed_burst, single_long | 99.94% | 99.94% |
| 8 | 8ms | shared_block | mixed_burst, single_long | 97.90% | 97.90% |
| 8 | 8ms | shared_block_ar1 | mixed_burst, single_long | 95.55% | 0.00% |
| 8 | 10ms | iid | mixed_burst, single_long | 98.83% | 98.83% |
| 8 | 10ms | shared_block | mixed_burst, single_long | 91.30% | 91.30% |
| 8 | 10ms | shared_block_ar1 | mixed_burst, single_long | 88.72% | 0.00% |
| 8 | 12ms | iid | mixed_burst | 0.00% | 0.00% |
| 8 | 12ms | shared_block | mixed_burst | 0.00% | 0.00% |
| 8 | 12ms | shared_block_ar1 | mixed_burst | 0.00% | 0.00% |
| 8 | 15ms | iid | mixed_burst | 0.00% | 0.00% |
| 8 | 15ms | shared_block | mixed_burst | 0.00% | 0.00% |
| 8 | 15ms | shared_block_ar1 | mixed_burst | 0.00% | 0.00% |

At10ms the single-long gap is only about30.2ms against a30ms eligibility
threshold. This is fragile. At12–15ms it is ineligible and the two-group gate
remains inconclusive even with a perfect M2. Do not lower thresholds or expand
the matrix after viewing held measurements. Historical revision7 had one group.

## Equal-performance and wrong-model controls

| Null | Sigma | Noise | Nominal t pass | Qualified pass |
|---|---:|---|---:|---:|
| midpoint | 8ms | iid | 0.14% | 0.14% |
| midpoint | 8ms | shared_block | 2.86% | 2.86% |
| midpoint | 8ms | shared_block_ar1 | 13.69% | 0.00% |
| baseline_truth | 8ms | iid | 0.00% | 0.00% |
| baseline_truth | 8ms | shared_block | 0.00% | 0.00% |
| baseline_truth | 8ms | shared_block_ar1 | 0.00% | 0.00% |
| midpoint | 10ms | iid | 0.21% | 0.21% |
| midpoint | 10ms | shared_block | 2.99% | 2.99% |
| midpoint | 10ms | shared_block_ar1 | 13.93% | 0.00% |
| baseline_truth | 10ms | iid | 0.00% | 0.00% |
| baseline_truth | 10ms | shared_block | 0.00% | 0.00% |
| baseline_truth | 10ms | shared_block_ar1 | 0.00% | 0.00% |
| midpoint | 12ms | iid | 0.00% | 0.00% |
| midpoint | 12ms | shared_block | 0.00% | 0.00% |
| midpoint | 12ms | shared_block_ar1 | 0.00% | 0.00% |
| baseline_truth | 12ms | iid | 0.00% | 0.00% |
| baseline_truth | 12ms | shared_block | 0.00% | 0.00% |
| baseline_truth | 12ms | shared_block_ar1 | 0.00% | 0.00% |
| midpoint | 15ms | iid | 0.00% | 0.00% |
| midpoint | 15ms | shared_block | 0.00% | 0.00% |
| midpoint | 15ms | shared_block_ar1 | 0.00% | 0.00% |
| baseline_truth | 15ms | iid | 0.00% | 0.00% |
| baseline_truth | 15ms | shared_block | 0.00% | 0.00% |
| baseline_truth | 15ms | shared_block_ar1 | 0.00% | 0.00% |

Cross-block AR1 midpoint errors have a high nominal false-pass rate (about14%).
They are blocked by the dependence qualification; do not quote an iid result
as assurance about a correlated GPU run. Five blocks cannot establish independence.

## Calibration-label noise and joint sensitivity

Three calibration trials per cell are generated. LOCO holds out the entire
cell including all its repeats; scales and fits never see that cell.
Multiplicative noise is mean-one lognormal (positive labels), unlike an exact
reproduction of the review Gaussian-label experiment. Only labels change;
true work positions and first-token membership stay fixed. This isolates label
uncertainty and does not simulate physical slowdowns changing scheduling.

| k | CV | Labels | M2 selections | Label-only pass | Joint iid10ms | Joint shared/AR10ms nominal | Joint iid15ms |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 1% | iid | 0/40 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0 | 2% | iid | 0/40 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0 | 5% | iid | 0/40 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0 | 1% | shared_trial | 0/40 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0 | 2% | shared_trial | 0/40 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0 | 5% | shared_trial | 0/40 | 0.00% | 0.00% | 0.00% | 0.00% |
| 1.5e-09 | 1% | iid | 40/40 | 70.00% | 69.12% | 62.11% | 0.00% |
| 1.5e-09 | 2% | iid | 40/40 | 57.50% | 56.49% | 50.65% | 0.00% |
| 1.5e-09 | 5% | iid | 26/40 | 37.50% | 34.49% | 31.10% | 0.00% |
| 1.5e-09 | 1% | shared_trial | 40/40 | 70.00% | 69.03% | 62.40% | 0.00% |
| 1.5e-09 | 2% | shared_trial | 31/40 | 60.00% | 58.20% | 52.60% | 0.00% |
| 1.5e-09 | 5% | shared_trial | 15/40 | 35.00% | 32.59% | 30.65% | 0.00% |

Shared/AR rates in the table are nominal diagnostics; qualified pass is zero.

The JSON archives every training seed, selected family, coefficients, LOCO
mean errors, predictions, eligibility and held-result counts. Shared training
fits mean held Monte Carlo draws are not independent full experimental runs.

## Calibration-only readiness

After the first three calibration blocks and before model/held-score freeze,
compute repeat CV within matched executed-work, request-position, mode and
padding strata. Freeze expected strata before opening labels; do not pool
different chunk sizes or discard unsupported/missing strata. Use one duration
per independent trial, not individual token gaps as repeat observations.
`calibration_noise_readiness` rejects incomplete/nonfinite repeats and stops
structural claims if any matched-stratum CV exceeds the preregistered5% limit.
Above2% is flagged. This is a conservative engineering limit with only three
repeats, not a population-CV confidence bound. Within range never accepts a model.
Keep held timing sealed; no resampling/retry or threshold tuning from held results.
Measured sigma/CV and actual two-group support must accompany any later claim.
A calibration-only serial-dependence assessment must be specified and reviewed
before executable freeze. Five held blocks cannot qualify independence; lack of
a significant autocorrelation test is not evidence of independence. Until that
assessment exists and supports the assumptions, dependence remains unresolved.

Native collector/source mapping, matched-stratum identity, scheduler parity,
mode-aware causal rollout and exact token manifests remain pre-freeze blockers.
Preserve Azure development transfer; controlled tests and real-load transfer
remain complementary. The fixed128-output replay scope is unchanged.

Reproduce: `python scripts/check_prefill_noise.py --output-dir /tmp/prefill-noise-fresh`.
All previous reports are historical snapshots at their recorded revisions.
