# PR32 round3: distribution checks and calibrated comparators

CPU-only revision5 responding to the round3 review of1c05760f. No new GPU capture, fitted physical model, C usage or MPC/controller result.

## Full histogram comparison

All299 public evidence files verify. Every exporter bucket is archived. Joint scores use disjoint intervals starting at25ms, normalized by total gaps with count floor5. The reported factor is exp(max absolute log rate ratio); factor2 is an engineering discrepancy flag, not model acceptance or a significance test. Snapshot sent-time endpoints are matched approximately; legacy nominal60–300s counts remain separately labelled.

| Capture | Linear B2048 worst-bin factor | k1.5e-9 factor |
|---|---:|---:|
| cap-range-v1-fixed-32-r1 | 27.41 | 1.74 |
| cap-range-v1-fixed-32-r2 | 17.58 | 2.67 |
| cap-range-v1-fixed-16-r1 | 2.30 | 5.80 |
| low-cap-switch-v1-fixed-32-r1 | 21.77 | 2.47 |
| low-cap-switch-v1-fixed-16-r1 | 3.78 | 4.10 |

The nonlinear old32r1 case is below the factor2 flag; this does NOT establish acceptance or support k. Other captures and both transfer windows show substantial joint discrepancies. A selected150–200ms match cannot confirm the nonlinear mechanism. The coefficient is retained only as a sensitivity; no tail-derived bound is supported.

| Transfer score window | Linear B2048 factor | k1.5e-9 factor |
|---|---:|---:|
| [60,120) | 8.41 | 2.97 |
| [180,240) | 6.42 | 3.47 |

Transfer vectors use the historical admitted/output path, empty at the20s lead-in origin. They are conditional diagnostics, not fresh-host admission forecasts. All five historical admitted paths have exactly128 outputs, so forced128 is identical and explicitly marked duplicate. Future natural-EOS transfer settings are preserved; fresh realized lengths and boundary state remain measured inputs for retrospective checks.

## Recalibration is not structural improvement

The synthetic counterexample rescales all three frozen linear coefficients by0.95/1.05/0.90/1.10. At a synthetic5ms jitter and25ms floor it creates6/6/8/8 eligible cells, and the OLD raw-baseline gate3 component passes in all four deterministic toy examples. No model is fitted and no GPU observation is involved.

New contracts require candidate-versus-calibrated-M0 and, for M2 retaining M1, candidate-versus-calibrated-M1. Same fitting data/weights/accounting and causal isolation are required. Selected M0 makes structural comparison not applicable. Raw-B2048 remains staleness-only. Component passing never passes capture integrity, bias/order, causal provenance or the complete experiment. Missing comparators, incomplete error matrices, noneligible regressions and empty structural plans fail closed.

## Position noise and graph confounding

Noise is measured across off trials at each request position, not pooled across positions. Complete20x1 and10x4 matrices are required. A fixed nearest-log-latency mapping from raw-B2048 scenarios is committed before collection. Four@50 uses uniform-D16 position3; four@400 uses position1. Each cell gets its own mapped jitter threshold. Mapping is a proxy, not a cell-specific measured variance or power claim.

Per-step graph mode, actual/padded work, capture/replay and execution IDs are added to the collector specification. Official v0.30.0 dispatcher/source confirms these runtime concepts, not their historical values or causal performance impact. Mode-aware M0-level accounting and same-mode/padding D16 context checks are required before mechanistic claims. The existing five-column toy rank check does not validate that encoding. Actual hooks/encoding/future-mode prediction remain pre-freeze blockers.

## Validation and scope

**122 related offline tests pass.**

Current tests cover full-vector arithmetic, internal resets/identity/schema/sample errors, neighbouring-bin misfit despite a matching tail, position-vs-noise separation, per-cell eligibility, calibrated comparator requirements, and prevention of incomplete/vacuous passes. Existing44 frozen-barrier checks and110 background-prefix checks remain. Fresh generated plans match committed outputs; independent ITL invocations reproduce core results.

Workload212 trials and160s replay is unchanged, estimated1317.78s under1800s active. Collector/hooks, mode-aware encoding/causal rollout, token manifests, trace slicing and native CPU/GPU acceptance are still pending. No user machine is needed for this CPU update.

[All histogram/source data](itl-scenarios.json) · [Synthetic comparator counterexample](comparator-counterexample.json) · [Tests](tests.txt) · [Validation](validation.json)

```bash
python scripts/build_prefill_design.py --output-dir /tmp/fresh-prefill-plan
python scripts/check_prefill_itl.py --old-evidence OLD --new-evidence NEW --output /tmp/fresh-itl.json
python scripts/check_prefill_comparators.py --output /tmp/fresh-comparators.json
python -m pytest -q tests/test_prefill_design.py tests/test_prefill_identifiability.py tests/test_prefill_review_gates.py tests/test_cap_miss_causes.py tests/test_greybox_simulator.py tests/test_low_cap_frozen.py
```
