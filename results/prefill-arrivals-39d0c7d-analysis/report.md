# First native delayed pilot: valid evidence, rejected candidate

**The pilot completed and its original independent audit is valid, but the candidate correctly fails one preregistered sentinel queue gate.** Work order, observation quality, all primary endpoint errors and both anchor phases pass. No model/threshold/acceptance is changed to erase this failure.

## Evidence

- Captured source `39d0c7d6aa02e9667a5b67bc94740f5d2148264a`; all148 source files match the reviewed Git inventory.
- Archive `1793057749935383344.tar.gz`: 9,089,762 bytes, SHA256 `43594328c4d355406ca97a11a55d60f53bc262b39203dc65ba24e50e7f76382b`. All156 extracted files match the archive; individual byte hashes are retained.
- Original native JUnit verifies1078 passing tests with required trace configurations. Source/manifest/installed38-file runtime identities and saved forecasts/report are checked by the ORIGINAL audit reconstructed from Git.
- All136 trials complete once, no collection exceptions, elapsed579.168s. Original exit2 is a valid candidate rejection, not an interrupted collection.

| Gate | Result |
|---|---|
| anchor_after | PASS |
| anchor_before | PASS |
| endpoints | PASS |
| quality | PASS |
| sentinel_queue | FAIL |
| structure_and_background_lifetime | PASS |

## Primary endpoints

Eight delayed cells cover40 foreground/sentinel positions, each scored by the median of four OFF repeats. These correlated positions are not40 independent experiments, and four repeats do not establish tail accuracy.

| Endpoint | Mean absolute error | Largest absolute error | Largest relative error |
|---|---:|---:|---:|
| Target to first output | 10.966ms | 39.722ms | 9.635% |
| Target to final output | 50.760ms | 74.183ms | 2.996% |

All original endpoint max(5ms,20%) TTFT / max(10ms,20%) completion limits pass. This does NOT override the independently preregistered queue gate. Anchors pass before AND after, and native/own-work/background-lifetime checks pass.

## Queue failure

| Arrival / background | Measured residence | Predicted residence | Error | Limit |
|---|---:|---:|---:|---:|
| mixed-plus-sentinel-at50-D0 | 62.713ms | 62.821ms | 0.108ms | 12.543ms PASS |
| mixed-plus-sentinel-at200-D0 | 25.804ms | 30.121ms | 4.317ms | 10.000ms PASS |
| mixed-plus-sentinel-at400-D0 | 107.055ms | 114.313ms | 7.258ms | 21.411ms PASS |
| mixed-plus-sentinel-at800-D0 | 116.731ms | 123.023ms | 6.292ms | 23.346ms PASS |
| mixed-plus-sentinel-at50-D16 | 78.079ms | 78.590ms | 0.511ms | 15.616ms PASS |
| mixed-plus-sentinel-at200-D16 | 50.885ms | 48.386ms | -2.499ms | 10.177ms PASS |
| mixed-plus-sentinel-at400-D16 | 148.476ms | 137.034ms | -11.441ms | 29.695ms PASS |
| mixed-plus-sentinel-at800-D16 | 29.092ms | 8.328ms | -20.765ms | 10.000ms FAIL |

At800ms/D16 the predicted8.328ms versus observed29.092ms differs by20.765ms, exceeding the fixed10ms limit. The other seven queue cells pass. The gate remains failed; no change to10ms or20%, no retry/best-run selection, no held refit.

## Three-clock and matched-work diagnosis

The four OFF posting lags are 0.116–0.139ms, well within5ms. Their engine enqueue times are 828.089–829.503ms after release; queues consistently wait about29ms. This is not explained by late generator posting.
The model enqueues at 808.328ms after its own predicted barrier. In LIGHT runs the nearest schedule boundary is the same executed-work index17, at 829.570–830.437ms; the model places the same work boundary at 808.328ms. Native schedule starts are only 0.093–0.128ms after the recorded injection.

Inference: the reduction gets discrete work order right but its event clock reaches this input-consumption boundary roughly21ms too early. That small fraction of overall1.2s first-output latency becomes a large error in the short residual queue wait. Endpoint feasibility accuracy is insufficient to qualify arrival-phase predictions. These observations implicate effective timing fidelity; they do not distinguish an inadequate four-feature law from endpoint-only fitting, or identify exclusive GPU/CPU/GC/driver causes.

## Changes and next step

This change adds a read-only source-bound analyzer and improves rejection reporting. Future failed runs now print the original failing gate/case, prediction, observation, error and threshold, and store rejection_reasons in the summary. No scoring formula, old data, frozen parameters, control policy, original full-suite gate or native GPU wrapper changes.

Next work can proceed on CPU with the retained original TRAINING LIGHT captures: test absolute schedule-start/output-commit event times as calibration targets, aligned from initial request injection and using own-generated work. Overlapping wall durations must never be summed. Training-only model development must freeze its objective and support before fresh confirmatory delayed trials. Current delayed held data is now diagnostic/development evidence and cannot be reused as independent confirmation after model changes. Do not request another rental merely to repeat the unchanged failed model. HTTP, dynamic cap and MPC remain unqualified.

## Reproduce

```bash
python scripts/analyze_prefill_arrival_capture.py --capture /path/to/1793057749935383344 --archive /path/to/1793057749935383344.tar.gz --output /tmp/new-arrival-analysis
```

The analyzer verifies raw archive bytes and exact148 Git source files, runs the ORIGINAL check_prefill_arrivals.py audit and accepts audit exit2 only when it agrees with a valid failing-candidate verdict. It reuses no changed source-identity gate and does not run GPU/native CPU acceptance locally. Phase comparisons are emitted only when work structure passes. Hashes are artifact consistency checks, not signatures. Local test scope is recorded in validation.json.
