# PR32 round4: support, magnitude and resolved wins

CPU design revision6 addresses [round4](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6331855173). No new GPU capture, model fitting, C usage or MPC result. Earlier reports remain immutable historical snapshots.

## ITL results with a shared step-scale reference

All299 public evidence files verify. Every disjoint exporter bucket remains archived, with matched first/last scrape endpoints. For EACH window, fix floor=5*(linear-B2048 total gaps / decode-bearing steps), then use that SAME floor for all candidate magnitude scores. This is a toy-equivalent-step normalization, not measured real steps, effective sample size or a noise envelope. Zero-support mismatches remain separate primary evidence even below the material flag. The legacy floor5 headline is superseded.

| Window | Shared floor (gaps) | Linear magnitude | k1.5 magnitude | Material support mismatch linear / k | Common-positive linear / k | Repeat context† |
|---|---:|---:|---:|---|---|---|
| old32 r1 | 86.07 | 1.59 | 1.74 | yes / no | 1.55 / 1.74 | 1.52 / 1.80 / 1.79 |
| old32 r2 | 82.97 | 1.55 | 2.67 | yes / no | 1.55 / 1.68 | 1.52 / 1.80 / 1.79 |
| old16 r1 | 42.03 | 2.30 | 2.50 | no / no | 2.30 / 2.50 | 1.52 / 1.80 / 1.79 |
| new32 r1 | 73.31 | 2.13 | 2.47 | yes / no | 2.13 / 2.47 | 1.52 / 1.80 / 1.79 |
| new16 r1 | 44.37 | 3.78 | 4.10 | no / no | 3.78 / 4.10 | 1.52 / 1.80 / 1.79 |
| transfer60–120 | 72.38 | 2.57 | 2.97 | no / no | 2.57 / 2.97 | 1.52 / 1.80 / 1.79 |
| transfer180–240 | 68.85 | 3.08 | 3.47 | no / no | 3.08 / 3.47 | 1.52 / 1.80 / 1.79 |

Magnitude=exp(max absolute normalized log-rate ratio) across ALL bins starting at25ms. Support flag means one side is zero and the other has at least5 reference-equivalent steps of gaps. Every sub-floor support mismatch is also archived; “no” is not an agreement claim. Common-positive sensitivity uses a shared observed/linear/k1.5 mask, excludes support differences and cannot replace the primary evidence. Floors1/5 remain sensitivity outputs. All window-specific floors and source hashes are archived.

†Repeated numbers are descriptive CONTEXT, not window-specific estimated noise or acceptance thresholds:

| Reference | Factor (both legacy5 and shared-step floor) | Scope |
|---|---:|---|
| same_host_old32_replicates | 1.5157 | Same host, one pair;60–300s requested |
| cross_host_32 | 1.8035 | Cross-host difference; not repeat noise;60–300s requested |
| cross_host_16 | 1.7887 | Cross-host difference; not repeat noise;60–300s requested |

These240s pairs are not matched60s transfer-window replicates; old32 same-host noise is not a measured cap16/new-host envelope. One same-host pair cannot calibrate factor2. No score below2 automatically accepts a model. The revised magnitude ranking favors linear over k1.5 in all seven windows; positive k has no demonstrated mechanistic support. Its ability to add a few long gaps does not distinguish nonlinear service from graph fallback, scheduling or observation effects.

## Gap clustering

The toy archives decode-bearing step totals, per-bin step counts and gaps/step ranges for every scenario. Its150–200ms gaps are correlated within steps. For the observed counts below, equivalent steps divide by the FIXED linear reference mean; these are NOT actual executed step counts.

| Window | Observed150–200ms gaps | Equivalent reference steps | Linear toy steps in bin | k1.5 toy steps in bin |
|---|---:|---:|---:|---:|
| old32 r1 | 137 | 7.96 | 0 | 13 |
| old32 r2 | 88 | 5.30 | 0 | 13 |
| old16 r1 | 0 | 0.00 | 0 | 5 |
| new32 r1 | 109 | 7.43 | 0 | 11 |
| new16 r1 | 0 | 0.00 | 0 | 0 |
| transfer60–120 | 42 | 2.90 | 0 | 2 |
| transfer180–240 | 32 | 2.32 | 0 | 3 |

Future uncertainty must resample measured steps or predeclared time blocks, never independent token gaps. Existing scores were descriptive, with no p-values or independent-gap confidence claims. The new normalization does not manufacture independence.

## Gate3: resolved improvements, not epsilon wins

A win now requires comparator MAE minus candidate MAE >max(3*jitter[cell],25ms), using exactly the cell’s calibration-only eligibility threshold. Numerical equality within1e-12 tolerance and near-ties are non-wins. Require>=4 eligible and>=ceil(.75*n) resolved wins for EACH calibrated comparator. Insufficient wins are inconclusive; any material regression on any of8 cells fails. All8×5 finite paired errors and calibrated comparator requirements remain mandatory. Component success never passes the complete experiment.

[Deterministic counterexamples](win-margin-counterexample.json) reproduce four eligible cells, three1µs improvements: the old win count meets3-of4, the revised component is inconclusive. Exactly25ms is also inconclusive;26ms passes the component; a regression on noneligible cell7 fails. This is not an experiment false-positive estimate. The review’s30.9% concerns only its synthetic old win-count component.

The [recalibration counterexample](comparator-counterexample.json) still passes the raw-baseline component even with meaningful margins; calibrated M0 remains structural N/A. This artifact labels its current-margin component explicitly rather than claiming to execute old code. Jitter mappings remain proxies: longer median latency does not guarantee greater variance or conservatism.

## Transfer replication and budget

Four future attempts: A-r1, B-r1, B-r2, A-r2. A replay40–120/score60–120; B replay160–240/score180–240. Each replays80s with20s unscored lead-in, natural EOS, identical engine/config, unique attempt ID, full drain and boundary/identity logs. One attempt allowed; retain all failures, no retries or outcome-based reordering. Two repeats per window remain descriptive, not a well-estimated statistical envelope. No C.

The212 micro/bridge/deep trials are unchanged. Transfer replay grows160→320s. Explicit transfer terminal reserve is40s (10s per attempt), alongside the original120s general frontend/flush reserve. Total1517.7774s=25.30min, +200s from revision5, under1800s. Hard deadline includes drains/flush; reserve is a planning allowance, not measured runtime. It is incorrect to call the replication change budget-neutral.

## Validation and remaining work

**130 related offline tests pass.** Tests cover meaningful/equal/epsilon margins, noise-specific resolution, all-cell regressions, shared reference/support arithmetic, real artifact source hashes, frozen plan regeneration, calibrated comparisons and prior simulator/identifiability checks. No learned-source or coefficient refitting. The7 full-vector results and3 repeat comparisons derive from verified archived captures; hashes/provenance remain committed.

Collector/hooks, graph-aware accounting/causal rollout, exact token manifests, trace slicing and exact-revision native collector/GPU acceptance remain pending. This is a draft design/CPU-tools checkpoint, not a ready-to-run GPU protocol. No rental is needed for these changes.

[Full vectors](itl-scenarios.json) · [Tests](tests.txt) · [Validation](validation.json)

```bash
python scripts/build_prefill_design.py --output-dir /tmp/fresh-prefill-plan
python scripts/check_prefill_itl.py --old-evidence OLD --new-evidence NEW --output /tmp/fresh-itl.json
python scripts/check_prefill_comparators.py --output /tmp/fresh-comparators.json
python scripts/check_prefill_win_margin.py --output /tmp/fresh-win-margin.json
python -m pytest -q tests/test_prefill_design.py tests/test_prefill_identifiability.py tests/test_prefill_review_gates.py tests/test_cap_miss_causes.py tests/test_greybox_simulator.py tests/test_low_cap_frozen.py
```
