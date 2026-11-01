# Native controlled queue confirmation at 2ebb185

**The prospectively sealed controlled confirmation passes, and its strict raw
source-bound audit is independently reproduced.** On this host the entry family
and F0 produce identical forecasts. This confirms the predeclared fixed-host
scope; it does not establish a correction advantage or reproduce the earlier
host's phase switch. HTTP, adaptive cap and MPC acceptance remain false.

## Identity and independent reconstruction

Archive `1793136029888154788.tar.gz`: 13,225,010 bytes, SHA256
`c569e2d78ccefdab3ecc8d275181bb320c07f51ba6e983a5fdd1b7e7027858b5`.
All 260 regular files (151,992,353 uncompressed bytes) match the uploaded archive.
All 191 captured source files match Git commit
`2ebb18566b0e4c4751bded54ab6f18fab73abcbb`. The copied native CPU acceptance,
JUnit and log are byte-identical to the archive: **1331 tests**, no unsuccessful
cases, with the required five trace configurations present exactly once.

The original checker is executed in an isolated snapshot of those exact Git
bytes. It revalidates manifest pairing, runtime/source pins, chronology, raw
rows, dispatcher parity, work/lifetimes, TRAIN fitting, the complete freeze,
seal and final report. The independently refitted coefficients, frozen model
and report match the native artifacts under the original strict comparisons.
The first local replay used SciPy 1.17.0 and failed the refit identity check.
Matching the captured NumPy 2.3.5 / SciPy 1.18.1 made the complete strict replay
pass. No data, coefficient, source guard, budget or equality rule was relaxed.

The model is sealed after all 88 warmup/TRAIN trials, before qc-089, the first
held anchor. The seal follows the final TRAIN capture by 25.498s and precedes
the first held capture by 126.883ms. It binds the source, manifest, TRAIN trial
IDs, base model and entry freeze. Native base-model SHA256 is
`0a19c928cdcb208e2fd6fafbe5441d75e96486293c9e89a50756a29bfb904ac5`;
entry-freeze SHA256 is
`5c1570d43dd82443b6b1b8092c35dfb6103e323bb833b7773bc4316cc854e8d9`.

## Complete frozen denominator and measured accuracy

All **240 trials** complete with no collection failures in **1003.371s**:
40 warmups, 48 TRAIN references, 4 before anchors, 144 held queue trials and
4 after anchors. No sample is removed or retrospectively reassigned to TRAIN.
Each of the eight original cells has six held samples in each of the baseline,
receipt and LIGHT arms. The following are individual baseline OFF sample
ranges/errors, not errors against a midpoint or a median queue target.

| Offset / background | Point (ms) | Observed OFF range (ms) | Largest absolute error (ms) |
|---|---:|---:|---:|
| 50 ms / D0 | 55.648 | 57.673–58.830 | 3.183 |
| 50 ms / D16 | 77.297 | 76.687–77.308 | 0.610 |
| 200 ms / D0 | 17.342 | 19.840–21.658 | 4.316 |
| 200 ms / D16 | 46.809 | 48.444–49.398 | 2.589 |
| 400 ms / D0 | 100.505 | 96.626–98.844 | 3.879 |
| 400 ms / D16 | 145.592 | 143.406–145.677 | 2.186 |
| 800 ms / D0 | 103.525 | 102.272–104.166 | 1.253 |
| 800 ms / D16 | 23.178 | 18.238–21.567 | 4.940 |

Baseline OFF queue MAE is **1.920ms**, largest absolute error **4.940ms**.
Receipt and LIGHT MAE are 1.842ms and 1.825ms; their largest errors are 6.444ms
and 4.380ms. All **144/144** held queue samples pass the original
max(10ms,20% observed) budget. F0 point, entry point and phase error counts are
all zero in every arm. All forecast points are shared between F0 and entry.

The endpoint gates retain their original per-case/request median estimand.
Across 40 baseline held request positions, TTFT median-error MAE is 2.415ms,
maximum 5.390ms (maximum relative error 2.650%). Completion median-error MAE is
49.651ms, maximum 75.967ms (maximum relative error 3.089%). These are not
individual-request tail accuracy, p99 or confidence bounds. Receipt/LIGHT
endpoints and before/after anchor endpoint gates also pass.

## Observer, branch, anchor and work checks

All **48/48** receipt/LIGHT pairs have comparable observed states and pass;
each cell has six comparable pairs, with no different-state pair. Largest
comparable queue delta is 3.474ms. Largest incremental endpoint delta is
7.353ms; the largest endpoint observer budget utilization is 34.563% under the
original max(2ms,10%) rule. This does not establish zero observer overhead.

Fresh TRAIN yields **one eligible branch in every cell**. Each has six
comparable passing control pairs. All branch coverage gates therefore pass,
without dropping or adding a branch after held scoring. There is no two-branch
confirmation on this host. OFF internal phase is still unobserved; nearest-point
assignments are not native phase labels or branch probabilities.

Both independent before/after anchor checks, including comparison against TRAIN,
pass. Largest anchor delta is 15.075ms; maximum budget utilization is 31.112%.
All 52 held LIGHT work checks and predicted/native background lifetime checks
pass. Runtime and source identities remain unchanged through collection.

## Interpretation and remaining limitation

Same-host TRAIN estimates P=11.656585ms and D=0.550195ms; the shared-prefill
sensitivity domain is 4.396682–17.719582ms. These are fresh estimates, not old
absolute coefficients copied to a new rental. All 24 TRAIN first-commit sanity
checks pass; maximum absolute error is 5.875ms. GPU coefficients are in the
unaltered native base-model artifact.

The previous host had P=44.914831ms and a much wider/slower return domain;
50ms/D16 baseline waits there split near 0–3ms and 76–77ms. Here all six baseline
50ms/D16 waits are 76.687–77.308ms. F0 and the entry family's endpoint, queue and
work forecasts are **exactly identical on all declared cases**. This run cannot
distinguish their benefit. It must not be reported as proof that a CPU-return
correction won, that the earlier phase switch is solved, or that an uncertainty
branch has measured probability coverage. No cross-host accuracy, unknown HTTP
ingress prediction, cap action validity or closed-loop MPC result follows.

The native `candidate_passed: true` is retained for its exact predeclared
controlled-arrival scope. Global production model/HTTP/cap/MPC acceptance stays
false. The next engineering scope is CPU integration of the frozen controlled
predictor into the separate HTTP state-bridge contract before a new GPU test.
The existing HTTP command does not automatically qualify this entry family.

## Reproduction and CPU checks

Use Python 3.12, NumPy 2.3.5 and SciPy 1.18.1 for exact numerical replay:

```bash
python scripts/analyze_prefill_queue_confirmation.py \
  --capture /path/to/1793136029888154788 \
  --archive /path/to/1793136029888154788.tar.gz \
  --output /path/to/new-confirmation-analysis
```

The output directory must be new and outside immutable evidence. The script
reconstructs the exact historical source from Git and refuses numerical-version
mismatch before fitting. Missing historical Git objects require an explicit
fetch of 2ebb185; no HEAD substitution or automatic fetch is allowed.

**77 related local CPU tests pass** (7.58s), covering summary denominators,
source rejection, identical-forecast limitations and existing confirmation,
audit, entry and receipt-analysis contracts. This count overlaps earlier local
selections and is separate from the captured full native 1331-case acceptance.
A committed metadata/hash regeneration check is recorded separately. No new
local GPU collection or full native CPU acceptance was performed.
