# Entry HTTP prerequisite rejected at TRAIN: efca3e4

**Original-source reconstruction reproduces TRAIN readiness rejection.**
This attempt completed 88 controlled trials (40 warmups plus 48 TRAIN), then
stopped before all held controlled and HTTP trials. No accepted freeze/seal or
held/HTTP report was written. The controlled prefix took 406.433s; the combined
runner took 406.598s. This is a failed prerequisite, not HTTP accuracy evidence.

## Immutable input and reconstruction

- Archive `1793156148422216020.tar.gz`: 19,833,493 bytes; SHA256
  `a281a36e0939a7181233f8c7edbc5dc86bec598cd33377c83a641d26f51e4d8f`.
- All 115 regular files match the archive (100,746,572 uncompressed bytes).
  All 199 captured source files match Git commit
  `efca3e4e5e1a1237e796b39d965e4fed233b5fdb`.
- Original 1414-test native CPU proof is copied byte-for-byte and passes the
  original validator, including the five required trace configurations.
- An isolated verified Git snapshot rebuilds all 88 raw rows, LIGHT dispatcher
  parity, manifests, runtime/source pins, actual combined launch flags,
  chronology, parent/subcapture identities, failed-prefix membership and
  absence of held/HTTP trials.
- Original TRAIN fitting and entry-family construction use matching
  NumPy2.3.5/SciPy1.18.1 and reproduce readiness=false. The original
  seal_training canonical row round trip is preserved; dictionary insertion
  order must not change quality-list order or downstream document hashes.
  Two independent canonical reconstructions give identical complete base and
  rejected-candidate documents. No original source, estimator or budget changes.

The native run **did not persist or seal this candidate**. These reconstructed
files are explicitly offline rejected TRAIN artifacts. `reconstruction.json`
records partial rejection reconstruction, not a complete-collection audit or
accuracy acceptance. Raw native verdict and every acceptance flag remain false.

## Exact failed gate

Phase construction succeeds (`phase_errors=[]`). First-commit readiness passes
21/24 TRAIN LIGHT checks and rejects three under max(10ms,20%):

| Trial | TRAIN case | Observed first commit | Predicted | Absolute error | Limit |
|---|---|---:|---:|---:|---:|
| qc-047 | composition-uniform-D16 | 32.731ms | 46.955ms | 14.225ms | 10.000ms |
| qc-060 | composition-uniform-D16 | 35.714ms | 46.955ms | 11.242ms | 10.000ms |
| qc-070 | single-L7168-D16 | 25.161ms | 46.955ms | 21.795ms | 10.000ms |

First commit means the first positive committed event after foreground release.
With async background work it can consume an already-submitted background
batch; it is not foreground first token or full foreground batch completion.
Entry P=46.955337ms, D=2.084162ms; the full TRAIN prefix P domain is
10.732867–55.953486ms. These are effective model terms, not physical CPU-service
identification.

## One constant cannot explain all repeats

The same planned single-L7168-D16 case has these first commits:

| Trial | Observed | Acceptable prediction interval |
|---|---:|---:|
| qc-052 | 51.639ms | 41.311–61.967ms |
| qc-057 | 50.539ms | 40.431–60.646ms |
| qc-070 | 25.161ms | 15.161–35.161ms |
| qc-083 | 49.974ms | 39.974–59.974ms |

The shared interval requires >=41.311ms and <=35.161ms: empty, with a 6.150ms
gap. No one fixed prediction for this planned case can pass all four under the
original budgets, even if selected after observing the outcomes. This is a
descriptive infeasibility check, not a new estimator or held result.

All four normalized recorded release scheduler states match. Each of the six
TRAIN cases has one recorded scheduler signature across its four repeats.
This covers recorded request progress and running/waiting order, and excludes
pending-copy readiness/debt or physical CPU/GPU state. It does not establish
identical physical states or rule out a separately qualified state-conditioned
predictor. No causal attribution to network, GPU fault, CPU contention, GIL or
GC is supported by this archive alone. Slow installation is a separate issue.

## Repair and model decision

The collector now preserves rejected base/candidate artifacts and a source-bound
training-readiness-failure.json containing failed checks. No accepted freeze or
seal is written; the exception still stops before held/HTTP. No retry, sample
deletion, budget relaxation or coefficient change is introduced.

**The predictor is not repaired or accepted by this diagnostic change.**
Another blind one-constant refit cannot solve this capture. The next model
design needs decision-time observable pending state/residual debt and/or
explicit uncertainty, then a separately frozen qualification plan. Future
worker-return spans, completions and held labels cannot become online features.
Existing pre-ADD HTTP snapshots contain more state than the controlled release
record, but their forecast validity remains untested in this attempt.

C16/C32 action validity and closed-loop benefit remain separate. Changing the
rental host alone is not established as a fix. Reconstructing this evidence
does not require another GPU rental.

## Reproduction and local validation

With the immutable archive extracted safely outside the repo, the pinned Git
source available and captured numerical versions installed:

```bash
git fetch --depth 1 origin efca3e4e5e1a1237e796b39d965e4fed233b5fdb
python scripts/analyze_prefill_entry_http_failure.py --capture /path/1793156148422216020 --archive /path/1793156148422216020.tar.gz --output /path/fresh-analysis
```

The analyzer rejects wrong source inventory/hash, numerical versions, raw
archive identity, extra/missing prefix evidence or altered verdicts.
**172 related local CPU checks pass, with zero failures/errors/skips.**
They cover failure preservation and no held progression, interval feasibility,
release-state normalization, source/version guards, existing entry HTTP,
confirmation, host and analysis regressions, and design/hash regeneration.
This selection overlaps earlier selections and is separate from the original
1414-case native CPU acceptance. No new local GPU or full native CPU run is
claimed. cpu-validation.json binds current source and artifact hashes.
