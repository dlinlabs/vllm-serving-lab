# First-entry clock refinement: CPU development and confirmation implementation

The new estimator improves nominal first-return timing on TRAIN. Independent
GPU confirmation has not run. The original six-term model and two-constant
submit predecessor remain reproducible and their historical verdicts unchanged.

## Immutable calibration evidence

Input archive `1793126846787142213.tar.gz`, SHA256
`985eeac690fc5c1055fe6663d92794baef3b1c710d6628754a333318972fabda`.
The reproduction command rechecks all 250 raw files, reconstructs all 179 source
files at 8bd96cb, reruns the original diagnostic audit and refits the original
TRAIN estimator in isolation. The raw archive, its 1250-case historical native
CPU proof and existing published results are unchanged. No local GPU was used.

Parameters come only from 48 TRAIN endpoints and 24 TRAIN LIGHT captures.
First foreground prefill work is identified by planned request identity and
prompt positions. No delayed-probe queue outcome selects a coefficient or
uncertainty bound. Family development has seen calibration failures, so these
results cannot be called independent validation.

## What changes

| Quantity | Previous submit family | First-entry family |
|---|---:|---:|
| Representative prefill CPU-return P | 35.155308 ms | 44.914831 ms |
| Pure-decode CPU-return D | 1.685664 ms | 1.685664 ms |
| Shared TRAIN prefill sensitivity domain | 14.984692–57.353666 ms | unchanged |
| D16 TRAIN first post-release commit error | -17.011 to -2.816 ms | -7.252 to +6.943 ms |

All 24 TRAIN first-commit readiness checks satisfy max(10ms,20% observed).
The 12 D16 checks have maximum absolute error 7.252ms. This is a sanity check on
TRAIN, not a new tolerance for held results or an identified physical CPU cost.
Six effective GPU coefficients remain unchanged. CPU/GPU overlap, own predicted
barrier, causal work and exact max-plus phase construction are retained.

The family uses the first-entry statistic as a representative for all prefill
returns. It does not establish that every later launch has that duration.
Host spans may include waiting/enqueue work. Existing service-law coefficients
may also compensate for omitted stages; no physical decomposition is claimed.

## What remains unresolved

All 144 calibration samples satisfy the nearest-frozen-point phase budget;
endpoint scores and LIGHT work signatures pass. Raw point errors still include
**3/48 baseline OFF misses and 1/48 LIGHT miss at 50ms/D16**; receipt has no miss.
Nominal first-return repair does not itself resolve this near-boundary queue
ambiguity. Baseline OFF internal phase remains unobserved.

The early branch at 50ms/D0 is absent from all three arms, and early 50ms/D16
has no receipt-only partner. These are missing controls, not validated
predictions or phase probabilities. Under the new confirmation contract,
passing queue errors with such missing branch controls remains inconclusive.
All production model, HTTP, adaptive-cap and MPC flags remain false.

## Independent confirmation implementation

The committed numeric plan has 240 new qc trials: 40 warmups, 48 TRAIN,
4 before anchors, 144 held three-arm trials and 4 after anchors. TRAIN fitting
and sealing occur before the first held anchor. The fresh host must pass
first-return/endpoint/work readiness before held collection. There is one
predeclared experimental family and no held-dependent fallback or model choice.

The collector seals source, exact-token manifest, model and all predictions.
It rejects source/freeze changes and retains cancellation/partial evidence.
Reports keep each sample's F0 point, entry point and phase errors, per-arm
counts, separate observer controls, branch coverage, endpoints, work/lifetimes
and independent before/after anchor comparisons to TRAIN. Original budgets and
all eight difficult cells are retained. Different-state comparisons are
unassessable rather than passing. A missing branch control blocks a pass.

After shutdown, an independent checker reconstructs raw rows, redoes TRAIN
fitting and the complete freeze, verifies seal chronology and rebuilds the
report. Canonical nested metric ordering makes live and saved-JSON results
identical. The helper archives complete failed/inconclusive runs as well as
passes, without retrying. Native GPU acceptance still requires a fresh run.

See `experiments/prefill-entry-clock-v1.md` for the full equations, assumptions,
numeric plan and short rental commands. HTTP and cap/MPC experiments remain
subsequent, separately qualified scopes.

## Reproduction

```bash
python scripts/analyze_prefill_entry_clock.py \
  --capture /path/to/1793126846787142213 \
  --archive /path/to/1793126846787142213.tar.gz \
  --output /path/to/new-entry-analysis
```

Output must be outside immutable evidence and use a new directory. This first
reproduces the original source-audited F0/submit development before constructing
the first-entry freeze and development scores. `freeze.json`,
`development-report.json` and `summary.json` are the resulting artifacts.

## CPU verification

341 related CPU tests passed in 84.89s, including entry/protocol/audit/host,
async/event/phase/receipt, collector/live and review-gate regressions. A separate
committed design/hash regeneration check passed (1 case). Total: 342 distinct
cases, no failures/errors/skips in the final reports. Focused subsets are not
added. The first local metadata check lacked an unchanged historical frozen
model fixture in the sparse checkout; restoring the original Git fixtures made
the complete check pass without exclusions. CPU results do not replace fresh
native CPU acceptance or GPU confirmation.
