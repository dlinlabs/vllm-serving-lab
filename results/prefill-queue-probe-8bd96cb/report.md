# Native receipt-probe audit and submit-clock development

The supplied 232-trial archive passes its original measurement audit. A fresh
TRAIN-only six-term fit still misses four queue observations. A new TRAIN-only
CPU-return sensitivity ablation explains the two queue branches in this
calibration dataset, but is not a selected or independently confirmed model.

## Original evidence

- Archive: `1793126846787142213.tar.gz`, 12,578,499 bytes; SHA256
  `985eeac690fc5c1055fe6663d92794baef3b1c710d6628754a333318972fabda`.
- All 250 raw files match the immutable archive. All 179 captured source files
  match Git commit `8bd96cbe07ba8e7f26aab9fd33ff31af5453e55f`.
- Complete original native CPU acceptance: **1250 cases**, verified against the
  retained original JUnit. This is the capture's evidence, not a new local run.
- All 232 trials completed without failures in **967.118s (16min7s)**, under the
  pinned Qwen3-4B/RTX4090/vLLM0.30.0/Torch2.13.0/CUDA13.0/B2048/seq32 scope.
- Endpoint observer controls pass. All **47 comparable receipt/LIGHT pairs pass**;
  one different-state pair is unassessable. All cells have comparable evidence;
  50ms/D16 has five matching pairs and the other cells six each.
- The largest comparable queue delta is **3.623ms**. Baseline OFF internal receipt
  phase remains unobserved. Control success does not prove that instrumentation
  leaves OFF branch probabilities unchanged.

The reproduction command reconstructs original source in a temporary tree,
re-audits the raw capture, and repeats the original offline TRAIN-only fitting
algorithm. No raw file, old policy/report, or acceptance verdict is changed.

## Fresh TRAIN fit

Only 48 TRAIN OFF/LIGHT endpoint rows and 24 TRAIN LIGHT event records enter the
six-coefficient fit. Delayed probes are excluded. This is CPU fitting of retained
native calibration measurements, not new GPU execution or held confirmation.

| Arrival / background | Baseline queue range (ms) | F0 point (ms) | F0 failed OFF samples | Submit branch points (ms) |
|---|---:|---:|---:|---|
| 50-D0 | 59.605–60.686 | 54.685 | 0/6 | 0.000, 54.685 |
| 50-D16 | 0.202–77.119 | 76.184 | 3/6 | 0.000, 76.184 |
| 200-D0 | 20.994–22.602 | 15.520 | 0/6 | 15.520 |
| 200-D16 | 47.880–49.856 | 44.627 | 0/6 | 44.627 |
| 400-D0 | 97.083–99.544 | 98.522 | 0/6 | 98.522 |
| 400-D16 | 141.886–143.879 | 142.776 | 0/6 | 142.776 |
| 800-D0 | 101.121–104.568 | 100.585 | 0/6 | 100.585 |
| 800-D16 | 17.481–23.384 | 19.100 | 0/6 | 19.100 |

F0's TRAIN-only ±3.5034% common-scale family does not explain 50ms/D16:
**3/48 baseline failures, 0/48 receipt failures and 1/48 LIGHT failure** remain.
Endpoint median scores and LIGHT work signatures pass across arms/cases.
The CPU ablation's point forecasts at these offsets remain identical and retain
these four misses; only its phase score changes, fitting **144/144 calibration
probe samples** under the unchanged max(10ms,20% observed residence) budget.
This is development fit, not accuracy on unseen data. F0's extra 169.068ms branch
at 800ms/D16 has zero observations; neither family has statistical tail coverage.

## What the receipt evidence resolves

At 50ms/D16 the six baseline OFF waits are **2.747, 76.424, 2.075, 76.736, 77.119,
0.202ms**. Their 39.586ms median falls between the modes and is not a typical
request. Receipt-only has six late samples; LIGHT has five late and one early.
These small counts do not establish branch probabilities or rule out observer
perturbation of unobserved OFF states.

| Block-5 trial | Queue-post after release | Relevant commit | Receipt | Receipt progress | Queue |
|---|---:|---:|---:|---|---:|
| qr-194 LIGHT | 50.136ms | commit10 at50.634ms | 50.899ms | scheduled11, committed10; pending11 | 0.756ms |
| qr-196 receipt | 50.114ms | commit11 at128.213ms | 128.320ms | scheduled12, committed11; pending12 | 78.199ms |

The late trial's commit10 was already at48.161ms, before queue-post; the request
reaches the next receive state. The early request is received before schedule12.
Thus this paired delta is a different observed state, not a77ms same-state
observer shift. These CPU boundaries bracket transitions but do not timestamp
exact input-poll entry or reveal OFF internals.

## Missing timing and ablation limits

TRAIN D16 first post-release commits occur at **37.972–52.167ms**, whereas F0
predicts **13.215ms**. TRAIN execute_model host spans are nonzero; F0 has zero
CPU launch-return parts. The new ablation places effective CPU unavailability in
forward.cpu_after, retaining overlap with the GPU lane. It does not add a bias
to measured queues. Host spans may contain waits/enqueue work; this is not a
physically identified CPU-only cost or proven physical placement.

TRAIN-only per-trial medians give prefill **P=35.155ms**, decode **D=1.686ms** and
an empirical shared-P domain of **14.985–57.354ms** from TRAIN prefix spans.
The analytic max-plus sensitivity family declares 50ms/D0 and50ms/D16 ambiguous,
with at most two nominal poll points. No probe outcome fits these constants or
bounds; the family is not combined with F0's uncertainty to widen the union.

**The nominal clock is still incomplete:** the submit point first commit is
35.155ms, **2.816–17.011ms earlier than TRAIN D16 observations**. The near-boundary
phase score succeeds partly because an early nominal boundary is clipped to
zero residence. This does not establish accurate predictions at arbitrary early
offsets. State/context-dependent submit timing is a remaining CPU development
question before admitting a confirmation candidate.

Unobserved submit branches: early50ms/D0 in all three arms and early50ms/D16 in
the receipt arm. The fast LIGHT branch has no same-state receipt partner here.
Successful late-branch controls do not qualify those gaps. No model is promoted.

## Reproduction and current decision

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analyze_prefill_queue_probe.py \
  --capture /path/to/1793126846787142213 \
  --archive /path/to/1793126846787142213.tar.gz \
  --output /path/to/new-output
```

The raw archive remains in the supplied Library file. This directory retains its
per-file SHA256 inventory, original full CPU/JUnit proof, exact-source TRAIN fit,
both development reports, event diagnosis, native measurement report and the new
submit freeze. Source/provenance details are in provenance.json and summary.json.

Decision: **measurement diagnostic passed; F0 queue prediction failed; submit
uncertainty remains an unselected development hypothesis with unresolved nominal
timing and branch controls**. Independent confirmation, HTTP, adaptive-cap and
MPC remain false. Further CPU development can use this retained data; another
GPU rental is not the next step. Complete/admit the model and its confirmation
collector before acquiring fresh held measurements.

Local regression/replay evidence is in cpu-validation.json. Local CPU counts
are separate from the original1250-case native acceptance and must not be added.
