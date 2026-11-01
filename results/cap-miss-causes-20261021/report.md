# Cap-32 miss diagnosis: public reproduction and new-host replication

This is a development diagnosis on archived CPU-readable captures; it ran no GPU.
It makes the analysis in the PR #30 [follow-up comment](https://github.com/dlinlabs/vllm-serving-lab/pull/30#issuecomment-6327867880)
reproducible from public evidence and repeats it on the new host. That comment's
scripts were not attached; the [state-model report](../state-model-20261021/report.md#diagnosis-and-model-changes)
therefore could not check them. Everything below is an **association**. No per-step
attention cost, exclusive cause or counterfactual cap effect is identified.

* **Reproduced.** Every number in the comment is recomputed: the goodput
  decomposition from the #30 frozen evaluation of the new-host captures, the rest
  from the archived old-host bundle. Two wording corrections are listed below.
* **Replicated and stronger on the new host.** At fixed cap 32, short prompts
  (< 2000 tokens) miss **0.8% / 3.8% / 15% / 63%** of the time by quartile of other
  prompt tokens accepted in the preceding second (old host: 0% / 0.5% / 4% / 33%).
* **Not supported as stated: the Σ L² step-time result.** vLLM's prompt-token
  counter advances when prefill completes. A *linear* term in prompt tokens accepted
  during the interval gives the same out-of-capture gain as Σ L². With a better
  timing control, a squared term still helps, but only by 2–9% leave-one-capture-out
  and 1–6% across hosts. Its size is not identified per step.

Code: [`diagnose_cap_misses.py`](../../diagnose_cap_misses.py). All numbers:
[summary.json](summary.json). Reproduction: [below](#reproduce).

## Data and definitions

| | Old host 9945b0a0debb | New host 7a1cab853d19 |
|---|---|---|
| Evidence | [cap-range raw bundle](../cap-range-20261020/raw-bundle/manifest.json), 161 files | [state-model raw bundle](../state-model-20261021/raw-bundle/index.json), 138 files |
| Captures | 7 fixed: 16 ×2, 32 ×2, 64, 96 ×2 | 6: fixed 16, fixed 32, A ×2, B ×2 (smoke excluded) |

Both hosts replay the same trace on an RTX 4090. The summary records the SHA-256 of
each capture's six input files. One new-host fixed-32 request failed in transport
before the gateway assigned an ID. It is kept under the client ID, using the same
rule as `evaluate_low_cap_frozen.load_capture_compatible`.

Segments are [60,120), [120,180), [180,240) and [240,300) s of scheduled time.
Requests are admitted, successful trace requests; a miss is TTFT > 0.5 s.
**Recent load** is the prompt tokens of *other* admitted requests accepted in the
preceding 1 s. Step-time analysis uses one-second scrape intervals with at least
four running requests at their end ("busy"), from all captures on a host. The exact
definitions are stored in `summary.json`.

## 1. The comment's numbers, recomputed

The comment drew on two sources. Its goodput decomposition reads the #30 frozen
evaluation ([full report](../low-cap-frozen-20261021/full-report.json.gz)), which
scored the new-host captures:

| Claim (#30 frozen evaluation, new host) | Posted | Recomputed |
|---|---|---|
| Cap-32 goodput error: total / admission / SLO-hit term | +0.32 / −0.08 / +0.40 | +0.3163 / −0.0812 / +0.3975 |
| Same, segment 1; segment 3 | +0.71 / +0.13 / +0.58; +0.52 / +0.03 / +0.49 | +0.7147 / +0.1333 / +0.5814; +0.5178 / +0.0264 / +0.4914 |
| Cap-32 pass ratio, simulated / actual | 0.83 / 0.77 | 0.8328 / 0.7699 |
| Cap-32 goodput error with oracle arrivals | +0.47 | +0.4661 |
| Matched-request TTFT error, segment 1 / 3 | −39 / −22 ms | −39.2 / −22.2 ms |
| Cap-16 pass ratio, simulated / actual | ≥ 0.98 | 0.983 / 0.986 |

Everything else comes from the archived old-host bundle:

| Claim (old-host bundle) | Posted | Recomputed |
|---|---|---|
| Fixed-32 r1 pass rate by segment | 0.84 / 0.90 / 0.90 / 0.95 | 0.835 / 0.904 / 0.901 / 0.947 |
| Fixed-32 r2 pass rate by segment | 0.72 / 0.89 / 0.79 / 0.95 | 0.723 / 0.893 / 0.791 / 0.949 |
| Fixed-32 admitted prompt tokens/s | ≈ 6.8k, flat | 6.6k–7.0k |
| Short-prompt miss rate by recent-load quartile | 0 / 0.5% / 4% / 33% | 0 / 0.54% / 3.96% / 33.3% |
| Top-quartile cut | ≈ 9.2k tokens | 9,226 |
| Recent-load p90, segment 1 / 4 | 15.0k / 11.0k | 14,974 / 11,028 |
| Segment-1 misses; short-prompt share | ≈ 150; ≈ 61% | 149; 61.7% |
| vLLM queue mean, fixed 32, segments 1–4 | 74 / 30 / 41 / 16 ms | 73.8 / 29.6 / 40.6 / 16.5 ms |
| vLLM prefill mean, fixed 32 | 191 / 165 / 175 / 159 ms | 191.5 / 164.9 / 175.3 / 159.1 ms |
| vLLM queue mean, fixed 16 | 4–18 ms | 4.2–18.0 ms |
| Frozen-model residual by Σ L² bin (≤p25 … >p90) | −0.041 / −0.041 / +0.004 / +0.069 / +0.109 s | −0.0412 / −0.0407 / +0.0040 / +0.0690 / +0.1095 s |
| Linear refit c0 / c1 / c2; RMSE | 0.00908 / 3.12e-4 / 5.29e-5; 0.116 | 0.00908 / 3.12e-4 / 5.29e-5; 0.1162 |
| With Σ L²: c3; RMSE; added time for one 8k prompt | 3.15e-9; 0.097; ≈ 0.20 s | 3.15e-9; 0.0971; 0.202 s |

Corrections:

* "Schedule → accept is only 2–5 ms": each part (schedule → gateway receipt,
  receipt → accept) is 2–5 ms. Their sum at fixed 32 is 5–8 ms. Accept → first
  content is 190–340 ms, so the conclusion stands.
* "At cap 16 short prompts essentially never miss": on the old host 2.1% of
  short prompts miss at fixed 16 (7.3% in the top recent-load quartile). On the new
  host, 0 of 931 miss.

The comment built intervals after dropping failed scrapes. A stricter construction
never bridges a failed scrape and requires 0.8–1.3 s spacing and continuous
counters. It gives identical busy-interval results on both hosts. One cap-16
exporter row loses one interval.

## 2. New-host replication

**Segments.** The fixed-32 run passes 0.78 / 0.71 / 0.64 / 0.81 by segment. Pooled
over all five captures with cap-32 segments, the rates are
0.71 / 0.83 / 0.73 / 0.81. These equal the actual pass ratios in #30; the old-host
pooled values are 0.78 / 0.90 / 0.85 / 0.95. Segments 1 and 3 are weakest when
pooled, but not in the single fixed run. Pooled switch segments also differ in
recent cap history. In the fixed-32 run, recent-load p90
(15.4k / 15.8k / 18.7k / 12.6k) ranks the segments in the same order as their misses
(69 / 94 / 109 / 62).

**Short-prompt misses by recent-load quartile.**

| Captures | Short requests | Q1 | Q2 | Q3 | Q4 | Q3/Q4 cut |
|---|---:|---:|---:|---:|---:|---:|
| Old, fixed 32 (r1 + r2) | 2,223 | 0% | 0.5% | 4.0% | 33.3% | 9,226 |
| New, fixed 32 | 960 | 0.8% | 3.8% | 15.4% | 63.3% | 11,590 |
| New, every cap-32 segment | 3,119 | 0.4% | 2.2% | 14.4% | 55.4% | 10,946 |
| Old, fixed 16 (r1 + r2) | 1,637 | 0.5% | 0% | 0.5% | 7.3% | 5,895 |
| New, fixed 16 | 931 | 0% | 0% | 0% | 0% | 5,135 |

Short prompts account for 52–68% of new-host fixed-32 misses in each segment. Old
host segment 1: 62%. Median accept → first content at fixed 32 by own prompt length
(< 1k, 1–2k, 2–4k, ≥ 4k) is 0.11 / 0.20 / 0.30 / 0.53 s on the old host and
0.21 / 0.29 / 0.40 / 0.56 s on the new host. Schedule → accept is 9–11 ms.

**Exporter histograms (new, fixed 32).** Queue means are 66 / 87 / 110 / 58 ms and
prefill means 208 / 226 / 235 / 192 ms by segment; at fixed 16 the queue is 1.6–4.6 ms.
The #30 diagnosis (62 / 82 / 111 / 60 and 204 / 223 / 237 / 194 ms) selected
intervals differently. Histogram increments are not the request cohorts above and
cannot be added to request spans.

**Frozen-model residual by Σ L² bin (new host):**
−0.056 / −0.047 / −0.006 / +0.043 / +0.142 s. The pattern replicates.

## 3. Step-time fits with a timing control

The busy-interval residual pattern does not by itself show super-linear cost. The
model uses the counter `prompt_tokens_total`; how that counter is timed matters:

| Correlation of interval prompt-counter increment with … | Old | New |
|---|---:|---:|
| prompt tokens whose gateway first content arrived in the interval | 0.990 | 0.990 |
| prompt tokens accepted in the interval | 0.401 | 0.300 |
| prompt tokens spread uniformly over each request's accept → first-content span | 0.792 | 0.757 |

The counter is credited when prefill finishes. An interval in which long prompts
were accepted is busy with prefill that the counter reports later. Σ L² of accepted
prompts is correlated with that unattributed work, whether the cost is linear or not.
The fits below separate timing from shape. Each is a no-intercept least-squares fit
of interval wall time; lower RMSE is better.

| Terms added to steps, decode and prompt counters | Old LOCO (7) | New LOCO (6) | Old → new | New → old |
|---|---:|---:|---:|---:|
| none (linear) | 0.1175 | 0.1030 | 0.1047 | 0.1314 |
| Σ L² accepted (posted) | 0.0995 | 0.0737 | 0.0789 | 0.1134 |
| tokens accepted (linear timing control) | 0.0982 | 0.0752 | 0.0791 | 0.1101 |
| tokens accepted + Σ L² accepted | 0.0982 | 0.0728 | 0.0780 | 0.1113 |
| tokens spread (linear timing control) | 0.0859 | 0.0601 | 0.0643 | 0.0970 |
| tokens spread + Σ L² spread | 0.0838 | 0.0549 | 0.0603 | 0.0956 |

LOCO is leave-one-capture-out; "old → new" fits on all old-host intervals and scores
new-host intervals. Every added-term model beats linear in every held-out capture.
All coefficients are nonnegative.

* The posted gain is reproduced by a **linear** timing term. Adding Σ L² to
  "tokens accepted" changes RMSE by 0% / −3% LOCO and −1% / +1% across hosts.
* With the spread control, a squared term still lowers RMSE by 2.5% / 8.7% LOCO and
  by 6.2% / 1.3% across hosts. Its coefficient (2.7e-9 old, 3.9e-9 new) is similar to
  the posted 3.15e-9: 0.17–0.25 s extra for an isolated 8k prompt. Its sign agrees
  with super-linear prefill. Its size is not identified: scrapes do not reveal
  per-step chunk composition. The spread uses gateway spans, which include queueing.
  Squared and linear spread terms are correlated.
* Total linear prompt cost with the spread control (2.97e-5 + 2.48e-5 = 5.45e-5 s/token,
  old host) is close to the frozen 5.34e-5. The frozen *average* prompt cost is not
  biased by the lag; *when* the cost lands is missing.

## What this means for the model fix

* The strongest replicated signal is head-of-line delay after recent long-prompt
  load. The simulator already models FIFO with an assumed 8192-token per-step budget
  and is still too optimistic. Either that assumption or per-step timing is wrong.
  Recording the vLLM scheduler configuration (chunked prefill,
  `max_num_batched_tokens`, partial-prefill limits) and a short per-step GPU
  microbenchmark are better evidence than a Σ L² coefficient fitted from scrapes.
* Prefill-backlog state (queued prompt tokens ahead, largest prompt in flight) is
  supported by the association.
* A super-linear prefill term may be tried as a sensitivity, like the #30 quadratic
  stress, not as an estimated coefficient.

**Limits.** Development data only. Two old-host fixed-32 runs and one new-host run.
Pooled switch segments mix histories. Associations, not causes. No GPU, no controller
or policy-gain claim.

## Reproduce

From a clean checkout of this branch, with Python 3.12+ on Linux:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/pip install -r requirements-cpu-tests.txt \
  -r results/state-model-20261021/requirements.txt
.venv-analysis/bin/python scripts/reproduce_cap_miss_causes.py \
  --work-dir /tmp/cap-miss-reproduction
```

The script restores both bundles into a fresh work directory and verifies every
listed file's size and SHA-256. It reruns the module and compares the result with
`summary.json`: floats within relative 1e-4, everything else exactly. It writes
`verification.json`. The cap-range extractor reopens a named temporary file, which
Windows does not allow. On Windows, restore that bundle once elsewhere and pass
`--old-evidence DIR`; `--new-evidence DIR` is also accepted. Both are hash-verified
before use. To run the analysis directly:

```bash
python diagnose_cap_misses.py \
  --host old=OLD_EVIDENCE/results/cap-range-v1 \
  --host new=NEW_EVIDENCE/results/low-cap-switch-v1 \
  --output summary.json
```

Unit and synthetic end-to-end tests: `python -m pytest tests/test_cap_miss_causes.py`.
