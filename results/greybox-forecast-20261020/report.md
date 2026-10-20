# Grey-box simulator for five-second goodput forecasts

Development analysis on CPU only. It uses the same eight qualified captures and the
same **768** chronological five-second test origins (t ≥ 200 s) as the
[combined service review](../service-review-20261020/report.md). There is no new GPU data, and
**no controller is enabled**. Numbers are 5 s goodput rates (req/s). MAE is the main metric,
and lower is better.

![Per-run MAE and open-loop cap curve](greybox-forecast.png)

## Model

[`greybox_simulator.py`](../../greybox_simulator.py) simulates the gateway and the engine one
vLLM step at a time from each origin:

- **Gateway.** Uses the exact admission rule: admit when reserved + cost ≤ cap, and release the
  reservation at completion.
- **Engine.** Uses continuous batching with an 8,192-token step budget. Each step decodes one
  token for every running request, then fills the rest of the budget with FIFO prefill.
  - Step time is 9.18 ms + 0.317 ms per decode token + 0.0534 ms per prompt token. This is
    fitted from `/metrics` counters for t < 200 s only.
  - Every successful request in these captures returns 128 tokens, so outputs are fixed at 128.
- **Origin state.** Built from gateway events and step counters that were already received at
  the origin.
- **Future arrivals.** Sixteen bootstrap replays of one random past 5 s window drawn from the
  previous 60 s. Only arrivals whose gateway decision is known at the origin are used.
- **TTFT calibration.** The excess TTFT that the step model does not explain is modeled as a
  nondecreasing function of the running count at admission. Its spread is robust, and it is
  fitted on first tokens observed by 200 s. An online bias then adds a shrunken mean of recent
  residuals, using only first tokens already observed at the origin.

[`evaluate_greybox.py`](../../evaluate_greybox.py) checks every raw file against the SHA-256
values in the archived review and confirms that each origin's actual count matches the
archive. It then scores these models against the review's best 5 s model (`service_no_plan`
ridge):

- A trailing-60 s nowcast. This is the met-SLO rate over the previous 60 s, counting only
  outcomes observable at the origin.
- Each simulator ablation.
- The replicate noise floor.

## Forecast accuracy

Paired differences use a block bootstrap (10 s blocks within each run, 4,000 reps).

| Model | MAE | Bias | Within-run corr | ΔMAE vs ridge (95% CI) |
|---|---:|---:|---:|---:|
| Ridge (`service_no_plan`) | 1.1304 | −0.375 | 0.23 | – |
| Trailing-60 s nowcast | 1.0865 | −0.259 | 0.05 | −0.044 (−0.128, +0.042) |
| **Simulator** | **1.0196** | −0.114 | 0.33 | **−0.111 (−0.193, −0.030)** |
| Simulator, no online bias | 1.0449 | −0.208 | 0.33 | – |
| Oracle arrivals, raw step TTFT | 1.8904 | +1.593 | 0.44 | – |
| Oracle arrivals + overhead | 1.0281 | −0.156 | 0.41 | – |
| Oracle arrivals + overhead + bias | 1.0007 | −0.062 | 0.40 | – |
| Each run's own test mean (oracle) | 1.0769 | 0.000 | – | – |

- **Against the nowcast**, the simulator's ΔMAE is −0.067 (−0.127, −0.005).
- **Per run**, the simulator beats ridge on 7/8 runs. The exception is shared-a: 1.204 vs 1.094.

| Run | Ridge | Nowcast | Simulator |
|---|---:|---:|---:|
| fixed-16-r1 | 0.579 | 0.541 | 0.435 |
| fixed-32-r1 | 1.085 | 1.197 | 0.926 |
| fixed-32-r2 | 1.176 | 1.211 | 1.005 |
| fixed-64-r1 | 1.345 | 1.253 | 1.257 |
| fixed-96-r1 | 1.175 | 1.001 | 0.964 |
| fixed-96-r2 | 1.393 | 1.293 | 1.311 |
| shared-a-r1 | 1.094 | 1.129 | 1.204 |
| shared-b-r1 | 1.195 | 1.067 | 1.056 |

- **Noise floor.** Two replays of the same trace at the same cap are nearly uncorrelated at
  5 s: corr 0.10 at cap 32 and 0.04 at cap 96. Their replicate-vs-replicate MAE is 1.40 and
  1.62. Most remaining 5 s error is therefore intrinsic.
- **Arrival uncertainty is not the main error source.** Oracle future arrivals improve the
  simulator by only 0.019.
- **Near cap changes, the simulator is worse.** "Near" means a change from 10 s before the
  origin through the end of the window, in shared A/B (n = 96). MAE is ridge 1.240, simulator
  1.357 and nowcast 1.359. In the steady segments of the same runs (n = 96) it is ridge 1.049,
  simulator 0.902 and nowcast 0.838.

## Why it still cannot choose a cap

The open-loop cap curve simulates the arrivals of fixed-32-r1 from an idle start for 300 s.

| Cap | 16 | 24 | 32 | 40 | 48 | 64 | 80 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Predicted admitted | 4.28 | 5.14 | 5.63 | 5.82 | 5.97 | 6.56 | 6.96 | 7.11 |
| Measured admitted | 4.23 | – | 5.84 / 5.69 | – | – | 6.52 | – | 7.28 / 7.42 |
| Predicted goodput | 4.20 | 4.49 | 4.21 | 2.84 | 2.21 | 1.79 | 1.79 | 1.31 |
| Measured goodput | 4.13 | – | 5.08 / 4.59 | – | – | 2.27 | – | 3.33 / 2.67 |
| Goodput, cap held out of both fits | 4.02 | – | 2.79 | – | – | 2.44 | – | 0.65 |

- **Throughput transfers across caps.** Predicted admissions are within about 4% of the
  measured values at every measured cap.
- **Goodput does not.** The predicted values depend almost entirely on the fitted
  concurrency → TTFT overhead curve. Holding a cap out of the fit moves its prediction a lot.
- **The predicted peak at 24 is untested.** No 24 / 40 / 48 data exists.

vLLM histogram means over 5–300 s, in run order:

| Run (order) | Queue time (s) | Prefill time (s) | Engine TTFT (s) | Gateway TTFT (s) |
|---|---:|---:|---:|---:|
| shared-a-r1 | 0.391 | 0.314 | 0.746 | 0.772 |
| shared-b-r1 | 0.327 | 0.296 | 0.665 | 0.688 |
| fixed-64-r1 | 0.304 | 0.293 | 0.651 | 0.669 |
| fixed-16-r1 | 0.009 | 0.117 | 0.146 | 0.154 |
| fixed-96-r1 | 0.228 | 0.291 | 0.559 | 0.583 |
| fixed-32-r1 | 0.037 | 0.172 | 0.236 | 0.246 |
| fixed-32-r2 | 0.061 | 0.192 | 0.281 | 0.290 |
| fixed-96-r2 | 0.337 | 0.304 | 0.683 | 0.696 |

- **The high-concurrency TTFT jump is engine queueing plus slower prefill.** It is not frontend
  lag:
  - Engine and gateway TTFT agree within about 8–26 ms.
  - There are no preemptions, and no prefix-cache hits.
- **The step model does not reproduce this queue wait.** The wait is also confounded with run
  order: in both replicate pairs the later run is worse.

## Differences from the PR #26 comment

[The comment](https://github.com/dlinlabs/vllm-serving-lab/pull/26#issuecomment-6300431263)
used a scratch script. With the median schedule-to-accept delay computed over the whole run,
this port reproduces the comment's MAE, per-run and cap-curve values. The committed default
computes that median only from requests accepted by 200 s, so no test-period information is
used.

The bootstrap CIs differ for a separate reason: the scratch script used a different block
resampler and random stream. Even the nowcast-vs-ridge CI moves, although neither model uses
that median. Changes:

| Number | Comment | Committed |
|---|---:|---:|
| Simulator MAE | 1.0199 | 1.0196 |
| Simulator without online bias | 1.0469 | 1.0449 |
| Oracle arrivals + overhead + bias | 1.0016 | 1.0007 |
| Simulator vs ridge CI | (−0.192, −0.027) | (−0.193, −0.030) |
| Simulator vs nowcast CI | (−0.129, −0.009) | (−0.127, −0.005) |
| Nowcast vs ridge CI | (−0.127, +0.044) | (−0.128, +0.042) |
| shared-a simulator MAE | 1.198 | 1.204 |
| Cap 32 held-out goodput | 2.80 | 2.79 |

No conclusion changes.

## Limitations

- **The test is not independent.** It uses the same eight development captures and a single
  trace. Test origins come from the same runs whose first 200 s fit the step time and the
  overhead curve.
- **The overhead curve is pooled.** One curve is fitted across caps and run order. It absorbs
  engine queueing that the step model does not explain, so it is not a causal model of that
  queueing.
- **Fixed 128-token outputs** hold for these captures only.
- **The cap curve is open-loop from an idle start** with one reference arrival sequence.
  Measured fixed-cap goodput is confounded with run order.

## Missing data for a model that can choose a cap

1. **Within-run cap switches in the 16–48 region.** All current switches are 48–96. The
   measured goodput curve peaks below 64 (32 is the best measured cap). MPC needs action
   effects where the optimum lies: 16↔24↔32↔40↔48 in both directions, with several step sizes,
   repeated.
2. **Fixed caps 24 / 40 / 48**, to locate the peak.
3. **Per-request vLLM timestamps** (arrival, queued, scheduled and first token, or OTel spans).
   These separate frontend lag from engine queueing.
4. **Randomized cap order, a warm-up run and interleaved replicates**, to break the
   order/drift confound.
5. **More replicates per condition.** The 5 s noise floor is close to zero correlation.
6. **More than one trace or load level.** Every run replays the same arrivals.
7. **A prompt-mix test at fixed concurrency.** This checks whether TTFT overhead depends on the
   running count or on prefill tokens.

## Reproduction

Use Linux or macOS. The cap-range extractor reopens a named temporary file, which Windows does
not allow.

```bash
python results/greybox-forecast-20261020/reproduce.py --output-dir /tmp/greybox-reproduction
python results/greybox-forecast-20261020/plot.py
python -m pytest tests/test_greybox_simulator.py -q
```

`reproduce.py` does four things:

1. Checks the archived report and scoring-code fingerprints.
2. Re-extracts both raw bundles and verifies every capture file against the archived hashes.
3. Reruns `evaluate_greybox.py` with the recorded settings.
4. Requires every number to match within 1e-9.

[`summary.json`](summary.json) holds the headline numbers. `full-report.json.gz` holds every
prediction, per-run score, overhead curve, cap curve and engine summary.
