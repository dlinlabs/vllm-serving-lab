# Frozen CPU comparison: MPC does not beat the selected fixed baseline

Source: `de36581f16fc39784297aadcd6186883aaf1d35a`.
Artifact: `results/mpc-research-cpu/policy-benchmark.json`.
Calibration identity is explicitly `synthetic-unqualified-v1`.
This is a negative exploratory synthetic result, not a native GPU conclusion.

## Protocol and integrity

The prospectively frozen protocol is `cpu-benchmark-plan.md`. Development
seeds1/2 selected fixed16 and feedback thresholds512/2048, averaging all three
workloads. Only then were fresh test seeds10001–10005 used: stationary, burst
and shift,15 episodes per arm,75 total, with identical offered traces and
deterministically randomized arm order. Each has12s offers, empty initial
state and up to20s drain. There was no held-out policy/parameter retuning.

The failed development invocation and two causal-grid regressions are retained.
The correction changed time representation, not chronology gates or performance
criteria. The evaluator's old seed101 mock prefix and its prospective test-seed
amendment are also retained. No rejected run was relabeled as passing.

## All policies, same denominator

Values below are means of the15 episode metrics, not pooled quantiles.
Goodput counts eventual-success requests whose scheduled TTFT is at most0.5s,
divided by the12s offered window. Nonattainment counts every offered request
that is not good, including rejections and failures.

| Policy | Accepted req/s | Good req/s | Rejection | All-offered nonattainment | Mean episode controller P95, ms | Deadline fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Fixed16 | 6.0222 | 5.5778 | 57.984% | 61.267% | 0.0021 | 0/180 |
| Fixed32 | 7.5500 | 2.3833 | 47.429% | 83.678% | 0.0025 | 0/180 |
| Feedback | 6.2444 | 3.8722 | 56.599% | 73.224% | 0.0253 | 0/180 |
| One control move | 5.9722 | 4.5444 | 58.312% | 68.461% | 21.2259 | 0/180 |
| Three-move MPC | 5.9944 | 4.4222 | 58.189% | 69.381% | 69.6505 | 10/180 |

Conditional latency summaries are **means of episode P50/P95/P99**, not global
request quantiles, confidence limits, individual token-gap quantiles or SLO
attainment probabilities. Full per-episode data are in the JSON.

| Policy | TTFT seconds, P50/P95/P99 | Per-request mean TPOT ms, P50/P95/P99 | Successful E2E seconds, P50/P95/P99 |
|---|---|---|---|
| Fixed16 | .200/.491/.593 | 13.172/14.176/14.678 | 1.051/2.816/3.282 |
| Fixed32 | .751/1.428/1.605 | 14.767/16.165/16.609 | 1.873/3.886/4.325 |
| Feedback | .359/1.183/1.475 | 13.644/15.061/15.514 | 1.420/3.413/3.855 |
| One move | .278/1.033/1.203 | 13.448/15.013/15.478 | 1.367/3.300/3.740 |
| MPC | .291/1.054/1.215 | 13.451/14.934/15.394 | 1.385/3.312/3.759 |

Mean controller CPU totals per episode were0.013/0.016/0.194/159.213/537.785ms,
respectively. Mean budget switches were0/0/7.4/3.2/3.333. Recovery was observed
under the protocol's descriptive definition in10/8/11/12/14 of15 episodes;
missing recovery is retained as null, not zero. All75 episodes had zero unresolved
accepted requests after drain; this is specific to these short synthetic outputs.
GPU utilization is null throughout. Per-episode failure rates, total budget
variation, latency samples and recovery times remain in the artifact.

## Paired seed-block differences

MPC minus each comparator, averaging the three workloads inside each seed block:

| Comparator | Mean difference, good req/s | Exploratory percentile95 interval |
|---|---:|---|
| Development-selected fixed16 | -1.1556 | [-1.4167, -0.9667] |
| Feedback | +0.5500 | [+0.1778, +1.0444] |
| Same-model one-move | -0.1222 | [-0.4000, +0.0333] |

Only five independent synthetic seed blocks and2,000 bootstrap resamples are
used, without multiplicity adjustment. These are exploratory intervals, not a
confirmatory statistical claim. An interval containing zero does not establish
equivalence. MPC's advantage against this feedback baseline cannot be presented
without its worse result against the selected fixed baseline.

Fixed16 had higher mean goodput in each workload: stationary6.5000 versus
MPC4.9167; burst6.1333 versus5.0333; shift4.1000 versus3.3167. Overall MPC was
about20.7% below fixed16 and offered no demonstrated benefit from three control
moves over one, while consuming more CPU and hitting the unchanged100ms limit
ten times.

## Interpretation and stop rule

Do not promote this candidate or declare a native benefit. Potential causes
include uncalibrated model/EOS/arrival assumptions, imperfect intervention
ranking, finite-tail scoring and deadline fallback. This experiment does not
causally separate those contributions. It does not prove all MPC architectures
inferior, nor that fixed16 is optimal on a real4090.

Further held-out tuning or a larger solver would not answer the unresolved
native identification question. Stop synthetic leaderboard optimization.
Keep fixed and simple feedback controls as serious alternatives; preserve the
one-move challenger. The next high-information work is the separately authorized
native state/action/HTTP qualification and randomized causal-identification
pilot in `limitations-and-next-experiment.md`, followed by untouched validation
and a newly frozen confirmatory benchmark.
