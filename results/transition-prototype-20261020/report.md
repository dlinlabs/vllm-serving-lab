# Offline planned-cap transition model: useful short steps, insufficient rollout

Eight existing qualified full captures were re-audited and converted to a
one-second table. All **2,400 one-step Q/D/R conservation checks** pass, and all
2,447 gateway snapshots agree with lifecycle reconstruction. This implements and
evaluates a predictive baseline, **not an MPC controller or an exact serving
simulator**. No GPU collection, original-data edits or C evaluation occurred.

## What changed

- Prediction controls use the cap plan known before measurement, with an explicit
  instantaneous-application assumption. Future actual application times are not
  features. Admin budget version and row/run identity are not predictors.
- Six one-second flow heads advance waiting-for-first, streaming and reservation
  cost. A separately fitted cohort-reward head estimates eventual SLO successes.
- Only origin-time causal information is used. Chronological training excludes
  labels whose time window crosses the split or whose cohort outcomes resolve
  afterward. Imputation/scaling use training rows only.
- Compare persistence, training mean, direct planned-cap ridge, flow rollout, and
  flow rollout with engine waiting/running/KV gauges. Engine counter-rate features
  are not implemented by this prototype. Alpha=1; no hyperparameter search.

The [method/runbook](../../experiments/transition-prototype-v1.md) explains
equations, assumptions, eligibility and limitations. The
[development settings](../../experiments/transition-prototype-v1.json) have no
acceptance thresholds and are not a frozen collection protocol.

## Main results

Chronological split: fit on each run's first 200 s (purged by label availability),
score the remaining arrival window. The following MAE is in **SLO requests/s**:
cohort count error divided by the horizon. Results pool equal-length per-run
evaluation sets; 1/2/5 s have 800/792/768 origins. Windows overlap, so these are
not independent repetitions or confidence intervals.

| Horizon | Persistence | Training mean | Direct ridge | Flow rollout | Flow + engine gauges |
|---|---:|---:|---:|---:|---:|
| 1 s | 2.776 | 2.770 | 2.174 | 2.174 | **2.136** |
| 2 s | 1.926 | 2.114 | **1.568** | 1.721 | 1.689 |
| 5 s | 1.366 | 1.503 | **1.168** | 1.396 | 1.427 |

![Chronological forecast errors](forecast-errors.png)

One-second flow rollout improves waiting-count MAE from 3.465 to 2.571 requests
and reservation-cost MAE from 3.047 to 2.328 cost units. At five seconds, its
waiting-count MAE is **3.587**, worse than persistence's **3.316** and direct
ridge's **2.783**. The engine variant improves five-second reservation-cost MAE,
but worsens waiting-count and goodput prediction: gauges are not a universal fix.

The five-second goodput rollout is worse than persistence in **5 of 8 runs**;
direct ridge is worse in **1 of 8** (cap32-r2). Thus the direct model's pooled
14.46% MAE improvement does not establish uniform reliability or correct action
ranking. This is an already examined development set, not final validation.

Near cap changes (A/B only), five-second rate MAE is 1.453 for persistence,
1.400 for direct ridge, 1.489 for flow rollout and 1.555 with engine gauges.
There are only 90 overlapping origins in this chronological subset.

### Cross-regime development transfer

Five-second cohort-goodput rate MAE:

| Train → score | Persistence | Direct ridge | Flow rollout | Flow + engine |
|---|---:|---:|---:|---:|
| Fixed captures → scheduled A/B | 1.325 | **1.241** | 1.476 | 1.465 |
| Scheduled A/B → fixed captures | **1.372** | 1.529 | 1.512 | 1.550 |

The second direction includes 888 origins with caps outside the
48–96 fitted range. Those diagnostic extrapolations are explicitly counted,
not accepted as supported operation. Hosts/runtime differ too. Leave-one-run-out,
all targets, tails, directional optimism and subsets are retained in
[metrics.csv](metrics.csv) and the full report.

## Interpretation and next action

The algebra is correct, but iterating separately learned one-step heads with
frozen history/engine covariates accumulates error. This is a plausible limitation
of the approximation, not a causal diagnosis proven by these scores. Projection
was needed in 7.16% of five-step forecasts without gauges and 17.06% with them;
its L1 magnitude mixes request and cost units and is descriptive only.

**Do not use this rollout as a validated MPC simulator.** Keep direct ridge as
the predictive comparator, investigate error by request mix/age and evolving
service history, and test whether service/throughput-counter structure improves
multi-step behavior before optimizing a controller. Fixed captures do not
validate low-cap switch dynamics; additional targeted measurement may still be
needed. A fixed-cap cohort maximum is not a dynamic-policy upper bound. C remains
untouched until the final model and evaluation protocol are frozen.

## Data and reproduction

Inputs: qualified shared-client A/B, plus cap16-r1, cap32-r1/r2, cap64-r1 and
cap96-r1/r2. Smoke and failed cap16-r2 are excluded from model scoring. The
original cap-range suite remains failed/incomplete. The cap64 service failure
remains accounted for in its labels. All inputs replay the same arrival window;
run holdout is replicate/host transfer, not an unseen-workload test.

- [Dataset manifest](dataset-manifest.json): all raw/table/code fingerprints,
  snapshot reconstruction evidence and flow verification counts.
- [Compact summary](summary.json): pooled and per-run errors, projection rates.
- [Full report](full-report.json.gz): all 576 score groups, 2,360 chronological
  prediction records, fitted coefficients and preprocessing. Decompress with
  Python's standard `gzip` module; the summary records compressed/plain SHA256.
- [Summary/figure generator](summarize.py): no fitting or model selection.
- [Reproducer](reproduce.py): complete raw re-audit, feature rebuild and scoring;
  refuses existing destinations and checks raw/table evidence and numeric scores.

After extracting the verified archives using the method/runbook commands:

```bash
python results/transition-prototype-20261020/reproduce.py \
  --cap-root /tmp/cap-transition-raw \
  --shared-root /tmp/shared-transition-raw \
  --output-dir /tmp/transition-reproduction
```

The execution environment used Python 3.12.14 / NumPy 2.3.5. Complete capture
audits were rerun for all eight inputs. Repeating model scoring produced identical
numeric results. Figure rendering, compressed-report integrity and relevant
offline regression tests were checked (**206 passed** across six relevant test
modules, including adversarial future-label, projection and reproducer checks).
The new code has **not** undergone a new
full native Linux CPU/mock acceptance run or any live GPU/controller validation;
the archived 504-case native record belongs to the earlier collection runtime.
