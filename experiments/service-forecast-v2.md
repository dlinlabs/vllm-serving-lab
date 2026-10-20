# Service-context forecast ablations and cap-response diagnostics

This is a retrospective offline follow-up to PR #26, using the same eight
qualified captures. It is not an MPC controller, new collection protocol or
acceptance test. C stays uncollected/unscored. No serving code, fixed cost formula,
capture gates or original files are changed. Settings are recorded in
[service-forecast-v2.json](service-forecast-v2.json).

## Why direct horizons

The previous independently learned one-second flow heads accumulated error when
rolled out with frozen history/engine inputs. Here separate heads predict states
and eventual cohort SLO-success counts directly at 1/2/5 seconds. This avoids
recursive error accumulation, but does **not** produce a physically consistent
trajectory simulator. Horizon predictions can disagree with each other; no
monotonicity, request-level admission or terminal-state feasibility guarantee is
introduced. Predictions are clipped at zero as in the original ridge baseline.

## Causal service observations

- Q/D token and cost composition: sum original input-token estimates and
  unreleased cost over admitted requests awaiting first content versus streaming.
  Streaming input tokens are not remaining token work or exact KV usage.
- Age sums: waiting age since admission; streaming age since first content.
  Existing oldest-waiting age remains receipt-anchored.
- Offered input tokens in the trailing five seconds: admission/rejection
  decisions only, because the gateway may not yet know a received request's cost.
- Counter rates: prompt tokens, generation tokens and preemptions. Use two
  consecutive scrapes whose entire records finished by decision time; require
  successful HTTP/metric evidence, matching exporter/epoch/series identities,
  nonnegative per-series deltas, receive-time gap <=3 s and latest send age <=3 s.
  Failed scrapes are not bridged. Invalid/reset/stale values become null with a
  missing flag, never zero. Creation-time metrics are not token increments.

Rates divide observed counter increments by body-receipt time differences. They
are observed exporter throughput, not intrinsic engine capacity, exclusive GPU
timing or per-request attribution. Exporter batching and scrape timing can distort
short intervals. Co-located state and collector visibility are assumed; no online
feature-delivery path is implemented. All 2,400 selected origins have valid rate
observations; missing-rate behavior is not empirically validated by these runs.

## Fixed ablations

| Model | Additional input over the previous stage |
|---|---|
| direct_ridge | Exact PR #26 planned first/mean/end-cap baseline |
| context_ridge | Cost/token/age composition and current applied cap/headroom |
| gauge_ridge | Latest causal waiting/running/KV gauges and missing flag |
| counter_ridge | Three counter rates and their missing flags |
| service_quadratic | Planned-cap square, delta, and cost-composition interactions |
| service_no_plan | counter_ridge static inputs, with all planned-cap columns removed |

The no-plan model retains **current observed** applied cap and reservation state.
It tests incremental planned-control information given existing state, not whether
cap affects the real system. The gauge-only comparator was added during self-review
to separate gauges from counter-rate additions; no existing model or parameter
was tuned in response. All are exploratory development comparisons, not a final
model-selection test. Ridge alpha=1 with train-only scaling and imputation.

The same purged chronological, leave-one-run-out and fixed/scheduled transfer
splits are retained. Reward labels are unavailable until eventual cohort outcomes
resolve. Direct state heads use that same conservative eligibility for matching
comparators. Row identity/time and future actual cap applications never enter the
predictor. Control inputs use the premeasurement plan with zero delay at
integer-second left endpoints. No random window split or hyperparameter search.

## Action diagnostics, not counterfactual evaluation

For each five-second chronological test origin, hold the origin's features fixed
and request predicted goodput under held caps 16/32/64/96. Preserve ties; a model
without planned-cap inputs must tie all candidates. Report those **predictions**
along with training hold counts, counts with the same origin applied cap, and
nearest training-state distance. Distances use training-standardized Q/D/R and
token-composition fields, not all possible hidden state. No support threshold is
chosen and no distance certifies causal validity, feasibility or safety.

In particular, two different fixed runs do not supply counterfactual outcomes
for the same state. The low-cap captures contain no 16↔32 switches; A/B uses
48–96 on different hosts/runtime. These records cannot directly validate low-cap
action ranking. The direct forecast API rejects numeric cap-range extrapolation
by default; evaluation explicitly marks diagnostic extrapolations. Being inside
that numeric range does not establish state/action coverage.

## Reproduce

Python 3.12 and `requirements-cpu-tests.txt` are sufficient. A checked derived
table pack avoids repeating the expensive original audits for every model run:

```bash
python results/service-forecast-20261020/reproduce.py --output-dir /tmp/service-reproduction
```

It verifies archive/manifest/table/code fingerprints, runs all fixed models, and
compares scores and action diagnostics with numeric tolerance 1e-8. Existing
destinations are refused. The cached pack preserves the original audit evidence;
using it is not a new raw audit. For an independent complete raw audit, first
extract the cap-range and shared A/B archives using their original verified
helpers, then provide both roots:

```bash
python results/service-forecast-20261020/reproduce.py \
  --cap-root /tmp/cap-transition-raw --shared-root /tmp/shared-transition-raw \
  --output-dir /tmp/service-fresh-audit
```

`service_features.py` can also build fresh tables from explicitly supplied raw
capture directories, using 1/2/4 worker processes (default 2). Every selected
capture must pass the original auditor and snapshot/flow checks before any output
is published; duplicate raw captures and inconsistent context costs are rejected.
`service_forecasts.py DATASET --output-dir OUTPUT` writes the full score report.
