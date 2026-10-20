# Combined offline model review

This retrospective revision combines the flow-transition prototype and the
service-context forecast follow-up in PR #26. The original numerical reports
remain in their own directories; this revision is a separate development analysis,
using the same eight captures and untouched C. Fixed settings are in
[service-forecast-reviewed-v3.json](service-forecast-reviewed-v3.json). No GPU
collection, serving-path change, cost-weight fit or controller is included.

## Review corrections

1. Counter features require finite clocks, well-formed counter samples, matching
   exporter/epoch/labels, monotone per-series increments, and two consecutive
   fully available scrapes. Send and receive gaps and both scrape RTTs must be
   at most three seconds; overlapping scrapes are rejected. Invalid observations
   remain missing, with explicit flags. This is a conservative throughput
   observation, not exclusive GPU service capacity.
2. Forecast inputs require finite nonnegative core state/context, consistent
   Q+D=in-flight, waiting-cost+streaming-cost=reserved-cost, and cap−reserved-cost
   headroom. Negative headroom is valid while work drains after a cap reduction.
   Engine observations can be imputed only with consistent missing flags.
3. Prediction rejects unseen `(observed applied cap, complete planned sequence)`
   patterns by default, even inside the numeric cap range. Two separate explicit
   overrides permit numeric extrapolation and unseen-pattern diagnostics. Reports
   flag both. A pattern observed once is **not** a safety or causal certificate;
   state coverage, host/runtime and hidden work remain limitations.
4. The fixed `service_sequence` comparator replaces first/mean/last control
   summaries with one column per future step. This distinguishes plan permutations
   that older encodings alias. It uses the same static context/gauges/counters as
   `service_no_plan`, ridge alpha=1, and train-only scaling/imputation. The six
   legacy comparators are retained without tuning. Extra plan columns cannot
   manufacture evidence of switches absent from the captures.
5. Additional grouped-cap holdouts omit every repeat at a fixed cap and remove
   all other training origins with that applied cap or a planned occurrence
   within the forecast horizon. This prevents a same-cap repeat leaking into the
   nominal held-out group. Earlier workload history may still reflect that cap;
   this is not an unseen-dynamics causal experiment or independent-workload test.
6. Extraction/scoring capture code fingerprints before work and verify them before
   publication. The historical v2 reproducer uses hash-verified source snapshots
   so advancing the implementation does not silently change the archived result.

## Evaluation boundaries

Prediction heads directly estimate Q/D/R and the eventual SLO-success count of the
future arrival cohort at 1/2/5 seconds. They are clipped at zero, but do not enforce
cross-horizon consistency, flow conservation, request-level feasibility or an SLO
constraint. The v1 flow rollout remains a separate baseline with its existing
projection; it is not replaced by the direct model.

Training labels that resolve beyond the chronological cutoff are purged. All
encodings use only origin observations and the premeasurement control plan,
assuming zero application delay at integer-second left endpoints. Current cap is
an observation; future actual application timing is never an input. Forecasting
accuracy and cap rankings on these reused development captures do not establish
policy gains. The counterfactual diagnostics explicitly override unsupported-action
guards and must not be used as controller validation.

## Reproduce

From the repository root, Python 3.12 with `requirements-cpu-tests.txt`:

```bash
python results/service-review-20261020/reproduce.py --output-dir /tmp/service-review-repro
```

The cached pack is hash-verified, with original raw-audit provenance and fresh
review extraction recorded. Cached replay does not re-audit raw captures. To
repeat that step, extract the archived raw captures and run `service_features.py`
with the eight qualified directories as documented in the historical v2 method;
then run `service_forecasts.py DATASET --output-dir FRESH_OUTPUT`.

Historical v2 scores remain reproducible separately:

```bash
python results/service-forecast-20261020/reproduce.py --output-dir /tmp/service-v2-repro
```

Native acceptance still requires a clean checkout and a verified native PID/procfs
namespace. The present review environment fails that preflight; its record is
retained rather than bypassed. Offline model/audit regression tests and archive
reproductions are reported separately and are not a native acceptance claim.
