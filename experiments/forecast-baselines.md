# Offline causal features and forecast baselines

Development tooling for the step after the
[qualified budget-identification capture](../results/gc-identification-20261015/report.md):
turn audited captures into a time-causal feature table and score simple
multi-step forecasts before any model or MPC work. It adds no telemetry,
solver, actuation or GPU run, and commits no results. The three existing
long captures are **development data**: they shaped this design, so their
scores choose what to measure next and cannot validate a model. Without
designated test runs, every report states `evaluation_role=development` and
`model_readiness=not_established`. A predeclared held-out test is scored with
`--acceptance` (see [excitation v1](excitation-v1.md) and below).

## Run

```bash
python3 results/gc-identification-20261015/extract_bundle.py \
  --output-dir /tmp/gc-identification-20261015

python identification_features.py \
  /tmp/gc-identification-20261015/results/gc-defer-fixed-96-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-fixed-64-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-step-96-64-96-r1 \
  --output-dir /tmp/forecast-features

python forecast_baselines.py /tmp/forecast-features --output-dir /tmp/forecast-baselines
```

Each run is first re-audited with `analyze_trace_dynamics.analyze_run`
(about 1.5 minutes per five-minute capture on one CPU core); unqualified captures
are refused unless `--allow-unqualified` is given, and stay marked: the forecast
report and summary then list each run's qualification and reconstruction verdicts,
whether the override was set, and a warning for every unqualified input. Raw
directories are never written. Defaults: 0.5 s grid (`--step`), horizons
1/2/5 s, trailing windows 1/5 s, engine samples older than 3 s treated as
missing. The grid is a table resolution, not a validated control period.

## Feature table contract (`identification_features.py`)

Outputs `features_summary.json` (parameters, input hashes, capture identity —
trace provenance, budget plan, replay settings and any collection-time protocol
binding —, column glossary, reconstruction check, label coverage),
`<run>-features.jsonl` (one row per decision time `t`) and
`<run>-request-labels.jsonl`.

- **Features at `t`** read only what a gateway-co-located controller had by `t`:
  gateway lifecycle events at their emission time, budget changes at their
  recorded *application* time, and engine scrapes whose body was fully received
  by `t` (with their age). An external poller would see staler state.
  - Applied cap/version, seconds since the last application and last value change.
  - Reserved admitted cost, headroom (negative after a cut below reserved work),
    in-flight, awaiting first content, oldest awaiting age, streaming, active.
  - Trailing `(t-w, t]` counts of received/accepted/rejected/first-content/
    completed/failed events, admitted input tokens and offered cost.
  - Receipt-anchored deadline proxies that resolve at the deadline itself
    (`resolved_ttft_met`, `resolved_admitted_ttft_missed`, and with an E2E SLO
    `resolved_joint_met`), so no outcome is used before it is knowable.
  - Engine waiting/running/KV usage, nulled with `engine_missing=true` when absent or stale.
- **Gateway state is reconstructed, then checked**: the event replay is compared
  with every gateway `snapshot` taken under the state lock (cost tolerance 1e-6,
  age 1 ms, counts exact). Every snapshot must have a finite sample time and a
  finite value for all eight compared fields; a missing, null, boolean or
  nonfinite value, an invalid timestamp, any mismatch or an inconsistent history
  marks the run `state_reconstruction.valid=false`, and the forecast step refuses
  it. The summary also records the snapshot span and largest gap.
- **Controls** (`control.<h>`): time-weighted mean and final cap over `(t, t+h]`
  from recorded application times. They are control inputs, not features, and
  they are *realized*: they include actuation timing that was unknown at `t`
  (a change planned for `t` but applied 10 ms or 0.8 s later gives different
  values). Reports label this `control_conditioning=realized_control_conditioned`:
  valid for system identification conditioned on the applied cap, not a
  deployable planned-control forecast. A deployable variant would build control
  inputs from decision-time information plus an explicit actuation-delay
  assumption and keep applications as evidence; it is not implemented. In these
  captures the step changes applied 22 ms and 1 ms after their planned times.
- **Labels** (`labels.<h>`), never features: gateway state at `t+h`; event counts in
  `(t, t+h]`; and the all-offered cohort scheduled in `[t, t+h)` — offered,
  admitted, unknown, TTFT met and admitted-but-missed, measured from scheduled
  arrival, including rejected and failed requests. Met/missed counts are null
  when any cohort outcome is unknown. `cohort_resolved_s` records when the last
  cohort outcome became observable. Drain-only windows are kept but flagged by
  `label_window_complete`; labels past the recorded log are null.
- **Request labels** hold each offered request's outcome, scheduled-arrival
  TTFT/E2E and `resolved_s`, so any E2E deadline can be applied offline later.

No E2E SLO is declared by these captures, and the tools have no default.
`--e2e-slo` adds joint labels recorded as `e2e_slo_source=analyst_offline`; it
is refused if a run declares a different value. It also adds
`resolved_joint_met` feature windows, so `--acceptance` refuses a table whose
effective deadline differs from the protocol's declared one (below).

## Baselines (`forecast_baselines.py`)

Outputs `forecast_report.json` and `forecast_summary.md`. Each target and
horizon is forecast directly (no recursion):

| Model | Forecast |
|---|---|
| Persistence | Current state; flows and cohorts extend the trailing rate from the window nearest the horizon (cohorts via the receipt-anchored proxies). |
| Training mean | Static mean of the training labels. |
| Ridge (ARX-style) | Standardized features plus realized cap mean/end over the horizon; training-mean imputation; clipped at zero; `--alpha` 1. |

Evaluation never shuffles neighboring rows:

- **Chronological**: cut each run at 2/3 of its duration (`--split-fraction`);
  train only on rows whose label was fully known by the cut (`t+h`, or
  `cohort_resolved_s` for cohort labels); evaluate rows at or after the cut.
  One model is fitted on all runs' training rows and scored per run and pooled.
- **Leave one run out**: train on the other runs, evaluate the whole held-out run.
- **Designated test** (`--test-runs NAME...`, or the `test_runs` of `--acceptance`):
  those runs never enter the two splits above; one model trained on every other
  run scores them once (per run, and pooled when there are several).
- Subsets: all rows; `near_budget_change` (within 10 s after a cap change or
  with a realized change inside the horizon); and its split by direction,
  `near_budget_increase` / `near_budget_decrease`.
- `budget_effects` records, per split, the ridge forecast's summed response to
  one unit of realized cap mean and end (a property of the fitted model).

### Predeclared acceptance (`--acceptance PROTOCOL.json`)

The protocol's `acceptance` section (schema 1) declares `status`
(`proposed`/`frozen`), `model: ridge`, `training_runs`, `test_runs`,
preconditions (`binding_headroom`, `min_binding_fraction`, `min_subset_n`) and
criteria, each with an id, horizon, target, subset, metric
(`skill_mae.ridge_vs_persistence`, `skill_mae.ridge_vs_mean`,
`models.ridge.mae|p99_abs|optimistic_p99`, or `budget_effect_per_unit`) and
exactly one of `min`, `max`, `greater_than`.
The protocol file also declares `traces`, `schedules`, `replay` settings,
`runs` (name, trace, schedule, role) and `analysis`: every feature-table
parameter, `split_fraction`, `alpha`, and `features.e2e_slo` (the offline E2E
deadline, or `null` for none; required); `training_runs` and `test_runs` must be
exactly its development and test runs. The run is refused (exit status 2, no
report) if the feature table's runs differ from the declared training plus test
runs, if any feature or forecast setting differs from or is missing in the
protocol's `analysis`, if a run's effective E2E deadline, its source or the
presence of joint-outcome features does not match `analysis.features.e2e_slo`, if
any capture's recorded trace identity, budget plan or replay settings differ
from its run's declaration (`experiment_protocol.py`; feature tables record this
as `capture` per run), or with `--allow-unqualified`. The verdict is
`inconclusive` when the binding fraction of test rows (headroom below the
threshold) is too low or a criterion has too few rows, otherwise `fail` if any
criterion fails, else `pass`. Predeclaration is verified when every test
capture's manifest records collection with `run_trace_pilot.py --protocol`
under the same protocol bytes, run and role while `status` was `frozen`. The
outcome is `passed` (exit status 0, `model_readiness`
`offline_acceptance_passed`) only for a pass with verified predeclaration;
otherwise a pass is `not_established` (5), and `fail` (3) and `inconclusive` (4)
are reported as such. Like every
report here, an acceptance run is `realized_control_conditioned`: it shows the
fitted dynamics predict held-out load given the applied cap, not that a forecast
from a plan known at `t` is deployable.

Metrics are in physical units: n, MAE, RMSE, bias, p90/p99/max absolute error,
and skill (1 − MAE ratio) against persistence and the training mean.
**Optimistic** errors — those that make admission look safer than it was —
are reported separately as fraction, p90, p99 and max: under-prediction of
backlog, oldest wait, reserved cost, in-flight, arrivals and admitted misses;
over-prediction of first content, completions and deadline successes.

## Limits

- Only one capture changes the cap (two applications), so budget effects are weakly
  identified and the `near_budget_change` subset is tiny.
- All captures replay one trace on one GPU; rows overlap in time, so `n`
  overstates independent evidence.
- No numeric error or optimistic-tail limits are established by the development
  data. [Excitation v1](excitation-v1.md) proposes limits at 5 s and a held-out
  window plus unseen schedule, following
  [What "ready for MPC" must mean](trace-identification.md#what-ready-for-mpc-must-mean);
  they must be approved and frozen before that test is captured.
