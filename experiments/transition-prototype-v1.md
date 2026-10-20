# Planned-cap state-transition prototype (development only)

This offline prototype asks whether short-step flow prediction can improve on
simple forecasts before building an MPC controller. It changes no serving,
gateway, collection protocol, threshold or original result. C is not collected,
loaded or scored. The [settings record](transition-prototype-v1.json) describes
retrospective development constants, not predeclared acceptance limits.

## Inputs and causal boundary

Use six qualified full cap-range captures (16-r1, 32-r1/r2, 64-r1, 96-r1/r2)
and the two qualified shared-client A/B captures. Exclude all smoke captures and
failed cap16-r2 from scoring. The partial cap-range suite remains incomplete;
qualifying its individual captures does not turn it into a complete experiment.
The existing auditor independently reproduces every selected saved report, and
the state replay must agree with every gateway snapshot. Raw files are read only.

At each integer second t, inputs are reconstructed gateway Q/D/R, trailing
five-second arrival/cost/first-token/completion/deadline proxies, and optionally
the latest causal engine waiting/running/KV gauges and their missing flag.
Q = admitted without first content, D = first content without a terminal event,
R = unreleased reservation cost. Q is not the engine's internal queue. Gauges
are aggregate observations, not request-linked prefill or service capacity.
Co-located gateway-state visibility is assumed; this is not a validated external
online feature channel. Waiting/streaming ages and request cost composition are
not modeled in this minimal baseline.

The control input comes from the premeasurement replay plan. Planned caps apply
at integer-second left endpoints with zero actuation delay. Observed future
applications are excluded, unlike the earlier realized-control identification
baseline. Nonzero actual delay remains a modeling error, not leaked knowledge.
The initial observed state can include work admitted under a previous cap.

## Transition and reward

Six separate ridge heads predict one-second admissions A, first-content F,
exits-before-first E_Q, streaming exits E_D, reservation cost in C_in and out C_out:

```
Q_next = Q + A - F - E_Q
D_next = D + F - E_D
R_next = R + C_in - C_out
```

Training labels reconcile these equations against independently reconstructed
states, including warmup survivors, failed/cancelled requests and same-second
admission/completion. Flows use (t,t+1]; scheduled-arrival cohorts use [t,t+h).

Forecast flows are projected to prevent negative states and impossible aggregate
event counts; an empty population releases all remaining reservation. Projection
magnitude is retained. Reserved cost is **not** clamped to a reduced cap.
This does not enforce request-level cost mixtures or admission ordering; it is
an aggregate transition baseline, not an exact grey-box serving simulator.
Only Q/D/R advance during rollouts. Other historical inputs and engine gauges
freeze at the origin; arrival forecasts are thus a persistence approximation.

A separate one-second reward head learns eventual completed requests with
scheduled-arrival TTFT <=0.5 s, summed over rollout steps. This is cohort SLO
success count, not a first-token-only count or a claim that those requests finish
within the prediction horizon. Receipt-deadline proxies used as historical
features do not require future knowledge and are not the same target. No new
TTFT violation-rate constraint or P99 guarantee is introduced.

## Evaluation

- Horizons: 1/2/5 s. Resolution does not establish a control period.
- Models: state/rate persistence, training mean, direct planned-cap ridge,
  conservation rollout ridge, and the rollout with causal engine gauges.
- Ridge alpha=1, train-only imputation/standardization, no parameter search.
- Direct ridge uses planned first/mean/end cap. Rollouts use each planned step.
- Chronological split: first two thirds of each run for fitting, last third for
  scoring. State-flow labels must finish by the cutoff; reward labels and direct
  heads must also wait for every cohort outcome to resolve. No random row split.
- Leave-one-run-out and fixed/scheduled transfer are additional development
  checks. They reuse already examined captures, not an untouched acceptance set.
- Score each run separately: MAE/RMSE, tail error, directional optimism, and
  skill against persistence. Also report within 10 s after an observed cap change
  or a horizon containing a planned change. Empty subsets stay empty.
- In transfer checks, explicit cap-range extrapolation is scored but counted.
  The forecast API rejects it by default. A fitted numeric range does not prove
  support for an unseen cap or state-action combination.

The fixed and A/B datasets differ in rental host/runtime, so transfer failures
are not attributable solely to cap or model structure. A/B cap range is 48–96;
there is no low-cap switching validation. Neighboring and overlapping windows
are correlated. No confidence interval treating seconds as independent trials
is appropriate. All captures replay the same arrival window: leave-one-run-out
tests replicate/host transfer, not generalization to an unseen workload. Pooled
chronological cuts are relative to each run, not a calendar-time online retraining
simulation. Forecast accuracy alone does not establish correct causal
action ranking or MPC goodput improvement.

## Reproduce without GPU

Python 3.12 and the CPU dependencies in `requirements-cpu-tests.txt` suffice.
All output directories must be fresh. From the repository root:

```bash
python results/cap-range-20261020/extract_bundle.py --output-dir /tmp/cap-transition-raw
python results/shared-client-20261018/extract_bundle.py \
  --archives shared-a shared-b --output-dir /tmp/shared-transition-raw

python transition_features.py \
  /tmp/shared-transition-raw/shared-a/results/shared-client-a-r1 \
  /tmp/shared-transition-raw/shared-b/results/shared-client-b-r1 \
  /tmp/cap-transition-raw/results/cap-range-v1/cap-range-v1-fixed-16-r1 \
  /tmp/cap-transition-raw/results/cap-range-v1/cap-range-v1-fixed-32-r1 \
  /tmp/cap-transition-raw/results/cap-range-v1/cap-range-v1-fixed-32-r2 \
  /tmp/cap-transition-raw/results/cap-range-v1/cap-range-v1-fixed-64-r1 \
  /tmp/cap-transition-raw/results/cap-range-v1/cap-range-v1-fixed-96-r1 \
  /tmp/cap-transition-raw/results/cap-range-v1/cap-range-v1-fixed-96-r2 \
  --output-dir /tmp/transition-features
python transition_baselines.py /tmp/transition-features --output-dir /tmp/transition-scores
python -m pytest -q tests/test_transition_model.py tests/test_identification_features.py tests/test_forecast_baselines.py
```

Extraction reruns complete capture audits and can take several minutes. It prints
progress after each qualified run; it does not start new GPU requests. Dataset
tables and reports retain raw/table/code fingerprints, fitted coefficients and
train-only preprocessing. Output creation refuses replacement of existing paths.

## Decision after this prototype

Inspect errors by state, cap, horizon and change direction before choosing a
controller. If the baseline is unreliable, improve the service model or identify
specific missing measurements; do not optimize a controller against a misleading
simulator. Counter rates, nonlinear interactions and evolving history are
possible follow-ups, not changes silently selected for this first score.
New low-cap switch data may still be needed. A final GPU protocol must separate
calibration from formal evaluation, preserve the original workload alongside
labeled synthetic variants, compare a strong fixed and reactive baseline, and
freeze the final model/metrics before touching C.
