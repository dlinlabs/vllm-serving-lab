# Excitation v1: multi-level budget steps and a held-out acceptance test

Stage 3/4 of the [staged identification protocol](trace-identification.md#staged-gpu-protocol-do-not-start-with-a-matrix),
declared before collection. The [qualified captures](../results/gc-identification-20261015/report.md)
contain only two budget switches (96→64→96) in one run, so the
[development forecasts](forecast-baselines.md) cannot separate the effect of the
cap from the load. This protocol adds 9 switches per run across four levels,
covers every directed transition, and holds out one load window **and** one
control sequence for a single scored test. It adds no telemetry, solver,
actuation or controller code, and this PR runs no GPU capture.

Machine-readable declaration: [excitation-v1.json](excitation-v1.json). The
acceptance limits there are **`status: "proposed"`** until approved.

## What is fixed

| Item | Value |
|---|---|
| Levels / anchor | 48, 64, 80, 96 admitted-work cost cap; every schedule opens and closes at 96 (warmup cap) |
| Holds | 10 × 30 s over the 300 s arrival window (9 switches per run) |
| Schedules | [A](../results/budget-schedules/ident-multilevel-30s-v1-a.json) 96 64 80 64 48 80 96 48 64 96 (seed 1); [B](../results/budget-schedules/ident-multilevel-30s-v1-b.json) = A reversed; [C](../results/budget-schedules/ident-multilevel-30s-v1-c.json) 96 80 64 96 48 80 48 64 80 96 (seed 2, excluding A and its reversal) |
| Coverage | A + B contain all 12 directed transitions; each level appears 2–3 times per run |
| Development trace | `results/trace-experiment-96-r1/trace.json` (experiment preset, `[00:01, 00:06)`), SHA-256 `62f3f182…bc4e2e`, the same bytes as the existing captures |
| Held-out trace | Custom window `[00:07:00, 00:12:00)` of the same 1 MiB Azure conversation prefix (SHA-256 `242489b8…fa66fb16`, which ends at 00:16:23.95); sample probability 1, seed 1, output cap 128, same model |
| Replay settings | As the qualified gc-defer captures: TTFT SLO 0.5 s, no E2E SLO, 50 ms scheduling and budget lag, cache disabled, gateway + runner diagnostics, engine observations off, runner GC `defer` with 1536 MiB RSS cap, process sampler, and the runner defaults (3 warmup requests, 1000 pending, 120 s request/drain/startup timeouts) |
| Analysis | `identification_features.py` defaults (0.5 s grid, horizons 1/2/5 s, windows 1/5 s, engine age 3 s) and no offline E2E deadline (`analysis.features.e2e_slo: null`, so no `--e2e-slo` and no joint-outcome features); `forecast_baselines.py` split 2/3, ridge α 1 |
| Primary horizon | 5 s (about 1–3 s cap response plus margin; backlog bursts recur about every 5 s) |

The schedule candidates are enumerated, not sampled: anchor at both ends, no
repeated neighbours, level counts within one, every level pair adjacent at least
once (312 candidates). A seed picks one through SHA-256, so the files can be
regenerated exactly; `tests/test_budget_schedule.py` checks this and the hashes.

```bash
python make_budget_schedule.py --budgets 48 64 80 96 --anchor 96 \
  --hold-seconds 30 --duration-seconds 300 --seed 1 \
  --output results/budget-schedules/ident-multilevel-30s-v1-a.json \
  --reverse-output results/budget-schedules/ident-multilevel-30s-v1-b.json
python make_budget_schedule.py --budgets 48 64 80 96 --anchor 96 \
  --hold-seconds 30 --duration-seconds 300 --seed 2 \
  --exclude results/budget-schedules/ident-multilevel-30s-v1-a.json \
  --output results/budget-schedules/ident-multilevel-30s-v1-c.json
```

The script refuses to overwrite: the committed files are frozen.

## Runs (in this order, one backend launch)

| Order | Output directory | Trace | Schedule | Role |
|---|---|---|---|---|
| 1 | `results/multilevel-v1-a-r1` | development | A | development |
| 2 | `results/multilevel-v1-b-r1` | development | B | development |
| 3 | `results/multilevel-v1-c-heldout-r1` | held-out | C | final test, scored once |

Run all three on the same backend launch, like the existing captures, after the
stage 0 CPU gate on the exact commit. Each needs approval on the GPU host.

The protocol also declares the three existing captures (`gc-defer-fixed-96-r1`,
`gc-defer-fixed-64-r1`, `gc-defer-step-96-64-96-r1`) as development runs with
their fixed-96, fixed-64 and 96→64→96 schedules, so every analysis input is
checked against a declaration.

**48 is not yet qualified on this trace.** Stage 2 only qualified 64 on the
Azure trace; the earlier cost-48 result used a synthetic workload. If run A fails
integrity, load/control fidelity or a diagnostic gate, keep the failed capture,
stop, and report it. Do not lower the cap further, shorten holds, drop 48 or
edit a schedule and call it the same protocol; any change starts `excitation-v2`.

### Commands

Verify the inputs, then prepare the held-out trace **without inspecting its
arrivals or outcomes** before run 3 is scored:

```bash
sha256sum results/trace-experiment-96-r1/trace.json   # 62f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e
sha256sum results/trace-source/azure-conversation-prefix.csv   # 242489b81db35abe68b5d15d5ff3a7cc0e75a06184d104c7a0fe89a5fa66fb16
python prepare_trace.py \
  --source results/trace-source/azure-conversation-prefix.csv \
  --start 2024-05-12T00:07:00+00:00 --duration-seconds 300 \
  --sample-probability 1 --seed 1 --output-cap 128 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/trace-heldout-0007-input
```

Custom mode does not check the source hash itself, hence the explicit check;
the prepared trace also records it. Download the prefix as in
[trace replay](trace-replay.md) if it is missing. Then, for each run in order
(`RUN`, `TRACE`, `SCHEDULE` from the table):

```bash
nohup .venv/bin/python -u run_trace_pilot.py \
  --trace "$TRACE" --output-dir "results/$RUN" \
  --budget-schedule "$SCHEDULE" \
  --ttft-slo 0.5 --max-scheduling-lag 0.05 --max-budget-lag 0.05 \
  --cache-policy disabled --gateway-diagnostics --sampler-mode process \
  --runner-diagnostics --runner-gc-policy defer --runner-max-rss-mib 1536 \
  --protocol experiments/excitation-v1.json --protocol-run "$RUN" \
  > "/workspace/$RUN.log" 2>&1 < /dev/null &
```

With `--protocol`, the runner refuses to start unless the output directory name
equals `--protocol-run` and the trace (hash, or source hash, start, duration,
sampling, seed, output cap and model), the budget plan (schedule file hash and
budgets) and every replay setting equal the run's declaration. A test run is
refused until the acceptance `status` is `"frozen"`. The manifest then records
`experiment_protocol`: the protocol's SHA-256 (of its LF bytes), run, role and
acceptance status at collection time.

Inspect every gate in `report.json` before starting the next run.

## Analysis

Features for all six qualified captures (three existing, A, B, then C):

```bash
python identification_features.py \
  /tmp/gc-identification-20261015/results/gc-defer-fixed-96-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-fixed-64-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-step-96-64-96-r1 \
  results/multilevel-v1-a-r1 results/multilevel-v1-b-r1 results/multilevel-v1-c-heldout-r1 \
  --output-dir /tmp/excitation-v1-features
```

Development work (model choices, plots) may use only the first five; to look at
them without C, build a table without C and run `forecast_baselines.py` without
`--acceptance`. The single scored test:

```bash
python forecast_baselines.py /tmp/excitation-v1-features \
  --output-dir /tmp/excitation-v1-acceptance --acceptance experiments/excitation-v1.json
```

With `--acceptance`, the feature table must contain exactly the declared
training and test runs, every feature and forecast setting must match the
declared `analysis`, each run's effective E2E deadline must match
`analysis.features.e2e_slo` (here none: a table built with `--e2e-slo` adds
joint-outcome inputs and is refused), every capture's recorded trace, budget
plan and replay settings must match its run's declaration, and unqualified
captures are refused (exit status 2, no report). C is excluded from
the chronological and leave-one-run-out splits; the `designated_test` split
trains on the five other runs and scores C once. The report records the
protocol file's SHA-256 (of its LF bytes, as committed) and a verdict:

- **inconclusive** if fewer than 25% of C's rows are binding (headroom < 5) or
  a criterion has fewer than 60 evaluated rows;
- **fail** if any criterion fails (and the binding precondition holds);
- **pass** otherwise.

Predeclaration is **verified** only if C's manifest records collection under
this exact protocol file (same SHA-256, run and role) with `frozen` limits. A
pass becomes outcome `passed` and `model_readiness` `offline_acceptance_passed`
only then; a pass against proposed limits, after the protocol was edited, or
without the collection record is `not_established`. The exit status is 0
passed, 3 fail, 4 inconclusive, 5 not established. The record is written by the
runner: it rules out changing the limits after collection, not deliberately
repeated captures, which this document forbids.

## Proposed limits (pending approval)

All at h = 5 s, ridge, physical units (requests). Skill is 1 − MAE ratio
against persistence (P) or the training mean (M); optimistic p99 is the 99th
percentile of errors that make admission look safer (under-predicted backlog or
misses, over-predicted TTFT successes).

| Target | Subset | Limit | Development evidence (pooled chronological; worst leave-one-run-out) |
|---|---|---|---|
| Awaiting first content (backlog at t+5) | all | skill vs P ≥ 0.20, vs M ≥ 0.10, optimistic p99 ≤ 18 | 0.44 / 0.24, p99 16.1 (persistence 22.0); worst LORO −0.24 / 0.13, p99 17.4 |
| Admitted but missed TTFT (cohort [t, t+5)) | all | skill vs P ≥ 0.25, vs M ≥ 0.15, optimistic p99 ≤ 17 | 0.54 / 0.45, p99 14.1 (persistence 32.3); worst LORO 0.08 / 0.23, p99 16.6 |
| TTFT met (cohort [t, t+5)) | all | skill vs P ≥ 0.10, vs M ≥ 0.05, optimistic p99 ≤ 10 | 0.28 / 0.11, p99 6.8 (persistence 14.0); worst LORO 0.10 / 0.11, p99 9.0 |
| Backlog and admitted misses | near cap increase; near cap decrease | skill vs P ≥ 0 each (4 criteria) | One decrease and one increase only; the decrease gave backlog skill −0.74 (n = 30) |
| Backlog and admitted misses | — | fitted per-unit cap effect > 0 | Sign check on the model trained without C; unstable on development data |

Rationale: skill limits are about half the pooled development skill; tail limits
are the worst development leave-one-run-out ridge value, rounded up. "Near" means
within 10 s after a cap change or with a realized change inside the horizon. The
per-unit effect is the summed ridge response to the realized cap mean and end; a
larger cap should admit more work and so raise backlog and admitted misses. It
describes the trained model, not test-set error. All criteria are
`realized_control_conditioned`: the controls are recorded cap applications, so a
pass shows the identified dynamics hold on unseen load given the applied cap,
not that a forecast from a plan known at t is deployable.

These limits are **not** easy on existing data: holding out the fixed-64 capture
would fail the backlog and missed-TTFT skill limits, and the only decrease near a
change was worse than persistence. Adding A and B to training is what the test
is meant to measure.

Before running C, the user approves or edits the limits, `status` is set to
`"frozen"` in the same commit, and that commit is recorded. C is then collected
with `--protocol` from that commit, so its manifest binds the frozen file.
Do not change the protocol after C is captured: any edit changes its hash and
the outcome becomes `not_established`. A fail or inconclusive result is
reported as such; it is not rerun with new limits under this protocol name.

## Decision after the test

- **Pass**: offline acceptance only. Next is a shadow-mode feed (no actuation)
  with its own approval, not a closed-loop claim.
- **Fail or inconclusive**: report which criteria failed. Further collection
  or model changes need a new protocol version with a new held-out window.

Limits of scope: one trace source, one GPU, a fixed output cap and one
repetition per schedule. A pass does not establish an optimal cap or a
transferable model.
