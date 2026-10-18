# Shared-client excitation: experiment and evidence index

Published 2026-10-18 UTC (2026-10-17 in Los Angeles). The rental GPUs have been
returned. **C remains uncollected and unscored; MPC has not been evaluated.**

## What changed and why

The original per-request HTTP-client excitation A stopped at 210 s: the scheduled
96→48 budget application lag was 52.884 ms against the 50 ms limit. The command
was applied, but the run correctly aborted and remains invalid for a whole-run
comparison. A 39.960 ms HTTP client construction overlapped that dispatch-to-
application interval; no gateway GC overlapped it. These are elapsed spans,
not exclusive CPU attribution.

The opt-in `--shared-upstream-client` mode creates one transport at gateway
startup, keeps request response/reservation cleanup independent, prevents
response-cookie carryover, and closes the transport at shutdown. It preserves
legacy mode and the original gates. Runtime `d0ccb2c` passed the complete native
Linux CPU/mock suite on two rental hosts: 476 cases on each, no failures/errors/
skips. That evidence is for the collection runtime, not an assertion that a
documentation-only publication commit was separately run on a rental GPU.

## Experiment ledger

All long scheduled captures use the existing Azure development trace,
7,643 offered requests / 300 s, fixed output cap 128, Qwen3-4B-Instruct-2507,
context 8192, prefix cache disabled, TTFT SLO 0.5 s, unchanged 50 ms load/control
gates, process sampler, gateway/runner diagnostics and bounded runner GC deferral.
The fixed-96 smokes replay 787 requests over 30 s.

| Capture | Declaration | Measurement qualification | Outcome / role |
|---|---|---|---|
| [Original smoke](captures/excitation-v1-smoke-96-r1/report.json) | Runtime smoke; no collection-time binding | Six gates pass | Original runtime check; separate legacy mode |
| [Original A](captures/multilevel-v1-a-r1/report.json) | frozen excitation-v1 | **Invalid / aborted** at 210 s | Failure evidence, excluded from shared-model training |
| [Shared smoke/A](captures/shared-client-a-r1/report.json) | shared-client-qualification-v1 | Six gates pass for both | Development A, 572 SLO successes / 1.907 per second |
| [Fresh-host smoke/B](captures/shared-client-b-r1/report.json) | shared-client-b-v1 | Six gates pass for both | Development B, 736 SLO successes / 2.453 per second |
| Held-out C | Not collected | Not applicable | Retained for a future, separately frozen model test |

A schedule: `96,64,80,64,48,80,96,48,64,96`.
B schedule: `96,64,48,96,80,48,64,80,64,96`.
Each hold lasts 30 s. Together they contain 18 noninitial changes and cover all
12 directed pairs among four levels; they are not hundreds of independent
actuation experiments. A/B use different rental hosts with the same RTX 4090,
595.91.07 driver and core runtime versions. Full-run score differences are
descriptive and do not isolate schedule or host effects.

![A/B dynamics](figures/ab-overview.png)

## What the data establish

- Both A/B saved reports reproduce exactly from raw files. All 15 recorded
  runtime/dependency file hashes match `d0ccb2c`; recorded `git_dirty=true`
  is retained. The [verification record](verification.json) contains every
  comparison, and original manifests remain unchanged.
- A's 306 and B's 307 snapshots match lifecycle reconstruction for all eight
  required state fields, with no missing fields, replay issues or mismatches.
- All 7,646 per-run request diagnostic summaries, including three warmups,
  complete cleanup. Per-request construction/close phases are `not_reached`
  in shared mode, as intended.
- The same 30 s 96→64 cut first reaches reserved cost≤64 after 0.947 s in A
  versus 2.744 s in B. This is a first crossing, not a validated settling time.
- B has gateway monitor delays of 343 and 233 ms overlapping loop-thread
  generation-2 GC spans of about 265 and 248 ms. Timing overlap is not exclusive
  attribution; these do not invalidate the passing load/control gates.

## Model findings and limits

The published [baseline report](analysis/forecast_summary.md) is explicitly
development-only. A/B each contribute 600 overlapping 0.5 s rows. Chronological
and leave-one-run-out validation exclude labels unavailable at the training cut;
the latter still reuses the same offered arrival trace, not new unseen traffic.

Five-second chronological pooled MAE (requests, lower is better):

| Variant | First-token backlog | SLO successes | Admitted SLO misses |
|---|---:|---:|---:|
| Persistence | 4.14 | 7.20 | 7.69 |
| Original ridge | 3.64 | 7.49 | 8.42 |
| Phase features, planned cap | 3.71 | 7.81 | 8.52 |
| Drop administrative budget version, planned cap | 3.72 | 6.98 | 7.72 |
| Fixed nonlinear baseline + phase/history, planned cap | 2.96 | 7.93 | 6.96 |
| State×cap interactions, planned cap | 3.24 | 7.50 | 7.58 |

![Forecast comparison](figures/forecast-comparison.png)

Planned-cap diagnostics use the predeclared schedule, observed state at t, and
a fixed 50 ms actuation-delay assumption; future actual applications are not
inputs. Original baseline controls remain `realized_control_conditioned`.
An explicitly labelled future-arrival-summary diagnostic is nondeployable and
never presented as an online feature. Better short-term forecasts or backlog
scores do not establish reliable action ranking. Cap-response coefficients in
the original ridge even change sign across training splits.

The next candidate is an offline short-step service/state-transition model with
conservation/nonnegative constraints, plus causal token-service features from
the already recorded engine counters. Validate multistep rollout and control
sensitivity on A/B before collecting C. No candidate here passed a final model
test, and no goodput improvement from closed-loop MPC is claimed.

Read the [full Chinese analysis](analysis-zh.md), [computed capture facts](facts.json),
[transition states](transitions.json), and diagnostic JSON/scripts under [analysis/](analysis/).

## Complete raw data and reproduction

The [raw-bundle index](raw-bundle/index.json) preserves three byte-identical
original uploads, split into 52 binary parts (53,739,633 bytes total), containing
139 original regular files. It records SHA256 for every part, reconstructed
archive and extracted file. `captures/`, `setup/` and `native-validation/` are
convenient browsing subsets, not complete audit inputs.

```bash
python3 results/shared-client-20261018/extract_bundle.py --output-dir /tmp/shared-client-captures
python3.12 -m venv /tmp/shared-client-analysis-env
/tmp/shared-client-analysis-env/bin/python -m pip install -r results/shared-client-20261018/requirements-analysis.txt
/tmp/shared-client-analysis-env/bin/python results/shared-client-20261018/reproduce.py \
  --raw-root /tmp/shared-client-captures --output-dir /tmp/shared-client-analysis --extended
/tmp/shared-client-analysis-env/bin/python results/shared-client-20261018/analysis/make_figures.py \
  --analysis-root /tmp/shared-client-analysis --raw-root /tmp/shared-client-captures \
  --output-dir /tmp/shared-client-figures
```

No GPU, vLLM installation or model download is required for offline reproduction.
The source-file check refuses a changed runtime. If main later changes those
files, use `--code-root` pointing to a clean `d0ccb2c` checkout while keeping this
published reproduction entry point. Never use `--allow-unqualified` to turn the
failed original A into a qualified training capture.

Frozen declarations retain their original hashes. The B declaration source
commit `ad2a3a826d75340773fc26fe3f492330b5fae8ba` is retained in merge history;
reproduction does not require keeping a temporary branch alive.
