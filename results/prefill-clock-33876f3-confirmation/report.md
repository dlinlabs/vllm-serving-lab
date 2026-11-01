# First fresh native event-clock confirmation

**The separately frozen B2048 event-clock pilot passes all declared gates.**
This is new GPU evidence after the training-only development repair. The original
four-term delayed pilot remains rejected. Full-model, physical-cost, HTTP, cap,
MPC and complete-original-suite acceptance remain false.

## Evidence identity and independent audit

Uploaded archive1793071036756838596.tar.gz:9,160,093 bytes, SHA256
2d1a78e03ca81705257910a1428e76eeb64c7904a9dcb17a0c034a417cdd9fe2.
All157 extracted files (109,119,397 bytes) match the archive exactly. No raw
capture, freeze, threshold or saved native report is changed.

The read-only analyzer verifies the COMPLETE156-file source inventory against
Git commit33876f34b8e540c7008ecf5e64880289a60e2b84, reconstructs those exact
sources in an isolated temporary tree and runs the ORIGINAL clock audit. It
reproduces audit_valid=true/candidate_passed=true/model_accepted=false. Original
native acceptance/JUnit validates1116 passing cases, zero failures/errors/skips,
including all five required trace configurations. Installed vLLM0.30.0 source
identities retain all38 pins. This audit does not execute the GPU again.

The recorded runtime is RTX4090, Torch2.13.0+cu130/CUDA13.0, B2048, max-seqs32,
context8192 and5594 KV blocks of16 tokens. Model Qwen/Qwen3-4B-Instruct-2507 is
pinned to cdbee75f17c01a7cc42f958dc650907174af0554. One native engine preserves
the reviewed V2 two-slot/FCFS/chunked-prefill configuration, without prefix cache
or speculation. Root-path CPU/memory cgroup limits are not reported by the host
metadata; no CPU quota is inferred from its96-entry affinity list.

## Collection and frozen model

All136 trials finish once in598.079s, no failures or rejection reasons:
16 LIGHT warmups,48 training OFF/LIGHT trials,4 anchor-before,64 delayed and4
anchor-after. Totals:60 OFF/76 LIGHT trials,1596 request instances and15,135 LIGHT
executed steps. Observed worker modes: NONE888/FULL14,239/PIECEWISE8. Mode counts
are not deep-trace qualification of exclusive GPU streams or graph service.

The source-bound prospective plan predates collection. New ct- trial IDs and
paired-content keys replace the development pools. This is fresh measurement
and content in the same workload cases, not previously unseen arrival patterns.
Only24 TRAIN OFF endpoint trials and24 TRAIN LIGHT event trajectories fit six
positive effective terms. The collector freezes before confirmation/anchors;
source/runtime/token, raw-label, training hashes, regenerated forecasts and the
saved report independently recompute. Hashes provide artifact consistency, not
signed attestation of actual execution order.

Both fixed optimizer starts converge (9/6 evaluations), agree and satisfy the
original boundary/sensitivity rules. Local rank6, normalized condition10.009909;
training trial/phase-balanced event MAE9.805ms. Recomputing training event scores
from raw labels/own rollout exactly matches the freeze, without fitting again.
All76 LIGHT own-work signatures, INCLUDING warmups, match the frozen independent
rollout. Work never enters prediction as a captured future covariate.

The six fresh normalized-second coefficients are:

`[0.009446870571235165, 0.09113373252069282, 0.005476838455750142, 0.01029740558578274, 0.0010552457540133641, 0.005439278740547326]`

The fresh freeze SHA256 is
3b9ce9241a11f19634a5452eb030761bd8cff67ee4a3642dd40897ac54907950.
It records fresh_native_training_events and differs from the retained development
freeze. Coefficients were refitted on this host. In particular the effective mixed
term changes about14% relative to development; this is not proof that coefficients
transfer unchanged across machines or identify exclusive physical service.

## Fresh delayed errors

Primary clock is declared target-to-EngineCore first/final output commit, with
forced128-token foreground/sentinel outputs. Scores use each request position's
median of four OFF repeats. Across8 delayed cases and40 positions:

| Metric | Mean absolute error | Maximum absolute error | Maximum relative error |
|---|---:|---:|---:|
| Target to first output | 3.174ms | 5.655ms | 5.076% |
| Target to final output | 50.225ms | 75.717ms | 3.004% |

These are position medians, not per-request tail bounds or p99. D0 first-output
MAE3.943ms/bias-3.376ms; D16 MAE2.404ms/bias+1.269ms. Completion error remains
systematic: D0 bias-27.100ms, D16+73.351ms, despite passing the original endpoint
gates. No uncertainty margin or production SLO guarantee follows from these
residuals.

All eight queue-residence gates retain max(10ms,20% of observed median):

| Case | Observed median ms | Prediction ms | Signed error ms | Limit ms | Verdict |
|---|---:|---:|---:|---:|---|
| at50-D0 | 61.399 | 55.734 | -5.665 | 12.280 | pass |
| at50-D16 | 77.431 | 77.398 | -0.033 | 15.486 | pass |
| at200-D0 | 23.800 | 17.662 | -6.138 | 10.000 | pass |
| at200-D16 | 50.024 | 47.053 | -2.970 | 10.005 | pass |
| at400-D0 | 100.821 | 103.181 | 2.360 | 20.164 | pass |
| at400-D16 | 145.662 | 147.969 | 2.307 | 29.132 | pass |
| at800-D0 | 108.827 | 108.908 | 0.081 | 21.765 | pass |
| at800-D16 | 25.099 | 26.945 | 1.846 | 10.000 | pass |

The previously failed800ms/D16 case now passes on fresh evidence: predicted
26.945ms, median25.099ms, error+1.846ms (limit10ms). Its four actual OFF queues
range23.981–25.523ms, with maximum posting lag0.121ms. Across all32 delayed OFF
sentinels the maximum posting lag is1.744ms, below the unchanged5ms gate.
This is not a paired comparison of old and new models on this new host: prior
observations/predictions came from a different run/model/host fit.

Primary endpoints, exact native/own-work structure, background lifetime,
training/held observer and repeat gates, and BEFORE/AFTER anchors separately
all pass. Anchor first-output MAEs are4.160ms before and3.056ms after; their
maxima are10.988ms and5.045ms. Largest held incremental OFF/LIGHT first-output
delta is5.551ms; largest OFF repeat spread6.551ms. These meet the declared
absolute/relative gates, but OFF retains control-hook overhead. They are not
measurements of uninstrumented production overhead.

## Interpretation, review and next step

The repaired effective clock has reproduced its controlled pilot accuracy with
fresh training and confirmation data. The original rejected four-term outcome
and earlier development attempts stay intact. Analysis only reads data and runs
the exact old auditor; it never changes coefficients or thresholds, retries
collection, chooses a best run or promotes the model to cap/MPC acceptance.

This is fixed B2048, D0/D16, known forced lengths and controlled arrivals from
an empty engine. The7643 Azure requests verified in setup were NOT replayed here.
No HTTP/gateway/stream timing, natural-EOS length uncertainty, changing budgets,
preemption or live-state initialization has been qualified. The21-channel
exclusive physical-service estimator remains blocked. Original serial-overlap
and full-experiment guards remain unchanged.

The next development step is HTTP timing and online state-observation/initialization
qualification, followed by fixed-versus-state-aware cap tests and MPC. Designing
that protocol can proceed on CPU; this archive can be fully analyzed after the
rental is returned. A new GPU run is needed when that separate validation is ready.

**64 local read-only analysis/design tests pass, zero failures/errors/skips**,
including12 new inventory/verdict/clock regressions. Compilation and registered
fingerprints pass. Results are recorded in validation.json and local-cpu-tests.xml. No new local native/full CPU/resource/GPU acceptance is
claimed; the1116 native tests belong to the original33876f3 capture.
