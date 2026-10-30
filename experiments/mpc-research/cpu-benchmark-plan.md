# Frozen exploratory CPU comparison

This plan is saved before the new benchmark is executed. It is not a native
acceptance protocol and cannot establish a vLLM performance gain.

Development seeds1/2 select a fixed cap from16/32 and backlog hysteresis from
(512,2048), (2000,6000), (4000,10000) prompt-token threshold pairs, averaging
stationary/burst/shift workloads. Test seeds101–105 remain untouched until
selection. Run both fixed policies, the frozen selected feedback, same-model
one-control-move and3-control-move MPC on identical offered traces.
Randomize arm order deterministically per seed/workload. Do not tune from test.

Each arrival window is12s, initially empty, with no separate warmup, identical
across arms. Drain up to20s beyond offers; unresolved requests are not successes.
The25ms fluid truth plant is separate from the event forecast implementation:
prefill FIFO rate14000/(1+.08D) tokens/s with200-token per-request overhead;
per-stream decode100/(1+.04D+W/100000) tokens/s. This intentionally creates
model mismatch. It is not a simulator calibrated to vLLM.

Marked arrivals use thinning of a40/s Poisson proposal. Stationary/shift offer
14/s; burst offers8/s except36/s in the middle third. Prompt marks initially
128/512/2048, shifting to2048/4096/8192 in the second half of shift episodes.
Requested output maxima are64/128/256; hidden actual lengths are uniform from
max/4 through max;2% latent failures are fixed per offer across policies.
Only past observed arrivals/marks reach the scenario sampler. Actual future
arrival tape, EOS and failures are never passed to the controller.
CPU fixtures provide past scheduled timestamps; this is not proof the live
gateway knows client schedule or transport delay.

Forecast parameters remain the shared synthetic defaults, not fitted to test.
Use4 common seeded scenarios per decision, service scales.8/1/1.3,1s cadence,
5s scoring window and10s tail, actions16/32 and100ms planner wall limit.
Only the number of optimized slots differs for one-move versus MPC.
All configured limits/fallbacks and whole controller CPU/wall time are reported;
an implementation missing its deadline is not given a larger threshold to win.

Primary score is eventual-success and scheduled-TTFT<=.5s count divided by12s.
Report throughput, rejection, all-offered nonattainment, conditional TTFT,
complete-success E2E, per-request mean TPOT, controller CPU/wall cost, budget
variation and fallback incidence together. TPOT is not an individual token-gap
distribution. Backlog recovery is a descriptive two-consecutive-control-sample
return below pre-middle-third P95; it is not an independently validated SLO.
GPU utilization is unavailable, not zero.

Paired goodput differences average workloads within each of5 independent
synthetic seed blocks;2,000 block bootstrap resamples provide descriptive
percentile95 intervals. With five blocks, no multiplicity correction, short
cold-start episodes and a misspecified uncalibrated model, these are exploratory,
not confirmatory superiority/equivalence/native-risk claims. Report negative
results as well as positive ones. Native experimental acceptance remains governed
by benchmark-protocol.md and separate authorization.
