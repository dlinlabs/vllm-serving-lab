# Delayed-arrival validation: next engineering design

Implementation follow-up: [the executable v2 pilot](prefill-effective-arrivals-v2.md) now exists with CPU validation; first native delayed qualification is pending. The original planning notes below retain their design-stage scope.
The v1 native pilot establishes a useful effective candidate for simultaneous
arrivals. This proposal tests its next missing mechanism: a new request entering
while prefill/decode work is already in flight. It does not change v1 data,
thresholds, frozen coefficients or acceptance flags.

Eight new development-screened cells use the previously evaluated mixed
foreground [1024,1024,7168,7168], plus a512-token/128-output sentinel arriving
at50/200/400/800ms after the foreground barrier, with D0/D16 backgrounds.
The main four foreground requests arrive together. All prompts stay <=7168
and the batch budget remains2048. The fifth foreground request is a NEW joint
composition, not qualified merely by marginal prompt-length support. All four D16
point forecasts keep their backgrounds alive, but actual lifetime still needs a
native integrity check; no background is artificially extended to force a pass.

The generated JSON proposal binds the original v1 freeze and runtime and includes
own-rollout forecasts, not new observations. It is produced by
scripts/analyze_prefill_effective_capture.py. That original proposal JSON is not consumed by an endpoint. The separate executable v2 JSON is selected by prefill_host.py arrivals; calibrate remains the v1 pilot.

Implementation checklist (covered by the v2 implementation and CPU tests):

1. A dedicated source-bound token manifest, trial order and endpoint mode. Four
   fresh OFF/LIGHT pairs per delayed cell, balanced directions, fixed randomized
   block order and zero-offset drift anchors before/after scored delayed trials.
2. On each NEW host, repeat original six-cell training and its quality/sensitivity
   gates, then freeze parameters and all delayed forecasts BEFORE scored delayed
   observations. Do not assume4090/model identity makes archived coefficients
   transferable across CPU quotas, clocks, KV capacities or runtime changes.
3. Keep the four-parameter architecture fixed for this test. Report prior D16
   underprediction explicitly. If structure or errors require a richer law,
   treat the failed new run as development and preregister fresh confirmation;
   do not refit from held delayed outcomes and relabel them a pass.
4. Audit three distinct clocks. Primary target-to-first latency matches the
   simulator's ideal offered arrival. Also report actual queue-post-to-first,
   posting lag and queue-post-to-engine-enqueue. Existing max5ms absolute posting
   lag integrity applies. Long engine queue residence is an outcome, not generator
   posting error. The observed v1 model predicts some >100ms queue residence in
   these cases; that is a mechanism to test, not permission to relax posting gates.
5. Preserve native progress/dispatcher/forced-output/source checks and fail on
   structure divergence, invalid backgrounds, event loss or changed runtime.
   Freeze quality, endpoint and signed-error reporting rules before execution.
   Do not derive a production admission safety margin from four repeated trials
   or the v1 maximum residual alone.

No HTTP/gateway or dynamic cap runs in this step. After delayed-arrival validation,
qualify the actual HTTP path and state observations needed for a fixed versus
state-aware cap comparison. MPC remains downstream of demonstrated predictive
support and controller benefit. A passed reduced endpoint pilot is not evidence
that all21 physical stage costs were identified.
