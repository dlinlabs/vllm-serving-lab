# PR32 collection implementation: revision10

The repository now includes an opt-in vLLM0.30.0 V1/V2 collector and complete
collection runner. [Native-host commands](../../experiments/prefill-service-v1.runbook.md)
cover exact token preparation, CPU acceptance and one-command collection of the
frozen212 trials plus four Azure development attempts.

Implemented: actual scheduler/worker input-position and count reconciliation,
engine-side batch release, timed sentinel queue posting, bounded async step/event
records, delayed CUDA event queries, graph/padding/capture/replay metadata,
source/config/GPU/KV guards, observer/drift checks, HTTP bridge, Azure replay,
deep traces, mode-aware calibration-only NNLS/LOCO with a plain-M0 ablation,
serial causal mode/padding rollout and held bias/order/structural scoring.

## Validation

- **140 related CPU tests passed**, including30 collector tests. The new tests
  cover token pairing/tampering, partial chunks, async step IDs and terminal
  guards, barriers, output isolation, overflow/cancellation, deferred CUDA events,
  missing/duplicate work, native configuration/KV checks, HTTP authentication,
  stage order/partial-stage retention, dispatcher candidates and a complete
  18-trial calibration freeze with held-label mutation isolation.
- Editable-package installation exposes both required vLLM entry points. The
  disabled plugin does not initialize CUDA in ordinary CPU test processes.
- All current generated artifacts and design-file hashes pass the existing
  provenance checks. Python compilation and git whitespace checks pass.
- The earlier full-suite attempt produced **759 passed,11 failed,27 errors**.
  Existing resource/diagnostic tests reject the nested PID/procfs namespace in
  this Work environment. The same unchanged `process_resources` guard reports
  `procfs_pid_namespace_unverified`. Native CPU acceptance therefore remains false;
  the runner stops at that preflight instead of filtering tests or relaxing it.
  Full JUnit cases/outcomes/tracebacks are preserved with trailing whitespace
  normalized and the original file hash recorded; they retain their earlier-working-tree scope.
- No installed vLLM/GPU runtime has been loaded here. Actual hook integration,
  stream/graph scope, observer overhead and GPU acceptance remain unverified.

[Validation/source hashes](validation.json) · [Related JUnit](related-tests-junit.xml) ·
[Initial full-suite JUnit](initial-full-suite-junit.xml).

## Acceptance scope

Source guards target exact reviewed upstream0.30.0 files. Actual-host runtime or
scheduler differences stop/invalidate the applicable stage. Normal V1/V2 selection,
graphs and async settings are preserved. Overlapping async wall spans cannot earn
serial cumulative model selection. GPU spans cover current-stream execute_model
preparation/forward; sampling and additional streams need separate qualification.
Block dependence stays unresolved until a defensible calibration-only assessment.
Neither collected trial counts nor nominal components accept a model or MPC.

Statistical revision9 is unchanged: two structural groups, existing eligibility,
fixed5ms minimum mean improvement and all-cell regression checks. Existing noise
screens remain historical and reproducible. Azure output policy remains cap128
with natural EOS. PR32 stays draft until native and actual-host qualification.
