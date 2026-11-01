# Opt-in native stage diagnostic implementation

This change implements development measurements for the next async model. It
does not implement that predictor, fit stage costs, accept a physical model or
evaluate cap/MPC. Actual GPU execution of these new hooks remains pending.

Use `prefill_host.py prepare`, then `prefill_host.py stages` on the updated clean
native checkout. The exact-revision complete CPU gate is required again; the
older876-test capture cannot authorize a changed source tree. The model pin and
download cache are reusable. This mode runs two frozen light warmups and two
deep cases under a180s active bound, without gateway/transfer/held scoring, into
`~/pr32-stage-diagnostics/<timestamp>`. Existing collect/diagnose do not enable
the added stage hooks. Launch records the explicit opt-in flag and separate scope.

## Implemented

- Stable step bindings from V2 forward to sample_tokens and native AsyncOutput.
  Reaping an older future uses its own binding even after another forward/sample.
- CPU spans for forward, sample_tokens, logits/sample, copy enqueue, post-update,
  output-ready and scheduler commit. Native output counts and first-output times
  are independently reconciled against the existing request capture.
- Stream-local CUDA event pairs for forward and the four GPU enqueue stages.
  Resolution uses query after drain. No new per-step synchronize/wait is added;
  native AsyncOutput stream waits/copy_event synchronization are preserved.
- Copy enqueue remains before native GPU post-update, allowing overlap.
  Primary sampled-token/num-sampled bytes and optional extra-payload presence are
  recorded. The copy span includes native stream waits and is not pure transfer
  cost. Nested spans are never declared additive.
- Deep profiling includes the fifth sample_tokens/copy/post-update tail, exact
  CPU stage markers and separate stop/export timestamps. Reap/commit can occur
  later; deep wall/TTFT remains diagnostic rather than a performance label.
- Strict V2/same-process UniProcExecutor/two-slot preflight and source-bound
  development protocol. The three additionally inspected files are AsyncOutput's
  gpu/async_utils.py, gpu/input_batch.py and gpu/sample/output.py. There are38
  installed source pins. Reviewed V2/core/executor content hashes were also
  reverified against their existing pins; no upstream source is modified.
- Source/protocol-bound offline audit for both old foreground and new stage
  captures, with no relabeling of old data or promotion of acceptance flags.

## Validation

[validation.json](validation.json) and [junit.xml](junit.xml) record214 passing
related CPU tests, zero failures/errors/skips, including39 new pipeline cases and
one new analyzer-tail case. Fixtures exercise the actual recorder/hook wrappers
with native-shaped CPU objects and deliberately fake CUDA/profiler interfaces.
They establish accounting/control contracts, not CUDA API execution or overhead.

Coverage includes two outstanding outputs, core commit instrumentation, unresolved
copy events after forward completion, fifth sampling before profiler stop, native
sampling errors, cancellation, absent/duplicated/malformed stages, NaN timing,
wrong stream/process/thread/forward identities, mismatched first-output/token
totals, overflow, disabled hooks, exact native gate/delegation, independent
diagnostic orchestration and mutually exclusive CLI modes. The uploaded6edb3e3
diagnostic also passes a fresh offline audit with the updated analyzer. Compilation,
registered source fingerprints and final staged whitespace checks pass.

The first implementation pass had212 passing related cases; two additional
forward-identity/stop-boundary regressions bring the final count to214. Neither
selected local run is the complete native CPU acceptance required on the rental
host. Native source/GPU/profiler/stream qualification and any overhead screen must
be established from the new actual-host capture. The existing serial overlap
guard and all CUDA/model/complete-experiment acceptance boundaries remain intact.

Development plan: [prefill-stage-diagnostic-v1.json](../../experiments/prefill-stage-diagnostic-v1.json).
Predictor proposal: [prefill-async-pipeline-v1.md](../../experiments/prefill-async-pipeline-v1.md).
Native usage: [runbook](../../experiments/prefill-service-v1.runbook.md).
