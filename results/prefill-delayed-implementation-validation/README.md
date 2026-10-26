# Delayed-arrival pilot implementation and self-review

Implemented a separate136-trial source-bound v2 protocol:16 warmups,48 original
training trials, freeze/forecasts,4 anchor-before,64 delayed and4 anchor-after.
The8 delayed cases add a512-token sentinel at50/200/400/800ms to the existing
mixed workload with D0/D16. All native GPU wrappers,38 source pins, the v1 fitter,
serial overlap guard, fixed B2048 and gateway/control behavior remain unchanged.

Native v2 collection is NOT yet demonstrated. CPU validation reaches the first
GPU-dependent step, not delayed/model/cap/MPC acceptance.

Final related CPU validation: **365 passed in74.16s**, zero failures/errors/skips,
including22 new regression cases. JUnit is retained. Tests cover target/post/enqueue
clock separation, legitimate long queue residence, posting/target/chronology/output/
background/pending-output violations, missing/duplicate/wrong held data, changed
forecasts/freezes, failed structure/queue/anchor gates, held-label isolation from
fitting, synthetic own-rollout recovery, matched anchor token content, endpoint
mode isolation, freeze-before-anchor transport failure, host delegation and
failed-attempt archiving. Existing v1/analysis/scheduler/dependency tests still pass.

Standalone noiseless synthetic recovery on saved scope/dispatcher metadata has
maximum coefficient error1.942890293094024e-16s and local condition4.107196167474088.
The frozen/report artifacts here are explicitly synthetic. No v1 held timing or
archived coefficient file is imported by native v2 fitting. Synthetic tests do
not qualify observation quality or native latency.

The first focused run had28 pass/1 fail: its queue-failure fixture changed one of
four samples while expecting a median gate to fail. The corrected fixture changes
all four relevant OFF samples; the preregistered median gate/threshold is unchanged.
A subsequent focused30-case run passes. This is not a per-trial tail guarantee;
raw clock diagnostics are retained and native endpoint repeat gates are separate.

Self-review checks:

- The old six-cell fitter is reused unchanged and only receives planned training
  membership. Delayed labels never enter optimizer/scales/parameter choice. Fresh
  host training and a freeze occur before all scored anchors/delayed trials.
- Primary target-to-first/final clocks match ideal offered-arrival predictions.
  Actual posted TTFT, posting lag and queue residence remain distinct.5ms posting
  integrity is not relaxed because engine residence can exceed100ms.
- Before and after anchor content is identical; each phase has a separate error
  gate so their averages cannot hide drift. Queue median accuracy has its own
  preregistered max(10ms,20%) gate, and signed residuals remain visible.
- Only the dedicated arrivals operation/manifest/endpoint mode can run v2.
  Conflicting endpoint flags fail and every launcher mode resets both flags.
  Existing calibrate/collect/diagnose/stages retain their own plans.
- Invalid native captures stop with partial evidence; failed held predictions
  produce a failing candidate with all completed evidence. No automatic retries,
  forced background extension, threshold adjustment or model refit on held data.
- Exact-source native acceptance is still mandatory, and original JUnit/log and
  all data are archived even on failed collection. Offline audit reconstructs raw
  clock labels and checks identities/forecasts/report, rather than trusting a
  stored pass bit. Hashes are consistency checks, not signatures.
- The v1 source-bound uploaded audit still passes in its isolated original Git
  snapshot. Its raw data and frozen candidate are unchanged.

Compilation and staged whitespace checks pass. No local native full CPU acceptance
or GPU run is claimed. The next step is documented in
experiments/prefill-effective-arrivals-v2.md: prepare a clean current checkout on a
Linux/Python3.12 RTX4090 host, then run prefill_host.py arrivals once. The four-
parameter reduction and known D16 bias may fail the new tests; preserve that result
instead of weakening the gates. HTTP, dynamic cap and MPC remain downstream.
