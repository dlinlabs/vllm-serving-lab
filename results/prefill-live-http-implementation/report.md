# Busy-state / HTTP bridge: implementation review

This change prepares the next native experiment. It has **no new GPU result**.
The previous controlled event-clock confirmation remains accepted within its
pilot scope; HTTP, production-model, adaptive-cap and MPC acceptance remain false.

## Resulting behavior

The predictor can now start from an observed EngineCore pre-ADD boundary with
running/waiting requests and an uncommitted batch. It generates subsequent work
from that state and the frozen six-term effective lane. It preserves slot order,
output placeholders, forced termination and native conditional idle yields.
Invalid state, overdue unready work or insufficient conservative KV headroom
produce an unavailable forecast. Such forecasts count as failures.

The opt-in fixed-32 gateway forwards exact token IDs and joins client, gateway
and engine milestones on an identified Linux monotonic clock. Prediction occurs
after ingress has already elapsed: only the future engine interval and a
TRAIN-only frozen egress allowance are forecast. This does not establish a
pre-admission decision or natural-EOS response-length model.

The `live-http` host operation first runs 136 unchanged clock trials with fresh
same-host fitting/freezing, then 98 HTTP trials. Ten LIGHT warmups must establish
snapshot availability and work-prefix agreement before training/held collection.
The held protocol, individual error budgets, observer controls and failure
denominators are frozen before native execution. The independent post-shutdown
audit regenerates forecasts and checks raw journals and transport freezing.
Successful and failed captures are both archived.

## Self-review findings and corrections

- A pending UniProc `AsyncOutputFuture.done()` is not GPU readiness. Nonempty
  work now queries the existing source-pinned D2H copy event without synchronizing
  or materializing outputs. Empty execute futures retain their separate guard.
- The copy-ready bit does not prove complete main-stream idleness. The declared
  commit-anchored residual is an effective approximation, not physical service
  identification. Its unknown/overdue states are retained and fail qualification.
- The existing chat SSE observer did not recognize raw-completion `text` chunks.
  Raw mode now has an explicit parser option; default chat parsing is unchanged.
- Clock domains, request IDs, exact native prompt hashes, both token-manifest
  identities, stale snapshots, missing milestones and failed/truncated streams
  are checked before scoring. No held transport label participates in fitting.
- Global/trial deadlines own subprocess termination and preserve partial records.
  Both localhost ports are checked before startup. Automatic independent audit
  runs after graceful shutdown, with a separate bounded CPU watchdog.
- Normal launch modes reset the opt-in live flag. The raw route is absent by
  default and requires protected admission, shared transport and benchmark IDs.
  No adaptive-cap behavior is enabled by this change.

## CPU evidence and limits

The final related regression run passed **551 cases** with one warning in
283.68 seconds. The final focused live/host/design run passed **91 cases** in
5.39 seconds. Both have zero failures, errors or skips. These counts overlap;
they are not summed. The new live module contains 52 parameterized cases.

See `validation.json` and the included JUnit reports for the exact test counts,
commands and source hashes. Tests cover synthetic busy continuations against the
independent original from-empty simulator, poisoned readiness/future fields,
causal clocks, fragmented SSE, raw/default gateway behavior, TRAIN-only fitting,
unavailable forecasts, the 98-trial orchestration and cancellation/audit cleanup.
These are local CPU regression results, **not** full native CPU acceptance.
The host's complete native PID/procfs and exact-source acceptance guards remain
mandatory; they are not weakened or substituted in this environment.

`cpu-smoke/forecast.json` uses the explicitly synthetic state/request alongside
the previously archived six coefficients and runtime metadata. Its
40.881383 ms forecast illustrates executable CPU continuation only. It is not a
measurement, a refit, or evidence of native busy-state accuracy.

`legacy-confirmation-recheck.json` is a fresh analysis of the original archived
33876f3 capture using its captured sources. That audit still passes without
refitting. It does not execute a GPU or validate the new live bridge. The six
complete upstream files reviewed for readiness/scheduling are listed in
`reviewed-source-pins.json`; all already belong to the existing 38-file map.

## Next required evidence

Run the documented `--next live-http` preparation and `prefill_host.py live-http`
on a fresh native RTX4090 host. Preparation runs complete native CPU acceptance;
collection uses newly fitted same-host coefficients, not the archived smoke
vector. Inspect the new warmup/held audit before any cap experiment. A valid
rejection must retain its raw evidence and cannot be repaired by excluding trials
or changing thresholds after seeing the result.

Protocol and short host commands:
[`prefill-live-http-v1.md`](../../experiments/prefill-live-http-v1.md).
The full prior PR description is preserved as historical material in
[`pr32-history-through-68c3be5/pr-description.md`](../pr32-history-through-68c3be5/pr-description.md).
