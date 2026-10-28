# Conditional state timing envelope: CPU development, not native qualification

The efca3e4 rejection shows that one fixed-case first-commit point cannot meet
every original TRAIN budget. This module replaces a point in a separate CPU
interface with a conditional timing envelope. It does not change or bypass
that failed point gate, make entry-http proceed, or accept a new native model.

## Inputs available at the decision boundary

Use an actual EngineCore pre-ADD `live-engine-state-v1` snapshot: clock domain,
observed_at, last observed commit, running/waiting order, current request
progress/placeholders, and pending batch work, scheduled_at and nonblocking
copy-ready bits. One unique proposed request supplies known input and forced
output bounds. No later requests may arrive in this continuation's scope.

A controlled `release_state` contains scheduler counts/order only. Its missing
pending state cannot be reconstructed from later commits or worker durations.
The new API returns missing_pending_time_state for that legacy shape. Extra
future timing fields are rejected by the strict snapshot validator. A supplied
evaluation clock must share the domain and satisfy the support's age policy.
Clock format/finite-value checks precede the domain check, but timestamps are
compared across the snapshot and evaluation only within the same domain.
Another boot returns `clock_domain_mismatch` even when its uptime is smaller;
the CLI still writes the unavailable artifact. Malformed clocks and backward
same-domain timestamps remain invalid inputs, not successful fallback records.

The scope remains V2/UniProc, TP=PP=DP=1, two slots, FCFS, B2048, seq32,
context8192, forced lengths, no prefix/speculation/preemption, and conservative
KV headroom. Natural future EOS and arrivals are not controller inputs here.

## Cost support and its limits

`state-time-support-v1` holds six nonnegative GPU coefficient intervals, prefill
and pure-decode CPU-return intervals, runtime/dispatcher/scope identities,
implementation hashes and frozen age/width policy. Every future batch may take
a different duration inside these intervals; a shared P/D scalar is not assumed.
The required `cost_definition` also binds the confirmation-plan digest, actual
feature scales and event-clock specification, both generated cost-document
hashes, and source dependencies for cost construction, feature extraction,
dispatch, runtime validation and continuation. Both corners use the captured
specification, not separate reads of a mutable current specification.

Older support documents without these bindings are rejected, not silently
upgraded or rebound. Rebuild support from the original provenance and intended
runtime, or supply explicitly labeled assumptions for a new runtime. A prepared
predictor validates these identities once before the observation path and keeps
its detached costs; this is not a per-request file-integrity monitor.

`from_training` requires exactly the declared TRAIN LIGHT set, a fresh native
base-model identity, and matching reconstructed TRAIN event hashes. It takes
all past TRAIN-prefix worker host-span extrema by prefill/decode and leaves the
six GPU coefficients fixed at their original development fit. It does not read
readiness error targets or fit on held data. Changes to runtime, dispatcher,
source or the hashed support document invalidate the replay.

These are empirical past-support ranges, **not confidence intervals, prediction
coverage estimates, or identified physical CPU/GPU service bounds**. Worker
host spans can include waits/enqueue work. Fixed effective GPU coefficients
also remain uncertain. A true trajectory outside this declared support is not
bounded by the calculation. Explicit-assumption boxes are separately labeled.

## Why the conditional envelope bounds the supported event graph

With fixed pre-ADD membership/progress, no later arrivals and forced outputs,
this FIFO/two-slot continuation's work, placeholder updates and output-reap
order depend on logical progress, not on wall time. Each event time is formed
only by a nonnegative service addition or a maximum of predecessor times.
Both operations are coordinatewise monotone.

Run the same generated logical graph with every future service at its lower
bound and then every service at its upper bound. Any sequence of independently
varying per-batch services inside the box is bounded by these extremal event
times. Compare generated work/output inventories at the corners and abstain
if they differ. The claim is limited to the stated time-independent graph;
endpoint corner sampling is not a general proof for arrival-dependent policies.

For pending work, ready=true fixes debt to zero; its CPU submission already
returned before the cut. For unready work, use the existing effective anchor
max(scheduled_at,last_commit_at). Subtract the observed cut from the lower/upper
anchored service deadlines. A negative lower bound is conservatively clipped
to zero, meaning arbitrarily small positive remaining work, not completion.
If the upper deadline has passed while the observed bit is unready, abstain:
unready_pending_outside_service_support. No measured future completion supplies
debt, and no service bound is enlarged to make a case pass.

CPU return overlaps the GPU lane. Charge future submissions only, then reap
the oldest output under the original two-slot order. Existing public point
forecasts retain their original overdue rejection and unchanged semantics.

## Outputs and fallback

The result contains cut-relative TTFT bounds, optionally completion bounds,
width, supplied snapshot age, pending debt bounds, state/support/work hashes,
and a conditional relation to an optional cut-relative engine deadline.
It is not client HTTP TTFT, a pre-admission forecast, or a cost-aware C action.
Age is reported, not subtracted to invent a later current engine state.

Default arithmetic policies are snapshot age <=5ms and interval width <=10ms.
They are frozen in the support and are not replacements for old accuracy
gates. A wide available interval remains in output with fallback_required=true;
missing/old/mixed-clock/unsupported pending state reports unavailable and
requires fallback. No unavailable case may be dropped from a later evaluation.

`usable_under_assumptions` means only that the conditional arithmetic interval
meets these age/width policies. Every output keeps actuation_authorized=false,
online_compute_age_qualified=false and native_coverage_qualified=false, plus all
model/HTTP/adaptive-cap/MPC acceptance flags false. In particular, an interval
fully before a supplied deadline does not authorize admission or prove SLO
safety. Real observer/forecast wall time, source/seal chronology, input scope,
coverage and availability rates still need independent native validation.

## CPU commands

Build past empirical support from the immutable failed capture's controlled
directory and its separately reconstructed base model:

```bash
python scripts/predict_prefill_envelope.py support --controlled /path/1793156148422216020/controlled --base-model /path/reconstructed-base-model.json --output /path/new-support
```

Replay a separately recorded real pre-ADD snapshot and explicit request:

```bash
python scripts/predict_prefill_envelope.py predict --snapshot /path/snapshot.json --request /path/request.json --runtime /path/runtime.json --support /path/new-support/support.json --replay-at RECORDED_EVALUATION_TIME --output /path/new-replay
```

Offline replay labels the clock as recorded_metadata_only. Without replay-at,
the CLI checks the current Linux boot/time namespace and monotonic time; it
still does not qualify end-to-end compute age or actuate a gateway budget.
Output directories must be fresh. Neither command launches a GPU workload.

## Native evidence still missing

The 88-trial archive supports reconstructing past training ranges, not native
envelope accuracy: none of its controlled release records includes the full
pre-ADD pending/time snapshot. Those inputs must remain unavailable rather than
receiving fabricated ready bits. The original 3/24 readiness failures remain.

Before another full rental run, the next native collection contract must seal
the support before fresh shadow probes, capture nonblocking pre-ADD pending
state, and retain paired observer controls, every unavailable/wide/missed
interval, input/work/runtime/source integrity and actual compute age. Freeze
that scope and analysis first. An observed-envelope miss is a model/support
failure; a wide interval is a usefulness failure. Coverage obtained by expanding
bounds after held outcomes is not a validation result.

The separate [short shadow workflow](prefill-envelope-shadow-v1.md) now supplies
that collection and raw-audit interface, with CPU/mock validation only. It
measures completion/enqueue age and retains misses, wide intervals and
unavailable cases without changing the old entry-http workflow. Native coverage
is still unmeasured. C16/C32 action ranking and closed-loop MPC benefit remain
separate gates in prefill-to-mpc-roadmap.md.
