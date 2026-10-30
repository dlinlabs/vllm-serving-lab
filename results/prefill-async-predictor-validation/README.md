# Causal async CPU prototype: implementation and development validation

An executable causal simulator now models the reviewed two-slot V2 pipeline.
The old serial model correctly rejects overlapping wall labels; it is retained
unchanged. The new separate prototype handles CPU/GPU overlap, copy/update
branching, native submit-first/FIFO-reap order, empty futures, own request
progress/placeholders and a simulated foreground barrier. It introduces no new
GPU collection, fitting, gateway cap or MPC behavior.

**Status: unfitted prototype.** Every physical-model, CUDA-scope, complete-
experiment, adaptive-cap and MPC acceptance flag remains false. The costs are
explicit illustrative coefficients, not values fitted to development/held traces.

## Executable entry points

```sh
python scripts/predict_prefill_async.py --synthetic-demo --output /tmp/pr32-async-demo
```

Runs all 22 declared cases without a GPU. Select one with `--case`, retain its
resource graph with `--timeline`, or provide a saved `--runtime` dispatcher.
Output directories must be fresh. Read
[the prototype contract](../../experiments/prefill-async-predictor-v1.md) for
input formats, assumptions and limits. `--costs` is an explicit parameter input,
not a fitting/acceptance mechanism.

For the original stage diagnostic directory, reproduce the committed comparison:

```sh
python scripts/validate_prefill_async.py --evidence /path/to/1792991194546315502 --output /tmp/pr32-async-development.json
```

The validator requires exact bytes/hashes for its five raw inputs against the
previously committed 29-file native-stage input index. It does not mutate raw
captures. No captured completion timestamps or measured stage durations enter
rollout, coefficient selection or structural comparison.

## Native development scheduling comparison

The original archive was captured at ab24416334174f31c0c8b686ff4d0e496f86a969.
It has 3,956,779 bytes and SHA256
`a5a446a27f0a991f8ee715ee4f0010ab32a5c9cdc3e3188ec97e54525c10571e`.
Its prior source, work, dispatcher and stage/profile integrity audits are
documented in [the measured stage report](../prefill-stages-ab24416-analysis/report.md).
That original native measurement/acceptance status is not changed here.

The prototype uses only case definitions, frozen runtime/dispatcher metadata
and synthetic costs. After independently finishing its prediction, comparison
checks the native ordered work and positions, selected request pre-state
(computed/committed tokens, placeholders, prompt/output limits), aggregate work
features, actual mode/padding, step-bound output emissions and final output counts.

| Native trial | Case | Executed steps | Work/state/features/mode/padding/output progression |
| --- | --- | ---: | --- |
| trial-014, light | single-L7168-D16 | 264 | Exact match |
| trial-007, light | composition-mixed-D16 | 264 | Exact match |
| trial-211, deep | single-L7168-D16 | 264 | Exact match |
| trial-212, deep | composition-mixed-D16 | 264 | Exact match |

All 1056 steps match. The two case definitions have zero-offset foregrounds and
no sentinel; their work order does not establish response-time accuracy for new
arrival offsets. Light and deep captures of these two cells are development
evidence, not four independent calibration/validation cases. The held-composition
development trace is not used to tune any coefficient or acceptance criterion.
Complete step identities and pipeline output evidence are required; missing or
corrupted state/work/mode/output rows fail the comparison. Clock-label changes do
not alter the result. The comparator cannot independently qualify capture source,
CUDA coverage or observer overhead; those remain separate audits.

## CPU checks and illustrative smoke run

**270 related CPU tests passed in 14.15s, with zero failures/errors/skips**, on
Python3.12.14. The final JUnit and machine-readable scope are committed alongside
this report. Forty-six new cases cover:

- Hand-calculated GPU execution before CPU return, last-enqueue dependencies,
  copy/post-update overlap, copy-stream FIFO and a first token before GPU update
  completion. Dependency waits are charged once, without a nested sampling parent.
- Submit-first behavior even when an old result is ready, FIFO reaping, two-slot
  occupancy, partial-prefill positions, delayed placeholders, exact forced output
  limits, empty futures and the native empty-iteration idle yield.
- Empty-engine background barriers, lexicographic injection, delayed input queue
  consumption, ideal posting versus engine TTFT, sequence/KV/context constraints,
  duplicate IDs, nonfinite/negative coefficients, unsupported metadata and bounds.
- All 22 declared cells draining deterministically, frozen cost laws, provenance,
  corruption-sensitive independent structure comparison, clock-label independence,
  exact archived-input identities and CLI failure-result retention/nonzero exit.

The CLI also ran all 22 synthetic eager cases successfully. The native-metadata
smoke run in development.json uses its frozen dispatcher and capacity, while
retaining illustrative costs. Every request reaches its forced output count and
all steps respect B2048. Three late-arrival D16 cells report that backgrounds
finish before foregrounds: sentinel-four-at400-D16, sentinel-four-at800-D16 and
sentinel-one-at200-D16. No artificial decoder extension or timing adjustment is
made. These smoke results do not satisfy the controlled native experiment's
background-outliving condition or qualify physical latency.

An earlier targeted unit attempt passed25/failed1 because its proposed idle-
yield fixture finished the last request and therefore did not leave scheduler
work. The fixture was corrected to the native condition: a pending request
blocked by the sequence limit after an empty iteration. The native1ms rule was
not relaxed. Subsequent targeted46 and final related270 runs pass. The earlier
related266 run predates the final added comparison/CLI tests; it is not the final
count. Native procfs/resource/full-suite acceptance was not attempted or bypassed.

Compilation, registered fingerprints and staged whitespace are checked. The
three full upstream files for EngineCore/Scheduler/AsyncScheduler were fetched
at v0.30.0 and their SHA256 values matched the existing captured source map before
implementation. The full original source text is not duplicated in this report.

## Remaining qualification

Five-part launch laws approximate a multi-enqueue CPU/GPU program. Existing
aggregate event spans cannot uniquely identify those parameters, and the measured
copy event includes dependency waits. New calibration must establish an
identifiable parameterization, supported mode/work coverage, overhead and
repeat/dependence checks before freezing coefficients. This prototype shares
coefficients across modes and models no HTTP/gateway path, final idle-worker
cleanup or CPU contention. See the contract for all exclusions.

Next work is a separately preregistered calibration/held-validation protocol.
After an accepted end-to-end latency model, compare fixed versus state-aware cap
on the same offered workload, retaining failures/rejections; MPC is a later
separate result. Another GPU run is not required to exercise this CPU milestone.
