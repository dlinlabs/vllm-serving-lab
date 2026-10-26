# Async service calibration contract v1

Status: executable CPU calibration/evaluation contract, NOT native collection ready.
No old development or held-composition trace has been used to fit costs. The
serial overlap guard, collector, acceptance thresholds and gateway remain intact.

## Executable workflow

```
python scripts/calibrate_prefill_async.py demo --output /tmp/pr32-cal-demo
python scripts/calibrate_prefill_async.py plan --output /tmp/pr32-cal-plan
```

The demo generates synthetic design points with known coefficients, freezes a
model from six declared training trials, then evaluates three separate held
trials. Output includes plan.json, synthetic-input.json, frozen.json, costs.json
and result.json. Output directories must be fresh. Synthetic features are an
algebraic excitation design, including fractional counts; they are not GPU
scheduler observations or a physical workload. A separate test also uses the
recovered costs in an actual causal rollout with a long prompt and sentinel.

Separate commands support a training-only input file and later held-only input:

```
python scripts/calibrate_prefill_async.py freeze --plan PLAN.json --input TRAIN.json --output /tmp/pr32-cal-freeze
python scripts/calibrate_prefill_async.py evaluate --frozen /tmp/pr32-cal-freeze/frozen.json --input HELD.json --output /tmp/pr32-cal-held
```

Inputs must carry the frozen plan digest and runtime identity. The plan assigns
trial IDs to splits; row-provided role labels never determine training membership.
Held feature/label values are not read during freezing, and training labels are
not read during held evaluation. Frozen outputs bind training content, runtime,
plan, coefficients, support and diagnostics by digest. Hashes detect accidental
mutation, not malicious forgery; they are not signatures or source qualification.

## Parameterization and identifiability

The model has 21 distinct service channels: five parts for each of forward,
logits_sample and post_update, plus schedule, empty submission, copy enqueue,
copy service, output materialization and commit. Forward GPU tail uses the fixed
basis bias/prefill/decode_context/attention_pairs. Other channels use
bias/num_requests. These are candidates to test, not claims that these laws
capture all hardware costs or runtime modes.

Each input row has a unique trial/step ID, a mode, feature dictionary and a
complete channel-label dictionary. Every label must explicitly be
`{"scope":"exclusive_service_s","seconds":...}`. Wait-inclusive CUDA intervals,
nested sample_tokens totals, negative/nonfinite durations and missing channels
are rejected. No heuristic subtracts forward from a copy event.

Each law is estimated with nonnegative least squares. Trial-balanced weights
prevent a long trial from dominating. Feature RMS scaling uses the same balanced
weights and training rows only. Each scaled weighted design must have full column
rank and condition number <=1000; this is a numerical rejection rule, not a noise
or power guarantee. Zero/rank-deficient/nearly-collinear designs stop fitting.

The freeze records scaling, rank, condition, modes and training feature ranges.
Held reports preserve per-trial and balanced MAE plus unseen modes/out-of-range
rows. Marginal range checks do not prove joint support. There is deliberately no
physical-model pass threshold in this CPU contract: numerical coefficient
recovery cannot grant physical or end-to-end TTFT acceptance. All exported costs
remain marked synthetic, even after a perfect synthetic held score.

## Self-review: unresolved native measurement blocker

The original five launch parameters are not independently observable from the
existing whole-stage CPU/CUDA elapsed spans. In particular:

- CPU-to-first/first-to-last/after-last enqueue need qualified boundaries.
- GPU prefix/tail service must exclude CPU starvation and dependency waiting;
  aggregate CUDA elapsed labels do not establish these pieces.
- Copy CUDA elapsed starts before a stream wait; it is not copy service.
- output_ready may include blocking; materialization-only cost is different.
- CPU and GPU timestamps/stream identities require launch-correlation evidence;
  overlapping activity unions and gaps must have a defined measurement scope.

The program therefore rejects ALL native-origin input today. A caller changing
only a scope string cannot turn the old stage archive into accepted native data.
No native qualifier/extractor has been implemented in this change. No GPU run
command is advertised as ready, and re-running `prefill_host.py stages` does not
supply this contract's missing labels. The CPU implementation is complete within
this stated boundary; a physically identifiable native parameterization and its
qualified extractor remain outstanding, not silently replaced by fixed guesses.

## Next native protocol requirements

Before another paid run, either qualify direct component measurements or replace
the five-part approximation with a demonstrably identifiable lower-dimensional
model. Test the proposed design's rank and sensitivity before collecting held
outcomes. A candidate workload matrix is calibration single1024/single7168 with
D0 and D16, plus uniform/reverse compositions; independent validation includes
single2048, mixed composition and sentinel offsets. Use fresh runs, repeated
whole cells, fixed source/model/runtime identity and randomized recorded blocks.
The CPU demo's train-N/held-N identifiers are NOT a GPU collection manifest.

The concrete native manifest, repetitions, mode coverage, observer OFF/light
pairs, allowed overhead, dependence treatment and stage/end-to-end error limits
must be preregistered after the measurement design is qualified. Freeze the
training model before independent held labels are inspected. Unsupported modes,
backgrounds finishing too early, work/arrival mismatch and insufficient numerical
support remain failures, not reasons to tune thresholds. Native acceptance must
also precede any rental-host collection. Cap/MPC experiments remain downstream.
