# PR32 collection implementation

The collector, trial runner, exact-token materializer, source guard, mode-aware
calibration fitting and held scorer are implemented. The adapter is experimental:
it has CPU/mock validation, but has not been loaded against an installed GPU vLLM
runtime in this environment. Keep PR32 draft and the executable protocol unfrozen
until native acceptance and a real-host qualification succeed.

## Supported boundary

The entry points target the reviewed **vLLM0.30.0 V1 and V2 GPUModelRunner**, one RTX4090,
TP/PP/DP1, OpenAI-server EngineCore, batch budget2048, context8192, FCFS, threshold0,
no prefix cache/speculation/KV transfer. The entire source files for every hooked
interface are SHA256-checked against the committed upstream source map. A different
wheel, unreviewed dispatcher, microbatching configuration or runtime defaults fail
before scored collection; there is no silent eager/async override.

[Source contracts](prefill-vllm-v030-source-map.json) record the exact inspected
upstream files and hook/counter update sites. Hash equality establishes the source
boundary; only actual-host collection establishes that the hooks work together.

## Short rental-host commands

After installing the pinned runtime and CPU dependencies in .venv, use:

```bash
.venv/bin/python scripts/prefill_host.py prepare
.venv/bin/python scripts/prefill_host.py collect
```

Prepare pins the model/tokenizer to one full HF commit before downloading, resumes
that pin on repeated calls, verifies every retained Azure prompt length and creates
the exact token manifest under ~/pr32-inputs. It runs complete native CPU acceptance
for the updated checkout/final environment, or reuses only already-valid evidence
matching that commit and current source fingerprints. Failed evidence is retained
in a separate directory; it cannot produce READY. Outputs stay outside the checkout.
Prepare does not launch collection. Collect reads ready.json and delegates to the
existing frozen runner with a fresh ~/pr32-captures/<timestamp> output directory.
Use --output-dir to select a separate input/pin directory.

## Native host commands

Use a clean checkout of the PR branch on Linux x86_64 / Python3.12. Install the
pinned runtime from requirements-v1.txt and CPU dependencies, then the opt-in plugin:

```bash
python -m pip install -r requirements-cpu-tests.txt
python -m pip install --no-deps -e .
```

Resolve the model and tokenizer to the same full40-character Hugging Face commit.
Provide a local tokenizer snapshot at that revision. Preparation reads only local
files, hashes their content and creates raw engine token IDs; no chat template or
hidden special tokens are added to these mechanistic trials.

```bash
python scripts/run_prefill_service.py prepare \
  --tokenizer /path/to/pinned-tokenizer-snapshot \
  --revision FULL_40_CHARACTER_HF_COMMIT \
  --output /tmp/prefill-tokens.json

python scripts/run_prefill_service.py cpu-check \
  --output /tmp/prefill-cpu-acceptance

python scripts/run_prefill_service.py collect \
  --manifest /tmp/prefill-tokens.json \
  --tokenizer /path/to/pinned-tokenizer-snapshot \
  --trace results/trace-experiment-96-r1/trace.json \
  --cpu-acceptance /tmp/prefill-cpu-acceptance/acceptance.json \
  --output /path/to/new-capture
```

`collect` starts one localhost OpenAI server and one fixed-cost-budget32 gateway,
waits for model/tokenizer load, validates source/config/GPU/KV capacity, and runs
the frozen212 trials plus four Azure attempts. Startup/model load precede the
1800s collection clock. CUDA graphs and async execution keep the engine defaults.
The plugin is registered under both `vllm.general_plugins` and
`vllm.endpoint_plugins`; `PREFILL_LAB_ENABLED=1` is set only in that server process.
The collection route uses a random environment-held bearer credential and the
server listens only on127.0.0.1. Normal OpenAI traffic remains isolated to that host.

Native CPU acceptance requires the whole test suite, all five existing trace
configurations, no skipped/failed tests, matching source hashes and commit, and a
clean tracked checkout. The existing manual Linux CPU workflow also runs the new
collector tests. A mismatched PID/procfs namespace is a failure, not something to
bypass with mocked resources or filtered tests. The current Work environment hits
that limitation; the committed local test record does not claim native acceptance.

## Collection behavior

- Exact token pools are deterministic, exclude tokenizer special IDs and preserve
  paired off/light content and common per-block content prefixes. Input lengths,
  output caps, roles, offsets, cases and order are checked before collection.
- Core utility calls reuse the server's existing transport. Foreground requests
  enter the scheduler in one engine-thread batch after all16 backgrounds have
  produced output. The real scheduler chooses chunks. A timer posts the sentinel
  onto the EngineCore's existing input queue, never mutating scheduler state from
  another thread. Its queue-posting lag must be<=5ms. Both queue arrival and later
  scheduler-consumption timestamps are retained; queue residence is not hidden as
  timer jitter. Sentinel TTFT is queue-arrival to matched first engine output.
- The collector pairs pre-step positions/counts with output commits using a hash
  of request IDs, scheduled counts and positions. Worker IDs use the same rule,
  including async batches. Off mode keeps minimal request/release logging; it
  does not build per-step snapshots or allocate CUDA events.
- Light mode records actual work, context, queue/running snapshots through parity
  diagnostics, mode/padding, graph capture/replay calls and stream-scoped CUDA
  forward events. CPU iteration wall and GPU forward spans are separate labels.
  GPU spans cover the current-stream work in `execute_model` (input preparation
  and forward), with sampling and unrelated streams outside that label. Delayed
  `query()` resolves them after drain; no per-step global/device synchronization.
- Buffers are bounded. Overflow, missing/duplicate events, progress gaps,
  cancellations, preemption/speculation, external traffic in a controlled trial,
  short forced outputs and scheduler/worker mismatches cannot pass integrity.
  Partial request/client/capture records survive timeouts where the engine remains
  reachable. A dead engine cannot supply records that never reached persistent
  storage; its logs and incomplete stage remain evidence, never a complete trial.
- Before/after each main block, drift cells and clock/thermal snapshots provide
  context. Observer acceptance uses the frozen20/10 paired mean-log-ratio t upper
  bound. HTTP bridge trials go through normal streaming `/v1/completions` and
  retain API send/first-byte times alongside engine state. These are diagnostic,
  not replacements for the engine barrier experiment.
- Azure transfer offers [40,120) and [160,240) in frozen ABBA order. It scores the
  final60 seconds of each80-second replay, retains natural EOS with cap128 and
  uses the normal fixed32 admission-cost gateway. Raw exporter buckets and gateway
  snapshots straddle the scored boundaries; every offer/rejection/failure is kept.
  Deep CPU/CUDA traces profile at most five execute calls in each of the two final
  deep trials and are written under `deep-traces/`.
- Each stage keeps its planned/attempted/completed IDs, integrity, deadline and
  failure reason. One attempt only, no retries or changing workload shapes. Later
  transfer/deep failures retain a separately complete main stage but cannot pass
  the entire suite. The source/config/dispatcher identity is checked again at end.

## Analysis and acceptance boundaries

`model-freeze.json` is built after calibration blocks1–3 from the18 `train` trials
only. Held and repeat-check labels cannot enter fitting, LOCO, scales or feature
encoding. Every fold derives its own mode vocabulary and constant-padding rule
from its training cells. M0/M1/M0Z/M2 share the same accounting basis in that fold.
NNLS weights give each calibration cell equal total squared row weight; full
column rank and scaled condition<=1e4 are required. Unknown modes or unsupported
padding cannot earn extrapolated held success.

The readiness screen has twelve preregistered strata: six calibration cells by
prefill-bearing/decode-only phase. Sum each phase's labels per trial only after
matching the complete ordered work/position/mode/padding signatures across all
three repeats. Missing work is inconclusive; phase-total CV>2% is flagged and >5%
stops structural claims. Per-step maximum CV is descriptive and cannot stop the
run. Aggregation reduces sensitivity to independent step noise; this is phase-total
repeatability, not the per-step CV used in the historical power screen. Actual
D16 mode/padding strata also report whether context varies independently of the
other work features. A coefficient alone never earns a unique KV mechanism claim.

The causal mode-aware rollout accepts only offered arrivals, forced lengths,
release background state and its own predicted progress. The installed dispatcher
lookup (V1) or ordered candidate descriptors (V2) is frozen before trials and checked against actual observed mode/padding.
Unexpected modes, microbatches or scheduler differences invalidate that rollout
scope. It is a restricted serial FIFO predictor: async overlapping wall spans
cannot be added into serial cumulative targets. Such captures retain their raw
measurements and get an inconclusive model freeze instead of invented serial costs.

At block3_end the runner also freezes eligibility using nominal release state and
OFF observer jitter (1.4826*MAD per position, assigned by the frozen position-reference mapping), plus
all nominal predictions/gaps and source identities. Measured held release states
only condition held predictions, never eligibility. The jitter proxy is not assumed
equal to held per-trial SD. Scoring verifies digest(model-freeze) against suite.json
and verifies training/OFF capture hashes before opening held data.

Held scoring checks this source-bound model freeze, all five blocks, calibrated M0 and
retained M1 comparisons, the existing two structural groups, fixed5ms improvement
floor, all-cell regressions, bias and offset ordering. Eligibility remains the
revision9 `max(3*jitter,25ms)` rule. The latest proposed eligibility change is not
silently folded into this collector implementation.

The dependence status stays **unresolved** until a defensible calibration-only
qualification is reviewed. Five held blocks cannot establish independence.
Likewise, GPU forward-span scope remains unqualified until the real deep traces
establish stream/graph coverage. Completed collection and a nominal statistical
component do not automatically accept the physical model or evaluate MPC.

## Files to inspect after a run

`launch.json`, `collection-sources.json`, `runtime.json`, `runtime-final.json`,
`host.json`, `suite.json`, per-trial captures/audits, HTTP client records, four
transfer files, gateway telemetry, `model-freeze.json`, `held-report.json`, and the
two deep traces. `suite.json.complete` means all frozen collection stages completed
with their scoped integrity checks. `model_accepted` and `gpu_validated` deliberately
remain false pending actual-host qualification; they are not inferred from counts.

Revision11: the raw-token HTTP bridge goes directly to OpenAI completions and
explicitly does not qualify gateway transport. Azure chat transfer uses the
cost-aware gateway. Controlled requests increment upstream cumulative metrics;
use per-trial snapshots and IDs rather than interpreting them as offered demand.
The utility transport is a pinned private API. EngineCore uses a0.5ms GIL switch
interval; the unchanged5ms posting deadline must still hold on the native host.
Invalid requests receive request-local aborts; recorder finish disables observations
before deferred event polling, and stale timer epochs cannot enter HTTP captures.

Revision12 resilience: a block3 analysis/import/archive-read exception is saved in
model-freeze.json as inconclusive with error_type/reason, and collection continues.
suite.complete describes collection only; inspect calibration_status separately.
Invalid captures stay invalid after drain. Output-observer failures preserve
request/output isolation and engine liveness; native engine exceptions still fail
the run. The source guard now checks35 upstream files including attn_utils.py.

The collection launcher prepends the current interpreter's bin directory to the
API-server and gateway child PATH. FlashInfer's EngineCore descendants can find
venv-local ninja even when invoking .venv/bin/python without shell activation.

The meta utility reply converts model and tokenizer revisions to builtin strings
before crossing EngineCore's msgspec transport. For Hugging Face ResolvedRevision,
it records the resolved commit rather than a requested alias such as main. Exact
manifest revision checks remain in force; no pickle fallback is enabled. Updating
this code requires prepare again for the new commit/source-bound CPU acceptance;
the pinned model snapshot and prepared exact tokens are reused.
