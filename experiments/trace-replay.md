# Real-trace replay: fixed-budget foundation for MPC

Replay production-derived arrival times and prompt token counts on our Qwen
server. Natural gateway stalls remain in primary results. This change does not
implement MPC, reproduce original conversations or source output lengths, or
claim a performance improvement. Root-cause profiling is not a prerequisite.

## Source and preparation

Use the [Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md),
Conversation subset, licensed CC-BY. Attribution: Jovan Stojkovic, Chaojie Zhang,
Íñigo Goiri, Josep Torrellas, Esha Choukse. *DynamoLLM: Designing LLM Inference
Clusters for Performance and Energy Efficiency*, HPCA 2025.
[Paper](https://arxiv.org/abs/2408.00741).
Fields: `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`; no original prompts.

The full Conversation download is about 1.1 GB. This bounded download keeps the
first 1 MiB's complete CSV lines and records the source URL and byte range. It
covers approximately the first 16 minutes. The converter checks that the supplied
source covers the complete requested window.

```bash
mkdir -p results/trace-source
python - <<'PY'
import json
from pathlib import Path
import urllib.request
url = 'https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv'
request = urllib.request.Request(url, headers={'Range': 'bytes=0-1048575'})
with urllib.request.urlopen(request, timeout=60) as response:
    raw = response.read(1048576)
    metadata = {'url': url, 'http_status': response.status,
                'content_range': response.headers.get('Content-Range')}
complete = raw[:raw.rfind(b'\n') + 1]
if not complete:
    raise RuntimeError('No complete CSV lines downloaded')
root = Path('results/trace-source')
(root / 'azure-conversation-prefix.csv').write_bytes(complete)
(root / 'download.json').write_text(json.dumps(metadata, indent=2) + '\n')
PY
```

After normal setup caches Qwen's tokenizer, prepare the fixed 30-second smoke input:

```bash
source .venv/bin/activate
python prepare_trace.py \
  --source results/trace-source/azure-conversation-prefix.csv \
  --preset smoke \
  --output-dir results/trace-smoke-input
```

Prepare the separate fixed five-minute comparison input:

```bash
python prepare_trace.py \
  --source results/trace-source/azure-conversation-prefix.csv \
  --preset experiment \
  --output-dir results/trace-experiment-input
```

Both presets are versioned in [trace-presets.json](trace-presets.json). They pin
the exact prefix CSV SHA-256, model, window, seed, output cap, and sampling
probability. Smoke selects `[00:00:00.001163, 00:00:30.001163)` on May 12, 2024;
experiment selects `[00:01:00, 00:06:00)` in source timestamp coordinates. They do
not overlap. These are convenient initial validation/calibration windows, not a
random or representative sample of the full dataset or held-out MPC evaluation.
Presets only prepare inputs; the replay command below actually runs the experiment.
Reuse one prepared trace across policies and repetitions, rather than selecting a
new window per run. The trace records preset name, complete settings, protocol
version, and configuration SHA-256; the runner copies it into every result.

Preset settings cannot be overridden. Revise the configuration and protocol
version for a new frozen protocol. For a custom source or window, retain the
explicit CLI instead:

```bash
python prepare_trace.py \
  --source results/trace-source/azure-conversation-prefix.csv \
  --start 2024-05-12T00:07:00+00:00 --duration-seconds 300 \
  --output-dir results/trace-custom-input
```

`--sample-probability 0.5` deterministically thins arrivals without compressing
time in custom mode. Select the probability from source statistics and a separate calibration
run, then freeze it for comparisons. Thinning changes fine-scale burst structure
as well as intensity: report it. Retained timestamps and ties preserve their
original order. Naive timestamps are interpreted as UTC for relative arithmetic;
do not infer users' local peak hours.

`trace.json` contains prepared inputs and source SHA-256, window, seed, thinning
probability, exclusion counts, tokenizer fingerprint, and fixed output cap.
`source_audit.jsonl` separately preserves source line numbers, historical output
lengths, and each in-window row's disposition. Keep both and `download.json`.
Historical `GeneratedTokens` never enter runnable request fields or determine the
output cap. Input counts include the full chat template. Varied synthetic text is
generated before replay and re-tokenized before launch, not on the scheduler path.

Context plus fixed cap must fit 8192. Rows shorter than an empty chat template or
exceeding context are explicitly excluded, never silently truncated. Preparation
fails for incomplete windows, no retained requests, or unsupported tokenization.
Report exclusions: the retained trace is not the entire production workload.

## Server and replay

For repeated comparisons, stop the owned server with Ctrl+C and restart using the
existing setup with prefix caching explicitly disabled:

```bash
bash setup_and_start.sh --skip-install --disable-prefix-caching
```

This optional flag uses vLLM's documented
[`--no-enable-prefix-caching`](https://docs.vllm.ai/en/v0.30.0/cli/serve/).
The setup default is unchanged. Keep its `launch.json`, runtime, and package
records. Replaying identical prompts against an evolving cache can bias run order.
Runner `--cache-policy` is self-reported, not verified through `/v1/models`.

In a second terminal:

```bash
source .venv/bin/activate
python run_trace_pilot.py \
  --trace results/trace-smoke-input/trace.json \
  --output-dir results/trace-smoke-96 \
  --budget 96 --ttft-slo 0.5 --cache-policy disabled
```

The 0.5-second SLO is an example, not an established product requirement. Declare
it before comparison. The maximum client scheduling lag is also predeclared
(`--max-scheduling-lag`, default 50 ms). Changing either threshold after seeing a
result requires a documented protocol revision and rerunning comparisons.

The runner owns one gateway, validates the existing model and 8192 context, warms
up, and drains before measurement. It schedules independently of responses,
allocates request IDs and outcome rows before warmup and before choosing the
measurement clock origin, never retries, and never silently skips traffic when
its pending limit is exceeded. Pending-limit overflow invalidates collection,
with explicit not-launched rows. Requests pending beyond the drain deadline retain
their cancellation outcome. Existing per-request upstream HTTP clients remain.

Requests use temperature zero, a fixed cap, and natural EOS. This is not exact
output-cost replay. Synthetic text changes output lengths and semantics, and the
source used another model/service. A later fixed-length service-demand experiment
must be labeled separately; never expose historical output lengths as a production
controller feature. Future controller interfaces must also hide future arrivals.

## Outputs and acceptance

Outputs: copied `trace.json`, `manifest.json`, `client.json`, `warmup.json`,
`gateway.jsonl`, `budget_changes.json`, `state_observations.json`, `report.json`,
`timeseries.png`, `telemetry_status.json`, and process log. Shutdown status confirms
the final flush completed without dropped records or write errors. Manifest records input hashes, code revision,
packages, GPU/driver, configuration, and shared-host monotonic boundaries.

The report separates:

- **Integrity** (`valid`): planned rows and unique IDs, lifecycle ordering,
  cumulative counters, token counts, budget application, timestamps, final drain,
  and telemetry loss. The shared-host monotonic clock also checks that gateway
  receipt follows client start and client first content follows gateway first
  content. Gateway terminal cleanup may finish after the client.
- **Load fidelity** (`load_fidelity.valid`): all traffic launched and scheduling
  lag stayed within the predeclared limit. Start time is not an exact wire-send
  probe; connection-pool/transport delay remains in client-to-handler latency.
- **Performance**: successful-only latency percentiles alongside all success,
  rejection, and failure counts. SLO goodput requires successful completion and
  `first_content_at - scheduled_at <= ttft_slo`; denominator is the arrival-window
  duration. This cohort metric is not steady-state completion throughput.

Recorded connection failure before the handler runs can pass integrity. A backend
completion with failed client delivery counts as a client failure, preserving both
observations. Backend HTTP 503 is a failure; admission-capacity 503 is a rejection.
Neither is removed from the denominator. Tail spikes stay in primary metrics.
Client and gateway use the same text-SSE field validation: non-text content,
malformed choices/usage, and explicit error events cannot count as success.
Correct records with inaccurate scheduling remain integrity-valid but
load-fidelity-invalid. CLI exits nonzero if either check fails.

The plot aligns planned arrivals, client starts, handler arrivals, reserved cost,
client TTFT, client-to-handler delay, scheduling lag, and event-loop lag. The loop
monitor samples every 100 ms, records delays after resuming, and skips catch-up
sampling. State polls record sample time, response time, and age. These reveal
timing disturbances, not their CPU/GC/GPU root cause. Keep instrumentation identical
across comparisons; its overhead has not yet been GPU-benchmarked.

Reanalyze with `python analyze_trace_pilot.py results/trace-smoke-96`.

For the five-minute preset, use `--trace results/trace-experiment-input/trace.json`
and a fresh output directory such as `results/trace-experiment-96-r1`. Keep budget,
SLO, cache policy, and run order explicit in replay commands; the preset freezes
the offered input, not the controller setting being compared.

## Comparison protocol after smoke passes

Freeze inputs and server settings. Reserve separate chronological source windows
for calibration, model validation, and final evaluation; never shuffle neighboring
seconds across train/test. Obtain additional windows beyond this short download
before generalization claims. Evaluate budgets 80, 96, 112 on the same five-minute
trace, initially three repetitions each in interleaved order. Report run-level
variation, not requests treated as independent repetitions. Include per-input-size
outcomes so rejecting expensive requests cannot hide behind aggregate percentiles.

This PR supplies replay and auditing. Budget excitation, model fitting, intentional
stall injection, feedback baselines, shadow MPC, and closed-loop evaluation follow
in separate changes. Real GPU results are required before claiming a benefit.
