# Token-Cost Calibration — 2026-10-04

## Goal

Measure how realized input-token and output-token counts affect request latency under low-load conditions before designing workload-aware admission control.

The calibration intentionally isolates request shape from queueing pressure:

- GPU: NVIDIA RTX 3090 24 GB
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- vLLM: `0.30.0`
- PyTorch: `2.13.0+cu132`
- Python: `3.12.3`
- Max model length: `8192`
- Gateway mode: baseline
- Arrival rate: `1 RPS`
- Duration per condition: `12 s`
- Repeats: `3`
- Streaming enabled
- Temperature: `0`

The benchmark records realized prompt/completion token usage from the OpenAI-compatible `usage` payload rather than assuming requested limits equal actual token counts.

## Calibration design

Three workload variables are treated separately:

```text
Load = f(RPS, input tokens, output tokens)
```

This phase holds RPS low and changes only one token dimension at a time.

### Input sweep

- Prompt size varied with deterministic prompt repetition.
- Output cap held at 128 tokens.
- Natural EOS behavior retained.
- Primary metric: TTFT, with E2E retained as supporting data.

### Output sweep

- Prompt held constant at approximately 98 realized input tokens.
- Output target varied across 32, 128, 256, 512, and 1024 tokens.
- `ignore_eos=true` used only for this sweep so realized output length matches the controlled target.
- Primary metric: E2E latency.

A smoke test caught an important issue before the formal run: without `ignore_eos`, output requests with `max_tokens` 256/512/1024 stopped naturally around 140–150 completion tokens. The calibration runner was therefore changed so output-only experiments disable EOS while normal/input-sweep traffic preserves natural EOS semantics.

## Formal results

### Input-token sweep

| Target prompt repeat | Avg realized input tokens | Avg realized output tokens | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32.0 | 124.67 | 0.06876 s | 0.08522 s | 1.55747 s | 1.57098 s |
| 8 | 98.0 | 128.0 | 0.06944 s | 0.08179 s | 1.56759 s | 1.58088 s |
| 32 | 362.0 | 128.0 | 0.07044 s | 0.08389 s | 1.57689 s | 1.59194 s |
| 64 | 714.0 | 128.0 | 0.07338 s | 0.09452 s | 1.60648 s | 1.63000 s |
| 128 | 1418.0 | 128.0 | 0.07606 s | 0.10157 s | 1.65693 s | 1.67845 s |

Observed behavior:

- TTFT increased gradually as realized input size increased.
- The increase was small at this model size and 1 RPS, but directionally consistent.
- A simple linear fit across these points gives approximately:
  - intercept: `68.8 ms`
  - slope: `0.0053 ms / input token`
  - equivalently, about `5.3 ms` additional TTFT per 1,000 input tokens
- The fit is strong enough to treat the trend as real for this environment, but the coefficient should not be generalized across hardware/models.

### Output-token sweep

| Target output tokens | Avg realized input tokens | Avg realized output tokens | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 98.0 | 32.0 | 0.05547 s | 0.08014 s | 0.41761 s | 0.44182 s |
| 128 | 98.0 | 128.0 | 0.07054 s | 0.08685 s | 1.57556 s | 1.58965 s |
| 256 | 98.0 | 256.0 | 0.07122 s | 0.08353 s | 3.11026 s | 3.12492 s |
| 512 | 98.0 | 512.0 | 0.07789 s | 0.09353 s | 6.41279 s | 6.45499 s |
| 1024 | 98.0 | 1024.0 | 0.08346 s | 0.09859 s | 13.97172 s | 14.16032 s |

Observed behavior:

- TTFT stayed relatively flat as output length changed.
- E2E latency scaled almost linearly with generated-token count.
- A simple five-point linear fit gives approximately `13.7 ms / output token` on this setup.
- The 1024-token point is slightly more expensive than a perfect linear extrapolation from 512 tokens, which may reflect longer-decode/KV-cache effects; this should not be over-interpreted from one low-load calibration.

## Interpretation

The calibration supports the core project thesis:

> One request is not a uniform unit of serving load.

Request cost depends strongly on token shape. Input tokens primarily affect prefill and TTFT, while output tokens dominate decode duration in this setup.

However, the calibration slopes are not yet a production admission-control cost function. TTFT and E2E are outcome metrics, while admission control needs a proxy for future GPU capacity consumed by an arriving request.

The next experiment therefore should not directly set:

```text
cost = alpha * input_tokens + beta * output_tokens
```

from the latency slopes alone.

Instead, the next phase will measure saturation behavior for fixed request shapes while increasing arrival rate.

## Next experiment

Use three controlled workload shapes:

```text
A: short       ~100 input / 128 output
B: long-input  ~1400 input / 128 output
C: long-output ~100 input / 512 output
```

Sweep arrival rate for each shape and find the region where queueing / P99 TTFT degrades materially.

Candidate RPS sweep:

```text
5, 10, 15, 20, 25, 30 RPS
```

The goal is to connect all three variables:

```text
RPS + input tokens + output tokens
```

and estimate relative workload cost from saturation capacity rather than relying only on isolated latency slopes.

That result will drive the first workload-aware / token-cost-aware admission policy.
