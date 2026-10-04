# vLLM Serving Benchmark Lab

A reproducible LLM serving and overload-control experiment using **vLLM** and **Qwen3-4B-Instruct** on a single NVIDIA RTX 3090.

The project studies two related questions:

1. How does increasing request concurrency affect LLM serving throughput and latency?
2. Under sustained overload, can bounded admission control protect tail latency without unnecessarily rejecting healthy traffic?

The second phase extends the original serving benchmark into a production-style reliability experiment using a streaming FastAPI gateway, bounded concurrency, bounded waiting capacity, and HTTP 503 load shedding.

---

## Architecture

### Baseline serving path

```text
Client / OpenAI-compatible API
        |
        v
HTTP /v1/chat/completions
        |
        v
vLLM OpenAI-Compatible API Server
        |
        v
vLLM Engine / Scheduler
        |
        v
PyTorch / CUDA
        |
        v
NVIDIA RTX 3090
        |
        v
Qwen3-4B-Instruct
```

### Overload-control path

```text
Open-loop benchmark client
        |
        v
FastAPI admission gateway :8080
        |
        +--> bounded in-flight requests
        +--> bounded waiting capacity
        +--> HTTP 503 load shedding
        |
        v
vLLM OpenAI-Compatible API Server :8000
        |
        v
vLLM Engine / Scheduler
        |
        v
RTX 3090
```

When admission capacity is exceeded, the gateway returns HTTP `503` instead of allowing backlog to grow without bound.

The gateway preserves streaming semantics so client-observed TTFT remains meaningful.

---

## Environment

### Original concurrency benchmark

- GPU: NVIDIA RTX 3090 24 GB
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- vLLM: `0.28.0`
- PyTorch: `2.13.0+cu132`
- Python: `3.12`
- Max model length: `8192`
- GPU memory utilization target: `0.90`

### Sustained-load / overload-control experiment

- GPU: NVIDIA RTX 3090 24 GB
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- vLLM: `0.29.0`
- PyTorch: `2.8.0+cu128`
- Python: `3.12.3`
- Max model length: `8192`
- GPU memory utilization target: `0.90`
- Cloud environment: RunPod

---

## Start the vLLM Server

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

The model's default context length was much larger than required for the benchmark. The maximum model length was reduced to 8,192 tokens to fit the available KV-cache budget on a single RTX 3090.

---

## Client Examples

`non_streaming.py` demonstrates a standard OpenAI-compatible chat-completion request.

`streaming.py` validates incremental token delivery with `stream=True`.

---

# Phase 1 — Concurrency Benchmark

## Methodology

The first experiment held the request workload constant while varying maximum request concurrency.

Fixed configuration:

- Requests per run: 100
- Input length: 256 tokens/request
- Output length: 128 tokens/request
- Temperature: 0
- EOS ignored
- Concurrency: 1, 4, 8, 16, 32, 64

Benchmark command template:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 100 \
  --random-input-len 256 \
  --random-output-len 128 \
  --max-concurrency <CONCURRENCY> \
  --ignore-eos \
  --temperature 0
```

## Results

| Concurrency | Output tok/s | Mean TTFT | P99 TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|
| 1 | 81.96 | 61.24 ms | 72.37 ms | 11.81 ms |
| 4 | 320.89 | 56.25 ms | 269.52 ms | 12.11 ms |
| 8 | 591.95 | 72.90 ms | 286.27 ms | 12.55 ms |
| 16 | 984.53 | 123.45 ms | 356.27 ms | 13.92 ms |
| 32 | 1632.44 | 202.30 ms | 350.69 ms | 14.68 ms |
| 64 | 2665.72 | 345.10 ms | 454.55 ms | 16.71 ms |

All formal benchmark runs completed with zero failed requests.

## Analysis

Increasing concurrency from 1 to 64 increased output throughput from `81.96` to `2665.72` tokens/s, approximately a **32.5x throughput increase** for a 64x increase in maximum concurrency.

The throughput gain came with increasing per-request latency:

```text
Mean TTFT: 61.24 -> 345.10 ms
Mean TPOT: 11.81 -> 16.71 ms
P99 TTFT: 72.37 -> 454.55 ms
```

At low concurrency, additional requests improve batching efficiency and aggregate throughput. At higher concurrency, throughput continues to increase but scaling becomes increasingly sublinear while TTFT and TPOT rise.

GPU utilization reached 100% during higher-concurrency testing, but throughput continued increasing afterward. Therefore:

> **GPU utilization alone is not sufficient evidence that an inference server has reached throughput saturation.**

This experiment established the throughput-versus-latency tradeoff, but did not establish steady-state overload behavior.

---

# Phase 2 — Sustained Arrival-Rate Benchmark

## Why a second benchmark was needed

The original benchmark used a fixed number of requests and varied maximum concurrency. That does not directly answer what happens when requests arrive continuously faster than the server can sustainably process them.

The second experiment therefore uses an **open-loop sustained arrival rate**. Requests are scheduled at fixed intervals:

```text
arrival interval = 1 / target RPS
```

New requests continue arriving regardless of whether previous requests have completed. This allows backlog and tail-latency collapse to emerge naturally when arrival rate exceeds sustainable service capacity.

## Sustained load generator

`sustained_load.py` sends streaming requests through the gateway and records:

- target RPS
- successful requests
- HTTP 503 rejected requests
- failed requests
- scheduling lag
- P50 / P95 / P99 TTFT
- P50 / P99 end-to-end latency

The load generator uses absolute monotonic-time scheduling so request completion time does not control the arrival rate. It also uses a large HTTP connection pool to avoid client-side connection limits becoming the bottleneck.

## Baseline sustained-load results

Fixed workload:

- Input: approximately 256 tokens
- Output: 128 tokens
- Temperature: 0
- Streaming enabled
- Duration: 60 seconds

| Arrival Rate | Success | Failed | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 10 RPS | 600 | 0 | 0.130 s | 0.236 s | 1.869 s | 2.004 s |
| 12 RPS | 720 | 0 | 0.133 s | 0.283 s | 1.970 s | 2.158 s |
| 15 RPS | 900 | 0 | 0.107 s | 1.077 s | 2.112 s | 3.262 s |
| 20 RPS | 1200 | 0 | 0.145 s | 1.549 s | 2.693 s | 4.383 s |
| 30 RPS | 1799 | 1 | 4.254 s | 13.449 s | 11.362 s | 18.772 s |

### Interpretation

The healthy low-tail-latency region was approximately `<= 12 RPS`. A clear latency knee appeared between `12–15 RPS`.

At 30 RPS, the system entered severe overload:

```text
P99 TTFT: 13.449 s
P99 E2E: 18.772 s
```

Median latency remained relatively stable at lower arrival rates while tail latency degraded first. This matters operationally because overload can become visible in P99 latency before request failures occur.

---

# Phase 3 — Admission Control

## Gateway design

The gateway supports two modes:

```text
ADMISSION_MODE=baseline
ADMISSION_MODE=protected
```

Protected mode uses:

- bounded in-flight concurrency
- bounded waiting capacity
- HTTP 503 load shedding
- streaming proxy behavior
- metrics for accepted, rejected, failed, waiting, and in-flight requests

Requests are rejected when total admitted work exceeds the configured capacity.

The purpose is not to make the GPU faster. The purpose is to prevent unbounded queue growth from degrading every admitted request.

## Threshold experiments

Three configurations were tested:

```text
32 in-flight / 8 waiting
48 in-flight / 8 waiting
64 in-flight / 8 waiting
```

### 32 / 8

At 20 RPS:

- 799 successful
- 401 rejected
- reject rate: ~33.4%
- P99 TTFT: 1.886 s
- P99 E2E: 3.808 s

At 30 RPS:

- 719 successful
- 1081 rejected
- reject rate: ~60.1%
- P99 TTFT: 2.999 s
- P99 E2E: 4.888 s

This configuration strongly protected latency under severe overload, but was too aggressive at moderate load.

### 48 / 8

At 20 RPS:

- 976 successful
- 224 rejected
- reject rate: ~18.7%
- P99 TTFT: 2.789 s
- P99 E2E: 4.870 s

At 30 RPS:

- 855 successful
- 945 rejected
- reject rate: ~52.5%
- P99 TTFT: 4.016 s
- P99 E2E: 6.186 s

This configuration reduced rejection compared with 32/8, but provided weaker latency protection and still performed poorly at moderate load.

### 64 / 8 — selected configuration

At 20 RPS:

- 1193 successful
- 7 rejected
- reject rate: ~0.58%
- P99 TTFT: 1.341 s
- P99 E2E: 3.874 s

At 30 RPS:

- 1136 successful
- 664 rejected
- reject rate: ~36.9%
- successful requests / arrival window: 18.93 req/s
- P99 TTFT: 3.037 s
- P99 E2E: 5.195 s

This configuration produced the best latency-versus-availability tradeoff among the tested static thresholds.

---

# Final Before / After Result

## 20 RPS

| Metric | Baseline | Protected 64 / 8 |
|---|---:|---:|
| Reject rate | 0% | 0.58% |
| P99 TTFT | 1.549 s | 1.341 s |
| P99 E2E | 4.383 s | 3.874 s |

At moderate load, admission control introduced almost no availability penalty while preserving similar or slightly better tail latency.

## 30 RPS

| Metric | Baseline | Protected 64 / 8 | Change |
|---|---:|---:|---:|
| Reject rate | 0% | 36.9% | intentional load shedding |
| P99 TTFT | 13.449 s | 3.037 s | **~77% lower** |
| P99 E2E | 18.772 s | 5.195 s | **~72% lower** |
| Successful requests / arrival window | 29.98 req/s | 18.93 req/s | bounded useful traffic |

Under severe overload, bounded admission control prevented unbounded queue growth and reduced P99 TTFT from `13.45 s` to `3.04 s`, while shedding excess work with controlled HTTP 503 responses.

---

# Phase 4 — Heterogeneous Workload Benchmark

The benchmark now includes a deterministic mixed workload instead of only one fixed request shape:

```text
short_interactive -> medium -> long_context -> long_output -> repeat
```

The four classes vary prompt size and output cap so the gateway is tested against non-uniform request costs.

### Baseline mixed-workload sweep

| Target RPS | Success | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 100/100 | 0% | 0.0777 s | 0.1047 s | 1.4815 s | 1.6855 s |
| 10 | 200/200 | 0% | 0.0822 s | 0.1278 s | 1.6489 s | 1.9107 s |
| 20 | 400/400 | 0% | 0.0878 s | 0.2275 s | 1.8687 s | 2.4990 s |
| 25 | 500/500 | 0% | 0.1657 s | 1.6064 s | 3.3102 s | 4.7673 s |
| 30 | 600/600 | 0% | 0.3160 s | 1.1534 s | 3.1606 s | 4.3891 s |

The mixed workload entered a clear queueing/degradation region around **20–30 RPS**. The 25 RPS and 30 RPS one-shot results were not monotonic, so the exact knee should not be treated as a single precise RPS without repeated trials.

### Static admission under heterogeneous load

At 25 RPS, a very small `4 in-flight / 2 waiting` bound was far too aggressive:

- success: 66 / 500
- reject rate: **86.8%**
- P99 TTFT: 1.6417 s
- P99 E2E: 3.0012 s

The low cap likely reduced continuous-batching efficiency while still adding gateway queueing.

A larger `64 in-flight / 8 waiting` bound behaved much better at 25 RPS:

- success: 500 / 500
- reject rate: **0%**
- P99 TTFT: 0.4124 s
- P99 E2E: 3.3049 s

However, the same `64/8` policy at 30 RPS did not generalize cleanly:

- success: 534 / 600
- reject rate: **11.0%**
- P99 TTFT: 1.9759 s
- P99 E2E: 4.2836 s

This is the key Phase 4 result: **a fixed request-count admission threshold can be strongly workload- and load-dependent.** A threshold that looks good at one operating point can reject too aggressively, reduce batching efficiency, or fail to protect TTFT at another.

The detailed environment notes, every run, every per-workload metric, and the exact static-policy comparisons are recorded in:

[experiments/heterogeneous-workload-2026-09-23.md](experiments/heterogeneous-workload-2026-09-23.md)

---


# Phase 5 — Token-Cost Calibration

Low-load calibration was added to separate the effects of request shape from queueing pressure. The benchmark treats serving load as a function of three core workload variables:

```text
Load = f(RPS, input tokens, output tokens)
```

At 1 RPS on the RTX 3090, realized input length increased from 32 to 1,418 tokens while output stayed near 128 tokens; P50 TTFT increased from 68.8 ms to 76.1 ms. In the output sweep, input stayed at 98 realized tokens while output was forced to 32, 128, 256, 512, and 1,024 tokens using `ignore_eos=true`. P50 E2E increased from 0.418 s to 13.972 s and was approximately linear with generated-token count.

The calibration confirms that equal request counts do not imply equal serving cost, but the isolated latency slopes are not used directly as admission weights. The next experiment will measure saturation RPS for controlled short, long-input, and long-output request shapes before deriving the first workload-aware admission policy.

Detailed methodology, complete results, and interpretation:

[experiments/token-cost-calibration-2026-10-04.md](experiments/token-cost-calibration-2026-10-04.md)

---

# Engineering Lessons

## 1. Tail latency reveals overload before outright failure

At 15–20 RPS, the server still completed all requests, but P99 TTFT degraded sharply. Availability alone would not have revealed the problem.

## 2. Admission control trades availability for bounded latency

Rejecting excess traffic can produce a better service than accepting every request and allowing all requests to experience multi-second queueing delay.

## 3. Thresholds must account for continuous batching

A low concurrency cap was not automatically better. `32/8` protected severe-overload latency but rejected too aggressively and likely reduced vLLM batching efficiency. `64/8` produced a better latency-versus-availability tradeoff.

## 4. TTFT is an outcome metric, not a per-request admission signal

TTFT is only known after a request has already been admitted. The gateway therefore uses current admitted work as the leading control signal and uses P99 TTFT to evaluate and tune the threshold.

## 5. GPU utilization is insufficient as an overload signal

The earlier concurrency benchmark showed throughput continuing to improve even after GPU utilization reached 100%.

## 6. Streaming semantics must be preserved

A proxy that buffers the complete model response would corrupt client-observed TTFT. The gateway forwards streamed chunks as they arrive.

## 7. Availability failure must be separated from overload

One attempted 12 RPS run produced 720 failures because both vLLM and the gateway were down. That run was excluded from performance analysis and rerun successfully after service health checks.

This reinforced the distinction between:

- service availability failure
- performance saturation
- overload-induced latency degradation

---

## Gateway Dependency Setup

```bash
pip install fastapi uvicorn httpx
```

## Start the Gateway

### Baseline mode

```bash
ADMISSION_MODE=baseline \
uvicorn gateway:app --host 0.0.0.0 --port 8080
```

### Protected mode

The selected configuration is:

```text
MAX_IN_FLIGHT=64
MAX_WAITING=8
```

Start it with:

```bash
ADMISSION_MODE=protected \
MAX_IN_FLIGHT=64 \
MAX_WAITING=8 \
uvicorn gateway:app --host 0.0.0.0 --port 8080
```

## Gateway Metrics

```bash
curl http://localhost:8080/metrics
```

Example metrics include:

- `current_in_flight`
- `current_waiting`
- `accepted_requests`
- `rejected_requests`
- `failed_requests`
- `max_waiting`

## Run Sustained-Load Benchmark

```bash
python3 sustained_load.py --rps 20 --duration 60
```

or:

```bash
python3 sustained_load.py --rps 30 --duration 60
```

---

## Current Limitations

- Single NVIDIA RTX 3090
- Single model
- Realized input/output token counts are recorded for calibration and benchmark analysis
- Static admission thresholds
- No deadline-aware scheduling
- No adaptive queue control
- No multi-GPU / tensor parallelism
- No distributed gateway
- No production retry/backoff policy
- `max_waiting` is supporting instrumentation rather than a precise measurement of vLLM's internal scheduler queue

The experiment intentionally focuses on one narrow production reliability question rather than building a complete serving platform.

## Next Steps

- Run workload-shape saturation sweeps across short, long-input, and long-output requests
- Derive workload-cost weights from saturation behavior
- Implement and compare workload-aware / token-cost-aware admission
- Add a baseline-vs-protected P99 TTFT graph
- Add benchmark health checks and richer telemetry
- Evaluate retry/backoff behavior
- Explore adaptive admission thresholds

The static overload-control experiment is complete. The next phase is workload-aware / token-cost-aware admission under heterogeneous traffic.
