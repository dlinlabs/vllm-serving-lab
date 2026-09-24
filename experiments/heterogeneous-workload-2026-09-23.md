# Heterogeneous Workload Admission-Control Experiments — 2026-09-23

This document records the full heterogeneous-workload experiment session run on a single NVIDIA RTX 3090.

The goal was to extend the earlier homogeneous sustained-load benchmark into a mixed-workload test and evaluate whether a static request-count admission policy generalizes when request costs differ.

## Environment

- Cloud: RunPod
- GPU: NVIDIA GeForce RTX 3090, 24 GB
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Python: `3.12.3`
- vLLM: `0.30.0`
- PyTorch after vLLM installation: `2.13.0+cu130`
- CUDA available: `True`
- vLLM server: `:8000`
- Admission gateway: `:8080`
- Model max length used for the experiments: `8192`

### Startup finding: KV-cache sizing

The first vLLM start attempted to honor the model's declared maximum sequence length of 262,144 tokens and failed during KV-cache initialization:

```text
model max seq len: 262144
KV cache required: 36.0 GiB
available KV cache memory: 11.89 GiB
estimated maximum model length from available memory: 86592
```

The experiment does not require a 262K-token context window, so the server was restarted with:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192
```

The server then started successfully and reported `max_model_len=8192`.

## Heterogeneous workload definition

The load generator cycles deterministically through four request classes:

| Workload | Input shape | max_tokens | Intended stress |
|---|---|---:|---|
| `short_interactive` | short prompt | 64 | light interactive request |
| `medium` | medium prompt repeated 3x | 128 | medium mixed cost |
| `long_context` | long prompt repeated 12x | 128 | heavier prefill/input cost |
| `long_output` | short prompt | 512 | intended longer decode/output cost |

The round-robin schedule is:

```text
short_interactive -> medium -> long_context -> long_output -> repeat
```

Important limitation: `max_tokens` is only an upper bound. The model may emit EOS earlier, so `long_output` is not yet guaranteed to actually generate 512 tokens. This is why the next phase should record realized input/output token counts.

## Metric semantics

The client records:

- client-side attempted requests
- success count
- HTTP 503 rejection count
- failed count
- rejection rate
- scheduling lag
- P50 / P95 / P99 TTFT
- P50 / P99 end-to-end latency
- per-workload success/rejection/latency

`attempted` means a client-side request attempt that produced a `RequestResult`; it does **not** prove the gateway received the request.

Also, the current metric:

```text
successful requests / arrival window
```

is `completed_successes / configured_arrival_duration`. Because the benchmark waits for all tasks to drain after scheduling stops, this is **not** the same as measured wall-clock completion throughput. Future runs should record benchmark wall time and post-arrival drain time.

---

# Smoke Tests

## Smoke A — 1 RPS, 8 seconds

- scheduled: 8
- success: 8
- rejected: 0
- failed: 0
- P50 scheduling lag: 0.0010 s
- P99 scheduling lag: 0.0696 s
- P50 TTFT: 0.0641 s
- P95 TTFT: 0.1339 s
- P99 TTFT: 0.1380 s
- P50 E2E: 1.3716 s
- P99 E2E: 1.6454 s

Per workload:

| Workload | Count | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|
| short_interactive | 2 | 0.0971 s | 0.1239 s | 0.8388 s | 0.8668 s |
| medium | 2 | 0.0555 s | 0.0577 s | 1.5912 s | 1.6267 s |
| long_context | 2 | 0.1024 s | 0.1383 s | 1.6117 s | 1.6460 s |
| long_output | 2 | 0.0623 s | 0.0623 s | 1.1871 s | 1.1883 s |

## Smoke B — 1 RPS, 8 seconds, after per-workload rejection accounting

- scheduled: 8
- success: 8
- rejected: 0
- failed: 0
- P50 scheduling lag: 0.0012 s
- P99 scheduling lag: 0.0602 s
- P50 TTFT: 0.0628 s
- P95 TTFT: 0.1025 s
- P99 TTFT: 0.1075 s
- P50 E2E: 1.3700 s
- P99 E2E: 1.5981 s

Per workload:

| Workload | Attempted | Success | Rejected | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 2 | 2 | 0 | 0% | 0.0884 s | 0.1083 s | 0.8326 s | 0.8549 s |
| medium | 2 | 2 | 0 | 0% | 0.0545 s | 0.0573 s | 1.5539 s | 1.5548 s |
| long_context | 2 | 2 | 0 | 0% | 0.0772 s | 0.0907 s | 1.5882 s | 1.5995 s |
| long_output | 2 | 2 | 0 | 0% | 0.0613 s | 0.0621 s | 1.1815 s | 1.1868 s |

---

# Baseline heterogeneous sweep

Gateway mode:

```bash
ADMISSION_MODE=baseline python gateway.py
```

No gateway admission rejection is applied in baseline mode.

## Overall results

| Target RPS | Duration | Scheduled | Success | Rejected | Failed | P50 TTFT | P95 TTFT | P99 TTFT | P50 E2E | P99 E2E | P99 scheduling lag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 20 s | 100 | 100 | 0 | 0 | 0.0777 s | 0.0927 s | 0.1047 s | 1.4815 s | 1.6855 s | 0.0028 s |
| 10 | 20 s | 200 | 200 | 0 | 0 | 0.0822 s | 0.0954 s | 0.1278 s | 1.6489 s | 1.9107 s | 0.0027 s |
| 20 | 20 s | 400 | 400 | 0 | 0 | 0.0878 s | 0.1293 s | 0.2275 s | 1.8687 s | 2.4990 s | 0.0058 s |
| 25 | 20 s | 500 | 500 | 0 | 0 | 0.1657 s | 1.3899 s | 1.6064 s | 3.3102 s | 4.7673 s | 0.0199 s |
| 30 | 20 s | 600 | 600 | 0 | 0 | 0.3160 s | 0.6894 s | 1.1534 s | 3.1606 s | 4.3891 s | 0.0159 s |

### Baseline interpretation

- 5–10 RPS remained low-latency.
- 20 RPS showed visible but still moderate tail growth.
- 25–30 RPS entered a clear queueing/degradation region.
- The 25 RPS run had worse P99 TTFT/E2E than the 30 RPS run, so the single-run results are not monotonic. This indicates meaningful run-to-run / batching variance; the exact knee should not be claimed as a single precise RPS from these one-shot runs.
- The defensible conclusion from this session is that the heterogeneous queueing knee is approximately in the **20–30 RPS region**.
- Load-generator scheduling lag stayed small relative to server latency, so the observed latency growth was not caused by the client falling far behind its own schedule.

## Per-workload baseline results

### 5 RPS

| Workload | Attempted | Success | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 25 | 25 | 0% | 0.0785 s | 0.1998 s | 0.8673 s | 1.0108 s |
| medium | 25 | 25 | 0% | 0.0776 s | 0.0916 s | 1.6698 s | 1.6833 s |
| long_context | 25 | 25 | 0% | 0.0768 s | 0.1018 s | 1.6681 s | 1.6880 s |
| long_output | 25 | 25 | 0% | 0.0781 s | 0.0947 s | 1.2670 s | 1.3307 s |

### 10 RPS

| Workload | Attempted | Success | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 50 | 50 | 0% | 0.0815 s | 0.0953 s | 0.9699 s | 1.0348 s |
| medium | 50 | 50 | 0% | 0.0792 s | 0.1630 s | 1.8722 s | 1.9789 s |
| long_context | 50 | 50 | 0% | 0.0836 s | 0.1134 s | 1.8790 s | 1.9225 s |
| long_output | 50 | 50 | 0% | 0.0847 s | 0.1268 s | 1.4358 s | 1.5609 s |

### 20 RPS

| Workload | Attempted | Success | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 100 | 100 | 0% | 0.0879 s | 0.2286 s | 1.2392 s | 1.3975 s |
| medium | 100 | 100 | 0% | 0.0862 s | 0.2056 s | 2.3943 s | 2.5461 s |
| long_context | 100 | 100 | 0% | 0.0889 s | 0.1785 s | 2.3951 s | 2.4953 s |
| long_output | 100 | 100 | 0% | 0.0886 s | 0.1987 s | 1.7961 s | 1.8547 s |

### 25 RPS

| Workload | Attempted | Success | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 125 | 125 | 0% | 0.1661 s | 1.6050 s | 2.1628 s | 2.9688 s |
| medium | 125 | 125 | 0% | 0.1658 s | 1.5810 s | 4.1668 s | 4.7857 s |
| long_context | 125 | 125 | 0% | 0.1624 s | 1.6109 s | 4.1606 s | 4.7762 s |
| long_output | 125 | 125 | 0% | 0.1691 s | 1.5905 s | 3.1240 s | 3.7747 s |

### 30 RPS

| Workload | Attempted | Success | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 150 | 150 | 0% | 0.3197 s | 1.1175 s | 2.1127 s | 2.6221 s |
| medium | 150 | 150 | 0% | 0.3127 s | 1.1482 s | 3.8971 s | 4.4984 s |
| long_context | 150 | 150 | 0% | 0.3164 s | 1.1454 s | 3.9152 s | 4.4702 s |
| long_output | 150 | 150 | 0% | 0.3231 s | 1.1341 s | 2.9208 s | 3.4807 s |

---

# Protected admission experiments

## Protected 4 in-flight / 2 waiting at 25 RPS

Configuration:

```text
MAX_IN_FLIGHT=4
MAX_WAITING=2
MAX_OUTSTANDING=6
```

Overall:

- scheduled: 500
- success: 66
- rejected: 434
- failed: 0
- reject rate: **86.8%**
- successful requests / arrival window: 3.3 req/s
- P50 scheduling lag: 0.0007 s
- P99 scheduling lag: 0.0021 s
- P50 TTFT: 0.7080 s
- P95 TTFT: 1.3435 s
- P99 TTFT: 1.6417 s
- P50 E2E: 1.9019 s
- P99 E2E: 3.0012 s

Per workload:

| Workload | Attempted | Success | Rejected | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 125 | 20 | 105 | 84.0% | 0.7975 s | 1.6737 s | 1.5644 s | 2.4619 s |
| medium | 125 | 15 | 110 | 88.0% | 0.6335 s | 1.2999 s | 2.1730 s | 2.8569 s |
| long_context | 125 | 15 | 110 | 88.0% | 0.5080 s | 1.4962 s | 2.0552 s | 3.0460 s |
| long_output | 125 | 16 | 109 | 87.2% | 0.7609 s | 1.1595 s | 1.8627 s | 2.1924 s |

Interpretation:

- This configuration is far too aggressive for this serving stack.
- It rejected 86.8% of requests.
- P99 TTFT was essentially not improved relative to the 25 RPS baseline (1.6417 s vs 1.6064 s).
- P50 TTFT was substantially worse.
- The low in-flight cap likely prevents vLLM from using enough concurrent sequences for efficient continuous batching.
- A concurrency bound must be large enough to preserve batching efficiency; "stricter" is not automatically "better."

## Protected 64 in-flight / 8 waiting at 25 RPS

Configuration:

```text
MAX_IN_FLIGHT=64
MAX_WAITING=8
MAX_OUTSTANDING=72
```

Overall:

- scheduled: 500
- success: 500
- rejected: 0
- failed: 0
- reject rate: **0%**
- P50 scheduling lag: 0.0011 s
- P99 scheduling lag: 0.0136 s
- P50 TTFT: 0.1223 s
- P95 TTFT: 0.3160 s
- P99 TTFT: 0.4124 s
- P50 E2E: 2.4172 s
- P99 E2E: 3.3049 s

Per workload:

| Workload | Attempted | Success | Rejected | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 125 | 125 | 0 | 0% | 0.1220 s | 0.3426 s | 1.5952 s | 1.8293 s |
| medium | 125 | 125 | 0 | 0% | 0.1234 s | 0.4227 s | 3.0741 s | 3.3294 s |
| long_context | 125 | 125 | 0 | 0% | 0.1230 s | 0.4105 s | 3.0586 s | 3.3092 s |
| long_output | 125 | 125 | 0 | 0% | 0.1216 s | 0.3723 s | 2.2451 s | 2.5780 s |

Compared with the single 25 RPS baseline run:

| Metric | Baseline | Protected 64/8 |
|---|---:|---:|
| Reject rate | 0% | 0% |
| P50 TTFT | 0.1657 s | 0.1223 s |
| P99 TTFT | 1.6064 s | 0.4124 s |
| P50 E2E | 3.3102 s | 2.4172 s |
| P99 E2E | 4.7673 s | 3.3049 s |

This one run showed strong improvement without rejection. However, because the baseline 25/30 RPS results were already non-monotonic, repeated trials are required before claiming a stable percentage improvement.

## Protected 64 in-flight / 8 waiting at 30 RPS

Overall:

- scheduled: 600
- success: 534
- rejected: 66
- failed: 0
- reject rate: **11.0%**
- successful requests / arrival window: 26.7 req/s
- P50 scheduling lag: 0.0008 s
- P99 scheduling lag: 0.0142 s
- P50 TTFT: 0.4678 s
- P95 TTFT: 1.3392 s
- P99 TTFT: 1.9759 s
- P50 E2E: 2.6191 s
- P99 E2E: 4.2836 s

Per workload:

| Workload | Attempted | Success | Rejected | Reject rate | P50 TTFT | P99 TTFT | P50 E2E | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_interactive | 150 | 131 | 19 | 12.67% | 0.4572 s | 1.9598 s | 1.7203 s | 3.0898 s |
| medium | 150 | 132 | 18 | 12.00% | 0.4431 s | 1.9579 s | 2.9355 s | 4.3508 s |
| long_context | 150 | 136 | 14 | 9.33% | 0.4864 s | 1.9759 s | 2.9342 s | 4.3877 s |
| long_output | 150 | 135 | 15 | 10.00% | 0.4694 s | 1.9570 s | 2.2834 s | 3.7486 s |

Compared with the single 30 RPS baseline run:

| Metric | Baseline | Protected 64/8 |
|---|---:|---:|
| Reject rate | 0% | 11.0% |
| P50 TTFT | 0.3160 s | 0.4678 s |
| P99 TTFT | 1.1534 s | 1.9759 s |
| P50 E2E | 3.1606 s | 2.6191 s |
| P99 E2E | 4.3891 s | 4.2836 s |

Interpretation:

- At 30 RPS the same static 64/8 policy rejected 11% of requests.
- Median E2E improved, but P99 E2E changed little.
- P50 and P99 TTFT were worse.
- A fixed request-count threshold that looked strong at 25 RPS did not generalize cleanly to the heavier heterogeneous load.

---

# Main findings

1. **Heterogeneous load changed the operating point.** The mixed workload remained healthy through roughly 20 RPS and entered a clear degradation region around 20–30 RPS.

2. **Static concurrency limits are sensitive to configuration.** A 4/2 policy was dramatically too restrictive and rejected 86.8% of 25 RPS traffic while failing to improve P99 TTFT.

3. **A larger static bound can preserve batching efficiency.** The 64/8 policy performed well in the 25 RPS run, with zero rejection and substantially lower observed tail latency.

4. **A fixed request-count policy does not generalize reliably.** The same 64/8 policy at 30 RPS rejected 11% and produced worse TTFT than baseline, while barely changing P99 E2E.

5. **Per-workload rejection was nearly uniform rather than cost-aware.** At 30 RPS, rejection rates were roughly 9–13% across workload classes. The gateway currently sees only "one request = one slot"; it does not reason about input tokens, expected output tokens, KV-cache occupancy, prefill cost, or decode duration.

6. **Run-to-run variance matters.** The one-shot 25 RPS baseline produced worse tail latency than the one-shot 30 RPS baseline. Final claims should therefore use repeated trials and report medians/confidence intervals rather than overfitting to a single run.

7. **The synthetic `long_output` class is not yet proven to be decode-heavy.** Its E2E latency was often lower than medium/long-context because `max_tokens=512` does not force 512 realized output tokens.

# Next experiment

Before implementing cost-aware admission:

1. Record actual input token count.
2. Record actual output token count.
3. Record total benchmark wall time and drain time.
4. Repeat key baseline/static-policy points multiple times.
5. Define a first request-cost estimate, for example:

```text
estimated_cost = input_tokens + alpha * max_output_tokens
```

6. Compare:
   - baseline
   - static bounded admission
   - workload-aware / cost-aware admission

The objective is to test whether token/cost-aware admission protects interactive TTFT more consistently than a fixed request-count threshold under heterogeneous traffic.
