# vLLM Serving Benchmark Lab

A reproducible LLM serving experiment using **vLLM** and **Qwen3-4B** on a single NVIDIA RTX 3090.

The goal of this project is to understand how request concurrency affects LLM serving throughput, time-to-first-token (TTFT), decode latency, and tail latency.

## Architecture

Client / OpenAI Python SDK  
↓ HTTP `/v1/chat/completions`  
vLLM OpenAI-Compatible API Server  
↓  
vLLM Engine / Scheduler  
↓  
PyTorch / CUDA  
↓  
NVIDIA RTX 3090  
↓  
Qwen3-4B-Instruct

Both non-streaming and streaming inference were validated through the OpenAI-compatible API.

## Environment

- GPU: NVIDIA RTX 3090 24 GB
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- vLLM: `0.28.0`
- PyTorch: `2.13.0+cu132`
- Python: `3.12`
- Max model length: `8192`
- GPU memory utilization target: `0.90`

## Start the Server

    vllm serve Qwen/Qwen3-4B-Instruct-2507 \
      --host 0.0.0.0 \
      --port 8000 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 8192

The model's default context length was 262,144 tokens. With the available GPU memory, this exceeded the KV-cache capacity required for a worst-case maximum-length request.

For this benchmark, the maximum model length was reduced to 8,192 tokens.

At startup, vLLM reported approximately:

- Model weights: 7.64 GiB
- Available KV-cache memory: 11.86 GiB
- GPU KV-cache capacity: 86,320 tokens

## Client Examples

`non_streaming.py` demonstrates a standard OpenAI-compatible chat-completion request.

`streaming.py` uses `stream=True` and consumes incremental response chunks through:

    chunk.choices[0].delta.content

## Benchmark Methodology

The workload was held constant while maximum request concurrency was varied.

Fixed configuration:

- Requests per run: 100
- Input length: 256 tokens/request
- Output length: 128 tokens/request
- Temperature: 0
- EOS ignored to keep output length consistent
- Concurrency: 1, 4, 8, 16, 32, 64

Benchmark command template:

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

Increasing concurrency from 1 to 64 increased output throughput from:

    81.96 -> 2665.72 tokens/s

This represents approximately a **32.5x throughput increase** for a 64x increase in maximum concurrency.

The throughput improvement came with increasing per-request latency:

    Mean TTFT: 61.24 -> 345.10 ms
    Mean TPOT: 11.81 -> 16.71 ms
    P99 TTFT: 72.37 -> 454.55 ms

At low concurrency, additional requests substantially improved batching efficiency and aggregate throughput.

At higher concurrency, throughput continued to increase but scaling became increasingly sublinear while TTFT and TPOT increased.

This demonstrates the fundamental serving trade-off:

    higher concurrency
          |
          +--> better batching / GPU utilization
          |          |
          |          +--> higher aggregate throughput
          |
          +--> more scheduling / execution contention
                     |
                     +--> higher per-request latency

GPU utilization reached 100% during higher-concurrency testing. However, throughput continued to increase substantially beyond the first observation of 100% utilization.

Therefore, **GPU utilization alone is not sufficient evidence that an inference server has reached throughput saturation**.

The tested range demonstrates a clear throughput-versus-latency trade-off and increasingly sublinear scaling, but it does not establish a hard throughput plateau.

## Operational Observations

Two failures encountered during the experiment were useful for distinguishing availability failures from performance saturation:

- A `ConnectionRefusedError` indicated that the API server was unavailable, rather than overloaded.
- A second attempted vLLM server failed with `OSError: [Errno 98] Address already in use` because an existing server was already bound to port 8000.

The existing server continued serving benchmark traffic successfully.

These incidents reinforced the distinction between process failure, service availability, and serving-performance degradation.

## Limitations

This experiment intentionally uses a bounded baseline:

- Single NVIDIA RTX 3090
- Single model
- Fixed 256-token input workload
- Fixed 128-token output workload
- 100 requests per benchmark
- No multi-GPU / tensor-parallel experiment
- No sustained-load saturation experiment

Because only 100 requests were used per run, increasing maximum concurrency beyond 64 would not provide a strong steady-state saturation measurement without redesigning the workload.

## Next Steps

The next phase extends this baseline toward production-oriented inference serving:

- Define latency and availability SLOs
- Add structured serving telemetry
- Test failure and recovery behavior
- Introduce overload / admission-control behavior
- Evaluate serving behavior against explicit reliability objectives
