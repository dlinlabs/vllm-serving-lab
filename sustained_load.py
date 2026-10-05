import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from statistics import quantiles
from pathlib import Path
from typing import Optional

import httpx

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"

@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    prompt: str
    max_tokens: int


SHORT_PROMPT = (
    "Explain admission control in an LLM serving system in a few sentences."
)

MEDIUM_PROMPT = " ".join(
    [
        "Explain how an inference gateway manages admission control, request scheduling, "
        "streaming token delivery, time-to-first-token measurement, throughput, and tail latency."
    ] * 3
)

LONG_PROMPT = " ".join(
    [
        "Explain how an inference gateway manages admission control, request scheduling, "
        "streaming token delivery, time-to-first-token measurement, throughput, tail latency, "
        "bounded concurrency, queueing, backpressure, batching efficiency, fairness, overload "
        "rejection, KV-cache pressure, and observability."
    ] * 12
)

WORKLOADS = [
    WorkloadSpec(
        name="short_interactive",
        prompt=SHORT_PROMPT,
        max_tokens=64,
    ),
    WorkloadSpec(
        name="medium",
        prompt=MEDIUM_PROMPT,
        max_tokens=128,
    ),
    WorkloadSpec(
        name="long_context",
        prompt=LONG_PROMPT,
        max_tokens=128,
    ),
    WorkloadSpec(
        name="long_output",
        prompt=SHORT_PROMPT,
        max_tokens=512,
    ),
]

@dataclass
class RequestResult:
    scheduled_at: float
    workload: str
    request_id: Optional[str] = None
    stream_complete: bool = False
    started_at: Optional[float] = None
    status: Optional[int] = None
    ttft: Optional[float] = None
    end_to_end: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    success: bool = False
    rejected: bool = False


async def consume_request(
    client: httpx.AsyncClient,
    scheduled_at: float,
    model: str,
    workload: WorkloadSpec,
    prompt_override: Optional[str] = None,
    max_tokens_override: Optional[int] = None,
    ignore_eos: bool = False,
) -> RequestResult:
    result = RequestResult(
        scheduled_at=scheduled_at,
        workload=workload.name,
        max_output_tokens=max_tokens_override or workload.max_tokens,
    )
    result.started_at = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_override or workload.prompt}],
        "temperature": 0,
        "max_tokens": max_tokens_override or workload.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }

    try:
        async with client.stream("POST", GATEWAY_URL, json=payload) as response:
            result.request_id = response.headers.get("x-request-id")
            result.status = response.status_code
            result.rejected = response.status_code == 503
            if result.rejected:
                await response.aread()
                result.end_to_end = time.monotonic() - result.started_at
                return result

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    result.stream_complete = True
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = event.get("usage")
                if usage:
                    result.input_tokens = usage.get("prompt_tokens")
                    result.output_tokens = usage.get("completion_tokens")
                    result.total_tokens = usage.get("total_tokens")

                choices = event.get("choices", [])
                content = (
                    choices[0].get("delta", {}).get("content")
                    if choices
                    else None
                )
                if content and result.ttft is None:
                    result.ttft = time.monotonic() - result.started_at

            result.success = response.is_success and result.stream_complete
    except httpx.HTTPError:
        result.success = False
    finally:
        result.end_to_end = time.monotonic() - result.started_at

    return result


async def run_benchmark(
    rps: float,
    duration: float,
    model: str,
    workload_name: Optional[str] = None,
    prompt_override: Optional[str] = None,
    max_tokens_override: Optional[int] = None,
    ignore_eos: bool = False,
) -> list[RequestResult]:
    interval = 1.0 / rps
    results: list[RequestResult] = []
    tasks: list[asyncio.Task[RequestResult]] = []
    start = time.monotonic()

    limits = httpx.Limits(
        max_connections=1000,
        max_keepalive_connections=1000,
    )
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        request_number = 0
        while True:
            scheduled_at = start + request_number * interval
            wait = scheduled_at - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            if scheduled_at >= start + duration:
                break
            if workload_name is None:
                workload = WORKLOADS[request_number % len(WORKLOADS)]
            else:
                workload = next(item for item in WORKLOADS if item.name == workload_name)
            print(
                f"request={request_number}, "
                f"workload={workload.name}, "
                f"max_tokens={max_tokens_override or workload.max_tokens}"
            )
            tasks.append(
                asyncio.create_task(
                    consume_request(
                        client,
                        scheduled_at,
                        model,
                        workload,
                        prompt_override,
                        max_tokens_override,
                        ignore_eos,
                    )
                )
            )
            request_number += 1

        if tasks:
            results = await asyncio.gather(*tasks)

    return results


def percentile(values: list[float], percent: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[int(percent) - 1]


def format_seconds(value: Optional[float]) -> str:
    return f"{value:.4f}s" if value is not None else "n/a"


def average(values: list[int]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def format_number(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def print_summary(results: list[RequestResult], rps: float, duration: float) -> None:
    successful = [result for result in results if result.success]
    ttft_values = [result.ttft for result in successful if result.ttft is not None]
    latency_values = [result.end_to_end for result in successful if result.end_to_end is not None]
    scheduling_lag_values = [
        result.started_at - result.scheduled_at
        for result in results
        if result.started_at is not None
    ]
    rejected = sum(result.rejected for result in results)
    completed = len(successful)

    print(f"target RPS: {rps}")
    print(f"duration: {duration}s")
    print(f"total requests scheduled: {len(results)}")
    print(f"successful requests: {completed}")
    print(f"rejected requests (503): {rejected}")
    print(f"failed requests: {len(results) - completed - rejected}")
    print(f"successful requests / arrival window: {completed / duration:.4f} requests/sec")
    print(f"P50 scheduling lag: {format_seconds(percentile(scheduling_lag_values, 50))}")
    print(f"P99 scheduling lag: {format_seconds(percentile(scheduling_lag_values, 99))}")
    print(f"P50 TTFT: {format_seconds(percentile(ttft_values, 50))}")
    print(f"P95 TTFT: {format_seconds(percentile(ttft_values, 95))}")
    print(f"P99 TTFT: {format_seconds(percentile(ttft_values, 99))}")
    print(f"P50 end-to-end latency: {format_seconds(percentile(latency_values, 50))}")
    print(f"P99 end-to-end latency: {format_seconds(percentile(latency_values, 99))}")
    print("\nPer-workload metrics:")

    for workload in WORKLOADS:
        # All client-side attempts for this workload.
        # This does NOT mean the gateway necessarily received every request.
        workload_all_results = [
            result
            for result in results
            if result.workload == workload.name
        ]

        attempted_count = len(workload_all_results)

        success_count = sum(
            result.success
            for result in workload_all_results
        )

        rejected_count = sum(
            result.rejected
            for result in workload_all_results
        )

        failed_count = attempted_count - success_count - rejected_count

        reject_rate = (
            rejected_count / attempted_count
            if attempted_count > 0
            else 0.0
        )

        # Latency metrics should only use successful requests.
        workload_successful_results = [
            result
            for result in workload_all_results
            if result.success
        ]

        workload_input_tokens = [
            result.input_tokens
            for result in workload_successful_results
            if result.input_tokens is not None
        ]

        workload_output_tokens = [
            result.output_tokens
            for result in workload_successful_results
            if result.output_tokens is not None
        ]

        usage_missing = sum(
            result.input_tokens is None or result.output_tokens is None
            for result in workload_successful_results
        )

        workload_ttft = [
            result.ttft
            for result in workload_successful_results
            if result.ttft is not None
        ]

        workload_latency = [
            result.end_to_end
            for result in workload_successful_results
            if result.end_to_end is not None
        ]

        print(
            f"{workload.name}: "
            f"attempted={attempted_count}, "
            f"success={success_count}, "
            f"rejected={rejected_count}, "
            f"failed={failed_count}, "
            f"reject_rate={reject_rate:.2%}, "
            f"avg_input_tokens={format_number(average(workload_input_tokens))}, "
            f"avg_output_tokens={format_number(average(workload_output_tokens))}, "
            f"usage_missing={usage_missing}, "
            f"P50 TTFT={format_seconds(percentile(workload_ttft, 50))}, "
            f"P99 TTFT={format_seconds(percentile(workload_ttft, 99))}, "
            f"P50 E2E={format_seconds(percentile(workload_latency, 50))}, "
            f"P99 E2E={format_seconds(percentile(workload_latency, 99))}"
        )


def export_results(
    path: str,
    results: list[RequestResult],
    rps: float,
    duration: float,
    model: str,
    workload_name: Optional[str],
    prompt_repeat: Optional[int] = None,
    max_tokens_override: Optional[int] = None,
    ignore_eos: bool = False,
) -> None:
    payload = {
        "config": {
            "target_rps": rps,
            "duration_seconds": duration,
            "model": model,
            "workload": workload_name or "heterogeneous",
            "prompt_repeat": prompt_repeat,
            "max_tokens_override": max_tokens_override,
            "ignore_eos": ignore_eos,
        },
        "requests": [asdict(result) for result in results],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sustained arrival-rate benchmark.")
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--workload",
        choices=[workload.name for workload in WORKLOADS],
        default=None,
        help="Run one workload class only. Omit for the deterministic heterogeneous mix.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=None,
        help="Repeat a deterministic calibration sentence N times as the request prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the workload output-token cap.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS and continue generation until max_tokens. Intended for controlled output calibration.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for machine-readable per-request benchmark results.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.rps <= 0 or args.duration <= 0:
        raise SystemExit("--rps and --duration must be greater than zero")
    if args.prompt_repeat is not None and args.prompt_repeat <= 0:
        raise SystemExit("--prompt-repeat must be greater than zero")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be greater than zero")

    prompt_override = None
    if args.prompt_repeat is not None:
        calibration_sentence = (
            "Explain one practical consideration when serving large language models efficiently. "
        )
        prompt_override = calibration_sentence * args.prompt_repeat

    results = await run_benchmark(
        args.rps,
        args.duration,
        args.model,
        args.workload,
        prompt_override,
        args.max_tokens,
        args.ignore_eos,
    )
    print_summary(results, args.rps, args.duration)
    if args.json_out:
        export_results(
            args.json_out,
            results,
            args.rps,
            args.duration,
            args.model,
            args.workload,
            args.prompt_repeat,
            args.max_tokens,
            args.ignore_eos,
        )


if __name__ == "__main__":
    asyncio.run(main())
