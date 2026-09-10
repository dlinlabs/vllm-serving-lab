import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from statistics import quantiles
from typing import Optional

import httpx


GATEWAY_URL = "http://localhost:8080/v1/chat/completions"
PROMPT = " ".join(
    [
        "Explain how an inference gateway manages admission control, request scheduling,"
        " streaming token delivery, time-to-first-token measurement, throughput, and tail"
        " latency under sustained load. Describe the trade-offs between bounded concurrency,"
        " queueing, backpressure, batching efficiency, fairness, overload rejection, and"
        " observability. Use precise technical language and discuss how a fixed prompt and"
        " fixed output limit improve benchmark comparability across repeated experiments."
    ]
    * 3
)


@dataclass
class RequestResult:
    scheduled_at: float
    started_at: Optional[float] = None
    status: Optional[int] = None
    ttft: Optional[float] = None
    end_to_end: Optional[float] = None
    success: bool = False
    rejected: bool = False


async def consume_request(
    client: httpx.AsyncClient,
    scheduled_at: float,
    model: str,
) -> RequestResult:
    result = RequestResult(scheduled_at=scheduled_at)
    result.started_at = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
    }

    try:
        async with client.stream("POST", GATEWAY_URL, json=payload) as response:
            result.status = response.status_code
            result.rejected = response.status_code == 503
            if result.rejected:
                await response.aread()
                result.end_to_end = time.monotonic() - result.started_at
                return result

            async for line in response.aiter_lines():
                if result.ttft is not None or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                content = choices[0].get("delta", {}).get("content") if choices else None
                if content:
                    result.ttft = time.monotonic() - result.started_at

            result.success = response.is_success
    except httpx.HTTPError:
        result.success = False
    finally:
        result.end_to_end = time.monotonic() - result.started_at

    return result


async def run_benchmark(rps: float, duration: float, model: str) -> list[RequestResult]:
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
            tasks.append(asyncio.create_task(consume_request(client, scheduled_at, model)))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sustained arrival-rate benchmark.")
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.rps <= 0 or args.duration <= 0:
        raise SystemExit("--rps and --duration must be greater than zero")
    results = await run_benchmark(args.rps, args.duration, args.model)
    print_summary(results, args.rps, args.duration)


if __name__ == "__main__":
    asyncio.run(main())
