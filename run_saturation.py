import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median


SHAPES = {
    "baseline": {
        "prompt_repeat": 8,
        "max_tokens": 128,
        "ignore_eos": True,
        "description": "~100 input / 128 output",
    },
    "prefill_heavy": {
        "prompt_repeat": 128,
        "max_tokens": 128,
        "ignore_eos": True,
        "description": "~1400 input / 128 output",
    },
    "decode_heavy": {
        "prompt_repeat": 8,
        "max_tokens": 512,
        "ignore_eos": True,
        "description": "~100 input / 512 output",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure saturation behavior for controlled LLM request shapes."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=list(SHAPES),
        default=list(SHAPES),
        help="Request shapes to sweep. Defaults to all controlled shapes.",
    )
    parser.add_argument(
        "--rps",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
    )
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="results/saturation")
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(path: Path, shape: str, repeat: int, duration: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload["requests"]
    successful = [r for r in requests if r["success"]]
    rejected = [r for r in requests if r["rejected"]]
    failed = [r for r in requests if not r["success"] and not r["rejected"]]

    ttft = [r["ttft"] for r in successful if r["ttft"] is not None]
    e2e = [r["end_to_end"] for r in successful if r["end_to_end"] is not None]
    input_tokens = [
        r["input_tokens"] for r in successful if r["input_tokens"] is not None
    ]
    output_tokens = [
        r["output_tokens"] for r in successful if r["output_tokens"] is not None
    ]

    config = payload["config"]
    return {
        "shape": shape,
        "target_rps": config["target_rps"],
        "repeat": repeat,
        "attempted": len(requests),
        "success": len(successful),
        "rejected": len(rejected),
        "failed": len(failed),
        "reject_rate": len(rejected) / len(requests) if requests else 0.0,
        "successful_requests_per_arrival_second": (
            len(successful) / duration if duration > 0 else None
        ),
        "avg_input_tokens": (
            sum(input_tokens) / len(input_tokens) if input_tokens else None
        ),
        "avg_output_tokens": (
            sum(output_tokens) / len(output_tokens) if output_tokens else None
        ),
        "p50_ttft": percentile(ttft, 50),
        "p95_ttft": percentile(ttft, 95),
        "p99_ttft": percentile(ttft, 99),
        "p50_e2e": percentile(e2e, 50),
        "p99_e2e": percentile(e2e, 99),
    }


def run_case(
    output_dir: Path,
    shape: str,
    rps: float,
    duration: float,
    repeat: int,
) -> dict:
    spec = SHAPES[shape]
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label = f"{shape}_rps{rps:g}_run{repeat}"
    json_path = raw_dir / f"{label}.json"

    command = [
        sys.executable,
        "sustained_load.py",
        "--rps",
        str(rps),
        "--duration",
        str(duration),
        "--workload",
        "short_interactive",
        "--prompt-repeat",
        str(spec["prompt_repeat"]),
        "--max-tokens",
        str(spec["max_tokens"]),
        "--json-out",
        str(json_path),
    ]
    if spec["ignore_eos"]:
        command.append("--ignore-eos")

    print(
        f"\n=== shape={shape} ({spec['description']}) "
        f"rps={rps:g} run={repeat} ===",
        flush=True,
    )
    subprocess.run(command, check=True)
    return summarize(json_path, shape, repeat, duration)


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.duration <= 0 or any(rps <= 0 for rps in args.rps):
        raise SystemExit("RPS, duration, and repeats must be greater than zero")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for shape in args.shapes:
        for rps in args.rps:
            for repeat in range(1, args.repeats + 1):
                rows.append(
                    run_case(
                        output_dir,
                        shape,
                        rps,
                        args.duration,
                        repeat,
                    )
                )

    fields = [
        "shape",
        "target_rps",
        "repeat",
        "attempted",
        "success",
        "rejected",
        "failed",
        "reject_rate",
        "successful_requests_per_arrival_second",
        "avg_input_tokens",
        "avg_output_tokens",
        "p50_ttft",
        "p95_ttft",
        "p99_ttft",
        "p50_e2e",
        "p99_e2e",
    ]

    runs_path = output_dir / "runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["shape"], row["target_rps"]), []).append(row)

    metric_fields = [
        "reject_rate",
        "successful_requests_per_arrival_second",
        "avg_input_tokens",
        "avg_output_tokens",
        "p50_ttft",
        "p95_ttft",
        "p99_ttft",
        "p50_e2e",
        "p99_e2e",
    ]
    median_path = output_dir / "medians.csv"
    with median_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["shape", "target_rps", *metric_fields],
        )
        writer.writeheader()
        for shape in args.shapes:
            for rps in sorted(args.rps):
                group = grouped[(shape, rps)]
                out = {"shape": shape, "target_rps": rps}
                for metric in metric_fields:
                    values = [r[metric] for r in group if r[metric] is not None]
                    out[metric] = median(values) if values else None
                writer.writerow(out)

    print(f"\nSaved raw request data to: {output_dir / 'raw'}")
    print(f"Saved per-run summaries to: {runs_path}")
    print(f"Saved median summaries to: {median_path}")


if __name__ == "__main__":
    main()
