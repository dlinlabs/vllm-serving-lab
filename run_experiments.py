import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated sustained-load experiment sweeps.")
    parser.add_argument("--rps", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--workload",
        choices=["heterogeneous", "short_interactive", "medium", "long_context", "long_output"],
        default="heterogeneous",
    )
    parser.add_argument("--output-dir", default="results")
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


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload["requests"]
    successful = [r for r in requests if r["success"]]
    rejected = [r for r in requests if r["rejected"]]
    failed = [r for r in requests if not r["success"] and not r["rejected"]]
    ttft = [r["ttft"] for r in successful if r["ttft"] is not None]
    e2e = [r["end_to_end"] for r in successful if r["end_to_end"] is not None]
    input_tokens = [r["input_tokens"] for r in successful if r["input_tokens"] is not None]
    output_tokens = [r["output_tokens"] for r in successful if r["output_tokens"] is not None]

    return {
        "target_rps": payload["config"]["target_rps"],
        "workload": payload["config"]["workload"],
        "attempted": len(requests),
        "success": len(successful),
        "rejected": len(rejected),
        "failed": len(failed),
        "reject_rate": len(rejected) / len(requests) if requests else 0.0,
        "avg_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else None,
        "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "p50_ttft": percentile(ttft, 50),
        "p95_ttft": percentile(ttft, 95),
        "p99_ttft": percentile(ttft, 99),
        "p50_e2e": percentile(e2e, 50),
        "p99_e2e": percentile(e2e, 99),
    }


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.duration <= 0 or any(rps <= 0 for rps in args.rps):
        raise SystemExit("RPS, duration, and repeats must be greater than zero")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for rps in args.rps:
        for repeat in range(1, args.repeats + 1):
            label = f"{args.workload}_rps{rps:g}_run{repeat}"
            json_path = raw_dir / f"{label}.json"
            command = [
                sys.executable,
                "sustained_load.py",
                "--rps",
                str(rps),
                "--duration",
                str(args.duration),
                "--json-out",
                str(json_path),
            ]
            if args.workload != "heterogeneous":
                command.extend(["--workload", args.workload])

            print(f"\n=== {label} ===", flush=True)
            subprocess.run(command, check=True)
            row = summarize(json_path)
            row["repeat"] = repeat
            rows.append(row)

    fieldnames = [
        "target_rps", "workload", "repeat", "attempted", "success", "rejected",
        "failed", "reject_rate", "avg_input_tokens", "avg_output_tokens",
        "p50_ttft", "p95_ttft", "p99_ttft", "p50_e2e", "p99_e2e",
    ]
    csv_path = output_dir / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["target_rps"], []).append(row)

    median_path = output_dir / "medians.csv"
    metric_fields = [
        "reject_rate", "avg_input_tokens", "avg_output_tokens",
        "p50_ttft", "p95_ttft", "p99_ttft", "p50_e2e", "p99_e2e",
    ]
    with median_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_rps", "workload", *metric_fields])
        writer.writeheader()
        for rps in sorted(grouped):
            group = grouped[rps]
            out = {"target_rps": rps, "workload": args.workload}
            for metric in metric_fields:
                values = [row[metric] for row in group if row[metric] is not None]
                out[metric] = median(values) if values else None
            writer.writerow(out)

    print(f"\nSaved raw request data to: {raw_dir}")
    print(f"Saved per-run summaries to: {csv_path}")
    print(f"Saved median summaries to: {median_path}")


if __name__ == "__main__":
    main()
