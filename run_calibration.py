import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median


INPUT_REPEAT_SWEEP = [2, 8, 32, 64, 128]
OUTPUT_TOKEN_SWEEP = [32, 128, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate request cost versus realized token counts.")
    parser.add_argument("--rps", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="results/calibration")
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


def summarize(path: Path, sweep: str, target: int, repeat: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    successful = [r for r in payload["requests"] if r["success"]]
    input_tokens = [r["input_tokens"] for r in successful if r["input_tokens"] is not None]
    output_tokens = [r["output_tokens"] for r in successful if r["output_tokens"] is not None]
    ttft = [r["ttft"] for r in successful if r["ttft"] is not None]
    e2e = [r["end_to_end"] for r in successful if r["end_to_end"] is not None]
    return {
        "sweep": sweep,
        "target": target,
        "repeat": repeat,
        "success": len(successful),
        "avg_realized_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else None,
        "avg_realized_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "p50_ttft": percentile(ttft, 50),
        "p99_ttft": percentile(ttft, 99),
        "p50_e2e": percentile(e2e, 50),
        "p99_e2e": percentile(e2e, 99),
    }


def run_case(
    output_dir: Path,
    rps: float,
    duration: float,
    sweep: str,
    target: int,
    repeat: int,
) -> dict:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"{sweep}_{target}_run{repeat}.json"

    command = [
        sys.executable,
        "sustained_load.py",
        "--rps", str(rps),
        "--duration", str(duration),
        "--workload", "short_interactive",
        "--json-out", str(json_path),
    ]

    if sweep == "input":
        command.extend(["--prompt-repeat", str(target), "--max-tokens", "128"])
    else:
        command.extend([
            "--prompt-repeat", "8",
            "--max-tokens", str(target),
            "--ignore-eos",
        ])

    print(f"\n=== {sweep} target={target} run={repeat} ===", flush=True)
    subprocess.run(command, check=True)
    return summarize(json_path, sweep, target, repeat)


def main() -> None:
    args = parse_args()
    if args.rps <= 0 or args.duration <= 0 or args.repeats <= 0:
        raise SystemExit("RPS, duration, and repeats must be greater than zero")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for target in INPUT_REPEAT_SWEEP:
        for repeat in range(1, args.repeats + 1):
            rows.append(run_case(output_dir, args.rps, args.duration, "input", target, repeat))

    for target in OUTPUT_TOKEN_SWEEP:
        for repeat in range(1, args.repeats + 1):
            rows.append(run_case(output_dir, args.rps, args.duration, "output", target, repeat))

    fields = [
        "sweep", "target", "repeat", "success",
        "avg_realized_input_tokens", "avg_realized_output_tokens",
        "p50_ttft", "p99_ttft", "p50_e2e", "p99_e2e",
    ]
    runs_path = output_dir / "runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    median_path = output_dir / "medians.csv"
    median_fields = [
        "sweep", "target", "avg_realized_input_tokens", "avg_realized_output_tokens",
        "p50_ttft", "p99_ttft", "p50_e2e", "p99_e2e",
    ]
    with median_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=median_fields)
        writer.writeheader()
        for sweep, targets in (("input", INPUT_REPEAT_SWEEP), ("output", OUTPUT_TOKEN_SWEEP)):
            for target in targets:
                group = [r for r in rows if r["sweep"] == sweep and r["target"] == target]
                out = {"sweep": sweep, "target": target}
                for metric in median_fields[2:]:
                    values = [r[metric] for r in group if r[metric] is not None]
                    out[metric] = median(values) if values else None
                writer.writerow(out)

    print(f"\nSaved calibration runs to: {runs_path}")
    print(f"Saved calibration medians to: {median_path}")
    print("Use realized token columns, not target/repeat counts, for cost-model analysis.")


if __name__ == "__main__":
    main()
