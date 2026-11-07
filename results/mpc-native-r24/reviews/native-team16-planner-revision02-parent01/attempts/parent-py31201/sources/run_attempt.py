"""Run one bounded validation attempt; retain its exact inputs and every log."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHONS = {
    "314": ROOT / ".venv/bin/python3.14",
    "312": Path(
        "/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d"
        "/files/r24-mpc-py312-venv/bin/python3.12"),
}
ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs():
    failures = []
    rows = (HERE / "INPUT_SHA256SUMS").read_text().splitlines()
    for line in rows:
        expected, relative = line.split("  ", 1)
        if sha(ROOT / relative) != expected:
            failures.append(relative)
    if failures:
        raise ValueError("frozen_inputs_changed:" + ",".join(failures))
    return len(rows)


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run_from_repository_root")
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("python", choices=PYTHONS)
    parser.add_argument("suite", choices=("predecessor", "witnesses", "derived"))
    args = parser.parse_args()
    if not args.label.replace("-", "").replace("_", "").isalnum():
        raise ValueError("simple_attempt_label_required")
    attempt = HERE.relative_to(ROOT) / "attempts" / args.label
    attempt.mkdir(parents=True, exist_ok=False)
    source_dir = attempt / "sources"
    source_dir.mkdir()
    for path in sorted(HERE.glob("*.py")):
        shutil.copyfile(path, source_dir / path.name)
    pinned = verify_inputs()
    runner = (
        ROOT / ".runtime/r24-mpc-parallel15-planner/run_tests.py"
        if args.suite == "predecessor" else HERE / "run_tests.py")
    command = ["nice", "-n", "19", str(PYTHONS[args.python]), "-B", str(runner)]
    if args.suite != "predecessor":
        command.append(args.suite)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(ENVIRONMENT)
    metadata = {
        "label": args.label, "suite": args.suite, "cwd": str(ROOT),
        "command": command, "env_overrides": ENVIRONMENT,
        "env_unset": ["PYTHONPATH"], "input_pins_verified_before": pinned,
        "runner_sha256": sha(runner),
        "source_snapshot_sha256": {
            path.name: sha(path) for path in sorted(source_dir.iterdir())},
        "started_unix_s": time.time(),
    }
    with (attempt / "command.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"attempt": args.label, "command": command}), flush=True)
    started = time.monotonic()
    with (attempt / "output.log").open("x") as stream:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            check=False)
    result = {
        "returncode": completed.returncode, "wall_s": time.monotonic() - started,
        "input_pins_verified_after": verify_inputs(),
        "output_sha256": sha(attempt / "output.log"),
    }
    with (attempt / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print((attempt / "output.log").read_text(), end="")
    print(json.dumps({"attempt": args.label, **result}, sort_keys=True))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
