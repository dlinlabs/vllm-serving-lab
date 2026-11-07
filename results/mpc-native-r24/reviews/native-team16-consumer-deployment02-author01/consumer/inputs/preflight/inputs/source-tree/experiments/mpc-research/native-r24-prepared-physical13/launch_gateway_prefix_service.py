"""Explicit source-bound factory handoff; preserve the pinned loopback launcher."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from gateway_prefix_bridge import SCHEMA, source_sha256


ORIGINAL_SHA256 = "cce5f87c37b5d0a6009d68b6311d8e20f61f066630d7039ea46ebd75a0563d59"


def handoff_arguments(executable, arguments):
    expected = [
        executable, "-m", "uvicorn", "gateway:app",
        "--host", "127.0.0.1", "--port", "8080", "--workers", "1",
    ]
    if arguments != expected:
        raise ValueError("gateway_prefix_unreviewed_exec_arguments")
    return [
        executable, "-m", "uvicorn", "gateway_prefix_bridge:application", "--factory",
        *arguments[4:],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("gateway",))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args, _ = parser.parse_known_args()
    root = Path(__file__).resolve().parents[3]
    original = root / "experiments/mpc-research/native-v2/launch_batch_service.py"
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("gateway_prefix_unreviewed_original_launcher")
    extension = json.loads(args.plan.read_text()).get("gateway_prefix_extension")
    launcher_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (not isinstance(extension, dict) or extension.get("schema") != SCHEMA
            or extension.get("source_sha256") != source_sha256()
            or extension.get("launcher_sha256") != launcher_sha):
        raise ValueError("gateway_prefix_prospectively_bound_extension_required")
    original_exec = os.execv

    def execute(executable, arguments):
        actual = handoff_arguments(executable, arguments)
        path = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] += os.pathsep + path
        receipt = {
            "schema": "r24-gateway-prefix-launch-v1",
            "source_commit": json.loads(args.plan.read_text())["source_commit"],
            "original_launcher_sha256": ORIGINAL_SHA256,
            "bridge_source_sha256": source_sha256(), "launcher_sha256": launcher_sha,
            "original_arguments": arguments, "actual_arguments": actual,
            "scope": "Same owned process, loopback and single worker. Explicit opt-in factory, no stock mutation or model qualification.",
        }
        with (args.evidence / "gateway-prefix-launch.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        return original_exec(executable, actual)

    os.execv = execute
    before_path = list(sys.path)
    sys.path.insert(0, str(original.parent))
    try:
        runpy.run_path(str(original), run_name="__main__")
    finally:
        os.execv = original_exec
        sys.path[:] = before_path


if __name__ == "__main__":
    main()
