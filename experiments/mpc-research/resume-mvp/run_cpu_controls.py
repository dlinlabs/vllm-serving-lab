"""Run the scoped MPC CPU/mock controls; never launch a GPU backend."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output = (args.output or ROOT / "cpu-ci-artifacts" / ("mpc-" + stamp)).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    names = (
        "native_services_smoke", "native_nominal_qualification", "nominal_episode",
        "nominal_runtime", "nominal_planner", "nominal_permit", "nominal_scenario",
        "nominal_metrics", "nominal_workload", "nominal_client",
    )
    files = [str((HERE / ("test_" + name + ".py")).relative_to(ROOT)) for name in names]
    files.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "tests").glob("test_mpc_*.py"))
                 if path.name not in ("test_mpc_native_joint_predictive.py",
                                      "test_mpc_native_transport_training.py"))
    files.extend(("tests/test_gateway_diagnostics.py", "tests/test_telemetry.py"))
    settings = {
        "PYTHONPATH": os.pathsep.join((str(HERE), str(ROOT))),
        "NATIVE_SMOKE_TEST_OUTPUT": str((scratch / "native").relative_to(ROOT)),
        "NATIVE_SMOKE_PROTOCOL": str(HERE / "nominal_protocol.json"),
        "NOMINAL_RUNTIME_TEST_DIR": str(scratch / "runtime"),
        "NOMINAL_EPISODE_TEST_DIR": str(scratch / "episode"),
        "NOMINAL_WORKLOAD_SCRATCH": str(scratch / "workload"),
    }
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml", str(output / "junit.xml"), *files]
    metadata = {
        "scope": "CPU/mock only; not Native acceptance or a full repository test run",
        "python": sys.version, "command": command, "environment_overrides": settings,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    environment = dict(os.environ, **settings)
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(command, cwd=ROOT, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "pytest.log").write_text(result.stdout)
    (output / "pytest.exit-code").write_text(str(result.returncode) + "\n")
    print(result.stdout, end="")
    print(f"CPU evidence: {output}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
