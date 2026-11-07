"""Metadata freeze and bounded red/green runner; never starts services."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
SCOPE = Path(".runtime/r24-pre-gpu-b3-role6")
PYTHON = "/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/r24-mpc-py312-venv/bin/python"
OWN = ("native_nominal_qualification.py", "test_native_nominal_qualification.py")
PROTECTED = ("nominal_episode.py", "nominal_client.py", "nominal_metrics.py", "nominal_runtime.py",
             "nominal_permit.py", "nominal_planner.py", "nominal_scenario.py", "nominal_workload.py",
             "nominal_services.py", "nominal_protocol.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(pid):
    try:
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": values[19], "state": values[0]}
    except FileNotFoundError:
        return None


assert Path.cwd() == ROOT
mode = sys.argv[1]
if mode == "baseline":
    paths = [PUBLIC / name for name in (*OWN, *PROTECTED)] + [Path("gateway.py")]
    pins = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths}
    assert pins[str(PUBLIC / OWN[0])]["sha256"] == "1b2ad22e9a73870827c285c7caa77e95cab0491ca7043d204c7233914fc9675c"
    assert pins[str(PUBLIC / "nominal_episode.py")]["sha256"] == "935c2679c4b9fffdffb23ce88ca5351f91fa84e25c8c1edccedddaea4e4262a8"
    for name in OWN:
        shutil.copyfile(PUBLIC / name, SCOPE / "sources/before" / name)
    save(SCOPE / "BASELINE.json", {
        "pins": pins, "plan_sha256": sha(SCOPE / "PLAN.json"),
        "identity": identity(os.getpid()), "python": sys.version,
        "concurrent_smoke_not_frozen": str(PUBLIC / "native_services_smoke.py"),
    })
    print(json.dumps({"baseline_pins": len(pins), "tests_executed": 0}))
    raise SystemExit(0)

assert mode in ("red", "green")
attempt = SCOPE / "attempts" / mode
attempt.mkdir()
(attempt / "artifacts").mkdir()
(attempt / "sources").mkdir()
baseline = json.loads((SCOPE / "BASELINE.json").read_text())["pins"]
for name, record in baseline.items():
    if Path(name).name not in OWN:
        assert sha(Path(name)) == record["sha256"], name
pins = {name: sha(Path(name)) for name in baseline}
if mode == "green":
    frozen = json.loads((SCOPE / "GREEN-FREEZE.json").read_text())
    assert pins == frozen["source_pins"]
    assert sha(SCOPE / "control.py") == frozen["control_sha256"]
    for name, value in frozen["data_pins"].items():
        assert sha(Path(name)) == value, name
    pins.update(frozen["data_pins"])
for name in OWN:
    shutil.copyfile(PUBLIC / name, attempt / "sources" / name)
save(attempt / "pins-before.json", pins)
selectors = ([f"test_native_nominal_qualification.Controls.{name}" for name in (
    "test_pre_control_error_keeps_not_run_capture_and_zero_offers",
    "test_preparation_error_keeps_primary_reason_and_zero_offers",
)] if mode == "red" else ["test_native_nominal_qualification"])
command = ["env", f"PYTHONPATH={ROOT / PUBLIC}:{ROOT}", "PYTHONDONTWRITEBYTECODE=1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "PYTHONHASHSEED=0", "OMP_NUM_THREADS=1",
           "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
           "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1",
           f"NATIVE_SMOKE_TEST_OUTPUT={attempt / 'artifacts'}",
           f"NATIVE_SMOKE_PROTOCOL={ROOT / PUBLIC / 'nominal_protocol.json'}",
           "nice", "-n19", PYTHON, "-B", "-m", "unittest", "-v", *selectors]
save(attempt / "command.json", {"argv": command, "cwd": str(ROOT), "runner": identity(os.getpid()),
                              "monotonic": time.monotonic(), "source_frozen_before_tests": True})
started = time.monotonic()
deadline = started + 60
timed_out = False
with (attempt / "stdout.log").open("xb") as out, (attempt / "stderr.log").open("xb") as err:
    child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
    original = identity(child.pid)
    save(attempt / "process-start.json", original)
    try:
        child.wait(timeout=55)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.terminate()
        try:
            child.wait(timeout=min(2, max(0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
current = identity(child.pid)
result = {"returncode": child.returncode, "elapsed_s": time.monotonic() - started,
          "deadline": deadline, "timed_out": timed_out, "joined": child.returncode is not None,
          "original_identity": original, "identity_after_wait": current,
          "original_identity_gone": current is None or current["start_ticks"] != original["start_ticks"],
          "python": sys.version, "attempt": mode, "Native": False}
save(attempt / "result.json", result)
after = {name: sha(Path(name)) for name in pins}
save(attempt / "pins-after.json", after)
assert after == pins and result["original_identity_gone"] and result["elapsed_s"] <= 60
print(json.dumps(result, sort_keys=True))
raise SystemExit(child.returncode or int(timed_out))
