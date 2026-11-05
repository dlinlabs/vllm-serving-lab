"""One <=60s focused CPU invocation; preserve all output and join the owned process."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode02")
ATTEMPT = SCOPE / "attempts/generation01-py312"
PYTHON = "/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/r24-mpc-py312-venv/bin/python3.12"


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}
    except FileNotFoundError:
        return None


assert Path.cwd() == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
pins = json.loads((SCOPE / "INPUT_PINS.json").read_text())
assert len(json.loads((SCOPE / "FREEZE.json").read_text())["selected_controls"]) == 4
ATTEMPT.mkdir(exist_ok=False)
(ATTEMPT / "artifacts").mkdir()


def check(name):
    values = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == item["sha256"]
              for path, item in pins.items()}
    save(ATTEMPT / name, values)
    assert all(values.values()), values


check("pins-before.json")
command = [
    "env", "-u", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
    "PYTHONHASHSEED=0", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1",
    f"NOMINAL_EPISODE_TEST_DIR={ATTEMPT / 'artifacts'}", "nice", "-n", "19", PYTHON,
    "-B", "-m", "unittest", "discover", "-s", "experiments/mpc-research/resume-mvp",
    "-p", "test_nominal_episode.py", "-v",
]
save(ATTEMPT / "command.json", {"argv": command, "cwd": str(Path.cwd())})
save(ATTEMPT / "runner-identity.json", identity(os.getpid()))
started = time.monotonic()
deadline = started + 60.0
timed_out = False
with (ATTEMPT / "stdout.log").open("xb") as stdout, (ATTEMPT / "stderr.log").open("xb") as stderr:
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    original = identity(child.pid)
    save(ATTEMPT / "process-start.json", original)
    try:
        child.wait(timeout=min(55, max(0, deadline - time.monotonic())))
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
gone = original is not None and (current is None or current["start_ticks"] != original["start_ticks"])
result = {
    "returncode": child.returncode, "elapsed_s": time.monotonic() - started,
    "absolute_test_deadline": deadline, "timed_out": timed_out, "joined": child.returncode is not None,
    "original_identity": original, "identity_after_wait": current, "original_identity_gone": gone,
    "python": sys.version, "attempt": 1, "selected_controls": 4,
    "peer_or_historical_suites_run": 0, "scope": "CPU-only new episode module, no Native permission",
}
save(ATTEMPT / "result.json", result)
check("pins-after.json")
print(json.dumps(result, sort_keys=True))
sys.exit(0 if child.returncode == 0 and gone and not timed_out and result["elapsed_s"] <= 60 else 1)
