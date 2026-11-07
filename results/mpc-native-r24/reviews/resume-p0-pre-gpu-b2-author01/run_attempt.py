"""Freeze and run one planned red/green CPU attempt, without retries."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CAP = Path(".runtime/r24-pre-gpu-b2-role3")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
phase = sys.argv[1]
assert phase in ("red01", "green01")
os.umask(0o077)
attempt = CAP / "attempts" / phase
attempt.mkdir(parents=True)
baseline = json.loads((CAP / "BASELINE.json").read_text())
sources = {}
for name in ("native_services_smoke.py", "test_native_services_smoke.py", "nominal_protocol.json"):
    data = (PUBLIC / name).read_bytes()
    sources[name] = hashlib.sha256(data).hexdigest()
    (attempt / name).write_bytes(data)
assert sources["nominal_protocol.json"] == baseline["bindings"][str(PUBLIC / "nominal_protocol.json")]["sha256"]
if phase == "red01":
    assert sources["native_services_smoke.py"] == baseline["bindings"][str(PUBLIC / "native_services_smoke.py")]["sha256"]
else:
    red = json.loads((CAP / "attempts/red01/SOURCE-FREEZE.json").read_text())
    red_result = json.loads((CAP / "attempts/red01/evidence/summary.json").read_text())
    assert sources["test_native_services_smoke.py"] == red["sources"]["test_native_services_smoke.py"]
    assert red_result["tests"] == 4 and red_result["failures"] > 0 and red_result["errors"] == 0
    assert (CAP / "attempts/red01/exit-code.txt").read_text().strip() == "1"
    assert sources["native_services_smoke.py"] != red["sources"]["native_services_smoke.py"]
specs = {name: hashlib.sha256((CAP / name).read_bytes()).hexdigest()
         for name in ("PLAN.json", "run_attempt.py", "execute_controls.py")}
(attempt / "SOURCE-FREEZE.json").write_text(json.dumps(dict(
    phase=phase, sources=sources, specifications=specs, frozen_before_execution=True),
    sort_keys=True, indent=2) + "\n")
started = time.monotonic()
deadline = started + 60


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return dict(pid=pid, start_ticks=int(text[text.rfind(")") + 2:].split()[19]))


env = dict(os.environ)
for key in ("PYTHONPROFILEIMPORTTIME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
    env.pop(key, None)
env.update(PYTHONPATH=str(Path.cwd()) + ":" + str((Path.cwd() / PUBLIC)),
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
           PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
           NATIVE_SMOKE_TEST_OUTPUT=str(attempt / "evidence"),
           NATIVE_SMOKE_PROTOCOL=str((Path.cwd() / PUBLIC / "nominal_protocol.json")))
argv = ["nice", "-n", "19", sys.executable, "-B", str(CAP / "execute_controls.py"), phase]
(attempt / "command.json").write_text(json.dumps(dict(
    argv=argv, cwd=str(Path.cwd()), environment={key: env[key] for key in (
        "PYTHONPATH", "NATIVE_SMOKE_TEST_OUTPUT", "NATIVE_SMOKE_PROTOCOL", "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD")},
    unset=["PYTHONPROFILEIMPORTTIME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"],
    runner_identity=identity(os.getpid()), original_deadline=deadline), sort_keys=True, indent=2) + "\n")
with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
    process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    row = identity(process.pid)
    (attempt / "process.json").write_text(json.dumps(row) + "\n")
    try:
        code = process.wait(timeout=max(0, deadline - 5 - time.monotonic()))
    except subprocess.TimeoutExpired:
        if identity(process.pid) == row and time.monotonic() < deadline:
            fd = os.pidfd_open(process.pid)
            try:
                if identity(process.pid) == row:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
            finally:
                os.close(fd)
        process.wait(timeout=max(0, deadline - time.monotonic()))
        code = 124
(attempt / "exit-code.txt").write_text(str(code) + "\n")
after = {name: hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() for name in sources}
assert after == sources, "source/input changed during attempt"
(attempt / "result.json").write_text(json.dumps(dict(
    phase=phase, exit=code, joined=process.poll() is not None, started=started, ended=time.monotonic(),
    original_deadline=deadline, source_bytes_unchanged=True), sort_keys=True, indent=2) + "\n")
print((attempt / "stdout.log").read_text(), flush=True)
print((attempt / "stderr.log").read_text(), flush=True)
raise SystemExit(code)
