"""One finite pair of mocked CPU-control attempts, with one original 60s bound."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


os.chdir(Path(__file__).resolve().parent)
os.umask(0o077)
assert not Path("attempts").exists()
Path("attempts").mkdir()
began = time.monotonic()
deadline = began + 60
freeze = json.loads(Path("CODE-FREEZE.json").read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return dict(pid=pid, start_ticks=int(text[text.rfind(")") + 2:].split()[19]))


save(Path("outer-budget.json"), dict(start=began, deadline=deadline, runner=identity(os.getpid())))
results = []
for label, python in (
    ("py314", "/home/daniel_lin/gt/pr32/mayor/mpc-local-r21/.venv/bin/python"),
    ("py312", "/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/r24-mpc-py312-venv/bin/python"),
):
    if time.monotonic() >= deadline - 5:
        results.append(dict(interpreter=label, status="NOT_RUN_ORIGINAL_BUDGET"))
        break
    for name, expected in freeze["new_sources"].items():
        assert hashlib.sha256((Path("frozen") / name).read_bytes()).hexdigest() == expected
    attempt = Path("attempts") / (label + "-01")
    attempt.mkdir()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONPROFILEIMPORTTIME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               NATIVE_SMOKE_TEST_OUTPUT=str(attempt / "evidence"),
               NATIVE_SMOKE_PROTOCOL="inputs/nominal_protocol.json")
    argv = ["nice", "-n", "19", python, "-B", "frozen/test_native_services_smoke.py"]
    save(attempt / "command.json", dict(argv=argv, cwd=str(Path.cwd()), outer_deadline=deadline,
                                       full_environment_recorded=False, evidence_kind="CPU_MOCK_ONLY"))
    with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        row = identity(process.pid)
        save(attempt / "process.json", row)
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
    results.append(dict(interpreter=label, exit=code, joined=process.poll() is not None, identity=row))
    print(label, code, flush=True)
    print((attempt / "stdout.log").read_text(), flush=True)
    print((attempt / "stderr.log").read_text(), flush=True)
    if code == 124:
        break
save(Path("control-results.json"), dict(start=began, deadline=deadline, end=time.monotonic(),
                                       attempts=results, no_retry=True))
raise SystemExit(0 if len(results) == 2 and all(r.get("exit") == 0 for r in results) else 1)
