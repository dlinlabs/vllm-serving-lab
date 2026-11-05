"""Single frozen two-interpreter CPU attempt set; one original outer deadline."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


HERE = Path(__file__).resolve().parent
os.chdir(HERE)
os.umask(0o077)
assert not Path("attempts").exists()
Path("attempts").mkdir()
outer = time.monotonic() + 110
test_end = outer - 10
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0
interpreters = {
    "py314": "/home/daniel_lin/gt/pr32/mayor/mpc-local-r21/.venv/bin/python",
    "py312": "/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/r24-mpc-py312-venv/bin/python",
}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(fields[19])}


save(Path("outer-budget.json"), {
    "outer_deadline": outer, "test_and_test_guardian_deadline": test_end,
    "runner_cleanup_margin_seconds": 10, "new_budget_on_expiry": False,
    "model": "gpt-6-astra", "reasoning_effort": "max", "owner": identity(os.getpid()),
})
outcomes = []
for name, python in interpreters.items():
    if time.monotonic() >= test_end - 5:
        outcomes.append({"interpreter": name, "status": "NOT_RUN_ORIGINAL_BUDGET"})
        break
    attempt = Path("attempts") / (name + "-01")
    attempt.mkdir(mode=0o700)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONPROFILEIMPORTTIME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               NOMINAL_TEST_OUTPUT=str(attempt / "evidence"), NOMINAL_TEST_DEADLINE=str(test_end))
    command = ["nice", "-n", "19", python, "-B", "frozen/test_nominal_services.py"]
    save(attempt / "command.json", {"argv": command, "cwd": str(HERE),
                                   "test_deadline": test_end, "full_environment_recorded": False})
    for filename in ("nominal_services.py", "test_nominal_services.py"):
        data = (Path("frozen") / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == json.loads(
            Path("CODE-FREEZE.json").read_text())["sha256"][filename]
    with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        process_identity = identity(process.pid)
        save(attempt / "process.json", process_identity)
        try:
            code = process.wait(timeout=max(0, test_end - time.monotonic()))
        except subprocess.TimeoutExpired:
            known = {}
            for path in attempt.rglob("ownership.json"):
                try:
                    for row in json.loads(path.read_text())["known"]:
                        known[(row["pid"], row["start_ticks"])] = row
                except (ValueError, KeyError):
                    pass
            known[(process_identity["pid"], process_identity["start_ticks"])] = process_identity
            for pid, ticks in known:
                if time.monotonic() >= outer:
                    break
                if identity(pid) == {"pid": pid, "start_ticks": ticks}:
                    fd = os.pidfd_open(pid)
                    try:
                        if identity(pid) == {"pid": pid, "start_ticks": ticks}:
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    finally:
                        os.close(fd)
            process.wait(timeout=max(0, outer - time.monotonic()))
            for pid, ticks in known:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            save(attempt / "outer-timeout-cleanup.json", {
                "known": list(known.values()), "remaining": [
                    row for row in known.values()
                    if identity(row["pid"]) == {"pid": row["pid"], "start_ticks": row["start_ticks"]}],
                "deadline": outer, "checked_at": time.monotonic(),
            })
            code = 124
    (attempt / "exit-code.txt").write_text(str(code) + "\n")
    outcomes.append({"interpreter": name, "exit": code, "joined": process.poll() is not None})
    print(name, "exit", code, flush=True)
    print((attempt / "stdout.log").read_text(), flush=True)
    print((attempt / "stderr.log").read_text(), flush=True)
    if code == 124:
        break
save(Path("control-attempts.json"), {"attempts": outcomes, "outer_deadline": outer,
                                    "finished": time.monotonic(), "no_retries": True})
raise SystemExit(0 if len(outcomes) == 2 and all(r.get("exit") == 0 for r in outcomes) else 1)
