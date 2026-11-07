"""One bounded selected integration run; never discovers legacy tests or live mains."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import composition_common01 as c


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    c.require(output.is_relative_to(HERE / "attempts") and not output.exists(),
              "exclusive_control_output")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    def alarm(signum, frame):
        raise TimeoutError("CPU_control_run_120s_absolute")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 120)
    c.write(output / "invocation.json", dict(
        argv=sys.argv, version=sys.version, executable=sys.executable, started_at=started,
        source_freeze_sha256=c.sha(HERE / "CONTROL_FREEZE.json"),
        no_live_entrypoint_invoked=True, expected_focused_controls=9,
        PYTHONPATH_unset="PYTHONPATH" not in os.environ,
        dont_write_bytecode=sys.dont_write_bytecode,
        plugins_disabled=os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"))
    c.verify_code()
    import pytest
    code = int(pytest.main([
        "-q", "--tb=short", "-p", "no:cacheprovider", "--basetemp", str(output / "cases"),
        "--junitxml", str(output / "junit.xml"), str(HERE / "test_deployment02.py")]))
    c.verify_code()
    signal.setitimer(signal.ITIMER_REAL, 0)
    helpers = []
    for path in sorted(output.rglob("owned-cleanup.json")):
        row = c.load(path)
        for pid, ticks in row["owned_pid_start_ticks"].items():
            pid = int(pid)
            current = c.owner().snapshot(pid)
            gone = current is None or current[0] != ticks
            helpers.append(dict(pid=pid, start_ticks=ticks, gone=gone,
                                receipt=str(path.relative_to(output))))
    threads = [thread.name for thread in threading.enumerate() if thread is not threading.main_thread()]
    c.write(output / "finished.json", dict(
        exit_code=code, finished_at=time.monotonic(), elapsed_s=time.monotonic() - started,
        helper_identities=helpers, independently_gone=all(row["gone"] for row in helpers),
        non_main_threads_remaining=threads, Native_execution=False, actual_application_at=None))
    c.require(all(row["gone"] for row in helpers) and not threads, "CPU_helpers_not_joined")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
