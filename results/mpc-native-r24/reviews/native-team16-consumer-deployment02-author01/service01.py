"""New private handoff; actual Gateway server retains this PID and session."""

import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition_common01 as c
from authorization02 import require_activation


def service_route(service):
    c.require(service in ("native", "gateway"), "owned_service_kind")
    directory = c.PUBLIC / service
    command = ["--config", str(c.CONFIG), "--plan", str(c.PLAN),
               "--python", str(c.PYTHON), "--evidence", str(directory)]
    if service == "gateway":
        return c.SOURCE13 / "launch_gateway_prefix_service.py", [
            "gateway", *command, "--identification-protocol", str(c.ORIGINAL_PROTOCOL)]
    return c.SOURCE13 / "launch_queue_service.py", command


def validate_exec(service, program, argv, environment):
    c.require(service in ("native", "gateway"), "owned_service_kind")
    if service == "gateway":
        expected = [str(c.PYTHON), "-m", "uvicorn", "gateway_prefix_bridge:application",
                    "--factory", "--host", "127.0.0.1", "--port", "8080", "--workers", "1"]
        c.require(program == str(c.PYTHON) and argv == expected, "source_exec_gateway_arguments")
    else:
        c.require(program == str(c.PYTHON.with_name("vllm"))
                  and argv[0:3] == [program, "serve", str(c.ALIAS)]
                  and environment["PYTHONPATH"].split(os.pathsep)[0] == str(c.SOURCE13)
                  and environment["PR33_R24_PREPARED_COPY_READY_ENABLED"] == "0"
                  and environment["PR33_R24_PREPARED_FORWARD_ENABLED"] == "1"
                  and environment["PR33_R24_PHASE_OBSERVER_ENABLED"] == "1"
                  and environment["PR33_R23_INPUT_POST_OBSERVER_ENABLED"] == "1",
                  "source_exec_native_route")
        for flag, expected in (("--max-num-seqs","32"),("--max-num-batched-tokens","2048")):
            c.require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
                      "Native_formation_limit_not_Gateway_admission")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--owned-service", action="store_true")
    parser.add_argument("--service", choices=("native", "gateway"))
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--owner-start-ticks", type=int)
    parser.add_argument("--mayor-preregistration", type=Path)
    parser.add_argument("--approval-sha256")
    args = parser.parse_args(argv)
    activation = require_activation(args.owned_service, args.mayor_preregistration, role="service",
                                    expected_sha256=args.approval_sha256)
    if activation is None:
        print('{"status":"disabled","native_execution":false,"actual_application_at":null}')
        return 0
    c.require(args.approval_sha256 is not None and args.service is not None
              and args.owner_pid is not None and args.owner_start_ticks is not None,
              "owned_service_arguments_required")
    c.remote_only()
    sys.pycache_prefix = str(c.STATE / "python-cache-evidence")
    c.owned_parent(args.owner_pid, args.owner_start_ticks)
    c.write(c.STATE / (args.service + "-consumer02-single-entry.json"), dict(pid=os.getpid()))
    c.verify_code()
    c.require(os.getsid(0) == os.getpid() == os.getpgrp(), "actual_owned_session_leader")
    c.private_config()
    c.require(os.environ["TMPDIR"] == str(c.STATE / "t"), "same_verified_TMPDIR")
    c.import_prerequisites()
    names = ("phase_observer", "input_post_observer", "queue_observer")
    before = {}
    for name in names:
        obj = importlib.import_module(name)
        expected = c.SOURCE13 / (name + ".py")
        c.require(Path(obj.__file__).resolve() == expected
                  and c.sha(expected) == c.sha(c.SOURCE / c.SOURCE13_REL / (name + ".py")),
                  "preexec_loaded_origin")
        before[name] = dict(path=str(expected), sha256=c.sha(expected))
    directory = c.PUBLIC / args.service
    c.write(directory / "preexec-origins.json", dict(pid=os.getpid(), modules=before,
            scope="Loaded in handoff, not proof of postexec hook registration"))
    entry, command = service_route(args.service)
    original_exec = os.execv

    def execute(program, argv):
        validate_exec(args.service, program, argv, os.environ)
        os.environ["PYTHONPATH"] += os.pathsep + str(c.HERE / "origin_guard")
        os.environ["R24_CUT_CONSUMER02_ORIGIN_POLICY"] = str(c.HERE / "ORIGIN_POLICY.json")
        os.environ["R24_CUT_CONSUMER02_ORIGIN_RECORDS"] = str(c.ORIGINS)
        c.write(directory / "direct-exec.json", dict(
            pid=os.getpid(), start_ticks=c.owner().snapshot(os.getpid())[0],
            executable=program, arguments=argv,
            source13_first=args.service != "native" or
                os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(c.SOURCE13),
            private_origin_guard_appended=True, spawned_redactor_process=False))
        original_exec(program, argv)

    os.execv = execute
    sys.argv = [str(entry), *command]
    # These are service exec launchers, not old spent probe/collector entrypoints.
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
