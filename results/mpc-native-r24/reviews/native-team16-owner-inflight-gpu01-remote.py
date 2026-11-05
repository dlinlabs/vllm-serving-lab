"""One prospective inflight component; retain Mayor lease through safe export."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time


BASE = Path("/workspace/pr32-r23")
STAGE = BASE / "r24-team16-native-owner-inflight-adapter01"
PUBLIC = BASE / "source/results/mpc-native-r24/r24-team16-native-owner-inflight-adapter01"
OUTPUT = BASE / "owner-inflight-gpu01-mayor"
AUTHORITY = BASE / "mayor-owner-inflight01-preregistration.json"
PLAN = BASE / "native-team16-owner-inflight-gpu01-preregistration.json"
SEAL = "08331c167a233cd164c3104cd1f18b50a82fc8ef0ae17eb13c05b6a0a9247e14"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--launch-on-authorized-existing-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.launch_on_authorized_existing_gpu:
        print('{"status":"disabled","Native_execution":false}')
        return 0
    if Path(sys.prefix).resolve() != BASE / "venv" or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("exact_remote_isolated_interpreter_required")
    plan = json.loads(PLAN.read_bytes())
    authority = json.loads(AUTHORITY.read_bytes())
    if (sha(Path(__file__)) != plan["frozen"]["remote_script_sha256"]
            or sha(STAGE / "SHA256SUMS") != SEAL
            or sha(STAGE / "PROTOCOL.json") != plan["frozen"]["protocol_sha256"]
            or sha(STAGE / "OFFERS.json") != plan["frozen"]["offers_sha256"]
            or sha(STAGE / "NATIVE_PLAN.revision01.json") != plan["frozen"]["native_plan_sha256"]
            or authority["adapter_manifest_sha256"] != SEAL):
        raise ValueError("preregistered_source_binding")
    os.umask(0o077)
    fd = os.open(BASE / "r24-team16-mayor-gpu.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1):
            raise ValueError("Mayor_lease_identity")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUTPUT.mkdir(mode=0o700)
        started = time.monotonic()
        command = [str(BASE / "venv/bin/python"), "-I", "-B", str(STAGE / "launch01.py"),
                   "--execute-on-existing-gpu", "--mayor-preregistration", str(AUTHORITY)]
        with (OUTPUT / "launcher.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=STAGE, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(fd,))
            fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            write(OUTPUT / "mayor-start.json", dict(
                command=command, pid=process.pid, start_ticks=fields[19],
                started_at=started, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                preregistration_commit=authority["preregistration_commit"],
                Mayor_lease_inherited_by_launcher=True, Native_budget_s=260))
            try:
                status = process.wait(timeout=285)
            except subprocess.TimeoutExpired:
                write(OUTPUT / "reconcile-only.json", dict(
                    pid=process.pid, start_ticks=fields[19], namespace_spent=True,
                    retry_permitted=False, reason="Launcher exceeded its own260s budget; read-only reconcile",
                    launcher_retains_inherited_Mayor_lease=True))
                print('{"status":"RECONCILE_ONLY_NO_RETRY","safe_export":false}', file=sys.stderr)
                return 124
        write(OUTPUT / "launcher-exit.json", dict(returncode=status, finished_at=time.monotonic()))
        gate = json.loads((PUBLIC / "export-gate.json").read_bytes())
        if (PUBLIC / "DO_NOT_EXPORT.json").exists() or gate.get("safe_to_export") is not True:
            print('{"status":"QUARANTINED_NO_EXPORT","retry_permitted":false}', file=sys.stderr)
            return 125
        files = []
        for path in sorted(PUBLIC.rglob("*")):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError("export_regular_files_only")
            if path.is_file():
                files.append(path)
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
            for path in files:
                archive.add(path, arcname=str(Path("native-output") / path.relative_to(PUBLIC)),
                            recursive=False)
            for name in ("mayor-start.json", "launcher-exit.json"):
                archive.add(OUTPUT / name, arcname="mayor/" + name, recursive=False)
        print(json.dumps(dict(status="SAFE_EVIDENCE_EXPORTED", launcher_returncode=status,
                              component_pass=gate["component_pass"], retries=0)), file=sys.stderr)
        return 0 if gate["component_pass"] and status == 0 else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
