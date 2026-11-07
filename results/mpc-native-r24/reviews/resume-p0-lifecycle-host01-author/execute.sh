#!/usr/bin/env bash
set -eu
export SSH_AUTH_SOCK=/home/daniel_lin/.ssh/pr32_gpu-agent.sock
timeout 530s ssh -T -p 18289 -i /home/daniel_lin/.ssh/pr32_gpu \
  -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=15 -o ConnectionAttempts=1 root@103.196.86.102 'bash -s' <<'REMOTE'
set -eu
umask 077
cd /workspace/pr32-r23/r24-resume-p0-lifecycle-6113b37701
/workspace/pr32-r23/venv/bin/python3.12 -B - <<'PY'
import hashlib, json, pathlib, tarfile
p = pathlib.Path("source.tar")
assert hashlib.sha256(p.read_bytes()).hexdigest() == "bf007372fe35f8a0017aa0787d56046ad1149c66f8b53fa5686376372a1f71fe"
assert not pathlib.Path("source").exists()
with tarfile.open(p) as t:
    members = t.getmembers()
    assert len(members) == 48
    assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute()
               and ".." not in pathlib.PurePosixPath(m.name).parts for m in members)
    t.extractall("source", filter="data")
print(json.dumps({"stage": "source_verified", "files": 48}), flush=True)
PY
test ! -e driver.exit-code.txt
set +e
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/pr32-r23/venv/bin/python3.12 -B \
  source/experiments/mpc-research/resume-mvp/native_services_smoke.py \
  --execute --source-root /workspace/pr32-r23/r24-resume-p0-lifecycle-6113b37701/source \
  --output-root /workspace/pr32-r23/r24-resume-p0-lifecycle-6113b37701/output \
  --seconds 480 --cleanup-reserve 30 --term-grace 10 \
  >driver.stdout.log 2>driver.stderr.log
code=$?
set -e
printf '%s\n' "$code" >driver.exit-code.txt
/workspace/pr32-r23/venv/bin/python3.12 -B - <<'PY'
import json, pathlib
p = pathlib.Path("output/report.json")
r = json.loads(p.read_text()) if p.is_file() else {}
print(json.dumps({"driver_exit": pathlib.Path("driver.exit-code.txt").read_text().strip(),
                  "report_exists": p.is_file(), "lifecycle_passed": r.get("lifecycle_passed"),
                  "errors": r.get("errors"), "known_closure_checked": r.get("known_closure_checked"),
                  "offered_requests_from_driver": r.get("offered_requests_from_driver")}), flush=True)
PY
REMOTE
