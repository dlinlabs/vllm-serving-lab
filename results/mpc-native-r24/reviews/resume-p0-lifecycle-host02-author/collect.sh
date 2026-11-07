#!/usr/bin/env bash
set -eu
export SSH_AUTH_SOCK=/home/daniel_lin/.ssh/pr32_gpu-agent.sock
timeout 90s ssh -T -p 18289 -i /home/daniel_lin/.ssh/pr32_gpu \
  -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=15 root@103.196.86.102 \
  '/workspace/pr32-r23/venv/bin/python3.12 -B -' <<'PY'
import hashlib, io, json, os, pathlib, sys, tarfile
root = pathlib.Path("/workspace/pr32-r23/r24-resume-p0-lifecycle-b70f70e102")
files = [root / n for n in ("driver.stdout.log", "driver.stderr.log", "driver.exit-code.txt")]
excluded = []
for current, dirs, names in os.walk(root / "output", followlinks=False):
    p = pathlib.Path(current)
    for name in list(dirs):
        q = p / name
        if q.is_symlink() or (p == root / "output" and name in ("cache", "scratch")):
            dirs.remove(name)
            excluded.append(str(q.relative_to(root)))
    for name in names:
        q = p / name
        if q.is_symlink():
            excluded.append(str(q.relative_to(root)))
        elif q.is_file():
            files.append(q)
records = []
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
    for p in sorted(files):
        assert p.is_file() and not p.is_symlink()
        data = p.read_bytes()
        name = p.relative_to(root).as_posix()
        info = tarfile.TarInfo(name)
        info.mode = 0o600
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    data = (json.dumps({"schema": "native-lifecycle-raw-collection-v1", "files": records,
                        "excluded_generated_cache_scratch_or_symlinks": excluded},
                       indent=2, sort_keys=True) + "\n").encode()
    info = tarfile.TarInfo("COLLECTION-MANIFEST.json")
    info.mode = 0o600
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
PY
