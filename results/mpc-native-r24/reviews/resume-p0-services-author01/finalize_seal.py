"""Seal this delivery only after verification and its logs have closed."""

import hashlib
import json
import os
from pathlib import Path


os.chdir(Path(__file__).resolve().parent)
assert not Path("SHA256SUMS").exists()
assert Path("verify01.exit-code.txt").read_text().strip() == "0"
delivery = json.loads(Path("DELIVERY.json").read_text())
assert delivery["status"] == "CPU_OWNERSHIP_CONTROLS_PASS_ONLY_PENDING_MAYOR_REVIEW"
paths = sorted(Path(".").rglob("*"))
assert not any(path.is_symlink() for path in paths)
files = [path for path in paths if path.is_file()]
manifest = "".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.as_posix() + "\n"
                   for path in files)
with Path("SHA256SUMS").open("x") as stream:
    stream.write(manifest)
seal = hashlib.sha256(Path("SHA256SUMS").read_bytes()).hexdigest()
for path in (*files, Path("SHA256SUMS")):
    path.chmod(path.stat().st_mode & 0o777 & ~0o222)
for path in reversed(paths):
    if path.is_dir():
        path.chmod(path.stat().st_mode & 0o777 & ~0o222)
Path(".").chmod(Path(".").stat().st_mode & 0o777 & ~0o222)
print(json.dumps({"indexed_files": len(files), "total_files_including_manifest": len(files) + 1,
                  "sha256sums_sha256": seal,
                  "total_bytes_including_manifest": sum(p.stat().st_size for p in (*files, Path("SHA256SUMS")))},
                 sort_keys=True))
