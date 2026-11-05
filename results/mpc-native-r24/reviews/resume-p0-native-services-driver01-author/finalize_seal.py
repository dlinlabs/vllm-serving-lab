"""Seal the bounded driver delivery after closed verification logs, without reruns."""

import hashlib
import json
import os
from pathlib import Path


os.chdir(Path(__file__).resolve().parent)
assert not Path("SHA256SUMS").exists()
assert Path("verify01.exit-code.txt").read_text().strip() == "0"
paths = sorted(Path(".").rglob("*"))
assert not any(path.is_symlink() for path in paths)
files = [path for path in paths if path.is_file()]
manifest = "".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.as_posix() + "\n"
                   for path in files)
Path("SHA256SUMS").write_text(manifest)
seal = hashlib.sha256(Path("SHA256SUMS").read_bytes()).hexdigest()
for path in (*files, Path("SHA256SUMS")):
    path.chmod(path.stat().st_mode & 0o777 & ~0o222)
for path in reversed(paths):
    if path.is_dir():
        path.chmod(path.stat().st_mode & 0o777 & ~0o222)
Path(".").chmod(Path(".").stat().st_mode & 0o777 & ~0o222)
print(json.dumps(dict(indexed_files=len(files), total_files=len(files) + 1,
                      total_bytes=sum(p.stat().st_size for p in (*files, Path("SHA256SUMS"))),
                      sha256sums_sha256=seal), sort_keys=True))
