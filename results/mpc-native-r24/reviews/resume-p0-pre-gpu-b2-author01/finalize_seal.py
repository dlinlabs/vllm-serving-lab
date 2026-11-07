"""Seal only this new B2 capsule after verification logs are closed."""

import hashlib
import json
from pathlib import Path


CAP = Path(".runtime/r24-pre-gpu-b2-role3")
assert not (CAP / "SHA256SUMS").exists()
assert (CAP / "verify01.exit-code.txt").read_text().strip() == "0"
paths = sorted(CAP.rglob("*"))
assert not any(path.is_symlink() for path in paths)
files = [path for path in paths if path.is_file()]
manifest = "".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  "
                   + path.relative_to(CAP).as_posix() + "\n" for path in files)
(CAP / "SHA256SUMS").write_text(manifest)
seal = hashlib.sha256((CAP / "SHA256SUMS").read_bytes()).hexdigest()
for path in (*files, CAP / "SHA256SUMS"):
    path.chmod(path.stat().st_mode & 0o777 & ~0o222)
for path in reversed(paths):
    if path.is_dir():
        path.chmod(path.stat().st_mode & 0o777 & ~0o222)
CAP.chmod(CAP.stat().st_mode & 0o777 & ~0o222)
print(json.dumps(dict(indexed_files=len(files), total_files=len(files) + 1,
                      total_bytes=sum(p.stat().st_size for p in (*files, CAP / "SHA256SUMS")),
                      sha256sums_sha256=seal), sort_keys=True))
