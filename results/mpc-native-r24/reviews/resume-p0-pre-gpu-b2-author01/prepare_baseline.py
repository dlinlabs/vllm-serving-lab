"""Retain the exact pre-edit sources and only the authorized gateway excerpt."""

import hashlib
import itertools
import json
import os
from pathlib import Path


ROOT = Path.cwd()
CAP = Path(".runtime/r24-pre-gpu-b2-role3")
os.umask(0o077)
originals = CAP / "originals"
originals.mkdir()
expected = {
    "native_services_smoke.py": "832c252ae0e7fc634a425acebd79edd796833bc93092e03bf1bf48190fc2b54c",
    "test_native_services_smoke.py": "05e2fcd7914aa037a7f58c332ffcb3ddf9327d075a292d6fdc5909a331cab3ec",
    "nominal_protocol.json": "08d145c8822a71e7acbb3cb0102bec29eda1de081c5af756140aa1fb3a9ee51a",
}
bindings = {}
for name, digest in expected.items():
    path = Path("experiments/mpc-research/resume-mvp") / name
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, str(path)
    (originals / name).write_bytes(data)
    bindings[str(path)] = dict(sha256=digest, bytes=len(data))
with Path("gateway.py").open("rb") as stream:
    excerpt = b"".join(itertools.islice(stream, 139, 205))
(originals / "gateway-lines140-205.txt").write_bytes(excerpt)
bindings["gateway.py:140-205"] = dict(sha256=hashlib.sha256(excerpt).hexdigest(),
                                      bytes=len(excerpt), full_file_hash=False)
record = dict(root=str(ROOT), bindings=bindings, before_public_edits=True,
              model_id="gpt-6-astra", reasoning_effort="max",
              plan_sha256=hashlib.sha256((CAP / "PLAN.json").read_bytes()).hexdigest())
(CAP / "BASELINE.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
for path in originals.iterdir():
    path.chmod(0o400)
print(json.dumps(record, sort_keys=True))
