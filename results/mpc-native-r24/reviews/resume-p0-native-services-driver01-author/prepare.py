"""Freeze one CPU-only mocked-control generation; no probe/module execution."""

import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path


os.chdir(Path(__file__).resolve().parent)
os.umask(0o077)
assert not Path("CODE-FREEZE.json").exists()
Path("frozen").mkdir()
Path("inputs").mkdir()
root = Path("../..")
public = root / "experiments/mpc-research/resume-mvp"
bindings = {}
for name in ("native_services_smoke.py", "test_native_services_smoke.py"):
    data = (public / name).read_bytes()
    (Path("frozen") / name).write_bytes(data)
    bindings[name] = hashlib.sha256(data).hexdigest()
inputs = {}
for path in (public / "nominal_services.py", public / "nominal_protocol.json", root / "gateway.py"):
    data = path.read_bytes()
    destination = Path("inputs") / path.name
    destination.write_bytes(data)
    inputs[str(path.resolve())] = dict(copy=str(destination), sha256=hashlib.sha256(data).hexdigest(),
                                      bytes=len(data))
with (root / "mpc_lab/model.py").open("rb") as stream:
    excerpt = b"".join(itertools.islice(stream, 160))
Path("inputs/model-lines1-160.txt").write_bytes(excerpt)
inputs[str((root / "mpc_lab/model.py").resolve())] = dict(
    copy="inputs/model-lines1-160.txt", sha256=hashlib.sha256(excerpt).hexdigest(),
    bytes=len(excerpt), range_inclusive=[1, 160], full_source_hash_claimed=False)
assert sum(row["bytes"] for row in inputs.values()) <= 131072
assert hashlib.sha256(Path("inputs/nominal_services.py").read_bytes()).hexdigest() == (
    "5249221b9bd795c8265039502746fe30a394e774a52040f9fcf6261146919bed")
specs = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
         for name in ("PLAN.json", "COMMANDS.txt", "prepare.py", "run_controls.py")}
record = dict(schema="native-services-smoke-freeze-v1", utc=datetime.datetime.now(
    datetime.timezone.utc).isoformat(), model_id="gpt-6-astra", reasoning_effort="max",
    new_sources=bindings, canonical_sources=inputs, specifications=specs,
    generation=1, attempts_per_interpreter=1, aggregate_original_deadline_seconds=60,
    controls_per_interpreter=5, actual_service_HTTP_GPU_execution=False,
    earlier_recursive_search_scope_deviation_retained=True,
    code_execution_not_begun=True)
Path("CODE-FREEZE.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
for path in (*Path("frozen").iterdir(), *Path("inputs").iterdir(), Path("CODE-FREEZE.json")):
    path.chmod(0o400)
print(json.dumps(record, sort_keys=True))
