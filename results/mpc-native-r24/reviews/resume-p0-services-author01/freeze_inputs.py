"""Freeze generation one before executing either interpreter's CPU controls."""

import datetime
import hashlib
import json
import os
from pathlib import Path


os.chdir(Path(__file__).resolve().parent)
os.umask(0o077)
assert not Path("CODE-FREEZE.json").exists()
Path("frozen").mkdir(mode=0o700)
source = Path("../../experiments/mpc-research/resume-mvp")
bindings = {}
for name in ("nominal_services.py", "test_nominal_services.py"):
    data = (source / name).read_bytes()
    with (Path("frozen") / name).open("xb") as stream:
        stream.write(data)
    bindings[name] = hashlib.sha256(data).hexdigest()
    (Path("frozen") / name).chmod(0o400)
specs = {}
for name in ("PLAN.json", "COMMANDS.txt", "run_controls.py", "freeze_inputs.py"):
    specs[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
record = {
    "schema": "role3-resume-p0-nominal-services-generation1",
    "frozen_before_controls": True,
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_id": "gpt-6-astra", "reasoning_effort": "max",
    "sha256": bindings, "spec_sha256": specs,
    "source_directory": str(source.resolve()),
    "generations": 1, "attempts_per_interpreter": 1, "focused_cases_per_interpreter": 6,
    "actual_model_native_GPU_network_authority": False,
    "absolute_budget_seconds": 110, "pre_reserved_runner_cleanup_seconds": 10,
    "public_and_frozen_must_remain_byte_identical": True,
}
with Path("CODE-FREEZE.json").open("x") as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write("\n")
Path("CODE-FREEZE.json").chmod(0o400)
print(json.dumps(record, sort_keys=True))
