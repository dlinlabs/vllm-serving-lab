"""Freeze the single corrected caller generation without importing its implementation."""

import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
SCOPE = Path(".runtime/r24-pre-gpu-b3-role6")
OWN = ("native_nominal_qualification.py", "test_native_nominal_qualification.py")
DATA = (
    "results/mpc-native-r24/reviews/resume-p0-probes02/inputs/warmup.json",
    "results/mpc-native-r24/reviews/resume-p0-probes02/inputs/admission-boundary.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert Path.cwd() == ROOT
baseline = json.loads((SCOPE / "BASELINE.json").read_text())
assert sha(SCOPE / "PLAN.json") == baseline["plan_sha256"]
pins = {name: sha(Path(name)) for name in baseline["pins"]}
for name, record in baseline["pins"].items():
    if Path(name).name not in OWN:
        assert pins[name] == record["sha256"], name
trees = {name: ast.parse((PUBLIC / name).read_text(), filename=str(PUBLIC / name)) for name in OWN}
tests = [
    node.name for node in ast.walk(trees[OWN[1]])
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
]
assert len(tests) == 13
source_dir = SCOPE / "sources/candidate"
source_dir.mkdir()
diffs = []
for name in OWN:
    shutil.copyfile(PUBLIC / name, source_dir / name)
    diffs.extend(difflib.unified_diff(
        (SCOPE / "sources/before" / name).read_text().splitlines(keepends=True),
        (PUBLIC / name).read_text().splitlines(keepends=True),
        fromfile="before/" + name, tofile="candidate/" + name))
with (SCOPE / "caller-changes.diff").open("x") as stream:
    stream.writelines(diffs)
record = {
    "schema": "B3-corrected-source-freeze-v1", "root": str(ROOT),
    "frozen_at_monotonic": time.monotonic(), "python": sys.version,
    "freezer_identity": {
        "pid": os.getpid(),
        "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
    },
    "plan_sha256": sha(SCOPE / "PLAN.json"), "source_pins": pins,
    "data_pins": {name: sha(Path(name)) for name in DATA},
    "control_sha256": sha(SCOPE / "control.py"), "freeze_sha256": sha(Path(__file__)),
    "syntax": "AST parsing only; no caller/test imports or corrected tests executed",
    "test_methods": tests, "method_count": len(tests),
    "generation": "one corrected targeted Python3.12 attempt, <=60s; no retry",
    "mock_scope": "Real caller, codec, runtime construction/protocol and MockTransport GETs; explicit warmup/episode execution boundary fixtures, no real replay or runtime.run.",
    "not_asserted": [
        "No Native qualification, inference, apply, admission effect or actual worker-cycle evidence.",
        "No real service lifecycle/full-wall or GPU/resource ownership evidence.",
        "Role3 concurrently owns native_services_smoke.py; its source is neither frozen nor independently accepted by this capsule.",
        "Caller SHA is external runtime-binding provenance, not an alteration to accepted immutable episode inputs or permit source-digest ABI.",
    ],
}
with (SCOPE / "GREEN-FREEZE.json").open("x") as stream:
    json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"frozen": True, "source_pins": len(pins), "data_pins": len(DATA),
                  "method_count": len(tests), "corrected_tests_executed": 0,
                  "caller_sha256": pins[str(PUBLIC / OWN[0])],
                  "test_sha256": pins[str(PUBLIC / OWN[1])]}, sort_keys=True))
