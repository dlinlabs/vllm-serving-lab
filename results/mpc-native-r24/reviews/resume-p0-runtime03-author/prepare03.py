"""Freeze one strict-root correction and 17 selected CPU controls; no execution."""

import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
SCOPE = Path(".runtime/r24-resume-p0-nominal-runtime03")
OLD = Path(".runtime/r24-resume-p0-nominal-runtime02")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
OWN = [PUBLIC / name for name in ("nominal_runtime.py", "test_nominal_runtime.py")]
COPIES = OWN + [PUBLIC / name for name in
                ("nominal_permit.py", "nominal_planner.py", "nominal_scenario.py", "nominal_workload.py")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


assert ROOT == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
baseline = json.loads((SCOPE / "BASELINE_INPUT_PINS.json").read_text())
assert sha(SCOPE / "PLAN.json") == baseline["plan_sha256"]
for name, row in baseline["unchanged_pre_edit_bindings"].items():
    if Path(name) not in OWN:
        assert sha(Path(name)) == row["sha256"], name
for source in OWN:
    assert sha(SCOPE / "baseline" / source.name) == baseline["unchanged_pre_edit_bindings"][str(source)]["sha256"]
assert sha(OLD / "SHA256SUMS") == baseline["old_seal_sha256"]
for line in (OLD / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert sha(OLD / name) == expected, name

trees = {path.name: ast.parse(path.read_text()) for path in OWN}
old_tests = ast.parse((SCOPE / "baseline/test_nominal_runtime.py").read_text())


def controls(tree):
    return {f"{cls.name}.{method.name}": method for cls in tree.body if isinstance(cls, ast.ClassDef)
            for method in cls.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name.startswith("test_")}


before = controls(old_tests)
after = controls(trees["test_nominal_runtime.py"])
added = set(after) - set(before)
assert len(before) == 16 and len(after) == 17
assert added == {"NominalRuntimeTests.test_root03_canonicalization_preserves_strict_endpoint_binding"}
assert all(ast.dump(before[name]) == ast.dump(after[name]) for name in before)
assert sum(".test_cli02_" in name for name in after) == 4
diff = "".join(
    "".join(difflib.unified_diff(
        (SCOPE / "baseline" / path.name).read_text().splitlines(keepends=True),
        path.read_text().splitlines(keepends=True),
        fromfile="runtime02/" + path.name, tofile=str(path),
    )) for path in OWN
)
with (SCOPE / "SOURCE_DIFF.patch").open("x") as stream:
    stream.write(diff)
paths = sorted(set(
    [Path(name) for name in baseline["unchanged_pre_edit_bindings"]] + COPIES +
    [OLD / name for name in ("SHA256SUMS", "FREEZE.json", "REPORT.json", "FAILURES.json")] +
    [SCOPE / name for name in ("PLAN.json", "BASELINE_INPUT_PINS.json", "prepare03.py", "run03.py")] +
    [SCOPE / "baseline" / path.name for path in OWN]
))
pins = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths}
for source in COPIES:
    with (SCOPE / "sources" / source.name).open("xb") as stream:
        stream.write(source.read_bytes())
save(SCOPE / "INPUT_PINS.json", pins)
save(SCOPE / "FREEZE.json", {
    "schema": "nominal-runtime03-freeze-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
    "source_ids": {str(path): pins[str(path)] for path in COPIES},
    "input_pins_sha256": sha(SCOPE / "INPUT_PINS.json"), "plan_sha256": sha(SCOPE / "PLAN.json"),
    "source_diff_sha256": sha(SCOPE / "SOURCE_DIFF.patch"), "selected_controls": sorted(after),
    "unchanged_old_control_bodies": sorted(before), "new_control": sorted(added),
    "source_generations": 1, "attempts": 1, "new_canonicalization_controls": 1,
    "unchanged_cli02_controls": 4, "old_runtime_controls": 12,
    "immutable_predecessor_seal": sha(OLD / "SHA256SUMS"),
    "preparation_identity": {
        "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
    },
})
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / PUBLIC))
import httpx
import nominal_runtime
from nominal_permit import source_digest
save(SCOPE / "SOURCE_METADATA.json", {
    "source_digest_after_freeze": source_digest(), "python": sys.version, "interpreter": sys.executable,
    "httpx_version": httpx.__version__, "httpx_module": str(Path(httpx.__file__).resolve()),
    "module_paths": nominal_runtime.module_paths(),
    "no_prior_or_forecast_generated": True, "not_an_approval": True,
})
for name, row in pins.items():
    assert sha(Path(name)) == row["sha256"], name
print(json.dumps({
    "freeze_sha256": sha(SCOPE / "FREEZE.json"), "bindings": len(pins),
    "source_copies": len(COPIES), "controls_executed": 0, "selected_future_controls": len(after),
    "unchanged_prior_control_bodies": len(before),
}, sort_keys=True))
