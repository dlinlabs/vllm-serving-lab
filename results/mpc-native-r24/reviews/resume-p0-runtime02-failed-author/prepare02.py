"""Freeze runtime02 changes and four selected controls without executing a forecast."""

import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
SCOPE = Path(".runtime/r24-resume-p0-nominal-runtime02")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
TRAIN = Path("results/mpc-native-r24/reviews/resume-p0-workload-author01/tapes-v1/train.tape.json")
EXPECTED = {
    PUBLIC / "nominal_permit.py": "05b7d0aebcede0851300ee70e9947e50ea6c94e6bd3ed2dc1283e10c5fcf6503",
    PUBLIC / "nominal_planner.py": "e492a2548f378dcf0511abc56bde4a6bc06bdc9c62854e9dd1053b5f95ec2ea1",
    PUBLIC / "nominal_scenario.py": "b15ccb25c4db2164a6bd44f3a4ec956a8e2aff1cfaa37040459948c4d7d477fc",
    PUBLIC / "nominal_workload.py": "614ae110e901110f875383b9ee0a66b08fe75e3a58a923b1d30dbd9c21d88bd4",
    TRAIN: "40bd755b441d6b38c0f2827cdb81afe50f637a579c471a2b1b4ef72b26c6192f",
    Path(".runtime/r24-resume-p0-nominal-runtime01/SHA256SUMS"): "367777a642f776e2f46df58513f447bb6de0cf11d0bb4dd72b36e0723d8fbdc5",
    Path(".runtime/r24-resume-p0-nominal-scenario01/SHA256SUMS"): "dee31c6b3792895fd74462b1efb2551628fbebbc0df928c151b563efb2cbd6ae",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


assert ROOT == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
for path, expected in EXPECTED.items():
    assert sha(path) == expected, str(path)
own = [PUBLIC / name for name in ("nominal_runtime.py", "test_nominal_runtime.py")]
copies = own + [PUBLIC / name for name in
                ("nominal_permit.py", "nominal_planner.py", "nominal_scenario.py", "nominal_workload.py")]
pins = sorted(set(
    copies + list(Path("mpc_lab").glob("*.py")) +
    [Path("gateway.py"), Path("telemetry.py"), Path("prefill_lab/live_http.py")] +
    list(EXPECTED) + [Path(".runtime/r24-resume-p0-nominal-runtime01/REPORT.json")] +
    [SCOPE / name for name in ("PLAN.json", "prepare02.py", "run02.py")] +
    [SCOPE / "baseline" / path.name for path in own]
))
bindings = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in pins}
for source in copies:
    with (SCOPE / "sources" / source.name).open("xb") as stream:
        stream.write(source.read_bytes())
tree = ast.parse((PUBLIC / "test_nominal_runtime.py").read_text())
controls = [node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_cli02_")]
assert len(controls) == 4, controls
diff = "".join(
    "".join(difflib.unified_diff(
        (SCOPE / "baseline" / path.name).read_text().splitlines(keepends=True),
        path.read_text().splitlines(keepends=True),
        fromfile="runtime01/" + path.name, tofile=str(path),
    )) for path in own
)
with (SCOPE / "SOURCE_DIFF.patch").open("x") as stream:
    stream.write(diff)
save(SCOPE / "INPUT_PINS.json", bindings)
save(SCOPE / "FREEZE.json", {
    "schema": "nominal-runtime02-freeze-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
    "source_ids": {str(path): bindings[str(path)] for path in copies},
    "input_pins_sha256": sha(SCOPE / "INPUT_PINS.json"), "plan_sha256": sha(SCOPE / "PLAN.json"),
    "source_diff_sha256": sha(SCOPE / "SOURCE_DIFF.patch"), "new_controls": controls,
    "selected_controls": 4, "old_controls_to_run": 0, "source_generations": 1,
    "preparation_identity": {
        "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
    },
})
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / PUBLIC))
from nominal_permit import source_digest
import nominal_runtime
save(SCOPE / "SOURCE_METADATA.json", {
    "source_digest_after_freeze": source_digest(), "python": sys.version,
    "interpreter": sys.executable, "module_paths": nominal_runtime.module_paths(),
    "no_prior_or_forecast_generated": True, "not_an_approval": True,
})
for name, row in bindings.items():
    assert sha(Path(name)) == row["sha256"], name
print(json.dumps({
    "freeze_sha256": sha(SCOPE / "FREEZE.json"), "bindings": len(bindings),
    "source_copies": len(copies), "controls_executed": 0, "selected_future_controls": 4,
}, sort_keys=True))
