"""Freeze the episode implementation, four controls and bounded peer dependencies."""

import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
SCOPE = Path(".runtime/r24-resume-p0-nominal-episode02")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
OWN = [PUBLIC / name for name in ("nominal_episode.py", "test_nominal_episode.py")]
EXTRA_HASH_INPUTS = {
    Path("trace_workload.py"): "98f6c818bc61d7f9e07392b6f79b77819b2b3748198e3f16dfbe710309eb8c3f",
    Path("experiments/mpc-research/native-r23/score_http.py"):
        "9a173ab99e668534dfe18df9be0b9e8447e06a1a673f63aebb16b9ef5217081d",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


assert ROOT == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
baseline = json.loads((SCOPE / "BASELINE_INPUT_PINS.json").read_text())
for name, row in baseline.items():
    assert sha(Path(name)) == row["sha256"], name
for path, expected in EXTRA_HASH_INPUTS.items():
    assert sha(path) == expected
budget = json.loads((SCOPE / "SOURCE_READ_BUDGET.json").read_text())
sources = {**budget["sources"], **{str(path): {"sha256": sha(path), "bytes": path.stat().st_size,
                                            "hash_only_no_content_view": True}
                                  for path in EXTRA_HASH_INPUTS}}
assert len(sources) == 6 and sum(row["bytes"] for row in sources.values()) == 200013
save(SCOPE / "SOURCE_READ_BUDGET_FINAL.json", {
    "sources": sources, "files": len(sources), "bytes": sum(row["bytes"] for row in sources.values()),
    "maximum_files": 6, "maximum_bytes": 384 * 1024,
    "additional_two": "Hash-only pins of direct recorder dependencies; no extra source content viewed.",
    "runtime_context_retained": True, "native_services_smoke_read": False,
})
for source in OWN:
    ast.parse(source.read_text())
tree = ast.parse(OWN[1].read_text())
controls = [node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
            and node.name.startswith("test_")]
assert len(controls) == 4, controls
copies = OWN + [PUBLIC / name for name in (
    "nominal_runtime.py", "nominal_client.py", "test_nominal_client.py", "nominal_metrics.py",
    "nominal_services.py", "nominal_planner.py", "nominal_scenario.py", "nominal_permit.py",
    "nominal_workload.py",
)]
paths = sorted(set([Path(name) for name in baseline] + copies + list(EXTRA_HASH_INPUTS) + [
    SCOPE / name for name in ("BASELINE_INPUT_PINS.json", "SOURCE_READ_BUDGET.json",
                              "SOURCE_READ_BUDGET_FINAL.json", "prepare02.py", "run02.py")
]))
pins = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths}
for source in copies:
    with (SCOPE / "sources" / source.name).open("xb") as stream:
        stream.write(source.read_bytes())
diff = "".join("".join(difflib.unified_diff(
    [], path.read_text().splitlines(keepends=True), fromfile="/dev/null", tofile=str(path),
)) for path in OWN)
with (SCOPE / "SOURCE_DIFF.patch").open("x") as stream:
    stream.write(diff)
save(SCOPE / "INPUT_PINS.json", pins)
save(SCOPE / "FREEZE.json", {
    "schema": "nominal-episode02-freeze-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
    "plan_sha256": sha(SCOPE / "PLAN.json"), "input_pins_sha256": sha(SCOPE / "INPUT_PINS.json"),
    "source_diff_sha256": sha(SCOPE / "SOURCE_DIFF.patch"),
    "source_ids": {str(path): pins[str(path)] for path in copies},
    "source_generations": 1, "attempts": 1, "selected_controls": controls,
    "peer_sources": len(sources), "peer_bytes": 200013,
    "preparation_identity": {
        "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
    },
})
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / PUBLIC))
import nominal_episode
save(SCOPE / "SOURCE_METADATA.json", {
    "python": sys.version, "interpreter": sys.executable,
    "episode_sources": nominal_episode._sources(),
    "module_paths": nominal_episode.runtime_api.module_paths(),
    "client_module": str(Path(nominal_episode.recorder.__file__).resolve()),
    "no_prior_or_forecast_generated": True, "no_capture_prepared": True, "not_an_approval": True,
})
for name, row in pins.items():
    assert sha(Path(name)) == row["sha256"], name
print(json.dumps({
    "freeze_sha256": sha(SCOPE / "FREEZE.json"), "source_copies": len(copies), "bindings": len(pins),
    "controls_executed": 0, "selected_future_controls": len(controls),
    "peer_source_inputs": len(sources), "peer_source_bytes": 200013,
}, sort_keys=True))
