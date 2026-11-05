"""Freeze exactly the authorized comparison and same-coroutine test changes."""

import ast
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode03")
OLD = Path(".runtime/r24-resume-p0-nominal-episode02")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
OWN = ("nominal_episode.py", "test_nominal_episode.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def save(name, value):
    with (SCOPE / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def dump(node):
    return ast.dump(node, include_attributes=False)


assert Path.cwd() == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
baseline = json.loads((SCOPE / "BASELINE_INPUT_PINS.json").read_text())
plan = json.loads((SCOPE / "PLAN.json").read_text())
own_paths = {str(PUBLIC / name) for name in OWN}
assert all(sha(Path(name)) == row["sha256"] for name, row in baseline.items() if name not in own_paths)
before = {name: (SCOPE / "sources/before" / name).read_text() for name in OWN}
after = {name: (PUBLIC / name).read_text() for name in OWN}
needle = 'and runtime.protocol["scenario_binding"] == prepared.scenario_binding,'
replacement = ('and runtime_api.digest(runtime.protocol["scenario_binding"])\n'
               '             == runtime_api.digest(prepared.scenario_binding),')
assert before["nominal_episode.py"].count(needle) == 1
assert after["nominal_episode.py"] == before["nominal_episode.py"].replace(needle, replacement)
old_test, new_test = (ast.parse(text) for text in
                      (before["test_nominal_episode.py"], after["test_nominal_episode.py"]))
old_class = next(node for node in old_test.body if isinstance(node, ast.ClassDef)
                 and node.name == "NominalEpisodeTests")
new_class = next(node for node in new_test.body if isinstance(node, ast.ClassDef)
                 and node.name == "NominalEpisodeTests")
assert [dump(node) for node in old_test.body if node is not old_class] == [
    dump(node) for node in new_test.body if node is not new_class]
old_methods = {node.name: node for node in old_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
new_methods = {node.name: node for node in new_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
assert set(new_methods) - set(old_methods) == {"checked_episode"}
assert set(old_methods) - set(new_methods) == set()
tests = [name for name in new_methods if name.startswith("test_")]
assert set(tests) == set(plan["selected_methods"]) and len(tests) == 4
for name in ("write", "prepare", "gateway", "transport", "runtime"):
    assert dump(old_methods[name]) == dump(new_methods[name]), name
unchanged_bodies = []
for name in tests:
    if "default_off" in name:
        continue
    normalized = copy.deepcopy(new_methods[name])
    for node in ast.walk(normalized):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                and node.func.attr == "checked_episode"):
            node.func = ast.Attribute(value=ast.Name(id="episode", ctx=ast.Load()),
                                      attr="run_episode", ctx=ast.Load())
    assert dump(normalized) == dump(old_methods[name]), name
    unchanged_bodies.append(name)
for text in after.values():
    ast.parse(text)
patch = "".join("".join(difflib.unified_diff(
    before[name].splitlines(keepends=True), after[name].splitlines(keepends=True),
    fromfile="episode02/" + str(PUBLIC / name), tofile="episode03/" + str(PUBLIC / name)))
    for name in OWN)
with (SCOPE / "SOURCE_DIFF.patch").open("x") as stream:
    stream.write(patch)
previous_sources = json.loads((OLD / "FREEZE.json").read_text())["source_ids"]
source_ids = {name: info(Path(name)) for name in previous_sources}
for name in source_ids:
    shutil.copyfile(name, SCOPE / "sources" / Path(name).name)
sys.path.insert(0, str(PUBLIC.absolute()))
import nominal_episode as episode
save("SOURCE_METADATA.json", {
    "interpreter": sys.executable, "python": sys.version,
    "episode_sources": episode._sources(), "module_paths": episode.runtime_api.module_paths(),
    "episode_module": episode.__file__, "client_module": episode.recorder.__file__,
    "no_prior_or_capture_prepared": True, "no_tests_executed": True,
    "implementation_change_exactly_authorized_digest_expression": True,
    "unchanged_test_bodies_except_signal_wrapper": unchanged_bodies,
    "unchanged_fixture_and_helper_AST": True,
    "three_new_subcases": ["binding_changed", "provider_changed", "policy_changed"],
})
pins = {name: info(Path(name)) for name in baseline}
for name in ("PLAN.json", "start03.py", "prepare03.py", "run03.py", "BASELINE_INPUT_PINS.json",
             "STARTED.json", "SOURCE_READ_BUDGET.json", "SOURCE_METADATA.json", "SOURCE_DIFF.patch"):
    path = SCOPE / name
    pins[str(path)] = info(path)
save("INPUT_PINS.json", pins)
save("FREEZE.json", {
    "schema": "nominal-episode03-freeze-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
    "plan_sha256": sha(SCOPE / "PLAN.json"), "input_pins_sha256": sha(SCOPE / "INPUT_PINS.json"),
    "source_diff_sha256": sha(SCOPE / "SOURCE_DIFF.patch"), "source_ids": source_ids,
    "source_generations": 1, "attempts": 1, "selected_controls": tests,
    "new_discrimination_subcases": 3, "peer_sources": 6, "peer_bytes": 200013,
    "preparation_identity": {"pid": os.getpid(),
                             "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]},
})
print(json.dumps({
    "freeze_sha256": sha(SCOPE / "FREEZE.json"), "bindings": len(pins),
    "selected_methods": len(tests), "new_discrimination_subcases": 3,
    "test_attempts_executed": 0, "implementation_change_only_digest_comparison": True,
    "source_ids": {str(PUBLIC / name): source_ids[str(PUBLIC / name)] for name in OWN},
}, sort_keys=True))
