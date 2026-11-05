"""Freeze only the typed pre-worker classification correction and one discrimination."""

import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode04")
OLD = Path(".runtime/r24-resume-p0-nominal-episode03")
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
needle = 'receipt["status"] = "refused" if not tasks and isinstance(error, EpisodeRefused) else "failed"'
replacement = ('receipt["status"] = (\n'
               '            "refused" if not tasks and isinstance(error, (EpisodeRefused, runtime_api.RuntimeRefused)) else "failed")')
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
assert set(new_methods) == set(old_methods)
tests = [name for name in new_methods if name.startswith("test_")]
assert set(tests) == set(plan["selected_methods"]) and len(tests) == 4
changed_test = "test_returned_failure_envelopes_cancel_peer_and_preserve_results"
for name in old_methods:
    if name != changed_test:
        assert dump(old_methods[name]) == dump(new_methods[name]), name
for text in after.values():
    ast.parse(text)
patch = "".join("".join(difflib.unified_diff(
    before[name].splitlines(keepends=True), after[name].splitlines(keepends=True),
    fromfile="episode03/" + str(PUBLIC / name), tofile="episode04/" + str(PUBLIC / name)))
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
    "implementation_change_exactly_typed_pre_worker_classification": True,
    "all_other_public_implementation_bytes_unchanged": True,
    "unchanged_test_methods": [name for name in tests if name != changed_test],
    "unchanged_fixture_helper_and_signal_check_AST": True,
    "new_subcase": "controller_exception_refused: same RuntimeRefused after shared start remains failure",
})
pins = {name: info(Path(name)) for name in baseline}
for name in ("PLAN.json", "start04.py", "prepare04.py", "run04.py", "BASELINE_INPUT_PINS.json",
             "STARTED.json", "SOURCE_READ_BUDGET.json", "SOURCE_METADATA.json", "SOURCE_DIFF.patch"):
    path = SCOPE / name
    pins[str(path)] = info(path)
save("INPUT_PINS.json", pins)
save("FREEZE.json", {
    "schema": "nominal-episode04-freeze-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
    "plan_sha256": sha(SCOPE / "PLAN.json"), "input_pins_sha256": sha(SCOPE / "INPUT_PINS.json"),
    "source_diff_sha256": sha(SCOPE / "SOURCE_DIFF.patch"), "source_ids": source_ids,
    "source_generations": 1, "attempts": 1, "selected_controls": tests,
    "new_discrimination_subcases": 1, "peer_sources": 6, "peer_bytes": 200013,
    "preparation_identity": {"pid": os.getpid(),
                             "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]},
})
print(json.dumps({
    "freeze_sha256": sha(SCOPE / "FREEZE.json"), "bindings": len(pins),
    "selected_methods": len(tests), "new_discrimination_subcases": 1,
    "test_attempts_executed": 0, "implementation_change_only_typed_pre_worker_classification": True,
    "source_ids": {str(PUBLIC / name): source_ids[str(PUBLIC / name)] for name in OWN},
}, sort_keys=True))
