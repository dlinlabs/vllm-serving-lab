"""Freeze, bound and seal the single F1/F2 correction; no service or model execution."""

import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
PUBLIC = Path("experiments/mpc-research/resume-mvp")
SCOPE = Path(".runtime/r24-pre-gpu-b3-f1-role6")
REVIEW = Path(".runtime/r24-pre-gpu-final-review-role4")
OWN = ("native_nominal_qualification.py", "test_native_nominal_qualification.py")
METHOD = "test_drain_tolerates_only_idle_cost_residue"
OLD = "        if all(metrics[name] == 0 for name in DRAIN_FIELDS):"
NEW = ('        if (metrics["current_in_flight"] == 0 and metrics["current_waiting"] == 0\n'
       '                and metrics["current_admitted_cost"] <= 1e-9):')
PYTHON = "/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/r24-mpc-py312-venv/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(pid):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": parts[19], "state": parts[0]}
    except FileNotFoundError:
        return None


def review_members():
    plan = load(SCOPE / "PLAN.json")
    assert sha(REVIEW / "SHA256SUMS") == plan["role4_manifest_sha256"]
    result = []
    for line in (REVIEW / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert not (REVIEW / path).is_symlink() and sha(REVIEW / path) == expected
        result.append((path, expected))
    assert len(result) == plan["role4_manifest_members"]
    return result


assert Path.cwd() == ROOT
mode = sys.argv[1]
if mode == "baseline":
    plan = load(SCOPE / "PLAN.json")
    members = review_members()
    copied = SCOPE / "prior-role4"
    copied.mkdir()
    for path, expected in members:
        target = copied / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REVIEW / path, target)
        assert sha(target) == expected
    shutil.copyfile(REVIEW / "SHA256SUMS", copied / "SHA256SUMS")
    report = load(copied / "REPORT.json")
    pins = report["pins"]["reviewed_consumers"] | report["pins"]["unchanged_dependencies_hash_verified"]
    assert pins[str(PUBLIC / OWN[0])] == plan["before_caller_sha256"]
    assert pins[str(PUBLIC / OWN[1])] == plan["before_test_sha256"]
    assert all(sha(Path(path)) == expected for path, expected in pins.items())
    (SCOPE / "sources/before").mkdir(parents=True)
    for name in OWN:
        shutil.copyfile(PUBLIC / name, SCOPE / "sources/before" / name)
    save(SCOPE / "BASELINE.json", {
        "source_pins": pins, "plan_sha256": sha(SCOPE / "PLAN.json"),
        "role4_seal": sha(REVIEW / "SHA256SUMS"), "role4_members_copied": len(members),
        "baseline_identity": identity(os.getpid()), "python": sys.version,
        "controls_executed": 0,
    })
    save(SCOPE / "FINDINGS.json", {
        "source": "prior-role4/REPORT.json", "source_sha256": sha(copied / "REPORT.json"),
        "findings": report["findings"], "checklist_errata": report["checklist_errata"],
        "interpretation": "Original F1 CONFIRMED on both Pythons; F2 originally static-only. Evidence copied, not rerun or altered.",
    })
    print(json.dumps({"baseline": True, "role4_sealed_members": len(members), "source_pins": len(pins)}))
elif mode == "freeze":
    baseline = load(SCOPE / "BASELINE.json")
    before = {name: (SCOPE / "sources/before" / name).read_text() for name in OWN}
    after = {name: (PUBLIC / name).read_text() for name in OWN}
    assert before[OWN[0]].count(OLD) == 1
    assert after[OWN[0]] == before[OWN[0]].replace(OLD, NEW)
    before_tree, after_tree = (ast.parse(value) for value in (before[OWN[1]], after[OWN[1]]))
    before_class = next(node for node in before_tree.body if isinstance(node, ast.ClassDef) and node.name == "Controls")
    after_class = next(node for node in after_tree.body if isinstance(node, ast.ClassDef) and node.name == "Controls")
    new_methods = [node for node in after_class.body if isinstance(node, ast.FunctionDef) and node.name == METHOD]
    assert len(new_methods) == 1
    after_class.body.remove(new_methods[0])
    assert ast.dump(before_class) == ast.dump(after_class)
    for tree in (before_tree, after_tree):
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "ROOT" for target in node.targets))]
    assert ast.dump(before_tree) == ast.dump(after_tree)
    pins = {path: sha(Path(path)) for path in baseline["source_pins"]}
    for path, expected in baseline["source_pins"].items():
        if Path(path).name not in OWN:
            assert pins[path] == expected
    (SCOPE / "sources/candidate").mkdir()
    changes = []
    for name in OWN:
        shutil.copyfile(PUBLIC / name, SCOPE / "sources/candidate" / name)
        changes.extend(difflib.unified_diff(before[name].splitlines(keepends=True),
                                           after[name].splitlines(keepends=True),
                                           fromfile="before/" + name, tofile="candidate/" + name))
    with (SCOPE / "correction.diff").open("x") as stream:
        stream.writelines(changes)
    save(SCOPE / "FREEZE.json", {
        "source_pins": pins, "plan_sha256": sha(SCOPE / "PLAN.json"),
        "capsule_sha256": sha(Path(__file__)), "worker_sha256": sha(SCOPE / "focused_worker.py"),
        "source_diff": "Exactly one drain predicate replacement; no other caller bytes changed.",
        "test_diff": "Only ROOT assignments and one new focused method; all other test AST unchanged.",
        "freeze_identity": identity(os.getpid()), "at_monotonic": time.monotonic(),
    })
    print(json.dumps({"frozen": True, "caller_sha256": pins[str(PUBLIC / OWN[0])],
                      "test_sha256": pins[str(PUBLIC / OWN[1])]}))
elif mode == "run":
    frozen = load(SCOPE / "FREEZE.json")
    assert sha(Path(__file__)) == frozen["capsule_sha256"]
    assert sha(SCOPE / "focused_worker.py") == frozen["worker_sha256"]
    assert all(sha(Path(path)) == expected for path, expected in frozen["source_pins"].items())
    attempt = SCOPE / "attempt"
    attempt.mkdir()
    (attempt / "outputs").mkdir()
    save(attempt / "pins-before.json", frozen["source_pins"])
    command = [
        "env", f"PYTHONPATH={ROOT / PUBLIC}:{ROOT}", "PYTHONDONTWRITEBYTECODE=1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "PYTHONHASHSEED=0", "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1",
        f"PYTHONPYCACHEPREFIX={ROOT / attempt / 'unused-pycache'}",
        f"NATIVE_SMOKE_TEST_OUTPUT={attempt / 'outputs/native-smoke'}",
        f"NATIVE_SMOKE_PROTOCOL={ROOT / PUBLIC / 'nominal_protocol.json'}",
        "nice", "-n19", PYTHON, "-B", str(SCOPE / "focused_worker.py"),
    ]
    save(attempt / "command.json", {"argv": command, "cwd": str(ROOT),
                                  "runner_identity": identity(os.getpid())})
    started = time.monotonic()
    timed_out = False
    with (attempt / "stdout.log").open("xb") as out, (attempt / "stderr.log").open("xb") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        original = identity(child.pid)
        save(attempt / "process-start.json", original)
        try:
            child.wait(timeout=55)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=max(0.01, started + 60 - time.monotonic()))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=max(0.01, started + 60 - time.monotonic()))
    current = identity(child.pid)
    result = {
        "returncode": child.returncode, "elapsed_s": time.monotonic() - started,
        "timed_out": timed_out, "joined": child.returncode is not None,
        "original_identity": original, "identity_after_wait": current,
        "original_identity_gone": current is None or current["start_ticks"] != original["start_ticks"],
        "python": sys.version, "native_qualification": False,
    }
    save(attempt / "result.json", result)
    after = {path: sha(Path(path)) for path in frozen["source_pins"]}
    save(attempt / "pins-after.json", after)
    assert after == frozen["source_pins"] and result["original_identity_gone"] and result["elapsed_s"] <= 60
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(child.returncode or int(timed_out))
else:
    raise ValueError("use baseline, freeze or run")
