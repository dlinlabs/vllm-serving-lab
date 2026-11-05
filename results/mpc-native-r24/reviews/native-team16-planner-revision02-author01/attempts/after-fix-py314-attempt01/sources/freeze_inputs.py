"""Verify the frozen critique and prior delivery; copy only into revision02."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / ".runtime/r24-team16-planner-revision01"
CRITIQUE = ROOT / ".runtime/r24-team16-comparison/planner16-critique"
PARENT_MAP = ROOT / (
    "results/mpc-native-r24/reviews/native-team16-numerical-crossreview-parent-seal01.json")
COPIED = (
    "policy_planner16.py", "test_policy_planner16.py",
    "test_policy_planner16_numerical_revision.py", "test_parent_numerical_review.py",
    "test_predecessor_witnesses.py", "run_attempt.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_manifest(path, base):
    paths = []
    for line in path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        target = base / relative
        if not target.resolve().is_relative_to(ROOT) or sha(target) != expected:
            raise ValueError("input_pin_mismatch:" + relative)
        paths.append(target)
    return paths


def copy_exclusive(source, destination):
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run_from_repository_root")
    destination = HERE.relative_to(ROOT)
    if (destination / "INPUT_SHA256SUMS").exists():
        raise SystemExit("refuse_to_overwrite_frozen_revision02_inputs")
    plan = json.loads((HERE / "PLAN.json").read_text())
    if (
        sha(OLD / "SHA256SUMS") != plan["predecessor_seal_sha256"]
        or sha(OLD / "policy_planner16.py") != plan["predecessor_planner_sha256"]
        or sha(CRITIQUE / "SHA256SUMS") != plan["critique_seal_sha256"]
    ):
        raise ValueError("unexpected_predecessor_or_critique")
    original = checked_manifest(OLD / "INPUT_SHA256SUMS", ROOT)
    sealed = checked_manifest(OLD / "SHA256SUMS", OLD)
    critique = checked_manifest(CRITIQUE / "SHA256SUMS", CRITIQUE)
    proofs = []
    for row in json.loads(PARENT_MAP.read_text())["copies"]:
        origin = row["original"]
        if not (
            origin.startswith(str(ROOT / ".runtime/r24-team16-planner"))
            or origin.startswith(str(CRITIQUE))
        ):
            continue
        path = ROOT / row["path"]
        if not path.resolve().is_relative_to(ROOT) or sha(path) != row["sha256"]:
            raise ValueError("parent_planner_proof_mismatch:" + row["path"])
        proofs.append(path)
    paths = sorted(set(
        original + sealed + critique + proofs
        + [OLD / "SHA256SUMS", CRITIQUE / "SHA256SUMS", PARENT_MAP, HERE / "PLAN.json"]))
    with (destination / "INPUT_SHA256SUMS").open("x") as stream:
        for path in paths:
            stream.write(f"{sha(path)}  {path.relative_to(ROOT)}\n")
    (destination / "originals").mkdir()
    (destination / "critique-evidence").mkdir()
    for name in COPIED:
        copy_exclusive(OLD / name, destination / name)
    copy_exclusive(
        OLD / "policy_planner16.py", destination / "originals/policy_planner16.py.txt")
    copy_exclusive(
        CRITIQUE / "controls/test_role7_planner_controls.py",
        destination / "test_role7_planner_controls.py")
    for name in ("CRITIQUE.md", "critique.json", "PLAN.json", "SHA256SUMS"):
        copy_exclusive(CRITIQUE / name, destination / "critique-evidence" / name)
    report = {
        "status": "inputs_verified_and_revision02_plan_frozen",
        "input_count": len(paths), "predecessor_seal_files_checked": len(sealed),
        "predecessor_input_files_checked": len(original),
        "critique_files_checked": len(critique),
        "planner_only_parent_proofs_checked": len(proofs),
        "predecessor_seal_sha256": sha(OLD / "SHA256SUMS"),
        "critique_seal_sha256": sha(CRITIQUE / "SHA256SUMS"),
        "predecessor_planner_sha256": sha(OLD / "policy_planner16.py"),
        "copied_unchanged_sha256": {name: sha(OLD / name) for name in COPIED},
        "role7_control_sha256": sha(CRITIQUE / "controls/test_role7_planner_controls.py"),
        "parent_map_sha256": sha(PARENT_MAP),
        "plan_sha256": sha(HERE / "PLAN.json"),
        "input_manifest_sha256": sha(HERE / "INPUT_SHA256SUMS"),
        "unrelated_role2_source_or_proofs_read": False,
    }
    with (destination / "freeze-report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
