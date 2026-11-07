import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
REVIEWS = ROOT / "results/mpc-native-r24/reviews"
AUTHOR = ROOT / ".runtime/r24-team16-planner-revision02"
PARENT = ROOT / ".runtime/r24-team16-planner-revision02-parent01"
SEAL = "14ff6f180a5eb16299f3f001c4ab9986c36ca0da0fa60f4e2c50a9960c175224"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(AUTHOR / "SHA256SUMS") == SEAL
    files = []
    for line in (AUTHOR / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        path = AUTHOR / relative
        assert path.is_file() and not path.is_symlink() and sha(path) == expected
        assert sha(PARENT / relative) == expected
        files.append(path)
    assert len(set(files)) == len(files) == 74
    assert set(files) == {p for p in AUTHOR.rglob("*")
                          if p.is_file() and p != AUTHOR / "SHA256SUMS"}
    pins = (AUTHOR / "INPUT_SHA256SUMS").read_text().splitlines()
    assert len(pins) == 532
    for line in pins:
        expected, relative = line.split("  ", 1)
        assert sha(ROOT / relative) == expected
    for name in ("test_policy_planner16.py", "test_parent_numerical_review.py",
                 "test_policy_planner16_numerical_revision.py"):
        assert sha(AUTHOR / name) == sha(
            ROOT / ".runtime/r24-team16-planner-revision01" / name)
    assert sha(AUTHOR / "test_role7_planner_controls.py") == sha(
        ROOT / ".runtime/r24-team16-comparison/planner16-critique/controls/"
        "test_role7_planner_controls.py")
    evidence = []
    for version in ("314", "312"):
        attempt = PARENT / "attempts" / f"parent-py{version}01"
        command = json.loads((attempt / "command.json").read_bytes())
        result = json.loads((attempt / "result.json").read_bytes())
        assert command["input_pins_verified_before"] == 532
        assert result["input_pins_verified_after"] == 532 and result["returncode"] == 0
        assert sha(attempt / "output.log") == result["output_sha256"]
        log = (attempt / "output.log").read_text()
        assert f"planner={PARENT}/policy_planner16.py" in log and "130 passed" in log
        for name, expected in command["source_snapshot_sha256"].items():
            assert sha(attempt / "sources" / name) == sha(AUTHOR / name) == expected
        evidence.extend(p for p in attempt.rglob("*") if p.is_file())
    pattern = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                         rb"sk-[A-Za-z0-9]{24,}|hf_[A-Za-z0-9]{24,})")
    for path in [*files, *evidence]:
        assert not pattern.search(path.read_bytes()), f"credential-pattern refusal: {path}"
    copies = []
    for base, paths, destination in (
        (AUTHOR, [*files, AUTHOR / "SHA256SUMS"],
         REVIEWS / "native-team16-planner-revision02-author01"),
        (PARENT, evidence, REVIEWS / "native-team16-planner-revision02-parent01"),
    ):
        assert not destination.exists()
        destination.mkdir()
        for source in sorted(paths):
            output = destination / source.relative_to(base)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
            assert sha(output) == sha(source)
            copies.append(dict(path=str(output.relative_to(ROOT)), original=str(source),
                               bytes=output.stat().st_size, sha256=sha(output)))
    report = dict(
        status="PARENT_ACCEPTED_EXACT_ORDERING_AND_FULL_ACTION_FIXED_REFERENCE_ONLY",
        seal_sha256=SEAL, source_sha256=sha(AUTHOR / "policy_planner16.py"),
        tests_each_python=130, unchanged_input_pins=532,
        scope="Exact rational ordering of frozen binary inputs; float metrics display only; "
              "Train uses exact per-panel means and all shared constant candidates incl24.",
        preserved_controls="Original102 and unchanged Role7 ten controls plus18 revision02 "
                           "cases, including common-term absorption, deadlines and versions.",
        bounds="Unrepresentable products may still refuse the full panel, not drop worlds. "
               "Fractions include JSON/selection work within cooperative solver deadline.",
        integration="Only planner numerical dependency can advance. Runtime integration, "
                    "comparison exact-key/time mapping, acquisition/CAS/TTL and native gates "
                    "are not accepted here. Comparison benchmark remains shared16/32.",
        native_model_qualified=False, native_performance_claim=False,
        native_launch_permitted=False, copies=copies,
    )
    with (REVIEWS / "native-team16-planner-revision02-parent-seal01.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], files=len(copies),
                          bytes=sum(row["bytes"] for row in copies))))


if __name__ == "__main__":
    main()
