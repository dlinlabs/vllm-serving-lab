"""Preserve first source generation,then freeze only the declared final harness fix."""

import argparse
import difflib
import json
from pathlib import Path
import time

import pins02 as p

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("phase",choices=("preserve","freeze"))
    args = parser.parse_args()
    if Path.cwd().resolve() != HERE:
        raise ValueError("exclusive_deployment02_scope")
    saved = HERE / "generations/generation01"
    if args.phase == "preserve":
        saved.mkdir(parents=True,exist_ok=False)
        old = p.load("CONTROL_FREEZE.json")
        for name,fingerprint in old["sha256"].items():
            if p.sha(name) != fingerprint:
                raise ValueError("first_generation_changed_before_preservation:" + name)
        names = set(old["sha256"]) | {"CONTROL_FREEZE.json","FREEZE.json"}
        core = p.load("consumer/CONTROL_FREEZE.json")
        names |= {"consumer/" + name for name in core["sha256"]}
        for name in sorted(names):
            target = saved / name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:
                stream.write(Path(name).read_bytes())
        p.write(saved / "SNAPSHOT.json",dict(
            source_freeze_sha256=p.sha("CONTROL_FREEZE.json"),
            shared_immutable_dependency_inputs="Original INPUTS/INPUT_FREEZE remain at capsule root;not copied again.",
            sha256={name:p.sha(name) for name in sorted(names)}))
        print(json.dumps(dict(preserved_files=len(names),source_freeze_sha256=p.sha("CONTROL_FREEZE.json"))))
        return
    old = p.load(saved / "CONTROL_FREEZE.json")
    old_files = old["sha256"]
    allowed = {"test_deployment02.py","run_controls02.py"}
    for name,fingerprint in old_files.items():
        if name not in allowed and p.sha(name) != fingerprint:
            raise ValueError("undeclared_generation02_change:" + name)
    changes = []
    for name in sorted(allowed):
        changes.extend(difflib.unified_diff(
            (saved / name).read_text().splitlines(keepends=True),
            Path(name).read_text().splitlines(keepends=True),
            fromfile="generation01/" + name,tofile="generation02/" + name))
    with Path("GENERATION02_EXACT_DIFF.patch").open("x") as stream:
        stream.writelines(changes)
    current = HERE / "generations/generation02"
    current.mkdir(parents=True,exist_ok=False)
    names = sorted(set(old_files) | {"revision02.py","AMENDMENT02.json","GENERATION02_EXACT_DIFF.patch"})
    value = dict(generation=2,frozen_at=time.time(),tests_started=False,
                 sha256={name:p.sha(name) for name in names})
    p.write(current / "CONTROL_FREEZE.json",value)
    Path("CONTROL_FREEZE.json").write_bytes((current / "CONTROL_FREEZE.json").read_bytes())
    report = dict(generation=2,source_freeze_sha256=p.sha("CONTROL_FREEZE.json"),
        input_freeze_sha256=p.sha("INPUT_FREEZE.json"),tests_started=False,prospective_only=True,
        predecessor_source_freeze_sha256=p.sha(saved / "CONTROL_FREEZE.json"),
        live_runtime_source_changed=False,selected_controls=["C07"])
    p.write(current / "FREEZE.json",report)
    Path("FREEZE.json").write_bytes((current / "FREEZE.json").read_bytes())
    print(json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
