"""Pin the first selection before any content read. Writes PINS.json exclusively; prints a summary."""

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
SUBJECT = RUNTIME / "r24-team16-native-cut-consumer-deployment02"
SUCCESSOR = RUNTIME / "r24-team16-native-cut-consumer-successor01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sums(root):
    entries = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def main():
    plan = json.loads((HERE / "PLAN.json").read_text())
    subject = plan["subject"]
    problems = []
    if sha(SUBJECT / "SHA256SUMS") != subject["SHA256SUMS_sha256"]:
        problems.append("subject_seal")
    sealed = sums(SUBJECT)
    if len(sealed) != subject["sealed_entries"]:
        problems.append("subject_entries")
    regular = [p for p in SUBJECT.rglob("*") if p.is_file() and not p.is_symlink()]
    if len(regular) != subject["regular_files_including_seal"]:
        problems.append("subject_regular_files")
    symlinks = sorted((p.relative_to(SUBJECT).as_posix(), os.readlink(p)) for p in SUBJECT.rglob("*") if p.is_symlink())
    selection = plan["first_selection"]
    pins, total = {}, 0
    for name in selection:
        path = SUBJECT / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"not_regular:{name}")
            continue
        digest = sha(path)
        if name != "SHA256SUMS" and sealed.get(name) != digest:
            problems.append(f"seal_mismatch:{name}")
        size = path.stat().st_size
        pins[name] = dict(sha256=digest, bytes=size)
        total += size
    limits = plan["budget"]
    if len(selection) != limits["first_selection_files"] or len(set(selection)) != len(selection) \
            or len(selection) > limits["selected_files_limit"] or total > limits["selected_bytes_limit"]:
        problems.append("selection_limits")

    # Hash-only identity metadata (no content review).
    identity = {}
    pairs = dict(
        consumer=("consumer", "candidate",
                  ["consumer01.py", "worker01.py", "validate01.py", "generation01.py", "ipc01.py", "fixture01.py",
                   "common01.py", "seal_inputs01.py", "NATIVE_PLAN.json", "OFFERS.json", "CONTROL_FREEZE.json"]),
        composition=("", "partial-composition",
                     ["launch01.py", "service01.py", "client01.py", "transport01.py", "owned01.py",
                      "composition_common01.py", "coverage01.py", "origin_guard/sitecustomize.py",
                      "PROTOCOL.json", "ORIGIN_POLICY.json"]))
    for label, (subdir, succ_dir, names) in pairs.items():
        for name in names:
            mine = SUBJECT / subdir / name if subdir else SUBJECT / name
            theirs = SUCCESSOR / succ_dir / name
            gen1 = SUBJECT / "generations/generation01" / (f"{subdir}/{name}" if subdir else name)
            identity[f"{label}:{name}"] = dict(
                deployment02=sha(mine), successor=sha(theirs) if theirs.is_file() else None,
                identical_to_successor=theirs.is_file() and sha(theirs) == sha(mine),
                identical_to_generation01=gen1.is_file() and sha(gen1) == sha(mine))
    copy = SUBJECT / "inputs/SUCCESSOR-SHA256SUMS"
    record = dict(
        schema="d2r01-pins-v1", subject_path=str(SUBJECT.relative_to(RUNTIME.parent)),
        subject_SHA256SUMS_sha256=sha(SUBJECT / "SHA256SUMS"), subject_sealed_entries=len(sealed),
        subject_regular_files=len(regular), subject_symlinks_not_sealed=symlinks,
        successor_SHA256SUMS_sha256=sha(SUCCESSOR / "SHA256SUMS"),
        deployment_copy_of_successor_SHA256SUMS_sha256=sha(copy),
        selection=pins, selected_files=len(pins), selected_bytes=total, hash_only_identity=identity,
        problems=problems)
    with open(HERE / "PINS.json", "x") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps(dict(problems=problems, selected_files=len(pins), selected_bytes=total,
                          symlinks=len(symlinks)), sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
