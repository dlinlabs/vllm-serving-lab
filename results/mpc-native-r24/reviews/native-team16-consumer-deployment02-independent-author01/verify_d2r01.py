"""Read-only join helper for the deployment02 independent source disposition (d2r01). Writes nothing; prints one JSON line.

Default mode expects the sealed package; --pre-seal expects no SHA256SUMS yet. Both modes re-hash the whole subject
seal, every pinned selection file, the 21 hash-only identity pairs and the related seal digests, and require
report.json to equal a fresh in-memory build. The builder is executed from source (no import, no bytecode).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
SUBJECT = RUNTIME / "r24-team16-native-cut-consumer-deployment02"
SUCCESSOR = RUNTIME / "r24-team16-native-cut-consumer-successor01"
SCR01 = RUNTIME / "r24-team16-native-cut-consumer-successor-independent-review01"
DISPOSITION01 = RUNTIME / "r24-team16-native-cut-consumer-successor-independent-review01-disposition01"
OWN_FILES = {
    "MARKER-review-start.txt", "MARKER-freeze.txt", "PLAN.json", "PINS.json", "pin_d2r01.py", "add_pins_d2r01.py",
    "add_pins_d2r01_02.py", "POST-FREEZE-ADDITIONS.json", "POST-FREEZE-ADDITIONS-02.json", "make_report_d2r01.py",
    "report.json", "REVIEW.md", "README.txt", "commands.sh", "verify_d2r01.py",
    "verification/pin-py314.log", "verification/pin-py314.exit",
    "verification/add-pins-py314.log", "verification/add-pins-py314.exit",
    "verification/add-pins-02-py314.log", "verification/add-pins-02-py314.exit",
    "verification/make-report-py314.log", "verification/make-report-py314.exit",
    "verification/no-writes-after-marker.txt",
    "verification/verify-pre-seal-py314.json", "verification/verify-pre-seal-py314.exit",
    "verification/verify-pre-seal-py312.json", "verification/verify-pre-seal-py312.exit",
}
IDENTITY_PAIRS = dict(
    consumer=("consumer", "candidate",
              ["consumer01.py", "worker01.py", "validate01.py", "generation01.py", "ipc01.py", "fixture01.py",
               "common01.py", "seal_inputs01.py", "NATIVE_PLAN.json", "OFFERS.json", "CONTROL_FREEZE.json"]),
    composition=("", "partial-composition",
                 ["launch01.py", "service01.py", "client01.py", "transport01.py", "owned01.py",
                  "composition_common01.py", "coverage01.py", "origin_guard/sitecustomize.py",
                  "PROTOCOL.json", "ORIGIN_POLICY.json"]))


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(root):
    regular, links = set(), set()
    for directory, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(directory) / name
            if path.is_symlink():
                links.add((path.relative_to(root).as_posix(), os.readlink(path)))
        for name in filenames:
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                regular.add(path.relative_to(root).as_posix())
    return regular, links


def sums(root, problems, label):
    entries = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in entries:
            problems.append(f"{label}_duplicate_entry:{name}")
        entries[name] = digest
    return entries


def load_builder():
    path = HERE / "make_report_d2r01.py"
    namespace = dict(__file__=str(path), __name__="make_report_d2r01_verifier")
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return namespace


def exit_zero(path):
    return path.is_file() and path.read_text().strip() in ("0", "exit 0")


def main():
    sealed = "--pre-seal" not in sys.argv[1:]
    problems = []
    builder = load_builder()
    pins = json.loads((HERE / "PINS.json").read_bytes())
    first = json.loads((HERE / "POST-FREEZE-ADDITIONS.json").read_bytes())
    second = json.loads((HERE / "POST-FREEZE-ADDITIONS-02.json").read_bytes())
    plan = json.loads((HERE / "PLAN.json").read_bytes())

    # Subject seal: digest, complete entry re-hash, exact regular-file set, and the known unsealed symlinks.
    if sha(SUBJECT / "SHA256SUMS") != builder["SUBJECT_SEAL"] or pins["subject_SHA256SUMS_sha256"] != builder[
            "SUBJECT_SEAL"] or plan["subject"]["SHA256SUMS_sha256"] != builder["SUBJECT_SEAL"]:
        problems.append("subject_seal_digest")
    manifest = sums(SUBJECT, problems, "subject")
    regular, links = walk(SUBJECT)
    if len(manifest) != pins["subject_sealed_entries"] or len(regular) != pins["subject_regular_files"] \
            or set(manifest) != regular - {"SHA256SUMS"}:
        problems.append("subject_file_set")
    for name, digest in sorted(manifest.items()):
        if sha(SUBJECT / name) != digest:
            problems.append(f"subject_hash_mismatch:{name}")
    if links != {tuple(row) for row in pins["subject_symlinks_not_sealed"]}:
        problems.append("subject_symlink_set")

    # Related sealed inputs: digests only.
    for label, root, expected in (("successor01", SUCCESSOR, builder["SUCCESSOR_SEAL"]),
                                  ("scr01", SCR01, builder["SCR01_SEAL"]),
                                  ("disposition01", DISPOSITION01, builder["DISPOSITION01_SEAL"])):
        if sha(root / "SHA256SUMS") != expected:
            problems.append(f"{label}_seal_digest")
    if not (pins["successor_SHA256SUMS_sha256"] == builder["SUCCESSOR_SEAL"]
            == pins["deployment_copy_of_successor_SHA256SUMS_sha256"]
            == sha(SUBJECT / "inputs/SUCCESSOR-SHA256SUMS")):
        problems.append("successor_copy_binding")

    # Pinned selection (first selection plus post-freeze additions), re-hashed on this Python.
    selection = {name: row for name, row in pins["selection"].items()}
    if pins["problems"] or first["problems"] or second["problems"]:
        problems.append("pin_step_problems")
    if first["pins_sha256"] != sha(HERE / "PINS.json") or second["pins_sha256"] != sha(HERE / "PINS.json") \
            or second["previous_additions_sha256"] != sha(HERE / "POST-FREEZE-ADDITIONS.json"):
        problems.append("addition_chain")
    for document in (first, second):
        if document["subject_manifest_sha256"] != builder["SUBJECT_SEAL"] \
                or document["pinned_before_content_read"] is not True:
            problems.append("addition_binding")
        for row in document["additions"]:
            if row["path"] in selection or row["manifest_sha256"] != row["sha256"]:
                problems.append(f"addition_row:{row['path']}")
            selection[row["path"]] = dict(sha256=row["sha256"], bytes=row["bytes"])
    total = 0
    for name, row in sorted(selection.items()):
        path = SUBJECT / name
        if path.is_symlink() or not path.is_file() or sha(path) != row["sha256"] \
                or path.stat().st_size != row["bytes"] or (name != "SHA256SUMS" and manifest.get(name) != row["sha256"]):
            problems.append(f"selection_pin:{name}")
        total += row["bytes"]
    budget = plan["budget"]
    if (len(selection), total) != (second["total_files"], second["total_bytes"]) \
            or len(selection) > budget["selected_files_limit"] or total > budget["selected_bytes_limit"] \
            or second["additions_used_total"] != len(first["additions"]) + len(second["additions"]) \
            or second["additions_used_total"] > budget["post_freeze_additions_limit"] \
            or sorted(pins["selection"]) != sorted(plan["first_selection"]):
        problems.append("selection_budget")

    # Hash-only identity metadata against successor01 and generation01.
    identity = {}
    for label, (subdir, successor_dir, names) in IDENTITY_PAIRS.items():
        for name in names:
            mine = SUBJECT / subdir / name if subdir else SUBJECT / name
            theirs = SUCCESSOR / successor_dir / name
            generation = SUBJECT / "generations/generation01" / (f"{subdir}/{name}" if subdir else name)
            identity[f"{label}:{name}"] = dict(
                deployment02=sha(mine), successor=sha(theirs) if theirs.is_file() else None,
                identical_to_successor=theirs.is_file() and sha(theirs) == sha(mine),
                identical_to_generation01=generation.is_file() and sha(generation) == sha(mine))
    if identity != pins["hash_only_identity"]:
        problems.append("hash_only_identity")

    # The report is a deterministic function of the pin records and the builder.
    text = builder["render"]()
    if (HERE / "report.json").read_text() != text:
        problems.append("report_not_fresh_build")
    report = json.loads(text)
    disposition = report["disposition"]
    if report["schema"] != "d2r01-report-v1" or disposition["code"] != builder["DISPOSITION_CODE"]:
        problems.append("schema_or_code")
    if disposition["is_acceptance"] is not False or disposition["not_a_proof"] is not True \
            or disposition["ownership_and_OS_failure_safety_universally_established"] is not False \
            or any(value is not False for value in report["permissions"].values()):
        problems.append("flag_invariants")
    if report["gates"] != dict(G06="not_closed", G16="pending", Native14="FAIL_UNCHANGED", gate_status_changes=[]):
        problems.append("gate_invariants")
    use = report["budget_use"]
    if use["controls_run_per_python"] != 0 or use["controls_limit_per_python"] != 4 \
            or (use["files"], use["bytes"]) != (len(selection), total) or report["failures_preserved"] != []:
        problems.append("budget_or_failures")
    note_ids = [row["id"] for row in report["notes"]]
    finding_ids = [row["id"] for row in report["findings"]]
    premise_ids = [row["id"] for row in report["premises"]]
    if note_ids != ["N%02d" % i for i in range(1, 15)] or finding_ids != ["D2-F1"] \
            or premise_ids != ["P%02d" % i for i in range(1, 9)] or disposition["conditional_on"] != premise_ids \
            or disposition["blockers"] != finding_ids:
        problems.append("ids")
    if set(report["answers"]) != set(report["questions"]):
        problems.append("answers_cover_questions")
    for key, answer in report["answers"].items():
        if not set(answer.get("notes", [])) <= set(note_ids) or not set(answer.get("findings", [])) <= set(finding_ids):
            problems.append(f"answer_reference:{key}")
    if {(row["path"], row["sha256"], row["bytes"]) for row in report["selection"]} != {
            (name, row["sha256"], row["bytes"]) for name, row in selection.items()}:
        problems.append("report_selection")

    # Own package: no symlinks, no bytecode caches, exact file list when sealed.
    own, own_links = walk(HERE)
    if own_links or any("__pycache__" in name.split("/") for name in own) or any(
            "__pycache__" in dirs for _, dirs, _ in os.walk(HERE)):
        problems.append("own_symlink_or_bytecode")
    for name in ("pin-py314", "add-pins-py314", "add-pins-02-py314"):
        if not exit_zero(HERE / "verification" / (name + ".exit")):
            problems.append(f"step_exit:{name}")
    if sealed:
        entries = sums(HERE, problems, "own")
        if set(entries) != OWN_FILES or own != OWN_FILES | {"SHA256SUMS"}:
            problems.append("own_sealed_file_list")
        for name, digest in sorted(entries.items()):
            if not (HERE / name).is_file() or sha(HERE / name) != digest:
                problems.append(f"own_hash_mismatch:{name}")
        if not exit_zero(HERE / "verification/make-report-py314.exit"):
            problems.append("step_exit:make-report-py314")
        for version in ("py314", "py312"):
            row = json.loads((HERE / "verification" / f"verify-pre-seal-{version}.json").read_text())
            if row["status"] != "PASS" or row["mode"] != "pre-seal" or row["problems"] \
                    or not row["python"].startswith("3.14." if version == "py314" else "3.12.") \
                    or not exit_zero(HERE / "verification" / f"verify-pre-seal-{version}.exit"):
                problems.append(f"pre_seal_{version}")
        lines = (HERE / "verification/no-writes-after-marker.txt").read_text().splitlines()
        if not lines or any(not (line.startswith("check") or line == "find_exit=0") for line in lines):
            problems.append("writes_after_marker")
    else:
        if "SHA256SUMS" in own:
            problems.append("pre_seal_but_SHA256SUMS_present")
        if not own <= OWN_FILES:
            problems.append("unexpected_own_files:" + ",".join(sorted(own - OWN_FILES)))
    print(json.dumps(dict(
        schema="d2r01-verification-v1", mode="sealed" if sealed else "pre-seal", python=sys.version.split()[0],
        status="FAIL" if problems else "PASS", problems=problems, subject_seal=builder["SUBJECT_SEAL"],
        subject_entries_rehashed=len(manifest), selection_files=len(selection), selection_bytes=total,
        report_sha256=hashlib.sha256(text.encode()).hexdigest(),
        own_seal=sha(HERE / "SHA256SUMS") if sealed and (HERE / "SHA256SUMS").is_file() else None,
        disposition=disposition["code"]), sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
