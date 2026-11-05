"""Preserve a source blocker and join selected pins without replaying controls."""

import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
REVIEWS = ROOT / "results/mpc-native-r24/reviews"
REVIEW = ROOT / ".runtime/r24-team16-native-cut-consumer-deployment02-independent-review01"
SUBJECT = ROOT / ".runtime/r24-team16-native-cut-consumer-deployment02"
ARCHIVE = REVIEWS / "native-team16-consumer-deployment02-independent-author01"
INDEX = REVIEWS / "native-team16-consumer-deployment02-independent-parent01.json"
REVIEW_SEAL = "b13e5b29a5312dd4dee09e0d6ebc2abd6bbdd83b3b560ad1468381c5288522d6"
SUBJECT_SEAL = "407bc2e88426f4997064fd7f62d09651af74a2c18c6e4989267f48a601a16ae8"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_regular(path):
    require(stat.S_ISREG(path.lstat().st_mode), "nonregular:" + str(path))
    return path.read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def manifest(data):
    result = {}
    for line in data.decode().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and name not in result, "manifest_path")
        result[name] = digest
    return result


def main():
    require(not ARCHIVE.exists() and not INDEX.exists(), "spent_archive")
    seal = read_regular(REVIEW / "SHA256SUMS")
    require(sha(seal) == REVIEW_SEAL, "review_seal")
    entries = manifest(seal)
    require(len(entries) == 28, "review_member_count")
    regular = set()
    for path in REVIEW.rglob("*"):
        require(not path.is_symlink(), "review_symlink")
        if path.is_file():
            regular.add(path.relative_to(REVIEW).as_posix())
    require(regular == set(entries) | {"SHA256SUMS"}, "review_file_set")
    retained = {}
    for name in sorted(regular):
        data = read_regular(REVIEW / name)
        require(name == "SHA256SUMS" or sha(data) == entries[name], "review_hash:" + name)
        retained[name] = data
    pins = json.loads(retained["PINS.json"])
    first = json.loads(retained["POST-FREEZE-ADDITIONS.json"])
    second = json.loads(retained["POST-FREEZE-ADDITIONS-02.json"])
    require(first["pins_sha256"] == second["pins_sha256"] == sha(retained["PINS.json"])
            and second["previous_additions_sha256"] == sha(
                retained["POST-FREEZE-ADDITIONS.json"]), "addition_chain")
    selection = dict(pins["selection"])
    for document in (first, second):
        require(not document["problems"]
                and document["subject_manifest_sha256"] == SUBJECT_SEAL, "addition_subject")
        for row in document["additions"]:
            require(row["path"] not in selection
                    and row["sha256"] == row["manifest_sha256"], "addition_duplicate")
            selection[row["path"]] = {"sha256": row["sha256"], "bytes": row["bytes"]}
    subject_manifest_bytes = read_regular(SUBJECT / "SHA256SUMS")
    require(sha(subject_manifest_bytes) == SUBJECT_SEAL, "subject_manifest")
    subject_manifest = manifest(subject_manifest_bytes)
    require(len(subject_manifest) == 479, "subject_manifest_entry_count")
    checked = {}
    for name, row in sorted(selection.items()):
        data = subject_manifest_bytes if name == "SHA256SUMS" else read_regular(SUBJECT / name)
        digest = sha(data)
        require(digest == row["sha256"] and len(data) == row["bytes"]
                and (name == "SHA256SUMS" or subject_manifest[name] == digest),
                "selected_pin:" + name)
        checked[name] = row
    require(len(checked) == 39 and sum(row["bytes"] for row in checked.values()) == 396016,
            "selection_counts")
    report = json.loads(retained["report.json"])
    require({row["path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
             for row in report["selection"]} == checked, "report_selection")
    disposition = report["disposition"]
    require(disposition["blockers"] == ["D2-F1"]
            and disposition["is_acceptance"] is False
            and report["budget_use"]["controls_run_per_python"] == 0
            and report["gates"]["G06"] == "not_closed"
            and report["gates"]["G16"] == "pending", "disposition")
    for version in ("py314", "py312"):
        preseal = json.loads(retained["verification/verify-pre-seal-" + version + ".json"])
        require(preseal["status"] == "PASS" and preseal["problems"] == []
                and preseal["subject_entries_rehashed"] == 479, "retained_preseal")
    ARCHIVE.mkdir(mode=0o700)
    for name, data in retained.items():
        destination = ARCHIVE / name
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        destination.chmod(stat.S_IMODE((REVIEW / name).lstat().st_mode) & ~0o222)
    result = {
        "status": "D2_F1_SOURCE_BLOCKER_ACCEPTED_NO_NATIVE_RELEASE",
        "subject_seal": SUBJECT_SEAL,
        "review_seal": REVIEW_SEAL,
        "review_archive": str(ARCHIVE.relative_to(ROOT)),
        "review_archive_regular_files_including_seal": 29,
        "report_sha256": sha(retained["report.json"]),
        "reviewer_model_id": "claude-opus-5.5",
        "reviewer_reasoning_effort": "max",
        "Mayor_selected_source_pins_joined": checked,
        "Mayor_selected_source_files": 39,
        "Mayor_selected_source_bytes": 396016,
        "Mayor_full_subject_inventory_rehashed": False,
        "new_runtime_controls_or_native_workloads": 0,
        "new_helper_processes_spawned": 0,
        "author_rehashed_whole_479_member_subject": True,
        "author_total_file_bytes_read": None,
        "scope_disclosure": (
            "39 is the selected content-review count, not the total files read. "
            "The author verifier additionally rehashed the whole 479-member subject, "
            "contrary to the no-old-inventory-repeat instruction. Hashing reads bytes. "
            "This excess is preserved, not retroactively authorized or repeated by Mayor."
        ),
        "finding": "D2-F1",
        "source_spans": {
            "launch01.py": ["18-24", "74-103", "294-360"],
            "owned01.py": ["66-83", "158-223"],
        },
        "finding_basis": "Source control flow, not an observed live orphan incident.",
        "failure_path": (
            "Pending monitor alarm or anchor publish/clock error can prevent owned.cleanup; "
            "flock can subsequently be released with owned processes potentially alive."
        ),
        "false_pass_claim": False,
        "risk_acceptance": False,
        "resolution": "Surgical successor plus independent review before any live attempt.",
        "original_engineering_acceptance": "c00d29d2296bda7e249578f2de7c08650a4d79c9",
        "original_engineering_acceptance_is_live_eligibility": False,
        "all_other_review_claims": "Conditional P01-P08; not a universal release-safety proof.",
        "G06": "not_closed",
        "G16": "pending",
        "Native14": "FAIL",
        "qualified_posterior_prediction_MPC": False,
        "Native_execution_permission": False,
    }
    with INDEX.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "Mayor_selected_source_pins_joined"}, sort_keys=True))


if __name__ == "__main__":
    main()
