"""Second disclosed post-freeze pin addition for the deployment02 independent review.

Pins the added file against the sealed deployment02 SHA256SUMS before its content is read and
counts the first additions file against the same 40-file / 4 MiB / 4-addition budget.
Writes POST-FREEZE-ADDITIONS-02.json exclusively; never modifies the subject.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUBJECT = ROOT / ".runtime/r24-team16-native-cut-consumer-deployment02"
SUBJECT_SEAL = "407bc2e88426f4997064fd7f62d09651af74a2c18c6e4989267f48a601a16ae8"
FILES_LIMIT, BYTES_LIMIT, ADDITIONS_LIMIT = 40, 4 * 1024 * 1024, 4

ADDITIONS = (
    ("consumer/inputs/base/inputs/launch02/inputs/source13-entry02.py",
     "Q-own: c.owner() helper used by owned01 (enable_subreaper/remember/snapshot/alive); needed to state "
     "whether launcher remember() discovers an unregistered consumer worker between client Popen and "
     "register_leaf, and what survives if launcher cleanup is skipped (D2-F1)."),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(SUBJECT / "SHA256SUMS") != SUBJECT_SEAL:
        raise SystemExit("subject_seal_mismatch")
    manifest = {line[66:]: line[:64] for line in (SUBJECT / "SHA256SUMS").read_text().splitlines()}
    pins = json.loads((HERE / "PINS.json").read_bytes())
    first = json.loads((HERE / "POST-FREEZE-ADDITIONS.json").read_bytes())
    if first["problems"] or first["pins_sha256"] != sha(HERE / "PINS.json"):
        raise SystemExit("first_additions_inconsistent")
    already = set(pins["selection"]) | {row["path"] for row in first["additions"]}
    used_files, used_bytes = first["total_files"], first["total_bytes"]
    rows, problems = [], []
    for relative, reason in ADDITIONS:
        path = SUBJECT / relative
        info = path.lstat()
        digest = sha(path)
        if not (relative not in already and manifest.get(relative) == digest
                and not path.is_symlink() and path.is_file()):
            problems.append(relative)
        rows.append(dict(path=relative, bytes=info.st_size, sha256=digest,
                         manifest_sha256=manifest.get(relative), reason=reason))
    additions = len(first["additions"]) + len(rows)
    total_files = used_files + len(rows)
    total_bytes = used_bytes + sum(row["bytes"] for row in rows)
    if additions > ADDITIONS_LIMIT or total_files > FILES_LIMIT or total_bytes > BYTES_LIMIT:
        problems.append("budget")
    document = dict(
        schema="team16-d2r01-post-freeze-additions-v1", sequence=2,
        added_at_unix_ns=time.time_ns(), python=sys.version.split()[0],
        subject_manifest_sha256=SUBJECT_SEAL, pins_sha256=sha(HERE / "PINS.json"),
        previous_additions_sha256=sha(HERE / "POST-FREEZE-ADDITIONS.json"),
        additions=rows, additions_used_total=additions, total_files=total_files, total_bytes=total_bytes,
        limits=dict(files=FILES_LIMIT, bytes=BYTES_LIMIT, additions=ADDITIONS_LIMIT),
        pinned_before_content_read=True, problems=problems)
    with (HERE / "POST-FREEZE-ADDITIONS-02.json").open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(dict(additions_used_total=additions, total_files=total_files,
                          total_bytes=total_bytes, problems=problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
