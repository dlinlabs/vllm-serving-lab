"""Disclosed post-freeze pin additions for the deployment02 independent review.

Pins each added file against the sealed deployment02 SHA256SUMS before its content is read.
Writes POST-FREEZE-ADDITIONS.json exclusively; never modifies the subject.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUBJECT = ROOT / ".runtime/r24-team16-native-cut-consumer-deployment02"
SUBJECT_SEAL = "407bc2e88426f4997064fd7f62d09651af74a2c18c6e4989267f48a601a16ae8"
FILES_LIMIT, BYTES_LIMIT, ADDITIONS_LIMIT = 40, 4 * 1024 * 1024, 4

ADDITIONS = (
    ("CONTROL_FREEZE.json",
     "Q-bind: composition_common01.verify_code (158-162) trusts only the top-level CONTROL_FREEZE sha256 map; "
     "need its exact coverage of runnable launcher/service/client/consumer files."),
    ("consumer/CONTROL_FREEZE.json",
     "Q-bind: consumer READY (consumer01 104) and launcher registration (launch01 84) bind this file's digest; "
     "need its exact coverage of the consumer worker-side runnable files."),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(SUBJECT / "SHA256SUMS") != SUBJECT_SEAL:
        raise SystemExit("subject_seal_mismatch")
    manifest = {line[66:]: line[:64] for line in (SUBJECT / "SHA256SUMS").read_text().splitlines()}
    pins = json.loads((HERE / "PINS.json").read_bytes())
    selected = pins["selection"]
    used_files, used_bytes = len(selected), sum(row["bytes"] for row in selected.values())
    if used_files != pins["selected_files"] or used_bytes != pins["selected_bytes"] or pins["problems"]:
        raise SystemExit("first_selection_pins_inconsistent")
    already = set(selected)
    rows, problems = [], []
    for relative, reason in ADDITIONS:
        path = SUBJECT / relative
        info = path.lstat()
        digest = sha(path)
        ok = (relative not in already and relative in manifest and manifest[relative] == digest
              and not path.is_symlink() and path.is_file())
        if not ok:
            problems.append(relative)
        rows.append(dict(path=relative, bytes=info.st_size, sha256=digest,
                         manifest_sha256=manifest.get(relative), reason=reason))
    total_files = used_files + len(rows)
    total_bytes = used_bytes + sum(row["bytes"] for row in rows)
    if len(rows) > ADDITIONS_LIMIT or total_files > FILES_LIMIT or total_bytes > BYTES_LIMIT:
        problems.append("budget")
    document = dict(
        schema="team16-d2r01-post-freeze-additions-v1",
        added_at_unix_ns=time.time_ns(), python=sys.version.split()[0],
        subject_manifest_sha256=SUBJECT_SEAL, pins_sha256=sha(HERE / "PINS.json"),
        additions=rows, first_selection_files=used_files, first_selection_bytes=used_bytes,
        total_files=total_files, total_bytes=total_bytes,
        limits=dict(files=FILES_LIMIT, bytes=BYTES_LIMIT, additions=ADDITIONS_LIMIT),
        pinned_before_content_read=True, problems=problems)
    with (HERE / "POST-FREEZE-ADDITIONS.json").open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(dict(total_files=total_files, total_bytes=total_bytes, problems=problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
