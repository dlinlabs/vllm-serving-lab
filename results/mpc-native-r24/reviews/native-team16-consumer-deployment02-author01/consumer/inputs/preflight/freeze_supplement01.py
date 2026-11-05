"""Retain the hash-only collector dependency exposed by the first CPU attempts."""

import json
from pathlib import Path
import tarfile
import time

from freeze_inputs01 import ARCHIVE, ARCHIVE_SHA256, SOURCE13, sha, verify, write


def main():
    before = verify(originals=True)
    if sha(ARCHIVE.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("archive_changed")
    name = "collect_azure.py"
    closure = json.loads(Path(f"inputs/source-tree/{SOURCE13}/source-closure.json").read_bytes())
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile(name).read()
    if sha(raw) != closure["source_sha256"][name]:
        raise ValueError("preregistered_source_closure_binding")
    destination = Path(f"inputs/source-tree/{SOURCE13}/{name}")
    with destination.open("xb") as handle:
        handle.write(raw)
    write("SUPPLEMENTAL_INPUTS01.json", dict(
        schema="team16-owner-acquisition-supplement-v1",
        reason="Both first development attempts had29PASS+1 harness missing-file failure. The original copy-ready declaration hashes collect_azure.py; it never imports or runs it. Its hash was already committed by the frozen Source13 closure.",
        no_protocol_or_original_input_change=True, before=before,
        bindings=[dict(copy=str(destination), source=str(ARCHIVE) + "::" + name,
                       sha256=sha(raw), bytes=len(raw))]))
    write("SUPPLEMENTAL_FREEZE01.json", dict(
        frozen_at_unix=time.time(), before_successor_tests=True, native_execution=False,
        frozen_sha256={name: sha(Path(name).read_bytes())
                       for name in ("SUPPLEMENTAL_INPUTS01.json", "freeze_supplement01.py")}))
    print(json.dumps(dict(supplemental_bindings=1, initial_bindings=before["bindings"],
                         input_sha256=sha(Path("SUPPLEMENTAL_INPUTS01.json").read_bytes()))))


if __name__ == "__main__":
    main()
