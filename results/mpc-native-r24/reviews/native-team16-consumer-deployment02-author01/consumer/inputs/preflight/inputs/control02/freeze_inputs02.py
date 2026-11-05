"""Verify narrow frozen inputs and the immutable failed predecessor lineage."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_manifest(directory, expected):
    directory = Path(directory)
    if sha(directory / "SHA256SUMS") != expected:
        raise ValueError("dependency_seal_changed")
    names = set()
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if (Path(name).is_absolute() or ".." in Path(name).parts or name in names
                or sha(directory / name) != digest):
            raise ValueError("dependency_content_changed")
        names.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and path != directory / "SHA256SUMS"}
    if names != actual:
        raise ValueError("dependency_inventory_changed")
    links = {str(path.relative_to(directory)): str(path.readlink())
             for path in directory.rglob("*") if path.is_symlink()}
    if links != json.loads((directory / "SYMLINKS.json").read_bytes()):
        raise ValueError("dependency_symlinks_changed")
    return dict(files=len(names), manifest_sha256=expected)


def verify(*, originals=True):
    pins = json.loads((HERE / "INPUTS.json").read_bytes())
    if sha(HERE / "AMENDMENT02.json") != pins["amendment_sha256"]:
        raise ValueError("preregistered_amendment_changed")
    if sha(HERE / "EXPERIMENT.json") != pins["experiment_sha256"]:
        raise ValueError("preregistered_experiment_changed")
    for row in pins["bindings"]:
        if sha(HERE / row["copied"]) != row["sha256"] or (
                originals and sha(row["source"]) != row["sha256"]):
            raise ValueError("input_binding_changed")
    if originals:
        for old in pins["old_capsules"]:
            verify_manifest(old["directory"], old["manifest_sha256"])
    return dict(bindings=len(pins["bindings"]), input_sha256=sha(HERE / "INPUTS.json"),
                amendment_sha256=pins["amendment_sha256"], experiment_sha256=pins["experiment_sha256"])
