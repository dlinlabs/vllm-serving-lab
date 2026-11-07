"""Entry-time closure check for an append-only metrology source snapshot."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys


def _relative(value):
    if (type(value) is not str or not value or value == "." or PurePosixPath(value).is_absolute()
            or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value):
        raise ValueError("research_source_relative_path")
    return value


def _hashes(value):
    if type(value) is not dict or not value:
        raise ValueError("research_source_hash_inventory")
    for name, fingerprint in value.items():
        _relative(name)
        if type(fingerprint) is not str or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("research_source_fingerprint")
    return value


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loader_inventory(root, directory, original):
    roots = {
        root, directory, root / "experiments/mpc-research",
        root / "experiments/mpc-research/native-v2",
        root / "experiments/mpc-research/native_observer",
        *((root / name).parent for name in original),
    }
    seen = set()
    while roots:
        current = roots.pop()
        if current in seen or not current.exists():
            continue
        if current.is_symlink():
            raise ValueError("research_source_import_directory_symlink")
        seen.add(current)
        for path in current.iterdir():
            executable = path.name.endswith((".py", ".pyc", ".pyo", ".so", ".pyd"))
            if path.is_symlink() and (path.name.isidentifier() or executable):
                raise ValueError("research_source_import_symlink")
            if path.is_dir() and path.name.isidentifier() and path.name != "__pycache__":
                roots.add(path)
            elif path.name.endswith((".pyc", ".pyo", ".so", ".pyd")):
                raise ValueError("research_source_unbound_loader")


def verify(directory, manifest_bytes, execution_seal):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("research_source_directory")
    directory = directory.resolve()
    root = directory.parents[2]
    cache = Path(sys.pycache_prefix) if sys.pycache_prefix else None
    if (cache is None or not cache.is_absolute() or cache.is_symlink()
            or not cache.is_dir() or any(cache.iterdir())
            or stat.S_IMODE(cache.stat().st_mode) != 0o700
            or cache.resolve().is_relative_to(root)):
        raise ValueError("research_source_empty_external_cache_required")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != "r24-append-only-metrology-source-closure-v2"
            or manifest.get("execution_seal") != execution_seal
            or type(execution_seal) is not str
            or not re.fullmatch(r"[0-9a-f]{40}", execution_seal)
            or manifest.get("prospective_target") != str(directory)):
        raise ValueError("research_source_identity")
    payload = _hashes(manifest["source_sha256"])
    original = _hashes(manifest["original_python_sha256"])
    # Successors retain all earlier immutable source snapshots.
    if len(original) not in (242, 263, 341, 437, 533, 629, 725, 822, 921, 1028, 1138, 1248) or any(
            not name.endswith(".py") for name in original):
        raise ValueError("research_source_original_inventory")
    if "source-closure.json" in payload:
        raise ValueError("research_source_manifest_not_payload")
    expected_files = set(payload) | {"source-closure.json"}
    expected_directories = {
        str(parent) for name in payload for parent in PurePosixPath(name).parents
        if str(parent) != "."}
    files, directories = set(), set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            files.add(relative)
        else:
            raise ValueError("research_source_nonregular_entry")
    if files != expected_files or directories != expected_directories:
        raise ValueError("research_source_closed_inventory")
    if (directory / "source-closure.json").read_bytes() != manifest_bytes:
        raise ValueError("research_source_manifest_changed")
    if any(_sha(directory / name) != expected for name, expected in payload.items()):
        raise ValueError("research_source_payload_changed")
    additions = {
        (directory / name).relative_to(root).as_posix(): expected
        for name, expected in payload.items() if name.endswith(".py")}
    if set(original) & set(additions):
        raise ValueError("research_source_original_collision")
    expected_python = {**original, **additions}
    actual_python = {}
    for path in root.rglob("*.py"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("research_source_original_nonregular")
        actual_python[path.relative_to(root).as_posix()] = _sha(path)
    if actual_python != expected_python:
        raise ValueError("research_source_append_only_inventory")
    _loader_inventory(root, directory, original)
    return {
        "schema": "r24-entry-source-binding-v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "execution_seal": execution_seal,
        "directory": str(directory),
        "original_python_files": len(original),
        "added_python_files": len(additions),
        "closed_payload_files": len(payload),
        "empty_private_pycache_prefix": str(cache),
        "scope": "Entry-time byte/inventory check, not filesystem immutability, "
                 "native ABI qualification or execution permission.",
    }
