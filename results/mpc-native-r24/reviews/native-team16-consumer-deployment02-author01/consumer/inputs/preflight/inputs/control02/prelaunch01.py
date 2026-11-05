"""CPU-only private TMPDIR preflight, to run on the actual host BEFORE launch."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time


SCHEMA = "team16-owned-short-TMPDIR-prelaunch-v1"
MAX_UNIX_PATH_BYTES = 107
UUID_SUFFIX_BYTES = 37


class PrelaunchRefused(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise PrelaunchRefused(reason)


def socket_path_budget(directory):
    raw = str(directory)
    try:
        size = len(raw.encode("ascii"))
    except UnicodeEncodeError as error:
        raise PrelaunchRefused("TMPDIR_must_be_ASCII") from error
    require(raw.startswith("/") and not raw.startswith("//")
            and str(PurePosixPath(raw)) == raw and ".." not in PurePosixPath(raw).parts
            and "\x00" not in raw, "TMPDIR_must_be_canonical_absolute")
    require(size + UUID_SUFFIX_BYTES <= MAX_UNIX_PATH_BYTES, "TMPDIR_UUID_exceeds_AF_UNIX_107_bytes")
    return dict(directory=raw, dirname_ascii_bytes=size, UUID_suffix_bytes=UUID_SUFFIX_BYTES,
                socket_address_max_bytes=size + UUID_SUFFIX_BYTES,
                maximum_AF_UNIX_path_bytes=MAX_UNIX_PATH_BYTES)


def directory_identity(directory, owned_root, *, require_empty):
    directory, owned_root = Path(directory), Path(owned_root)
    require(directory.is_absolute() and owned_root.is_absolute()
            and directory != owned_root and owned_root in directory.parents, "private_owned_root_required")
    require(all(not path.is_symlink() for path in (directory, *directory.parents)),
            "TMPDIR_symlink_not_permitted")
    require(directory.resolve(strict=True) == directory and owned_root.resolve(strict=True) == owned_root,
            "canonical_private_directory_required")
    info, root_info = directory.stat(), owned_root.stat()
    require(stat.S_ISDIR(info.st_mode) and stat.S_ISDIR(root_info.st_mode), "private_directories_required")
    require(info.st_uid == root_info.st_uid == os.geteuid(), "private_directory_owner_changed")
    require(stat.S_IMODE(info.st_mode) == stat.S_IMODE(root_info.st_mode) == 0o700,
            "private_directory_mode_must_be_0700")
    require(os.access(directory, os.W_OK | os.X_OK), "TMPDIR_not_writable_searchable")
    if require_empty:
        require(not any(directory.iterdir()), "TMPDIR_must_be_empty_before_launch")
    return dict(
        directory=str(directory), owned_root=str(owned_root), uid=info.st_uid, mode="0700",
        device=info.st_dev, inode=info.st_ino,
        owned_root_device=root_info.st_dev, owned_root_inode=root_info.st_ino,
        owned_root_uid=root_info.st_uid, owned_root_mode="0700")


def clock_domain():
    return ("linux-monotonic:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            + ":" + os.readlink("/proc/self/ns/time"))


def make_receipt(directory, owned_root):
    budget = socket_path_budget(directory)
    code = Path(__file__).resolve().parent
    require(code != Path(owned_root) and code not in Path(owned_root).parents,
            "private_state_must_be_outside_sealed_code")
    identity = directory_identity(directory, owned_root, require_empty=True)
    require(os.environ.get("TMPDIR") == budget["directory"], "TMPDIR_environment_must_match_checked_directory")
    return dict(
        schema=SCHEMA, **budget, **{key: value for key, value in identity.items() if key != "directory"},
        checked_at=time.monotonic(), clock_domain=clock_domain(),
        empty_before_launch=True, environment_TMPDIR_matches=True,
        socket_created=False, native_launch_performed=False,
        cache_paths_changed=False, whole_environment_recorded=False)


def verify_receipt(receipt):
    require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA, "TMPDIR_receipt_schema")
    require(set(receipt) == {
        "schema", "directory", "dirname_ascii_bytes", "UUID_suffix_bytes",
        "socket_address_max_bytes", "maximum_AF_UNIX_path_bytes", "owned_root",
        "uid", "mode", "device", "inode", "owned_root_device", "owned_root_inode",
        "owned_root_uid", "owned_root_mode", "checked_at", "clock_domain",
        "empty_before_launch", "environment_TMPDIR_matches", "socket_created",
        "native_launch_performed", "cache_paths_changed", "whole_environment_recorded",
    }, "TMPDIR_receipt_fields_not_permitted")
    code = Path(__file__).resolve().parent
    require(code != Path(receipt["owned_root"]) and code not in Path(receipt["owned_root"]).parents,
            "private_state_must_be_outside_sealed_code")
    budget = socket_path_budget(receipt["directory"])
    identity = directory_identity(receipt["directory"], receipt["owned_root"], require_empty=False)
    for key, value in {**budget, **identity}.items():
        require(type(receipt.get(key)) is type(value) and receipt[key] == value,
                "TMPDIR_receipt_or_directory_identity_changed")
    require(receipt.get("empty_before_launch") is True
            and receipt.get("environment_TMPDIR_matches") is True
            and receipt.get("native_launch_performed") is False
            and receipt.get("socket_created") is False
            and receipt.get("cache_paths_changed") is False
            and receipt.get("whole_environment_recorded") is False, "TMPDIR_prelaunch_contract")
    require(os.environ.get("TMPDIR") == receipt["directory"],
            "TMPDIR_environment_must_match_prelaunch")
    require(receipt.get("clock_domain") == clock_domain()
            and type(receipt.get("checked_at")) in (int, float)
            and 0 <= receipt["checked_at"] <= time.monotonic(), "TMPDIR_prelaunch_clock")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--owned-root", type=Path, required=True)
    parser.add_argument("--tmpdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output == args.owned_root / "tmpdir-preflight.json", "owned_preflight_output_required")
    receipt = make_receipt(args.tmpdir, args.owned_root)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    args.output.chmod(0o600)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
