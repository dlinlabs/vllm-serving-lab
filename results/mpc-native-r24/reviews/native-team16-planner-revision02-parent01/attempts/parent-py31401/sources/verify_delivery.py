"""Read-only verification of the full seal, tested source and preserved inputs."""

import json
import sys

from run_attempt import HERE, sha, verify_inputs


def main():
    covered = set()
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = HERE / relative
        if (
            not path.resolve().is_relative_to(HERE) or path.is_symlink()
            or relative in covered or sha(path) != expected
        ):
            raise ValueError("revision_pin_mismatch:" + relative)
        covered.add(relative)
    actual = {
        str(path.relative_to(HERE)) for path in HERE.rglob("*")
        if path.is_file() and path != HERE / "SHA256SUMS"
    }
    if covered != actual:
        raise ValueError("unsealed_or_missing_revision_files")
    if any(path.is_symlink() or path.name in ("__pycache__", ".pytest_cache")
           for path in HERE.rglob("*")):
        raise ValueError("unexpected_symlink_or_cache")
    report = json.loads((HERE / "report.json").read_text())
    for name, expected in report["validation"]["final_tested_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("revision_not_final_tested_bytes:" + name)
    print(json.dumps({
        "status": "verified",
        "python": sys.version.split()[0],
        "sealed_files_checked": len(covered),
        "preserved_input_files_checked": verify_inputs(),
        "before_fix": report["validation"]["before_fix"],
        "after_fix": report["validation"]["after_fix"],
        "parent_setup_errors_distinguished": report["inherited_product_rounding_evidence"][
            "parent_attempts01_02_are_not_algorithm_failures"],
        "native_mpc_permitted": report["native_qualification"]["native_mpc_permitted"],
        "parent_acceptance": report["parent_acceptance"],
        "owned_running_helpers": report["execution"]["owned_running_helpers"],
        "revised_planner_sha256": sha(HERE / "policy_planner16.py"),
        "manifest_sha256": sha(HERE / "SHA256SUMS"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
