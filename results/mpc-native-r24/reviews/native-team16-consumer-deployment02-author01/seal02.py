"""Close this capsule once,or independently verify it without further writes."""

import argparse
import json
import os
from pathlib import Path
import time

import composition_common01 as c

HERE = Path(__file__).resolve().parent


def inventory():
    files, aliases = [], []
    for folder, directories, names in os.walk(HERE, followlinks=False):
        for name in tuple(directories):
            path = Path(folder) / name
            if path.is_symlink():
                aliases.append(dict(path=str(path.relative_to(HERE)),target=os.readlink(path),
                                    provenance_only=True,not_followed=True))
                directories.remove(name)
        for name in names:
            path = Path(folder) / name
            if path.is_symlink():
                aliases.append(dict(path=str(path.relative_to(HERE)),target=os.readlink(path),
                                    provenance_only=True,not_followed=True))
            else:
                c.require(path.is_file(),"regular_capsule_file")
                files.append(path)
    return sorted(files), sorted(aliases,key=lambda row:row["path"])


def verify(expected=None):
    manifest = HERE / "SHA256SUMS"
    if expected is not None:
        c.require(c.sha(manifest) == expected,"manifest_digest_changed")
    rows = {}
    for line in manifest.read_text().splitlines():
        fingerprint,name = line[:64],line[66:]
        path = HERE / name
        c.require(not path.is_symlink() and path.is_file()
                  and c.sha(path) == fingerprint,"sealed_file_changed:" + name)
        rows[name] = fingerprint
    files,aliases = inventory()
    c.require(set(rows) == {str(path.relative_to(HERE)) for path in files if path != manifest},
              "extra_or_missing_after_seal")
    c.require(aliases == c.load(HERE / "ALIASES.json")["aliases"],"aliases_changed")
    return dict(entries=len(rows),manifest_sha256=c.sha(manifest),verified=True,
                aliases_provenance_only=len(aliases))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--expected")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.expected),sort_keys=True))
        return
    c.require(Path.cwd().resolve() == HERE and not (HERE / "SHA256SUMS").exists(),
              "exclusive_first_final_seal")
    c.verify(originals=True)
    c.verify_code()
    for name,fingerprint in c.load(HERE / "POSTPROCESS_FREEZE.json")["sha256"].items():
        c.require(c.sha(HERE / name) == fingerprint,"postprocessing_source_changed:" + name)
    for version in ("py314","py312"):
        row = c.load(HERE / ("POSTCHECK-" + version + ".json"))
        c.require(row["semantic_projection_equal"] and row["original_sources_unchanged"],
                  "dual_Python_postcheck_required")
        for item in row["helper_identities"]:
            current = c.owner().snapshot(item["pid"])
            c.require(current is None or current[0] != item["start_ticks"],"owned_helper_still_present")
    c.require(c.load(HERE / "FAILURES.json")["unresolved_test_failures"] == 0,
              "unresolved_controls_require_blocker")
    for item in c.load(HERE / "COMMANDS.json")["actual_commands"]:
        if "exit_file" in item:
            expected = 1 if item["id"].startswith("controls01") else 0
            c.require(int((HERE / item["exit_file"]).read_text()) == expected,"command_exit_binding")
    for version in ("py314","py312"):
        c.require(int((HERE / f"outputs/postprocess-{version}.exit").read_text()) == 0,
                  "postprocessing_command_failed")
    files,aliases = inventory()
    c.write(HERE / "ALIASES.json",dict(aliases=aliases,followed=False,Gitlinks_created=False))
    sources = [dict(path=str(path.relative_to(HERE)),sha256=c.sha(path),bytes=path.stat().st_size)
               for path in files if path.suffix == ".py"]
    c.write(HERE / "SOURCE_INDEX.json",dict(
        sources=sources,
        actual_deployment_sources=["launch01.py","client01.py","service01.py","authorization02.py",
            "composition_common01.py","owned01.py","transport01.py","coverage01.py","origin_guard/sitecustomize.py",
            "consumer/common01.py","consumer/consumer01.py","consumer/generation01.py","consumer/ipc01.py",
            "consumer/validate01.py","consumer/worker01.py","consumer/seal_inputs01.py"],
        old_inputs_and_generations_are_provenance_only=True,old_mains_never_called=True))
    c.write(HERE / "HANDOFF.json",dict(
        state="idle_after_seal",scope=HERE.name,model_id="gpt-6-astra",reasoning_effort="max",
        source_freeze_sha256=c.sha(HERE / "CONTROL_FREEZE.json"),
        dependency_status="UNACCEPTED_FOR_DEPLOYMENT_PENDING_INDEPENDENT_DELTA_AND_SEPARATE_MAYOR_APPROVAL",
        source_successor_manifest_sha256=c.CONSUMER_SEAL,
        Parent_engineering_commit="5317b0223cb96d5c9f53a0ffae5c71745cbf2e32",
        focused_controls_per_Python="9 unique resolved;8 gen01 +1 corrected gen02,10 executions total",
        preserved_initial_harness_failures=2,owned_CPU_identities=4,all_helpers_joined=True,
        legacy42_migration_declared=True,legacy42_suite_executed=False,
        Native_authorization_created=False,Native_entrypoints_executed=False,
        source_only_prospective_candidate=True,no_readiness_or_fairness_claim=True,
        actual_application_at=None,planner_entry_at=None,decision_at=None,
        qualified_model=False,observer_deployable=False,native_mpc_permitted=False,
        Native14="FAIL",G06="not_closed",G16="pending",
        remaining="REMAINING_PREREGISTRATION.json"))
    c.write(HERE / "SEAL_PREPARATION.json",dict(
        all_known_helpers_gone_rechecked_at=time.monotonic(),
        no_live_authority_or_execution=True,source_and_protocol_unchanged_after_declared_final_freeze=True,
        original_sealed_scopes_never_written=True,aliases=len(aliases),
        output_policy="Full regular files sealed;pytest aliases metadata only,not followed. No later writes."))
    files,_ = inventory()
    c.write(HERE / "FILE_INDEX.json",dict(
        excluded_self_and_manifest=["FILE_INDEX.json","SHA256SUMS"],
        files=[dict(path=str(path.relative_to(HERE)),sha256=c.sha(path),bytes=path.stat().st_size)
               for path in files]))
    files,_ = inventory()
    with (HERE / "SHA256SUMS").open("x") as stream:
        for path in files:
            stream.write(c.sha(path) + "  " + str(path.relative_to(HERE)) + "\n")
    print(json.dumps(verify(),sort_keys=True))


if __name__ == "__main__":
    main()
