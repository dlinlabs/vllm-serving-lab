"""Freeze one inflight question, then package only committed authorization."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / ".runtime/r24-team16-native-owner-inflight-adapter01"
WORK = ROOT / ".runtime/r24-team16-owner-inflight-gpu01-mayor"
SEAL = "08331c167a233cd164c3104cd1f18b50a82fc8ef0ae17eb13c05b6a0a9247e14"
STEM = "native-team16-owner-inflight-gpu01"
PLAN = HERE / (STEM + "-preregistration.json")
REMOTE = HERE / (STEM + "-remote.py")


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    assert sha(SOURCE / "SHA256SUMS") == SEAL
    parent = HERE / "native-team16-owner-inflight-parent01.json"
    accepted = load(parent)
    assert accepted["source_seal"] == SEAL and accepted["source_bindings_unchanged"] == 271
    assert accepted["status"] == "CPU_INFLIGHT_ADAPTER_ACCEPTED_REQUIRES_NEW_PREREGISTRATION"
    write(PLAN, dict(
        schema="Mayor-preregistered-owner-inflight-component-v1",
        question="Can a target nonempty scheduler/issued-work cut be available while its managed HTTP outcome is pending?",
        hypotheses=[
            "Source acquisition no longer queues behind the entire pending managed POST",
            "The single fixed20ms trigger produces source-bound target inventory, not merely pending intent",
            "Source collection AND collector availability precede each offer's terminal receipt",
            "Concurrent reservations preserve unique IDs, immutable terminal receipts and global1900 accounting",
            "Existing clock/closure/prefix/ghost/privacy/owned-cleanup/source contracts remain valid",
        ],
        frozen=dict(
            adapter_manifest_sha256=SEAL, protocol_sha256=sha(SOURCE / "PROTOCOL.json"),
            offers_sha256=sha(SOURCE / "OFFERS.json"),
            native_plan_sha256=sha(SOURCE / "NATIVE_PLAN.revision01.json"),
            plan_amendment_sha256=sha(SOURCE / "PLAN_AMENDMENT01.json"),
            parent_acceptance_sha256=sha(parent), remote_script_sha256=sha(REMOTE)),
        protocol=load(SOURCE / "PROTOCOL.json"), deployment=load(SOURCE / "DEPLOYMENT.json"),
        explicit_seed_only_amendment=load(SOURCE / "PLAN_AMENDMENT01.json"),
        explicit_private_source_declaration=load(SOURCE / "SOURCE_DECLARATION01.json"),
        source_reviewed=[
            "launch01.py", "service01.py", "owned01.py", "common01.py", "acquire01.py",
            "transport01.py", "coverage01.py", "replay01.py", "origin_guard/sitecustomize.py",
            "test_adapter01.py", "test_transport01.py", "run_local.py", "run_checks01.py",
            "smoke01.py", "freeze01.py", "package01.py"],
        decision="Authorize exactly one fresh existing-GPU component after source review and dual-Python Parent acceptance",
        GPU_schedule="Mayor external exclusive lease through cleanup/export plus launcher internal lock",
        existing_gpu_only=True, rent_install_push_merge_or_product_SDK_edits=False,
        full_original_sources_and_SDK_before_after_required=True,
        raw_success_failure_unsent_refused_records_required=True,
        on_disconnect="Read-only reconcile the same spent namespace; never restart",
        wrapper_watchdog="285s transport/supervision watch; Native260s budget unchanged",
        on_watchdog="No retry/export; launcher inherits Mayor lease; reconcile PID/start ticks",
        pass_meaning="Only two prescribed source-bound usable-inflight overlaps plus engineering contracts",
        failure_meaning="Retain misses, partial coverage and failures; no outcome-driven resampling or fallback",
        unchanged_limits="fixed32;1warmup+2serialmanaged,cap16each,total48;260/180/20/3/40/17;cleanup7+30+2+1;age.5/cutHTTP.2",
        actual_application_at=None, Native14="FAIL unchanged", science_parameters_selected=False,
        model_prediction_copy_ready_or_MPC_qualification_promoted=False,
        MPC_comparison_authorized=False,
        commit_binding="Authority must name a local commit containing this exact preregistration and wrapper"))
    print(json.dumps(dict(status="FROZEN_REQUIRES_LOCAL_COMMIT", plan_sha256=sha(PLAN))))


def pack():
    assert sha(SOURCE / "SHA256SUMS") == SEAL
    plan = load(PLAN)
    assert sha(REMOTE) == plan["frozen"]["remote_script_sha256"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for path in (PLAN, REMOTE, Path(__file__).resolve()):
        saved = subprocess.check_output(["git", "show", commit + ":" + str(path.relative_to(ROOT))], cwd=ROOT)
        assert saved == path.read_bytes()
    WORK.mkdir()
    authority = dict(
        schema="mayor-owner-inflight-preregistration-v1", execute_once=True,
        adapter_manifest_sha256=SEAL, protocol_sha256=plan["frozen"]["protocol_sha256"],
        gpu_uuid=plan["deployment"]["GPU_UUID"], source_root=plan["deployment"]["source_root"],
        existing_Mayor_exclusive_GPU_lock_held=True, preregistration_commit=commit)
    authority_path = WORK / "mayor-owner-inflight01-preregistration.json"
    write(authority_path, authority)
    members = {"SHA256SUMS": SEAL}
    for line in (SOURCE / "SHA256SUMS").read_text().splitlines():
        digest, raw_name = line.split("  ", 1)
        name = Path(raw_name)
        assert not name.is_absolute() and ".." not in name.parts and name.as_posix() not in members
        members[name.as_posix()] = digest
    assert len(members) == 5105
    with tarfile.open(WORK / "deployment.tar.gz", "w:gz") as archive:
        for name, digest in sorted(members.items()):
            path = SOURCE / name
            assert not path.is_symlink() and path.is_file() and sha(path) == digest
            info = archive.gettarinfo(str(path), arcname=SOURCE.name + "/" + name)
            info.mode = 0o444
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        for path in (REMOTE, PLAN, authority_path):
            archive.add(path, arcname=path.name, recursive=False)
    write(WORK / "handoff-index.json", dict(
        local_preregistration_commit=commit, capsule_seal=SEAL,
        deployment_archive_sha256=sha(WORK / "deployment.tar.gz"), capsule_entries=5105,
        authority_sha256=sha(authority_path), no_Native_execution=True))
    print(json.dumps(load(WORK / "handoff-index.json"), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "pack"))
    if parser.parse_args().phase == "freeze":
        freeze()
    else:
        pack()
