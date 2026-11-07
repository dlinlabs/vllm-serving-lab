"""Freeze prospective source/templates before the bounded CPU controls."""

import difflib
import hashlib
import json
from pathlib import Path
import time

import pins02 as p

HERE = Path(__file__).resolve().parent


def main():
    if Path.cwd().resolve() != HERE or (HERE / "CONTROL_FREEZE.json").exists():
        raise ValueError("exclusive_first_source_freeze_required")
    p.verify(True)
    old = p.ROOT / ".runtime/r24-team16-native-cut-consumer-successor01"
    original = old / "candidate/seal_inputs01.py"
    manifest = {line[66:]:line[:64] for line in (HERE / "inputs/SUCCESSOR-SHA256SUMS").read_text().splitlines()}
    if p.sha(original) != manifest["candidate/seal_inputs01.py"]:
        raise ValueError("original_input_verifier_changed")
    destination = HERE / "inputs/reference-consumer/seal_inputs01.py"
    with destination.open("xb") as stream:
        stream.write(original.read_bytes())
    p.write("ADDITIONAL_INPUTS.json", dict(bindings=[dict(source=str(original),
        copy=str(destination.relative_to(HERE)), sha256=p.sha(original))],
        reason="Exact before-source for explicitly replaced private data verifier;literal preflight guards unchanged."))
    proof = dict(accepted=False, path=None, sha256=None, subject_sha256=None)
    p.write("MAYOR_APPROVAL_TEMPLATE_INVALID.json", dict(
        schema="mayor-cut-consumer-deployment02-preregistration-v1",
        scope="one_model_free_ordinal2_consumer_component_only", execute_once=False,
        deployment_manifest_sha256=None, protocol_sha256=p.sha("PROTOCOL.json"),
        native_plan_sha256=p.sha("NATIVE_PLAN.json"), offers_sha256=p.sha("OFFERS.json"),
        successor_manifest_sha256=p.load("PLAN.json")["source_successor_manifest_sha256"],
        capture_id="nv2-team16-cut-consumer02-fixed32", operation_id="opteam16-cut-consumer02",
        gpu_uuid="GPU-8610519d-3680-ae5f-cdfc-40037bfb5013",
        source_root="/workspace/pr32-r23/source",
        native_python="/workspace/pr32-r23/venv/bin/python3.12", preregistration_commit=None,
        existing_Mayor_exclusive_GPU_lock_held=False, fresh_host_receipt=None,
        fresh_host_receipt_sha256=None,
        acceptances={key:dict(proof) for key in ("successor_Mayor_engineering",
            "successor_independent_delta","deployment_Mayor_review","observational_policy_Mayor_review")},
        actual_actuation_authorized=False, model_qualification_authorized=False))
    pairs = [(HERE / "inputs/reference-composition" / name, HERE / name) for name in
             ("client01.py","launch01.py","service01.py","composition_common01.py","transport01.py",
              "owned01.py","coverage01.py","origin_guard/sitecustomize.py","PROTOCOL.json","ORIGIN_POLICY.json")]
    pairs += [(HERE / "inputs/reference-consumer" / name, HERE / "consumer" / name)
              for name in ("common01.py","consumer01.py","generation01.py","ipc01.py","validate01.py",
                           "worker01.py","fixture01.py","seal_inputs01.py","NATIVE_PLAN.json","OFFERS.json")]
    pairs += [(HERE / "inputs/reference-tests/test_consumer01.py", HERE / "migration/test_consumer01.py")]
    differences, unchanged = [], []
    for before, after in pairs:
        if before.read_bytes() == after.read_bytes():
            unchanged.append(str(after.relative_to(HERE)))
        differences.extend(difflib.unified_diff(
            before.read_text().splitlines(keepends=True), after.read_text().splitlines(keepends=True),
            fromfile=str(before.relative_to(HERE)), tofile=str(after.relative_to(HERE))))
    with Path("EXACT_SOURCE_DIFF.patch").open("x") as stream:
        stream.writelines(differences)
    legacy_before, legacy_after = pairs[-1]
    with Path("LEGACY42_REASON_MIGRATION.patch").open("x") as stream:
        stream.writelines(difflib.unified_diff(
            legacy_before.read_text().splitlines(keepends=True),
            legacy_after.read_text().splitlines(keepends=True),
            fromfile="candidate/test_consumer01.py", tofile="candidate/test_consumer01.py"))
    p.write("DERIVATION.json", dict(
        dependency_aggregate_status=p.load("PLAN.json")["dependency_status"],
        exact_unchanged=unchanged, differences="EXACT_SOURCE_DIFF.patch",
        private_origin_guard_changes=["Both environment keys now R24_CUT_CONSUMER02_*",
                                      "Literal destination cut-consumer-state02/origins-consumer02"],
        original_SDK_product_Source13_changed=False,
        new_files=["authorization02.py","pins02.py","prepare02.py","cpu_helper02.py","test_deployment02.py",
                   "run_controls02.py","consumer/seal_inputs01.py"],
        no_42_suite_execution=True))
    for folder in ("consumer", "."):
        directory = HERE / folder
        names = [str(path.relative_to(directory)) for path in sorted(directory.glob("*.py"))]
        if folder == "consumer":
            names += ["NATIVE_PLAN.json","OFFERS.json"]
        else:
            names += ["NATIVE_PLAN.json","OFFERS.json","PROTOCOL.json","ORIGIN_POLICY.json",
                      "origin_guard/sitecustomize.py","PLAN.json","PLAN_FREEZE.json","INPUTS.json",
                      "INPUT_FREEZE.json","ADDITIONAL_INPUTS.json","CONTROL_PLAN.json",
                      "SOURCE_BINDING.json","MAYOR_APPROVAL_TEMPLATE_INVALID.json",
                      "LEGACY42_REASON_MIGRATION.json","LEGACY42_REASON_MIGRATION.patch",
                      "migration/test_consumer01.py","INTEGRATION.txt","REMAINING_PREREGISTRATION.json",
                      "DERIVATION.json","EXACT_SOURCE_DIFF.patch","consumer/CONTROL_FREEZE.json"]
        p.write(directory / "CONTROL_FREEZE.json", dict(
            generation=1, frozen_at=time.time(), tests_started=False,
            sha256={name:p.sha(directory / name) for name in names}))
        if folder == "consumer":
            p.write(HERE / "SOURCE_BINDING.json", dict(
                author_launch_permitted=False,
                successor_manifest_sha256=p.load("PLAN.json")["source_successor_manifest_sha256"],
                candidate_control_freeze_sha256=p.sha(directory / "CONTROL_FREEZE.json"),
                original_runtime_logic_unchanged_except_namespace_and_private_input_verifier=True))
    p.write("FREEZE.json", dict(generation=1, source_freeze_sha256=p.sha("CONTROL_FREEZE.json"),
        input_freeze_sha256=p.sha("INPUT_FREEZE.json"), prospective_only=True, tests_started=False))
    print(json.dumps(p.load("FREEZE.json"), sort_keys=True))


if __name__ == "__main__":
    main()
