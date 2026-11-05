"""Read-only source/proof checks and reporting,not a third control generation."""

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import composition_common01 as c

HERE = Path(__file__).resolve().parent


def cases(label, generation):
    root = ET.parse(HERE / f"attempts/generation{generation:02d}-{label}/junit.xml").getroot()
    result = {}
    for row in root.iter("testcase"):
        name = row.attrib["name"]
        issue = row.find("failure")
        if issue is None:
            issue = row.find("error")
        result[name.split("_")[1]] = dict(name=name, status="pass" if issue is None else "fail",
            failure=None if issue is None else issue.attrib.get("message"), seconds=float(row.attrib["time"]))
    return result


def proof_file(label, code, name, generation=1):
    matches = list((HERE / f"attempts/generation{generation:02d}-{label}/cases").glob(
        "test_" + code + "_*0/" + name))
    c.require(len(matches) == 1, "one_named_control_receipt:" + code + ":" + name)
    return matches[0]


def projection(label):
    first, second = cases(label, 1), cases(label, 2)
    combined = {**first, **second}
    bootstrap = c.load(proof_file(label, "C02", "new-bootstrap-binding.json"))
    placement = c.load(proof_file(label, "C05", "actual-protocol-placement.json"))
    raw = c.load(proof_file(label, "C07", "exact-new-bundle.json", generation=2))
    close = c.load(proof_file(label, "C08", "host-actual-close-proof.json"))
    inventory = c.load(proof_file(label, "C08", "owned-cleanup.json"))
    privacy = c.load(proof_file(label, "C09", "private-close-guards.json"))
    return dict(control_status={key:value["status"] for key,value in combined.items()},
        bootstrap_events=bootstrap["events"], fixture_raw_unchanged=bootstrap["fixture_raw_unchanged"],
        new_plan_sha256=bootstrap["plan_sha256"], new_offers_sha256=bootstrap["offers_sha256"],
        lifecycle_calls=placement["calls"], actual_mock_overlap=raw["actual_mock_overlap"],
        source_capture=raw["bundle"]["current"]["journal"]["cut"]["native_observation"]["native_phase"]["capture"],
        registered_owned_count=len(inventory["owned_pid_start_ticks"]),
        all_joined=inventory["all_joined"], capture_complete=inventory["capture_complete"],
        true_close_byte_equal=close["staged_sha256"] == close["persisted"]["persisted_sha256"],
        CPU_no_cut_component_pass=close["persisted"]["receipt"]["component_pass"],
        privacy_negative_guards=[privacy["unjoined_refused"],privacy["decoded_unsafe_refused"]],
        actual_application_at=None, Native_execution=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--label", choices=("py314","py312"), required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    pins = c.verify(originals=True)
    c.verify_code()
    old = c.load(HERE / "generations/generation01/CONTROL_FREEZE.json")["sha256"]
    allowed = {"run_controls02.py","test_deployment02.py"}
    unchanged = []
    for name,fingerprint in old.items():
        if name not in allowed:
            c.require(c.sha(HERE / name) == fingerprint, "unexpected_post_control_change:" + name)
            unchanged.append(name)
    snapshot = c.load(HERE / "generations/generation01/SNAPSHOT.json")
    for name,fingerprint in snapshot["sha256"].items():
        c.require(c.sha(HERE / "generations/generation01" / name) == fingerprint,
                  "first_generation_snapshot_changed:" + name)
    notice = c.load(HERE / "PARENT_ENGINEERING_NOTICE.json")
    c.require(c.sha(notice["source"]) == notice["sha256"], "Parent_engineering_index_changed")
    c.require(c.sha(HERE / "inputs/verified-model-request01.txt") ==
              "a653c68e9d5e8ddb03c35894be20e319310fa1532ff0e4104f8c8b74d22e4aca",
              "retained_verified_model_request")
    for name in old:
        path = HERE / name
        if path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")
    compile((HERE / "migration/test_consumer01.py").read_bytes(),
            str(HERE / "migration/test_consumer01.py"), "exec")
    before = ast.parse((HERE / "inputs/reference-tests/test_consumer01.py").read_bytes())
    after = ast.parse((HERE / "migration/test_consumer01.py").read_bytes())
    target = "test_actual_worker_checks_authority_and_malformed_IPC"
    for left,right in zip(before.body,after.body,strict=True):
        if isinstance(left,ast.FunctionDef) and left.name == target:
            c.require(ast.dump(left.decorator_list[0]) == ast.dump(right.decorator_list[0]),
                      "legacy_parameter_count_changed")
        else:
            c.require(ast.dump(left) == ast.dump(right), "unrelated_legacy_test_definition_changed")
    helper_rows = []
    for version in ("py314","py312"):
        for generation in (1,2):
            output = HERE / f"attempts/generation{generation:02d}-{version}"
            finished = c.load(output / "finished.json")
            c.require(finished["independently_gone"] and not finished["non_main_threads_remaining"],
                      "owned_control_helpers_unjoined")
            for identity in finished["helper_identities"]:
                current = c.owner().snapshot(identity["pid"])
                gone = current is None or current[0] != identity["start_ticks"]
                c.require(gone,"owned_identity_still_present")
                helper_rows.append(dict(identity, version=version, generation=generation,
                                        freshly_gone=gone, checked_at=time.monotonic()))
    c.require(len(helper_rows) == 4, "expected_four_owned_control_identities")
    projections = {label:projection(label) for label in ("py314","py312")}
    # Capture start clock is real per-fixture evidence,not deterministic source identity.
    for value in projections.values():
        value["source_capture"] = {key:item for key,item in value["source_capture"].items()
                                   if key not in ("started_at","started","start_at")}
    c.require(projections["py314"] == projections["py312"], "semantic_cross_Python_projection")
    c.require(set(projections["py314"]["control_status"].values()) == {"pass"},
              "focused_control_not_resolved")
    check = dict(label=args.label, version=sys.version, executable=sys.executable,
        verified_inputs=pins, original_sources_unchanged=True,
        unchanged_generation01_nonharness_files=len(unchanged),
        helper_identities=helper_rows, legacy42_definitions_executed=False,
        legacy42_parameter_decorator_unchanged=True, semantic_projection_equal=True,
        no_new_control_execution=True, Native_execution=False,
        elapsed_s=time.monotonic() - started)
    c.write(HERE / ("POSTCHECK-" + args.label + ".json"),check)
    if args.report:
        validation = {}
        failures = []
        for label in ("py314","py312"):
            first, second = cases(label,1), cases(label,2)
            c.require(len(first) == 9 and len(second) == 1 and set(second) == {"C07"},
                      "bounded_nine_unique_ten_executions")
            c.require(first["C07"]["status"] == "fail" and "exclusive_candidate_cwd_required" in
                      first["C07"]["failure"], "exact_retained_first_attempt_failure")
            validation[label] = dict(generation01=first,generation02=second,
                unique_passing_controls=9,total_executions=10,
                statement="8 unchanged passes generation01 plus corrected C07 generation02;not a fresh nine-test generation02 suite.")
            failures.append(dict(Python=label, generation=1, control="C07",
                classification="Harness omitted literal required cwd;correct guard refusal,not runtime/Native defect.",
                reason=first["C07"]["failure"], log=f"outputs/generation01-{label}.log",
                exit_file=f"outputs/generation01-{label}.exit", retained=True, resolved_in=2))
        c.write(HERE / "VALIDATION.json",dict(versions=validation, semantic_projection_equal=True,
            old14_6_42_suites_executed=False, live_mains_executed=False,
            invalid_approval_negative_only=True, valid_approval_created=False))
        c.write(HERE / "FAILURES.json",dict(failures=failures,unexpected_failures_total=2,
            unresolved_test_failures=0,all_first_attempts_preserved=True,no_third_generation=True))
        c.write(HERE / "SEMANTIC_PROJECTIONS.json",dict(
            projections=projections,exact_equal=True,
            exclusion="Actual per-run process/clocks/IDs/raw hashes are retained elsewhere,not asserted byte-identical across Pythons."))
        c.write(HERE / "FRESH_OWNED_GONE.json",dict(identities=helper_rows,count=4,
            all_freshly_gone=True,no_global_process_scan=True,all_helpers_joined=True))
        c.write(HERE / "MODEL_PROVENANCE.json",dict(
            retained_role=6,provider="OpenAI",model_id="gpt-6-astra",reasoning_effort="max",
            authority="Original Parent task-tool model selection and verified request,not fresh self-attestation.",
            request="inputs/verified-model-request01.txt",request_sha256=c.sha(HERE / "inputs/verified-model-request01.txt"),
            original_request_commit="c457b5aa6860ca84842343059ed2065697f582d6",
            new_catalog_or_model_probe=False,new_agents=False,model_change=False))
        c.write(HERE / "REPORT.json",dict(
            schema="consumer-deployment02-CPU-source-candidate-report-v1",
            status="SEALED_REVIEW_CANDIDATE_NOT_MAYOR_DEPLOYMENT_ACCEPTED",
            source_freeze=c.load(HERE / "FREEZE.json"),
            exact_dependency=c.CONSUMER_SEAL, Parent_CPU_engineering_notice=notice,
            dependency_aggregate_status=c.load(HERE / "PLAN.json")["dependency_status"],
            deliverable="Actual default-OFF fresh client/launcher/service,external Mayor gate,matching plan/offers/op/capture,owned registration/true-close/privacy paths.",
            validation="9 unique controls/Python resolved:8 generation01 plus1 corrected generation02;10 executions/Python,2 original harness failures preserved.",
            source_copy_bindings=128,additional_original_verifier_bindings=1,
            actual_owned_CPU_identities=4,all_helpers_joined=True,
            original_product_SDK_Source13_changed=False,accepted_successor_runtime_logic_changed=False,
            author_Native_or_network_or_GPU_or_git_or_install=False,
            old42suite_run=False,legacy42_reason_migration="LEGACY42_REASON_MIGRATION.patch",
            not_live_readiness=True,not_live_fairness_proof=True,
            remaining_prerequisites="REMAINING_PREREGISTRATION.json",
            source_differences=["EXACT_SOURCE_DIFF.patch","GENERATION02_EXACT_DIFF.patch"],
            actual_application_at=None,planner_entry_at=None,decision_at=None,
            qualified_model=False,prediction_qualified=False,observer_deployable=False,
            native_mpc_permitted=False,Native14="FAIL",G06="not_closed",G16="pending",
            model_id="gpt-6-astra",reasoning_effort="max"))
    print(json.dumps(dict(label=args.label,inputs=129,owned_identities_gone=4,
                         semantic_projection_equal=True,control_generations=2,Native_execution=False)))


if __name__ == "__main__":
    main()
