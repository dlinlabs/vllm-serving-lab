"""Verify exact-ranking repair evidence and seal only the exclusive revision."""

import difflib
import json
from pathlib import Path
import re
import shlex

from run_attempt import ENVIRONMENT, HERE, ROOT, PYTHONS, sha, verify_inputs


OLD = ROOT / ".runtime/r24-team16-planner-revision01"
CRITIQUE = ROOT / ".runtime/r24-team16-comparison/planner16-critique"
BEFORE = ("before-fix-py314-attempt01", "before-fix-py312-attempt01")
AFTER = ("after-fix-py314-attempt01", "after-fix-py312-attempt01")
TESTED_FILES = (
    "policy_planner16.py", "test_policy_planner16.py",
    "test_parent_numerical_review.py", "test_policy_planner16_numerical_revision.py",
    "test_predecessor_witnesses.py", "test_role7_planner_controls.py",
    "test_policy_planner16_revision02.py", "run_tests.py", "run_attempt.py",
    "freeze_inputs.py",
)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inspect_attempt(path):
    log = (path / "output.log").read_text()
    counts = {
        key: int(match.group(1)) if (
            match := re.search(rf"(\d+) {key}(?:\W|$)", log)) else 0
        for key in ("passed", "failed", "error", "skipped")
    }
    return {
        "command": json.loads((path / "command.json").read_text()),
        "result": json.loads((path / "result.json").read_text()),
        "counts": counts,
        "failed_tests": re.findall(r"^FAILED (.+?)\s*$", log, re.MULTILINE),
        "role7_observations": [
            json.loads(row) for row in re.findall(r"ROLE7_OBSERVATION (\{[^\n]*\})", log)],
        "output_log": str((path / "output.log").relative_to(HERE)),
    }


def launch_environment(extra=False):
    values = ["env", "-u", "PYTHONPATH"]
    for key, value in ENVIRONMENT.items():
        if key != "PYTHONHASHSEED":
            values.append(key + "=" + value)
        if extra and key == "NUMEXPR_NUM_THREADS":
            values.append("NUMEXPR_MAX_THREADS=1")
    return values


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run_from_repository_root")
    destination = HERE.relative_to(ROOT)
    if any((destination / name).exists() for name in (
            "report.json", "source_changes.patch", "commands.txt", "SHA256SUMS")):
        raise SystemExit("refuse_to_overwrite_revision_seal")
    input_count = verify_inputs()
    freeze = json.loads((HERE / "freeze-report.json").read_text())
    plan = json.loads((HERE / "PLAN.json").read_text())
    attempts = {
        path.name: inspect_attempt(path) for path in sorted((HERE / "attempts").iterdir())
        if path.is_dir()
    }
    for label in BEFORE:
        row = attempts[label]
        if (
            row["result"]["returncode"] != 1
            or row["counts"] != {"passed": 3, "failed": 7, "error": 0, "skipped": 0}
            or len(row["role7_observations"]) != 10
            or row["command"]["source_snapshot_sha256"]["policy_planner16.py"]
            != freeze["predecessor_planner_sha256"]
        ):
            raise ValueError("valid_before_fix_reproduction_missing:" + label)
    for label in AFTER:
        row = attempts[label]
        if (
            row["result"]["returncode"] != 0
            or row["counts"] != {"passed": 130, "failed": 0, "error": 0, "skipped": 0}
        ):
            raise ValueError("complete_after_fix_validation_missing:" + label)
        for name in TESTED_FILES:
            if sha(HERE / name) != row["command"]["source_snapshot_sha256"][name]:
                raise ValueError("source_changed_since_test:" + name)
    for label, row in attempts.items():
        if (
            sha(HERE / row["output_log"]) != row["result"]["output_sha256"]
            or row["command"]["input_pins_verified_before"] != input_count
            or row["result"]["input_pins_verified_after"] != input_count
        ):
            raise ValueError("attempt_log_or_pins_changed:" + label)
        for name, expected in row["command"]["source_snapshot_sha256"].items():
            if sha(HERE / "attempts" / label / "sources" / name) != expected:
                raise ValueError("attempt_source_snapshot_changed:" + label + ":" + name)
    unchanged = {}
    for name, expected in freeze["copied_unchanged_sha256"].items():
        if name == "policy_planner16.py":
            continue
        if sha(HERE / name) != expected or sha(OLD / name) != expected:
            raise ValueError("expected_unchanged_file_changed:" + name)
        unchanged[name] = expected
    if (
        sha(HERE / "test_role7_planner_controls.py") != freeze["role7_control_sha256"]
        or sha(HERE / "originals/policy_planner16.py.txt")
        != freeze["predecessor_planner_sha256"]
    ):
        raise ValueError("preserved_control_or_original_source_changed")
    for name in ("CRITIQUE.md", "critique.json", "PLAN.json", "SHA256SUMS"):
        if sha(HERE / "critique-evidence" / name) != sha(CRITIQUE / name):
            raise ValueError("preserved_critique_copy_changed:" + name)
    diff = "".join(difflib.unified_diff(
        (OLD / "policy_planner16.py").read_text().splitlines(keepends=True),
        (HERE / "policy_planner16.py").read_text().splitlines(keepends=True),
        fromfile=".runtime/r24-team16-planner-revision01/policy_planner16.py",
        tofile=str(destination / "policy_planner16.py")))
    with (destination / "source_changes.patch").open("x") as stream:
        stream.write(diff)
    previous_report = json.loads((OLD / "report.json").read_text())
    inherited_parent_evidence = []
    for row in previous_report["defect"]["parent_logs"]:
        evidence = OLD / row["file"]
        if sha(evidence) != row["sha256"]:
            raise ValueError("inherited_parent_evidence_changed")
        inherited_parent_evidence.append({
            **row, "file": str(evidence.relative_to(ROOT)),
        })
    report = {
        "schema": "r24-team16-planner-numerical-revision02-delivery-v1",
        "role": 5, "scope": str(destination),
        "status": "sealed_synthetic_exact_ordering_repair_pending_parent_acceptance",
        "entrypoint": "policy_planner16.PolicyPlanner16",
        "parent_acceptance": False,
        "source_changes": {
            "implementation_files_changed": ["policy_planner16.py"],
            "previous_planner_sha256": freeze["predecessor_planner_sha256"],
            "revised_planner_sha256": sha(HERE / "policy_planner16.py"),
            "exact_diff": "source_changes.patch",
            "exact_diff_sha256": sha(HERE / "source_changes.patch"),
            "unchanged_copied_files": unchanged,
            "previous_102_controls_byte_identical": True,
            "role7_controls_byte_identical": True,
            "role7_controls_sha256": freeze["role7_control_sha256"],
            "new_focused_tests_sha256": sha(HERE / "test_policy_planner16_revision02.py"),
        },
        "frozen_inputs": {
            "count": input_count, "manifest": "INPUT_SHA256SUMS",
            "manifest_sha256": sha(HERE / "INPUT_SHA256SUMS"),
            "plan_sha256": sha(HERE / "PLAN.json"),
            "predecessor_seal_sha256": sha(OLD / "SHA256SUMS"),
            "critique_seal_sha256": sha(CRITIQUE / "SHA256SUMS"),
            "all_previous_pinned_seals_and_files_unchanged": True,
        },
        "parent_evidence_provenance": {
            "commit_reported_by_parent": plan["parent_evidence_commit_reported"],
            "git_verification_performed": False,
            "map": plan["parent_evidence_map"],
            "map_sha256": freeze["parent_map_sha256"],
            "planner_only_proof_copies_verified": freeze["planner_only_parent_proofs_checked"],
            "unrelated_role2_source_or_proofs_read": False,
            "parent_reported_planner_revision01_counts_per_python": {
                "intentional_defect_failures": 7, "passes": 3,
            },
            "all_cross_role_proofs_independently_revalidated": False,
        },
        "findings": {
            "I1": {
                "classification": "decision_ordering_defect",
                "mechanism": "float_sum_absorbs_exact_positive_terms_into_nonzero_common_term",
                "weights": ["32*math.ulp(0.0)", "2**-60"],
                "keys": ["natural_slo_good", "cohort_wall"],
                "affected_policies": ["mpc", "strong_one_move", "training_fixed_selection"],
                "oracle_path": [32, 32], "before_accepted_path": [16, 16],
                "new_acceptance_tests_require_exact_path_not_blanket_refusal": True,
                "below_dominant_term_ulp_not_native_performance_evidence": True,
            },
            "I2": {
                "classification": "conditional_shared_action_set_baseline_weakening",
                "witness_budgets": [16, 24, 32],
                "before_frozen_budget": 16, "strongest_constant_budget": 24,
                "repair": "enumerate_every_contract_constant_and_bind_canonical_candidate_set",
                "actual_comparison_already_locks_all_arms_to": [16, 32],
                "actual_inflated_native_win_claimed": False,
            },
            "I3": {
                "classification": "existing_runtime_integration_constraint_not_new_planner_defect",
                "requirement": "actual_same_domain_dequeue_clock_or_external_queue_IPC_delivery_TTL",
                "default_acquisition_clock_characterization_retained": True,
            },
            "I4": {
                "classification": "existing_comparison_integration_constraint_not_new_planner_defect",
                "requirement": (
                    "Actual fixed arms apply frozen constants independently of model success/"
                    "cost; retain holds, failures, timing and all offers in denominators."),
            },
        },
        "repair": {
            "ranking_arithmetic": "exact-rational-frozen-binary-inputs-v1",
            "exact_scope": "positive_frozen_weights_and_finite_frozen_per_world_numeric_metrics",
            "exact_key_encoding": ["reduced_numerator_decimal_string", "positive_denominator_decimal_string"],
            "floating_objectives_and_candidate_summaries": "display_only_not_ordering_keys",
            "old_product_underflow_and_boundary_refusals_retained": True,
            "exact_within_panel_training_aggregation": True,
            "complete_validation_before_sort_and_best_row_replacement": True,
            "all_positive_weight_worlds_in_accepted_plan_exact_keys": True,
            "no_world_deleted_or_ratio_changed_to_break_ties": True,
            "corrected_prior_claim": (
                "Preserved entries and exact weight ratios did not prevent prior float sums "
                "from erasing a positive world's decision influence."),
            "physical_real_arithmetic_or_universal_input_acceptance_claimed": False,
            "original_mathematical_objective_and_strong_baselines_preserved": True,
            "explicit_fixed16_fixed32_and_legacy_caller_choice_interface_retained": True,
            "old_training_artifacts_rejected_by_new_arithmetic_contract": True,
            "source_version_and_full_cooperative_deadline_guards_retained": True,
            "bounds": plan["unchanged_contract"],
        },
        "refusal_cases": [
            "Previous zero/nonfinite or inexact underflowed weighted objective product.",
            "Previous invalid, partial, mixed-source or mutated complete common panel.",
            "Previous stale/reused source or changed observation version.",
            "Wrong/missing cached exact objective keys during train-panel validation.",
            "Frozen candidate set omits a shared action or contract configuration differs.",
            "Exact arithmetic/sorting pushes entry+solver+actuation beyond deadline.",
            "Observation version changes during exact arithmetic or output preparation.",
        ],
        "inherited_product_rounding_evidence": {
            "previous_report": str((OLD / "report.json").relative_to(ROOT)),
            "previous_report_sha256": sha(OLD / "report.json"),
            "parent_logs": inherited_parent_evidence,
            "parent_attempts01_02_are_not_algorithm_failures": True,
            "parent_attempt03_is_valid_algorithm_counterexample": True,
            "original_parent_oracle_unchanged_and_passing": True,
        },
        "validation": {
            "before_fix_attempts": list(BEFORE), "after_fix_attempts": list(AFTER),
            "before_fix": {
                "python314": {"failed": 7, "passed": 3},
                "python312": {"failed": 7, "passed": 3},
            },
            "after_fix": {"python314_passed": 130, "python312_passed": 130},
            "unchanged_previous_controls": 102, "unchanged_role7_controls": 10,
            "new_focused_cases": 18,
            "new_focused_breakdown": {
                "accepted_exact_absorption_order": 6,
                "exact_training_and_frozen_heldout": 3,
                "canonical_full_action_set_training": 3,
                "partial_candidate_artifact_refusal": 1,
                "tampered_exact_key_refusal": 1,
                "exact_arithmetic_and_sort_deadline_refusals": 2,
                "version_change_during_exact_ordering": 1,
                "primary_good_count_over_secondary_wall": 1,
            },
            "all_attempts": attempts,
            "all_observed_failures_preserved": True,
            "new_revision_harness_errors": 0,
            "final_tested_sha256": {name: sha(HERE / name) for name in TESTED_FILES},
            "algorithmic_fixtures_use_frozen_clock": True,
            "native_solver_speed_or_performance_claim": False,
        },
        "native_qualification": previous_report["native_qualification"],
        "execution": {
            "environment_overrides": ENVIRONMENT,
            "before_fix_py312_outer_extra_environment": {"NUMEXPR_MAX_THREADS": "1"},
            "PYTHONPATH_unset": True, "python_B": True, "nice": 19,
            "threads_per_numerical_library": 1,
            "pytest_plugin_autoload_and_cache_disabled": True,
            "owned_test_children_synchronously_joined": True,
            "owned_running_helpers": [],
            "agents": 0, "broadcasts": 0, "git_operations": 0, "network_experiments": 0,
            "GPU_use": 0, "live_CAS_HTTP_actuation": 0, "actual_D_MCMC_transitions": 0,
            "product_native_runtime_or_comparison_peer_edits": 0,
            "post_seal_read_only_verification": "run_verify_delivery_with_both_prescribed_pythons",
        },
        "remaining_blockers": [
            "Synthetic exact-order controls do not qualify real native MPC, predict wins or identify an unpredictability cause.",
            "Native14 remains rejected under unchanged 1.01/400/400 gates; no accepted posterior is supplied.",
            "Shared normalized native/client law, public-only filter, heldout calibration/PIT and native action response remain unqualified.",
            "Role6/Role7 must meet I3/I4 integration requirements; no peer implementation was changed.",
            "Parent independently reviews, accepts, integrates and commits; this delivery is not acceptance.",
        ],
    }
    write_json(destination / "report.json", report)
    launcher = [*launch_environment(), "nice", "-n", "19", ".venv/bin/python3.14", "-B"]
    cd = "cd " + shlex.quote(str(ROOT)) + " && "
    commands = [
        "# Executed commands; attempt labels are once-only. Stdout tails did not truncate stored logs.",
        "# Complete child argv/environment, input hashes and outputs are in attempts/.",
        cd + shlex.join([*launcher, str(destination / "freeze_inputs.py")]),
    ]
    for label, row in sorted(
            attempts.items(), key=lambda item: item[1]["command"]["started_unix_s"]):
        command = row["command"]
        version = "314" if command["command"][3].endswith("python3.14") else "312"
        wrapper = [
            *launch_environment(extra=label == "before-fix-py312-attempt01"),
            "nice", "-n", "19", ".venv/bin/python3.14", "-B",
            str(destination / "run_attempt.py"), label, version, command["suite"],
        ]
        tail = 45 if label in BEFORE else 90
        commands.append("set -o pipefail; " + cd + shlex.join(wrapper) + f" | tail -n {tail}")
    commands.append(cd + shlex.join([*launcher, str(destination / "seal_revision.py")]))
    commands.append("# Read-only post-seal verification commands:")
    commands.extend(cd + shlex.join([
        *launch_environment(), "nice", "-n", "19",
        ".venv/bin/python3.14" if version == "314" else str(python), "-B",
        str(destination / "verify_delivery.py")]) for version, python in PYTHONS.items())
    with (destination / "commands.txt").open("x") as stream:
        stream.write("\n".join(commands) + "\n")
    files = []
    for path in sorted(HERE.rglob("*")):
        if path.is_symlink() or path.name in ("__pycache__", ".pytest_cache"):
            raise ValueError("unexpected_symlink_or_cache")
        if path.is_file() and path != HERE / "SHA256SUMS":
            files.append(f"{sha(path)}  {path.relative_to(HERE)}\n")
    with (destination / "SHA256SUMS").open("x") as stream:
        stream.writelines(files)
    print(json.dumps({
        "status": report["status"], "sealed_files": len(files),
        "input_pins_verified": input_count,
        "revised_planner_sha256": sha(HERE / "policy_planner16.py"),
        "manifest_sha256": sha(HERE / "SHA256SUMS"),
        "passed_per_python": 130,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
