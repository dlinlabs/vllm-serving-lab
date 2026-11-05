"""Verify this delivery's frozen sources, mocked artifacts and joined identities only."""

import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET


os.chdir(Path(__file__).resolve().parent)
os.umask(0o077)
freeze = json.loads(Path("CODE-FREEZE.json").read_text())
bindings = []
for name, expected in freeze["new_sources"].items():
    for path in (Path("frozen") / name, Path("../../experiments/mpc-research/resume-mvp") / name):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, str(path)
        bindings.append(dict(path=str(path), sha256=digest))
for path, row in freeze["canonical_sources"].items():
    digest = hashlib.sha256(Path(row["copy"]).read_bytes()).hexdigest()
    assert digest == row["sha256"]
    bindings.append(dict(path=row["copy"], sha256=digest))
for path, expected in freeze["specifications"].items():
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest == expected
    bindings.append(dict(path=path, sha256=digest))
assert hashlib.sha256(Path("../../experiments/mpc-research/resume-mvp/nominal_services.py").read_bytes()).hexdigest() == (
    "5249221b9bd795c8265039502746fe30a394e774a52040f9fcf6261146919bed")
assert Path("generation01.exit-code.txt").read_text().strip() == "0"
results = json.loads(Path("control-results.json").read_text())
budget = json.loads(Path("outer-budget.json").read_text())
assert len(results["attempts"]) == 2 and results["no_retry"] is True
assert results["start"] <= results["end"] < results["deadline"]
rows = [budget["runner"]]
summary = {}
for entry in results["attempts"]:
    assert entry["exit"] == 0 and entry["joined"] is True
    rows.append(entry["identity"])
    base = Path("attempts") / (entry["interpreter"] + "-01")
    assert (base / "exit-code.txt").read_text().strip() == "0"
    value = json.loads((base / "evidence/summary.json").read_text())
    assert value["tests"] == 5 and value["errors"] == value["failures"] == 0
    assert value["actual_servers_HTTP_GPU_NVML_SSH_model_or_asset_calls"] == 0
    junit = ET.parse(base / "evidence/junit.xml").getroot()
    assert junit.attrib == dict(tests="5", failures="0", errors="0")
    assert len(junit.findall("testcase")) == 5
    assert not junit.findall(".//error") and not junit.findall(".//failure")
    cases = {}
    for case, success in (("success", True), ("source-mismatch", False), ("poison", False)):
        report = json.loads((base / "evidence" / case / "output/report.json").read_text())
        assert report["source"]["evidence_kind"] == "CPU_MOCK_ONLY_NOT_HOST_VALIDATION"
        assert report["lifecycle_passed"] is success and report["native_qualification"] is False
        assert report["inference_or_MPC_or_benchmark_result"] is False
        if case == "poison":
            assert report["post_gpu"] is None and report["ownership_receipt"]["status"] == "POISON"
        cases[case] = dict(mocked_lifecycle_passed=success, real_host_validation=False)
    summary[entry["interpreter"]] = dict(python=value["python"], tests=5, failures=0, errors=0, cases=cases)

fresh = []
for row in rows:
    try:
        text = Path(f"/proc/{row['pid']}/stat").read_text()
        actual = int(text[text.rfind(")") + 2:].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        actual = None
    assert actual != row["start_ticks"], "owned control process remains"
    fresh.append(dict(row, current_start_ticks=actual, original_identity_gone=True))
result = dict(
    schema="role3-native-services-driver-delivery-v1", model_id="gpt-6-astra", reasoning_effort="max",
    status="CPU_MOCK_CONTROLS_PASS_ONLY_PENDING_MAYOR_REVIEW_AND_SCOPE_DEVIATION_DISPOSITION",
    source_bindings=bindings, source_sha256=freeze["new_sources"], interpreters=summary,
    aggregate_control_wall_seconds=results["end"] - results["start"],
    source_lines={name: len((Path("frozen") / name).read_text().splitlines()) for name in freeze["new_sources"]},
    generation=1, attempts_per_interpreter=1, failed_test_attempts=0,
    process_closure=fresh, closure_checked_monotonic=time.monotonic(),
    source_scope_deviation_preserved=True, strict_source_search_budget_compliance=False,
    host_execution_held=True, actual_host_bootstrap_or_HTTP_or_GPU_or_service_validation=False,
    actual_offered_inference_or_admin_POSTs=0, Native_B_or_model_qualification=False,
    old_helper_unchanged=True, callback_added=False,
    inprocess_handoff="Mayor uses accepted public NominalServices directly after independent gates",
    full_real_source_model_identity_path_executed=False,
)
with Path("DELIVERY.json").open("x") as stream:
    json.dump(result, stream, sort_keys=True, indent=2)
    stream.write("\n")
print(json.dumps(dict(status=result["status"], source_lines=result["source_lines"],
                      fresh_owned_identities_gone=len(fresh), source_bindings=len(bindings),
                      aggregate_control_wall_seconds=result["aggregate_control_wall_seconds"],
                      controls={name: row["tests"] for name, row in summary.items()}), sort_keys=True))
