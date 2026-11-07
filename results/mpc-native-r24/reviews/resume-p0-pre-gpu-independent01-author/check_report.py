"""Read-only consistency check of REPORT.json for the Role4 final pre-GPU source review capsule."""

import hashlib
import json
import math
import sys
from pathlib import Path

CAPSULE = Path(__file__).resolve().parent
WORKTREE = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
RUNTIME_SEALS = {
    "CHECKLIST.json": WORKTREE / ".runtime/r24-pre-gpu-review-plan-role4/CHECKLIST.json",
    "B2 REPORT.json": WORKTREE / ".runtime/r24-pre-gpu-b2-review-role4/REPORT.json",
}
ROW_IDS = ["B3-%d" % i for i in range(1, 12)]
failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict(raw):
    def constant(value):
        raise ValueError("nonfinite constant " + value)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key " + key)
            result[key] = value
        return result

    return json.loads(raw, parse_constant=constant, object_pairs_hook=pairs)


raw = (CAPSULE / "REPORT.json").read_bytes()
check(all(byte < 128 for byte in raw), "REPORT.json is not ASCII")
report = strict(raw)
for key in ("schema", "status", "verdict", "pins", "rows", "findings", "checklist_errata", "witnesses",
            "preconditions", "nonclaims", "limitations", "disclosures"):
    check(key in report, "missing top-level key " + key)

verdict = report["verdict"]
check(verdict["native_qualification"] is False, "native_qualification must be false")
check(verdict["gpu_authorized_by_this_report"] is False, "GPU authorization must be false")
check(verdict["performance_or_mpc_win_claim"] is False, "no performance claim allowed")

rows = report["rows"]
check([row["id"] for row in rows] == ROW_IDS, "rows must be exactly B3-1..B3-11 in order")
for row in rows:
    allowed = {"PASS", "PASS_WITH_GAP", "FAIL"} if row["level"] == "MUST" else {"PASS", "PARTIAL", "FAIL"}
    check(row["status"] in allowed, "bad status for " + row["id"])
must_fail = [row["id"] for row in rows if row["level"] == "MUST" and row["status"] == "FAIL"]
blocking = [f["id"] for f in report["findings"] if f["severity"].startswith("BLOCKING")]
check(must_fail == ["B3-5"], "expected exactly B3-5 MUST FAIL, got %r" % must_fail)
check(blocking == verdict["blocking_findings"] == ["F1"], "blocking findings mismatch")
check((verdict["overall"] == "FAIL") == bool(must_fail or blocking), "overall verdict inconsistent with rows/findings")
for finding in report["findings"]:
    if finding["id"] in blocking:
        check("smallest_fix" in finding and "regression_control" in finding, finding["id"] + " lacks fix/control")

caller = WORKTREE / "experiments/mpc-research/resume-mvp/native_nominal_qualification.py"
line_111 = caller.read_text().splitlines()[110]
fix = report["findings"][0]["smallest_fix"]
check(fix["replace_line_111"] == line_111, "F1 fix anchor does not equal frozen caller line 111")
check(len(fix["with_lines"]) == 2 and "<= 1e-9" in fix["with_lines"][1], "F1 replacement text unexpected")

pins = report["pins"]
verified = 0
for group in ("reviewed_consumers", "frozen_diffs", "unchanged_dependencies_hash_verified",
              "author_capsule_seals_SHA256SUMS_digest"):
    for relative, expected in pins[group].items():
        check(sha(WORKTREE / relative) == expected, "pin mismatch " + relative)
        verified += 1
for relative, expected in pins["prior_role4_seals_SHA256SUMS_digest"].items():
    path = RUNTIME_SEALS.get(relative, WORKTREE / relative)
    check(sha(path) == expected, "seal/report mismatch " + relative)
    verified += 1

witness = report["witnesses"][0]
check(sha(CAPSULE / witness["prereg"]["file"]) == witness["prereg"]["sha256"], "prereg hash mismatch")
check(sha(CAPSULE / witness["script"]["file"]) == witness["script"]["sha256"], "witness script hash mismatch")
for run in witness["runs"]:
    check(sha(CAPSULE / run["log"]) == run["log_sha256"], "witness log hash mismatch " + run["log"])
    check(sha(CAPSULE / run["result"]) == run["result_sha256"], "witness result hash mismatch " + run["result"])
    result = strict((CAPSULE / run["result"]).read_bytes())
    residue, zero = result["P2_residue"], result["P2_zero_control"]
    check(residue["error"]["message"] == "original work deadline exhausted", "residue run did not refuse")
    check(residue["caller_drains"] is None and residue["mock_GETs"] >= 2, "residue run drained or too short")
    check(residue["metrics_cost_served"] == 2.7755575615628914e-17 and math.isfinite(residue["metrics_cost_served"]),
          "unexpected served residue")
    check(zero["caller_drains"]["final"]["readings"] == 1 and zero["error"] is None, "zero control did not drain")
    check(all(residue["hypothetical_fixed_predicate_on_readings"]), "fixed predicate not true on residue readings")

summary = {"python": sys.version.split()[0], "pins_verified": verified, "rows": len(rows),
           "must_fail": must_fail, "blocking": blocking, "overall": verdict["overall"],
           "failures": failures}
print(json.dumps(summary, sort_keys=True))
print("CHECK_REPORT_PASS" if not failures else "CHECK_REPORT_FAIL")
sys.exit(0 if not failures else 1)
