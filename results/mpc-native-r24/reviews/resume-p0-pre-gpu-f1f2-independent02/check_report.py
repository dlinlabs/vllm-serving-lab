import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
RES = W / "experiments/mpc-research/resume-mvp"
failures = []


def check(ok, text):
    if not ok:
        failures.append(text)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def refuse_constant(name):
    raise ValueError("non-finite constant " + name)


raw = (HERE / "REPORT.json").read_bytes()
check(raw.isascii(), "REPORT.json not ASCII")
report = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates, parse_constant=refuse_constant)
verdict = report["verdict"]
check(verdict["overall"] == "PASS_SOURCE_F1F2", "overall")
check(verdict["role4_reviewed_scope_cpu_source_blockers"] == [], "blockers")
check(verdict["native_qualification"] is False and verdict["gpu_authorized"] is False, "authority flags")
check([row["id"] for row in report["rows"]] == [f"R{i}" for i in range(1, 11)], "row ids")
check(all(row["status"] == "PASS" for row in report["rows"]), "row statuses")
check([row["id"] for row in report["should"]] == ["S1", "S2", "S3", "S4"], "should ids")
check(report["combined_pre_gpu_source_disposition"]["role4_cpu_source_blockers_remaining"] == [], "combined")

expected = {
    RES / "native_nominal_qualification.py": "9018acee9f659c821929d69443621bfd482edb5a6dbed15ae8bcba5a140894ec",
    RES / "test_native_nominal_qualification.py": "76b7e93688f0ba766456368716ded4cfe1809491c27f32fa5e76e812d2959d83",
    HERE / "inputs/new/native_nominal_qualification.py": "9018acee9f659c821929d69443621bfd482edb5a6dbed15ae8bcba5a140894ec",
    HERE / "inputs/new/test_native_nominal_qualification.py": "76b7e93688f0ba766456368716ded4cfe1809491c27f32fa5e76e812d2959d83",
    HERE / "inputs/old/native_nominal_qualification.py": "629caa23cb7d7cdedc4103499e44ab51968bac0ead3cc640ed4601d87d8e4122",
    HERE / "inputs/old/test_native_nominal_qualification.py": "7d087a88a1c6fed3e4a71e52739294f77bfdb00572b5f311c1b4a95c5203acac",
    HERE / "inputs/role6-correction.diff": "eda50b2a681ed52f4e0a8acb2fef492a0dbc797149538cca291e7b9ceaa6c7b0",
    W / ".runtime/r24-pre-gpu-b3-f1-role6/correction.diff": "eda50b2a681ed52f4e0a8acb2fef492a0dbc797149538cca291e7b9ceaa6c7b0",
    W / ".runtime/r24-pre-gpu-b3-f1-role6/SHA256SUMS": "0dbfc4601693147588342f28904edbc5a81d8877290463c1237ac39053230efa",
    W / ".runtime/r24-pre-gpu-b3-role6/SHA256SUMS": "b08511ad3048fdecaf4bbc45eeff198ef0bcc958255b86b96ec681777c037b0e",
    W / ".runtime/r24-pre-gpu-b2-role3/SHA256SUMS": "3f7e6804a0d5007201ed4c742a430d5e260d39b60acfe19b4859846eddcaa26f",
    W / ".runtime/r24-pre-gpu-final-review-role4/SHA256SUMS": "0f0b0e7cb489bbfa3102bc4dbab5262151095e353e7e1735dd35b647417ac403",
    W / ".runtime/r24-pre-gpu-f1f2-recheck-prereg-role4/SHA256SUMS": "4737fc5e1a1a485c03321d414d86e9c3f7b2966e9882a67c163999e5c1bec731",
    W / ".runtime/r24-pre-gpu-f1f2-recheck-prereg-role4/PREREG.json": "011165b578bc3e85894344c7f5e4400bd85d5d152e7aa25e8a284e7c6869b462",
}
for name, digest in report["pins"]["unchanged"].items():
    expected[(W / name) if name == "gateway.py" else (RES / name)] = digest
for path, digest in expected.items():
    check(sha(path) == digest, f"pin {path}")

new = (HERE / "inputs/new/native_nominal_qualification.py").read_text().splitlines()
old = (HERE / "inputs/old/native_nominal_qualification.py").read_text().splitlines()
test = (HERE / "inputs/new/test_native_nominal_qualification.py").read_text().splitlines()
check(old[110] == "        if all(metrics[name] == 0 for name in DRAIN_FIELDS):", "old line 111")
check(new[110] == '        if (metrics["current_in_flight"] == 0 and metrics["current_waiting"] == 0', "new 111")
check(new[111] == '                and metrics["current_admitted_cost"] <= 1e-9):', "new 112")
check(new[106:110] == old[106:110], "guard lines 107-110 unchanged")
check(test[23:26] == ['ROOT = Path(os.environ["NATIVE_SMOKE_TEST_OUTPUT"])',
                      'ROOT = ROOT.with_name(ROOT.name + "-qualification")',
                      "ROOT.mkdir(parents=True, exist_ok=True)"], "test 24-26")
check(test[359] == "    def test_drain_tolerates_only_idle_cost_residue(self):", "test 360")

check("ROUNDTRIP-EXACT" in (HERE / "roundtrip01.log").read_text(), "roundtrip")
check("CALLER-HUNK-IDENTICAL" in (HERE / "hunkcompare01.log").read_text(), "caller hunk compare")
check("TEST-HUNKS-IDENTICAL" in (HERE / "hunkcompare02.log").read_text(), "test hunk compare")

w2_sha = sha(HERE / "w2.py")
check(report["witness_w2"]["script_sha256"].startswith(w2_sha), "w2 sha in report")
check(w2_sha in (HERE / "commands.log").read_text(), "w2 sha logged before execution")
expect = {
    "fixed": {"c1": "drained", "c2": "drained", "c3": "deadline", "c4": "deadline",
              "c5": "deadline", "c6": "deadline", "c7": "fallback", "c8": "invalid"},
    "old": {"c1": "deadline", "c2": "deadline", "c3": "deadline", "c4": "deadline",
            "c5": "deadline", "c6": "deadline", "c7": "fallback", "c8": "invalid"},
}
for tag in ("py3.14.4", "py3.12.16"):
    result = json.loads((HERE / "w2-out" / tag / "result.json").read_text(), parse_constant=refuse_constant)
    check(result["status"] == "CONFIRMED" and result["mismatches"] == [], f"w2 {tag} status")
    check(result["pins_unchanged"] is True, f"w2 {tag} pins")
    for version, table in expect.items():
        for case, outcome in table.items():
            check(result["outcomes"][version][case]["outcome"] == outcome, f"w2 {tag} {version} {case}")

print(json.dumps({"python": sys.version.split()[0], "pins_verified": len(expected), "failures": failures},
                 sort_keys=True))
informational = {name: (W / name).is_file() and sha(W / name) == digest
                 for name, digest in report["pins"]["hashed_not_reviewed"].items()}
print(json.dumps({"hashed_not_reviewed_still_matching_informational": informational}, sort_keys=True))
print("CHECK_REPORT_FAIL" if failures else "CHECK_REPORT_PASS")
sys.exit(1 if failures else 0)
