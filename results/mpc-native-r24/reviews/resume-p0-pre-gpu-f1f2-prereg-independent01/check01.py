import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRED_SEAL = "0f0b0e7cb489bbfa3102bc4dbab5262151095e353e7e1735dd35b647417ac403"
failures = []


def check(ok, text):
    if not ok:
        failures.append(text)


def no_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def refuse_constant(name):
    raise ValueError("non-finite constant " + name)


raw = (HERE / "PREREG.json").read_bytes()
check(raw.isascii(), "PREREG.json not ASCII")
prereg = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates, parse_constant=refuse_constant)
check(prereg["verdict"] is None, "prereg carries a verdict")
check(prereg["native_qualification"] is False and prereg["gpu_authorized"] is False, "authority flags")
check([row["id"] for row in prereg["must"]] == [f"R{i}" for i in range(1, 11)], "must rows R1-R10")
check(prereg["predecessor"]["seal_sha256"] == PRED_SEAL, "predecessor seal text")
seal = HERE.parent / "r24-pre-gpu-final-review-role4" / "SHA256SUMS"
check(hashlib.sha256(seal.read_bytes()).hexdigest() == PRED_SEAL, "predecessor seal digest")
check(len(seal.read_text().splitlines()) == prereg["predecessor"]["members"], "predecessor member count")

residue = (0.1 + 0.2) - 0.1 - 0.2
above = math.nextafter(1e-9, math.inf)


def gateway_idle(value):
    return math.isclose(value, 0.0, rel_tol=1e-12, abs_tol=1e-9)


check(residue == 2.7755575615628914e-17, "residue value")
check(residue != 0 and residue > 0, "residue positive")
for value, expected in ((0.0, True), (residue, True), (1e-9, True), (above, False), (1e-6, False), (0.1, False)):
    check(gateway_idle(value) is expected, f"gateway isclose {value!r}")
    check((value <= 1e-9) is expected, f"inclusive predicate {value!r}")

print(json.dumps({"python": sys.version.split()[0], "residue": residue, "above": above,
                  "failures": failures}, sort_keys=True))
print("PREREG_CHECK_FAIL" if failures else "PREREG_CHECK_PASS")
sys.exit(1 if failures else 0)
