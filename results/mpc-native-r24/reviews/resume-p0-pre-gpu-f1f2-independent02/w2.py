"""Role4 W2, preregistered as w2_plan_only_if_needed in sealed
.runtime/r24-pre-gpu-f1f2-recheck-prereg-role4/PREREG.json (SHA256SUMS digest 4737fc5e).

Drives the frozen fixed caller drain() (9018acee) and the frozen predecessor drain() (629caa23)
through their unchanged observe() with httpx.MockTransport /metrics bodies. CPU/mock only:
no sockets, no GPU, no Train/Eval data. Exit 0 CONFIRMED, 3 REFUTED, 2 ERROR.
"""

import asyncio
import hashlib
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

W = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
RES = W / "experiments/mpc-research/resume-mvp"
CAPSULE = Path(__file__).resolve().parent
FIXED = CAPSULE / "inputs/new/native_nominal_qualification.py"
OLD = CAPSULE / "inputs/old/native_nominal_qualification.py"
PINS = {
    FIXED: "9018acee9f659c821929d69443621bfd482edb5a6dbed15ae8bcba5a140894ec",
    OLD: "629caa23cb7d7cdedc4103499e44ab51968bac0ead3cc640ed4601d87d8e4122",
    RES / "native_nominal_qualification.py": "9018acee9f659c821929d69443621bfd482edb5a6dbed15ae8bcba5a140894ec",
    RES / "native_services_smoke.py": "585e606f7dd60d5154f0d3f0f8bbac9094732e00b8e4ad2a60d93bdbb6f69d4c",
}
RESIDUE = (0.1 + 0.2) - 0.1 - 0.2
ABOVE = math.nextafter(1e-9, math.inf)


def body(cost, in_flight=0, waiting=0, fallback=0):
    return {"current_in_flight": in_flight, "current_waiting": waiting,
            "current_admitted_cost": cost, "tokenizer_fallback_requests": fallback}


CASES = {
    "c1": body(RESIDUE),
    "c2": body(1e-9),
    "c3": body(ABOVE),
    "c4": body(1e-6),
    "c5": body(0.0, in_flight=1),
    "c6": body(0.0, waiting=1),
    "c7": body(0.0, fallback=1),
    "c8": body(-1e-12),
}
EXPECTED = {
    "fixed": {"c1": "drained", "c2": "drained", "c3": "deadline", "c4": "deadline",
              "c5": "deadline", "c6": "deadline", "c7": "fallback", "c8": "invalid"},
    "old": {"c1": "deadline", "c2": "deadline", "c3": "deadline", "c4": "deadline",
            "c5": "deadline", "c6": "deadline", "c7": "fallback", "c8": "invalid"},
}
MESSAGES = {"deadline": "original work deadline exhausted",
            "fallback": "tokenizer fallback after traffic",
            "invalid": "invalid drain/fallback metric"}


def hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in PINS}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(error, readings, drains):
    if error is None:
        if drains is not None and readings == 1 and drains.get("final", {}).get("readings") == 1:
            return "drained"
        return "other"
    if error["type"] != "ProbeError":
        return "other"
    for kind, message in MESSAGES.items():
        if error["message"] == message:
            if kind == "deadline" and readings >= 2 and drains is None:
                return kind
            if kind != "deadline" and readings == 1 and drains is None:
                return kind
    return "other"


async def run_case(caller, httpx, case_dir, metrics):
    gets = []

    def handler(request):
        gets.append(request.url.path)
        return httpx.Response(200, content=json.dumps(metrics, allow_nan=False).encode(),
                              headers={"content-type": "application/json"})

    case_dir.mkdir(parents=True)
    owner = types.SimpleNamespace(deadline=time.monotonic() + 0.6, reserve=0.1, check=lambda: None)
    report = {"caller_errors": []}
    error = None
    started = time.monotonic()
    async with httpx.AsyncClient(base_url="http://127.0.0.1:18001", transport=httpx.MockTransport(handler),
                                 trust_env=False, follow_redirects=False) as client:
        try:
            await caller.drain(client, owner=owner, deadline=owner.deadline, reserve=owner.reserve,
                               output=case_dir, key="w2-redaction-key", report=report, phase="final")
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc),
                     "is_services_ProbeError": isinstance(exc, caller.services.ProbeError)}
    elapsed = time.monotonic() - started
    observations = report.get("caller_observations", [])
    values = [o["value"] for o in observations if "value" in o]
    drains = report.get("caller_drains")
    first = case_dir / "final-metrics-0001.json"
    outcome = classify(error, len(values), drains)
    return {
        "served": metrics, "served_cost_repr": repr(metrics["current_admitted_cost"]),
        "outcome": outcome, "error": error, "mock_GETs": len(gets), "readings_with_value": len(values),
        "caller_drains": drains, "elapsed_s": elapsed,
        "evidence_files_written": len(list(case_dir.iterdir())),
        "extra_drain_record_keys": None if drains is None else sorted(drains["final"]),
        "extra_drain_record_strict_json": None if drains is None else bool(json.dumps(drains, allow_nan=False)),
        "extra_first_evidence_has_served_cost_text": first.exists()
        and json.dumps(metrics["current_admitted_cost"]) in first.read_text(),
    }


def write_result(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n")


def main():
    tag = "py" + ".".join(map(str, sys.version_info[:3]))
    out = CAPSULE / "w2-out" / tag
    out.mkdir(parents=True)
    before = hashes()
    result = {"schema": "role4-W2-result-v1", "prereg": "r24-pre-gpu-f1f2-recheck-prereg-role4/PREREG.json",
              "python": sys.version, "executable": sys.executable,
              "dont_write_bytecode": sys.flags.dont_write_bytecode, "isolated": sys.flags.isolated,
              "label": "CPU/mock witness only; not Native evidence", "pins_before": before,
              "pins_ok_before": all(before[str(p)] == s for p, s in PINS.items()),
              "residue_repr": repr(RESIDUE), "above_repr": repr(ABOVE)}
    if not result["pins_ok_before"]:
        result["status"] = "ERROR"
        write_result(out / "result.json", result)
        print("ERROR pin mismatch before import")
        return 2
    sys.path.insert(0, str(RES))
    import httpx
    callers = {"fixed": load("w2_caller_fixed", FIXED), "old": load("w2_caller_old", OLD)}
    services_file = str(Path(callers["fixed"].services.__file__).resolve())
    result["services_module"] = services_file
    result["services_same_object"] = callers["fixed"].services is callers["old"].services
    outcomes = {}
    for version, caller in callers.items():
        outcomes[version] = {}
        for case, metrics in CASES.items():
            outcomes[version][case] = asyncio.run(run_case(caller, httpx, out / version / case, metrics))
    result["outcomes"] = outcomes
    after = hashes()
    result["pins_after"] = after
    result["pins_unchanged"] = after == before
    mismatches = [f"{version}/{case}: expected {expected} got {outcomes[version][case]['outcome']}"
                  for version, table in EXPECTED.items() for case, expected in table.items()
                  if outcomes[version][case]["outcome"] != expected]
    if services_file != str(RES / "native_services_smoke.py"):
        mismatches.append("services module not from RES: " + services_file)
    if not result["pins_unchanged"]:
        mismatches.append("pins changed during witness")
    result["mismatches"] = mismatches
    result["status"] = "CONFIRMED" if not mismatches else "REFUTED"
    write_result(out / "result.json", result)
    print(result["status"], json.dumps(mismatches))
    for version in outcomes:
        print(version, json.dumps({case: [r["outcome"], r["readings_with_value"], round(r["elapsed_s"], 3)]
                                   for case, r in outcomes[version].items()}, sort_keys=True))
    fixed_c1 = outcomes["fixed"]["c1"]
    print("extra fixed c1", fixed_c1["extra_drain_record_keys"], fixed_c1["extra_drain_record_strict_json"],
          fixed_c1["extra_first_evidence_has_served_cost_text"])
    return 0 if result["status"] == "CONFIRMED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
