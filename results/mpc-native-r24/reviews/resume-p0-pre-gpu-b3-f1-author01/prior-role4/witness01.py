"""Role4 W1 (preregistered in PREREG-W1.json): frozen drain() exact cost predicate vs gateway float residue.

CPU/mock only: httpx.MockTransport, no sockets, no GPU, no Train/Eval data. Exit 0 CONFIRMED, 3 REFUTED, 2 ERROR.
"""

import asyncio
import hashlib
import itertools
import json
import sys
import time
import types
from pathlib import Path

W = Path("/home/daniel_lin/gt/pr32/mayor/mpc-p0-pr-20261105")
RES = W / "experiments/mpc-research/resume-mvp"
CAPSULE = Path(__file__).resolve().parent
PINS = {
    RES / "native_nominal_qualification.py": "629caa23cb7d7cdedc4103499e44ab51968bac0ead3cc640ed4601d87d8e4122",
    RES / "native_services_smoke.py": "585e606f7dd60d5154f0d3f0f8bbac9094732e00b8e4ad2a60d93bdbb6f69d4c",
    W / "gateway.py": "0b1a80ade258d8875ba80a072c4f360bfa5d4a7ac312bd222c0571380b341570",
}
EXPECTED_ERROR = {"type": "ProbeError", "message": "original work deadline exhausted"}


def hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in PINS}


def admit(current, cost):
    return current + cost  # replica of gateway 456 and 467


def finish(current, cost):
    return max(0.0, current - cost)  # replica of gateway 518-521


def overlapped(costs, order):
    current = 0.0
    for cost in costs:
        current = admit(current, cost)
    for index in order:
        current = finish(current, costs[index])
    return current


def part1():
    out = {"label": "SYNTHETIC literal costs; replica arithmetic, not gateway execution"}
    pair = (0.1, 0.2)
    out["pair_costs"] = list(pair)
    out["pair_fifo_residue"] = overlapped(pair, (0, 1))
    out["pair_lifo_residue"] = overlapped(pair, (1, 0))
    tokens = [(100, 128), (250, 64), (37, 512), (1000, 16), (512, 256)]
    costs = [max(0.1, 0.259 + 0.000758 * i + 0.0052083333 * o) for i, o in tokens]
    out["synthetic_tokens_in_out"] = [list(t) for t in tokens]
    out["synthetic_costs"] = costs
    residues = [overlapped(costs, order) for order in itertools.permutations(range(5))]
    positive = [r for r in residues if r > 0]
    out["five_overlapped_all_finish_orders"] = {
        "orders": len(residues), "positive": len(positive), "exact_zero": sum(r == 0.0 for r in residues),
        "min_positive": min(positive) if positive else None, "max_positive": max(positive) if positive else None}
    pairs = [overlapped((costs[a], costs[b]), order)
             for a, b in itertools.combinations(range(5), 2) for order in ((0, 1), (1, 0))]
    pair_positive = [r for r in pairs if r > 0]
    out["pairs_fifo_and_lifo"] = {
        "cases": len(pairs), "positive": len(pair_positive), "exact_zero": sum(r == 0.0 for r in pairs),
        "max_positive": max(pair_positive) if pair_positive else None}
    current, sequential = 0.0, []
    for cost in costs:
        current = finish(admit(current, cost), cost)
        sequential.append(current)
    out["sequential_non_overlap_after_each_finish"] = sequential
    return out


async def part2(case_dir, cost):
    import httpx
    if str(RES) not in sys.path:
        sys.path.insert(0, str(RES))
    import native_nominal_qualification as caller
    if Path(caller.__file__).resolve() != RES / "native_nominal_qualification.py":
        raise RuntimeError("wrong caller module imported: " + str(caller.__file__))
    gets = []

    def handler(request):
        gets.append({"path": request.url.path, "at": time.monotonic()})
        body = {"current_in_flight": 0, "current_waiting": 0, "current_admitted_cost": cost,
                "tokenizer_fallback_requests": 0, "admission_mode": "cost_aware"}
        return httpx.Response(200, content=json.dumps(body, allow_nan=False).encode(),
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
                               output=case_dir, key="w1-redaction-key", report=report, phase="final")
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.monotonic() - started
    observations = report.get("caller_observations", [])
    values = [o["value"] for o in observations if "value" in o]
    return {
        "metrics_cost_served": cost, "metrics_cost_repr": repr(cost), "error": error,
        "mock_GETs": len(gets), "observations": len(observations), "readings_with_value": len(values),
        "served_cost_reprs_seen": sorted({repr(v["current_admitted_cost"]) for v in values}),
        "counters_all_zero": all(v["current_in_flight"] == 0 and v["current_waiting"] == 0
                                 and v["tokenizer_fallback_requests"] == 0 for v in values),
        "observation_error_messages": [o["error"]["message"] for o in observations if "error" in o],
        "caller_drains": report.get("caller_drains"), "caller_errors": report["caller_errors"],
        "evidence_files_written": len(list(case_dir.iterdir())), "elapsed_s": elapsed,
        "hypothetical_fixed_predicate_on_readings": [
            v["current_in_flight"] == 0 and v["current_waiting"] == 0 and v["tokenizer_fallback_requests"] == 0
            and abs(v["current_admitted_cost"]) <= 1e-9 for v in values],
    }


def write_result(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n")


def main():
    tag = "py" + ".".join(map(str, sys.version_info[:3]))
    out = CAPSULE / "witness01-out" / tag
    out.mkdir(parents=True)
    before = hashes()
    result = {"schema": "role4-W1-result-v1", "prereg": "PREREG-W1.json", "python": sys.version,
              "executable": sys.executable, "dont_write_bytecode": sys.flags.dont_write_bytecode,
              "isolated": sys.flags.isolated, "label": "CPU/mock witness only; not Native evidence",
              "pins_before": before, "pins_ok_before": all(before[str(p)] == s for p, s in PINS.items())}
    if not result["pins_ok_before"]:
        result["status"] = "ERROR"
        write_result(out / "result.json", result)
        print("ERROR pin mismatch before import")
        return 2
    result["P1"] = part1()
    residue = result["P1"]["pair_fifo_residue"]
    result["P2_residue"] = asyncio.run(part2(out / "residue", residue))
    result["P2_zero_control"] = asyncio.run(part2(out / "zero", 0.0))
    after = hashes()
    result["pins_after"] = after
    result["pins_unchanged"] = after == before
    r, z = result["P2_residue"], result["P2_zero_control"]
    checks = {
        "P1_positive_residue": residue > 0,
        "P2_residue_refused_at_deadline": r["error"] == EXPECTED_ERROR,
        "P2_residue_at_least_two_readings": r["readings_with_value"] >= 2,
        "P2_residue_counters_zero_and_cost_exact": r["counters_all_zero"]
        and r["served_cost_reprs_seen"] == [repr(residue)],
        "P2_residue_no_drain_record": r["caller_drains"] is None,
        "P2_zero_control_one_reading_drained": z["error"] is None and z["readings_with_value"] == 1
        and z["caller_drains"] is not None and z["caller_drains"]["final"]["drained"] is True,
        "P3_fixed_predicate_true_on_residue_readings": all(r["hypothetical_fixed_predicate_on_readings"])
        and bool(r["hypothetical_fixed_predicate_on_readings"]),
        "pins_unchanged": result["pins_unchanged"],
    }
    result["checks"] = checks
    result["status"] = "CONFIRMED" if all(checks.values()) else "REFUTED"
    write_result(out / "result.json", result)
    print(result["status"], json.dumps(checks, sort_keys=True))
    print("residue", repr(residue), "readings", r["readings_with_value"], "files", r["evidence_files_written"],
          "elapsed", round(r["elapsed_s"], 3), "error", r["error"])
    print("P1 five-order", json.dumps(result["P1"]["five_overlapped_all_finish_orders"]),
          "pairs", json.dumps(result["P1"]["pairs_fifo_and_lifo"]))
    return 0 if result["status"] == "CONFIRMED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
