"""Copy only exact dependencies; never execute a spent entrypoint."""

import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify(originals=False):
    doc = json.loads((HERE / "INPUTS.json").read_bytes())
    for row in doc["bindings"]:
        if sha(HERE / row["copy"]) != row["sha256"] or originals and sha(row["source"]) != row["sha256"]:
            raise ValueError("input_binding:" + row["copy"])
    for name, fingerprint in json.loads((HERE / "INPUT_FREEZE.json").read_bytes())["sha256"].items():
        if sha(HERE / name) != fingerprint:
            raise ValueError("input_freeze:" + name)
    return dict(bindings=len(doc["bindings"]), originals=originals)


def main():
    if Path.cwd().resolve() != HERE:
        raise ValueError("exclusive_scope")
    plan = json.loads(Path("PLAN.json").read_bytes())
    base = ROOT / ".runtime/r24-team16-native-owner-inflight-adapter01"
    gap = ROOT / ".runtime/r24-team16-native-inflight-runtime-gap01"
    if sha(base / "SHA256SUMS") != plan["accepted_inflight_source_seal"] or sha(gap / "SHA256SUMS") != plan["gap_seal"]:
        raise ValueError("lineage_seal")
    bindings = []
    def retain(source, name, expected):
        source, target = Path(source), HERE / name
        if source.is_symlink() or sha(source) != expected:
            raise ValueError("source_pin")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        bindings.append(dict(source=str(source), copy=name, sha256=expected))
    for root, group, names in (
        (base, "base", ["owned01.py", "acquire01.py", "transport01.py", "common01.py",
                        "PROTOCOL.json", "OFFERS.json", "NATIVE_PLAN.revision01.json",
                        "inputs/launch02/inputs/source13-entry02.py",
                        "inputs/launch02/inputs/clock02/safe_exec.py"]),
        (gap, "gap", ["NEXT_COMPONENT.json", "FINDINGS.json", "ACTUATOR_TARGET.json", "REPORT.json"])):
        pins = {line[66:]: line[:64] for line in (root / "SHA256SUMS").read_text().splitlines()}
        selected = set(names)
        if group == "base":
            selected.update(name for name in pins if name.startswith("inputs/preflight/"))
        retain(root / "SHA256SUMS", "inputs/" + group + "/SHA256SUMS", sha(root / "SHA256SUMS"))
        for name in sorted(selected):
            output = ("inputs/preflight/" + name[len("inputs/preflight/"):]
                      if name.startswith("inputs/preflight/") else "inputs/" + group + "/" + name)
            retain(root / name, output, pins[name])
    native = json.loads(Path("inputs/base/NATIVE_PLAN.revision01.json").read_bytes())
    old_trial = native["trials"][0]["trial_id"]
    native["trials"][0]["trial_id"] = plan["capture_id"]
    write("NATIVE_PLAN.json", native)
    offers = json.loads(Path("inputs/base/OFFERS.json").read_bytes())
    for row in offers["rows"]:
        row["request_id"] = hashlib.sha256(
            (plan["namespace"] + ":offer:" + str(row["ordinal"])).encode()).hexdigest()[:32]
    write("OFFERS.json", offers)
    write("PROSPECTIVE_INPUT_DIFF.json", dict(
        scope="INPUTS_ONLY_NOT_NATIVE_AUTHORIZATION", old_trial_id=old_trial,
        new_trial_id=plan["capture_id"], plan_other_fields_unchanged=True,
        offer_payload_and_cap_unchanged=True, only_offer_IDs_renewed=True,
        source_plan_and_payload="inputs/base"))
    write("INPUTS.json", dict(bindings=bindings, gap_acceptance="pending Parent",
                             previous_inflight_source="accepted by Mayor"))
    names = ("PLAN.json", "NATIVE_PLAN.json", "OFFERS.json", "INPUTS.json",
             "PROSPECTIVE_INPUT_DIFF.json", "seal_inputs01.py")
    write("INPUT_FREEZE.json", dict(frozen_at=time.time(), tests_started=False,
                                   sha256={name:sha(name) for name in names}))
    print(json.dumps(verify(True), sort_keys=True))


if __name__ == "__main__":
    main()
