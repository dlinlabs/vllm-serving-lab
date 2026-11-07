"""Narrow exact source/input copies and validation; never run predecessor mains."""

import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify(originals=False):
    doc = load(HERE / "INPUTS.json")
    for row in doc["bindings"]:
        if sha(HERE / row["copy"]) != row["sha256"] or originals and sha(row["source"]) != row["sha256"]:
            raise ValueError("deployment_exact_input_changed:" + row["copy"])
    for name, fingerprint in load(HERE / "INPUT_FREEZE.json")["sha256"].items():
        if sha(HERE / name) != fingerprint:
            raise ValueError("deployment_input_freeze_changed:" + name)
    return dict(bindings=len(doc["bindings"]), originals=originals)


def main():
    if Path.cwd().resolve() != HERE:
        raise ValueError("exclusive_deployment02_scope")
    for name, fingerprint in load("PLAN_FREEZE.json")["sha256"].items():
        if sha(name) != fingerprint:
            raise ValueError("prefrozen_plan_changed")
    plan = load("PLAN.json")
    source = ROOT / ".runtime/r24-team16-native-cut-consumer-successor01"
    if sha(source / "SHA256SUMS") != plan["source_successor_manifest_sha256"]:
        raise ValueError("exact_unaccepted_successor_required")
    manifest = {line[66:]:line[:64] for line in (source / "SHA256SUMS").read_text().splitlines()}
    bindings = []
    def retain(relative, target, *, pinned=True):
        original, output = source / relative, Path(target)
        if original.is_symlink() or sha(original) != manifest[relative]:
            raise ValueError("successor_selected_source_changed:" + relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(original.read_bytes())
        if pinned:
            bindings.append(dict(source=str(original), copy=str(output), sha256=manifest[relative]))
    (HERE / "inputs").mkdir()
    with (HERE / "inputs/SUCCESSOR-SHA256SUMS").open("xb") as stream:
        stream.write((source / "SHA256SUMS").read_bytes())
    bindings.append(dict(source=str(source / "SHA256SUMS"), copy="inputs/SUCCESSOR-SHA256SUMS",
                         sha256=plan["source_successor_manifest_sha256"]))
    core = ("common01.py","consumer01.py","generation01.py","ipc01.py","validate01.py","worker01.py","fixture01.py")
    composition = ("composition_common01.py","client01.py","transport01.py","coverage01.py","owned01.py",
                   "launch01.py","service01.py","origin_guard/sitecustomize.py","ORIGIN_POLICY.json","PROTOCOL.json")
    for name in core:
        retain("candidate/" + name, "inputs/reference-consumer/" + name)
        retain("candidate/" + name, "consumer/" + name, pinned=False)
    for name in composition:
        retain("partial-composition/" + name, "inputs/reference-composition/" + name)
        retain("partial-composition/" + name, name, pinned=False)
    for name in ("NATIVE_PLAN.json","OFFERS.json"):
        retain("candidate/" + name, "inputs/reference-consumer/" + name)
    for relative in sorted(manifest):
        if relative.startswith("candidate/inputs/preflight/"):
            retain(relative, "consumer/" + relative[len("candidate/"):])
    needed = (
        "owned01.py", "common01.py", "acquire01.py", "PROTOCOL.json",
        "inputs/host-readonly01.json", "inputs/launch02/inputs/source13-entry02.py",
        "inputs/launch02/inputs/source13/h16-original-protocol.json",
        "inputs/launch02/inputs/clock02/probe_common.py",
        "inputs/launch02/inputs/clock02/probe_entry.py",
        "inputs/launch02/inputs/clock02/safe_exec.py",
        "inputs/launch02/inputs/clock02/inputs/source13/research_source_guard.py")
    for name in needed:
        retain("partial-composition/inputs/base/" + name, "consumer/inputs/base/" + name)
    retain("candidate/test_consumer01.py", "inputs/reference-tests/test_consumer01.py")
    retain("candidate/test_consumer01.py", "migration/test_consumer01.py", pinned=False)
    retain("evidence/verified-model-request01.txt", "inputs/verified-model-request01.txt")
    parent = ROOT / plan["parent_engineering_update"]["index"]
    write("PARENT_ENGINEERING_NOTICE.json", dict(
        source=str(parent), sha256=sha(parent), commit=plan["parent_engineering_update"]["commit"],
        scope="CPU engineering only;not independent delta/deployment/Native/G06",
        full_parent_archive_not_copied=True, aggregate_dependency_status=plan["dependency_status"]))
    offers = load("inputs/reference-consumer/OFFERS.json")
    for row in offers["rows"]:
        row["request_id"] = hashlib.sha256(
            (plan["names"]["namespace"] + ":offer:" + str(row["ordinal"])).encode()).hexdigest()[:32]
    write("OFFERS.json", offers)
    native = load("inputs/reference-consumer/NATIVE_PLAN.json")
    native.update(experiment_id=HERE.name, trace_sha256=sha("OFFERS.json"),
                  scope="Prospective fixed32 one ordinal2 model-free consumer;not model/MPC authorization.")
    native["trials"][0].update(trial_id=plan["names"]["capture"],
                              request_ids=[row["request_id"] for row in offers["rows"]])
    write("NATIVE_PLAN.json", native)
    for name in ("NATIVE_PLAN.json","OFFERS.json"):
        with (HERE / "consumer" / name).open("xb") as stream:
            stream.write((HERE / name).read_bytes())
    write("INPUT_CHANGES.json", dict(
        original_inputs="inputs/reference-consumer", new_capture=plan["names"]["capture"],
        request_IDs_renewed_in_both_plan_and_offers=True, experiment_ID_renewed=True,
        trace_sha256_means_exact_OFFERS_file_not_training_data=True,
        original_prompt_payload_caps_seed_and_source_extensions_unchanged=True,
        no_actual_training_or_model_data_read=True))
    write("INPUTS.json", dict(bindings=bindings, dependency_status=plan["dependency_status"],
                             only_selected_source_and_literal_guard_closure_copied=True))
    names = ("PLAN.json","PLAN_FREEZE.json","pins02.py","INPUTS.json","NATIVE_PLAN.json","OFFERS.json",
             "consumer/NATIVE_PLAN.json","consumer/OFFERS.json","INPUT_CHANGES.json","PARENT_ENGINEERING_NOTICE.json")
    write("INPUT_FREEZE.json", dict(frozen_at=time.time(), tests_started=False,
                                   sha256={name:sha(name) for name in names}))
    (HERE / "outputs").mkdir()
    (HERE / "attempts").mkdir()
    print(json.dumps(verify(True), sort_keys=True))


if __name__ == "__main__":
    main()
