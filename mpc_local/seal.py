"""Bind CPU candidate bytes and development inputs, never native qualification."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(root):
    root = Path(root).resolve()
    paths = sorted((root / "mpc_local").glob("*.py"))
    paths += sorted((root / "mpc_lab").glob("*.py"))
    paths += sorted((root / "scripts").glob("*r21*.py"))
    paths += sorted((root / "experiments/mpc-research/native-v3").glob("r21*.json"))
    paths += [root / "gateway.py", root / "telemetry.py", root / "pyproject.toml",
              root / "requirements-cpu-tests.txt"]
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def _training_records(document, training_inputs):
    splits = [set(document[k]) for k in ("train_seeds", "validation_seeds", "heldout_seeds")]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("overlapping_splits")
    if not training_inputs:
        raise ValueError("missing_training_provenance")
    paths = [str(Path(path).resolve()) for path in training_inputs]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate_training_inputs")
    all_records = []
    for path in paths:
        records = json.loads(Path(path).read_text())
        if not isinstance(records, list) or not records:
            raise ValueError("training_input_must_be_nonempty_record_list")
        if any(not isinstance(row, dict) or type(row.get("seed")) is not int
               or row["seed"] not in splits[0] for row in records):
            raise ValueError("nontrain_fit_input")
        all_records.extend(records)
    return splits, paths, all_records


def _bind_fit(root, weights, document, records):
    from . import model as fitted

    for name in ("model.py", "contracts.py", "seal.py"):
        if file_hash(Path(root) / "mpc_local" / name) != file_hash(Path(__file__).parent / name):
            raise ValueError("model_code_context_mismatch")
    if file_hash(Path(root) / "mpc_lab/contracts.py") != file_hash(
        Path(sys.modules[fitted.CostSpec.__module__].__file__)
    ):
        raise ValueError("cost_code_context_mismatch")
    data = json.loads(Path(weights).read_text())
    model = fitted.Model.from_dict(data)
    normalized = fitted.normalize_records(records)
    record_hash = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    manifest = data["manifest"]
    if (manifest["records_sha256"] != record_hash
            or manifest["n_records"] != len(records)
            or manifest["seeds"] != sorted({row["seed"] for row in records})
            or manifest["allowed_seeds"] != sorted(document["train_seeds"])):
        raise ValueError("model_training_records_mismatch")
    specification = document.get("fitting")
    if specification and manifest["seeds"] != sorted(document["train_seeds"]):
        raise ValueError("incomplete_formal_train")
    if specification and (
        data["features"] != specification["features"]
        or model.dynamics != specification["dynamics"]
        or data["n_scenarios"] != specification["n_scenarios"]
        or manifest["ridge"] != specification["ridge"]
    ):
        raise ValueError("fit_outside_preregistered_specification")
    kwargs = dict(
        features=data["features"], allowed_seeds=document["train_seeds"],
        data_source=manifest["data_source"], ridge=manifest["ridge"],
        n_scenarios=data["n_scenarios"], dynamics=model.dynamics,
    )
    if model.dynamics == "decode_load":
        kwargs["at_cap_unlabeled"] = data["finish_at_cap"]["policy"]
    elif model.dynamics == "decode_load_terminal":
        kwargs["at_cap_unlabeled"] = "refuse"
    replay = fitted.fit(records, cost_spec=model.cost_spec, **kwargs)
    if replay.fingerprint != model.fingerprint:
        raise ValueError("weights_not_reproduced_from_declared_train")
    return model.fingerprint


def seal_candidate(root, weights, protocol, training_inputs, output):
    root = Path(root).resolve()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True,
    ).strip()
    if dirty:
        raise ValueError("commit_source_before_sealing")
    document = json.loads(Path(protocol).read_text())
    splits, training_inputs, records = _training_records(document, training_inputs)
    fingerprint = _bind_fit(root, weights, document, records)
    source = source_manifest(root)
    subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", *source],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    receipt = {
        "schema": "r21-cpu-candidate-seal-v2",
        "candidate": document["candidate"],
        "scope": "cpu_synthetic_only",
        "git_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "python": sys.version,
        "packages": {name: version(name) for name in ("numpy", "httpx", "fastapi", "pytest")},
        "weights_path": str(Path(weights).resolve()),
        "weights_sha256": file_hash(weights),
        "protocol_path": str(Path(protocol).resolve()),
        "protocol_sha256": file_hash(protocol),
        "training_inputs": {str(Path(p).resolve()): file_hash(p) for p in training_inputs},
        "training_input_order": training_inputs,
        "model_fingerprint": fingerprint,
        "train_fit_reproduced": True,
        "source_files": source,
        "source_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest(),
        "heldout_seeds": sorted(splits[2]),
        "native_runtime_qualified": False,
        "calibrated": False,
        "native_mpc_permitted": False,
    }
    with Path(output).open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
    return receipt


def verify_candidate(root, receipt):
    receipt = json.loads(Path(receipt).read_text())
    if (receipt.get("schema") != "r21-cpu-candidate-seal-v2"
            or receipt.get("train_fit_reproduced") is not True):
        raise ValueError("candidate_fit_binding_missing")
    if receipt["scope"] != "cpu_synthetic_only" or any(
        receipt[key] is not False for key in
        ("native_mpc_permitted", "native_runtime_qualified", "calibrated")
    ):
        raise ValueError("invalid_candidate_scope")
    source = source_manifest(root)
    if source != receipt["source_files"]:
        raise ValueError("source_changed_since_seal")
    if hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() != receipt["source_sha256"]:
        raise ValueError("source_fingerprint_mismatch")
    if receipt["python"] != sys.version or receipt["packages"] != {
        name: version(name) for name in ("numpy", "httpx", "fastapi", "pytest")
    }:
        raise ValueError("candidate_environment_mismatch")
    if file_hash(receipt["weights_path"]) != receipt["weights_sha256"]:
        raise ValueError("weights_changed_since_seal")
    if file_hash(receipt["protocol_path"]) != receipt["protocol_sha256"]:
        raise ValueError("protocol_changed_since_seal")
    for path, expected in receipt["training_inputs"].items():
        if file_hash(path) != expected:
            raise ValueError("training_input_changed_since_seal")
    order = receipt["training_input_order"]
    if len(order) != len(receipt["training_inputs"]) or set(order) != set(receipt["training_inputs"]):
        raise ValueError("training_input_order_mismatch")
    document = json.loads(Path(receipt["protocol_path"]).read_text())
    splits, _, records = _training_records(document, order)
    if receipt["candidate"] != document["candidate"] or receipt["heldout_seeds"] != sorted(splits[2]):
        raise ValueError("protocol_binding_mismatch")
    if _bind_fit(root, receipt["weights_path"], document, records) != receipt["model_fingerprint"]:
        raise ValueError("model_fingerprint_mismatch")
    return receipt
