"""Pinned local consumer support; no endpoint, model or Native launch interface."""

import base64
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types

from seal_inputs01 import verify as input_verify

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "inputs/preflight"
SOURCE = PREFLIGHT / "inputs/source-tree"
NAMESPACE = "team16-cut-consumer02"
CAPTURE = "nv2-team16-cut-consumer02-fixed32"
MAX_BYTES = 1_048_576
PIPELINE = .080


def require(value, reason):
    if not value:
        raise ValueError(reason)


def number(value):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "finite_clock")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def raw_write(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def permissions():
    return dict(planner_entry_at=None, decision_at=None, actual_application_at=None,
                acquisition_token=None, request_token=None, qualified_model=False,
                prediction_qualified=False, observer_deployable=False,
                native_mpc_permitted=False, actual_actuation_performed=False,
                exact_application_time_gate=False)


def module(name, path):
    path = Path(path).resolve()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    result = sys.modules[name]
    require(Path(result.__file__).resolve() == path, "module_origin")
    return result


def preflight():
    if str(PREFLIGHT) not in sys.path:
        sys.path.insert(0, str(PREFLIGHT))
    pre = importlib.import_module("preflight01")
    require(Path(pre.__file__).resolve() == PREFLIGHT / "preflight01.py", "literal_preflight_origin")
    pre.frozen.install_paths()
    return pre


def candidate_bootstrap(startup, *, cutoff):
    original = preflight().bootstrap_inputs
    require(Path(original.__code__.co_filename).resolve() == PREFLIGHT / "preflight01.py",
            "literal_bootstrap_code")
    def document(name):
        return load(HERE / "NATIVE_PLAN.json" if name == "PROPOSED_NATIVE_PLAN.json" else PREFLIGHT / name)
    bound = types.FunctionType(original.__code__, {**original.__globals__, "document": document},
                               original.__name__, original.__defaults__, original.__closure__)
    return bound(startup, cutoff=cutoff)


def proc_api():
    value = preflight().frozen.module("process_clock_provenance")
    require(sha((SOURCE / "prefill_lab/live_http.py").read_bytes()) == value.CLOCK_SOURCE_SHA256,
            "literal_clock_source")
    return value


def current_identity():
    api = proc_api()
    proc = api.LinuxProc()
    process = proc.read(os.getpid())
    from dataclasses import asdict
    return dict(process=asdict(process), boot_id=proc.boot_id(),
                clock_domain="linux-monotonic:" + proc.boot_id() + ":" + process.time_namespace)


def redaction():
    return module("_consumer_literal_redaction", HERE / "inputs/base/inputs/launch02/inputs/clock02/safe_exec.py")


def ownership():
    return module("_consumer_literal_ownership", HERE / "inputs/base/inputs/launch02/inputs/source13-entry02.py")


def reader_type():
    return module("_consumer_literal_reader", HERE / "inputs/base/owned01.py").Reader


def verify_code(*, originals=False, literal_cwd=False):
    inputs = input_verify(originals)
    for name, fingerprint in load(HERE / "CONTROL_FREEZE.json")["sha256"].items():
        require(sha((HERE / name).read_bytes()) == fingerprint, "consumer_source_pin:" + name)
    if literal_cwd:
        preflight().frozen.verify()
    return inputs


def raw64(value):
    require(type(value) is str and len(value) <= 4 * (MAX_BYTES + 2) // 3, "raw_byte_capacity")
    return base64.b64decode(value, validate=True)


def no_secret(value, secret):
    if not secret:
        return
    marker = secret.encode()
    require(marker not in canonical(value), "credential_material_refused")
    def inspect(row):
        if isinstance(row, dict):
            for key, item in row.items():
                if key.endswith("_base64") and type(item) is str:
                    require(marker not in raw64(item), "encoded_credential_material_refused")
                inspect(item)
        elif isinstance(row, list):
            for item in row:
                inspect(item)
    inspect(value)
