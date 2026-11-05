"""Specific frozen paths, dependencies and privacy checks for this one launch."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys

from freeze_inputs import sha, verify, verify_manifest, write

HERE = Path(__file__).resolve().parent
ROOT = Path("/workspace/pr32-r23/source")
NATIVE = Path("/workspace/pr32-r23/venv/bin/python")
STAGE = ROOT.parent / "r24-team16-native-control-launch02"
CLIENT = ROOT.parent / "r24-team16-native-control-probe02"
STATE = ROOT.parent / "control-state02"
PUBLIC = ROOT / "results/mpc-native-r24/r24-team16-native-control-launch02"
CLIENT_OUTPUT = ROOT / "results/mpc-native-r24/native-team16-control-probe02-launch02"
SOURCE13 = ROOT / "experiments/mpc-research/native-r24-prepared-physical13"
PROTOCOL = ROOT / "results/mpc-native-r23/runtime/h16-original-protocol.json"
ALIAS = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554")
MODEL = ROOT.parent / "models/Qwen3-4B-Instruct-2507"
SAFE_EXEC = HERE / "inputs/clock02/safe_exec.py"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_bytes())


def module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def clock_helpers():
    return module("_fixed_control_clock02_helpers", HERE / "inputs/clock02/probe_entry.py")


def owners():
    return module("_fixed_control_Source13_owned_helpers", HERE / "inputs/source13-entry02.py")


def client_library(directory=CLIENT):
    sys.path.insert(0, str(directory))
    return module("_fixed_control_revision02_client", directory / "native_probe02.py")


def verify_runnable(client=CLIENT):
    proof = verify(originals=False)
    pins = load(HERE / "RUNNABLES.json")
    require(all(sha(HERE / name) == digest for name, digest in pins.items()), "launcher_code_changed")
    binding = load(HERE / "CLIENT_BINDING02.json")
    proof["client"] = verify_manifest(client, binding["manifest_sha256"])
    return proof


def require_remote():
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize
            and HERE == STAGE and Path.cwd().resolve() == STAGE
            and Path(sys.prefix).resolve() == NATIVE.parent.parent.resolve(),
            "Mayor_only_exact_remote_isolated_namespace")


def create_attempt():
    require(not any(path.exists() or path.is_symlink() for path in
                    (STATE, PUBLIC, CLIENT_OUTPUT)), "exclusive_one_attempt_no_retry")
    STATE.mkdir(mode=0o700)
    PUBLIC.mkdir(mode=0o700)
    for name in ("t", "python-cache-evidence", "native-artifacts"):
        (STATE / name).mkdir(mode=0o700)


def private_config(protocol_path=PROTOCOL, destination=None):
    destination = destination or STATE / "control.private.json"
    require(sha(protocol_path) == sha(HERE / "inputs/source13/h16-original-protocol.json"),
            "original_native_cost_protocol_changed")
    cost = load(protocol_path)["cost_spec"]
    config = dict(
        MODEL_NAME=str(ALIAS), ADMIN_API_KEY=secrets.token_hex(32),
        COST_INTERCEPT=cost["intercept"], INPUT_TOKEN_COST=cost["input_per_token"],
        OUTPUT_TOKEN_COST=cost["output_per_token"], MIN_REQUEST_COST=cost["minimum"],
        DEFAULT_MAX_TOKENS=cost["default_max_tokens"],
        MPC_TOKENIZER_IDENTITY="gateway-token-estimate-v1;Qwen3-4B-2507",
        MPC_OWNER_LEDGER_ENABLED="true")
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(config, stream)
    return config


def guarded(command, config_path=None):
    return [str(NATIVE), "-I", "-B", str(SAFE_EXEC),
            str(config_path or STATE / "control.private.json"), "--", *map(str, command)]


def public_credentials_absent(directories, secret):
    token = secret.encode()
    require(token, "nonempty_credential_scan_token")
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.is_symlink() or path.name.endswith(".private.json") or token in path.read_bytes():
                return False
            if path.suffix == ".json":
                value = load(path)
                pending = [value]
                while pending:
                    value = pending.pop()
                    if isinstance(value, dict):
                        encoded = value.get("response_body_base64")
                        if encoded is not None and token in base64.b64decode(encoded, validate=True):
                            return False
                        pending.extend(value.values())
                    elif isinstance(value, list):
                        pending.extend(value)
    return True


def evidence_index():
    records = []
    for directory in (PUBLIC, CLIENT_OUTPUT):
        if directory.exists():
            records.extend(dict(directory=directory.name, path=str(path.relative_to(directory)),
                                sha256=sha(path), bytes=path.stat().st_size)
                           for path in sorted(directory.rglob("*")) if path.is_file())
    return records
