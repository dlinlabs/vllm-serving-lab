"""Fixed paths and source-bound dependencies for this single component."""

import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys

from freeze01 import sha, verify as verify_initial, write

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "inputs/preflight"
SOURCE = PREFLIGHT / "inputs/source-tree"
SOURCE13_REL = "experiments/mpc-research/native-r24-prepared-physical13"
ROOT = Path("/workspace/pr32-r23/source")
SOURCE13 = ROOT / SOURCE13_REL
PYTHON = ROOT.parent / "venv/bin/python"
STAGE = ROOT.parent / HERE.name
STATE = ROOT.parent / "owner-inflight-state01"
ORIGINS = STATE / "origins-inflight01"
PUBLIC = ROOT / "results/mpc-native-r24/r24-team16-native-owner-inflight-adapter01"
CONFIG = STATE / "owner-inflight.private.json"
LOCK = ROOT.parent / "native-component-GPU-8610519d.lock"
ALIAS = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554")
MODEL = ROOT.parent / "models/Qwen3-4B-Instruct-2507"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ORIGINAL_PROTOCOL = ROOT / "results/mpc-native-r23/runtime/h16-original-protocol.json"
PLAN = HERE / "NATIVE_PLAN.revision01.json"
OFFERS = HERE / "OFFERS.json"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def publish(path, value):
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(), "exclusive_marker")
    with partial.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    partial.rename(path)


def protocol():
    return load(HERE / "PROTOCOL.json")


def verify(originals=False):
    result = verify_initial(originals)
    amendment = load(HERE / "PLAN_AMENDMENT01.json")
    require(amendment["effective_plan"] == PLAN.name
            and amendment["effective_sha256"] == sha(PLAN)
            and amendment["original_sha256"] == sha(HERE / "NATIVE_PLAN.json")
            and amendment["source_sha256"] == sha(HERE / amendment["source"]),
            "seed_amendment_source_binding")
    return dict(result, effective_native_plan_sha256=sha(PLAN),
                plan_amendment_sha256=sha(HERE / "PLAN_AMENDMENT01.json"))


def module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        obj = importlib.util.module_from_spec(spec)
        sys.modules[name] = obj
        spec.loader.exec_module(obj)
    require(Path(sys.modules[name].__file__).resolve() == Path(path).resolve(), "module_name_collision")
    return sys.modules[name]


def preflight():
    sys.path.insert(0, str(PREFLIGHT))
    result = importlib.import_module("preflight01")
    require(Path(result.__file__).resolve() == PREFLIGHT / "preflight01.py", "preflight_origin")
    result.frozen.install_paths()
    # Only the prospective plan input changes; every original validation assertion remains.
    result.document = lambda name: load(PLAN if name == "PROPOSED_NATIVE_PLAN.json" else PREFLIGHT / name)
    return result


def owner():
    return module("_owner_adapter_source13_ownership", HERE / "inputs/launch02/inputs/source13-entry02.py")


def redaction():
    return module("_owner_adapter_safe_redaction", HERE / "inputs/launch02/inputs/clock02/safe_exec.py")


def clock_helpers():
    directory = HERE / "inputs/launch02/inputs/clock02"
    sys.path.insert(0, str(directory))
    return module("_owner_adapter_clock_helpers", directory / "probe_entry.py")


def domain():
    return ("linux-monotonic:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            + ":" + os.readlink("/proc/self/ns/time"))


def remote_only():
    require(HERE == STAGE and Path(sys.prefix).resolve() == PYTHON.parent.parent.resolve()
            and sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,
            "Mayor_only_exact_remote_isolated_interpreter")


def owned_parent(pid, ticks):
    require(type(pid) is int and pid == os.getppid()
            and owner().snapshot(pid)[0] == ticks
            and os.getsid(0) == os.getpid() == os.getpgrp(), "actual_owned_parent_and_session")


def require_idle(inventory):
    clock_helpers().require_idle(inventory)
    require(float(inventory["gpu"].split(",")[-1].strip()) == 0, "GPU_utilization_not_idle")


def verify_code():
    result = verify(False)
    for name, fingerprint in load(HERE / "RUNNABLES.json").items():
        require(sha(HERE / name) == fingerprint, "runnable_changed:" + name)
    return result


def create_config():
    require(sha(ORIGINAL_PROTOCOL) == sha(
        HERE / "inputs/launch02/inputs/source13/h16-original-protocol.json"), "original_cost_protocol")
    cost = load(ORIGINAL_PROTOCOL)["cost_spec"]
    config = dict(MODEL_NAME=str(ALIAS), ADMIN_API_KEY=secrets.token_hex(32),
                  COST_INTERCEPT=cost["intercept"], INPUT_TOKEN_COST=cost["input_per_token"],
                  OUTPUT_TOKEN_COST=cost["output_per_token"], MIN_REQUEST_COST=cost["minimum"],
                  DEFAULT_MAX_TOKENS=cost["default_max_tokens"],
                  MPC_TOKENIZER_IDENTITY="gateway-token-estimate-v1;Qwen3-4B-2507",
                  MPC_OWNER_LEDGER_ENABLED="true")
    fd = os.open(CONFIG, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(config, stream)
    return config


def private_config(path=CONFIG):
    info = path.lstat()
    require(not path.is_symlink() and info.st_mode & 0o777 == 0o600
            and info.st_uid == os.getuid() and info.st_nlink == 1, "private_config_identity")
    value = load(path)
    require(value["MODEL_NAME"] == str(ALIAS) and len(value["ADMIN_API_KEY"]) == 64,
            "private_config_schema")
    return value


def sdk_snapshot():
    specification = load(SOURCE / "experiments/prefill-vllm-v030-source-map.json")
    spec = importlib.util.find_spec("vllm")
    require(spec is not None and spec.origin, "installed_vllm_required")
    directory = Path(spec.origin).resolve().parent
    require(directory.is_relative_to(PYTHON.parent.parent), "installed_sdk_origin")
    version = importlib.metadata.version("vllm")
    actual = {name: sha(directory / name) for name in specification["sha256"]}
    require(version == specification["version"] and actual == specification["sha256"],
            "installed_sdk_source_map_mismatch")
    full = {}
    for path in sorted(directory.rglob("*.py")):
        require(not path.is_symlink() and path.is_file(), "SDK_regular_sources_required")
        full[str(path.relative_to(directory))] = sha(path)
    return dict(version=version, directory=str(directory), source_map=actual, all_python=full)


def source_snapshot():
    helper = clock_helpers()
    source = helper.source_binding(STATE / "python-cache-evidence")
    sdk = sdk_snapshot()
    sys.path[:0] = [str(SOURCE13), str(ROOT), str(ROOT / "experiments/mpc-research/native-v2")]
    original = importlib.import_module("run_batch_qualification")
    require(Path(original.__file__).resolve() == SOURCE13 / "run_batch_qualification.py",
            "original_verifier_origin")
    deployment = original.verify_original_sources(ROOT, ROOT.parent / "deployment-source.tar.gz")
    return dict(source13_and_product=source, installed_sdk=sdk, original_deployment=deployment)


def import_prerequisites():
    declarations = load(HERE / "inputs/host-readonly01.json")["entrypoints"]
    found = []
    for row in declarations:
        matches = [ep for ep in importlib.metadata.entry_points(group=row["group"])
                   if ep.name == row["name"]]
        require(len(matches) == 1 and matches[0].value == row["value"]
                and matches[0].dist.version == row["version"]
                and matches[0].dist.metadata["Name"] == row["distribution"], "entrypoint_changed")
        found.append(dict(row))
    paths = [str(SOURCE13), str(ROOT), str(ROOT / "experiments/mpc-research/native-v2"),
             str(ROOT / "experiments/mpc-research/native_observer")]
    sys.path[:0] = paths
    origins = {}
    for name in ("phase_observer", "input_post_observer", "queue_observer"):
        spec = importlib.util.find_spec(name)
        expected = SOURCE13 / (name + ".py")
        require(spec is not None and Path(spec.origin).resolve() == expected
                and sha(expected) == sha(SOURCE / SOURCE13_REL / (name + ".py")), "module_origin_changed")
        origins[name] = dict(path=str(expected), sha256=sha(expected))
    return dict(entrypoints=found, origins=origins, hooks_registered_by_check=False)


def estimate_prompt(payload, tokenizer):
    source = SOURCE / "gateway.py"
    tree = ast.parse(source.read_bytes(), filename=str(source))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in
                 ("_message_text", "estimate_input_tokens")]
    import math
    from typing import Any, Dict
    namespace = dict(tokenizer=tokenizer, json=json, math=math, Any=Any, Dict=Dict)
    exec(compile(tree, str(source), "exec"), namespace)
    count, fallback = namespace["estimate_input_tokens"](payload)
    require(type(count) is int and count > 0 and not fallback, "exact_cached_tokenizer_no_fallback")
    return count


def privacy_scan(directories, secret, *, no_key_created=False):
    import base64
    token = secret.encode() if secret else None
    require(bool(token) or no_key_created is True, "privacy_scan_nonempty_key")
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_symlink():
                return False
            if not path.is_file():
                continue
            if path.name.endswith(".private.json") or (token and token in path.read_bytes()):
                return False
            if path.suffix == ".json":
                try:
                    pending = [load(path)]
                except (ValueError, UnicodeError):
                    return False
                while pending:
                    value = pending.pop()
                    if isinstance(value, dict):
                        if "response_body_base64" in value:
                            try:
                                decoded = base64.b64decode(value["response_body_base64"], validate=True)
                            except (ValueError, TypeError):
                                return False
                            if token and token in decoded:
                                return False
                        pending.extend(value.values())
                    elif isinstance(value, list):
                        pending.extend(value)
    return True


def failure(error, secret):
    return dict(error_type=type(error).__name__,
                reason=str(error).replace(secret, "[REDACTED_AUTH]")[:4096] if secret else type(error).__name__)


def load_origins(directory):
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        require(not path.is_symlink() and path.stat().st_size <= 1048576, "origin_receipt_file")
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            require(row.get("valid") is True and type(row.get("pid")) is int
                    and row["pid"] > 0 and type(row.get("start_ticks")) is int
                    and row.get("event") in ("loaded_after_exec", "inherited_after_fork"),
                    "origin_receipt_refused")
            records.append(row)
    return records


def require_origins(records, *, native_pid, gateway_pid, process_reader=None):
    policy = load(HERE / "ORIGIN_POLICY.json")["origins"]
    reader = process_reader or (lambda pid: owner().snapshot(pid)[0])
    for pid, names in ((native_pid, ("phase_observer", "input_post_observer", "queue_observer")),
                       (gateway_pid, ("gateway_prefix_bridge", "gateway"))):
        ticks = reader(pid)
        for name in names:
            matching = [r for r in records if r["pid"] == pid and r["module"] == name
                        and r["start_ticks"] == ticks and r["path"] == policy[name]["path"]
                        and r["sha256"] == policy[name]["sha256"] and r["valid"] is True]
            require(bool(matching), "missing_actual_running_origin:" + name)
    return True
