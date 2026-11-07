"""Small source-bound utilities shared by the independent component probe."""

import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/workspace/pr32-r23/source")
SOURCE13 = SOURCE_ROOT / "experiments/mpc-research/native-r24-prepared-physical13"
NATIVE_PYTHON = Path("/workspace/pr32-r23/venv/bin/python")
GPU_UUID = "GPU-8610519d-3680-ae5f-cdfc-40037bfb5013"
EXPERIMENT = "r24-team16-native-clock-probe02"
PUBLIC_RESULT = SOURCE_ROOT / "results/mpc-native-r24" / EXPERIMENT
SOURCE_MANIFEST = "94fa09623feed131fb24c4c3054483f56387e64d48a59f9de556f4bbefab1de8"
EXECUTION_SEAL = "961b8520efd2656b6494895030fa43b0f873b664"
ENTRY_SHA = "fe9cf85739aa06ed534eb4871e0152ae9cefb1a7425af44194aad43c56b7fbb4"
TOOL_FILES = ("probe_common.py", "probe_replay.py", "probe_native.py",
              "probe_entry.py", "safe_exec.py")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_inputs():
    pins = load(HERE / "source-pins.json")
    require(sha(HERE / "PLAN.json") == pins["plan_sha256"], "prefrozen_plan_changed")
    for row in pins["sources"] + load(HERE / "support-pins.json")["sources"]:
        require(sha(HERE / row["copy"]) == row["sha256"], "frozen_input_changed:" + row["copy"])
    return pins


def verify_candidate():
    verify_inputs()
    prereg = load(HERE / "prereg.json")
    require(prereg["schema"] == "team16-native-copy-clock-prereg-draft-v1",
            "independent_component_prereg_required")
    for name, expected in prereg["runnable_sha256"].items():
        require(sha(HERE / name) == expected, "candidate_source_changed:" + name)
    for name, expected in prereg["input_sha256"].items():
        require(sha(HERE / name) == expected, "candidate_input_changed:" + name)
    require(prereg["native_execution_by_author"] is False
            and prereg["mayor_only_scheduler"] is True, "separate_native_authority_required")
    return prereg


def source13_library():
    directory = HERE / "inputs/source13"
    sys.path.insert(0, str(directory))
    module = importlib.import_module("prepared_copy_ready_plugin")
    require(sha(module.__file__) == sha(directory / "prepared_copy_ready_plugin.py"),
            "source13_copy_plugin_identity")
    return module


def diagnostic():
    name = "_team16_clock_reference_diagnostic"
    if name not in sys.modules:
        load_module(name, HERE / "inputs/reference/trace_diagnostic.py")
    return sys.modules[name]


def prepared_worker_gate():
    """Execute the unchanged small worker gate without importing the eight-arm fitter."""
    path = HERE / "inputs/source13/native_physics_support.py"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    node, = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "validate_prepared_worker"]
    namespace = {"json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["validate_prepared_worker"]
