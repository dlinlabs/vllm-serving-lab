"""Canonical imports from the narrow, unchanged Source13/product copies."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys

from freeze_inputs01 import sha, verify as verify_initial


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "inputs/source-tree"
NATIVE = SOURCE / "experiments/mpc-research/native-r24-prepared-physical13"
CONTROL = HERE / "inputs/control02"


def verify(*, originals=False):
    result = verify_initial(originals=originals)
    supplement = HERE / "SUPPLEMENTAL_INPUTS01.json"
    if supplement.is_file():
        declaration = json.loads(supplement.read_bytes())
        seal = json.loads((HERE / "SUPPLEMENTAL_FREEZE01.json").read_bytes())
        for name, fingerprint in seal["frozen_sha256"].items():
            if sha((HERE / name).read_bytes()) != fingerprint:
                raise ValueError("supplement_changed")
        for row in declaration["bindings"]:
            if sha((HERE / row["copy"]).read_bytes()) != row["sha256"]:
                raise ValueError("supplemental_source_changed")
        result.update(supplemental_bindings=len(declaration["bindings"]),
                      supplemental_sha256=sha(supplement.read_bytes()))
    return result


def install_paths():
    paths = (NATIVE, SOURCE, SOURCE / "experiments/mpc-research/native-v2",
             SOURCE / "experiments/mpc-research/native_observer", CONTROL)
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def module(name):
    install_paths()
    value = importlib.import_module(name)
    path = Path(value.__file__).resolve()
    if not path.is_relative_to(SOURCE):
        raise ValueError("noncanonical_frozen_import:" + name)
    return value


def control():
    install_paths()
    name = "_owner_frozen_control02"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CONTROL / "native_probe02.py")
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def settings():
    return json.loads((CONTROL / "EXPERIMENT.json").read_bytes())


def sources():
    import hashlib
    digest = lambda name: hashlib.sha256((NATIVE / name).read_bytes()).hexdigest()
    return dict(
        source_sha256=digest("phase_observer.py"),
        input_post_source_sha256=digest("input_post_observer.py"),
        queue_observer_sha256=digest("queue_observer.py"),
        core_source_sha256=module("input_post_observer").CORE_SHA256,
    )
