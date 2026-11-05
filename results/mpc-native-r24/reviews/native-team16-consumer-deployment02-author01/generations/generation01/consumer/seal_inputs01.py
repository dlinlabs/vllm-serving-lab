"""Private deployment input verifier; original literal preflight guards stay intact."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = "_consumer_deployment02_pins"


def verify(originals=False):
    if NAME not in sys.modules:
        specification = importlib.util.spec_from_file_location(NAME, ROOT / "pins02.py")
        value = importlib.util.module_from_spec(specification)
        sys.modules[NAME] = value
        specification.loader.exec_module(value)
    value = sys.modules[NAME]
    if Path(value.__file__).resolve() != ROOT / "pins02.py":
        raise ValueError("deployment_input_verifier_origin")
    return value.verify(originals)
