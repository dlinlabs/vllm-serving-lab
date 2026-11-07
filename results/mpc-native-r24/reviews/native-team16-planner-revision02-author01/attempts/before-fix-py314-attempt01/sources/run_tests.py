"""Run frozen Role7 red controls or all preserved and revision02 controls."""

import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE), str(ROOT / ".runtime/r24-mpc-parallel15-planner"), str(ROOT),
    str(ROOT / "tests"), str(ROOT / "experiments/mpc-research/native-r23"),
]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

import pytest
import policy_planner16


if __name__ == "__main__":
    if Path(policy_planner16.__file__).resolve().parent != HERE:
        raise SystemExit("wrong_planner_imported")
    suite = sys.argv[1]
    selected = {
        "witnesses": ("test_role7_planner_controls.py",),
        "derived": (
            "test_policy_planner16.py", "test_parent_numerical_review.py",
            "test_policy_planner16_numerical_revision.py",
            "test_role7_planner_controls.py", "test_policy_planner16_revision02.py",
        ),
    }[suite]
    print(f"planner={policy_planner16.__file__}", flush=True)
    print(f"interpreter={sys.executable} version={sys.version}", flush=True)
    raise SystemExit(pytest.main([
        "-q", *(["-s"] if suite == "witnesses" else []),
        "-p", "no:cacheprovider", "--confcutdir=" + str(HERE),
        *[str(HERE / name) for name in selected],
    ]))
