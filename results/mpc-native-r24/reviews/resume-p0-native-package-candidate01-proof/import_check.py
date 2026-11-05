import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
expected_path = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(root), str(root / "experiments/mpc-research/resume-mvp")]

import native_nominal_qualification
import nominal_client
import nominal_episode
import nominal_runtime
from mpc_lab.contracts import CostSpec, ModelParameters
from dataclasses import asdict

expected = json.loads(expected_path.read_text())["source"]
parameters = ModelParameters(cost_spec=CostSpec(**expected["cost_spec"]))
assert asdict(parameters.identity) == expected["identity"]
paths = nominal_runtime.module_paths()
for module in tuple(sys.modules.values()):
    name = getattr(module, "__name__", "")
    if name.startswith(("mpc_lab", "prefill_lab", "nominal_")) or name in (
            "native_nominal_qualification", "native_services_smoke", "telemetry", "trace_workload"):
        assert Path(module.__file__).resolve().is_relative_to(root), name
factory, binding = nominal_runtime.load_train_scenario(nominal_episode.TRAIN, purpose="engineering")
assert callable(factory) and binding["purpose"] == "engineering"
print(json.dumps({
    "scope": "CPU_IMPORT_AND_TRAIN_BINDING_ONLY_NO_FORECAST_OR_GPU",
    "modules_from_candidate_tree": True, "actual_host_identity_reconstructed": True,
    "prior_content_sha256": binding["prior_content_sha256"],
    "module_paths": paths, "Native_qualification": False,
}, sort_keys=True))
