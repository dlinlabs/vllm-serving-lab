
import json
from pathlib import Path
import sys
from dataclasses import asdict
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / "experiments/mpc-research/resume-mvp")]
import native_nominal_qualification
import nominal_client
import nominal_episode
import nominal_runtime
from mpc_lab.contracts import CostSpec, ModelParameters
expected = json.loads(Path(sys.argv[2]).read_text())["source"]
parameters = ModelParameters(cost_spec=CostSpec(**expected["cost_spec"]))
assert asdict(parameters.identity) == expected["identity"]
for module in tuple(sys.modules.values()):
    name = getattr(module, "__name__", "")
    if name.startswith(("mpc_lab", "prefill_lab", "nominal_")) or name in (
        "native_nominal_qualification", "native_services_smoke", "telemetry", "trace_workload"):
        assert Path(module.__file__).resolve().is_relative_to(root), name
factory, binding = nominal_runtime.load_train_scenario(nominal_episode.TRAIN, purpose="engineering")
assert callable(factory) and binding["purpose"] == "engineering"
print(json.dumps({
    "scope": "ISOLATED_CPU_IMPORT_AND_TRAIN_BINDING_ONLY",
    "modules_from_package": True,
    "historical_identity_fixture_reconstructed": True,
    "reference_is_current_live_authority": False,
    "prior_content_sha256": binding["prior_content_sha256"],
    "module_paths": nominal_runtime.module_paths(),
    "Native_qualified": False,
}, sort_keys=True))
