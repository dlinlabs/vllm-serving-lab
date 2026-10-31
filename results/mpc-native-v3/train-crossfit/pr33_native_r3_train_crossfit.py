"""Fixed two-block Train diagnostic of frozen R3 marginal release, not MPC."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
REPLICA = BASE / "pr33-native-v3-long01-replica"
SOURCE = BASE / "pr33-native-v3-round3-frozen/opus/experiments/mpc-research/native-v3/native_release_model.py"
EXPECTED_SOURCE = "98edb2df3b0bc9f5f4c8e9bedc0df5c4e08070f65408d32f63666004dc55ade3"
BLOCKS = (181001, 181002)
OUTPUT = BASE / "pr33-native-v3-r3-two-train-block-crossfit.json"


def main():
    assert not OUTPUT.exists(), "Preserve the original diagnostic"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SOURCE
    spec = importlib.util.spec_from_file_location("frozen_opus_r3", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    index = json.loads((REPLICA / "latest-index.json").read_text())
    assert index["source_commit"] == "6c11fe7e683439f379f0dfdbd21b05f5873b2c90"
    assert not index["mpc_active"]
    verified = {}
    for line in (REPLICA / "verified-artifact-sha256.jsonl").read_text().splitlines():
        row = json.loads(line)
        verified[row["artifact"]] = row["sha256"]
    episodes, receipts = {}, {}
    for block in BLOCKS:
        names = sorted(name for name in verified
                       if name.startswith(f"nv3-{block}-") and name.endswith(".json"))
        assert len(names) == 10, "Require every paired cell in a complete seed block"
        episodes[block] = []
        for name in names:
            raw = (REPLICA / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == verified[name]
            document = json.loads(raw)
            assert document["split"] == "train"
            episodes[block].append(module.episode_from_replica(document, index))
            receipts[name] = verified[name]
    started = time.monotonic()
    rows, stats = {}, {}
    for block in BLOCKS:
        stats[block] = {}
        rows[block] = module.build_dataset(episodes[block], stats[block])
        print("Parsed Train block", block, "origins", len(rows[block]), flush=True)
    result = dict(
        schema="native-v3-frozen-r3-two-train-block-crossfit-v1",
        scope="Train-only grouped diagnostic, not Calibration/Validation/Final, "
              "not a free-running joint transition or native MPC performance result",
        model_sha256=EXPECTED_SOURCE,
        collector_source_commit=index["source_commit"],
        artifact_sha256=receipts,
        fixed_blocks=BLOCKS,
        folds=[],
        origin_stats=stats,
        acceptance_threshold=None,
        native_performance_accepted=False,
    )
    for train, evaluate in ((181001, 181002), (181002, 181001)):
        print("Fitting Train block", train, flush=True)
        parameters = module.fit(rows[train], sensitivity=False)
        report = module.score(parameters, rows[evaluate])
        fold = dict(
            train_block=train, evaluate_train_block=evaluate,
            label_counts=parameters["label_counts"],
            fit_diagnostics=parameters["logistic"],
            scores=report,
        )
        result["folds"].append(fold)
        print(json.dumps(dict(train=train, evaluate=evaluate, scores=report)), flush=True)
    result["cpu_elapsed_s"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved", OUTPUT, "elapsed", result["cpu_elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
