"""Derive fixed engineering probes from the verified Train replay, never Eval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import trace_workload as tw


TRAIN_REPLAY_SHA256 = "5e418b8939179bac16d146d08f7acfa4d617b78adc4c324caca8ec0893ba52e7"


def build_probes(raw):
    if hashlib.sha256(raw).hexdigest() != TRAIN_REPLAY_SHA256:
        raise ValueError("Expected the frozen real-tokenizer Train replay")
    train = tw.validate_trace(json.loads(raw))
    result = {}
    selections = (
        ("warmup", [next(row for row in train["requests"] if row["input_tokens"] == n)
                    for n in (256, 1024, 3072)], (0.0, 1.0, 2.0), 3,
         "first Train row of each token class"),
        ("qualification", train["requests"][:4], (0.25, 1.25, 2.25, 3.25), 5,
         "first four Train rows"),
        ("admission-boundary",
         [row for row in train["requests"] if row["input_tokens"] == 3072][:5],
         (1.25,) * 5, 5, "first five Train rows of class 3072 in one offered burst"),
    )
    for kind, selected, offsets, duration, derivation in selections:
        trace = {key: train[key] for key in ("schema_version", "max_model_len", "model", "output_cap")}
        trace["duration_seconds"] = duration
        trace["requests"] = [
            dict(row, trace_id=f"{kind}-{row['trace_id']}", arrival_seconds=offset)
            for row, offset in zip(selected, offsets)
        ]
        trace["provenance"] = {
            "kind": "ENGINEERING_PROBE_FROM_FROZEN_TRAIN_REPLAY",
            "purpose": "engineering",
            "probe": kind,
            "source_train_replay_sha256": TRAIN_REPLAY_SHA256,
            "source_request_ids": [row["trace_id"] for row in selected],
            "tokenizer_sha256": train["provenance"]["tokenizer_sha256"],
            "derivation": derivation,
            "not_a_held_out_or_performance_comparison": True,
        }
        tw.validate_trace(trace)
        result[f"{kind}.json"] = (
            json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--train-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = build_probes(args.train_replay.read_bytes())
    args.output_dir.mkdir(mode=0o700)
    summary = {}
    for name, data in outputs.items():
        with (args.output_dir / name).open("xb") as stream:
            stream.write(data)
        summary[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    print(json.dumps({"scope": "engineering_probes_not_Native_results", "files": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
