"""Convert a frozen offered tape using a real, already-cached tokenizer.

Run only in the measurement harness, never in the controller/forecaster.
This performs CPU tokenization, not an inference or GPU performance experiment.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]

import nominal_workload as nw
import trace_workload as tw


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--purpose", choices=nw.PURPOSES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tape = nw.load_tape(args.tape, purpose=args.purpose)
    tokenizer_source = tw.MODEL
    if args.tokenizer_path is not None:
        if not args.tokenizer_path.is_dir():
            raise NotADirectoryError(args.tokenizer_path)
        tokenizer_source = str(args.tokenizer_path.resolve())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, trust_remote_code=False,
    )
    fingerprint = tw.tokenizer_fingerprint(tokenizer)
    if fingerprint is None:
        raise ValueError("A fingerprinted tokenizer is required")
    trace = nw.to_replay_trace(
        tape, model=tw.MODEL,
        prompt_for=lambda n, key: tw.synthetic_prompt(tokenizer, n, key),
        count_tokens=lambda prompt: tw.input_tokens(
            tokenizer, [{"role": "user", "content": prompt}],
        ),
        tokenizer_sha256=fingerprint,
    )
    tw.validate_trace(trace)
    tw.verify_prompts(trace, tokenizer)
    data = nw.canonical_bytes(trace)
    with args.output.open("xb") as handle:
        handle.write(data)
    print(json.dumps({
        "status": "REAL_TOKENIZER_CPU_VALIDATED_NOT_GPU_PERFORMANCE",
        "purpose": args.purpose,
        "model": tw.MODEL,
        "tokenizer_source": tokenizer_source,
        "tokenizer_sha256": fingerprint,
        "transformers_version": importlib.metadata.version("transformers"),
        "offered_tape_sha256": trace["provenance"]["offered_tape_sha256"],
        "replay_sha256": hashlib.sha256(data).hexdigest(),
        "replay_bytes": len(data),
        "offers": len(trace["requests"]),
        "measured_equals_requested": trace["provenance"]["measured_equals_requested_count"],
        "sources_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("prepare_replay.py", Path(__file__)),
                ("nominal_workload.py", HERE / "nominal_workload.py"),
                ("trace_workload.py", ROOT / "trace_workload.py"),
            )
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
