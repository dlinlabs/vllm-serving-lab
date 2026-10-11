"""Prepare a continuous Azure window; preserve timestamps, use a fixed output cap."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from trace_workload import (MODEL, MAX_MODEL_LEN, sha256_file, synthetic_prompt,
                            validate_trace, input_tokens, tokenizer_fingerprint)

SOURCE_URL = 'https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md'
PRESET_FILE = Path(__file__).resolve().parent / 'experiments' / 'trace-presets.json'
DEFAULTS = dict(duration_seconds=300, sample_probability=1, seed=1, output_cap=128, model=MODEL)


def timestamp(value):
    dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    # Only relative intervals are used. No inference about local usage hours.
    return dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()


def prepare(source, start, duration, sample_probability, seed, output_cap, tokenizer,
            model=MODEL, tokenizer_revision=None):
    if not math.isfinite(duration) or not 0 < sample_probability <= 1 or duration <= 0:
        raise ValueError('Duration and sampling probability must be positive; probability <= 1')
    if type(output_cap) is not int or not 0 < output_cap < MAX_MODEL_LEN:
        raise ValueError('Invalid fixed output cap')
    source_hash = sha256_file(source)
    beginning = timestamp(start)
    end = beginning + duration
    rows, audit = [], []
    counts = dict(in_window=0, sampled_out=0, context_overflow=0, context_too_short=0, retained=0)
    overhead = input_tokens(tokenizer, [{'role': 'user', 'content': ''}])
    previous = None
    covers_end = False
    with Path(source).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'TIMESTAMP', 'ContextTokens', 'GeneratedTokens'} <= set(reader.fieldnames or []):
            raise ValueError('Source must have Azure TIMESTAMP, ContextTokens, GeneratedTokens columns')
        for number, raw in enumerate(reader, 2):
            at = timestamp(raw['TIMESTAMP'])
            if previous is None and at > beginning:
                raise ValueError('Requested window starts before the first supplied source timestamp')
            if previous is not None and at < previous:
                raise ValueError(f'Source timestamps are not sorted at CSV line {number}')
            previous = at
            if at >= end:
                covers_end = True
                break
            if not beginning <= at < end:
                continue
            context, generated = int(raw['ContextTokens']), int(raw['GeneratedTokens'])
            if context < 0 or generated < 0:
                raise ValueError(f'Negative source token count at line {number}')
            counts['in_window'] += 1
            if counts['in_window'] > 1000000:
                raise ValueError('Source window exceeds one million rows; select a shorter window')
            identifier = f'{source_hash[:16]}:{number}'
            draw = int(hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest(), 16) / 2**256
            reason = ('sampled_out' if draw >= sample_probability else
                      'context_overflow' if context + output_cap > MAX_MODEL_LEN else
                      'context_too_short' if context < overhead else 'retained')
            counts[reason] += 1
            audit.append(dict(trace_id=identifier, source_line=number, timestamp=raw['TIMESTAMP'],
                              context_tokens=context, generated_tokens=generated, disposition=reason))
            if reason == 'retained':
                prompt = synthetic_prompt(tokenizer, context, f'{seed}:{identifier}')
                rows.append(dict(trace_id=identifier, arrival_seconds=at - beginning,
                                 input_tokens=context, max_tokens=output_cap, prompt=prompt))
                if len(rows) > 90000:
                    raise ValueError('Too many retained requests; select a shorter window or thin the source')
    if not covers_end:
        raise ValueError('Supplied source does not cover the complete requested window')
    trace = dict(schema_version=1, model=model, max_model_len=MAX_MODEL_LEN,
                 duration_seconds=duration, output_cap=output_cap, requests=rows,
                 provenance=dict(source_url=SOURCE_URL, source_sha256=source_hash, window_start=start,
                                 sampling_probability=sample_probability, seed=seed, time_scale=1.0,
                                 tokenizer_revision=tokenizer_revision, selection_counts=counts,
                                 tokenizer_sha256=tokenizer_fingerprint(tokenizer),
                                 source_coverage_checked_through=previous,
                                 timestamp_basis='Naive timestamps use UTC for relative arithmetic only',
                                 output_policy='fixed cap, natural EOS; source output lengths are audit-only'))
    validate_trace(trace)
    return trace, audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--start', help='Custom ISO source timestamp; inclusive')
    selection.add_argument('--preset', choices=('smoke', 'experiment'),
                           help='Versioned, fixed source window and preparation settings')
    parser.add_argument('--duration-seconds', type=float)
    parser.add_argument('--sample-probability', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-cap', type=int)
    parser.add_argument('--model')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    args.preset_record = None
    if args.preset:
        overrides = [name for name in DEFAULTS if getattr(args, name) is not None]
        if overrides:
            parser.error('Preset settings are frozen; use --start for custom settings instead of overriding '
                         + ', '.join('--' + name.replace('_', '-') for name in overrides))
        # Hash the same bytes that we parse so the recorded protocol is unambiguous.
        raw = PRESET_FILE.read_bytes()
        config = json.loads(raw)
        if config['schema_version'] != 1:
            parser.error('Unsupported preset schema version')
        profile = config['presets'][args.preset]
        for name in ('start', *DEFAULTS):
            setattr(args, name, profile[name])
        args.preset_record = dict(name=args.preset, protocol_version=config['protocol_version'],
                                  config_sha256=hashlib.sha256(raw).hexdigest(), settings=profile)
        if sha256_file(args.source) != profile['source_sha256']:
            parser.error('Preset source checksum mismatch; use the documented prefix download, '
                         'or --start for a custom source')
    else:
        for name, value in DEFAULTS.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    return args


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    revision = tokenizer.init_kwargs.get('_commit_hash')
    trace, audit = prepare(args.source, args.start, args.duration_seconds, args.sample_probability,
                           args.seed, args.output_cap, tokenizer, args.model, revision)
    if args.preset_record:
        if trace['provenance']['source_sha256'] != args.preset_record['settings']['source_sha256']:
            raise ValueError('Preset source changed during preparation')
        trace['provenance']['preset'] = args.preset_record
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'trace.json').write_text(json.dumps(trace, indent=2, allow_nan=False) + '\n')
    (output / 'source_audit.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in audit))
    print(json.dumps({'trace': str(output / 'trace.json'), **trace['provenance']['selection_counts']}, indent=2))


if __name__ == '__main__':
    main()
