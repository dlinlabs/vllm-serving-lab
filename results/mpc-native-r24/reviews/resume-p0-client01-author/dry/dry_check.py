"""Parse-only dry check (SELECTION-AMENDMENT-04): recorder loader/purpose/binding path on frozen inputs.

Never prints prompts, request rows or outcomes; never loads a tokenizer/model; never opens a socket; Eval never read.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(ROOT / 'experiments' / 'mpc-research' / 'resume-mvp'), str(ROOT)]
import nominal_client as nc  # noqa: E402
import trace_workload as tw  # noqa: E402

ALIAS = 'Qwen/Qwen3-4B-Instruct-2507'
LOCAL = '/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507'
TOKENIZER = '4daabcf7fab3bce457d30dbd14e7eb9c5e79d77c15916f6abdb85b7b8bf0d2c5'
INPUTS = {
    'train_replay': ('.runtime/r24-resume-p0-tokenizer02/returned/train.replay.json',
                     '5e418b8939179bac16d146d08f7acfa4d617b78adc4c324caca8ec0893ba52e7', 'train'),
    'qualification': ('results/mpc-native-r24/reviews/resume-p0-probes01/inputs/qualification.json',
                      'fe480796606b86aec3b2d4c91388a4846d98035b5a9810131e39fd40b627b333', 'engineering'),
    'warmup': ('results/mpc-native-r24/reviews/resume-p0-probes01/inputs/warmup.json',
               '3a893f485e11204088f03cf08415d1e9d56028ce0a7ee476477ad3d24c2808b2', 'engineering'),
}
BINDING = {
    'schema': nc.BINDING_SCHEMA, 'served_alias': ALIAS, 'local_model_path': LOCAL, 'tokenizer_sha256': TOKENIZER,
    'served_model_names': [ALIAS],
    'backend_argv': ['vllm', 'serve', LOCAL, '--served-model-name', ALIAS, '--scheduling-policy', 'fcfs',
                     '--no-enable-prefix-caching', '--max-model-len', '8192'],
    'backend_argv_source': 'literal dry-check argv (no launched backend; NOT launch proof)',
}


def shape(value):
    if isinstance(value, str):
        return value[:200]
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, dict):
        return {'type': 'dict', 'keys': sorted(value)}
    if isinstance(value, list):
        return {'type': 'list', 'len': len(value)}
    return {'type': type(value).__name__}


report = {'python': sys.version.split()[0], 'binding_literal': BINDING, 'inputs': {}}
for name, (relative, expected, declared) in INPUTS.items():
    entry = report['inputs'][name] = {'path': relative, 'sha256_expected': expected, 'declared_purpose': declared}
    try:
        loaded, data = nc._strict_load(ROOT / relative)
    except nc.ClientRefusal as exc:
        entry['strict_load'] = f'ClientRefusal: {str(exc)[:300]}'
        continue
    entry.update(strict_load='ok', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    entry['sha256_match'] = entry['sha256'] == expected
    try:
        trace = tw.validate_trace(loaded)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        entry['validate_trace'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        continue
    provenance = trace.get('provenance') if isinstance(trace.get('provenance'), dict) else {}
    entry.update(validate_trace='ok', model=trace['model'], n_requests=len(trace['requests']),
                 duration_seconds=trace['duration_seconds'], output_cap=trace['output_cap'],
                 provenance_keys=sorted(provenance), tape={key: shape(provenance.get(key)) for key in nc.TAPE_KEYS})
    try:
        entry['declared_purpose_cli'] = nc._purpose(declared)
    except Exception as exc:  # argparse.ArgumentTypeError
        entry['declared_purpose_cli'] = f'refused: {exc}'
    entry['provenance_purpose_equals_declared'] = provenance.get('purpose') == declared
    entry['finalizable_core_purpose'] = provenance.get('purpose') in nc.nm.PURPOSES
    try:
        evidence = nc.check_binding(trace, BINDING)
        entry['check_binding'] = {key: evidence[key] for key in ('served_alias', 'local_model_path',
                                                                 'argv_scheduling_policy', 'argv_max_model_len')}
    except nc.ClientRefusal as exc:
        entry['check_binding'] = f'ClientRefusal: {exc}'
with open(OUT, 'x') as handle:
    json.dump(report, handle, sort_keys=True, indent=1)
    handle.write('\n')
print(json.dumps({name: {key: entry.get(key) for key in ('sha256_match', 'validate_trace', 'n_requests',
                                                         'provenance_purpose_equals_declared', 'check_binding')}
                  for name, entry in report['inputs'].items()}, sort_keys=True))
