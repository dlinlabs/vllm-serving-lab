import hashlib
import json
import math
import sys

REQUIRED = ('schema', 'author', 'mandate', 'verdicts', 'pins', 'f_closure', 'new_findings', 'observations',
            'driver_obligations_not_component_defects', 'owner_disclosed_untested_paths', 'witness_w01',
            'review_coverage', 'read_disclosure', 'limitations', 'nonclaims', 'artifacts')


def pairs(items):
    keys = [key for key, _ in items]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate JSON object key')
    return dict(items)


def reject_constant(token):
    raise ValueError(f'non-finite JSON constant {token}')


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f'non-finite float literal {text}')
    return value


raw = open(sys.argv[1], 'rb').read()
doc = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject_constant,
                 parse_float=finite_float)
missing = [key for key in REQUIRED if key not in doc]
verdicts = {name: doc['verdicts'][name]['verdict'] for name in ('RECORDER', 'METRICS')}
ids = [entry['id'] for entry in doc['f_closure']]
problems = []
if missing:
    problems.append(f'missing keys {missing}')
if any(value not in ('PASS', 'FAIL') for value in verdicts.values()):
    problems.append('verdict not PASS/FAIL')
if ids != [f'F{index}' for index in range(1, 10)]:
    problems.append(f'f_closure ids {ids}')
if doc['author'].get('model_id') != 'claude-opus-5.5' or doc['author'].get('reasoning_effort') != 'max':
    problems.append('author identity')
print(json.dumps({'python': sys.version.split()[0], 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'ascii': raw.isascii(), 'verdicts': verdicts, 'f_ids': ids,
                  'new_findings': [entry['id'] for entry in doc['new_findings']],
                  'observations': [entry['id'] for entry in doc['observations']],
                  'problems': problems, 'ok': not problems}, sort_keys=True))
sys.exit(1 if problems else 0)
