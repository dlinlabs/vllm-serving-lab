import hashlib
import json
import sys

path = sys.argv[1]
raw = open(path, 'rb').read()
raw.decode('ascii')


def no_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate key in object: %r' % keys)
    return dict(pairs)


def reject_constant(name):
    raise ValueError('non-finite constant ' + name)


doc = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
required = ['schema', 'written_utc', 'status', 'purpose', 'preregistration_note', 'worktree', 'author', 'basis',
            'gateway_facts_140_205', 'recorder_clock_facts', 'l5_erratum', 'prior_fix_refinements',
            'b2_requirements', 'b3_requirements', 'acceptance_procedure', 'decision_rule', 'nonclaims', 'limitations']
missing = [key for key in required if key not in doc]
assert not missing, missing
author = doc['author']
assert author['model_id'] == 'claude-opus-5.5' and author['reasoning_effort'] == 'max', author
assert author['child_agents'] == 0, author
assert doc['status'] == 'PLAN_ONLY_NOT_ACCEPTANCE', doc['status']
assert doc['l5_erratum']['status'].startswith('WITHDRAWN'), doc['l5_erratum']['status']
ids = []
levels = {'MUST': 0, 'SHOULD': 0}
for section in ('b2_requirements', 'b3_requirements'):
    for row in doc[section]:
        for key in ('id', 'level', 'requirement', 'pass_if', 'fail_if'):
            assert isinstance(row.get(key), str) and row[key], (section, row.get('id'), key)
        assert set(row) == {'id', 'level', 'requirement', 'pass_if', 'fail_if'}, row['id']
        assert row['level'] in levels, row['id']
        levels[row['level']] += 1
        ids.append(row['id'])
assert len(ids) == len(set(ids)), 'duplicate requirement ids'
print('OK', sys.version.split()[0], len(raw), hashlib.sha256(raw).hexdigest(), 'rows', len(ids), levels)
