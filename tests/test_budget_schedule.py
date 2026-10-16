"""The excitation schedules are regenerable, constrained and bound to the protocol by hash."""
import hashlib
import json
from pathlib import Path

import pytest

import make_budget_schedule
from make_budget_schedule import candidates, generate, schedule, select
from trace_control import validate_schedule

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments' / 'excitation-v1.json'
LEVELS = (48., 64., 80., 96.)


def protocol():
    return json.loads(PROTOCOL.read_text())


def lf_sha256(path):
    # Windows checkouts may convert line endings; the protocol hashes the committed LF bytes.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def budgets(steps):
    return tuple(float(step['budget']) for step in steps)


def directed(sequence):
    return set(zip(sequence, sequence[1:]))


def test_candidates_are_anchored_balanced_and_cover_every_pair():
    options = candidates(LEVELS, 10, 96.)
    assert len(options) == len(set(options)) == 312
    for sequence in options:
        assert sequence[0] == sequence[-1] == 96. and len(sequence) == 10
        assert all(a != b for a, b in zip(sequence, sequence[1:]))
        counts = [sequence.count(level) for level in LEVELS]
        assert max(counts) - min(counts) <= 1
        assert {tuple(sorted(pair)) for pair in directed(sequence)} == {
            (a, b) for a in LEVELS for b in LEVELS if a < b}


def test_selection_is_deterministic_and_excludes_reversals():
    options = candidates(LEVELS, 10, 96.)
    first = select(options, 1)
    assert select(list(options), 1) == first and select(options, 2) != first
    for seed in range(20):
        assert select(options, seed, [first]) not in (first, tuple(reversed(first)))
    with pytest.raises(ValueError, match='No candidate'):
        select(options[:1], 1, options[:1])


def test_frozen_schedules_regenerate_and_match_the_protocol():
    spec = protocol()
    settings = spec['generator']
    args = (settings['budgets'], settings['anchor'], settings['hold_seconds'], settings['duration_seconds'])
    generated = {name: item for name, item in spec['schedules'].items() if name in ('a', 'b', 'c')}
    files = {name: json.loads((ROOT / item['path']).read_text()) for name, item in generated.items()}
    a, count = generate(*args, spec['schedules']['a']['seed'])
    assert count == 312 and files['a'] == a
    assert files['b'] == schedule(list(reversed(budgets(a))), settings['hold_seconds'])
    c, _ = generate(*args, spec['schedules']['c']['seed'], [budgets(a)])
    assert files['c'] == c
    for name, item in generated.items():
        assert lf_sha256(ROOT / item['path']) == item['sha256']
        assert budgets(files[name]) == tuple(map(float, item['budgets']))
        assert validate_schedule(files[name], settings['duration_seconds'])
        assert files[name][0]['budget'] == settings['anchor']
        assert (ROOT / item['path']).read_bytes().replace(b'\r\n', b'\n').decode() == \
            make_budget_schedule.dumps(files[name])
    pairs = {(x, y) for x in LEVELS for y in LEVELS if x != y}
    assert directed(budgets(files['a'])) | directed(budgets(files['b'])) == pairs
    assert budgets(files['c']) not in (budgets(files['a']), budgets(files['b']))


def test_protocol_runs_and_acceptance_are_consistent():
    spec = protocol()
    runs = {run['name']: run for run in spec['runs']}
    assert all(run['schedule'] in spec['schedules'] and run['trace'] in spec['traces'] for run in runs.values())
    acceptance = spec['acceptance']
    assert acceptance['status'] in ('proposed', 'frozen') and acceptance['model'] == 'ridge'
    assert acceptance['test_runs'] == [name for name, run in runs.items() if run['role'] == 'test']
    assert all(runs[name]['trace'] == 'held_out' for name in acceptance['test_runs'])
    development = [name for name, run in runs.items() if run['role'] == 'development']
    assert set(development) <= set(acceptance['training_runs'])
    assert not set(acceptance['training_runs']) & set(acceptance['test_runs'])
    assert [run['name'] for run in spec['runs']][-1] == acceptance['test_runs'][-1]


def test_generate_rejects_unusable_settings():
    with pytest.raises(ValueError, match='whole number'):
        generate(LEVELS, 96, 30, 290, 1)
    with pytest.raises(ValueError, match='anchor'):
        generate(LEVELS, 100, 30, 300, 1)
    with pytest.raises(ValueError, match='Too few'):
        generate(LEVELS, 96, 60, 240, 1)
    with pytest.raises(ValueError, match='Too many'):
        generate(LEVELS, 96, 10, 300, 1)


def test_cli_writes_lf_schedules_once(tmp_path, monkeypatch, capsys):
    forward, reverse = tmp_path / 'a.json', tmp_path / 'b.json'
    argv = ['make_budget_schedule.py', '--budgets', '48', '64', '80', '96', '--anchor', '96',
            '--hold-seconds', '30', '--duration-seconds', '300', '--seed', '1',
            '--output', str(forward), '--reverse-output', str(reverse)]
    monkeypatch.setattr('sys.argv', argv)
    make_budget_schedule.main()
    assert b'\r' not in forward.read_bytes()
    assert json.loads(forward.read_text()) == generate(LEVELS, 96, 30, 300, 1)[0]
    assert budgets(json.loads(reverse.read_text())) == tuple(reversed(budgets(json.loads(forward.read_text()))))
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed[0]['sha256'] == hashlib.sha256(forward.read_bytes()).hexdigest()
    before = forward.read_bytes()
    with pytest.raises(SystemExit):
        make_budget_schedule.main()
    assert forward.read_bytes() == before
    monkeypatch.setattr('sys.argv', argv[:-4] + ['--output', str(tmp_path / 'c.json'), '--exclude', str(forward)])
    make_budget_schedule.main()
    excluded = budgets(json.loads((tmp_path / 'c.json').read_text()))
    assert excluded not in (budgets(json.loads(forward.read_text())), budgets(json.loads(reverse.read_text())))
