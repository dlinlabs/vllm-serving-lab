"""Frozen CLI inputs must not change silently or be mislabeled in output."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys
import types

import pytest

import prepare_trace
from trace_workload import sha256_file
from test_trace_replay import Tokenizer


@pytest.fixture
def source_and_presets(tmp_path, monkeypatch):
    source = tmp_path / 'source.csv'
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['TIMESTAMP', 'ContextTokens', 'GeneratedTokens'])
        for at in (0, .1, .5, 1, 1.1, 1.5, 2):
            writer.writerow([datetime.fromtimestamp(at, timezone.utc).isoformat(), 8, 5000])
    config = json.loads(prepare_trace.PRESET_FILE.read_text())
    for name, start in (('smoke', '1970-01-01T00:00:00Z'),
                        ('experiment', '1970-01-01T00:00:01Z')):
        config['presets'][name].update(start=start, duration_seconds=.5,
                                       source_sha256=sha256_file(source))
    preset_file = tmp_path / 'presets.json'
    preset_file.write_text(json.dumps(config))
    monkeypatch.setattr(prepare_trace, 'PRESET_FILE', preset_file)
    return source, preset_file


@pytest.mark.parametrize('name,expected_start', [('smoke', '1970-01-01T00:00:00Z'),
                                               ('experiment', '1970-01-01T00:00:01Z')])
def test_preset_cli_writes_frozen_provenance_and_exact_window(source_and_presets, tmp_path, monkeypatch,
                                                           name, expected_start):
    source, preset_file = source_and_presets
    class CachedTokenizer(Tokenizer):
        init_kwargs = {'_commit_hash': 'mock-tokenizer-revision'}
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs['local_files_only']
            return cls()
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=CachedTokenizer))
    output = tmp_path / 'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare_trace.py', '--source', str(source),
                                    '--preset', name, '--output-dir', str(output)])
    prepare_trace.main()
    data = json.loads((output / 'trace.json').read_text())
    provenance = data['provenance']
    record = provenance['preset']
    assert provenance['window_start'] == expected_start
    assert record['name'] == name
    assert record['config_sha256'] == hashlib.sha256(preset_file.read_bytes()).hexdigest()
    assert record['settings'] == json.loads(preset_file.read_text())['presets'][name]
    assert provenance['source_sha256'] == record['settings']['source_sha256']
    assert [r['arrival_seconds'] for r in data['requests']] == pytest.approx([0, .1])
    assert len(data['requests']) == 2 and data['duration_seconds'] == .5
    assert all(r['max_tokens'] == 128 for r in data['requests'])
    assert len((output / 'source_audit.jsonl').read_text().splitlines()) == 2
    # Re-preparing the same profile must produce identical inputs and provenance.
    second = tmp_path / 'prepared-again'
    monkeypatch.setattr(sys, 'argv', ['prepare_trace.py', '--source', str(source),
                                    '--preset', name, '--output-dir', str(second)])
    prepare_trace.main()
    assert (output / 'trace.json').read_bytes() == (second / 'trace.json').read_bytes()
    assert (output / 'source_audit.jsonl').read_bytes() == (second / 'source_audit.jsonl').read_bytes()


@pytest.mark.parametrize('override', [('--duration-seconds', '30'), ('--seed', '1'),
                                   ('--sample-probability', '1'), ('--output-cap', '128'),
                                   ('--model', 'other')])
def test_preset_rejects_even_matching_explicit_overrides(source_and_presets, override, capsys):
    source, _ = source_and_presets
    with pytest.raises(SystemExit) as exc:
        prepare_trace.parse_args(['--source', str(source), '--preset', 'smoke', '--output-dir', 'unused',
                                  *override])
    assert exc.value.code == 2
    assert 'Preset settings are frozen' in capsys.readouterr().err


def test_wrong_source_is_rejected_and_custom_mode_remains_available(source_and_presets, capsys):
    source, _ = source_and_presets
    source.write_text(source.read_text() + '\n')
    common = ['--source', str(source), '--output-dir', 'unused']
    with pytest.raises(SystemExit) as exc:
        prepare_trace.parse_args([*common, '--preset', 'smoke'])
    assert exc.value.code == 2 and 'checksum mismatch' in capsys.readouterr().err
    args = prepare_trace.parse_args([*common, '--start', '1970-01-01T00:00:00Z', '--seed', '42'])
    assert args.preset_record is None and args.preset is None
    assert args.seed == 42 and args.duration_seconds == 300 and args.output_cap == 128
    with pytest.raises(SystemExit):
        prepare_trace.parse_args([*common, '--start', args.start, '--preset', 'smoke'])
