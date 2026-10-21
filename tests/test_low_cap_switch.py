import copy
import json

import pytest

import analyze_low_cap_switch as analysis
import run_cap_sweep as sweep
from experiment_protocol import load_protocol, normalized_sha256
from trace_workload import sha256_file
from test_cap_sweep import good_report, supervisor, write

PROTOCOL = sweep.ROOT / 'experiments/low-cap-switch-v1.json'


def test_frozen_low_cap_plan_swaps_phase_repeats_and_never_sends_both_budget_flags(tmp_path):
    document, _ = load_protocol(PROTOCOL)
    original = PROTOCOL.read_bytes()
    jobs = sweep.plan(document, tmp_path / 'raw', 'full', PROTOCOL)
    assert len(jobs) == 7 and jobs[0]['cap'] == 32 and jobs[0]['duration_s'] == 30
    assert [j['cap'] for j in jobs[1:]] == [32, [16, 32, 16, 32, 16], [32, 16, 32, 16, 32],
                                          16, [32, 16, 32, 16, 32], [16, 32, 16, 32, 16]]
    assert sum(j['duration_s'] for j in jobs[1:]) == 1800
    for job in jobs:
        config, argv = job['config'], job['command']
        assert '--shared-upstream-client' in argv and '--runner-gc-policy' in argv
        assert argv[argv.index('--protocol') + 1] == str(PROTOCOL)
        assert ('--budget' in argv) != ('--budget-schedule' in argv)
        assert config['budget'] == (job['cap'][0] if isinstance(job['cap'], list) else job['cap'])
    assert not (tmp_path / 'raw').exists() and PROTOCOL.read_bytes() == original
    assert len(sweep.plan(document, tmp_path / 'raw', 'smoke', PROTOCOL)) == 1


def test_schedule_hash_is_checked_before_any_collection(tmp_path):
    document, _ = load_protocol(PROTOCOL)
    bad = tmp_path / 'wrong.json'
    write(bad, [dict(at_seconds=0, budget=16), dict(at_seconds=60, budget=32)])
    document['schedules']['a']['path'] = str(bad)
    declared = tmp_path / 'protocol.json'
    write(declared, document)
    with pytest.raises(ValueError, match='sha256'):
        sweep.plan(document, tmp_path / 'raw', 'full', declared)
    assert not (tmp_path / 'raw').exists()


def test_supervisor_supports_list_cap_labels_and_explicit_protocol(supervisor, tmp_path):
    jobs = sweep.plan(None, None, None)
    jobs[0]['cap'] = [16., 32., 16.]
    sweep.run_suite(supervisor.output, 'full', tmp_path / 'frozen.json')
    result = sweep.read(supervisor.output / 'suite.json')
    assert result['valid'] and result['runs'][0]['cap'] == [16., 32., 16.]


@pytest.fixture
def captures(tmp_path, monkeypatch):
    source = tmp_path / 'raw'
    source.mkdir()
    document, _ = load_protocol(PROTOCOL)
    write(source / 'protocol.json', document)
    sha = normalized_sha256(source / 'protocol.json')
    backend = dict(host='same-host')
    records, manifests = [], {}
    for definition in document['runs']:
        path = source / definition['name']
        path.mkdir()
        declaration = document['schedules'][definition['schedule']]
        schedule = ([dict(at_seconds=0, budget=declaration['fixed_budget'])] if 'fixed_budget' in declaration
                    else sweep.read(sweep.ROOT / declaration['path']))
        smoke = definition['trace'] == 'smoke'
        times = [1., 16.] if smoke else [float(s + t) for s in range(0, 300, 60) for t in (1, 16)]
        rows = []
        for i, at in enumerate(times):
            cap = next(s['budget'] for s in reversed(schedule) if s['at_seconds'] <= at)
            met = cap == 32
            rows.append(dict(workload='trace', trace_id=str(i), scheduled_at=100. + at,
                             first_content_at=100. + at + .1 if met else None, finished_at=102. + at,
                             success=met, stream_complete=met, rejected=not met, ttft=.1 if met else None,
                             end_to_end=2., expected_input_tokens=800))
        trace = dict(duration_seconds=30. if smoke else 300.,
                     requests=[dict(trace_id=str(i), arrival_seconds=at) for i, at in enumerate(times)])
        write(path / 'trace.json', trace)
        write(path / 'client.json', dict(requests=rows))
        write(path / 'report.json', good_report())
        records.append(dict(name=definition['name'], status='passed', exitcode=0, backend=backend,
                            cap=schedule[0]['budget'] if len(schedule) == 1 else [s['budget'] for s in schedule],
                            artifact_sha256={p.name: sha256_file(p) for p in path.iterdir()}))
        manifests[definition['name']] = dict(host='same-host', code_sha256={'runtime': 'sha'},
                                            packages={'fastapi': 'test'}, trace_sha256='same')
    suite = dict(stage='full', status='collected', valid=True, protocol_sha256=sha, commit='rev', backend=backend,
                 planned=document['qualification']['order'], runs=records, packages={'fastapi': 'test'},
                 cpu_acceptance=dict(acceptance=dict(valid=True, commit='rev')))
    write(source / 'suite.json', suite)
    monkeypatch.setattr(analysis, 'qualify_capture', lambda path, *args: (manifests[path.name], good_report()))
    monkeypatch.setattr(analysis, 'audit_copy', lambda path: good_report())
    return source, tmp_path / 'analysis'


def test_analysis_pairs_same_arrivals_with_cap32_always_on_left_and_keeps_all_windows(captures):
    source, output = captures
    result = analysis.analyze(source, output, plot=True)
    assert result['valid'] and result['model_readiness'] == 'not_established'
    assert len(result['per_run']) == 6 and len(result['arrival_holds']) == 90
    assert len(result['arrival_5s_bins']) == 360
    assert len(result['cap32_minus_cap16_matched_phases']) == 30
    assert len(result['same_schedule_repeat2_minus_repeat1']) == 10
    for row in result['cap32_minus_cap16_matched_phases']:
        expected = 2 / 60 if row['phase'] == 'all' else 1 / (15 if row['phase'] == 'early' else 45)
        assert row['left_minus_right_goodput'] == pytest.approx(expected)
        assert row['right_only_met'] == 0
    assert all(r['left_minus_right_goodput'] == 0 for r in result['same_schedule_repeat2_minus_repeat1'])
    assert (output / 'phase-response.png').stat().st_size > 0
    assert (output / 'report.md').is_file()


@pytest.mark.parametrize('damage', ['raw', 'cap', 'host', 'missing', 'incomplete', 'cpu', 'binding'])
def test_analysis_refuses_damaged_or_incomplete_suites(captures, damage):
    source, output = captures
    suite = sweep.read(source / 'suite.json')
    if damage == 'raw':
        (source / suite['runs'][0]['name'] / 'client.json').write_text('{}')
    elif damage == 'cap':
        suite['runs'][0]['cap'] = 16
    elif damage == 'host':
        suite['runs'][0]['backend'] = dict(host='other')
    elif damage == 'missing':
        suite['runs'].pop()
    elif damage == 'incomplete':
        suite['status'] = 'failed'
    elif damage == 'cpu':
        suite['cpu_acceptance']['acceptance']['commit'] = 'other'
    else:
        suite['protocol_sha256'] = 'wrong'
    write(source / 'suite.json', suite)
    with pytest.raises(ValueError):
        analysis.analyze(source, output, plot=False)
    assert not output.exists()


def test_outcome_requires_completed_stream_and_unique_identity():
    row = dict(trace_id='a', success=True, stream_complete=False, scheduled_at=1., first_content_at=1.1)
    assert analysis.offered_outcomes([row], .5) == {'a': False}
    with pytest.raises(ValueError, match='Duplicate'):
        analysis.offered_outcomes([row, copy.copy(row)], .5)
