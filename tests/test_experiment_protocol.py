import copy
import json
from pathlib import Path

import pytest

from experiment_protocol import (NOT_REPLAY, binding, capture_identity, capture_mismatches, collection_binding,
                                 expected_schedule, load_protocol, normalized_sha256, predeclaration_problems, same,
                                 validate_declarations)
from forecast_baselines import load_acceptance
from trace_control import make_plan

ROOT = Path(__file__).resolve().parents[1]
SHA256 = 'f' * 64
STEP = [dict(at_seconds=0, budget=10)]


def config(**overrides):
    """Runner settings as run_trace_pilot.py passes them, including the non-identity ones."""
    return dict(dict(trace='trace.json', output_dir='out/x', vllm_url='http://localhost:8000', protocol='p.json',
                     protocol_run='x', budget=10., budget_schedule=None, ttft_slo=.5, e2e_slo=None,
                     max_scheduling_lag=.05, max_budget_lag=.05, runner_gc_policy='automatic',
                     runner_max_rss_mib=None, runner_diagnostics=True, warmup_requests=3), **overrides)


def trace(**provenance):
    return dict(duration_seconds=100, model='m', output_cap=128, provenance=dict(dict(
        window_start='2024-05-11T17:01:00-07:00', preset=dict(name='experiment'), seed=1, sampling_probability=1.,
        time_scale=1., source_sha256='e' * 64), **provenance))


def document(status='proposed'):
    replay = {key: value for key, value in config().items() if key not in NOT_REPLAY}
    replay.update(max_scheduling_lag=.05, warmup_requests=3.)
    return dict(protocol='synthetic', replay=replay, schedules={'fixed-10': dict(fixed_budget=10)},
                traces=dict(dev=dict(sha256=SHA256, preset='experiment', seed=1, sample_probability=1, model='m',
                                     output_cap=128, window=['2024-05-12T00:01:00+00:00', '2024-05-12T00:02:40Z'])),
                runs=[dict(name='x', trace='dev', schedule='fixed-10', role='development'),
                      dict(name='t', trace='dev', schedule='fixed-10', role='test')],
                acceptance=dict(status=status))


def identity(settings=None, source=None, sha256=SHA256, schedule=STEP):
    settings, source = settings or config(), source or trace()
    return capture_identity(settings, source, sha256, make_plan(settings, source, sha256, schedule))


def test_same_compares_numbers_by_value_but_keeps_booleans_and_null_distinct():
    assert same(3, 3.) and same(.5, .5) and same('a', 'a') and same(None, None) and same(True, True)
    assert not same(True, 1) and not same(0, False) and not same(None, 0) and not same(1, '1')


def test_capture_matches_its_declaration_across_time_zones_and_numeric_types():
    capture = identity()
    assert capture['trace']['window_start'] == '2024-05-11T17:01:00-07:00' and capture['budget_mode'] == 'fixed'
    assert not set(capture['replay']) & NOT_REPLAY and capture['experiment_protocol'] is None
    assert capture_mismatches(document(), 'x', capture) == []


@pytest.mark.parametrize('change, problem', [
    (dict(sha256='e' * 64), 'trace sha256'),
    (dict(source=trace(window_start='2024-05-12T00:01:30+00:00')), 'trace window starts'),
    (dict(source=trace(window_start='not a time')), 'trace window starts'),
    (dict(source=dict(trace(), duration_seconds=90)), 'trace lasts 90 s, declared window 100 s'),
    (dict(source=trace(seed=2)), 'trace seed'),
    (dict(source=trace(preset=None)), 'trace preset'),
    (dict(source=dict(trace(), output_cap=64)), 'trace output_cap'),
    (dict(settings=config(budget=96.), schedule=[dict(at_seconds=0, budget=96)]), 'budget plan'),
    (dict(settings=config(ttft_slo=1.)), 'replay ttft_slo is 1.0, declared 0.5'),
    (dict(settings=config(cache_policy='enabled')), "replay cache_policy is 'enabled', declared 'nothing'"),
    (dict(settings={k: v for k, v in config().items() if k != 'warmup_requests'}), "replay warmup_requests is 'missing'"),
    (dict(settings=config(runner_diagnostics=1)), 'replay runner_diagnostics is 1, declared True'),
])
def test_capture_mismatches_name_each_differing_setting(change, problem):
    problems = capture_mismatches(document(), 'x', identity(**change))
    assert any(item.startswith('x: ' + problem) for item in problems), problems


def test_capture_mismatches_need_a_declared_run_and_recorded_provenance():
    assert capture_mismatches(document(), 'y', identity()) == ['y is not declared in the protocol runs']
    assert 'no capture provenance' in capture_mismatches(document(), 'x', None)[0]


def test_scheduled_declarations_hash_lf_normalized_files(tmp_path):
    steps = [dict(at_seconds=0, budget=10), dict(at_seconds=50, budget=5)]
    path = tmp_path / 'schedule.json'
    text = json.dumps(steps, indent=1)
    path.write_bytes(text.replace('\n', '\r\n').encode())
    declared = dict(path=str(path), budgets=[10, 5], sha256=normalized_sha256(path))
    path.write_bytes(text.encode())
    assert normalized_sha256(path) == declared['sha256']
    assert expected_schedule(declared, 100) == ('scheduled', [dict(at_seconds=0., budget=10.),
                                                              dict(at_seconds=50., budget=5.)])
    protocol = document()
    protocol['schedules']['step'] = declared
    protocol['runs'][0]['schedule'] = 'step'
    scheduled = identity(config(budget_schedule='schedule.json'), schedule=steps)
    assert capture_mismatches(protocol, 'x', scheduled) == []
    assert 'budget plan' in capture_mismatches(protocol, 'x', identity())[0]
    with pytest.raises(ValueError, match='declared budgets'):
        expected_schedule(dict(declared, budgets=[10, 6]), 100)
    path.write_text(json.dumps([dict(at_seconds=0, budget=10), dict(at_seconds=50, budget=6)]))
    with pytest.raises(ValueError, match='declared sha256'):
        expected_schedule(declared, 100)
    assert 'cannot be checked' in capture_mismatches(protocol, 'x', scheduled)[0]


def broken(edit):
    protocol = document()
    edit(protocol)
    return protocol


@pytest.mark.parametrize('protocol', [
    broken(lambda d: d.pop('traces')), broken(lambda d: d.update(schedules={})), broken(lambda d: d.pop('replay')),
    broken(lambda d: d['traces']['dev'].pop('sha256')), broken(lambda d: d['traces']['dev'].update(colour='red')),
    broken(lambda d: d['traces']['dev'].update(window=['2024-05-12T00:02:00Z', '2024-05-12T00:01:00Z'])),
    broken(lambda d: d['traces']['dev'].update(window=['yesterday', '2024-05-12T00:01:00Z'])),
    broken(lambda d: d['traces']['dev'].update(start=5)),
    broken(lambda d: d['schedules'].update(both=dict(fixed_budget=1, path='s.json', sha256=SHA256))),
    broken(lambda d: d['schedules'].update(unhashed=dict(path='s.json'))),
    broken(lambda d: d.update(runs=[])), broken(lambda d: d['runs'].append(dict(d['runs'][0]))),
    broken(lambda d: d['runs'][0].update(trace='other')), broken(lambda d: d['runs'][0].update(role='train')),
    broken(lambda d: d['runs'][0].pop('name')),
])
def test_validate_declarations_rejects_incomplete_protocols(protocol):
    with pytest.raises(ValueError):
        validate_declarations(protocol)


def test_collection_binding_refuses_undeclared_drifted_or_premature_captures(tmp_path):
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(document()))

    def bind(name='x', output=None, settings=None):
        return collection_binding(path, name, tmp_path / (output or name), settings or config(), trace(), SHA256, STEP)

    assert bind() == dict(path=str(path), protocol='synthetic', sha256=normalized_sha256(path), run='x',
                          role='development', acceptance_status='proposed')
    for kwargs, message in ((dict(name='y'), 'not declared'), (dict(output='other'), 'output directory'),
                            (dict(name='t'), 'frozen'), (dict(settings=config(ttft_slo=1.)), 'ttft_slo')):
        with pytest.raises(ValueError, match=message):
            bind(**kwargs)
    path.write_text(json.dumps(document('frozen')))
    assert bind('t')['acceptance_status'] == 'frozen' and bind('t')['role'] == 'test'


def test_predeclaration_requires_a_frozen_binding_to_the_same_bytes():
    frozen, proposed = document('frozen'), document('proposed')
    recorded = dict(path='p.json', **binding(frozen, SHA256, 't'))
    assert predeclaration_problems(frozen, SHA256, 't', dict(experiment_protocol=recorded)) == []
    assert 'no collection-time protocol binding' in predeclaration_problems(frozen, SHA256, 't', None)[0]
    assert 'no collection-time' in predeclaration_problems(frozen, SHA256, 't', dict(experiment_protocol=None))[0]
    edited = predeclaration_problems(frozen, 'e' * 64, 't', dict(experiment_protocol=recorded))
    assert len(edited) == 1 and 'recorded sha256' in edited[0]
    early = dict(experiment_protocol=dict(path='p.json', **binding(proposed, SHA256, 't')))
    assert predeclaration_problems(proposed, SHA256, 't', early) == [
        't was collected before the acceptance limits were frozen']
    refrozen = predeclaration_problems(frozen, SHA256, 't', early)
    assert len(refrozen) == 2 and "acceptance_status 'proposed', now 'frozen'" in refrozen[0]
    renamed = copy.deepcopy(recorded)
    renamed['run'] = 'x'
    assert 'recorded run' in predeclaration_problems(frozen, SHA256, 't', dict(experiment_protocol=renamed))[0]


def test_repository_protocol_declarations_load_and_resolve():
    protocol, sha256 = load_acceptance(ROOT / 'experiments' / 'excitation-v1.json', [1., 2., 5.])
    assert (protocol, sha256) == load_protocol(ROOT / 'experiments' / 'excitation-v1.json')
    assert protocol['acceptance']['status'] in ('proposed', 'frozen')
    duration = protocol['generator']['duration_seconds']
    for name, declared in protocol['schedules'].items():
        mode, steps = expected_schedule(declared, duration)
        budgets = declared.get('budgets', [declared.get('fixed_budget')])
        assert mode == ('fixed' if 'fixed_budget' in declared else 'scheduled'), name
        assert [step['budget'] for step in steps] == [float(budget) for budget in budgets], name
    for declared in protocol['traces'].values():
        assert declared.get('duration_seconds', duration) == duration
