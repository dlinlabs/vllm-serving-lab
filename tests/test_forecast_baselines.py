import json
import math
import warnings

import numpy as np
import pytest

from forecast_baselines import (Ridge, analysis_mismatches, change_directions, evaluate, judge, load, load_acceptance,
                                main, metrics, near_change, persistence, splits)
from identification_features import PROTOCOL, SCHEMA_VERSION


def level(k, phase=0.):
    return 6. + 4. * math.sin((k + phase) / 5.)


def synthetic_rows(count=200, step=.5, delay=20., budget=10., phase=0., change_at=None, headroom=None):
    """awaiting(t+1) = 2 awaiting(t) + 1 exactly; cohort labels resolve `delay` seconds late."""
    rows = []
    for k in range(count):
        t = k * step
        now = budget if change_at is None or t < change_at else budget / 2
        later = budget if change_at is None or t + 1 < change_at else budget / 2
        awaiting = level(k, phase)
        rows.append(dict(
            t_s=t,
            features=dict(budget=now, since_budget_value_change_s=None if change_at is None or t < change_at
                          else t - change_at, awaiting_first_content=awaiting, received_1s=awaiting + 2.,
                          received_5s=5. * awaiting, resolved_ttft_met_1s=1.,
                          engine_waiting=None if k % 3 == 0 else awaiting / 2, engine_missing=k % 3 == 0),
            control={'1': dict(budget_mean=later, budget_end=later)},
            labels={'1': dict(awaiting_first_content=2. * awaiting + 1., received=awaiting + 2.,
                              cohort_ttft_met=None if k == 7 else 1., cohort_resolved_s=t + 1. + delay)},
            label_window_complete={'1': t + 1. <= count * step}))
        if headroom is not None:
            rows[-1]['features']['budget_headroom'] = headroom(k)
    return rows


def synthetic_runs(**variants):
    variants = variants or dict(only={})
    return {name: dict(meta=dict(duration_s=100.), rows=synthetic_rows(**options), sha256=name)
            for name, options in variants.items()}


def find(evaluation, **criteria):
    return next(row for row in evaluation['results'] if all(row[k] == v for k, v in criteria.items()))


def test_ridge_recovers_linear_dynamics_and_beats_persistence():
    evaluation = evaluate(synthetic_runs(a={}, b=dict(phase=3.)), horizons=[1.], windows=[1., 5.], alpha=1e-6)
    row = find(evaluation, split='chronological', scope='pooled', target='awaiting_first_content', subset='all')
    assert row['models']['ridge']['mae'] < 1e-3
    assert row['skill_mae']['ridge_vs_persistence'] > .99 and row['skill_mae']['ridge_vs_mean'] > .99
    expected = [level(k, phase) + 1. for phase in (0., 3.) for k in range(200) if k * .5 >= 200 / 3 and k * .5 <= 99.]
    assert row['models']['persistence']['mae'] == pytest.approx(np.mean(expected))
    assert evaluation['targets'] == ['awaiting_first_content', 'received', 'cohort_ttft_met']
    assert evaluation['feature_columns'][-2:] == ['control.budget_mean', 'control.budget_end']


def test_chronological_split_embargoes_labels_resolved_after_the_cut():
    evaluation = evaluate(synthetic_runs(), horizons=[1.], windows=[1.])
    sizes = {row['target']: row['n_train'] for row in evaluation['training'] if row['split'] == 'chronological'}
    cut = 2 / 3 * 100.
    assert sizes['awaiting_first_content'] == sum(k * .5 + 1. <= cut for k in range(200))
    assert sizes['cohort_ttft_met'] == sum(k * .5 + 21. <= cut for k in range(200)) - 1
    row = find(evaluation, split='chronological', scope='only', target='awaiting_first_content', subset='all')
    assert row['models']['ridge']['n'] == sum(k * .5 >= cut and k * .5 + 1. <= 100. for k in range(200))
    prepared = dict(x=dict(t=np.array([0., 50., 70.]), known=np.array([1., 80., 71.]), duration=100.))
    train, test = next(splits(prepared, .5))[1](prepared)
    assert train['x'].tolist() == [True, False, False] and test['x'].tolist() == [False, True, True]


def test_leave_one_run_out_never_trains_on_the_held_out_run():
    evaluation = evaluate(synthetic_runs(a={}, b=dict(phase=2.), c=dict(phase=4.)), horizons=[1.], windows=[1.])
    held = {row['split']: row['train_runs'] for row in evaluation['training']
            if row['split'].startswith('leave_one_run_out') and row['target'] == 'received'}
    assert held == {'leave_one_run_out:a': ['b', 'c'], 'leave_one_run_out:b': ['a', 'c'],
                    'leave_one_run_out:c': ['a', 'b']}
    scopes = {row['scope'] for row in evaluation['results'] if row['split'] == 'leave_one_run_out:b'}
    assert scopes == {'b'}


def test_persistence_holds_state_and_extends_the_nearest_trailing_rate():
    rows = synthetic_rows(count=3)
    assert persistence(rows, 'awaiting_first_content', 5., [1.]).tolist() == [level(k) for k in range(3)]
    assert persistence(rows, 'received', 2., [1., 5.]) == pytest.approx([2. * (level(k) + 2.) for k in range(3)])
    assert persistence(rows, 'received', 4., [1., 5.]) == pytest.approx([4. * level(k) for k in range(3)])
    assert persistence(rows, 'cohort_ttft_met', 3., [1., 5.]) == pytest.approx([3.] * 3)


def test_near_change_uses_past_changes_and_in_horizon_changes_only():
    rows = synthetic_rows(count=40, change_at=10.)
    mask = near_change(rows, '1')
    times = [row['t_s'] for row in rows]
    assert [t for t, flag in zip(times, mask) if flag] == [t for t in times if 9. <= t <= 20.]


def test_change_directions_partition_near_change_by_step_direction():
    down = synthetic_rows(count=40, change_at=10.)
    increase, decrease = change_directions(down, '1')
    assert not increase.any() and decrease.tolist() == near_change(down, '1').tolist()
    up = synthetic_rows(count=40, change_at=10.)
    for row in up:
        row['features']['budget'] = 15. - row['features']['budget']
        row['control']['1'] = {name: 15. - value for name, value in row['control']['1'].items()}
    increase, decrease = change_directions(up, '1')
    assert not decrease.any() and increase.tolist() == near_change(up, '1').tolist()


def test_designated_test_runs_never_enter_development_splits():
    evaluation = evaluate(synthetic_runs(a={}, b=dict(phase=2.), c=dict(phase=4., change_at=60.)),
                          horizons=[1.], windows=[1.], test_runs=['c'])
    trained = {row['split']: row['train_runs'] for row in evaluation['training'] if row['target'] == 'received'}
    assert trained == {'chronological': ['a', 'b'], 'leave_one_run_out:a': ['b'], 'leave_one_run_out:b': ['a'],
                       'designated_test': ['a', 'b']}
    scored = {(row['split'], row['scope']) for row in evaluation['results']}
    assert ('designated_test', 'c') in scored and ('designated_test', 'pooled') not in scored
    assert not any(scope == 'c' for split, scope in scored if split != 'designated_test')
    row = find(evaluation, split='designated_test', target='awaiting_first_content', subset='near_budget_decrease')
    assert row['models']['ridge']['n'] and not find(
        evaluation, split='designated_test', target='awaiting_first_content',
        subset='near_budget_increase')['models']['ridge']['n']
    assert {e['split'] for e in evaluation['budget_effects']} == set(trained)


def criterion(name, metric, subset='all', target='awaiting_first_content', **bound):
    return dict(id=name, horizon_s=1, target=target, metric=metric,
                **({} if metric == 'budget_effect_per_unit' else dict(subset=subset)), **bound)


def acceptance(criteria, status='frozen', training_runs=('a', 'b'), test_runs=('c',), **preconditions):
    return dict(schema_version=1, status=status, model='ridge', training_runs=list(training_runs),
                test_runs=list(test_runs), criteria=criteria,
                preconditions=dict(dict(binding_headroom=5., min_binding_fraction=.25, min_subset_n=10),
                                   **preconditions))


def test_judge_scores_the_designated_test_against_declared_limits():
    runs = synthetic_runs(a={}, b=dict(phase=2.), c=dict(phase=4., headroom=lambda k: 0. if k % 2 else 9.))
    for name in ('a', 'b'):
        for row in runs[name]['rows']:
            row['features']['budget_headroom'] = 9.
    evaluation = evaluate(runs, horizons=[1.], windows=[1.], alpha=1e-6, test_runs=['c'])
    passing = [criterion('skill', 'skill_mae.ridge_vs_persistence', min=.9),
               criterion('tail', 'models.ridge.optimistic_p99', max=1e-3),
               criterion('effect', 'budget_effect_per_unit', greater_than=-1e9)]
    result = judge(acceptance(passing), evaluation, runs)
    assert result['verdict'] == 'pass' and result['scope'] == 'c' and result['binding_fraction'] == .5
    assert [item['outcome'] for item in result['criteria']] == ['pass'] * 3
    assert result['criteria'][0]['n'] == find(evaluation, split='designated_test', target='awaiting_first_content',
                                              subset='all')['models']['ridge']['n']
    failing = passing + [criterion('impossible', 'skill_mae.ridge_vs_mean', min=1.5)]
    assert judge(acceptance(failing), evaluation, runs)['verdict'] == 'fail'
    sparse = passing + [criterion('rare', 'skill_mae.ridge_vs_persistence', subset='near_budget_increase', min=0.)]
    result = judge(acceptance(sparse), evaluation, runs)
    assert result['verdict'] == 'inconclusive' and result['criteria'][-1]['outcome'] == 'insufficient'
    unbound = judge(acceptance(passing, min_binding_fraction=.6), evaluation, runs)
    assert unbound['verdict'] == 'inconclusive' and not unbound['binding_met']
    assert judge(acceptance(failing, min_binding_fraction=.6), evaluation, runs)['verdict'] == 'inconclusive'


def test_load_acceptance_rejects_incomplete_or_ambiguous_protocols(tmp_path):
    good = acceptance([criterion('skill', 'skill_mae.ridge_vs_persistence', min=.1)])
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(dict(acceptance=good, analysis=dict(forecast=dict(alpha=1.)))))
    spec, analysis, sha = load_acceptance(path, [1.])
    assert spec['test_runs'] == ['c'] and analysis == dict(forecast=dict(alpha=1.)) and len(sha) == 64
    text = json.dumps(dict(acceptance=good), indent=1)
    path.write_bytes(text.encode())
    lf = load_acceptance(path, [1.])[2]
    path.write_bytes(text.replace('\n', '\r\n').encode())
    assert load_acceptance(path, [1.])[2] == lf
    bad = [dict(good, status='draft'), dict(good, model='persistence'), dict(good, training_runs=['a', 'c']),
           dict(good, test_runs=[]), dict(good, preconditions=dict(binding_headroom=5.)),
           dict(good, criteria=good['criteria'] * 2),
           dict(good, criteria=[criterion('x', 'skill_mae.mean_vs_persistence', min=0.)]),
           dict(good, criteria=[criterion('x', 'skill_mae.ridge_vs_mean', subset='other', min=0.)]),
           dict(good, criteria=[criterion('x', 'skill_mae.ridge_vs_mean', min=0., max=1.)]),
           dict(good, criteria=[dict(criterion('x', 'skill_mae.ridge_vs_mean', min=0.), horizon_s=5)])]
    for spec in bad:
        path.write_text(json.dumps(dict(acceptance=spec)))
        with pytest.raises(ValueError):
            load_acceptance(path, [1.])
    assert analysis_mismatches(dict(features=dict(horizons=[1.], step=.5), forecast=dict(alpha=1.)),
                               dict(horizons=[1., 2.], windows=[1.]), 2 / 3, 2.) == ['alpha', 'horizons', 'step']
    assert analysis_mismatches(None, dict(horizons=[1.]), .5, 1.) == []


def test_optimistic_errors_follow_the_controller_risk_direction():
    under = metrics(np.array([1., 5.]), np.array([3., 3.]), 'under')
    assert under['optimistic_fraction'] == .5 and under['optimistic_max'] == 2. and under['bias'] == 0.
    over = metrics(np.array([1., 5.]), np.array([3., 3.]), 'over')
    assert over['optimistic_max'] == 2. and over['optimistic_direction'] == 'over'
    neutral = metrics(np.array([1.]), np.array([3.]), None)
    assert 'optimistic_fraction' not in neutral and neutral['mae'] == 2.
    assert metrics(np.array([]), np.array([]), 'under') == dict(n=0)


def test_ridge_imputes_missing_and_constant_columns_without_warnings():
    x = np.array([[np.nan, 1., 0.], [np.nan, 1., 1.], [np.nan, 1., 2.]])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model = Ridge(1e-9).fit(x, np.array([1., 3., 5.]))
        prediction = model.predict(np.array([[7., 1., 3.], [np.nan, 9., -10.]]))
    assert prediction[0] == pytest.approx(7.) and prediction[1] == 0.


def write_table(directory, runs, qualified=True, protocol=PROTOCOL, reconstructed=True):
    directory.mkdir()
    meta = [dict(name=name, qualified_for_comparison=qualified, state_reconstruction=dict(valid=reconstructed),
                 raw_sha256={}, duration_s=100., e2e_slo=None, e2e_slo_source=None, initial_budget=10.,
                 table=dict(budget_applications=[])) for name in runs]
    (directory / 'features_summary.json').write_text(json.dumps(dict(
        schema_version=SCHEMA_VERSION, protocol=protocol, parameters=dict(horizons=[1.], windows=[1., 5.]),
        runs=meta)))
    for name, run in runs.items():
        (directory / f'{name}-features.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in run['rows']))


def test_load_rejects_unqualified_runs_and_unknown_tables(tmp_path):
    runs = synthetic_runs(a={})
    write_table(tmp_path / 'ok', runs)
    assert load(tmp_path / 'ok')[1]['a']['rows'] == runs['a']['rows']
    write_table(tmp_path / 'unqualified', runs, qualified=False)
    with pytest.raises(ValueError, match='unqualified'):
        load(tmp_path / 'unqualified')
    assert load(tmp_path / 'unqualified', allow_unqualified=True)[1]['a']['rows']
    write_table(tmp_path / 'old', runs, protocol='other')
    with pytest.raises(ValueError, match='protocol'):
        load(tmp_path / 'old')
    mixed = synthetic_runs(a={})
    del mixed['a']['rows'][3]['features']['engine_waiting']
    write_table(tmp_path / 'mixed', mixed)
    with pytest.raises(ValueError, match='Feature columns differ'):
        evaluate(load(tmp_path / 'mixed')[1], horizons=[1.], windows=[1.])


def test_cli_reports_development_only_baselines(tmp_path, monkeypatch):
    write_table(tmp_path / 'features', synthetic_runs(a={}, b=dict(phase=3., change_at=60.)))
    output = tmp_path / 'forecast'
    monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(tmp_path / 'features'), '--output-dir', str(output)])
    main()
    report = json.loads((output / 'forecast_report.json').read_text())
    assert report['evaluation_role'] == 'development' and report['model_readiness'] == 'not_established'
    assert report['control_conditioning'] == 'realized_control_conditioned' and report['control_limits']
    assert report['input_qualification'] == dict(allow_unqualified=False, all_qualified=True, unqualified_runs=[])
    assert not report['warnings'] and report['inputs']['a']['qualified_for_comparison']
    assert report['inputs']['a']['state_reconstruction_valid']
    assert report['inputs']['b']['features_sha256'] and report['parameters']['horizons'] == [1.]
    assert any(row['subset'] == 'near_budget_change' and row['models']['ridge']['n']
               for row in report['evaluation']['results'])
    summary = (output / 'forecast_summary.md').read_text()
    assert 'Chronological hold-out (pooled, all)' in summary and 'Held-out run a' in summary
    assert 'realized_control_conditioned' in summary and 'Warning' not in summary
    assert 'a (qualified yes, state reconstructed yes)' in summary
    for argv in (['--horizons', '2'], ['--split-fraction', '1'], ['--alpha', '0']):
        monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(tmp_path / 'features'),
                                         '--output-dir', str(tmp_path / 'bad')] + argv)
        with pytest.raises(SystemExit):
            main()
        assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('qualified, reconstructed', [(False, True), (True, False)])
def test_cli_override_report_keeps_unqualified_verdicts_and_warnings(tmp_path, monkeypatch, qualified,
                                                                     reconstructed):
    features, output = tmp_path / 'features', tmp_path / 'forecast'
    write_table(features, synthetic_runs(a={}, b=dict(phase=3.)), qualified=qualified, reconstructed=reconstructed)
    monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(features), '--output-dir', str(output)])
    with pytest.raises(SystemExit):
        main()
    assert not output.exists()
    monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(features), '--output-dir', str(output),
                                     '--allow-unqualified'])
    main()
    report = json.loads((output / 'forecast_report.json').read_text())
    assert report['input_qualification'] == dict(allow_unqualified=True, all_qualified=False,
                                                 unqualified_runs=['a', 'b'])
    assert len(report['warnings']) == 2 and 'invalid evidence' in report['warnings'][0]
    assert report['inputs']['a']['qualified_for_comparison'] is qualified
    assert report['inputs']['a']['state_reconstruction_valid'] is reconstructed
    summary = (output / 'forecast_summary.md').read_text()
    assert summary.splitlines()[2].startswith('> **Warning: unqualified evidence.**')
    assert '--allow-unqualified set' in summary and ' NO' in summary


def test_cli_scores_the_predeclared_test_once_and_refuses_drift(tmp_path, monkeypatch):
    write_table(tmp_path / 'features', synthetic_runs(a={}, b=dict(phase=2., change_at=60.), c=dict(phase=4.)))
    criteria = [criterion('skill', 'skill_mae.ridge_vs_persistence', min=-1e9),
                criterion('effect', 'budget_effect_per_unit', greater_than=-1e9)]
    protocol = tmp_path / 'protocol.json'
    analysis = dict(features=dict(horizons=[1], windows=[1, 5]), forecast=dict(split_fraction=2 / 3, alpha=1))

    def run(spec, *extra, output='forecast'):
        protocol.write_text(json.dumps(dict(analysis=analysis, acceptance=spec)))
        monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(tmp_path / 'features'), '--output-dir',
                                         str(tmp_path / output), '--acceptance', str(protocol), *extra])
        main()
        return json.loads((tmp_path / output / 'forecast_report.json').read_text())

    report = run(acceptance(criteria, min_binding_fraction=0.))
    assert report['evaluation_role'] == 'designated_test' and report['parameters']['test_runs'] == ['c']
    assert report['model_readiness'] == 'offline_acceptance_passed'
    assert report['acceptance']['judgement']['verdict'] == 'pass' and len(report['acceptance']['protocol_sha256']) == 64
    summary = (tmp_path / 'forecast' / 'forecast_summary.md').read_text()
    assert '# Forecast baselines (designated test)' in summary and 'Acceptance (frozen limits): **pass**' in summary
    assert 'Designated test (c, all)' in summary and 'Held-out run c' not in summary
    proposed = run(acceptance(criteria, status='proposed', min_binding_fraction=0.), output='proposed')
    assert proposed['acceptance']['judgement']['verdict'] == 'pass'
    assert proposed['model_readiness'] == 'not_established'
    unbound = run(acceptance(criteria), output='unbound')
    assert unbound['acceptance']['judgement']['verdict'] == 'inconclusive'
    assert unbound['model_readiness'] == 'not_established'
    for spec, extra in ((acceptance(criteria, training_runs=['a']), ()),
                        (acceptance(criteria), ('--alpha', '2')),
                        (acceptance(criteria), ('--test-runs', 'b')),
                        (acceptance(criteria), ('--allow-unqualified',))):
        with pytest.raises(SystemExit):
            run(spec, *extra, output='bad')
        assert not (tmp_path / 'bad').exists()
    monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(tmp_path / 'features'), '--output-dir',
                                     str(tmp_path / 'bad'), '--test-runs', 'a', 'b', 'c'])
    with pytest.raises(SystemExit):
        main()
    monkeypatch.setattr('sys.argv', ['forecast_baselines.py', str(tmp_path / 'features'), '--output-dir',
                                     str(tmp_path / 'plain'), '--test-runs', 'c'])
    main()
    plain = json.loads((tmp_path / 'plain' / 'forecast_report.json').read_text())
    assert plain['evaluation_role'] == 'designated_test' and plain['acceptance'] is None
    assert plain['model_readiness'] == 'not_established'
