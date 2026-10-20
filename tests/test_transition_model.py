import copy
import hashlib
import json
import math

import numpy as np
import pytest

import transition_baselines as model
import transition_features as features
from identification_features import GatewayReplay
from test_identification_features import START, records
from trace_control import validate_schedule


def test_flow_labels_reconcile_survivors_first_tokens_and_failure():
    windows = features.flow_windows(records(), START)
    state = GatewayReplay(records(), START, 10.).state(.6)
    flows = {key: value.total(.6, 2.7) for key, value in windows.items()}
    assert flows == dict(admitted=1., first=1., exit_awaiting=1., exit_streaming=2., cost_in=2., cost_out=11.)
    assert features.conservation(state, flows) == dict(awaiting_first_content=0., streaming=0., reserved_cost=0.)


def test_flow_labels_use_right_closed_boundaries():
    windows = features.flow_windows(records(), START)
    assert windows['first'].total(.4, 1.) == 1
    assert windows['first'].total(1., 2.) == 0


def test_duplicate_or_unmatched_first_content_is_rejected():
    for damaged in (records() + [dict(event='accepted', request_id='b', monotonic=START + .5, request_cost=1)],
                    [dict(event='first_content', request_id='absent', monotonic=START)]):
        with pytest.raises(ValueError, match='Invalid'):
            features.flow_windows(damaged, START)


def test_plans_use_targets_not_later_observed_applications():
    plan = validate_schedule([dict(at_seconds=0, budget=96), dict(at_seconds=2, budget=32)], 30)
    assert features.planned_caps(plan, 1, 5) == [96, 32, 32, 32, 32]
    assert features.planned_caps(plan, 2, 1) == [32]
    with pytest.raises(ValueError):
        features.planned_caps(plan, True, 1)


def state():
    return dict(awaiting_first_content=2., streaming=3., reserved_cost=10.)


def test_advance_conserves_and_preserves_over_cap_work():
    nxt, correction = model.advance(state(), [1, 2, 0, 1, 4, 3])
    assert nxt == dict(awaiting_first_content=1., streaming=4., reserved_cost=11.)
    assert correction == 0
    # There is deliberately no clamp to a new budget: existing requests must drain.
    assert model.vector(dict(nxt), 4)[5] == -7


def test_advance_projects_impossible_flows_and_zero_population_cost():
    nxt, correction = model.advance(state(), [-1, 100, 100, 100, 50, 50])
    assert all(value >= 0 for value in nxt.values()) and correction > 0
    assert nxt == dict.fromkeys(features.STATES, 0.)
    nxt, _ = model.advance(state(), [0, 0, 0, 0, 5, 8])
    assert nxt == state()  # no admissions/exits means no cost in/out


@pytest.mark.parametrize('bad', [[1, 2], [1, 2, 0, 0, math.nan, 1], [1, 2, 0, 0, True, 1]])
def test_advance_rejects_bad_flow_predictions(bad):
    with pytest.raises(ValueError):
        model.advance(state(), bad)


def synthetic_runs():
    runs = {}
    for name, cap, mode in (('fixed-low', 16., 'fixed'), ('fixed-high', 96., 'fixed'), ('scheduled', 64., 'scheduled')):
        rows = []
        for t in range(90):
            f = {key: 0. for key in model.BASE + model.ENGINE}
            f.update(state(), budget=cap, budget_headroom=cap - 10., in_flight=5.,
                     resolved_ttft_met_5s=5., engine_missing=True, since_budget_value_change_s=None)
            labels = {str(h): dict(state(), cohort_ttft_met=float(h), cohort_resolved_s=t + h + 8.)
                      for h in features.HORIZONS}
            rows.append(dict(t_s=t, features=f, planned_caps=[cap] * 5,
                next_flows=dict(admitted=1., first=1., exit_awaiting=0., exit_streaming=1., cost_in=2., cost_out=2.),
                labels=labels, label_window_complete={str(h): t + h <= 90 for h in features.HORIZONS}))
        runs[name] = dict(meta=dict(name=name, duration_s=90, budget_mode=mode,
            ttft_slo=.5, e2e_slo=None,
            qualified_for_comparison=True, state_reconstruction=dict(valid=True)), rows=rows)
    return runs


def test_training_purges_unresolved_future_rewards_and_crossing_windows():
    rows = synthetic_runs()['fixed-low']['rows']
    assert max(r['t_s'] for r in model.eligible(rows, 1, 60.)) == 59
    assert max(r['t_s'] for r in model.eligible(rows, 1, 60., True)) == 51
    assert all(r['t_s'] + 5 <= 60 for r in model.eligible(rows, 5, 60.))


def test_feature_allowlist_excludes_labels_time_identity_and_versions():
    row = synthetic_runs()['fixed-low']['rows'][0]
    original = model.vector(row['features'], 32)
    row['features'].update(cohort_ttft_met=9999, future_engine_running=9999, budget_version=9999,
                           time=9999, run_id='leak')
    assert model.vector(row['features'], 32) == original
    assert original[model.BASE.index('in_flight')] == 5
    assert original[model.BASE.index('budget')] == 32


def test_rollout_uses_only_origin_features_and_explicit_caps():
    runs = synthetic_runs()
    rows = runs['fixed-low']['rows'] + runs['fixed-high']['rows']
    fitted = model.FlowModel().fit(model.eligible(rows, 1), model.eligible(rows, 1, reward=True))
    f = runs['fixed-low']['rows'][10]['features']
    before = copy.deepcopy(f)
    prediction = fitted.forecast(f, [32] * 5)
    assert f == before
    assert prediction[-1]['cohort_ttft_met'] == pytest.approx(5.)
    assert prediction[-1]['reserved_cost'] == pytest.approx(10.)
    with pytest.raises(ValueError, match='outside fitted range'):
        fitted.forecast(f, [8])
    assert len(fitted.forecast(f, [8], allow_extrapolation=True)) == 1
    with pytest.raises(ValueError):
        fitted.forecast(f, [math.nan])


def test_all_splits_are_disjoint_except_purged_chronological_prefix():
    runs = synthetic_runs()
    for name, cuts, testing in model.splits(runs):
        if name != 'chronological':
            assert not set(cuts).intersection(testing)


def test_evaluate_complete_pipeline_and_explicit_extrapolation():
    report = model.evaluate(synthetic_runs())
    assert report['acceptance'] is None and report['c_used'] is False
    assert report['scores'] and report['chronological_predictions']
    assert any(r['diagnostic_extrapolation_rows'] > 0 for r in report['scores'])
    scores = [r for r in report['scores'] if r['target'] == 'cohort_ttft_met' and r['models']['persistence']['n']]
    assert all(r['models']['flow_ridge']['mae'] < 1e-8 for r in scores)
    json.dumps(report, allow_nan=False)


def test_chronological_fit_and_preprocessing_cannot_see_test_values():
    original = synthetic_runs()
    changed = copy.deepcopy(original)
    for run in changed.values():
        for row in run['rows']:
            if row['t_s'] >= 60:
                row['features']['received_5s'] = 1e9
                row['features']['engine_running'] = 1e9
                row['next_flows']['cost_in'] = 1e9
            for h in features.HORIZONS:
                if row['labels'][str(h)]['cohort_resolved_s'] > 60:
                    row['labels'][str(h)]['cohort_ttft_met'] = 1e9
    assert model.evaluate(original)['fitted']['chronological'] == model.evaluate(changed)['fitted']['chronological']


def test_projection_stays_finite_and_nonnegative_over_random_rollouts():
    rng = np.random.default_rng(25)
    current = state()
    for _ in range(500):
        current, correction = model.advance(current, rng.uniform(-100, 100, 6).tolist())
        assert all(math.isfinite(v) and v >= 0 for v in current.values())
        assert math.isfinite(correction) and correction >= 0
        if current['awaiting_first_content'] + current['streaming'] == 0:
            assert current['reserved_cost'] == 0


def test_recorded_development_settings_match_implementation():
    from pathlib import Path
    spec = json.loads((Path(__file__).resolve().parents[1] / 'experiments/transition-prototype-v1.json').read_text())
    assert spec['protocol'] == features.PROTOCOL
    assert spec['horizons_s'] == list(features.HORIZONS)
    assert spec['models'] == list(model.MODELS)
    assert spec['acceptance'] is None and spec['c_used'] is False


def write_dataset(root):
    root.mkdir()
    entries = []
    for name, run in synthetic_runs().items():
        filename = name + '.jsonl'
        data = ''.join(json.dumps(row) + '\n' for row in run['rows']).encode()
        (root / filename).write_bytes(data)
        entries.append(dict(meta=run['meta'], file=filename, rows=len(run['rows']), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(protocol=features.PROTOCOL, step_s=1., horizons=list(features.HORIZONS), runs=entries)
    (root / 'dataset.json').write_text(json.dumps(manifest))
    return manifest


def test_loader_checks_fingerprint_qualification_and_path(tmp_path):
    root = tmp_path / 'data'
    manifest = write_dataset(root)
    assert len(model.load(root)[1]) == 3
    (root / 'fixed-low.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='fingerprint'):
        model.load(root)
    manifest['runs'][0]['meta']['qualified_for_comparison'] = False
    (root / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unqualified'):
        model.load(root)
    manifest['runs'][0]['file'] = '../escape.jsonl'
    (root / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe'):
        model.load(root)


def test_loader_refuses_mixed_objectives(tmp_path):
    root = tmp_path / 'data'
    manifest = write_dataset(root)
    manifest['runs'][0]['meta']['ttft_slo'] = 1.
    (root / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='TTFT-only objective'):
        model.load(root)


def test_cli_preserves_existing_outputs(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    write_dataset(data)
    output = tmp_path / 'out'
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_text('original')
    monkeypatch.setattr('sys.argv', ['transition_baselines.py', str(data), '--output-dir', str(output)])
    with pytest.raises(SystemExit) as error:
        model.main()
    assert error.value.code == 2 and sentinel.read_text() == 'original'


def reproducer():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'results/transition-prototype-20261020/reproduce.py'
    spec = importlib.util.spec_from_file_location('transition_reproducer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproducer_numeric_tolerance_and_structural_checks():
    module = reproducer()
    module.compare({'metric': [1., None]}, {'metric': [1. + 1e-10, None]})
    for bad in ({'metric': [2., None]}, {'metric': [1., None, 3]}, {'wrong': [1., None]}):
        with pytest.raises(ValueError):
            module.compare({'metric': [1., None]}, bad)


def test_reproducer_cli_wiring_and_existing_destination(tmp_path, monkeypatch):
    import gzip
    module = reproducer()
    archive = tmp_path / 'archive'
    archive.mkdir()
    expected = dict(analysis_code_sha256={'model': 'abc'}, source_runs=[{'raw': 'hash'}], scores=[{'mae': 1.}])
    raw = json.dumps(expected).encode()
    compressed = gzip.compress(raw, mtime=0)
    (archive / 'full-report.json.gz').write_bytes(compressed)
    (archive / 'summary.json').write_text(json.dumps(dict(full_report_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_report_sha256=hashlib.sha256(compressed).hexdigest())))
    monkeypatch.setattr(module, 'HERE', archive)
    output = tmp_path / 'out'
    monkeypatch.setattr('sys.argv', ['reproduce.py', '--cap-root', str(tmp_path/'caps'),
        '--shared-root', str(tmp_path/'shared'), '--output-dir', str(output)])
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        destination = __import__('pathlib').Path(command[command.index('--output-dir') + 1])
        destination.mkdir()
        if 'transition_baselines.py' in command[1]:
            (destination / 'transition_report.json').write_bytes(raw)
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    module.main()
    assert len(calls) == 3
    assert len([arg for arg in calls[0] if 'fixed-' in arg]) == 6
    assert json.loads((output / 'verification.json').read_text())['scores_match']
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
