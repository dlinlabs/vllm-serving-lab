import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import service_features as features
import service_forecasts as forecasts
from test_identification_features import START, records
from test_transition_model import synthetic_runs
from transition_baselines import eligible


def point(sent, amount, **kwargs):
    def metric(value):
        return dict(available=True, type='counter', samples=[dict(name='vllm:sample_total', labels={'engine': '0'}, value=value)])
    return dict(sent_at=START + sent, received_at=START + sent + .05, finished_at=START + sent + .1,
        http_status=200, error_type=None, counter_epoch=0, exporter_identity={'started': 99},
        metrics={key: metric(amount) for key in features.COUNTERS}, **kwargs)


def rates(points, t):
    series = features.counter_series(points, START)
    return features.rate_features(series, [p['available_s'] for p in series], t)


def test_counter_samples_must_be_fully_available_and_then_become_stale():
    points = [point(0., 100.), point(1., 300.)]
    assert rates(points, 1.06)['prompt_tokens_missing']  # body received, resource sampling not finished
    result = rates(points, 1.2)
    assert result['prompt_tokens_rate'] == pytest.approx(200.)
    assert result['generation_tokens_rate'] == pytest.approx(200.)
    assert not result['prompt_tokens_missing']
    assert rates(points, 4.2)['prompt_tokens_missing']


@pytest.mark.parametrize('change', ['epoch', 'exporter', 'labels', 'reset_marker', 'gap'])
def test_counter_continuity_requires_matching_series_and_epoch(change):
    points = [point(0., 100.), point(1., 300.)]
    if change == 'epoch':
        points[1]['counter_epoch'] = 1
    elif change == 'exporter':
        points[1]['exporter_identity'] = {'started': 100}
    elif change == 'labels':
        for metric in points[1]['metrics'].values():
            metric['samples'][0]['labels']['engine'] = '1'
    elif change == 'reset_marker':
        points[1]['counter_resets'] = ['other counter']
    else:
        points[1] = point(4., 300.)
    assert rates(points, 4.2 if change == 'gap' else 1.2)['prompt_tokens_missing']


def test_failed_scrape_is_not_bridged_or_replaced_by_a_future_sample():
    points = [point(0., 100.), point(1., 200.), point(2., 300.), point(3., 400.)]
    points[1].update(error_type='ReadTimeout', http_status=None)
    assert rates(points, 1.2)['prompt_tokens_missing']
    assert rates(points, 2.2)['prompt_tokens_missing']
    assert rates(points, 3.2)['prompt_tokens_rate'] == pytest.approx(100.)


def test_reset_of_one_engine_cannot_hide_behind_increase_of_another():
    points = [point(0., 100.), point(1., 200.)]
    points[1]['metrics']['prompt_tokens']['samples'][0]['value'] = 90.
    for i, extra in enumerate((100., 200.)):
        points[i]['metrics']['prompt_tokens']['samples'].append(
            dict(name='vllm:sample_total', labels={'engine': '1'}, value=extra))
    result = rates(points, 1.2)
    assert result['prompt_tokens_missing'] and not result['generation_tokens_missing']


@pytest.mark.parametrize('bad', [None, True, math.inf, -1.])
def test_invalid_counter_values_are_missing(bad):
    p = point(0., 100.)
    p['metrics']['prompt_tokens']['samples'][0]['value'] = bad
    assert features.counter_values(p, 'prompt_tokens') is None


def test_created_metrics_and_duplicate_series_are_not_counter_deltas():
    p = point(0., 100.)
    p['metrics']['prompt_tokens']['samples'][0]['name'] = 'vllm:sample_created'
    assert features.counter_values(p, 'prompt_tokens') is None
    p = point(0., 100.)
    p['metrics']['prompt_tokens']['samples'] *= 2
    assert features.counter_values(p, 'prompt_tokens') is None


def context_records():
    rows = records()
    for row in rows:
        if row['event'] == 'rejected':
            row['estimated_input_tokens'] = 30
    return rows


def test_context_reconstructs_cost_tokens_ages_and_failed_release():
    a, b = features.contexts(context_records(), START, [.6, 2.7])
    assert a['awaiting_cost'] == 5 and a['streaming_cost'] == 4
    assert a['awaiting_input_tokens'] == 50 and a['streaming_input_tokens'] == 40
    assert a['awaiting_age_sum_s'] == pytest.approx(.25)
    assert a['streaming_age_sum_s'] == pytest.approx(.2)
    assert a['offered_input_tokens_5s'] == 120
    assert b['awaiting_cost'] == b['streaming_cost'] == 0


def test_context_is_invariant_to_future_request_changes():
    original = context_records()
    changed = copy.deepcopy(original)
    for row in changed:
        if row['event'] == 'accepted' and row['monotonic'] > START + .6:
            row['request_cost'] = 1e6
            row['estimated_input_tokens'] = 1_000_000
    assert features.contexts(original, START, [.6]) == features.contexts(changed, START, [.6])
    with pytest.raises(ValueError, match='nondecreasing'):
        features.contexts(original, START, [1., 0.])


def service_runs():
    runs = synthetic_runs()
    for run in runs.values():
        run['meta']['service_context'] = dict(verified_rows=len(run['rows']))
        for row in run['rows']:
            row['features'].update({key: 0. for key in features.CONTEXT})
            row['features'].update(awaiting_cost=4., streaming_cost=6., oldest_awaiting_age_s=.2)
            for key in features.COUNTERS:
                row['features'][key + '_rate'] = 100.
                row['features'][key + '_missing'] = False
    return runs


def test_no_plan_ablation_is_invariant_to_all_candidate_caps():
    f = service_runs()['fixed-low']['rows'][0]['features']
    original = forecasts.design(f, [16.] * 5, 'service_no_plan')
    assert original == forecasts.design(f, [96.] * 5, 'service_no_plan')
    assert forecasts.design(f, [16.] * 5, 'service_quadratic') != forecasts.design(f, [96.] * 5, 'service_quadratic')
    f.update(future_tokens=1e9, cohort_ttft_met=1e9, budget_version=999)
    assert original == forecasts.design(f, [16.] * 5, 'service_no_plan')


def test_predictor_requires_exact_plan_length_and_explicit_extrapolation():
    rows = service_runs()['fixed-low']['rows']
    model = forecasts.DirectModel(5, 'counter_ridge').fit(eligible(rows, 5, reward=True))
    f = rows[20]['features']
    assert model.predict(f, [16.] * 5)['cohort_ttft_met'] == pytest.approx(5.)
    with pytest.raises(ValueError, match='length'):
        model.predict(f, [16.])
    with pytest.raises(ValueError, match='training range'):
        model.predict(f, [32.] * 5)
    assert model.predict(f, [32.] * 5, allow_extrapolation=True)['cohort_ttft_met'] >= 0


def test_support_distinguishes_a_hold_from_an_unseen_switch():
    runs = service_runs()
    rows = runs['fixed-low']['rows'] + runs['fixed-high']['rows']
    support = forecasts.SupportIndex(rows, 5)
    f = runs['fixed-low']['rows'][0]['features']
    assert support.at(f, 16.)['same_applied_cap_rows'] == 90
    assert support.at(f, 96.)['training_hold_rows'] == 90
    assert support.at(f, 96.)['same_applied_cap_rows'] == 0
    assert support.at(f, 32.)['nearest_normalized_state_distance'] is None


def test_chronological_fit_does_not_learn_future_values_and_ranks_keep_ties():
    original = service_runs()
    changed = copy.deepcopy(original)
    for run in changed.values():
        for row in run['rows']:
            if row['t_s'] >= 60:
                row['features']['prompt_tokens_rate'] = 1e8
                row['features']['awaiting_input_tokens'] = 1e8
            for h in (1, 2, 5):
                if row['labels'][str(h)]['cohort_resolved_s'] > 60:
                    row['labels'][str(h)]['cohort_ttft_met'] = 1e8
    a, b = forecasts.evaluate(original), forecasts.evaluate(changed)
    assert a['fitted']['chronological'] == b['fitted']['chronological']
    assert all(len(r['best_candidates']['service_no_plan']) == 4 for r in a['cap_responses'])
    json.dumps(a, allow_nan=False)


def write_dataset(path):
    path.mkdir()
    entries = []
    for name, run in service_runs().items():
        filename = name + '.jsonl'
        data = ''.join(json.dumps(row) + '\n' for row in run['rows']).encode()
        (path / filename).write_bytes(data)
        entries.append(dict(meta=run['meta'], file=filename, rows=len(run['rows']), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(protocol='service-transition-development-v1', service_protocol=features.SERVICE_PROTOCOL,
                    step_s=1., horizons=[1, 2, 5], runs=entries)
    (path / 'dataset.json').write_text(json.dumps(manifest))
    return manifest


def test_dataset_requires_verified_context_and_consistent_counter_flags(tmp_path):
    root = tmp_path / 'data'
    manifest = write_dataset(root)
    assert len(forecasts.load_service(root)[1]) == 3
    filename = manifest['runs'][0]['file']
    rows = [json.loads(s) for s in (root / filename).read_text().splitlines()]
    rows[0]['features']['prompt_tokens_missing'] = True
    data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    (root / filename).write_bytes(data)
    manifest['runs'][0]['sha256'] = hashlib.sha256(data).hexdigest()
    (root / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='missing flag'):
        forecasts.load_service(root)


def test_settings_match_model_inventory():
    path = Path(__file__).resolve().parents[1] / 'experiments/service-forecast-v2.json'
    spec = json.loads(path.read_text())
    assert spec['models'] == list(forecasts.MODELS)
    assert spec['cap_response_candidates'] == list(forecasts.CANDIDATE_CAPS)
    assert spec['counter_max_age_and_gap_s'] == features.MAX_COUNTER_AGE_S
    assert spec['acceptance'] is None and spec['c_used'] is False


def test_gauge_counter_and_plan_ablations_have_exact_column_differences():
    assert set(forecasts.columns('counter_ridge'))-set(forecasts.columns('gauge_ridge')) == set(features.RATE_FIELDS)
    assert set(forecasts.columns('counter_ridge'))-set(forecasts.columns('service_no_plan')) == set(forecasts.PLAN_COLUMNS)
    assert set(forecasts.columns('service_quadratic'))-set(forecasts.columns('counter_ridge')) == set(forecasts.INTERACTIONS)


def helper():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'results/service-forecast-20261020/reproduce.py'
    spec=importlib.util.spec_from_file_location('service_reproducer',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cached_pack(tmp_path,monkeypatch):
    import gzip
    source=tmp_path/'source'
    manifest=write_dataset(source)
    wrapper=dict(format='audited-service-tables-v2',manifest_text=(source/'dataset.json').read_text(),
                 tables={e['file']:(source/e['file']).read_text() for e in manifest['runs']})
    archive=tmp_path/'archive'
    archive.mkdir()
    data=gzip.compress(json.dumps(wrapper).encode(),mtime=0)
    (archive/'dataset.json.gz').write_bytes(data)
    (archive/'summary.json').write_text(json.dumps(dict(dataset_pack_sha256=hashlib.sha256(data).hexdigest(),
        dataset_sha256=hashlib.sha256(wrapper['manifest_text'].encode()).hexdigest())))
    module=helper()
    monkeypatch.setattr(module,'HERE',archive)
    return module,archive,wrapper


def test_cached_table_roundtrip_and_existing_destination(tmp_path,monkeypatch):
    module,archive,_=cached_pack(tmp_path,monkeypatch)
    destination=tmp_path/'restored'
    module.restore(destination)
    assert len(forecasts.load_service(destination)[1])==3
    with pytest.raises(ValueError,match='fresh'):
        module.restore(destination)
    data=bytearray((archive/'dataset.json.gz').read_bytes());data[-5]^=1
    (archive/'dataset.json.gz').write_bytes(data)
    with pytest.raises(ValueError,match='archive fingerprint'):
        module.restore(tmp_path/'corrupt')
    assert not (tmp_path/'corrupt').exists()


def test_cached_table_member_fingerprint_checked_before_writes(tmp_path,monkeypatch):
    import gzip
    module,archive,wrapper=cached_pack(tmp_path,monkeypatch)
    key=next(iter(wrapper['tables']))
    wrapper['tables'][key]='changed table'
    data=gzip.compress(json.dumps(wrapper).encode(),mtime=0)
    (archive/'dataset.json.gz').write_bytes(data)
    summary=json.loads((archive/'summary.json').read_text());summary['dataset_pack_sha256']=hashlib.sha256(data).hexdigest()
    (archive/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='table fingerprint'):
        module.restore(tmp_path/'bad-table')
    assert not (tmp_path/'bad-table').exists()
