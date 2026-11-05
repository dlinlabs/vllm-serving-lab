"""Focused literal-event CPU tests for nominal_metrics (no native data, no network)."""
import copy
import hashlib
import json
import math
import re

import pytest

import nominal_metrics as nm

COMPARABILITY = {
    'hardware': {'gpu': 'literal GPU model', 'gpu_count': 1},
    'model': {'local_model_path': '/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507', 'asset_sha256': '1' * 64,
              'served_alias': 'Qwen/Qwen3-4B-Instruct-2507', 'served_alias_verified': True,
              'tokenizer_sha256': '2' * 64},
    'backend': {'vllm': 'literal', 'scheduling_policy': 'fcfs'}, 'gateway': {'source_sha256': '3' * 64},
    'resources': {'cpu': 'literal'}, 'code': {'client_sha256': '4' * 64},
    'client_generation': {'ignore_eos': False, 'max_tokens_policy': 'trace max_tokens'}, 'clock': 'literal monotonic',
}


def canonical_sha(value):
    """Independent canonical JSON digest (sorted keys, compact separators)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def slo(status='complete', finish='stop', ignore=False, done=True, error=False):
    return {'status': status, 'finish_reason': finish, 'requested_ignore_eos': ignore, 'stream_done': done,
            'stream_error': error}


def row(rid, fate, sched, sent=None, first=None, term=None, tokens=100, ct=None, slo_fields=None, cap=128):
    return {'request_id': rid, 'group_tokens': tokens, 'max_output_tokens': cap, 'scheduled_at': sched,
            'sent_at': sent, 'first_content_at': first, 'terminal_at': term, 'fate': fate,
            'completion_tokens': ct, 'slo': slo_fields}


def run(rows, arm='strong_one_move', purpose='train', missing=('finish_reason: literal test',), control=None,
        control_missing='no controller journal in literal test', start=10.0, duration=5.0, envelope=None,
        identity='a' * 64, scope='offered_cohort_observed', comparability=None, selection=None):
    if envelope is None:
        envelope = (start, max([start + duration] + [r['terminal_at'] for r in rows if r['terminal_at'] is not None]))
    provenance = {'offered_trace_sha256': identity, 'offered_count': len(rows),
                  'offered_request_ids_sha256': canonical_sha([r['request_id'] for r in rows]),
                  'comparability': copy.deepcopy(comparability), 'train_selection_sha256': selection}
    return {'arm': arm, 'purpose': purpose, 'clock_domain': 'literal', 'offered_start': start,
            'offered_duration_s': duration, 'envelope_start': envelope[0], 'envelope_end': envelope[1],
            'envelope_scope': scope, 'rows': rows, 'slo_missing': None if missing is None else list(missing),
            'control': control, 'control_missing': control_missing, 'provenance': provenance}


def static(budget):
    return {'kind': 'static_fixed', 'budget': budget, 'basis': 'literal static install before offered start'}


def keys_recursive(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_recursive(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_recursive(item)


def test_nearest_rank_counts_and_sample_max_flags():
    block = nm.percentile_block([float(v) for v in range(20, 0, -1)])
    assert (block['n'], block['p50'], block['p95'], block['p99']) == (20, 10.0, 19.0, 20.0)
    assert (block['p95_rank'], block['p95_is_sample_max'], block['p99_is_sample_max']) == (19, False, True)
    small = nm.percentile_block([float(v) for v in range(1, 20)])
    assert (small['p50'], small['p95'], small['p95_is_sample_max']) == (10.0, 19.0, True)
    empty = nm.percentile_block([])
    assert (empty['n'], empty['p50'], empty['null_reason']) == (0, None, 'no_samples')


def test_all_offered_fates_ttft_origins_penalty_and_usage_bounds():
    rows = [row('a', 'complete', 10.0, 10.01, 10.21, 11.0, ct=20),
            row('b', 'complete', 10.5, 10.5, 10.9, 11.5, ct=None),
            row('c', 'rejected', 11.0, 11.0, None, 11.01),
            row('d', 'not_launched', 12.0),
            row('e', 'cancelled', 13.0, 13.0, None, 20.0),
            row('f', 'unknown', 14.0),
            row('g', 'stream_error', 14.5, 14.5, 14.6, 15.0, ct=3)]
    metrics = nm.compute(run(rows))
    outcomes = metrics['outcomes']
    assert metrics['all_offered'] == 7
    assert outcomes['fates'] == {'complete': 2, 'rejected': 1, 'http_error': 0, 'stream_error': 1,
                                 'transport_error': 0, 'cancelled': 1, 'not_launched': 1, 'unknown': 1}
    assert outcomes['rejection_rate'] == pytest.approx(1 / 7)  # not_launched is never merged into rejection
    assert outcomes['failure_rate'] == pytest.approx(2 / 7)
    assert outcomes['not_launched_rate'] == pytest.approx(1 / 7) and outcomes['unknown_rate'] == pytest.approx(1 / 7)
    ttft = metrics['ttft_s']
    assert ttft['conditional_scheduled']['n'] == 2
    assert ttft['conditional_scheduled']['p50'] == pytest.approx(0.21)
    assert ttft['conditional_client_send']['p50'] == pytest.approx(0.20)
    assert ttft['conditional_scheduled']['p95'] == pytest.approx(0.4)
    assert ttft['conditional_scheduled']['p95_is_sample_max'] is True
    penalized = ttft['all_offered_penalized_scheduled']
    assert (penalized['n'], penalized['n_failure_penalty'], penalized['p50_rank']) == (7, 4, 4)
    assert penalized['p50'] is None and penalized['p50_null_reason'] == 'rank_falls_on_failure_penalty'
    assert 'NOT a survival-analysis' in penalized['estimand']
    assert metrics['send_lag_s']['n'] == 5 and metrics['send_lag_s']['n_not_sent'] == 2
    throughput = metrics['throughput']
    assert throughput['delivered'] == 2 and throughput['cohort_normalized_delivered_rps'] == pytest.approx(0.4)
    assert throughput['wall_envelope_delivered_rps'] == pytest.approx(0.2)
    assert (throughput['completion_tokens_observed'], throughput['completion_tokens_upper_bound']) == (20, 148)
    assert throughput['cohort_normalized_completion_tokens_per_s'] == {'lower': 4.0, 'upper': 29.6, 'exact': False}
    assert throughput['wall_envelope_completion_tokens_per_s'] == {'lower': 2.0, 'upper': 14.8, 'exact': False}
    assert throughput['undelivered_reported_completion_tokens'] == 3
    natural = metrics['natural_slo']
    assert natural['available'] is False and natural['natural_slo_good'] is None
    assert natural['null_reason'] == 'natural_slo_fields_unavailable' and natural['missing']
    assert metrics['controller_overhead']['available'] is False
    assert metrics['controller_overhead']['null_reason'] == 'no controller journal in literal test'
    windows = metrics['windows']
    assert windows['envelope_scope'] == 'offered_cohort_observed' and windows['full_wall_s'] is None
    assert 'full wall not recorded' in windows['full_wall_null_reason']
    json.dumps(metrics, allow_nan=False)


def test_unchanged_natural_slo_length_censoring_groups_and_jain():
    rows = [row('g1', 'complete', 0.0, 0.0, 0.3, 0.66, tokens=512, ct=10, slo_fields=slo()),
            row('g2', 'complete', 1.0, 1.0, 1.3, 1.66, tokens=513, ct=10, slo_fields=slo(finish='length')),
            row('g3', 'complete', 2.0, 2.0, 2.6, 2.96, tokens=2048, ct=10, slo_fields=slo()),
            row('g4', 'rejected', 3.0, 3.0, None, 3.01, tokens=2049,
                slo_fields=slo(status='rejected', finish=None, done=False)),
            row('g5', 'complete', 4.0, 4.0, 4.2, 4.45, tokens=100, ct=1, slo_fields=slo()),
            row('g6', 'complete', 4.9, 4.9, 5.2, 5.56, tokens=3000, ct=10, slo_fields=slo())]
    metrics = nm.compute(run(rows, missing=None, start=0.0, duration=5.0, envelope=(0.0, 8.0), scope='full_wall'))
    natural = metrics['natural_slo']
    assert natural['score_http_sha256'] == nm.SCORE_HTTP_SHA256
    assert natural['rule']['schema'] == 'r23-natural-endpoint-slo-v1'
    assert (natural['all_offered'], natural['natural_slo_good']) == (6, 3)
    assert natural['slo_attainment_fraction'] == pytest.approx(0.5)
    assert natural['cohort_slo_goodput_rps'] == pytest.approx(0.6)
    assert natural['in_window_slo_goodput_rps'] == pytest.approx(0.4)
    assert natural['wall_envelope_slo_goodput_rps'] == pytest.approx(0.375)
    assert natural['censored_cap_completions'] == 1 and natural['finish_reasons']['length'] == 1
    assert natural['one_token_complete'] == 1
    groups = metrics['groups']
    assert [groups[name]['n_offered'] for name in nm.GROUPS] == [2, 2, 2]
    assert [groups[name]['slo_attainment_fraction'] for name in nm.GROUPS] == [1.0, 0.0, 0.5]
    spread = metrics['fairness']['slo_attainment_fraction']
    assert spread['jain'] == pytest.approx(0.6)
    assert (spread['worst_min'], spread['best_max'], spread['gap'], spread['min_over_max']) == (0.0, 1.0, 1.0, 0.0)
    assert metrics['fairness']['worst_group_rejection_rate'] == 0.5
    assert metrics['fairness']['worst_group_ttft_conditional_scheduled_p95_group'] == '513to2048'


def test_group_edges_zero_denominators_and_all_zero_jain():
    assert [nm.group_of(n) for n in (1, 512, 513, 2048, 2049)] == ['le512', 'le512', '513to2048', '513to2048',
                                                                 'gt2048']
    rejected = [row('r1', 'rejected', 10.0, 10.0, None, 10.1, tokens=100),
                row('r2', 'rejected', 11.0, 11.0, None, 11.1, tokens=3000)]
    metrics = nm.compute(run(rejected))
    spread = metrics['fairness']['delivered_fraction']
    assert (spread['k'], spread['jain'], spread['min_over_max']) == (2, None, None)
    assert spread['null_reason'].startswith('all_zero') and spread['worst_min'] == 0.0
    assert metrics['fairness']['slo_attainment_fraction']['null_reason'] == 'metric_unavailable'
    assert metrics['groups']['513to2048']['rejection_rate'] is None  # empty group: null, never 0 or 1
    assert metrics['throughput']['cohort_normalized_delivered_rps'] == 0.0
    assert metrics['throughput']['cohort_normalized_completion_tokens_per_s'] == {'lower': 0.0, 'upper': 0.0,
                                                                                  'exact': True}
    single = nm.compute(run(rejected[:1]))['fairness']['delivered_fraction']
    assert single['null_reason'] == 'fewer_than_two_nonempty_groups' and single['jain'] is None
    empty = nm.compute(run([]))
    assert empty['outcomes']['rejection_rate'] is None and empty['outcomes']['delivered_fraction'] is None
    assert empty['ttft_s']['all_offered_penalized_scheduled']['null_reason'] == 'no_samples'


@pytest.mark.parametrize('mutate, message', [
    (lambda r: r['rows'][0].update(first_content_at=10.005), 'clock order'),
    (lambda r: r['rows'][0].update(sent_at=9.0, first_content_at=9.5), 'clock order'),
    (lambda r: r['rows'][0].update(terminal_at=10.1), 'clock order'),
    (lambda r: r['rows'][0].update(first_content_at=None), 'complete requires sent_at, first_content_at'),
    (lambda r: r['rows'][1].update(sent_at=12.0, terminal_at=12.5), 'not_launched carries'),
    (lambda r: r['rows'][0].update(scheduled_at=True), 'finite int/float'),
    (lambda r: r['rows'][0].update(scheduled_at=math.nan), 'finite int/float'),
    (lambda r: r['rows'][1].update(request_id='a'), 'unique'),
    (lambda r: r['rows'][1].update(fate='length'), 'fate must be one of'),
    (lambda r: r.update(envelope_end=14.0), 'envelope must cover'),
    (lambda r: r.update(envelope_scope='offered_window_only'), 'envelope_scope must be one of'),
    (lambda r: r.update(arm='fixed16'), 'fixed arms require'),
    (lambda r: r.update(arm='train_selected_fixed'), 'table alias'),
    (lambda r: r['rows'][0].update(slo=slo()), 'every row or on none'),
    (lambda r: r.update(control=static(16.0), control_missing=None), 'static_fixed control must belong'),
])
def test_strict_input_refusals(mutate, message):
    literal = run([row('a', 'complete', 10.0, 10.01, 10.21, 11.0, ct=20), row('b', 'not_launched', 12.0)])
    nm.compute(literal)
    mutate(literal)
    with pytest.raises(nm.MetricsInputError, match=message):
        nm.compute(literal)


def test_slo_status_must_coincide_with_fate():
    literal = run([row('a', 'rejected', 10.0, 10.0, None, 10.1, slo_fields=slo())], missing=None)
    with pytest.raises(nm.MetricsInputError, match='coincide'):
        nm.compute(literal)


def test_early_send_is_refused_and_lag_extremes_reported():
    literal = run([row('a', 'complete', 10.0, 10.0, 10.2, 10.5), row('b', 'rejected', 11.0, 11.25, None, 11.3)])
    lag = nm.compute(literal)['send_lag_s']
    assert (lag['min'], lag['max'], lag['n_negative'], lag['early_send_tolerance_s']) == (0.0, 0.25, 0, 0.0)
    literal['rows'][1]['sent_at'] = 10.999999
    with pytest.raises(nm.MetricsInputError, match='early send'):
        nm.compute(literal)


def test_all_offered_cohort_is_bound_to_offered_count_and_ordered_ids():
    literal = run([row('a', 'complete', 10.0, 10.0, 10.2, 10.5), row('b', 'not_launched', 11.0),
                   row('c', 'unknown', 12.0)])
    cohort = nm.compute(literal)['offered_cohort']
    assert (cohort['offered_count'], cohort['request_ids'], cohort['request_ids_sha256']) == (
        3, ['a', 'b', 'c'], canonical_sha(['a', 'b', 'c']))
    assert nm.request_ids_sha256(iter(['a', 'b', 'c'])) == cohort['request_ids_sha256']
    for mutate, message in (
            (lambda r: r['rows'].pop(), 'all-offered cohort: 2 rows'),
            (lambda r: r['rows'].reverse(), 'request ids'),
            (lambda r: r['rows'][2].update(request_id='z'), 'request ids'),
            (lambda r: r['provenance'].pop('offered_count'), 'offered_count must be an int'),
            (lambda r: r['provenance'].update(offered_count=3.0), 'offered_count must be an int'),
            (lambda r: r['provenance'].update(offered_request_ids_sha256='A' * 64), '64-hex')):
        mutated = copy.deepcopy(literal)
        mutate(mutated)
        with pytest.raises(nm.MetricsInputError, match=message):
            nm.compute(mutated)


def test_offered_window_and_declared_envelope_bind_every_observation():
    rows = [row('a', 'complete', 10.0, 10.0, 10.2, 10.5), row('b', 'cancelled', 14.0, 14.0, None, 18.0)]
    assert nm.compute(run(rows))['windows']['envelope_end'] == 18.0
    for kwargs, message in (
            ({'envelope': (10.0, 17.0)}, 'terminal_at lies outside the declared envelope'),
            ({'envelope': (10.0, 19.0)}, 'offered_cohort_observed envelope must be'),
            ({'envelope': (9.0, 18.0)}, 'offered_cohort_observed envelope must be'),
            ({'envelope': (10.0, 17.0), 'scope': 'full_wall'}, 'outside the declared envelope')):
        with pytest.raises(nm.MetricsInputError, match=message):
            nm.compute(run(copy.deepcopy(rows), **kwargs))
    late = copy.deepcopy(rows)
    late[1].update(scheduled_at=15.5, sent_at=15.5)
    with pytest.raises(nm.MetricsInputError, match='offered window'):
        nm.compute(run(late))
    full = nm.compute(run(copy.deepcopy(rows), envelope=(9.0, 19.0), scope='full_wall'))
    assert (full['windows']['full_wall_s'], full['throughput']['wall_rate_direction']) == (10.0, 'full_wall_rate')


def test_group_identity_ties_and_rate_scope_labels():
    rows = [row('a', 'rejected', 10.0, 10.0, None, 10.1, tokens=100),
            row('b', 'complete', 10.5, 10.5, 10.7, 11.0, tokens=100, ct=5),
            row('c', 'rejected', 11.0, 11.0, None, 11.1, tokens=3000),
            row('d', 'complete', 11.5, 11.5, 11.7, 12.0, tokens=3000, ct=5),
            row('e', 'complete', 12.0, 12.0, 12.2, 12.5, tokens=1000, ct=5)]
    metrics = nm.compute(run(rows))
    fairness, summary = metrics['fairness'], nm.summarize(metrics)
    assert (fairness['worst_group_rejection_rate'], fairness['worst_group_rejection_rate_group'],
            fairness['worst_group_rejection_rate_n'], fairness['worst_group_rejection_rate_tied_groups']) == (
        0.5, 'le512', 2, ['le512', 'gt2048'])
    spread = fairness['delivered_fraction']
    assert (spread['k'], spread['n_offered_by_group']) == (3, {'le512': 2, '513to2048': 1, 'gt2048': 2})
    assert 'per-group n' in spread['note']
    assert (summary['group_k'], summary['worst_group_rejection_rate_group'], summary['wall_rate_direction']) == (
        3, 'le512', 'upper_bound_on_full_wall_rate')
    assert 'delivered rows only' in summary['token_rate_scope']
    assert 'UPPER bounds' in metrics['throughput']['wall_envelope_label']


def test_controller_overhead_union_static_and_null():
    control = {'kind': 'measured_intervals', 'source': 'literal journal',
               'intervals': [[11.0, 11.5], [11.25, 12.0], [19.5, 21.0], [8.0, 9.0]]}
    overhead = nm.compute(run([], control=control, control_missing=None, envelope=(10.0, 20.0),
                              scope='full_wall'))['controller_overhead']
    assert overhead['decisions'] == 4 and overhead['union_total_s'] == pytest.approx(3.5)
    assert overhead['union_within_envelope_s'] == pytest.approx(1.5)
    assert overhead['union_fraction_of_envelope'] == pytest.approx(0.15)
    assert overhead['durations_s']['n'] == 4 and overhead['durations_s']['max'] == pytest.approx(1.5)
    fixed = nm.compute(run([], arm='fixed32', control=static(32.0), control_missing=None))['controller_overhead']
    assert (fixed['decisions'], fixed['union_total_s'], fixed['kind']) == (0, 0.0, 'static_fixed')
    with pytest.raises(nm.MetricsInputError, match='no controller decision intervals'):
        nm.compute(run([], arm='fixed16', control=control, control_missing=None))
    full = nm.compute(run([], control=control, control_missing=None, envelope=(8.0, 22.0), scope='full_wall'))
    assert (full['windows']['full_wall_s'], full['windows']['full_wall_null_reason']) == (14.0, None)
    assert full['controller_overhead']['union_within_envelope_s'] == pytest.approx(3.5)


def write_raw(directory, mutate=None):
    model = '/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507'
    trace = {'schema_version': 1, 'max_model_len': 8192, 'model': model,
             'duration_seconds': 5.0, 'output_cap': 128,
             'requests': [{'trace_id': f'p0w1-train-000{i}', 'arrival_seconds': arrival, 'input_tokens': tokens,
                           'max_tokens': 128, 'prompt': 'literal'}
                          for i, (arrival, tokens) in enumerate([(0.1, 300), (0.2, 1100), (0.3, 3100), (0.4, 600),
                                                                 (4.9, 700)])],
             'provenance': {'kind': 'SYNTHETIC_OFFERED_WORKLOAD_TAPE', 'offered_tape_schema': nm.OFFERED_TAPE_SCHEMA,
                            'offered_tape_sha256': 'b' * 64, 'purpose': 'train', 'source_id': 'literal:train',
                            'protocol_id': 'literal', 'requested_prompt_tokens': [256, 1024, 3072, 256, 1024]}}
    start = 1000.0

    def launched(i, **fields):
        request = trace['requests'][i]
        base = {'scheduled_at': start + request['arrival_seconds'], 'workload': 'trace', 'request_id': f'gw-{i}',
                'client_request_id': f'c{i}', 'stream_complete': False, 'started_at': None, 'first_content_at': None,
                'finished_at': None, 'status': 200, 'ttft': None, 'end_to_end': None, 'input_tokens': None,
                'output_tokens': None, 'total_tokens': None, 'max_output_tokens': 128, 'error_type': None,
                'error_message': None, 'error_phase': None, 'success': False, 'rejected': False,
                'trace_id': request['trace_id'], 'expected_input_tokens': request['input_tokens']}
        base.update(fields)
        return base

    def placeholder(i, message):
        request = trace['requests'][i]
        return {'trace_id': request['trace_id'], 'client_request_id': f'c{i}', 'request_id': None,
                'scheduled_at': start + request['arrival_seconds'], 'started_at': None, 'first_content_at': None,
                'finished_at': None, 'status': None, 'ttft': None, 'end_to_end': None, 'stream_complete': False,
                'input_tokens': None, 'output_tokens': None, 'max_output_tokens': 128,
                'expected_input_tokens': request['input_tokens'], 'success': False, 'rejected': False,
                'error_type': 'NotLaunched', 'error_message': message, 'error_phase': 'scheduler'}

    client = {'requests': [
        launched(0, stream_complete=True, started_at=1000.1005, first_content_at=1000.3, finished_at=1001.0,
                 ttft=1000.3 - 1000.1005, input_tokens=300, output_tokens=50, total_tokens=350, success=True),
        launched(1, status=503, rejected=True, started_at=1000.2, finished_at=1000.21),
        launched(2, started_at=1000.3, finished_at=1006.0, error_type='CancelledError',
                 error_message='replay drain timeout or collection interrupted', error_phase='response_body'),
        placeholder(3, 'collection interrupted'),
        placeholder(4, 'KeyError: literal implementation failure')]}
    manifest = {'schema_version': 1, 'status': 'collected', 'git_commit': 'c' * 40, 'git_dirty': False,
                'config': {'budget': 16.0, 'budget_schedule': None, 'cache_policy': 'disabled', 'ttft_slo': 0.5,
                           'e2e_slo': None, 'trace': 'literal.json'},
                'measurement_start': start, 'measurement_end': start + 5.0,
                'initial_metrics': {'tokenizer_available': True},
                'gateway_launch': {'settings': {'ADMISSION_MODE': 'cost_aware', 'MAX_ADMITTED_COST': '16.0',
                                                'MODEL_NAME': model}}}
    changes = [{'sent': 999.5, 'acknowledged': 999.6, 'target': start}]
    files = {'trace.json': trace, 'client.json': client, 'manifest.json': manifest, 'budget_changes.json': changes}
    if mutate is not None:
        mutate(files)
    texts = {name: value if isinstance(value, str) else json.dumps(value) for name, value in files.items()}
    if 'trace_sha256' not in manifest:
        manifest['trace_sha256'] = hashlib.sha256(texts['trace.json'].encode()).hexdigest()
        texts['manifest.json'] = json.dumps(manifest) if not isinstance(files['manifest.json'], str) \
            else texts['manifest.json']
    directory.mkdir()
    for name, text in texts.items():
        (directory / name).write_text(text)
    return directory


def test_adapter_exact_mapping_static_fixed_and_refused_natural_slo(tmp_path):
    adapted = nm.adapt_run_trace_pilot(write_raw(tmp_path / 'raw'), arm='fixed16', purpose='train')
    rows = adapted['rows']
    assert [r['fate'] for r in rows] == ['complete', 'rejected', 'cancelled', 'not_launched', 'unknown']
    assert rows[0]['sent_at'] == 1000.1005 and rows[0]['terminal_at'] == 1001.0 and rows[0]['completion_tokens'] == 50
    assert rows[0]['group_tokens'] == 300 and rows[4]['sent_at'] is None and rows[4]['terminal_at'] is None
    assert all(r['slo'] is None for r in rows) and 'finish_reason' in adapted['slo_missing'][0]
    assert adapted['envelope_end'] == 1006.0 and adapted['control']['budget'] == 16.0
    provenance = adapted['provenance']
    assert provenance['manifest']['git_commit'] == 'c' * 40 and provenance['offered_tape']['purpose'] == 'train'
    assert provenance['raw_error_type_counts'] == {'None': 2, 'CancelledError': 1, 'NotLaunched': 2}
    assert provenance['client_config_slo_not_used'] == {'ttft_slo': 0.5, 'e2e_slo': None}
    metrics = nm.compute(adapted)
    assert metrics['natural_slo']['available'] is False and metrics['controller_overhead']['decisions'] == 0
    assert metrics['outcomes']['fates']['not_launched'] == 1 and metrics['outcomes']['fates']['unknown'] == 1
    json.dumps(metrics, allow_nan=False)


def _set(path, value):
    def mutate(files):
        target = files
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


@pytest.mark.parametrize('mutate, kwargs, message', [
    (_set(['manifest.json', 'status'], 'failed'), {}, 'collected'),
    (_set(['manifest.json', 'trace_sha256'], 'd' * 64), {}, 'trace.json sha256'),
    (None, {'purpose': 'eval'}, 'purpose'),
    (None, {'arm': 'mpc'}, 'closed-loop'),
    (_set(['manifest.json', 'config', 'cache_policy'], 'unknown'), {}, 'cache_policy'),
    (_set(['client.json', 'requests', 1, 'scheduled_at'], 1000.21), {}, 'scheduled_at'),
    (_set(['manifest.json', 'config', 'budget_schedule'], 'schedule.json'), {}, 'budget-schedule'),
    (_set(['budget_changes.json', 0, 'acknowledged'], 1000.5), {}, 'before measurement_start'),
    (_set(['manifest.json', 'gateway_launch', 'settings', 'MAX_ADMITTED_COST'], '32.0'), {}, 'launch settings'),
    (_set(['trace.json', 'model'], 'Qwen/Qwen3-4B-Instruct-2507'), {}, 'absolute local model path'),
    (_set(['manifest.json', 'gateway_launch', 'settings', 'MODEL_NAME'], '/other/model'), {}, 'MODEL_NAME'),
    (_set(['manifest.json', 'initial_metrics', 'tokenizer_available'], False), {}, 'tokenizer_available'),
    (_set(['client.json', 'requests', 0, 'finish_reason'], 'stop'), {}, 'schema drift'),
    (_set(['client.json', 'requests', 0, 'ttft'], 0.5), {}, 'ttft'),
    (_set(['client.json'], '{"requests": [], "requests": []}'), {}, 'duplicate JSON object key'),
])
def test_adapter_refusals(tmp_path, mutate, kwargs, message):
    arguments = dict({'arm': 'fixed16', 'purpose': 'train'}, **kwargs)
    with pytest.raises(nm.AdapterRefusal, match=message):
        nm.adapt_run_trace_pilot(write_raw(tmp_path / 'raw', mutate), **arguments)


def _fixed(arm, outcomes, identity='e' * 64, purpose='train', missing=None, scope='offered_cohort_observed',
           start=10.0, spacing=0.1, prefix='r', comparability=COMPARABILITY, selection=None):
    rows = []
    for index, kind in enumerate(outcomes):
        sched, rid = start + index * spacing, f'{prefix}{index}'
        if kind == 'good':
            rows.append(row(rid, 'complete', sched, sched, sched + 0.1, sched + 0.46, ct=10,
                            slo_fields=None if missing else slo()))
        elif kind == 'slow':
            rows.append(row(rid, 'complete', sched, sched, sched + 0.9, sched + 1.3, ct=10,
                            slo_fields=None if missing else slo()))
        else:
            rows.append(row(rid, 'rejected', sched, sched, None, sched + 0.01,
                            slo_fields=None if missing else slo(status='rejected', finish=None, done=False)))
    return nm.compute(run(rows, arm=arm, purpose=purpose, missing=missing, control=static(nm.FIXED_BUDGETS[arm]),
                          control_missing=None, identity=identity, scope=scope, start=start,
                          comparability=comparability, selection=selection))


def test_train_selection_primary_ties_and_refusals():
    chosen = nm.select_train_fixed({'fixed16': _fixed('fixed16', ['good', 'rej', 'rej']),
                                    'fixed32': _fixed('fixed32', ['good', 'good', 'rej'])})
    assert (chosen['status'], chosen['selected_arm'], chosen['decided_by']) == ('SELECTED', 'fixed32',
                                                                                'primary_natural_slo_good')
    tie1 = nm.select_train_fixed({'fixed16': _fixed('fixed16', ['good', 'slow', 'rej']),
                                  'fixed32': _fixed('fixed32', ['good', 'rej', 'rej'])})
    assert (tie1['selected_arm'], tie1['decided_by']) == ('fixed16', 'tie_1_fewer_not_delivered')
    tie2 = nm.select_train_fixed({'fixed16': _fixed('fixed16', ['good', 'rej']),
                                  'fixed32': _fixed('fixed32', ['rej', 'good'])})
    assert (tie2['selected_arm'], tie2['decided_by']) == ('fixed16', 'tie_2_lower_budget')
    refusals = [
        {'fixed16': _fixed('fixed16', ['good'], missing=['finish_reason']), 'fixed32': _fixed('fixed32', ['good'])},
        {'fixed16': _fixed('fixed16', ['good'], purpose='eval'), 'fixed32': _fixed('fixed32', ['good'])},
        {'fixed16': _fixed('fixed16', ['good']), 'fixed32': _fixed('fixed32', ['good'], identity='f' * 64)},
        {'fixed16': _fixed('fixed16', ['good'])},
    ]
    reasons = [nm.select_train_fixed(item) for item in refusals]
    assert [item['status'] for item in reasons] == ['REFUSED'] * 4
    assert 'no TTFT-only substitute' in reasons[0]['reason'] and 'Train only' in reasons[1]['reason']


def test_comparison_table_alias_refusals_and_no_winner():
    f16, f32 = _fixed('fixed16', ['good', 'rej']), _fixed('fixed32', ['good', 'good'])
    selection = nm.select_train_fixed({'fixed16': f16, 'fixed32': f32})
    table = nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': f16},
                                 'fixed32': {'status': 'MEASURED', 'metrics': f32},
                                 'mpc': {'status': 'REFUSED', 'reason': 'model not qualified'}},
                                purpose='train', train_selection=selection)
    assert [r['arm'] for r in table['rows']] == list(nm.ARMS)
    by_arm = {r['arm']: r for r in table['rows']}
    assert by_arm['train_selected_fixed']['alias_of'] == 'fixed32'
    assert by_arm['train_selected_fixed']['status'] == 'ALIAS'
    assert by_arm['mpc']['status'] == 'REFUSED' and by_arm['strong_one_move']['status'] == 'NOT_RUN'
    assert by_arm['fixed32']['summary']['slo_attainment_fraction'] == 1.0
    assert table['fifo_label'] == 'PENDING_VERIFIED_BACKEND_FCFS_CONFIG'
    assert not [key for key in keys_recursive(table) if 'winner' in key or 'best_arm' in key]
    other = _fixed('fixed32', ['good', 'good'], identity='f' * 64)
    with pytest.raises(nm.MetricsInputError, match='incomparable'):
        nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': f16},
                             'fixed32': {'status': 'MEASURED', 'metrics': other}}, purpose='train')
    full_wall = _fixed('fixed32', ['good', 'good'], scope='full_wall')
    with pytest.raises(nm.MetricsInputError, match='envelope scope: incomparable'):
        nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': f16},
                             'fixed32': {'status': 'MEASURED', 'metrics': full_wall}}, purpose='train')
    with pytest.raises(nm.MetricsInputError, match='this arm/split'):
        nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': f16}}, purpose='eval')


def test_comparability_facts_validated_and_required_for_selection_and_tables():
    rows = [row('a', 'complete', 10.0, 10.0, 10.2, 10.5, slo_fields=slo())]
    declared = nm.compute(run(rows, missing=None, comparability=COMPARABILITY))['comparability']
    assert (declared['status'], declared['sha256']) == ('DECLARED', canonical_sha(COMPARABILITY))
    model = COMPARABILITY['model']
    for change, message in (
            ({'clock': ''}, 'comparability.clock'),
            ({'hardware': {}}, 'hardware must be a non-empty object'),
            ({'extra': {'x': 1}}, 'keys must be exactly'),
            ({'model': dict(model, served_alias_verified=False)}, 'served_alias_verified must be true'),
            ({'model': dict(model, local_model_path='models/Qwen3-4B-Instruct-2507')}, 'absolute local asset path'),
            ({'model': dict(model, asset_sha256='abc')}, '64-hex'),
            ({'gateway': {'settings': {'MAX_ADMITTED_COST': '16.0'}}}, 'per-arm MAX_ADMITTED_COST'),
            ({'client_generation': {'ignore_eos': True, 'max_tokens_policy': 'trace max_tokens'}},
             'requested_ignore_eos differs')):
        with pytest.raises(nm.MetricsInputError, match=message):
            nm.compute(run(copy.deepcopy(rows), missing=None, comparability=dict(COMPARABILITY, **change)))
    base16 = _fixed('fixed16', ['good', 'rej'])
    shifted = nm.select_train_fixed({'fixed16': base16, 'fixed32': _fixed('fixed32', ['good', 'good'], start=500.0)})
    assert (shifted['status'], shifted['selected_arm']) == ('SELECTED', 'fixed32')  # absolute start is not identity
    for candidate, message in (
            (_fixed('fixed32', ['good', 'good'], comparability=None), "comparability UNVERIFIED for ['fixed32']"),
            (_fixed('fixed32', ['good', 'good'], comparability=dict(COMPARABILITY, hardware={'gpu': 'other'})),
             "incomparable: comparability facts differ ['hardware']"),
            (_fixed('fixed32', ['good', 'good'], spacing=0.2), 'relative schedule')):
        refused = nm.select_train_fixed({'fixed16': base16, 'fixed32': candidate})
        assert refused['status'] == 'REFUSED' and message in refused['reason']
        assert 'selected_arm' not in refused
        with pytest.raises(nm.MetricsInputError, match=re.escape(message)):
            nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': base16},
                                 'fixed32': {'status': 'MEASURED', 'metrics': candidate}}, purpose='train')


def test_selection_record_binding_and_eval_alias_reuse_refusals():
    f16, f32 = _fixed('fixed16', ['good', 'rej']), _fixed('fixed32', ['good', 'good'])
    selection = nm.select_train_fixed({'fixed16': f16, 'fixed32': f32})
    unsealed = {key: value for key, value in selection.items() if key != 'selection_sha256'}
    assert selection['selection_sha256'] == canonical_sha(unsealed)
    assert (selection['decided_by'], selection['candidates']['fixed16']['metrics_sha256']) == (
        'primary_natural_slo_good', canonical_sha(f16))
    train = {'fixed16': {'status': 'MEASURED', 'metrics': f16}, 'fixed32': {'status': 'MEASURED', 'metrics': f32}}
    tampered = dict(selection, selected_arm='fixed16', selected_budget=16.0)
    contradictory = dict(tampered, decided_by='tie_2_lower_budget')
    contradictory['selection_sha256'] = canonical_sha({key: value for key, value in contradictory.items()
                                                       if key != 'selection_sha256'})
    for forged, message in (({'schema': nm.SELECTION_SCHEMA, 'status': 'SELECTED', 'selected_arm': 'fixed32'},
                             'keys must be exactly'), (tampered, 'tampered or forged'),
                            (contradictory, 'contradicts')):
        with pytest.raises(nm.MetricsInputError, match=message):
            nm.comparison_table(train, purpose='train', train_selection=forged)
    with pytest.raises(nm.MetricsInputError, match='not the measured fixed16'):
        nm.comparison_table(dict(train, fixed16={'status': 'MEASURED', 'metrics': _fixed('fixed16', ['good', 'good'])}),
                            purpose='train', train_selection=selection)

    def eval_entry(arm, outcomes=('good', 'rej'), identity='9' * 64, prefix='e', **kwargs):
        return {'status': 'MEASURED', 'metrics': _fixed(arm, list(outcomes), identity=identity, purpose='eval',
                                                        prefix=prefix, **kwargs)}

    def alias(entries, chosen=selection):
        table = nm.comparison_table(entries, purpose='eval', train_selection=chosen)
        return next(item for item in table['rows'] if item['arm'] == 'train_selected_fixed')

    sha = selection['selection_sha256']
    bound = {arm: eval_entry(arm, selection=sha) for arm in nm.FIXED_BUDGETS}
    accepted = alias(bound)
    assert (accepted['status'], accepted['alias_of'], accepted['selection']['selection_sha256']) == (
        'ALIAS', 'fixed32', sha)
    unbound = alias(dict(bound, fixed16=eval_entry('fixed16')))
    assert unbound['status'] == 'REFUSED' and 'selection provenance UNVERIFIED' in unbound['reason']
    refused_selection = nm.select_train_fixed({'fixed16': f16})
    assert alias(bound, refused_selection)['status'] == 'REFUSED'
    other_environment = dict(COMPARABILITY, hardware={'gpu': 'other'})
    for entries, message in (
            (dict(bound, fixed16=eval_entry('fixed16', selection='0' * 64)), 'different train_selection_sha256'),
            ({arm: eval_entry(arm, identity='e' * 64, selection=sha) for arm in nm.FIXED_BUDGETS},
             'positive Train/Eval reuse signal'),
            ({arm: eval_entry(arm, prefix='r', selection=sha) for arm in nm.FIXED_BUDGETS},
             'positive Train/Eval reuse signal'),
            ({arm: eval_entry(arm, ('good', 'rej', 'rej'), prefix='r', selection=sha) for arm in nm.FIXED_BUDGETS},
             'positive Train/Eval reuse signal'),
            ({arm: eval_entry(arm, comparability=other_environment, selection=sha) for arm in nm.FIXED_BUDGETS},
             'Train selection comparability facts')):
        with pytest.raises(nm.MetricsInputError, match=message):
            alias(entries)


def test_cli_measures_fixed_run_and_reports_refusal(tmp_path, capsys):
    directory = write_raw(tmp_path / 'raw')
    assert nm.main(['--output-dir', str(directory), '--arm', 'fixed16', '--purpose', 'train']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['outcomes']['fates']['complete'] == 1 and printed['natural_slo']['available'] is False
    assert nm.main(['--output-dir', str(directory), '--arm', 'fixed16', '--purpose', 'eval']) == 1
    assert '"status": "REFUSED"' in capsys.readouterr().err
