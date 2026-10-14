from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

import pytest

from analyze_trace_dynamics import analyze_cpu, analyze_run, figures, hashes, interval_overlap, join_diagnostics, join_requests, main, request_windows


RUN_ID = '12345678123456781234567812345678'


def fixture_rows():
    trace = dict(requests=[dict(trace_id=str(i), arrival_seconds=i, input_tokens=8)
                           for i in range(8)])
    clients, histories = [], defaultdict(lambda: defaultdict(list))
    for i in range(7):
        identifier = uuid.uuid5(uuid.UUID(RUN_ID), str(i)).hex
        start = 100. + i
        client = dict(trace_id=str(i), client_request_id=identifier, scheduled_at=start,
                      started_at=start+.1, first_content_at=start+.3, finished_at=start+1.,
                      ttft=.2, end_to_end=.9, expected_input_tokens=8, input_tokens=8,
                      output_tokens=128, max_output_tokens=128, stream_complete=True,
                      success=True, rejected=False, error_type=None, error_phase=None)
        for name, delta in [('received', .11), ('accepted', .12), ('backend_sent', .13),
                            ('first_content', .29), ('completed', .99)]:
            histories[identifier][name].append(dict(event=name, request_id=identifier,
                monotonic=start+delta, estimated_input_tokens=8))
        clients.append(client)
    clients[1].update(first_content_at=101.8, ttft=.7)
    histories[clients[1]['client_request_id']]['first_content'][0]['monotonic'] = 101.79
    clients[2].update(success=False, rejected=True, first_content_at=None, ttft=None,
                      stream_complete=False)
    history = histories[clients[2]['client_request_id']]
    for name in ('accepted', 'backend_sent', 'first_content', 'completed'):
        del history[name]
    history['rejected'].append(dict(event='rejected', request_id=clients[2]['client_request_id'], monotonic=102.12))
    clients[3].update(success=False, first_content_at=None, ttft=None, stream_complete=False,
                      error_type='ConnectError', error_phase='before_response_headers')
    del histories[clients[3]['client_request_id']]
    clients[4].update(success=False, stream_complete=False, error_type='IncompleteStream')
    clients[5].update(success=False, stream_complete=False, error_type='CancelledError')
    clients[6].update(success=False, started_at=None, finished_at=None, end_to_end=None,
                      first_content_at=None, ttft=None, stream_complete=False, error_type='NotLaunched')
    del histories[clients[6]['client_request_id']]
    return trace, clients, histories


def test_all_offered_outcomes_and_distinct_e2e():
    trace, clients, histories = fixture_rows()
    before = deepcopy(clients)
    rows, accounting = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert accounting['outcomes'] == dict(successful_within_ttft=1, successful_over_ttft=1,
        rejected=1, failed=2, cancelled=1, never_launched=1, missing_client_outcome=1)
    assert accounting['planned'] == accounting['ledger_rows'] == sum(accounting['outcomes'].values()) == 8
    assert clients == before
    assert rows[0]['end_to_end'] == .9
    assert rows[0]['scheduled_e2e'] == 1.
    assert rows[0]['scheduled_ttft'] == pytest.approx(.3)
    assert sum(rows[0]['stages'].values()) == pytest.approx(.2)
    assert rows[0]['deadline_remaining_at_admission_s'] == pytest.approx(.38)
    assert rows[3]['stages'] == dict.fromkeys(rows[3]['stages'])
    assert rows[3]['gateway_outcome'] is None
    assert rows[4]['scheduled_ttft'] < .5 and rows[4]['outcome'] == 'failed'
    assert rows[6]['started_s'] is None and rows[6]['scheduled_e2e'] is None
    assert rows[7]['schedule_source'] == 'planned' and rows[7]['scheduled_s'] == 7


def test_partial_stages_remain_unknown_and_duplicates_do_not_disappear():
    trace, clients, histories = fixture_rows()
    identifier = clients[0]['client_request_id']
    del histories[identifier]['backend_sent']
    clients.append(dict(clients[0]))
    rows, accounting = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert accounting['duplicate_trace_ids'] == {'0': 2}
    assert accounting['ledger_rows'] == 9
    assert accounting['extra_client_rows'] == 1
    assert rows[0]['stages']['client_build'] is None
    assert rows[0]['stages']['post_dispatch'] is None
    assert rows[0]['stages']['delivery'] == pytest.approx(.01)
    assert not rows[0]['stages_complete']
    histories[identifier]['accepted'].append(dict(histories[identifier]['accepted'][0]))
    rows, _ = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert rows[0]['stages']['admission'] is None
    assert 'duplicate accepted' in rows[0]['measurement_issues']


def test_deadline_equality_missing_first_content_and_invalid_durations():
    trace, clients, histories = fixture_rows()
    clients[0].update(first_content_at=100.5, ttft=.4)
    rows, _ = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert rows[0]['outcome'] == 'successful_within_ttft'
    clients[0]['first_content_at'] = None
    rows, _ = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert rows[0]['outcome'] == 'successful_ttft_unknown'
    assert rows[0]['stages']['delivery'] is None
    clients[0]['first_content_at'] = 99.
    clients[0]['finished_at'] = 98.
    rows, _ = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert rows[0]['scheduled_ttft'] is None and rows[0]['scheduled_e2e'] is None
    assert 'negative delivery duration' in rows[0]['measurement_issues']
    assert not rows[0]['stages_complete']


def test_joint_deadlines_reconcile_all_offered_cohorts_without_changing_legacy_outcomes():
    trace, clients, histories = fixture_rows()
    legacy, legacy_accounting = join_requests(clients, trace, histories, 100., .5, RUN_ID)
    rows, accounting = join_requests(clients, trace, histories, 100., .5, RUN_ID, 1.)
    assert accounting['outcomes'] == legacy_accounting['outcomes']
    assert accounting['joint_slo'] == dict(met=1, missed=6, unknown=1)
    assert all('joint_slo_met' not in row for row in legacy)
    assert rows[0]['e2e_slo_met'] is True and rows[0]['joint_slo_met'] is True
    assert rows[0]['e2e_deadline_remaining_at_admission_s'] == pytest.approx(.88)
    assert rows[1]['e2e_slo_met'] is True and rows[1]['joint_slo_met'] is False
    assert all(row['joint_slo_met'] is False for row in rows[2:7])
    assert rows[7]['joint_slo_met'] is None
    events = [e for history in histories.values() for group in history.values() for e in group]
    windows = request_windows(rows, events, 100., 108.)
    assert sum(w['cohort_slo_successes'] for w in windows['windows']) == 1
    assert sum(w['cohort_slo_misses'] for w in windows['windows']) == 6
    assert sum(w['cohort_slo_unknown'] for w in windows['windows']) == 1
    assert windows['windows'][0]['cohort_slo_successes'] == 1
    assert windows['windows'][1]['completed'] == 1
    assert 'cohort_slo_successes' not in request_windows(legacy, events, 100., 108.)['windows'][0]


def test_joint_deadline_uses_scheduled_completion_and_preserves_unknowns():
    trace, clients, histories = fixture_rows()
    clients[0].update(finished_at=101.05, end_to_end=.95)
    rows, _ = join_requests(clients, trace, histories, 100., .5, RUN_ID, 1.)
    assert rows[0]['end_to_end'] < 1. and rows[0]['scheduled_e2e'] > 1.
    assert rows[0]['outcome'] == 'successful_within_ttft'
    assert rows[0]['e2e_slo_met'] is False and rows[0]['joint_slo_met'] is False
    clients[0].update(first_content_at=None, finished_at=100.9, end_to_end=.8)
    rows, _ = join_requests(clients, trace, histories, 100., .5, RUN_ID, 1.)
    assert rows[0]['e2e_slo_met'] is True and rows[0]['joint_slo_met'] is None
    clients[0].update(finished_at=101.05, end_to_end=.95)
    rows, _ = join_requests(clients, trace, histories, 100., .5, RUN_ID, 1.)
    assert rows[0]['e2e_slo_met'] is False and rows[0]['joint_slo_met'] is False
    clients[0].update(first_content_at=100.3, finished_at=None, end_to_end=None)
    rows, _ = join_requests(clients, trace, histories, 100., .5, RUN_ID, 1.)
    assert rows[0]['e2e_slo_met'] is None and rows[0]['joint_slo_met'] is None
    with pytest.raises(ValueError):
        join_requests(clients, trace, histories, 100., .5, RUN_ID, float('nan'))


@pytest.mark.parametrize('damage', ['completion_order', 'content_order', 'success_error', 'missing_success'])
def test_inconsistent_timing_or_success_is_not_joint_compliance(damage):
    trace, clients, histories = fixture_rows()
    if damage == 'completion_order':
        clients[0].update(finished_at=100.2, end_to_end=.1)
    elif damage == 'content_order':
        clients[0].update(first_content_at=100.05, ttft=-.05)
    elif damage == 'success_error':
        clients[0]['error_type'] = 'ReadError'
    else:
        del clients[0]['success']
    rows, _ = join_requests(clients, trace, histories, 100., .5, RUN_ID, 1.)
    assert rows[0]['joint_slo_met'] is None
    if damage in {'completion_order', 'content_order'}:
        assert rows[0]['measurement_issues']


def test_duplicate_planned_rows_and_truncated_gateway_log_are_visible(tmp_path):
    trace, clients, histories = fixture_rows()
    trace['requests'].append(dict(trace['requests'][-1]))
    rows, accounting = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    assert len(rows) == accounting['planned'] == 9
    assert accounting['duplicate_planned_ids'] == {'7': 2}
    assert accounting['missing_client_outcomes'] == 2
    for name, value in {
        'manifest.json': dict(run_id=RUN_ID, measurement_start=100., measurement_end=108.,
                              config=dict(ttft_slo=.5, max_scheduling_lag=.05, budget=96.)),
        'trace.json': trace, 'client.json': dict(requests=clients),
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / 'gateway.jsonl').write_text('{"event":')
    summary, rows, _, _ = analyze_run(tmp_path, .3)
    assert summary['malformed_gateway_lines'] == [1]
    assert len(rows) == 9 and not summary['report']['valid']


def test_windows_reconcile_outcomes_tokens_boundaries_and_drain():
    trace, clients, histories = fixture_rows()
    rows, _ = join_requests(clients, trace, histories, 100, .5, RUN_ID)
    events = [e for history in histories.values() for group in history.values() for e in group]
    events.append(dict(event='accepted', request_id='warmup', monotonic=99., estimated_input_tokens=100))
    windows = request_windows(rows, events, 100, 108)
    assert sum(w['scheduled'] for w in windows['windows']) == 8
    assert sum(w['started'] for w in windows['windows']) + windows['unbinned']['started'] == 8
    assert sum(w['accepted'] for w in windows['windows']) == 4
    assert sum(w['admitted_input_tokens_known'] for w in windows['windows']) == 32
    events.append(dict(event='accepted', request_id=rows[0]['request_id'], monotonic=108.))
    windows = request_windows(rows, events, 100, 108)
    assert windows['windows'][-1]['start_s'] == 8
    assert not windows['windows'][-1]['arrival_window']
    assert windows['windows'][-1]['admitted_input_tokens_unknown_requests'] == 1


def test_diagnostic_actual_times_union_cpu_and_async_wait_are_not_causal_sums():
    joined = [dict(request_id='one', dispatch_s=0., gateway_content_s=1.)]
    records = [
        dict(event='gc_span', span_start=100.2, span_end=100.6, monotonic=102.,
             callback_thread_cpu_seconds=.1, on_loop_thread=True),
        dict(event='gc_span', span_start=100.4, span_end=100.8, monotonic=103.,
             callback_thread_cpu_seconds=.2, on_loop_thread=False),
        dict(event='event_loop_lag', scheduled_at=100., monotonic=101., lag_seconds=1.),
        dict(event='request_diagnostics', request_id='one', cleanup_complete=True,
             phases=dict(http_send_headers=dict(start=100., end=100.9, status='completed'))),
        dict(event='diagnostic_cpu', sampled_at=100., loop_thread_cpu_seconds=5., process_cpu_seconds=7., loop_thread_id=1),
        dict(event='diagnostic_cpu', sampled_at=101., loop_thread_cpu_seconds=5.1, process_cpu_seconds=7.2, loop_thread_id=1),
    ]
    manifest = dict(investigation_protocol='periodic-peaks-v1', gateway_diagnostics_requested=True)
    report = dict(diagnostic_integrity=dict(valid=True, gc_coverage_complete=True, external_mode='disabled'))
    result = join_diagnostics(joined, records, [], 100., manifest, report)
    assert joined[0]['diagnostics']['post_dispatch_gc_overlap_s'] == pytest.approx(.6)
    assert joined[0]['diagnostics']['post_dispatch_loop_gc_overlap_s'] == pytest.approx(.4)
    monitor, = result['monitor_overdue_intervals']
    assert monitor['gc_overlap_s'] == pytest.approx(.6)
    assert monitor['observed_phase_overlap_s']['http_send_headers'] == pytest.approx(.9)
    assert result['cpu_intervals'][0]['loop_thread_cpu_seconds'] == pytest.approx(.1)
    assert result['request_linked_prefill'] == 'unavailable'
    assert interval_overlap(None, 1, [(0, 1)]) is None
    # A measured non-GC CPU interval is not fabricated GC or exclusive request CPU.
    no_gc = [r for r in records if r['event'] != 'gc_span']
    result = join_diagnostics(joined, no_gc, [], 100., manifest, report)
    assert result['monitor_overdue_intervals'][0]['gc_overlap_s'] == 0
    # An unqualified/disabled/legacy absence is different from measured zero GC.
    for current_manifest, current_report in [
        (manifest, dict(diagnostic_integrity=dict(valid=False, gc_coverage_complete=False))),
        (dict(investigation_protocol='periodic-peaks-v1', gateway_diagnostics_requested=False), {}),
        ({}, {}),
    ]:
        join_diagnostics(joined, no_gc, [], 100., current_manifest, current_report)
        assert joined[0]['diagnostics']['post_dispatch_gc_overlap_s'] is None


def test_incomplete_zero_success_run_retains_invalid_audit_and_plots(tmp_path, monkeypatch):
    trace, clients, _ = fixture_rows()
    clients = clients[3:]
    root, output = tmp_path / 'raw', tmp_path / 'analysis'
    root.mkdir()
    output.mkdir()
    for name, value in {
        'manifest.json': dict(run_id=RUN_ID, measurement_start=100., measurement_end=108.,
                              config=dict(ttft_slo=.5, max_scheduling_lag=.05, budget=96.)),
        'trace.json': trace, 'client.json': dict(requests=clients),
    }.items():
        (root / name).write_text(json.dumps(value))
    # A delayed monitor with no adjacent lifecycle must not invent a gap.
    (root / 'gateway.jsonl').write_text(json.dumps(dict(event='event_loop_lag',
        monotonic=101., scheduled_at=100.5, lag_seconds=.5)) + '\n')
    before = hashes(root)
    data = analyze_run(root, .3)
    summary, rows, _, _ = data
    assert not summary['report']['valid'] and not summary['qualified_for_comparison']
    assert not summary['original_report_reproduced']
    assert summary['accounting']['ledger_rows'] == 8
    assert summary['slowest_one_percent']['count'] == 0
    assert summary['slowest_one_percent']['post_dispatch_fraction'] is None
    assert summary['large_monitor_delays'][0]['gap_s'] is None
    assert 'telemetry_status.json' in summary['missing_raw_files']
    assert len(rows) == 8
    figures(output, [data], .3)
    assert (output / 'latency_stages.png').stat().st_size > 1000
    assert (output / 'stall_alignment.png').stat().st_size > 1000
    assert (output / 'cpu_scheduling.png').stat().st_size > 1000
    assert summary['cpu_analysis']['sources']['runner']['status'] == 'legacy_unavailable'
    assert hashes(root) == before
    monkeypatch.setattr('sys.argv', ['analyze_trace_dynamics.py', str(root), '--output-dir', str(output)])
    main()
    exported = [json.loads(line) for line in (output / 'raw-requests.jsonl').read_text().splitlines()]
    assert len(exported) == 8
    assert json.loads((output / 'summary.json').read_text())['schema_version'] == 2
    assert hashes(root) == before
    monkeypatch.setattr('sys.argv', ['analyze_trace_dynamics.py', str(root), '--output-dir', str(root / 'nested')])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize('name,successes,misses', [
    ('trace-smoke-96', 248, None), ('trace-experiment-96-r1', 2239, 1518),
    ('trace-experiment-96-r2', 2220, 1561)])
def test_frozen_reports_reproduce_without_mutating_raw(name, successes, misses):
    root = Path(os.environ.get('TRACE_BASELINE_ROOT', Path(__file__).parents[1] / 'results')) / name
    if not root.is_dir():
        pytest.fail(f'Frozen bundle required: {root}')
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    summary, rows, _, _ = analyze_run(root, .3)
    assert summary['original_report_reproduced']
    assert summary['report'] == json.loads((root / 'report.json').read_text())
    assert summary['report']['valid']
    assert summary['accounting']['ledger_rows'] == summary['accounting']['planned'] == len(rows)
    assert sum(row['success'] for row in rows) == successes
    if misses is not None:
        assert summary['accounting']['outcomes']['successful_over_ttft'] == misses
        assert not summary['report']['load_fidelity']['valid']
        assert not summary['qualified_for_comparison']
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()} == before


def cpu_fixture():
    manifest = dict(investigation_protocol='periodic-peaks-v1', run_id=RUN_ID,
                    gateway_diagnostics_requested=True, sampler=dict(enabled=True), host='test-host',
                    runner_process_id=11, gateway_launch=dict(process_id=22),
                    config=dict(max_scheduling_lag=.05))
    report = dict(diagnostic_integrity=dict(valid=True, gateway_mode='enabled', external_mode='enabled'))
    observations, records = [], []
    for i in range(2):
        at = 100. + 2*i
        def process(pid, user, system):
            return dict(available=True, sample_start=at+.1, sample_end=at+.3, pid=pid,
                        process_created_unix=10., user_cpu_seconds=user, system_cpu_seconds=system)
        observations.append(dict(run_id=RUN_ID, host='test-host', counter_epoch=i+1,
            sent_at=at-10, received_at=at-5, finished_at=at+5,
            resources=dict(runner=process(11, 10.+2*i, 2.+i),
                           gateway=process(22, 4.+1.5*i, 1.+.5*i),
                           host_cpu=dict(available=True, sample_start=at+.1, sample_end=at+.3,
                               logical_cpus=8, cpu_seconds=dict(user=100.+4*i, nice=10.+i,
                                   system=50.+2*i, idle=840.+8*i, iowait=10.+i,
                                   guest=5.+2*i, guest_nice=2.+.5*i)))))
        records.append(dict(event='diagnostic_cpu', run_id=RUN_ID, sampled_at=at+.7,
                            monotonic=at+10, loop_thread_id=3, loop_thread_cpu_seconds=3.+.6*i))
    joined = [dict(request_id='same-id', trace_id=str(i), outcome=outcome,
                   scheduled_s=1., started_s=1.1, scheduling_lag_s=.1)
              for i, outcome in enumerate(('successful_within_ttft', 'rejected', 'failed'))]
    return manifest, report, records, observations, joined


def test_cpu_read_times_units_guest_accounting_and_all_late_outcomes():
    manifest, report, records, observations, joined = cpu_fixture()
    before = deepcopy((records, observations, joined))
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    runner, = result['sources']['runner']['intervals']
    assert runner['start_s'] == pytest.approx(.2)
    assert runner['end_s'] == pytest.approx(2.2)
    assert runner['elapsed_s'] == pytest.approx(2.)
    assert runner['cpu_seconds'] == dict(user_cpu_seconds=2., system_cpu_seconds=1.)
    assert runner['cpu_cores'] == pytest.approx(1.5)  # Not divided by eight host CPUs.
    assert not runner['issues']  # Engine counter epochs do not reset process CPU.
    host, = result['sources']['host']['intervals']
    assert host['host_nonidle_fraction'] == pytest.approx(7/16)
    assert host['cpu_cores'] is None and host['cpu_seconds']['guest'] == 2
    loop, = result['sources']['gateway_loop']['intervals']
    assert loop['start_s'] == pytest.approx(.7)  # Never the later emission time.
    assert loop['cpu_cores'] == pytest.approx(.3)
    assert [r['ledger_index'] for r in result['late_launches']] == [0, 1, 2]
    assert [r['outcome'] for r in result['late_launches']] == [r['outcome'] for r in joined]
    for late in result['late_launches']:
        match, = late['cpu_interval_overlaps']['runner']
        assert match['interval_index'] == 0 and match['possible_overlap_s'] == pytest.approx(.1)
        assert 'cpu_seconds' not in late  # No proportional CPU attribution.
    assert (records, observations, joined) == before


def test_cpu_overlap_uses_outer_read_bounds_and_excludes_touching_intervals():
    manifest, report, records, observations, joined = cpu_fixture()
    joined[0].update(scheduled_s=.11, started_s=.19, scheduling_lag_s=.08)
    joined[1].update(scheduled_s=2.3+1e-6, started_s=2.4, scheduling_lag_s=.1-1e-6)
    joined[2].update(started_s=None, scheduling_lag_s=None, outcome='never_launched')
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    match, = result['late_launches'][0]['cpu_interval_overlaps']['runner']
    assert match['possible_overlap_s'] == pytest.approx(.08)
    assert result['late_launches'][1]['cpu_interval_overlaps']['runner'] == []
    assert result['launch_timing_unavailable_count'] == 1
    joined[0]['scheduling_lag_s'] = .05
    assert len(analyze_cpu(joined, records, observations, 100., manifest, report)['late_launches']) == 1


@pytest.mark.parametrize('field,value', [
    ('available', False), ('user_cpu_seconds', 9.), ('user_cpu_seconds', float('nan')),
    ('system_cpu_seconds', True), ('pid', 999), ('process_created_unix', 11.),
    ('sample_start', None), ('sample_end', 102.), ('sample_start', 100.2),
])
def test_bad_cpu_pairs_remain_visible_and_unknown(field, value):
    manifest, report, records, observations, joined = cpu_fixture()
    observations[1]['resources']['runner'][field] = value
    if field == 'user_cpu_seconds' and value == 9.:
        observations[1]['resources']['runner']['system_cpu_seconds'] = 100.
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    source = result['sources']['runner']
    interval, = source['intervals']
    assert source['status'] == 'unavailable' and source['invalid_interval_count'] == 1
    assert interval['issues'] and interval['cpu_cores'] is None and interval['cpu_seconds'] is None
    json.dumps(result, allow_nan=False)


def test_cpu_missing_sample_is_not_bridged_and_zero_process_cpu_is_measured():
    manifest, report, records, observations, joined = cpu_fixture()
    missing = deepcopy(observations[0])
    missing['resources']['runner'].update(available=False, sample_start=101., sample_end=101.1,
                                        error_reason='procfs_pid_namespace_unverified')
    final = deepcopy(observations[1])
    final['resources']['runner'].update(sample_start=104.1, sample_end=104.3)
    observations = [observations[0], missing, observations[1], final]
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    source = result['sources']['runner']
    assert source['status'] == 'partial'
    assert [r['cpu_cores'] for r in source['intervals']] == [None, None, 0.]
    assert source['intervals'][2]['previous_sample_index'] == 2
    observations[1]['resources'].pop('runner')
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    assert [r['cpu_cores'] for r in result['sources']['runner']['intervals']] == [None, None, 0.]


@pytest.mark.parametrize('damage', ['count', 'field_set', 'counter_decrease', 'zero_delta', 'unknown_field', 'host'])
def test_host_cpu_changes_are_unknown_not_idle(damage):
    manifest, report, records, observations, joined = cpu_fixture()
    host = observations[1]['resources']['host_cpu']
    if damage == 'count':
        host['logical_cpus'] = 16
    elif damage == 'field_set':
        host['cpu_seconds'].pop('guest')
    elif damage == 'counter_decrease':
        host['cpu_seconds']['idle'] = 839.
    elif damage == 'zero_delta':
        host['cpu_seconds'] = deepcopy(observations[0]['resources']['host_cpu']['cpu_seconds'])
    elif damage == 'unknown_field':
        host['cpu_seconds']['unrecognized'] = 1.
    else:
        observations[1]['host'] = 'other-host'
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    interval, = result['sources']['host']['intervals']
    assert interval['host_nonidle_fraction'] is None and interval['issues']


def test_windows_host_interrupt_and_dpc_are_not_counted_twice():
    manifest, report, records, observations, joined = cpu_fixture()
    for i, row in enumerate(observations):
        row['resources']['host_cpu']['cpu_seconds'] = dict(
            user=10.+2*i, system=20.+3*i, idle=30.+5*i, interrupt=1.+i, dpc=1.+.5*i)
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    interval, = result['sources']['host']['intervals']
    assert interval['host_nonidle_fraction'] == pytest.approx(.5)


def test_idle_host_is_measured_zero_without_subtractive_roundoff():
    manifest, report, records, observations, joined = cpu_fixture()
    for i, row in enumerate(observations):
        row['resources']['host_cpu']['cpu_seconds'] = dict(user=0., system=0., idle=1.11*i, iowait=.22*i)
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    interval, = result['sources']['host']['intervals']
    assert not interval['issues'] and interval['host_nonidle_fraction'] == 0.


@pytest.mark.parametrize('mode', ['legacy', 'disabled', 'failed', 'external_only', 'missing', 'missing_verdict'])
def test_cpu_qualification_modes_do_not_fabricate_zero_cpu(mode):
    manifest, report, records, observations, joined = cpu_fixture()
    if mode == 'legacy':
        manifest.pop('investigation_protocol')
    elif mode == 'disabled':
        manifest.update(gateway_diagnostics_requested=False, sampler=dict(enabled=False))
    elif mode == 'failed':
        report['diagnostic_integrity']['valid'] = False
    elif mode == 'missing_verdict':
        report['diagnostic_integrity'] = None
    elif mode == 'external_only':
        manifest['gateway_diagnostics_requested'] = False
    else:
        observations, records = [], []
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    loop = result['sources']['gateway_loop']
    assert loop['status'] != 'available'
    assert loop['intervals'] == ([] if mode == 'missing' else None)
    runner = result['sources']['runner']
    if mode == 'external_only':
        assert runner['status'] == 'available'
    elif mode == 'missing':
        assert runner['intervals'] == [] and runner['reason'] == 'fewer_than_two_samples'
    else:
        assert runner['intervals'] is None
        assert result['late_launches'][0]['cpu_interval_overlaps']['runner'] is None


def test_cpu_reordered_reads_do_not_create_overlapping_valid_intervals():
    manifest, report, records, observations, joined = cpu_fixture()
    base = observations[0]
    observations = []
    for at in (100., 103., 102., 104., 105.):
        row = deepcopy(base)
        row['resources']['runner'].update(sample_start=at, sample_end=at+.01)
        observations.append(row)
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    intervals = result['sources']['runner']['intervals']
    assert [bool(row['issues']) for row in intervals] == [False, True, True, False]


def test_cpu_invalid_selection_limit_is_explicit():
    manifest, report, records, observations, joined = cpu_fixture()
    manifest['config']['max_scheduling_lag'] = float('nan')
    result = analyze_cpu(joined, records, observations, 100., manifest, report)
    assert result['late_launches'] is None
    assert result['selection_issue'] == 'invalid_scheduling_lag_limit'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mode,label', [
    ('thread', 'Runner process (includes sampler)'),
    ('process', 'Runner process (sampler separate)'),
    ('legacy', 'Runner process (includes sampler)'),
    ('disabled', 'Runner process: disabled'),
])
def test_cpu_scope_and_figure_follow_sampler_execution(tmp_path, monkeypatch, mode, label):
    import matplotlib.pyplot as plt
    from analyze_trace_dynamics import cpu_figure
    manifest, report, records, observations, joined = cpu_fixture()
    if mode == 'disabled':
        manifest['sampler']['enabled'] = False
    elif mode != 'legacy':
        manifest['sampler']['mode'] = mode
    cpu = analyze_cpu(joined, records, observations, 100., manifest, report)
    expected_mode = None if mode == 'disabled' else 'thread' if mode == 'legacy' else mode
    assert cpu['sampler_mode'] == expected_mode
    closed, close = [], plt.close
    def remember_figure(figure=None):
        if hasattr(figure, 'axes'):
            closed.append(figure)
        close(figure)
    monkeypatch.setattr(plt, 'close', remember_figure)
    summary = dict(cpu_analysis=cpu, name='scope', duration_s=3.,
                   report=dict(load_fidelity=dict(valid=False)))
    cpu_figure(tmp_path, [(summary, joined, None, None)])
    assert label in closed[-1].axes[0].get_legend_handles_labels()[1]
    assert (tmp_path / 'cpu_scheduling.png').stat().st_size > 0
