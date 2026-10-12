from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

import pytest

from analyze_trace_dynamics import analyze_run, figures, hashes, interval_overlap, join_diagnostics, join_requests, main, request_windows


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
