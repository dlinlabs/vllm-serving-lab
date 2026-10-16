import json

import pytest

from analyze_trace_dynamics import hashes
from identification_features import (GatewayReplay, budget_applications, build_table, cohort_labels,
                                     engine_series, event_windows, extract, main, request_labels,
                                     validate_snapshots)
from test_trace_dynamics import RUN_ID, fixture_rows

START = 1000.


def lifecycle(kind, identifier, moment, **extra):
    return dict(event=kind, request_id=identifier, monotonic=START + moment, **extra)


def records():
    return [
        dict(event='budget_updated', applied_at=START - .1, budget=10., budget_version=1),
        lifecycle('received', 'a', .1), lifecycle('accepted', 'a', .15, request_cost=4., estimated_input_tokens=40),
        lifecycle('received', 'b', .3), lifecycle('accepted', 'b', .35, request_cost=5., estimated_input_tokens=50),
        lifecycle('first_content', 'a', .4),
        lifecycle('received', 'c', .5), lifecycle('rejected', 'c', .55, request_cost=3.),
        dict(event='snapshot', state_sampled_at=START + .6, current_admitted_cost=9., current_in_flight=2,
             awaiting_first_content=1, oldest_awaiting_first_seconds=.3, streaming_requests=1,
             active_requests=2, admission_budget=10., budget_version=1),
        lifecycle('first_content', 'b', 1.), lifecycle('completed', 'a', 1.2),
        dict(event='budget_updated', applied_at=START + 1.5, budget=6., budget_version=2),
        lifecycle('received', 'd', 1.6), lifecycle('accepted', 'd', 1.65, request_cost=2., estimated_input_tokens=20),
        lifecycle('completed', 'b', 2.), lifecycle('failed', 'd', 2.6),
    ]


def engine_point(sent, received, waiting, running):
    metric = lambda value: dict(available=True, samples=[dict(value=value)])
    return dict(sent_s=sent, received_s=received,
                metrics=dict(waiting=metric(waiting), running=metric(running), kv_cache=metric(.5)))


ENGINE = [engine_point(.2, .25, 1., 2.), engine_point(1., 1.4, 3., 4.)]
OPTIONS = dict(start=START, initial_budget=10., initial_version=0, ttft_slo=.5, step=.5,
               horizons=(.5, 1.), windows=(.5, 1.), engine_max_age=1., coverage_end=4.)


def event_time(record):
    return next(record[key] for key in ('applied_at', 'monotonic', 'state_sampled_at') if key in record) - START


def test_replay_matches_gateway_snapshot_and_detects_mismatch():
    state = GatewayReplay(records(), START, 10.).state(.6)
    assert state['reserved_cost'] == 9. and state['in_flight'] == 2 and state['active_requests'] == 2
    assert state['awaiting_first_content'] == 1 and state['streaming'] == 1
    assert state['oldest_awaiting_age_s'] == pytest.approx(.3)
    assert state['budget_headroom'] == 1. and state['since_budget_value_change_s'] is None
    assert validate_snapshots(records(), START, 10.)['valid']
    damaged = records()
    damaged[8]['current_admitted_cost'] = 8.
    result = validate_snapshots(damaged, START, 10.)
    assert not result['valid'] and result['mismatches'] == dict(current_admitted_cost=1)
    assert result['examples'][0]['reconstructed'] == 9.


def test_replay_reports_inconsistent_histories_and_budget_versions():
    damaged = records() + [lifecycle('received', 'a', 3.), lifecycle('first_content', 'e', 3.)]
    damaged.append(dict(event='budget_updated', applied_at=START + 3., budget=8., budget_version=4))
    applications, issues = budget_applications(damaged, START, 10., 0)
    assert [version for _, _, version in applications] == [1, 2]
    assert issues == ['nonconsecutive budget version 4 at 3.000000s']
    result = validate_snapshots(damaged, START, 10.)
    assert not result['valid'] and result['replay_issue_count'] == 3
    replay = GatewayReplay(records(), START, 10.)
    replay.state(1.)
    with pytest.raises(ValueError, match='must not decrease'):
        replay.state(.5)


def test_features_never_read_events_after_the_decision_time():
    full, _ = build_table(records(), [], ENGINE, duration=3., **OPTIONS)
    assert [row['t_s'] for row in full] == [0., .5, 1., 1.5, 2., 2.5]
    for row in full:
        t = row['t_s']
        prefix = [record for record in records() if event_time(record) <= t]
        engine = [point for point in ENGINE if point['received_s'] <= t]
        truncated, _ = build_table(prefix, [], engine, duration=t + .5, **OPTIONS)
        assert truncated[-1]['t_s'] == t
        assert truncated[-1]['features'] == row['features']


def test_engine_samples_are_used_only_after_receipt_and_expire():
    rows, info = build_table(records(), [], ENGINE, duration=3., **OPTIONS)
    by_time = {row['t_s']: row['features'] for row in rows}
    assert info['engine_samples'] == 2
    assert by_time[0.]['engine_missing'] and by_time[0.]['engine_waiting'] is None
    assert by_time[1.]['engine_waiting'] == 1. and by_time[1.]['engine_age_s'] == pytest.approx(.8)
    assert by_time[1.5]['engine_waiting'] == 3. and by_time[1.5]['engine_running'] == 4.
    assert by_time[2.5]['engine_missing'] and by_time[2.5]['engine_running'] is None
    broken = dict(ENGINE[0], received_s=.1)
    assert engine_series([broken, dict(ENGINE[1], metrics={})]) == []


def test_controls_and_labels_are_separate_from_features():
    rows, info = build_table(records(), [], ENGINE, duration=3., **OPTIONS)
    row = {row['t_s']: row for row in rows}[1.]
    assert row['features']['budget'] == 10. and 'budget_end' not in row['features']
    assert row['control']['1'] == dict(budget_mean=pytest.approx(8.), budget_end=6.)
    assert row['labels']['1']['reserved_cost'] == 2. and row['labels']['1']['received'] == 1
    assert row['labels']['0.5']['completed'] == 1 and row['labels']['0.5']['first_content'] == 0
    assert not any(key.startswith('cohort') for key in row['features'])
    assert [a['budget_version'] for a in info['budget_applications']] == [1, 2]
    last = rows[-1]
    assert last['label_window_complete'] == {'0.5': True, '1': False}
    late, _ = build_table(records(), [], ENGINE, duration=3., **dict(OPTIONS, coverage_end=3.))
    assert late[-1]['labels']['1'] is None and late[-1]['labels']['0.5'] is not None


def test_receipt_anchored_proxies_resolve_at_their_deadlines():
    flows = event_windows(records(), START, .5, e2e_slo=2.)
    assert flows['resolved_ttft_met'].total(.5, .65) == 1 and flows['resolved_ttft_met'].total(0, .55) == 0
    assert flows['resolved_admitted_ttft_missed'].total(0, .85) == 1
    assert flows['resolved_admitted_ttft_missed'].total(0, 2.15) == 2
    assert flows['resolved_joint_met'].total(0, 2.05) == 0 and flows['resolved_joint_met'].total(0, 2.15) == 1
    assert flows['accepted_input_tokens'].total(0, 1) == 90. and flows['offered_cost'].total(0, 1) == 12.
    assert flows['failed'].total(0, 3) == 1 and flows['rejected'].total(0, 3) == 1
    assert 'resolved_joint_met' not in event_windows(records(), START, .5)


def test_cohort_labels_keep_all_offered_unknowns_and_resolution_time():
    ledger = [dict(scheduled_s=0., admitted_s=.1, outcome='successful_within_ttft', finished_s=1., joint_slo_met=True),
              dict(scheduled_s=.2, admitted_s=.3, outcome='successful_over_ttft', finished_s=3., joint_slo_met=False),
              dict(scheduled_s=.4, admitted_s=None, outcome='rejected', gateway_terminal_s=.45, joint_slo_met=False),
              dict(scheduled_s=.9, admitted_s=None, outcome='successful_ttft_unknown', finished_s=2.,
                   joint_slo_met=None)]
    scheduled = [row['scheduled_s'] for row in ledger]
    assert cohort_labels(ledger, scheduled, 0., .5, None) == dict(
        cohort_offered=3, cohort_admitted=2, cohort_unknown=0, cohort_ttft_met=1,
        cohort_admitted_ttft_missed=1, cohort_resolved_s=3.)
    labels = cohort_labels(ledger, scheduled, .5, 1., 2.)
    assert labels['cohort_unknown'] == 1 and labels['cohort_ttft_met'] is None
    assert labels['cohort_joint_met'] is None and labels['cohort_joint_unknown'] == 1
    assert labels['cohort_resolved_s'] == 2.
    assert cohort_labels(ledger, scheduled, 0., .5, 2.)['cohort_joint_met'] == 1
    exported = request_labels(list(reversed(ledger)) + [dict(scheduled_s=None, outcome='missing_client_outcome')])
    assert [row['scheduled_s'] for row in exported] == scheduled
    assert [row['resolved_s'] for row in exported] == [1., 3., .45, 2.]


def write_run(root, e2e_slo=None):
    trace, clients, histories = fixture_rows()
    config = dict(ttft_slo=.5, max_scheduling_lag=.05, budget=96.)
    if e2e_slo is not None:
        config['e2e_slo'] = e2e_slo
    events = sorted((event for history in histories.values() for items in history.values() for event in items),
                    key=lambda event: event['monotonic'])
    for event in events:
        if event['event'] == 'accepted':
            event['request_cost'] = 1.
    root.mkdir()
    for name, value in {'manifest.json': dict(run_id=RUN_ID, measurement_start=100., measurement_end=108.,
                                              config=config),
                        'trace.json': trace, 'client.json': dict(requests=clients)}.items():
        (root / name).write_text(json.dumps(value))
    (root / 'gateway.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    return root


def test_extract_refuses_unqualified_runs_and_records_analyst_e2e(tmp_path):
    root = write_run(tmp_path / 'raw')
    before = hashes(root)
    options = dict(step=.5, horizons=(1.,), windows=(1.,))
    with pytest.raises(ValueError, match='not qualified'):
        extract(root, **options)
    run, rows, requests = extract(root, allow_unqualified=True, **options)
    assert not run['qualified_for_comparison'] and len(rows) == 16 and len(requests) == 8
    assert run['e2e_slo'] is None and run['e2e_slo_source'] is None
    assert not run['state_reconstruction']['valid'] and run['state_reconstruction']['snapshots'] == 0
    assert 'joint_slo_met' not in requests[0] and requests[0]['scheduled_ttft'] == pytest.approx(.3)
    run, rows, requests = extract(root, allow_unqualified=True, e2e_slo=1.5, **options)
    assert run['e2e_slo'] == 1.5 and run['e2e_slo_source'] == 'analyst_offline'
    assert requests[0]['joint_slo_met'] is True and 'cohort_joint_met' in rows[0]['labels']['1']
    assert hashes(root) == before
    declared = write_run(tmp_path / 'declared', e2e_slo=2.)
    assert extract(declared, allow_unqualified=True, **options)[0]['e2e_slo_source'] == 'declared'
    with pytest.raises(ValueError, match='declares E2E SLO'):
        extract(declared, allow_unqualified=True, e2e_slo=1.5, **options)


def test_cli_writes_separate_feature_and_label_files(tmp_path, monkeypatch):
    root, output = write_run(tmp_path / 'raw'), tmp_path / 'features'
    monkeypatch.setattr('sys.argv', ['identification_features.py', str(root), '--output-dir', str(output),
                                     '--horizons', '1', '--windows', '1', '--allow-unqualified'])
    main()
    summary = json.loads((output / 'features_summary.json').read_text())
    assert summary['protocol'] == 'causal-gateway-features-v1' and summary['runs'][0]['name'] == 'raw'
    assert len((output / 'raw-features.jsonl').read_text().splitlines()) == 16
    assert len((output / 'raw-request-labels.jsonl').read_text().splitlines()) == 8
    for argv, target in ((['--step', '.3'], tmp_path / 'step'), (['--e2e-slo', '0'], tmp_path / 'slo'),
                         ([], root / 'nested')):
        monkeypatch.setattr('sys.argv', ['identification_features.py', str(root), '--output-dir',
                                         str(target), '--allow-unqualified'] + argv)
        with pytest.raises(SystemExit):
            main()
        assert not target.exists()
