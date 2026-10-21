import json
import math
from collections import deque

import numpy as np
import pytest

import evaluate_greybox as ev
import greybox_simulator as gb

COEF = (0.01, 0.001, 0.0001)


def arrival(at, cost=1., tokens=100., rid=None):
    return gb.Arrival(at, at, cost, tokens, str(at) if rid is None else rid)


def requests(sched, acc, first, done, cost=1., tokens=100.):
    sched, acc, first, done = (np.array(v, float) for v in (sched, acc, first, done))
    n = len(sched)
    r = dict(sched=sched, recv=sched.copy(), acc=acc, first=first, done=done, cost=np.full(n, cost),
             est_in=np.full(n, tokens), success=~np.isnan(done), ttft=first - sched, c_first=first.copy(),
             trace_id=np.array([str(i) for i in range(n)], dtype=object),
             request_id=np.array([str(i) for i in range(n)], dtype=object), rejected_at=np.full(n, np.nan))
    r['admitted'] = ~np.isnan(acc)
    r['met'] = r['success'] & (r['ttft'] <= gb.SLO)
    return r


def run_for(req, steps_per_second=10., cut=None):
    t = np.arange(0., 120., 1.)
    scrapes = dict(t=t, recv=t + .1, steps=steps_per_second * t)
    meta = dict(schedule=[dict(at_seconds=0., budget=8.), dict(at_seconds=60., budget=4.)])
    return gb.Run('synthetic', meta, req, scrapes, cut=cut)


def test_single_request_timing_and_release():
    out = gb.simulate(gb.EmptyStart(1.), 0., [arrival(0.), arrival(1.4), arrival(1.5)], COEF, horizon=2.)
    by = {s: (a, f) for s, a, f, _, _ in out}
    # An arrival at an idle engine joins at the next step boundary (one empty step of c0), then
    # its prompt is prefilled in one step of c0 + c2 * tokens.
    assert by[0.][0] and by[0.][1] == pytest.approx(0.01 + 0.01 + 0.0001 * 100)
    # It holds the whole cap until 127 decode steps of c0 + c1 * 1 have completed.
    release = 0.03 + 127 * 0.011
    assert release < 1.5
    assert by[1.4][0] is False
    assert by[1.5][0] and by[1.5][1] == pytest.approx(1.5 + 0.01 + 0.02)


def test_admission_rule_is_reserved_plus_cost_within_cap():
    arrivals = [arrival(0., cost=c, rid=str(i)) for i, c in enumerate((2., 1.5, 0.5))]
    out = gb.simulate(gb.EmptyStart(2.5), 0., arrivals, COEF)
    assert [a for _, a, _, _, _ in out[:1]] == [False]
    assert sorted(a for _, a, _, _, _ in out) == [False, True, True]


def test_prefill_respects_token_budget_and_fifo():
    arrivals = [arrival(0., tokens=gb.B), arrival(0., tokens=10., rid='second')]
    out = gb.simulate(gb.EmptyStart(10.), 0., arrivals, COEF)
    first = sorted(f for _, a, f, _, _ in out if a)
    big = 0.01 + 0.01 + 0.0001 * gb.B
    # The second prompt waits for the next step, which also decodes the first request.
    assert first == pytest.approx([big, big + 0.01 + 0.001 + 0.0001 * 10])


def test_unreached_arrivals_are_reported_as_unknown():
    slow = (1., 0., 0.)
    out = gb.simulate(gb.EmptyStart(10.), 0., [arrival(0.), gb.Arrival(5.25, 4.95, 1., 100., 'late')], slow, horizon=5., extra=0.)
    assert (0., True, 2.) == out[0][:3]
    assert out[1][:2] == (4.95, None)


def test_origin_uses_only_events_known_at_origin():
    req = requests(sched=[0, 1, 2, 5], acc=[.1, 1.1, 2.1, 5.1], first=[.3, 3.5, 4.0, 5.4], done=[6, 9, 10, 12])
    run = run_for(req)
    running, queue, reserved = run.origin(3.)
    assert len(running) == 1 and len(queue) == 2 and reserved == 3.
    moved = requests(sched=[0, 1, 2, 5], acc=[.1, 1.1, 2.1, 5.9], first=[.3, 7.5, 8.0, 9.4], done=[6, 19, 30, 32])
    assert run_for(moved).origin(3.)[1:] == (queue, reserved) and run_for(moved).origin(3.)[0] == running
    # Remaining decode steps come from step counters received by the origin.
    assert running[0][0] == pytest.approx(gb.OUT - 1 - 10 * (3. - .3))


def test_bootstrap_and_nowcast_ignore_the_future():
    sched = np.arange(0., 100., .25)
    first = sched + 8.2
    req = requests(sched, sched + 8., first, sched + 11.)
    t = 70.
    undecided = req['acc'] > t
    changed = {k: v.copy() for k, v in req.items()}
    changed['cost'][undecided], changed['est_in'][undecided] = 99., 1.
    replayed = 0
    for seed in range(20):
        base = run_for(req).bootstrap_arrivals(t, np.random.default_rng(seed))
        assert run_for(changed).bootstrap_arrivals(t, np.random.default_rng(seed)) == base
        assert all(t <= a[1] < t + gb.HORIZON and a[0] == a[1] + 8. and a[2] == 1. for a in base)
        replayed += len(base)
    assert replayed > 0
    late = {k: v.copy() for k, v in req.items()}
    unobserved = first > t
    late['c_first'][unobserved], late['ttft'][unobserved] = np.inf, 0.
    assert ev.nowcast(late, t) == ev.nowcast(req, t)
    quick = requests(sched, sched + .05, sched + .2, sched + 3.)
    assert ev.nowcast(quick, t) == pytest.approx(((sched >= t - 60) & (sched < t - .5)).sum() / 59.5)


def test_causal_pre_uses_only_acceptances_before_cut():
    req = requests([0, 1, 2, 3], [.1, 1.1, 2.9, 3.9], [.3, 1.3, 3.1, 4.1], [5, 6, 7, 8])
    assert run_for(req, cut=1.5).pre == pytest.approx(.1)
    assert run_for(req).pre == pytest.approx(.5)


def test_step_fit_recovers_coefficients():
    rng = np.random.default_rng(1)
    steps, gen, prompt = rng.uniform(20, 80, 50), rng.uniform(100, 2000, 50), rng.uniform(0, 9000, 50)
    dt = 0.009 * steps + 0.0003 * gen + 0.00005 * prompt
    fit = gb.fit_step_time([dict(steps=steps, gen=gen, prompt=prompt, dt=dt)])
    assert fit == pytest.approx((0.009, 0.0003, 0.00005))


def test_step_intervals_keep_busy_one_second_windows():
    scrapes = dict(t=np.array([0., 4., 5., 6., 8., 9.]), steps=np.array([0., 0., 10., 10., 20., 30.]),
                   prompt=np.zeros(6), gen=np.zeros(6), recv=np.array([0., 4., 5., 6., 8., 9.]) + .1)
    assert list(gb.step_intervals(scrapes)['t']) == [5., 9.]


def test_isotonic_pools_violators_and_averages_ties():
    x, y = gb.isotonic([1, 2, 2, 3, 4], [1., 3., 1., 1., 5.])
    assert list(x) == [1, 2, 3, 4]
    assert list(y) == pytest.approx([1., 5 / 3, 5 / 3, 5.])
    x, y = gb.isotonic([1, 2, 3], [1., 2., 3.])
    assert list(y) == [1., 2., 3.]


def test_overhead_is_monotone_and_expected_met_uses_it():
    n = np.repeat(np.arange(1., 41.), 5)
    residual = .01 * n + np.tile([-.02, -.01, 0., .01, .02], 40)
    overhead = gb.Overhead(n, residual)
    grid = np.linspace(0, 60, 61)
    assert np.all(np.diff(overhead.mean(grid)) >= 0) and np.all(np.diff(overhead.sigma(grid)) >= 0)
    out = [gb.Simulated(0., a, f, n, str(i)) for i, (a, f, n) in enumerate([
        (True, .1, 1.), (True, .1, 40.), (False, math.nan, math.nan), (True, math.inf, 3.)])]
    assert gb.expected_met(out) == pytest.approx(2.)
    assert 1. < gb.expected_met(out, overhead) < 2.
    assert gb.expected_met(out, overhead, bias=5.) == pytest.approx(0.)


def test_online_bias_ignores_unobserved_first_tokens():
    overhead = gb.Overhead([1., 2.], [0., 0.])
    rows = np.array([[50., 50.1, .1, .6, 1., 50.7, 8., 55.8], [55., 55.1, .1, .6, 1., 70., 8., 60.8]])
    assert gb.online_bias(rows, overhead, 60.) == pytest.approx(.5 / 21)
    assert gb.online_bias(rows[1:], overhead, 60.) == 0.


def test_block_bootstrap_and_scores():
    runs = np.repeat(np.array(['a', 'b']), 30)
    t = np.tile(np.arange(30.), 2)
    rng = np.random.default_rng(0)
    zero = ev.paired_block_bootstrap(np.zeros(60), runs, t, rng, 200, 10.)
    assert zero['delta_mae'] == 0 and zero['ci95'] == [0., 0.]
    shift = ev.paired_block_bootstrap(np.full(60, -.2), runs, t, rng, 200, 10.)
    assert shift['ci95'] == pytest.approx([-.2, -.2]) and shift['share_below_zero'] == 1.
    actual = np.tile(np.arange(30.) % 3, 2)
    s = ev.score(actual + 1., actual, runs)
    assert s['mae'] == 1. and s['bias'] == 1. and s['within_run_corr'] == pytest.approx(1.)


def test_near_change_window():
    run = run_for(requests([0], [.1], [.2], [1.]))
    assert ev.changes(run) == [60.]
    assert [t for t in range(40, 80) if ev.near_change(run, t)] == list(range(55, 71))


def write_capture(root, cost=2.):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'manifest.json').write_text(json.dumps(dict(measurement_start=100., created_at=1.)))
    (root / 'replay_plan.json').write_text(json.dumps(dict(duration_seconds=300., budget_schedule=[
        dict(at_seconds=0., budget=32.)])))
    events = [dict(request_id='a', event='received', monotonic=101.), dict(
        request_id='a', event='accepted', monotonic=101.1, request_cost=cost, estimated_input_tokens=50),
        dict(request_id='a', event='first_content', monotonic=101.3),
        dict(request_id='a', event='completed', monotonic=103.), dict(request_id='b', event='received', monotonic=102.),
        dict(request_id='b', event='rejected', monotonic=102.01, request_cost=5., estimated_input_tokens=70)]
    (root / 'gateway.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n')
    (root / 'client.json').write_text(json.dumps(dict(requests=[
        dict(request_id='a', workload='trace', scheduled_at=101., first_content_at=101.4, success=True,
             stream_complete=True, trace_id='x'),
        dict(request_id='b', workload='trace', scheduled_at=102., success=False, trace_id='y'),
        dict(request_id='w', workload='warmup', scheduled_at=99., trace_id='z')])))
    metrics = ('vllm:iteration_tokens_total_count{engine="0"} %s\nvllm:prompt_tokens_total{engine="0"} 50\n'
               'vllm:time_to_first_token_seconds_sum{engine="0"} %s\n'
               'vllm:time_to_first_token_seconds_count{engine="0"} %s\n')
    scrapes = [dict(sent_at=100. + t, received_at=100.05 + t, raw_metrics=metrics % (10 * t, .2 * t, t))
               for t in (0., 5., 300.)] + [dict(sent_at=150., raw_metrics='')]
    (root / 'engine_observations.jsonl').write_text('\n'.join(map(json.dumps, scrapes)) + '\n')


def test_capture_parsing(tmp_path):
    write_capture(tmp_path / 'run-a')
    meta, req = gb.load_capture(tmp_path / 'run-a')
    assert meta['schedule'][0]['budget'] == 32. and meta['created'] == 1.
    assert list(req['admitted']) == [True, False] and list(req['met']) == [True, False]
    assert req['ttft'][0] == pytest.approx(.4) and req['cost'][1] == 5. and req['est_in'][1] == 70.
    scrapes = gb.load_scrapes(tmp_path / 'run-a', meta['start'])
    assert list(scrapes['t']) == [0., 5., 50., 300.]
    assert list(scrapes['valid']) == [1., 1., 0., 1.]
    assert scrapes['steps'][3] == 3000.
    assert math.isnan(scrapes['gen'][0])
    summary = ev.engine_summary(meta, req, scrapes, start=1.)
    assert summary['time_to_first_token_seconds'] == pytest.approx(.2)
    assert summary['gateway_ttft'] == pytest.approx(.2) and summary['created_at'] == 1.


def test_find_run_checks_archived_hashes(tmp_path):
    write_capture(tmp_path / 'bundle' / 'nested' / 'run-a')
    root = tmp_path / 'bundle' / 'nested' / 'run-a'
    hashes = {f: ev.digest((root / f).read_bytes()) for f in ev.RAW}
    assert ev.find_run([tmp_path / 'bundle'], 'run-a', hashes) == root.resolve()
    with pytest.raises(ValueError, match='differs'):
        ev.find_run([tmp_path / 'bundle'], 'run-a', {**hashes, 'client.json': '0' * 64})
    with pytest.raises(ValueError, match='expected one'):
        ev.find_run([tmp_path / 'bundle'], 'run-b', hashes)


def test_clean_makes_json_safe():
    value = ev.clean(dict(a=np.float64(math.nan), b=np.array([1, 2]), c=(np.bool_(True), math.inf), d=None))
    assert json.loads(json.dumps(value, allow_nan=False)) == dict(a=None, b=[1, 2], c=[True, None], d=None)


def test_empty_start_has_no_state():
    assert gb.EmptyStart(3.).origin(10.) == ([], deque(), 0.) and gb.EmptyStart(3.).cap(10.) == 3.


def test_rejected_cost_is_unavailable_until_rejection_decision():
    sched = np.arange(60., 70., .1)
    req = requests(sched, np.full(len(sched), np.nan), np.full(len(sched), np.nan), np.full(len(sched), np.nan))
    req['rejected_at'][:] = 80.
    changed = {k: v.copy() for k, v in req.items()}
    changed['cost'][:] = 99.
    changed['est_in'][:] = 9000.
    for seed in range(20):
        meta = dict(schedule=[dict(at_seconds=0., budget=8.)])
        scrapes = dict(t=np.array([0., 1.]), recv=np.array([.1, 1.1]), steps=np.array([0., 10.]))
        a = gb.Run('a', meta, req, scrapes, pre=.1)
        b = gb.Run('a', meta, changed, scrapes, pre=.1)
        assert a.bootstrap_arrivals(70., np.random.default_rng(seed)) == []
        assert b.bootstrap_arrivals(70., np.random.default_rng(seed)) == []
    assert np.all(a.decision == 80.)
    assert any(a.bootstrap_arrivals(90., np.random.default_rng(seed)) for seed in range(20))


def test_duplicate_scheduled_times_keep_distinct_request_labels():
    req = requests([5.1, 5.1], [5.11, 5.12], [5.2, 6.], [8., 9.])
    residual, sims = gb.residuals(run_for(req), COEF, [5.])
    assert {o.request_id for o in sims[5.]} == {'0', '1'}
    assert residual[:, 3] == pytest.approx([.1, .9])
    assert residual.shape == (2, 8)


def test_tied_arrivals_preserve_input_order_instead_of_sorting_cost():
    arrivals = gb.ordered_arrivals([arrival(0., cost=1.5, rid='first'), arrival(0., cost=1., rid='second')])
    out = gb.simulate(gb.EmptyStart(2.), 0., arrivals, COEF)
    assert {o.request_id: o.admitted for o in out} == {'first': True, 'second': False}
    with pytest.raises(ValueError, match='Duplicate'):
        gb.simulate(gb.EmptyStart(2.), 0., [arrival(0.), arrival(0.)], COEF)


def test_capture_rejects_duplicate_client_identity(tmp_path):
    write_capture(tmp_path)
    path = tmp_path / 'client.json'
    data = json.loads(path.read_text())
    data['requests'].append(data['requests'][0].copy())
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='duplicate client request IDs'):
        gb.load_capture(tmp_path)


def test_scrape_sent_before_cut_but_received_or_parsed_after_cut_is_excluded(tmp_path):
    write_capture(tmp_path)
    path = tmp_path / 'engine_observations.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()][:1]
    base = events[0]
    # A response arrives before cut, but parsing finishes after it.
    events = [{**base, 'sent_at': 103. + t, 'received_at': 103.01 + t, 'finished_at': 103.02 + t,
               'raw_metrics': base['raw_metrics'].replace(' 0.0', ' ' + str(10. * t))}
              for t in (0., 1., 2.)]
    events[-1]['finished_at'] = 106.
    path.write_text('\n'.join(map(json.dumps, events)) + '\n')
    s = gb.load_scrapes(tmp_path, 100.)
    s['prompt'] = np.array([0., 10., 20.])
    s['gen'] = np.array([0., 10., 20.])
    assert list(gb.step_intervals(s, end=5.5)['t']) == [4.]


@pytest.mark.parametrize('problem', ['reset', 'missing', 'failed', 'epoch', 'identity'])
def test_step_intervals_do_not_bridge_bad_counter_samples(problem):
    t = np.arange(3., 9.)
    s = dict(t=t, recv=t + .01, steps=t * 10., gen=t * 20., prompt=t * 30., valid=np.ones(6),
             epoch=np.zeros(6), identity=np.full(6, 100.))
    if problem == 'reset':
        s['steps'][2:] -= 100.
    elif problem == 'missing':
        s['prompt'][2] = np.nan
    elif problem == 'failed':
        s['valid'][2] = 0.
    elif problem == 'epoch':
        s['epoch'][2:] = 1.
    else:
        s['identity'][2:] = 200.
    interval = gb.step_intervals(s)
    assert 5. not in interval['t']
    if problem in ('missing', 'failed'):
        assert 6. not in interval['t']


def test_zero_spread_calibration_has_finite_predictions():
    overhead = gb.Overhead([1., 2.], [0., 0.])
    assert np.all(overhead.sigma([1., 2.]) == gb.SIGMA_MIN)
    assert gb.expected_met([gb.Simulated(0., True, .5, 1., 'a')], overhead) == pytest.approx(.5)
    with pytest.raises(ValueError, match='Nonphysical'):
        gb.simulate(gb.EmptyStart(2.), 0., [arrival(0.)], (0., 0., 0.))


def test_online_bias_waits_for_entire_historical_window():
    overhead = gb.Overhead([1., 2.], [0., 0.])
    res = np.array([[55., 55.1, .1, .6, 1., 55.7, 8., 60.8]])
    assert gb.online_bias(res, overhead, 60.) == 0.


def calibration_fixture():
    t = np.arange(0., 33.)
    steps = 30. + 5. * np.sin(t[1:])
    gen = 100. + 20. * np.cos(t[1:])
    prompt = (1. - COEF[0] * steps - COEF[1] * gen) / COEF[2]
    s = dict(t=t, recv=t + .1, steps=np.r_[0., steps.cumsum()], gen=np.r_[0., gen.cumsum()],
             prompt=np.r_[0., prompt.cumsum()], valid=np.ones(len(t)))
    sched = np.arange(0., 32., .25)
    r = requests(sched, sched + .1, sched + .3, sched + 3.)
    # An arrival inside a training window whose cost is not known until after cut.
    i = np.flatnonzero(sched == 12.)[0]
    r['acc'][i], r['first'][i], r['c_first'][i], r['done'][i], r['ttft'][i] = 21., 22., 22., 25., 10.
    m = dict(schedule=[dict(at_seconds=0., budget=8.)])
    return {'synthetic': (m, r, s)}


def test_training_parameters_are_invariant_to_post_cut_information():
    import copy
    data = calibration_fixture()
    modified = copy.deepcopy(data)
    _, r, s = modified['synthetic']
    unknown = r['acc'] > 20.
    r['cost'][unknown] = 999.
    r['est_in'][unknown] = 7000.
    for key in ('first', 'done', 'c_first'):
        r[key][r[key] > 20.] += 500.
    r['ttft'][r['c_first'] > 20.] = 888.
    r['success'][r['done'] > 20.] = False
    for key in ('steps', 'gen', 'prompt'):
        s[key][s['recv'] > 20.] += 1e6
    a, _, oa, train_a = ev.calibrate(data, cut=20.)
    b, _, ob, train_b = ev.calibrate(modified, cut=20.)
    assert a == pytest.approx(b)
    np.testing.assert_array_equal(train_a, train_b)
    assert train_a[:, 7].max() <= 20.
    for key in ('knots', 'values', 'sigma_knots', 'sigma_values'):
        np.testing.assert_array_equal(getattr(oa, key), getattr(ob, key))


def test_origin_with_unknown_first_token_keeps_request_in_prefill_queue():
    req = requests([0.], [.1], [np.nan], [np.nan])
    running, queue, reserved = run_for(req).origin(1.)
    assert running == [] and len(queue) == 1 and reserved == 1.


def test_admission_count_includes_accepted_requests_without_first_content():
    sim = [gb.Simulated(0., True, math.inf, 1., 'waiting'),
           gb.Simulated(0., True, .1, 1., 'fast'),
           gb.Simulated(0., False, math.nan, math.nan, 'rejected'),
           gb.Simulated(0., None, math.nan, math.nan, 'unreached')]
    rates = ev.simulation_rates(sim, None, 2.)
    assert rates['admitted'] == 1.
    assert rates['goodput_step_only'] == .5
    assert rates['goodput'] == pytest.approx(.5)
    assert rates['unreached_requests'] == 1
