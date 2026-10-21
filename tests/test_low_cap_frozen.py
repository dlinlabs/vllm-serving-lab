import copy
import gzip
import json
import math

import numpy as np
import pytest

import evaluate_low_cap_frozen as lf
import greybox_simulator as gb
from test_greybox_simulator import write_capture


def example_run():
    sched = np.arange(0., 100., .25)
    n = len(sched)
    r = dict(sched=sched, recv=sched + .01, acc=sched + .02, first=sched + .15,
             done=sched + 2., cost=np.ones(n), est_in=np.full(n, 100.),
             success=np.ones(n, bool), ttft=np.full(n, .15), c_first=sched + .15,
             trace_id=np.array([str(i) for i in range(n)], object),
             request_id=np.array([str(i) for i in range(n)], object),
             rejected_at=np.full(n, np.nan), admitted=np.ones(n, bool), met=np.ones(n, bool))
    t = np.arange(0., 101.)
    s = dict(t=t, recv=t + .1, steps=t * 50.)
    meta = dict(schedule=[dict(at_seconds=0., budget=16.), dict(at_seconds=62., budget=32.)])
    return gb.Run('synthetic', meta, r, s, pre=.005)


def test_frozen_arrays_load_without_any_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('New data must not fit parameters')
    monkeypatch.setattr(gb, 'fit_step_time', forbidden)
    monkeypatch.setattr(gb.Overhead, '__init__', forbidden)
    coef, overhead, ref, archive = lf.frozen_parameters()
    assert tuple(coef) == tuple(archive['step_coefficients'][k] for k in
                                ('step_s', 'per_decode_token_s', 'per_prompt_token_s'))
    np.testing.assert_array_equal(overhead.values, archive['overhead']['mean'])
    np.testing.assert_array_equal(overhead.sigma_values, archive['overhead']['sigma'])
    assert ref['pre_s'] == .00545710029109614


def test_frozen_source_or_archive_mutation_is_rejected(tmp_path, monkeypatch):
    calibration = json.loads(lf.CALIBRATION.read_text())
    calibration['source_report_sha256'] = '0' * 64
    p = tmp_path / 'calibration.json'
    p.write_text(json.dumps(calibration))
    monkeypatch.setattr(lf, 'CALIBRATION', p)
    with pytest.raises(ValueError, match='provenance'):
        lf.frozen_parameters()


def test_future_outcomes_and_undecided_costs_do_not_change_forecast():
    base = example_run()
    changed = copy.deepcopy(base)
    future = changed.req['sched'] >= 60.
    for k, value in [('cost', 100.), ('est_in', 8000.), ('first', 999.),
                     ('done', 999.), ('ttft', 0.), ('c_first', 999.)]:
        changed.req[k][future] = value
    changed.req['success'][future] = False
    # Rebuild the cached admitted table as an adversarial full-record change.
    changed = gb.Run('changed', changed.meta, changed.req, changed.scrapes, pre=base.pre)
    coef = (.01, .001, .00005)
    overhead = gb.Overhead([1., 2.], [.01, .02])
    past, _ = gb.residuals(base, coef, np.arange(5., 55.))
    after, _ = gb.residuals(changed, coef, np.arange(5., 55.))
    np.testing.assert_allclose(past, after)
    assert lf.forecast(base, 60., coef, overhead, past) == lf.forecast(changed, 60., coef, overhead, after)


def test_candidate_caps_share_draws_and_do_not_use_future_plan(monkeypatch):
    base = example_run()
    other = gb.Run('other', dict(schedule=[dict(at_seconds=0., budget=16.)]),
                   base.req, base.scrapes, pre=base.pre)
    coef, overhead, _, _ = lf.frozen_parameters()
    residuals = np.empty((0, 8))
    p = lf.forecast(base, 60., coef, overhead, residuals)
    q = lf.forecast(other, 60., coef, overhead, residuals)
    assert p['cap16'] == q['cap16'] and p['cap32'] == q['cap32']
    assert p['online_bias_s'] == 0.


def test_target_is_successfully_completed_arrival_cohort():
    run = example_run()
    selected = (run.req['sched'] >= 60) & (run.req['sched'] < 65)
    assert lf.actual(run, 60.) == dict(goodput=4., admission=4., offered=20)
    run.req['met'][selected] = False
    assert lf.actual(run, 60.) == dict(goodput=0., admission=4., offered=20)


def test_matched_gap_is_factual_not_same_state_counterfactual():
    captures = [dict(name=s + str(rep), definition=dict(schedule=s, replicate=rep))
                for rep in (1, 2) for s in ('a', 'b')]
    rows = [dict(run=c['name'], t=t, cap=32 if c['definition']['schedule'] == 'a' else 16,
                 actual_goodput=4. if c['definition']['schedule'] == 'a' else 3.,
                 simulator=2. if c['definition']['schedule'] == 'a' else 3.)
            for c in captures for t in range(lf.START, lf.STOP + 1, lf.STRIDE)]
    gaps = lf.matched_phase_differences(rows, captures)
    assert len(gaps) == 96
    assert all(g['observed_gap'] == 1. and g['predicted_gap'] == -1. and g['same_direction'] is False for g in gaps)


def test_no_candidate_policy_value_is_reported():
    # A single-state hypothetical choice cannot be scored as an observed policy rollout.
    run = example_run()
    coef, overhead, _, _ = lf.frozen_parameters()
    result = lf.forecast(run, 60., coef, overhead, np.empty((0, 8)))
    assert 'cap16' in result and 'cap32' in result
    assert not any('gain' in key or 'regret' in key or 'policy_value' in key for key in result)


def test_audit_rejects_unqualified_suite_before_loading_captures(tmp_path):
    (tmp_path / 'suite.json').write_text(json.dumps(dict(valid=False)))
    (tmp_path / 'protocol.json').write_bytes((lf.ROOT / 'experiments/low-cap-switch-v1.json').read_bytes())
    with pytest.raises(ValueError, match='qualified'):
        lf.audit_suite(tmp_path)


def test_transport_failure_keeps_denominator_and_original_bytes(tmp_path):
    path = tmp_path / 'capture'
    write_capture(path)
    client = json.loads((path / 'client.json').read_text())
    client['requests'].append(dict(request_id=None, client_request_id='failed-client',
                                   workload='trace', scheduled_at=104., success=False,
                                   rejected=False, stream_complete=False, trace_id='failed-trace',
                                   expected_input_tokens=512, error_type='ReadError'))
    (path / 'client.json').write_text(json.dumps(client))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    meta, req, replacements = lf.load_capture_compatible(path)
    assert replacements == 1 and len(req['sched']) == 3
    assert req['request_id'][-1] == 'failed-client'
    assert not req['met'][-1] and not req['admitted'][-1]
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    client['requests'][-1]['success'] = True
    (path / 'client.json').write_text(json.dumps(client))
    with pytest.raises(ValueError, match='transport failure'):
        lf.load_capture_compatible(path)
