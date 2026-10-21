import copy
from collections import deque
import numpy as np
import pytest
import greybox_simulator as gb
import iterate_state_model as it
import state_model as sm
from test_low_cap_frozen import example_run


@pytest.mark.parametrize('seed', range(12))
def test_instrumentation_preserves_frozen_mechanics(seed):
    rng = np.random.default_rng(seed)
    at = np.sort(rng.uniform(0, 5, 80))
    arrivals = [gb.Arrival(t, t, float(rng.uniform(.5, 3)), float(rng.integers(20, 8192)), str(i)) for i, t in enumerate(at)]
    coef = (.01, .001, .00005)
    old = gb.simulate(gb.EmptyStart(16), 0., arrivals, coef)
    new, info = sm.simulate(gb.EmptyStart(16), 0., arrivals, coef)
    assert [x[:2] + x[4:] for x in new] == [x[:2] + x[4:] for x in old]
    np.testing.assert_allclose([x[2:4] for x in new], [x[2:4] for x in old], equal_nan=True)
    assert all(x.request_id in info for x in new if x.admitted)


def test_parity_with_existing_work_and_cap_changes():
    class State:
        def origin(self, t):
            return [(30., 2.)], deque([[9000., 2., np.nan, np.nan, None]]), 4.
        def cap(self, t):
            return 8. if t < 2. else 4.
    arrivals = [gb.Arrival(t, t, 1., 800., str(i)) for i, t in enumerate([.1, .1, 1., 2., 3.])]
    old = gb.simulate(State(), 0., arrivals, (.01, .001, .00005))
    new, _ = sm.simulate(State(), 0., arrivals, (.01, .001, .00005))
    assert [x[:2] + x[4:] for x in new] == [x[:2] + x[4:] for x in old]
    np.testing.assert_allclose([x[2:4] for x in new], [x[2:4] for x in old], equal_nan=True)


def test_features_ignore_unobserved_future_events_and_scrapes():
    run = example_run(); run.scrapes['waiting'] = np.arange(len(run.scrapes['t']), dtype=float)
    changed = copy.deepcopy(run)
    for key in ('first', 'done', 'recv', 'acc'):
        changed.req[key][changed.req[key] > 60.] += 1000.
    changed.req['cost'][changed.decision > 60.] = 999.
    changed.req['est_in'][changed.decision > 60.] = 99999.
    changed.scrapes['waiting'][changed.scrapes['recv'] > 60.] = 9999.
    assert sm.observed_state(run, 60.) == sm.observed_state(changed, 60.)
    assert sm.observed_state(run, 60.)['engine_waiting'] == 59.


def test_prefix_fit_uses_only_fully_available_intervals():
    rng = np.random.default_rng(3); t = np.arange(0., 101.)
    s = dict(t=t, recv=t + .1, steps=np.cumsum(rng.uniform(20, 40, len(t))),
             gen=np.cumsum(rng.uniform(100, 300, len(t))), prompt=np.cumsum(rng.uniform(100, 1000, len(t))))
    changed = {k: v.copy() for k, v in s.items()}
    for k in ('steps', 'gen', 'prompt'):
        changed[k][s['recv'] > 60.] += 1e9
    a, b = (it.prefix_coefficients(v, (.01, .001, .00005)) for v in (s, changed))
    np.testing.assert_allclose(a['nnls'], b['nnls']); np.testing.assert_allclose(a['shrink'], b['shrink'])
    assert a['latest_available_s'] <= 60.


@pytest.mark.parametrize('factory', [sm.LogisticCalibration, sm.TreeCalibration])
def test_calibration_probabilities_are_monotone_in_margin(factory):
    rng = np.random.default_rng(4); x = rng.normal(size=(300, 9))
    y = (x[:, 0] + .4 * x[:, 2] > rng.normal(size=300)).astype(float)
    model = factory(range(9), 10.).fit(x, y)
    high = x.copy(); high[:, 0] += 1.
    p, q = model.predict(x), model.predict(high)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and (q >= p - 1e-12).all()
    before = model.parameters(); model.predict(high)
    assert model.parameters() == before


def test_rate_retains_all_draws_in_denominator():
    class Half:
        def predict(self, x):
            return np.full(len(x), .5)
    assert it.rate(Half(), [np.zeros((32, 9)), np.zeros((0, 9))]).tolist() == [.2, 0.]


@pytest.mark.parametrize('bad', [-1., 0., float('nan')])
def test_invalid_calibration_penalty_rejected(bad):
    with pytest.raises(ValueError, match='penalty'):
        sm.LogisticCalibration([0], bad)


def test_nonphysical_step_model_rejected():
    with pytest.raises(ValueError, match='Nonphysical'):
        sm.simulate(gb.EmptyStart(16), 0., [], (.01, -.001, .00005))


def test_grouped_outer_labels_cannot_choose_model_or_penalty(monkeypatch):
    names = ['fixed-16', 'fixed-32', 'x-a-r1', 'x-a-r2', 'x-b-r1', 'x-b-r2']
    data = {n: dict(points=dict(x=np.zeros((20, 21)), y=np.full(20, i / 10)),
                    draws=[np.zeros((10, 21))], rows=[dict(actual=i / 10)]) for i, n in enumerate(names)}
    class Dummy:
        def __init__(self, train, p):
            self.p = np.mean([data[n]['points']['y'].mean() for n in train]); self.train = train
        def predict(self, x):
            return np.full(len(x), self.p)
        def parameters(self):
            return dict(p=self.p, train=self.train)
    monkeypatch.setattr(it, 'fit_calibration', lambda data_arg, train, kind, penalty: Dummy(train, penalty))
    base, folds = it.nested_calibration(data, 'state', grouped=True)
    for n in ['x-a-r1', 'x-a-r2']:
        data[n]['points']['y'][:] = .99; data[n]['rows'][0]['actual'] = .99
    after, changed_folds = it.nested_calibration(data, 'state', grouped=True)
    for n in ['x-a-r1', 'x-a-r2']:
        np.testing.assert_array_equal(base[n], after[n])
    a = next(f for f in folds if 'x-a-r1' in f['held_runs'])
    b = next(f for f in changed_folds if 'x-a-r1' in f['held_runs'])
    assert a == b and not any('-a-' in n for n in a['training_runs'])
