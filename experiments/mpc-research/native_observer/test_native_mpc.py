from dataclasses import replace
from itertools import product

import pytest

from mpc_lab.contracts import CostSpec, Identity, RequestMark, Snapshot
from native_mpc import (EOSCell, Forecast, NativeModel, Offer, Scenario, choose_plan,
                        rollout, scenarios, train_eos_cell)
from state_sample import ServingObservation, WorkObservation


def model(**changes):
    values = dict(cost_spec=CostSpec(intercept=4, input_per_token=0, output_per_token=0),
                  theta_ms=(20., 0., 0.), prefill_overhead_s=.001,
                  prefill_s_per_token=.0001,
                  cells=(EOSCell("short", 10, 8, 64, (16, 24, 32)),),
                  max_decode=32, max_prefill_ktokens=100,
                  decode_source="synthetic-test-only", prefill_source="required-test-coefficients",
                  eos_source="synthetic-train", observer_sha256="observer",
                  training_source_sha256="fixture-not-native-evidence", train_cutoff="fixture")
    values.update(changes)
    return NativeModel(**values)


def observation(m, work=(), budget=16):
    marks = tuple(RequestMark(w.request_id, w.charged_cost, w.cost_prompt_tokens,
                              w.max_output_tokens, 9., 9.) for w in work)
    ident = Identity("native-proxy", "unaccepted", "support", m.cost_spec.fingerprint, "source")
    gateway = Snapshot("gateway", "clock", 10., budget, 0, marks, ident)
    return ServingObservation(gateway, "native", "observer", 10.01, 10.02,
                              tuple(work), len(work), 0, 0, .1)


def forecast(rate=30.):
    return Forecast(10., "clock", "past-offers-only", (rate, rate, rate), (1.,))


def choose(m, obs=None, f=None, **kwargs):
    return choose_plan(m, obs or observation(m), f or forecast(),
                       now=10.03, clock_domain="clock", **kwargs)


def test_reservation_release_once_and_cap_decrease_never_evicts():
    m = model(cost_spec=CostSpec(intercept=8, input_per_token=0, output_per_token=0))
    work = tuple(WorkObservation(str(i), 8., 8, 10, 1, 64, 9.) for i in range(3))
    obs = observation(m, work, budget=32)
    scenario = Scenario((Offer(0., 0, 16), Offer(1., 0, 16)), .5)
    result = rollout(m, obs, scenario, (16, 16, 16))
    assert result.reserved_trace[:4] == ((0., 24.), (.46, 16.), (.46, 8.), (.46, 0.))
    assert result.rejected == 1 and result.admitted == 1
    assert result.terminal == 4 and result.good == 1  # old scheduled times are unknown
    assert result.reserved_trace[-1][1] == 0


def test_physical_prompt_never_reprices_charged_cost():
    cost = CostSpec(intercept=0, input_per_token=1, output_per_token=0)
    m = model(cost_spec=cost, cells=(EOSCell("mismatch", 100, 8, 64, (16,)),))
    work = (WorkObservation("old", 8., 8, 100, 1, 64, 9.),)
    obs = observation(m, work)
    decision = choose(m, obs, forecast(1), clock=lambda: 0.)
    assert decision.reason == "research_only_unqualified"
    result = rollout(m, obs, Scenario((Offer(0., 0, 16),), .5), (16, 16, 16))
    assert result.admitted == 1 and result.reserved_trace[1] == (0., 16.)
    invalid = replace(obs, observations=(replace(work[0], charged_cost=100.),))
    assert choose(m, invalid).budget == 16
    assert choose(m, invalid).reason == "work_cost_or_progress_mismatch"


def test_workload_changes_binding_action_and_all_plans_share_scenarios():
    short = model(cells=(EOSCell("short", 10, 8, 64, (16,)),))
    long = replace(short, cells=(EOSCell("long", 1000, 8, 64, (16,)),))
    small = choose(short, clock=lambda: 0.)
    large = choose(long, clock=lambda: 0.)
    assert small.budget == 32 and large.budget == 16
    assert len(small.plan_scores) == len(large.plan_scores) == 8
    shared = scenarios(short, forecast())
    hashes = {tuple(rollout(short, observation(short), s, plan).scenario_hash for s in shared)
              for plan in product((16, 32), repeat=3)}
    assert hashes == {small.scenario_hashes}
    assert max(s for _, s in small.plan_scores) > min(s for _, s in small.plan_scores)


def test_offered_cohort_rejections_and_complete_drain_not_last_slot_reward():
    m = model()
    offers = tuple(Offer(4.9, 0, 16) for _ in range(10))
    r = rollout(m, observation(m), Scenario(offers, .5), (32, 32, 32))
    assert r.offered == 10 and r.rejected == 6 and r.good == 4
    assert r.terminal == r.admitted == 4 and r.reserved_trace[-1][0] > 5
    rejecting = replace(m, cost_spec=CostSpec(intercept=40, input_per_token=0, output_per_token=0))
    r = rollout(rejecting, observation(rejecting), Scenario(offers, .5), (32, 32, 32))
    assert r.offered == r.rejected == 10 and r.good == 0


def test_composite_requires_request_mean_tpot_even_when_ttft_passes():
    m = model(theta_ms=(60., 0., 0.))
    s = Scenario((Offer(0., 0, 16),), .5)
    r = rollout(m, observation(m), s, (32, 32, 32))
    assert r.admitted == r.terminal == 1 and r.good == 0
    one_token = rollout(m, observation(m), Scenario((Offer(0., 0, 1),), .5), (16, 16, 16))
    assert one_token.good == 1  # no inter-token intervals: explicitly defined zero


def test_future_prefill_can_spoil_old_started_requests_composite_good():
    m = model(theta_ms=(20., 0., 100.), prefill_s_per_token=.00015,
              cells=(EOSCell("long", 2000, 8, 64, (8,)),))
    work = (WorkObservation("old", 4., 8, 2000, 1, 64, 9.9),)
    obs = observation(m, work)
    mark = replace(obs.gateway.requests[0], first_at=9.99, scheduled_at=9.9,
                   received_at=9.9, admitted_at=9.9)
    obs = replace(obs, gateway=replace(obs.gateway, requests=(mark,)))
    isolated = rollout(m, obs, Scenario((), .5), (16, 16, 16))
    interfered = rollout(m, obs, Scenario((Offer(0., 0, 8),), .5), (16, 16, 16))
    assert isolated.good == isolated.old_good == 1 and isolated.terminal == 1
    assert interfered.old_good == 0 and interfered.future_good == 1 and interfered.terminal == 2


def test_train_eos_filters_future_and_conditions_only_on_observed_survival():
    records = [dict(split="train", available_at=t, output_tokens=n)
               for t, n in ((1, 10), (2, 20), (4, 60))]
    c = train_eos_cell("short", 10, 8, 64, records, cutoff=3)
    assert c.train_outputs == (10, 20) and c.eos(.25, observed=10) == 20
    with pytest.raises(ValueError, match="unknown_conditional"):
        c.eos(.5, observed=20)
    with pytest.raises(ValueError, match="non_train"):
        train_eos_cell("short", 10, 8, 64, [dict(split="test", available_at=1, output_tokens=1)],
                       cutoff=3)


def test_deadline_fallback_and_no_normal_actuation_authority():
    times = iter((0., .101, .102))
    d = choose(model(), clock=lambda: next(times))
    assert d.actions == (16, 16, 16) and d.deadline_exceeded and d.score is None
    assert not d.actuation_authority and not d.native_calibration_accepted and not d.performance_accepted
    assert d.latency_s >= .1 and d.model_identity
    ticks = iter(i * .00002 for i in range(10000))
    interrupted = choose(model(), clock=lambda: next(ticks))
    assert interrupted.deadline_exceeded and interrupted.actions == (16, 16, 16)
    assert interrupted.score is None and 0 < len(interrupted.plan_scores) < 8


@pytest.mark.parametrize("failure", ["unknown", "stale", "future", "support", "nonfinite", "drain"])
def test_incomplete_stale_unsupported_nonfinite_and_undrained_fallback(failure):
    m, f = model(), forecast()
    obs = observation(m)
    if failure == "unknown":
        obs = observation(m, (WorkObservation("old", 4., 8, None, None, 64, 9.),))
    elif failure == "stale":
        obs = replace(obs, native_received_at=9.)
    elif failure == "future":
        f = replace(f, observed_through=11.)
    elif failure == "support":
        m = replace(m, max_decode=1)
    elif failure == "nonfinite":
        f = replace(f, rates=(float("nan"), 30., 30.))
    else:
        m = replace(m, theta_ms=(100000., 0., 0.))
    d = choose(m, obs, f, clock=lambda: 0.)
    assert d.actions == (16, 16, 16) and d.score is None
    assert d.reason != "research_only_unqualified"


def test_ties_keep_current_cap_and_identity_binds_physics_cost_and_sources():
    m = model()
    d = choose(m, observation(m, budget=32), forecast(0), clock=lambda: 0.)
    assert d.actions == (32, 32, 32)
    variants = (replace(m, prefill_overhead_s=.002), replace(m, eos_source="other-train"),
                replace(m, cells=(replace(m.cells[0], train_outputs=(12,)),)),
                replace(m, cost_spec=replace(m.cost_spec, intercept=5)))
    assert all(v.identity != m.identity for v in variants)
    assert m.spec()["rules"]["native_calibration_accepted"] is False
    with pytest.raises(ValueError, match="required_native_source"):
        replace(m, prefill_source="")
