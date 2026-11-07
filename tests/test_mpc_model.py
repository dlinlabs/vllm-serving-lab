import math
import time
from dataclasses import replace

import pytest

from mpc_lab.contracts import (
    CostSpec,
    FutureOffer,
    Identity,
    ModelParameters,
    Plan,
    PlanningConfig,
    RequestMark,
    RolloutError,
    Scenario,
    Snapshot,
)
from mpc_lab.model import generate_scenarios, rollout

# Synthetic dyadic cost spec: cost == prompt_tokens / 1024 exactly, so every old
# mark's sealed ledger cost is formula-consistent (no weakened check).
UNIT = 1.0 / 1024
P = ModelParameters(cost_spec=CostSpec(intercept=0.0, input_per_token=UNIT, output_per_token=0.0,
                                       minimum=UNIT, estimator_version="test-dyadic-v1"))
CFG = PlanningConfig()
T0 = 100.0


def ident(params=P):
    return Identity("m", "c", "s", params.cost_spec.fingerprint, "i")


def snap(requests=(), budget=16.0, **kw):
    return Snapshot("run", "mono", T0, budget, 0, tuple(requests), ident(), **kw)


def mark(rid, cost, max_tokens=128, admitted=99.0, first=None, scheduled=None):
    prompt = round(cost / UNIT)
    assert P.cost_spec.estimate(prompt, max_tokens) == cost  # exact ledger fixture
    return RequestMark(rid, cost, prompt, max_tokens, admitted - 0.01, admitted, first, scheduled)


def offer(rid, at, prompt=100, m=128, out=None, success=True, ingress=0.0, egress=0.0):
    return FutureOffer(rid, at, prompt, m, m if out is None else out, success, ingress, egress)


def plan(*actions, cont=16.0):
    return Plan(tuple(actions), 1.0, cont)


def by_id(result):
    return {o.request_id: o for o in result.outcomes}


def test_exact_boundary_admits_and_one_more_rejects():
    c = P.cost_spec.estimate(100, 128)
    # Old reservation leaves exactly c of room under 16 (no release before arrivals).
    old = mark("old", 16.0 - c, first=99.9)
    sc = Scenario((offer("a", 0.0), offer("b", 0.0)))
    res = rollout(snap([old]), plan(16.0), sc, P, CFG)
    o = by_id(res)
    assert o["a"].accepted and not o["b"].accepted
    assert max(r for _, r in res.reserved_trace) <= 16.0 + 1e-12


def test_cap_decrease_never_evicts_and_release_exactly_once():
    marks = [mark(f"o{i}", 4.0, first=99.95) for i in range(8)]  # R = 32
    res = rollout(snap(marks, budget=32.0), plan(16.0, 16.0, 16.0), Scenario(()), P, CFG)
    o = by_id(res)
    assert res.reserved_trace[0] == (T0, 32.0)
    # No release before each old request's own terminal event.
    releases = [(t, r) for (t, r), (_, prev) in zip(res.reserved_trace[1:], res.reserved_trace) if r < prev]
    assert len(releases) == 8
    assert all(o[f"o{i}"].terminal_at is not None and o[f"o{i}"].terminal_at > T0 for i in range(8))
    assert res.reserved_trace[-1][1] == pytest.approx(0.0)


def test_decrease_blocks_new_admission_until_release():
    marks = [mark("big", 20.0, first=99.95)]
    sc = Scenario((offer("x", 0.1),))
    res = rollout(snap(marks, budget=32.0), plan(16.0), sc, P, CFG)
    assert not by_id(res)["x"].accepted
    res32 = rollout(snap(marks, budget=32.0), plan(32.0), sc, P, CFG)
    assert by_id(res32)["x"].accepted


def test_causal_action_change_affects_only_later_arrivals():
    marks = [mark("big", 20.0, max_tokens=4096, first=99.95)]  # streams beyond the test window
    sc = Scenario((offer("early", 0.5), offer("late", 1.5)))
    o = by_id(rollout(snap(marks), plan(16.0, 32.0), sc, P, CFG))
    assert not o["early"].accepted and o["late"].accepted
    assert o["big"].terminal_at is None or o["big"].terminal_at > T0 + 1.5


def test_old_cohort_late_failure_is_not_success_and_known_first_respected():
    m = mark("old", 2.0, admitted=99.86, first=99.9, scheduled=99.8)  # received 99.85
    ok = by_id(rollout(snap([m]), plan(16.0), Scenario(()), P, CFG))["old"]
    bad_res = rollout(snap([m]), plan(16.0), Scenario((), old_success=False), P, CFG)
    bad = by_id(bad_res)["old"]
    assert ok.first_at == pytest.approx(99.9) and ok.good() and ok.success is True
    assert bad.first_at == pytest.approx(99.9) and bad.success is False and not bad.good()
    assert bad_res.score_lower == 0.0 and bad_res.score_upper == 0.0


def test_old_scheduled_unknown_uses_explicit_ingress_and_egress_assumption():
    m = mark("old", 2.0, admitted=99.7, first=99.9)  # received 99.69
    near = by_id(rollout(snap([m]), plan(16.0), Scenario(()), P, CFG))["old"]
    far = by_id(rollout(snap([m]), plan(16.0), Scenario((), ingress_delay_s=0.2, egress_delay_s=0.1), P, CFG))["old"]
    assert near.scheduled_at == pytest.approx(99.69) and near.good()
    assert far.scheduled_at == pytest.approx(99.49) and far.first_at == pytest.approx(100.0)
    assert not far.good()


def test_new_offer_first_hit_then_failure_and_zero_output_failure():
    sc = Scenario((offer("f", 0.0, out=5, success=False), offer("z", 0.0, out=0, success=False)))
    o = by_id(rollout(snap(), plan(32.0), sc, P, CFG))
    assert o["f"].first_at is not None and o["f"].first_at - o["f"].scheduled_at <= 0.5
    assert o["f"].success is False and not o["f"].good()
    assert o["z"].first_at is None and o["z"].success is False and o["z"].terminal_at is not None


def test_zero_prompt_has_positive_work_and_completes():
    res = rollout(snap(), plan(16.0), Scenario((offer("z", 0.0, prompt=0, out=1),)), P, CFG)
    o = by_id(res)["z"]
    assert o.first_at == pytest.approx(T0 + P.first_overhead_s)
    assert o.terminal_at == o.first_at and o.good()
    assert res.event_count < 10


def test_only_window_offers_scored_but_tail_offers_interfere():
    cfg = replace(CFG, evaluation_window_s=1.0, tail_s=3.0)
    window_only = Scenario((offer("w", 1.5),))
    res = rollout(snap(), plan(16.0), window_only, P, cfg)
    assert res.outcomes == () and res.score_lower == 0.0
    # A tail offer is admitted and reserved, but is never an outcome.
    tail = Scenario((offer("w", 0.5), offer("tail", 1.2, prompt=15000)))
    r2 = rollout(snap(), plan(16.0), tail, P, cfg)
    assert [o.request_id for o in r2.outcomes] == ["w"]
    assert any(abs(t - (T0 + 1.2)) < 1e-9 for t, _ in r2.reserved_trace)
    # Interference: an unscored tail-period arrival delays a scored request's service.
    base = rollout(snap(), plan(32.0), Scenario((offer("w", 0.95),)), P, cfg)
    busy = rollout(snap(), plan(32.0, cont=32.0), Scenario((offer("w", 0.95), offer("tail", 1.0))), P, cfg)
    assert [o.request_id for o in busy.outcomes] == ["w"]
    assert by_id(busy)["w"].first_at == by_id(base)["w"].first_at
    assert by_id(busy)["w"].terminal_at > by_id(base)["w"].terminal_at


def test_unresolved_bounds_at_tail_end():
    cfg = replace(CFG, evaluation_window_s=1.0, tail_s=0.5)
    sc = Scenario((offer("long", 0.0, m=4096, out=4096),))
    res = rollout(snap(), plan(32.0), sc, P, cfg)
    o = by_id(res)["long"]
    assert o.first_at is not None and o.terminal_at is None and o.success is None
    assert res.score_lower == 0.0 and res.score_upper == 1.0
    failing = Scenario((offer("long", 0.0, m=4096, out=4096, success=False),))
    assert rollout(snap(), plan(32.0), failing, P, cfg).score_upper == 0.0


def test_rejected_offers_and_scores_match_good_outcomes():
    sc = generate_scenarios(snap(), arrival_rate=25.0, prompt_tokens=(200, 1500, 6000), max_tokens=(128,),
                            count=3, seed=1, duration_s=15.0)
    for s in sc:
        res = rollout(snap(), plan(16.0, 32.0, 16.0), s, P, CFG)
        assert res.score_lower == sum(o.good(CFG.slo_s) for o in res.outcomes)
        assert res.score_lower <= res.score_upper <= len(res.outcomes)
        assert all(r >= -1e-12 for _, r in res.reserved_trace)
        assert all(o.offset_s < CFG.evaluation_window_s for o in s.offers
                   if o.request_id in {x.request_id for x in res.outcomes})


def test_generate_scenarios_deterministic_and_past_only():
    s = snap([mark("old", 2.0)])
    kw = dict(arrival_rate=10.0, prompt_tokens=(10, 20), max_tokens=(64, 128), count=4, seed=7,
              duration_s=5.0, service_scales=(1.0, 1.5))
    a, b = generate_scenarios(s, **kw), generate_scenarios(s, **kw)
    assert a == b and len(a) == 4
    assert generate_scenarios(s, **{**kw, "seed": 8}) != a
    assert [x.service_scale for x in a] == [1.0, 1.5, 1.0, 1.5]
    for x in a:
        assert all(0 <= o.offset_s < 5.0 and o.output_tokens == o.max_tokens for o in x.offers)
        assert all(o.prompt_tokens in (10, 20) and o.max_tokens in (64, 128) for o in x.offers)
    # The API has no parameter for a realized future tape or realized output lengths.
    import inspect
    names = set(inspect.signature(generate_scenarios).parameters)
    assert not names & {"offers", "trace", "future", "outputs", "tape"}
    assert generate_scenarios(s, **{**kw, "arrival_rate": 0.0})[0].offers == ()


def test_rollout_deterministic_and_common_scenarios_across_plans():
    sc = generate_scenarios(snap(), arrival_rate=25.0, prompt_tokens=(500, 3000), max_tokens=(128,),
                            count=1, seed=3, duration_s=15.0)[0]
    r1 = rollout(snap(), plan(16.0, 16.0, 16.0), sc, P, CFG)
    assert r1 == rollout(snap(), plan(16.0, 16.0, 16.0), sc, P, CFG)
    r2 = rollout(snap(), plan(32.0, 32.0, 32.0), sc, P, CFG)
    assert {o.request_id for o in r1.outcomes} == {o.request_id for o in r2.outcomes}


@pytest.mark.parametrize("change,reason", [
    (lambda s, sc, p, c, pl: (s, sc, replace(p, first_overhead_s=0.0), c, pl), "invalid_first_overhead_s"),
    (lambda s, sc, p, c, pl: (s, sc, replace(p, decode_s_per_token=math.nan), c, pl), "invalid_decode_s_per_token"),
    (lambda s, sc, p, c, pl: (s, sc, p, c, plan(24.0)), "unsupported_plan_budget"),
    (lambda s, sc, p, c, pl: (replace(s, complete=False), sc, p, c, pl), "incomplete_snapshot"),
    (lambda s, sc, p, c, pl: (s, Scenario((offer("a", 0.0, m=128, out=129),)), p, c, pl), "output_exceeds_declared_max"),
    (lambda s, sc, p, c, pl: (s, Scenario((offer("a", 1.0), offer("b", 0.5))), p, c, pl), "unsorted_offers"),
    (lambda s, sc, p, c, pl: (s, Scenario((offer("a", 0.0, prompt=10**6),)), p, c, pl), "unsupported_prompt_tokens"),
    (lambda s, sc, p, c, pl: (s, Scenario((offer("a", 0.0, out=0, success=True),)), p, c, pl), "invalid_offer_success"),
    (lambda s, sc, p, c, pl: (s, Scenario((), service_scale=0.0), p, c, pl), "invalid_service_scale"),
    (lambda s, sc, p, c, pl: (replace(s, identity=Identity("m", "c", "s", "other", "i")), sc, p, c, pl),
     "cost_identity_mismatch"),
    (lambda s, sc, p, c, pl: (snap([mark("x", 1.0, admitted=101.0)]), sc, p, c, pl), "noncausal_request_times"),
    (lambda s, sc, p, c, pl: (s, Scenario((offer("old", 0.0),)), p, c, pl), "duplicate_request_id"),
    (lambda s, sc, p, c, pl: (snap([RequestMark("x", 1.5, 1024, 128, 98.99, 99.0)]), sc, p, c, pl),
     "ledger_cost_mismatch"),
])
def test_invalid_inputs_raise_stable_reasons(change, reason):
    args = change(snap([mark("old", 1.0)]), Scenario(()), P, CFG, plan(16.0))
    s, sc, p, c, pl = args
    with pytest.raises(RolloutError, match=f"^{reason}$"):
        rollout(s, pl, sc, p, c)


def test_limits_and_deadline_raise_instead_of_truncating():
    sc = generate_scenarios(snap(), arrival_rate=50.0, prompt_tokens=(10,), max_tokens=(16,),
                            count=1, seed=2, duration_s=15.0)[0]
    with pytest.raises(RolloutError, match="^event_limit$"):
        rollout(snap(), plan(32.0, cont=32.0), sc, P, replace(CFG, max_events=50))
    with pytest.raises(RolloutError, match="^active_limit$"):
        rollout(snap(), plan(32.0, cont=32.0), sc, P, replace(CFG, max_active_requests=2))
    with pytest.raises(RolloutError, match="^active_limit$"):
        rollout(snap([mark(f"o{i}", 1.0) for i in range(3)]), plan(16.0), Scenario(()), P,
                replace(CFG, max_active_requests=2))
    with pytest.raises(RolloutError, match="^deadline_exceeded$"):
        rollout(snap(), plan(32.0), sc, P, CFG, deadline=time.monotonic() - 1.0)


def test_generate_scenarios_rejects_invalid_inputs():
    for kw in ({"arrival_rate": -1.0}, {"prompt_tokens": ()}, {"max_tokens": (0,)}, {"count": 0},
               {"duration_s": 0.0}, {"service_scales": (0.0,)}):
        base = dict(arrival_rate=1.0, prompt_tokens=(1,), max_tokens=(1,), count=1, seed=0, duration_s=1.0)
        with pytest.raises(ValueError):
            generate_scenarios(snap(), **{**base, **kw})


def test_sealed_ledger_cost_must_match_claimed_formula():
    good = mark("old", 2.0)
    rollout(snap([good]), plan(16.0), Scenario(()), P, CFG)
    # Same marks under the default formula: cost_hash would claim a formula they violate.
    d = ModelParameters()
    s = Snapshot("run", "mono", T0, 16.0, 0, (good,), ident(d))
    with pytest.raises(RolloutError, match="^ledger_cost_mismatch$"):
        rollout(s, plan(16.0), Scenario(()), d, CFG)
    ok = replace(good, cost=d.cost_spec.estimate(good.prompt_tokens, good.max_tokens))
    rollout(replace(s, requests=(ok,)), plan(16.0), Scenario(()), d, CFG)


def test_client_terminal_includes_egress_and_release_stays_at_gateway():
    sc = Scenario((offer("one", 0.0, prompt=0, out=1, egress=0.2),))
    res = rollout(snap(), plan(16.0), sc, P, CFG)
    o = by_id(res)["one"]
    gateway_end = T0 + P.first_overhead_s
    assert o.first_at == pytest.approx(gateway_end + 0.2) and o.terminal_at == o.first_at
    assert o.terminal_at >= o.first_at and o.good()
    assert res.reserved_trace[-1] == (pytest.approx(gateway_end), 0.0)


def test_transport_tail_beyond_cut_leaves_success_unresolved():
    cfg = replace(CFG, evaluation_window_s=1.0, tail_s=0.0)
    sc = Scenario((offer("edge", 0.9, prompt=0, out=1, egress=0.2),))
    res = rollout(snap(), plan(16.0), sc, P, cfg)
    o = by_id(res)["edge"]
    gateway_end = T0 + 0.9 + P.first_overhead_s  # < cut, client end > cut
    assert res.reserved_trace[-1] == (pytest.approx(gateway_end), 0.0)  # released at gateway
    assert o.accepted and o.first_at is None and o.terminal_at is None and o.success is None
    assert len(res.outcomes) == 1 and res.score_lower == 0.0 and res.score_upper == 1.0


def test_late_failure_with_egress_ordering_and_beyond_cut():
    sc = Scenario((offer("f", 0.0, out=5, success=False, egress=0.1),))
    o = by_id(rollout(snap(), plan(32.0), sc, P, CFG))["f"]
    assert o.success is False and o.first_at <= o.terminal_at and not o.good()
    cfg = replace(CFG, evaluation_window_s=1.0, tail_s=0.0)
    late = Scenario((offer("f", 0.9, prompt=0, out=1, success=False, egress=0.2),))
    res = rollout(snap(), plan(32.0), late, P, cfg)
    assert by_id(res)["f"].success is None  # client terminal beyond cut
    assert res.score_lower == 0.0 and res.score_upper == 0.0  # known gateway failure


def test_generate_scenarios_bounded_for_huge_finite_inputs():
    base = dict(arrival_rate=1.0, prompt_tokens=(1,), max_tokens=(1,), count=1, seed=0, duration_s=1.0)
    start = time.monotonic()
    for kw, reason in (({"arrival_rate": 1e300}, "offer_limit"),
                       ({"duration_s": 1e300}, "offer_limit"),
                       ({"count": 10**12}, "scenario_count_limit"),
                       ({"count": 32, "arrival_rate": 1000.0}, "offer_limit")):
        with pytest.raises(ValueError, match=f"^{reason}$"):
            generate_scenarios(snap(), **{**base, **kw})
    assert time.monotonic() - start < 2.0
    assert len(generate_scenarios(snap(), **{**base, "count": 32})) == 32
