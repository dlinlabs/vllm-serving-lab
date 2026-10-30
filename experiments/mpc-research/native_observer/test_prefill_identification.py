import pytest

from prefill_identification import extract, fit, serial_capacity_screen


def sources():
    initial = dict(run_id="gateway",clock_domain="boot")
    episode = dict(seed=160009,initial=initial,start=10,prelude=False,budget=16,
        native_frames=[dict(clock_domain="boot",sampled_at=10.05,running=1,waiting=0)],
        rows=[dict(status="complete",request_id="a",clock_domain="boot",
                   first_content_at=10.1,usage={"prompt_tokens":128})])
    events = [dict(event="backend_sent",run_id="gateway",request_id="a",monotonic=10.01),
              dict(event="mpc_ledger",run_id="gateway",effective_at=10,running=1,
                   complete=True,clock_domain="boot")]
    return [([episode],events)]


def test_isolated_label_preserves_http_overhead_and_excludes_shared_work():
    source = sources()
    rows, counts = extract(source)
    assert counts["observed_isolated"]==1
    assert rows[0]["first_residence_s"]==pytest.approx(.09)
    source[0][1][-1]["running"]=2
    rows, counts = extract(source)
    assert not rows and counts["shared_reservation_excluded"]==1


def test_foreign_clock_is_rejected_before_timing_comparison():
    source = sources()
    source[0][0][0]["native_frames"][0].update(clock_domain="other-boot",sampled_at=None)
    with pytest.raises(ValueError,match="clock mismatch"):
        extract(source)


def test_unidentifiable_single_prompt_level_is_not_given_a_slope():
    rows = [dict(seed=seed,prompt_tokens=128,first_residence_s=.03) for seed in (160009,160010)]
    with pytest.raises(ValueError,match="two prompt"):
        fit(rows)
    rows += [dict(seed=seed,prompt_tokens=4096,first_residence_s=.05) for seed in (160009,160010)]
    parameters = fit(rows)
    assert parameters["prefill_s_per_token"]==pytest.approx(.02/(4096-128))


def test_prelude_is_retained_in_accounting_but_not_fitted():
    source = sources()
    source[0][0][0]["prelude"]=True
    rows, counts = extract(source)
    assert not rows and counts["warmup_excluded"]==counts["offered"]==1


def test_http_proxy_serial_interpretation_is_falsified_by_empty_start_throughput():
    source = sources()
    episode = source[0][0][0]
    episode.update(schedule=[{"budget":16}],prompt_tokens=128,episode_id="train",
                   initial_drain=[{"drained":True},{"drained":True}])
    episode["initial"]["requests"]=[]
    episode["rows"][0]["terminal_at"]=10.2
    result = serial_capacity_screen(dict(first_overhead_s=100,prefill_s_per_token=0),source)
    assert result["rejected"]
    assert result["episodes"][0]["serial_complete_upper_bound"]==0
    episode["initial"]["requests"]=[{"request_id":"old"}]
    with pytest.raises(ValueError,match="empty initial"):
        serial_capacity_screen(dict(first_overhead_s=100,prefill_s_per_token=0),source)
