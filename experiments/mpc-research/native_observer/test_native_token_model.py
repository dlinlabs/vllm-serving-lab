import copy

import numpy as np
import pytest

import native_token_model as ntm


def snap(t, requests, run="r", age=.01):
    return dict(clock_domain="clk", run_id=run, observer_sha256="sha", complete=True,
                running=len(requests), waiting=0, pending_batches=0, sampled_at=t,
                received_at=t + age, frontend_sampled_at=t + .001,
                requests=[dict(request_id=rid, prompt_tokens=p, output_tokens=o, computed_tokens=p + o,
                               maximum_output_tokens=128) for rid, p, o in requests])


def episode(frames, seed=1, arm="arm0"):
    return dict(seed=seed, arm=arm, episode=f"{seed}-{arm}", cell="c", frames=frames)


def test_labels_are_output_deltas_and_features_are_current_only():
    frames = [snap(0.0, [("a", 128, 3), ("b", 4096, 0)]),
              snap(0.1, [("a", 128, 3), ("b", 4096, 1), ("c", 128, 0)]),
              snap(0.2, [("b", 4096, 4), ("c", 128, 0)])]
    rows, count = ntm.episode_intervals(episode(frames))
    first = [r for r in rows if r["request_id"] == "a"][0]
    assert (first["y"], first["D"], first["Pk"]) == (0, 1, 4.096)  # zero increment kept
    mutated = copy.deepcopy(frames)
    for f in mutated:
        for r in f["requests"]:
            r["computed_tokens"] = 999999
    mutated[2]["requests"].append(dict(request_id="z", prompt_tokens=8192, output_tokens=0,
                                       computed_tokens=0, maximum_output_tokens=128))
    mutated[2]["running"] += 1
    rows2, _ = ntm.episode_intervals(episode(mutated))
    assert [(r["y"], r["D"], r["Pk"], r["dt"]) for r in rows] == [(r["y"], r["D"], r["Pk"], r["dt"]) for r in rows2]
    assert count["zero_increment_intervals"] == 1
    assert count["prefill_phase_survivor_intervals_out_of_scope"] == 2  # b at 0.0, c at 0.1
    assert count["disappeared_requests_censored"] == 1 and count["new_request_appearances"] == 1
    assert [r["persisted_rate"] for r in rows if r["request_id"] == "b"] == [pytest.approx(10.0)]


def test_bad_intervals_are_excluded_with_reasons_and_regressions_fail_closed():
    frames = [snap(0.0, [("a", 128, 1)]), snap(0.1, [("a", 128, 2)], age=.6),
              snap(0.2, [("a", 128, 3)]), snap(0.8, [("a", 128, 9)]), snap(0.9, [("a", 128, 9)], run="other")]
    rows, count = ntm.episode_intervals(episode(frames))
    assert rows == []
    assert count["excluded_pair_stale_or_negative_age"] == 2
    assert count["excluded_pair_gap_exceeds_bound"] == 1
    assert count["excluded_pair_source_identity"] == 1
    with pytest.raises(ValueError, match="non_monotonic"):
        ntm.episode_intervals(episode([snap(0.0, [("a", 128, 5)]), snap(0.1, [("a", 128, 4)])]))


def synthetic(seed, arm, rng, theta=(12.0, 0.05, 9.0)):
    rows = []
    for k in range(300):
        D, Pk, dt = int(rng.integers(1, 30)), float(rng.choice([0, 4.096, 8.192])), float(rng.uniform(.05, .15))
        lam = 1000 * dt / (theta[0] + theta[1] * D + theta[2] * Pk)
        rows.append(dict(seed=seed, arm=arm, episode=f"{seed}-{arm}", request_id=f"r{k % 40}", dt=dt,
                         y=float(rng.poisson(lam)), D=D, Pk=Pk, running=D, persisted_rate=None))
    return rows


def test_poisson_service_fit_recovers_law_and_grouped_folds_never_see_held_labels():
    rng = np.random.default_rng(1)
    rows = [r for s in (1, 2, 3) for a in ("arm0", "arm1") for r in synthetic(s, a, rng)]
    theta = ntm.fit(rows)["candidate"]
    assert theta[0] == pytest.approx(12, rel=.15) and theta[2] == pytest.approx(9, rel=.25)
    assert min(theta) >= 0
    folds = ntm.leave_one_seed_out(rows)
    poisoned = ntm.leave_one_seed_out([dict(r, y=r["y"] + 50) if r["seed"] == 2 else r for r in rows])
    for fold, other in zip(folds, poisoned):
        assert fold["held_arms"] == ["arm0", "arm1"] and fold["held_seed"] not in fold["train_seeds"]
        assert (fold["params_ms_per_token"] == other["params_ms_per_token"]) == (fold["held_seed"] == 2)
    with pytest.raises(ValueError, match="invalid_grouped_fold"):
        ntm.leave_one_seed_out([r for r in rows if not (r["seed"] == 3 and r["arm"] == "arm1")])


def test_weights_balance_seed_blocks_episodes_and_requests():
    rows = [dict(seed=1, episode="e1", request_id="a")] * 3 + [dict(seed=1, episode="e1", request_id="b")] \
        + [dict(seed=2, episode="e2", request_id="c")]
    w = ntm.weights(rows)
    assert w.sum() == pytest.approx(1)
    assert w[:3].sum() == pytest.approx(w[3]) == pytest.approx(.25) and w[4] == pytest.approx(.5)
