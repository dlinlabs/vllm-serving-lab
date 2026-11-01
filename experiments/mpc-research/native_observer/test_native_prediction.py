import copy

import numpy as np
import pytest

import native_prediction as npred

A = "a" * 32
B = "b" * 32
R = "c" * 32


def frame(t, running=(), waiting=0):
    return dict(sampled_at=t - .01, received_at=t, waiting=waiting, running=len(running),
                requests=[dict(request_id=f"chatcmpl-{rid}-0000abcd", output_tokens=out)
                          for rid, out in running])


def episode(seed=1, budget=16):
    gateway = {
        A: dict(accepted=10.0, first_content=10.3, completed=11.0, input_tokens=4096, max_output_tokens=128),
        B: dict(accepted=10.2, first_content=10.9, completed=11.5, input_tokens=4096, max_output_tokens=128),
    }
    rows = [dict(request_id=A, status="complete", first_content_at=10.31, terminal_at=11.01),
            dict(request_id=B, status="complete", first_content_at=10.91, terminal_at=11.51),
            dict(request_id=R, status="rejected", first_content_at=None, terminal_at=10.1)]
    frames = [frame(10.2, [(A, 0)], waiting=1), frame(10.5, [(A, 5), (B, 0)])]
    return dict(seed=seed, budget=budget, prompt_tokens=4096, max_tokens=128, start=10.0,
                rows=rows, frames=frames, gateway=gateway)


def test_features_use_only_events_known_at_origin():
    e = episode()
    t0 = 10.5
    before = npred.features_at(t0, e["frames"][1], e["gateway"], e["start"])
    stripped = {rid: {k: v for k, v in r.items() if not isinstance(v, float) or v <= t0 or k == "accepted"}
                for rid, r in e["gateway"].items()}
    mutated = copy.deepcopy(e["gateway"])
    mutated[B]["first_content"] = 99.0
    mutated[A]["completed"] = 50.0
    for other in (stripped, mutated):
        assert npred.features_at(t0, e["frames"][1], other, e["start"]) == before
    started = {f["request_id"]: f["started"] for f in before}
    assert started == {A: True, B: False}
    with pytest.raises(ValueError, match="future_native_frame"):
        npred.features_at(10.0, e["frames"][1], e["gateway"], e["start"])


def test_labels_are_separate_and_rejections_are_counted_not_sampled():
    e = episode()
    samples, accounting = npred.build_samples([e])
    shifted = copy.deepcopy(e)
    for row in shifted["rows"]:
        if row["status"] == "complete":
            row["terminal_at"] += 5
            row["first_content_at"] += 1
    samples2, _ = npred.build_samples([shifted])
    assert [s["x"] for s in samples] == [s["x"] for s in samples2]
    assert [s["y"] for s in samples] != [s["y"] for s in samples2]
    assert R not in {s["request_id"] for s in samples}
    count = accounting[0]
    assert (count["offered"], count["complete"], count["rejected"]) == (3, 2, 1)
    # A and B (accepted 10.2) are both pre-first-content at the 10.2 frame.
    assert count["first_content_sampled_requests"] == 2
    assert count["terminal_sampled_requests"] == 2


def test_boundaries_censoring_and_unlinkable_native_ids():
    e = episode()
    e["gateway"][B]["accepted"] = 10.5  # accepted exactly at t0 is known
    e["gateway"][A]["completed"] = 10.5  # completed exactly at t0 is no longer in flight
    ids = {f["request_id"] for f in npred.features_at(10.5, e["frames"][1], e["gateway"], e["start"])}
    assert ids == {B}
    e = episode()
    e["rows"][1]["terminal_at"] = None
    _, accounting = npred.build_samples([e])
    assert accounting[0]["terminal_right_censored_samples"] == 2  # B in both frames
    bad = frame(10.5, [(A, 0)])
    bad["requests"][0]["request_id"] = "chatcmpl-not-linked"
    with pytest.raises(ValueError, match="unlinkable"):
        npred.native_running(bad)
    short = frame(10.5, [(A, 0)])
    short["running"] = 2
    with pytest.raises(ValueError, match="incomplete"):
        npred.native_running(short)


def synthetic(seed, budget, n, rng):
    out = []
    for i in range(n):
        x = (1.0, *rng.uniform(0, 2, size=len(npred.FEATURES) - 1))
        out.append(dict(seed=seed, budget=budget, request_id=f"{seed}-{budget}-{i}", target="terminal",
                        x=x, y=float(np.dot(x, [.1, .2, .1, .0, .3, .0]) + rng.normal(0, .01)),
                        elapsed=float(rng.uniform(0, 1)), occupancy=float(rng.integers(0, 9)),
                        persisted=None if i == 0 else float(rng.uniform(0, 1))))
    return out


def test_grouped_folds_hold_both_arms_and_never_fit_held_labels():
    rng = np.random.default_rng(0)
    samples = [s for seed in (1, 2, 3) for budget in (16, 32) for s in synthetic(seed, budget, 20, rng)]
    folds = npred.leave_one_block_out(samples, "terminal")
    poisoned = [dict(s, y=s["y"] + 100) if s["seed"] == 2 else s for s in samples]
    poisoned_folds = npred.leave_one_block_out(poisoned, "terminal")
    for fold, other in zip(folds, poisoned_folds):
        assert fold["held_seed"] not in fold["train_seeds"]
        assert fold["held_budgets"] == [16, 32]
        if fold["held_seed"] == 2:
            assert fold["params"] == other["params"]
        else:
            assert fold["params"] != other["params"]
    assert all(c >= 0 for fold in folds for c in fold["params"]["candidate"])
    assert folds[0]["persistence_fallback_samples"] == 2


def intervention_document(seeds=(160001, 160002)):
    episodes, telemetry = [], []
    for seed in seeds:
        for arm in (0, 1):
            run_id, base = f"run{seed}{arm}", float(seed % 100) * 100 + arm * 20
            ids = [f"{seed:08d}{arm:024d}", f"{seed:08d}{arm + 2:024d}"]
            rows, frames = [], []
            for i, rid in enumerate(ids):
                accepted = base + 1 + i * .2
                for event, at in (("accepted", accepted), ("first_content", accepted + .3),
                                  ("completed", accepted + 1)):
                    telemetry.append(dict(event=event, run_id=run_id, request_id=rid, monotonic=at,
                                          estimated_input_tokens=128, max_output_tokens=128))
                rows.append(dict(request_id=rid, status="complete", clock_domain="clk",
                                 first_content_at=accepted + .31, terminal_at=accepted + 1.01))
            for t in (base + 1.1, base + 1.25, base + 1.5):
                f = frame(t, [(ids[0], 1)])
                f["clock_domain"] = "clk"
                frames.append(f)
            frames.reverse()  # received order is asynchronous; reader must sort by received_at
            episodes.append(dict(seed_block=seed, arm=arm, source_script="synthetic", split="train",
                                 summary=dict(offered=2, complete=2, rejected=0, other=0),
                                 schedule=[dict(at_s=0, budget=16), dict(at_s=1, budget=32)],
                                 initial=dict(run_id=run_id, clock_domain="clk", sampled_at=base),
                                 final=dict(sampled_at=base + 5), rows=rows, native_frames=frames))
    return dict(scope="TRAIN only; no MPC or calibration acceptance", episodes=episodes), telemetry


def test_intervention_reader_pairs_arms_and_folds_hold_whole_seed():
    document, telemetry = intervention_document()
    episodes, extra = npred.parse_interventions(document, telemetry, forbidden_seeds={159901})
    assert [(e["seed"], e["budget"]) for e in episodes] == [
        (160001, "arm0"), (160001, "arm1"), (160002, "arm0"), (160002, "arm1")]
    assert all([f["received_at"] for f in e["frames"]] == sorted(f["received_at"] for f in e["frames"])
               for e in episodes)
    samples, accounting = npred.build_samples(episodes)
    assert all(a["complete"] == a["gateway_admitted_joined"] == 2 for a in accounting)
    assert {a["dataset"] for a in accounting} == {"train-interventions"}
    assert [a["arm"] for a in accounting] == [0, 1, 0, 1]
    assert {tuple(a["schedule_budgets"]) for a in accounting} == {(16, 32)}
    for fold in npred.leave_one_block_out(samples, "terminal"):
        assert fold["held_budgets"] == ["arm0", "arm1"]
        assert fold["held_seed"] not in fold["train_seeds"]
    incomplete = intervention_document()[0]
    incomplete["episodes"][1]["rows"] = []
    kept, extra = npred.parse_interventions(incomplete, telemetry)
    assert {e["seed"] for e in kept} == {160002}
    assert extra["excluded_seed_blocks"] == [dict(seed=160001, reason="fewer than two complete arms")]


@pytest.mark.parametrize("mutate, forbidden", [
    (lambda d: d["episodes"][0].update(seed_block=160021), ()),
    (lambda d: d.update(split="calibration"), ()),
    (lambda d: d["episodes"][2].pop("split"), ()),
    (lambda d: d["episodes"][0]["summary"].update(rejected=1), ()),
    (lambda d: d["episodes"][1].update(arm=0), ()),
    (lambda d: None, {160001}),
])
def test_intervention_reader_fails_closed_on_leaky_or_ambiguous_input(mutate, forbidden):
    document, telemetry = intervention_document()
    mutate(document)
    with pytest.raises(ValueError):
        npred.parse_interventions(document, telemetry, forbidden_seeds=forbidden)
