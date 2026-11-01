"""Offline one-step native token-progress (decode service) diagnostic; Train-only, unaccepted.

Uses native read-only snapshots ONLY (no Gateway events). For consecutive valid
snapshots k -> k+1 of one episode, every request running in BOTH snapshots with
output_tokens > 0 at k (a decode-phase survivor) contributes one interval whose label
is output_tokens[k+1] - output_tokens[k] (zero increments included). Features are
current-snapshot only: D = running requests with output > 0, Pk = physical prompt
tokens / 1000 of running requests still at output 0, and the known source-clock gap
dt. num_computed_tokens is never used.

Candidate service law per decode request (milliseconds per token):
    s = theta0 + thetaD * D + thetaP * Pk,  E[delta] = 1000 * dt / s,  theta >= 0,
fitted by weighted Poisson likelihood with fixed weights (each seed block equal, each
episode equal within block, each request equal within episode). No held-fold tuning.

Conditional-survivor limitation: requests that finish (disappear) inside an interval,
new arrivals, and prefill-phase requests are excluded and counted; the model describes
progress of requests that survive the interval, not global throughput or causal effects.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

FROZEN_IMPLEMENTATION_HASH = "b4f234020105f69f608da45dba7317a8bb632ecd84aeb0e310362740e5fa083a"
MAX_FRESHNESS_S = 0.5      # frozen qualification: received_at - sampled_at
MAX_SAMPLE_WIDTH_S = 0.05  # frozen qualification: frontend_sampled_at - sampled_at
MAX_GAP_S = 0.5            # one-step interval bound, same as frozen freshness
MIN_THETA0_MS = 0.1
RESERVED_TEST_SEEDS = frozenset(range(160021, 160025))
MODELS = ("constant", "queue", "persistence", "candidate")


def frame_problem(frame, reference):
    if frame["clock_domain"] != reference["clock_domain"]:
        return "clock_domain"
    if frame["run_id"] != reference["run_id"] or frame["observer_sha256"] != reference["observer_sha256"]:
        return "source_identity"
    if not frame["complete"] or len(frame["requests"]) != frame["running"]:
        return "incomplete_request_list"
    if not 0 <= frame["received_at"] - frame["sampled_at"] <= MAX_FRESHNESS_S:
        return "stale_or_negative_age"
    if not 0 <= frame["frontend_sampled_at"] - frame["sampled_at"] <= MAX_SAMPLE_WIDTH_S:
        return "sample_width"
    return None


def state(frame):
    """Current-snapshot-only features."""
    requests = {r["request_id"]: r for r in frame["requests"]}
    if len(requests) != len(frame["requests"]):
        raise ValueError("duplicate_native_request")
    decode = sum(1 for r in requests.values() if r["output_tokens"] > 0)
    prefill_k = sum(r["prompt_tokens"] for r in requests.values() if r["output_tokens"] == 0) / 1000
    return requests, dict(D=decode, Pk=prefill_k, running=frame["running"], waiting=frame["waiting"],
                          pending_batches=frame["pending_batches"])


def episode_intervals(episode):
    """One-step decode-survivor intervals and explicit exclusion accounting."""
    frames = sorted(episode["frames"], key=lambda f: f["sampled_at"])
    if any(b["sampled_at"] <= a["sampled_at"] for a, b in zip(frames, frames[1:])):
        raise ValueError("non_increasing_sample_time")
    count = Counter(snapshots=len(frames), snapshot_pairs=max(len(frames) - 1, 0))
    reference = frames[0]
    problems = [frame_problem(f, reference) for f in frames]
    intervals, previous_rate = [], {}
    for k in range(len(frames) - 1):
        a, b = frames[k], frames[k + 1]
        dt = b["sampled_at"] - a["sampled_at"]
        reason = problems[k] or problems[k + 1] or ("gap_exceeds_bound" if dt > MAX_GAP_S else None)
        if reason:
            count[f"excluded_pair_{reason}"] += 1
            previous_rate = {}
            continue
        count["valid_pairs"] += 1
        now, features = state(a)
        nxt, _ = state(b)
        count["new_request_appearances"] += len(nxt.keys() - now.keys())
        count["disappeared_requests_censored"] += len(now.keys() - nxt.keys())
        rates = {}
        for rid in sorted(now.keys() & nxt.keys()):
            out, out_next = now[rid]["output_tokens"], nxt[rid]["output_tokens"]
            if out_next < out or out_next > nxt[rid]["maximum_output_tokens"]:
                raise ValueError("non_monotonic_or_over_cap_output")
            if now[rid]["prompt_tokens"] != nxt[rid]["prompt_tokens"]:
                raise ValueError("prompt_tokens_changed")
            delta = out_next - out
            rates[rid] = delta / dt
            if out == 0:
                count["prefill_phase_survivor_intervals_out_of_scope"] += 1
                continue
            count["decode_survivor_intervals"] += 1
            count["zero_increment_intervals"] += delta == 0
            intervals.append(dict(seed=episode["seed"], arm=episode["arm"], episode=episode["episode"],
                                  request_id=rid, dt=dt, y=delta, persisted_rate=previous_rate.get(rid),
                                  **features))
        previous_rate = rates
    count["decode_survivor_requests"] = len({i["request_id"] for i in intervals})
    if frames:
        count["requests_present_at_first_snapshot"] = len(frames[0]["requests"])
        count["requests_present_at_last_snapshot_uncensored_by_window"] = len(frames[-1]["requests"])
    count["distinct_native_requests"] = len({r["request_id"] for f in frames for r in f["requests"]})
    return intervals, count


def weights(intervals):
    """Seed blocks equal; episodes equal within block; requests equal within episode."""
    per_request = Counter((i["episode"], i["request_id"]) for i in intervals)
    requests = defaultdict(set)
    episodes = defaultdict(set)
    for i in intervals:
        requests[i["episode"]].add(i["request_id"])
        episodes[i["seed"]].add(i["episode"])
    seeds = len(episodes)
    return np.array([1 / per_request[i["episode"], i["request_id"]] / len(requests[i["episode"]])
                     / len(episodes[i["seed"]]) / seeds for i in intervals])


def arrays(intervals):
    return {k: np.array([i[k] for i in intervals], float) for k in ("dt", "y", "D", "Pk", "running")}


def fit_service(X, dt, y, w):
    """Weighted Poisson MLE of lambda = 1000*dt/(X @ theta), theta >= 0, theta0 >= MIN_THETA0_MS."""
    w = w / w.sum()

    def objective(theta):
        s = X @ theta
        lam = 1000 * dt / s
        value = np.sum(w * (lam - y * np.log(lam)))
        grad = (w * (1 - y / lam) * (-lam / s)) @ X
        return value, grad

    start = np.zeros(X.shape[1])
    start[0] = 1000 * np.sum(w * dt) / max(np.sum(w * y), 1e-9)
    bounds = [(MIN_THETA0_MS, None)] + [(0, None)] * (X.shape[1] - 1)
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(maxiter=2000, ftol=1e-15, gtol=1e-12))
    if not result.success:
        raise RuntimeError(f"fit_failed:{result.message}")
    return result.x


def design(a, model):
    one = np.ones_like(a["dt"])
    if model == "constant":
        return one[:, None]
    if model == "queue":
        return np.column_stack([one, a["running"]])
    return np.column_stack([one, a["D"], a["Pk"]])


def fit(train):
    a, w = arrays(train), weights(train)
    return {m: fit_service(design(a, m), a["dt"], a["y"], w).tolist()
            for m in ("constant", "queue", "candidate")}


def predict(params, intervals):
    a = arrays(intervals)
    lam = {m: 1000 * a["dt"] / (design(a, m) @ np.array(params[m])) for m in ("constant", "queue", "candidate")}
    rates = [i["persisted_rate"] for i in intervals]
    lam["persistence"] = np.array([lam["constant"][k] if r is None else r * a["dt"][k]
                                   for k, r in enumerate(rates)])
    return lam, sum(r is None for r in rates)


def deviance(y, lam):
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / lam), 0.0)
    return 2 * (term - (y - lam))


def scores(intervals, lam):
    a, w = arrays(intervals), weights(intervals)
    w = w / w.sum()
    y = a["y"]
    out = {}
    for m in MODELS:
        e = lam[m] - y
        entry = dict(weighted_mae_tokens=float(np.sum(w * np.abs(e))),
                     weighted_bias_tokens=float(np.sum(w * e)))
        if m != "persistence":  # persistence is a point forecast, not a likelihood model
            low, high = poisson.ppf(.05, lam[m]), poisson.ppf(.95, lam[m])
            entry.update(weighted_poisson_deviance=float(np.sum(w * deviance(y, lam[m]))),
                         weighted_pearson_dispersion=float(np.sum(w * e * e / lam[m])),
                         uncalibrated_poisson_90_interval_coverage=float(
                             np.sum(w * ((y >= low) & (y <= high)))))
        out[m] = entry
    return out


def support(intervals):
    a = arrays(intervals)
    return {k: [float(a[k].min()), float(np.median(a[k])), float(a[k].max())]
            for k in ("dt", "y", "D", "Pk", "running")} | dict(zero_fraction=float(np.mean(a["y"] == 0)))


def leave_one_seed_out(intervals):
    folds = []
    for held in sorted({i["seed"] for i in intervals}):
        train = [i for i in intervals if i["seed"] != held]
        test = [i for i in intervals if i["seed"] == held]
        if not train or len({i["arm"] for i in test}) != 2:
            raise ValueError("invalid_grouped_fold")
        params = fit(train)
        lam, fallback = predict(params, test)
        Xh, Xt = arrays(test), arrays(train)
        inside = np.all([(Xh[k] >= Xt[k].min()) & (Xh[k] <= Xt[k].max()) for k in ("D", "Pk", "dt")], 0)
        overall = scores(test, lam)
        best = min((m for m in MODELS if m != "candidate"), key=lambda m: overall[m]["weighted_mae_tokens"])
        per_episode = {}
        for episode in sorted({i["episode"] for i in test}):
            index = [k for k, i in enumerate(test) if i["episode"] == episode]
            per_episode[episode] = dict(intervals=len(index),
                                        scores=scores([test[k] for k in index],
                                                      {m: v[index] for m, v in lam.items()}))
        folds.append(dict(held_seed=held, held_arms=sorted({str(i["arm"]) for i in test}),
                          train_seeds=sorted({i["seed"] for i in train}),
                          held_intervals=len(test),
                          held_requests=len({(i["episode"], i["request_id"]) for i in test}),
                          persistence_fallback_intervals=fallback,
                          held_intervals_inside_train_D_Pk_dt_box=float(inside.mean()),
                          params_ms_per_token=params, scores=overall, best_baseline_by_mae=best,
                          candidate_beats_best_baseline_mae=overall["candidate"]["weighted_mae_tokens"]
                          < overall[best]["weighted_mae_tokens"],
                          candidate_beats_constant_deviance=overall["candidate"]["weighted_poisson_deviance"]
                          < overall["constant"]["weighted_poisson_deviance"],
                          per_episode=per_episode))
    return folds


def load_scout(archive):
    with tarfile.open(archive, "r:gz") as bundle:
        def document(name):
            with bundle.extractfile("./" + name) as stream:
                return json.loads(stream.read().decode())
        first = document("train-saturation-scout-v1.json")
        completion = document("train-saturation-scout-completion.json")
    raw = [e for e in first["episodes"] if "summary" in e] + completion["episodes"]
    return [dict(seed=e["seed"], arm=f"budget{e['budget']}", episode=f"scout-{e['seed']}-{e['budget']}",
                 cell=f"p{e['prompt_tokens']}-m{e['max_tokens']}", frames=e["native_frames"]) for e in raw]


def load_interventions(directory, forbidden_seeds):
    document = json.loads((Path(directory) / "interventions.json").read_text())
    if str(document.get("split", "train")).lower() != "train":
        raise ValueError("non_train_dataset")
    episodes = []
    for e in document["episodes"]:
        if str(e.get("split")).lower() != "train":
            raise ValueError("non_train_episode")
        if e["seed_block"] in RESERVED_TEST_SEEDS or e["seed_block"] in forbidden_seeds:
            raise ValueError("forbidden_or_overlapping_seed")
        episodes.append(dict(seed=e["seed_block"], arm=f"arm{e['arm']}",
                             episode=e.get("episode_id", f"train-{e['seed_block']}-{e['arm']}"),
                             cell=e.get("cell"), frames=e["native_frames"]))
    return episodes


def run(episodes, implementation_hash):
    if implementation_hash != FROZEN_IMPLEMENTATION_HASH:
        raise ValueError("implementation_hash_changed")
    if len({e["episode"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate_episode")
    intervals, accounting = [], []
    for e in episodes:
        rows, count = episode_intervals(e)
        intervals += rows
        accounting.append(dict(seed=e["seed"], arm=e["arm"], episode=e["episode"], cell=e["cell"], **count))
    folds = leave_one_seed_out(intervals)
    params = fit(intervals)
    lam, _ = predict(params, intervals)
    seed_mean = {m: {k: float(np.mean([f["scores"][m][k] for f in folds]))
                     for k in folds[0]["scores"][m]} for m in MODELS}
    candidate = params["candidate"]
    return dict(
        schema="native-token-progress-diagnostic-v1",
        scope="offline retrospective Train-only one-step decode-survivor token progress diagnostic",
        implementation_hash=implementation_hash,
        model_accepted=False, calibration_generated=False, native_test_generated=False,
        online_issued_forecasts=False, causal_claim=False, global_throughput_claim=False,
        inputs="native read-only snapshots only; no Gateway events; num_computed_tokens unused",
        label="output_tokens[k+1] - output_tokens[k] for requests running in both snapshots with "
              "output_tokens[k] > 0; zero increments included",
        features=dict(D="running requests with output_tokens > 0 at k (includes self)",
                      Pk="sum physical prompt_tokens/1000 of running requests with output_tokens == 0 at k",
                      dt="snapshot sampled_at gap (engine monotonic source clock), known at forecast"),
        models=dict(
            constant="lambda = 1000*dt/theta0 (single per-token service time)",
            queue="lambda = 1000*dt/(theta0 + thetaR*running)",
            persistence="lambda = same request's previous valid-interval rate * dt; constant fallback",
            candidate="lambda = 1000*dt/(theta0 + thetaD*D + thetaP*Pk)"),
        fit="weighted Poisson MLE, L-BFGS-B, theta>=0, theta0>=0.1ms; weights fixed: seed blocks "
            "equal, episodes equal within block, requests equal within episode; no tuning",
        freshness_bounds_s=dict(age=MAX_FRESHNESS_S, sample_width=MAX_SAMPLE_WIDTH_S, gap=MAX_GAP_S),
        accounting=accounting,
        totals=dict(sum((Counter({k: v for k, v in a.items() if isinstance(v, int) and k != "seed"})
                         for a in accounting), Counter())),
        support_all=support(intervals),
        leave_one_seed_block_out=dict(
            folds=folds, seed_balanced_mean=seed_mean,
            candidate_wins_mae=sum(f["candidate_beats_best_baseline_mae"] for f in folds),
            candidate_beats_constant_deviance=sum(f["candidate_beats_constant_deviance"] for f in folds)),
        in_sample_all_blocks=dict(
            label="in-sample training fit; not a generalisation estimate",
            params_ms_per_token=params,
            candidate_interpretation=dict(
                theta0_ms_per_token=candidate[0], thetaD_ms_per_token_per_decode_request=candidate[1],
                thetaP_ms_per_token_per_kilo_prefill_prompt_token=candidate[2],
                isolated_decode_rate_tokens_per_s=1000 / (candidate[0] + candidate[1])),
            scores=scores(intervals, lam)),
        limitations=[
            "Conditional survivors only: requests finishing inside an interval, new arrivals and "
            "prefill-phase requests are excluded and counted; not a global throughput model.",
            "Retrospective Train-only fit under fixed/scheduled budgets; no causal effect of admission "
            "actions is identified and no online forecasts were issued.",
            "Intervals within a request/episode are dependent; scores are weighted per request, "
            "episode and seed block; no standard errors are claimed.",
            "Poisson 90% interval coverage is a descriptive, uncalibrated dispersion check.",
            "Waiting requests are counts only; their prompts are not visible natively and are not in Pk.",
            "Output counters are engine output tokens; client streaming delay is not modelled.",
            "Each held seed block also has one other-seed block in its prompt/max-output cell only.",
        ])


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from mpc_lab.contracts import implementation_hash
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(root / "results/mpc-native-4090-scout/native-evidence.tar.gz"))
    parser.add_argument("--interventions-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    before = implementation_hash()
    scout = load_scout(args.evidence)
    episodes = scout + load_interventions(args.interventions_dir, {e["seed"] for e in scout})
    result = run(episodes, before)
    result["inputs_sha256"] = dict(
        evidence=hashlib.sha256(Path(args.evidence).read_bytes()).hexdigest(),
        interventions=hashlib.sha256((Path(args.interventions_dir) / "interventions.json").read_bytes()).hexdigest(),
        script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if implementation_hash() != before:
        raise RuntimeError("implementation_hash_changed_during_run")
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lo = result["leave_one_seed_block_out"]
    print(json.dumps(dict(totals=result["totals"], seed_balanced_mean=lo["seed_balanced_mean"],
                          wins_mae=lo["candidate_wins_mae"], wins_dev=lo["candidate_beats_constant_deviance"],
                          params=result["in_sample_all_blocks"]["candidate_interpretation"]), indent=1))
