"""Offline, Train-only native residence prediction diagnostic (not an accepted model).

Prediction origin t0 is the client receipt time of a native read-only frame.
Features use only the native frame (sampled <= t0) and Gateway telemetry events
whose recorded effective monotonic time is <= t0. Those Gateway event times are a
retrospective (oracle) reconstruction: no Gateway-snapshot receipt time was
recorded, so availability of these events to an online consumer at t0 is
unverified and they are not qualified online/causal inputs. Labels are the later
client HTTP first-content and terminal times; they are built separately and never
enter feature construction. Evaluation is leave-one-seed-block-out; each held
block contains both arms (fixed budgets for the scout, schedule arms for the
optional Train intervention dataset).
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np
from scipy.optimize import nnls

FROZEN_IMPLEMENTATION_HASH = "b4f234020105f69f608da45dba7317a8bb632ecd84aeb0e310362740e5fa083a"
NATIVE_ID = re.compile(r"chatcmpl-([0-9a-f]{32})-([0-9a-f]{8})")
TARGETS = ("first_content", "terminal")
# Predeclared before any fit; fixed for every fold and target.
FEATURES = (
    "intercept_s",
    "own_unstarted_prompt_ktokens",   # own prompt/1000 while Gateway has not seen first content
    "gateway_prefill_backlog_ktokens",  # admitted in-flight prompts/1000 without Gateway first content
    "native_waiting",                 # native scheduler waiting count
    "own_remaining_output_cap_htokens",  # (max_tokens - native output tokens)/100, an upper bound
    "own_cap_x_native_decode_batch",  # remaining cap * running requests with output>0 / 10
)
RIDGE = 1e-2  # fixed, relative to total training weight; never tuned


def gateway_requests(lines, run_id):
    """Per-request Gateway-observed event times; ordering is only by event time."""
    events = defaultdict(dict)
    for frame in lines:
        if frame.get("run_id") != run_id or frame.get("event") not in (
                "accepted", "first_content", "completed"):
            continue
        record = events[frame["request_id"]]
        if frame["event"] in record:
            raise ValueError("duplicate_gateway_event")
        record[frame["event"]] = frame["monotonic"]
        if frame["event"] == "accepted":
            record["input_tokens"] = frame["estimated_input_tokens"]
            record["max_output_tokens"] = frame["max_output_tokens"]
    return dict(events)


def native_running(frame):
    """Map visible running native requests to Gateway UUID -> output tokens."""
    visible = {}
    for request in frame["requests"]:
        match = NATIVE_ID.fullmatch(request["request_id"])
        if not match or match[1] in visible:
            raise ValueError("unlinkable_native_request")
        visible[match[1]] = request["output_tokens"]
    if len(visible) != frame["running"]:
        raise ValueError("incomplete_native_request_list")
    return visible


def known(record, event, t0):
    return record.get(event) is not None and record[event] <= t0


def features_at(t0, frame, gateway, start):
    """Feature rows for Gateway-admitted in-flight requests at t0; past-only inputs."""
    if frame["sampled_at"] > t0:
        raise ValueError("future_native_frame")
    visible = native_running(frame)
    decode_batch = sum(1 for tokens in visible.values() if tokens > 0)
    members = {rid: r for rid, r in gateway.items()
               if start <= r["accepted"] <= t0 and not known(r, "completed", t0)}
    backlog = sum(r["input_tokens"] for r in members.values()
                  if not known(r, "first_content", t0)) / 1000
    result = []
    for rid in sorted(members):
        record = members[rid]
        started = known(record, "first_content", t0)
        cap = (record["max_output_tokens"] - visible.get(rid, 0)) / 100
        x = (1.0, 0.0 if started else record["input_tokens"] / 1000, backlog,
             float(frame["waiting"]), cap, cap * decode_batch / 10)
        result.append(dict(request_id=rid, x=x, started=started, visible=rid in visible,
                           elapsed=t0 - record["accepted"],
                           occupancy=float(frame["running"] + frame["waiting"])))
    return result


def recent_realized(t0, gateway, start, event):
    """Most recent residence realized at Gateway in this episode before t0, else None."""
    done = [(r[event], r[event] - r["accepted"]) for r in gateway.values()
            if r.get("accepted") is not None and start <= r["accepted"] and known(r, event, t0)]
    return max(done)[1] if done else None


def label(row, t0, target):
    at = row["first_content_at"] if target == "first_content" else row["terminal_at"]
    return None if at is None else at - t0


def build_samples(episodes):
    samples, accounting = [], []
    for episode in episodes:
        gateway, start = episode["gateway"], episode["start"]
        rows = {r["request_id"]: r for r in episode["rows"]}
        admitted = {rid for rid, r in gateway.items() if start <= r.get("accepted", -1) and rid in rows}
        count = Counter(offered=len(rows),
                        complete=sum(r["status"] == "complete" for r in rows.values()),
                        rejected=sum(r["status"] == "rejected" for r in rows.values()),
                        gateway_admitted_joined=len(admitted), native_frames=len(episode["frames"]))
        if any(rows[rid]["status"] != "complete" for rid in admitted):
            raise ValueError("admitted_request_without_complete_row")
        sampled = {t: set() for t in TARGETS}
        for frame in episode["frames"]:
            t0 = frame["received_at"]
            rows_at = features_at(t0, frame, gateway, start)
            count["frames_without_admitted_inflight"] += not rows_at
            persisted = {t: recent_realized(t0, gateway, start, e)
                         for t, e in zip(TARGETS, ("first_content", "completed"))}
            for item in rows_at:
                rid = item["request_id"]
                if rid not in rows:
                    raise ValueError("gateway_admission_without_http_row")
                for target in TARGETS:
                    if target == "first_content" and item["started"]:
                        continue
                    y = label(rows[rid], t0, target)
                    if y is None:
                        count[f"{target}_right_censored_samples"] += 1
                        continue
                    count[f"{target}_nonpositive_labels"] += y <= 0
                    sampled[target].add(rid)
                    samples.append(dict(item, target=target, y=y, seed=episode["seed"],
                                        cell=(episode["prompt_tokens"], episode["max_tokens"]),
                                        budget=episode["budget"], persisted=persisted[target]))
        for target in TARGETS:
            count[f"{target}_samples"] = sum(
                s["target"] == target and s["seed"] == episode["seed"] and s["budget"] == episode["budget"]
                for s in samples)
            count[f"{target}_sampled_requests"] = len(sampled[target])
            count[f"{target}_admitted_never_in_risk_set"] = len(admitted - sampled[target])
        accounting.append(dict(seed=episode["seed"], budget=episode.get("accounting_budget", episode["budget"]),
                               prompt_tokens=episode["prompt_tokens"],
                               max_tokens=episode["max_tokens"], **count,
                               **{k: episode[k] for k in ("dataset", "source_script", "arm", "episode_id", "cell",
                                                       "schedule_budgets") if k in episode}))
    return samples, accounting


def request_weights(samples):
    """Each (episode, request, target) contributes total weight 1 across its samples."""
    n = Counter((s["seed"], s["budget"], s["request_id"], s["target"]) for s in samples)
    return np.array([1 / n[s["seed"], s["budget"], s["request_id"], s["target"]] for s in samples])


def fit_nonnegative(X, y, w, ridge=RIDGE):
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    root = np.sqrt(w)
    penalty = np.sqrt(ridge * w.sum()) * np.eye(X.shape[1])[1:]
    A = np.vstack([X * root[:, None], penalty])
    b = np.concatenate([y * root, np.zeros(X.shape[1] - 1)])
    return nnls(A, b, maxiter=100 * X.shape[1])[0]


def fit(train):
    """All fitted quantities use training samples only."""
    w = request_weights(train)
    y = np.array([s["y"] for s in train])
    total = np.array([s["y"] + s["elapsed"] for s in train])
    occupancy = np.array([[1.0, s["occupancy"]] for s in train])
    return dict(constant_s=float(np.average(y, weights=w)),
                total_residence_s=float(np.average(total, weights=w)),
                queue=fit_nonnegative(occupancy, y, w).tolist(),
                candidate=fit_nonnegative([s["x"] for s in train], y, w).tolist())


def predict(params, samples):
    fallback = [s["persisted"] is None for s in samples]
    return dict(
        constant=np.full(len(samples), params["constant_s"]),
        elapsed=np.array([max(params["total_residence_s"] - s["elapsed"], 0) for s in samples]),
        persistence=np.array([max((params["total_residence_s"] if s["persisted"] is None
                                   else s["persisted"]) - s["elapsed"], 0) for s in samples]),
        queue=np.array([params["queue"][0] + params["queue"][1] * s["occupancy"] for s in samples]),
        candidate=np.array([s["x"] for s in samples]) @ np.array(params["candidate"]),
    ), sum(fallback)


def errors(samples, predictions):
    w = request_weights(samples)
    y = np.array([s["y"] for s in samples])
    out = {}
    for name, value in predictions.items():
        e = value - y
        out[name] = dict(request_weighted_mae_s=float(np.average(np.abs(e), weights=w)),
                         request_weighted_bias_s=float(np.average(e, weights=w)),
                         sample_mae_s=float(np.mean(np.abs(e))),
                         sample_rmse_s=float(np.sqrt(np.mean(e * e))),
                         sample_abs_error_p50_p90_s=np.quantile(np.abs(e), [.5, .9]).tolist())
    return out


def leave_one_block_out(samples, target):
    rows = [s for s in samples if s["target"] == target]
    blocks = sorted({s["seed"] for s in rows})
    folds = []
    for held in blocks:
        train = [s for s in rows if s["seed"] != held]
        test = [s for s in rows if s["seed"] == held]
        if not train or not test or {s["seed"] for s in train} & {s["seed"] for s in test} \
                or len({s["budget"] for s in test}) != 2:
            raise ValueError("invalid_grouped_fold")
        params = fit(train)
        predictions, fallback = predict(params, test)
        arms = {}
        for budget in sorted({s["budget"] for s in test}):
            index = [i for i, s in enumerate(test) if s["budget"] == budget]
            arms[str(budget)] = dict(
                samples=len(index),
                requests=len({test[i]["request_id"] for i in index}),
                errors=errors([test[i] for i in index], {k: v[index] for k, v in predictions.items()}))
        overall = errors(test, predictions)
        best_baseline = min((k for k in overall if k != "candidate"),
                            key=lambda k: overall[k]["request_weighted_mae_s"])
        folds.append(dict(held_seed=held, held_budgets=sorted({s["budget"] for s in test}),
                          train_seeds=sorted({s["seed"] for s in train}),
                          train_samples=len(train), held_samples=len(test),
                          held_requests=len({(s["budget"], s["request_id"]) for s in test}),
                          persistence_fallback_samples=fallback, params=params, errors=overall,
                          best_baseline=best_baseline,
                          candidate_beats_best_baseline=overall["candidate"]["request_weighted_mae_s"]
                          < overall[best_baseline]["request_weighted_mae_s"],
                          per_budget_arm=arms))
    return folds


def load(archive):
    with tarfile.open(archive, "r:gz") as bundle:
        def text(name):
            with bundle.extractfile("./" + name) as stream:
                return stream.read().decode()
        first = json.loads(text("train-saturation-scout-v1.json"))
        completion = json.loads(text("train-saturation-scout-completion.json"))
        telemetry = [json.loads(line) for line in text("gateway-observed-telemetry.jsonl").splitlines()]
    failed = [e for e in first["episodes"] if "summary" not in e]
    raw = [e for e in first["episodes"] if "summary" in e] + completion["episodes"]
    episodes = []
    for e in raw:
        run_id, clock = e["initial"]["run_id"], e["initial"]["clock_domain"]
        if any(r["clock_domain"] != clock for r in e["rows"]) or any(
                f["clock_domain"] != clock for f in e["native_frames"]):
            raise ValueError("clock_domain_mismatch")
        episodes.append(dict(seed=e["seed"], budget=e["budget"], prompt_tokens=e["prompt_tokens"],
                             max_tokens=e["max_tokens"], start=e["start"], rows=e["rows"],
                             frames=sorted(e["native_frames"], key=lambda f: f["received_at"]),
                             gateway=gateway_requests(telemetry, run_id)))
    # Restrict each episode's Gateway view to requests accepted in its own drained window.
    ordered = sorted(episodes, key=lambda e: e["start"])
    starts = [e["start"] for e in ordered]
    for e in episodes:
        end = starts[bisect_right(starts, e["start"])] if bisect_right(starts, e["start"]) < len(starts) \
            else float("inf")
        e["gateway"] = {rid: r for rid, r in e["gateway"].items()
                        if r.get("accepted") is not None and e["start"] <= r["accepted"] < end}
    accepted_total = sum(1 for f in telemetry if f.get("event") == "accepted")
    return episodes, dict(failed_preoffer_attempts=[dict(seed=f["seed"], budget=f["budget"],
                                                         offers=0, status=f["action"].get("status"))
                                                    for f in failed],
                          gateway_accepted_events_total=accepted_total)


RESERVED_TEST_SEEDS = frozenset(range(160021, 160025))


def load_interventions(directory, forbidden_seeds=()):
    """Optional second Train dataset: interventions.json + gateway-telemetry.jsonl.

    Episodes carry seed_block and arm (0/1); budget may vary inside an episode, so the
    arm (not a budget) labels the paired unit. Fails closed on non-Train scope, reserved
    Test seeds, seed overlap, duplicate arms or clock mismatch. A seed block is used only
    when both arms are complete; otherwise the whole block is excluded and recorded.
    """
    directory = Path(directory)
    document = json.loads((directory / "interventions.json").read_text())
    telemetry = [json.loads(line) for line in
                 (directory / "gateway-telemetry.jsonl").read_text().splitlines() if line.strip()]
    episodes, extra = parse_interventions(document, telemetry, forbidden_seeds)
    return episodes, dict(
        directory=str(directory),
        interventions_sha256=hashlib.sha256((directory / "interventions.json").read_bytes()).hexdigest(),
        telemetry_sha256=hashlib.sha256((directory / "gateway-telemetry.jsonl").read_bytes()).hexdigest(),
        **extra)


def parse_interventions(document, telemetry, forbidden_seeds=()):
    # Free-text scope may legitimately say "no calibration acceptance"; splits must be exactly Train.
    if str(document.get("split", "train")).lower() != "train":
        raise ValueError("non_train_intervention_dataset")
    complete, incomplete, seen = defaultdict(dict), [], set()
    for e in document["episodes"]:
        seed, arm = e["seed_block"], e["arm"]
        if seed in RESERVED_TEST_SEEDS or seed in set(forbidden_seeds):
            raise ValueError("forbidden_or_overlapping_seed")
        if str(e.get("split")).lower() != "train":
            raise ValueError("non_train_intervention_episode")
        if (seed, arm) in seen:
            raise ValueError("duplicate_intervention_arm")
        seen.add((seed, arm))
        if not all(e.get(k) for k in ("initial", "final", "rows", "native_frames")):
            incomplete.append(dict(seed=seed, arm=arm, reason="missing initial/final/rows/native_frames"))
            continue
        run_id, clock = e["initial"]["run_id"], e["initial"]["clock_domain"]
        if any(r["clock_domain"] != clock for r in e["rows"]) or any(
                f["clock_domain"] != clock for f in e["native_frames"]):
            raise ValueError("clock_domain_mismatch")
        status = Counter(r["status"] for r in e["rows"])
        summary = e.get("summary")
        if summary is not None and (summary["offered"], summary["complete"], summary["rejected"],
                                    summary.get("other", 0)) != (
                len(e["rows"]), status["complete"], status["rejected"],
                len(e["rows"]) - status["complete"] - status["rejected"]):
            raise ValueError("intervention_summary_disagrees_with_rows")
        if e["initial"].get("reserved", 0) or e["final"].get("reserved", 0) or not all(
                d.get("drained") for d in e.get("initial_drain", []) + e.get("final_drain", [])):
            raise ValueError("intervention_episode_not_drained")
        start = e.get("start", e["initial"]["sampled_at"])
        end = e["final"]["sampled_at"]
        gateway = {rid: r for rid, r in gateway_requests(telemetry, run_id).items()
                   if r.get("accepted") is not None and start <= r["accepted"] <= end}
        complete[seed][arm] = dict(
            seed=seed, budget=arm if str(arm).startswith("arm") else f"arm{arm}", prompt_tokens=e.get("prompt_tokens"),
            max_tokens=e.get("max_tokens"), start=start, rows=e["rows"],
            frames=sorted(e["native_frames"], key=lambda f: f["received_at"]), gateway=gateway,
            dataset="train-interventions", source_script=e.get("source_script"),
            arm=arm, episode_id=e.get("episode_id"), cell=e.get("cell"),
            accounting_budget="varies within episode (see schedule_budgets)",
            schedule_budgets=sorted({step["budget"] for step in e.get("schedule", [])}))
    episodes, excluded_blocks = [], []
    for seed in sorted(complete):
        if len(complete[seed]) == 2:
            episodes.extend(complete[seed][arm] for arm in sorted(complete[seed]))
        else:
            excluded_blocks.append(dict(seed=seed, reason="fewer than two complete arms"))
    excluded_blocks += [dict(seed=seed, reason="no complete arm")
                        for seed in sorted({i["seed"] for i in incomplete} - set(complete))]
    ids = [r["request_id"] for e in episodes for r in e["rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_intervention_request_id")
    return episodes, dict(
        incomplete_episodes=incomplete, excluded_seed_blocks=excluded_blocks,
        gateway_accepted_events_total=sum(1 for f in telemetry if f.get("event") == "accepted"))


def run(archive, implementation_hash, interventions_dir=None):
    if implementation_hash != FROZEN_IMPLEMENTATION_HASH:
        raise ValueError("implementation_hash_changed")
    episodes, extra = load(archive)
    if len(episodes) != 8 or sorted(Counter(e["seed"] for e in episodes).values()) != [2] * 4:
        raise ValueError("expected_four_paired_seed_blocks")
    scout_episodes = len(episodes)
    intervention_extra = None
    if interventions_dir is not None:
        added, intervention_extra = load_interventions(
            interventions_dir, forbidden_seeds={e["seed"] for e in episodes})
        if not added:
            raise ValueError("no_complete_intervention_seed_blocks")
        episodes = episodes + added
    samples, accounting = build_samples(episodes)
    joined = sum(a["gateway_admitted_joined"] for a in accounting[:scout_episodes])
    result = dict(
        schema="native-prediction-diagnostic-v1" if interventions_dir is None
        else "native-prediction-diagnostic-combined-train-v1",
        scope="offline retrospective Train-only leave-one-seed-block-out diagnostic",
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_hash=implementation_hash,
        model_accepted=False, calibration_generated=False, native_test_generated=False,
        online_issued_forecasts=False, causal_control_model_identified=False,
        prediction_origin="native frame received_at (client monotonic, same clock domain)",
        labels=dict(first_content="client HTTP first_content_at - t0, only while Gateway has "
                    "not observed first content by t0",
                    terminal="client HTTP terminal_at - t0 for Gateway-admitted in-flight requests"),
        features=list(FEATURES), ridge_relative=RIDGE,
        fit="non-negative least squares, fixed ridge on non-intercept coefficients, request-weighted",
        baselines=dict(
            constant="training request-weighted mean remaining residence",
            elapsed="max(training mean total residence since Gateway accept - elapsed, 0)",
            persistence="max(latest realized Gateway residence in this episode before t0 - elapsed, 0);"
                        " training mean total residence if none realized yet",
            queue="NNLS a + b*(native running + waiting)"),
        num_computed_tokens_used=False,
        feature_availability=dict(
            native_frame="native snapshot sampled_at <= frame received_at (t0); received_at is the "
                         "asynchronous client receipt time, not a synchronized Gateway decision time",
            gateway_events="UNVERIFIED oracle retrospective reconstruction: Gateway telemetry effective "
                           "timestamps <= t0 do not establish that a client/controller could consume "
                           "those events at t0; no Gateway-snapshot receipt times were recorded. These "
                           "are not qualified online-available causal inputs.",
            online_qualification_requirement="future collection must record Gateway-snapshot receipt "
                                             "times; until then this diagnostic remains unaccepted"),
        feature_value_ranges={t: {name: [float(min(s["x"][i] for s in samples if s["target"] == t)),
                                         float(max(s["x"][i] for s in samples if s["target"] == t))]
                                  for i, name in enumerate(FEATURES)} for t in TARGETS},
        coefficient_note="A zero/negligible fitted coefficient (e.g. own_cap_x_native_decode_batch) "
                         "means the constrained fit assigned it no weight; the feature values themselves "
                         "are not always zero (see feature_value_ranges).",
        eligibility="Residence samples cover only Gateway-admitted requests in flight at a frame "
                    "(first-content additionally only before Gateway first content); rejected and "
                    "between-frame requests are counted, not predicted.",
        exclusions=dict(extra, gateway_accepted_events_outside_episode_windows=(
            extra["gateway_accepted_events_total"] - joined),
            rejected_requests="no residence label; excluded from residence samples, retained in counts",
            per_episode=accounting),
        sample_counts={t: sum(s["target"] == t for s in samples) for t in TARGETS},
        leave_one_seed_block_out={}, in_sample_all_blocks={},
        **({} if intervention_extra is None else dict(
            interventions=dict(intervention_extra, gateway_admitted_joined=sum(
                a["gateway_admitted_joined"] for a in accounting[scout_episodes:])),
            seed_blocks=sorted({e["seed"] for e in episodes}),
            combined_limitations=[
                "Intervention arms are schedule arms (budget varies inside an episode); arm labels, "
                "not fixed budgets, define the pair held out with each seed block.",
                "Scout cells are each covered by one scout seed; whether a held scout block has an "
                "independent same-cell seed depends on the intervention cells (see per_episode).",
                "Eligible admitted requests only; outcomes of rejected offers are not predicted.",
                "Native frame received_at timestamps are asynchronous client receipt times.",
                "Train-only; no Calibration/Validation/Test data are read or produced.",
            ])),
        limitations=[
            "Four independent seed blocks; each held block is also a different prompt/max-output "
            "cell, so every fold is an out-of-cell (OOD) extrapolation, not a held-out replicate.",
            "Samples within a request/frame/episode are strongly dependent; sample counts are not "
            "independent replicates. No standard errors, intervals or calibration are claimed.",
            "Error quantiles are descriptive held-out residuals, not statistically guaranteed bounds.",
            "Retrospective Train-only data under fixed budgets 16/32; no online-issued forecasts, "
            "no native Calibration/Test split, no causal effect of actions identified.",
            "Remaining output cap is an upper bound; natural EOS length is not observed at t0.",
            "Waiting requests are counts only in native frames; per-request waiting state is Gateway-side.",
            "Client-side labels include detokenisation/streaming delay after engine work.",
        ])
    for target in TARGETS:
        folds = leave_one_block_out(samples, target)
        result["leave_one_seed_block_out"][target] = dict(
            folds=folds,
            candidate_wins=sum(f["candidate_beats_best_baseline"] for f in folds),
            mean_block_request_weighted_mae_s={
                name: float(np.mean([f["errors"][name]["request_weighted_mae_s"] for f in folds]))
                for name in folds[0]["errors"]})
        rows = [s for s in samples if s["target"] == target]
        params = fit(rows)
        result["in_sample_all_blocks"][target] = dict(
            label="in-sample training fit; not a generalisation estimate",
            params=params, candidate_coefficients=dict(zip(FEATURES, params["candidate"])),
            errors=errors(rows, predict(params, rows)[0]))
    if intervention_extra is not None:
        result["limitations"][0] = (
            f"{len(result['seed_blocks'])} independent seed blocks (4 scout + Train interventions); "
            "see combined_support for whether each held block has same-cell training seeds.")
        result["limitations"][3] = (
            "Retrospective Train-only data: scout fixed budgets 16/32 plus intervention arms whose "
            "budget varies within episode; no online-issued forecasts, no native Calibration/Test "
            "split, no causal effect of actions identified.")
        result["combined_support"] = combined_support(samples, accounting, result)
    return result


def combined_support(samples, accounting, result):
    """Descriptive support overlap, worst biases, unpredicted requests and offer accounting."""
    support = {}
    for target in TARGETS:
        rows = [s for s in samples if s["target"] == target]
        folds = []
        for fold in result["leave_one_seed_block_out"][target]["folds"]:
            held = [s for s in rows if s["seed"] == fold["held_seed"]]
            train = [s for s in rows if s["seed"] != fold["held_seed"]]
            cells = {s["cell"] for s in held}
            X, T = np.array([s["x"] for s in held]), np.array([s["x"] for s in train])
            inside = (X >= T.min(0)) & (X <= T.max(0))
            folds.append(dict(
                held_seed=fold["held_seed"], held_cells=sorted(f"p{p}-m{m}" for p, m in cells),
                same_cell_train_seeds=sorted({s["seed"] for s in train if s["cell"] in cells}),
                held_samples_inside_train_feature_box=float(inside.all(1).mean()),
                per_feature_inside_fraction=dict(zip(FEATURES, inside.mean(0).tolist())),
                candidate_request_weighted_bias_s=fold["errors"]["candidate"]["request_weighted_bias_s"]))
        biases = [(abs(f["errors"][m]["request_weighted_bias_s"]), f["held_seed"], m,
                   f["errors"][m]["request_weighted_bias_s"])
                  for f in result["leave_one_seed_block_out"][target]["folds"] for m in f["errors"]]
        worst = max(biases)
        worst_candidate = max(b for b in biases if b[2] == "candidate")
        support[target] = dict(
            folds=folds,
            worst_bias=dict(seed=worst[1], model=worst[2], request_weighted_bias_s=worst[3]),
            worst_candidate_bias=dict(seed=worst_candidate[1], request_weighted_bias_s=worst_candidate[3]))
    total = Counter()
    for a in accounting:
        for key in ("offered", "complete", "rejected", "gateway_admitted_joined",
                    "first_content_admitted_never_in_risk_set", "terminal_admitted_never_in_risk_set"):
            total[key] += a[key]
    total["other"] = total["offered"] - total["complete"] - total["rejected"]
    return dict(
        per_target=support,
        offer_accounting=dict(total, note="every offer is complete or rejected; rejected offers "
                              "and admitted requests never in a risk set are counted, not predicted"),
        unpredicted_fast_first_content_requests=total["first_content_admitted_never_in_risk_set"])


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from mpc_lab.contracts import implementation_hash
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(root / "results/mpc-native-4090-scout/native-evidence.tar.gz"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--interventions-dir", help="optional Train intervention dataset directory "
                        "(interventions.json + gateway-telemetry.jsonl); written to a separate output")
    args = parser.parse_args()
    scout_output = root / "results/mpc-native-4090-scout/native-prediction-diagnostic.json"
    if args.interventions_dir and Path(args.output).resolve() == scout_output.resolve():
        parser.error("combined output must not overwrite the scout-only diagnostic")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_evidence import analyze
    before = implementation_hash()
    audit = analyze(args.evidence)  # inherit the established archive integrity assertions
    result = run(args.evidence, before, args.interventions_dir)
    if any(a["complete"] != a["gateway_admitted_joined"] for a in result["exclusions"]["per_episode"]):
        raise RuntimeError("admitted_requests_not_fully_joined")
    accounting = [a for a in result["exclusions"]["per_episode"] if "dataset" not in a]
    for key in ("complete", "rejected"):
        if audit[key] != sum(a[key] for a in accounting):
            raise RuntimeError("accounting_disagrees_with_evidence_audit")
    if audit["complete"] != sum(a["gateway_admitted_joined"] for a in accounting):
        raise RuntimeError("admitted_requests_not_fully_joined")
    result["evidence_audit"] = {k: audit[k] for k in ("physical_offers", "complete", "rejected")}
    if implementation_hash() != before:
        raise RuntimeError("implementation_hash_changed_during_run")
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(samples=result["sample_counts"], **{
        t: v["mean_block_request_weighted_mae_s"] | {"candidate_wins": v["candidate_wins"]}
        for t, v in result["leave_one_seed_block_out"].items()}), indent=2))
