"""Observed-isolated HTTP first-residence proxy, not a GPU kernel service law."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np
from scipy.optimize import nnls


def load_sources(archive, directories):
    sources = []
    with tarfile.open(archive,"r:gz") as bundle:
        def read(name):
            return bundle.extractfile("./"+name).read().decode()
        first = json.loads(read("train-saturation-scout-v1.json"))
        last = json.loads(read("train-saturation-scout-completion.json"))
        episodes = [e for e in first["episodes"] if "summary" in e]+last["episodes"]
        sources.append((episodes,[json.loads(line) for line in
                                  read("gateway-observed-telemetry.jsonl").splitlines()]))
    for directory in directories:
        directory = Path(directory)
        report_path = directory/"interventions.json"
        if not report_path.exists():
            report_path = directory/"isolated.json"
        report = json.loads(report_path.read_text())
        if not report.get("completed") or any(e["split"]!="train" for e in report["episodes"]):
            raise ValueError("only complete Train intervention sources")
        sources.append((report["episodes"],[json.loads(line) for line in
                       (directory/"gateway-telemetry.jsonl").read_text().splitlines()]))
    return sources


def extract(sources):
    rows,accounting = [],defaultdict(int)
    for episodes,events in sources:
        backend = {(e["run_id"],e["request_id"]):e["monotonic"]
                   for e in events if e.get("event")=="backend_sent"}
        by_run = defaultdict(list)
        for event in events:
            if event.get("event")=="mpc_ledger":
                by_run[event["run_id"]].append(event)
        for episode in episodes:
            seed = episode.get("seed",episode.get("seed_block"))
            if not (159901 <= seed <= 159904 or 160001 <= seed <= 160012):
                raise ValueError("unexpected non-Train seed")
            initial = episode["initial"]
            ledger = by_run[initial["run_id"]]
            times = [f["effective_at"] for f in ledger]
            if any(f["clock_domain"]!=initial["clock_domain"] for f in ledger):
                raise ValueError("ledger prefill clock mismatch")
            if times != sorted(times):
                raise ValueError("ledger chronology")
            if any(f["clock_domain"]!=initial["clock_domain"] for f in episode["native_frames"]):
                raise ValueError("native prefill clock mismatch")
            for row in episode["rows"]:
                accounting["offered"] += 1
                if row["status"]!="complete":
                    accounting["not_admitted_complete"] += 1
                    continue
                if row["clock_domain"] != initial["clock_domain"]:
                    raise ValueError("prefill clock mismatch")
                begin = backend.get((initial["run_id"],row["request_id"]))
                end = row["first_content_at"]
                if begin is None or not 0 <= begin <= end:
                    raise ValueError("missing backend-send label")
                if (episode.get("prelude") is True
                        or ("prelude" not in episode and begin < episode["start"]+1)):
                    accounting["warmup_excluded"] += 1
                    continue
                position = bisect_right(times,begin)-1
                window = ledger[position:bisect_right(times,end)]
                if (position < 0 or not window or any(f["clock_domain"]!=initial["clock_domain"]
                    for f in window)):
                    raise ValueError("unobserved prefill reservation interval")
                if any(f["running"]!=1 or f["complete"] is not True for f in window):
                    accounting["shared_reservation_excluded"] += 1
                    continue
                native = [f for f in episode["native_frames"] if begin <= f["sampled_at"] <= end]
                if any(f["running"]>1 or f["waiting"]>0 for f in native):
                    accounting["shared_native_excluded"] += 1
                    continue
                rows.append(dict(seed=seed,episode_id=episode.get("episode_id") or f'{seed}-{episode["budget"]}',
                    request_id=row["request_id"],prompt_tokens=row["usage"]["prompt_tokens"],
                    first_residence_s=end-begin,native_inside_first=len(native)))
                accounting["observed_isolated"] += 1
    return rows,dict(accounting)


def fit(rows):
    if len({r["prompt_tokens"] for r in rows}) < 2:
        raise ValueError("prefill design lacks two prompt levels")
    counts = defaultdict(int)
    for r in rows:
        counts[r["seed"]] += 1
    x = np.array([[1,r["prompt_tokens"]/1000] for r in rows],float)
    y = np.array([r["first_residence_s"] for r in rows])
    weights = np.array([1/counts[r["seed"]] for r in rows])
    coefficients,_ = nnls(x*np.sqrt(weights[:,None]),y*np.sqrt(weights))
    return dict(first_overhead_s=float(coefficients[0]),
                prefill_s_per_token=float(coefficients[1]/1000))


def serial_capacity_screen(parameters, sources):
    """Falsify interpreting the full HTTP proxy as exclusive FIFO service."""
    episodes = []
    for group,_ in sources:
        for episode in group:
            if "schedule" not in episode:
                continue
            initial = episode["initial"]
            if initial["requests"] or not all(d["drained"] for d in episode["initial_drain"]):
                raise ValueError("capacity screen requires empty initial state")
            prompt = episode["prompt_tokens"]
            service = parameters["first_overhead_s"]+parameters["prefill_s_per_token"]*prompt
            if service <= 0:
                raise ValueError("invalid serial service")
            completed = 0
            for row in episode["rows"]:
                if row["clock_domain"] != initial["clock_domain"]:
                    raise ValueError("capacity screen clock mismatch")
                if row["status"]=="complete" and row["terminal_at"] <= episode["start"]+90:
                    completed += 1
            bound = int(90/service)
            episodes.append(dict(episode_id=episode["episode_id"],window_s=90,
                prompt_tokens=prompt,complete_inside_window=completed,
                exclusive_serial_first_service_s=service,serial_complete_upper_bound=bound,
                violates_bound=completed>bound))
    return dict(interpretation="entire HTTP first-residence proxy used as exclusive serial prefill service",
        rejected=any(e["violates_bound"] for e in episodes),episodes=episodes,
        limitation="Not a rejection of MPC or an estimate of GPU prefill service; HTTP overhead and batching are not separated.")


def analyze(archive,directories):
    sources = load_sources(archive,directories)
    rows,accounting = extract(sources)
    folds = []
    for seed in sorted({r["seed"] for r in rows}):
        train = [r for r in rows if r["seed"]!=seed]
        held = [r for r in rows if r["seed"]==seed]
        try:
            parameters = fit(train)
        except ValueError as exc:
            folds.append(dict(held_seed=seed,requests=len(held),identifiable=False,reason=str(exc)))
            continue
        errors = [parameters["first_overhead_s"]+
                  parameters["prefill_s_per_token"]*r["prompt_tokens"]-r["first_residence_s"]
                  for r in held]
        folds.append(dict(held_seed=seed,requests=len(held),
            mae_s=statistics.mean(abs(e) for e in errors),bias_s=statistics.mean(errors)))
    try:
        parameters = fit(rows)
    except ValueError:
        parameters = None
    return dict(schema="native-observed-isolated-prefill-proxy-v1",parameters=parameters,
        identifiable=parameters is not None,
        serial_interpretation_screen=serial_capacity_screen(parameters,sources) if parameters else None,
        accounting=accounting,folds=folds,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,native_calibration_accepted=False,performance_accepted=False,
        limitations=["Backend send to client content includes HTTP/front-end overhead, not pure GPU time.",
                     "Isolation is sealed Gateway reservation count plus observed native counts; unsampled hidden work is unproven.",
                     "Intervention first1s and predeclared isolated preludes excluded; no clipping adverse labels.",
                     "Retrospective Train-only proxy; paired-seed holdout, no live prediction or SLO guarantee.",
                     "Two prompt levels; linear interpolation outside exact128/4096 support unvalidated."])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence",required=True)
    parser.add_argument("--interventions-dir",action="append",default=[])
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = analyze(args.evidence,args.interventions_dir)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"parameters":result["parameters"],"accounting":result["accounting"]},indent=2))
