"""Score the grey-box simulator on PR #26's archived five-second goodput origins (CPU only).

Reads the extracted cap-range and shared-client captures, checks every raw file used against the
SHA-256 recorded in the archived service review, and compares the simulator with that review's best
5 s model, a trailing-60 s nowcast and the replicate noise floor. Also reports the open-loop cap
curve and vLLM histogram means that limit the simulator as a cap-choice model.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import greybox_simulator as gb

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'results/service-review-20261020/full-report.json.gz'
MODEL = 'service_no_plan'
RAW = ('manifest.json', 'replay_plan.json', 'client.json', 'gateway.jsonl', 'engine_observations.jsonl')
CUT, NEAR, LOOKBACK = 200., 10., 60.
CAPS, HELD = (16, 24, 32, 40, 48, 64, 80, 96), (None, 16, 32, 64, 96)
REFERENCE = 'cap-range-v1-fixed-32-r1'
COMPARISONS = (('simulator', 'ridge'), ('simulator', 'nowcast'), ('nowcast', 'ridge'))
MODELS = ('ridge', 'nowcast', 'run_mean_oracle', 'oracle_arrivals_raw', 'oracle_arrivals_overhead',
          'oracle_arrivals_overhead_bias', 'simulator_no_bias', 'simulator')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def code_sha256(path):
    return digest(Path(path).read_bytes().replace(b'\r\n', b'\n'))


def clean(value):
    """JSON-safe copy: NaN/inf become null and numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def load_archive(path):
    raw = Path(path).read_bytes()
    report = json.loads(gzip.decompress(raw))
    hashes = {s['meta']['name']: s['meta']['raw_sha256'] for s in report['source_runs']}
    rows = [(r['run'], float(r['t_s']), float(r['actual']['cohort_ttft_met']),
             float(r['predictions'][MODEL]['cohort_ttft_met']))
            for r in report['chronological_predictions'] if r['horizon_s'] == gb.HORIZON]
    return digest(raw), hashes, rows


def find_run(roots, name, hashes):
    """The unique extracted capture directory for name, verified against the archived hashes."""
    found = {p.parent.resolve() for root in roots for p in Path(root).rglob('manifest.json')
             if p.parent.name == name and all((p.parent / f).is_file() for f in RAW)}
    if len(found) != 1:
        raise ValueError(f'{name}: expected one capture directory, found {len(found)}')
    root = found.pop()
    for f in RAW:
        if digest((root / f).read_bytes()) != hashes[f]:
            raise ValueError(f'{name}/{f} differs from the archived SHA-256')
    return root


def nowcast(req, t, lookback=LOOKBACK):
    """Met-SLO rate over the trailing window, counting only outcomes observable at t."""
    sel = ((req['sched'] >= t - lookback) & (req['sched'] < t - gb.SLO) & (req['ttft'] <= gb.SLO)
           & (req['c_first'] <= t))
    return sel.sum() / (lookback - gb.SLO)


def score(pred, actual, runs):
    error = pred - actual
    per_run, corr = {}, []
    for name in dict.fromkeys(runs):
        m = runs == name
        per_run[name] = float(np.abs(error[m]).mean())
        ok = pred[m].std() > 0 and actual[m].std() > 0
        corr.append(np.corrcoef(pred[m], actual[m])[0, 1] if ok else math.nan)
    return dict(mae=float(np.abs(error).mean()), bias=float(error.mean()),
                within_run_corr=float(np.nanmean(corr)), per_run_mae=per_run)


def paired_block_bootstrap(diff, runs, t, rng, reps, block):
    """Mean paired difference with a 95% CI from resampling fixed time blocks within each run."""
    groups = []
    for name in dict.fromkeys(runs):
        m = runs == name
        b = np.floor((t[m] - t[m].min()) / block).astype(int)
        counts = np.bincount(b)
        groups.append((np.bincount(b, weights=diff[m])[counts > 0], counts[counts > 0]))
    stats = np.empty(reps)
    for k in range(reps):
        total = n = 0.
        for sums, counts in groups:
            pick = rng.integers(0, len(sums), len(sums))
            total += sums[pick].sum()
            n += counts[pick].sum()
        stats[k] = total / n
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(delta_mae=float(diff.mean()), ci95=[float(lo), float(hi)], share_below_zero=float((stats < 0).mean()))


def changes(run):
    return [at for (at, cap), (_, prev) in zip(run.caps[1:], run.caps) if cap != prev]


def near_change(run, t):
    """A cap change from NEAR s before the origin through the end of the forecast window."""
    return any(t - NEAR <= at <= t + gb.HORIZON for at in changes(run))


def engine_summary(meta, req, scrapes, start=5., end=300.):
    """vLLM histogram means and counters between the scrapes nearest start and end."""
    i, j = (int(np.argmin(np.abs(scrapes['t'] - x))) for x in (start, end))
    delta = {k: scrapes[k][j] - scrapes[k][i] for k in scrapes}
    out = {h: delta[h + '_sum'] / delta[h + '_count'] for h in gb.HISTOGRAMS}
    out['preemptions'] = delta['preemptions']
    out['prefix_cache_queried_tokens'] = delta['prefix_queries']
    out['prefix_cache_hit_tokens'] = delta['prefix_hits']
    g = req['success'] & (req['acc'] > start) & (req['acc'] < end)
    out['gateway_ttft'] = float(np.mean(req['first'][g] - req['acc'][g]))
    out['created_at'] = meta['created']
    return out


def cap_curve(data, runs, held, reference=REFERENCE, caps=CAPS):
    """Whole-run counterfactuals from an idle start, with the held cap removed from every fit."""
    intervals = []
    for name, (_, _, scrapes) in data.items():
        i, run = gb.step_intervals(scrapes), runs[name]
        keep = np.array([run.cap(t - 1) != held and run.cap(t) != held for t in i['t']], bool)
        intervals.append({k: v[keep] for k, v in i.items()})
    coef = gb.fit_step_time(intervals)
    res = np.concatenate([gb.residuals(run, coef, np.arange(5, 296, 5.))[0] for run in runs.values()])
    if held is not None:
        res = res[res[:, 6] != held]
    overhead = gb.Overhead(res[:, 4], res[:, 3] - res[:, 2])
    ref = runs[reference]
    q = ref.req
    arrivals = sorted((s + ref.pre, s, c, e) for s, c, e in zip(q['sched'], q['cost'], q['est_in']))
    duration = 300.
    rows = []
    for cap in caps:
        sim = gb.simulate(gb.EmptyStart(cap), 0., arrivals, coef, horizon=duration)
        first = [(s, f) for s, a, f, _ in sim if a and math.isfinite(f)]
        rows.append(dict(held=held, cap=cap, goodput=gb.expected_met(sim, overhead) / duration,
                         goodput_step_only=sum(f - s <= gb.SLO for s, f in first) / duration,
                         admitted=len(first) / duration))
    return dict(held=held, step_coefficients=coef, overhead_at=dict(zip((10, 25, 45), overhead.mean([10, 25, 45]))),
                rows=rows)


def load(cap_range_dir, shared_client_dir, archive):
    archive_sha, hashes, rows = load_archive(archive)
    data, roots = {}, (cap_range_dir, shared_client_dir)
    for name in dict.fromkeys(r[0] for r in rows):
        root = find_run(roots, name, hashes[name])
        meta, req = gb.load_capture(root)
        data[name] = (meta, req, gb.load_scrapes(root, meta['start']))
    inputs = dict(archive_sha256=archive_sha, model=MODEL,
                  raw_sha256={n: {f: hashes[n][f] for f in RAW} for n in data})
    return data, rows, inputs


def evaluate(data, rows, boot_k=16, seed=1, reps=4000, block=10., pre_cut=CUT, curve=True):
    coef = gb.fit_step_time([gb.step_intervals(s, end=CUT) for _, _, s in data.values()])
    runs = {n: gb.Run(n, m, r, s, cut=pre_cut) for n, (m, r, s) in data.items()}
    res, sims = {}, {}
    for name, run in runs.items():
        res[name], sims[name] = gb.residuals(run, coef, np.arange(5, 296, 1.))
    pooled = np.concatenate(list(res.values()))
    train = pooled[pooled[:, 5] <= CUT]
    overhead = gb.Overhead(train[:, 4], train[:, 3] - train[:, 2])

    rng = np.random.default_rng(seed)
    table = {k: [] for k in ('run', 't', 'actual', 'online_bias') + MODELS if k != 'run_mean_oracle'}
    for name, t, met, ridge in rows:
        run = runs[name]
        r = run.req
        count = int(((r['sched'] >= t) & (r['sched'] < t + gb.HORIZON) & r['met']).sum())
        if count != met:
            raise ValueError(f'{name} t={t}: {count} met requests, archive has {met}')
        u = gb.online_bias(res[name], overhead, t)
        out = sims[name][t]
        boots = [gb.simulate(run, t, run.bootstrap_arrivals(t, rng), coef) for _ in range(boot_k)]
        # Every forecast is a rate in requests/s over the five-second cohort.
        values = dict(run=name, t=t, actual=count / gb.HORIZON, online_bias=u, ridge=ridge / gb.HORIZON,
                      nowcast=nowcast(r, t),
                      oracle_arrivals_raw=gb.expected_met(out) / gb.HORIZON,
                      oracle_arrivals_overhead=gb.expected_met(out, overhead) / gb.HORIZON,
                      oracle_arrivals_overhead_bias=gb.expected_met(out, overhead, u) / gb.HORIZON,
                      simulator_no_bias=np.mean([gb.expected_met(o, overhead) for o in boots]) / gb.HORIZON,
                      simulator=np.mean([gb.expected_met(o, overhead, u) for o in boots]) / gb.HORIZON)
        for k, v in values.items():
            table[k].append(v)
    names = np.array(table.pop('run'))
    p = {k: np.array(v, float) for k, v in table.items()}
    p['run_mean_oracle'] = np.array([p['actual'][names == n].mean() for n in names])
    t = p['t']

    scores = {m: score(p[m], p['actual'], names) for m in MODELS}
    boot_rng = np.random.default_rng(seed + 1)
    paired = {f'{a}_vs_{b}': paired_block_bootstrap(np.abs(p[a] - p['actual']) - np.abs(p[b] - p['actual']),
                                                   names, t, boot_rng, reps, block) for a, b in COMPARISONS}
    noise = {}
    for cap in sorted({int(runs[n].cap(0)) for n in runs if not changes(runs[n])}):
        reps_ = [n for n in runs if not changes(runs[n]) and runs[n].cap(0) == cap]
        if len(reps_) == 2:
            x, y = (p['actual'][names == n] for n in reps_)
            noise[str(cap)] = dict(runs=reps_, mae=float(np.abs(x - y).mean()), corr=float(np.corrcoef(x, y)[0, 1]))
    switching = np.array([bool(changes(runs[n])) for n in names])
    near = np.array([near_change(runs[n], x) for n, x in zip(names, t)]) & switching
    subsets = {}
    for label, mask in (('near_change', near), ('steady', switching & ~near)):
        subsets[label] = dict(n=int(mask.sum()), mae={m: float(np.abs(p[m][mask] - p['actual'][mask]).mean())
                                                      for m in ('ridge', 'nowcast', 'simulator')})

    fixed = {n: data[n] for n in runs if not changes(runs[n])}
    measured = [dict(run=n, cap=int(runs[n].cap(0)), goodput=float(r['met'].sum() / m['duration']),
                     admitted=float(r['admitted'].sum() / m['duration'])) for n, (m, r, _) in fixed.items()]
    engine = {n: engine_summary(*data[n]) for n in sorted(runs, key=lambda n: data[n][0]['created'])}

    report = dict(
        protocol=dict(horizon_s=gb.HORIZON, slo_s=gb.SLO, cut_s=CUT, batch_tokens=gb.B, output_tokens=gb.OUT,
                      boot_k=boot_k, seed=seed, block_s=block, bootstrap_reps=reps, nowcast_lookback_s=LOOKBACK,
                      near_change_s=NEAR, causal_pre=pre_cut is not None, reference_run=REFERENCE,
                      test_origins=len(rows), training='step fit t<200 s; overhead fit on first tokens <=200 s'),
        step_coefficients=dict(zip(('step_s', 'per_decode_token_s', 'per_prompt_token_s'), coef)),
        overhead=dict(knots=overhead.knots, mean=overhead.values, sigma_knots=overhead.sigma_knots,
                      sigma=overhead.sigma_values),
        actual_counts_match_archive=True,
        scores=scores, paired=paired, noise_floor=noise, cap_change_subsets=subsets,
        measured_fixed_caps=measured, engine=engine,
        cap_curves=[cap_curve(data, runs, h) for h in HELD] if curve else [],
        predictions=dict(run=names, **p))
    return clean(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cap-range-dir', required=True, type=Path, help='Extracted cap-range raw bundle')
    parser.add_argument('--shared-client-dir', required=True, type=Path, help='Extracted shared-client raw bundle')
    parser.add_argument('--archived-report', type=Path, default=ARCHIVE)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--boot-k', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--reps', type=int, default=4000)
    parser.add_argument('--block', type=float, default=10.)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a fresh output directory')
    data, rows, inputs = load(args.cap_range_dir, args.shared_client_dir, args.archived_report)
    report = evaluate(data, rows, args.boot_k, args.seed, args.reps, args.block)
    report['inputs'] = inputs
    report['code_sha256'] = {f: code_sha256(HERE / f) for f in ('greybox_simulator.py', 'evaluate_greybox.py')}
    raw = json.dumps(report, sort_keys=True).encode()
    compressed = gzip.compress(raw, mtime=0)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'full-report.json.gz').write_bytes(compressed)
    s = report['scores']
    summary = dict(
        full_report_sha256=digest(raw), compressed_report_sha256=digest(compressed),
        archive_sha256=inputs['archive_sha256'], code_sha256=report['code_sha256'],
        test_origins=report['protocol']['test_origins'], step_coefficients=report['step_coefficients'],
        mae={m: s[m]['mae'] for m in MODELS}, within_run_corr={m: s[m]['within_run_corr'] for m in MODELS},
        paired=report['paired'], noise_floor=report['noise_floor'], cap_change_subsets=report['cap_change_subsets'],
        cap_curve_all_caps=report['cap_curves'][0]['rows'] if report['cap_curves'] else [],
        measured_fixed_caps=report['measured_fixed_caps'])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    for m in MODELS:
        print(f"{m:30s} MAE {s[m]['mae']:.4f} bias {s[m]['bias']:+.3f} within-run corr {s[m]['within_run_corr']:.2f}")


if __name__ == '__main__':
    main()
