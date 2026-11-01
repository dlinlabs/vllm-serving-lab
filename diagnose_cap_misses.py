"""Request- and scrape-level associations behind cap-32 TTFT misses (development diagnosis, CPU only).

Makes the analysis behind the PR #30 follow-up comment reproducible from the public
evidence bundles and repeats it on the new host. Everything here is a descriptive
association on development captures. Nothing identifies a per-step attention cost, an
exclusive cause, or a counterfactual cap effect.

* stages: per-capture, per-segment admission, TTFT and gateway-observed lifecycle spans;
* misses: SLO-miss rate by a request's own prompt length and by recent load, the prompt
  tokens of *other* requests accepted in the preceding second;
* exporter: vLLM queue/prefill histogram increments (not the same cohorts as requests);
* step: frozen linear step-model residual versus a sum-of-squared-prompt-length proxy,
  then refits that separate *when* prefill work happens (linear request-timed terms) from
  super-linear prompt cost, with in-sample, leave-one-capture-out and cross-host scores;
* decomposition: the frozen #30 goodput error split into admission and SLO-hit terms,
  with the archived oracle-arrival diagnostics.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

import greybox_simulator as gb

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'results/low-cap-frozen-20261021'
SEGMENTS = ((60., 120.), (120., 180.), (180., 240.), (240., 300.))
LENGTH_BINS = (0, 1000, 2000, 4000, 8193)
SHORT, RECENT, BUSY_RUNNING = 2000, 1.0, 4
QUANTILES = (.25, .5, .75, .9)
HASHED = ('manifest.json', 'replay_plan.json', 'client.json', 'gateway.jsonl',
          'engine_observations.jsonl', 'trace.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rounded(value):
    """JSON-safe, platform-stable values: six significant digits, NaN as null."""
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else float(f'{float(value):.6g}')
    return value


def discover(suite):
    """Full-length captures in a suite directory; smoke and short runs are excluded."""
    out = []
    for path in sorted(Path(suite).iterdir()):
        if not ((path / 'manifest.json').exists() and (path / 'replay_plan.json').exists()):
            continue
        if 'smoke' in path.name or read(path / 'replay_plan.json').get('duration_seconds', 0) < 300:
            continue
        out.append(path)
    if not out:
        raise ValueError('No full-length captures in ' + str(suite))
    return out


def load(path):
    """gb.load_capture, keeping identified transport failures.

    Same identity rule as evaluate_low_cap_frozen.load_capture_compatible, but the reader
    copies inputs instead of symlinking them so it also runs on Windows.
    """
    path = Path(path)
    client = read(path / 'client.json')
    patched = 0
    for row in client['requests']:
        if row.get('workload') != 'trace' or row.get('request_id'):
            continue
        if row.get('success') or row.get('rejected') or not row.get('client_request_id'):
            raise ValueError('Missing identity is allowed only for an identified transport failure')
        row['request_id'] = row['client_request_id']
        patched += 1
    if not patched:
        return (*gb.load_capture(path), 0)
    with tempfile.TemporaryDirectory(prefix='cap-miss-') as directory:
        reader = Path(directory)
        for name in ('manifest.json', 'replay_plan.json', 'gateway.jsonl'):
            shutil.copyfile(path / name, reader / name)
        (reader / 'client.json').write_text(json.dumps(client))
        meta, table = gb.load_capture(reader)
    return meta, table, patched


def budget_at(schedule, t):
    at = np.array([s['at_seconds'] for s in schedule], float)
    level = np.array([s['budget'] for s in schedule], float)
    i = np.searchsorted(at, np.asarray(t, float), 'right') - 1
    return np.where(i >= 0, level[np.clip(i, 0, None)], np.nan)


def segment_of(t):
    t = np.asarray(t, float)
    seg = np.full(t.shape, -1)
    for k, (a, b) in enumerate(SEGMENTS):
        seg[(t >= a) & (t < b)] = k
    return seg


def window_sum(times, values, lo, hi):
    """Sum of values whose finite time lies in [lo, hi), vectorized over intervals."""
    keep = np.isfinite(times)
    order = np.argsort(times[keep], kind='stable')
    t, v = times[keep][order], values[keep][order]
    cs = np.concatenate([[0.], np.cumsum(v)])
    return cs[np.searchsorted(t, hi, 'left')] - cs[np.searchsorted(t, lo, 'left')]


def recent_tokens(acc, tokens, window=RECENT):
    """Prompt tokens of other admitted requests accepted in [acc - window, acc).

    Requests accepted at exactly the same instant (including the request itself) are
    excluded. Unadmitted requests get NaN.
    """
    acc = np.asarray(acc, float)
    out = window_sum(acc, np.asarray(tokens, float), acc - window, acc)
    return np.where(np.isfinite(acc), out, np.nan)


def capture(path):
    """Per-request arrays plus the raw scrape table for one capture."""
    meta, t, patched = load(path)
    lookup = {r['trace_id']: r['input_tokens'] for r in read(Path(path) / 'trace.json')['requests']}
    tokens = np.array([lookup[x] for x in t['trace_id']], float)
    return dict(
        name=Path(path).name, path=Path(path), meta=meta, patched=patched,
        levels=sorted({float(s['budget']) for s in meta['schedule']}),
        cap=budget_at(meta['schedule'], t['sched']), seg=segment_of(t['sched']),
        tokens=tokens, ok=t['admitted'] & t['success'], admitted=t['admitted'], met=t['met'],
        sched=t['sched'], recv=t['recv'], acc=t['acc'], first=t['first'], ttft=t['ttft'],
        recent=recent_tokens(t['acc'], tokens), scrapes=gb.load_scrapes(path, meta['start']))


def mean(x):
    x = np.asarray(x, float)
    return float(np.mean(x)) if len(x) else math.nan


def quantile(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else math.nan


def stages(c):
    """Per-segment admission, pass rate, TTFT and lifecycle spans for one capture."""
    window = (c['sched'] >= SEGMENTS[0][0]) & (c['sched'] < SEGMENTS[-1][1])
    span = SEGMENTS[-1][1] - SEGMENTS[0][0]
    rows = []
    for k, (a, b) in enumerate(SEGMENTS):
        m = c['ok'] & (c['seg'] == k)
        met, miss = c['met'][m], ~c['met'][m]
        rows.append(dict(
            segment=k + 1, cap=float(c['cap'][(c['sched'] >= a) & (c['sched'] < b)][0]),
            admitted_per_s=m.sum() / (b - a), pass_rate=mean(met),
            ttft_p50_s=quantile(c['ttft'][m], .5), ttft_p90_s=quantile(c['ttft'][m], .9),
            sched_to_recv_ms=1e3 * mean(c['recv'][m] - c['sched'][m]),
            recv_to_accept_ms=1e3 * mean(c['acc'][m] - c['recv'][m]),
            accept_to_first_ms=1e3 * mean(c['first'][m] - c['acc'][m]),
            admitted_prompt_tokens_per_s=c['tokens'][m].sum() / (b - a),
            mean_prompt_tokens=mean(c['tokens'][m]),
            missed_mean_prompt_tokens=mean(c['tokens'][m][miss]),
            met_mean_prompt_tokens=mean(c['tokens'][m][met])))
    return dict(capture=c['name'], levels=c['levels'],
                goodput_60_300=float(c['met'][window].sum()) / span, segments=rows)


def misses(captures, cap):
    """Miss rates for admitted, successful requests scheduled while the cap was in effect."""
    pick = lambda c, extra: c['ok'] & (c['cap'] == cap) & (c['seg'] >= 0) & extra
    cat = lambda key, extra: np.concatenate([c[key][pick(c, extra(c))] for c in captures])
    bins = list(zip(LENGTH_BINS, LENGTH_BINS[1:]))
    segments = []
    for k in range(len(SEGMENTS)):
        own = cat('tokens', lambda c: c['seg'] == k)
        met = cat('met', lambda c: c['seg'] == k)
        recent = cat('recent', lambda c: c['seg'] == k)
        if not len(met):
            continue
        segments.append(dict(
            segment=k + 1, n=len(met), misses=int((~met).sum()),
            short_share_of_misses=mean(own[~met] < SHORT) if (~met).any() else math.nan,
            miss_rate_by_own_tokens=[dict(lo=a, hi=b, n=int(((own >= a) & (own < b)).sum()),
                                          miss_rate=mean(~met[(own >= a) & (own < b)]))
                                     for a, b in bins],
            recent_tokens_mean=mean(recent), recent_tokens_p90=quantile(recent, .9)))
    own = cat('tokens', lambda c: True)
    met = cat('met', lambda c: True)
    recent = cat('recent', lambda c: True)
    engine = cat('first', lambda c: True) - cat('acc', lambda c: True)
    short = own < SHORT
    cuts = [quantile(recent[short], q) for q in (.25, .5, .75)]
    edges = [-math.inf, *cuts, math.inf]
    by_recent = []
    for lo, hi in zip(edges, edges[1:]):
        k = short & (recent > lo) & (recent <= hi)
        by_recent.append(dict(lo=lo, hi=hi, n=int(k.sum()), miss_rate=mean(~met[k])))
    by_length = []
    for a, b in bins:
        k = (own >= a) & (own < b)
        by_length.append(dict(lo=a, hi=b, n=int(k.sum()), accept_to_first_median_s=quantile(engine[k], .5),
                              accept_to_first_p90_s=quantile(engine[k], .9)))
    return dict(cap=cap, captures=[c['name'] for c in captures], n=len(met), segments=segments,
                short_miss_rate_by_recent_quartile=by_recent, accept_to_first_by_own_tokens=by_length)


def spread_sum(start, end, values, t0, t1):
    """Sum of values spread uniformly over each [start, end) span, overlapped with [t0, t1)."""
    ok = np.isfinite(start) & np.isfinite(end)
    s, v = start[ok], values[ok]
    e = np.maximum(end[ok], s + 1e-6)
    overlap = np.clip(np.minimum(e, t1[:, None]) - np.maximum(s, t0[:, None]), 0., None)
    return (overlap / (e - s)) @ v


def intervals(c, coef, strict=False):
    """Consecutive scrape intervals inside the evaluated window with at least one step.

    strict=False reproduces the posted analysis: failed samples are dropped first, so an
    interval may bridge one. strict=True never bridges a failed sample and also applies
    gb.step_intervals' 0.8-1.3 s duration and counter-continuity guards.
    """
    s = c['scrapes']
    rows = np.arange(len(s['t'])) if strict else np.flatnonzero(s['valid'] > 0)
    i, j = rows[:-1], rows[1:]
    d = lambda k: s[k][j] - s[k][i]
    t0, t1 = s['t'][i], s['t'][j]
    dt, steps, prompt, gen = t1 - t0, d('steps'), d('prompt'), d('gen')
    mid = (t0 + t1) / 2
    keep = (steps > 0) & (mid >= SEGMENTS[0][0]) & (mid < SEGMENTS[-1][1])
    if strict:
        keep &= s['valid'][i].astype(bool) & s['valid'][j].astype(bool) & (dt > .8) & (dt < 1.3)
        for k in ('steps', 'prompt', 'gen'):
            keep &= np.isfinite(s[k][i]) & np.isfinite(s[k][j]) & (d(k) >= 0)
        for k in ('epoch', 'identity'):
            keep &= (s[k][i] == s[k][j]) | (np.isnan(s[k][i]) & np.isnan(s[k][j]))
    t0, t1 = t0[keep], t1[keep]
    tok, sq = c['tokens'], c['tokens'] ** 2
    out = dict(
        dt=dt, steps=steps, prompt=prompt, gen=gen, mid=mid, running=s['running'][j],
        cap=budget_at(c['meta']['schedule'], mid), seg=segment_of(mid),
        queue_sum=d('request_queue_time_seconds_sum'), queue_n=d('request_queue_time_seconds_count'),
        prefill_sum=d('request_prefill_time_seconds_sum'), prefill_n=d('request_prefill_time_seconds_count'))
    out = {k: v[keep] for k, v in out.items()}
    out.update(tok_accepted=window_sum(c['acc'], tok, t0, t1), sq_accepted=window_sum(c['acc'], sq, t0, t1),
               tok_first=window_sum(c['first'], tok, t0, t1),
               tok_spread=spread_sum(c['acc'], c['first'], tok, t0, t1),
               sq_spread=spread_sum(c['acc'], c['first'], sq, t0, t1))
    out['linear'] = coef[0] * out['steps'] + coef[1] * out['gen'] + coef[2] * out['prompt']
    out['capture'] = np.array([c['name']] * len(t0), dtype=object)
    return out


def stack(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def exporter(rows):
    """vLLM queue/prefill histogram means by cap and segment (ratio of increments)."""
    out = []
    for cap in sorted(set(rows['cap'].tolist())):
        for k in range(len(SEGMENTS)):
            m = (rows['cap'] == cap) & (rows['seg'] == k)
            if not m.any():
                continue
            out.append(dict(
                cap=cap, segment=k + 1, intervals=int(m.sum()),
                queue_ms=1e3 * rows['queue_sum'][m].sum() / rows['queue_n'][m].sum(),
                prefill_ms=1e3 * rows['prefill_sum'][m].sum() / rows['prefill_n'][m].sum(),
                prompt_tokens_per_s=rows['prompt'][m].sum() / rows['dt'][m].sum(),
                linear_busy_fraction=rows['linear'][m].sum() / rows['dt'][m].sum()))
    return out


BASE = ('steps', 'gen', 'prompt')
# Interval wall time regressed on counter increments plus optional request-timed terms.
# tok_* terms are linear controls for *when* prefill work happens; sq_* terms add
# super-linear prompt cost. Only sq beyond its matching tok control suggests super-linearity.
STEP_MODELS = {
    'linear': BASE,
    'sq_accepted': BASE + ('sq_accepted',),
    'tok_accepted': BASE + ('tok_accepted',),
    'tok_sq_accepted': BASE + ('tok_accepted', 'sq_accepted'),
    'tok_spread': BASE + ('tok_spread',),
    'tok_sq_spread': BASE + ('tok_spread', 'sq_spread'),
}


def design(rows, model):
    return np.column_stack([rows[k] for k in STEP_MODELS[model]])


def fit(rows, model):
    return np.linalg.lstsq(design(rows, model), rows['dt'], rcond=None)[0]


def rmse(rows, coef, model):
    return float(np.sqrt(np.mean((rows['dt'] - design(rows, model) @ coef) ** 2)))


def busy(rows):
    m = rows['running'] >= BUSY_RUNNING
    return {k: v[m] for k, v in rows.items()}


def fit_report(rows, model):
    coef = fit(rows, model)
    terms = STEP_MODELS[model]
    out = dict(model=model, terms=list(terms), n=len(rows['dt']), rmse_s=rmse(rows, coef, model),
               coefficients=dict(zip(terms, coef)), nonnegative=bool((coef >= 0).all()))
    sq = [k for k in terms if k.startswith('sq_')]
    if sq:
        out['squared_term_s_for_one_8000_token_prompt'] = coef[terms.index(sq[0])] * 8000 ** 2
    return out


def loco(rows, model):
    """Leave-one-capture-out pooled RMSE, and captures where the model beats linear."""
    err, better, captures = [], 0, sorted(set(rows['capture'].tolist()))
    for name in captures:
        test = rows['capture'] == name
        train = {k: v[~test] for k, v in rows.items()}
        held = {k: v[test] for k, v in rows.items()}
        score = {m: rmse(held, fit(train, m), m) for m in ('linear', model)}
        err.append(held['dt'] - design(held, model) @ fit(train, model))
        better += int(score[model] < score['linear'])
    return dict(model=model, captures=len(captures), rmse_s=float(np.sqrt(np.mean(np.concatenate(err) ** 2))),
                captures_better_than_linear=better)


def residual_bins(rows, proxy='sq_accepted'):
    """Frozen linear residual (wall - predicted busy) by proxy quantile bins."""
    residual = rows['dt'] - rows['linear']
    cuts = [quantile(rows[proxy], q) for q in QUANTILES]
    edges = [-math.inf, *cuts, math.inf]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        k = (rows[proxy] > lo) & (rows[proxy] <= hi)
        out.append(dict(lo=lo, hi=hi, n=int(k.sum()), mean_residual_s=mean(residual[k]),
                        prompt_tokens_per_interval=mean(rows['prompt'][k])))
    return dict(proxy=proxy, mean_residual_s=mean(residual), bins=out)


def step_analysis(rows):
    return dict(
        intervals=len(rows['dt']),
        frozen_linear_rmse_s=float(np.sqrt(np.mean((rows['dt'] - rows['linear']) ** 2))),
        prompt_counter_correlation=dict(
            (k, float(np.corrcoef(rows['prompt'], rows[k])[0, 1])) for k in ('tok_first', 'tok_accepted', 'tok_spread')),
        residual_by_proxy=[residual_bins(rows, p) for p in ('sq_accepted', 'sq_spread')],
        in_sample=[fit_report(rows, m) for m in STEP_MODELS],
        leave_one_capture_out=[loco(rows, m) for m in STEP_MODELS])


def cross_host(step_rows):
    return [dict(train=a, test=b, model=m, rmse_s=rmse(step_rows[b], fit(step_rows[a], m), m))
            for a in step_rows for b in step_rows if a != b for m in STEP_MODELS]


def decomposition(report_path):
    """Frozen #30 per-window goodput error, split exactly into admission and hit-rate terms.

    sim_g - act_g = (sim_adm - act_adm) * act_pass + sim_adm * (sim_pass - act_pass).
    """
    report = json.loads(gzip.decompress(Path(report_path).read_bytes()))
    oracle = {(o['run'], o['t']): o for o in report['oracle_diagnostics']}
    groups = defaultdict(list)
    for p in report['predictions']:
        groups[(int(p['cap']), None)].append(p)
        groups[(int(p['cap']), int((p['t'] - 60) // 60) + 1)].append(p)
    out = []
    for (cap, segment), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        adm, hit = [], []
        for x in rows:
            act_pass = x['actual_goodput'] / x['actual_admission'] if x['actual_admission'] else 0.
            sim_pass = x['simulator'] / x['admission'] if x['admission'] else 0.
            adm.append((x['admission'] - x['actual_admission']) * act_pass)
            hit.append(x['admission'] * (sim_pass - act_pass))
        o = [oracle[(x['run'], x['t'])] for x in rows]
        matched = sum(y['ttft_matched_n'] for y in o)
        sim_adm, act_adm = mean([x['admission'] for x in rows]), mean([x['actual_admission'] for x in rows])
        out.append(dict(
            cap=cap, segment=segment, n=len(rows),
            goodput_error=mean([x['simulator'] - x['actual_goodput'] for x in rows]),
            no_bias_goodput_error=mean([x['no_bias'] - x['actual_goodput'] for x in rows]),
            admission_term=mean(adm), hit_rate_term=mean(hit),
            simulated_pass_ratio=mean([x['simulator'] for x in rows]) / sim_adm,
            actual_pass_ratio=mean([x['actual_goodput'] for x in rows]) / act_adm,
            oracle_goodput_error=mean([y['goodput'] - x['actual_goodput'] for x, y in zip(rows, o)]),
            oracle_admission_error=mean([y['admission'] - x['actual_admission'] for x, y in zip(rows, o)]),
            oracle_matched_ttft_error_ms=1e3 * sum(y['ttft_signed_error_sum'] for y in o) / max(matched, 1),
            oracle_matched_requests=matched,
            online_bias_ms=1e3 * mean([x['online_bias_s'] for x in rows])))
    path = Path(report_path).resolve()
    source = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    return dict(source=source, source_sha256=sha(path), groups=out)


def frozen_coefficients(path):
    c = read(path)['frozen_step_coefficients']
    return c['step_s'], c['per_decode_token_s'], c['per_prompt_token_s']


def analyze(hosts, coef, report_path=None):
    result = dict(
        schema_version=1,
        scope='Development diagnosis on archived captures; associations only, not identified causes.',
        definitions=dict(
            segments='[60,120), [120,180), [180,240), [240,300) s of scheduled time',
            requests='admitted and successful trace requests; miss = TTFT > 0.5 s',
            recent_tokens=f'prompt tokens of other admitted requests accepted in the preceding {RECENT} s',
            short_prompt=f'own prompt < {SHORT} tokens',
            busy_interval=f'scrape interval with num_requests_running >= {BUSY_RUNNING} at its end',
            tok_accepted='prompt tokens of requests accepted (gateway) in the interval',
            sq_accepted='sum of squared prompt tokens of requests accepted in the interval (posted proxy)',
            tok_first='prompt tokens of requests whose gateway first content arrived in the interval',
            tok_spread='prompt tokens spread uniformly over each request\'s [accept, first content) span',
            sq_spread='squared prompt tokens spread the same way',
            step_models='interval wall time ~ no-intercept least squares on the listed terms',
            posted_intervals='failed scrapes dropped before differencing (as in the #30 comment)',
            strict_intervals='no bridging of failed scrapes; 0.8-1.3 s; counter continuity'),
        frozen_step_coefficients=dict(zip(('step_s', 'per_decode_token_s', 'per_prompt_token_s'), coef)),
        hosts={})
    step_rows = {}
    for label, suite in hosts.items():
        caps = [capture(p) for p in discover(suite)]
        fixed = [c for c in caps if len(c['levels']) == 1]
        with32 = [c for c in caps if 32. in c['levels']]
        host = dict(
            suite=Path(suite).name,
            captures=[dict(name=c['name'], levels=c['levels'], transport_identity_patches=c['patched'],
                           sha256={n: sha(c['path'] / n) for n in HASHED}) for c in caps],
            stages=[stages(c) for c in caps],
            misses_fixed=[misses([c for c in fixed if c['levels'] == [cap]], cap)
                          for cap in sorted({c['levels'][0] for c in fixed}) if cap in (16., 32.)],
            misses_all_cap32_segments=misses(with32, 32.) if with32 else None)
        modes = {}
        for strict in (False, True):
            parts = {c['name']: intervals(c, coef, strict) for c in caps}
            b = busy(stack(list(parts.values())))
            modes['strict' if strict else 'posted'] = dict(
                exporter_fixed=exporter(stack([parts[c['name']] for c in fixed])), busy=step_analysis(b))
            if not strict:
                step_rows[label] = b
        host['intervals'] = modes
        result['hosts'][label] = host
    result['cross_host_posted_busy'] = cross_host(step_rows) if len(step_rows) > 1 else []
    if report_path is not None:
        result['frozen_decomposition'] = decomposition(report_path)
    return rounded(result)


def host_arg(value):
    label, sep, path = value.partition('=')
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError('Use LABEL=SUITE_DIR')
    return label, Path(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--host', action='append', type=host_arg, required=True,
                   help='LABEL=SUITE_DIR, e.g. old=.../results/cap-range-v1 (repeatable)')
    p.add_argument('--frozen-summary', type=Path, default=FROZEN / 'summary.json')
    p.add_argument('--frozen-report', type=Path, default=FROZEN / 'full-report.json.gz')
    p.add_argument('--no-decomposition', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    hosts = dict(args.host)
    if len(hosts) != len(args.host):
        p.error('Duplicate host label')
    result = analyze(hosts, frozen_coefficients(args.frozen_summary),
                     None if args.no_decomposition else args.frozen_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
