"""Step-level grey-box simulator for five-second cohort TTFT goodput (development analysis, CPU only).

Gateway: the exact admission rule (admit iff reserved + cost <= cap; release at completion).
Engine: vLLM-style continuous batching. Each step decodes one token for every running request and
fills the remaining token budget B with FIFO prefill; step time is c0 + c1*decode + c2*prefill
tokens, fitted from /metrics counters. A request emits its first token at the end of the step that
finishes its prompt and completes OUT-1 decode steps later (every successful capture output is
OUT tokens). TTFT excess that the step model does not explain is calibrated as a nondecreasing
function of the running count at admission, with an optional causal online bias.
"""
import json
import math
import re
from collections import deque
from pathlib import Path

import numpy as np

B, OUT, HORIZON, SLO = 8192, 128, 5.0, 0.5
TERMINAL = {'completed', 'failed', 'cancelled'}
EVENTS = {'received', 'accepted', 'rejected', 'first_content'} | TERMINAL
COUNTERS = {'steps': 'iteration_tokens_total_count', 'prompt': 'prompt_tokens_total',
            'gen': 'generation_tokens_total', 'running': 'num_requests_running',
            'waiting': 'num_requests_waiting', 'preemptions': 'num_preemptions_total',
            'prefix_queries': 'prefix_cache_queries_total', 'prefix_hits': 'prefix_cache_hits_total'}
HISTOGRAMS = ('time_to_first_token_seconds', 'request_queue_time_seconds',
              'request_prefill_time_seconds', 'request_decode_time_seconds')
SIGMA_BINS = (-1, 5, 10, 15, 20, 25, 30, 40, 50, 70)


def metric(raw, name):
    match = re.search(rf'^vllm:{name}\{{[^}}]*\}} (\S+)', raw, re.M)
    return float(match.group(1)) if match else math.nan


def number(value):
    return math.nan if value is None else float(value)


def load_capture(root):
    """Per-request table (seconds from measurement start) and run metadata for one capture."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    t0 = manifest['measurement_start']
    plan = json.loads((root / 'replay_plan.json').read_text())
    gateway = {}
    for line in (root / 'gateway.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        rid, kind = event.get('request_id'), event.get('event')
        if rid is None or kind not in EVENTS:
            continue
        g = gateway.setdefault(rid, {})
        g[kind] = event['monotonic'] - t0
        if kind in ('accepted', 'rejected'):
            g['cost'], g['est_in'] = event.get('request_cost'), event.get('estimated_input_tokens')
        if kind in TERMINAL:
            g['done'] = event['monotonic'] - t0
    rows = []
    for c in json.loads((root / 'client.json').read_text())['requests']:
        if c.get('workload') != 'trace':
            continue
        g = gateway.get(c['request_id'], {})
        first = c.get('first_content_at')
        rows.append((c['scheduled_at'] - t0, g.get('received', math.nan), g.get('accepted', math.nan),
                     g.get('first_content', math.nan), g.get('done', math.nan), number(g.get('cost')),
                     number(g['est_in'] if 'est_in' in g else c.get('expected_input_tokens')),
                     bool(c.get('success')) and bool(c.get('stream_complete')),
                     first - c['scheduled_at'] if first is not None else math.nan,
                     first - t0 if first is not None else math.nan, c['trace_id']))
    rows.sort(key=lambda r: r[0])
    keys = ('sched', 'recv', 'acc', 'first', 'done', 'cost', 'est_in', 'success', 'ttft', 'c_first', 'trace_id')
    table = {k: np.array([r[i] for r in rows], dtype=object if k == 'trace_id' else bool if k == 'success' else float)
             for i, k in enumerate(keys)}
    table['admitted'] = ~np.isnan(table['acc'])
    table['met'] = table['success'] & (table['ttft'] <= SLO)
    meta = dict(start=t0, created=manifest.get('created_at'), duration=plan.get('duration_seconds', 300.),
                schedule=plan['budget_schedule'])
    return meta, table


def load_scrapes(root, t0):
    """Counters and histogram sums/counts from every vLLM /metrics scrape, sorted by send time."""
    rows = []
    for line in (Path(root) / 'engine_observations.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        raw = event.get('raw_metrics')
        if not raw:
            continue
        row = dict(t=event['sent_at'] - t0, recv=event.get('received_at', event['sent_at']) - t0)
        row.update({k: metric(raw, name) for k, name in COUNTERS.items()})
        for name in HISTOGRAMS:
            row[name + '_sum'], row[name + '_count'] = metric(raw, name + '_sum'), metric(raw, name + '_count')
        rows.append(row)
    order = sorted(range(len(rows)), key=lambda i: rows[i]['t'])
    return {k: np.array([rows[i][k] for i in order], float) for k in rows[0]}


def step_intervals(scrapes, start=3., end=300.):
    """Roughly one-second busy intervals between consecutive scrapes."""
    d = {k: np.diff(scrapes[k]) for k in ('t', 'steps', 'prompt', 'gen')}
    t = scrapes['t'][1:]
    ok = (t > start) & (t < end) & (d['steps'] > 0) & (d['t'] > .8) & (d['t'] < 1.3)
    return dict(t=t[ok], dt=d['t'][ok], steps=d['steps'][ok], prompt=d['prompt'][ok], gen=d['gen'][ok])


def fit_step_time(intervals):
    """Least squares dt = c0*steps + c1*generated + c2*prompt tokens, without intercept."""
    x = np.concatenate([np.c_[i['steps'], i['gen'], i['prompt']] for i in intervals])
    y = np.concatenate([i['dt'] for i in intervals])
    return tuple(float(c) for c in np.linalg.lstsq(x, y, rcond=None)[0])


class Run:
    """Observed state of one capture; every origin view uses only events known at that origin."""

    def __init__(self, name, meta, requests, scrapes, cut=None):
        self.name, self.req = name, requests
        self.caps = sorted((s['at_seconds'], s['budget']) for s in meta['schedule'])
        adm = requests['admitted']
        order = np.argsort(requests['acc'][adm], kind='stable')
        a = {k: v[adm][order] for k, v in requests.items()}
        a['first_f'] = np.where(np.isnan(a['first']), a['done'], a['first'])
        a['done_f'] = np.where(np.isnan(a['done']), np.inf, a['done'])
        self.adm = a
        known = adm & (requests['acc'] <= (math.inf if cut is None else cut))
        self.pre = float(np.median(requests['acc'][known] - requests['sched'][known]))
        self.decision = np.where(adm, requests['acc'], requests['recv'])
        self.e_recv, self.e_t, self.e_steps = scrapes['recv'], scrapes['t'], scrapes['steps']

    def cap(self, t):
        cap = self.caps[0][1]
        for at, budget in self.caps:
            if at <= t:
                cap = budget
        return cap

    def steps_between(self, a, t):
        """Engine steps in [a, t] from cumulative step counters received by t."""
        k = np.searchsorted(self.e_recv, t, side='right')
        if k < 2:
            return 0.
        et, es = self.e_t[:k], self.e_steps[:k]
        rate = (es[-1] - es[-2]) / max(et[-1] - et[-2], 1e-3)
        f = lambda x: np.interp(x, et, es) if x <= et[-1] else es[-1] + rate * (x - et[-1])
        return max(f(t) - f(a), 0.)

    def origin(self, t):
        a = self.adm
        inflight = (a['acc'] <= t) & (a['done_f'] > t)
        streaming, awaiting = inflight & (a['first_f'] <= t), inflight & (a['first_f'] > t)
        rem = [max(OUT - 1 - self.steps_between(f, t), 1.) for f in a['first_f'][streaming]]
        running = list(zip(rem, a['cost'][streaming]))
        queue = deque([float(tok), float(c), math.nan, math.nan]
                      for tok, c in zip(a['est_in'][awaiting], a['cost'][awaiting]))
        return running, queue, float(np.nansum(a['cost'][inflight]))

    def oracle_arrivals(self, t, horizon=HORIZON):
        """Measured future arrivals: a non-deployable diagnostic."""
        r = self.req
        sel = (r['sched'] < t + horizon) & (self.decision > t)
        return sorted((max(t, s + self.pre), s, c, e) for s, c, e in zip(r['sched'][sel], r['cost'][sel], r['est_in'][sel]))

    def bootstrap_arrivals(self, t, rng, lookback=60., horizon=HORIZON):
        """Replay one random past window of arrivals already decided by the origin."""
        r = self.req
        u = rng.uniform(t - lookback, t - horizon)
        sel = (r['sched'] >= u) & (r['sched'] < u + horizon) & (self.decision <= t)
        return sorted((s - u + t + self.pre, s - u + t, c, e)
                      for s, c, e in zip(r['sched'][sel], r['cost'][sel], r['est_in'][sel]))


class EmptyStart:
    """Idle engine and gateway at a fixed cap, for whole-run counterfactuals."""

    def __init__(self, cap):
        self.fixed = cap

    def origin(self, t):
        return [], deque(), 0.

    def cap(self, t):
        return self.fixed


def simulate(run, t, arrivals, coef, cap_fn=None, horizon=HORIZON, extra=SLO + .3):
    """Return (sched, admitted, simulated first-token time, running count at admission) per cohort arrival.

    admitted is None when the arrival is not reached before the simulation ends.
    """
    c0, c1, c2 = coef
    cap_fn = cap_fn or run.cap
    running, queue, reserved = run.origin(t)
    rem = np.array([r for r, _ in running], dtype=float)
    cost = np.array([c for _, c in running], dtype=float)
    out, ai, tau, end = [], 0, t, t + horizon + extra
    na = len(arrivals)
    while tau < end:
        if not len(rem) and not queue:
            if ai >= na:
                break
            tau = max(tau, arrivals[ai][0])
        budget = B - len(rem)
        p, finished = 0., []
        for item in queue:
            if budget <= 0:
                break
            take = min(budget, item[0])
            item[0] -= take
            p += take
            budget -= take
            if item[0] <= 0:
                finished.append(item)
        for _ in finished:
            queue.popleft()
        tau_end = tau + c0 + c1 * len(rem) + c2 * p
        # Arrivals during a step see reservations before that step's releases.
        while ai < na and arrivals[ai][0] < tau_end:
            at, sched, c, tok = arrivals[ai]
            ai += 1
            cohort = t <= sched < t + horizon
            if reserved + c <= cap_fn(at) + 1e-9:
                reserved += c
                queue.append([float(tok), float(c), sched if cohort else math.nan, len(rem)])
            elif cohort:
                out.append((sched, False, math.nan, math.nan))
        rem = rem - 1
        done = rem <= 0
        if done.any():
            reserved -= cost[done].sum()
            rem, cost = rem[~done], cost[~done]
        if finished:
            rem = np.concatenate([rem, np.full(len(finished), OUT - 1.)])
            cost = np.concatenate([cost, [f[1] for f in finished]])
            for f in finished:
                if not math.isnan(f[2]):
                    out.append((f[2], True, tau_end, f[3]))
        tau = tau_end
    for item in queue:
        if not math.isnan(item[2]):
            out.append((item[2], True, math.inf, item[3]))
    while ai < na:
        at, sched, c, tok = arrivals[ai]
        ai += 1
        if t <= sched < t + horizon:
            out.append((sched, None, math.nan, math.nan))
    return out


def isotonic(x, y):
    """Least-squares nondecreasing fit (pool adjacent violators) on tie-averaged points."""
    ux, inverse = np.unique(np.asarray(x, float), return_inverse=True)
    weight = np.bincount(inverse).astype(float)
    mean = np.bincount(inverse, weights=np.asarray(y, float)) / weight
    values, weights, sizes = [], [], []
    for v, w in zip(mean, weight):
        values.append(v), weights.append(w), sizes.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, s2 = values.pop(), weights.pop(), sizes.pop()
            v1, w1, s1 = values.pop(), weights.pop(), sizes.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2)), weights.append(w1 + w2), sizes.append(s1 + s2)
    return ux, np.repeat(values, sizes)


class Overhead:
    """Nondecreasing TTFT excess and robust spread as functions of running count at admission."""

    def __init__(self, n, residual):
        n, residual = np.asarray(n, float), np.asarray(residual, float)
        self.knots, self.values = isotonic(n, np.clip(residual, -1, 2))
        error = residual - self.mean(n)
        mids, spreads = [], []
        for lo, hi in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
            e = error[(n > lo) & (n <= hi)]
            if len(e):
                mids.append((lo + hi) / 2)
                spreads.append(1.4826 * np.median(np.abs(e - np.median(e))))
        self.sigma_knots, self.sigma_values = isotonic(mids, spreads)

    def mean(self, n):
        return np.interp(n, self.knots, self.values)

    def sigma(self, n):
        return np.interp(n, self.sigma_knots, self.sigma_values)


def normal_cdf(z):
    return np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in np.atleast_1d(z)])


def expected_met(out, overhead=None, bias=0.):
    """Expected count of simulated cohort requests with TTFT <= SLO."""
    kept = [(s, f, n) for s, a, f, n in out if a and math.isfinite(f)]
    if not kept:
        return 0.
    s, f, n = (np.array(v, float) for v in zip(*kept))
    if overhead is None:
        return float(normal_cdf((SLO - (f - s + bias)) / .05).sum())
    return float(normal_cdf((SLO - (f - s + overhead.mean(n) + bias)) / overhead.sigma(n)).sum())


def residuals(run, coef, origins):
    """Oracle-arrival simulations and measured-minus-simulated TTFT for admitted requests.

    Columns: origin, sched, simulated TTFT, measured TTFT, running count, client first-content time,
    cap at acceptance.
    """
    index = {s: i for i, s in enumerate(run.req['sched'])}
    rows, sims = [], {}
    for t in origins:
        out = simulate(run, t, run.oracle_arrivals(t), coef)
        sims[float(t)] = out
        for s, a, f, n in out:
            if a and math.isfinite(f):
                i = index[s]
                if run.req['admitted'][i] and math.isfinite(run.req['ttft'][i]):
                    rows.append((t, s, f - s, run.req['ttft'][i], n, run.req['c_first'][i], run.cap(run.req['acc'][i])))
    return np.array(rows, float).reshape(-1, 7), sims


def online_bias(res, overhead, t, k0=20., lookback=60.):
    """Shrunken mean of recent TTFT residuals whose first token was observed by t."""
    o = res[:, 0]
    sel = (o <= t - HORIZON) & (o >= t - lookback) & (np.mod(o - t, HORIZON) == 0) & (res[:, 5] <= t)
    if not sel.any():
        return 0.
    e = np.clip(res[sel, 3] - res[sel, 2] - overhead.mean(res[sel, 4]), -1, 1)
    return float(e.sum() / (len(e) + k0))
