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
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

B, OUT, HORIZON, SLO = 8192, 128, 5.0, 0.5
EXTRA = SLO + .3
SIGMA_MIN = 1e-3
TERMINAL = {'completed', 'failed', 'cancelled'}
EVENTS = {'received', 'accepted', 'rejected', 'first_content'} | TERMINAL
COUNTERS = {'steps': 'iteration_tokens_total_count', 'prompt': 'prompt_tokens_total',
            'gen': 'generation_tokens_total', 'running': 'num_requests_running',
            'waiting': 'num_requests_waiting', 'preemptions': 'num_preemptions_total',
            'prefix_queries': 'prefix_cache_queries_total', 'prefix_hits': 'prefix_cache_hits_total'}
HISTOGRAMS = ('time_to_first_token_seconds', 'request_queue_time_seconds',
              'request_prefill_time_seconds', 'request_decode_time_seconds')
SIGMA_BINS = (-1, 5, 10, 15, 20, 25, 30, 40, 50, 70)


class Arrival(NamedTuple):
    at: float
    sched: float
    cost: float
    tokens: float
    request_id: str


class Simulated(NamedTuple):
    sched: float
    admitted: bool | None
    first: float
    running: float
    request_id: str


def ordered_arrivals(arrivals):
    """Stable event order; equal arrival times must not be reordered by cost or ID."""
    return sorted(arrivals, key=lambda a: a.at)


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
                     first - t0 if first is not None else math.nan, c['trace_id'], c['request_id'],
                     g.get('rejected', math.nan)))
    rows.sort(key=lambda r: r[0])
    keys = ('sched', 'recv', 'acc', 'first', 'done', 'cost', 'est_in', 'success', 'ttft', 'c_first',
            'trace_id', 'request_id', 'rejected_at')
    ids = [r[keys.index('request_id')] for r in rows]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate client request IDs')
    table = {k: np.array([r[i] for r in rows], dtype=object if k in ('trace_id', 'request_id') else bool if k == 'success' else float)
             for i, k in enumerate(keys)}
    table['admitted'] = ~np.isnan(table['acc'])
    table['met'] = table['success'] & (table['ttft'] <= SLO)
    meta = dict(start=t0, created=manifest.get('created_at'), duration=plan.get('duration_seconds', 300.),
                schedule=plan['budget_schedule'])
    return meta, table


def load_scrapes(root, t0):
    """Retain failed samples as barriers; values become available after receipt and parsing."""
    rows = []
    for line in (Path(root) / 'engine_observations.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        raw = event.get('raw_metrics')
        received = number(event.get('received_at'))
        finished = number(event.get('finished_at', received))
        usable = (bool(raw) and math.isfinite(received) and math.isfinite(finished)
                  and event.get('http_status', 200) == 200 and not event.get('error_type')
                  and not event.get('counter_resets') and not event.get('disappeared_counters')
                  and not event.get('exporter_identity_changed')
                  and event['sent_at'] <= received <= finished)
        row = dict(t=event['sent_at'] - t0, recv=finished - t0, valid=float(usable),
                   epoch=event.get('counter_epoch', 0),
                   identity=number((event.get('exporter_identity') or {}).get('process_start_time_seconds')))
        raw = raw or ''
        row.update({k: metric(raw, name) for k, name in COUNTERS.items()})
        for name in HISTOGRAMS:
            row[name + '_sum'], row[name + '_count'] = metric(raw, name + '_sum'), metric(raw, name + '_count')
        rows.append(row)
    if not rows:
        raise ValueError('No engine observations')
    order = sorted(range(len(rows)), key=lambda i: rows[i]['t'])
    return {k: np.array([rows[i][k] for i in order], float) for k in rows[0]}


def step_intervals(scrapes, start=3., end=300.):
    """Busy consecutive intervals wholly available by end, excluding resets and failures."""
    d = {k: np.diff(scrapes[k]) for k in ('t', 'steps', 'prompt', 'gen')}
    t = scrapes['t'][1:]
    ok = (t > start) & (t < end) & (scrapes['recv'][1:] <= end) & (scrapes['recv'][:-1] <= end)
    ok &= (d['steps'] > 0) & (d['t'] > .8) & (d['t'] < 1.3)
    for k in ('steps', 'prompt', 'gen'):
        ok &= np.isfinite(scrapes[k][:-1]) & np.isfinite(scrapes[k][1:]) & (d[k] >= 0)
    if 'valid' in scrapes:
        ok &= scrapes['valid'][:-1].astype(bool) & scrapes['valid'][1:].astype(bool)
    for key in ('epoch', 'identity'):
        if key in scrapes:
            a, b = scrapes[key][:-1], scrapes[key][1:]
            ok &= (a == b) | (np.isnan(a) & np.isnan(b))
    return dict(t=t[ok], dt=d['t'][ok], steps=d['steps'][ok], prompt=d['prompt'][ok], gen=d['gen'][ok])


def fit_step_time(intervals):
    """Least squares dt = c0*steps + c1*generated + c2*prompt tokens, without intercept."""
    x = np.concatenate([np.c_[i['steps'], i['gen'], i['prompt']] for i in intervals])
    y = np.concatenate([i['dt'] for i in intervals])
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Step fit requires finite busy intervals')
    coef = tuple(float(c) for c in np.linalg.lstsq(x, y, rcond=None)[0])
    if coef[0] <= 0 or any(c < 0 or not math.isfinite(c) for c in coef):
        raise ValueError('Nonphysical step coefficients; do not run this simulator')
    return coef


class Run:
    """Lifecycle views gate observations at the origin; delay calibration uses cut.

    cut=None is a whole-capture diagnostic, not causal training for test-period forecasts.
    """

    def __init__(self, name, meta, requests, scrapes, cut=None, pre=None):
        self.name, self.req = name, requests
        self.meta, self.scrapes, self.cut = meta, scrapes, cut
        self.caps = sorted((s['at_seconds'], s['budget']) for s in meta['schedule'])
        adm = requests['admitted']
        order = np.argsort(requests['acc'][adm], kind='stable')
        a = {k: v[adm][order] for k, v in requests.items()}
        a['first_f'] = np.where(np.isnan(a['first']), np.inf, a['first'])
        a['done_f'] = np.where(np.isnan(a['done']), np.inf, a['done'])
        self.adm = a
        known = adm & (requests['acc'] <= (math.inf if cut is None else cut))
        if pre is None and not known.any():
            raise ValueError('No acceptance available for schedule-to-accept calibration')
        self.pre = float(np.median(requests['acc'][known] - requests['sched'][known])) if pre is None else pre
        self.decision = np.where(adm, requests['acc'], requests['rejected_at'])
        self.e_recv, self.e_t, self.e_steps = scrapes['recv'], scrapes['t'], scrapes['steps']

    def observed_prefix(self, end):
        """Censor event attributes not yet observed; retain the already fitted delay parameter."""
        r = {k: v.copy() for k, v in self.req.items()}
        for key in ('recv', 'acc', 'rejected_at', 'first', 'done', 'c_first'):
            r[key][r[key] > end] = np.nan
        r['admitted'] = np.isfinite(r['acc'])
        decided = r['admitted'] | np.isfinite(r['rejected_at'])
        r['cost'][~decided] = np.nan
        r['est_in'][~decided] = np.nan
        r['ttft'][~np.isfinite(r['c_first'])] = np.nan
        r['success'] &= np.isfinite(r['done'])
        r['met'] = r['success'] & (r['ttft'] <= SLO)
        return Run(self.name, self.meta, r, self.scrapes, cut=self.cut, pre=self.pre)

    def cap(self, t):
        cap = self.caps[0][1]
        for at, budget in self.caps:
            if at <= t:
                cap = budget
        return cap

    @lru_cache(maxsize=1)
    def _step_segment(self, t):
        """Cache the immutable counter view shared by all requests at an origin."""
        # Use only the latest contiguous, available counter segment. Never bridge a reset.
        known = np.isfinite(self.e_recv) & (self.e_recv <= t)
        indices = np.flatnonzero(known)
        if len(indices) < 2:
            return None
        k = indices[-1]
        valid = self.scrapes.get('valid', np.ones(len(self.e_t))).astype(bool)
        if not valid[k] or not math.isfinite(self.e_steps[k]):
            return None
        j = k
        while j > 0 and known[j - 1] and valid[j - 1] and self.e_t[j] > self.e_t[j - 1] and self.e_steps[j] >= self.e_steps[j - 1]:
            if any(key in self.scrapes and self.scrapes[key][j] != self.scrapes[key][j - 1]
                   and not (np.isnan(self.scrapes[key][j]) and np.isnan(self.scrapes[key][j - 1]))
                   for key in ('epoch', 'identity')):
                break
            j -= 1
        if j == k:
            return None
        et, es = self.e_t[j:k + 1], self.e_steps[j:k + 1]
        rate = (es[-1] - es[-2]) / max(et[-1] - et[-2], 1e-3)
        return et, es, rate

    def steps_between(self, a, t):
        """Engine steps in [a, t] from a contiguous counter segment available by t."""
        segment = self._step_segment(t)
        if segment is None:
            return 0.
        et, es, rate = segment
        f = lambda x: np.interp(x, et, es) if x <= et[-1] else es[-1] + rate * (x - et[-1])
        return max(f(t) - f(a), 0.)

    def origin(self, t):
        a = self.adm
        inflight = (a['acc'] <= t) & (a['done_f'] > t)
        streaming, awaiting = inflight & (a['first_f'] <= t), inflight & (a['first_f'] > t)
        rem = [max(OUT - 1 - self.steps_between(f, t), 1.) for f in a['first_f'][streaming]]
        running = list(zip(rem, a['cost'][streaming]))
        queue = deque([float(tok), float(c), math.nan, math.nan, None]
                      for tok, c in zip(a['est_in'][awaiting], a['cost'][awaiting]))
        return running, queue, float(np.nansum(a['cost'][inflight]))

    def oracle_arrivals(self, t, horizon=HORIZON):
        """Measured future arrivals: a non-deployable diagnostic."""
        r = self.req
        sel = (r['sched'] < t + horizon) & (self.decision > t)
        return ordered_arrivals(Arrival(max(t, s + self.pre), s, c, e, rid)
                                for s, c, e, rid in zip(r['sched'][sel], r['cost'][sel], r['est_in'][sel], r['request_id'][sel]))

    def bootstrap_arrivals(self, t, rng, lookback=60., horizon=HORIZON):
        """Replay one random past window of arrivals already decided by the origin."""
        r = self.req
        u = rng.uniform(t - lookback, t - horizon)
        sel = (r['sched'] >= u) & (r['sched'] < u + horizon) & (self.decision <= t)
        return ordered_arrivals(Arrival(s - u + t + self.pre, s - u + t, c, e, rid)
                                for s, c, e, rid in zip(r['sched'][sel], r['cost'][sel], r['est_in'][sel], r['request_id'][sel]))


class EmptyStart:
    """Idle engine and gateway at a fixed cap, for whole-run counterfactuals."""

    def __init__(self, cap):
        self.fixed = cap

    def origin(self, t):
        return [], deque(), 0.

    def cap(self, t):
        return self.fixed


def simulate(run, t, arrivals, coef, cap_fn=None, horizon=HORIZON, extra=EXTRA):
    """Return identity-linked Simulated records for cohort arrivals.

    admitted is None when the arrival is not reached before the simulation ends.
    """
    c0, c1, c2 = coef
    if c0 <= 0 or any(c < 0 or not math.isfinite(c) for c in coef):
        raise ValueError('Nonphysical step coefficients')
    if len({a.request_id for a in arrivals}) != len(arrivals):
        raise ValueError('Duplicate simulated request IDs')
    if any(not all(math.isfinite(v) for v in a[:4]) or a.cost <= 0 or a.tokens < 0 for a in arrivals):
        raise ValueError('Invalid arrival attributes')
    if any(a.at > b.at for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError('Arrivals must be in stable time order')
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
            at, sched, c, tok, rid = arrivals[ai]
            ai += 1
            cohort = t <= sched < t + horizon
            if reserved + c <= cap_fn(at) + 1e-9:
                reserved += c
                queue.append([float(tok), float(c), sched if cohort else math.nan, len(rem), rid])
            elif cohort:
                out.append(Simulated(sched, False, math.nan, math.nan, rid))
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
                    out.append(Simulated(f[2], True, tau_end, f[3], f[4]))
        tau = tau_end
    for item in queue:
        if not math.isnan(item[2]):
            out.append(Simulated(item[2], True, math.inf, item[3], item[4]))
    while ai < na:
        at, sched, c, tok, rid = arrivals[ai]
        ai += 1
        if t <= sched < t + horizon:
            out.append(Simulated(sched, None, math.nan, math.nan, rid))
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
        if not len(n) or len(n) != len(residual) or not np.isfinite(n).all() or not np.isfinite(residual).all():
            raise ValueError('Overhead fit requires finite paired observations')
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
        return np.maximum(SIGMA_MIN, np.interp(n, self.sigma_knots, self.sigma_values))


def normal_cdf(z):
    return np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in np.atleast_1d(z)])


def expected_met(out, overhead=None, bias=0.):
    """Expected count of simulated cohort requests with TTFT <= SLO."""
    kept = [(o.sched, o.first, o.running) for o in out if o.admitted and math.isfinite(o.first)]
    if not kept:
        return 0.
    s, f, n = (np.array(v, float) for v in zip(*kept))
    if overhead is None:
        return float(normal_cdf((SLO - (f - s + bias)) / .05).sum())
    return float(normal_cdf((SLO - (f - s + overhead.mean(n) + bias)) / overhead.sigma(n)).sum())


def residuals(run, coef, origins, available_by=math.inf):
    """Oracle-arrival simulations and measured-minus-simulated TTFT for admitted requests.

    Columns: origin, sched, simulated TTFT, measured TTFT, running count, client first-content time,
    cap at acceptance, window availability time. Request matching uses IDs, never timestamps.
    Each historical simulation is censored at its own horizon + drain boundary; late decisions
    and first tokens cannot enter its calibration residual. Oracle arrivals are diagnostic only.
    """
    index = {rid: i for i, rid in enumerate(run.req['request_id'])}
    if len(index) != len(run.req['request_id']):
        raise ValueError('Duplicate request IDs in residual calibration')
    rows, sims = [], {}
    for t in origins:
        end = t + HORIZON + EXTRA
        if end > available_by:
            continue
        view = run.observed_prefix(end)
        out = simulate(view, t, view.oracle_arrivals(t), coef)
        sims[float(t)] = out
        for s, a, f, n, rid in out:
            if a and math.isfinite(f):
                i = index[rid]
                if view.req['admitted'][i] and math.isfinite(view.req['ttft'][i]):
                    rows.append((t, s, f - s, view.req['ttft'][i], n, view.req['c_first'][i],
                                 run.cap(view.req['acc'][i]), end))
    return np.array(rows, float).reshape(-1, 8), sims


def online_bias(res, overhead, t, k0=20., lookback=60.):
    """Shrunken mean of recent TTFT residuals whose first token was observed by t."""
    o = res[:, 0]
    sel = (o <= t - HORIZON) & (o >= t - lookback) & (np.mod(o - t, HORIZON) == 0) & (res[:, 5] <= t) & (res[:, 7] <= t)
    if not sel.any():
        return 0.
    e = np.clip(res[sel, 3] - res[sel, 2] - overhead.mean(res[sel, 4]), -1, 1)
    return float(e.sum() / (len(e) + k0))
