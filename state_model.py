"""Development state-aware variants. Frozen PR #28 and serving code stay unchanged."""
import math
import numpy as np
from scipy.special import expit
import greybox_simulator as gb


def observed_state(run, t):
    r = run.req
    active = (r['acc'] <= t) & (~np.isfinite(r['done']) | (r['done'] > t))
    awaiting = active & (~np.isfinite(r['first']) | (r['first'] > t))
    past = (r['recv'] > t - 10.) & (r['recv'] <= t)
    known = past & (run.decision <= t) & np.isfinite(r['est_in'])
    buckets = np.histogram(r['recv'][past], bins=np.arange(t - 10, t + 1))[0]
    available = np.flatnonzero((run.scrapes['recv'] <= t) & run.scrapes.get('valid', np.ones(len(run.scrapes['t']))).astype(bool))
    waiting = float(run.scrapes['waiting'][available[-1]]) if len(available) else 0.
    if not math.isfinite(waiting):
        waiting = 0.
    observed = (r['c_first'] <= t) & (r['c_first'] > t - 30.) & np.isfinite(r['ttft'])
    ttft = r['ttft'][observed]
    segment = run._step_segment(t)
    return dict(reserved=float(np.nansum(r['cost'][active])), awaiting=int(awaiting.sum()),
                queued_tokens=float(np.nansum(r['est_in'][awaiting])), streaming=int((active & ~awaiting).sum()),
                oldest=float(np.max(t - r['acc'][awaiting])) if awaiting.any() else 0.,
                arrival_rate=float(past.sum() / 10.), burst=float(buckets.max()) if len(buckets) else 0.,
                input_rate=float(np.sum(r['est_in'][known]) / 10.), engine_waiting=waiting,
                observed_ttft_mean=float(np.mean(ttft)) if len(ttft) else 0.,
                observed_ttft_p90=float(np.quantile(ttft, .9)) if len(ttft) else 0.,
                observed_first_hit_fraction=float(np.mean(ttft <= gb.SLO)) if len(ttft) else 0.,
                observed_first_count=len(ttft), engine_steps_rate=segment[2] if segment else 0.,
                remaining_decode_tokens=float(sum(rem for rem, _ in run.origin(t)[0])))


def simulate(run, t, arrivals, coef, cap_fn=None, quadratic=0.):
    """Original mechanics with admission features; q is a structural stress ablation.

    Added cost is c2*q*sum(prefill_chunk_tokens^2)/B. Aggregate scrapes cannot identify
    that per-step term independently; it is not a measured physical attention cost.
    """
    c0, c1, c2 = coef
    if c0 <= 0 or min(c1, c2, quadratic) < 0 or not all(map(math.isfinite, (*coef, quadratic))):
        raise ValueError('Nonphysical simulator parameters')
    if len({a.request_id for a in arrivals}) != len(arrivals):
        raise ValueError('Duplicate simulated request IDs')
    if any(not all(math.isfinite(v) for v in a[:4]) or a.cost <= 0 or a.tokens < 0 for a in arrivals):
        raise ValueError('Invalid arrival attributes')
    if any(a.at > b.at for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError('Arrivals must be in stable time order')
    cap_fn = cap_fn or run.cap
    running, queue, reserved = run.origin(t)
    rem, cost = np.array([r for r, _ in running], float), np.array([c for _, c in running], float)
    out, info, ai, tau, end = [], {}, 0, t, t + gb.HORIZON + gb.EXTRA
    while tau < end:
        if not len(rem) and not queue:
            if ai >= len(arrivals):
                break
            tau = max(tau, arrivals[ai].at)
        budget, p, p2, finished = gb.B - len(rem), 0., 0., []
        for item in queue:
            if budget <= 0:
                break
            take = min(budget, item[0]); item[0] -= take
            p += take; p2 += take * take; budget -= take
            if item[0] <= 0:
                finished.append(item)
        for _ in finished:
            queue.popleft()
        tau_end = tau + c0 + c1 * len(rem) + c2 * p + c2 * quadratic * p2 / gb.B
        while ai < len(arrivals) and arrivals[ai].at < tau_end:
            at, sched, c, tok, rid = arrivals[ai]; ai += 1
            if reserved + c <= cap_fn(at) + 1e-9:
                info[rid] = dict(tokens=tok, ahead_tokens=sum(x[0] for x in queue), ahead_count=len(queue),
                                 reserved=reserved, largest_ahead=max([x[0] for x in queue] + [0.]))
                reserved += c
                queue.append([float(tok), float(c), sched if t <= sched < t + gb.HORIZON else math.nan, len(rem), rid])
            elif t <= sched < t + gb.HORIZON:
                out.append(gb.Simulated(sched, False, math.nan, math.nan, rid))
        rem -= 1
        done = rem <= 0
        if done.any():
            reserved -= cost[done].sum(); rem, cost = rem[~done], cost[~done]
        if finished:
            rem = np.concatenate([rem, np.full(len(finished), gb.OUT - 1.)])
            cost = np.concatenate([cost, [f[1] for f in finished]])
            out.extend(gb.Simulated(f[2], True, tau_end, f[3], f[4]) for f in finished if not math.isnan(f[2]))
        tau = tau_end
    out.extend(gb.Simulated(x[2], True, math.inf, x[3], x[4]) for x in queue if not math.isnan(x[2]))
    out.extend(gb.Simulated(a.sched, None, math.nan, math.nan, a.request_id)
               for a in arrivals[ai:] if t <= a.sched < t + gb.HORIZON)
    return out, info


POINT_NAMES = ('ttft_margin', 'running', 'prompt', 'ahead_tokens', 'ahead_count', 'reserved',
               'largest_ahead', 'origin_engine_waiting', 'origin_oldest', 'recent_ttft_mean',
               'recent_ttft_p90', 'recent_first_hit_fraction', 'recent_first_count', 'engine_steps_rate',
               'remaining_decode_tokens', 'recent_input_rate', 'recent_burst', 'running_x_prompt',
               'running_x_ahead', 'ahead_x_largest', 'prompt_squared')


def point_features(out, info, state, overhead, bias):
    kept = [o for o in out if o.admitted and math.isfinite(o.first)]
    x = []
    for o in kept:
        f = info[o.request_id]
        predicted = o.first - o.sched + float(overhead.mean(o.running)) + bias
        prompt, ahead, largest = (np.log1p(f[k] / scale) for k, scale in
                                  [('tokens', 1024.), ('ahead_tokens', 8192.), ('largest_ahead', 1024.)])
        x.append([(gb.SLO - predicted) / .25, o.running / 16., prompt, ahead, f['ahead_count'] / 8.,
                  f['reserved'] / 32., largest, state['engine_waiting'] / 4., state['oldest'],
                  state['observed_ttft_mean'] / gb.SLO, state['observed_ttft_p90'] / gb.SLO,
                  state['observed_first_hit_fraction'], state['observed_first_count'] / 100.,
                  state['engine_steps_rate'] / 50., state['remaining_decode_tokens'] / 1024.,
                  state['input_rate'] / 10000., state['burst'] / 50., o.running / 16. * prompt,
                  o.running / 16. * ahead, ahead * largest, (f['tokens'] / 8192.) ** 2])
    return kept, np.array(x, float).reshape(-1, len(POINT_NAMES))


class LogisticCalibration:
    """Training-only scaling and ridge logistic fit, monotone in TTFT margin."""
    def __init__(self, columns, penalty):
        self.columns, self.penalty = np.asarray(columns, int), float(penalty)
        if self.penalty <= 0 or not math.isfinite(self.penalty) or not len(self.columns):
            raise ValueError('Positive finite penalty and nonempty features are required')

    def fit(self, x, y):
        x, y = np.asarray(x, float)[:, self.columns], np.asarray(y, float)
        if len(y) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Calibration requires finite training observations')
        self.center, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
        z = np.c_[np.ones(len(x)), (x - self.center) / self.scale]
        beta = np.zeros(z.shape[1]); mean = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        beta[0] = np.log(mean / (1 - mean))
        ridge = np.eye(z.shape[1]) * self.penalty; ridge[0, 0] = 1e-8
        for _ in range(40):
            p = expit(z @ beta); grad = z.T @ (p - y) + ridge @ beta
            h = (z.T * np.maximum(p * (1 - p), 1e-6)) @ z + ridge
            step = np.linalg.solve(h, grad)
            loss = np.logaddexp(0, z @ beta).sum() - y @ (z @ beta) + .5 * beta @ ridge @ beta
            factor = 1.
            while factor > 1e-6:
                candidate = beta - factor * step
                if 0 in self.columns:
                    j = 1 + int(np.flatnonzero(self.columns == 0)[0]); candidate[j] = max(candidate[j], 0.)
                eta = z @ candidate
                next_loss = np.logaddexp(0, eta).sum() - y @ eta + .5 * candidate @ ridge @ candidate
                if next_loss <= loss + 1e-9:
                    break
                factor *= .5
            beta = candidate
            if np.max(np.abs(factor * step)) < 1e-6:
                break
        self.beta = beta
        return self

    def predict(self, x):
        x = np.asarray(x, float)[:, self.columns]
        return expit(np.c_[np.ones(len(x)), (x - self.center) / self.scale] @ self.beta)

    def parameters(self):
        return dict(columns=self.columns.tolist(), penalty=self.penalty, center=self.center.tolist(),
                    scale=self.scale.tolist(), coefficients=self.beta.tolist())


class TreeCalibration:
    def __init__(self, columns, penalty):
        self.columns, self.penalty = list(columns), float(penalty)

    def fit(self, x, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, min_samples_leaf=40,
            l2_regularization=self.penalty, monotonic_cst=[1] + [0] * (len(self.columns) - 1),
            early_stopping=False, random_state=1)
        self.model.fit(np.asarray(x)[:, self.columns], y)
        return self

    def predict(self, x):
        return self.model.predict_proba(np.asarray(x)[:, self.columns])[:, 1]

    def parameters(self):
        return dict(columns=self.columns, penalty=self.penalty, max_iter=80, max_leaf_nodes=7,
                    min_samples_leaf=40, monotone_margin=True, seed=1)
