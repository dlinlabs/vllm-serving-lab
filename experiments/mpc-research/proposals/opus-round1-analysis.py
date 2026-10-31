"""Offline development diagnostics for opus-round1.md (pure Python, no GPU).

Inputs are the existing archived captures restored by
results/cap-range-20261020/extract_bundle.py and
results/state-model-20261021/extract_evidence.py into fresh directories.
Everything here is development reuse of already-inspected captures; nothing is
a held-out, counterfactual or policy-gain measurement.

Usage:
  python opus-round1-analysis.py CAP_RANGE_ROOT LOW_CAP_ROOT > out.json
"""
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

SLO = 0.5
COST = (0.259, 0.000758, 0.0052083333)


def load_run(d):
    d = Path(d)
    client = json.loads((d / 'client.json').read_text())['requests']
    trace = json.loads((d / 'trace.json').read_text())
    tok = {r['trace_id']: r['input_tokens'] for r in trace['requests']}
    ev = {}
    budgets = []
    for line in (d / 'gateway.jsonl').open():
        e = json.loads(line)
        k = e['event']
        if k == 'budget_updated':
            budgets.append((e['applied_at'], e['budget']))
            continue
        rid = e.get('request_id')
        if rid is None or k not in ('received', 'accepted', 'first_content', 'completed', 'failed', 'cancelled', 'rejected'):
            continue
        rec = ev.setdefault(rid, {})
        if k == 'received':
            rec['recv'] = e['monotonic']
        elif k == 'accepted':
            rec['acc'] = e['monotonic']
            rec['cost'] = e['request_cost']
            rec['budget'] = e['budget']
            rec['pg'] = e['estimated_input_tokens']
        elif k == 'first_content':
            rec['fc'] = e['monotonic']
        elif k == 'rejected':
            rec['rej_t'] = e['monotonic']
        else:
            rec['end'] = e['monotonic']
            rec['outcome'] = k
    t0 = min(r['scheduled_at'] for r in client)
    rows = []
    for r in client:
        g = ev.get(r['request_id'], {})
        hit = bool(r['success'] and r['first_content_at'] is not None
                   and r['first_content_at'] - r['scheduled_at'] <= SLO)
        rows.append(dict(s=r['scheduled_at'] - t0, sched=r['scheduled_at'], hit=hit,
                         ok=bool(r['success']), rej=bool(r['rejected']),
                         p=tok.get(r['trace_id'], r.get('expected_input_tokens')),
                         ttft=(r['first_content_at'] - r['scheduled_at']) if r['first_content_at'] else None,
                         e2e=(r['finished_at'] - r['scheduled_at']) if r['success'] else None,
                         **g))
        # Gateway-side token estimate is the causally available size when present.
        if 'pg' in g:
            rows[-1]['p'] = g['pg']
    return rows, budgets, t0


def gateway_state(rows):
    """State at each acceptance from gateway events strictly before it.

    All quantities are internally known by the gateway at its own decision
    point (no engine scrape, no future information)."""
    adm = sorted((r for r in rows if 'acc' in r), key=lambda r: r['acc'])
    feats = []
    for i, r in enumerate(adm):
        t = r['acc']
        wn = wt = dn = 0
        rc = 0.0
        oldest = 0.0
        # A window of 600 earlier admissions is far larger than any population.
        for q in adm[max(0, i - 600):i]:
            if q.get('end', math.inf) <= t:
                continue
            rc += q['cost']
            if q.get('fc', math.inf) > t:
                wn += 1
                wt += q['p']
                oldest = max(oldest, t - q['acc'])
            else:
                dn += 1
        feats.append(dict(r=r, wn=wn, wt=wt, dn=dn, rc=rc, oldest=oldest,
                          lag=t - r['sched'], gwlag=t - r['recv']))
    return feats


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def lstsq(X, y, ridge=1e-6):
    n = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) + (ridge if i == j else 0) for j in range(n)] for i in range(n)]
    b = [sum(x[i] * v for x, v in zip(X, y)) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        b[c], b[p] = b[p], b[c]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * cc for a, cc in zip(A[r], A[c])]
                b[r] -= f * b[c]
    return [b[i] / A[i][i] for i in range(n)]


def design(f, kind):
    p = f['r']['p'] / 1000
    if kind == 'cap_only':      # what a pure reserved-cost view knows
        return [1.0, f['rc'], p]
    if kind == 'queue':         # includes client schedule lag: NOT gateway-causal
        return [1.0, f['wt'] / 1000 + p, f['wn'], f['dn'], f['oldest'], f['lag']]
    if kind == 'queue_causal':  # strictly gateway-internal at its decision point
        return [1.0, f['wt'] / 1000 + p, f['wn'], f['dn'], f['oldest'], f['gwlag']]
    if kind == 'backlog_only':  # prefill tokens ahead plus own prompt
        return [1.0, f['wt'] / 1000 + p]
    if kind == 'backlog_decode':
        return [1.0, f['wt'] / 1000 + p, f['dn']]
    raise ValueError(kind)


def lorocv(runs, kind):
    """Leave-one-run-out linear TTFT model; score held runs only."""
    out = {}
    for held in runs:
        X, y = [], []
        for name, fs in runs.items():
            if name == held:
                continue
            for f in fs:
                if f['r'].get('ttft') is not None:
                    X.append(design(f, kind))
                    y.append(f['r']['ttft'])
        th = lstsq(X, y)
        pred, lab, err = [], [], []
        for f in runs[held]:
            yh = sum(a * b for a, b in zip(th, design(f, kind)))
            pred.append(-yh)
            lab.append(f['r']['hit'])
            if f['r'].get('ttft') is not None:
                err.append(abs(yh - f['r']['ttft']))
        # Precision/recall of a "predicted miss" rule at 0.5 s on held run.
        flag = [(-s) > SLO for s in pred]
        tp = sum(1 for g, h in zip(flag, lab) if g and not h)
        fp = sum(1 for g, h in zip(flag, lab) if g and h)
        fn = sum(1 for g, h in zip(flag, lab) if not g and not h)
        out[held] = dict(auc=auc(pred, lab), mae=st.mean(err), coef=[round(c, 5) for c in th],
                         flagged=sum(flag), flagged_true_miss=tp, flagged_but_hit=fp, missed_misses=fn)
    return out


def drain(rows, budgets, t0):
    """After each applied budget decrease, time until the gateway's reserved
    cost first fits under the new budget, reconstructed from gateway events."""
    adm = [r for r in rows if 'acc' in r]
    res = []
    for k in range(1, len(budgets)):
        (ta, b), (_, prev) = budgets[k], budgets[k - 1]
        if b >= prev:
            continue
        evs = sorted([(r['acc'], r['cost']) for r in adm] +
                     [(r['end'], -r['cost']) for r in adm if 'end' in r])
        rc = 0.0
        t_fit = None
        for t, dc in evs:
            rc += dc
            if t > ta and rc <= b and t_fit is None:
                t_fit = t - ta
        first_adm = min((r['acc'] - ta for r in adm if r['acc'] > ta), default=None)
        res.append(dict(at=round(ta - t0, 2), from_=prev, to=b,
                        reserved_fits_after_s=t_fit, first_admission_after_s=first_adm))
    return res


def acceptance_by_cost(rows):
    q = sorted(COST[0] + COST[1] * r['p'] + COST[2] * 128 for r in rows)
    cuts = [q[int(len(q) * x)] for x in (0.25, 0.5, 0.75)]
    bins = [[0, 0] for _ in range(4)]
    for r in rows:
        c = COST[0] + COST[1] * r['p'] + COST[2] * 128
        b = sum(c > x for x in cuts)
        bins[b][0] += 1
        bins[b][1] += 'acc' in r
    return [round(a / n, 3) for n, a in bins]


def windows(rows, w=30):
    n = int(300 // w)
    g = [0] * n
    for r in rows:
        if r['hit']:
            g[min(int(r['s'] // w), n - 1)] += 1
    return [v / w for v in g]


def fcfs_after_test(fs):
    """F1: add prompt tokens admitted after i (before its first content)."""
    import bisect
    adm = sorted((f for f in fs if f['r'].get('ttft') is not None and 'fc' in f['r']),
                 key=lambda f: f['r']['acc'])
    accs = [f['r']['acc'] for f in adm]
    X, y = [], []
    for f in adm:
        r = f['r']
        lo, hi = bisect.bisect_right(accs, r['acc']), bisect.bisect_left(accs, r['fc'])
        after = sum(adm[k]['r']['p'] for k in range(lo, hi)) / 1000
        X.append([1.0, (f['wt'] + r['p']) / 1000, f['dn'], after])
        y.append(r['ttft'])
    full, base = lstsq(X, y), lstsq([x[:3] for x in X], y)
    mae = lambda th, XX: st.mean(abs(v - sum(a * b for a, b in zip(th, x))) for x, v in zip(XX, y))
    return dict(ahead_s_per_ktok=round(full[1], 5), after_s_per_ktok=round(full[3], 5),
                mae_without=round(mae(base, [x[:3] for x in X]), 5), mae_with=round(mae(full, X), 5))


def per_run_fit(fs, d):
    X = [design(f, 'backlog_only') for f in fs if f['r'].get('ttft') is not None]
    y = [f['r']['ttft'] for f in fs if f['r'].get('ttft') is not None]
    th = lstsq(X, y)
    out = [r['output_tokens'] for r in json.loads((Path(d) / 'client.json').read_text())['requests'] if r['success']]
    return dict(theta0_s=round(th[0], 5), theta1_s_per_ktok=round(th[1], 5),
                output_tokens_mean=st.mean(out), frac_output_128=sum(o >= 128 for o in out) / len(out))


def main():
    cap_root, low_root = Path(sys.argv[1]), Path(sys.argv[2])
    runs = {}
    for d in sorted((cap_root / 'results/cap-range-v1').iterdir()):
        if 'smoke' not in d.name and d.is_dir():
            runs[d.name] = d
    for d in sorted((low_root / 'results/low-cap-switch-v1').iterdir()):
        if 'smoke' not in d.name and d.is_dir():
            runs[d.name] = d
    summary, feats = {}, {}
    for name, d in runs.items():
        rows, budgets, t0 = load_run(d)
        adm = [r for r in rows if 'acc' in r]
        e2e = sorted(r['end'] - r['acc'] for r in adm if 'end' in r)
        ttft = sorted(r['ttft'] for r in adm if r.get('ttft') is not None)
        summary[name] = dict(
            goodput=sum(r['hit'] for r in rows) / 300,
            admitted=len(adm),
            admitted_miss=sum(1 for r in adm if not r['hit']),
            admitted_miss_frac=round(sum(1 for r in adm if not r['hit']) / len(adm), 4),
            accept_rate_by_cost_quartile=acceptance_by_cost(rows),
            residence_s_p50_p90_p99=[round(e2e[int(len(e2e) * q)], 3) for q in (0.5, 0.9, 0.99)],
            ttft_admitted_p50_p90_p99=[round(ttft[int(len(ttft) * q)], 3) for q in (0.5, 0.9, 0.99)],
            goodput_30s=[round(v, 2) for v in windows(rows)],
            drain=drain(rows, budgets, t0))
        feats[name] = gateway_state(rows)
        fs = feats[name]
        summary[name]['per_run_backlog_fit'] = per_run_fit(fs, d)
        if 'fixed-16-r2' not in name:
            summary[name]['fcfs_after_test'] = fcfs_after_test(fs)
        summary[name]['mean_state'] = dict(
            wn=round(st.mean(f['wn'] for f in fs), 2), wt=round(st.mean(f['wt'] for f in fs)),
            dn=round(st.mean(f['dn'] for f in fs), 2), rc=round(st.mean(f['rc'] for f in fs), 2))
    qualified = {k: v for k, v in feats.items() if 'fixed-16-r2' not in k}
    out = dict(summary=summary,
               cv_cap_only=lorocv(qualified, 'cap_only'),
               cv_queue=lorocv(qualified, 'queue'),
               cv_queue_causal=lorocv(qualified, 'queue_causal'),
               cv_backlog_only=lorocv(qualified, 'backlog_only'),
               cv_backlog_decode=lorocv(qualified, 'backlog_decode'))
    # Within-run 30 s window SD at fixed caps vs between-session SD.
    fixed = {k: v['goodput_30s'] for k, v in summary.items() if 'fixed' in k and 'fixed-16-r2' not in k}
    out['within_run_30s_sd'] = {k: round(st.pstdev(v), 3) for k, v in fixed.items()}
    random.seed(0)
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == '__main__':
    main()
