"""Round 3 offline checks for opus-round3.md (pure Python stdlib, no GPU/network).

K1 corrected drain diagnostic on the 12 development captures (replaces the
   frozen Round 1 `drain` helper, which ignored already-fitting state).
K2 three actuator contracts on one deterministic event sequence.
K3 cohort-partition boundary invariance (and a double-count counterexample).
K4 CAS + server expiry semantics on the late-write schedule.
K5 same-model myopic (N=1) versus N=3 under pure reservation opportunity cost.

Usage:
  python3 -B opus-round3-checks.py CAP_RANGE_ROOT LOW_CAP_ROOT
All K2-K5 rows are constructed examples; only K1 reads measured artifacts,
and K1 remains development reuse, not a policy or performance measurement.
"""
import importlib.util
import itertools
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r1', HERE / 'opus-round1-analysis.py')
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
COST = r1.COST


def runs(cap_root, low_root):
    out = {}
    for base, sub in ((cap_root, 'results/cap-range-v1'), (low_root, 'results/low-cap-switch-v1')):
        for d in sorted((Path(base) / sub).iterdir()):
            if d.is_dir() and 'smoke' not in d.name:
                out[d.name] = d
    return out


def corrected_drain(rows, budgets):
    """For each applied decrease: R at update (inclusive of all events with
    t <= update), delay until R<=b, and delay until the *next received offer*
    fits (R + c_next <= b) or is accepted. Ties: events at t==update count as
    before the update (gateway applies under lock; documented convention)."""
    adm = [r for r in rows if 'acc' in r]
    evs = sorted([(r['acc'], r['cost']) for r in adm] + [(r['end'], -r['cost']) for r in adm if 'end' in r])
    offers = sorted((r['recv'], r) for r in rows if 'recv' in r)
    out = []
    for k in range(1, len(budgets)):
        (ta, b), (_, prev) = budgets[k], budgets[k - 1]
        if b >= prev:
            continue
        rc = sum(dc for t, dc in evs if t <= ta)
        t_r = 0.0 if rc <= b + 1e-9 else None
        if t_r is None:
            run = rc
            for t, dc in evs:
                if t > ta:
                    run += dc
                    if run <= b + 1e-9:
                        t_r = t - ta
                        break
        nxt = next((r for t, r in offers if t > ta), None)
        t_adm = None
        if nxt is not None:
            t_adm = (nxt['acc'] - ta) if 'acc' in nxt else 'rejected'
        frozen = r1.drain(rows, [(-1e18, prev), (ta, b)], 0.0)[0]['reserved_fits_after_s']
        out.append(dict(from_=prev, to=b, R_at_update=round(rc, 3),
                        fits_after_s=None if t_r is None else round(t_r, 4),
                        frozen_helper_s=None if frozen is None else round(frozen, 4),
                        next_offer_outcome=t_adm if not isinstance(t_adm, float) else round(t_adm, 4)))
    return out


def k1(cap_root, low_root):
    res = {}
    for name, d in runs(cap_root, low_root).items():
        rows, budgets, t0 = r1.load_run(d)
        c = corrected_drain(rows, budgets)
        if c:
            res[name] = c
    allc = [x for v in res.values() for x in v]
    fits = [x['fits_after_s'] for x in allc if x['fits_after_s'] is not None]
    disagree = sum(1 for x in allc if x['fits_after_s'] != x['frozen_helper_s'])
    return dict(per_run=res, n_decreases=len(allc), n_fit_observed=len(fits),
                already_fit_at_update=sum(1 for v in fits if v == 0.0),
                fits_after_min_max=[min(fits), max(fits)] if fits else None,
                fits_after_median=st.median(fits) if fits else None,
                n_disagree_with_frozen=disagree)


def cost(p, m=128):
    return COST[0] + COST[1] * p + COST[2] * m


def k2():
    """Sample R at t=0; three decoders (prompt 1153) terminate at t=10ms; the
    command applies at t=20ms; a 7000-token prompt arrives at t=30ms; three more
    decoders terminate at 35ms; a second 7000-token prompt arrives at 40ms.
    Backlog W_tok = K exactly at sample (zero intended prefill allowance)."""
    R0, K, W0, b = 40.0, 6000, 6000, cost(0)
    s = b  # Round 1 slack rule leaves one base cost when K-W<=0
    rel1, rel2 = 3 * cost(1153), 3 * cost(1153)
    out = {}
    # (a) absolute C = R_sample + s, R read at t=0
    C = R0 + s
    R = R0 - rel1
    adm = 0
    R_after = R
    if R_after + cost(7000) <= C:
        adm += 7000; R_after += cost(7000)
    R_after -= rel2
    if R_after + cost(7000) <= C:
        adm += 7000; R_after += cost(7000)
    out['absolute_C'] = adm
    # (b) server-side relative slack: C = R_at_apply + s (atomic), then fixed
    R = R0 - rel1
    C = R + s
    adm, R_after = 0, R
    if R_after + cost(7000) <= C:
        adm += 7000; R_after += cost(7000)
    R_after -= rel2
    if R_after + cost(7000) <= C:
        adm += 7000; R_after += cost(7000)
    out['atomic_relative_slack'] = adm
    # (c) per-request predicate W_tok + p <= K evaluated at each arrival
    W = W0
    adm = 0
    for p in (7000, 7000):
        if W + p <= K:
            adm += p; W += p
    out['per_request_backlog_gate'] = adm
    return dict(excess_prompt_tokens_admitted_beyond_K=out,
                note='atomic relative slack removes the read-to-apply race but not post-apply releases')


def k3():
    """Cohort partition. Requests: (arrival, accepted, first_ok, final_ok, resolve_time).
    Score = explicit sum of eventual g_i for K (old + arrivals <= H) whose resolution
    is <= T, + continuation value V on *unresolved* K members only (their exact g),
    + future-cohort term F (arrivals > H) independent of the cut T. Moving T must not
    change the total. A naive V that re-adds all K reward double counts."""
    reqs = [(-1.0, 1, 1, 1, 0.8), (-0.5, 1, 1, 0, 2.5), (0.2, 1, 1, 1, 3.4),
            (0.9, 1, 0, 0, 1.4), (1.5, 0, 0, 0, 1.5), (2.7, 1, 1, 1, 6.0)]
    H = 3.0
    K = [r for r in reqs if r[0] <= H]
    g = lambda r: r[1] * r[2] * r[3]
    F_value = 1.0  # stand-in differential value of post-H cohorts (same for every cut)
    rows = []
    for T in (3.0, 3.5, 4.0, 7.0):
        explicit = sum(g(r) for r in K if r[4] <= T)
        V_unres = sum(g(r) for r in K if r[4] > T)
        naive_V = sum(g(r) for r in K)
        rows.append(dict(T=T, partition_total=explicit + V_unres + F_value,
                         naive_total=explicit + naive_V + F_value))
    return rows


def k4():
    """Server holds (budget, version). Commands carry expected_version and not_after."""
    def run(cas, expiry, with_fallback):
        state = dict(budget=16, version=0)
        log = []
        cmds = [(100, dict(b=32, ev=0, na=50, name='A'))]
        if with_fallback:
            cmds.append((70, dict(b=16, ev=0, na=120, name='B')))
        for t, c in sorted(cmds, key=lambda x: x[0]):
            if expiry and t > c['na']:
                log.append((t, c['name'], 'expired')); continue
            if cas and state['version'] != c['ev']:
                log.append((t, c['name'], 'cas_reject')); continue
            state['budget'] = c['b']; state['version'] += 1
            log.append((t, c['name'], 'applied'))
        return dict(final=state['budget'], log=log)
    return {f'cas={c},expiry={e},fallback={f}': run(c, e, f)
            for c, e, f in itertools.product((False, True), (False, True), (True, False))}


def k5():
    """Strict reservation plant, TTFT constant 0.1 s (zero decode interference).
    t=0: long request cost 20, holds reservation 4 s, hit. t=1,2,3: short cost 13,
    hold 0.5 s, hit. Caps in {16, 32}. N=1 same model scores only the current
    decision's arrivals over a 1-s window; N=4 scores the 4-s window."""
    arrivals = [(0, 20, 4.0), (1, 13, 0.5), (2, 13, 0.5), (3, 13, 0.5)]
    def simulate(plan):
        active, hits = [], 0
        for k, (t, c, hold) in enumerate(arrivals):
            active = [(e, cc) for e, cc in active if e > t]
            if sum(cc for _, cc in active) + c <= plan[k]:
                active.append((t + hold, c)); hits += 1
        return hits
    plans = list(itertools.product((16, 32), repeat=4))
    best4 = max(plans, key=simulate)
    # myopic: choose first action maximizing hits of the t=0 cohort only, then repeat
    # N=1 greedy: each decision maximizes only its own arrival's hit; ties -> 32.
    def own_hit(prefix, u):
        k = len(prefix)
        active = []
        sub = arrivals[:k + 1]
        for j, (tj, cj, hj) in enumerate(sub):
            active = [(e, cc) for e, cc in active if e > tj]
            cap = (list(prefix) + [u])[j]
            ok = sum(cc for _, cc in active) + cj <= cap
            if ok:
                active.append((tj + hj, cj))
        return int(ok)
    prefix = []
    for k in range(4):
        prefix.append(max((32, 16), key=lambda u: own_hit(prefix, u)))
    myopic_plan = tuple(prefix)
    myopic_first = myopic_plan[0]
    return dict(myopic_first_action=myopic_first, myopic_hits=simulate(myopic_plan),
                N4_best_plan=best4, N4_hits=simulate(best4))


def main():
    out = dict(K1=k1(sys.argv[1], sys.argv[2]), K2=k2(), K3=k3(), K4=k4(), K5=k5())
    k1r = out['K1']
    print('K1 decreases', k1r['n_decreases'], 'fit_observed', k1r['n_fit_observed'],
          'already_fit', k1r['already_fit_at_update'], 'min/max', k1r['fits_after_min_max'],
          'median', k1r['fits_after_median'], 'disagree_frozen', k1r['n_disagree_with_frozen'])
    print('K2', out['K2']['excess_prompt_tokens_admitted_beyond_K'])
    for r in out['K3']:
        print('K3', r)
    for k, v in out['K4'].items():
        print('K4', k, v['final'], v['log'])
    print('K5', out['K5'])
    assert all(r['partition_total'] == out['K3'][0]['partition_total'] for r in out['K3'])
    assert len({r['naive_total'] for r in out['K3']}) > 1
    assert out['K4']['cas=True,expiry=False,fallback=True']['final'] == 16
    assert out['K4']['cas=False,expiry=True,fallback=False']['final'] == 16
    assert out['K4']['cas=False,expiry=False,fallback=True']['final'] == 32
    assert out['K5']['N4_hits'] > out['K5']['myopic_hits']
    Path('/tmp/obs-mpc/r3.json').write_text(json.dumps(out, indent=1, default=str))
    print('assertions passed; JSON at /tmp/obs-mpc/r3.json')


if __name__ == '__main__':
    main()
