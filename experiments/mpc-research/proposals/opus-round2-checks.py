"""Offline Round 2 counterexample checks for opus-round2.md (pure Python 3, no GPU).

Usage:
  python3 opus-round2-checks.py CAP_RANGE_ROOT LOW_CAP_ROOT > out.json

Roots are the restored development captures used by opus-round1-analysis.py.
All data checks are development reuse of already-inspected captures. C6 timings
are wall-clock on the executing CPU and are the only non-deterministic output.
"""
import asyncio
import bisect
import importlib.util
import itertools
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a1', HERE / 'opus-round1-analysis.py')
a1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a1)

C0, CP, CO = 0.259, 0.000758, 0.0052083333
cost = lambda p, m=128: max(0.1, C0 + CP * p + CO * m)


def runs(cap_root, low_root):
    out = {}
    for root, sub in ((cap_root, 'results/cap-range-v1'), (low_root, 'results/low-cap-switch-v1')):
        for d in sorted((Path(root) / sub).iterdir()):
            if d.is_dir() and 'smoke' not in d.name and 'fixed-16-r2' not in d.name:
                out[d.name] = d
    return out


def c1_old_cohort(fs):
    """Upper estimate of hits lost by old cohorts to *all* later admissions."""
    adm = sorted((f for f in fs if f['r'].get('ttft') is not None and 'fc' in f['r']),
                 key=lambda f: f['r']['acc'])
    accs = [f['r']['acc'] for f in adm]
    X, y, after = [], [], []
    for f in adm:
        r = f['r']
        lo, hi = bisect.bisect_right(accs, r['acc']), bisect.bisect_left(accs, r['fc'])
        a = sum(adm[k]['r']['p'] for k in range(lo, hi)) / 1000
        after.append(a)
        X.append([1.0, (f['wt'] + r['p']) / 1000, f['dn'], a])
        y.append(r['ttft'])
    b = a1.lstsq(X, y)[3]
    hits = sum(1 for f in adm if f['r']['hit'])
    flips = sum(1 for f, a, t in zip(adm, after, y) if f['r']['ok'] and t > 0.5 and t - b * a <= 0.5)
    first_hit_then_fail = sum(1 for f, t in zip(adm, y) if t <= 0.5 and not f['r']['ok'])
    return dict(b_after=round(b, 5), hits=hits, flips_if_no_later_admissions=flips,
                flip_frac_of_hits=round(flips / hits, 4),
                mean_after_ktok=round(st.mean(after), 3),
                first_token_hit_then_failed=first_hit_then_fail)


def c2_little(rows, fs):
    """Little's-law admission capacity if residence capacity went only to small requests."""
    adm = [r for r in rows if 'acc' in r and 'end' in r]
    R = st.mean(f['rc'] for f in fs)
    ks = sorted(adm, key=lambda r: r['cost'])
    out = dict(mean_R=round(R, 2), measured_admit_rate=round(len([r for r in rows if 'acc' in r]) / 300, 3))
    for name, sub in (('all', ks), ('cheapest_decile', ks[:len(ks) // 10])):
        kS = st.mean(r['cost'] * (r['end'] - r['acc']) for r in sub)
        out[name] = dict(mean_k=round(st.mean(r['cost'] for r in sub), 3),
                         mean_S=round(st.mean(r['end'] - r['acc'] for r in sub), 3),
                         little_rate=round(R / kS, 2))
    return out


def c3_cost():
    out = dict(c_p907_m128=round(cost(907), 4), c_p907_m2048=round(cost(907, 2048), 4),
               max_prompt_admissible_at_C16_m2048=math.floor((16 - C0 - CO * 2048) / CP),
               max_prompt_admissible_at_C16_m128=math.floor((16 - C0 - CO * 128) / CP))
    # Tokens resident per cost unit is increasing in p, so the max is at the largest prompt.
    for pmax in (6674, 8192, 32768):
        tpc = (pmax + 128) / cost(pmax)
        out[f'max_tokens_per_cost_pmax{pmax}'] = round(tpc, 1)
        out[f'resident_token_bound_C32_pmax{pmax}'] = math.floor(32 * tpc)
    return out


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def c4_stats():
    out = {}
    for n in (50, 100, 200, 1000):
        k = math.ceil(0.88 * n) - 1  # coverage < 0.88  <=>  hits <= k
        out[f'P_fail_gate_true_cov_0.90_n{n}'] = round(binom_cdf(k, n, 0.90), 4)
    out['zero_fail_n_for_95pct_upper_le_0.001'] = math.ceil(math.log(0.05) / math.log(1 - 0.001))
    out['zero_fail_n_for_95pct_upper_le_0.01'] = math.ceil(math.log(0.05) / math.log(1 - 0.01))
    out['astra_299_upper'] = round(1 - 0.05 ** (1 / 299), 5)
    # Gemini static-domination test: replicate SD vs SE of a difference of two means.
    return out


def gemini_plans(levels=(16, 24, 32), H=4):
    plans = set()
    for u in itertools.product(levels, repeat=H):
        d = [u[i + 1] - u[i] for i in range(H - 1)]
        if all(x in (-8, 0, 8) for x in d) and sum(x != 0 for x in d) <= 1:
            plans.add(u)
    return len(plans)


def one_switch(levels, n):
    return len({tuple([a] * s + [b] * (n - s)) for a in levels for b in levels for s in range(n + 1)})


def c5_counts():
    return dict(gemini=gemini_plans(), astra_one_switch_20_slots=one_switch((16, 32), 20),
                gpt_all_5=2 ** 5, opus_all_3=2 ** 3, opus_all_3_three_levels=3 ** 3)


def rollout(plan, dt, arrivals, th0=0.065, th1=0.05e-3, S=2.8, R0=0.0, inflight=()):
    """Cheapest event rollout consistent with all four proposals' admission rule."""
    R, hits = R0, 0
    ends = list(inflight)  # (end_time, cost)
    fcs = []  # (fc_time, prompt) for W_tok
    for t, p in arrivals:
        while ends and ends[0][0] <= t:
            R -= ends.pop(0)[1]
        fcs = [x for x in fcs if x[0] > t]
        c = cost(p)
        u = plan[min(int(t / dt), len(plan) - 1)]
        if R + c <= u:
            R += c
            ttft = th0 + th1 * (sum(q for _, q in fcs) + p)
            hits += ttft <= 0.5
            bisect.insort(ends, (t + S, c))
            fcs.append((t + ttft, p))
    return hits


def c6_timing(prompts, rng):
    def scen(H):
        out, t = [], 0.0
        while True:
            t += rng.expovariate(25.48)
            if t >= H:
                return out
            out.append((t, rng.choice(prompts)))
    cases = dict(
        gemini_15x8_H4=(15, 8, 4.0, 1.0),
        opus_8x32_H3=(8, 32, 3.0, 1.0),
        gpt_32x16_H5=(32, 16, 5.0, 1.0),
        astra_40x32_H5=(40, 32, 5.0, 0.25),
    )
    res = {}
    for name, (nplans, nscen, H, dt) in cases.items():
        scens = [scen(H) for _ in range(nscen)]
        plan = [32] * int(H / dt)
        reps = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(nplans):
                for s in scens:
                    rollout(plan, dt, s)
            reps.append((time.perf_counter() - t0) * 1000)
        res[name] = dict(rollouts=nplans * nscen, arrivals_per_rollout=round(st.mean(len(s) for s in scens), 1),
                         ms_median=round(st.median(reps), 1), ms_min=round(min(reps), 1))
    return res


async def c7_race():
    """Client timeout does not cancel a server-side budget write; last lock wins."""
    state = dict(budget=32.0, version=0)
    lock = asyncio.Lock()
    log = []

    async def post(budget, transit_s):
        await asyncio.sleep(transit_s)  # request still in flight / queued in the server
        async with lock:
            state['budget'] = budget
            state['version'] += 1
            log.append((round(loop.time() - t0, 3), budget, state['version']))

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    plan_write = asyncio.ensure_future(post(16.0, 0.060))  # MPC command, slow transit
    try:
        await asyncio.wait_for(asyncio.shield(plan_write), timeout=0.020)
        timed_out = False
    except asyncio.TimeoutError:
        timed_out = True
    await post(32.0, 0.001)  # controller "falls back" to C*=32 and gets an ack
    after_fallback_ack = dict(state)
    await plan_write
    race = dict(client_timed_out=timed_out, state_at_fallback_ack=after_fallback_ack,
                final_state=dict(state), apply_log=log)

    # In-process synchronous solver: wait_for cannot preempt it.
    def solver():
        end = time.perf_counter() + 0.050
        while time.perf_counter() < end:
            pass
        return 24.0

    async def solve():
        return solver()

    t1 = time.perf_counter()
    try:
        val = await asyncio.wait_for(solve(), timeout=0.020)
        outcome = f'returned {val} (no timeout raised)'
    except asyncio.TimeoutError:
        outcome = 'timeout'
    race['inprocess_sync_solver'] = dict(watchdog_s=0.020, outcome=outcome,
                                         elapsed_ms=round((time.perf_counter() - t1) * 1000, 1),
                                         event_loop_blocked=True)
    return race


def main():
    cap_root, low_root = sys.argv[1], sys.argv[2]
    out = dict(c1_old_cohort={}, c2_little={})
    prompts = None
    for name, d in runs(cap_root, low_root).items():
        rows, _, _ = a1.load_run(d)
        fs = a1.gateway_state(rows)
        out['c1_old_cohort'][name] = c1_old_cohort(fs)
        if 'fixed-96' in name or 'fixed-16' in name:
            out['c2_little'][name] = c2_little(rows, fs)
        if prompts is None:
            prompts = [r['input_tokens'] for r in json.loads((d / 'trace.json').read_text())['requests']]
    out['c3_cost'] = c3_cost()
    out['c4_stats'] = c4_stats()
    out['c5_plan_counts'] = c5_counts()
    out['c6_rollout_timing_ms'] = c6_timing(prompts, random.Random(20261029))
    out['c7_actuation'] = asyncio.run(c7_race())
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
