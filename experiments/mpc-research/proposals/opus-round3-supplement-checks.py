"""Round 3 supplement checks against the Mayor's provisional
mpc-mathematical-specification.md (pure stdlib; constructed examples only).

S1 horizon-edge cohort truncation: reward only C_k (unresolved + offers in the
   N=3 window) versus a fixed evaluation window that also rewards tail offers.
S2 kernel surrogate identifiability under saturated packing q = B - d.
S3 randomized check of the reservation bound R(t) <= max(R0, u_max) and
   count <= max(R0, u_max)/c_min under exact serial accounting.
S4 plan-count arithmetic for N = 1/3/5 over {16,32}.

Usage: python3 -B opus-round3-supplement-checks.py
"""
import itertools
import random
from fractions import Fraction


def s1():
    # Strict reservation plant, TTFT constant 0.1 s (always on time if admitted).
    # Slots of 1 s; N=3 planning window covers offers at t in [0,3).
    # Decision at t=0 (the executed first action). Long: t=0, cost 20, holds 6 s.
    # Shorts: t=3,4,5 (just beyond the N=3 window), cost 13, hold 0.5 s.
    offers = [(0, 20, 6.0), (3, 13, 0.5), (4, 13, 0.5), (5, 13, 0.5)]
    backup = 32

    def run(plan):
        active, admitted = [], []
        for t, c, h in offers:
            active = [(e, cc) for e, cc in active if e > t]
            cap = plan[t] if t < len(plan) else backup
            ok = sum(cc for _, cc in active) + c <= cap
            if ok:
                active.append((t + h, c))
            admitted.append((t, ok))
        return admitted

    rows = {}
    for plan in itertools.product((16, 32), repeat=3):
        adm = run(plan)
        cohort_only = sum(ok for t, ok in adm if t < 3)
        fixed_window = sum(ok for t, ok in adm)
        rows[plan] = (cohort_only, fixed_window)
    cohort_best = max(v[0] for v in rows.values())
    window_best = max(v[1] for v in rows.values())
    cohort_first = {p[0] for p in rows if rows[p][0] == cohort_best}
    window_first = {p[0] for p in rows if rows[p][1] == window_best}
    true_after_cohort = max(rows[p][1] for p in rows if rows[p][0] == cohort_best)
    return dict(scores={str(k): v for k, v in rows.items()},
                cohort_only_first_actions=sorted(cohort_first), cohort_only_true_value=true_after_cohort,
                fixed_window_first_actions=sorted(window_first), fixed_window_value=window_best)


def rank(rows):
    m = [[Fraction(x) for x in r] for r in rows]
    r, cols = 0, len(m[0])
    for c in range(cols):
        piv = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c] / m[r][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
    return r


def s2():
    B = 2048
    sat = [[1, B - d, d, (B - d) * d] for d in range(1, 257, 5)]
    mixed = sat + [[1, q, d, q * d] for q, d in ((0, 32), (100, 32), (512, 8), (0, 64), (300, 1))]
    return dict(params=4, rank_saturated=rank(sat), rank_with_unsaturated_iterations=rank(mixed))


def s3(trials=2000, seed=7):
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(trials):
        c_min, u_max = 0.1, rng.choice((16.0, 32.0))
        active = [rng.uniform(c_min, 6) for _ in range(rng.randint(0, 20))]
        R0 = sum(active)
        bound = max(R0, u_max)
        for _ in range(200):
            if active and rng.random() < 0.4:
                active.pop(rng.randrange(len(active)))
            else:
                c = rng.uniform(c_min, 6)
                u = rng.choice((16.0, 32.0)) if u_max == 32.0 else 16.0
                if sum(active) + c <= u:
                    active.append(c)
            R = sum(active)
            assert R <= bound + 1e-9
            assert len(active) <= bound / c_min + 1e-9
            worst = max(worst, R / bound)
    return dict(trials=trials, max_R_over_bound=round(worst, 6), violations=0)


def s4():
    return {f'N={n}': 2 ** n for n in (1, 3, 5)}


def main():
    out = dict(S1=s1(), S2=s2(), S3=s3(), S4=s4())
    for k, v in out.items():
        print(k, v)
    assert out['S1']['cohort_only_true_value'] < out['S1']['fixed_window_value']
    assert out['S2']['rank_saturated'] == 3 and out['S2']['rank_with_unsaturated_iterations'] == 4
    print('assertions passed')


if __name__ == '__main__':
    main()
