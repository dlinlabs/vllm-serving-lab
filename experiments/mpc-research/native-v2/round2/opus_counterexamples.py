"""Round2 (Opus) executable counterexamples and one unprofiled native check.

Stdlib only. TRAIN artifacts only; held-out 160021-160024 is refused by the
Round1 loader that this module reuses. Nothing here fits a controller, chooses
a budget or simulates a benchmark. Each function is a witness for one claim in
`opus-critique.md`:

* reservation_invariant_witness  -- Gemini 5.3(1) inequality is false after a cap cut.
* sign_flip_p_floor              -- exact paired sign-flip p cannot reach 0.05 with 4 blocks.
* horizon_witness                -- "multi-step adds value iff T_settle > T_slot" is false;
                                    the value is a property of the backup policy.
* required_tflops / flop_floor_ms -- physical compute floor for prefill claims.
* decode_wall_*                  -- unprofiled decode iteration time from Delta output_tokens;
                                    refutes Opus Round1 F3 (host gap) and the profiled D-slope.
* loglinear_vs_hinge_loco        -- Astra's literal log-linear duration family vs the hinge.
"""

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round1"))
import opus_batch_models as r1  # noqa: E402  (frozen Round1 module, read-only reuse)


# ----------------------------------------------------- ledger / statistics witnesses

def reservation_invariant_witness(costs=(10.0, 10.0, 10.0), budgets=(32.0, 16.0)):
    """Admit under budgets[0], then apply budgets[1] (no eviction).

    Gemini round1 5.3(1) claims R(t) <= max(R(0), u_applied(t)) for all t.
    The sound invariant is R(t) <= max(R(0), max_{s<=t} u_applied(s))."""
    R0, R, applied = 0.0, 0.0, []
    for c in costs:
        if R + c <= budgets[0]:
            R += c
    applied.append(budgets[0])
    applied.append(budgets[1])          # cap reduction: no eviction, no repricing
    return dict(R=R, u_now=budgets[1],
                gemini_bound=max(R0, budgets[1]), gemini_holds=R <= max(R0, budgets[1]),
                history_bound=max(R0, max(applied)), history_holds=R <= max(R0, max(applied)))


def sign_flip_p(diffs, two_sided=False):
    """Exact paired sign-flip permutation p-value (all 2^n sign patterns)."""
    n = len(diffs)
    if n == 0:
        raise ValueError("no blocks")
    observed = sum(diffs) / n
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = sum(s * d for s, d in zip(signs, diffs)) / n
        hits += (abs(m) >= abs(observed) - 1e-12) if two_sided else (m >= observed - 1e-12)
    return hits / 2 ** n


def sign_flip_p_floor(n, two_sided=False):
    """Smallest attainable p with n nonzero paired differences."""
    return (2 if two_sided else 1) / 2 ** n


# ------------------------------------------------------------- horizon witness

def _plan_value(x, t, plan, backup, arrivals, step):
    total = 0.0
    for k in range(t, len(arrivals)):
        u = plan[k - t] if k - t < len(plan) else backup(x, k)
        r, x = step(x, u, arrivals[k])
        total += r
    return total


def _first_actions(x, t, n_moves, backup, arrivals, step, actions):
    moves = min(n_moves, len(arrivals) - t)
    scores = {p: _plan_value(x, t, p, backup, arrivals, step) for p in itertools.product(actions, repeat=moves)}
    best = max(scores.values())
    return frozenset(p[0] for p, v in scores.items() if v >= best - 1e-12)


def toy_step(x, u, a, capacity=3):
    """Admit k=min(a,u); admitted work persists for one further slot (memory 2 slots
    > 1 slot). A slot whose new plus carried load exceeds capacity spoils its admits."""
    k = min(a, u)
    return (float(k) if k + x <= capacity else 0.0), k


def memoryless_step(x, u, a, capacity=3):
    k = min(a, u)
    return (float(k) if k <= capacity else 0.0), 0


def optimal_backup(arrivals, step, actions):
    """Finite-horizon DP policy (ties -> smallest action)."""
    states = sorted({0} | {min(a, u) for a in arrivals for u in actions})
    V = {(len(arrivals), x): 0.0 for x in states}
    pol = {}
    for t in reversed(range(len(arrivals))):
        for x in states:
            best = None
            for u in actions:
                r, nx = step(x, u, arrivals[t])
                v = r + V[(t + 1, nx)]
                if best is None or v > best[0] + 1e-12:
                    best = (v, u)
            V[(t, x)], pol[(t, x)] = best
    return lambda x, t: pol[(t, x)]


def horizon_witness(arrivals=(2, 2, 2, 1, 1), actions=(1, 3)):
    """Compare first-action sets of One-move (N=1) and N=3 under the same backup."""
    states = sorted({0} | {min(a, u) for a in arrivals for u in actions})
    out = {}
    for name, step in (("memory_two_slots", toy_step), ("memoryless", memoryless_step)):
        backups = {"optimal_dp": optimal_backup(arrivals, step, actions),
                   "fixed_low": lambda x, t: actions[0], "fixed_high": lambda x, t: actions[-1]}
        for bname, backup in backups.items():
            disagree = [dict(t=t, x=x,
                             n1=sorted(_first_actions(x, t, 1, backup, arrivals, step, actions)),
                             n3=sorted(_first_actions(x, t, 3, backup, arrivals, step, actions)))
                        for t in range(len(arrivals)) for x in states]
            disagree = [d for d in disagree if d["n1"] != d["n3"]]
            out[f"{name}/{bname}"] = dict(disagreements=len(disagree), examples=disagree[:3])
    return dict(arrivals=list(arrivals), actions=list(actions), capacity=3, results=out)


# ------------------------------------------------------------- compute floor

# Qwen/Qwen3-4B-Instruct-2507 public config (assumption, not read from the GPU):
QWEN3_4B = dict(hidden=2560, layers=36, intermediate=9728, q_heads=32, kv_heads=8, head_dim=128)
RTX4090_BF16_FP32ACC_TFLOPS = 165.2   # Ada whitepaper dense tensor peak (assumption)


def matmul_params(cfg=QWEN3_4B):
    h, qd, kvd = cfg["hidden"], cfg["q_heads"] * cfg["head_dim"], cfg["kv_heads"] * cfg["head_dim"]
    per_layer = h * qd + 2 * h * kvd + qd * h + 3 * h * cfg["intermediate"]
    return cfg["layers"] * per_layer


def prefill_flops(q, context_end, cfg=QWEN3_4B):
    """Dense matmul + causal attention FLOPs for one q-token chunk ending at context_end
    (lm_head excluded: only sampled positions need logits)."""
    if not 0 < q <= context_end:
        raise ValueError("need 0 < q <= context_end")
    keys = q * context_end - q * (q - 1) / 2
    attn = 4 * cfg["layers"] * cfg["q_heads"] * cfg["head_dim"] * keys
    return 2 * matmul_params(cfg) * q + attn


def flop_floor_ms(q, context_end, peak_tflops=RTX4090_BF16_FP32ACC_TFLOPS):
    return prefill_flops(q, context_end) / (peak_tflops * 1e12) * 1000


def required_tflops(q, context_end, ms):
    return prefill_flops(q, context_end) / (ms / 1000) / 1e12


def gemini_affine_iteration_ms(decode, prefill_backlog_tokens, theta=(12.31, 0.049, 8.73)):
    """Gemini round1 4.1 law with P_k = prompt tokens awaiting/undergoing prefill."""
    return theta[0] + theta[1] * decode + theta[2] * prefill_backlog_tokens / 1000


# ------------------------------------------------- unprofiled decode wall time

def decode_wall_intervals(episodes, max_gap_s=0.3):
    """Pure-decode snapshot intervals: no waiting, no prefill-phase request in the first
    frame, no new request in the second. Each decoder gains one token per scheduler
    iteration, so median Delta output_tokens of persisting decoders counts iterations."""
    rows = []
    for e in episodes:
        frames = [f for f in e["native_frames"] if f.get("complete") and f.get("requests") is not None]
        for a, b in zip(frames, frames[1:]):
            ra = {r["request_id"]: r for r in a["requests"]}
            rb = {r["request_id"]: r for r in b["requests"]}
            if not ra or a["waiting"] or b["waiting"] or not set(rb) <= set(ra):
                continue
            if any(r["output_tokens"] == 0 for r in ra.values()):
                continue
            gains = [rb[k]["output_tokens"] - ra[k]["output_tokens"] for k in rb]
            dt = b["sampled_at"] - a["sampled_at"]
            if not gains or min(gains) < 1 or not 0 < dt <= max_gap_s:
                continue
            rows.append(dict(seed_block=e["seed_block"], prompt_tokens=e["prompt_tokens"],
                             dt_ms=dt * 1000, iterations=statistics.median(gains), decode=len(ra),
                             kv_tokens=sum(r["prompt_tokens"] + r["output_tokens"] for r in ra.values())))
    return rows


def _wall_features(r):
    n = r["iterations"]
    return [n, n * r["decode"], n * r["kv_tokens"] / 1000]


def fit_decode_wall(rows):
    beta = r1.nnls([_wall_features(r) for r in rows], [r["dt_ms"] for r in rows])
    return dict(intercept_ms=beta[0], per_decode_request_ms=beta[1], per_kilo_kv_ms=beta[2])


def _pred(model, r):
    return r["iterations"] * (model["intercept_ms"] + model["per_decode_request_ms"] * r["decode"]
                              + model["per_kilo_kv_ms"] * r["kv_tokens"] / 1000)


def decode_wall_lobo(rows, profile_law, token_theta=(12.31, 0.049), host_gap_ms=4.118):
    """Leave-one-seed-block-out interval MAE (ms) of competing per-iteration laws."""
    blocks = sorted({r["seed_block"] for r in rows})
    prof = dict(intercept_ms=profile_law["intercept_ms"],
                per_decode_request_ms=profile_law["per_decode_request_ms"],
                per_kilo_kv_ms=profile_law["per_kilo_decode_kv_ms"])
    fixed = {"profiled_kernel_busy_law": prof,
             "profiled_law_plus_round1_F3_host_gap": dict(prof, intercept_ms=prof["intercept_ms"] + host_gap_ms),
             "token_model_theta0_thetaD": dict(intercept_ms=token_theta[0], per_decode_request_ms=token_theta[1],
                                               per_kilo_kv_ms=0.0)}
    err = defaultdict(list)
    for held in blocks:
        test = [r for r in rows if r["seed_block"] == held]
        fitted = fit_decode_wall([r for r in rows if r["seed_block"] != held])
        for r in test:
            err["unprofiled_fit_lobo"].append(abs(_pred(fitted, r) - r["dt_ms"]))
            for name, model in fixed.items():
                err[name].append(abs(_pred(model, r) - r["dt_ms"]))
    return {k: dict(mae_ms=statistics.mean(v), intervals=len(v)) for k, v in err.items()}


def decode_wall_by_bin(rows, profile_law):
    bins = defaultdict(list)
    for r in rows:
        bins[(r["prompt_tokens"], min(r["decode"] // 4 * 4, 28))].append(r)
    out = []
    for (p, d), rs in sorted(bins.items()):
        if len(rs) < 20:
            continue
        D = statistics.mean(r["decode"] for r in rs)
        K = statistics.mean(r["kv_tokens"] for r in rs)
        wall = sum(r["dt_ms"] for r in rs) / sum(r["iterations"] for r in rs)
        busy = r1.predict_ms(profile_law, dict(scheduled_tokens=D, decode_requests=D, decode_key_tokens_sum=K,
                                               prefill_query_key_product_sum=0))
        out.append(dict(prompt_tokens=p, decode_bin=d, intervals=len(rs), mean_decode=D, mean_kv_tokens=K,
                        unprofiled_wall_ms=wall, profiled_law_busy_ms=busy, wall_minus_busy_ms=wall - busy))
    return out


def snapshot_prefill_slope(episodes, max_gap_s=0.3):
    """Exploratory: dt = iterations*(a + b*D + c*K) + p*prefill_tokens, where prefill
    tokens come from scheduler `computed_tokens` deltas. Reported per prompt cell to show
    the slope is NOT stable (bookkeeping lead / knee), not as a calibration."""
    data = defaultdict(lambda: ([], []))
    for e in episodes:
        frames = [f for f in e["native_frames"] if f.get("complete") and f.get("requests") is not None]
        for a, b in zip(frames, frames[1:]):
            ra = {r["request_id"]: r for r in a["requests"]}
            rb = {r["request_id"]: r for r in b["requests"]}
            dec = [k for k in ra if ra[k]["output_tokens"] > 0 and k in rb]
            dt = b["sampled_at"] - a["sampled_at"]
            if not dec or not 0 < dt <= max_gap_s:
                continue
            n = statistics.median(rb[k]["output_tokens"] - ra[k]["output_tokens"] for k in dec)
            if n < 1:
                continue
            P = sum(min(r["computed_tokens"], r["prompt_tokens"])
                    - (min(ra[k]["computed_tokens"], ra[k]["prompt_tokens"]) if k in ra else 0)
                    for k, r in rb.items())
            live = [r for r in ra.values() if r["output_tokens"] > 0]
            K = sum(r["prompt_tokens"] + r["output_tokens"] for r in live)
            X, y = data[e["prompt_tokens"]]
            X.append([n, n * len(live), n * K / 1000, P / 1000])
            y.append(dt * 1000)
    out = {}
    for p, (X, y) in sorted(data.items()):
        beta = r1.nnls(X, y)
        out[p] = dict(intervals=len(y), intercept_ms=beta[0], per_kilo_prefill_token_ms=beta[3])
    return out


def decode_residence(episodes):
    by = defaultdict(list)
    for e in episodes:
        by[e["cell"]] += [r["terminal_at"] - r["first_content_at"] for r in e["rows"] if r["status"] == "complete"]
    return {c: dict(median_s=statistics.median(v), max_s=max(v), n=len(v)) for c, v in sorted(by.items())}


# ----------------------------------------------- Astra log-linear duration family

def _loglinear_features(b):
    return [1.0, b["scheduled_tokens"] / 1000, float(b["decode_requests"]),
            b["decode_key_tokens_sum"] / 1000, b["prefill_query_key_product_sum"] / 1e6]


def fit_loglinear(batches):
    X = [_loglinear_features(b) for b in batches]
    return r1._lstsq(X, [math.log(b["kernel_busy_union_us"] / 1000) for b in batches], list(range(5)))


def predict_loglinear(beta, b):
    return math.exp(sum(c * x for c, x in zip(beta, _loglinear_features(b))))


def loglinear_vs_hinge_loco(batches, knee=32):
    cells = defaultdict(list)
    for b in batches:
        cells[r1.condition_key(b)].append(b)
    err = []
    for key, held in cells.items():
        beta = fit_loglinear([b for k, g in cells.items() if k != key for b in g])
        err += [abs(predict_loglinear(beta, b) - b["kernel_busy_union_us"] / 1000) for b in held]
    hinge = r1.leave_one_condition_out(batches, knee)["hinge_batch_law"]
    beta, law = fit_loglinear(batches), r1.fit_law(batches, knee)
    curve = []
    for q in (128, 512, 1024, 2048, 4096):
        b = dict(scheduled_tokens=q, decode_requests=0, decode_key_tokens_sum=0,
                 prefill_query_key_product_sum=q * (q + 1) / 2)
        curve.append(dict(prefill_tokens=q, loglinear_ms=predict_loglinear(beta, b),
                          hinge_ms=r1.predict_ms(law, b), flop_floor_ms=flop_floor_ms(q, q)))
    return dict(loglinear=dict(mae_ms=statistics.mean(err), max_abs_ms=max(err)),
                hinge=dict(mae_ms=hinge["mae_ms"], max_abs_ms=hinge["max_abs_ms"]), curve=curve)


# ------------------------------------------------------------------- driver

def run(profile_path, archive_path):
    batches = json.loads(Path(profile_path).read_text())["batches"]
    law = r1.fit_law(batches, 32)
    episodes = r1.load_train_archive(archive_path)
    rows = decode_wall_intervals(episodes)
    gem48 = gemini_affine_iteration_ms(0, 4096)
    return dict(
        schema="opus-round2-counterexamples-v1", calibration_accepted=False, performance_accepted=False,
        reservation_invariant=reservation_invariant_witness(),
        sign_flip=dict(p_floor_one_sided={n: sign_flip_p_floor(n) for n in (4, 5, 8)},
                       p_floor_two_sided={n: sign_flip_p_floor(n, True) for n in (4, 5, 8)}),
        horizon=horizon_witness(),
        compute_floor=dict(
            matmul_params=matmul_params(),
            chunk2048_floor_ms=flop_floor_ms(2048, 2048), chunk2048_profiled_busy_ms=103.0,
            chunk2048_profiled_required_tflops=required_tflops(2048, 2048, 103.0),
            gemini_iteration_ms_with_4096_backlog=gem48,
            gemini_required_tflops_for_2048_chunk=required_tflops(2048, 2048, gem48),
            gemini_thetaP_marginal_tflops=required_tflops(1000, 1000, 8.73)),
        loglinear=loglinear_vs_hinge_loco(batches),
        decode_wall=dict(intervals=len(rows), fit_all=fit_decode_wall(rows),
                         lobo=decode_wall_lobo(rows, law), by_bin=decode_wall_by_bin(rows, law)),
        snapshot_prefill_slope_exploratory=snapshot_prefill_slope(episodes),
        tpot_max_long_insertions_round1_vs_corrected={
            str(L): dict(round1_wall_12_31=r1.max_insertions_under_tpot(law, 12.31, L, 4096, decode_kv=4200),
                         unprofiled_wall_10_0=r1.max_insertions_under_tpot(law, 10.0, L, 4096, decode_kv=4200))
            for L in (50, 112)},
        decode_residence=decode_residence(episodes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    R = "results/mpc-native-4090-scout"
    p.add_argument("--profile", default=f"{R}/native-profile-diagnostic.json")
    p.add_argument("--train-archive", default=f"{R}/native-identification-train-evidence.tar.gz")
    p.add_argument("--output")
    a = p.parse_args()
    text = json.dumps(run(a.profile, a.train_archive), indent=2, sort_keys=True, allow_nan=False)
    Path(a.output).write_text(text) if a.output else print(text)


if __name__ == "__main__":
    main()
