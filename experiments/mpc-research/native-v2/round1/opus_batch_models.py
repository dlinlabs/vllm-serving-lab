"""Round1 Opus CPU diagnostic: physical batch service, remaining work, identifiability.

Standard library only. Reads existing Train/exploratory artifacts; never the
final held-out seeds 160021-160024. Outputs are diagnostics, not calibration,
controller acceptance or performance evidence.

Contents
  * Lawson-Hanson NNLS (tiny dense problems).
  * Batch iteration law fitted to profiled GPU batches:
        tau_ms = a + c_d*D + c_k*K/1000 + b*max(0, s-knee)/1000 + c_att*QK/1e6
    s = scheduled tokens, D = decode requests, K = decode KV tokens,
    QK = sum prefill query*key products.  Knee profiled; ambiguity reported.
  * Batched vs exclusive-serial capacity bounds on window completions.
  * FIFO chunked-prefill backlog time and long-prefill TPOT interference.
  * EOS: paired-arm length agreement, age-conditional mean residual life with
    leave-one-seed-block-out value of age.
  * KV fraction per resident token.
  * Observational send-to-first-content by number of prefills already ahead.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

HELDOUT_SEEDS = frozenset(range(160021, 160025))
MAX_BATCHED_TOKENS = 2048
MAX_SEQS = 32
TRAIN_MEMBERS = ("native-id-train/interventions.json", "native-id-low-train/interventions.json")


# ---------------------------------------------------------------- linear algebra

def _solve(A, b):
    n = len(A)
    M = [list(map(float, row)) + [float(v)] for row, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise ValueError("singular design")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _lstsq(X, y, cols):
    A = [[sum(r[i] * r[j] for r in X) + (1e-12 if i == j else 0) for j in cols] for i in cols]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in cols]
    return _solve(A, b)


def nnls(X, y, iterations=200):
    """Lawson-Hanson active set: argmin ||X beta - y|| subject to beta >= 0."""
    n = len(X[0])
    beta, passive = [0.0] * n, []
    for _ in range(iterations):
        r = [v - sum(b * x for b, x in zip(beta, row)) for row, v in zip(X, y)]
        w = [sum(row[j] * e for row, e in zip(X, r)) for j in range(n)]
        free = [j for j in range(n) if j not in passive and w[j] > 1e-10]
        if not free:
            return beta
        passive.append(max(free, key=lambda j: w[j]))
        while True:
            z = dict(zip(passive, _lstsq(X, y, passive)))
            if all(v > 0 for v in z.values()):
                beta = [z.get(j, 0.0) for j in range(n)]
                break
            alpha = min(beta[j] / (beta[j] - z[j]) for j in passive if z[j] <= 0)
            beta = [beta[j] + alpha * (z.get(j, 0.0) - beta[j]) for j in range(n)]
            passive = [j for j in passive if beta[j] > 1e-12]
    raise RuntimeError("nnls did not converge")


# ------------------------------------------------------------ batch iteration law

FEATURES = ("intercept_ms", "per_decode_request_ms", "per_kilo_decode_kv_ms",
            "per_kilo_token_above_knee_ms", "per_mega_prefill_qk_ms")


def batch_features(batch, knee):
    s = batch["scheduled_tokens"]
    return [1.0, float(batch["decode_requests"]), batch["decode_key_tokens_sum"] / 1000,
            max(0.0, s - knee) / 1000, batch["prefill_query_key_product_sum"] / 1e6]


def baseline_features(batch, kind):
    if kind == "per_request_serial":   # additive per-request service, no batch sharing
        return [float(batch["prefill_requests"]), float(batch["decode_requests"])]
    if kind == "affine_tokens":        # no memory-bound floor / knee
        return [1.0, batch["scheduled_tokens"] / 1000]
    raise ValueError(kind)


def _target(batch, target):
    return batch[target] / 1000


def fit_law(batches, knee, target="kernel_busy_union_us"):
    beta = nnls([batch_features(b, knee) for b in batches], [_target(b, target) for b in batches])
    return dict(knee_tokens=knee, target=target, **dict(zip(FEATURES, beta)))


def predict_ms(law, batch):
    return sum(law[f] * x for f, x in zip(FEATURES, batch_features(batch, law["knee_tokens"])))


def sse(law, batches):
    return sum((predict_ms(law, b) - _target(b, law["target"])) ** 2 for b in batches)


def knee_profile(batches, knees=range(0, 1025, 32), tolerance=0.10, target="kernel_busy_union_us"):
    """Profile SSE over the knee. Knees within (1+tolerance)*min SSE are not
    distinguishable by this design: report a set, not a point estimate."""
    rows = [(k, sse(fit_law(batches, k, target), batches)) for k in knees]
    best_sse = min(v for _, v in rows)
    best = min(rows, key=lambda r: r[1])[0]
    ambiguous = [k for k, v in rows if v <= best_sse * (1 + tolerance) + 1e-9]
    support = sorted({b["scheduled_tokens"] for b in batches})
    below = max((s for s in support if s <= best), default=None)
    above = min((s for s in support if s > best), default=None)
    # A grid-flat SSE is not enough: with no observed batch sizes near the
    # knee, any knee inside the support gap fits the same points.
    bracketed = below is not None and above is not None and best - below <= 64 and above - best <= 64
    return dict(best_knee=best, indistinguishable_knees=[min(ambiguous), max(ambiguous)], tolerance=tolerance,
                scheduled_token_support=support, support_gap_around_knee=[below, above],
                knee_identified=bracketed and max(ambiguous) - min(ambiguous) <= 64,
                profile=[dict(knee=k, sse_ms2=v) for k, v in rows])


def condition_key(b):
    return (b["prefill_requests"], b["prefill_query_tokens"], b["decode_requests"])


def leave_one_condition_out(batches, knee, target="kernel_busy_union_us"):
    """Extrapolation test: each batch-composition cell is held out in turn."""
    cells = defaultdict(list)
    for b in batches:
        cells[condition_key(b)].append(b)
    errors = defaultdict(list)
    for key, held in sorted(cells.items()):
        train = [b for k, g in cells.items() if k != key for b in g]
        law = fit_law(train, knee, target)
        for kind in ("per_request_serial", "affine_tokens"):
            coef = nnls([baseline_features(b, kind) for b in train], [_target(b, target) for b in train])
            for b in held:
                pred = sum(c * x for c, x in zip(coef, baseline_features(b, kind)))
                errors[kind].append(abs(pred - _target(b, target)))
        errors["hinge_batch_law"] += [abs(predict_ms(law, b) - _target(b, target)) for b in held]
    return {k: dict(mae_ms=statistics.mean(v), max_abs_ms=max(v), batches=len(v)) for k, v in errors.items()}


# ------------------------------------------------------------- capacity screens

def serial_completion_bound(window_s, service_s):
    if service_s <= 0:
        raise ValueError("serial service must be positive")
    return int(window_s / service_s)


def batch_time_lower_bound_s(n, prompt_tokens, mean_output_tokens, law,
                             max_batched_tokens=MAX_BATCHED_TOKENS, max_seqs=MAX_SEQS):
    """Lower bound on GPU busy time to finish n requests from empty.

    Uses only tau >= a + b*max(0, s-knee) (other law terms are nonnegative) and
    convexity: sum_j max(0, s_j-knee) >= max(0, S-N*knee).  Iteration count N
    obeys token-budget and seq-limit constraints and per-request sequential
    decode.  Valid only if the law's terms lower-bound wall time per iteration."""
    if n <= 0:
        return 0.0
    a, b, knee = law["intercept_ms"], law["per_kilo_token_above_knee_ms"] / 1000, law["knee_tokens"]
    total_tokens = n * (prompt_tokens + mean_output_tokens - 1)
    n_min = max(math.ceil(total_tokens / max_batched_tokens), math.ceil(n * mean_output_tokens / max_seqs),
                math.ceil(prompt_tokens / max_batched_tokens) + math.ceil(mean_output_tokens) - 1)
    candidates = {n_min}
    if knee > 0 and total_tokens / knee > n_min:
        candidates |= {math.floor(total_tokens / knee), math.ceil(total_tokens / knee)}
    return min(N * a + b * max(0.0, total_tokens - N * knee) for N in candidates if N >= n_min) / 1000


def batch_completion_bound(window_s, prompt_tokens, mean_output_tokens, law, **limits):
    lo, hi = 0, 1
    while batch_time_lower_bound_s(hi, prompt_tokens, mean_output_tokens, law, **limits) <= window_s:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if batch_time_lower_bound_s(mid, prompt_tokens, mean_output_tokens, law, **limits) <= window_s:
            lo = mid
        else:
            hi = mid
    return lo


# ------------------------------------------------- prefill backlog / interference

def chunk_plan(prompt_tokens, decode_running, max_batched_tokens=MAX_BATCHED_TOKENS):
    """Chunks of one prompt under decode-first chunked prefill: (q, context)."""
    room = max_batched_tokens - decode_running
    if room <= 0:
        raise ValueError("no prefill room")
    chunks, done = [], 0
    while done < prompt_tokens:
        q = min(room, prompt_tokens - done)
        done += q
        chunks.append((q, done))
    return chunks


def chunk_ms(law, q, context, decode_running, decode_kv):
    return predict_ms(law, dict(scheduled_tokens=q + decode_running, decode_requests=decode_running,
                                decode_key_tokens_sum=decode_kv, prefill_query_key_product_sum=q * context))


def prefill_backlog_s(law, prompts_fifo, decode_running=0, decode_kv=0, host_floor_ms=0.0):
    """Time until the last FIFO prompt finishes prefill, one prompt per chunk.
    vLLM may pack several short prompts per iteration, so this over-states short
    backlogs; for long (compute-bound) prompts packing does not save GPU time."""
    total = 0.0
    for p in prompts_fifo:
        for q, ctx in chunk_plan(p, decode_running):
            total += max(host_floor_ms, chunk_ms(law, q, ctx, decode_running, decode_kv))
    return total / 1000


def interference_tpot_ms(law, decode_wall_ms, output_tokens, insertions, prompt_tokens,
                         decode_running=1, decode_kv=0):
    """Request-mean TPOT when `insertions` prompts are chunk-prefilled during a
    decoder's lifetime; each chunk iteration replaces one decode iteration."""
    if output_tokens < 2:
        return 0.0
    extra = sum(max(0.0, chunk_ms(law, q, c, decode_running, decode_kv) - decode_wall_ms)
                for q, c in chunk_plan(prompt_tokens, decode_running))
    return decode_wall_ms + insertions * extra / (output_tokens - 1)


def max_insertions_under_tpot(law, decode_wall_ms, output_tokens, prompt_tokens, slo_ms=50.0, **kw):
    if interference_tpot_ms(law, decode_wall_ms, output_tokens, 10001, prompt_tokens, **kw) <= slo_ms:
        return math.inf
    k = 0
    while interference_tpot_ms(law, decode_wall_ms, output_tokens, k + 1, prompt_tokens, **kw) <= slo_ms:
        k += 1
    return k


# ------------------------------------------------------------- remaining work/EOS

def paired_length_agreement(pairs):
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no paired completions")
    diffs = [abs(a - b) for a, b in pairs]
    return dict(pairs=len(pairs), identical_fraction=sum(d == 0 for d in diffs) / len(pairs),
                mean_abs_diff_tokens=statistics.mean(diffs),
                mean_abs_diff_when_different=statistics.mean([d for d in diffs if d] or [0]),
                over_10pct_fraction=sum(d > 0.1 * max(a, b) for d, (a, b) in zip(diffs, pairs)) / len(pairs))


def mean_residual_life(lengths):
    """Age a = output tokens already generated; MRL(a) = E[L-a | L>a]."""
    return {a: statistics.mean(L - a for L in lengths if L > a) for a in range(max(lengths))}


def age_samples(lengths):
    """Each decode age a request occupies, with its remaining tokens."""
    return [(a, L - a) for L in lengths for a in range(L)]


def age_value_held_out(by_block):
    """Leave-one-seed-block-out remaining-token MAE: unconditional mean residual
    vs age-conditional MRL from other blocks. Ages beyond training support use
    the last supported MRL and are counted as outside support."""
    out = []
    for held, lengths in sorted(by_block.items()):
        train = [L for k, v in by_block.items() if k != held for L in v]
        mrl = mean_residual_life(train)
        flat = statistics.mean(r for _, r in age_samples(train))
        samples = age_samples(lengths)
        top = max(mrl)
        out.append(dict(held_block=held, samples=len(samples),
                        mae_unconditional=statistics.mean(abs(r - flat) for _, r in samples),
                        mae_age_conditional=statistics.mean(abs(r - mrl[min(a, top)]) for a, r in samples),
                        outside_support=sum(a > top for a, _ in samples)))
    return out


def ttft_by_prefill_ahead(episodes, prompt_tokens, slo_s=0.5, warmup_s=1.0):
    """Send-to-first-content grouped by how many earlier-sent complete requests
    of the same episode had not yet produced first content at send time.
    Observational (Train): the count is a Gateway-observable proxy for prefill
    work ahead; it is not a randomized causal contrast."""
    table = defaultdict(list)
    for e in episodes:
        if e["prompt_tokens"] != prompt_tokens:
            continue
        rows = sorted((r for r in e["rows"] if r["status"] == "complete"
                       and r["sent_at"] >= e["start"] + warmup_s), key=lambda r: r["sent_at"])
        for i, r in enumerate(rows):
            ahead = sum(1 for o in rows[:i] if o["first_content_at"] > r["sent_at"])
            table[ahead].append(r["first_content_at"] - r["sent_at"])
    return [dict(ahead=k, requests=len(v), median_s=statistics.median(v),
                 within_slo_fraction=sum(x <= slo_s for x in v) / len(v)) for k, v in sorted(table.items())]


def kv_fraction_per_token(frames):
    xs, ys = [], []
    for f in frames:
        if f.get("complete") and f.get("requests") and f.get("kv_cache_usage") is not None:
            xs.append(sum(r["prompt_tokens"] + r["output_tokens"] for r in f["requests"]))
            ys.append(f["kv_cache_usage"])
    if not xs:
        raise ValueError("no KV frames")
    k = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    return dict(frames=len(xs), fraction_per_token=k, tokens_at_full=1 / k,
                max_abs_residual=max(abs(y - k * x) for x, y in zip(xs, ys)), max_observed=max(ys))


# --------------------------------------------------------------- data loading

def train_episodes(document):
    episodes = []
    for e in document["episodes"]:
        seed = e.get("seed_block", e.get("seed"))
        if seed in HELDOUT_SEEDS:
            raise ValueError(f"held-out seed {seed} must not be read")
        if e.get("split") != "train":
            raise ValueError("only Train episodes")
        episodes.append(e)
    return episodes


def load_train_archive(path):
    episodes = []
    with tarfile.open(path, "r:gz") as bundle:
        for name in TRAIN_MEMBERS:
            episodes += train_episodes(json.load(bundle.extractfile(name)))
    return episodes


def eos_analysis(episodes, window_s=90):
    pairs, cells, capacity = defaultdict(dict), defaultdict(lambda: defaultdict(list)), []
    for e in episodes:
        index = {rid: i for i, rid in enumerate(e["expected_ids"])}
        done = [r for r in e["rows"] if r["status"] == "complete"]
        for r in done:
            if r["finish_reason"] != "stop":
                raise ValueError("EOS analysis requires natural EOS")
            L = r["usage"]["completion_tokens"]
            pairs[(e["seed_block"], e["cell"])].setdefault(index[r["request_id"]], {})[e["arm"]] = L
            cells[e["cell"]][e["seed_block"]].append(L)
        inside = [r for r in done if r["terminal_at"] <= e["start"] + window_s]
        capacity.append(dict(episode_id=e["episode_id"], cell=e["cell"], prompt_tokens=e["prompt_tokens"],
                             complete_inside_window=len(inside),
                             mean_output_tokens=statistics.mean(r["usage"]["completion_tokens"] for r in inside)
                             if inside else 0.0))
    agreement = defaultdict(list)
    for (_, cell), m in pairs.items():
        agreement[cell] += [(v[0], v[1]) for v in m.values() if len(v) == 2]
    return dict(paired_agreement={c: paired_length_agreement(p) for c, p in sorted(agreement.items())},
                age_value={c: age_value_held_out(b) for c, b in sorted(cells.items())},
                length_support={c: [min(min(v) for v in b.values()), max(max(v) for v in b.values())]
                                for c, b in sorted(cells.items())}), capacity


def run(profile_path, archive_path=None, serial_service_s=None, token_theta0_ms=12.31):
    """token_theta0_ms: unprofiled per-iteration wall intercept from the PR33
    decode token-progress model (native-identification report, section 2)."""
    profile = json.loads(Path(profile_path).read_text())
    batches = profile["batches"]
    knees = knee_profile(batches)
    law = fit_law(batches, knees["best_knee"])
    decode1 = dict(scheduled_tokens=1, decode_requests=1, decode_key_tokens_sum=200,
                   prefill_query_key_product_sum=0)
    p4096 = [chunk_ms(law, q, c, 1, 4200) for q, c in chunk_plan(4096, 1)]
    result = dict(schema="opus-round1-batch-diagnostic-v1",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  profile_sha256=hashlib.sha256(Path(profile_path).read_bytes()).hexdigest(),
                  calibration_accepted=False, performance_accepted=False, batches=len(batches),
                  knee=knees, law_kernel_busy=law,
                  law_gpu_span=fit_law(batches, knees["best_knee"], target="gpu_span_us"),
                  loco=leave_one_condition_out(batches, knees["best_knee"]),
                  unprofiled_wall_minus_profiled_busy_ms_D1=token_theta0_ms - predict_ms(law, decode1),
                  long_prompt=dict(chunks_ms=p4096, exclusive_prefill_s=sum(p4096) / 1000,
                      max_insertions_tpot50={str(L): max_insertions_under_tpot(
                          law, token_theta0_ms, L, 4096, decode_kv=4200) for L in (50, 112)},
                      ttft_backlog_s={str(n): prefill_backlog_s(law, [4096] * n, 1, 4200) for n in (1, 2, 3)}))
    if archive_path:
        episodes = load_train_archive(archive_path)
        result["kv"] = kv_fraction_per_token(f for e in episodes for f in e["native_frames"])
        result["eos"], capacity = eos_analysis(episodes)
        result["ttft_by_prefill_ahead"] = {str(p): ttft_by_prefill_ahead(episodes, p) for p in (128, 4096)}
        for c in capacity:
            c["batch_bound"] = batch_completion_bound(90, c["prompt_tokens"], max(1.0, c["mean_output_tokens"]), law)
            c["batch_consistent"] = c["complete_inside_window"] <= c["batch_bound"]
            if serial_service_s:
                c["serial_bound"] = serial_completion_bound(90, serial_service_s[str(c["prompt_tokens"])])
                c["serial_consistent"] = c["complete_inside_window"] <= c["serial_bound"]
        result["capacity"] = capacity
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, help="native-profile-diagnostic.json")
    parser.add_argument("--train-archive", help="native-identification-train-evidence.tar.gz (Train only)")
    parser.add_argument("--prefill-proxy", help="native-prefill-combined.json (falsified serial reading)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    serial = None
    if args.prefill_proxy:
        p = json.loads(Path(args.prefill_proxy).read_text())["parameters"]
        serial = {str(t): p["first_overhead_s"] + p["prefill_s_per_token"] * t for t in (128, 4096)}
    out = run(args.profile, args.train_archive, serial)
    Path(args.output).write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    summary = {k: out[k] for k in ("law_kernel_busy", "loco", "unprofiled_wall_minus_profiled_busy_ms_D1",
                                   "long_prompt", "kv") if k in out}
    summary["knee"] = {k: v for k, v in out["knee"].items() if k != "profile"}
    print(json.dumps(summary, indent=1))
