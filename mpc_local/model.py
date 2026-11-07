"""R21 Opus candidate: fitted at-admission public-cost/age/service model.

Local CPU candidate only; no native calibration, adequacy or MPC permit.

Fitted components (from offline Train records only):

* TTFT and terminal token interval as ridge-linear functions of public
  admission features (prompt size, admitted public cost stock and active
  count at admission), each with an empirical residual quantile table;
* realised output as an empirical ratio-to-requested-``max_tokens`` quantile
  table, so ``output <= max_tokens`` always holds and the mass at the cap is
  a learned *length* (non-stop) probability, not a natural EOS;
* cleanup (last content to whole-cost release) as an empirical quantile table.

Rollout (``Model.predict``) uses the exact Source144 cost admission rule:
admit iff ``stock + cost <= budget`` else reject immediately; the whole
original public cost is held until predicted terminal plus cleanup; a lowered
budget never evicts or clips admitted cost. Explicit plan budgets cover the
first ``len(plan)`` slots; later slots use the frozen protocol continuation
(stock above ``reactive_threshold`` -> lowest action, else highest action)
evaluated on the *predicted* stock. Future offers come only from the supplied
``Forecast`` (rate and marks) with a fixed RNG; no actual future arrival,
output or service is read. Old requests are reconstructed only from causal
public fields: their admission stock is the observed cost still held by
requests admitted no later than them (a lower bound), and remaining
first-content / output are sampled from the fitted distributions conditioned
on the observed elapsed age and observed output tokens.

Rollout clock: ``state.now`` is the caller decision time in the declared
clock domain; old requests are conditioned on ``sampled_at`` (defaulting to
``now`` only for ``clock_domain == "simulation"``; native states without it
are refused, as are future or older-than-``max_snapshot_age_s`` samples).
Offers arrive only during ``horizon_s``. The rollout ends at
``now + horizon_s + tail_s``: ``cohort_good`` counts cohort members (old plus
horizon offers) predicted good by then, ``window_good`` only those terminal by
``now + horizon_s``. Tail offers are omitted because under the at-admission
service law they cannot change any reported field.

Uncertainty: ``n_scenarios`` stratified quantile levels, permuted per request
by a stable hash, shared across plans (common random numbers). Prediction
fields are scenario means.

Known shortcoming of v1 (falsified on the Opus demo plant at h3/h5): service
is fixed at admission; a later admission does not slow requests in service.

v2 (``fit(..., dynamics="decode_load")``, schema ``r21-opus-model-v2``) keeps
the v1 TTFT/output/cleanup components but replaces the at-admission token
interval with an event-driven decode law: each decoding request emits tokens
at ``1 / max(min, w . f(prompt, stock(t), nactive(t)) + residual_i)`` where
``stock(t)``/``nactive(t)`` are the *instantaneous* public held cost/count,
so admissions slow everyone already decoding and releases speed them up. ``w``
is fitted on offline token-weighted public-load labels
(``decode_load_stock``/``decode_load_nactive``: the public load at each emitted
token), which makes ``E[token interval] = w . mean_token_weighted(f)`` exact
for a linear law. v2 requires offline ``finish_reason`` labels for every
record that reached ``max_tokens``: reaching the cap alone does not identify
API ``length`` vs natural ``stop`` (refused unless labelled or the explicit
``at_cap_unlabeled="not_good"`` lower-bound policy is chosen). Offers after
``horizon_s`` are still omitted (an optimistic omission under v2: tail offers
would slow tail decoding).

v3 (``fit(..., dynamics="decode_load_terminal")``, schema
``r21-opus-model-v3``; preregistration r21-candidate-fit-preregistration-v1)
separates three post-first phases: decode spacing (event-driven, fitted on
token-weighted instantaneous public held stock/count INCLUDING the subject
owner), and TWO distinct endpoints from the last decode: API terminal overhead
(last decode -> client API terminal) and owner release (last decode ->
whole-cost gateway release). The label ``owner_release_extra_s`` is the SIGNED
offset release - API terminal: H16 Q3 permits gateway owner release BEFORE the
client API terminal, so release is NOT forced >= API terminal; only
release >= last decode is required (fit refuses api+extra < 0). The two
endpoint offsets are sampled as independent empirical marginals (their joint
dependence is not modelled). The canonical
``token_interval_s`` label is terminal-inclusive; decode spacing is
``token_interval_s - api_terminal_overhead_s / (output - 1)``. SLO TPOT is
scored terminal-inclusively at the API terminal; cost is retained until owner
release. Output priors are empirical per public (prompt, max) geometry (no
cross-geometry cap-ratio extrapolation; unseen geometry or observed progress
beyond the geometry support is refused), and at-cap finish keeps the offline
stop/length/unlabelled split. Received ``terminal_at``/``finish_reason`` are
honoured (unknown finish is never scored good) and old-output age uses each
request's own ``output_sampled_at``. Forecast offers continue into the tail
(until ``Config.offered_end_s`` if declared) only as congestion: cohort
scores count original owners plus offers inside the horizon, and
admitted/rejected/cost/stock stop at the horizon. Only v3 reads the new
received fields; v1/v2 refuse received terminals and older output counters.

All versions cap forecast offers at ``Config.offered_end_s`` when declared
(default ``None`` preserves earlier behaviour).

All versions carry every represented owner to the caller's decision time:
when ``sampled_at < now`` no owner is released in ``(sampled_at, now]``, so
``final_stock == stock(state) + cost_in - cost_out`` exactly (clamped release,
never a silent pre-now disappearance).

Both versions refuse (``pending_age_ood``) an old pending request whose
elapsed age exceeds every fitted TTFT outcome: such a request is outside the
fitted support and no instant first content/release is imputed.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import platform
import random
from dataclasses import asdict, replace
from typing import Any, Iterable, Mapping

import numpy as np

from mpc_lab.contracts import CostSpec

from .contracts import Config, Forecast, Prediction, PublicRequest, PublicState

__all__ = [
    "MODEL_VERSION", "MODEL_VERSION_V3", "FEATURE_SETS", "TRAIN_SEEDS", "HELDOUT_SEEDS",
    "ModelError", "Model", "fit", "normalize_records",
]

MODEL_VERSION = "r21-opus-admission-service-v1"
SCHEMA = "r21-opus-model-v1"
MODEL_VERSION_V2 = "r21-opus-admission-service-v2"
SCHEMA_V2 = "r21-opus-model-v2"
MODEL_VERSION_V3 = "r21-opus-admission-service-v3"
SCHEMA_V3 = "r21-opus-model-v3"
DYNAMICS = {SCHEMA: None, SCHEMA_V2: "decode_load", SCHEMA_V3: "decode_load_terminal"}
V3_LABELS = ("decode_load_stock", "decode_load_nactive", "api_terminal_overhead_s",
             "owner_release_extra_s", "finish_reason")
DYN_FEATURES = ("one", "prompt_k", "stock_16", "nactive_16")
FINISH_REASONS = ("stop", "length")
AT_CAP_POLICIES = ("refuse", "not_good")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "constant": ("one",),
    "load": ("one", "prompt_k", "stock_16", "nactive_16"),
    "load_x": ("one", "prompt_k", "stock_16", "nactive_16", "prompt_x_stock"),
}
TRAIN_SEEDS = frozenset(range(2101001, 2101009))
VALIDATION_SEEDS = frozenset(range(2102001, 2102005))
HELDOUT_SEEDS = frozenset(range(2103001, 2103013))
RECORD_FIELDS = (
    "seed", "prompt_tokens", "max_tokens", "output_tokens", "stock_at_admission",
    "nactive_at_admission", "ttft_s", "token_interval_s", "cleanup_s",
)
N_QUANTILES = 33
COST_REL_TOL = 1e-12
_MIN_TTFT = 1e-3
_MIN_TPOT = 1e-4


class ModelError(ValueError):
    pass


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def _features(names: tuple[str, ...], prompt: float, stock: float, nactive: float) -> list[float]:
    vals = {
        "one": 1.0, "prompt_k": prompt / 1024.0, "stock_16": stock / 16.0,
        "nactive_16": nactive / 16.0, "prompt_x_stock": (prompt / 1024.0) * (stock / 16.0),
    }
    return [vals[n] for n in names]


def _get(rec: Any, name: str) -> Any:
    if isinstance(rec, Mapping):
        return rec.get(name)
    return getattr(rec, name, None)


def normalize_records(records: Iterable[Any]) -> list[dict]:
    """Validate offline Train records; labels are never live policy inputs."""
    out = []
    for rec in records:
        row = {k: _get(rec, k) for k in RECORD_FIELDS}
        for k in ("seed", "prompt_tokens", "max_tokens", "output_tokens", "nactive_at_admission"):
            if type(row[k]) is not int:
                raise ModelError(f"invalid_record_{k}")
        if row["prompt_tokens"] < 0 or row["max_tokens"] <= 0 or row["nactive_at_admission"] < 0:
            raise ModelError("invalid_record_range")
        if not 1 <= row["output_tokens"] <= row["max_tokens"]:
            raise ModelError("invalid_record_output_tokens")
        for k in ("stock_at_admission", "ttft_s", "cleanup_s"):
            if not _finite(row[k]) or row[k] < 0:
                raise ModelError(f"invalid_record_{k}")
        ti = row["token_interval_s"]
        if row["output_tokens"] > 1:
            if not _finite(ti) or ti < 0:
                raise ModelError("invalid_record_token_interval_s")
        elif ti is not None and not _finite(ti):
            raise ModelError("invalid_record_token_interval_s")
        row = {k: (float(v) if isinstance(v, float) else v) for k, v in row.items()}
        out.append(row)
    return out


def _quantiles(values: np.ndarray) -> list[float]:
    grid = np.linspace(0.0, 1.0, N_QUANTILES)
    return [float(v) for v in np.quantile(values, grid)]


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    a = x.T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(a, x.T @ y)


def fit(
    records: Iterable[Any],
    cost_spec: CostSpec = CostSpec(),
    *,
    features: str = "load_x",
    allowed_seeds: Iterable[int] | None = None,
    data_source: str = "unspecified",
    ridge: float = 1e-6,
    n_scenarios: int = 8,
    dynamics: str | None = None,
    at_cap_unlabeled: str = "refuse",
) -> "Model":
    """Fit the candidate on Train records only (heldout seeds always refused).

    ``dynamics=None`` reproduces v1 exactly; ``"decode_load"`` fits v2;
    ``"decode_load_terminal"`` fits v3.
    """
    if dynamics not in DYNAMICS.values():
        raise ModelError("unknown_dynamics")
    if at_cap_unlabeled not in AT_CAP_POLICIES:
        raise ModelError("unknown_at_cap_policy")
    records = list(records)
    if features not in FEATURE_SETS:
        raise ModelError("unknown_feature_set")
    if type(n_scenarios) is not int or not 1 <= n_scenarios <= 64:
        raise ModelError("invalid_n_scenarios")
    allowed = TRAIN_SEEDS if allowed_seeds is None else frozenset(allowed_seeds)
    if allowed & HELDOUT_SEEDS:
        raise ModelError("heldout_seed_refused")
    rows = normalize_records(records)
    for r in rows:
        if r["seed"] in HELDOUT_SEEDS:
            raise ModelError("heldout_seed_refused")
        if r["seed"] not in allowed:
            raise ModelError("seed_not_allowed")
    names = FEATURE_SETS[features]
    if len(rows) < len(names) + 2:
        raise ModelError("too_few_records")
    x = np.array([_features(names, r["prompt_tokens"], r["stock_at_admission"],
                            r["nactive_at_admission"]) for r in rows])
    ttft = np.array([r["ttft_s"] for r in rows])
    w_ttft = _ridge(x, ttft, ridge)
    multi = [i for i, r in enumerate(rows) if r["output_tokens"] > 1]
    if len(multi) < len(names) + 2:
        raise ModelError("too_few_decode_records")
    xm = x[multi]
    tpot = np.array([rows[i]["token_interval_s"] for i in multi])
    w_tpot = _ridge(xm, tpot, ridge)
    ratio = np.array([r["output_tokens"] / r["max_tokens"] for r in rows])
    params = {
        "schema": SCHEMA,
        "version": MODEL_VERSION,
        "features": features,
        "feature_names": list(names),
        "n_scenarios": n_scenarios,
        "cost_spec": asdict(cost_spec),
        "cost_fingerprint": cost_spec.fingerprint,
        "ttft": {"weights": [float(v) for v in w_ttft],
                 "residual_q": _quantiles(ttft - x @ w_ttft)},
        "tpot": {"weights": [float(v) for v in w_tpot],
                 "residual_q": _quantiles(tpot - xm @ w_tpot)},
        "output_ratio_q": _quantiles(ratio),
        "p_length": float(np.mean([r["output_tokens"] >= r["max_tokens"] for r in rows])),
        "cleanup_q": _quantiles(np.array([r["cleanup_s"] for r in rows])),
        "manifest": {
            "data_source": data_source,
            "n_records": len(rows),
            "seeds": sorted({r["seed"] for r in rows}),
            "records_sha256": _sha(rows),
            "allowed_seeds": sorted(allowed),
            "ridge": ridge,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "flags": {"native_calibrated": False, "mpc_permit": False},
    }
    if dynamics is None:
        return Model(params)
    if dynamics == "decode_load_terminal":
        return Model(_v3_params(params, records, rows, ridge, at_cap_unlabeled))
    return Model(_v2_params(params, records, rows, ridge, at_cap_unlabeled))


def _v3_params(params: dict, records: list, rows: list[dict], ridge: float,
               at_cap_unlabeled: str) -> dict:
    cost_spec = CostSpec(**params["cost_spec"])
    labels, dec_x, dec_y = [], [], []
    for rec, r in zip(records, rows):
        ls, ln, api, extra, fr = (_get(rec, k) for k in V3_LABELS)
        if not (_finite(api) and _finite(extra) and api >= 0):
            raise ModelError("missing_phase_labels")
        if api + extra < -1e-9:  # signed extra; release may precede API terminal
            raise ModelError("release_before_last_decode")
        if fr is not None and fr not in FINISH_REASONS:
            raise ModelError("invalid_finish_reason")
        at_cap = r["output_tokens"] >= r["max_tokens"]
        if fr == "length" and not at_cap:
            raise ModelError("invalid_finish_reason")
        if fr is None and not at_cap:
            raise ModelError("missing_finish_reason")
        if r["output_tokens"] > 1:
            if not (_finite(ls) and _finite(ln)):
                raise ModelError("missing_decode_load_labels")
            own = cost_spec.estimate(r["prompt_tokens"], r["max_tokens"])
            if ln < 1 or ls < own * (1 - 1e-9):
                raise ModelError("decode_load_excludes_owner")
            d = r["token_interval_s"] - api / (r["output_tokens"] - 1)
            if d < -1e-9:
                raise ModelError("inconsistent_terminal_interval")
            dec_x.append(_features(DYN_FEATURES, r["prompt_tokens"], ls, ln))
            dec_y.append(max(0.0, d))
        labels.append((ls, ln, api, extra, fr))
    if len(dec_y) < len(DYN_FEATURES) + 2:
        raise ModelError("too_few_decode_records")
    xd, yd = np.array(dec_x), np.array(dec_y)
    w_dec = _ridge(xd, yd, ridge)
    geoms: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(rows):
        geoms.setdefault((r["prompt_tokens"], r["max_tokens"]), []).append(i)
    table = []
    for (pt, mt), idx in sorted(geoms.items()):
        at_cap = [i for i in idx if rows[i]["output_tokens"] >= mt]
        n_unlab = sum(labels[i][4] is None for i in at_cap)
        if n_unlab and at_cap_unlabeled == "refuse":
            raise ModelError("finish_reason_ambiguous_at_cap")
        n_stop = sum(labels[i][4] == "stop" for i in at_cap)
        table.append({
            "prompt_tokens": pt, "max_tokens": mt, "n": len(idx),
            "output_q": _quantiles(np.array([rows[i]["output_tokens"] for i in idx], float)),
            "n_at_cap": len(at_cap), "n_stop_at_cap": n_stop, "n_unlabeled_at_cap": n_unlab,
            "p_stop_at_cap": (n_stop / len(at_cap)) if at_cap else 0.0,
        })
    return {
        "schema": SCHEMA_V3, "version": MODEL_VERSION_V3, "dynamics": "decode_load_terminal",
        "features": params["features"], "feature_names": params["feature_names"],
        "n_scenarios": params["n_scenarios"], "cost_spec": params["cost_spec"],
        "cost_fingerprint": params["cost_fingerprint"], "ttft": params["ttft"],
        "decode": {"feature_names": list(DYN_FEATURES), "weights": [float(v) for v in w_dec],
                   "residual_q": _quantiles(yd - xd @ w_dec),
                   "target": "token_interval_s - api_terminal_overhead_s/(output-1)",
                   "load": "token-weighted public held stock/count including the owner"},
        "api_terminal_overhead_q": _quantiles(np.array([l[2] for l in labels])),
        "owner_release_extra_q": _quantiles(np.array([l[3] for l in labels])),
        "owner_release_offset_q": _quantiles(np.array([max(0.0, l[2] + l[3]) for l in labels])),
        "output_prior": {"kind": "geometry_empirical", "policy": at_cap_unlabeled,
                         "geometries": table},
        "manifest": dict(params["manifest"], labels_sha256=_sha(labels)),
        "environment": params["environment"], "flags": params["flags"],
    }


def _v2_params(params: dict, records: list, rows: list[dict], ridge: float,
               at_cap_unlabeled: str) -> dict:
    labels = []
    for rec, r in zip(records, rows):
        ls, ln, fr = (_get(rec, k) for k in ("decode_load_stock", "decode_load_nactive",
                                              "finish_reason"))
        if r["output_tokens"] > 1 and not (_finite(ls) and _finite(ln) and ls >= 0 and ln >= 0):
            raise ModelError("missing_decode_load_labels")
        if fr is not None and fr not in FINISH_REASONS:
            raise ModelError("invalid_finish_reason")
        if fr == "length" and r["output_tokens"] < r["max_tokens"]:
            raise ModelError("invalid_finish_reason")
        labels.append((ls, ln, fr))
    multi = [i for i, r in enumerate(rows) if r["output_tokens"] > 1]
    xd = np.array([_features(DYN_FEATURES, rows[i]["prompt_tokens"], labels[i][0], labels[i][1])
                   for i in multi])
    tpot = np.array([rows[i]["token_interval_s"] for i in multi])
    w_dec = _ridge(xd, tpot, ridge)
    at_cap = [i for i, r in enumerate(rows) if r["output_tokens"] >= r["max_tokens"]]
    n_unlab = sum(labels[i][2] is None for i in at_cap)
    if n_unlab and at_cap_unlabeled == "refuse":
        raise ModelError("finish_reason_ambiguous_at_cap")
    n_stop = sum(labels[i][2] == "stop" for i in at_cap)
    p = dict(params)
    p.pop("tpot")
    p.update({
        "schema": SCHEMA_V2,
        "version": MODEL_VERSION_V2,
        "dynamics": "decode_load",
        "decode": {"feature_names": list(DYN_FEATURES), "weights": [float(v) for v in w_dec],
                   "residual_q": _quantiles(tpot - xd @ w_dec)},
        "finish_at_cap": {"n_at_cap": len(at_cap), "n_stop": n_stop, "n_unlabeled": n_unlab,
                          "policy": at_cap_unlabeled,
                          "p_stop": (n_stop / len(at_cap)) if at_cap else 0.0},
    })
    p["manifest"] = dict(p["manifest"], labels_sha256=_sha(labels))
    return p


def _stable_unit(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


class Model:
    def __init__(self, params: dict):
        if not isinstance(params, dict) or params.get("schema") not in DYNAMICS:
            raise ModelError("invalid_model_schema")
        self.dynamics = params.get("dynamics")
        if DYNAMICS[params["schema"]] != self.dynamics:
            raise ModelError("invalid_model_dynamics")
        if params.get("features") not in FEATURE_SETS or \
                tuple(params.get("feature_names", ())) != FEATURE_SETS[params["features"]]:
            raise ModelError("invalid_model_features")
        if params.get("flags") != {"native_calibrated": False, "mpc_permit": False}:
            raise ModelError("invalid_model_flags")
        self._p = copy.deepcopy(params)
        self._names = FEATURE_SETS[params["features"]]
        self.cost_spec = CostSpec(**params["cost_spec"])
        if self.cost_spec.fingerprint != params["cost_fingerprint"]:
            raise ModelError("cost_fingerprint_mismatch")
        k = len(self._names)
        self._grid = np.linspace(0.0, 1.0, N_QUANTILES)
        self.n_scenarios = int(params["n_scenarios"])
        if self.dynamics == "decode_load_terminal":
            self._init_v3(params, k)
            return
        dec = params["decode"] if self.dynamics else params["tpot"]
        self._tpot_names = DYN_FEATURES if self.dynamics else self._names
        if self.dynamics:
            fa = params.get("finish_at_cap", {})
            if tuple(dec.get("feature_names", ())) != DYN_FEATURES or \
                    fa.get("policy") not in AT_CAP_POLICIES or \
                    not (_finite(fa.get("p_stop")) and 0.0 <= fa["p_stop"] <= 1.0):
                raise ModelError("invalid_model_weights")
            self._p_stop = float(fa["p_stop"])
        self._w_ttft = np.array(params["ttft"]["weights"], dtype=float)
        self._w_tpot = np.array(dec["weights"], dtype=float)
        tables = [params["ttft"]["residual_q"], dec["residual_q"],
                  params["output_ratio_q"], params["cleanup_q"]]
        if self._w_ttft.shape != (k,) or self._w_tpot.shape != (len(self._tpot_names),) or \
                any(len(t) != N_QUANTILES or not all(_finite(v) for v in t) for t in tables) or \
                not np.all(np.isfinite(self._w_ttft)) or not np.all(np.isfinite(self._w_tpot)):
            raise ModelError("invalid_model_weights")
        self._q_ttft, self._q_tpot, self._q_ratio, self._q_clean = (np.array(t) for t in tables)

    def _init_v3(self, params: dict, k: int) -> None:
        dec = params["decode"]
        self._tpot_names = DYN_FEATURES
        self._w_ttft = np.array(params["ttft"]["weights"], dtype=float)
        self._w_tpot = np.array(dec["weights"], dtype=float)
        tables = [params["ttft"]["residual_q"], dec["residual_q"],
                  params["api_terminal_overhead_q"], params["owner_release_extra_q"],
                  params["owner_release_offset_q"]]
        prior = params.get("output_prior", {})
        geoms = {}
        for g in prior.get("geometries", ()):
            q = g.get("output_q", ())
            if len(q) != N_QUANTILES or not all(_finite(v) and 1 <= v <= g["max_tokens"]
                                                for v in q) or \
                    not (_finite(g.get("p_stop_at_cap")) and 0 <= g["p_stop_at_cap"] <= 1):
                raise ModelError("invalid_model_weights")
            geoms[(g["prompt_tokens"], g["max_tokens"])] = (np.array(q), float(g["p_stop_at_cap"]))
        if tuple(dec.get("feature_names", ())) != DYN_FEATURES or not geoms or \
                prior.get("kind") != "geometry_empirical" or \
                prior.get("policy") not in AT_CAP_POLICIES or \
                self._w_ttft.shape != (k,) or self._w_tpot.shape != (len(DYN_FEATURES),) or \
                any(len(t) != N_QUANTILES or not all(_finite(v) for v in t) for t in tables) or \
                not np.all(np.isfinite(self._w_ttft)) or not np.all(np.isfinite(self._w_tpot)):
            raise ModelError("invalid_model_weights")
        self._geoms = geoms
        self._q_ttft, self._q_tpot, self._q_api, self._q_extra, self._q_rel = (
            np.array(t) for t in tables)

    # -- identity -------------------------------------------------------
    def to_dict(self) -> dict:
        d = copy.deepcopy(self._p)
        d["fingerprint"] = self.fingerprint
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        d = copy.deepcopy(data)
        fp = d.pop("fingerprint", None)
        m = cls(d)
        if fp != m.fingerprint:
            raise ModelError("fingerprint_mismatch")
        return m

    @property
    def fingerprint(self) -> str:
        body = {k: v for k, v in self._p.items() if k != "environment"}
        return _sha(body)

    # -- components -----------------------------------------------------
    def ttft_mean(self, prompt: float, stock: float, nactive: float) -> float:
        return float(np.dot(self._w_ttft, _features(self._names, prompt, stock, nactive)))

    def tpot_mean(self, prompt: float, stock: float, nactive: float) -> float:
        """v1: at-admission load; v2: instantaneous public load."""
        return float(np.dot(self._w_tpot, _features(self._tpot_names, prompt, stock, nactive)))

    def _q(self, table: np.ndarray, u: float) -> float:
        return float(np.interp(u, self._grid, table))

    def expected_output(self, max_tokens: int, prompt_tokens: int | None = None) -> float:
        if self.dynamics == "decode_load_terminal":
            return float(np.mean([self._sample_out3(prompt_tokens, max_tokens, 0, u)
                                  for u in self._levels()]))
        return float(np.mean([self._sample_output(max_tokens, 0, u) for u in self._levels()]))

    def _sample_out3(self, prompt: int, max_tokens: int, observed: int, u: float,
                     *, request_id: str | None = None) -> int:
        g = self._geoms.get((prompt, max_tokens))
        if g is None:  # no cap-ratio extrapolation across distinct geometries
            raise ModelError("output_geometry_ood")
        if observed >= max_tokens:
            return max_tokens
        cand = np.clip(np.round(g[0]), 1, max_tokens)
        cand = cand[cand > observed]
        if cand.size == 0:  # observed progress beyond fitted support: no imputed finish
            raise ModelError("output_progress_ood")
        return int(np.quantile(cand, u, method="inverted_cdf"))

    def _sample_cond(self, table: np.ndarray, age: float, u: float, reason: str) -> float:
        cand = np.maximum(table, 0.0)
        cand = cand[cand > age]
        if cand.size == 0:
            raise ModelError(reason)
        return float(np.quantile(cand, u)) - age

    def _levels(self) -> list[float]:
        s = self.n_scenarios
        return [(j + 0.5) / s for j in range(s)]

    def _sample_output(self, max_tokens: int, observed: int, u: float) -> int:
        cand = np.clip(np.round(self._q_ratio * max_tokens), 1, max_tokens)
        cand = cand[cand > observed]
        if cand.size == 0:
            return min(max_tokens, observed + 1) if observed < max_tokens else max_tokens
        return int(np.quantile(cand, u, method="inverted_cdf"))

    def _sample_ttft_remaining(self, base: float, elapsed: float, u: float) -> float:
        cand = np.maximum(base + self._q_ttft, _MIN_TTFT)
        cand = cand[cand > elapsed]
        if cand.size == 0:  # outside fitted support: never impute instant first content
            raise ModelError("pending_age_ood")
        return float(np.quantile(cand, u)) - elapsed

    def _service(self, prompt: int, max_tokens: int, stock: float, nactive: int,
                 u: tuple[float, float, float, float]) -> tuple[float, int, float, float]:
        ttft = max(_MIN_TTFT, self.ttft_mean(prompt, stock, nactive) + self._q(self._q_ttft, u[0]))
        out = self._sample_output(max_tokens, 0, u[1])
        tpot = max(_MIN_TPOT, self.tpot_mean(prompt, stock, nactive) + self._q(self._q_tpot, u[2]))
        clean = max(0.0, self._q(self._q_clean, u[3]))
        return ttft, out, tpot, clean

    def _u(self, key: str, j: int) -> tuple[float, float, float, float]:
        s = self.n_scenarios
        h = _stable_unit(key)
        return tuple(((j + (h >> (16 * c)) % s) % s + 0.5) / s for c in range(4))  # type: ignore

    # -- rollout --------------------------------------------------------
    def _validate(self, state: PublicState, plan: tuple[float, ...], forecast: Forecast,
                  config: Config) -> None:
        if not isinstance(state, PublicState) or not state.complete:
            raise ModelError("incomplete_state")
        if not _finite(state.now) or not _finite(state.budget):
            raise ModelError("invalid_state")
        obs = self._observed_at(state)
        if not _finite(obs) or obs > state.now or state.now - obs > config.max_snapshot_age_s:
            raise ModelError("stale_or_future_state")
        if not plan or any(b not in config.actions for b in plan):
            raise ModelError("unsupported_plan_budget")
        if not _finite(forecast.rate) or forecast.rate < 0:
            raise ModelError("invalid_forecast_rate")
        if forecast.rate > 0 and not forecast.marks:
            raise ModelError("forecast_marks_missing")
        for p, m in forecast.marks:
            if type(p) is not int or type(m) is not int or p < 0 or m <= 0:
                raise ModelError("invalid_forecast_mark")
        if not _finite(config.tail_s) or config.tail_s < 0:
            raise ModelError("invalid_tail_s")
        for name in ("slot_s", "horizon_s"):
            if not _finite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ModelError(f"invalid_{name}")
        if config.offered_end_s is not None and not _finite(config.offered_end_s):
            raise ModelError("invalid_offered_end_s")
        v3 = self.dynamics == "decode_load_terminal"
        seen = set()
        for r in state.requests:
            if r.request_id in seen:
                raise ModelError("duplicate_request_id")
            seen.add(r.request_id)
            expected = self.cost_spec.estimate(r.prompt_tokens, r.max_tokens)
            if not math.isclose(r.cost, expected, rel_tol=COST_REL_TOL, abs_tol=0.0):
                raise ModelError("original_cost_mismatch")
            if r.admitted_at > obs or r.scheduled_at > r.admitted_at:
                raise ModelError("request_chronology")
            if r.first_at is not None and not r.admitted_at <= r.first_at <= obs:
                raise ModelError("request_chronology")
            if not 0 <= r.observed_output_tokens <= r.max_tokens:
                raise ModelError("invalid_observed_output")
            if r.first_at is None and r.observed_output_tokens:
                raise ModelError("output_without_first")
            ts = r.output_sampled_at
            if ts is not None and (not _finite(ts) or not r.admitted_at <= ts <= obs or (
                    r.first_at is not None and ts < r.first_at)):
                raise ModelError("request_chronology")
            if r.first_at is not None and ts is None and state.clock_domain != "simulation":
                raise ModelError("missing_output_sampled_at")  # never relabel an old counter
            if r.terminal_at is None:
                if r.finish_reason is not None:
                    raise ModelError("finish_without_terminal")
            elif r.first_at is None or not _finite(r.terminal_at) or \
                    not r.first_at <= r.terminal_at <= obs:
                raise ModelError("request_chronology")
            if r.finish_reason is not None and (
                    r.finish_reason not in FINISH_REASONS or (
                        r.finish_reason == "length" and r.observed_output_tokens != r.max_tokens)):
                raise ModelError("invalid_finish_reason")
            if v3:
                if r.first_at is not None and r.observed_output_tokens < 1:
                    raise ModelError("invalid_observed_output")
            else:
                if r.terminal_at is not None:
                    raise ModelError("unsupported_received_terminal")
                if ts is not None and ts != obs:
                    raise ModelError("unsupported_output_age")

    @staticmethod
    def _observed_at(state: PublicState) -> float:
        if state.sampled_at is not None:
            return state.sampled_at
        if state.clock_domain == "simulation":
            return state.now
        raise ModelError("missing_sampled_at")

    def _old(self, state: PublicState, j: int, config: Config) -> list[tuple]:
        # Old requests are conditioned on what was observed at sampled_at.
        now = self._observed_at(state)
        reqs = sorted(state.requests, key=lambda r: (r.admitted_at, r.request_id))
        rows = []
        for i, r in enumerate(reqs):
            stock_adm = math.fsum(q.cost for q in reqs[:i])
            n_adm = i
            u = self._u("old:" + r.request_id, j)
            tpot = max(_MIN_TPOT, self.tpot_mean(r.prompt_tokens, stock_adm, n_adm)
                       + self._q(self._q_tpot, u[2]))
            if r.first_at is None:
                base = self.ttft_mean(r.prompt_tokens, stock_adm, n_adm)
                first = now + self._sample_ttft_remaining(base, now - r.admitted_at, u[0])
                out = self._sample_output(r.max_tokens, 0, u[1])
                terminal = first + (out - 1) * tpot
            else:
                first = r.first_at
                k = r.observed_output_tokens
                if k >= 4:  # observed per-request rate replaces the fitted mean
                    tpot = max(_MIN_TPOT, (now - r.first_at) / (k - 1))
                out = self._sample_output(r.max_tokens, k, u[1]) if k < r.max_tokens else k
                terminal = max(now, r.first_at + (out - 1) * tpot) if out > k else now
            # a represented owner is still held at the caller's decision time
            # (state.now >= sampled_at); its whole cost leaves strictly later
            release = max(state.now + 1e-6,
                          terminal + max(0.0, self._q(self._q_clean, u[3])))
            good = self._good(out, r.max_tokens, first - r.scheduled_at, first, terminal,
                              r.scheduled_at, config)
            rows.append((release, r.cost, terminal, good))
        return rows

    @staticmethod
    def _good(out: int, max_tokens: int, ttft: float, first: float, terminal: float,
              scheduled: float, config: Config) -> bool:
        if out <= 1 or out >= max_tokens:  # cap reached = length, not API stop
            return False
        if ttft > config.slo_ttft_s or terminal - scheduled > config.http_deadline_s:
            return False
        return (terminal - first) / (out - 1) <= config.slo_tpot_s

    def _tape(self, state: PublicState, forecast: Forecast, config: Config, j: int,
              until: float | None = None) -> list:
        rng = random.Random(forecast.seed * 1009 + j)
        t, tape = state.now, []
        end = state.now + config.horizon_s if until is None else until
        if config.offered_end_s is not None:  # public offered boundary, shared by all arms
            end = min(end, config.offered_end_s)
        if forecast.rate <= 0:
            return tape
        while True:
            t += rng.expovariate(forecast.rate)
            if t >= end:
                return tape
            p, m = forecast.marks[rng.randrange(len(forecast.marks))]
            tape.append((t, p, m))

    def _scenario(self, state: PublicState, plan: tuple[float, ...], forecast: Forecast,
                  config: Config, j: int) -> dict:
        now, end = state.now, state.now + config.horizon_s
        rollout_end = end + config.tail_s
        held: list[tuple[float, int, float]] = []  # (release, uid, cost)
        stock_items: dict[int, float] = {}
        uid = 0
        cohort = window = 0
        for release, cost, terminal, good in self._old(state, j, config):
            heapq.heappush(held, (release, uid, cost))
            stock_items[uid] = cost
            uid += 1
            cohort += good and terminal <= rollout_end
            window += good and terminal <= end
        cost_out = 0.0

        def release_until(t: float) -> None:
            nonlocal cost_out
            while held and held[0][0] <= t:
                rel, k, c = heapq.heappop(held)
                del stock_items[k]
                if rel > now:
                    cost_out += c

        lo, hi = min(config.actions), max(config.actions)
        nslots = max(1, math.ceil(config.horizon_s / config.slot_s - 1e-12))
        budgets: dict[int, float] = {i: b for i, b in enumerate(plan[:nslots])}

        def budget_at(t: float) -> float:
            i = min(int((t - now) / config.slot_s), nslots - 1)
            for s in range(len(budgets), i + 1):
                release_until(now + s * config.slot_s)
                stock = math.fsum(stock_items.values())
                budgets[s] = lo if stock > config.reactive_threshold else hi
            return budgets[i]

        admitted = rejected = 0
        cost_in = 0.0
        for n, (t, p, m) in enumerate(self._tape(state, forecast, config, j)):
            b = budget_at(t)
            release_until(t)
            stock = math.fsum(stock_items.values())
            c = self.cost_spec.estimate(p, m)
            if stock + c > b:
                rejected += 1
                continue
            ttft, out, tpot, clean = self._service(p, m, stock, len(stock_items),
                                                   self._u(f"new:{n}", j))
            first = t + ttft
            terminal = first + (out - 1) * tpot
            heapq.heappush(held, (terminal + clean, uid, c))
            stock_items[uid] = c
            uid += 1
            admitted += 1
            cost_in += c
            good = self._good(out, m, ttft, first, terminal, t, config)
            cohort += good and terminal <= rollout_end
            window += good and terminal <= end
        budget_at(end - 1e-9)
        release_until(end)
        return {"cohort_good": cohort, "window_good": window, "admitted": admitted,
                "rejected": rejected, "final_stock": math.fsum(stock_items.values()),
                "cost_in": cost_in, "cost_out": cost_out}

    def _u_stop(self, key: str, j: int) -> float:
        s = self.n_scenarios
        return ((j + _stable_unit("stop:" + key) % s) % s + 0.5) / s

    def _good2(self, r: dict, terminal: float, config: Config) -> bool:
        out = r["out"]
        if out <= 1 or (out >= r["max"] and not r["stop"]):
            return False
        if r["first"] - r["sched"] > config.slo_ttft_s or \
                terminal - r["sched"] > config.http_deadline_s:
            return False
        return (terminal - r["first"]) / (out - 1) <= config.slo_tpot_s

    def _scenario_dyn(self, state: PublicState, plan: tuple[float, ...], forecast: Forecast,
                      config: Config, j: int) -> dict:
        """Event-driven v2 rollout; decode rates change only at admit/release."""
        obs = self._observed_at(state)
        now, end = state.now, state.now + config.horizon_s
        rollout_end = end + config.tail_s
        w0, w1, w2, w3 = (float(v) for v in self._w_tpot)
        held: dict[int, dict] = {}
        stock0 = math.fsum(r.cost for r in state.requests)
        reqs = sorted(state.requests, key=lambda r: (r.admitted_at, r.request_id))
        for i, q in enumerate(reqs):
            key = "old:" + q.request_id
            u = self._u(key, j)
            r = {"cost": q.cost, "prompt": q.prompt_tokens, "max": q.max_tokens,
                 "sched": q.scheduled_at, "first": q.first_at,
                 "clean": max(0.0, self._q(self._q_clean, u[3])),
                 "resid": self._q(self._q_tpot, u[2])}
            if q.first_at is None:
                stock_adm = math.fsum(x.cost for x in reqs[:i])
                base = self.ttft_mean(q.prompt_tokens, stock_adm, i)
                r["first"] = obs + self._sample_ttft_remaining(base, obs - q.admitted_at, u[0])
                r["out"] = self._sample_output(q.max_tokens, 0, u[1])
                r["phase"], r["rem"] = "pre", float(r["out"] - 1)
            else:
                k = max(1, q.observed_output_tokens)
                r["out"] = self._sample_output(q.max_tokens, k, u[1]) if k < q.max_tokens else k
                if r["out"] > k:
                    r["phase"], r["rem"] = "dec", float(r["out"] - k)
                    if k >= 4:  # anchor this request's residual on its observed rate
                        r["resid"] = (obs - q.first_at) / (k - 1) - (
                            w0 + w1 * q.prompt_tokens / 1024.0 + w2 * stock0 / 16.0
                            + w3 * len(reqs) / 16.0)
                else:
                    r["phase"], r["terminal"] = "cln", obs
                    r["release"] = max(now + 1e-6, obs + r["clean"])
            r["stop"] = r["out"] < r["max"] or self._u_stop(key, j) < self._p_stop
            held[i] = r
        uid = len(held)
        lo, hi = min(config.actions), max(config.actions)
        nslots = max(1, math.ceil(config.horizon_s / config.slot_s - 1e-12))
        budgets: list[float] = []
        tape = self._tape(state, forecast, config, j)
        ai = 0
        admitted = rejected = cohort = window = 0
        cost_in = cost_out = 0.0
        final_stock = None
        t = obs
        stock, n = stock0, len(held)

        def finish(r: dict, at: float) -> None:
            nonlocal cohort, window
            r["phase"], r["terminal"] = "cln", at
            # represented owners are carried to decision time, never lost pre-now
            r["release"] = max(now + 1e-6, at + r["clean"])
            g = self._good2(r, at, config)
            cohort += g and at <= rollout_end
            window += g and at <= end

        for r in list(held.values()):
            if r["phase"] == "cln" and r["terminal"] == obs:
                g = self._good2(r, obs, config)
                cohort += g
                window += g
        while True:
            common = w0 + w2 * stock / 16.0 + w3 * n / 16.0
            best = (math.inf, 9, None)
            for k_, r in held.items():
                ph = r["phase"]
                if ph == "cln":
                    cand = (r["release"], 0, k_)
                elif ph == "pre":
                    cand = (r["first"], 1, k_)
                else:
                    r["tpot"] = max(_MIN_TPOT, common + w1 * r["prompt"] / 1024.0 + r["resid"])
                    cand = (t + r["rem"] * r["tpot"], 1, k_)
                if cand < best:
                    best = cand
            if len(budgets) < nslots:
                best = min(best, (now + len(budgets) * config.slot_s, 2, None))
            if ai < len(tape):
                best = min(best, (tape[ai][0], 3, None))
            if final_stock is None:
                best = min(best, (end, 4, None))
            t_new, kind, k_ = best
            if t_new > rollout_end or t_new == math.inf:
                break
            t_new = max(t_new, t)
            for r in held.values():
                if r["phase"] == "dec":
                    r["rem"] -= (t_new - t) / r["tpot"]
            t = t_new
            if kind == 0:
                r = held.pop(k_)
                if now < t <= end:
                    cost_out += r["cost"]
                stock, n = math.fsum(x["cost"] for x in held.values()), len(held)
            elif kind == 1:
                r = held[k_]
                if r["phase"] == "pre":
                    if r["rem"] <= 0:
                        finish(r, t)
                    else:
                        r["phase"] = "dec"
                else:
                    r["rem"] = 0.0
                    finish(r, t)
            elif kind == 2:
                budgets.append(plan[len(budgets)] if len(budgets) < len(plan)
                               else (lo if stock > config.reactive_threshold else hi))
            elif kind == 3:
                _, p, m = tape[ai]
                b = budgets[min(int((t - now) / config.slot_s), nslots - 1)]
                c = self.cost_spec.estimate(p, m)
                if stock + c > b:
                    rejected += 1
                else:
                    key = f"new:{ai}"
                    u = self._u(key, j)
                    ttft = max(_MIN_TTFT, self.ttft_mean(p, stock, n) + self._q(self._q_ttft, u[0]))
                    out = self._sample_output(m, 0, u[1])
                    held[uid] = {"cost": c, "prompt": p, "max": m, "sched": t, "first": t + ttft,
                                 "out": out, "phase": "pre", "rem": float(out - 1),
                                 "resid": self._q(self._q_tpot, u[2]),
                                 "clean": max(0.0, self._q(self._q_clean, u[3])),
                                 "stop": out < m or self._u_stop(key, j) < self._p_stop}
                    uid += 1
                    admitted += 1
                    cost_in += c
                    stock, n = math.fsum(x["cost"] for x in held.values()), len(held)
                ai += 1
            else:
                final_stock = stock
                if not held:
                    break
        if final_stock is None:
            final_stock = stock
        return {"cohort_good": cohort, "window_good": window, "admitted": admitted,
                "rejected": rejected, "final_stock": final_stock,
                "cost_in": cost_in, "cost_out": cost_out}

    @staticmethod
    def _good3(r: dict, api_terminal: float, config: Config) -> bool:
        """API-endpoint SLO: terminal-inclusive TPOT; unknown finish is never good."""
        out = r["out"]
        if out <= 1 or r["stop"] is not True:
            return False
        if r["first"] - r["sched"] > config.slo_ttft_s or \
                api_terminal - r["sched"] > config.http_deadline_s:
            return False
        return (api_terminal - r["first"]) / (out - 1) <= config.slo_tpot_s

    def _scenario_v3(self, state: PublicState, plan: tuple[float, ...], forecast: Forecast,
                     config: Config, j: int) -> dict:
        obs = self._observed_at(state)
        now, end = state.now, state.now + config.horizon_s
        rollout_end = end + config.tail_s
        w0, w1, w2, w3 = (float(v) for v in self._w_tpot)
        stock0 = math.fsum(r.cost for r in state.requests)
        n0 = len(state.requests)
        reqs = sorted(state.requests, key=lambda r: (r.admitted_at, r.request_id))
        cnt = {"cohort": 0, "window": 0, "undecided": 0}

        def rate(r: dict, stock: float, n: int) -> float:
            return max(_MIN_TPOT, w0 + w1 * r["prompt"] / 1024.0 + w2 * stock / 16.0
                       + w3 * n / 16.0 + r["resid"])

        def finish(r: dict, last: float) -> None:
            api_t = last + r["api"]
            g = self._good3(r, api_t, config)
            if r["cohort"]:
                cnt["cohort"] += g and api_t <= rollout_end
                cnt["window"] += g and api_t <= end
                cnt["undecided"] -= 1
            r["phase"] = "post"
            # represented owners are carried to decision time, never lost pre-now
            r["release"] = max(now + 1e-6, last + r["rel"])  # not forced >= api_t

        held: dict[int, dict] = {}
        for i, q in enumerate(reqs):
            key = "old:" + q.request_id
            u, u2 = self._u(key, j), self._u(key + "#v3", j)
            r = {"cost": q.cost, "prompt": q.prompt_tokens, "max": q.max_tokens,
                 "sched": q.scheduled_at, "first": q.first_at, "cohort": True,
                 "resid": self._q(self._q_tpot, u[2]),
                 "api": max(0.0, self._q(self._q_api, u[3])),
                 "rel": max(0.0, self._q(self._q_rel, u2[0]))}
            held[i] = r
            if q.terminal_at is not None:  # received API terminal: honour actual clocks
                r["out"], r["stop"] = q.observed_output_tokens, q.finish_reason == "stop"
                g = self._good3(r, q.terminal_at, config)
                cnt["cohort"] += g
                cnt["window"] += g
                age = obs - q.terminal_at
                r["phase"] = "post"
                r["release"] = max(now + 1e-6, q.terminal_at + self._sample_cond(
                    self._q_extra, age, u2[0], "release_age_ood") + age)
                continue
            cnt["undecided"] += 1
            if q.first_at is None:
                stock_adm = math.fsum(x.cost for x in reqs[:i])
                base = self.ttft_mean(q.prompt_tokens, stock_adm, i)
                r["first"] = obs + self._sample_ttft_remaining(base, obs - q.admitted_at, u[0])
                r["out"] = self._sample_out3(q.prompt_tokens, q.max_tokens, 0, u[1],
                                           request_id=q.request_id)
                r["phase"], r["rem"] = "pre", float(r["out"] - 1)
            else:
                k = q.observed_output_tokens
                ts = obs if q.output_sampled_at is None else q.output_sampled_at
                r["out"] = self._sample_out3(q.prompt_tokens, q.max_tokens, k, u[1],
                                           request_id=q.request_id)
                if k >= 4:  # anchor on this request's own observed rate, at its own sample time
                    r["resid"] = 0.0
                    r["resid"] = (ts - q.first_at) / (k - 1) - rate(r, stock0, n0)
                if r["out"] <= k:
                    r["stop"] = self._u_stop(key, j) < self._p_stop3(q)
                    finish(r, ts)
                    continue
                tp = rate(r, stock0, n0)
                rem = (r["out"] - k) - (obs - ts) / tp  # advance its own counter to obs
                if rem <= 0:
                    r["stop"] = r["out"] < r["max"] or self._u_stop(key, j) < self._p_stop3(q)
                    finish(r, ts + (r["out"] - k) * tp)
                    continue
                r["phase"], r["rem"] = "dec", rem
            r["stop"] = r["out"] < r["max"] or self._u_stop(key, j) < self._p_stop3(q)
        uid = len(held)
        lo, hi = min(config.actions), max(config.actions)
        nslots = max(1, math.ceil(config.horizon_s / config.slot_s - 1e-12))
        tape = self._tape(state, forecast, config, j, until=rollout_end)
        offer_end = tape[-1][0] if tape else now
        budgets: list[float] = []
        ai = admitted = rejected = 0
        cost_in = cost_out = 0.0
        final_stock = None
        t = obs
        stock, n = stock0, len(held)
        while True:
            if final_stock is not None and cnt["undecided"] == 0:
                break
            best = (math.inf, 9, None)
            for k_, r in held.items():
                ph = r["phase"]
                if ph == "post":
                    cand = (r["release"], 0, k_)
                elif ph == "pre":
                    cand = (r["first"], 1, k_)
                else:
                    r["tpot"] = rate(r, stock, n)
                    cand = (t + r["rem"] * r["tpot"], 1, k_)
                if cand < best:
                    best = cand
            nxt = now + len(budgets) * config.slot_s
            if nxt <= offer_end:
                best = min(best, (nxt, 2, None))
            if ai < len(tape):
                best = min(best, (tape[ai][0], 3, None))
            if final_stock is None:
                best = min(best, (end, 4, None))
            t_new, kind, k_ = best
            if t_new > rollout_end or t_new == math.inf:
                break
            t_new = max(t_new, t)
            for r in held.values():
                if r["phase"] == "dec":
                    r["rem"] -= (t_new - t) / r["tpot"]
            t = t_new
            if kind == 0:
                r = held.pop(k_)
                if now < t <= end:
                    cost_out += r["cost"]
                stock, n = math.fsum(x["cost"] for x in held.values()), len(held)
            elif kind == 1:
                r = held[k_]
                if r["phase"] == "pre" and r["rem"] > 0:
                    r["phase"] = "dec"
                else:
                    r["rem"] = 0.0
                    finish(r, t)
            elif kind == 2:
                s_ = len(budgets)
                budgets.append(plan[s_] if s_ < min(len(plan), nslots)
                               else (lo if stock > config.reactive_threshold else hi))
            elif kind == 3:
                _, p, m = tape[ai]
                inside = t < end
                b = budgets[min(int((t - now) / config.slot_s), len(budgets) - 1)]
                c = self.cost_spec.estimate(p, m)
                if stock + c > b:
                    rejected += inside
                else:
                    key = f"new:{ai}"
                    u, u2 = self._u(key, j), self._u(key + "#v3", j)
                    ttft = max(_MIN_TTFT, self.ttft_mean(p, stock, n) + self._q(self._q_ttft, u[0]))
                    out = self._sample_out3(p, m, 0, u[1])
                    held[uid] = {"cost": c, "prompt": p, "max": m, "sched": t, "first": t + ttft,
                                 "out": out, "phase": "pre", "rem": float(out - 1),
                                 "resid": self._q(self._q_tpot, u[2]),
                                 "api": max(0.0, self._q(self._q_api, u[3])),
                                 "rel": max(0.0, self._q(self._q_rel, u2[0])),
                                 "cohort": inside,
                                 "stop": out < m or self._u_stop(key, j) < self._geoms[(p, m)][1]}
                    uid += 1
                    if inside:  # tail offers are congestion only, never scored or costed
                        admitted += 1
                        cost_in += c
                        cnt["undecided"] += 1
                    stock, n = math.fsum(x["cost"] for x in held.values()), len(held)
                ai += 1
            else:
                final_stock = stock
        if final_stock is None:
            final_stock = stock
        return {"cohort_good": cnt["cohort"], "window_good": cnt["window"], "admitted": admitted,
                "rejected": rejected, "final_stock": final_stock,
                "cost_in": cost_in, "cost_out": cost_out}

    def _p_stop3(self, q: PublicRequest) -> float:
        return self._geoms[(q.prompt_tokens, q.max_tokens)][1]

    def predict(self, state: PublicState, plan: tuple[float, ...], forecast: Forecast,
                config: Config = Config()) -> Prediction:
        plan = tuple(float(b) for b in plan)
        self._validate(state, plan, forecast, config)
        run = {None: self._scenario, "decode_load": self._scenario_dyn,
               "decode_load_terminal": self._scenario_v3}[self.dynamics]
        rows = [run(state, plan, forecast, config, j) for j in range(self.n_scenarios)]
        mean = {k: math.fsum(r[k] for r in rows) / len(rows) for k in rows[0]}
        return Prediction(**mean)

    def with_scenarios(self, n: int) -> "Model":
        d = copy.deepcopy(self._p)
        d["n_scenarios"] = n
        return Model(d)


def config_with_horizon(config: Config, horizon_s: float) -> Config:
    return replace(config, horizon_s=horizon_s)
