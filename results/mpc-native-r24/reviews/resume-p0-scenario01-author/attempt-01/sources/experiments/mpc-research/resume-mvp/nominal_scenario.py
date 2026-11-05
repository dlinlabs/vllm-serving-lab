"""One causal nominal forecast, never the offered Eval tape or its generator.

Freeze the authorized Train prior once before decisions. Future arrivals are
Poisson(88/5); lengths are draws from the frozen Train histogram, not known
future requests. Mayor reported tokenizer validation for the exact Train replay
and asset fingerprint below; that proof is not re-executed here. It establishes
Train chat-template counts, NOT future tokenization, gateway cost or calibration.

Reuse ONE returned .causal / planner panel for MPC and OneMove. Neither policy
nor candidate path is an RNG input. Production callers must supply the actual
same-domain monotonic clock. The coordinator's source_digest must also include
this module AND the imported nominal_workload.py, not just its old three files.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from random import Random
import sys
import time
from typing import Callable

import nominal_workload as workload
from nominal_planner import CausalScenario, StateVersion
from mpc_lab.contracts import FutureOffer, Scenario, Snapshot, digest
from mpc_lab.controller import validate_snapshot


TRAIN_SHA256 = "40bd755b441d6b38c0f2827cdb81afe50f637a579c471a2b1b4ef72b26c6192f"
WORKLOAD_SHA256 = "614ae110e901110f875383b9ee0a66b08fe75e3a58a923b1d30dbd9c21d88bd4"
TOKENIZER_FINGERPRINT = "4daabcf7fab3bce457d30dbd14e7eb9c5e79d77c15916f6abdb85b7b8bf0d2c5"
TRAIN_REPLAY_SHA256 = "5e418b8939179bac16d146d08f7acfa4d617b78adc4c324caca8ec0893ba52e7"
MARKS = ((256, 128, 33), (1024, 128, 26), (3072, 128, 29))
RNG_NAMESPACE = "resume-p0-nominal-forecast-v1"
RNG_SEED = 5105
MAX_OFFERS = 512
HORIZON_S = 15.0  # scored 5s plus contention-only 10s tail
ASSUMPTIONS = (
    "Unqualified cpu-lifecycle-v1 / synthetic-unqualified-v1; no Native permission.",
    "Future lengths sampled from measured Train counts for the pinned tokenizer only.",
    "Mayor-reported Train tokenizer proof; TRANSFORMERS 5.22.1; proof commit pending at delivery.",
    "Future output=max_output=128, success=True, service_scale=1, zero transport delays.",
    "Old output fraction=1 and old success=True; no oracle completion/progress information.",
)


class ScenarioRefused(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise ScenarioRefused(reason)


def _time(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    _require(valid, "invalid_clock")
    return value


@dataclass(frozen=True)
class TrainPrior:
    clock_domain: str
    available_at: float
    marks: tuple[tuple[int, int, int], ...] = MARKS
    count: int = 88
    duration_s: float = 5.0
    train_sha256: str = TRAIN_SHA256
    workload_sha256: str = WORKLOAD_SHA256
    tokenizer_fingerprint: str = TOKENIZER_FINGERPRINT
    train_replay_sha256: str = TRAIN_REPLAY_SHA256

    @property
    def content_sha256(self):
        return digest({key: value for key, value in asdict(self).items()
                       if key not in ("clock_domain", "available_at")})


@dataclass(frozen=True)
class NominalForecast:
    causal: CausalScenario
    prior: TrainPrior
    rng_seed_material: str
    rng_runtime: str
    producer_sources: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...] = ASSUMPTIONS


def freeze_train_prior(train_path: str | Path, *, purpose: str, clock_domain: str,
                       clock: Callable[[], float] = time.monotonic) -> TrainPrior:
    """Read/hash ONLY the frozen Train file; do not call workload.load_tape (it regenerates)."""
    _require(purpose == "train", "train_only_prior")
    _require(type(clock_domain) is str and bool(clock_domain.strip()), "missing_clock_domain")
    path = Path(train_path)
    _require(path.name == "train.tape.json", "train_filename_required")
    started = _time(clock())
    with path.open("rb") as handle:
        raw = handle.read(16385)
    _require(hashlib.sha256(raw).hexdigest() == TRAIN_SHA256, "train_source_hash_mismatch")
    _require(hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest() == WORKLOAD_SHA256,
             "workload_source_hash_mismatch")
    tape = json.loads(raw)
    _require(tape["split"]["purpose"] == "train" and tape["schema"] == workload.TAPE_SCHEMA
             and tape["columns"] == list(workload.COLUMNS), "train_schema_or_purpose_mismatch")
    marks = tuple((p, m, n) for (p, m), n in
                  sorted(Counter((row[2], row[3]) for row in tape["rows"]).items()))
    _require(tape["count"] == len(tape["rows"]) == 88
             and tape["protocol"]["duration_ns"] == 5 * workload.NS_PER_SECOND
             and marks == MARKS, "unfrozen_train_prior")
    available = _time(clock())
    _require(available >= started, "nonmonotonic_clock")
    return TrainPrior(clock_domain, available, marks)


def make_scenario(prior: TrainPrior, snapshot: Snapshot, *, purpose: str,
                  clock: Callable[[], float] = time.monotonic) -> NominalForecast:
    """No tape I/O/refit here. Availability is sampled AFTER generation and hashing."""
    _require(purpose in ("train", "evaluation", "engineering"), "invalid_forecast_purpose")
    _require(type(prior) is TrainPrior and type(snapshot) is Snapshot, "typed_prior_snapshot_required")
    _require(prior == TrainPrior(prior.clock_domain, prior.available_at), "unfrozen_train_prior")
    started = _time(clock())
    reason = validate_snapshot(snapshot, now=started, clock_domain=prior.clock_domain,
                               expected_identity=snapshot.identity, max_age_s=0.5)
    _require(reason is None, reason or "invalid_snapshot")
    _require(_time(prior.available_at) <= snapshot.sampled_at, "prior_not_available_at_source_time")
    state = StateVersion.from_snapshot(snapshot)
    seed_material = json.dumps(
        [RNG_NAMESPACE, RNG_SEED, prior.content_sha256, asdict(state), snapshot.sampled_at],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    rng = Random(seed_material)
    offers, offset = [], 0.0
    seen = {request.request_id for request in snapshot.requests}
    while True:
        gap = rng.expovariate(prior.count / prior.duration_s)
        _require(type(gap) in (int, float) and math.isfinite(gap)
                 and gap > 0 and offset + gap > offset, "nonadvancing_or_invalid_arrival")
        offset += gap
        if offset >= HORIZON_S:
            break
        _require(len(offers) < MAX_OFFERS, "future_offer_cap_exceeded")
        rank = rng.randrange(prior.count)
        for prompt, maximum, count in prior.marks:
            if rank < count:
                break
            rank -= count
        rid = f"{RNG_NAMESPACE}:{snapshot.sequence}:{len(offers)}"
        _require(rid not in seen, "forecast_id_collision")
        offers.append(FutureOffer(rid, offset, prompt, maximum, maximum))
    scenario = Scenario(tuple(offers))
    sources = (
        ("nominal_scenario.py", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        ("nominal_workload.py", hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest()),
    )
    _require(sources[1][1] == prior.workload_sha256, "workload_source_hash_mismatch")
    runtime = f"Python {sys.version.split()[0]} stdlib random.Random(str, version=2)/expovariate/randrange"
    source_sha = digest({
        "prior": asdict(prior), "seed_material": seed_material, "rng_runtime": runtime,
        "sources": sources, "scenario": asdict(scenario), "assumptions": ASSUMPTIONS,
    })
    available = _time(clock())
    _require(available >= started, "nonmonotonic_clock")
    reason = validate_snapshot(snapshot, now=available, clock_domain=prior.clock_domain,
                               expected_identity=snapshot.identity, max_age_s=0.5)
    _require(reason is None, reason or "invalid_snapshot")
    causal = CausalScenario(scenario, state, source_sha, snapshot.sampled_at, available,
                            "development", purpose)
    return NominalForecast(causal, prior, seed_material, runtime, sources)
