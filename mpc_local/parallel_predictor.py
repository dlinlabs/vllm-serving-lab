"""Owned CPU prefix workers; no native authority or hard real-time guarantee."""

from dataclasses import asdict
import hashlib
from itertools import product
import json
import multiprocessing
import os
import time

from mpc_lab.contracts import finite
from mpc_local.contracts import Arrival, Config, Forecast, Prediction, PublicRequest, PublicState


def _initialize(model_type, model_bytes, fingerprint, barrier, warm_timeout):
    global _MODEL_TYPE, _MODEL_BYTES, _FINGERPRINT, _BARRIER, _WARM_TIMEOUT
    _MODEL_TYPE, _MODEL_BYTES, _FINGERPRINT = model_type, model_bytes, fingerprint
    _BARRIER, _WARM_TIMEOUT = barrier, warm_timeout


def _fresh_model():
    model = _MODEL_TYPE.from_dict(json.loads(_MODEL_BYTES))
    if model.fingerprint != _FINGERPRINT:
        raise ValueError("worker_model_fingerprint_mismatch")
    return model


def _warm():
    _fresh_model()
    _BARRIER.wait(timeout=_WARM_TIMEOUT)
    return os.getpid()


def _prediction(payload, prefix):
    data = json.loads(payload)
    state = data["state"]
    state["requests"] = tuple(PublicRequest(**r) for r in state["requests"])
    state["arrivals"] = tuple(Arrival(**a) for a in state["arrivals"])
    forecast = data["forecast"]
    forecast["marks"] = tuple(tuple(mark) for mark in forecast["marks"])
    config = data["config"]
    config["actions"] = tuple(config["actions"])
    model = _fresh_model()
    result = model.predict(PublicState(**state), prefix, Forecast(**forecast), Config(**config))
    if model.fingerprint != _FINGERPRINT:
        raise ValueError("worker_model_fingerprint_changed")
    if not isinstance(result, Prediction):
        raise ValueError("worker_invalid_prediction_type")
    return os.getpid(), result


class ParallelPredictor:
    """Use ``with ParallelPredictor(model) as backend: choose(state, backend)``.

    The supplied model type must have importable, spawn-safe ``from_dict`` and
    ``to_dict`` methods and a stable serialized fingerprint. The model is
    captured once; each prefix reconstructs a fresh instance. Metadata is a
    CPU mechanics receipt, not proof of fitting, calibration or native identity.

    Four workers are ready before entering the context. Both OneMove and MPC
    use this backend. Only one caller may use an instance at a time; use a
    separate context per independently evaluated arm. Timeouts/failures close
    the instance, and context exit terminates and joins only its owned pool.
    Caller and planner still account actual wall time and fence dispatch.
    """

    def __init__(self, model, *, startup_timeout_s=10.0):
        if not finite(startup_timeout_s, positive=True):
            raise ValueError("invalid_startup_timeout")
        if not callable(getattr(model, "to_dict", None)):
            raise ValueError("serialized_cpu_model_required")
        parameters = model.to_dict()
        fingerprint = model.fingerprint
        if (
            not isinstance(parameters, dict)
            or not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
            or parameters.get("fingerprint") != fingerprint
            or not callable(getattr(type(model), "from_dict", None))
            or parameters.get("model_version") == "native-id-unfitted-v1"
            or parameters.get("version") == "native-id-unfitted-v1"
        ):
            raise ValueError("serialized_cpu_model_required")
        self._model_bytes = json.dumps(parameters, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()
        self._model_type = type(model)
        self._fingerprint = fingerprint
        self._startup_timeout = startup_timeout_s
        self._pool = None
        self._closed = False
        self._pids = ()
        self._startup_elapsed = 0.0
        self._batches = 0
        self._submitted = 0
        self._last_error = None
        self._key = None
        self._results = {}
        self._consumed = set()
        self._deadline = None

    @property
    def fingerprint(self):
        return self._fingerprint

    @property
    def metadata(self):
        return {
            "backend": "r21-owned-parallel-prefix-v1",
            "start_method": "spawn", "worker_count": len(self._pids),
            "worker_pids": list(self._pids),
            "ready": self._pool is not None and not self._closed,
            "closed": self._closed, "model_fingerprint": self._fingerprint,
            "serialized_model_sha256": hashlib.sha256(self._model_bytes).hexdigest(),
            "model_type": self._model_type.__module__ + "." + self._model_type.__qualname__,
            "fresh_model_per_prediction": True,
            "startup_elapsed_s": self._startup_elapsed,
            "startup_excluded_from_solve": True,
            "batches": self._batches, "predictions_submitted": self._submitted,
            "last_error": self._last_error, "native_mpc_permitted": False,
            "hard_deadline_guarantee": False,
        }

    def __enter__(self):
        if self._closed or self._pool is not None:
            raise RuntimeError("parallel_context_not_reentrant")
        started = time.perf_counter()
        try:
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(
                processes=4, initializer=_initialize,
                initargs=(self._model_type, self._model_bytes, self._fingerprint,
                          context.Barrier(4), self._startup_timeout),
            )
            tasks = [self._pool.apply_async(_warm) for _ in range(4)]
            pids = [task.get(timeout=max(0.0, self._startup_timeout -
                                        (time.perf_counter() - started))) for task in tasks]
            if time.perf_counter() - started >= self._startup_timeout:
                raise multiprocessing.TimeoutError("parallel_startup_deadline")
            if len(set(pids)) != 4:
                raise RuntimeError("four_owned_workers_not_ready")
            self._pids = tuple(sorted(pids))
        except BaseException as exc:
            self._last_error = type(exc).__name__
            self.close()
            raise
        finally:
            self._startup_elapsed = time.perf_counter() - started
        return self

    def close(self):
        pool, self._pool = self._pool, None
        self._closed = True
        self._key, self._deadline = None, None
        self._results.clear()
        self._consumed.clear()
        if pool is not None:
            try:
                pool.terminate()
            finally:
                pool.join()

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def predict(self, state, plan, forecast, config=Config()):
        if self._pool is None or self._closed:
            raise RuntimeError("parallel_context_not_ready")
        try:
            prefix = tuple(plan)
            if (
                len(prefix) not in (1, 2)
                or any(not finite(b) or b not in (16.0, 32.0) for b in prefix)
                or not isinstance(state, PublicState) or not isinstance(forecast, Forecast)
                or not isinstance(config, Config)
                or not finite(config.solver_budget_s, positive=True)
            ):
                raise ValueError("invalid_parallel_input")
            key = json.dumps({
                "model_fingerprint": self._fingerprint, "state": asdict(state),
                "forecast": asdict(forecast), "config": asdict(config),
                "prefix_length": len(prefix),
            }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            # A repeated prefix starts a new solve, never a zero-cost cache replay.
            if key != self._key or prefix in self._consumed:
                self._key = key
                self._deadline = time.perf_counter() + config.solver_budget_s
                self._consumed = set()
                self._results = {}
                for candidate in product((16.0, 32.0), repeat=len(prefix)):
                    self._results[candidate] = self._pool.apply_async(_prediction, (key, candidate))
                    self._submitted += 1
                self._batches += 1
            remaining = self._deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("parallel_batch_deadline")
            pid, result = self._results[prefix].get(timeout=remaining)
            if pid not in self._pids:
                raise RuntimeError("unwarmed_replacement_worker")
            if time.perf_counter() >= self._deadline:
                raise TimeoutError("parallel_batch_deadline")
            self._consumed.add(prefix)
            return result
        except BaseException as exc:
            self._last_error = type(exc).__name__
            self.close()
            raise
