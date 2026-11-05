"""Default-off worker registration; no product source or model replacement."""

from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading

from prepared_forward_capture import Resolver
from prepared_forward_hook import install
from prepared_forward_region import ForwardRegionRecorder


FLAG = "PR33_R24_PREPARED_FORWARD_ENABLED"
MARKER = "_r24_prepared_forward_registration"
SCHEMA = "r24-prepared-forward-worker-extension-v1"
FILES = (
    "prepared_forward_plugin.py", "prepared_forward_region.py",
    "prepared_forward_hook.py", "prepared_forward_capture.py", "queue_observer.py",
    "launch_queue_service.py",
)
NATIVE_PINS = {
    "prefill_lab.plugin": "7bdb7585047e42e8ec035e543fc6b1f79df425fc6706348989dd0856d42b82e2",
    "vllm.v1.worker.gpu_worker": "6994436e4547c0996ab555b3e814e8ccea26b650dfc1b378b97d640745309347",
    "vllm.v1.worker.gpu.model_runner":
        "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
}
RPC_LINES = {"prefill_lab_worker_begin": 430, "prefill_lab_worker_finish": 440}
EXECUTE_LINES = {"observed": 480, "native": 1630}
WIRE_LINE = 128
_REGISTRATION = None


def declaration():
    directory = Path(__file__).parent
    return {
        "schema": SCHEMA, "enabled": True, "observer": "light",
        "sources": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in FILES},
        "native_sources": dict(NATIVE_PINS),
    }


def validate_extension(extension):
    if (not isinstance(extension, dict) or extension.get("enabled") is not True
            or extension != declaration()):
        raise ValueError("prepared_worker_exact_source_extension_required")


def validate_plan(plan):
    validate_extension(plan.get("prepared_forward_extension"))
    trials = plan.get("trials")
    if (type(trials) is not list or not trials
            or any(not isinstance(trial, dict) or trial.get("observer") != "light"
                   for trial in trials)):
        raise ValueError("prepared_worker_all_light_plan_required")


def _file(path, digest):
    path = Path(path)
    if (any(part.is_symlink() for part in (path, *path.parents))
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
        raise ValueError("prepared_worker_source_changed")


def _function(function, digest, line):
    if not inspect.isfunction(function) or function.__code__.co_firstlineno != line:
        raise ValueError("prepared_worker_function_changed")
    _file(function.__code__.co_filename, digest)


def _research_sources():
    directory = Path(__file__).parent.resolve()
    for name, reference in {
        "prepared_forward_region.py": ForwardRegionRecorder,
        "prepared_forward_capture.py": Resolver,
        "prepared_forward_hook.py": install,
    }.items():
        module = sys.modules.get(reference.__module__)
        path = Path(inspect.getsourcefile(reference))
        if (getattr(module, reference.__name__, None) is not reference
                or path.resolve() != directory / name):
            raise ValueError("prepared_worker_imported_source_not_bound")
        _file(path, declaration()["sources"][name])
    module = sys.modules.get("queue_observer")
    if module is not None and Path(module.__file__).resolve() != directory / "queue_observer.py":
        raise ValueError("prepared_worker_imported_queue_source_not_bound")


def _load_sources():
    import prefill_lab.plugin as base

    _file(base.__file__, NATIVE_PINS["prefill_lab.plugin"])
    base.register()
    if not base._REGISTERED:
        raise ValueError("prepared_worker_parent_registration_required")
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR

    return base, Worker, NO_OP_KV_CONNECTOR


def register():
    global _REGISTRATION
    flag = os.environ.get(FLAG, "0")
    if flag not in ("0", "1"):
        raise ValueError("prepared_worker_boolean_launch_flag_required")
    if flag == "0":
        return
    if (os.environ.get("PREFILL_LAB_ENABLED") != "1"
            or os.environ.get("PR33_R23_QUEUE_OBSERVER_ENABLED") != "1"
            or any(os.environ.get("PREFILL_LAB_" + name, "0") != "0"
                   for name in ("STAGES", "LIVE_STATE", "ENVELOPE_SHADOW", "EVENT_CLOCK",
                                "EFFECTIVE_CALIBRATION", "DELAYED_ARRIVALS", "QUEUE_PROBE",
                                "QUEUE_CONFIRMATION", "ENTRY_HTTP"))):
        raise ValueError("prepared_worker_light_parent_mode_required")
    plan = json.loads(Path(os.environ["PR33_NATIVE_BATCH_PLAN"]).read_bytes())
    validate_plan(plan)
    _research_sources()
    if _REGISTRATION is not None:
        _REGISTRATION.check_owner()
        return
    base, worker, noop = _load_sources()
    registration = Registration(base, worker, noop)
    registration.install()
    _REGISTRATION = registration


class Registration:
    def __init__(self, base, worker, noop):
        self.base, self.worker, self.noop = base, worker, noop
        self.lease = None
        self.discard_required = False
        if getattr(sys.modules.get(worker.__module__), worker.__name__, None) is not worker:
            raise ValueError("prepared_worker_registry_identity_required")
        _file(inspect.getsourcefile(worker), NATIVE_PINS[worker.__module__])
        if MARKER in vars(worker):
            raise ValueError("prepared_worker_foreign_registration")
        self.originals = {}
        for name, line in RPC_LINES.items():
            function = vars(worker).get(name)
            _function(function, NATIVE_PINS["prefill_lab.plugin"], WIRE_LINE)
            original = inspect.unwrap(function)
            _function(original, NATIVE_PINS["prefill_lab.plugin"], line)
            self.originals[name] = function
        if (tuple(inspect.signature(self.originals["prefill_lab_worker_begin"]).parameters)
                != ("worker", "trial_id", "mode")
                or tuple(inspect.signature(self.originals["prefill_lab_worker_finish"]).parameters)
                != ("worker",)):
            raise ValueError("prepared_worker_rpc_signature_changed")
        @wraps(self.originals["prefill_lab_worker_begin"])
        def begin(worker, trial_id, mode):
            return self.begin(worker, trial_id, mode)

        @wraps(self.originals["prefill_lab_worker_finish"])
        def finish(worker):
            return self.finish(worker)

        self.callbacks = {
            "prefill_lab_worker_begin": base.wire_reply(begin),
            "prefill_lab_worker_finish": base.wire_reply(finish),
        }

    def install(self):
        try:
            for name, callback in self.callbacks.items():
                setattr(self.worker, name, callback)
            setattr(self.worker, MARKER, self)
        except Exception:
            for name, callback in self.callbacks.items():
                if vars(self.worker).get(name) is callback:
                    setattr(self.worker, name, self.originals[name])
            if vars(self.worker).get(MARKER) is self:
                delattr(self.worker, MARKER)
            raise

    def check_owner(self):
        if (vars(self.worker).get(MARKER) is not self
                or any(vars(self.worker).get(name) is not callback
                       for name, callback in self.callbacks.items())):
            raise ValueError("prepared_worker_registration_ownership_changed")

    def begin(self, worker, trial_id, mode):
        self.check_owner()
        if self.discard_required:
            raise ValueError("prepared_worker_setup_failed_discard_required")
        if (type(worker) is not self.worker or type(trial_id) is not str or not trial_id
                or type(mode) is not str or mode != "light"):
            raise ValueError("prepared_worker_explicit_light_trial_required")
        if self.lease is not None and not self.lease.removed:
            raise ValueError("prepared_worker_previous_lease_not_drained")
        runner = worker.model_runner
        execute = runner.execute_model
        if ("execute_model" in vars(runner)
                or getattr(execute, "__func__", None) is not type(runner).execute_model):
            raise ValueError("prepared_worker_pristine_runner_execute_required")
        _function(execute.__func__, NATIVE_PINS["prefill_lab.plugin"], EXECUTE_LINES["observed"])
        _function(inspect.unwrap(execute.__func__),
                  NATIVE_PINS[type(runner).__module__], EXECUTE_LINES["native"])
        reply = self.originals["prefill_lab_worker_begin"](worker, trial_id, mode)
        record = self.base._WORKER
        if (type(record) is not self.base.WorkerRecorder or record.trial_id != trial_id
                or record.mode != mode or record.active is not True or record.stages is not None
                or "resolve" in vars(record)):
            raise ValueError("prepared_worker_pristine_parent_recorder_required")
        try:
            self.lease = Lease(self.base, worker, record, execute, self.noop)
            self.lease.install()
        except BaseException:
            self.discard_required = True
            record.active = False
            raise
        return reply

    def finish(self, worker):
        self.check_owner()
        if self.lease is None or self.lease.removed or worker is not self.lease.worker:
            raise ValueError("prepared_worker_current_lease_required")
        if vars(self.lease.record).get("resolve") is not self.lease.resolve_callback:
            self.discard_required = True
            self.lease.native_errors.append({"type": "prepared_worker_resolve_ownership_changed"})
            raise ValueError("prepared_worker_resolve_ownership_changed")
        self.lease.barrier()
        return self.originals["prefill_lab_worker_finish"](worker)


class Lease:
    def __init__(self, base, worker, record, execute, noop):
        self.base, self.worker, self.record, self.original_execute = base, worker, record, execute
        self.runner = worker.model_runner
        self.thread = threading.get_ident()
        self.lock = threading.RLock()
        self.executing = 0
        self.native_errors = []
        self.removed = False
        self.region = ForwardRegionRecorder(record.torch, base._ACTIVE.get,
                                            capacity=record.records.capacity)
        self.hook = install(self.runner, self.region, noop_connector=noop, enabled=True)
        try:
            @wraps(execute)
            def execute_callback(*args, **kwargs):
                return self.execute(*args, **kwargs)

            self.execute_callback = execute_callback
            self.resolve_callback = self.resolve
            self.resolver = Resolver(record, self.region, self, native_errors=self.native_errors)
        except BaseException:
            self.hook.remove()
            raise

    def install(self):
        try:
            self.runner.execute_model = self.execute_callback
            self.record.resolve = self.resolve_callback
        except Exception:
            if vars(self.runner).get("execute_model") is self.execute_callback:
                del self.runner.execute_model
            if vars(self.record).get("resolve") is self.resolve_callback:
                del self.record.resolve
            self.hook.remove()
            raise

    def barrier(self):
        with self.lock:
            if (threading.get_ident() != self.thread or self.executing
                    or self.base._ACTIVE.get() is not None
                    or self.base._WORKER is not self.record
                    or self.worker.model_runner is not self.runner):
                self.native_errors.append({"type": "prepared_worker_same_actor_quiescence_required"})
                raise ValueError("prepared_worker_same_actor_quiescence_required")

    def execute(self, *args, **kwargs):
        entered = False
        try:
            with self.lock:
                self.barrier()
                if self.removed or self.record.active is not True:
                    raise ValueError("prepared_worker_active_lease_required")
                self.executing += 1
                entered = True
            return self.original_execute(*args, **kwargs)
        except BaseException as error:
            with self.lock:
                self.native_errors.append({"type": type(error).__name__})
            raise
        finally:
            if entered:
                with self.lock:
                    self.executing -= 1

    def resolve(self):
        self.barrier()
        capture = self.resolver.resolve()
        capture["native_exception_log"] = list(self.native_errors)
        capture["prepared_worker_barrier"] = {
            "same_actor_thread": True, "executing": self.executing,
            "active_context_empty": True, "parent_recorder_owned": True,
        }
        return capture

    def remove(self):
        with self.lock:
            if self.removed:
                return
            self.barrier()
            if (vars(self.runner).get("execute_model") is not self.execute_callback
                    or vars(self.record).get("resolve") is not self.resolve_callback):
                raise ValueError("prepared_worker_instance_ownership_changed")
            self.hook.remove()
            del self.runner.execute_model
            del self.record.resolve
            self.removed = True
