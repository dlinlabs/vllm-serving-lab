"""Source-pinned, default-off hooks; install/remove at a quiescent worker barrier."""

import hashlib
import inspect
from pathlib import Path
import sys


MARKER = "_r24_prepared_forward_hook"
TYPE_PINS = {
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner":
        "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
    "vllm.v1.worker.gpu.async_utils.StepTimingCollector":
        "77e17a4570ead2be30ae9b00888cf077a3b873018c12ca0549b853bfda02c1ba",
    "vllm.v1.worker.gpu.kv_connector.KVConnector":
        "b89a4333f1e552d86e1815674b59e9f1335f0f49c0a7b46082439076f62c6a81",
}
METHOD_LINES = {"forward_start": 94, "finish_forward": 35}


def _source(instance):
    kind = type(instance)
    name = f"{kind.__module__}.{kind.__qualname__}"
    module = sys.modules.get(kind.__module__)
    if name not in TYPE_PINS or getattr(module, kind.__qualname__, None) is not kind:
        raise ValueError("prepared_hook_unreviewed_type")
    path = Path(inspect.getsourcefile(kind))
    if (any(part.is_symlink() for part in (path, *path.parents))
            or hashlib.sha256(path.read_bytes()).hexdigest() != TYPE_PINS[name]):
        raise ValueError("prepared_hook_native_source_changed")


def _method(instance, name):
    method = getattr(instance, name)
    function = getattr(method, "__func__", None)
    if (name in vars(instance) or function is not getattr(type(instance), name)
            or function.__code__.co_firstlineno != METHOD_LINES[name]
            or function.__module__ != type(instance).__module__
            or Path(function.__code__.co_filename).resolve()
            != Path(inspect.getsourcefile(type(instance))).resolve()
            or tuple(inspect.signature(method).parameters)):
        raise ValueError("prepared_hook_native_method_changed")
    return method


def install(runner, recorder, *, noop_connector, enabled=False):
    if type(enabled) is not bool:
        raise ValueError("prepared_hook_boolean_enable_required")
    if not enabled:
        return None
    with recorder._lock:
        timing, connector = runner.step_timing, runner.kv_connector
        for instance in (runner, timing, connector):
            _source(instance)
        if (connector is not noop_connector or vars(connector)
                or type(connector).__init__ is not object.__init__
                or type(connector).__new__ is not object.__new__):
            raise ValueError("prepared_hook_bare_shared_noop_required")
        if (MARKER in vars(runner) or MARKER in vars(timing)
                or MARKER in vars(recorder)
                or recorder.pending or recorder.errors or recorder._closed):
            raise ValueError("prepared_hook_owner_or_recorder_not_pristine")
        start = _method(timing, "forward_start")
        finish = _method(connector, "finish_forward")
        hook = Hook(runner, recorder, timing, connector, start, finish)
        try:
            timing.forward_start = hook.start
            runner.kv_connector = hook.connector
            setattr(timing, MARKER, hook)
            setattr(runner, MARKER, hook)
            setattr(recorder, MARKER, hook)
        except Exception:
            hook._restore_owned()
            raise
        return hook


class Hook:
    def __init__(self, runner, recorder, timing, original_connector, start, finish):
        self.runner, self.recorder, self.timing = runner, recorder, timing
        self.original_connector = original_connector
        self.original_start, self.original_finish = start.__func__, finish.__func__
        self.start, self.finish = recorder.wrap(start, finish)
        self.connector = type(original_connector)()
        if self.connector is original_connector:
            raise ValueError("prepared_hook_shared_clone_refused")
        self.connector.finish_forward = self.finish
        self.removed = False

    def _restore_owned(self):
        if self.runner.kv_connector is self.connector:
            self.runner.kv_connector = self.original_connector
        if vars(self.timing).get("forward_start") is self.start:
            del self.timing.forward_start
        for instance in (self.runner, self.timing, self.recorder):
            if vars(instance).get(MARKER) is self:
                delattr(instance, MARKER)

    def remove(self):
        with self.recorder._lock:
            if self.removed:
                return
            if (self.recorder.pending or self.runner.step_timing is not self.timing
                    or self.runner.kv_connector is not self.connector
                    or vars(self.timing).get("forward_start") is not self.start
                    or vars(self.connector).get("finish_forward") is not self.finish
                    or any(vars(instance).get(MARKER) is not self
                           for instance in (self.runner, self.timing, self.recorder))
                    or type(self.timing).forward_start is not self.original_start
                    or type(self.original_connector).finish_forward is not self.original_finish):
                raise ValueError("prepared_hook_live_or_ownership_changed")
            self._restore_owned()
            self.removed = True
