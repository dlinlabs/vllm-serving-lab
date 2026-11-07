"""Default-off instance observation of the pinned two-slot V2 sampling route."""

from collections import deque
from concurrent.futures import Future
from copy import deepcopy
from functools import wraps
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from copy_ready_intervals import CopyReadyIntervals, V2CopyEventBinding


MARKER = "_r24_prepared_copy_ready_hook"
SCHEMA = "r24-prepared-copy-ready-hook-v2"
METROLOGY_SCHEMA = "r24-native-getter-metrology-v1"
METROLOGY_MODES = ("off", "brackets", "query")
PINS = {
    "vllm.v1.engine.core": "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5",
    "vllm.v1.executor.uniproc_executor":
        "3cb039b80fe9b03dc1122f93c1f8a4098ef1355104b317ff6df25358f4349494",
    "vllm.v1.worker.gpu.model_runner":
        "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
    "vllm.v1.worker.gpu.async_utils":
        "77e17a4570ead2be30ae9b00888cf077a3b873018c12ca0549b853bfda02c1ba",
    "vllm.v1.outputs": "65ab7698bbdea1d53502ef8df3689bec860521ab6e2fab3c7bff63151be039b1",
    "prefill_lab.contracts": "4d5cda37f5d5f765f286476da675f4155fffcce5e8b22d35d8623f8d8429a23f",
    "prefill_lab.runtime": "c136380a889d83fba0c0a058657d4c238883c50461fd4934d4162933b00a426c",
    "prefill_lab.live_http": "cbbd1f338e0d078d5c5eacdfbcec725dcfe2ed9b46c1ee1ed68ad2f26de24e1f",
    "batch_capture_bridge": "5d76d994921bb16088cabf9b8f9ebe8843c17f7bacb4b1dc979bf255ad8ccef8",
    "process_clock_provenance": "685117d4d8603d6b2cbcf08b3b34d402a1ee8d06e5b8436b67f051f19131c051",
    "pr33_native_observer": "77633dc7652066e45b7680d297d3e36ddb97a3e12222de9f526a76b9a01d66b4",
}


def require(condition, reason):
    if not condition:
        raise ValueError("prepared_copy:" + reason)


def function(method):
    return getattr(method, "__func__", method)


def check_file(path, digest):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents))
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest, "source_changed")


def load_sources():
    modules = {name: importlib.import_module(name) for name in PINS}
    for name, module in modules.items():
        check_file(module.__file__, PINS[name])
    return modules


class CaptureContext:
    def __init__(self, core, bridge, epoch):
        modules = load_sources()
        require(type(bridge) is modules["batch_capture_bridge"].BatchCapture
                and bridge.core is core and core._pr33_batch_capture is bridge,
                "direct_batch_capture_required")
        capture = core._prefill_trial
        require(type(capture) is modules["prefill_lab.runtime"].ExternalCapture
                and bridge.active == capture.id and bridge.trials[capture.id] == "light",
                "authoritative_light_capture_required")
        base = modules["pr33_native_observer"].observe(core)
        clock = modules["process_clock_provenance"]
        identity = clock.self_process_metadata()
        clock.metadata_identity(identity)
        clock.domain_parts(base["clock_domain"])
        require(base["complete"] is True
                and base["observer_sha256"] == PINS["pr33_native_observer"]
                and base["clock_domain"] == bridge.domain
                == modules["prefill_lab.live_http"].clock_domain()
                and base["run_id"] == core._pr33_observer_run, "native_clock_context_required")
        self.core, self.bridge, self.capture = core, bridge, capture
        self.context = (base["clock_domain"], base["run_id"], identity)
        self.bridge_context = (bridge.run_id, bridge.plan_hash, bridge.domain)
        self.controller = SimpleNamespace(clock=capture.clock,
            step_key=modules["prefill_lab.contracts"].step_key,
            current=SimpleNamespace(capture=capture, epoch=epoch))

    def guard(self):
        require(self.core._pr33_batch_capture is self.bridge and self.bridge.core is self.core
                and (self.bridge.run_id, self.bridge.plan_hash, self.bridge.domain)
                    == self.bridge_context, "batch_context_changed")
        require(self.bridge.active == (self.capture.id if self.core._prefill_trial is not None else None),
                "batch_active_capture_changed")
        require(self.core._pr33_observer_run == self.context[1]
                and self.bridge.domain == self.context[0], "native_run_clock_changed")


class NativeBinding:
    def __init__(self, core, host):
        self.modules = load_sources()
        self.sources = {module.__file__: PINS[name] for name, module in self.modules.items()}
        engine = self.modules["vllm.v1.engine.core"]
        executor = self.modules["vllm.v1.executor.uniproc_executor"]
        runner = self.modules["vllm.v1.worker.gpu.model_runner"]
        output = self.modules["vllm.v1.worker.gpu.async_utils"]
        self.runner = core.model_executor.driver_worker.worker.model_runner
        require(type(core) in (engine.EngineCore, engine.EngineCoreProc)
                and type(core.model_executor) is executor.UniProcExecutor
                and type(self.runner) is runner.GPUModelRunner
                and type(host) is CaptureContext,
                "registry_objects_required")
        self.host = host
        self.event = V2CopyEventBinding(executor.AsyncOutputFuture, output.AsyncOutput)
        require(executor.Future is Future, "native_plain_future_registry_required")
        self.plain_future_type = Future
        self.model_output_type = self.modules["vllm.v1.outputs"].ModelRunnerOutput
        self.future_result_code = executor.AsyncOutputFuture.result.__code__
        self.state_type = runner.ExecuteModelState
        self.core_method = engine.EngineCore.step_with_batch_queue
        require(function(core.step_with_batch_queue) is self.core_method
                and function(core.step_fn) is self.core_method, "native_core_step_required")
        self.core_code = self.core_method.__code__
        self.methods = [
            (executor.UniProcExecutor, "execute_model", executor.UniProcExecutor.execute_model),
            (executor.UniProcExecutor, "sample_tokens", executor.UniProcExecutor.sample_tokens),
            (executor.UniProcExecutor, "collective_rpc", executor.UniProcExecutor.collective_rpc),
            (runner.GPUModelRunner, "sample_tokens", runner.GPUModelRunner.sample_tokens),
            (output.AsyncOutput, "__init__", output.AsyncOutput.__init__),
            (output.AsyncOutput, "get_output", output.AsyncOutput.get_output),
            (executor.AsyncOutputFuture, "__init__", executor.AsyncOutputFuture.__init__),
            (executor.AsyncOutputFuture, "result", executor.AsyncOutputFuture.result),
        ]
        require(function(core.model_executor.execute_model) is executor.UniProcExecutor.execute_model
                and function(core.model_executor.sample_tokens) is executor.UniProcExecutor.sample_tokens
                and function(self.runner.sample_tokens) is runner.GPUModelRunner.sample_tokens
                and function(core.model_executor.collective_rpc)
                    is executor.UniProcExecutor.collective_rpc,
                "pristine_sampling_methods_required")
        self.executor = core.model_executor
        self.rpc = executor.UniProcExecutor.collective_rpc
        self.codes = {}
        for _, _, method in self.methods:
            original = inspect.unwrap(method)
            require(inspect.isfunction(original), "native_python_method_required")
            path = original.__code__.co_filename
            require(path in self.sources, "native_method_file_required")
            self.codes[method] = (method.__code__, original, original.__code__)
        self.sample_body = inspect.unwrap(runner.GPUModelRunner.sample_tokens)
        require(self.sample_body.__code__.co_firstlineno == 1995
                and self.sample_body.__globals__["AsyncOutput"] is self.event.output_type,
                "v2_sample_constructor_required")
        require(function(host.controller.step_key).__code__.co_filename in self.sources,
                "host_step_key_source_required")

    def guard(self):
        self.host.guard()
        require(self.modules["vllm.v1.executor.uniproc_executor"].Future is self.plain_future_type
                and self.modules["vllm.v1.outputs"].ModelRunnerOutput is self.model_output_type,
                "native_no_copy_registry_changed")
        require(function(self.executor.collective_rpc) is self.rpc
                and self.core_method.__code__ is self.core_code,
                "native_rpc_or_core_code_changed")
        require(all(getattr(cls, name) is method for cls, name, method in self.methods),
                "native_method_identity_changed")
        require(all(method.__code__ is code and inspect.unwrap(method) is body
                    and body.__code__ is body_code
                    for method, (code, body, body_code) in self.codes.items()),
                "native_method_code_changed")
        require(self.sample_body.__globals__["AsyncOutput"] is self.event.output_type,
                "native_output_constructor_changed")

    def check_sources(self):
        for path, digest in self.sources.items():
            check_file(path, digest)


def install(core, host, *, enabled=False, metrology="off"):
    if type(enabled) is not bool:
        raise ValueError("prepared_copy_boolean_enable_required")
    require(type(metrology) is str and metrology in METROLOGY_MODES,
            "getter_metrology_mode_required")
    if not enabled:
        require(metrology == "off", "getter_metrology_requires_parent")
        return None
    require(getattr(core, MARKER, None) is None, "duplicate_or_foreign_hook")
    binding = NativeBinding(core, host)
    hook = Hook(core, host, binding, metrology=metrology)
    hook.install()
    return hook


class Hook:
    def __init__(self, core, host, binding, *, metrology="off"):
        self.core, self.host, self.binding = core, host, binding
        self.executor, self.scheduler, self.runner = (
            core.model_executor, core.scheduler, binding.runner)
        self.capture, self.clock = host.capture, host.capture.clock
        self.context = deepcopy(host.context)
        self.started, self.epoch = self.capture.started, host.controller.current.epoch
        self.thread = threading.get_ident()
        self.pending, self.entered, self.consumed = deque(), [], []
        self.routes, self.completed_routes = [], []
        self.entered_route_count = self.completed_route_count = 0
        self.active = self.candidate = self.sample = None
        self.errors, self.targets = [], []
        self.getters = {}
        self.metrology = self.bound_metrology = metrology
        self.metrology_copy, self.metrology_empty, self.empty_getters = {}, {}, {}
        self.closed = self.removed = self.discard_required = False
        self.rollback_failed = False
        self.snapshot = None
        require(not core.batch_queue and not self.capture.steps.rows,
                "empty_capture_boundary_required")
        self.trace = CopyReadyIntervals(binding.event, capture_id=self.capture.id,
            clock_domain=self.context[0], clock=self.clock, limit=8192)
        self.guard(owner=False)

    def fail(self, operation, error):
        name = type.__dict__["__name__"].__get__(type(error))
        reason = operation + ":" + str.__getitem__(name, slice(None, 64))
        if reason not in self.errors and len(self.errors) < 16:
            self.errors.append(reason)

    def safe(self, operation, callback):
        try:
            return callback()
        except BaseException as error:
            self.fail(operation, error)
            return None

    def guard(self, *, owner=True):
        require(not self.closed and not self.errors and not self.trace.errors,
                "closed_or_refused_capture")
        require(self.metrology == self.bound_metrology, "getter_metrology_mode_changed")
        require(threading.get_ident() == self.thread
                and self.host.core is self.core and self.host.capture is self.capture
                and self.host.context == self.context
                and self.capture.clock is self.clock and self.host.controller.clock is self.clock
                and self.capture.id == self.trace.capture_id and self.capture.started == self.started
                and self.host.controller.current.capture is self.capture
                and self.host.controller.current.epoch == self.epoch
                and self.capture.status == "running" and self.capture.mode == "light"
                and self.core._prefill_trial is self.capture, "capture_actor_changed")
        require(self.core.model_executor is self.executor and self.core.scheduler is self.scheduler
                and self.executor.driver_worker.worker.model_runner is self.runner,
                "execution_objects_changed")
        require(self.core.async_scheduling is True and self.core.is_pooling_model is False
                and self.core.is_ec_consumer is True and self.core.check_for_draft_tokens is False
                and self.core.batch_queue_size == 2 and type(self.core.batch_queue) is deque
                and self.core.batch_queue.maxlen == 2
                and self.core.vllm_config.parallel_config.world_size == 1
                and self.runner.pcp_manager is None and self.runner.speculator is None,
                "supported_two_slot_route_required")
        self.binding.guard()
        if owner:
            self.check_owner()

    def check_owner(self):
        require(vars(self.core).get(MARKER) is self
                and all(vars(obj).get(name) is wrapper
                        for obj, name, _, _, wrapper in self.targets),
                "hook_ownership_changed")

    def install(self):
        for obj, name, factory in (
            (self.core, "step_with_batch_queue", self.step),
            (self.executor, "execute_model", self.execute),
            (self.executor, "sample_tokens", self.sample_tokens),
            (self.runner, "sample_tokens", self.runner_sample),
            (self.scheduler, "update_from_output", self.update),
        ):
            original = getattr(obj, name)
            present, value = name in vars(obj), vars(obj).get(name)
            self.targets.append((obj, name, present, value, factory(original)))
        self.targets.append((self.core, "step_fn", "step_fn" in vars(self.core),
                             vars(self.core).get("step_fn"), self.targets[0][4]))
        try:
            for obj, name, _, _, wrapper in self.targets:
                setattr(obj, name, wrapper)
            setattr(self.core, MARKER, self)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        for obj, name, present, value, wrapper in reversed(self.targets):
            if vars(obj).get(name) is wrapper:
                if present:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)
        if vars(self.core).get(MARKER) is self:
            delattr(self.core, MARKER)

    def caller(self, frame):
        self.guard()
        require(self.active is not None and frame.f_code is self.binding.core_code
                and frame.f_locals.get("self") is self.core, "native_core_caller_required")

    def step(self, original):
        @wraps(original)
        def call(*args, **kwargs):
            if self.core._prefill_trial is None:
                return original(*args, **kwargs)
            def enter():
                self.guard()
                require(self.active is None and self.candidate is None, "reentrant_core_step")
                queued = list(reversed(self.core.batch_queue))
                require(len(queued) == len(self.pending)
                        and all(q[0] is p["future"] and q[1] is p["scheduler"]
                                for q, p in zip(queued, self.pending)), "fifo_inventory_changed")
                self.active = object()
            self.safe("step_enter", enter)
            try:
                return original(*args, **kwargs)
            except BaseException as error:
                self.fail("native_step", error)
                raise
            finally:
                if self.candidate is not None:
                    self.fail("unsampled_execute", ValueError())
                self.active = self.candidate = None
        return call

    def execute(self, original):
        @wraps(original)
        def call(*args, **kwargs):
            if self.core._prefill_trial is None:
                return original(*args, **kwargs)
            frame = sys._getframe(1)
            def before():
                self.caller(frame)
                require(len(args) == 1 and kwargs == {"non_block": True}
                        and self.candidate is None and self.runner.execute_model_state is None,
                        "fresh_execute_required")
                scheduler = args[0]
                tokens = scheduler.total_num_scheduled_tokens
                require(type(tokens) is int and tokens >= 0
                        and not scheduler.pending_structured_output_tokens
                        and not scheduler.scheduled_spec_decode_tokens, "unsupported_work")
                require(type(scheduler.num_scheduled_tokens) is dict
                        and (tokens > 0 or not scheduler.num_scheduled_tokens),
                        "empty_no_copy_schedule_required")
                key = self.host.controller.step_key(scheduler)
                require(type(key) is str and (tokens == 0 or key not in self.entered)
                        and self.entered_route_count < self.trace.limit, "execute_inventory")
                route = {
                    "sequence": self.entered_route_count, "kind": "copy" if tokens else "no_copy",
                    "step_id": key, "scheduled_tokens": tokens,
                    "scheduled_requests": len(scheduler.num_scheduled_tokens),
                    "execute_started_at": self.trace._now(), "execute_finished_at": None,
                    "update_started_at": None, "update_finished_at": None, "status": "entered",
                }
                self.entered_route_count += 1
                self.routes.append(route)
                if tokens:
                    self.entered.append(key)
                self.candidate = {"scheduler": scheduler, "key": key, "route": route}
            self.safe("execute_enter", before)
            result = original(*args, **kwargs)
            def after():
                self.guard()
                require(self.candidate is not None, "execute_candidate_required")
                route = self.candidate["route"]
                route["execute_finished_at"] = self.trace._now()
                if route["kind"] == "no_copy":
                    require(self.runner.execute_model_state is None
                            and type(result) is self.binding.plain_future_type
                            and result._state == "FINISHED" and result._exception is None
                            and type(result._result) is self.binding.model_output_type
                            and result._result.req_ids == []
                            and len(self.pending) < 2, "native_no_copy_future_required")
                    pending = {**self.candidate, "future": result}
                    if self.metrology != "off":
                        self.observe_empty_result(pending)
                    self.pending.append(pending)
                    self.candidate = None
                    return
                require(type(self.runner.execute_model_state) is self.binding.state_type,
                        "v2_execute_state_required")
                state = self.runner.execute_model_state
                self.candidate.update(state=state, req_ids=tuple(state.input_batch.req_ids))
            self.safe("execute_exit", after)
            return result
        return call

    def sample_tokens(self, original):
        @wraps(original)
        def call(*args, **kwargs):
            if self.core._prefill_trial is None:
                return original(*args, **kwargs)
            frame = sys._getframe(1)
            def before():
                self.caller(frame)
                require(args == (None,) and kwargs == {"non_block": True}
                        and self.candidate is not None and self.sample is None,
                        "immediate_sample_required")
                self.sample = {**self.candidate, "calls": 0, "output": None}
            self.safe("sample_enter", before)
            try:
                future = original(*args, **kwargs)
                def after():
                    self.guard()
                    sample = self.sample
                    require(sample is not None and sample["calls"] == 1
                            and type(future) is self.binding.event.future_type
                            and future.async_output is sample["output"]
                            and future._state == "PENDING"
                            and tuple(sample["output"].model_runner_output.req_ids)
                                == sample["req_ids"]
                            and len(self.pending) < 2, "sample_future_binding")
                    self.observe_getter(sample, future)
                    self.trace.register(sample["key"], future,
                                        sample_started_at=sample["started"])
                    require(not self.trace.errors, "interval_registration_refused")
                    sample["future"] = future
                    self.pending.append(sample)
                self.safe("sample_exit", after)
                return future
            finally:
                self.sample = self.candidate = None
        return call

    def observe_getter(self, sample, future):
        output = sample["output"]
        require("get_output" not in vars(output)
                and function(output.get_output) is self.binding.event.output_type.get_output,
                "pristine_output_getter_required")
        original = output.get_output
        sample.update(getter_calls=0, getter_returned=False,
                      event=self.binding.event.event(future))
        if self.metrology != "off":
            sample["metrology"] = {}
            self.metrology_copy[sample["key"]] = sample["metrology"]
            self.observe_synchronize(sample, function(original).__code__)

        @wraps(original)
        def get_output(*args, **kwargs):
            frame = sys._getframe(1)
            def before():
                self.guard()
                require(not args and not kwargs and sample["getter_calls"] == 0
                        and frame.f_code is self.binding.future_result_code
                        and frame.f_locals.get("self") is future
                        and frame.f_back is not None
                        and frame.f_back.f_code is self.binding.core_code
                        and frame.f_back.f_locals.get("self") is self.core
                        and output.copy_event is sample["event"], "normal_native_getter_caller")
                sample["getter_calls"] += 1
                if self.metrology == "query":
                    observation = sample["metrology"]
                    observation["query_started_at"] = self.trace._now()
                    ready = sample["event"].query()
                    observation["query_finished_at"] = self.trace._now()
                    require(type(ready) is bool, "getter_boolean_query_required")
                    observation["query_ready"] = ready
                sample["getter_begin_at"] = self.trace._now()
            self.safe("get_output_enter", before)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                self.fail("native_get_output", error)
                raise
            def after():
                if self.metrology != "off":
                    sample["metrology"]["get_return_raw_at"] = self.trace._now()
                self.guard()
                require(sample["getter_calls"] == 1
                        and result is output.model_runner_output
                        and output.copy_event is sample["event"], "native_getter_result_changed")
                sample["getter_return_at"] = self.trace._now()
                sample["getter_returned"] = True
            self.safe("get_output_exit", after)
            return result

        sample["getter_wrapper"] = get_output
        try:
            output.get_output = get_output
        except BaseException:
            try:
                self.rollback_attribute(output, "get_output", get_output)
            finally:
                if self.bound_metrology != "off":
                    self.rollback_attribute(sample["event"], "synchronize", sample["sync_wrapper"])
            raise
        self.getters[sample["key"]] = sample

    def rollback_attribute(self, obj, name, wrapper):
        if vars(obj).get(name) is wrapper:
            try:
                delattr(obj, name)
            except BaseException:
                self.rollback_failed = True
                raise

    def restore_getter(self, sample):
        output = sample["output"]
        require(vars(output).get("get_output") is sample["getter_wrapper"],
                "output_getter_ownership_changed")
        if self.bound_metrology != "off":
            require(vars(sample["event"]).get("synchronize") is sample["sync_wrapper"],
                    "event_synchronize_ownership_changed")
            del sample["event"].synchronize
        del output.get_output
        del self.getters[sample["key"]]

    def observe_synchronize(self, sample, native_getter_code):
        event = sample["event"]
        require("synchronize" not in vars(event), "pristine_event_synchronize_required")
        original, descriptor = event.synchronize, type(event).synchronize
        observation = sample["metrology"]
        observation["sync_calls"] = 0

        @wraps(original)
        def synchronize(*args, **kwargs):
            frame = sys._getframe(1)
            def before():
                observation["sync_observer_enter_at"] = self.trace._now()
                self.guard()
                require(not args and not kwargs and observation["sync_calls"] == 0
                        and sample["getter_calls"] == 1
                        and frame.f_code is native_getter_code
                        and frame.f_locals.get("self") is sample["output"]
                        and frame.f_back is not None
                        and frame.f_back.f_code is sample["getter_wrapper"].__code__
                        and sample["output"].copy_event is event
                        and vars(event).get("synchronize") is synchronize
                        and type(event).synchronize is descriptor,
                        "natural_native_synchronize_required")
                observation["sync_calls"] += 1
                observation["sync_begin_at"] = self.trace._now()
            self.safe("synchronize_enter", before)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                self.fail("native_synchronize", error)
                raise
            def after():
                observation["sync_return_at"] = self.trace._now()
                self.guard()
                require(observation["sync_calls"] == 1
                        and vars(event).get("synchronize") is synchronize
                        and sample["output"].copy_event is event
                        and type(event).synchronize is descriptor,
                        "natural_synchronize_completion_required")
                observation["sync_observer_return_at"] = self.trace._now()
            self.safe("synchronize_exit", after)
            return result

        sample["sync_wrapper"] = synchronize
        try:
            event.synchronize = synchronize
        except BaseException:
            self.rollback_attribute(event, "synchronize", synchronize)
            raise

    def observe_empty_result(self, pending):
        future, sequence = pending["future"], pending["route"]["sequence"]
        require("result" not in vars(future)
                and function(future.result) is self.binding.plain_future_type.result,
                "pristine_empty_result_required")
        original = future.result
        observation = {"result_calls": 0}
        self.metrology_empty[str(sequence)] = observation

        @wraps(original)
        def result(*args, **kwargs):
            frame = sys._getframe(1)
            def before():
                observation["result_observer_enter_at"] = self.trace._now()
                self.caller(frame)
                require(not args and not kwargs and self.pending
                        and self.pending[0] is pending and observation["result_calls"] == 0
                        and vars(future).get("result") is result,
                        "natural_empty_result_required")
                observation["result_calls"] += 1
                observation["result_begin_at"] = self.trace._now()
            self.safe("empty_result_enter", before)
            try:
                value = original(*args, **kwargs)
            except BaseException as error:
                self.fail("native_empty_result", error)
                raise
            def after():
                observation["result_return_at"] = self.trace._now()
                self.guard()
                require(observation["result_calls"] == 1 and value is future._result
                        and vars(future).get("result") is result,
                        "natural_empty_result_completion_required")
                observation["result_observer_return_at"] = self.trace._now()
            self.safe("empty_result_exit", after)
            return value

        pending["result_wrapper"] = result
        try:
            future.result = result
        except BaseException:
            self.rollback_attribute(future, "result", result)
            raise
        self.empty_getters[sequence] = pending

    def restore_empty_result(self, pending):
        future = pending["future"]
        require(vars(future).get("result") is pending["result_wrapper"],
                "empty_result_ownership_changed")
        del future.result
        del self.empty_getters[pending["route"]["sequence"]]

    def runner_sample(self, original):
        @wraps(original)
        def call(*args, **kwargs):
            if self.core._prefill_trial is None:
                return original(*args, **kwargs)
            def before():
                self.guard()
                require(self.sample is not None and self.sample["calls"] == 0
                        and ((args == (None,) and not kwargs)
                             or (not args and kwargs == {"grammar_output": None}))
                        and self.runner.execute_model_state is self.sample["state"],
                        "same_execute_state_required")
                self.sample["calls"] += 1
                self.sample["started"] = self.trace._now()
            self.safe("runner_sample_enter", before)
            result = original(*args, **kwargs)
            def after():
                self.guard()
                require(self.sample is not None and self.runner.execute_model_state is None
                        and type(result) is self.binding.event.output_type
                        and tuple(result.model_runner_output.req_ids) == self.sample["req_ids"],
                        "fresh_async_output_required")
                self.sample["output"] = result
                del self.sample["state"]
            self.safe("runner_sample_exit", after)
            return result
        return call

    def update(self, original):
        @wraps(original)
        def call(*args, **kwargs):
            if self.core._prefill_trial is None:
                return original(*args, **kwargs)
            frame = sys._getframe(1)
            pending = None
            def before():
                nonlocal pending
                self.caller(frame)
                require(len(args) == 2 and not kwargs and self.pending, "native_update_required")
                pending = self.pending[0]
                route = pending["route"]
                require(route["sequence"] == self.completed_route_count
                        and args[0] is pending["scheduler"]
                        and args[0].total_num_scheduled_tokens == route["scheduled_tokens"]
                        and len(args[0].num_scheduled_tokens) == route["scheduled_requests"],
                        "all_route_fifo_binding")
                route["update_started_at"] = self.trace._now()
                if route["kind"] == "no_copy":
                    future = pending["future"]
                    require(route["scheduled_tokens"] == route["scheduled_requests"] == 0
                            and self.runner.execute_model_state is None
                            and type(future) is self.binding.plain_future_type
                            and future._state == "FINISHED" and future._exception is None
                            and args[1] is future._result
                            and type(args[1]) is self.binding.model_output_type
                            and args[1].req_ids == [], "no_copy_consumption_binding")
                    if self.metrology != "off":
                        require(self.metrology_empty[str(route["sequence"])]["result_calls"] == 1,
                                "empty_result_observation_required")
                        self.restore_empty_result(pending)
                    return
                require(args[0] is pending["scheduler"]
                        and args[1] is pending["output"].model_runner_output
                        and tuple(args[1].req_ids) == pending["req_ids"]
                        and pending["getter_calls"] == 1 and pending["getter_returned"],
                        "fifo_consumption_binding")
                self.trace.consumed(pending["key"])
                require(not self.trace.errors, "native_consumption_refused")
                row = self.trace.records[pending["key"]]
                row["native_get_output_begin_at"] = pending["getter_begin_at"]
                row["native_get_output_return_at"] = pending["getter_return_at"]
                row["ready_upper_at"] = min(row["ready_upper_at"], pending["getter_return_at"])
                require(row["ready_lower_at"] < row["ready_upper_at"]
                        or row["ready_lower_at"] == row["ready_upper_at"]
                        and row["ready_lower_inclusive"] is True,
                        "nonempty_native_ready_bracket_required")
                self.restore_getter(pending)
                self.consumed.append(pending["key"])
            self.safe("update_enter", before)
            result = original(*args, **kwargs)
            def after():
                self.guard()
                require(pending is not None and self.pending and self.pending[0] is pending
                        and pending["route"]["sequence"] == self.completed_route_count,
                        "all_route_update_completion_required")
                pending["route"].update(update_finished_at=self.trace._now(), status="consumed")
                self.completed_route_count += 1
                self.completed_routes.append(pending["route"]["sequence"])
                self.pending.popleft()
            self.safe("update_exit", after)
            return result
        return call

    def close(self):
        if self.snapshot is not None:
            return deepcopy(self.snapshot)
        try:
            require(self.metrology == self.bound_metrology, "getter_metrology_mode_changed")
            require(threading.get_ident() == self.thread and self.active is None
                    and self.candidate is None and self.sample is None and not self.pending
                    and not self.core.batch_queue and self.core._prefill_trial is None
                    and self.capture.status == "complete", "drained_post_end_barrier_required")
            require(self.host.core is self.core and self.host.capture is self.capture
                    and self.host.context == self.context
                    and self.capture.clock is self.clock and self.host.controller.clock is self.clock
                    and self.capture.id == self.trace.capture_id
                    and self.capture.started == self.started
                    and self.host.controller.current.capture is self.capture
                    and self.host.controller.current.epoch == self.epoch,
                    "post_end_capture_identity_required")
            self.check_owner()
            self.binding.guard()
            self.binding.check_sources()
            expected = [row["step_id"] for row in self.capture.steps.rows]
            require(expected == self.entered == self.consumed
                    and len(self.routes) == self.entered_route_count == self.completed_route_count
                    and self.completed_routes == list(range(len(self.routes)))
                    and all(route["status"] == "consumed" for route in self.routes)
                    and not self.capture.errors and not self.capture.steps.overflow,
                    "whole_native_inventory_required")
            if self.bound_metrology != "off":
                validate_metrology(self.metrology_snapshot(), self.routes, self.trace.records,
                                   mode=self.metrology)
        except BaseException as error:
            self.fail("close", error)
        self.closed = True
        intervals = self.trace.finish(self.entered)
        try:
            require(self.active is None, "cleanup_midflight")
            self.check_owner()
            for sample in list(self.getters.values()):
                self.restore_getter(sample)
            for pending in list(self.empty_getters.values()):
                self.restore_empty_result(pending)
            self.restore()
            require(not self.rollback_failed, "partial_install_rollback_failed")
            self.removed = True
        except BaseException as error:
            self.fail("cleanup", error)
        self.discard_required = bool(self.errors or not intervals["observations_complete"]
                                     or not self.removed)
        self.snapshot = {
            "schema": SCHEMA, "intervals": intervals,
            "entered_step_ids": list(self.entered), "consumed_step_ids": list(self.consumed),
            "routes": deepcopy(self.routes), "completed_route_sequences": list(self.completed_routes),
            "entered_route_count": self.entered_route_count,
            "completed_route_count": self.completed_route_count,
            "whole_capture_complete": not self.discard_required,
            "errors": list(self.errors), "hook_removed": self.removed,
            "discard_required": self.discard_required,
            "scope": "Pinned parent sampling/FIFO identity and existing copy-event intervals; "
                     "native registry execution, collector integration and calibration unqualified.",
            "native_mpc_permitted": False, "deployable": False,
        }
        if self.metrology != "off":
            self.snapshot["getter_metrology"] = self.metrology_snapshot()
        return deepcopy(self.snapshot)

    def metrology_snapshot(self):
        return {
            "schema": METROLOGY_SCHEMA, "mode": self.bound_metrology,
            "copy": deepcopy(self.metrology_copy), "no_copy": deepcopy(self.metrology_empty),
            "scope": "Instance-only brackets around existing natural synchronize/result calls. "
                     "No added synchronization, getter consumption, event creation or stream wait. "
                     "Query mode adds exactly one nonblocking query before native getter entry. "
                     "Raw return and observer-return clocks distinguish measured native wall "
                     "brackets from observer CPU overhead; neither is exclusive CPU/GPU service. "
                     "False getter query is not proof of waiting at later sync entry.",
            "native_mpc_permitted": False, "deployable": False,
        }


def validate_metrology(value, routes, records, *, mode):
    require(type(value) is dict and value.get("schema") == METROLOGY_SCHEMA
            and value.get("mode") == mode and mode in ("brackets", "query")
            and value.get("native_mpc_permitted") is False
            and value.get("deployable") is False, "getter_metrology_identity_required")
    copies, empty = value.get("copy"), value.get("no_copy")
    require(type(copies) is dict and type(empty) is dict
            and set(copies) == set(records)
            and set(empty) == {str(route["sequence"]) for route in routes
                               if route["kind"] == "no_copy"},
            "getter_metrology_complete_inventory_required")
    sync_times = ("sync_observer_enter_at", "sync_begin_at", "sync_return_at",
                  "sync_observer_return_at", "get_return_raw_at")
    result_times = ("result_observer_enter_at", "result_begin_at", "result_return_at",
                    "result_observer_return_at")
    def ordered(values):
        return (all(type(v) in (int, float) and 0 <= v <= sys.float_info.max for v in values)
                and values == sorted(values))
    for key, row in copies.items():
        extra = {"query_started_at", "query_finished_at", "query_ready"} if mode == "query" else set()
        require(type(row) is dict and set(row) == {*sync_times, "sync_calls"} | extra
                and type(row["sync_calls"]) is int and row["sync_calls"] == 1,
                "exact_natural_synchronize_observation_required")
        native = records[key]
        require(ordered([native["native_get_output_begin_at"],
                         *[row[name] for name in sync_times],
                         native["native_get_output_return_at"],
                         native["native_update_begin_at"]]),
                "natural_synchronize_clock_chain_required")
        lower = native["ready_lower_at"]
        upper = min(native["ready_upper_at"], row["sync_return_at"])
        if mode == "query":
            require(type(row["query_ready"]) is bool
                    and ordered([native["query_finished_at"], row["query_started_at"],
                                 row["query_finished_at"], native["native_get_output_begin_at"]])
                    and (not native["query_ready"] or row["query_ready"]),
                    "natural_getter_query_chain_required")
            if row["query_ready"]:
                upper = min(upper, row["query_finished_at"])
            else:
                lower = row["query_started_at"]
        require(lower < upper or lower == upper and native["ready_lower_inclusive"] is True
                and (mode != "query" or row["query_ready"]),
                "nonempty_getter_ready_interval_required")
    for route in routes:
        if route["kind"] != "no_copy":
            continue
        row = empty[str(route["sequence"])]
        require(type(row) is dict and set(row) == {*result_times, "result_calls"}
                and type(row["result_calls"]) is int and row["result_calls"] == 1
                and ordered([route["execute_finished_at"], *[row[name] for name in result_times],
                             route["update_started_at"]]),
                "exact_natural_empty_result_observation_required")
    return {"copy_count": len(copies), "no_copy_count": len(empty),
            "mode": mode, "native_mpc_permitted": False, "deployable": False}
