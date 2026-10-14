"""Optional external observations, collected on a worker thread or a spawned process."""
import asyncio
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

import httpx

from telemetry import DIAGNOSTICS_SCHEMA

MAX_METRICS_BYTES = 4 * 1024 * 1024
METRICS_BODY_TIMEOUT_SECONDS = 5.0

REQUIRED_METRICS = {
    'waiting': ('gauge', ('vllm:num_requests_waiting',)),
    'running': ('gauge', ('vllm:num_requests_running',)),
    'queue_time': ('histogram', ('vllm:request_queue_time_seconds',)),
    'prefill_time': ('histogram', ('vllm:request_prefill_time_seconds',)),
    'decode_time': ('histogram', ('vllm:request_decode_time_seconds',)),
    'kv_cache': ('gauge', ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')),
    'preemptions': ('counter', ('vllm:num_preemptions',)),
    'prompt_tokens': ('counter', ('vllm:prompt_tokens',)),
    'generation_tokens': ('counter', ('vllm:generation_tokens',)),
}


def metric_inventory(text):
    from prometheus_client.parser import text_string_to_metric_families
    families = list(text_string_to_metric_families(text))
    inventory, counters = {}, {}
    for key, (expected_type, names) in REQUIRED_METRICS.items():
        family = next((f for f in families if f.name in names), None)
        samples = [] if family is None else [
            dict(name=s.name, labels=dict(s.labels), value=s.value) for s in family.samples]
        sample_names = {s['name'] for s in samples}
        reason = ('not_exported' if family is None else 'unexpected_type' if family.type != expected_type
                  else 'no_samples' if not samples else 'nonfinite_samples' if
                  any(not math.isfinite(s['value']) for s in samples) else 'negative_samples' if
                  any(s['value'] < 0 for s in samples) else 'incomplete_histogram' if
                  expected_type == 'histogram' and not
                  {family.name + suffix for suffix in ('_bucket', '_count', '_sum')} <= sample_names else None)
        inventory[key] = dict(available=reason is None, reason=reason,
                              name=family.name if family else None,
                              type=family.type if family else None,
                              sample_names=sorted({s['name'] for s in samples}),
                              samples=samples if reason is None else [])
        if reason is None and family.type in {'counter', 'histogram'}:
            for sample in samples:
                if sample['name'].endswith(('_total', '_count', '_sum', '_bucket')):
                    identity = json.dumps([sample['name'], sample['labels']], sort_keys=True)
                    counters[identity] = sample['value']
    exporter = {s.name: s.value for f in families for s in f.samples
                if s.name in {'process_start_time_seconds', 'process_id'} and math.isfinite(s.value)}
    return inventory, counters, exporter


class ObservationSampler:
    def __init__(self, output, endpoint, gateway_pid, run_id, interval=1.0, *,
                 runner_pid=None, stop=None, ready=None):
        self.output, self.endpoint = Path(output), endpoint.rstrip('/') + '/metrics'
        self.gateway_pid, self.run_id, self.interval = gateway_pid, run_id, interval
        self.runner_pid = os.getpid() if runner_pid is None else runner_pid
        parsed = urlsplit(self.endpoint)
        self.source = dict(scheme=parsed.scheme, host=parsed.hostname, port=parsed.port, path=parsed.path)
        self.stop = threading.Event() if stop is None else stop
        self.ready = threading.Event() if ready is None else ready
        self.previous_counters, self.previous_exporter = {}, None
        self.counter_epoch = 0
        self.status = dict(diagnostics_schema_version=DIAGNOSTICS_SCHEMA, run_id=run_id,
                           status='running', samples=0, sample_errors=0, write_errors=0,
                           interval_seconds=interval)

    @staticmethod
    def process_resources(pid):
        import psutil
        start = time.monotonic()
        if sys.platform == 'linux':
            try:
                status = (Path(psutil.PROCFS_PATH) / 'self' / 'status').read_bytes()
            except OSError as exc:
                return dict(available=False, pid=pid, sample_start=start, sample_end=time.monotonic(),
                            error_reason='procfs_pid_namespace_unverified', error_type=type(exc).__name__)
            namespace_pids = [line.split()[1:] for line in status.splitlines() if line.startswith(b'NStgid:')]
            # NStgid starts at the procfs mount's namespace; equal numbers alone can collide.
            if namespace_pids != [[str(os.getpid()).encode('ascii')]]:
                return dict(available=False, pid=pid, sample_start=start, sample_end=time.monotonic(),
                            error_reason='procfs_pid_namespace_unverified')
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                created = process.create_time()
                rss = process.memory_info().rss
                cpu = process.cpu_times()
            return dict(available=True, pid=pid, process_created_unix=created,
                        sample_start=start, sample_end=time.monotonic(), rss_bytes=rss,
                        user_cpu_seconds=cpu.user, system_cpu_seconds=cpu.system)
        except psutil.Error as exc:
            return dict(available=False, pid=pid, sample_start=start, sample_end=time.monotonic(),
                        error_type=type(exc).__name__)

    def sample(self, client):
        import psutil
        sent = time.monotonic()
        record = dict(diagnostics_schema_version=DIAGNOSTICS_SCHEMA, run_id=self.run_id,
                      source=self.source, host=socket.gethostname(), sent_at=sent, sampler_process_id=os.getpid(),
                      raw_metrics=None, metrics={}, unavailable_metrics=list(REQUIRED_METRICS),
                      exporter_identity={}, counter_epoch=self.counter_epoch,
                      counter_resets=[], disappeared_counters=[], error_type=None)
        try:
            with client.stream('GET', self.endpoint, headers={'Accept-Encoding': 'identity'}) as response:
                record.update(headers_at=time.monotonic(), http_status=response.status_code)
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() - record['headers_at'] > METRICS_BODY_TIMEOUT_SECONDS:
                        record['error_reason'] = 'metrics_body_time_limit'
                        raise ValueError('Metrics body exceeded collection time limit')
                    if len(body) + len(chunk) > MAX_METRICS_BYTES:
                        record['error_reason'] = 'metrics_body_size_limit'
                        raise ValueError('Metrics body exceeded collection size limit')
                    body.extend(chunk)
                record.update(received_at=time.monotonic(), raw_metrics=body.decode('utf-8'))
            inventory, counters, exporter = metric_inventory(record['raw_metrics'])
            resets = [name for name, value in counters.items()
                      if name in self.previous_counters and value < self.previous_counters[name]]
            disappeared = sorted(self.previous_counters.keys() - counters.keys())
            identity_changed = self.previous_exporter is not None and exporter != self.previous_exporter
            if resets or disappeared or identity_changed:
                self.counter_epoch += 1
            record.update(metrics=inventory, unavailable_metrics=[k for k, v in inventory.items() if not v['available']],
                          exporter_identity=exporter, exporter_identity_changed=identity_changed,
                          counter_epoch=self.counter_epoch, counter_resets=resets, disappeared_counters=disappeared)
            self.previous_counters, self.previous_exporter = counters, exporter
        except (httpx.HTTPError, ValueError) as exc:
            record.setdefault('received_at', time.monotonic())
            record['error_type'] = type(exc).__name__
        host_start = time.monotonic()
        try:
            host_cpu = dict(available=True, sample_start=host_start, cpu_seconds=psutil.cpu_times()._asdict(),
                            logical_cpus=psutil.cpu_count(), sample_end=time.monotonic())
        except (OSError, NotImplementedError) as exc:
            host_cpu = dict(available=False, sample_start=host_start, sample_end=time.monotonic(),
                            error_type=type(exc).__name__)
        record['resources'] = dict(gateway=self.process_resources(self.gateway_pid),
                                   runner=self.process_resources(self.runner_pid), host_cpu=host_cpu)
        record['finished_at'] = time.monotonic()
        record['interval_seconds'] = self.interval
        record['late_by_seconds'] = max(0., record['finished_at'] - sent - self.interval)
        return record

    def write_status(self):
        status_path = self.output / 'engine_observations_status.json'
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.status, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(status_path)

    def run(self):
        path = self.output / 'engine_observations.jsonl'
        self.status.update(started_at=time.monotonic(), sampler_process_id=os.getpid(),
                           runner_process_id=self.runner_pid)
        try:
            with path.open('w', encoding='utf-8') as stream, httpx.Client(timeout=2, trust_env=False) as client:
                while True:
                    final_sample = self.stop.is_set()
                    sample = self.sample(client)
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
                    stream.flush()
                    self.status['samples'] += 1
                    self.status['sample_errors'] += bool(sample['error_type'] or sample['unavailable_metrics'] or
                        not all(sample['resources'][name]['available'] for name in ('gateway', 'runner', 'host_cpu')))
                    if self.status['samples'] == 1:
                        self.write_status()
                    self.ready.set()
                    if final_sample:
                        break
                    self.stop.wait(max(0., self.interval - (time.monotonic() - sample['sent_at'])))
                    # Even stop during an in-flight sample gets a fresh post-stop sample.
                self.status['status'] = 'closed'
        except OSError as exc:
            self.status.update(write_errors=self.status['write_errors'] + 1, error_type=type(exc).__name__)
            raise
        finally:
            if self.status['status'] != 'closed':
                self.status['status'] = 'failed'
            self.status['closed_at'] = time.monotonic()
            try:
                self.write_status()
            finally:
                self.ready.set()


def _run_sampler(output, endpoint, gateway_pid, runner_pid, run_id, interval, stop, ready):
    ObservationSampler(output, endpoint, gateway_pid, run_id, interval,
                       runner_pid=runner_pid, stop=stop, ready=ready).run()


class _ProcessSignal:
    """A one-way, single-byte signal without a shared cross-process mutex."""

    def __init__(self, connection):
        self.connection, self.signalled = connection, False

    def is_set(self):
        if not self.signalled and self.connection.poll():
            if self.connection.recv_bytes(1) != b'\1':
                raise ValueError('Invalid sampler control signal')
            self.signalled = True
        return self.signalled

    def set(self):
        if not self.signalled:
            self.connection.send_bytes(b'\1')
            self.signalled = True

    def wait(self, timeout):
        return self.is_set() or (self.connection.poll(timeout) and self.is_set())


def _run_sampler_process(output, endpoint, gateway_pid, runner_pid, run_id, interval, stop_reader, ready_writer):
    with stop_reader, ready_writer:
        _run_sampler(output, endpoint, gateway_pid, runner_pid, run_id, interval,
                     _ProcessSignal(stop_reader), _ProcessSignal(ready_writer))


class SamplerWorker:
    """Own one collector; only readiness and shutdown read its small health file."""

    def __init__(self, output, endpoint, gateway_pid, runner_pid, run_id, mode='thread', interval=1.0):
        if mode not in ('thread', 'process'):
            raise ValueError('Sampler mode must be thread or process')
        if type(runner_pid) is not int or runner_pid <= 0:
            raise ValueError('The original runner PID is required')
        self.output, self.run_id, self.runner_pid, self.mode = Path(output), run_id, runner_pid, mode
        self.error, self.closing = None, None
        self.state = dict(started=False, stopped=False, forced_termination=False, unexpected_exit=False,
                          process_id=None, exitcode=None, stop_requested_at=None, ready_at=None,
                          start_method='spawn' if mode == 'process' else None)
        if mode == 'process':
            import multiprocessing
            context = multiprocessing.get_context('spawn')
            self.stop_reader, stop_writer = context.Pipe(duplex=False)
            ready_reader, self.ready_writer = context.Pipe(duplex=False)
            self.stop, self.ready = _ProcessSignal(stop_writer), _ProcessSignal(ready_reader)
            self.worker = context.Process(target=_run_sampler_process,
                args=(output, endpoint, gateway_pid, runner_pid, run_id, interval, self.stop_reader, self.ready_writer),
                name='engine-observation-sampler', daemon=True)
        else:
            self.stop, self.ready = threading.Event(), threading.Event()
            self.worker = threading.Thread(target=self._run_thread,
                args=(output, endpoint, gateway_pid, runner_pid, run_id, interval),
                name='engine-observation-sampler', daemon=True)

    def _run_thread(self, *args):
        try:
            _run_sampler(*args, self.stop, self.ready)
        except BaseException as exc:
            # Transport thread failures to the owner, which must reject the capture.
            self.error = exc

    def start(self):
        self.worker.start()
        self.state.update(started=True, process_id=self.worker.pid if self.mode == 'process' else os.getpid())
        if self.mode == 'process':
            self.stop_reader.close()
            self.ready_writer.close()

    def read_status(self, expected):
        status = json.loads((self.output / 'engine_observations_status.json').read_text(encoding='utf-8'))
        if (not isinstance(status, dict) or
                status.get('run_id') != self.run_id or status.get('diagnostics_schema_version') != DIAGNOSTICS_SCHEMA or
                type(status.get('sampler_process_id')) is not int or type(status.get('runner_process_id')) is not int or
                status.get('sampler_process_id') != self.state['process_id'] or
                status.get('runner_process_id') != self.runner_pid or status.get('status') != expected or
                type(status.get('samples')) is not int or status['samples'] < 1 or
                type(status.get('sample_errors')) is not int or not 0 <= status['sample_errors'] <= status['samples'] or
                type(status.get('write_errors')) is not int or status['write_errors'] != 0):
            raise RuntimeError('External diagnostic sampler identity/persistence/health mismatch')
        return status

    async def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                if self.ready.is_set():
                    break
            except (EOFError, OSError) as exc:
                raise RuntimeError('External diagnostic sampler readiness channel failed') from exc
            if not self.worker.is_alive():
                raise RuntimeError('External diagnostic sampler exited before readiness') from self.error
            if time.monotonic() >= deadline:
                raise TimeoutError('External diagnostic sampler startup timed out')
            await asyncio.sleep(.02)
        if not self.worker.is_alive() or self.error is not None:
            raise RuntimeError('External diagnostic sampler failed before warmup') from self.error
        self.read_status('running')
        self.state['ready_at'] = time.monotonic()

    def request_stop(self):
        if self.state['started'] and self.state['stop_requested_at'] is None:
            self.state['unexpected_exit'] = not self.worker.is_alive()
            self.state['stop_requested_at'] = time.monotonic()
            if self.mode == 'thread':
                self.stop.set()
            elif not self.state['unexpected_exit']:
                try:
                    self.stop.set()
                except OSError as exc:
                    self.error = exc
                    self.state['unexpected_exit'] = True

    async def wait_stopped(self, timeout):
        deadline = time.monotonic() + timeout
        while self.worker.is_alive():
            if time.monotonic() >= deadline:
                raise TimeoutError('External diagnostic sampler shutdown timed out')
            await asyncio.sleep(.02)

    def _release_process(self):
        self.worker.close()
        for connection in (self.stop_reader, self.ready_writer, self.stop.connection, self.ready.connection):
            connection.close()

    async def _close(self, timeout):
        if not self.state['started']:
            if self.mode == 'process':
                self._release_process()
            return
        try:
            try:
                await self.wait_stopped(timeout)
            except TimeoutError:
                if self.mode == 'process':
                    self.state['forced_termination'] = True
                    self.worker.terminate()
                    try:
                        await self.wait_stopped(2.)
                    except TimeoutError:
                        self.worker.kill()
                        await self.wait_stopped(2.)
                raise
        finally:
            self.state['stopped'] = not self.worker.is_alive()
            if self.state['stopped']:
                self.worker.join(0)
                if self.mode == 'process':
                    self.state['exitcode'] = self.worker.exitcode
                    self._release_process()
        if self.error is not None:
            raise RuntimeError(f'External diagnostic sampler failed: {type(self.error).__name__}') from self.error
        if self.state['unexpected_exit'] or (self.mode == 'process' and self.state['exitcode'] != 0):
            raise RuntimeError('External diagnostic sampler exited unexpectedly')
        self.read_status('closed')

    async def close(self, timeout):
        if self.closing is None:
            self.request_stop()
            self.closing = asyncio.create_task(self._close(timeout))
        cancelled = False
        while not self.closing.done():
            try:
                await asyncio.shield(self.closing)
            except asyncio.CancelledError:
                cancelled = True
        self.closing.result()
        if cancelled:
            raise asyncio.CancelledError
