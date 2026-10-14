import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from types import SimpleNamespace
import uuid

import httpx
import pytest

from diagnostic_sampling import ObservationSampler, SamplerWorker
from test_diagnostic_sampling import metrics_text


@pytest.fixture
def metrics_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.calls += 1
            if self.server.calls == self.server.block_at:
                self.server.blocked.set()
                self.server.release.wait(10)
            body = metrics_text().encode()
            try:
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Forced-stop tests deliberately disconnect an in-flight collector.
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = False
    server.calls, server.block_at = 0, None
    server.blocked, server.release = threading.Event(), threading.Event()
    server.endpoint = f'http://127.0.0.1:{server.server_port}'
    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=.01))
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


async def wait_event(event):
    deadline = time.monotonic() + 5
    while not event.is_set():
        assert time.monotonic() < deadline, 'Mock request did not reach its expected boundary'
        await asyncio.sleep(.01)


@pytest.mark.parametrize('mode', ['thread', 'process'])
def test_worker_identity_readiness_and_fresh_post_stop_sample(tmp_path, metrics_server, mode):
    async def scenario():
        metrics_server.block_at = 2
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(),
                               uuid.uuid4().hex, mode=mode, interval=.01)
        worker.start()
        try:
            await worker.wait_ready(10)
            startup = json.loads((tmp_path / 'engine_observations_status.json').read_text())
            assert startup['status'] == 'running' and startup['samples'] >= 1
            assert startup['runner_process_id'] == os.getpid()
            assert (startup['sampler_process_id'] == os.getpid()) is (mode == 'thread')
            await wait_event(metrics_server.blocked)
            worker.request_stop()
            metrics_server.release.set()
        finally:
            metrics_server.release.set()
            await worker.close(5)
        await worker.close(5)  # Closing twice must not touch an already closed process handle.
        rows = [json.loads(line) for line in (tmp_path / 'engine_observations.jsonl').read_text().splitlines()]
        status = json.loads((tmp_path / 'engine_observations_status.json').read_text())
        assert len(rows) == status['samples'] >= 3 and status['sample_errors'] == status['write_errors'] == 0
        assert status['status'] == 'closed' and worker.state['stopped']
        assert not worker.state['forced_termination'] and not worker.state['unexpected_exit']
        assert worker.state['exitcode'] == (0 if mode == 'process' else None)
        assert rows[-2]['sent_at'] <= worker.state['stop_requested_at'] <= rows[-1]['sent_at']
        for row in rows:
            assert row['sampler_process_id'] == status['sampler_process_id'] == worker.state['process_id']
            assert row['resources']['runner']['available'] and row['resources']['runner']['pid'] == os.getpid()
            assert row['resources']['gateway']['pid'] == os.getpid()
    asyncio.run(scenario())


def test_sampler_uses_explicit_runner_pid_not_collector_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(ObservationSampler, 'process_resources', staticmethod(
        lambda pid: dict(available=True, pid=pid)))
    sampler = ObservationSampler(tmp_path, 'http://engine', 123, 'test', runner_pid=456)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=metrics_text()))) as client:
        row = sampler.sample(client)
    assert row['resources']['runner']['pid'] == 456 and row['resources']['gateway']['pid'] == 123
    assert row['sampler_process_id'] == os.getpid()


@pytest.mark.parametrize('mode', ['thread', 'process'])
@pytest.mark.parametrize('failure', ['startup_write', 'startup_status_write', 'final_status_write', 'early_exit'])
def test_worker_failures_are_not_successful_shutdowns(tmp_path, metrics_server, mode, failure):
    async def scenario():
        if failure == 'startup_write':
            (tmp_path / 'engine_observations.jsonl').mkdir()
        elif failure == 'startup_status_write':
            (tmp_path / 'engine_observations_status.tmp').mkdir()
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test', mode=mode)
        worker.start()
        try:
            if failure in ('startup_write', 'startup_status_write'):
                with pytest.raises((RuntimeError, OSError)):
                    await worker.wait_ready(10)
            else:
                await worker.wait_ready(10)
                if failure == 'final_status_write':
                    path = tmp_path / 'engine_observations_status.json'
                    path.unlink()
                    path.mkdir()
                else:
                    # Bypass the owner's recorded stop: premature exit, even with exit code zero.
                    worker.stop.set()
                    await worker.wait_stopped(5)
        finally:
            with pytest.raises(RuntimeError):
                await worker.close(5)
        assert worker.state['stopped'] and not worker.state['forced_termination']
        if failure == 'early_exit':
            assert worker.state['unexpected_exit']
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['thread', 'process'])
def test_start_failure_can_be_cleaned_without_a_worker(tmp_path, monkeypatch, mode):
    worker = SamplerWorker(tmp_path, 'http://unused', os.getpid(), os.getpid(), 'test', mode=mode)
    def failed_start():
        raise OSError('Cannot start worker')
    monkeypatch.setattr(worker.worker, 'start', failed_start)
    with pytest.raises(OSError, match='Cannot start'):
        worker.start()
    asyncio.run(worker.close(1))
    assert worker.state['started'] is False
    if mode == 'process':
        with pytest.raises(ValueError, match='closed'):
            worker.worker.is_alive()


def test_process_waits_for_ready_signal_not_just_a_visible_status_file(tmp_path, monkeypatch):
    from telemetry import DIAGNOSTICS_SCHEMA
    async def scenario():
        worker = SamplerWorker(tmp_path, 'http://unused', 2, 1, 'test', mode='process')
        worker.state['process_id'] = 3
        monkeypatch.setattr(worker.worker, 'is_alive', lambda: True)
        status = dict(run_id='test', diagnostics_schema_version=DIAGNOSTICS_SCHEMA, sampler_process_id=3,
                      runner_process_id=1, status='running', samples=1, sample_errors=0, write_errors=0)
        (tmp_path / 'engine_observations_status.json').write_text(json.dumps(status))
        try:
            with pytest.raises(TimeoutError, match='startup timed out'):
                await worker.wait_ready(.03)
            assert worker.state['ready_at'] is None
            worker.ready_writer.send_bytes(b'\1')
            await worker.wait_ready(1)
            assert worker.state['ready_at'] is not None
        finally:
            await worker.close(1)
    asyncio.run(scenario())


@pytest.mark.parametrize('damage', ['not_object', 'run_id', 'schema', 'sampler_pid', 'runner_pid',
                                  'status', 'samples', 'sample_errors', 'write_errors'])
def test_startup_health_requires_valid_identity_and_counters(tmp_path, damage):
    from telemetry import DIAGNOSTICS_SCHEMA
    worker = SamplerWorker(tmp_path, 'http://unused', 2, 1, 'test')
    worker.state['process_id'] = 1
    status = dict(run_id='test', diagnostics_schema_version=DIAGNOSTICS_SCHEMA, sampler_process_id=1,
                  runner_process_id=1, status='running', samples=1, sample_errors=0, write_errors=0)
    if damage == 'not_object':
        status = []
    else:
        key, value = dict(run_id=('run_id', 'other'), schema=('diagnostics_schema_version', -1),
            sampler_pid=('sampler_process_id', True), runner_pid=('runner_process_id', True),
            status=('status', 'closed'), samples=('samples', True),
            sample_errors=('sample_errors', 2), write_errors=('write_errors', True))[damage]
        status[key] = value
    (tmp_path / 'engine_observations_status.json').write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match='identity/persistence/health'):
        worker.read_status('running')


def test_process_crash_after_readiness_is_an_unexpected_exit(tmp_path, metrics_server):
    async def scenario():
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test', mode='process')
        worker.start()
        try:
            await worker.wait_ready(10)
            worker.worker.kill()
            await worker.wait_stopped(5)
        finally:
            with pytest.raises(RuntimeError, match='exited unexpectedly'):
                await worker.close(5)
        assert worker.state['stopped'] and worker.state['unexpected_exit']
        assert worker.state['exitcode'] != 0 and not worker.state['forced_termination']
    asyncio.run(scenario())


def test_process_termination_escalates_without_hiding_timeout(tmp_path, monkeypatch):
    async def scenario():
        calls = []
        class UnresponsiveProcess:
            alive, exitcode = True, None
            def is_alive(self):
                return self.alive
            def terminate(self):
                calls.append('terminate')
            def kill(self):
                calls.append('kill')
                self.alive, self.exitcode = False, -9
            def join(self, timeout):
                calls.append('join')
            def close(self):
                calls.append('close')
        worker = SamplerWorker(tmp_path, 'http://unused', os.getpid(), os.getpid(), 'test', mode='process')
        worker.worker.close()
        worker.worker = UnresponsiveProcess()
        worker.state['started'] = True
        async def wait_stopped(timeout):
            calls.append(timeout)
            if worker.worker.alive:
                raise TimeoutError('External diagnostic sampler shutdown timed out')
        monkeypatch.setattr(worker, 'wait_stopped', wait_stopped)
        with pytest.raises(TimeoutError, match='shutdown timed out'):
            await worker.close(.1)
        assert calls == [.1, 'terminate', 2., 'kill', 2., 'join', 'close']
        assert worker.state['forced_termination'] and worker.state['stopped'] and worker.state['exitcode'] == -9
    asyncio.run(scenario())


def test_process_owner_channel_loss_is_not_a_clean_stop(tmp_path, metrics_server):
    async def scenario():
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test', mode='process')
        worker.start()
        try:
            await worker.wait_ready(10)
            worker.stop.connection.close()
            await worker.wait_stopped(5)
        finally:
            with pytest.raises(RuntimeError, match='exited unexpectedly'):
                await worker.close(5)
        assert worker.state['stopped'] and worker.state['unexpected_exit'] and worker.state['exitcode'] != 0
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['thread', 'process'])
def test_startup_timeout_still_stops_owned_worker(tmp_path, metrics_server, mode):
    async def scenario():
        metrics_server.block_at = 1
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test', mode=mode)
        worker.start()
        try:
            await wait_event(metrics_server.blocked)
            with pytest.raises(TimeoutError, match='startup timed out'):
                await worker.wait_ready(.03)
        finally:
            metrics_server.release.set()
            await worker.close(5)
        assert worker.state['stopped'] and worker.state['ready_at'] is None
    asyncio.run(scenario())


def test_process_shutdown_timeout_is_forced_and_unqualified(tmp_path, metrics_server):
    async def scenario():
        metrics_server.block_at = 2
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test',
                               mode='process', interval=.01)
        worker.start()
        try:
            await worker.wait_ready(10)
            await wait_event(metrics_server.blocked)
        finally:
            with pytest.raises(TimeoutError, match='shutdown timed out'):
                await worker.close(.03)
            metrics_server.release.set()
        assert worker.state['stopped'] and worker.state['forced_termination']
        assert worker.state['exitcode'] != 0
        status = json.loads((tmp_path / 'engine_observations_status.json').read_text())
        assert status['status'] == 'running'
    asyncio.run(scenario())


def test_thread_timeout_is_explicit_not_a_fake_thread_kill(tmp_path, metrics_server):
    async def scenario():
        metrics_server.block_at = 2
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test', interval=.01)
        worker.start()
        try:
            await worker.wait_ready(10)
            await wait_event(metrics_server.blocked)
            with pytest.raises(TimeoutError):
                await worker.close(.03)
            assert not worker.state['stopped'] and not worker.state['forced_termination']
        finally:
            worker.request_stop()
            metrics_server.release.set()
            worker.worker.join(5)
            assert not worker.worker.is_alive()
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['thread', 'process'])
def test_repeated_cancellation_waits_for_worker_cleanup(tmp_path, metrics_server, mode):
    async def scenario():
        metrics_server.block_at = 2
        worker = SamplerWorker(tmp_path, metrics_server.endpoint, os.getpid(), os.getpid(), 'test',
                               mode=mode, interval=.01)
        worker.start()
        closing = None
        try:
            await worker.wait_ready(10)
            await wait_event(metrics_server.blocked)
            closing = asyncio.create_task(worker.close(5))
            await asyncio.sleep(.01)
            closing.cancel()
            await asyncio.sleep(.01)
            closing.cancel()
            metrics_server.release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert worker.state['stopped'] and not worker.state['forced_termination']
        finally:
            metrics_server.release.set()
            await worker.close(5)
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['thread', 'process'])
@pytest.mark.parametrize('outcome', ['success', 'warmup_failure', 'cancelled', 'cancel_before_close'])
def test_runner_closes_sampler_before_gateway_without_gpu_or_fixed_ports(
        tmp_path, metrics_server, monkeypatch, mode, outcome):
    import run_trace_pilot
    from test_trace_replay import sse, trace

    async def scenario():
        manifest = {}
        alive = False
        warmup_started = asyncio.Event()

        class GatewayProcess:
            pid = os.getpid()
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                assert manifest['sampler']['worker']['stopped']
                status = json.loads((tmp_path / 'engine_observations_status.json').read_text())
                assert status['status'] == 'closed'
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

        def launch(*args, **kwargs):
            nonlocal alive
            alive = True
            return GatewayProcess()

        async def serve(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'mock-model', 'max_model_len': 8192}]})
            if request.url.path == '/metrics':
                if not alive:
                    raise httpx.ConnectError('No gateway is running yet')
                return httpx.Response(200, json=dict(tokenizer_available=True,
                    telemetry_run_id='12345678123456781234567812345678', gateway_diagnostics=False,
                    current_in_flight=0, current_admitted_cost=0., state_sampled_at=time.monotonic()))
            if request.url.path == '/admin/budget':
                return httpx.Response(200, json=dict(applied_at=time.monotonic(), budget=96))
            warmup_started.set()
            if outcome == 'cancelled':
                await asyncio.Event().wait()
            if outcome == 'warmup_failure':
                return httpx.Response(500)
            return sse(request.headers['x-benchmark-request-id'])

        real_client = httpx.AsyncClient
        monkeypatch.setattr(run_trace_pilot.httpx, 'AsyncClient',
                            lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs))
        monkeypatch.setattr(run_trace_pilot, 'subprocess', SimpleNamespace(Popen=launch, STDOUT=-2))
        args = SimpleNamespace(budget=96, vllm_url=metrics_server.endpoint, gateway_diagnostics=False,
            engine_observations=True, sampler_mode=mode, startup_timeout=10, drain_timeout=5,
            max_pending=10, request_timeout=1, warmup_requests=1)
        if outcome == 'cancel_before_close':
            metrics_server.block_at = 2
            request_stop = SamplerWorker.request_stop
            cancelling = False
            def stop_and_cancel(worker):
                nonlocal cancelling
                request_stop(worker)
                if not cancelling:
                    cancelling = True
                    asyncio.get_running_loop().call_soon(task.cancel)
            monkeypatch.setattr(SamplerWorker, 'request_stop', stop_and_cancel)
        task = asyncio.create_task(run_trace_pilot.run(args, trace(), tmp_path, manifest))
        if outcome == 'cancel_before_close':
            try:
                await wait_event(metrics_server.blocked)
                task.cancel()
            finally:
                metrics_server.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == 'cancelled':
            await asyncio.wait_for(warmup_started.wait(), timeout=15)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == 'warmup_failure':
            with pytest.raises(RuntimeError, match='Warmup request failed'):
                await task
        else:
            await task
        assert manifest['sampler']['mode'] == mode and manifest['sampler']['worker']['stopped']
        assert (manifest['sampler']['worker']['process_id'] == os.getpid()) is (mode == 'thread')
        assert all((tmp_path / name).is_file() for name in ('client.json', 'warmup.json', 'state_observations.json'))
    asyncio.run(scenario())


def test_process_mode_cannot_silently_enable_or_skip_observations(tmp_path, monkeypatch):
    import run_trace_pilot
    monkeypatch.setattr('sys.argv', ['run_trace_pilot.py', '--trace', 'not-read.json',
        '--output-dir', str(tmp_path / 'run'), '--ttft-slo', '.5', '--sampler-mode', 'process'])
    with pytest.raises(SystemExit) as exc:
        run_trace_pilot.main()
    assert exc.value.code == 2 and not (tmp_path / 'run').exists()
