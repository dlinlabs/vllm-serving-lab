"""Open-loop trace replay with a fixed or predeclared admission-cost budget."""
import argparse
import asyncio
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid

import httpx

from run_budget_pilot import save, validate_model, finalize, set_budget
from sustained_load import consume_request, WorkloadSpec, WORKLOADS
from trace_control import (CONTROL_PROTOCOL, PLAN_FILE, initial_control_version, make_plan,
                           read_plan, validate_response, validate_schedule)
from trace_workload import load_trace, sha256_file, verify_prompts, tokenizer_fingerprint

ROOT = Path(__file__).resolve().parent


def unlaunched(row, scheduled, identifier, reason):
    return dict(trace_id=row['trace_id'], client_request_id=identifier, request_id=None,
                scheduled_at=scheduled, started_at=None, first_content_at=None, finished_at=None,
                status=None, ttft=None, end_to_end=None, stream_complete=False,
                input_tokens=None, output_tokens=None, max_output_tokens=row['max_tokens'],
                expected_input_tokens=row['input_tokens'], success=False, rejected=False,
                error_type='NotLaunched', error_message=reason, error_phase='scheduler')


def prepare_replay(trace, run_id):
    """Allocate IDs and outcomes before choosing the measurement clock origin."""
    namespace = uuid.UUID(run_id)
    return [unlaunched(row, None, uuid.uuid5(namespace, row['trace_id']).hex, 'collection interrupted')
            for row in trace['requests']]


def complete_schedules(rows, trace, start):
    for result, row in zip(rows, trace['requests']):
        if result['scheduled_at'] is None:
            result['scheduled_at'] = start + row['arrival_seconds']


async def replay(client, trace, start, run_id, drain_timeout, max_pending, rows):
    """Never wait for a response to schedule the next arrival; preserve every outcome."""
    active, errors = set(), []
    # Keep direct callers supported; the production runner preallocates before warmup.
    if not rows:
        rows.extend(prepare_replay(trace, run_id))
    if len(rows) != len(trace['requests']):
        raise ValueError('Prepared replay row count differs from trace')

    async def send(index, row):
        try:
            result = await consume_request(
                client, start + row['arrival_seconds'], trace['model'],
                WorkloadSpec('trace', row['prompt'], row['max_tokens']),
                client_request_id=rows[index]['client_request_id'], capture_cancelled=True)
            rows[index] = dict(asdict(result), trace_id=row['trace_id'],
                               expected_input_tokens=row['input_tokens'])
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            rows[index]['error_message'] = errors[-1]

    try:
        for index, row in enumerate(trace['requests']):
            await asyncio.sleep(max(0, start + row['arrival_seconds'] - time.monotonic()))
            if len(active) >= max_pending:
                raise RuntimeError('Load generator pending limit exceeded; trace was not fully offered')
            task = asyncio.create_task(send(index, row))
            active.add(task)
            task.add_done_callback(active.discard)
        await asyncio.sleep(max(0, start + trace['duration_seconds'] - time.monotonic()))
        if active:
            _, pending = await asyncio.wait(list(active), timeout=drain_timeout)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if errors:
            raise RuntimeError('Unexpected client implementation failure: ' + errors[0])
    finally:
        remaining = list(active)
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        # Fill schedules for unlaunched/implementation-error rows after collection,
        # rather than scanning the full trace on the arrival scheduler's path.
        complete_schedules(rows, trace, start)


async def apply_budget_step(client, key, plan, index, start, base_version, changes):
    step = plan['budget_schedule'][index]
    target = start + step['at_seconds'] if start is not None else None
    command = dict(step_index=index, requested_budget=step['budget'], target=target,
                   sent=None, acknowledged=None, status='waiting')
    changes.append(command)
    try:
        if index:
            if changes[-2]['acknowledged'] > target:
                raise RuntimeError('Previous budget acknowledgement crossed the next target; no catch-up commands')
            await asyncio.sleep(max(0, target - time.monotonic()))
        sent = time.monotonic()
        if index and (sent < target - 1e-6 or sent - target > plan['max_budget_lag'] or
                      sent >= start + plan['duration_seconds']):
            raise RuntimeError('Budget command missed its predeclared send window')
        command.update(sent=sent, status='sent')
        response = await set_budget(client, key, step['budget'])
        command['acknowledged'] = time.monotonic()
        previous = plan['budget_schedule'][max(0, index - 1)]['budget']
        command.update(validate_response(response, step['budget'], previous, base_version + index + 1,
                                         command['sent'], command['acknowledged']))
        if index and (command['applied_at'] - target > plan['max_budget_lag'] or
                      command['applied_at'] >= start + plan['duration_seconds']):
            raise RuntimeError('Budget application missed its predeclared window')
        command['status'] = 'acknowledged'
    except asyncio.CancelledError:
        command.update(status='cancelled', error_type='CancelledError',
                       error='Control cancelled; an in-flight application may be unknown')
        raise
    except (httpx.HTTPError, ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
        command.update(status='failed', error_type=type(exc).__name__,
                       error='Budget HTTP request failed (URL omitted)' if isinstance(exc, httpx.HTTPError) else str(exc))
        raise


async def scheduled_budgets(client, key, plan, start, base_version, changes):
    for index in range(1, len(plan['budget_schedule'])):
        await apply_budget_step(client, key, plan, index, start, base_version, changes)


async def poll_state(client, stop, records, interval=1.0):
    while not stop.is_set():
        sent = time.monotonic()
        try:
            response = await client.get('http://127.0.0.1:8080/metrics')
            response.raise_for_status()
            metrics = response.json()
            received = time.monotonic()
            records.append(dict(sent_at=sent, received_at=received,
                                state_age_seconds=received - metrics['state_sampled_at'], metrics=metrics))
        except httpx.HTTPError as exc:
            records.append(dict(sent_at=sent, received_at=time.monotonic(), error_type=type(exc).__name__))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def gateway_environment(args, trace, output, key):
    return dict(os.environ, ADMISSION_MODE='cost_aware', MAX_ADMITTED_COST=str(args.budget),
               ADMIN_API_KEY=key, MODEL_NAME=trace['model'], VLLM_URL=args.vllm_url.rstrip('/'),
               TELEMETRY_JSONL=str(output / 'gateway.jsonl'),
               TELEMETRY_STATUS_JSON=str(output / 'telemetry_status.json'),
               BENCHMARK_REQUEST_IDS='1', LOOP_LAG_TELEMETRY='1',
               GATEWAY_DIAGNOSTICS='1' if args.gateway_diagnostics else '0')


async def run(args, trace, output, manifest):
    from telemetry import DIAGNOSTICS_SCHEMA, INVESTIGATION_PROTOCOL
    plan = read_plan(output, manifest, trace)
    if plan is not None and sha256_file(output / 'trace.json') != manifest['trace_sha256']:
        raise ValueError('Replay trace hash mismatch')
    key = secrets.token_hex(32)
    env = gateway_environment(args, trace, output, key)
    sampler_enabled = args.gateway_diagnostics or args.engine_observations
    if args.sampler_mode == 'process' and not sampler_enabled:
        raise ValueError('Process sampler requires --engine-observations or --gateway-diagnostics')
    manifest.update(investigation_protocol=INVESTIGATION_PROTOCOL, diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                    gateway_diagnostics_requested=args.gateway_diagnostics,
                    runner_process_id=os.getpid(), host=socket.gethostname(),
                    sampler=dict(enabled=sampler_enabled, mode=args.sampler_mode, interval_seconds=1.0,
                                 resources=['gateway_current_rss_cpu', 'runner_current_rss_cpu', 'host_cpu'],
                                 gpu='not_collected', request_linked_engine_trace='not_collected'),
                    package_scope='Runner Python environment; not proof of a remote backend version',
                    backend_launch_provenance='Not observed by this runner; archive and verify backend launch records separately')
    process = None
    rows, warmup, changes, observations = [], [], [], []
    stop, poll_task = asyncio.Event(), None
    sampler = None
    runner_diagnostics = None
    gc_control = None
    async with httpx.AsyncClient(timeout=10, trust_env=False) as control:
        try:
            await control.get('http://127.0.0.1:8080/metrics')
        except httpx.ConnectError:
            pass
        else:
            raise RuntimeError('Port 8080 already responds; stop the existing gateway first')
        response = await control.get(args.vllm_url.rstrip('/') + '/v1/models')
        response.raise_for_status()
        validate_model(response.json(), trace['model'])
        manifest['models'] = response.json()
        with (output / 'gateway-process.log').open('w') as log:
            try:
                process = subprocess.Popen([sys.executable, str(ROOT / 'gateway.py')], env=env,
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Gateway exited; inspect gateway-process.log')
                    try:
                        response = await control.get('http://127.0.0.1:8080/metrics')
                        response.raise_for_status()
                        metrics = response.json()
                        break
                    except httpx.HTTPError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Gateway startup timed out')
                        await asyncio.sleep(.2)
                if not metrics['tokenizer_available']:
                    raise RuntimeError('Gateway tokenizer is unavailable')
                manifest['run_id'] = metrics['telemetry_run_id']
                manifest['initial_metrics'] = metrics
                manifest['gateway_diagnostics_observed'] = metrics.get('gateway_diagnostics')
                if metrics.get('gateway_diagnostics') is not args.gateway_diagnostics:
                    raise RuntimeError('Requested and observed gateway diagnostic modes differ')
                manifest['gateway_launch'] = dict(process_id=process.pid, executable=sys.executable,
                    entrypoint='gateway.py', settings={name: env[name] for name in
                        ('ADMISSION_MODE', 'MAX_ADMITTED_COST', 'MODEL_NAME', 'GATEWAY_DIAGNOSTICS',
                         'BENCHMARK_REQUEST_IDS', 'LOOP_LAG_TELEMETRY')})
                rows.extend(prepare_replay(trace, manifest['run_id']))
                if getattr(args, 'runner_diagnostics', False):
                    from runner_diagnostics import RunnerDiagnostics, SCHEMA, LOOP_INTERVAL
                    manifest['runner_diagnostics'] = dict(schema_version=SCHEMA, enabled=True,
                        process_id=os.getpid(), interval_seconds=LOOP_INTERVAL)
                    runner_diagnostics = RunnerDiagnostics(output, manifest['run_id'])
                    await runner_diagnostics.start()
                if sampler_enabled:
                    from diagnostic_sampling import SamplerWorker
                    sampler = SamplerWorker(output, args.vllm_url, process.pid, manifest['runner_process_id'],
                                            manifest['run_id'], mode=args.sampler_mode)
                    manifest['sampler']['worker'] = sampler.state
                    sampler.start()
                    await sampler.wait_ready(args.startup_timeout)
                limits = httpx.Limits(max_connections=args.max_pending, max_keepalive_connections=args.max_pending)
                async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits, trust_env=False) as traffic:
                    manifest['warmup_start'] = time.monotonic()
                    for _ in range(args.warmup_requests):
                        result = await consume_request(traffic, time.monotonic(), trace['model'], WORKLOADS[0],
                                                       client_request_id=uuid.uuid4().hex)
                        warmup.append(asdict(result))
                        if not result.success:
                            raise RuntimeError('Warmup request failed')
                    # Client finishing is not proof the gateway released its reservation.
                    deadline = time.monotonic() + args.drain_timeout
                    while True:
                        response = await control.get('http://127.0.0.1:8080/metrics')
                        response.raise_for_status()
                        if response.json()['current_in_flight'] == 0:
                            manifest['post_warmup_metrics'] = response.json()
                            manifest['warmup_end'] = time.monotonic()
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Warmup failed to drain')
                        await asyncio.sleep(.05)
                    if plan is None:
                        before = time.monotonic()
                        applied = await set_budget(control, key, args.budget)
                        changes.append(dict(sent=before, acknowledged=time.monotonic(), **applied))
                    else:
                        base_version = initial_control_version(manifest, plan)
                        await apply_budget_step(control, key, plan, 0, None, base_version, changes)
                    from runner_gc_control import gc_spec, RunnerGCControl
                    spec = gc_spec(vars(args), trace['duration_seconds'])
                    if spec is not None:
                        manifest['runner_gc_control'] = spec
                        gc_control = RunnerGCControl(output, manifest['run_id'], spec)
                        await gc_control.prepare()
                    start = time.monotonic() + .2
                    manifest['measurement_start'] = start
                    manifest['measurement_end'] = start + trace['duration_seconds']
                    changes[0]['target'] = start
                    poll_task = asyncio.create_task(poll_state(control, stop, observations))
                    async def traffic_and_controls():
                        if plan is not None and len(plan['budget_schedule']) > 1:
                            async with asyncio.TaskGroup() as tasks:
                                tasks.create_task(scheduled_budgets(control, key, plan, start, base_version, changes))
                                tasks.create_task(replay(traffic, trace, start, manifest['run_id'],
                                                         args.drain_timeout, args.max_pending, rows))
                        else:
                            await replay(traffic, trace, start, manifest['run_id'], args.drain_timeout, args.max_pending, rows)
                    if gc_control is not None:
                        await gc_control.supervise(traffic_and_controls())
                    else:
                        await traffic_and_controls()
                deadline = time.monotonic() + args.drain_timeout
                while True:
                    response = await control.get('http://127.0.0.1:8080/metrics')
                    response.raise_for_status()
                    metrics = response.json()
                    if metrics['current_in_flight'] == 0 and abs(metrics['current_admitted_cost']) < 1e-6:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Gateway failed to drain')
                    await asyncio.sleep(.1)
                manifest['final_metrics'] = metrics
            finally:
                async def cleanup():
                    if gc_control is not None:
                        gc_control.restore()
                    stop.set()
                    if sampler is not None:
                        sampler.request_stop()
                    try:
                        tasks = [poll_task] if poll_task is not None else []
                        if sampler is not None:
                            tasks.append(sampler.close(args.drain_timeout))
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        save(output / 'state_observations.json', observations)
                        errors = [result for result in results if isinstance(result, BaseException)]
                        if errors:
                            manifest['collector_errors'] = [type(error).__name__ for error in errors]
                            raise errors[0]
                    finally:
                        try:
                            if 'measurement_start' in manifest:
                                complete_schedules(rows, trace, manifest['measurement_start'])
                        finally:
                            try:
                                try:
                                    if gc_control is not None:
                                        gc_control.close()
                                finally:
                                    if runner_diagnostics is not None:
                                        await runner_diagnostics.close()
                            finally:
                                await finalize(process, output, manifest, rows, warmup, changes)
                    # Some Uvicorn versions re-raise SIGTERM after graceful shutdown.
                    if process is not None and process.returncode not in (0, -signal.SIGTERM):
                        raise RuntimeError(f'Gateway exited with status {process.returncode}')

                cleanup_task = asyncio.create_task(cleanup())
                cancellation = None
                # Protect the entire ordering, even cancellation before sampler.close starts.
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError as exc:
                        cancellation = exc
                cleanup_task.result()
                if cancellation is not None:
                    raise cancellation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--output-dir', required=True)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--budget', type=float, help='Fixed cost budget (default: 96)')
    budget.add_argument('--budget-schedule', help='JSON array of {at_seconds, budget}; first offset must be zero')
    parser.add_argument('--vllm-url', default='http://127.0.0.1:8000')
    parser.add_argument('--warmup-requests', type=int, default=3)
    parser.add_argument('--max-pending', type=int, default=1000)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--drain-timeout', type=float, default=120)
    parser.add_argument('--startup-timeout', type=float, default=120)
    parser.add_argument('--ttft-slo', required=True, type=float)
    parser.add_argument('--e2e-slo', type=float, help='Optional scheduled-arrival completion deadline in seconds')
    parser.add_argument('--max-scheduling-lag', type=float, default=.05)
    parser.add_argument('--max-budget-lag', type=float, default=.05,
                        help='Maximum scheduled budget send/application lag in seconds')
    parser.add_argument('--cache-policy', choices=['disabled', 'enabled', 'unknown'], default='unknown',
                        help='Self-reported server prefix-cache setting; not verified by /v1/models')
    parser.add_argument('--gateway-diagnostics', action='store_true',
                        help='Opt in to gateway phases/GC/CPU and external engine/resource observations')
    parser.add_argument('--engine-observations', action='store_true',
                        help='External sampling without enabling gateway diagnostics (for a later matched control)')
    parser.add_argument('--runner-diagnostics', action='store_true',
                        help='Opt in to runner GC spans, loop delays and CPU timing; does not change GC policy')
    parser.add_argument('--runner-gc-policy', choices=['automatic', 'defer'], default='automatic',
                        help='Opt in to bounded GC deferral in the request runner during replay/drain')
    parser.add_argument('--runner-max-rss-mib', type=float,
                        help='Required sampled RSS ceiling for deferred GC; no implicit memory allowance')
    parser.add_argument('--sampler-mode', choices=['thread', 'process'], default='thread',
                        help='External sampler execution; process uses spawn and retains the original runner PID')
    parser.add_argument('--protocol',
                        help='Experiment protocol JSON: refuse a capture that differs from the declared run and '
                             'record the protocol hash and acceptance status in the manifest')
    parser.add_argument('--protocol-run', help='Run name declared in --protocol; must equal the output directory name')
    args = parser.parse_args()
    if (args.protocol is None) != (args.protocol_run is None):
        parser.error('--protocol and --protocol-run go together')
    if args.sampler_mode == 'process' and not (args.gateway_diagnostics or args.engine_observations):
        parser.error('Process sampler requires --engine-observations or --gateway-diagnostics')
    for value in (args.request_timeout, args.drain_timeout, args.startup_timeout,
                  args.ttft_slo, args.max_scheduling_lag, args.max_budget_lag,
                  *([args.budget] if args.budget is not None else []),
                  *([args.e2e_slo] if args.e2e_slo is not None else [])):
        if not math.isfinite(value) or value <= 0:
            parser.error('Numeric settings must be finite and positive')
    if not 1 <= args.max_pending <= 90000 or not 1 <= args.warmup_requests <= 1000:
        parser.error('Invalid max-pending or warmup count')
    trace = load_trace(args.trace)
    from runner_gc_control import gc_spec
    try:
        gc_spec(vars(args), trace['duration_seconds'])
    except ValueError as exc:
        parser.error(str(exc))
    if args.budget_schedule is not None:
        schedule = validate_schedule(json.loads(Path(args.budget_schedule).read_text()), trace['duration_seconds'])
        args.budget = schedule[0]['budget']
    else:
        args.budget = 96. if args.budget is None else args.budget
        schedule = validate_schedule([dict(at_seconds=0, budget=args.budget)], trace['duration_seconds'])
    protocol_binding = None
    if args.protocol is not None:
        from experiment_protocol import collection_binding
        try:
            protocol_binding = collection_binding(args.protocol, args.protocol_run, args.output_dir, dict(vars(args)),
                                                  trace, sha256_file(args.trace), schedule)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(trace['model'], local_files_only=True)
    expected_tokenizer = trace.get('provenance', {}).get('tokenizer_sha256')
    if expected_tokenizer and tokenizer_fingerprint(tokenizer) != expected_tokenizer:
        raise ValueError('Prepared trace tokenizer differs from the local replay tokenizer')
    verify_prompts(trace, tokenizer)  # Before measurement; never tokenize on scheduler path.
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'trace.json', trace)
    config = dict(vars(args))
    config['vllm_url'] = str(httpx.URL(args.vllm_url).copy_with(username='', password='', query=None, fragment=None))
    manifest = dict(schema_version=1, config=config, created_at=time.time(), status='running',
                    trace_sha256=sha256_file(output / 'trace.json'),
                    input_trace_sha256=sha256_file(args.trace), python=sys.version,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    packages={})
    save(output / PLAN_FILE, make_plan(config, trace, manifest['trace_sha256'], schedule))
    manifest.update(control_protocol=CONTROL_PROTOCOL, replay_plan_sha256=sha256_file(output / PLAN_FILE))
    if protocol_binding is not None:
        manifest['experiment_protocol'] = protocol_binding
    manifest['code_sha256'] = {name: sha256_file(ROOT / name) for name in
                              ('run_trace_pilot.py', 'trace_workload.py', 'trace_control.py', 'gateway.py',
                               'telemetry.py', 'sustained_load.py', 'run_budget_pilot.py',
                               'diagnostic_sampling.py', 'runner_diagnostics.py', 'runner_gc_control.py', 'analyze_budget_pilot.py',
                               'analyze_trace_pilot.py', 'analyze_trace_dynamics.py', 'experiment_protocol.py',
                               'requirements-v1.txt')}
    manifest['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
    for package in ('vllm', 'torch', 'transformers', 'fastapi', 'httpx', 'httpcore',
                    'uvicorn', 'psutil', 'prometheus-client', 'numpy', 'matplotlib'):
        try:
            manifest['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest['packages'][package] = None
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10)
        manifest['gpu'] = gpu.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        manifest['gpu'] = None
    save(output / 'manifest.json', manifest)
    try:
        asyncio.run(run(args, trace, output, manifest))
        manifest['status'] = 'collected'
    except (Exception, KeyboardInterrupt) as exc:
        manifest['status'] = 'failed'
        if isinstance(exc, httpx.HTTPError):
            manifest['error'] = f'{type(exc).__name__}: HTTP request failed (URL omitted)'
            if isinstance(exc, httpx.HTTPStatusError):
                manifest['error_http_status'] = exc.response.status_code
        else:
            manifest['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        save(output / 'manifest.json', manifest)
    from analyze_trace_pilot import analyze
    report = analyze(output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['valid'] and report['load_fidelity']['valid'] and
                     report.get('control_fidelity', {'valid': True})['valid'] and
                     report.get('runner_diagnostic_integrity', {'valid': True})['valid'] and
                     report.get('runner_gc_integrity', {'valid': True})['valid'] and
                     report.get('diagnostic_integrity', {'valid': True})['valid'] else 1)


if __name__ == '__main__':
    main()
