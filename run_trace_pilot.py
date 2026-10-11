"""Open-loop replay of a prepared trace against one fixed-budget gateway."""
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
import subprocess
import sys
import time
import uuid

import httpx

from run_budget_pilot import save, validate_model, finalize, set_budget
from sustained_load import consume_request, WorkloadSpec, WORKLOADS
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
        for result, row in zip(rows, trace['requests']):
            if result['scheduled_at'] is None:
                result['scheduled_at'] = start + row['arrival_seconds']


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


async def run(args, trace, output, manifest):
    key = secrets.token_hex(32)
    env = dict(os.environ, ADMISSION_MODE='cost_aware', MAX_ADMITTED_COST=str(args.budget),
               ADMIN_API_KEY=key, MODEL_NAME=trace['model'], VLLM_URL=args.vllm_url.rstrip('/'),
               TELEMETRY_JSONL=str(output / 'gateway.jsonl'),
               TELEMETRY_STATUS_JSON=str(output / 'telemetry_status.json'),
               BENCHMARK_REQUEST_IDS='1', LOOP_LAG_TELEMETRY='1')
    process = None
    rows, warmup, changes, observations = [], [], [], []
    stop, poll_task = asyncio.Event(), None
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
                rows.extend(prepare_replay(trace, manifest['run_id']))
                limits = httpx.Limits(max_connections=args.max_pending, max_keepalive_connections=args.max_pending)
                async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits, trust_env=False) as traffic:
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
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Warmup failed to drain')
                        await asyncio.sleep(.05)
                    before = time.monotonic()
                    applied = await set_budget(control, key, args.budget)
                    acknowledged = time.monotonic()
                    start = time.monotonic() + .2
                    manifest['measurement_start'] = start
                    manifest['measurement_end'] = start + trace['duration_seconds']
                    changes.append(dict(target=start, sent=before, acknowledged=acknowledged, **applied))
                    poll_task = asyncio.create_task(poll_state(control, stop, observations))
                    await replay(traffic, trace, start, manifest['run_id'], args.drain_timeout, args.max_pending, rows)
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
                stop.set()
                try:
                    if poll_task is not None:
                        await poll_task
                    save(output / 'state_observations.json', observations)
                finally:
                    await finalize(process, output, manifest, rows, warmup, changes)
                # Uvicorn re-raises handled SIGTERM after graceful shutdown on some versions.
                if process is not None and process.returncode not in (0, -signal.SIGTERM):
                    raise RuntimeError(f'Gateway exited with status {process.returncode}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--budget', type=float, default=96)
    parser.add_argument('--vllm-url', default='http://127.0.0.1:8000')
    parser.add_argument('--warmup-requests', type=int, default=3)
    parser.add_argument('--max-pending', type=int, default=1000)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--drain-timeout', type=float, default=120)
    parser.add_argument('--startup-timeout', type=float, default=120)
    parser.add_argument('--ttft-slo', required=True, type=float)
    parser.add_argument('--max-scheduling-lag', type=float, default=.05)
    parser.add_argument('--cache-policy', choices=['disabled', 'enabled', 'unknown'], default='unknown',
                        help='Self-reported server prefix-cache setting; not verified by /v1/models')
    args = parser.parse_args()
    for value in (args.budget, args.request_timeout, args.drain_timeout, args.startup_timeout,
                  args.ttft_slo, args.max_scheduling_lag):
        if not math.isfinite(value) or value <= 0:
            parser.error('Numeric settings must be finite and positive')
    if not 1 <= args.max_pending <= 90000 or not 1 <= args.warmup_requests <= 1000:
        parser.error('Invalid max-pending or warmup count')
    trace = load_trace(args.trace)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(trace['model'], local_files_only=True)
    expected_tokenizer = trace.get('provenance', {}).get('tokenizer_sha256')
    if expected_tokenizer and tokenizer_fingerprint(tokenizer) != expected_tokenizer:
        raise ValueError('Prepared trace tokenizer differs from the local replay tokenizer')
    verify_prompts(trace, tokenizer)  # Before measurement; never tokenize on scheduler path.
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'trace.json', trace)
    manifest = dict(schema_version=1, config=vars(args), created_at=time.time(), status='running',
                    trace_sha256=sha256_file(output / 'trace.json'),
                    input_trace_sha256=sha256_file(args.trace), python=sys.version,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    packages={})
    manifest['code_sha256'] = {name: sha256_file(ROOT / name) for name in
                              ('run_trace_pilot.py', 'trace_workload.py', 'gateway.py',
                               'telemetry.py', 'sustained_load.py')}
    manifest['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
    for package in ('vllm', 'torch', 'transformers', 'fastapi', 'httpx', 'uvicorn'):
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
    try:
        asyncio.run(run(args, trace, output, manifest))
        manifest['status'] = 'collected'
    except (Exception, KeyboardInterrupt) as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        save(output / 'manifest.json', manifest)
    from analyze_trace_pilot import analyze
    report = analyze(output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['valid'] and report['load_fidelity']['valid'] else 1)


if __name__ == '__main__':
    main()
