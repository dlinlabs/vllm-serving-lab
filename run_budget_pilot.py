"""Single-host pilot: owns one gateway process; requires an existing vLLM server."""
import argparse
import asyncio
from dataclasses import asdict
import json
import importlib.metadata
import shutil
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

import httpx
from sustained_load import consume_request, WORKLOADS

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


async def load_window(client, model, rps, duration, start, drain_timeout, rows):
    tasks = []
    try:
        number = 0
        while number / rps < duration:
            scheduled = start + number / rps
            await asyncio.sleep(max(0, scheduled - time.monotonic()))
            task = asyncio.create_task(consume_request(client, scheduled, model, WORKLOADS[number % len(WORKLOADS)]))
            tasks.append(task)
            number += 1
        await asyncio.sleep(max(0, start + duration - time.monotonic()))
        _, pending = await asyncio.wait(tasks, timeout=drain_timeout)
        if pending:
            raise RuntimeError(f'{len(pending)} requests exceeded drain timeout')
        # Surface unexpected task errors rather than silently dropping requests.
        for task in tasks:
            task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and task.exception() is None:
                rows.append(asdict(task.result()))


async def set_budget(client, key, budget):
    response = await client.post('http://127.0.0.1:8080/admin/budget',
                                 headers={'X-Admin-API-Key': key}, json={'budget': budget})
    response.raise_for_status()
    return response.json()


async def steps(client, key, budgets, seconds, start, changes):
    # The first budget is set before the shared measurement start.
    for index, budget in enumerate(budgets[1:], 1):
        target = start + index * seconds
        await asyncio.sleep(max(0, target - time.monotonic()))
        before = time.monotonic()
        result = await set_budget(client, key, budget)
        changes.append(dict(target=target, sent=before, acknowledged=time.monotonic(), **result))


def validate_model(models, model):
    served = next((item for item in models.get('data', []) if item.get('id') == model), None)
    if served is None or served.get('max_model_len') != 8192:
        raise RuntimeError('Requested model must be served with max_model_len=8192')


async def finalize(process, output, manifest, rows, warmup, changes):
    # Saving can fail (e.g. disk full); that must never skip process cleanup.
    try:
        save(output / 'client.json', {'requests': rows})
        save(output / 'warmup.json', {'requests': warmup})
        save(output / 'budget_changes.json', changes)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
                manifest['forced_gateway_kill'] = True


async def run(args, output, manifest):
    key = secrets.token_hex(32)
    env = dict(os.environ, ADMISSION_MODE='cost_aware', MAX_ADMITTED_COST=str(args.budgets[0]),
               ADMIN_API_KEY=key, MODEL_NAME=args.model, VLLM_URL=args.vllm_url,
               TELEMETRY_JSONL=str(output / 'gateway.jsonl'))
    process = None
    rows, warmup, changes = [], [], []
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=10) as control:
        # Refuse to interfere with another gateway using the fixed benchmark port.
        try:
            await control.get('http://127.0.0.1:8080/metrics')
        except httpx.ConnectError:
            pass
        else:
            raise RuntimeError('Port 8080 already responds; stop the existing gateway first')
        response = await control.get(args.vllm_url.rstrip('/') + '/v1/models')
        response.raise_for_status()
        models = response.json()
        validate_model(models, args.model)
        manifest['models'] = models
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
                    raise RuntimeError('Tokenizer unavailable; use the same model cache/environment as vLLM')
                manifest['initial_metrics'] = metrics
                manifest['run_id'] = metrics['telemetry_run_id']
                async with httpx.AsyncClient(timeout=httpx.Timeout(120), limits=limits) as traffic:
                    await load_window(traffic, args.model, args.warmup_rps, args.warmup_seconds,
                                      time.monotonic(), args.drain_timeout, warmup)
                    if not warmup or not all(r['success'] and r['request_id'] for r in warmup):
                        raise RuntimeError('Warmup/smoke test failed')
                    initial = await set_budget(control, key, args.budgets[0])
                    start = time.monotonic() + .2
                    manifest['measurement_start'] = start
                    manifest['measurement_end'] = start + args.step_seconds * len(args.budgets)
                    changes.append(dict(target=start, acknowledged=time.monotonic(), **initial))
                    # A control failure cancels traffic; a traffic failure cancels control.
                    async with asyncio.TaskGroup() as group:
                        group.create_task(steps(control, key, args.budgets, args.step_seconds, start, changes))
                        group.create_task(load_window(traffic, args.model, args.rps,
                                          args.step_seconds * len(args.budgets), start,
                                          args.drain_timeout, rows))
                deadline = time.monotonic() + args.drain_timeout
                while True:
                    response = await control.get('http://127.0.0.1:8080/metrics')
                    response.raise_for_status()
                    metrics = response.json()
                    if metrics['current_in_flight'] == 0 and abs(metrics['current_admitted_cost']) < 1e-6:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Gateway failed to drain')
                    await asyncio.sleep(.2)
                manifest['final_metrics'] = metrics
                for field in ('tokenizer_fallback_requests', 'telemetry_dropped_records', 'telemetry_write_errors'):
                    if metrics[field]:
                        raise RuntimeError(f'Invalid experiment: {field}={metrics[field]}')
                # Allow a post-drain snapshot before graceful shutdown flushes the log.
                await asyncio.sleep(1.1)
            finally:
                await finalize(process, output, manifest, rows, warmup, changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--vllm-url', default='http://127.0.0.1:8000')
    parser.add_argument('--rps', type=float, default=30)
    parser.add_argument('--budgets', type=float, nargs='+', default=[80, 96, 112, 96, 80])
    parser.add_argument('--step-seconds', type=float, default=30)
    parser.add_argument('--warmup-seconds', type=float, default=15)
    parser.add_argument('--warmup-rps', type=float, default=1)
    parser.add_argument('--drain-timeout', type=float, default=120)
    parser.add_argument('--startup-timeout', type=float, default=120)
    args = parser.parse_args()
    for value in [args.rps, args.step_seconds, args.warmup_seconds, args.warmup_rps,
                  args.drain_timeout, args.startup_timeout, *args.budgets]:
        if not math.isfinite(value) or value <= 0:
            parser.error('All numeric parameters must be finite and positive')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'config': vars(args), 'created_at': time.time(), 'status': 'running',
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    manifest['python'] = sys.version
    manifest['packages'] = {}
    for package in ('vllm', 'torch', 'transformers', 'fastapi', 'httpx', 'uvicorn'):
        try:
            manifest['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest['packages'][package] = None
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        if shutil.which('nvidia-smi'):
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                  '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
            manifest['gpu'] = gpu.stdout.strip()
        asyncio.run(run(args, output, manifest))
        if manifest.get('forced_gateway_kill'):
            raise RuntimeError('Gateway needed forced shutdown; telemetry may be incomplete')
        manifest['status'] = 'collected'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        save(output / 'manifest.json', manifest)
    from analyze_budget_pilot import analyze
    report = analyze(output)
    print(json.dumps(report, indent=2))
    if not report['valid']:
        raise SystemExit('Pilot failed data validation; inspect report.json')


if __name__ == '__main__':
    main()
