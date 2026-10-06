import asyncio
from dataclasses import asdict
import json
import time

import httpx
import pytest

from run_budget_pilot import load_window, steps
from analyze_budget_pilot import analyze


def test_open_loop_steps_and_bounded_drain():
    async def scenario():
        updates = []
        async def handler(request):
            if request.url.path == '/admin/budget':
                budget = json.loads(request.content)['budget']
                updates.append(budget)
                return httpx.Response(200, json={'budget': budget})
            await asyncio.sleep(.02)
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n', headers={'x-request-id': str(time.monotonic())})
        rows, changes = [], []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            start = time.monotonic()
            await asyncio.gather(load_window(client, 'model', 50, .12, start, 1, rows),
                                 steps(client, 'secret', [80, 96, 112], .04, start, changes))
        assert len(rows) == 6 and all(r['success'] for r in rows)
        assert updates == [96, 112]
        assert len(changes) == 2
        async def hanging(request):
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(hanging)) as client:
            with pytest.raises(RuntimeError, match='drain timeout'):
                await load_window(client, 'model', 10, .01, time.monotonic(), .01, [])
    asyncio.run(scenario())


def test_analysis_rejects_missing_and_mismatched_events(tmp_path):
    manifest = {'status': 'collected', 'run_id': 'run', 'measurement_start': 1, 'measurement_end': 2,
                'config': {'budgets': [80], 'rps': 1, 'step_seconds': 1}}
    client = {'request_id': 'r', 'success': True, 'rejected': False, 'ttft': .1, 'started_at': 1, 'scheduled_at': 1}
    records = [dict(event=event, run_id='run', request_id='r', monotonic=1.1) for event in ('received', 'accepted', 'first_content', 'completed')]
    records += [dict(event='budget_updated', run_id='run', budget=80, monotonic=1),
                dict(event='snapshot', run_id='run', active_requests=0, current_admitted_cost=0, monotonic=2)]
    for name, obj in [('manifest.json', manifest), ('client.json', {'requests': [client]}), ('warmup.json', {'requests': []})]:
        (tmp_path / name).write_text(json.dumps(obj))
    def write():
        (tmp_path / 'gateway.jsonl').write_text('\n'.join(map(json.dumps, records)))
    write()
    assert analyze(tmp_path, plot=False)['valid']
    records.pop(3)
    write()
    assert not analyze(tmp_path, plot=False)['valid']


def test_full_runner_with_mock_vllm(tmp_path):
    """Exercise real gateway lifecycle, files, controller and plot without a GPU."""
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    for port in (8000, 8080):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                pytest.skip(f'Integration port {port} is in use')
    # Only the subprocess test environment sees this deterministic tokenizer.
    (tmp_path / 'transformers.py').write_text('''class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def apply_chat_template(self, *args, **kwargs): return {"input_ids": [1] * 98}
''')
    (tmp_path / 'mock_vllm.py').write_text('''from fastapi import FastAPI
from fastapi.responses import StreamingResponse
app = FastAPI()
@app.get('/v1/models')
def models(): return {"data": [{"id": "mock-model"}]}
@app.post('/v1/chat/completions')
def completion():
    async def stream():
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\\n\\n'
        yield b'data: [DONE]\\n\\n'
    return StreamingResponse(stream(), media_type='text/event-stream')
''')
    env = dict({k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}, PYTHONPATH=str(tmp_path) + os.pathsep + str(root))
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'mock_vllm:app', '--port', '8000'],
                               cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                httpx.get('http://127.0.0.1:8000/v1/models', trust_env=False).raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Mock startup timed out')
                time.sleep(.05)
        output = tmp_path / 'pilot'
        result = subprocess.run([sys.executable, 'run_budget_pilot.py', '--output-dir', str(output),
                                 '--model', 'mock-model', '--rps', '10', '--step-seconds', '.2',
                                 '--warmup-seconds', '.1', '--warmup-rps', '10', '--drain-timeout', '5'],
                                cwd=root, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((output / 'report.json').read_text())
        assert report['valid'] and report['measurement_requests'] == 10
        assert (output / 'timeseries.png').stat().st_size > 1000
    finally:
        process.terminate()
        process.wait(timeout=5)
