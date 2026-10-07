"""Bootstrap the recorded V1 core environment and supervise one vLLM server."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
CONTEXT_LENGTH = 8192


def command(python):
    return [str(python.parent / 'vllm'), 'serve', MODEL, '--host', '127.0.0.1',
            '--port', '8000', '--max-model-len', str(CONTEXT_LENGTH)]


def check_models(payload):
    model = next((x for x in payload.get('data', []) if x.get('id') == MODEL), None)
    if model is None:
        raise RuntimeError('Expected Qwen model is not served')
    if model.get('max_model_len') != CONTEXT_LENGTH:
        raise RuntimeError(f'Server context length must be {CONTEXT_LENGTH}; got {model.get("max_model_len")}')


def wait_ready(process, timeout, fetch, sleep=time.sleep):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('vLLM exited during startup; inspect vllm.log')
        try:
            payload = fetch()
        except (OSError, urllib.error.URLError, ValueError):
            sleep(1)
            continue
        check_models(payload)
        return payload
    raise RuntimeError('Model startup timed out; inspect vllm.log for download, CUDA or KV-cache errors')


def check_port(port=8000):
    # A live listener still blocks this bind, but TIME_WAIT from a prior run does not.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


def run(args):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('This setup targets Linux x86_64 NVIDIA rental machines')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 (V1 used 3.12.3); set PYTHON_BIN before running')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    logs = ROOT / 'results' / ('setup-' + stamp)
    logs.mkdir(parents=True)
    def record(name, value):
        (logs / name).write_text(json.dumps(value, indent=2) + '\n')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                          '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=15)
    if not gpu.stdout.strip():
        raise RuntimeError('No NVIDIA GPU detected')
    print('NVIDIA GPU / driver / VRAM:\n' + gpu.stdout, flush=True)
    record('host.json', {'python': sys.version, 'gpu': gpu.stdout, 'platform': platform.platform()})
    # Do not collide with or terminate an existing server.
    check_port()
    venv = ROOT / '.venv'
    python = venv / 'bin' / 'python'
    if not venv.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    if not python.is_file():
        raise RuntimeError('.venv exists but is not a Linux venv; move it aside explicitly')
    subprocess.run([str(python), '-c', 'import sys; assert sys.version_info[:2] == (3,12), "Existing .venv must use Python 3.12"'], check=True)
    if not args.skip_install:
        # Resolve together so vLLM cannot silently replace the required Torch build.
        subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:',
                        '--extra-index-url', 'https://download.pytorch.org/whl/cu130',
                        '-r', str(ROOT / 'requirements-v1.txt'),
                        '--report', str(logs / 'install-report.json')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    probe = '''import json, importlib.metadata as m, torch
assert m.version('vllm') == '0.30.0', 'Wrong vLLM version'
assert torch.__version__ == '2.13.0+cu130', 'Wrong Torch build'
assert torch.version.cuda == '13.0', 'Wrong Torch CUDA runtime'
assert torch.cuda.is_available(), 'CUDA unavailable: check host driver and GPU passthrough'
a = torch.ones((32,32), device='cuda'); b = a @ a; torch.cuda.synchronize()
assert b[0,0].item() == 32, 'CUDA compute check failed'
print(json.dumps({'vllm':m.version('vllm'), 'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
                  'visible_gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
'''
    result = subprocess.run([str(python), '-c', probe], check=True, capture_output=True, text=True)
    record('runtime.json', json.loads(result.stdout))
    with (logs / 'pip-freeze.txt').open('w') as handle:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], check=True, stdout=handle)
    launch = command(python)
    record('launch.json', {'argv': launch, 'model': MODEL, 'max_model_len': CONTEXT_LENGTH})
    print(f'Starting Qwen with explicit context length {CONTEXT_LENGTH}. Logs: {logs}', flush=True)
    # Keep the parent running; Ctrl+C terminates only the process group we own.
    with (logs / 'vllm.log').open('w') as handle:
        process = subprocess.Popen(launch, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def fetch():
                with opener.open('http://127.0.0.1:8000/v1/models', timeout=3) as response:
                    return json.load(response)
            payload = wait_ready(process, args.startup_timeout, fetch)
            record('models.json', payload)
            # Verify gateway's local-only tokenizer path after the model download.
            subprocess.run([str(python), '-c',
                'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(' + repr(MODEL) + ',local_files_only=True)'], check=True)
            print('READY: Qwen is loaded; server max_model_len=8192 verified.\n'
                  'Keep this terminal open. In a second terminal: source .venv/bin/activate\n'
                  'Then run the smoke/pilot commands. Ctrl+C stops this vLLM server.', flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f'vLLM exited with status {code}; inspect {logs / "vllm.log"}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-install', action='store_true', help='Reuse .venv, still verify all core versions and CUDA')
    parser.add_argument('--startup-timeout', type=int, default=1800, help='Model download/load timeout in seconds')
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error('startup-timeout must be positive')
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Stopped setup / owned vLLM server.', file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
