"""Run frozen same-host fixed-cap or low-cap switching collections; stop on first failure."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time

from experiment_protocol import (capture_identity, capture_mismatches, collection_binding, expected_schedule,
                                 load_protocol, normalized_sha256)
from trace_control import make_plan, validate_schedule
from trace_workload import load_trace, sha256_file

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'experiments/cap-range-v1.json'
BACKEND = 'http://127.0.0.1:8000'
GATES = ('load_fidelity', 'control_fidelity', 'diagnostic_integrity',
         'runner_diagnostic_integrity', 'runner_gc_integrity')
COST = dict(cost_intercept=.259, input_token_cost=.000758, output_token_cost=.0052083333)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def configuration(document, run, output, protocol=None):
    declared = document['schedules'][run['schedule']]
    fixed = 'fixed_budget' in declared
    return dict(document['replay'], trace=str(ROOT / document['traces'][run['trace']]['path']),
                output_dir=str(output), budget=float(declared['fixed_budget'] if fixed else read(ROOT / declared['path'])[0]['budget']),
                budget_schedule=None if fixed else str(ROOT / declared['path']),
                vllm_url=BACKEND, protocol=str(protocol or PROTOCOL), protocol_run=run['name'])


def command(config):
    argv = [sys.executable, '-u', str(ROOT / 'run_trace_pilot.py')]
    for key, value in config.items():
        if key == 'budget' and config.get('budget_schedule') is not None:
            continue  # The child derives this normalized initial value from the schedule.
        if value is True:
            argv.append('--' + key.replace('_', '-'))
        elif value is not None and value is not False:
            argv.extend(['--' + key.replace('_', '-'), str(value)])
    return argv


def plan(document, output, stage, protocol=None):
    protocol = Path(protocol or PROTOCOL)
    if document.get('status') != 'frozen' or document.get('protocol') not in ('cap-range-v1', 'low-cap-switch-v1'):
        raise ValueError('Use a supported frozen collection declaration')
    by_name = {run['name']: run for run in document['runs']}
    order = document['qualification']['order']
    if set(order) != set(by_name) or len(order) != len(by_name):
        raise ValueError('Qualification order must contain every declared run exactly once')
    selected = order[:1] if stage == 'smoke' else order
    jobs = []
    for name in selected:
        run = by_name[name]
        if run['role'] != 'development':
            raise ValueError('Only development runs belong to this suite')
        if document['protocol'] == 'cap-range-v1' and 'fixed_budget' not in document['schedules'][run['schedule']]:
            raise ValueError('Only fixed-cap development runs belong to cap-range-v1')
        config = configuration(document, run, output / name, protocol)
        trace = load_trace(config['trace'])
        _, schedule = expected_schedule(document['schedules'][run['schedule']], trace['duration_seconds'])
        # Validate the exact same identity contract as the child, before collecting anything.
        collection_binding(protocol, name, output / name, config, trace, sha256_file(config['trace']), schedule)
        cap_label = config['budget'] if config['budget_schedule'] is None else [s['budget'] for s in schedule]
        jobs.append(dict(name=name, trace=run['trace'], cap=cap_label,
                         duration_s=trace['duration_seconds'], config=config, command=command(config)))
    return jobs


def cpu_acceptance(commit):
    from scripts.run_linux_cpu_tests import validate_report
    for path in sorted((ROOT / 'cpu-ci-artifacts').glob('*/acceptance.json'), reverse=True):
        record = read(path)
        if record.get('valid') is True and record.get('commit') == commit:
            coverage = validate_report(path.parent / 'junit.xml')
            if record.get('test_cases') == coverage['test_cases']:
                return dict(path=str(path), acceptance=record,
                            acceptance_sha256=sha256_file(path), junit_sha256=sha256_file(path.parent / 'junit.xml'))
    raise RuntimeError('No native CPU acceptance for this exact commit; run scripts/run_linux_cpu_tests.py first')


def backend_identity():
    """A local listener and engine process identity; no credentials or full process argv are retained."""
    import httpx
    import psutil
    if platform.system() != 'Linux':
        raise RuntimeError('Collection requires the Linux rental host')
    listeners = {item.pid for item in psutil.net_connections(kind='tcp')
                 if item.status == psutil.CONN_LISTEN and item.laddr.port == 8000}
    if len(listeners) != 1 or None in listeners:
        raise RuntimeError('Cannot identify one local backend listener on port 8000')
    process = psutil.Process(listeners.pop())
    argv = process.cmdline()
    model = 'Qwen/Qwen3-4B-Instruct-2507'
    if (model not in argv or '--no-enable-prefix-caching' not in argv or
            '--max-model-len' not in argv or argv[argv.index('--max-model-len') + 1] != '8192'):
        raise RuntimeError('Backend must use Qwen3-4B, explicit context 8192 and --no-enable-prefix-caching')
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(BACKEND + '/v1/models')
        response.raise_for_status()
        models = response.json()['data']
        if not any(item.get('id') == model and item.get('max_model_len') == 8192 for item in models):
            raise RuntimeError('Served model/context differ from the launch')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                   '--format=csv,noheader'], text=True, timeout=15).strip()
    # Linux comm may truncate VLLM::EngineCore to VLLM::EngineCor (15 bytes).
    engines = sorted((child.pid, child.create_time()) for child in process.children(recursive=True)
                     if 'EngineCor' in child.name())
    if not engines:
        raise RuntimeError('Cannot identify the local vLLM EngineCore process')
    return dict(host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                gpu=gpu, backend_pid=process.pid, backend_created=process.create_time(), engines=engines,
                model=model, max_model_len=8192, prefix_caching=False)


def idle_counts(text):
    result = {}
    for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
        values = [float(match.group(1)) for line in text.splitlines()
                  if (match := re.fullmatch(re.escape(name) + r'(?:\{.*\})?\s+([^\s]+)(?:\s+\S+)?', line))]
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            raise RuntimeError(f'Missing or invalid backend gauge: {name}')
        result[name] = sum(values)
    return result


def wait_idle(timeout=120):
    """Three consecutive idle samples, outside measurement. Never terminate other workloads."""
    import httpx
    deadline, consecutive = time.monotonic() + timeout, 0
    with httpx.Client(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            response = client.get(BACKEND + '/metrics')
            response.raise_for_status()
            counts = idle_counts(response.text)
            consecutive = consecutive + 1 if not any(counts.values()) else 0
            if consecutive == 3:
                return dict(sampled_at=time.monotonic(), counts=counts, consecutive=consecutive)
            time.sleep(.5)
    raise RuntimeError('Backend did not drain; stop collection and inspect other traffic')


def gate_issues(report):
    issues = [] if report.get('valid') is True else ['data integrity']
    return issues + [name for name in GATES if report.get(name, {}).get('valid') is not True]


def qualify_capture(path, document, protocol_sha, name, commit, expected_code=None):
    """Check identity as well as the six gates. Saved reports are re-audited by the analyzer."""
    manifest, trace, replay = read(path / 'manifest.json'), read(path / 'trace.json'), read(path / 'replay_plan.json')
    config = manifest['config']
    capture = capture_identity(config, trace, sha256_file(path / 'trace.json'), replay,
                               manifest.get('experiment_protocol'))
    issues = capture_mismatches(document, name, capture)
    recorded = manifest.get('experiment_protocol', {})
    if (recorded.get('sha256') != protocol_sha or recorded.get('run') != name or
            recorded.get('role') != 'development' or manifest.get('git_commit') != commit or
            manifest.get('status') != 'collected'):
        issues.append('Capture revision/status/protocol binding differs from this suite')
    code = manifest.get('code_sha256')
    if not isinstance(code, dict) or not code or (expected_code is not None and code != expected_code):
        issues.append('Runtime source hashes are missing or differ across runs')
    elif any(sha256_file(ROOT / filename) != digest for filename, digest in code.items()):
        issues.append('Recorded runtime sources differ from this checkout')
    for key in ('initial_metrics', 'post_warmup_metrics', 'final_metrics'):
        metrics = manifest.get(key, {})
        if any(metrics.get(field) != value for field, value in COST.items()):
            issues.append(f'{key}: cost coefficients differ from V1')
        if (metrics.get('current_in_flight') != 0 or abs(metrics.get('current_admitted_cost', math.inf)) > 1e-6
                or metrics.get('tokenizer_fallback_requests') != 0 or metrics.get('shared_upstream_client') is not True):
            issues.append(f'{key}: requests did not drain or admission path differs')
    report = read(path / 'report.json')
    issues.extend(gate_issues(report))
    if issues:
        raise RuntimeError('; '.join(issues))
    return manifest, report


def run_suite(output, stage, protocol=None):
    protocol = Path(protocol or PROTOCOL)
    document, protocol_sha = load_protocol(protocol)
    jobs = plan(document, output, stage, protocol)
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}; no automatic resume/retry')
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Tracked source must be clean')
    acceptance = cpu_acceptance(commit)
    reference = backend_identity()
    output.mkdir(parents=True)
    evidence = dict(schema_version=1, protocol=document['protocol'], protocol_sha256=protocol_sha,
                    stage=stage, commit=commit, status='running', valid=False, backend=reference,
                    cpu_acceptance=acceptance, planned=[job['name'] for job in jobs], runs=[],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    note=document.get('purpose', 'Development collection, not model acceptance or MPC benefit.'))
    save(output / 'suite.json', evidence)
    save(output / 'protocol.json', document)
    child, expected_code, expected_packages = None, None, None
    try:
        for index, job in enumerate(jobs):
            if (git('rev-parse', 'HEAD') != commit or git('status', '--porcelain', '--untracked-files=no') or
                    normalized_sha256(protocol) != protocol_sha or backend_identity() != reference):
                raise RuntimeError('Checkout, protocol, host or backend changed; stop this series')
            idle = wait_idle()
            record = dict(name=job['name'], cap=job['cap'], status='running', preflight_idle=idle,
                          backend=reference,
                          command=job['command'], started_at=datetime.now(timezone.utc).isoformat())
            evidence['runs'].append(record)
            save(output / 'suite.json', evidence)
            print(f'[{index + 1}/{len(jobs)}] {job["name"]}: cap={job["cap"]}, {job["duration_s"]:g}s', flush=True)
            with (output / (job['name'] + '.log')).open('w') as log:
                # The child owns its gateway cleanup. The supervisor never kills a shared backend.
                env = dict(os.environ, COST_INTERCEPT=str(COST['cost_intercept']),
                           INPUT_TOKEN_COST=str(COST['input_token_cost']),
                           OUTPUT_TOKEN_COST=str(COST['output_token_cost']), MIN_REQUEST_COST='0.1')
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                child = None
            record['exitcode'] = code
            if code:
                raise RuntimeError(f'{job["name"]} exited {code}; inspect its log/report; no later runs started')
            manifest, report = qualify_capture(output / job['name'], document, protocol_sha,
                                                job['name'], commit, expected_code)
            expected_code = manifest['code_sha256']
            packages = manifest.get('packages')
            if not packages or (expected_packages is not None and packages != expected_packages):
                raise RuntimeError('Recorded runtime package versions are missing or changed across runs')
            expected_packages = packages
            evidence['packages'] = packages
            if (backend_identity() != reference or git('rev-parse', 'HEAD') != commit or
                    git('status', '--porcelain', '--untracked-files=no') or normalized_sha256(protocol) != protocol_sha):
                raise RuntimeError('Backend, checkout or protocol changed during capture')
            record.update(status='passed', performance=report['performance'], postflight_idle=wait_idle(),
                          artifact_sha256={p.name: sha256_file(p) for p in (output / job['name']).iterdir() if p.is_file()})
            save(output / 'suite.json', evidence)
            print(f'PASS: {job["name"]}', flush=True)
        evidence.update(status='collected', valid=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=150)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if evidence['runs'] and evidence['runs'][-1]['status'] == 'running':
            evidence['runs'][-1]['status'] = 'failed'
        evidence.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        evidence['ended_at'] = datetime.now(timezone.utc).isoformat()
        save(output / 'suite.json', evidence)
    print(f'Suite complete: {output / "suite.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stage', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--plan-only', action='store_true', help='Validate/print commands without a server or output writes')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL, help='Frozen cap-range or low-cap-switch declaration')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if args.plan_only:
        document, _ = load_protocol(args.protocol)
        print(json.dumps(plan(document, output, args.stage, args.protocol), indent=2))
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        run_suite(output, args.stage, args.protocol)


if __name__ == '__main__':
    main()
