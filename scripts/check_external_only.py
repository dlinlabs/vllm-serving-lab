#!/usr/bin/env python3
"""Run the repository's CPU subprocess scenario with external-only observations.

Preserve each valid capture before the scenario intentionally corrupts its
temporary originals. No PID remapping, serving changes or audit bypasses.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def wait_for_ports(timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        try:
            for port in (8000, 8080):
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(('127.0.0.1', port))
            return
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError('8000/8080 are occupied; stop your vLLM/gateway first') from exc
            time.sleep(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if not (root / 'tests/test_trace_replay.py').is_file():
        parser.error('Run from vllm-serving-lab or provide --repo-root')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    output = (args.output_dir or root.parent / f'external-only-validation-{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', captures=[], repo_root=str(root),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        sys.path[:0] = [str(root), str(root / 'tests')]
        import os
        from pytest import MonkeyPatch, skip
        from diagnostic_sampling import ObservationSampler
        from test_trace_replay import test_full_trace_runner_and_adversarial_analysis
        own = ObservationSampler.process_resources(os.getpid())
        if not own['available']:
            raise RuntimeError(f'Native process-resource identity unavailable: {own}')
        wait_for_ports()
        status['repo_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        original_run = subprocess.run

        def external_run(command, **kwargs):
            command = list(command)
            replay = 'run_trace_pilot.py' in command
            if replay:
                command.append('--engine-observations')
            result = original_run(command, **kwargs)
            if replay:
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                capture = Path(command[command.index('--output-dir') + 1])
                report = json.loads((capture / 'report.json').read_text())
                verdict = report['diagnostic_integrity']
                if not (report['valid'] and report['load_fidelity']['valid'] and verdict['valid']
                        and verdict['gateway_mode'] == 'disabled'
                        and verdict['external_mode'] == 'enabled'
                        and not verdict['gc_coverage_complete']
                        and (capture / 'engine_observations.jsonl').stat().st_size > 0):
                    raise RuntimeError(f'External-only qualification failed: {report}')
                shutil.copytree(capture, output / capture.name)
                status['captures'].append(capture.name)
                print(f'PASS: {capture.name}', flush=True)
            return result

        with tempfile.TemporaryDirectory(prefix='external-check-') as scratch:
            with MonkeyPatch.context() as patch:
                patch.setattr(subprocess, 'run', external_run)
                try:
                    test_full_trace_runner_and_adversarial_analysis(Path(scratch), False)
                except skip.Exception as exc:
                    raise RuntimeError(f'Integration skipped; this is not a pass: {exc}') from exc
        if len(status['captures']) != 3 or set(status['captures']) != {'success', 'rejected', 'failed'}:
            raise RuntimeError('Missing or duplicate capture cases')
        status['status'] = 'passed'
        print('external-only: PASS', flush=True)
        print(f'Artifacts: {output}', flush=True)
        return 0
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        print(f'external-only: FAIL: {exc}', file=sys.stderr, flush=True)
        print(f'Artifacts: {output}', file=sys.stderr, flush=True)
        return 1
    finally:
        (output / 'validation_status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
