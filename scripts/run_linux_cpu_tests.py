"""Run complete native CPU acceptance without launching a GPU backend."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TRACE_CASES = {
    f'test_full_trace_runner_and_adversarial_analysis[{diagnostics}-{external}-{mode}]'
    for diagnostics, external, mode in (
        (False, False, 'thread'), (True, True, 'thread'), (True, True, 'process'),
        (False, True, 'thread'), (False, True, 'process'))
}


def validate_report(path):
    cases = ET.parse(path).findall('.//testcase')
    unsuccessful = [case.get('name') for case in cases
                    if any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped'))]
    if unsuccessful:
        raise RuntimeError(f'CPU acceptance has failed, errored or skipped cases: {unsuccessful}')
    found = Counter(case.get('name') for case in cases
                    if case.get('classname', '').split('.')[-1] == 'test_trace_replay'
                    and case.get('name', '').startswith('test_full_trace_runner_and_adversarial_analysis['))
    if found != Counter(TRACE_CASES):
        raise RuntimeError(f'Expected each of the five trace configurations exactly once; found {dict(found)}')
    return dict(test_cases=len(cases), trace_configurations=sorted(found))


def git(*arguments):
    return subprocess.run(['git', *arguments], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    output = ROOT / 'cpu-ci-artifacts' / stamp
    output.mkdir(parents=True)

    def record(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    info = dict(valid=False, python=sys.version, platform=platform.platform())
    record('acceptance.json', info)
    print(f'CPU acceptance evidence: {output}', flush=True)
    if (platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64')
            or platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 12)):
        raise RuntimeError('Acceptance requires Linux x86_64 and CPython 3.12; no GPU is required')
    info['commit'] = git('rev-parse', 'HEAD')
    record('acceptance.json', info)
    if git('status', '--porcelain'):
        raise RuntimeError('Use a clean checkout of the exact revision being validated')
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        if os.environ.get(name):
            raise RuntimeError(f'Unset {name}; acceptance must not inherit test filters or extra plugins')

    sys.path.insert(0, str(ROOT))
    from diagnostic_sampling import ObservationSampler
    from scripts.setup_v1 import check_port
    info['process_resources'] = ObservationSampler.process_resources(os.getpid())
    record('acceptance.json', info)
    if not info['process_resources']['available']:
        raise RuntimeError(f'Native PID/procfs resource preflight failed: {info["process_resources"]}')
    for port in (8000, 8080):
        try:
            check_port(port)
        except OSError as exc:
            raise RuntimeError(f'Port {port} is unavailable; run acceptance before starting backend/gateway services') from exc

    packages = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json',
                               '--disable-pip-version-check'], cwd=ROOT, check=True,
                              capture_output=True, text=True)
    record('packages.json', json.loads(packages.stdout))
    report = output / 'junit.xml'
    command = [sys.executable, '-m', 'pytest', '-q', '-ra', 'tests', '-o', 'addopts=',
               '--junitxml', str(report)]
    info['command'] = command
    record('acceptance.json', info)
    env = dict(os.environ, PYTHONUTF8='1', MPLBACKEND='Agg', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               TRACE_BASELINE_ROOT=str(ROOT / 'results'))
    print('Running the complete CPU/mock suite without selection filters...', flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    (output / 'pytest.log').write_text(result.stdout, encoding='utf-8')
    print(result.stdout, end='', flush=True)
    info['pytest_exitcode'] = result.returncode
    record('acceptance.json', info)
    if result.returncode:
        return result.returncode
    coverage = validate_report(report)
    if git('rev-parse', 'HEAD') != info['commit'] or git('status', '--porcelain'):
        raise RuntimeError('The checkout changed during validation; this is not evidence for one clean revision')
    info.update(coverage, valid=True)
    record('acceptance.json', info)
    print(f'Native CPU acceptance passed; retain {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
