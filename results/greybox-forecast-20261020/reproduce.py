"""Verify the archived grey-box report, re-extract both raw bundles and reproduce every number.

Run on Linux or macOS (the cap-range extractor reopens a named temporary file, which Windows does
not allow).
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(expected, actual, path='report', tol=1e-9):
    """Recursive comparison with a relative/absolute tolerance for numbers only."""
    number = (int, float)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(f'{path}: keys differ')
        for key in expected:
            close(expected[key], actual[key], f'{path}.{key}', tol)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f'{path}: lengths differ')
        for i, (e, a) in enumerate(zip(expected, actual)):
            close(e, a, f'{path}[{i}]', tol)
    elif (isinstance(expected, number) and isinstance(actual, number)
          and not isinstance(expected, bool) and not isinstance(actual, bool)):
        if not math.isclose(expected, actual, rel_tol=tol, abs_tol=tol):
            raise ValueError(f'{path}: {expected} != {actual}')
    elif expected != actual:
        raise ValueError(f'{path}: {expected!r} != {actual!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Choose a fresh output directory')
    summary = json.loads((HERE / 'summary.json').read_text())
    compressed = (HERE / 'full-report.json.gz').read_bytes()
    if digest(compressed) != summary['compressed_report_sha256']:
        raise ValueError('Archived report fingerprint mismatch')
    raw = gzip.decompress(compressed)
    if digest(raw) != summary['full_report_sha256']:
        raise ValueError('Archived report content fingerprint mismatch')
    expected = json.loads(raw)
    for name, checksum in expected['code_sha256'].items():
        if digest((REPO / name).read_bytes().replace(b'\r\n', b'\n')) != checksum:
            raise ValueError('Use the exact scoring revision: ' + name)

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], check=True, cwd=REPO)

    run(REPO / 'results/cap-range-20261020/extract_bundle.py', '--output-dir', output / 'cap-range')
    run(REPO / 'results/shared-client-20261018/extract_bundle.py', '--output-dir', output / 'shared-client',
        '--archives', 'shared-a', 'shared-b')
    p = expected['protocol']
    run(REPO / 'evaluate_greybox.py', '--cap-range-dir', output / 'cap-range',
        '--shared-client-dir', output / 'shared-client', '--output-dir', output / 'scores',
        '--boot-k', p['boot_k'], '--seed', p['seed'], '--reps', p['bootstrap_reps'], '--block', p['block_s'])
    actual = json.loads(gzip.decompress((output / 'scores/full-report.json.gz').read_bytes()))
    close(expected, actual)
    (output / 'verification.json').write_text(json.dumps(dict(
        raw_files_match_archive_hashes=True, report_matches=True, tolerance=1e-9,
        archived_full_report_sha256=summary['full_report_sha256']), indent=2) + '\n')
    print('Raw captures verified; every archived grey-box number reproduces.')


if __name__ == '__main__':
    main()
