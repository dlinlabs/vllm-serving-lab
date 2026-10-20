"""Re-audit raw captures and reproduce the archived development scores without GPU."""
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
CAP_NAMES = ('16-r1', '32-r1', '32-r2', '64-r1', '96-r1', '96-r2')


def compare(expected, actual):
    if type(expected) is not type(actual):
        raise ValueError('Score type mismatch')
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError('Score fields differ')
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError('Score lengths differ')
        for a, b in zip(expected, actual):
            compare(a, b)
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'Score differs: {expected} != {actual}')
    elif expected != actual:
        raise ValueError('Score value differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cap-root', required=True, type=Path,
                        help='Extracted cap archive root containing results/cap-range-v1')
    parser.add_argument('--shared-root', required=True, type=Path,
                        help='Extracted shared archive root containing shared-a and shared-b')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Choose a fresh output directory')
    roots = [args.shared_root.resolve() / archive / 'results' / name
             for archive, name in (('shared-a', 'shared-client-a-r1'), ('shared-b', 'shared-client-b-r1'))]
    roots += [args.cap_root.resolve() / 'results/cap-range-v1' / ('cap-range-v1-fixed-' + suffix) for suffix in CAP_NAMES]
    if any(output == root or root in output.parents for root in roots):
        parser.error('Output must be outside captures')
    summary = json.loads((HERE / 'summary.json').read_text())
    compressed = (HERE / 'full-report.json.gz').read_bytes()
    raw = gzip.decompress(compressed)
    if (hashlib.sha256(raw).hexdigest() != summary['full_report_sha256'] or
            hashlib.sha256(compressed).hexdigest() != summary['compressed_report_sha256']):
        raise ValueError('Archived report fingerprint differs')
    expected = json.loads(raw)
    output.mkdir(parents=True)
    subprocess.run([sys.executable, str(REPO / 'transition_features.py'), *map(str, roots),
                    '--output-dir', str(output / 'dataset')], check=True, cwd=REPO)
    subprocess.run([sys.executable, str(REPO / 'transition_baselines.py'), str(output / 'dataset'),
                    '--output-dir', str(output / 'scores')], check=True, cwd=REPO)
    actual = json.loads((output / 'scores/transition_report.json').read_text())
    compare(expected['analysis_code_sha256'], actual['analysis_code_sha256'])
    # These include exact raw/table fingerprints and state reconstruction evidence.
    compare(expected['source_runs'], actual['source_runs'])
    compare(expected['scores'], actual['scores'])
    subprocess.run([sys.executable, str(HERE / 'summarize.py'), str(output / 'scores/transition_report.json'),
                    '--output-dir', str(output / 'publication')], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL)
    (output / 'verification.json').write_text(json.dumps(dict(
        raw_and_table_fingerprints_match=True, scores_match=True, relative_tolerance=1e-8,
        absolute_tolerance=1e-8, c_used=False, acceptance=None), indent=2) + '\n')
    print('Verified raw/table fingerprints and all development scores; C untouched.')


if __name__ == '__main__':
    main()
