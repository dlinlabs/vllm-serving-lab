"""Re-audit original low-cap evidence and reproduce the frozen-model report exactly."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a fresh output directory')
    summary = json.loads((HERE / 'summary.json').read_text())
    archived = gzip.decompress((HERE / 'full-report.json.gz').read_bytes())
    if hashlib.sha256(archived).hexdigest() != summary['full_report_sha256']:
        raise ValueError('Archived report fingerprint differs')
    subprocess.run([sys.executable, '-u', str(REPO / 'evaluate_low_cap_frozen.py'),
                    '--suite-dir', str(args.suite_dir.resolve()),
                    '--output-dir', str(args.output_dir.resolve())], cwd=REPO, check=True)
    actual = gzip.decompress((args.output_dir / 'full-report.json.gz').read_bytes())
    if actual != archived:
        raise ValueError('Frozen evaluation differs; do not replace the expected report')
    (args.output_dir / 'verification.json').write_text(json.dumps(dict(
        all_capture_gates_reaudited=True, full_report_byte_identical=True,
        full_report_sha256=summary['full_report_sha256']), indent=2) + '\n')
    print('PASS: all capture gates re-audited; every frozen forecast reproduces byte-for-byte.')


if __name__ == '__main__':
    main()
