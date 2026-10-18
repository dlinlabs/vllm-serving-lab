"""Rebuild qualified A/B features and development forecasts from verified raw captures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path,
                        help='Output of extract_bundle.py, containing shared-a and shared-b')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--code-root', type=Path, default=HERE.parents[1],
                        help='Checkout with runtime/analysis code matching the recorded hashes')
    parser.add_argument('--extended', action='store_true', help='Also run the exploratory A/B diagnostics')
    args = parser.parse_args()
    raw, output, code = args.raw_root.resolve(), args.output_dir.resolve(), args.code_root.resolve()
    if output.exists():
        parser.error('Use a fresh output directory')
    if any(output == root or root in output.parents for root in (raw, code)):
        parser.error('Keep analysis output outside the raw and code trees')
    captures = [raw / f'shared-{side}' / 'results' / f'shared-client-{side}-r1' for side in ('a', 'b')]
    for capture in captures:
        manifest = json.loads((capture / 'manifest.json').read_text())
        for name, expected in manifest['code_sha256'].items():
            path = code / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                parser.error(f'{name} differs from captured code; use --code-root at d0ccb2c')
    output.mkdir(parents=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg',
               SHARED_CLIENT_RAW_ROOT=str(raw), SHARED_CLIENT_ANALYSIS_OUTPUT=str(output),
               SHARED_CLIENT_CODE_ROOT=str(code))

    def run(script, *arguments):
        subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, cwd=code, env=env)

    run(code / 'identification_features.py', *captures, '--output-dir', output / 'features')
    run(code / 'forecast_baselines.py', output / 'features', '--output-dir', output / 'forecast')
    if args.extended:
        for name in ('diagnose_models.py', 'version_ablation.py', 'nonlinear_diagnostic.py', 'interaction_diagnostic.py'):
            run(HERE / 'analysis' / name)
    print(json.dumps(dict(output=str(output), C_used=False, development_only=True)))


if __name__ == '__main__':
    main()
