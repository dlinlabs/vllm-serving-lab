"""Offline reproduction of results/cap-miss-causes-20261021 from both public evidence bundles.

The old-host captures come from results/cap-range-20261020/raw-bundle and the new-host
captures from results/state-model-20261021/raw-bundle. Either may instead be passed as an
already restored directory; every file listed by that bundle is then hash-checked before
use. (The cap-range extractor reopens a named temporary file, which Windows does not allow,
so Windows users restore it once on Linux/WSL or with tar and pass --old-evidence.)
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/cap-miss-causes-20261021'
OLD = ROOT / 'results/cap-range-20261020'
NEW = ROOT / 'results/state-model-20261021'
OLD_SUITE, NEW_SUITE = 'results/cap-range-v1', 'results/low-cap-switch-v1'
# Summary values are rounded to six significant digits; least-squares results may differ
# in the last digit across BLAS builds.
REL_TOL, ABS_TOL = 1e-4, 1e-14


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def old_files():
    manifest = json.loads((OLD / 'raw-bundle/manifest.json').read_text())
    return {name: (f['bytes'], f['sha256']) for name, f in manifest['files'].items()}


def new_files():
    index = json.loads((NEW / 'raw-bundle/index.json').read_text())
    return {f['path']: (f['bytes'], f['sha256']) for f in index['files']}


def verify(root, files):
    """Every bundle file must exist under root with the archived size and SHA-256."""
    for name, (size, sha) in files.items():
        path = Path(root) / name
        if not path.is_file() or path.stat().st_size != size or digest(path) != sha:
            raise ValueError('Evidence file differs: ' + name)
    return len(files)


def restore(given, fresh, files, extractor):
    if given is None:
        load_module(extractor).extract(fresh)
        given = fresh
    root = Path(given).resolve()
    return root, verify(root, files)


def differences(actual, expected, path='$'):
    """JSON paths where actual differs; floats compare with REL_TOL/ABS_TOL, all else exactly."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            return [path]
        return [d for k in expected for d in differences(actual[k], expected[k], f'{path}.{k}')]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path]
        return [d for i, (a, e) in enumerate(zip(actual, expected)) for d in differences(a, e, f'{path}[{i}]')]
    if isinstance(expected, float) and type(actual) is float:
        return [] if math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL) else [path]
    return [] if type(actual) is type(expected) and actual == expected else [path]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--old-evidence', type=Path, help=f'restored cap-range bundle root (contains {OLD_SUITE})')
    p.add_argument('--new-evidence', type=Path, help=f'restored state-model bundle root (contains {NEW_SUITE})')
    args = p.parse_args(argv)
    work = args.work_dir.resolve()
    if work.exists():
        p.error('Choose a fresh work directory')
    work.mkdir(parents=True)
    old, old_n = restore(args.old_evidence, work / 'old-evidence', old_files(), OLD / 'extract_bundle.py')
    new, new_n = restore(args.new_evidence, work / 'new-evidence', new_files(), NEW / 'extract_evidence.py')
    output = work / 'summary.json'
    subprocess.run([sys.executable, str(ROOT / 'diagnose_cap_misses.py'), '--host', f'old={old / OLD_SUITE}',
                    '--host', f'new={new / NEW_SUITE}', '--output', str(output)], cwd=ROOT, check=True)
    expected = (ARCHIVE / 'summary.json').read_bytes().replace(b'\r\n', b'\n')
    diff = differences(json.loads(output.read_bytes()), json.loads(expected))
    verification = dict(
        old_evidence_files_verified=old_n, new_evidence_files_verified=new_n,
        summary_byte_identical=output.read_bytes() == expected,
        float_tolerance=dict(relative=REL_TOL, absolute=ABS_TOL),
        differing_paths=diff[:50], reproduced=not diff)
    (work / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', newline='\n')
    print(json.dumps(verification, indent=2))
    if diff:
        raise SystemExit(f'Reproduction differs at {len(diff)} paths, first: {diff[0]}')


if __name__ == '__main__':
    main()
