"""Offline reproduction from the repository's original-file evidence bundle."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/state-model-20261021'


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--work-dir', required=True, type=Path)
    args = p.parse_args(); work = args.work_dir.resolve()
    if work.exists():
        p.error('Choose a fresh work directory')
    work.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('evidence', ARCHIVE / 'extract_evidence.py')
    evidence = importlib.util.module_from_spec(spec); spec.loader.exec_module(evidence)
    verification = evidence.extract(work / 'evidence')
    subprocess.run([sys.executable, str(ROOT / 'iterate_state_model.py'), '--suite-dir',
        str(work / 'evidence/results/low-cap-switch-v1'), '--output-dir', str(work / 'scores')], cwd=ROOT, check=True)
    for name in ('summary.json', 'diagnosis.json', 'full-report.json.gz'):
        actual, expected = (work / 'scores' / name).read_bytes(), (ARCHIVE / name).read_bytes()
        if actual != expected:
            raise RuntimeError('Reproduction differs: ' + name)
    verification['numerical_artifacts_byte_identical'] = True
    (work / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
