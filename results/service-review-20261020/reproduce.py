"""Verify archived tables and reproduce the combined offline development review."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def historical_helper():
    spec=importlib.util.spec_from_file_location('historical_service_reproducer',REPO/'results/service-forecast-20261020/reproduce.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE=HERE
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output.exists():parser.error('Choose a fresh output directory')
    summary=json.loads((HERE/'summary.json').read_text())
    compressed=(HERE/'full-report.json.gz').read_bytes()
    if hashlib.sha256(compressed).hexdigest()!=summary['compressed_report_sha256']:
        raise ValueError('Archived report fingerprint mismatch')
    raw=gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest()!=summary['full_report_sha256']:
        raise ValueError('Archived report content fingerprint mismatch')
    expected=json.loads(raw)
    for name,checksum in expected['code_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Use the exact reviewed scoring revision: '+name)
    helper=historical_helper()
    # Complete table/manifest verification occurs before the restored dataset is published.
    helper.restore(output/'dataset')
    subprocess.run([sys.executable,str(REPO/'service_forecasts.py'),str(output/'dataset'),'--output-dir',str(output/'scores')],check=True,cwd=REPO)
    actual=json.loads((output/'scores/service_report.json').read_text())
    for field in ('source_runs','scores','cap_responses','fitted'):
        helper.close(expected[field],actual[field])
    (output/'verification.json').write_text(json.dumps(dict(scores_match=True,cap_diagnostics_match=True,
        fitted_coefficients_match=True,raw_reaudited_this_invocation=False,c_used=False,acceptance=None),indent=2)+'\n')
    print('Reviewed scores, coefficients and cap diagnostics reproduce; C untouched.')


if __name__=='__main__':main()
