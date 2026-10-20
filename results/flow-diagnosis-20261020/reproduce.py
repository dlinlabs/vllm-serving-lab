"""Restore existing audited tables and reproduce diagnostic oracles, without GPU."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output.exists():parser.error('Choose a fresh output')
    summary=json.loads((HERE/'summary.json').read_text())
    compressed=(HERE/'full-report.json.gz').read_bytes();raw=gzip.decompress(compressed)
    if hashlib.sha256(compressed).hexdigest()!=summary['compressed_report_sha256'] or hashlib.sha256(raw).hexdigest()!=summary['full_report_sha256']:
        raise ValueError('Archived diagnosis fingerprint mismatch')
    expected=json.loads(raw)
    for name,checksum in expected['code_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Use the exact diagnostic source revision: '+name)
    archive=REPO/'results/transition-prototype-20261020/full-report.json.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected['archived_report_sha256']:
        raise ValueError('Original forecast archive changed')
    spec=importlib.util.spec_from_file_location('restore_service_tables',REPO/'results/service-forecast-20261020/reproduce.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.HERE=REPO/'results/service-review-20261020'
    helper.restore(output/'dataset')
    subprocess.run([sys.executable,str(REPO/'diagnose_flow_rollout.py'),str(output/'dataset'),'--archived-report',str(archive),'--output-dir',str(output/'scores')],check=True,cwd=REPO)
    actual=json.loads((output/'scores/diagnosis.json').read_text())
    for field in ('scores','predictions','observed_input_flow_residual_correlations','one_step_flow_errors_with_observed_inputs','origin_only_repeat_goodput'):
        helper.close(expected[field],actual[field])
    (output/'verification.json').write_text(json.dumps(dict(diagnostic_scores_match=True,predictions_match=True,
        archived_coefficients_match=True,original_rollouts_match=True,raw_reaudited_this_invocation=False,
        deployable_improvement_claim=False,c_used=False,acceptance=None),indent=2)+'\n')
    print('Frozen-model diagnostic scores and predictions reproduce; no deployable oracle claim.')


if __name__=='__main__':main()
