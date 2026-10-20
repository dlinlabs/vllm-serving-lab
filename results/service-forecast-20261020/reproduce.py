"""Restore verified derived tables and reproduce development scores, without GPU.

Optional --cap-root and --shared-root rebuild from the complete raw archives.
The default uses bundled tables whose original raw audits and hashes are recorded.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
AUDIT_SOURCES=('transition_features.py','identification_features.py','analyze_trace_dynamics.py','analyze_trace_pilot.py','trace_control.py')


def digest(data):return hashlib.sha256(data).hexdigest()


def close(a,b):
    if type(a) is not type(b):raise ValueError('Reproduction type mismatch')
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Reproduction fields mismatch')
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Reproduction length mismatch')
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):
        if not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Reproduction numeric mismatch')
    elif a!=b:raise ValueError('Reproduction value mismatch')


def restore(destination):
    if destination.exists():raise ValueError('Choose a fresh dataset destination')
    summary=json.loads((HERE/'summary.json').read_text())
    data=(HERE/'dataset.json.gz').read_bytes()
    if digest(data)!=summary['dataset_pack_sha256']:raise ValueError('Derived-table archive fingerprint mismatch')
    pack=json.loads(gzip.decompress(data))
    if pack.get('format')!='audited-service-tables-v2':raise ValueError('Unsupported cached table format')
    if digest(pack['manifest_text'].encode())!=summary['dataset_sha256']:raise ValueError('Dataset manifest fingerprint mismatch')
    manifest=json.loads(pack['manifest_text'])
    names=[entry['file'] for entry in manifest['runs']]
    if len(set(names))!=len(names) or set(names)!=set(pack['tables']):raise ValueError('Derived-table inventory mismatch')
    for entry in manifest['runs']:
        name=entry['file']
        if Path(name).name!=name or name!=entry['meta']['name']+'.jsonl':raise ValueError('Unsafe derived-table name')
        if digest(pack['tables'][name].encode())!=entry['sha256']:raise ValueError('Derived-table fingerprint mismatch')
    # Validate the complete pack before writing any table.
    destination.mkdir(parents=True)
    (destination/'dataset.json').write_text(pack['manifest_text'])
    for name,text in pack['tables'].items():(destination/name).write_text(text)


def historical_sources(destination, checksums, audit_checksums=None):
    """Use hash-verified archived code when the current implementation has advanced."""
    if set(checksums) != {'service_features.py', 'service_forecasts.py', 'transition_baselines.py', 'forecast_baselines.py'}:
        raise ValueError('Unexpected historical source inventory')
    sources=json.loads(gzip.decompress((HERE/'source-code.json.gz').read_bytes()))
    if set(sources)!=set(checksums):raise ValueError('Historical source inventory mismatch')
    for name,text in sources.items():
        if not isinstance(text,str) or digest(text.encode())!=checksums[name]:
            raise ValueError('Historical source fingerprint mismatch: '+name)
    if audit_checksums is not None:
        if set(audit_checksums)!=set(AUDIT_SOURCES):raise ValueError('Historical audit source inventory mismatch')
        for name,checksum in audit_checksums.items():
            text=(REPO/name).read_text()
            if digest(text.encode())!=checksum:raise ValueError('Historical audit dependency changed: '+name)
            sources[name]=text
    destination.mkdir()
    for name,text in sources.items():(destination/name).write_text(text)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--cap-root',type=Path)
    parser.add_argument('--shared-root',type=Path)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output.exists():parser.error('Choose a fresh output directory')
    if bool(args.cap_root)!=bool(args.shared_root):parser.error('Provide both raw archive roots or neither')
    summary=json.loads((HERE/'summary.json').read_text())
    compressed=(HERE/'full-report.json.gz').read_bytes();raw=gzip.decompress(compressed)
    if digest(compressed)!=summary['compressed_report_sha256'] or digest(raw)!=summary['full_report_sha256']:
        raise ValueError('Archived forecast report fingerprint mismatch')
    expected=json.loads(raw)
    packed=(HERE/'dataset.json.gz').read_bytes()
    if digest(packed)!=summary['dataset_pack_sha256']:raise ValueError('Derived-table archive fingerprint mismatch')
    manifest_text=json.loads(gzip.decompress(packed))['manifest_text']
    if digest(manifest_text.encode())!=summary['dataset_sha256']:raise ValueError('Dataset manifest fingerprint mismatch')
    extraction=json.loads(manifest_text)['code_sha256']
    roots=[]
    if args.cap_root:
        roots=[args.shared_root.resolve()/archive/'results'/name for archive,name in
               (('shared-a','shared-client-a-r1'),('shared-b','shared-client-b-r1'))]
        roots += [args.cap_root.resolve()/'results/cap-range-v1'/('cap-range-v1-fixed-'+suffix)
                  for suffix in ('16-r1','32-r1','32-r2','64-r1','96-r1','96-r2')]
        if any(root==output or root in output.parents for root in roots):parser.error('Output must be outside raw captures')
    # Materialize all four original modules even for raw re-audits. Additional
    # audit dependencies are imported from the repository, not the snapshot.
    with tempfile.TemporaryDirectory(prefix='service-v2-sources-') as temporary:
        scoring=historical_sources(Path(temporary)/'sources',expected['code_sha256'],
                                   {name:extraction[name] for name in AUDIT_SOURCES})
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None,(str(REPO),os.environ.get('PYTHONPATH')))))
        output.mkdir(parents=True)
        if roots:
            subprocess.run([sys.executable,str(scoring/'service_features.py'),*map(str,roots),'--output-dir',str(output/'dataset')],
                           check=True,cwd=REPO,env=env)
        else:restore(output/'dataset')
        subprocess.run([sys.executable,str(scoring/'service_forecasts.py'),str(output/'dataset'),'--output-dir',str(output/'scores')],
                       check=True,cwd=REPO,env=env)
    actual=json.loads((output/'scores/service_report.json').read_text())
    close(expected['source_runs'],actual['source_runs'])
    close(expected['scores'],actual['scores'])
    close(expected['cap_responses'],actual['cap_responses'])
    (output/'verification.json').write_text(json.dumps(dict(scores_match=True,cap_diagnostics_match=True,
        raw_reaudited_this_invocation=bool(roots),c_used=False,acceptance=None),indent=2)+'\n')
    print('Forecast scores and cap-response diagnostics reproduce; C untouched.')


if __name__=='__main__':main()
