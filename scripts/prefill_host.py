"""Short rental-host commands: prepare pinned inputs, then explicitly collect."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import save
from scripts.run_prefill_service import code_identity,cpu_check,prepare
from trace_workload import MODEL,load_trace,verify_prompts


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def accepted(path,sha):
    if not path.is_file():return False
    record=json.loads(path.read_text())
    return bool(record.get('valid') and record.get('commit')==sha and record.get('source_sha256')==code_identity())


def prepare_host(output):
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoTokenizer
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Use a clean tracked checkout before preparing the host')
    output.mkdir(parents=True,exist_ok=True)
    pin=output/'model.json'
    if pin.exists():spec=json.loads(pin.read_text())
    else:
        spec=dict(model=MODEL,revision=HfApi().model_info(MODEL).sha)
    if spec.get('model')!=MODEL or not re.fullmatch('[0-9a-f]{40}',spec.get('revision','')):
        raise ValueError('Invalid existing model pin; use a separate output directory for another model/revision')
    # Persist before download, so interrupted/repeated setup never follows a new HEAD.
    save(pin,spec)
    print('Downloading pinned model revision '+spec['revision'],flush=True)
    snapshot=snapshot_download(repo_id=MODEL,revision=spec['revision'],max_workers=4)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    trace_path=ROOT/'results/trace-experiment-96-r1/trace.json'
    trace=load_trace(trace_path)
    print('Verifying Azure trace token counts...',flush=True)
    verify_prompts(trace,tokenizer)
    tokens=output/'tokens.json'
    prepare(SimpleNamespace(tokenizer=str(snapshot),model=MODEL,revision=spec['revision'],output=tokens))
    sha=commit();ready_path=output/'ready.json';previous={}
    if ready_path.exists():previous=json.loads(ready_path.read_text())
    acceptance=Path(previous.get('cpu_acceptance',str(output/'no-prior-acceptance.json')))
    if not accepted(acceptance,sha):
        acceptance=output/f'cpu-acceptance-{sha[:12]}-{time.time_ns()}'/'acceptance.json'
        print('Validating this exact checkout and final installed environment...',flush=True)
        if cpu_check(acceptance.parent)!=0:
            raise RuntimeError('CPU acceptance failed; evidence: '+str(acceptance.parent))
    if not accepted(acceptance,sha):raise RuntimeError('CPU acceptance does not match the current source tree')
    ready=dict(model=MODEL,revision=spec['revision'],tokenizer=str(snapshot),manifest=str(tokens),
               trace=str(trace_path),cpu_acceptance=str(acceptance),commit=sha,
               trials=len(json.loads(tokens.read_text())['trials']),verified_trace_requests=len(trace['requests']))
    save(ready_path,ready)
    print(json.dumps(ready,indent=2),flush=True)
    print('READY. Next: .venv/bin/python scripts/prefill_host.py collect',flush=True)
    return ready


def collect_host(output):
    ready=json.loads((output/'ready.json').read_text())
    if not accepted(Path(ready['cpu_acceptance']),commit()):
        raise RuntimeError('Run prefill_host.py prepare for this exact checkout before collection')
    capture=Path.home()/'pr32-captures'/str(time.time_ns())
    capture.parent.mkdir(parents=True,exist_ok=True)
    print('Capture output: '+str(capture),flush=True)
    args=[sys.executable,str(ROOT/'scripts/run_prefill_service.py'),'collect']
    for name in ('manifest','tokenizer','trace'):
        args.extend(['--'+name,ready[name]])
    args.extend(['--cpu-acceptance',ready['cpu_acceptance'],'--output',str(capture)])
    subprocess.run(args,cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','collect'))
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'pr32-inputs')
    args=parser.parse_args();output=args.output_dir.expanduser().resolve()
    if args.operation=='prepare':prepare_host(output)
    else:collect_host(output)


if __name__=='__main__':main()
