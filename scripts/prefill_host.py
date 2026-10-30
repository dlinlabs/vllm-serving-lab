"""Short rental-host commands: prepare pinned inputs, then explicitly collect."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import tarfile
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


def prepare_host(output,*,envelope_shadow=False):
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
    if not envelope_shadow:
        trace_path=ROOT/'results/trace-experiment-96-r1/trace.json'
        trace=load_trace(trace_path)
        print('Verifying Azure trace token counts...',flush=True)
        verify_prompts(trace,tokenizer)
    tokens=output/('envelope-shadow-tokens.json' if envelope_shadow else 'tokens.json')
    prepare(SimpleNamespace(tokenizer=str(snapshot),model=MODEL,revision=spec['revision'],output=tokens,
                            envelope_shadow_tokens=envelope_shadow))
    sha=commit();ready_path=output/('shadow-ready.json' if envelope_shadow else 'ready.json');previous={}
    if ready_path.exists():previous=json.loads(ready_path.read_text())
    acceptance=Path(previous.get('cpu_acceptance',str(output/'no-prior-acceptance.json')))
    if not accepted(acceptance,sha):
        acceptance=output/f'cpu-acceptance-{sha[:12]}-{time.time_ns()}'/'acceptance.json'
        print('Validating this exact checkout and final installed environment...',flush=True)
        if cpu_check(acceptance.parent)!=0:
            raise RuntimeError('CPU acceptance failed; evidence: '+str(acceptance.parent))
    if not accepted(acceptance,sha):raise RuntimeError('CPU acceptance does not match the current source tree')
    ready=dict(model=MODEL,revision=spec['revision'],tokenizer=str(snapshot),manifest=str(tokens),
               cpu_acceptance=str(acceptance),commit=sha,trials=len(json.loads(tokens.read_text())['trials']))
    if envelope_shadow:ready['mode']='envelope-shadow'
    else:ready.update(trace=str(trace_path),verified_trace_requests=len(trace['requests']))
    save(ready_path,ready)
    print(json.dumps(ready,indent=2),flush=True)
    print('READY. Next: .venv/bin/python scripts/prefill_host.py '+
          ('envelope-shadow --support /path/to/support.json' if envelope_shadow else 'collect'),flush=True)
    return ready


def collect_host(output,diagnostics=False,*,stages=False,calibration=False,arrivals=False,event_clock=False,live_http=False,queue_probe=False,queue_confirmation=False,entry_http=False,envelope_shadow=False,support=None):
    if sum((diagnostics,stages,calibration,arrivals,event_clock,live_http,queue_probe,queue_confirmation,entry_http,envelope_shadow))>1:raise ValueError('Choose one host collection mode')
    if bool(envelope_shadow)!=bool(support):raise ValueError('Envelope shadow requires --support, and vice versa')
    if envelope_shadow:
        support=Path(support).expanduser().resolve()
        if not support.is_file():raise ValueError('Shadow support input must be an existing JSON file')
    ready=json.loads((output/('shadow-ready.json' if envelope_shadow else 'ready.json')).read_text())
    if not accepted(Path(ready['cpu_acceptance']),commit()):
        raise RuntimeError('Run prefill_host.py prepare for this exact checkout before collection')
    if envelope_shadow and ready.get('mode')!='envelope-shadow':raise ValueError('Use dedicated envelope-shadow preparation')
    manifest=ready['manifest']
    if calibration or arrivals or event_clock or live_http or queue_probe or queue_confirmation or entry_http:
        manifest=str(output/('entry-controlled-tokens.json' if entry_http else 'queue-confirmation-tokens.json' if queue_confirmation else 'queue-probe-tokens.json' if queue_probe else 'live-clock-tokens.json' if live_http else 'clock-tokens.json' if event_clock else 'arrival-tokens.json' if arrivals else 'calibration-tokens.json'))
        prepare(SimpleNamespace(tokenizer=ready['tokenizer'],model=ready['model'],revision=ready['revision'],
                                output=Path(manifest),calibration_only=calibration,arrivals_only=arrivals,event_clock_only=event_clock or live_http,queue_probe_only=queue_probe,queue_confirmation_only=queue_confirmation or entry_http))
        if live_http or entry_http:
            http_manifest=output/('entry-http-tokens.json' if entry_http else 'live-http-tokens.json')
            prepare(SimpleNamespace(tokenizer=ready['tokenizer'],model=ready['model'],revision=ready['revision'],
                                    output=http_manifest,live_http_tokens=not entry_http,entry_http_tokens=entry_http))
    capture=Path.home()/('pr32-envelope-shadows' if envelope_shadow else 'pr32-entry-http' if entry_http else 'pr32-queue-confirmations' if queue_confirmation else 'pr32-queue-probes' if queue_probe else 'pr32-live-http' if live_http else 'pr32-clocks' if event_clock else 'pr32-arrivals' if arrivals else 'pr32-calibrations' if calibration else 'pr32-stage-diagnostics' if stages else 'pr32-diagnostics' if diagnostics else 'pr32-captures')/str(time.time_ns())
    capture.parent.mkdir(parents=True,exist_ok=True)
    print('Capture output: '+str(capture),flush=True)
    args=[sys.executable,str(ROOT/'scripts/run_prefill_service.py'),'collect']
    for name in (('manifest','tokenizer') if envelope_shadow else ('manifest','tokenizer','trace')):
        args.extend(['--'+name,manifest if name=='manifest' else ready[name]])
    args.extend(['--cpu-acceptance',ready['cpu_acceptance'],'--output',str(capture)])
    if envelope_shadow:args.extend(['--envelope-shadow','--support',str(support)])
    elif entry_http:args.extend(['--entry-http-only','--http-manifest',str(http_manifest)])
    elif queue_confirmation:args.append('--queue-confirmation-only')
    elif queue_probe:args.append('--queue-probe-only')
    elif live_http:args.extend(['--live-http-only','--http-manifest',str(http_manifest)])
    elif event_clock:args.append('--event-clock-only')
    elif arrivals:args.append('--arrivals-only')
    elif calibration:args.append('--calibration-only')
    elif stages:args.append('--stage-diagnostics-only')
    elif diagnostics:args.append('--diagnostics-only')
    shadow_audit=[sys.executable,str(ROOT/'scripts/check_prefill_envelope_shadow.py'),'--capture',str(capture)]
    try:
        try:
            result=subprocess.run(args,cwd=ROOT,check=not (queue_confirmation or entry_http))
        except subprocess.CalledProcessError:
            partial_inputs=('shadow-summary.json','shadow-config.json','shadow-seal.json','runtime.json',
                            'gateway-start.json','gateway-final.json','gateway.jsonl','telemetry-status.json')
            if envelope_shadow and all((capture/name).is_file() for name in partial_inputs):
                try:subprocess.run(shadow_audit,cwd=ROOT,check=True,timeout=300)
                except (subprocess.SubprocessError,OSError) as exc:
                    save(capture/'shadow-audit-failure.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
        if envelope_shadow:
            subprocess.run(shadow_audit,cwd=ROOT,check=True,timeout=300)
        if queue_confirmation and (capture/'confirmation-report.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/check_prefill_queue_confirmation.py'),str(capture)],cwd=ROOT,check=True,timeout=300)
        if entry_http and (capture/'entry-http-report.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/check_prefill_entry_http.py'),str(capture)],cwd=ROOT,check=True,timeout=600)
        if queue_confirmation or entry_http:result.check_returncode()
        if queue_probe:
            subprocess.run([sys.executable,str(ROOT/'scripts/check_prefill_queue_probe.py'),str(capture)],cwd=ROOT,check=True,timeout=180)
        if live_http:
            subprocess.run([sys.executable,str(ROOT/'scripts/check_prefill_live_http.py'),str(capture)],cwd=ROOT,check=True,timeout=180)
    finally:
        if (calibration or arrivals or event_clock or live_http or queue_probe or queue_confirmation or entry_http or envelope_shadow) and capture.is_dir():
            archive=capture.with_suffix('.tar.gz')
            with tarfile.open(archive,'x:gz') as bundle:
                bundle.add(capture,arcname=capture.name)
            print('Download evidence: '+str(archive),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','collect','diagnose','stages','calibrate','arrivals','clock','live-http','queue-probe','queue-confirmation','entry-http','envelope-shadow'))
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'pr32-inputs')
    parser.add_argument('--envelope-shadow',action='store_true',help='Prepare independent shadow tokens without the Azure trace')
    parser.add_argument('--support',type=Path,help='Exact-runtime support or explicit shadow timing assumptions')
    args=parser.parse_args();output=args.output_dir.expanduser().resolve()
    if args.envelope_shadow and args.operation!='prepare':parser.error('--envelope-shadow is only a prepare option')
    if (args.operation=='envelope-shadow')!=bool(args.support):parser.error('The envelope-shadow operation requires --support, and vice versa')
    if args.operation=='prepare':prepare_host(output,envelope_shadow=args.envelope_shadow)
    elif args.operation=='envelope-shadow':collect_host(output,envelope_shadow=True,support=args.support)
    elif args.operation=='entry-http':collect_host(output,entry_http=True)
    elif args.operation=='queue-confirmation':collect_host(output,queue_confirmation=True)
    elif args.operation=='queue-probe':collect_host(output,queue_probe=True)
    elif args.operation=='live-http':collect_host(output,live_http=True)
    elif args.operation=='clock':collect_host(output,event_clock=True)
    elif args.operation=='arrivals':collect_host(output,arrivals=True)
    elif args.operation=='calibrate':collect_host(output,calibration=True)
    elif args.operation=='stages':collect_host(output,stages=True)
    else:collect_host(output,diagnostics=args.operation=='diagnose')


if __name__=='__main__':main()
