"""Prepare exact tokens, validate CPU code, and collect the complete PR32 suite."""
import argparse
import asyncio
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import digest,save
from prefill_lab.suite import inputs,Client,run
from prefill_lab.tokens import materialize,validate


def runtime_environment():
    """Make interpreter-local tools visible to EngineCore/FlashInfer descendants."""
    env=os.environ.copy()
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH',os.defpath)
    env.pop('PYTHONHOME',None)
    if sys.prefix!=sys.base_prefix:env['VIRTUAL_ENV']=sys.prefix
    return env


def failure_log_tails(output,secret):
    """Show child failures alongside the wrapper traceback, without credentials."""
    for name in ('server.log','gateway.log'):
        try:
            with (Path(output)/name).open(errors='replace') as stream:
                tail=''.join(deque(stream,maxlen=80))
        except OSError:continue
        if tail:
            if secret:tail=tail.replace(secret,'[REDACTED]')
            print(f'\n{name} (last80 lines):\n'+tail,file=sys.stderr,flush=True)


def code_identity():
    paths=sorted([*ROOT.glob('prefill_lab/*.py'),ROOT/'scripts/run_prefill_service.py',
                  ROOT/'experiments/prefill-vllm-v030-source-map.json',ROOT/'gateway.py',
                  ROOT/'experiments/prefill-stage-diagnostic-v1.json',ROOT/'experiments/prefill-effective-calibration-v1.json',ROOT/'experiments/prefill-effective-arrivals-v2.json',ROOT/'experiments/prefill-event-clock-v1.json',ROOT/'experiments/prefill-event-clock-confirmation-v1.json',ROOT/'experiments/prefill-live-http-v1.json',
                  ROOT/'prefill_grouped_validation.py',ROOT/'prefill_validation.py',
                  *ROOT.glob('experiments/prefill-service-v1.*'),*ROOT.glob('*.py'),
                  *ROOT.glob('scripts/*.py'),*ROOT.glob('tests/*.py'),ROOT/'pyproject.toml'])
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def cpu_check(output):
    if platform.system()!='Linux' or sys.version_info[:2]!=(3,12):raise RuntimeError('Native Linux/Python3.12 required')
    if os.environ.get('PYTEST_ADDOPTS') or os.environ.get('PYTEST_PLUGINS'):
        raise RuntimeError('Unset inherited pytest filters/plugins for complete acceptance')
    output.mkdir(parents=True,exist_ok=False)
    before=code_identity()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked_clean=not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip()
    from diagnostic_sampling import ObservationSampler
    resources=ObservationSampler.process_resources(os.getpid())
    if not resources['available']:
        save(output/'acceptance.json',dict(valid=False,commit=commit,source_sha256=before,process_resources=resources,
                                          reason='Native PID/procfs namespace is not verified; use the rental host',tests=0))
        print('Native process-resource preflight failed; no test filtering or procfs substitution is permitted.')
        return 1
    command=[sys.executable,'-m','pytest','-q','tests','-o','addopts=','--junitxml',str(output/'junit.xml')]
    env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg',TRACE_BASELINE_ROOT=str(ROOT/'results'))
    process=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/'pytest.log').write_text(process.stdout);print(process.stdout,end='')
    import xml.etree.ElementTree as ET
    cases=ET.parse(output/'junit.xml').findall('.//testcase')
    from scripts.run_linux_cpu_tests import validate_report
    try:coverage=validate_report(output/'junit.xml')
    except RuntimeError as exc:coverage=dict(error=str(exc))
    unsuccessful=[c.get('name') for c in cases if any(c.find(t) is not None for t in ('failure','error','skipped'))]
    result=dict(valid=process.returncode==0 and bool(cases) and not unsuccessful and before==code_identity() and 'error' not in coverage and tracked_clean,
                source_sha256=before,commit=commit,tracked_clean=tracked_clean,python=sys.version,platform=platform.platform(),tests=len(cases),
                unsuccessful=unsuccessful,coverage=coverage,command=command,vllm_gpu_validated=False)
    save(output/'acceptance.json',result)
    return 0 if result['valid'] else 1


def prepare(args):
    from transformers import AutoTokenizer
    directory=Path(args.tokenizer).resolve()
    if not directory.is_dir():raise ValueError('Use a local pinned tokenizer snapshot')
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file() and (p.suffix in ('.json','.model') or p.name.endswith('.txt'))}
    cases,order,_=inputs()
    if getattr(args,'live_http_tokens',False):
        from prefill_lab.live_http import protocol
        plan=protocol();cases,order=plan['cases'],plan['order']
    elif getattr(args,'event_clock_only',False) or getattr(args,'live_http_only',False):
        from prefill_lab.event_clock_calibration import protocol
        plan=protocol();cases,order=plan['cases'],plan['order']
    elif getattr(args,'arrivals_only',False):
        from prefill_lab.delayed_calibration import protocol
        plan=protocol();cases,order=plan['cases'],plan['order']
    elif getattr(args,'calibration_only',False):
        from prefill_lab.effective_calibration import protocol
        plan=protocol();cases,order=plan['cases'],plan['order']
    manifest=materialize(cases,order,tokenizer,model=args.model,revision=args.revision,tokenizer_sha256=digest(hashes))
    manifest['tokenizer_files']=hashes
    save(args.output,manifest)
    print(f'Prepared {len(order)} exact-token trials: {args.output}')


async def collect(args):
    import httpx
    from trace_workload import load_trace,verify_prompts
    from transformers import AutoTokenizer
    if platform.system()!='Linux' or sys.version_info[:2]!=(3,12):raise RuntimeError('Collect requires native Linux/Python3.12')
    from diagnostic_sampling import ObservationSampler
    if not ObservationSampler.process_resources(os.getpid())['available']:
        raise RuntimeError('Current collection host PID/procfs view is not verified')
    acceptance=json.loads(Path(args.cpu_acceptance).read_text())
    current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not acceptance.get('valid') or acceptance.get('commit')!=current_commit or acceptance['source_sha256']!=code_identity():
        raise RuntimeError('Run cpu-check for this exact source tree before collection')
    manifest=json.loads(Path(args.manifest).read_text());cases,order,_=inputs()
    bridge=getattr(args,'live_http_only',False)
    clock=getattr(args,'event_clock_only',False) or bridge
    arrivals=getattr(args,'arrivals_only',False) or clock
    calibration=getattr(args,'calibration_only',False) or arrivals
    if calibration:
        if clock:
            from prefill_lab.event_clock_calibration import protocol,validate_manifest
        elif arrivals:
            from prefill_lab.delayed_calibration import protocol,validate_manifest
        else:
            from prefill_lab.effective_calibration import protocol,validate_manifest
        plan=protocol();cases,order=plan['cases'],plan['order'];validate_manifest(manifest)
    validate(manifest,cases,order)
    if bridge:
        from prefill_lab.live_experiment import validate_manifest as validate_http,validate_pair
        http_manifest=json.loads(Path(args.http_manifest).read_text());validate_http(http_manifest);validate_pair(manifest,http_manifest)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    for name,sha in manifest['tokenizer_files'].items():
        if hashlib.sha256((Path(args.tokenizer)/name).read_bytes()).hexdigest()!=sha:raise ValueError('Tokenizer snapshot changed')
    trace=load_trace(args.trace);verify_prompts(trace,tokenizer)
    if trace['model']!=manifest['model'] or trace['duration_seconds']<240 or trace['output_cap']!=128:
        raise ValueError('Use the frozen Azure development replay with natural-EOS128 policy')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'token-manifest.json',manifest);save(output/'collection-sources.json',code_identity())
    if calibration:
        evidence=output/'native-cpu-acceptance';evidence.mkdir()
        acceptance_dir=Path(args.cpu_acceptance).parent
        for name in ('acceptance.json','junit.xml','pytest.log'):
            shutil.copyfile(acceptance_dir/name,evidence/name)
    save(output/'trace-identity.json',dict(sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),provenance=trace.get('provenance')))
    key=secrets.token_hex(32)
    stages=getattr(args,'stage_diagnostics_only',False)
    diagnostics=getattr(args,'diagnostics_only',False) or stages or calibration
    env=dict(runtime_environment(),PREFILL_LAB_ENABLED='1',PREFILL_LAB_KEY=key,
             PREFILL_LAB_MANIFEST=str(output/'token-manifest.json'),PREFILL_LAB_ARTIFACT_DIR=str(output/'deep-traces'),
             VLLM_PLUGINS='prefill_lab',PREFILL_LAB_STAGES='1' if stages else '0',
             PREFILL_LAB_EFFECTIVE_CALIBRATION='1' if calibration and not arrivals else '0',
             PREFILL_LAB_DELAYED_ARRIVALS='1' if arrivals else '0',
             PREFILL_LAB_EVENT_CLOCK='1' if clock else '0',
             PREFILL_LAB_LIVE_STATE='1' if bridge else '0')
    command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',manifest['model'],
             '--revision',manifest['revision'],'--tokenizer',str(Path(args.tokenizer).resolve()),
             '--tokenizer-revision',manifest['revision'],'--host','127.0.0.1','--port','8000',
             '--max-model-len','8192','--max-num-batched-tokens','2048','--max-num-seqs','32',
             '--tensor-parallel-size','1','--no-enable-prefix-caching']
    # Credentials stay in the environment on an isolated localhost benchmark server.
    # The instrument endpoint authenticates itself; this runtime is never remotely bound.
    save(output/'launch.json',dict(argv=command,source_sha256=code_identity(),cpu_acceptance=acceptance,
                                  scope='live_http_state_bridge' if bridge else 'effective_event_clock_confirmation' if clock else 'effective_delayed_arrivals_pilot' if arrivals else 'effective_calibration_pilot' if calibration else 'targeted_stage_diagnostic_only' if stages else 'targeted_foreground_diagnostic_only' if diagnostics else 'complete_suite',
                                  environment={k:env[k] for k in ('PREFILL_LAB_ENABLED','VLLM_PLUGINS','PREFILL_LAB_STAGES','PREFILL_LAB_EFFECTIVE_CALIBRATION','PREFILL_LAB_DELAYED_ARRIVALS','PREFILL_LAB_EVENT_CLOCK','PREFILL_LAB_LIVE_STATE')}))
    from scripts.setup_v1 import check_port
    for port in ((8000,) if diagnostics and not bridge else (8000,8080)):check_port(port)
    server=gateway=None
    server_log=(output/'server.log').open('w');gateway_log=(output/'gateway.log').open('w')
    try:
        server=subprocess.Popen(command,cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        async with httpx.AsyncClient(headers={'authorization':'Bearer '+key},timeout=45,trust_env=False,
                                     limits=httpx.Limits(max_connections=None,max_keepalive_connections=100)) as http:
            client=Client(http,'http://127.0.0.1:8000')
            deadline=time.monotonic()+args.startup_timeout
            while True:
                if server.poll() is not None:raise RuntimeError('Server exited during model/plugin startup; see server.log')
                try:
                    response=await http.get(client.url+'/health')
                    if response.status_code==200:
                        try:await client.command('meta')
                        except httpx.HTTPError as exc:raise RuntimeError('Engine is healthy but prefill plugin/source contract failed; see server.log') from exc
                        break
                except httpx.HTTPError:pass
                if time.monotonic()>deadline:raise TimeoutError('Initial model load deadline expired')
                await asyncio.sleep(1)
            if bridge:
                combined_deadline=time.monotonic()+1800
                gateway_env=dict(runtime_environment(),ADMISSION_MODE='protected',MAX_IN_FLIGHT='32',MAX_WAITING='0',
                    MODEL_NAME=str(Path(args.tokenizer).resolve()),VLLM_URL=client.url,SHARED_UPSTREAM_CLIENT='1',
                    GATEWAY_DIAGNOSTICS='1',BENCHMARK_REQUEST_IDS='1',PREFILL_LAB_RAW_COMPLETIONS='1',
                    TELEMETRY_JSONL=str(output/'gateway.jsonl'),TELEMETRY_STATUS_JSON=str(output/'telemetry-status.json'))
                gateway=subprocess.Popen([sys.executable,'-m','uvicorn','gateway:app','--host','127.0.0.1','--port','8080'],cwd=ROOT,env=gateway_env,stdout=gateway_log,stderr=subprocess.STDOUT,start_new_session=True)
                for _ in range(120):
                    if gateway.poll() is not None:raise RuntimeError('Live HTTP gateway startup failed')
                    try:
                        response=await http.get('http://127.0.0.1:8080/metrics')
                        if response.status_code==200:
                            from prefill_lab.live_http import gateway_contract
                            gateway_start=response.json();gateway_contract(gateway_start)
                            if gateway_start['accepted_requests']!=0:raise ValueError('Gateway was not empty at bridge startup')
                            save(output/'gateway-start.json',gateway_start);break
                    except httpx.HTTPError:pass
                    await asyncio.sleep(1)
                else:raise TimeoutError('Live HTTP gateway startup timeout')
                from prefill_lab.live_experiment import run as run_live
                summary=await asyncio.wait_for(run_live(client,http,client.url,manifest,http_manifest,output,'http://127.0.0.1:8080',combined_deadline),max(.001,combined_deadline-time.monotonic()))
                print('Live HTTP result: '+json.dumps(summary),flush=True)
                return 0 if summary['candidate_passed'] else 2
            if calibration:
                if arrivals:
                    from prefill_lab.delayed_calibration import run as run_calibration
                else:
                    from prefill_lab.effective_calibration import run as run_calibration
                summary=await run_calibration(client,http,client.url,manifest,output,event_clock=True) if clock else await run_calibration(client,http,client.url,manifest,output)
                print('Calibration result: '+json.dumps(summary),flush=True)
                return 0 if summary['candidate_passed'] else 2
            if diagnostics:
                from prefill_lab.deep_diagnostic import run as run_diagnostic
                summary=await run_diagnostic(client,http,client.url,manifest,output,stages=True) if stages else await run_diagnostic(client,http,client.url,manifest,output)
                return 0 if summary['diagnostic_complete'] else 1
            gateway_env=dict(runtime_environment(),ADMISSION_MODE='cost_aware',MAX_ADMITTED_COST='32',
                             MODEL_NAME=str(Path(args.tokenizer).resolve()),VLLM_URL=client.url,
                             ADMIN_API_KEY=secrets.token_hex(32),SHARED_UPSTREAM_CLIENT='1',PREFILL_LAB_RAW_COMPLETIONS='0',
                             GATEWAY_DIAGNOSTICS='1',BENCHMARK_REQUEST_IDS='1',LOOP_LAG_TELEMETRY='1',
                             TELEMETRY_JSONL=str(output/'gateway.jsonl'),TELEMETRY_STATUS_JSON=str(output/'telemetry-status.json'))
            gateway=subprocess.Popen([sys.executable,'-m','uvicorn','gateway:app','--host','127.0.0.1','--port','8080'],cwd=ROOT,env=gateway_env,stdout=gateway_log,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(120):
                if gateway.poll() is not None:raise RuntimeError('Gateway startup failed')
                try:
                    r=await http.get('http://127.0.0.1:8080/metrics')
                    if r.status_code==200 and r.json()['tokenizer_available']:break
                except httpx.HTTPError:pass
                await asyncio.sleep(1)
            else:raise TimeoutError('Gateway tokenizer/startup deadline')
            summary=await run(client,http,client.url,manifest,trace,output)
            if not summary['complete']:
                print('Collection incomplete: '+json.dumps(summary['failures']),file=sys.stderr,flush=True)
            if summary['complete']:
                from prefill_lab.scoring import score
                frozen=json.loads((output/'model-freeze.json').read_text())
                meta=json.loads((output/'runtime.json').read_text())
                refs=json.loads((ROOT/'experiments/prefill-service-v1.discrimination.json').read_text())['jitter_references']['targets']
                mapping={r['case']:r['reference'] for r in refs if r['request_id']=='sentinel'}
                if frozen.get('fits'):
                    try:report=score(output,cases,order,frozen,meta['workers'][0]['dispatch_table'],mapping)
                    except ValueError as exc:report=dict(status='inconclusive',reason=str(exc),complete_experiment_passed=False)
                else:report=dict(status='inconclusive',reason=frozen.get('reason'),complete_experiment_passed=False)
                save(output/'held-report.json',report)
            return 0 if summary['complete'] else 1
    except BaseException as exc:
        save(output/'startup-or-run-failure.json',dict(type=type(exc).__name__,message=str(exc),complete=False))
        failure_log_tails(output,key)
        raise
    finally:
        for process in (gateway,server):
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        server_log.close();gateway_log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--tokenizer',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--model',default='Qwen/Qwen3-4B-Instruct-2507');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('cpu-check');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('collect')
    for name in ('manifest','tokenizer','trace','cpu-acceptance','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--startup-timeout',type=float,default=900.)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--live-http-only',action='store_true',help='Fresh clock calibration then busy-state HTTP qualification')
    p.add_argument('--http-manifest',help='Dedicated exact-token HTTP input manifest')
    group.add_argument('--event-clock-only',action='store_true',help='Fresh training-event clock fit, freeze and independent delayed confirmation')
    group.add_argument('--arrivals-only',action='store_true',help='Delayed arrivals: fresh training, freeze, anchors, held sentinel trials')
    group.add_argument('--calibration-only',action='store_true',help='Reduced effective-service pilot; train/freeze/fresh held endpoints')
    group.add_argument('--diagnostics-only',action='store_true',help='Two frozen light warmups and two foreground deep traces; no suite/model acceptance')
    group.add_argument('--stage-diagnostics-only',action='store_true',help='Opt-in V2 stage timing diagnostic; no suite/model acceptance')
    args=parser.parse_args()
    if args.command=='collect' and bool(args.live_http_only)!=bool(args.http_manifest):parser.error('--live-http-only requires --http-manifest, and vice versa')
    if args.command=='prepare':prepare(args);return 0
    if args.command=='cpu-check':return cpu_check(args.output)
    return asyncio.run(collect(args))


if __name__=='__main__':raise SystemExit(main())
