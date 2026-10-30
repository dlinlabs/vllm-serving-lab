"""Reconstruct the efca3e4 TRAIN rejection; never promote partial collection."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import digest,finite,save
from scripts.analyze_prefill_effective_capture import archive_identity,source_paths
from scripts.analyze_prefill_queue_confirmation import EXTRAS as CONFIRMATION_EXTRAS

COMMIT='efca3e4e5e1a1237e796b39d965e4fed233b5fdb'
EXTRAS=CONFIRMATION_EXTRAS|{'experiments/prefill-entry-http-v1.json','experiments/prefill-entry-http-v1.plan.json'}


def snapshot(capture,destination):
    load=lambda n:json.loads((capture/n).read_text());launch=load('launch.json')
    if launch['scope']!='entry_http_state_bridge' or launch['cpu_acceptance']['commit']!=COMMIT:raise ValueError('Unsupported failed capture source/scope')
    def git(*args):
        try:return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:raise ValueError('Pinned source unavailable; run git fetch --depth 1 origin '+COMMIT) from exc
    names=git('ls-tree','-r','--name-only',COMMIT).decode().splitlines();sources=load('collection-sources.json')
    if set(sources)!=source_paths(names)|EXTRAS:raise ValueError('Failed capture source inventory differs')
    for name,sha in sources.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe source path')
        raw=git('show',COMMIT+':'+name)
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Git/native source mismatch: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return sources


# Runs against ONLY the pinned original modules in a verified temporary tree.
# Reconstructs a rejection, not the complete-collection checker or acceptance.
WORKER=r'''
import json,sys,time
from pathlib import Path
from prefill_lab import queue_confirmation as qc,event_clock_calibration as clock,entry_clock,entry_http
from prefill_lab.contracts import canonical,digest,finite,save
from prefill_lab.suite import runtime_preflight
from prefill_lab.causal import dispatcher_parity
from scripts.predict_prefill_async import runtime_inputs
from scripts.run_prefill_service import code_identity
from scripts.run_linux_cpu_tests import validate_report
root,out=map(Path,sys.argv[1:]);controlled=root/'controlled'
load=lambda n:json.loads((root/n).read_text());read=lambda n:json.loads((controlled/n).read_text())
for n in ('launch.json','collection-sources.json','token-manifest.json','native-cpu-acceptance/acceptance.json','native-cpu-acceptance/junit.xml','native-cpu-acceptance/pytest.log'):
    if (root/n).read_bytes()!=(controlled/n).read_bytes():raise ValueError('Parent/controlled evidence differs: '+n)
p=qc.protocol();boundary=next(i for i,t in enumerate(p['order']) if t['stage'] in qc.HELD)
expected=[t['trial_id'] for t in p['order'][:boundary]];summary=read('confirmation.json');http=load('entry-http-summary.json')
message='TRAIN entry-clock/phase readiness failed; held phase not attempted'
failure=[dict(type='ValueError',message=message)]
if boundary!=88 or summary['completed']!=expected or summary['failures']!=failure or summary['collection_complete'] or summary['candidate_passed']:raise ValueError('Failed TRAIN prefix differs')
if http['completed'] or http['failures']!=failure or http['collection_complete'] or http['candidate_passed']:raise ValueError('HTTP was attempted or failure differs')
if any(s.get(k) is not False for s in (summary,http) for k in qc.ACCEPTANCE):raise ValueError('Partial run promoted acceptance')
if load('startup-or-run-failure.json')!=dict(type='ValueError',message=message,complete=False):raise ValueError('Outer failure differs')
if {f.stem for f in controlled.glob('qc-*.json')}!=set(expected) or list((root/'bridge').glob('*')):raise ValueError('Unexpected controlled/HTTP captures')
if any((controlled/n).exists() for n in ('base-model.json','entry-freeze.json','seal.json','confirmation-report.json')):raise ValueError('Rejected candidate was sealed or scored')
source=read('collection-sources.json');launch=read('launch.json');accept=read('native-cpu-acceptance/acceptance.json');env=launch['environment']
if source!=code_identity() or launch['source_sha256']!=source or launch['cpu_acceptance']!=accept or accept['source_sha256']!=source or not accept['valid'] or not accept['tracked_clean']:raise ValueError('Native source/CPU proof differs')
if validate_report(controlled/'native-cpu-acceptance/junit.xml')['test_cases']!=accept['tests']:raise ValueError('CPU denominator differs')
if env.get('PREFILL_LAB_ENABLED')!='1' or env.get('VLLM_PLUGINS')!='prefill_lab' or any(env.get('PREFILL_LAB_'+k)!=v for k,v in dict(QUEUE_CONFIRMATION='1',ENTRY_HTTP='1',LIVE_STATE='1',QUEUE_PROBE='0',STAGES='0',EFFECTIVE_CALIBRATION='0',DELAYED_ARRIVALS='0',EVENT_CLOCK='0').items()):raise ValueError('Combined launch flags differ')
manifest=read('token-manifest.json');qc.validate_manifest(manifest);entry_http.validate_manifest(load('http-token-manifest.json'));entry_http.old.validate_pair(manifest,load('http-token-manifest.json'))
if digest(read('queue-confirmation-plan.json'))!=digest(p) or digest(load('entry-http-plan.json'))!=digest(entry_http.protocol()):raise ValueError('Numeric plan differs')
runtime=read('runtime.json');runtime_preflight(runtime,manifest);runtime_inputs(runtime)
pins=json.loads(Path('experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
if any(s['version']!='0.30.0' or s['files']!=pins for s in [runtime['source'],*[w['source'] for w in runtime['workers']]]) or any(w.get('pipeline_stages') for w in runtime['workers']):raise ValueError('Native installed source/stages differ')
rows=[];caps=[];previous=None;cases={c['id']:c for c in p['cases']}
for t in p['order'][:boundary]:
    cap=read(t['trial_id']+'.json')
    if not all(finite(cap.get(k)) for k in ('start','end')) or cap['start']>cap['end'] or previous is not None and cap['start']<previous:raise ValueError('Prefix chronology differs')
    previous=cap['end'];rows.append(qc.extract(cap,t,cases[t['case']],p))
    if t['observer']=='light' and not dispatcher_parity(cap,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('Native dispatcher differs')
    if t['stage']=='train' and t['observer']=='light':caps.append(cap)
if digest(rows)!=digest(read('confirmation-rows.json')):raise ValueError('Prefix rows differ from raw captures')
# Match seal_training's canonical round trip before fitting. Dict insertion
# order otherwise changes diagnostic-list order and downstream document hashes.
rows=json.loads(canonical(rows))
events=[clock.event_labels(c,t,cases[t['case']]) for t in p['order'] if t['stage']=='train' and t['observer']=='light' for c in caps if c['trial_id']==t['trial_id']]
model=clock.fit(rows,events,p,runtime,deadline=time.monotonic()+240,origin='fresh_native_training_events')
frozen=entry_clock.freeze(caps,p,runtime,model)
if frozen['readiness_passed']:raise ValueError('Original TRAIN failure did not reconstruct')
save(out/'reconstructed-base-model.json',model);save(out/'reconstructed-rejected-entry.json',frozen)
print(json.dumps(dict(rejection_reconstructed=True,completed_prefix=len(rows),train_light_captures=len(caps),native_cpu_tests=accept['tests'],source_files=len(source),complete_collection_audited=False)))
'''


def reconstruct(capture,output):
    import numpy,scipy
    native=json.loads((capture/'controlled/host-before.json').read_text())['packages']
    if (numpy.__version__,scipy.__version__)!=(native['numpy'],native['scipy']):
        raise ValueError('Exact numerical replay requires captured numpy='+native['numpy']+' scipy='+native['scipy'])
    with tempfile.TemporaryDirectory(prefix='pr32-rejected-training-') as temp:
        source=Path(temp);snapshot(capture,source)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source)+os.pathsep+os.environ.get('PYTHONPATH',''))
        result=subprocess.run([sys.executable,'-c',WORKER,str(capture),str(output)],cwd=source,env=env,text=True,capture_output=True,timeout=300)
        if result.returncode:raise ValueError('Original-source rejected TRAIN reconstruction failed: '+result.stdout+result.stderr)
    return dict(commit=COMMIT,original_worker_sha256=hashlib.sha256(WORKER.encode()).hexdigest(),
        numerical_versions=dict(numpy=numpy.__version__,scipy=scipy.__version__),**json.loads(result.stdout))


def summarize(frozen):
    checks=frozen['first_commit_readiness']['checks'];groups=defaultdict(list)
    if len(checks)!=24 or len({r['trial_id'] for r in checks})!=24:raise ValueError('TRAIN first-commit denominator differs')
    for r in checks:
        if not all(finite(r[k]) for k in ('observed_s','predicted_s','limit_s')) or r['limit_s']<0:raise ValueError('Invalid TRAIN clock/budget')
        if r['passed']!=(abs(r['predicted_s']-r['observed_s'])<=r['limit_s']):raise ValueError('TRAIN first-commit verdict differs')
        groups[r['case']].append(r)
    if len(groups)!=6 or any(len(v)!=4 for v in groups.values()):raise ValueError('TRAIN cell/repeat denominator differs')
    passed=all(r['passed'] for r in checks)
    if frozen['first_commit_readiness']['passed']!=passed or frozen['readiness_passed']!=(not frozen['errors'] and passed):raise ValueError('TRAIN readiness verdict differs')
    if frozen['readiness_passed'] is not False:raise ValueError('Expected rejected TRAIN candidate')
    intervals=[]
    for case,rs in sorted(groups.items()):
        lower=max(r['observed_s']-r['limit_s'] for r in rs);upper=min(r['observed_s']+r['limit_s'] for r in rs)
        intervals.append(dict(case=case,samples=len(rs),observed_s=[r['observed_s'] for r in rs],
            lower_s=lower,upper_s=upper,one_case_constant_feasible=lower<=upper,gap_s=max(0.,lower-upper)))
    return dict(schema='entry-http-rejected-training-summary-v1',readiness_passed=frozen['readiness_passed'],
        first_commit_samples=len(checks),first_commit_passed=sum(r['passed'] for r in checks),
        first_commit_failures=[r for r in checks if not r['passed']],phase_errors=frozen['errors'],
        constants_s=frozen['labels']['constants_s'],case_constant_intersections=intervals,
        scope='TRAIN-only reconstruction of rejected/unsealed candidate; no held or HTTP accuracy measured',
        interpretation='Empty intersection rules out one fixed prediction for that planned case under the original budgets; it does not rule out a separately qualified state-conditioned predictor or identify physical CPU/network contention',
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


def release_controls(capture,checks):
    """Descriptive scheduler-state equality; no pending readiness is inferred."""
    groups=defaultdict(list)
    for check in checks:
        cap=json.loads((capture/'controlled'/(check['trial_id']+'.json')).read_text());state=cap['release_state'];prefix='plab/'+check['trial_id']+'/'
        ids=[*state['requests'],*state['running'],*state['waiting']]
        if any(not rid.startswith(prefix) for rid in ids):raise ValueError('Release scheduler request identity differs')
        normal=dict(requests={rid.removeprefix(prefix):r for rid,r in state['requests'].items()},
            running=[r.removeprefix(prefix) for r in state['running']],waiting=[r.removeprefix(prefix) for r in state['waiting']])
        groups[check['case']].append(dict(trial_id=check['trial_id'],scheduler_sha256=digest(normal),observed_first_commit_s=check['observed_s']))
    return [dict(case=case,rows=rows,distinct_recorded_scheduler_states=len({r['scheduler_sha256'] for r in rows}),
        scope='Normalized recorded release scheduler state only; excludes pending readiness/debt and physical CPU/GPU state') for case,rows in sorted(groups.items())]


def run(capture,archive,output):
    capture,archive,output=map(lambda p:Path(p).resolve(),(capture,archive,output))
    if output==capture or capture in output.parents:raise ValueError('Output must be outside immutable capture')
    identity,index=archive_identity(archive,capture);output.mkdir(parents=True,exist_ok=False)
    save(output/'archive.json',identity);save(output/'input-files.json',index)
    proof=reconstruct(capture,output);save(output/'reconstruction.json',proof)
    frozen=json.loads((output/'reconstructed-rejected-entry.json').read_text());summary=summarize(frozen)
    summary['release_scheduler_controls']=release_controls(capture,frozen['first_commit_readiness']['checks']);save(output/'summary.json',summary)
    return dict(reconstruction=proof,summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=run(args.capture,args.archive,args.output)
    print(json.dumps(dict(rejection_reconstructed=result['reconstruction']['rejection_reconstructed'],readiness_passed=result['summary']['readiness_passed'],model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)))
