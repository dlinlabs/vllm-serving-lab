"""Read-only foreground trace audit; never fit service costs or grant GPU acceptance."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefill_lab.contracts import audit_capture, digest, save
from prefill_lab.causal import dispatcher_parity
from prefill_lab.deep_diagnostic import profile_audit
from prefill_lab.tokens import validate

GPU_CATEGORIES = ('kernel', 'gpu_memcpy', 'gpu_memset')
PREFIX = 'prefill_lab_step/'


def union_us(events):
    spans = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
    total = 0.
    right = float('-inf')
    for a, b in spans:
        if b < a:
            raise ValueError('Negative trace duration')
        total += max(0., b - max(a, right))
        right = max(right, b)
    return total


def correlate(events, step_ids):
    """Join GPU activity to its launch API and CPU step, not GPU clock overlap.

    Graph nodes share the graph-launch correlation. Runtime/driver correlation
    collisions and missing launches are rejected rather than guessed. CUDA stream
    handles and CUPTI stream IDs are intentionally never numerically compared.
    """
    names = [PREFIX + key for key in step_ids]
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name') in names]
    if Counter(e['name'] for e in markers) != Counter(names) or len(set(names)) != len(names):
        raise ValueError('Missing/duplicate CPU step markers')
    markers.sort(key=lambda e: e['ts'])
    if [e['name'] for e in markers] != names:
        raise ValueError('CPU marker order mismatch')
    if any(e['dur'] <= 0 for e in markers):
        raise ValueError('Nonpositive CPU step marker')
    apis = {}
    for e in events:
        if e.get('cat') not in ('cuda_runtime', 'cuda_driver') or 'correlation' not in e.get('args', {}):
            continue
        key = e['args']['correlation']
        if key in apis:
            raise ValueError('Ambiguous CUDA correlation ID')
        apis[key] = e
    def owner(api):
        matches = [i for i, m in enumerate(markers) if (api['pid'], api['tid']) == (m['pid'], m['tid'])
                   and m['ts'] <= api['ts'] < m['ts'] + m['dur']]
        if len(matches) > 1:
            raise ValueError('Overlapping CPU markers')
        return matches[0] if matches else None
    assigned = defaultdict(list)
    outside = []
    for e in events:
        if e.get('cat') not in GPU_CATEGORIES:
            continue
        api = apis.get(e.get('args', {}).get('correlation'))
        if api is None:
            raise ValueError('GPU activity has no unambiguous launch correlation')
        i = owner(api)
        (outside if i is None else assigned[i]).append(e)
    per_step = []
    for i, m in enumerate(markers):
        gpu = assigned[i]
        if not gpu:
            raise ValueError('Step marker has no correlated GPU work')
        annotations = [e for e in events if e.get('cat') == 'gpu_user_annotation' and e.get('name') == m['name']]
        if 'External id' not in m.get('args', {}) or not annotations or any(e['args'].get('External id') != m['args']['External id'] for e in annotations):
            raise ValueError('Missing/mismatched GPU annotation identity')
        a, b = min(e['ts'] for e in gpu), max(e['ts'] + e['dur'] for e in gpu)
        outside_annotation = [e for e in gpu if not any(e['args']['stream']==g['tid']
            and g['ts']-.01 <= e['ts'] and e['ts']+e['dur'] <= g['ts']+g['dur']+.01 for g in annotations)]
        per_step.append(dict(step_id=step_ids[i],cpu_start_us=m['ts'],cpu_duration_us=m['dur'],
            gpu_start_us=a,gpu_envelope_us=b-a,gpu_activity_union_us=union_us(gpu),
            gpu_streams=sorted({e['args']['stream'] for e in gpu}),
            activity_counts=dict(Counter(e['cat'] for e in gpu)),
            correlated_activities_outside_gpu_annotation=len(outside_annotation),
            gpu_annotations=[dict(start_us=e['ts'],duration_us=e['dur'],stream=e['tid']) for e in annotations]))
    syncs = [dict(name=e['name'],start_us=e['ts'],duration_us=e['dur']) for e in apis.values()
             if e['name'] == 'cudaDeviceSynchronize']
    return dict(steps=per_step,origin_us=markers[0]['ts'],
        gpu_activity_count=sum(len(v) for v in assigned.values())+len(outside),
        unmatched_gpu_activities=0,outside_forward_counts=dict(Counter(e['cat'] for e in outside)),
        outside_forward_streams=sorted({e['args']['stream'] for e in outside}),
        outside_forward_kernel_names=dict(Counter(e['name'] for e in outside if e['cat']=='kernel')),
        outside_forward_timeline=[dict(category=e['cat'],start_us=e['ts'],duration_us=e['dur'],stream=e['args']['stream']) for e in outside],
        device_synchronizations=syncs,profile_stop_precedes_fifth_sampling=True,
        cuda_scope_qualified=False,model_accepted=False,
        limitations=['Only the five captured calls; absence outside this window is not evidence of absence.',
                    'Fifth-call sampling follows profiler.stop and is not captured.',
                    'CUDA event endpoints have no CUPTI timestamps in this artifact.',
                    'Forward GPU envelopes and event elapsed values include idle gaps; neither is kernel busy time.'])


def analyze(capture, output, archive):
    capture, output = Path(capture), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    load = lambda name: json.loads((capture/name).read_text())
    launch = load('launch.json')
    acceptance = launch['cpu_acceptance']
    ref = acceptance['commit']
    source = lambda path: subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
    declared = load('collection-sources.json')
    if not acceptance.get('valid') or not acceptance.get('tracked_clean'):
        raise ValueError('Missing clean native CPU acceptance')
    if declared != launch['source_sha256'] or declared != acceptance['source_sha256']:
        raise ValueError('Launch/acceptance/source identity mismatch')
    if any(hashlib.sha256(source(p)).hexdigest()!=h for p,h in declared.items()):
        raise ValueError('Captured source does not match its Git commit')
    meta, final = load('runtime.json'), load('runtime-final.json')
    tokens, summary = load('token-manifest.json'), load('diagnostic.json')
    order = json.loads(source('experiments/prefill-service-v1.order.json'))['order']
    cases = json.loads(source('experiments/prefill-service-v1.cases.json'))['cases']
    validate(tokens,cases,order)
    if digest(meta)!=digest(final) or digest(tokens)!=meta['token_manifest_sha256']:
        raise ValueError('Runtime/token identity mismatch')
    pins = json.loads(source('experiments/prefill-vllm-v030-source-map.json'))['sha256']
    if meta['source']['files']!=pins or any(w['source']['files']!=pins for w in meta['workers']):
        raise ValueError('Installed engine/worker source identity mismatch')
    if load('trace-identity.json')['sha256']!=hashlib.sha256(source('results/trace-experiment-96-r1/trace.json')).hexdigest():
        raise ValueError('Azure trace input identity mismatch')
    deep = [t for t in order if t['stage']=='deep_diagnostic']
    warm = [next(t for t in order if t['stage']=='warmup' and t['case']==d['case']) for d in deep]
    expected = [t['trial_id'] for t in warm+deep]
    if launch['scope']!='targeted_foreground_diagnostic_only' or summary['planned']!=expected or summary['completed']!=expected:
        raise ValueError('Targeted diagnostic order/scope mismatch')
    if not summary['diagnostic_complete'] or summary['failures']:
        raise ValueError('Incomplete diagnostic')
    if any(summary.get(k) is not False for k in ('cuda_scope_qualified','model_accepted','gpu_validated','complete_experiment_passed')):
        raise ValueError('Diagnostic incorrectly claims broader acceptance')
    report = dict(capture_commit=ref,archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files_verified=len(declared),installed_source_files=len(pins),
        recorded_native_cpu_acceptance={k:acceptance[k] for k in ('valid','tests','tracked_clean')},
        original_junit_in_archive=False,scope='Offline diagnostic identity/work/profile/correlation audit only',
        diagnostic_complete=True,elapsed_seconds=summary['elapsed_seconds'],
        runtime={k:meta[k] for k in ('async_scheduling','max_concurrent_batches','scheduler_class','num_gpu_blocks','revision')},
        host=load('host.json'),trials=[],profiles=[],model_accepted=False,cuda_scope_qualified=False,gpu_validated=False)
    for trial in warm+deep:
        tid=trial['trial_id'];c=load(tid+'.json');workers=[s for w in c['workers'] for s in w['rows']]
        if c['trial_id']!=tid or c['observer']!=trial['observer']:
            raise ValueError('Capture trial/mode identity mismatch')
        audit=audit_capture(c,workers);audit['dispatcher']=dispatcher_parity(c,meta['workers'][0]['dispatch_table'])
        if trial['stage']=='deep_diagnostic':
            audit['profile']=profile_audit(c,capture/'deep-traces')
            if not audit['profile']['valid']:raise ValueError('Invalid foreground profile')
        if not audit['valid'] or not audit['dispatcher']['matches'] or audit!=load(tid+'-audit.json'):
            raise ValueError('Invalid/mismatched diagnostic work audit')
        report['trials'].append(dict(trial_id=tid,case=trial['case'],observer=c['observer'],audit=audit,
            foreground_ttft_s={k.rsplit('/',1)[-1]:v['first']-c['release'] for k,v in c['requests'].items() if v['role']=='foreground'}))
        if trial['stage']!='deep_diagnostic':continue
        proof=c['workers'][0]['deep_profile'];path=capture/'deep-traces'/proof['artifact']['name']
        events=json.loads(path.read_text())['traceEvents'];correlation=correlate(events,proof['step_ids'])
        by_id={s['step_id']:s for s in workers}
        for row in correlation['steps']:
            worker=by_id[row['step_id']]
            row.update(runtime_mode=worker['runtime_mode'],actual_tokens=worker['actual_tokens'],
                cuda_event_elapsed_s=worker['gpu_forward_s'],worker_host_elapsed_s=worker['host_end']-worker['host_start'])
        report['profiles'].append(dict(trial_id=tid,case=trial['case'],trace_sha256=proof['artifact']['sha256'],
            categories=dict(Counter(e.get('cat','metadata') for e in events)),**correlation))
    save(output/'analysis.json',report)
    save(output/'input-files.json',[dict(path=str(p.relative_to(capture)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(capture.rglob('*')) if p.is_file()])
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams['svg.hashsalt']='pr32-foreground-diagnostic'
    fig, axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    for ax,p in zip(axes,report['profiles']):
        origin=p['origin_us'];ms=lambda x:(x-origin)/1000
        for i,row in enumerate(p['steps']):
            ax.broken_barh([(ms(row['cpu_start_us']),row['cpu_duration_us']/1000)],(2.7,.55),facecolors='#3478b8')
            ax.broken_barh([(ms(row['gpu_start_us']),row['gpu_envelope_us']/1000)],(1.7,.55),facecolors='#2c9873')
            ax.text(ms(row['cpu_start_us']),3.4,str(i+11),fontsize=8)
        for e in p['outside_forward_timeline']:
            y,color=(.7,'#d19324') if e['category']!='gpu_memcpy' else (-.3,'#9d68ac')
            ax.vlines(ms(e['start_us']),y,y+.55,color=color,linewidth=.65)
        for e in p['device_synchronizations']:
            ax.axvspan(ms(e['start_us']),ms(e['start_us']+e['duration_us']),color='#d64d4d',alpha=.13)
        ax.set(title=p['case']+' (worker steps 11–15)',yticks=[3,2,1,0],
            yticklabels=['CPU forward','GPU forward envelope','Other GPU activity starts','DtoH copy starts'],
            ylim=(-.5,3.9),xlabel='Time from first foreground CPU marker (ms)')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Foreground traces: forward timing omits later token/output work',fontsize=14)
    fig.legend(handles=[Patch(color='#d64d4d',alpha=.2,label='Observed profiler-stop device synchronization')],loc='outside lower center')
    for suffix in ('png','svg'):
        path=Path(output)/('foreground-timeline.'+suffix)
        fig.savefig(path,dpi=150,metadata={'Date':None} if suffix=='svg' else None)
        if suffix=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--plot',action='store_true')
    a=p.parse_args();result=analyze(a.capture,a.output,a.archive)
    if a.plot:plot(result,a.output)
    print(json.dumps(dict(source_files_verified=result['source_files_verified'],trials=len(result['trials']),profiles=len(result['profiles']),model_accepted=False)))
