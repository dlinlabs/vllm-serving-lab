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
LEAF_STAGES = ('forward','logits_sample','copy_enqueue','post_update')


def correlate_stages(events, step_ids, pipeline_rows):
    """Partition GPU launches into non-nested stages; never sum parent/wait spans."""
    by_id={r['step_id']:r for r in pipeline_rows}
    if len(set(step_ids))!=len(step_ids):raise ValueError('Duplicate stage-profile step ID')
    markers={}
    for key in step_ids:
        for stage in LEAF_STAGES:
            name=PREFIX+key if stage=='forward' else 'prefill_lab_stage/'+stage+'/'+key
            found=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')==name]
            if len(found)!=1 or found[0]['dur']<=0:raise ValueError('Missing/duplicate stage CPU marker')
            markers[(key,stage)]=found[0]
    apis={}
    for e in events:
        if e.get('cat') not in ('cuda_runtime','cuda_driver') or 'correlation' not in e.get('args',{}):continue
        key=e['args']['correlation']
        if key in apis:raise ValueError('Ambiguous stage launch correlation')
        apis[key]=e
    groups=defaultdict(list)
    for e in events:
        if e.get('cat') not in GPU_CATEGORIES:continue
        api=apis.get(e.get('args',{}).get('correlation'))
        if api is None:raise ValueError('Unmatched stage GPU launch')
        owners=[key for key,m in markers.items() if (api['pid'],api['tid'])==(m['pid'],m['tid'])
                and m['ts']<=api['ts']<m['ts']+m['dur']]
        if len(owners)!=1:raise ValueError('GPU launch outside/ambiguous leaf-stage markers')
        groups[owners[0]].append(e)
    rows=[]
    for key in step_ids:
        leaf={};record=by_id[key]
        for stage in LEAF_STAGES:
            gpu=groups[(key,stage)]
            if not gpu:raise ValueError('Leaf stage lacks GPU activity')
            a=min(e['ts'] for e in gpu);b=max(e['ts']+e['dur'] for e in gpu)
            leaf[stage]=dict(start_us=a,envelope_us=b-a,activity_union_us=union_us(gpu),
                cpu_start_us=markers[(key,stage)]['ts'],cpu_duration_us=markers[(key,stage)]['dur'],
                activity_counts=dict(Counter(e['cat'] for e in gpu)),
                runtime_stream_handle=record['stages'][stage]['stream_handle'],
                cupti_streams=sorted({e['args']['stream'] for e in gpu}),
                event_elapsed_s=record['stages'][stage]['gpu_elapsed_s'])
        copies=groups[(key,'copy_enqueue')];updates=groups[(key,'post_update')]
        # Inclusion/exclusion gives union intersection, without double counting.
        overlap=max(0.,union_us(copies)+union_us(updates)-union_us(copies+updates))
        rows.append(dict(step_id=key,stages=leaf,copy_post_update_overlap_us=overlap,
            nested_sample_tokens_event_s=record['stages']['sample_tokens']['gpu_elapsed_s'],
            primary_copy_bytes=record['primary_copy_bytes'],extra_copy_payloads=record['extra_copy_payloads']))
    return dict(step_count=len(rows),steps=rows,
        gpu_activity_count=sum(len(es) for es in groups.values()),unmatched_gpu_activities=0,
        leaf_activity_counts={stage:dict(Counter(e['cat'] for (key,s),es in groups.items() if s==stage for e in es)) for stage in LEAF_STAGES},
        cuda_scope_qualified=False,model_accepted=False,
        interpretation='Observed launch-to-leaf attribution only; elapsed event endpoints and unprofiled work are not independently qualified')


def stage_statistics(capture):
    import statistics
    rows=capture['workers'][0]['pipeline']['rows']
    workers={r['step_id']:r for r in capture['workers'][0]['rows']}
    stages={}
    for name in rows[0]['stages']:
        host=[r['stages'][name]['host_end']-r['stages'][name]['host_start'] for r in rows]
        gpu=[r['stages'][name].get('gpu_elapsed_s') for r in rows]
        item=dict(cpu_median_s=statistics.median(host),cpu_max_s=max(host))
        if gpu[0] is not None:
            index=max(range(len(rows)),key=lambda i:gpu[i])
            item.update(event_median_s=statistics.median(gpu),event_max_s=max(gpu),max_event_worker_step=index+1,
                        max_event_mode=workers[rows[index]['step_id']]['runtime_mode'])
        stages[name]=item
    return dict(step_count=len(rows),cpu_spans=sum(len(r['stages']) for r in rows),
        cuda_spans=sum('gpu_elapsed_s' in s for r in rows for s in r['stages'].values()),
        modes=dict(Counter(w['runtime_mode'] for w in workers.values())),
        extra_copy_payload_steps=sum(r['extra_copy_payloads'] for r in rows),stages=stages)


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


def correlate(events, step_ids, *, sampling_profiled=False):
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
        device_synchronizations=syncs,profile_stop_precedes_fifth_sampling=not sampling_profiled,
        cuda_scope_qualified=False,model_accepted=False,
        limitations=['Only the five captured calls; absence outside this window is not evidence of absence.',
                    'Fifth-call sampling follows profiler.stop and is not captured.' if not sampling_profiled
                    else 'Fifth sampling/copy enqueue is recorded; CPU output reap/commit may lie outside the profile window.',
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
    stages=launch['scope']=='targeted_stage_diagnostic_only'
    if launch['scope'] not in ('targeted_foreground_diagnostic_only','targeted_stage_diagnostic_only') or summary['scope']!=launch['scope'] or summary['planned']!=expected or summary['completed']!=expected:
        raise ValueError('Targeted diagnostic order/scope mismatch')
    if stages:
        from prefill_lab.pipeline import preflight
        preflight(meta)
        if load('stage-protocol.json')!=json.loads(source('experiments/prefill-stage-diagnostic-v1.json')):
            raise ValueError('Stage protocol does not match its source-bound development plan')
    if not summary['diagnostic_complete'] or summary['failures']:
        raise ValueError('Incomplete diagnostic')
    if any(summary.get(k) is not False for k in ('cuda_scope_qualified','model_accepted','gpu_validated','complete_experiment_passed')):
        raise ValueError('Diagnostic incorrectly claims broader acceptance')
    report = dict(capture_commit=ref,archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files_verified=len(declared),installed_source_files=len(pins),
        recorded_native_cpu_acceptance={k:acceptance[k] for k in ('valid','tests','tracked_clean')},
        original_junit_in_archive=False,scope='Offline diagnostic identity/work/profile/correlation audit only',
        diagnostic_complete=True,diagnostic_scope=launch['scope'],elapsed_seconds=summary['elapsed_seconds'],
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
        if stages:
            from prefill_lab.pipeline import audit as pipeline_audit
            audit['pipeline']=pipeline_audit(c,capture/'deep-traces')
            if not audit['pipeline']['valid']:raise ValueError('Invalid pipeline-stage capture')
        if not audit['valid'] or not audit['dispatcher']['matches'] or audit!=load(tid+'-audit.json'):
            raise ValueError('Invalid/mismatched diagnostic work audit')
        trial_report=dict(trial_id=tid,case=trial['case'],observer=c['observer'],audit=audit,
            foreground_ttft_s={k.rsplit('/',1)[-1]:v['first']-c['release'] for k,v in c['requests'].items() if v['role']=='foreground'})
        if stages:trial_report['pipeline_statistics']=stage_statistics(c)
        report['trials'].append(trial_report)
        if trial['stage']!='deep_diagnostic':continue
        proof=c['workers'][0]['deep_profile'];path=capture/'deep-traces'/proof['artifact']['name']
        events=json.loads(path.read_text())['traceEvents'];correlation=correlate(events,proof['step_ids'],sampling_profiled=stages)
        by_id={s['step_id']:s for s in workers}
        for row in correlation['steps']:
            worker=by_id[row['step_id']]
            row.update(runtime_mode=worker['runtime_mode'],actual_tokens=worker['actual_tokens'],
                cuda_event_elapsed_s=worker['gpu_forward_s'],worker_host_elapsed_s=worker['host_end']-worker['host_start'])
        profile_report=dict(trial_id=tid,case=trial['case'],trace_sha256=proof['artifact']['sha256'],
            first_foreground_worker_step=audit['profile']['first_foreground_worker_step'],
            categories=dict(Counter(e.get('cat','metadata') for e in events)),**correlation)
        if stages:
            profile_report['leaf_stages']=correlate_stages(events,proof['step_ids'],c['workers'][0]['pipeline']['rows'])
            profile_report['profiler_stop_span']=proof['profile_stop_span']
        report['profiles'].append(profile_report)
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
        for i,row in enumerate(p['steps'],p.get('first_foreground_worker_step',11)):
            ax.broken_barh([(ms(row['cpu_start_us']),row['cpu_duration_us']/1000)],(2.7,.55),facecolors='#3478b8')
            ax.broken_barh([(ms(row['gpu_start_us']),row['gpu_envelope_us']/1000)],(1.7,.55),facecolors='#2c9873')
            ax.text(ms(row['cpu_start_us']),3.4,str(i),fontsize=8)
        for e in p['outside_forward_timeline']:
            y,color=(.7,'#d19324') if e['category']!='gpu_memcpy' else (-.3,'#9d68ac')
            ax.vlines(ms(e['start_us']),y,y+.55,color=color,linewidth=.65)
        for e in p['device_synchronizations']:
            ax.axvspan(ms(e['start_us']),ms(e['start_us']+e['duration_us']),color='#d64d4d',alpha=.13)
        first=p.get('first_foreground_worker_step',11)
        ax.set(title=p['case']+f' (worker steps {first}–{first+4})',yticks=[3,2,1,0],
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


def plot_stages(report, output):
    """Common-clock trace spans plus an explicitly separate duration comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams['svg.hashsalt']='pr32-stage-wait-audit'
    colors=dict(forward='#3478b8',logits_sample='#df982a',copy_enqueue='#9d68ac',post_update='#2c9873')
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),gridspec_kw={'width_ratios':[1.6,1]},layout='constrained')
    for (left,right),p in zip(axes,report['profiles']):
        origin=p['origin_us'];ms=lambda x:(x-origin)/1000
        for step in p['leaf_stages']['steps']:
            for name,s in step['stages'].items():
                left.broken_barh([(ms(s['cpu_start_us']),s['cpu_duration_us']/1000)],(2.6,.55),facecolors=colors[name])
                if name=='copy_enqueue':left.vlines(ms(s['start_us']),.6,1.15,color=colors[name],linewidth=.8)
                else:left.broken_barh([(ms(s['start_us']),s['envelope_us']/1000)],(1.6,.55),facecolors=colors[name])
        for e in p['device_synchronizations']:
            left.axvspan(ms(e['start_us']),ms(e['start_us']+e['duration_us']),color='#d64d4d',alpha=.12)
        left.set(title=p['case'],yticks=[2.85,1.85,.85],yticklabels=['CPU enqueue ranges','GPU main envelopes','GPU DtoH starts'],
            ylim=(.3,3.6),xlabel='Time from first foreground CPU marker (ms)')
        left.grid(axis='x',alpha=.2)
        example=p['leaf_stages']['steps'][2];s=example['stages']['copy_enqueue']
        values=[s['event_elapsed_s']*1000,s['envelope_us']/1000,s['activity_union_us']/1000]
        bars=right.barh(['Copy event span','Transfer envelope','Transfer active union'],values,
            color=[colors['copy_enqueue'],'#777777','#aaaaaa'],log=True)
        right.bar_label(bars,labels=[f'{v:.3f} ms' if v>1 else f'{v:.6f} ms' for v in values],fontsize=8,padding=4)
        number=p.get('first_foreground_worker_step',11)+2
        right.invert_yaxis();right.set(xlim=(.0005,10000),xlabel='Duration (ms, logarithmic scale)',
            title=f'Worker step {number}: {example["primary_copy_bytes"]} B primary copy')
        right.grid(axis='x',alpha=.2)
    fig.suptitle('Stage trace audit: copy timing includes waiting, rather than only data transfer',fontsize=13)
    fig.legend(handles=[Patch(color=c,label=n.replace('_',' ')) for n,c in colors.items()]+[
        Patch(color='#d64d4d',alpha=.2,label='Profiler-stop device synchronization')],loc='outside lower center',ncol=5,fontsize=8)
    for suffix in ('png','svg'):
        path=Path(output)/('stage-timing.'+suffix)
        fig.savefig(path,dpi=150,metadata={'Date':None} if suffix=='svg' else None)
        if suffix=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--plot',action='store_true')
    a=p.parse_args();result=analyze(a.capture,a.output,a.archive)
    if a.plot:plot(result,a.output)
    if a.plot and result['diagnostic_scope']=='targeted_stage_diagnostic_only':plot_stages(result,a.output)
    print(json.dumps(dict(source_files_verified=result['source_files_verified'],trials=len(result['trials']),profiles=len(result['profiles']),model_accepted=False)))
