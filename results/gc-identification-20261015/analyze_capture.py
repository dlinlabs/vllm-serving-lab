import argparse, json, hashlib, math, bisect
from pathlib import Path
from collections import Counter
import numpy as np

parser=argparse.ArgumentParser(description="Recompute matched-window findings from the complete uploaded capture, without a GPU.")
parser.add_argument('--capture-root',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
BASE=Path(__file__).resolve().parents[2]
RAW=args.capture_root.resolve()
OUT=args.output_dir.resolve()
if OUT==RAW or RAW in OUT.parents:
    parser.error('Output must be outside the original capture')
if OUT.exists() and any(OUT.iterdir()):
    parser.error('Output directory must be new or empty')
OUT.mkdir(exist_ok=True)
NAMES=['gc-defer-fixed-96-r1','gc-defer-fixed-64-r1','gc-defer-step-96-64-96-r1']
def read(p): return json.loads(p.read_text())
def lines(p):
    with p.open() as f: return [json.loads(s) for s in f if s.strip()]
def stat(vals):
    a=np.array(vals,float)
    return {'n':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),'p99':float(np.quantile(a,.99)),'max':float(a.max())} if len(a) else None
def ov(a,b,c,d): return max(0,min(b,d)-max(a,c))
def metric(e,k,suffix=None):
    m=e['metrics'][k]
    if not m.get('available'):return None
    vals=[s['value'] for s in m['samples'] if suffix is None or s['name'].endswith(suffix)]
    return sum(vals) if vals else None
def window(run,a,b):
    rows=[x for x in run['rows'] if a<=x['scheduled_s']<b]
    good=[x for x in rows if x['success'] and x['scheduled_ttft']<=.5]
    succ=[x for x in rows if x['success']]
    e=[x for x in run['engine'] if a<=x['received_at']-run['start']<b]
    states=[x['metrics'] for x in run['states'] if a<=x['metrics']['state_sampled_at']-run['start']<b]
    return dict(start=a,end=b,offered=len(rows),input_tokens=sum(x['input_tokens'] for x in rows),
        offered_input_mean=sum(x['input_tokens'] for x in rows)/len(rows) if rows else None,
        success=len(succ),rejected=sum(x['rejected'] for x in rows),slo=len(good),goodput=len(good)/(b-a),
        ttft=stat([x['scheduled_ttft'] for x in succ]),e2e=stat([x['scheduled_e2e'] for x in succ]),
        admissions=sum(a<=x['admitted_s']<b for x in run['rows'] if x['admitted_s'] is not None),
        completions=sum(a<=x['gateway_terminal_s']<b for x in run['rows'] if x['success']),
        admitted_tokens=sum(x['input_tokens'] for x in run['rows'] if x['admitted_s'] is not None and a<=x['admitted_s']<b),
        awaiting=stat([x['awaiting_first_content'] for x in states]),streaming=stat([x['streaming_requests'] for x in states]),
        cost=stat([x['current_admitted_cost'] for x in states]),
        engine_waiting=stat([metric(x,'waiting') for x in e]),engine_running=stat([metric(x,'running') for x in e]))

result={'archive_sha256':'08901e9e18c180d53a1008f21e989708205faa777d6af40b01b2899fe5ea3eec','files':{},'runs':{}}
for p in RAW.rglob('*'):
    if p.is_file():result['files'][str(p.relative_to(RAW))]=hashlib.sha256(p.read_bytes()).hexdigest()
data={}
saved=read(RAW/'results/analysis-gc-defer-ident-r1/summary.json')
for name in NAMES:
    root=RAW/'results'/name
    m=read(root/'manifest.json');start=m['measurement_start']
    rows=lines(RAW/'results/analysis-gc-defer-ident-r1'/f'{name}-requests.jsonl')
    # Independently reconcile all cohort labels against raw client and prepared trace.
    clients=read(root/'client.json')['requests'];trace=read(root/'trace.json')['requests']
    cl={x['trace_id']:x for x in clients}
    ledger={x['trace_id']:x for x in rows}
    assert len(cl)==len(ledger)==len(trace)==7643
    for t in trace:
        c,l=cl[t['trace_id']],ledger[t['trace_id']]
        assert c['success']==l['success'] and c['rejected']==l['rejected']
        assert abs((c['scheduled_at']-start)-l['scheduled_s'])<1e-8
        assert abs(l['scheduled_s']-t['arrival_seconds'])<1e-8
        assert l['input_tokens']==t['input_tokens']
        if c['success']:
            assert abs(c['first_content_at']-c['scheduled_at']-l['scheduled_ttft'])<1e-8
            assert (l['scheduled_ttft']<=.5)==(l['outcome']=='successful_within_ttft')
    engine=lines(root/'engine_observations.jsonl')
    events=lines(root/'gateway.jsonl')
    states=read(root/'state_observations.json')
    run=dict(rows=rows,engine=engine,events=events,states=states,start=start,manifest=m)
    data[name]=run
    summary=next(x for x in saved['runs'] if x['name']==name)
    gens=[x for x in events if x['event']=='gc_span' and x['generation']==2 and 0<=x['span_start']-start<300]
    stalls=summary['large_monitor_delays']
    stall_gc=[]
    for s in stalls:
        spans=[x for x in gens if ov(s['due_s'],s['resumed_s'],x['span_start']-start,x['span_end']-start)>0]
        stall_gc.append(dict(due=s['due_s'],resumed=s['resumed_s'],lag=s['lag_s'],gen2_overlap=sum(ov(s['due_s'],s['resumed_s'],x['span_start']-start,x['span_end']-start) for x in spans)))
    delta={}
    for k in ('queue_time','prefill_time','decode_time'):
        count=metric(engine[-1],k,'_count')-metric(engine[0],k,'_count')
        sec=metric(engine[-1],k,'_sum')-metric(engine[0],k,'_sum')
        delta[k]={'count':count,'sum':sec,'mean':sec/count if count else None}
    g=read(root/'runner_gc_control.json')
    rr=lines(root/'runner_observations.jsonl')
    active_gc=[x for x in rr if x['event']=='gc_span' and ov(g['disabled_at'],g['restore_started_at'],x['span_start'],x['span_end'])>0]
    result['runs'][name]=dict(report=read(root/'report.json'),
        provenance={'host':m['host'],'commit':m['git_commit'],'dirty':m['git_dirty'],'gpu':m['gpu'],'code_matches':{k:hashlib.sha256((BASE/k).read_bytes()).hexdigest()==v for k,v in m['code_sha256'].items()},'backend':m['backend_launch_provenance'],'packages':m['packages'],'trace_sha256':m['trace_sha256']},
        segments=[window(run,a,a+100) for a in (0,100,200)],
        windows10=[window(run,a,a+10) for a in range(0,300,10)],
        stalls=stall_gc,gateway_gen2=[dict(start=x['span_start']-start,end=x['span_end']-start,duration=x['span_end']-x['span_start'],cpu=x['callback_thread_cpu_seconds'],collected=x['collected']) for x in gens],
        slowtail=summary['slowest_one_percent'],engine_delta=delta,
        engine_extrema={k:stat([metric(x,k) for x in engine]) for k in ('waiting','running','kv_cache')},
        preemptions_delta=metric(engine[-1],'preemptions')-metric(engine[0],'preemptions'),
        successful_output_tokens=Counter(x['output_tokens'] for x in rows if x['success']),
        runner_active_gc_count=len(active_gc),runner_gc_spans=[dict(start=x['span_start']-start,end=x['span_end']-start,generation=x['generation']) for x in rr if x['event']=='gc_span'],
        gateway_gc_overlap_fraction=sum(x['gen2_overlap'] for x in stall_gc)/sum(x['lag'] for x in stall_gc))

step=data[NAMES[2]];start=step['start']
changes=read(RAW/'results'/NAMES[2]/'budget_changes.json')
result['transitions']=[]
for c in changes[1:]:
    t=c['applied_at']-start
    nearby={n:[window(run,t+a,t+b) for a,b in [(-10,0),(0,2),(2,5),(5,10),(10,20),(20,40)]] for n,run in data.items()}
    result['transitions'].append(dict(application_second=t,command=c,windows=nearby,
        step_states=[{'t':x['metrics']['state_sampled_at']-start,**{k:x['metrics'][k] for k in ('admission_budget','current_admitted_cost','awaiting_first_content','streaming_requests','active_requests','oldest_awaiting_first_seconds')}} for x in step['states'] if t-3<=x['metrics']['state_sampled_at']-start<t+12]))
(OUT/'facts.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print('facts saved',OUT/'facts.json')
for name,r in result['runs'].items():
    print('\n',name)
    print('segments',[(w['start'],w['offered'],w['success'],w['slo'],round(w['goodput'],3)) for w in r['segments']])
    print('gen2',[(round(x['start'],3),round(x['duration']*1000,2)) for x in r['gateway_gen2']])
    print('gc stall overlap',r['gateway_gc_overlap_fraction'])
    print('engine delta',r['engine_delta'],'max_wait',r['engine_extrema']['waiting']['max'],'preempt',r['preemptions_delta'])
for tr in result['transitions']:
    print('\nTRANSITION',tr['application_second'],'cost',tr['command']['current_admitted_cost'])
    for n,ws in tr['windows'].items():
        print(n,[(round(w['start'],1),round(w['end'],1),w['offered'],w['success'],w['slo'],w['admissions'],w['completions'],round(w['awaiting']['mean'],2) if w['awaiting'] else None) for w in ws])
    print('states',tr['step_states'])
