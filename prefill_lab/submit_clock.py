"""TRAIN-bound CPU-return sensitivity ablation; no physical/native acceptance.

The measured worker execute_model host span may contain waits and enqueue work.
Placing it in cpu_after is an explicit effective-model hypothesis, not a physical
stage identification. GPU coefficients remain fixed; no probe outcome is fitted.
"""
from dataclasses import dataclass,field
import statistics
import hashlib
from pathlib import Path
from . import event_clock_calibration as clock,queue_phase as phase
from .async_predictor import CostLaw,PipelineCosts,rollout
from .contracts import digest,finite


@dataclass(frozen=True)
class SubmitLaw(CostLaw):
    prefill_s:float=0.
    decode_s:float=0.

    def __post_init__(self):
        super().__post_init__()
        if not all(finite(v) and v>=0 for v in (self.prefill_s,self.decode_s)):raise ValueError('Invalid submit constants')

    def __call__(self,row):return self.prefill_s if row['prefill'] else self.decode_s


@dataclass(frozen=True)
class SubmitCosts(PipelineCosts):
    base_cost_document:dict=field(default_factory=dict)

    def to_dict(self):
        law=self.launches['forward']['cpu_after']
        return dict(schema='effective-submit-ablation-costs-v1',base=self.base_cost_document,
            forward_cpu_after=dict(rule='prefill>0 else pure decode',prefill_s=law.prefill_s,decode_s=law.decode_s))


def costs(theta,plan,prefill_s,decode_s):
    base=clock.costs(theta,plan,clock.specification());launches={k:dict(v) for k,v in base.launches.items()}
    launches['forward']['cpu_after']=SubmitLaw({},prefill_s,decode_s)
    return SubmitCosts(launches,dict(base.scalar),'development host-return hypothesis, not physical pipeline costs',base.to_dict())


def training_labels(captures,plan):
    wanted={t['trial_id']:t for t in plan['order'] if t['stage']=='train' and t['observer']=='light'}
    samples=[];medians=[];seen=set()
    for cap in captures:
        tid=cap.get('trial_id')
        if tid not in wanted:continue
        if tid in seen or cap['observer']!='light':raise ValueError('Duplicate/wrong TRAIN submit capture')
        if cap.get('status')!='complete' or cap.get('errors') or cap.get('overflow',0):raise ValueError('Invalid TRAIN submit capture')
        seen.add(tid);workers={}
        for w in cap['workers']:
            for row in w['rows']:
                if row['step_id'] in workers:raise ValueError('Duplicate worker step')
                workers[row['step_id']]=row
        groups={'prefill':[],'decode':[]}
        for s in cap['steps']:
            if s['end']>cap['foreground_done']:continue
            w=workers[s['step_id']];a,b=w['host_start'],w['host_end']
            if not all(finite(x) for x in (a,b,s['start'],s['end'])) or not s['start']<=a<=b<=s['end']:
                raise ValueError('Invalid TRAIN host-return clock')
            kind='prefill' if s['prefill'] else 'decode';span=b-a
            groups[kind].append(span)
            samples.append(dict(trial_id=tid,step_id=s['step_id'],kind=kind,host_span_s=span))
        if not all(groups.values()):raise ValueError('Missing TRAIN prefill/decode spans')
        medians.extend(dict(trial_id=tid,kind=k,median_s=statistics.median(v)) for k,v in groups.items())
    if seen!=set(wanted):raise ValueError('Missing TRAIN submit captures')
    if not samples:raise ValueError('Empty TRAIN submit evidence')
    prefill=[x['host_span_s'] for x in samples if x['kind']=='prefill']
    constants={k:statistics.median(x['median_s'] for x in medians if x['kind']==k) for k in ('prefill','decode')}
    lower,upper=min(prefill),max(prefill)
    if not 0<=lower<=constants['prefill']<=upper<=.1 or lower==upper:raise ValueError('Unsupported submit range')
    return dict(schema='submit-label-freeze-v1',constants_s=constants,prefill_lower_s=lower,prefill_upper_s=upper,
        training_sample_sha256=digest(samples),training_trial_ids=sorted(seen),trial_medians=sorted(medians,key=lambda r:(r['trial_id'],r['kind'])),
        samples=samples,rule='median of per-trial TRAIN LIGHT medians; prefill min/max uses all TRAIN prefix samples',
        probe_labels_used=False,physical_stage_identified=False,model_accepted=False)


def own_rollouts(theta,plan,runtime,prefill_s,decode_s):
    from scripts.predict_prefill_async import runtime_inputs
    scope,dispatch=runtime_inputs(runtime);model=costs(theta,plan,prefill_s,decode_s)
    return {c['id']:rollout(c,model,dispatch,scope=scope) for c in plan['cases']}


def reference_curves(reference,lower,upper,nominal,decode):
    if not all(finite(x) for x in (lower,upper,nominal,decode)) or not 0<=lower<upper<=.1 or not lower<=nominal<=upper or decode<0:
        raise ValueError('Unsupported submit curve domain')
    zero=[(lower,upper,0.,0.)];ends=[];starts=[]
    prefill={s['batch']:bool(s['features']['prefill']) for s in reference['steps']}
    for n in reference['nodes']:
        start=zero
        for dep in n['dependencies']:start=phase.combine(start,ends[dep],True)
        variable=n['name']=='forward.cpu_after' and prefill.get(n['batch'],False)
        a,b=(1.,0.) if variable else (0.,n['service_s'])
        end=[(l,r,x+a,y+b) for l,r,x,y in start]
        if abs(phase.value(start,nominal)-n['start_s'])>1e-8 or abs(phase.value(end,nominal)-n['end_s'])>1e-8:
            raise ValueError('Unsupported graph readiness')
        if len(end)>256:raise ValueError('Submit curve complexity')
        starts.append(start);ends.append(end)
    release=zero if reference['release_s']==0 else ends[phase.commit_node(reference,reference['release_s'])]
    return [(n['batch'],phase.combine(starts[n['id']],release)) for n in reference['nodes'] if n['name']=='schedule'],release,ends


def phase_cell(case,theta,plan,runtime,labels,point):
    nominal=labels['constants_s']['prefill'];decode=labels['constants_s']['decode']
    lower,upper=labels['prefill_lower_s'],labels['prefill_upper_s']
    if not case['sentinel'] or any(case['foreground_offsets_ms']):raise ValueError('Zero-offset burst reference required')
    one=dict(plan,cases=[dict(case,sentinel=None)])
    ref=own_rollouts(theta,one,runtime,nominal,decode)[case['id']]
    arrival=case['sentinel']['arrival_ms']/1000.
    polls,release,ends=reference_curves(ref,lower,upper,nominal,decode)
    done=max(r['finished_s'] for r in ref['requests'].values() if r['role']!='background')
    done_curve=phase.combine(ends[phase.commit_node(ref,done)],release)
    if arrival>=min(a*x+b for l,r,a,b in done_curve for x in (l,r)):raise ValueError('Probe after foreground completion')
    cuts={lower,upper}
    for _,curve in polls:
        for l,r,a,b in curve:
            cuts.update((l,r))
            if a:
                x=(arrival-b)/a
                if l<=x<=r:cuts.add(x)
    if len(cuts)>4096:raise ValueError('Submit partition complexity')
    cuts=sorted(cuts);consume=lambda x:next((j for j,c in polls if phase.value(c,x)>=arrival-1e-12),None)
    ids={consume(x) for x in cuts}|{consume((a+b)/2) for a,b in zip(cuts,cuts[1:])}
    if None in ids or not 1<=len(ids)<=2:raise ValueError('Unsupported submit phase branches')
    times={j:phase.value(c,nominal) for j,c in polls}
    nominal_point=max(0.,times[consume(nominal)]-arrival)
    if abs(nominal_point-point)>1e-8:raise ValueError('Reference point mismatch')
    branches=[dict(iteration=j,queue_s=max(0.,times[j]-arrival)) for j in sorted(ids)]
    if len({b['queue_s'] for b in branches})!=len(branches):raise ValueError('Collapsed submit phase points')
    return dict(status='phase_ambiguous' if len(branches)>1 else 'point',branches=branches,
        nominal_queue_s=point,parameter_boundary_assignments=[dict(prefill_submit_s=x,iteration=consume(x)) for x in cuts])


def freeze(captures,plan,runtime,model):
    clock.predict_frozen(model,plan,runtime)
    captures=list(captures)
    wanted={t['trial_id']:t for t in plan['order'] if t['stage']=='train' and t['observer']=='light'}
    cases={c['id']:c for c in plan['cases']}
    events=[clock.event_labels(c,wanted[c['trial_id']],cases[wanted[c['trial_id']]['case']]) for c in captures if c.get('trial_id') in wanted]
    if digest(clock.select_events(events,plan))!=model['training_events_sha256']:raise ValueError('Base model and submit TRAIN captures differ')
    labels=training_labels(captures,plan);g=labels['constants_s']
    predictions=clock.forecasts(own_rollouts(model['parameters_s'],plan,runtime,g['prefill'],g['decode']))
    cells={};errors=[]
    for case in plan['cases']:
        if not case['sentinel']:continue
        try:cells[case['id']]=phase_cell(case,model['parameters_s'],plan,runtime,labels,predictions[case['id']]['sentinel_queue_residence_s'])
        except (ValueError,StopIteration) as exc:errors.append(case['id']+': '+str(exc))
    result=dict(schema='submit-clock-development-freeze-v1',implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),labels=labels,cells=cells,errors=errors,readiness_passed=not errors,
        base_model_sha256=model['sha256'],plan_sha256=digest(plan),runtime_sha256=digest(runtime),predictions=predictions,
        probe_labels_used_for_fit=False,independent_confirmation_performed=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    result['sha256']=digest(result);return result


def verify(frozen,captures,plan,runtime,model):
    """Rebuild the entire TRAIN-bound hypothesis, not just a self hash."""
    rebuilt=freeze(captures,plan,runtime,model)
    if digest(frozen)!=digest(rebuilt):raise ValueError('Submit freeze/training/plan/runtime changed')
    return rebuilt
