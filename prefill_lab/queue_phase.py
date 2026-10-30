"""Training-bound phase margins and separately scored measurement/model checks.

Exact piecewise-affine common-scale timing on the effective lane's OWN no-probe
graph. No finite sweep is used to certify the continuum. This narrow uncertainty
family is a hypothesis, not measured phase or probabilistic/tail coverage.
"""
import copy
import json
from pathlib import Path
import math
import hashlib

from . import event_clock_calibration as clock,effective_calibration as base,delayed_calibration as delayed
from .contracts import digest,finite

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'experiments/prefill-queue-phase-v1.json'


def policy():return json.loads(POLICY.read_text())


def protocol():
    p=copy.deepcopy(clock.protocol());spec=policy()
    p.update(schema='queue-phase-confirmation-v1',scope=spec['scope'],queue_phase_policy_sha256=digest(spec))
    for t in p['order']:
        t['trial_id']=spec['trial_prefix']+t['trial_id'].split('-',1)[1]
        t['paired_content_key']=spec['content_prefix']+t['paired_content_key']
    return p


def validate_manifest(manifest):
    from .tokens import validate
    p=protocol()
    if (manifest['model'],manifest['revision'])!=(p['model']['name'],p['model']['revision']):raise ValueError('Phase model/revision differs')
    return validate(manifest,p['cases'],p['order'])


# Curves contain (left,right,slope,intercept), with exact linear crossings.
def value(curve,x):
    for l,r,a,b in curve:
        if l-1e-12<=x<=r+1e-12:return a*x+b
    raise ValueError('Scale outside curve domain')


def combine(f,g,maximum=False):
    points=sorted({v for c in (f,g) for l,r,a,b in c for v in (l,r)});out=[]
    for l,r in zip(points,points[1:]):
        m=(l+r)/2;a,b=next((a,b) for x,y,a,b in f if x<=m<=y);c,d=next((a,b) for x,y,a,b in g if x<=m<=y)
        cuts=[l,r]
        if maximum and a!=c:
            cross=(d-b)/(a-c)
            if l<cross<r:cuts=[l,cross,r]
        for x,y in zip(cuts,cuts[1:]):
            mid=(x+y)/2
            A,B=((a,b) if a*mid+b>=c*mid+d else (c,d)) if maximum else (a-c,b-d)
            if out and out[-1][2:]==(A,B):out[-1]=(out[-1][0],y,A,B)
            else:out.append((x,y,A,B))
    return out


def commit_node(reference,at):
    found=[n['id'] for n in reference['nodes'] if n['name']=='commit' and n['end_s']==at]
    if len(found)!=1:raise ValueError('Ambiguous reference commit anchor')
    return found[0]


def timing(reference,spread,spec):
    lo,hi=1-spread,1+spread;zero=[(lo,hi,0.,0.)];ends=[];starts=[]
    for n in reference['nodes']:
        start=zero
        for dep in n['dependencies']:start=combine(start,ends[dep],True)
        duration=n['service_s']
        if duration and n['name'] not in ('native_idle_yield','forward.gpu_tail'):
            raise ValueError('Only the effective lane plus fixed native idle yield is supported')
        slope,intercept=(0.,duration) if n['name']=='native_idle_yield' else (duration,0.)
        end=[(l,r,a+slope,b+intercept) for l,r,a,b in start]
        # Busy zero-offset reference has no external readiness. Fail rather than
        # silently discard an unsupported dynamic ready value/dependency.
        if not math.isclose(value(start,1.),n['start_s'],abs_tol=1e-9,rel_tol=1e-9) or not math.isclose(value(end,1.),n['end_s'],abs_tol=1e-9,rel_tol=1e-9):
            raise ValueError('Graph readiness is not represented by dependency/resource order')
        if len(end)>spec['maximum_curve_pieces']:raise ValueError('Phase curve complexity bound')
        starts.append(start);ends.append(end)
    release=zero if reference['release_s']==0 else ends[commit_node(reference,reference['release_s'])]
    polls=[(n['batch'],combine(starts[n['id']],release)) for n in reference['nodes'] if n['name']=='schedule']
    return polls,release,ends


def phase_cell(case,theta,plan,runtime,point,spec,spread):
    if any(case['foreground_offsets_ms']) or not case['sentinel']:raise ValueError('Phase reference needs a zero-offset burst and one delayed probe')
    no_probe=dict(case,sentinel=None)
    ref=clock.rollouts(theta,dict(plan,cases=[no_probe]),runtime,clock.specification())[case['id']]
    arrival=case['sentinel']['arrival_ms']/1000.;nominal=[(n['batch'],n['start_s']-ref['release_s']) for n in ref['nodes'] if n['name']=='schedule']
    nominal_j=next(j for j,t in nominal if t>=arrival)
    nominal_times=dict(nominal)
    if arrival>=max(r['finished_s'] for r in ref['requests'].values() if r['role']!='background')-ref['release_s']:
        raise ValueError('Probe arrives after reference foreground completion')
    if not math.isclose(max(0.,nominal_times[nominal_j]-arrival),point,abs_tol=1e-9,rel_tol=1e-9):raise ValueError('No-probe poll does not reproduce independent point rollout')
    if spread==0:return dict(status='point',branches=[dict(iteration=nominal_j,queue_s=point)],scale_regions=[],nominal_queue_s=point)
    polls,release,ends=timing(ref,spread,spec);lo,hi=1-spread,1+spread
    done=max(r['finished_s'] for r in ref['requests'].values() if r['role']!='background')
    done_node=commit_node(ref,done)
    relative_done=combine(ends[done_node],release)
    if arrival>=min(a*x+b for l,r,a,b in relative_done for x in (l,r)):raise ValueError('Probe can arrive after reference foreground completion')
    cuts={lo,hi}
    for _,curve in polls:
        for l,r,a,b in curve:
            cuts.update((l,r))
            if a:
                root=(arrival-b)/a
                if l<=root<=r:cuts.add(root)
    cuts=sorted(cuts)
    if len(cuts)>spec['maximum_partition_points']:raise ValueError('Phase partition complexity bound')
    consumed=lambda x:next((j for j,c in polls if value(c,x)>=arrival-1e-12),None)
    regions=[dict(lower=l,upper=r,iteration=consumed((l+r)/2)) for l,r in zip(cuts,cuts[1:])]
    ids={consumed(x) for x in cuts}|{r['iteration'] for r in regions}
    if None in ids or not ids or len(ids)>spec['maximum_branches']:raise ValueError('Unsupported/drained/more-than-two phase branches')
    branches=[dict(iteration=j,queue_s=max(0.,nominal_times[j]-arrival)) for j in sorted(ids)]
    if len({b['queue_s'] for b in branches})!=len(branches):raise ValueError('Collapsed phase branch points')
    return dict(status='phase_ambiguous' if len(branches)>1 else 'point',branches=branches,scale_regions=regions,
        scale_boundary_assignments=[dict(scale=x,iteration=consumed(x)) for x in cuts],nominal_queue_s=point)


def calibration_spread(rows,plan,predicted):
    training=base.select(rows,plan,{'train'},'off');samples=[];spread=0.
    for r in training:
        ratios=[v/predicted[r['case']]['endpoints'][rid][metric] for rid,e in r['endpoints'].items() for metric,v in e.items()]
        if not ratios or not all(finite(v) and v>0 for v in ratios):raise ValueError('Invalid training clock ratios')
        error=max(abs(v-1) for v in ratios);spread=max(spread,error)
        samples.append(dict(trial_id=r['trial_id'],minimum_ratio=min(ratios),maximum_ratio=max(ratios),max_absolute_relative_residual=error))
    return (0. if spread<1e-12 else spread),samples,training


def freeze(rows,plan,runtime,model,predicted):
    spec=policy()
    if set(spec['timing_source_sha256'])!={'prefill_lab/async_predictor.py','prefill_lab/event_clock_calibration.py',
            'prefill_lab/effective_calibration.py','prefill_lab/causal.py','prefill_lab/contracts.py','scripts/predict_prefill_async.py'} or any(
            hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha for name,sha in spec['timing_source_sha256'].items()):
        raise ValueError('Unreviewed timing source for analytic phase algebra')
    if spec['queue_absolute_s']!=plan['queue_absolute_s'] or spec['queue_relative']!=plan['queue_relative']:raise ValueError('Queue budgets changed')
    if digest(clock.predict_frozen(model,plan,runtime))!=digest(predicted):raise ValueError('Point prediction identity differs')
    spread,samples,training=calibration_spread(rows,plan,predicted)
    if digest(training)!=model['training_sha256']:raise ValueError('Phase training differs from model training')
    cells={};errors=[]
    if spread>spec['maximum_scale_half_width']:errors.append('Calibration scale exceeds declared maximum')
    else:
        for case in plan['cases']:
            if not case['sentinel']:continue
            try:cells[case['id']]=phase_cell(case,model['parameters_s'],plan,runtime,predicted[case['id']]['sentinel_queue_residence_s'],spec,spread)
            except (ValueError,StopIteration) as exc:errors.append(case['id']+': '+str(exc))
    result=dict(schema='queue-phase-freeze-v1',policy_sha256=digest(spec),plan_sha256=digest(plan),runtime_sha256=digest(runtime),
        model_sha256=model['sha256'],point_predictions_sha256=digest(predicted),training_sha256=digest(training),
        scale_half_width=spread,calibration_samples=samples,cells=cells,readiness_passed=not errors,errors=errors,
        held_labels_used=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    result['sha256']=digest(result);return result


def assign(observed,cell,spec):
    if not finite(observed) or observed<0:raise ValueError('Invalid queue observation')
    errors=[abs(observed-b['queue_s']) for b in cell['branches']];best=min(errors)
    ids=[i for i,e in enumerate(errors) if abs(e-best)<1e-12]
    limit=max(spec['queue_absolute_s'],spec['queue_relative']*observed)
    return dict(branch=ids[0] if len(ids)==1 else None,observed_s=observed,error_s=best,limit_s=limit,passed=best<=limit)


def evaluate(rows,plan,runtime,model,predicted,frozen):
    spec=policy()
    # Rebuild branch eligibility and points using TRAIN only. A rehashed wider
    # branch set cannot manufacture coverage after seeing held observations.
    if digest(frozen)!=digest(freeze(rows,plan,runtime,model,predicted)):
        raise ValueError('Phase freeze/training/plan/runtime identity differs')
    if not frozen['readiness_passed']:raise ValueError('Phase readiness failed')
    point_report=delayed.score(rows,plan,predicted,model['sha256'])
    off=delayed.select(rows,plan,delayed.HELD,'off');light=delayed.select(rows,plan,delayed.HELD,'light')
    pairs={(r['stage'],r['case'],r['block']):r for r in light};checks=[];observer=[];groups={};counts={}
    for r in off:
        name=r['case']
        if name not in frozen['cells']:continue
        cell=frozen['cells'][name];other=pairs[(r['stage'],name,r['block'])]
        a=assign(r['clocks']['sentinel']['queue_residence_s'],cell,spec);b=assign(other['clocks']['sentinel']['queue_residence_s'],cell,spec)
        counts.setdefault(name,{mode:{str(i):0 for i in range(len(cell['branches']))}|{'unresolved':0} for mode in ('off','light')})
        for row,result in ((r,a),(other,b)):
            checks.append(dict(trial_id=row['trial_id'],case=name,observer=row['observer'],phase_status=cell['status'],**result))
            counts[name][row['observer']][str(result['branch']) if result['branch'] is not None else 'unresolved']+=1
        delta=abs(a['observed_s']-b['observed_s']);limit=max(spec['queue_absolute_s'],spec['queue_relative']*a['observed_s'])
        if a['branch'] is None or b['branch'] is None:category='unresolved_assignment';passed=False
        elif a['branch']==b['branch']:category='same_phase_observer_shift' if delta>limit else 'same_phase_agreement';passed=delta<=limit
        else:category='expected_phase_difference' if a['passed'] and b['passed'] else 'unexplained_cross_phase';passed=category=='expected_phase_difference'
        observer.append(dict(trial_id=r['trial_id'],paired_trial=other['trial_id'],case=name,category=category,delta_s=delta,limit_s=limit,passed=passed))
        if a['branch'] is not None:groups.setdefault((name,a['branch']),[]).append(a['observed_s'])
    repeats=[]
    for (name,branch),values in sorted(groups.items()):
        median=sorted(values)[len(values)//2] if len(values)%2 else sum(sorted(values)[len(values)//2-1:len(values)//2+1])/2
        limit=max(spec['queue_absolute_s'],spec['queue_relative']*median);width=max(values)-min(values)
        repeats.append(dict(case=name,branch=branch,samples=len(values),assessable=len(values)>1,spread_s=width,limit_s=limit,passed=width<=limit))
    endpoints=point_report['quality'];endpoint_quality=all(r['passed'] for r in endpoints['paired_observer']+endpoints['repeat_spread'])
    model_passed=all(r['passed'] for r in checks) and all(r['passed'] for r in point_report['sentinel_queue_scores'] if frozen['cells'][r['case']]['status']=='point')
    measurement=all(r['passed'] for r in observer+repeats)
    base_passed=endpoint_quality and point_report['endpoint_scores']['passed'] and all(v['passed'] for v in point_report['anchor_phase_scores'].values()) and point_report['structure_and_background_lifetime_valid']
    return dict(schema='queue-phase-confirmation-report-v1',candidate_passed=base_passed and model_passed and measurement,
        endpoint_and_structure_passed=base_passed,queue_prediction_budget_passed=model_passed,measurement_qualification_passed=measurement,
        endpoint_scores=point_report['endpoint_scores'],anchor_phase_scores=point_report['anchor_phase_scores'],
        endpoint_quality=dict(passed=endpoint_quality,paired_observer=endpoints['paired_observer'],repeat_spread=endpoints['repeat_spread']),
        point_baseline_candidate_passed=point_report['candidate_passed'],raw_queue_quality=endpoints['queue_residence'],
        sample_checks=checks,observer_checks=observer,within_phase_repeat_checks=repeats,branch_counts=counts,
        unobserved_branches=[dict(case=case,observer=mode,branch=k) for case,modes in counts.items()
            for mode,branches in modes.items() for k,n in branches.items() if k!='unresolved' and n==0],
        underpowered_repeat_groups=[r for r in repeats if not r['assessable']],
        assignment_scope='nearest frozen model branch, not directly measured native phase; no validation for zero-count branches',
        frozen_sha256=frozen['sha256'],model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


def rejection_reasons(report):
    return ([dict(gate='endpoint_or_structure')] if not report['endpoint_and_structure_passed'] else [])+[
        dict(gate=gate,**r) for gate,key in (('queue_prediction_budget','sample_checks'),('queue_observer','observer_checks'),
        ('queue_within_phase_repeat','within_phase_repeat_checks')) for r in report[key] if not r['passed']]
