"""Training-only development of an effective batch clock; no physical/control acceptance.

Native schedule/commit timestamps are absolute event labels relative to the
observed barrier. Rollouts subtract their OWN predicted barrier. Captured work
is an integrity comparator, never a prediction input. Overlapping spans are not
summed. LIGHT event labels are not qualified uninstrumented physical costs.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from . import effective_calibration as base
from .async_predictor import CostLaw, PipelineCosts, rollout, nonnegative
from .contracts import digest, finite

SPEC = Path(__file__).resolve().parents[1] / 'experiments/prefill-event-clock-v1.json'
PARAMETERS = (*base.PARAMETERS, 'decode', 'prefill_decode')


def specification():
    return json.loads(SPEC.read_text())


def validate_spec(spec):
    if (tuple(spec['parameters'])!=PARAMETERS
            or any(not finite(spec[k]) or spec[k]<=0 for k in
                ('decode_scale','mixed_scale','event_absolute_scale_s','event_relative_scale'))
            or len(spec['bounds_s'])!=2 or not all(finite(v) for v in spec['bounds_s'])
            or not 0<spec['bounds_s'][0]<spec['bounds_s'][1]
            or type(spec['fit_max_nfev']) is not int or spec['fit_max_nfev']<1
            or len(spec['starts_s'])<2 or any(len(x)!=len(PARAMETERS) or any(
                not finite(v) or not spec['bounds_s'][0]<v<spec['bounds_s'][1] for v in x) for x in spec['starts_s'])):
        raise ValueError('Invalid event-clock specification')


@dataclass(frozen=True)
class MixedCostLaw(CostLaw):
    """Additive interference from simultaneously scheduled prefill and decode."""
    mixed: float = 0.

    def __post_init__(self):
        super().__post_init__()
        nonnegative(self.mixed)

    def __call__(self, row):
        return nonnegative(super().__call__(row) + self.mixed * row['prefill'] * row['decode'])


@dataclass(frozen=True)
class EventClockCosts(PipelineCosts):
    def to_dict(self):
        # A separate schema prevents silently dropping the mixed term when
        # loading a generic four-parameter or physical-stage cost document.
        return dict(schema='effective-event-clock-costs-v1', base=super().to_dict(),
                    prefill_decode=self.launches['forward']['gpu_tail'].mixed)


def costs(theta, plan, spec):
    validate_spec(spec)
    if tuple(spec['parameters']) != PARAMETERS or len(theta) != len(PARAMETERS):
        raise ValueError('Wrong event-clock parameterization')
    if not all(finite(v) and v >= 0 for v in theta):raise ValueError('Invalid clock coefficient')
    lane = base.costs(theta[:4], plan)
    launches = {k: dict(v) for k,v in lane.launches.items()}
    coefficients = dict(launches['forward']['gpu_tail'].coefficients)
    coefficients['decode'] = float(theta[4]) / spec['decode_scale']
    launches['forward']['gpu_tail'] = MixedCostLaw(coefficients, float(theta[5]) / spec['mixed_scale'])
    return EventClockCosts(launches, lane.scalar,
        'effective LIGHT-event/OFF-endpoint development fit; not physical GPU/CPU costs')


def rollouts(theta, plan, runtime, spec):
    from scripts.predict_prefill_async import runtime_inputs
    scope, dispatch = runtime_inputs(runtime)
    if (scope.token_budget, scope.max_seqs, scope.max_model_len) != (2048,32,8192):
        raise ValueError('Event-clock candidate requires B2048/seq32/context8192')
    model = costs(theta,plan,spec)
    return {c['id']: rollout(c,model,dispatch,scope=scope) for c in plan['cases']}


def forecasts(predicted):
    result = {}
    for case,pred in predicted.items():
        result[case] = dict(endpoints={rid:dict(ttft_s=r['ttft_s'], completion_s=r['finished_s']-r['arrival_s'])
            for rid,r in pred['requests'].items() if r['role']!='background'},
            work_signature=base.work_signature(pred['steps']),
            background_outlives_foreground=pred['background_outlives_foreground'])
        if 'sentinel' in pred['requests']:
            result[case]['sentinel_queue_residence_s']=pred['requests']['sentinel']['input_queue_wait_s']
    return result


def event_labels(capture, trial, case):
    """Only TRAIN LIGHT, with native integrity and exact request/endpoints audited."""
    if trial['stage']!='train' or trial['observer']!='light':
        raise ValueError('Event fitting accepts training LIGHT only')
    row=base.extract(capture,trial,case)
    steps=capture['steps'];release=capture['release'];done=capture['foreground_done']
    starts=[s['start'] for s in steps];ends=[s['end'] for s in steps]
    if (not steps or not all(finite(v) for v in starts+ends) or starts!=sorted(starts)
            or ends!=sorted(ends) or any(a>b for a,b in zip(starts,ends))):
        raise ValueError('Invalid event chronology')
    # Retain the ordered prefix through the commit completing the foreground;
    # background-only drain after this point is not the prediction target.
    stop=[i+1 for i,s in enumerate(steps) if abs(s['end']-done)<1e-9]
    if len(stop)!=1:raise ValueError('Ambiguous foreground completion event')
    prefix=steps[:stop[0]]
    return dict(trial_id=trial['trial_id'],case=trial['case'],stage=trial['stage'],
        block=trial['block'],observer=trial['observer'],capture_sha256=row['capture_sha256'],
        work_signature=row['work_signature'],prefill=[bool(s['prefill']) for s in prefix],
        times_s=[[s['start']-release,s['end']-release] for s in prefix])


def select_events(rows, plan):
    wanted={t['trial_id']:t for t in plan['order'] if t['stage']=='train' and t['observer']=='light'}
    found={}
    for r in rows:
        tid=r['trial_id']
        if tid not in wanted:continue
        if tid in found:raise ValueError('Duplicate training event trial')
        t=wanted[tid]
        if any(r[k]!=t[k] for k in ('case','stage','block','observer')):
            raise ValueError('Event split membership mismatch')
        values=r['times_s'];groups=r['prefill']
        if (not values or len(values)!=len(groups) or any(type(g) is not bool for g in groups)
                or any(len(v)!=2 or not all(finite(x) for x in v) or v[0]>v[1] for v in values)
                or any(values[i][k]>values[i+1][k] for i in range(len(values)-1) for k in (0,1))):
            raise ValueError('Invalid training event values')
        found[tid]=r
    if set(found)!=set(wanted):raise ValueError('Incomplete training event split')
    return [found[k] for k in sorted(wanted)]


def event_predictions(pred, row):
    if base.work_signature(pred['steps'])!=row['work_signature']:
        raise ValueError('Own rollout differs from training work')
    done=max(r['finished_s'] for r in pred['requests'].values() if r['role']!='background')
    stop=[i+1 for i,s in enumerate(pred['steps']) if s['committed_s']==done]
    if len(stop)!=1 or stop[0]!=len(row['times_s']):raise ValueError('Foreground event prefix differs')
    steps=pred['steps'][:stop[0]]
    if [bool(s['features']['prefill']) for s in steps]!=row['prefill']:
        raise ValueError('Event phase differs from own work')
    return np.array([[s[k]-pred['release_s'] for k in ('scheduled_s','committed_s')] for s in steps])


def event_weights(row, spec):
    labels=np.array(row['times_s']);groups=np.array(row['prefill'])
    count=np.array([sum(groups==g) for g in groups])[:,None]
    # Each trial gives equal total squared weight to prefill and decode groups;
    # 127 decode steps cannot drown out a handful of prefill boundaries.
    return 1/(np.maximum(spec['event_absolute_scale_s'],spec['event_relative_scale']*np.abs(labels))
              * np.sqrt(len(set(groups))*2*count))


def event_scores(rows,predicted):
    cells=[]
    for row in rows:
        errors=event_predictions(predicted[row['case']],row)-np.array(row['times_s'])
        for phase in (False,True):
            subset=errors[np.array(row['prefill'])==phase]
            if not len(subset):continue
            cells.append(dict(trial_id=row['trial_id'],case=row['case'],phase='prefill' if phase else 'decode',
                events=int(subset.size),mae_s=float(np.mean(np.abs(subset))),bias_s=float(np.mean(subset)),
                max_absolute_error_s=float(np.max(np.abs(subset)))))
    return dict(scope='training LIGHT absolute events; retrospective development, not independent validation',cells=cells,
        trial_phase_balanced_mae_s=float(np.mean([r['mae_s'] for r in cells])))


def fit(rows, events, plan, runtime, spec=None, *, deadline=None, origin='synthetic'):
    if origin not in ('synthetic','retained_native_training_development','fresh_native_training_events'):
        raise ValueError('Unknown event-clock measurement origin')
    spec=specification() if spec is None else spec
    validate_spec(spec)
    training=base.select(rows,plan,{'train'},'off');light=base.select(rows,plan,{'train'},'light')
    observed=select_events(events,plan);quality=base.quality(rows,plan,{'train'})
    if not quality['passed']:raise ValueError('Training observer/repeat gate failed')
    light_by_id={r['trial_id']:r for r in light}
    if any(any(r[k]!=light_by_id[r['trial_id']][k] for k in ('capture_sha256','work_signature')) for r in observed):
        raise ValueError('Event and endpoint capture identity differs')
    cases={r['case'] for r in training};train_plan=dict(plan,cases=[c for c in plan['cases'] if c['id'] in cases])
    if any(c['sentinel'] or any(c['foreground_offsets_ms']) for c in train_plan['cases']):
        raise ValueError('Event-clock training requires original zero-offset cells')
    keys=[];labels=[];weights=[]
    for r in training:
        for rid,endpoint in sorted(r['endpoints'].items()):
            for metric,value in sorted(endpoint.items()):
                floor=plan['gates']['ttft_absolute_s' if metric=='ttft_s' else 'completion_absolute_s']
                keys.append((r['case'],rid,metric));labels.append(value)
                weights.append(1/(max(value,floor)*math.sqrt(len(r['endpoints'])*2)))
    labels=np.array(labels);weights=np.array(weights);cache={}
    def residual(theta):
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Event-clock fit deadline')
        key=tuple(theta)
        if key not in cache:
            pred=rollouts(theta,train_plan,runtime,spec);end=forecasts(pred)
            result=[(np.array([end[c]['endpoints'][rid][m] for c,rid,m in keys])-labels)*weights]
            for r in observed:
                result.append(((event_predictions(pred[r['case']],r)-np.array(r['times_s']))*event_weights(r,spec)).ravel())
            cache[key]=np.concatenate(result)
        return cache[key]
    fits=[least_squares(residual,x,bounds=spec['bounds_s'],max_nfev=spec['fit_max_nfev'],
        diff_step=1e-4,ftol=1e-9,xtol=1e-9,gtol=1e-9) for x in spec['starts_s']]
    if any(not f.success for f in fits):raise ValueError('Event-clock optimizer did not converge')
    best=min(fits,key=lambda f:float(f.fun@f.fun));theta=best.x
    if any(np.max(np.abs(f.x-theta)/np.maximum(theta,1e-6))>.05 for f in fits):raise ValueError('Event-clock multistart disagreement')
    scale=np.linalg.norm(best.jac,axis=0)
    if np.any(scale<=1e-12):raise ValueError('Zero clock sensitivity')
    singular=np.linalg.svd(best.jac/scale,compute_uv=False);rank=int(np.linalg.matrix_rank(best.jac/scale));condition=float(singular[0]/singular[-1])
    if rank!=len(PARAMETERS) or condition>plan['gates']['condition_limit']:raise ValueError('Clock rank/condition gate failed')
    lower,upper=spec['bounds_s']
    if np.any(theta<lower*1.01) or np.any(theta>upper*.99):raise ValueError('Clock coefficient hits bound')
    pred=rollouts(theta,train_plan,runtime,spec);forecast=forecasts(pred);scores=base.endpoint_scores(training,forecast,plan)
    if not scores['passed'] or not base.structure_check(light,forecast):raise ValueError('Training endpoint/work gate failed')
    frozen=dict(schema='effective-event-clock-freeze-v1',parameters=list(PARAMETERS),parameters_s=theta.tolist(),
        costs=costs(theta,plan,spec).to_dict(),spec_sha256=digest(spec),plan_sha256=digest(plan),runtime_sha256=digest(runtime),
        training_sha256=digest(training),training_events_sha256=digest(observed),
        local_sensitivity=dict(rank=rank,condition=condition,singular_values=singular.tolist()),
        optimizer_runs=[dict(success=bool(f.success),nfev=f.nfev,parameters_s=f.x.tolist(),sse=float(f.fun@f.fun)) for f in fits],
        train_scores=scores,train_event_scores=event_scores(observed,pred),train_quality=quality,
        data_origin='training_OFF_endpoints_and_LIGHT_events',measurement_origin=origin,held_labels_used_for_fit=False,
        development_design_informed_by_previous_delayed_failure=True,
        status='development_candidate_frozen_requires_fresh_confirmation',model_accepted=False,
        physical_costs_identified=False,adaptive_cap_accepted=False,mpc_accepted=False)
    frozen['sha256']=digest(frozen)
    return frozen


def predict_frozen(frozen, plan, runtime, spec=None):
    spec=specification() if spec is None else spec
    if (frozen['schema']!='effective-event-clock-freeze-v1'
            or frozen['sha256']!=digest({k:v for k,v in frozen.items() if k!='sha256'})
            or frozen['parameters']!=list(PARAMETERS)
            or any(frozen[k]!=digest(v) for k,v in [('spec_sha256',spec),('plan_sha256',plan),('runtime_sha256',runtime)])
            or digest(frozen['costs'])!=digest(costs(frozen['parameters_s'],plan,spec).to_dict())):
        raise ValueError('Event-clock freeze/plan/runtime/spec/cost identity changed')
    return forecasts(rollouts(frozen['parameters_s'],plan,runtime,spec))


def protocol():
    return json.loads((SPEC.parent/'prefill-event-clock-confirmation-v1.json').read_text())


def validate_manifest(manifest):
    from .tokens import validate
    plan=protocol()
    if (manifest['model'],manifest['revision'])!=(plan['model']['name'],plan['model']['revision']):
        raise ValueError('Clock confirmation model/revision differs')
    return validate(manifest,plan['cases'],plan['order'])


def evaluate(rows, events, plan, runtime, frozen, predicted):
    from .delayed_calibration import score
    if (digest(predicted)!=digest(predict_frozen(frozen,plan,runtime))
            or frozen['training_sha256']!=digest(base.select(rows,plan,{'train'},'off'))
            or frozen['training_events_sha256']!=digest(select_events(events,plan))):
        raise ValueError('Clock forecasts/training identity changed')
    result=score(rows,plan,predicted,frozen['sha256'])
    result.update(schema='effective-event-clock-confirmation-report-v1',
        scope='fresh native event-clock pilot only; physical/full-model/HTTP/cap/MPC not accepted')
    return result
