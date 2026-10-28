"""CPU conditional timing envelopes, not native coverage or admission safety.

Fixed observed pre-ADD state, forced outputs, no later arrivals, FIFO/two slots.
Every future CPU return and GPU service may vary independently inside the box.
The logical work order is time-independent in this scope; event times use only
nonnegative additions and maxima. Extremal runs bound that fixed event graph.
Empirical TRAIN extrema do not establish probability or physical guarantees.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
from . import event_clock_calibration as clock,submit_clock,live_predictor as live
from .contracts import digest,finite
from scripts.predict_prefill_async import runtime_inputs

FLAGS=('model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted')
MEANING='conditional cost-box envelope; empirical support is not coverage qualification'


def pair(values):
    if not isinstance(values,(list,tuple)) or len(values)!=2 or not all(finite(v) and v>=0 for v in values) or values[0]>values[1]:raise ValueError('Invalid nonnegative timing interval')
    return list(values)


def make_support(gpu_lower,gpu_upper,prefill,decode,runtime,*,origin='explicit_assumptions',training=None,max_age_s=.005,max_width_s=.010):
    scope,dispatch=runtime_inputs(runtime)
    if (scope.token_budget,scope.max_seqs,scope.max_model_len)!=(2048,32,8192):raise ValueError('Envelope scope requires B2048/seq32/context8192')
    if len(gpu_lower)!=6 or len(gpu_upper)!=6:raise ValueError('Six GPU coefficients required')
    for a,b in zip(gpu_lower,gpu_upper):pair([a,b])
    if origin not in ('explicit_assumptions','train_empirical_support'):raise ValueError('Unsupported support origin')
    if origin=='explicit_assumptions' and training is not None:raise ValueError('Assumed support cannot claim TRAIN provenance')
    if origin=='train_empirical_support':
        keys={'trial_ids','base_model_sha256','plan_sha256','events_sha256','samples_sha256','sample_count','rule','held_labels_used','coverage_claim'}
        if not isinstance(training,dict) or set(training)!=keys or training['held_labels_used'] is not False or training['coverage_claim'] is not False:raise ValueError('Invalid TRAIN support provenance')
        if not training['trial_ids'] or training['trial_ids']!=sorted(set(training['trial_ids'])) or type(training['sample_count']) is not int or training['sample_count']<1:raise ValueError('Invalid TRAIN support membership')
    if not all(finite(v) and v>0 for v in (max_age_s,max_width_s)):raise ValueError('Positive age/width policy required')
    doc=dict(schema='state-time-support-v1',meaning=MEANING,scope=asdict(scope),runtime_sha256=digest(runtime),dispatcher_sha256=digest(dispatch),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),continuation_sha256=hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest(),
        gpu_lower_s=list(gpu_lower),gpu_upper_s=list(gpu_upper),prefill_return_s=pair(prefill),decode_return_s=pair(decode),
        policy=dict(max_age_s=max_age_s,max_width_s=max_width_s),origin=origin,training=training,
        native_coverage_qualified=False,**{k:False for k in FLAGS})
    doc['sha256']=digest(doc);return doc


def models(support,runtime):
    required={'schema','meaning','scope','runtime_sha256','dispatcher_sha256','implementation_sha256','continuation_sha256','gpu_lower_s','gpu_upper_s','prefill_return_s','decode_return_s','policy','origin','training','native_coverage_qualified','sha256',*FLAGS}
    if set(support)!=required or support['schema']!='state-time-support-v1' or support['meaning']!=MEANING or support['sha256']!=digest({k:v for k,v in support.items() if k!='sha256'}):raise ValueError('Support document identity differs')
    if support['native_coverage_qualified'] is not False or any(support[k] is not False for k in FLAGS):raise ValueError('Unqualified support cannot promote acceptance')
    policy=support['policy']
    if set(policy)!={'max_age_s','max_width_s'}:raise ValueError('Support policy differs')
    rebuilt=make_support(support['gpu_lower_s'],support['gpu_upper_s'],support['prefill_return_s'],support['decode_return_s'],runtime,
        origin=support['origin'],training=support['training'],**policy)
    if rebuilt!=support:raise ValueError('Support runtime/scope/dispatcher differs')
    # The fixed original plan supplies scaling units, not future cases/work.
    from .queue_confirmation import protocol
    p=protocol();lo=submit_clock.costs(support['gpu_lower_s'],p,support['prefill_return_s'][0],support['decode_return_s'][0])
    hi=submit_clock.costs(support['gpu_upper_s'],p,support['prefill_return_s'][1],support['decode_return_s'][1])
    return lo,hi


def from_training(captures,plan,runtime,base_model,**policy):
    """Past TRAIN extrema only; no rejected readiness/held target is fitted."""
    from .queue_confirmation import protocol
    if digest(plan)!=digest(protocol()):raise ValueError('Support construction requires the frozen confirmation plan')
    captures=list(captures);wanted={t['trial_id']:t for t in plan['order'] if t['stage']=='train' and t['observer']=='light'}
    if len(captures)!=len(wanted) or {c['trial_id'] for c in captures}!=set(wanted):raise ValueError('Exactly the declared TRAIN LIGHT captures required')
    if base_model.get('measurement_origin')!='fresh_native_training_events':raise ValueError('Fresh native TRAIN model required')
    clock.predict_frozen(base_model,plan,runtime)
    cases={c['id']:c for c in plan['cases']}
    events=[clock.event_labels(c,wanted[c['trial_id']],cases[wanted[c['trial_id']]['case']]) for c in captures]
    if digest(clock.select_events(events,plan))!=base_model['training_events_sha256']:raise ValueError('TRAIN event identity differs from base model')
    labels=submit_clock.training_labels(captures,plan);samples=labels['samples']
    ranges={kind:[min(r['host_span_s'] for r in samples if r['kind']==kind),max(r['host_span_s'] for r in samples if r['kind']==kind)] for kind in ('prefill','decode')}
    training=dict(trial_ids=sorted(wanted),base_model_sha256=base_model['sha256'],plan_sha256=digest(plan),
        events_sha256=base_model['training_events_sha256'],samples_sha256=labels['training_sample_sha256'],sample_count=len(samples),
        rule='all TRAIN prefix worker host-span extrema by prefill/decode; GPU lane fixed at original base parameters',
        held_labels_used=False,coverage_claim=False)
    return make_support(base_model['parameters_s'],base_model['parameters_s'],ranges['prefill'],ranges['decode'],runtime,
        origin='train_empirical_support',training=training,**policy)


def unavailable(snapshot,support,reason):
    return dict(schema='state-time-envelope-v1',available=False,usable_under_assumptions=False,fallback_required=True,reason=reason,actuation_authorized=False,online_compute_age_qualified=False,
        state_sha256=digest(snapshot),support_sha256=support['sha256'],conditional_only=True,native_coverage_qualified=False,**{k:False for k in FLAGS})


def forecast(snapshot,proposed,support,runtime,*,now,domain,deadline_s=None,until_first=True):
    lower,upper=models(support,runtime);scope,dispatch=runtime_inputs(runtime)
    if set(snapshot)=={'running','waiting','requests'}:return unavailable(snapshot,support,'missing_pending_time_state')
    live.validate(snapshot,scope)
    if not finite(now) or now<snapshot['observed_at']:raise ValueError('Invalid envelope evaluation time')
    if deadline_s is not None and (not finite(deadline_s) or deadline_s<0):raise ValueError('Invalid relative deadline')
    if snapshot['clock_domain']!=domain:return unavailable(snapshot,support,'clock_domain_mismatch')
    age=now-snapshot['observed_at']
    if age>support['policy']['max_age_s']:return unavailable(snapshot,support,'stale_snapshot')
    debts=[]
    for job in snapshot['pending']:
        if job['ready']:debts.append([0.,0.]);continue
        row=live.feature_row(job['work'],snapshot['requests'],dispatch)
        anchor=max(job['scheduled_at'],snapshot['last_commit_at'] or job['scheduled_at'])
        low=anchor+live.service(lower,row)-snapshot['observed_at'];high=anchor+live.service(upper,row)-snapshot['observed_at']
        if high<=0:return unavailable(snapshot,support,'unready_pending_outside_service_support')
        # Zero is a closed lower bound for arbitrarily small positive remaining
        # work, never a claim that the observed unready job has completed.
        debts.append([max(0.,low),high])
    corners=[live._continue(snapshot,proposed,model,dispatch,scope=scope,until_first=until_first,pending_residuals=[r[i] for r in debts]) for i,model in enumerate((lower,upper))]
    work=lambda r:[s['work'] for s in r['steps']]
    if work(corners[0])!=work(corners[1]) or corners[0]['output_tokens']!=corners[1]['output_tokens']:return unavailable(snapshot,support,'unsupported_timing_dependent_structure')
    bounds={}
    for key in ('ttft_from_cut_s','completion_from_cut_s'):
        values=[r[key] for r in corners]
        if values==[None,None]:bounds[key]=None;continue
        if any(v is None for v in values) or values[0]>values[1]:raise ValueError('Nonmonotone conditional timing endpoints')
        bounds[key]=values
    lo,hi=bounds['ttft_from_cut_s'];width=hi-lo
    # Age is reported, not used to invent a new current engine state by time
    # subtraction. These are cut-relative forecasts, not post-admission actions.
    relation=None if deadline_s is None else 'before' if hi<=deadline_s else 'after' if lo>deadline_s else 'straddles'
    return dict(schema='state-time-envelope-v1',available=True,usable_under_assumptions=width<=support['policy']['max_width_s'],actuation_authorized=False,online_compute_age_qualified=False,
        fallback_required=width>support['policy']['max_width_s'],reason=None if width<=support['policy']['max_width_s'] else 'interval_too_wide',
        **bounds,width_s=width,snapshot_age_s=age,pending_residual_bounds_s=debts,
        deadline_from_cut_s=deadline_s,conditional_deadline_relation=relation,
        state_sha256=digest(snapshot),support_sha256=support['sha256'],work_sha256=digest(work(corners[0])),executed_steps=len(corners[0]['steps']),
        assumptions='fixed observed state; no later arrivals; forced lengths; every GPU/CPU service inside supplied box; effective pending anchor valid',
        conditional_only=True,native_coverage_qualified=False,**{k:False for k in FLAGS})
