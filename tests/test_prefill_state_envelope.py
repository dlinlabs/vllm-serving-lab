"""Conditional timing bounds need independent graph checks, not point wins."""
import copy
import json
import random
from pathlib import Path
import pytest
from prefill_lab import state_envelope as env,live_predictor as live
from prefill_lab.async_predictor import Request,PipelineCosts,CostLaw,simulate
from prefill_lab.contracts import digest,save
from test_prefill_async_predictor import runtime_metadata
from test_prefill_live import empty,busy,D

THETA=[.01,.1,.002,.003,.001,.004]


def support(*,p=(.005,.06),d=(.0005,.004),factor=(.8,1.2),**kw):
    return env.make_support([v*factor[0] for v in THETA],[v*factor[1] for v in THETA],p,d,runtime_metadata(),**kw)


def evaluate(s,b=None,request=None,**kw):
    return env.forecast(s,request or Request('z',1024,4),b or support(),runtime_metadata(),now=kw.pop('now',s.get('observed_at',10.)),domain=kw.pop('domain','test'),**kw)


def varied_costs(lower,upper,rng):
    def law(a,b):
        class Interior(CostLaw):
            def __call__(self,row):return a(row)+rng.random()*(b(row)-a(row))
        return Interior({})
    parts={k:dict(v) for k,v in lower.launches.items()}
    for name in ('gpu_tail','cpu_after'):parts['forward'][name]=law(lower.launches['forward'][name],upper.launches['forward'][name])
    return PipelineCosts(parts,dict(lower.scalar),'independent varying per-batch costs inside the declared box')


@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('prompts',[(1024,),(4096,512),(7168,1024,2048)])
def test_independently_varying_cpu_gpu_batches_stay_inside_graph_bounds(seed,prompts):
    b=support();lower,upper=env.models(b,runtime_metadata());s=empty();requests=[]
    for i,tokens in enumerate(prompts[:-1]):
        rid=chr(ord('a')+i);requests.append(Request(rid,tokens,4));s['waiting'].append(rid)
        s['requests'][rid]=dict(prompt_tokens=tokens,max_tokens=4,num_tokens=tokens,computed=0,placeholders=0)
    request=Request('z',prompts[-1],4);result=evaluate(s,b,request,until_first=False)
    # Independent full EventGraph, with each batch drawn separately. A single
    # global P/D scalar is not imposed on this reference run.
    actual=simulate(requests+[request],varied_costs(lower,upper,random.Random(seed)),D)
    assert result['available']
    for key,observed in [('ttft_from_cut_s',actual['requests']['z']['ttft_s']),('completion_from_cut_s',actual['requests']['z']['finished_s'])]:
        lo,hi=result[key];assert lo-1e-10<=observed<=hi+1e-10
    assert not result['actuation_authorized'] and not result['native_coverage_qualified']


@pytest.mark.parametrize('prefill',[.006,.15])
def test_busy_cut_matches_independent_graph_with_ready_or_inflight_output(prefill):
    from prefill_lab import submit_clock,queue_confirmation
    p=queue_confirmation.protocol();model=submit_clock.costs(THETA,p,prefill,.002)
    reference=simulate([Request('a',4096,8),Request('z',512,4,.005)],model,D)
    cut=reference['requests']['z']['enqueue_s'];at=next(s for s in reference['steps'] if s['scheduled_s']==cut)
    requests={k:{f:v[f] for f in ('prompt_tokens','max_tokens','num_tokens','computed','placeholders')} for k,v in at['state_before'].items() if k!='z'}
    jobs=[s for s in reference['steps'] if s['scheduled_s']<cut and s['committed_s']>cut]
    s=dict(schema='live-engine-state-v1',clock_domain='test',observed_at=cut,
        last_commit_at=max((s['committed_s'] for s in reference['steps'] if s['committed_s']<=cut),default=None),
        running=list(requests),waiting=[],requests=requests,pending=[dict(work=s['work'],scheduled_at=s['scheduled_s'],ready=reference['nodes'][s['copy_node']]['end_s']<=cut) for s in jobs])
    result=evaluate(s,support(p=(prefill*.8,prefill*1.2)),Request('z',512,4),until_first=False)
    assert result['available'] and len(jobs)==1
    if s['pending'][0]['ready']:assert result['pending_residual_bounds_s']==[[0.,0.]]
    else:assert result['pending_residual_bounds_s'][0][1]>0
    for key,observed in [('ttft_from_cut_s',reference['requests']['z']['first_s']-cut),('completion_from_cut_s',reference['requests']['z']['finished_s']-cut)]:
        lo,hi=result[key];assert lo-1e-10<=observed<=hi+1e-10


def test_zero_width_box_equals_legacy_point_without_recharging_pending_return():
    b=support(p=(.02,.02),d=(.001,.001),factor=(1.,1.));model,_=env.models(b,runtime_metadata());s=busy(True)
    result=evaluate(s,b);point=live.forecast(s,Request('z',1024,4),model,D)
    assert result['ttft_from_cut_s']==[point['ttft_from_cut_s']]*2 and result['pending_residual_bounds_s']==[[0.,0.]]
    assert result['usable_under_assumptions'] and not result['fallback_required']
    assert s==busy(True)


def test_unready_job_with_zero_lower_debt_remains_observed_unready():
    s=busy(False);result=evaluate(s,support(factor=(0.,1.2)))
    assert result['available'] and result['pending_residual_bounds_s'][0][0]==0
    assert result['pending_residual_bounds_s'][0][1]>0 and s['pending'][0]['ready'] is False


def test_wide_support_is_retained_and_requests_fallback_even_far_from_deadline():
    result=evaluate(empty(),support(p=(.001,.15)),deadline_s=1.)
    assert result['available'] and result['width_s']>.010 and result['fallback_required']
    assert result['reason']=='interval_too_wide' and not result['usable_under_assumptions']
    assert result['conditional_deadline_relation']=='before' and not result['actuation_authorized']
    lo,hi=result['ttft_from_cut_s']
    assert evaluate(empty(),support(p=(.001,.15)),deadline_s=(lo+hi)/2)['conditional_deadline_relation']=='straddles'
    assert evaluate(empty(),support(p=(.001,.15)),deadline_s=0.)['conditional_deadline_relation']=='after'


@pytest.mark.parametrize('fault',['release_only','stale','clock','overdue'])
def test_missing_or_unsupported_state_abstains_without_imputing_future_readiness(fault):
    s=busy();kwargs={}
    if fault=='release_only':s={k:s[k] for k in ('requests','running','waiting')}
    if fault=='stale':kwargs['now']=10.006
    if fault=='clock':kwargs['domain']='another-boot'
    if fault=='overdue':s['pending'][0]['scheduled_at']=9.;s['last_commit_at']=9.
    result=evaluate(s,**kwargs)
    assert not result['available'] and result['fallback_required'] and not result['actuation_authorized']
    assert all(result[k] is False for k in env.FLAGS)
    if fault=='overdue':
        lo,_=env.models(support(),runtime_metadata())
        with pytest.raises(ValueError,match='overdue'):live.forecast(s,Request('z',1024,4),lo,D)


@pytest.mark.parametrize('fault',['future_label','scope','gpu','submit','policy','implementation','acceptance'])
def test_mutated_state_and_resealed_support_do_not_gain_silent_defaults(fault):
    b=support();s=empty()
    if fault=='future_label':s['measured_completion']=11.
    if fault=='scope':b['scope']['token_budget']=4096
    if fault=='gpu':b['gpu_lower_s'][0]=b['gpu_upper_s'][0]+1
    if fault=='submit':b['prefill_return_s']=[.1,.01]
    if fault=='policy':b['policy']['max_age_s']=-1
    if fault=='implementation':b['implementation_sha256']='changed'
    if fault=='acceptance':b['native_coverage_qualified']=True
    b['sha256']=digest({k:v for k,v in b.items() if k!='sha256'})
    with pytest.raises(ValueError):evaluate(s,b)


def test_train_extrema_and_event_identity_do_not_fit_readiness_or_held_targets(monkeypatch):
    from test_prefill_submit_clock import evidence
    from prefill_lab.queue_confirmation import protocol
    _,caps=evidence();p=protocol()
    for cap,t in zip(caps,[t for t in p['order'] if t['stage']=='train' and t['observer']=='light']):cap['trial_id']=t['trial_id']
    runtime=runtime_metadata();events=[dict(trial_id=c['trial_id']) for c in caps]
    base=dict(parameters_s=THETA,sha256='base',measurement_origin='fresh_native_training_events',training_events_sha256=digest(events))
    monkeypatch.setattr(env.clock,'predict_frozen',lambda *a:None)
    monkeypatch.setattr(env.clock,'event_labels',lambda c,*a:dict(trial_id=c['trial_id']))
    monkeypatch.setattr(env.clock,'select_events',lambda e,*a:sorted(e,key=lambda r:r['trial_id']))
    # Bind in declared sorted order; native version reconstructs full labels.
    base['training_events_sha256']=digest(sorted(events,key=lambda r:r['trial_id']))
    result=env.from_training(caps,p,runtime,base)
    raw=env.submit_clock.training_labels(caps,p)['samples']
    assert result['prefill_return_s']==[min(r['host_span_s'] for r in raw if r['kind']=='prefill'),max(r['host_span_s'] for r in raw if r['kind']=='prefill')]
    assert not result['native_coverage_qualified'] and not result['training']['coverage_claim']
    poisoned=copy.deepcopy(caps)
    for c in poisoned:c['future_readiness_target']='must not enter support'
    assert env.from_training(poisoned,p,runtime,base)==result
    with pytest.raises(ValueError,match='Exactly'):env.from_training(caps+[dict(trial_id='held')],p,runtime,base)
    base['training_events_sha256']='different'
    with pytest.raises(ValueError,match='event identity'):env.from_training(caps,p,runtime,base)


def test_cli_replay_cannot_claim_live_clock_or_native_qualification(tmp_path):
    from scripts.predict_prefill_envelope import main
    for name,doc in [('state',empty()),('request',dict(id='z',prompt_tokens=1024,max_tokens=4)),('runtime',runtime_metadata()),('support',support())]:save(tmp_path/(name+'.json'),doc)
    args=['predict','--replay-at','10','--output',str(tmp_path/'output')]
    for flag,name in [('snapshot','state'),('request','request'),('runtime','runtime'),('support','support')]:args+=['--'+flag,str(tmp_path/(name+'.json'))]
    main(args);r=json.loads((tmp_path/'output/envelope.json').read_text())
    assert r['clock_check']=='recorded_metadata_only' and not r['online_compute_age_qualified'] and not r['actuation_authorized']
