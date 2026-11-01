"""Event-clock algebra, split isolation, native label boundary and freeze tests."""
import copy
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from prefill_lab import event_clock_calibration as clock,delayed_calibration as delayed,effective_calibration as base
from prefill_lab.async_predictor import PipelineCosts
from prefill_lab.contracts import digest
from scripts import calibrate_prefill_event_clock as cli
from test_prefill_async_predictor import runtime_metadata


def synthetic():
    plan=delayed.protocol();runtime=runtime_metadata();spec=clock.specification()
    theta=[.009,.09,.005,.01,.001,.004]
    cases={c['id'] for c in plan['cases'] if c['split']=='calibration'}
    train_plan=dict(plan,cases=[c for c in plan['cases'] if c['id'] in cases])
    pred=clock.rollouts(theta,train_plan,runtime,spec);forecast=clock.forecasts(pred);rows=[];events=[]
    for t in plan['order']:
        if t['stage']!='train':continue
        r=pred[t['case']];end=forecast[t['case']]
        rows.append(dict(**{k:t[k] for k in ('trial_id','case','observer','stage','block')},
            endpoints=copy.deepcopy(end['endpoints']),work_signature=end['work_signature'] if t['observer']=='light' else None,
            capture_sha256='synthetic-'+t['trial_id']))
        if t['observer']=='light':
            done=max(x['finished_s'] for x in r['requests'].values() if x['role']!='background')
            stop=next(i+1 for i,s in enumerate(r['steps']) if s['committed_s']==done)
            events.append(dict(**{k:t[k] for k in ('trial_id','case','observer','stage','block')},
                capture_sha256='synthetic-'+t['trial_id'],work_signature=end['work_signature'],
                prefill=[bool(s['features']['prefill']) for s in r['steps'][:stop]],
                times_s=[[s[k]-r['release_s'] for k in ('scheduled_s','committed_s')] for s in r['steps'][:stop]]))
    return plan,runtime,spec,theta,rows,events


@pytest.fixture(scope='module')
def example():
    p,r,s,theta,rows,events=synthetic();f=clock.fit(rows,events,p,r,s)
    return p,r,s,theta,rows,events,f


def test_independent_synthetic_recovery_and_mixed_cost_identity(example):
    p,r,s,theta,rows,events,f=example
    assert np.allclose(f['parameters_s'],theta,rtol=1e-6,atol=1e-8)
    assert f['local_sensitivity']['rank']==6 and f['local_sensitivity']['condition']<1000
    assert len(f['optimizer_runs'])==2 and all(x['success'] for x in f['optimizer_runs'])
    assert f['train_event_scores']['trial_phase_balanced_mae_s']<1e-8
    assert not f['model_accepted'] and not f['mpc_accepted']
    assert not f['physical_costs_identified'] and not f['adaptive_cap_accepted']
    model=clock.costs(theta,p,s);row=dict(bias=1,prefill=2048,decode=16,decode_context=32768,attention_pairs=2097152)
    assert model.launches['forward']['gpu_tail'](row)==pytest.approx(sum(theta))
    assert model.to_dict()['prefill_decode']==pytest.approx(.004/32768)
    assert model.to_dict()!=clock.costs(theta[:-1]+[.005],p,s).to_dict()
    with pytest.raises(ValueError):PipelineCosts.from_dict(model.to_dict())
    forecast=clock.predict_frozen(f,p,r,s)
    assert set(forecast)=={c['id'] for c in p['cases']}


def test_held_rows_and_events_never_enter_fitting(example):
    p,r,s,_,rows,events,f=copy.deepcopy(example)
    rows += [dict(trial_id=t['trial_id'],endpoints=None,work_signature=None) for t in p['order'] if t['stage']!='train']
    events += [dict(trial_id=t['trial_id'],times_s=None,prefill=None) for t in p['order'] if t['stage']!='train']
    assert clock.fit(rows,events,p,r,s)==f


@pytest.mark.parametrize('fault',['missing','duplicate','split','nan','reversed','phase','work','capture','prefix'])
def test_event_corruption_is_rejected_before_freezing(example,fault):
    p,r,s,_,rows,events,_=copy.deepcopy(example);e=events[0]
    if fault=='missing':events.pop()
    if fault=='duplicate':events.append(copy.deepcopy(e))
    if fault=='split':e['stage']='held_delayed'
    if fault=='nan':e['times_s'][0][0]=float('nan')
    if fault=='reversed':e['times_s'][0].reverse()
    if fault=='phase':e['prefill'][0]=not e['prefill'][0]
    if fault=='work':e['work_signature']='bad'
    if fault=='capture':e['capture_sha256']='bad'
    if fault=='prefix':e['times_s'].pop();e['prefill'].pop()
    with pytest.raises(ValueError):clock.fit(rows,events,p,r,s)


@pytest.mark.parametrize('fault',['parameter','runtime','plan','spec','cost','digest'])
def test_frozen_cost_identity_cannot_silently_drop_mixed_term(example,fault):
    p,r,s,_,_,_,f=copy.deepcopy(example)
    if fault=='parameter':f['parameters_s'][-1]+=.1
    if fault=='runtime':r['process_id']+=1
    if fault=='plan':p['queue_absolute_s']+=1
    if fault=='spec':s['mixed_scale']*=2
    if fault=='cost':f['costs']['prefill_decode']*=2
    if fault=='digest':f['sha256']='bad'
    with pytest.raises(ValueError):clock.predict_frozen(f,p,r,s)


def test_phase_balancing_is_independent_of_number_of_decode_steps():
    s=clock.specification();row=dict(times_s=[[1.,1.]]*3,prefill=[True,False,False]);w=clock.event_weights(row,s)
    assert np.sum(w[:1]**2)==pytest.approx(np.sum(w[1:]**2))


def test_training_reader_never_opens_unselected_trial_files(example,monkeypatch,tmp_path):
    p,_,_,_,_,_,_=example
    monkeypatch.setattr(base,'extract',lambda raw,t,c:dict(trial_id=t['trial_id']))
    monkeypatch.setattr(clock,'event_labels',lambda raw,t,c:dict(trial_id=t['trial_id']))
    for t in p['order']:
        if t['stage']=='train':(tmp_path/(t['trial_id']+'.json')).write_text('{}')
    rows,events=cli.training_inputs(tmp_path,p)
    assert len(rows)==48 and len(events)==24


def test_native_label_extraction_uses_event_points_and_rejects_wrong_split(monkeypatch):
    t=dict(trial_id='train',stage='train',observer='light',case='case',block=1)
    cap=dict(release=10.,foreground_done=12.,steps=[dict(start=9.,end=11.,prefill=10),
        dict(start=9.5,end=12.,prefill=0),dict(start=11.,end=13.,prefill=0)])
    monkeypatch.setattr(base,'extract',lambda *a:dict(capture_sha256='hash',work_signature='signature'))
    row=clock.event_labels(cap,t,{})
    assert row['times_s']==[[-1.,1.],[-.5,2.]]  # Overlaps retained, never added.
    cap['release']+=100;cap['foreground_done']+=100
    for step in cap['steps']:step['start']+=100;step['end']+=100
    assert clock.event_labels(cap,t,{})==row
    with pytest.raises(ValueError):clock.event_labels(cap,dict(t,stage='held_delayed'),{})
    cap['steps'][1]['end']=cap['steps'][0]['end']
    with pytest.raises(ValueError):clock.event_labels(cap,t,{})


@pytest.mark.parametrize('fault',['deadline','nonconverged','rank','multistart','bound'])
def test_failed_identification_does_not_produce_a_candidate(example,monkeypatch,fault):
    p,r,s,theta,rows,events,_=example
    if fault=='deadline':
        with pytest.raises(TimeoutError):clock.fit(rows,events,p,r,s,deadline=time.monotonic()-1)
        return
    calls=[]
    def optimizer(*args,**kwargs):
        calls.append(1);x=np.array(theta)
        if fault=='multistart' and len(calls)==2:x*=2
        if fault=='bound':x[0]=s['bounds_s'][0]
        return NS(success=fault!='nonconverged',x=x,fun=np.zeros(6),jac=np.zeros((6,6)) if fault=='rank' else np.eye(6),nfev=1)
    monkeypatch.setattr(clock,'least_squares',optimizer)
    with pytest.raises(ValueError):clock.fit(rows,events,p,r,s)


def test_prospective_manifest_uses_fresh_content_and_same_frozen_rules():
    from prefill_lab.tokens import materialize
    p=clock.protocol();old=delayed.protocol();assert len(p['order'])==136 and p['cases']==old['cases']
    for key in ('gates','queue_absolute_s','queue_relative','posting_lag_limit_s'):assert p[key]==old[key]
    assert all(t['trial_id'].startswith('ct-') for t in p['order'])
    assert {t['paired_content_key'] for t in p['order']}.isdisjoint({t['paired_content_key'] for t in old['order']})
    tok=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    manifest=materialize(p['cases'],p['order'],tok,model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)
    clock.validate_manifest(manifest)
    with pytest.raises(ValueError):delayed.validate_manifest(manifest)


def test_clock_endpoint_mode_is_isolated_from_delayed_and_normal(monkeypatch,tmp_path):
    import asyncio,json
    from prefill_lab.endpoint import Endpoint
    from prefill_lab.tokens import materialize
    p=clock.protocol();tok=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    manifest=materialize(p['cases'],p['order'],tok,model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)
    path=tmp_path/'tokens.json';path.write_text(json.dumps(manifest));monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(path))
    monkeypatch.setenv('PREFILL_LAB_EVENT_CLOCK','1');monkeypatch.setenv('PREFILL_LAB_DELAYED_ARRIVALS','1');monkeypatch.setenv('PREFILL_LAB_EFFECTIVE_CALIBRATION','0')
    engine=NS(engine_core=NS(call_utility_async=lambda:None));endpoint=Endpoint();asyncio.run(endpoint.init_state(engine,None,None))
    assert set(endpoint.trials)=={t['trial_id'] for t in p['order']} and not endpoint.external_ids
    monkeypatch.setenv('PREFILL_LAB_DELAYED_ARRIVALS','0')
    with pytest.raises(ValueError,match='conflicting'):asyncio.run(Endpoint().init_state(engine,None,None))


def test_clock_host_retains_partial_archive_and_uses_own_mode(monkeypatch,tmp_path):
    import subprocess
    from pathlib import Path
    from scripts import prefill_host as host
    from test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[];prepared=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path));monkeypatch.setattr(host,'prepare',lambda args:prepared.append(args))
    def fail(args,**kw):
        calls.append(args);capture=Path(args[args.index('--output')+1]);capture.mkdir();(capture/'calibration.json').write_text('{}')
        raise subprocess.CalledProcessError(2,args)
    monkeypatch.setattr(host.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',event_clock=True)
    assert prepared[0].event_clock_only and not prepared[0].arrivals_only and not prepared[0].calibration_only
    assert '--event-clock-only' in calls[0] and '--arrivals-only' not in calls[0]
    assert calls[0][calls[0].index('--manifest')+1].endswith('/clock-tokens.json')
    assert len(list((tmp_path/'pr32-clocks').glob('*.tar.gz')))==1
    with pytest.raises(ValueError):host.collect_host(tmp_path/'inputs',event_clock=True,arrivals=True)


def test_native_clock_run_freezes_training_events_before_any_confirmation(monkeypatch,tmp_path):
    import asyncio,json
    from prefill_lab import suite,causal
    p=clock.protocol();runtime=runtime_metadata()
    monkeypatch.setattr(clock,'validate_manifest',lambda *a:None);monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(suite,'host_identity',lambda:{});monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:{'matches':True})
    monkeypatch.setattr(delayed,'extract',lambda cap,t,c,p:dict(trial_id=t['trial_id']))
    monkeypatch.setattr(clock,'event_labels',lambda cap,t,c:dict(trial_id=t['trial_id']))
    def fit(rows,events,*args,**kw):
        assert len(rows)==64 and len(events)==24 and kw['origin']=='fresh_native_training_events'
        return dict(costs={'schema':'test-only'})
    monkeypatch.setattr(clock,'fit',fit)
    monkeypatch.setattr(clock,'predict_frozen',lambda *a:{c['id']:dict(background_outlives_foreground=True) for c in p['cases']})
    async def trial(client,t,deadline,path):
        if t['stage'] in delayed.HELD:
            assert (tmp_path/'frozen.json').exists() and (tmp_path/'pre-held-predictions.json').exists()
            assert len(json.loads((tmp_path/'training-events.json').read_text()))==24
        return {}
    monkeypatch.setattr(suite,'controlled_trial',trial)
    report=dict(candidate_passed=True,quality={'passed':True},structure_and_background_lifetime_valid=True,
                endpoint_scores={'cells':[]},anchor_phase_scores={},sentinel_queue_scores=[])
    monkeypatch.setattr(clock,'evaluate',lambda *a:report)
    async def command(*args):return runtime
    summary=asyncio.run(delayed.run(NS(command=command),None,None,{},tmp_path,event_clock=True))
    assert summary['collection_complete'] and len(summary['completed'])==136
    assert not summary['model_accepted'] and summary['scope']==p['scope']


def test_clock_evaluation_keeps_v2_rules_and_rejects_changed_training(example):
    p,r,s,_,rows,events,f=copy.deepcopy(example);pred=clock.predict_frozen(f,p,r,s)
    for t in p['order']:
        if t['stage']=='train':continue
        forecast=pred[t['case']]
        row=dict(**{k:t[k] for k in ('trial_id','case','observer','stage','block')},endpoints=copy.deepcopy(forecast['endpoints']),
            work_signature=forecast['work_signature'] if t['observer']=='light' else None)
        if 'sentinel_queue_residence_s' in forecast:
            row['clocks']={'sentinel':dict(queue_residence_s=forecast['sentinel_queue_residence_s'],posting_lag_s=0.,
                posted_ttft_s=forecast['endpoints']['sentinel']['ttft_s'],target_ttft_s=forecast['endpoints']['sentinel']['ttft_s'])}
        rows.append(row)
    report=clock.evaluate(rows,events,p,r,f,pred)
    assert report['candidate_passed'] and not report['model_accepted'] and not report['mpc_accepted']
    events[0]['times_s'][0][0]-=.01
    with pytest.raises(ValueError,match='identity'):clock.evaluate(rows,events,p,r,f,pred)
