"""CPU recovery/leakage/fail-closed checks; no native timing claims."""
import asyncio
import copy
import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from prefill_lab import effective_calibration as ec
from prefill_lab.contracts import digest, save
from scripts.check_prefill_effective_calibration import synthetic_rows
from test_prefill_async_predictor import runtime_metadata


@pytest.fixture(scope='module')
def example():
    p=ec.protocol();runtime=runtime_metadata();truth=[.0015,.015,.002,.003]
    rows=synthetic_rows(p,runtime,truth);frozen=ec.fit(rows,p,runtime)
    forecast=ec.predictions(frozen['parameters_s'],p,runtime)
    return p,runtime,rows,frozen,forecast,truth


def test_recovery_with_own_rollout_and_fresh_held_endpoints(example):
    p,runtime,rows,frozen,forecast,truth=example
    assert np.allclose(frozen['parameters_s'],truth,rtol=1e-6,atol=1e-8)
    assert frozen['local_sensitivity']['rank']==4
    assert frozen['local_sensitivity']['condition']<1000
    report=ec.evaluate(rows,p,runtime,frozen,forecast)
    assert report['candidate_passed'] and not report['model_accepted']
    assert not report['physical_costs_identified'] and not report['mpc_accepted']


def test_held_poison_never_enters_fit_and_training_poison_never_enters_evaluation(example):
    p,runtime,rows,frozen,forecast,_=example
    poisoned=copy.deepcopy(rows)
    for r in poisoned:
        if r['stage'].startswith('held'):r['endpoints']=None;r['work_signature']=None
    assert ec.fit(poisoned,p,runtime)==frozen
    poisoned=copy.deepcopy(rows)
    for r in poisoned:
        if r['stage']=='train':r['endpoints']=None
    assert ec.evaluate(poisoned,p,runtime,frozen,forecast)==ec.evaluate(rows,p,runtime,frozen,forecast)


@pytest.mark.parametrize('fault',['missing','duplicate','membership','nan','negative','request'])
def test_training_labels_fail_closed(example,fault):
    p,_,rows,_,_,_=example;rows=copy.deepcopy(rows)
    r=next(r for r in rows if r['stage']=='train' and r['observer']=='off')
    if fault=='missing':rows.remove(r)
    if fault=='duplicate':rows.append(r)
    if fault=='membership':r['stage']='held'
    if fault=='nan':r['endpoints']['foreground-0']['ttft_s']=float('nan')
    if fault=='negative':r['endpoints']['foreground-0']['completion_s']=-1
    if fault=='request':r['endpoints']['foreign']=r['endpoints']['foreground-0']
    with pytest.raises(ValueError):ec.select(rows,p,{'train'},'off')


@pytest.mark.parametrize('fault',['observer','repeat','prediction','structure','freeze','runtime'])
def test_held_gate_cannot_pass_corrupted_evidence(example,fault):
    p,runtime,rows,frozen,forecast,_=copy.deepcopy(example)
    r=next(r for r in rows if r['stage']=='held' and r['observer']=='light')
    if fault=='observer':r['endpoints']['foreground-0']['ttft_s']+=1
    if fault=='repeat':
        r=next(r for r in rows if r['stage']=='held' and r['observer']=='off')
        r['endpoints']['foreground-0']['ttft_s']+=1
    if fault=='structure':r['work_signature']='bad'
    if fault=='prediction':forecast[r['case']]['endpoints']['foreground-0']['ttft_s']+=1
    if fault=='freeze':frozen['parameters_s'][0]+=1
    if fault=='runtime':runtime['process_id']+=1
    if fault in ('prediction','freeze','runtime'):
        with pytest.raises(ValueError):ec.evaluate(rows,p,runtime,frozen,forecast)
    else:assert not ec.evaluate(rows,p,runtime,frozen,forecast)['candidate_passed']


@pytest.mark.parametrize('fault',['rank','bound','nonconverged','multistart'])
def test_unidentified_or_unconverged_fit_rejected(example,monkeypatch,fault):
    p,runtime,rows,_,_,truth=example;calls=[]
    def optimizer(*a,**kw):
        calls.append(1);x=np.array(truth)
        if fault=='bound':x[0]=p['parameter_bounds_s'][0]
        if fault=='multistart' and len(calls)==2:x=x*2
        return NS(success=fault!='nonconverged',x=x,fun=np.zeros(4),jac=np.zeros((4,4)) if fault=='rank' else np.eye(4))
    monkeypatch.setattr(ec,'least_squares',optimizer)
    with pytest.raises(ValueError):ec.fit(rows,p,runtime)


def test_fit_deadline_and_training_quality_stop_before_acceptance(example):
    p,runtime,rows,_,_,_=example
    with pytest.raises(TimeoutError):ec.fit(rows,p,runtime,deadline=time.monotonic()-1)
    rows=copy.deepcopy(rows)
    r=next(r for r in rows if r['stage']=='train' and r['observer']=='light')
    r['endpoints']['foreground-0']['ttft_s']+=1
    with pytest.raises(ValueError,match='observer/repeat'):ec.fit(rows,p,runtime)


def off_capture():
    p=ec.protocol();t=next(t for t in p['order'] if t['observer']=='off' and t['case']=='single-L1024-D0')
    case=next(c for c in p['cases'] if c['id']==t['case'])
    capture=dict(trial_id=t['trial_id'],observer='off',status='complete',release=10.,foreground_done=11.,steps=[],errors=[],workers=[],
        requests={f'plab/{t["trial_id"]}/foreground-0':dict(input_tokens=1024,max_tokens=128,output_tokens=128,
            role='foreground',queue_posted=10.,enqueue=10.001,first=10.02,finished=11.,finish_reason='length')})
    return capture,t,case


@pytest.mark.parametrize('fault',[None,'namespace','time','release','missing','short','observer','pending','done'])
def test_native_endpoint_extractor_never_sums_overlapping_spans(fault):
    capture,t,case=off_capture();r=next(iter(capture['requests'].values()))
    if fault=='namespace':capture['requests']={'foreign/foreground-0':r}
    if fault=='time':r['first']=float('nan')
    if fault=='release':r['queue_posted']-=.01
    if fault=='missing':capture['requests']={}
    if fault=='short':r['output_tokens']-=1
    if fault=='observer':capture['observer']='light'
    if fault=='pending':capture['pending_step_ids']=['unresolved']
    if fault=='done':capture['foreground_done']=12.
    if fault:
        with pytest.raises(ValueError):ec.extract(capture,t,case)
    else:
        row=ec.extract(capture,t,case)
        assert row['endpoints']['foreground-0']['ttft_s']==pytest.approx(.02)
        assert row['endpoints']['foreground-0']['completion_s']==1.


def test_plan_freezes_zero_offsets_splits_and_paired_content():
    p=ec.protocol();order=p['order'];assert len(order)==138 and len(p['cases'])==14
    assert len({t['trial_id'] for t in order})==138
    assert all(c['sentinel'] is None and set(c['foreground_offsets_ms'])=={0} for c in p['cases'])
    phases=[t['stage'] for t in order]
    assert max(i for i,s in enumerate(phases) if s=='train')<min(i for i,s in enumerate(phases) if s=='held')
    from prefill_lab.tokens import materialize,validate
    tokenizer=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    manifest=materialize(p['cases'],order,tokenizer,model='m',revision='a'*40,tokenizer_sha256='b'*64)
    assert validate(manifest,p['cases'],order)
    for i,t in enumerate(order):
        if t['observer']=='off':
            pair=next(x for x in order if x['paired_content_key']==t['paired_content_key'] and x['observer']=='light')
            assert manifest['trials'][t['trial_id']]==manifest['trials'][pair['trial_id']]


def test_parameter_names_survive_json_key_sorting(example):
    p,runtime,_,frozen,forecast,_=example
    p=json.loads(json.dumps(p,sort_keys=True))
    assert ec.predictions(frozen['parameters_s'],p,runtime)==forecast


def test_collector_freezes_before_held_and_persists_failure(example,monkeypatch,tmp_path):
    p,runtime,rows,frozen,forecast,_=example
    from prefill_lab import suite,causal
    monkeypatch.setattr(ec,'validate_manifest',lambda *a:None)
    # Exercise full declared order but mock the native transport and numerical work.
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(suite,'host_identity',lambda:{'fake':True})
    monkeypatch.setattr(ec,'extract',lambda c,t,case:next(r for r in rows if r['trial_id']==t['trial_id']))
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:{'matches':True})
    monkeypatch.setattr(ec,'fit',lambda *a,**kw:frozen)
    monkeypatch.setattr(ec,'predictions',lambda *a:forecast)
    firstheld=next(t['trial_id'] for t in p['order'] if t['stage']=='held')
    async def controlled(client,t,deadline,path):
        if t['trial_id']==firstheld:
            assert (tmp_path/'frozen.json').is_file() and (tmp_path/'pre-held-predictions.json').is_file()
            raise RuntimeError('injected transport failure')
        return {}
    monkeypatch.setattr(suite,'controlled_trial',controlled)
    async def command(*a,**kw):return runtime
    with pytest.raises(RuntimeError,match='injected'):
        asyncio.run(ec.run(NS(command=command),None,None,{},tmp_path))
    status=json.loads((tmp_path/'calibration.json').read_text())
    assert not status['collection_complete'] and not status['candidate_passed']
    assert status['failures'][0]['message']=='injected transport failure'
    assert firstheld not in status['completed']


def test_endpoint_mode_selects_only_frozen_calibration_trials(monkeypatch,tmp_path):
    from prefill_lab.endpoint import Endpoint
    from prefill_lab.tokens import materialize
    p=ec.protocol();tokenizer=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    manifest=materialize(p['cases'],p['order'],tokenizer,model=p['model']['name'],
                         revision=p['model']['revision'],tokenizer_sha256='b'*64)
    path=tmp_path/'manifest.json';save(path,manifest)
    monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(path))
    monkeypatch.setenv('PREFILL_LAB_EFFECTIVE_CALIBRATION','1')
    endpoint=Endpoint();engine=NS(engine_core=NS(call_utility_async=lambda:None))
    asyncio.run(endpoint.init_state(engine,None,None))
    assert set(endpoint.trials)==set(manifest['trials']) and endpoint.external_ids==set()
    monkeypatch.setenv('PREFILL_LAB_EFFECTIVE_CALIBRATION','0')
    with pytest.raises(ValueError,match='frozen cases/order'):asyncio.run(Endpoint().init_state(engine,None,None))
    monkeypatch.setenv('PREFILL_LAB_EFFECTIVE_CALIBRATION','invalid')
    with pytest.raises(ValueError,match='Invalid calibration mode'):asyncio.run(Endpoint().init_state(engine,None,None))
    manifest['revision']='a'*40
    with pytest.raises(ValueError,match='model/revision'):ec.validate_manifest(manifest)


@pytest.mark.parametrize('fault',['early_finish','late_first'])
def test_native_background_lifetime_recomputed_not_just_trusted_errors(fault):
    capture,t,case=off_capture();case=copy.deepcopy(case);case['background_decoders']=16
    for i in range(16):
        capture['requests'][f'plab/{t["trial_id"]}/background-{i}']=dict(input_tokens=1024,max_tokens=256,
            output_tokens=256,role='background',queue_posted=9.,enqueue=9.,first=9.5,finished=12.,finish_reason='length')
    ec.extract(capture,t,case)
    r=capture['requests'][f'plab/{t["trial_id"]}/background-0']
    if fault=='early_finish':r['finished']=10.5
    else:r['first']=10.1
    with pytest.raises(ValueError,match='barrier/lifetime'):ec.extract(capture,t,case)
