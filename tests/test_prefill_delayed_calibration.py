"""Delayed pilot clock isolation, split freezing, queue and integration contracts."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from prefill_lab import delayed_calibration as da,effective_calibration as base
from prefill_lab.contracts import save,digest
from scripts.check_prefill_arrivals import synthetic_rows
from test_prefill_async_predictor import runtime_metadata


@pytest.fixture(scope='module')
def example():
    p=da.protocol();r=runtime_metadata();truth=[.00935,.1006,.00733,.00873]
    rows=synthetic_rows(p,r,truth);f=base.fit(rows,p,r);pred=da.forecasts(f['parameters_s'],p,r)
    return p,r,rows,f,pred,truth


def test_synthetic_recovery_and_all_delayed_held_cases(example):
    p,r,rows,f,pred,truth=example
    assert np.allclose(f['parameters_s'],truth,atol=1e-7)
    report=da.evaluate(rows,p,r,f,pred)
    assert report['candidate_passed'] and not report['model_accepted'] and not report['mpc_accepted']
    assert len(report['sentinel_queue_scores'])==8
    assert len(report['endpoint_scores']['cells'])==80
    assert all(x['background_outlives_foreground'] for x in pred.values())


def test_fitting_does_not_read_delayed_labels(example):
    p,r,rows,f,_,_=copy.deepcopy(example)
    for row in rows:
        if row['stage'] in da.HELD:row['endpoints']=None;row['clocks']=None
    assert base.fit(rows,p,r)==f


@pytest.mark.parametrize('fault',['queue','posting','structure','missing','duplicate','forecast','freeze','anchor'])
def test_invalid_delayed_evidence_cannot_pass(example,fault):
    p,r,rows,f,pred,_=copy.deepcopy(example)
    row=next(x for x in rows if x['stage']=='held_delayed' and x['observer']=='off')
    if fault=='queue':
        for other in rows:
            if other['stage']=='held_delayed' and other['observer']=='off' and other['case']==row['case']:
                other['clocks']['sentinel']['queue_residence_s']+=1
    if fault=='posting':row['clocks']['sentinel']['posting_lag_s']=.006
    if fault=='structure':next(x for x in rows if x['stage']=='held_delayed' and x['observer']=='light')['work_signature']='bad'
    if fault=='missing':rows.remove(row)
    if fault=='duplicate':rows.append(row)
    if fault=='forecast':pred[row['case']]['sentinel_queue_residence_s']+=.2
    if fault=='freeze':f['parameters_s'][0]+=.1
    if fault=='anchor':
        row=next(x for x in rows if x['stage']=='held_anchor_before' and x['observer']=='off')
        row['endpoints']['foreground-0']['ttft_s']+=1
    if fault in ('posting','missing','duplicate','forecast','freeze'):
        with pytest.raises(ValueError):da.evaluate(rows,p,r,f,pred)
    else:assert not da.evaluate(rows,p,r,f,pred)['candidate_passed']


def capture_fixture():
    p=da.protocol();t=next(t for t in p['order'] if t['stage']=='held_delayed' and t['observer']=='off' and t['case'].endswith('at800-D16'))
    c=next(c for c in p['cases'] if c['id']==t['case']);requests={}
    for rid,(n,out,role,offset) in da.identities(c).items():
        target=10.+offset;posted=target+(.002 if offset else 0.)
        if role=='background':target=None;posted=9.
        requests[f'plab/{t["trial_id"]}/{rid}']=dict(input_tokens=n,max_tokens=out,output_tokens=out,role=role,
            target=target,queue_posted=posted,enqueue=posted+.1,first=posted+.3,finished=15. if role=='background' else 13.5,finish_reason='length')
    return p,t,c,dict(trial_id=t['trial_id'],observer='off',status='complete',requests=requests,
                      release=10.,foreground_done=13.5,steps=[],workers=[],errors=[],pending_step_ids=[])


@pytest.mark.parametrize('fault',[None,'lag','target','missing','chronology','background','output','pending'])
def test_three_clocks_and_native_integrity(fault):
    p,t,c,capture=capture_fixture();s=capture['requests'][f'plab/{t["trial_id"]}/sentinel']
    if fault=='lag':s['queue_posted']+=.01
    if fault=='target':s['target']+=.01
    if fault=='missing':capture['requests'].pop(f'plab/{t["trial_id"]}/foreground-0')
    if fault=='chronology':s['enqueue']=s['queue_posted']-.1
    if fault=='background':capture['requests'][f'plab/{t["trial_id"]}/background-0']['finished']=13.
    if fault=='output':s['output_tokens']=127
    if fault=='pending':capture['pending_step_ids']=['unresolved']
    if fault:
        with pytest.raises(ValueError):da.extract(capture,t,c,p)
    else:
        row=da.extract(capture,t,c,p);clock=row['clocks']['sentinel']
        assert clock['posting_lag_s']==pytest.approx(.002)
        assert clock['queue_residence_s']==pytest.approx(.1)
        assert clock['posted_ttft_s']==pytest.approx(.3)
        assert row['endpoints']['sentinel']['ttft_s']==pytest.approx(.302)
        assert clock['queue_residence_s']>.005  # Long residence is not a posting violation.
        assert capture['requests'][f'plab/{t["trial_id"]}/sentinel']['target']==10.8


def test_frozen_plan_token_pairs_and_phase_order():
    from prefill_lab.tokens import materialize
    p=da.protocol();assert len(p['cases'])==16 and len(p['order'])==136
    order=p['order'];assert max(i for i,t in enumerate(order) if t['stage']=='train')<min(i for i,t in enumerate(order) if t['stage'] in da.HELD)
    tok=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    m=materialize(p['cases'],order,tok,model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)
    da.validate_manifest(m)
    for t in order:
        if t['stage']=='held_anchor_before':
            matches=[v for v in order if v['case']==t['case'] and v['stage']=='held_anchor_after']
            assert all(m['trials'][t['trial_id']]==m['trials'][v['trial_id']] for v in matches)


def test_freeze_precedes_anchors_and_partial_failure_is_retained(example,monkeypatch,tmp_path):
    from prefill_lab import suite,causal
    p,r,rows,f,pred,_=example
    monkeypatch.setattr(da,'validate_manifest',lambda *a:None);monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(suite,'host_identity',lambda:{})
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:{'matches':True})
    monkeypatch.setattr(base,'fit',lambda *a,**kw:f);monkeypatch.setattr(da,'forecasts',lambda *a:pred)
    monkeypatch.setattr(da,'extract',lambda cap,t,c,p:next(row for row in rows if row['trial_id']==t['trial_id']))
    async def controlled(client,t,deadline,path):
        if t['stage'] in da.HELD:
            assert (tmp_path/'frozen.json').exists() and (tmp_path/'pre-held-predictions.json').exists()
            raise RuntimeError('held transport failed')
        return {}
    monkeypatch.setattr(suite,'controlled_trial',controlled)
    async def command(*a,**kw):return r
    with pytest.raises(RuntimeError,match='held transport'):asyncio.run(da.run(NS(command=command),None,None,{},tmp_path))
    summary=json.loads((tmp_path/'calibration.json').read_text())
    assert not summary['collection_complete'] and summary['failures']
    assert len(summary['completed'])==64


def test_endpoint_delayed_mode_isolated_from_v1(monkeypatch,tmp_path):
    from prefill_lab.endpoint import Endpoint
    from prefill_lab.tokens import materialize
    p=da.protocol();tok=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    m=materialize(p['cases'],p['order'],tok,model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)
    path=tmp_path/'tokens.json';save(path,m);monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(path))
    monkeypatch.setenv('PREFILL_LAB_DELAYED_ARRIVALS','1');monkeypatch.setenv('PREFILL_LAB_EFFECTIVE_CALIBRATION','0')
    engine=NS(engine_core=NS(call_utility_async=lambda:None));endpoint=Endpoint()
    asyncio.run(endpoint.init_state(engine,None,None));assert len(endpoint.trials)==136 and endpoint.external_ids==set()
    monkeypatch.setenv('PREFILL_LAB_EFFECTIVE_CALIBRATION','1')
    with pytest.raises(ValueError,match='conflicting'):asyncio.run(Endpoint().init_state(engine,None,None))
