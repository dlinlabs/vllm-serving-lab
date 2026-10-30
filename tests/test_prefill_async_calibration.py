import copy
import json
import numpy as np
import pytest
from prefill_lab.async_calibration import plan,freeze,evaluate,CHANNELS
from prefill_lab.async_predictor import PipelineCosts,Request,simulate
from prefill_lab.contracts import digest
from scripts.calibrate_prefill_async import synthetic,main

@pytest.fixture
def setup():
    p=plan();data,truth=synthetic(p)
    return p,data,truth


def test_known_coefficients_recovered_and_costs_run_causal_predictor(setup):
    p,data,truth=setup;f=freeze(data,p)
    recovered={**{s+'.'+k:v for s,parts in f['costs']['launches'].items() for k,v in parts.items()},**f['costs']['scalar']}
    for c,cs in truth.items():
        for k,v in cs.items():assert recovered[c][k]==pytest.approx(v,rel=1e-6,abs=1e-14)
    real=copy.deepcopy(f['costs'])
    for c,cs in truth.items():
        if '.' in c:s,k=c.split('.');real['launches'][s][k]=cs
        else:real['scalar'][c]=cs
    dispatcher=dict(schema='v2-dispatch-candidates-v1',graphs_captured=False,candidates={})
    requests=[Request('long',7168,4),Request('sentinel',512,2,.01,'sentinel')]
    a=simulate(requests,PipelineCosts.from_dict(real),dispatcher)
    b=simulate(requests,PipelineCosts.from_dict(f['costs']),dispatcher)
    for rid in a['requests']:assert a['requests'][rid]['ttft_s']==pytest.approx(b['requests'][rid]['ttft_s'],abs=1e-10)
    report=evaluate(data,f)
    assert max(s['balanced_mae_s'] for s in report['scores'].values())<1e-12
    assert report['within_marginal_support'] and not report['model_accepted']


def test_held_poison_cannot_change_frozen_model(setup):
    p,d,_=setup;a=freeze(d,p)
    for r in d['rows']:
        if p['trials'][r['trial_id']]=='held':r['labels']=None;r['features']=None
    assert freeze(d,p)==a


def test_evaluation_does_not_read_training_labels(setup):
    p,d,_=setup;f=freeze(d,p);a=evaluate(d,f)
    for r in d['rows']:
        if p['trials'][r['trial_id']]=='train':r['labels']=None
    assert evaluate(d,f)==a


@pytest.mark.parametrize('change',['constant','zero','near_collinear'])
def test_nonidentifiable_design_is_rejected(setup,change):
    p,d,_=setup
    for r in d['rows']:
        if p['trials'][r['trial_id']]!='train':continue
        x=r['features']
        if change=='constant':x['num_requests']=3
        elif change=='zero':x['attention_pairs']=0
        else:x['attention_pairs']=x['prefill']*100+1e-10*x['decode_context']
    with pytest.raises(ValueError,match='identifiable|conditioned'):freeze(d,p)


@pytest.mark.parametrize('bad',[-1,float('nan'),float('inf'),True])
def test_bad_labels_rejected(setup,bad):
    p,d,_=setup;d['rows'][0]['labels']['copy_service']['seconds']=bad
    with pytest.raises(ValueError,match='label'):freeze(d,p)


@pytest.mark.parametrize('bad',['wait_inclusive','sample_tokens_parent','raw_cuda_elapsed'])
def test_wrong_label_scope_cannot_be_fit(setup,bad):
    p,d,_=setup;d['rows'][0]['labels']['copy_service']['scope']=bad
    with pytest.raises(ValueError,match='label rejected'):freeze(d,p)


def test_no_unqualified_native_or_old_diagnostic_fit(setup):
    p,d,_=setup;d['origin']='native'
    with pytest.raises(ValueError,match='qualification'):freeze(d,p)


@pytest.mark.parametrize('change',['missing','duplicate','unknown','plan','runtime'])
def test_identity_or_split_failure(setup,change):
    p,d,_=setup
    if change=='missing':d['rows']=[r for r in d['rows'] if r['trial_id']!='train-0']
    elif change=='duplicate':d['rows'].append(copy.deepcopy(d['rows'][0]))
    elif change=='unknown':d['rows'][0]['trial_id']='mystery'
    elif change=='plan':p['condition_limit']=100
    else:d['runtime_sha256']=''
    with pytest.raises(ValueError):freeze(d,p)


def test_frozen_tampering_and_runtime_change_fail(setup):
    p,d,_=setup;f=freeze(d,p)
    damaged=copy.deepcopy(f);damaged['costs']['scalar']['copy_service']['bias']=10
    with pytest.raises(ValueError,match='integrity'):evaluate(d,damaged)
    d['runtime_sha256']='1'*64
    with pytest.raises(ValueError,match='Runtime'):evaluate(d,f)


def test_held_out_of_support_is_reported_without_refitting(setup):
    p,d,_=setup;f=freeze(d,p);before=copy.deepcopy(f)
    held=next(r for r in d['rows'] if p['trials'][r['trial_id']]=='held')
    held['mode']='FULL';held['features']['prefill']=999999
    r=evaluate(d,f)
    assert not r['within_marginal_support'] and r['unseen_modes']==['FULL']
    assert f==before and not r['model_accepted']


def test_repeated_steps_do_not_overweight_one_trial(setup):
    p,d,_=setup
    for r in d['rows']:
        if r['trial_id']=='train-0':
            for label in r['labels'].values():label['seconds']+=1e-7
    a=freeze(d,p)
    extra=[]
    for row in d['rows']:
        if row['trial_id']=='train-0':
            for i in range(3):
                r=copy.deepcopy(row);r['step_id']+='-copy-'+str(i);extra.append(r)
    d['rows']+=extra;b=freeze(d,p)
    for c in CHANNELS:
        def law(f):
            if '.' in c:s,k=c.split('.');return f['costs']['launches'][s][k]
            return f['costs']['scalar'][c]
        for k,v in law(a).items():assert law(b)[k]==pytest.approx(v,rel=1e-5,abs=1e-12)


def test_cli_demo_and_export_are_reusable_and_cannot_overwrite(tmp_path):
    out=tmp_path/'demo';assert main(['demo','--output',str(out)])==0
    assert json.loads((out/'result.json').read_text())['max_coefficient_error']<1e-12
    PipelineCosts.from_dict(json.loads((out/'costs.json').read_text()))
    assert main(['evaluate','--input',str(out/'synthetic-input.json'),'--frozen',str(out/'frozen.json'),'--output',str(tmp_path/'score')])==0
    with pytest.raises(SystemExit):main(['demo','--output',str(out)])
