import copy
import numpy as np
import pytest
from prefill_design import simulate,scored_steps,requests_for
from prefill_identifiability import features,calibration_support,matrix_diagnostics
from scripts.build_prefill_design import cases


def calibration_rows():
    rows={}
    for c in cases():
        if c['split']!='calibration':continue
        steps=[];r=simulate(requests_for(c),2048,backgrounds=c['background_decoders'],steps_out=steps)
        rows[c['id']]=[features(s) for s in scored_steps(steps,r)]
    return rows


def test_scoring_excludes_preparation_and_background_only_drain():
    requests=[dict(id='foreground',tokens=1024,output=3,at=0.)]
    steps=[];r=simulate(requests,2048,backgrounds=16,steps_out=steps)
    window=r['scored_step_window'];selected=scored_steps(steps,r)
    assert len(selected)==3
    assert steps[0]['start']<window['start_s']
    assert selected[0]['start']==window['start_s']
    assert selected[-1]['end']==window['end_s']
    assert window['end_s']==r['release_s']+r['foreground']['foreground']['completed_s']
    assert steps[-1]['end']>window['end_s']
    assert len(scored_steps(steps,r))<len(steps)


def test_full_feature_rank_detects_what_q_z_rank_misses():
    rows=np.array([[1,0,100,0,5050],[1,0,200,0,20100],[1,0,300,0,45150]],float)
    assert np.linalg.matrix_rank(rows[:,[2,4]])==2
    d=matrix_diagnostics(rows,5)
    assert d['rank']==3 and not d['supported'] and d['scaled_condition_number'] is None
    assert d['zero_columns']==['decode_tokens','decode_context']


def test_proposed_calibration_supports_all_families_in_every_training_fold():
    d=calibration_support(calibration_rows())
    assert d['all_families_supported'] and len(d['folds'])==7
    for fold in d['folds']:
        assert fold['held_cell'] not in fold['training_cells']
        for name,n in [('M0',3),('M1',4),('M2',5)]:
            assert fold['models'][name]['rank']==n
            assert fold['models'][name]['scaled_condition_number']<100


def test_loco_scaling_uses_only_that_folds_training_cells():
    rows=calibration_rows();name='single-L7168-D16'
    a=next(f for f in calibration_support(rows)['folds'] if f['held_cell']==name)
    changed=copy.deepcopy(rows)
    changed[name]=(np.asarray(changed[name])*1e9).tolist()
    b=next(f for f in calibration_support(changed)['folds'] if f['held_cell']==name)
    assert a==b


def test_background_shortening_preserves_scored_model_rows():
    for c in cases():
        a=[];b=[]
        old=simulate(requests_for(c),2048,backgrounds=c['background_decoders'],background_output=512,steps_out=a)
        new=simulate(requests_for(c),2048,backgrounds=c['background_decoders'],background_output=256,steps_out=b)
        assert scored_steps(a,old)==scored_steps(b,new)
        assert old['scored_step_window']==new['scored_step_window']


@pytest.mark.parametrize('rows',[[],[[1,2,3]],[[1,2,3,4,float('nan')]]])
def test_invalid_feature_rows_rejected(rows):
    with pytest.raises(ValueError):matrix_diagnostics(rows,5)


@pytest.mark.parametrize('kwargs',[dict(quadratic=float('nan')),dict(background_output=2.5)])
def test_bad_service_parameters_fail_before_simulation(kwargs):
    with pytest.raises(ValueError):simulate([dict(id='r',tokens=100,output=3,at=0)],2048,**kwargs)


def test_fractional_output_rejected_instead_of_never_draining():
    with pytest.raises(ValueError):simulate([dict(id='r',tokens=100,output=2.5,at=0)],2048)
