"""Rejection reporting never relaxes the original delayed pilot decision."""
import copy
import pytest
from prefill_lab.delayed_calibration import rejection_reasons
from scripts.analyze_prefill_arrival_capture import gates


def report():
    return dict(candidate_passed=True,quality={'passed':True},endpoint_scores={'passed':True,'cells':[]},
        structure_and_background_lifetime_valid=True,
        anchor_phase_scores={s:{'passed':True,'cells':[]} for s in ('held_anchor_before','held_anchor_after')},
        sentinel_queue_scores=[dict(case='case',passed=True,error_s=0.,limit_s=.01)])


def test_valid_report_has_no_rejections_and_is_not_mutated():
    r=report();before=copy.deepcopy(r)
    assert rejection_reasons(r)==[] and all(gates(r).values())
    assert r==before


def test_queue_failure_is_preserved_with_actionable_values():
    r=report();r['candidate_passed']=False;r['sentinel_queue_scores'][0].update(passed=False,error_s=-.0207646435)
    before=copy.deepcopy(r);result=rejection_reasons(r)
    assert len(result)==1 and result[0]['gate']=='sentinel_queue'
    assert result[0]['error_s']==-.0207646435 and result[0]['limit_s']==.01
    assert not gates(r)['sentinel_queue'] and r==before


@pytest.mark.parametrize('gate',['quality','structure','endpoint','anchor_before','anchor_after'])
def test_each_original_failure_is_reported(gate):
    r=report();r['candidate_passed']=False
    if gate=='quality':r['quality']['passed']=False
    elif gate=='structure':r['structure_and_background_lifetime_valid']=False
    elif gate=='endpoint':r['endpoint_scores']={'passed':False,'cells':[{'passed':False,'request':'sentinel'}]}
    else:r['anchor_phase_scores']['held_'+gate]={'passed':False,'cells':[{'passed':False,'request':'foreground-0'}]}
    assert len(rejection_reasons(r))==1
    assert not all(gates(r).values())


def test_inconsistent_stored_gate_verdict_cannot_pass_diagnosis():
    r=report();r['candidate_passed']=False
    with pytest.raises(ValueError,match='inconsistency'):gates(r)
