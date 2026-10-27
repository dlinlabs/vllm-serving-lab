"""Calibration isolation, exact common-scale algebra and split outcome reporting."""
import copy
import json
from pathlib import Path

import pytest
from prefill_lab import queue_phase as q,event_clock_calibration as clock
from prefill_lab.contracts import digest


@pytest.fixture(scope='module')
def example():
    root=Path(__file__).resolve().parents[1]/'results/prefill-clock-4c89188-queue-phase/native'
    read=lambda name:json.loads((root/name).read_text())
    plan=read('calibration-plan.json');rt=read('runtime.json');model=read('frozen.json');rows=read('endpoints.json')
    predicted=clock.predict_frozen(model,plan,rt);f=q.freeze(rows,plan,rt,model,predicted)
    return rows,plan,rt,model,predicted,f


def test_common_scale_uses_training_only_and_keeps_historical_rejection(example):
    rows,p,rt,m,pred,f=copy.deepcopy(example)
    assert f['readiness_passed'] and f['scale_half_width']==pytest.approx(.04626523891083423)
    cell=f['cells']['mixed-plus-sentinel-at400-D16']
    assert cell['status']=='phase_ambiguous'
    assert [b['queue_s'] for b in cell['branches']]==pytest.approx([.006698045544983433,.175183260914491])
    for r in rows:
        if r['stage']!='train':r['endpoints']=None;r['clocks']=None;r['work_signature']=None
    assert q.freeze(rows,p,rt,m,pred)==f
    assert not f['held_labels_used'] and not f['model_accepted']


def test_piecewise_maximum_keeps_a_narrow_interior_switch():
    f=[(0.,2.,1.,0.)];g=[(0.,2.,-1.,2.000002)]
    h=q.combine(f,g,True)
    assert len(h)==2 and h[0][1]==pytest.approx(1.000001)
    for x in (0.,1.,1.0000009,1.0000011,2.):assert q.value(h,x)==pytest.approx(max(x,2.000002-x))


@pytest.mark.parametrize('name',['mixed-plus-sentinel-at400-D16','mixed-plus-sentinel-at800-D0','mixed-plus-sentinel-at50-D16'])
def test_analytic_poll_curves_match_independent_existing_rollouts(example,name):
    _,p,rt,m,_,f=example;case=next(c for c in p['cases'] if c['id']==name);s=f['scale_half_width']
    ref=clock.rollouts(m['parameters_s'],dict(p,cases=[dict(case,sentinel=None)]),rt,clock.specification())[name]
    polls,_,_=q.timing(ref,s,q.policy());at=case['sentinel']['arrival_ms']/1000.
    for scale in (1-s,1-s/2,1.,1+s/2,1+s):
        boundary=next(q.value(curve,scale) for _,curve in polls if q.value(curve,scale)>=at-1e-12)
        own=clock.rollouts([v*scale for v in m['parameters_s']],dict(p,cases=[case]),rt,clock.specification())[name]
        assert boundary-at==pytest.approx(own['requests']['sentinel']['input_queue_wait_s'],abs=1e-9)


@pytest.mark.parametrize('fault',['point','eligibility','spread','training','runtime','plan'])
def test_even_rehashed_held_friendly_changes_cannot_change_freeze(example,fault):
    rows,p,rt,m,pred,f=copy.deepcopy(example)
    if fault=='point':f['cells']['mixed-plus-sentinel-at400-D16']['branches'][0]['queue_s']=.021
    if fault=='eligibility':f['cells']['mixed-plus-sentinel-at50-D16']['status']='phase_ambiguous'
    if fault=='spread':f['scale_half_width']=.03
    if fault=='training':next(r for r in rows if r['stage']=='train' and r['observer']=='off')['endpoints']['foreground-0']['ttft_s']*=1.01
    if fault=='runtime':rt['num_gpu_blocks']+=1
    if fault=='plan':p['queue_absolute_s']*=2
    f['sha256']=digest({k:v for k,v in f.items() if k!='sha256'})
    with pytest.raises(ValueError):q.evaluate(rows,p,rt,m,pred,f)


def test_branch_assignment_is_not_permission_to_expand_budgets():
    spec=q.policy();cell=dict(branches=[dict(queue_s=.0067),dict(queue_s=.1752)])
    assert q.assign(.1639,cell,spec)['passed']
    assert not q.assign(.0204,cell,spec)['passed']
    tie=q.assign((.0067+.1752)/2,cell,spec)
    assert tie['branch'] is None


def test_residuals_observer_shift_and_phase_difference_stay_separate(example):
    rows,p,rt,m,pred,f=example;r=q.evaluate(rows,p,rt,m,pred,f)
    assert not r['candidate_passed'] and not r['queue_prediction_budget_passed'] and not r['measurement_qualification_passed']
    pairs={v['trial_id']:v for v in r['observer_checks']}
    assert pairs['ct-069']['category']=='expected_phase_difference'
    assert pairs['ct-107']['category']=='expected_phase_difference'
    assert pairs['ct-086']['category']=='same_phase_observer_shift'
    assert pairs['ct-118']['category']=='same_phase_observer_shift'
    # The calibration-derived envelope does not explain every historical outlier.
    assert any(v['trial_id']=='ct-120' and not v['passed'] for v in r['sample_checks'])
    original=json.loads((Path(__file__).resolve().parents[1]/'results/prefill-clock-4c89188-queue-phase/native/held-report.json').read_text())
    assert r['endpoint_scores']==original['endpoint_scores'] and r['anchor_phase_scores']==original['anchor_phase_scores']
    assert len(r['sample_checks'])==64 and len(r['observer_checks'])==32
    assert any(v['case']=='mixed-plus-sentinel-at400-D16' and v['observer']=='light' and v['branch']=='1' for v in r['unobserved_branches'])
    assert not r['model_accepted'] and not r['http_accepted'] and not r['mpc_accepted']


def test_proposed_inputs_are_distinct_without_changing_old_protocol():
    from scripts.run_prefill_service import code_identity
    original=clock.protocol();p=q.protocol()
    assert len(p['order'])==136 and all(t['trial_id'].startswith('qp-') for t in p['order'])
    assert not {t['paired_content_key'] for t in original['order']}&{t['paired_content_key'] for t in p['order']}
    assert p['cases']==original['cases'] and p['queue_absolute_s']==original['queue_absolute_s']
    assert p['queue_phase_policy_sha256']==digest(q.policy())
    assert clock.protocol()==original
    assert 'experiments/prefill-queue-phase-v1.json' in code_identity()


def test_expected_phase_switches_can_pass_without_widening_any_budget(example):
    # Synthetic held observations test scorer semantics, not native accuracy.
    rows,p,rt,m,pred,f=copy.deepcopy(example)
    for r in rows:
        if r['stage']=='held_delayed':
            cell=f['cells'][r['case']];i=(r['block']+int(r['observer']=='light'))%len(cell['branches'])
            r['clocks']['sentinel']['queue_residence_s']=cell['branches'][i]['queue_s']
    report=q.evaluate(rows,p,rt,m,pred,f)
    assert report['candidate_passed'] and report['queue_prediction_budget_passed'] and report['measurement_qualification_passed']
    assert not report['point_baseline_candidate_passed']
    assert any(r['category']=='expected_phase_difference' for r in report['observer_checks'])
    assert not report['model_accepted'] and not report['mpc_accepted']


def test_unreviewed_timing_source_cannot_receive_an_analytic_claim(example,monkeypatch):
    rows,p,rt,m,pred,_=example;spec=q.policy();spec['timing_source_sha256']['prefill_lab/async_predictor.py']='bad'
    monkeypatch.setattr(q,'policy',lambda:spec)
    with pytest.raises(ValueError,match='Unreviewed'):q.freeze(rows,p,rt,m,pred)


def test_too_many_branches_and_zero_width_have_explicit_semantics(example):
    _,p,rt,m,pred,f=example;name='mixed-plus-sentinel-at400-D16';case=next(c for c in p['cases'] if c['id']==name)
    spec=q.policy();spec['maximum_branches']=1
    with pytest.raises(ValueError,match='branches'):q.phase_cell(case,m['parameters_s'],p,rt,pred[name]['sentinel_queue_residence_s'],spec,f['scale_half_width'])
    cell=q.phase_cell(case,m['parameters_s'],p,rt,pred[name]['sentinel_queue_residence_s'],q.policy(),0.)
    assert cell['status']=='point' and len(cell['branches'])==1


def test_coincident_commit_anchors_are_not_arbitrarily_chosen():
    reference={'nodes':[dict(id=0,name='commit',end_s=.001),dict(id=1,name='commit',end_s=.001)]}
    with pytest.raises(ValueError,match='Ambiguous'):q.commit_node(reference,.001)
