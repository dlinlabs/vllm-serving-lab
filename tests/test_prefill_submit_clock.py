"""CPU submit-return hypothesis contracts, not native model acceptance."""
import copy
import json
from pathlib import Path
import pytest
from prefill_lab import submit_clock as submit,event_clock_calibration as clock,queue_probe as probe,queue_phase as phase
from prefill_lab.contracts import digest
from prefill_lab.async_predictor import PipelineCosts
from test_prefill_async_predictor import runtime_metadata


def evidence():
    p=probe.protocol();captures=[]
    for i,t in enumerate(p['order']):
        if t['stage']!='train' or t['observer']!='light':continue
        spans=[.02+(i%4)*.001,.002+(i%3)*.0001]
        steps=[dict(step_id=str(j),start=j*.1,end=j*.1+.08,prefill=1024 if j==0 else 0) for j in range(2)]
        workers=[dict(step_id=str(j),host_start=j*.1+.001,host_end=j*.1+.001+v) for j,v in enumerate(spans)]
        captures.append(dict(trial_id=t['trial_id'],observer='light',status='complete',errors=[],foreground_done=.18,steps=steps,workers=[dict(rows=workers)]))
    return p,captures


def test_only_training_labels_enter_constants_and_domain():
    p,caps=evidence();f=submit.training_labels(caps,p)
    assert len(f['training_trial_ids'])==24 and len(f['samples'])==48
    assert f['prefill_lower_s']<=f['constants_s']['prefill']<=f['prefill_upper_s']
    poisoned=[dict(trial_id=t['trial_id'],workers='unreadable') for t in p['order'] if t['stage']!='train']
    assert submit.training_labels(caps+poisoned,p)==f
    assert not f['probe_labels_used'] and not f['physical_stage_identified'] and not f['model_accepted']


@pytest.mark.parametrize('damage',['missing','duplicate','mode','worker','negative','nan','span','empty','range','invalid'])
def test_invalid_submit_evidence_fails(damage):
    p,caps=evidence();w=caps[0]['workers'][0]['rows'][0]
    if damage=='missing':caps.pop()
    if damage=='duplicate':caps.append(copy.deepcopy(caps[0]))
    if damage=='mode':caps[0]['observer']='off'
    if damage=='worker':caps[0]['workers'][0]['rows'].append(copy.deepcopy(w))
    if damage=='negative':w['host_end']=w['host_start']-.001
    if damage=='nan':w['host_end']=float('nan')
    if damage=='span':w['host_end']=1.
    if damage=='empty':caps[0]['steps']=[]
    if damage=='range':
        w['host_end']=.15;caps[0]['steps'][0]['end']=.2
    if damage=='invalid':caps[0]['status']='invalid'
    with pytest.raises(ValueError):submit.training_labels(caps,p)


def test_cost_serialization_cannot_drop_mixed_or_submit_constants():
    p=probe.protocol();theta=[.009,.09,.005,.01,.001,.004]
    a=submit.costs(theta,p,.03,.002);b=submit.costs(theta,p,.04,.002)
    assert digest(a.to_dict())!=digest(b.to_dict())
    assert a.to_dict()['base']['prefill_decode']==clock.costs(theta,p,clock.specification()).to_dict()['prefill_decode']
    assert a.launches['forward']['cpu_after'](dict(prefill=0))==.002
    assert a.launches['forward']['cpu_after'](dict(prefill=1))==.03
    with pytest.raises(ValueError):PipelineCosts.from_dict(a.to_dict())
    with pytest.raises(ValueError):submit.costs(theta,p,-.01,.002)


@pytest.mark.parametrize('background',[0,16])
def test_affine_cpu_curves_match_separate_own_rollouts(background):
    p=probe.protocol();runtime=runtime_metadata();theta=[.009,.09,.005,.01,.001,.004]
    case=next(c for c in p['cases'] if c['sentinel'] and c['background_decoders']==background)
    one=dict(p,cases=[dict(case,sentinel=None)]);nominal=.035;decode=.002
    reference=submit.own_rollouts(theta,one,runtime,nominal,decode)[case['id']]
    curves,release,ends=submit.reference_curves(reference,.015,.06,nominal,decode)
    for value in (.015,.028,.035,.049,.06):
        other=submit.own_rollouts(theta,one,runtime,value,decode)[case['id']]
        assert phase.value(release,value)==pytest.approx(other['release_s'],abs=1e-9)
        polls={n['batch']:n['start_s']-other['release_s'] for n in other['nodes'] if n['name']=='schedule'}
        assert set(polls)=={j for j,c in curves}
        assert all(phase.value(curve,value)==pytest.approx(polls[j],abs=1e-9) for j,curve in curves)


def test_zero_and_overwide_domains_do_not_produce_branch_claims():
    with pytest.raises(ValueError):submit.reference_curves(dict(nodes=[],steps=[],release_s=0.),.1,.01,.02,.002)
