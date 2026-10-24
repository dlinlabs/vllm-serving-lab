"""Round8: preserve collection and invalid evidence when analysis/observers fail."""
import asyncio
import json
import math
from types import SimpleNamespace as NS
import pytest

from prefill_lab.contracts import digest
from prefill_lab.runtime import Trial,ExternalCapture
from tests.test_prefill_collector import FakeCore,Clock,Tokenizer,small_plan,good_capture,output


@pytest.mark.parametrize('error',[FileNotFoundError('archived input absent'),ModuleNotFoundError('analysis import absent'),RuntimeError('analysis failed')])
def test_freeze_error_retains_blocks4_5_and_later_stages(monkeypatch,tmp_path,error):
    import prefill_lab.suite as s
    import prefill_lab.deep_diagnostic as diagnostic
    monkeypatch.setattr(diagnostic,'profile_audit',lambda *a:dict(valid=True))  # This test mocks every capture.
    from prefill_lab.tokens import materialize
    cases,_=small_plan()
    order=[dict(trial_id=tid,case='single',stage=stage,observer='light',**extra) for tid,stage,extra in (
        ('warm','warmup',{}),('observer','observer_check',{}),
        ('train3','main',dict(block=3,analysis_role='train')),
        ('freeze3','drift_reference',dict(block=3,position='end')),
        ('held4','main',dict(block=4,analysis_role='held_case')),
        ('held5','main',dict(block=5,analysis_role='held_case')),
        ('bridge','http_bridge',{}),('deep','deep_diagnostic',{}))]
    transfers=[dict(attempt_id='transfer')]
    manifest=materialize(cases,order,Tokenizer(),model='m',revision='a'*40,tokenizer_sha256='h')
    monkeypatch.setattr(s,'inputs',lambda:(cases,order,transfers));monkeypatch.setattr(s,'runtime_preflight',lambda *a:{})
    monkeypatch.setattr(s,'host_identity',lambda:{});monkeypatch.setattr(s,'audit_capture',lambda *a,**k:dict(valid=True))
    monkeypatch.setattr(s,'dispatcher_parity',lambda *a:dict(matches=True));monkeypatch.setattr(s,'observer_gate',lambda *a:dict(status='passed'))
    monkeypatch.setattr(s,'drift_report',lambda *a:{})
    monkeypatch.setattr(s,'freeze_models',lambda *a:{})
    def fail(*a):raise error
    monkeypatch.setattr(s,'freeze_eligibility',fail)
    history=[]
    async def controlled(client,t,*args):
        history.append(t['trial_id']);c,_=good_capture();c.update(trial_id=t['trial_id'],start=1.)
        return c
    async def bridge(client,http,url,t,*args):return dict(capture=await controlled(client,t))
    async def transfer(*a):history.append('transfer')
    async def sleep(*a):pass
    monkeypatch.setattr(s,'controlled_trial',controlled);monkeypatch.setattr(s,'http_bridge',bridge)
    monkeypatch.setattr(s,'transfer',transfer);monkeypatch.setattr(s.asyncio,'sleep',sleep)
    class Client:
        async def command(self,*a,**kw):return dict(workers=[dict(dispatch_table={})])
    summary=asyncio.run(s.run(Client(),None,'http://test',manifest,{},tmp_path))
    assert history==['warm','observer','train3','freeze3','held4','held5','bridge','transfer','deep']
    assert summary['complete'] and summary['failures']==[] and not summary['model_accepted']
    frozen=json.loads((tmp_path/'model-freeze.json').read_text())
    assert frozen['status']=='inconclusive' and frozen['error_type']==type(error).__name__
    assert frozen['reason']==str(error) and frozen['failed_stage']=='model_freeze'
    assert not frozen['held_labels_used'] and summary['model_freeze_sha256']==digest(frozen)
    assert (tmp_path/'held4.json').exists() and (tmp_path/'held5.json').exists()


@pytest.mark.parametrize('prior_invalid',[True,False])
def test_finished_requests_cannot_relabel_invalid_trial(prior_invalid):
    core=FakeCore();clock=Clock()
    trial=Trial(core,dict(trial_id='t',observer='off',epoch='e',requests=[dict(id='foreground-0',role='foreground',input_ids=[2]*4,max_tokens=1,at=0)]),lambda r:r,clock=clock)
    trial.begin();rid=next(iter(trial.requests))
    if prior_invalid:trial.status='invalid'
    else:trial.errors.append('earlier integrity failure')
    batch=NS(outputs=[NS(request_id=rid,new_token_ids=[2],finish_reason='length')])
    trial.after_output(output([]),{0:batch})
    assert trial.status=='invalid' and trial.cancel.is_set() and batch.outputs==[]
    assert trial.requests[rid]['finished'] is not None


@pytest.mark.parametrize('external',[True,False])
def test_observer_exception_does_not_kill_engine_or_leak_private_outputs(monkeypatch,external):
    import prefill_lab.plugin as p
    core=FakeCore();monkeypatch.setattr(p,'installed_sources',lambda:{})
    monkeypatch.setattr(p.sys,'setswitchinterval',lambda *a:None)
    own=NS(request_id='plab/t/foreground-0',new_token_ids=[2],finish_reason='length')
    other=NS(request_id='ordinary',new_token_ids=[3],finish_reason=None)
    batch=NS(outputs=[own,other]);native_result={0:batch};calls=[]
    core.scheduler.schedule=lambda:None
    def native_update(*a,**kw):calls.append('native');return native_result
    core.scheduler.update_from_output=native_update;p.core_runtime(core)
    if external:
        trial=ExternalCapture(core,'external','light')
        trial.requests[own.request_id]={}
    else:
        trial=Trial(core,dict(trial_id='t',observer='off',epoch='e',requests=[dict(id='foreground-0',role='foreground',input_ids=[2],max_tokens=1,at=0)]),lambda r:r)
    def broken(*a):raise AttributeError('observer failure')
    trial.after_output=broken;core._prefill_trial=trial
    assert core.scheduler.update_from_output(output([])) is native_result
    assert trial.status=='invalid' and trial.cancel.is_set()
    assert 'AttributeError' in trial.errors[-1] and 'observer failure' in trial.errors[-1]
    assert batch.outputs==([own,other] if external else [other])
    # Further native updates still execute; real engine failures are not hidden.
    core._prefill_trial=None;assert core.scheduler.update_from_output(output([])) is native_result
    assert calls==['native','native']


def test_native_engine_update_errors_are_not_hidden(monkeypatch):
    import prefill_lab.plugin as p
    core=FakeCore();monkeypatch.setattr(p,'installed_sources',lambda:{})
    monkeypatch.setattr(p.sys,'setswitchinterval',lambda *a:None)
    core.scheduler.schedule=lambda:None
    def broken(*a):raise RuntimeError('native engine failure')
    core.scheduler.update_from_output=broken;p.core_runtime(core)
    core._prefill_trial=NS(after_output=lambda *a:pytest.fail('observer must not run'))
    with pytest.raises(RuntimeError,match='native engine failure'):core.scheduler.update_from_output(output([]))


def test_abort_cannot_release_new_foreground_after_late_background_output():
    core=FakeCore();trial=Trial(core,dict(trial_id='t',observer='off',epoch='e',requests=[
        dict(id='background-0',role='background',input_ids=[2],max_tokens=2,at=0),
        dict(id='foreground-0',role='foreground',input_ids=[2],max_tokens=1,at=0)]),lambda r:r)
    trial.begin();trial.abort('cancelled');rid=next(iter(trial.backgrounds))
    trial.after_output(output([]),{0:NS(outputs=[NS(request_id=rid,new_token_ids=[2],finish_reason='length')])})
    assert trial.release is None and len(core.added)==1 and trial.status=='invalid'


def test_generator_comparison_preserves_types_ids_and_meaningful_values():
    from tests.test_prefill_design import assert_generated_json_equal
    expected=dict(tokens=2048,valid=True,id='trial-1',offsets=[.05,1518.29432734343])
    rounded=dict(expected,offsets=[math.nextafter(.05,math.inf),math.nextafter(1518.29432734343,math.inf)])
    assert_generated_json_equal(rounded,expected)
    for changed in (dict(expected,tokens=2049),dict(expected,tokens=2048.),dict(expected,valid=1),
                    dict(expected,id='trial-2'),dict(expected,offsets=[.05001,1518.29432734343])):
        with pytest.raises(AssertionError):assert_generated_json_equal(changed,expected)
