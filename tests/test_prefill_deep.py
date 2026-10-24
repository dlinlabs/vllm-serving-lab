"""Foreground profiler coverage and diagnostic scope; no real GPU qualification."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from prefill_lab.plugin import WorkerRecorder
from prefill_lab.deep_diagnostic import profile_audit
from tests.test_prefill_collector import FakeTorch,output


def recorded(monkeypatch,tmp_path,steps=5):
    monkeypatch.setenv('PREFILL_LAB_ARTIFACT_DIR',str(tmp_path))
    state=NS(starts=0,stops=0,steps=0,events=[])
    class Profile:
        def start(self):state.starts+=1
        def stop(self):state.stops+=1
        def step(self):state.steps+=1
        def export_chrome_trace(self,path):Path(path).write_text(json.dumps({'traceEvents':state.events+[{'cat':'kernel','name':'mock-kernel'}]}))
    class Range:
        def __init__(self,name):self.name=name
        def __enter__(self):return self
        def __exit__(self,*args):state.events.append(dict(cat='user_annotation',name=self.name))
    torch=FakeTorch();torch.profiler=NS(profile=lambda **kw:Profile(),record_function=Range,ProfilerActivity=NS(CPU=0,CUDA=1))
    recorder=WorkerRecorder(torch,'deep','deep');bg='plab/deep/background-0';fg='plab/deep/foreground-0'
    assert recorder.begin(output([(fg,0,1,16)],['fg']),dummy_run=True) is None
    for i in range(10):
        entry=recorder.begin(output([(bg,i,1,16)],['bg'] if i==0 else []));recorder.end(entry);entry[2].ready=True
    assert state.starts==0
    for i in range(steps):
        entry=recorder.begin(output([(bg,10+i,1,16),(fg,i,1,16)],['fg'] if i==0 else []))
        recorder.end(entry);entry[2].ready=True
    if steps<5:recorder.stop_profile()
    capture=dict(trial_id='deep',requests={bg:dict(role='background'),fg:dict(role='foreground')},workers=[recorder.resolve()])
    return capture,recorder,state


def test_profile_waits_through_background_and_marks_exact_foreground_steps(monkeypatch,tmp_path):
    capture,recorder,state=recorded(monkeypatch,tmp_path)
    assert (state.starts,state.stops,state.steps)==(1,1,5)
    verdict=profile_audit(capture,tmp_path)
    assert verdict['valid'] and verdict['first_foreground_worker_step']==11
    assert not verdict['cuda_scope_qualified'] and not verdict['model_accepted']
    assert not capture['workers'][0]['cuda_scope_qualified']
    recorder.end(recorder.begin(output([('plab/deep/foreground-0',6,1,16)])))
    assert state.starts==1 and state.steps==5 and not recorder.profile_ranges


def test_early_stop_cannot_pass_foreground_profile_audit(monkeypatch,tmp_path):
    capture,_,state=recorded(monkeypatch,tmp_path,2)
    assert state.stops==1 and not profile_audit(capture,tmp_path)['valid']


@pytest.mark.parametrize('damage',['legacy','wrong_steps','hash','markers','missing'])
def test_wrong_or_missing_profile_evidence_is_rejected(monkeypatch,tmp_path,damage):
    capture,_,_=recorded(monkeypatch,tmp_path);proof=capture['workers'][0]['deep_profile']
    path=tmp_path/proof['artifact']['name']
    if damage=='legacy':capture['workers'][0].pop('deep_profile')
    elif damage=='wrong_steps':proof['step_ids']=proof['step_ids'][::-1]
    elif damage=='hash':proof['artifact']['sha256']='0'*64
    elif damage=='markers':
        import hashlib
        path.write_text(json.dumps({'traceEvents':[{'cat':'kernel','name':'mock'}]}))
        proof['artifact']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage=='missing':path.unlink()
    assert not profile_audit(capture,tmp_path)['valid']


@pytest.mark.parametrize('bad_profile',[False,True])
def test_targeted_run_keeps_warmups_and_never_accepts_suite_or_gpu(monkeypatch,tmp_path,bad_profile):
    from prefill_lab import deep_diagnostic as diagnostic,suite
    order=[dict(trial_id=tid,stage=stage,case=case) for tid,stage,case in
        [('w1','warmup','one'),('w2','warmup','two'),('d1','deep_diagnostic','one'),('d2','deep_diagnostic','two')]]
    monkeypatch.setattr(suite,'inputs',lambda:([],order,[]))
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:{'valid':True})
    monkeypatch.setattr(suite,'host_identity',lambda:{})
    async def metrics(*a):return {}
    monkeypatch.setattr(suite,'metrics_sample',metrics)
    history=[]
    async def controlled(client,trial,*args):
        history.append(trial['trial_id'])
        return dict(requests={'r':{'finished':1}},workers=[{'rows':[]}])
    monkeypatch.setattr(suite,'controlled_trial',controlled)
    monkeypatch.setattr(diagnostic,'audit_capture',lambda *a:{'valid':True})
    monkeypatch.setattr(diagnostic,'dispatcher_parity',lambda *a:{'matches':True})
    monkeypatch.setattr(diagnostic,'profile_audit',lambda *a:{'valid':not bad_profile,'cuda_scope_qualified':False})
    class Client:
        async def command(self,*a,**kw):return {'workers':[{'dispatch_table':{}}]}
    if bad_profile:
        with pytest.raises(RuntimeError,match='profile integrity'):asyncio.run(diagnostic.run(Client(),None,'url',{},tmp_path))
    else:asyncio.run(diagnostic.run(Client(),None,'url',{},tmp_path))
    summary=json.loads((tmp_path/'diagnostic.json').read_text())
    assert history==(['w1','w2','d1'] if bad_profile else ['w1','w2','d1','d2'])
    assert summary['diagnostic_complete'] is (not bad_profile)
    assert not summary['model_accepted'] and not summary['gpu_validated'] and not summary['complete_experiment_passed']
    assert not (tmp_path/'suite.json').exists() and not (tmp_path/'model-freeze.json').exists()


def test_overlap_union_does_not_treat_async_durations_as_additive():
    from scripts.analyze_prefill_capture import intervals
    result=intervals([dict(start=0,end=10),dict(start=5,end=15)])
    assert result['sum_s']==20 and result['union_s']==15 and result['max_inflight']==2
    nested=intervals([dict(start=0,end=20),dict(start=5,end=10)])
    assert nested['span_s']==20 and nested['sum_over_union']==1.25
