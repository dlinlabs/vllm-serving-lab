"""Round7 regressions: trial boundaries, pre-held freeze, and engine survival."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest

from prefill_lab.contracts import digest,save
from prefill_lab.analysis import errors,rows_from_capture,freeze_models,calibration_readiness,basis,fit,FAMILIES
from prefill_lab.scoring import freeze_eligibility,verify_freeze
from tests.test_prefill_collector import calibration_fixture,FakeCore,FakeTorch,output,small_plan,Tokenizer,good_capture


def test_loco_cumulative_resets_at_each_trial_with_noisy_labels():
    captures,order,_=calibration_fixture();rng=np.random.default_rng(32)
    for c in captures:
        # Preserve nonoverlapping spans; perturb labels within each observed span.
        for s in c['steps']:s['start']=s['end']-(s['end']-s['start'])*rng.uniform(.96,1.)
    frozen=freeze_models(captures,order)
    training={}
    for c,t in zip(captures,order):training.setdefault(t['case'],[]).extend(rows_from_capture(c))
    for family in FAMILIES:
        if family not in frozen['fits']:continue
        for held,rows in training.items():
            part={k:v for k,v in training.items() if k!=held};model=fit(part,family,basis(part))
            by_trial={tid:[r for r in rows if r['trial_id']==tid] for tid in {r['trial_id'] for r in rows}}
            expected={k:np.mean([errors(rs,model)[k] for rs in by_trial.values()]) for k in ('decode','prefill','cumulative')}
            assert frozen['loco_errors'][family][held]==pytest.approx(expected)
            first=next(iter(by_trial.values()))
            twice=first+[dict(r,trial_id='another',work=[dict(w,request_id=w['request_id'].replace('plab/','repeat/')) for w in r['work']]) for r in first]
            assert errors(twice,model)==pytest.approx(errors(first,model))


def test_fixed_phase_readiness_does_not_maximize_hundreds_of_step_cvs():
    captures,order,_=calibration_fixture();training={}
    for c,t in zip(captures,order):training.setdefault(t['case'],[]).extend(rows_from_capture(c))
    rows=next(iter(training.values()))
    # One noisy matched step used to trip the maximum-step rule, even when the
    # whole phase is repeatable. The maximum remains visible as a diagnostic.
    noisy=next(r for r in rows if r['prefill']==0)
    noisy['y']*=1.3
    r=calibration_readiness(training,order)
    assert r['stratum_count']==12 and r['descriptive_max_step_cv']>.05
    assert r['status']=='within_tested_noise_range' and not r['step_cv_controls_gate']
    # A sustained phase-level instability still stops structural claims.
    for row in rows:
        if row['trial_id']==rows[0]['trial_id']:row['y']*=1.4
    assert calibration_readiness(training,order)['status']=='stop_structural_claim'
    rows[0]['work'][0]['position']+=1
    assert calibration_readiness(training,order)['status']=='inconclusive'


def freeze_fixture():
    from scripts.build_prefill_design import cases
    captures,order,_=calibration_fixture();frozen=freeze_models(captures,order)
    for case,n in (('single-L1024-D0',20),('composition-uniform-D16',10)):
        for i in range(n):
            tid=f'off-{case}-{i}';order.append(dict(trial_id=tid,stage='observer_check',observer='off',case=case))
            captures.append(dict(trial_id=tid,status='complete',errors=[],release=0.,requests={f'{tid}/foreground-{j}':dict(role='foreground',first=.1+i*.0001+j*.001) for j in range(1 if n==20 else 4)}))
    table={f'{n}:{u}':['NONE',n,None,False] for n in range(1,2049) for u in (0,1)}
    return cases(),order,captures,table,frozen


def test_eligibility_is_nominal_frozen_before_blocks4_5_and_hash_bound(tmp_path):
    cases,order,captures,dispatch,frozen=freeze_fixture()
    from prefill_lab.scoring import reference_mapping
    mapping=reference_mapping()
    f=freeze_eligibility(frozen,cases,order,captures,dispatch,mapping)
    assert f['eligibility_plan']['frozen_at']=='block3_end'
    assert 'MAD' in f['eligibility_plan']['jitter_definition']
    bad=dict(trial_id='held-later',release_state='unreadable',steps='invalid')
    assert freeze_eligibility(frozen,cases,order,[*captures,bad],dispatch,mapping)==f
    for c in captures:save(tmp_path/f"{c['trial_id']}.json",c)
    save(tmp_path/'suite.json',dict(model_freeze_sha256=digest(f)))
    verify_freeze(tmp_path,cases,order,f,dispatch,mapping)
    changed=copy.deepcopy(f);changed['eligibility_plan']['frozen_at']='after-held'
    with pytest.raises(ValueError,match='hash'):verify_freeze(tmp_path,cases,order,changed,dispatch,mapping)
    with pytest.raises(ValueError,match='eligibility'):verify_freeze(tmp_path,cases,order,f,{},mapping)
    save(tmp_path/f"{captures[-1]['trial_id']}.json",dict(tampered=True))
    with pytest.raises(ValueError,match='capture changed'):verify_freeze(tmp_path,cases,order,f,dispatch,mapping)


def test_bad_external_traffic_and_late_timer_do_not_kill_engine():
    from prefill_lab.runtime import ExternalCapture,Trial
    from prefill_lab.plugin import guarded_add,handle_arrival
    core=FakeCore();replies=[];core._send_abort_outputs_to_client=lambda ids,client:replies.append((ids,client))
    c=ExternalCapture(core,'http','light');core._prefill_trial=c
    request=NS(request_id='bad',client_index=3,sampling_params=NS(extra_args={'prefill_lab_role':'wrong'}),num_prompt_tokens=4,max_tokens=1)
    adds=[];original=lambda *a,**kw:adds.append(a)
    guarded_add(core,request,original)
    assert c.errors and c.status=='invalid' and not adds and replies==[(['bad'],3)]
    # A stale controlled-trial timer can arrive during an external capture.
    handle_arrival(core,('old-epoch',['not-present'],0.))
    request.request_id='valid';request.sampling_params.extra_args={}
    c=ExternalCapture(core,'next','light');core._prefill_trial=c
    guarded_add(core,request,original);assert len(adds)==1
    guarded_add(core,request,original)
    assert len(adds)==1 and c.status=='invalid' and 'Duplicate' in c.errors[-1]
    trial=Trial(core,dict(trial_id='controlled',observer='off',epoch='e',requests=[]),lambda r:r)
    core._prefill_trial=trial;guarded_add(core,request,original)
    assert trial.status=='invalid' and len(adds)==1


def test_worker_finish_disables_between_trials_even_while_events_pending(monkeypatch):
    import prefill_lab.plugin as p
    recorder=p.WorkerRecorder(FakeTorch(),'one','light');monkeypatch.setattr(p,'_WORKER',recorder)
    entry=recorder.begin(output([('r',0,4,4)],['r']));recorder.end(entry)
    assert p.finish_worker_recorder()['pending']==1
    assert recorder.begin(output([('unrelated',0,4,4)],['unrelated'])) is None
    assert len(recorder.pending)==1
    entry[2].ready=True
    assert p.finish_worker_recorder()['pending']==0 and p._WORKER is None
    next_recorder=p.WorkerRecorder(FakeTorch(),'two','light')
    assert next_recorder.begin(output([('dummy',0,4,4)],['dummy']),dummy_run=True) is None
    assert not next_recorder.pending


def test_decode_context_matches_actual_pre_step_position():
    from prefill_lab.causal import rollout
    case=dict(background_decoders=1,foreground_tokens=[4],foreground_offsets_ms=[0],sentinel=None)
    model=dict(features=['mode:NONE','decode','prefill','decode_context'],coefficients=[.001,.001,.001,1e-8])
    table={f'{n}:{u}':['NONE',n,None,False] for n in range(1,2049) for u in (0,1)}
    result=rollout(case,model,table)
    assert result['steps'][0]['decode_context']==1024
    # Next step includes background position1025 and foreground position4.
    assert result['steps'][1]['decode_context']==1029


def test_graph_context_absent_and_v2_backend_metadata(monkeypatch):
    import sys
    import prefill_lab.plugin as p
    monkeypatch.setitem(sys.modules,'vllm.forward_context',NS(is_forward_context_available=lambda:False,get_forward_context=lambda:pytest.fail('must not assert')))
    assert p.forward_context_or_none() is None
    assert p.attention_backends(NS(attn_groups=[[NS(backend='FLASH_ATTN')]]))==['FLASH_ATTN']


def test_main_failure_closes_interleaved_drift_stage(monkeypatch,tmp_path):
    import prefill_lab.suite as s
    from prefill_lab.tokens import materialize
    cases,_=small_plan();order=[dict(trial_id=tid,case='single',stage=stage,observer='light') for tid,stage in (
        ('observer','observer_check'),('drift-start','drift_reference'),('main-failure','main'),('drift-end','drift_reference'))]
    manifest=materialize(cases,order,Tokenizer(),model='m',revision='a'*40,tokenizer_sha256='h')
    monkeypatch.setattr(s,'inputs',lambda:(cases,order,[]));monkeypatch.setattr(s,'runtime_preflight',lambda *a:{})
    monkeypatch.setattr(s,'host_identity',lambda:{});monkeypatch.setattr(s,'audit_capture',lambda *a,**k:dict(valid=True))
    monkeypatch.setattr(s,'dispatcher_parity',lambda *a:dict(matches=True));monkeypatch.setattr(s,'observer_gate',lambda *a:dict(status='passed'))
    async def controlled(client,t,*a):
        if t['stage']=='main':raise RuntimeError('bad main')
        c,_=good_capture();c.update(trial_id=t['trial_id'],start=1.)
        return c
    async def sleep(*a):pass
    monkeypatch.setattr(s,'controlled_trial',controlled);monkeypatch.setattr(s.asyncio,'sleep',sleep)
    class Client:
        async def command(self,op,**kw):
            if op=='finish':raise RuntimeError('No active trial')
            return dict(workers=[dict(dispatch_table={})])
    summary=asyncio.run(s.run(Client(),None,'http://test',manifest,{},tmp_path))
    assert not summary['complete']
    for stage in ('main','drift_reference'):
        assert summary['stages'][stage]['integrity']=='incomplete'
        assert summary['stages'][stage]['failure_reason']=='bad main'


def test_engine_runtime_sets_submillisecond_gil_interval(monkeypatch):
    import sys
    import prefill_lab.plugin as p
    calls=[];monkeypatch.setattr(sys,'setswitchinterval',calls.append)
    monkeypatch.setattr(p,'installed_sources',lambda:{})
    core=FakeCore();core.scheduler.schedule=lambda:None;core.scheduler.update_from_output=lambda *a:None
    p.core_runtime(core);p.core_runtime(core)
    assert calls==[.0005]
