"""Synthetic receipt/protocol/audit tests, not native measurement acceptance."""
import asyncio
import copy
import itertools
import json
from types import SimpleNamespace as NS
import pytest
from prefill_lab import queue_probe as qp,receipt
from prefill_lab.contracts import digest,save
from test_prefill_collector import Clock,FakeCore,output


def observed():
    c=Clock();rec=receipt.ReceiptRecorder(c);c.now=10.;rec.release()
    a=output([('x',0,1,1)]);b=output([('x',1,1,1)])
    c.now=10.01;rec.schedule(a);c.now=10.02;rec.schedule(b)
    c.now=10.1;rec.commit(a);c.now=10.11
    rec.receive('plab/t/sentinel',10.05,10.11,NS(running=[],waiting=[],requests={}))
    c.now=10.2;rec.commit(b)
    cap=dict(start=9.,end=11.,release=10.,requests={'plab/t/sentinel':dict(role='sentinel',queue_posted=10.05,enqueue=10.11,first=10.3)})
    return rec.capture(),cap


def test_overlap_receipt_and_independent_light_crosscheck():
    record,cap=observed();r=receipt.validate(record,cap)['plab/t/sentinel']
    assert r['pending_sequences']==[2] and r['scheduled']==2 and r['committed']==1
    cap.update(observer='light',steps=[dict(start=10.,end=10.1),dict(start=10.015,end=10.2)])
    receipt.validate(record,cap);cap['steps'][0]['end']=10.09
    with pytest.raises(ValueError,match='LIGHT chronology'):receipt.validate(record,cap)


@pytest.mark.parametrize('fault',['missing','duplicate','counter','commit','order','release','queue','state','overflow','pending','nan','lifetime'])
def test_corrupt_receipt_rejected(fault):
    record,cap=observed();events=record['events'];r=next(e for e in events if e['kind']=='receipt')
    if fault=='missing':events.remove(r)
    if fault=='duplicate':events.insert(events.index(r),copy.deepcopy(r))
    if fault=='counter':r['scheduled']+=1
    if fault=='commit':events[-1]['sequence']=1
    if fault=='order':events[-1]['at']=9.
    if fault=='release':cap['release']=9.
    if fault=='queue':r['posted']+=.01
    if fault=='state':r['state']['requests']['sentinel']=dict(computed=0,tokens=512,placeholders=0)
    if fault=='overflow':record['overflow']=1
    if fault=='pending':record['pending']=1
    if fault=='nan':r['observation_end']=float('nan')
    if fault=='lifetime':cap['end']=10.
    with pytest.raises(ValueError):receipt.validate(record,cap)


def test_bounds_and_unmatched_commit():
    r=receipt.ReceiptRecorder(lambda:1.,1);r.release()
    with pytest.raises(RuntimeError):r.release()
    assert r.capture()['overflow']==1
    with pytest.raises(KeyError):r.commit(output([('x',0,1,1)]))


def test_frozen_protocol_balances_cases_and_keeps_training_before_probes():
    p=qp.protocol();assert len(p['cases'])==16 and len(p['order'])==232
    assert sum(t['stage']=='train' for t in p['order'])==48
    assert max(i for i,t in enumerate(p['order']) if t['stage']=='train')<min(i for i,t in enumerate(p['order']) if t['stage']=='calibration_probe')
    assert {c['sentinel']['arrival_ms'] for c in p['cases'] if c['sentinel']}=={50,200,400,800}
    for c in p['cases']:
        if not c['sentinel']:continue
        orders={tuple(t['arm'] for t in p['order'] if t['case']==c['id'] and t['block']==b) for b in range(1,7)}
        assert orders==set(itertools.permutations(('baseline','receipt','light')))
    assert not {t['paired_content_key'] for t in qp.clock.protocol()['order']} & {t['paired_content_key'] for t in p['order']}


@pytest.fixture(scope='module')
def manifest():
    from prefill_lab.tokens import materialize
    p=qp.protocol();return materialize(p['cases'],p['order'],NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[]),
        model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)


def test_manifest_rejects_rehashed_arm_content_swap(manifest):
    qp.validate_manifest(manifest);bad=copy.deepcopy(manifest)
    row=bad['trials']['qr-066'][0];pool=copy.deepcopy(bad['pools'][row['token_key']])
    pool['input_ids'][0]=3-pool['input_ids'][0];pool['sha256']=digest(pool['input_ids'])
    bad['pools']['forged']=pool;row['token_key']='forged'
    with pytest.raises(ValueError):qp.validate_manifest(bad)


def rows(p):
    result=[]
    for t in p['order']:
        state=dict(scheduled=2,committed=1,pending_count=1,state={})
        result.append(dict(**{k:t[k] for k in ('trial_id','case','stage','block','observer','arm')},
            endpoints={'sentinel':dict(ttft_s=.2,completion_s=1.)},clocks={'sentinel':dict(queue_residence_s=.05,posting_lag_s=.001)},
            receipt=None if not t['receipt_observer'] else dict(state=state,state_sha256=digest(state))))
    return result


def test_different_state_is_unassessable_and_no_vacuous_pass():
    p=qp.protocol();data=rows(p);r=qp.analyze(data,p)
    assert len(r['samples'])==144 and len(r['excluded_warmup_ids'])==40 and len(r['clock_training_ids'])==48
    assert r['same_state_pairs']==48 and r['same_state_queue_controls_passed']
    for row in data:
        if row['arm']=='light' and row['receipt']:
            row['clocks']['sentinel']['queue_residence_s']=.2
    assert not qp.analyze(data,p)['same_state_queue_controls_passed']
    for row in data:
        if row['arm']=='light' and row['receipt']:
            row['receipt']['state']['committed']=2;row['receipt']['state_sha256']=digest(row['receipt']['state'])
    r=qp.analyze(data,p);assert r['different_state_pairs']==48 and not r['same_state_queue_controls_passed']
    assert all(c['passed'] is None for c in r['receipt_pair_checks'])
    assert not any(r[k] for k in ('candidate_passed','model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted'))


@pytest.mark.parametrize('fault',['missing','duplicate','arm','state'])
def test_report_denominators_and_membership(fault):
    p=qp.protocol();data=rows(p)
    if fault=='missing':data.pop()
    if fault=='duplicate':data[-1]=copy.deepcopy(data[0])
    if fault=='arm':data[0]['arm']='baseline'
    if fault=='state':next(r for r in data if r['receipt'])['receipt']['state']['committed']=9
    with pytest.raises(ValueError):qp.analyze(data,p)


def test_baseline_is_unobserved_and_receipt_state_precedes_add():
    from prefill_lab.runtime import Trial
    for flag in (False,True):
        c=Clock();core=FakeCore();t=Trial(core,dict(trial_id='t',observer='off',epoch='e',receipt_observer=flag,
            requests=[dict(id='sentinel',role='sentinel',at=.05,input_ids=[2],max_tokens=1)]),lambda r:r,clock=c)
        t.release=c()
        if flag:t.receipt.release()
        c.advance(.06);t.inject(['plab/t/sentinel'],c.now-.01);cap=t.capture()
        assert ('receipt' in cap)==flag and len(core.added)==1
        if flag:assert cap['receipt']['events'][-1]['state']['requests']=={}


def test_native_hook_identity_and_observer_error_isolation(monkeypatch):
    from prefill_lab import plugin
    monkeypatch.setattr(plugin,'installed_sources',lambda:None)
    rec=receipt.ReceiptRecorder(Clock());o=output([('x',0,1,1)])
    trial=NS(mode='off',receipt=rec,errors=[],after_output=lambda *a:None)
    core=NS(scheduler=NS(schedule=lambda:o,update_from_output=lambda out:{}))
    plugin.core_runtime(core);core._prefill_trial=trial
    assert core.scheduler.schedule() is o and core.scheduler.update_from_output(o)=={}
    assert [e['kind'] for e in rec.capture()['events']]==['schedule','commit']
    rec.schedule=lambda *a:(_ for _ in ()).throw(RuntimeError('observer failed'))
    assert core.scheduler.schedule() is o and trial.errors


def test_endpoint_uses_frozen_probe_plan_and_conflict_guard(monkeypatch,tmp_path,manifest):
    from prefill_lab.endpoint import Endpoint
    save(tmp_path/'tokens.json',manifest);monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(tmp_path/'tokens.json'))
    for f in ('EFFECTIVE_CALIBRATION','DELAYED_ARRIVALS','EVENT_CLOCK','LIVE_STATE'):monkeypatch.setenv('PREFILL_LAB_'+f,'0')
    monkeypatch.setenv('PREFILL_LAB_QUEUE_PROBE','1');e=Endpoint();engine=NS(engine_core=NS(call_utility_async=lambda:None))
    asyncio.run(e.init_state(engine,None,None));assert len(e.trials)==232 and not e.external_ids and e.trials['qr-066']['receipt_observer']
    monkeypatch.setenv('PREFILL_LAB_LIVE_STATE','1')
    with pytest.raises(ValueError):asyncio.run(e.init_state(engine,None,None))


@pytest.mark.parametrize('failed',[False,True])
def test_runner_collects_all_or_retains_partial_without_retry(monkeypatch,tmp_path,failed):
    from prefill_lab import suite,causal
    from test_prefill_async_predictor import runtime_metadata
    p=qp.protocol();data=rows(p);meta=runtime_metadata();calls=[]
    monkeypatch.setattr(qp,'validate_manifest',lambda *a:None);monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(suite,'host_identity',lambda:{})
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True))
    monkeypatch.setattr(qp.clock,'event_labels',lambda cap,t,c:dict(trial_id=t['trial_id']))
    monkeypatch.setattr(qp,'extract',lambda cap,t,*a:next(r for r in data if r['trial_id']==t['trial_id']))
    async def controlled(client,t,deadline,path):
        calls.append(t['trial_id'])
        if failed and len(calls)==3:
            save(path,dict(status='invalid'));raise RuntimeError('interrupted')
        return dict(status='complete')
    async def command(*a,**kw):return meta
    monkeypatch.setattr(suite,'controlled_trial',controlled)
    if failed:
        with pytest.raises(RuntimeError):asyncio.run(qp.run(NS(command=command),None,None,{},tmp_path))
    else:assert asyncio.run(qp.run(NS(command=command),None,None,{},tmp_path))['diagnostic_complete']
    summary=json.loads((tmp_path/'queue-probe.json').read_text())
    assert len(summary['completed'])==(2 if failed else 232) and bool(summary['failures'])==failed
    assert len(calls)==len(set(calls))


@pytest.mark.parametrize('failed',[False,True])
def test_offline_fit_excludes_probe_labels_and_preserves_failure(monkeypatch,tmp_path,failed):
    from scripts import fit_prefill_queue_probe as fitter
    root=tmp_path/'capture';root.mkdir();out=tmp_path/'fit';data=rows(qp.protocol())
    for r in data:
        if r['stage']!='train':r['endpoints']='poison'
    save(root/'runtime.json',{});save(root/'queue-probe-rows.json',data);save(root/'training-events.json',[])
    monkeypatch.setattr(fitter,'audit',lambda *a:None)
    def fit(rows,*a,**kw):
        assert len(rows)==48 and all(r['stage']=='train' for r in rows)
        if failed:raise ValueError('training rejected')
        return dict(sha256='frozen')
    monkeypatch.setattr(qp.clock,'fit',fit);monkeypatch.setattr(qp.clock,'predict_frozen',lambda *a:{})
    with pytest.raises(ValueError):fitter.fit(root,root/'fit')
    if failed:
        with pytest.raises(ValueError):fitter.fit(root,out)
    else:assert fitter.fit(root,out)['fit_complete']
    result=json.loads((out/'fit.json').read_text());assert bool(result['failures'])==failed and not result['delayed_probe_labels_used']


def test_host_own_mode_and_failure_archive(monkeypatch,tmp_path):
    import subprocess
    from scripts import prefill_host as host
    from test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    def run(args,**kw):
        calls.append(args);p=host.Path(args[args.index('--output')+1]);p.mkdir();save(p/'queue-probe.json',{})
        raise subprocess.CalledProcessError(1,args)
    monkeypatch.setattr(host.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',queue_probe=True)
    assert '--queue-probe-only' in calls[0] and len(list((tmp_path/'pr32-queue-probes').glob('*.tar.gz')))==1
    with pytest.raises(ValueError):host.collect_host(tmp_path/'inputs',queue_probe=True,live_http=True)


def test_receipt_extraction_post_bracket_not_forecast(monkeypatch):
    record,cap=observed();cap['receipt']=record
    monkeypatch.setattr(qp.delayed,'extract',lambda *a:{})
    row=qp.extract(cap,dict(trial_id='t',arm='receipt',receipt_observer=True),{}, {})
    assert row['receipt']['post_context']['commit']['next']['relative_to_post_s']==pytest.approx(.05)
    assert row['receipt']['post_context']['schedule']['previous']['sequence']==2


@pytest.mark.parametrize('damage',[None,'report','rows','source','cpu','summary','mode'])
def test_independent_audit_rebuild_and_corruption(monkeypatch,tmp_path,damage):
    from scripts import check_prefill_queue_probe as checker,run_prefill_service as runner,run_linux_cpu_tests as cpu
    from prefill_lab import suite,causal
    from test_prefill_async_predictor import runtime_metadata
    p=qp.protocol();data=rows(p);meta=runtime_metadata()
    pins=json.loads((qp.ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    source=dict(version='0.30.0',files=pins);meta['source']=source;meta['workers'][0]['source']=source
    sources={'exact':'source'};acceptance=dict(valid=True,tracked_clean=True,source_sha256=sources,tests=10)
    env={k:'0' for k in ('PREFILL_LAB_STAGES','PREFILL_LAB_EFFECTIVE_CALIBRATION','PREFILL_LAB_DELAYED_ARRIVALS','PREFILL_LAB_EVENT_CLOCK','PREFILL_LAB_LIVE_STATE')};env['PREFILL_LAB_QUEUE_PROBE']='1'
    report=qp.analyze(data,p)
    summary=dict(diagnostic_complete=True,failures=[],completed=[t['trial_id'] for t in p['order']],endpoint_observer_controls_passed=True,same_state_queue_controls_passed=True,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    files={'queue-probe-plan.json':p,'queue-probe.json':summary,'runtime.json':meta,'runtime-final.json':meta,
        'token-manifest.json':{},'collection-sources.json':sources,
        'training-events.json':[dict(trial_id=t['trial_id']) for t in p['order'] if t['stage']=='train' and t['observer']=='light'],
        'launch.json':dict(source_sha256=sources,cpu_acceptance=acceptance,environment=env,scope='queue_receipt_calibration_probe'),
        'native-cpu-acceptance/acceptance.json':acceptance,'queue-probe-rows.json':data,'queue-probe-report.json':report}
    for name,value in files.items():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);save(path,value)
    for r in data:save(tmp_path/(r['trial_id']+'.json'),r)
    # Hardware/transport boundaries mocked; saved/raw comparisons and report
    # reconstruction run through the actual offline audit implementation.
    monkeypatch.setattr(runner,'code_identity',lambda:sources);monkeypatch.setattr(cpu,'validate_report',lambda *a:dict(test_cases=10))
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None);monkeypatch.setattr(qp,'validate_manifest',lambda *a:None)
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True))
    monkeypatch.setattr(qp,'extract',lambda cap,*a:cap);monkeypatch.setattr(qp.clock,'event_labels',lambda cap,t,c:dict(trial_id=t['trial_id']))
    if damage:
        name={'report':'queue-probe-report.json','rows':'queue-probe-rows.json','source':'collection-sources.json','cpu':'native-cpu-acceptance/acceptance.json','summary':'queue-probe.json','mode':'launch.json'}[damage]
        d=json.loads((tmp_path/name).read_text())
        if damage=='report':d['candidate_passed']=True
        if damage=='rows':d[0]['endpoints']['sentinel']['ttft_s']=9.
        if damage=='source':d['exact']='changed'
        if damage=='cpu':d['tests']=9
        if damage=='summary':d['model_accepted']=True
        if damage=='mode':d['environment']['PREFILL_LAB_EVENT_CLOCK']='1'
        save(tmp_path/name,d)
        with pytest.raises(ValueError):checker.audit(tmp_path)
    else:assert checker.audit(tmp_path)==report


def test_new_policy_bound_into_cpu_identity():
    from scripts.run_prefill_service import code_identity
    assert 'experiments/prefill-queue-probe-v1.json' in code_identity()


def test_one_unassessable_cell_blocks_whole_state_control():
    p=qp.protocol();data=rows(p)
    case=next(r['case'] for r in data if r['stage']=='calibration_probe')
    for r in data:
        if r['stage']=='calibration_probe' and r['case']==case and r['arm']=='light':
            r['receipt']['state']['committed']=9;r['receipt']['state_sha256']=digest(r['receipt']['state'])
    report=qp.analyze(data,p)
    assert report['same_state_pairs']==42 and report['different_state_pairs']==6
    assert report['unassessable_state_control_cases']==[case]
    assert not report['same_state_queue_controls_passed']
    assert next(r for r in report['state_control_coverage'] if r['case']==case)['comparable_pairs']==0
