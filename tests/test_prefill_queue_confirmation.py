"""Independent confirmation denominator, sealing, controls and failure paths."""
import asyncio
import copy
import itertools
import json
import time
from types import SimpleNamespace as NS
import pytest
from prefill_lab import queue_confirmation as qc,queue_probe
from prefill_lab.contracts import digest,save
from scripts.check_prefill_queue_confirmation import verify_seal


def fixture():
    p=qc.protocol();rows=[];pred={};cells={};cases={c['id']:c for c in p['cases']}
    for c in p['cases']:
        end={f'foreground-{i}':dict(ttft_s=.2,completion_s=1.) for i in range(len(c['foreground_tokens']))}
        if c['sentinel']:end['sentinel']=dict(ttft_s=.2,completion_s=1.)
        pred[c['id']]=dict(endpoints=end,work_signature='same',sentinel_queue_residence_s=.05,background_outlives_foreground=True)
        if c['sentinel']:cells[c['id']]=dict(status='point',branches=[dict(queue_s=.05)])
    for t in p['order']:
        state=dict(scheduled=2,committed=1,pending_count=1,state={})
        rows.append(dict(**{k:t[k] for k in ('trial_id','case','stage','block','observer','arm')},
            endpoints=copy.deepcopy(pred[t['case']]['endpoints']),work_signature='same' if t['observer']=='light' else None,
            receipt=dict(state=state,state_sha256=digest(state)) if t['receipt_observer'] else None,
            clocks={'sentinel':dict(queue_residence_s=.05,posting_lag_s=.001)} if cases[t['case']]['sentinel'] else {}))
    return p,rows,dict(sha256='freeze',plan_sha256=digest(p),readiness_passed=True,predictions=pred,f0_predictions=copy.deepcopy(pred),cells=cells)


def test_frozen_240_order_balances_all_arms_and_separates_namespaces():
    p=qc.protocol();assert len(p['order'])==240 and len({t['trial_id'] for t in p['order']})==240
    stages=[t['stage'] for t in p['order']]
    assert stages[:40].count('clock_warmup')==16 and stages[:40].count('warmup')==24
    assert stages[40:88]==['train']*48
    assert stages[88:92]==['held_anchor_before']*4 and stages[92:236]==['held_queue']*144
    assert stages[236:]==['held_anchor_after']*4
    for c in p['cases']:
        if c['sentinel']:
            assert {tuple(t['arm'] for t in p['order'] if t['case']==c['id'] and t['stage']=='held_queue' and t['block']==b) for b in range(1,7)}==set(itertools.permutations(('baseline','receipt','light')))
    assert not {t['paired_content_key'] for t in p['order']} & {t['paired_content_key'] for t in queue_probe.protocol()['order']}


def test_full_confirmation_has_no_cap_or_production_promotion():
    p,rows,f=fixture();r=qc.evaluate(rows,p,f)
    assert r['candidate_passed'] and r['status']=='passed'
    assert len(r['sample_checks'])==144 and len(r['work_checks'])==52 and len(r['branch_coverage'])==8
    assert all(sum(v.values())==6 for arms in r['branch_counts'].values() for v in arms.values())
    assert all(r[k] is False for k in qc.ACCEPTANCE) and not r['baseline_native_phase_observed']


@pytest.mark.parametrize('fault',['missing','duplicate','stage','arm','freeze','cell','nan'])
def test_invalid_confirmation_never_loses_a_denominator(fault):
    p,rows,f=fixture();row=next(r for r in rows if r['stage']=='held_queue')
    if fault=='missing':rows.remove(row)
    if fault=='duplicate':rows[-1]=copy.deepcopy(row)
    if fault=='stage':row['stage']='train'
    if fault=='arm':row['arm']='receipt'
    if fault=='freeze':f['plan_sha256']='wrong'
    if fault=='cell':f['cells'].pop(row['case'])
    if fault=='nan':row['clocks']['sentinel']['queue_residence_s']=float('nan')
    with pytest.raises(ValueError):qc.evaluate(rows,p,f)


@pytest.mark.parametrize('fault',['queue','endpoint','work','lifetime','observer','anchor_before','anchor_after','tie','uncovered','different_states'])
def test_separate_gates_cannot_hide_failures(fault):
    p,rows,f=fixture();row=next(r for r in rows if r['stage']=='held_queue' and r['arm']=='baseline')
    if fault=='queue':row['clocks']['sentinel']['queue_residence_s']=.2
    if fault=='endpoint':
        for r in rows:
            if r['stage']=='held_queue' and r['case']==row['case']:r['endpoints']['sentinel']['ttft_s']=.5
    if fault=='work':next(r for r in rows if r['stage']=='held_queue' and r['arm']=='light')['work_signature']='wrong'
    if fault=='lifetime':f['predictions'][row['case']]['background_outlives_foreground']=False
    if fault=='observer':next(r for r in rows if r['stage']=='held_queue' and r['arm']=='receipt')['endpoints']['sentinel']['ttft_s']=.3
    if fault.startswith('anchor_'):
        side='held_'+fault
        for r in rows:
            if r['stage']==side:
                for e in r['endpoints'].values():e['completion_s']=1.3
    if fault in ('tie','uncovered'):
        f['cells'][row['case']]['branches'].append(dict(queue_s=.15))
        if fault=='tie':row['clocks']['sentinel']['queue_residence_s']=.1
    if fault=='different_states':
        for r in rows:
            if r['stage']=='held_queue' and r['arm']=='light':
                r['receipt']['state']['committed']=2;r['receipt']['state_sha256']=digest(r['receipt']['state'])
    r=qc.evaluate(rows,p,f)
    assert len(r['sample_checks'])==144 and not r['candidate_passed']
    if fault in ('uncovered','different_states'):assert r['status']=='inconclusive' and r['queue_budget_passed'] and not r['branch_controls_passed']
    else:assert r['status']=='failed'


def test_manifest_tokens_and_endpoint_share_confirmation_protocol(monkeypatch,tmp_path):
    from prefill_lab.tokens import materialize
    from prefill_lab.endpoint import Endpoint
    p=qc.protocol();manifest=materialize(p['cases'],p['order'],NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[]),
        model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)
    qc.validate_manifest(manifest);save(tmp_path/'tokens.json',manifest)
    for flag in ('QUEUE_PROBE','EVENT_CLOCK','DELAYED_ARRIVALS','EFFECTIVE_CALIBRATION','LIVE_STATE'):monkeypatch.setenv('PREFILL_LAB_'+flag,'0')
    monkeypatch.setenv('PREFILL_LAB_QUEUE_CONFIRMATION','1');monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(tmp_path/'tokens.json'))
    e=Endpoint();asyncio.run(e.init_state(NS(engine_core=NS(call_utility_async=lambda:None)),None,None))
    assert len(e.trials)==240 and not e.external_ids
    monkeypatch.setenv('PREFILL_LAB_QUEUE_PROBE','1')
    with pytest.raises(ValueError):asyncio.run(e.init_state(NS(engine_core=NS(call_utility_async=lambda:None)),None,None))
    row=manifest['trials']['qc-093'][0];row['token_key']='old_namespace'
    with pytest.raises((KeyError,ValueError)):qc.validate_manifest(manifest)


@pytest.mark.parametrize('fault',['early','late','source','membership','manifest','model','freeze'])
def test_audit_rejects_seal_with_wrong_boundary_or_identity(fault):
    p=qc.protocol();caps=[dict(start=i*2.,end=i*2.+1.) for i in range(240)]
    model={'sha256':'base'};frozen={'sha256':'entry'};manifest={'a':1};source={'b':'hash'}
    seal=dict(completed=[t['trial_id'] for t in p['order'][:88]],next_trial='qc-089',source_sha256=source,
        manifest_sha256=digest(manifest),model_sha256='base',freeze_sha256='entry',at_monotonic=175.5)
    verify_seal(seal,p,caps,source,manifest,model,frozen)
    if fault=='early':seal['at_monotonic']=174.
    if fault=='late':seal['at_monotonic']=177.
    if fault=='source':seal['source_sha256']={}
    if fault=='membership':seal['completed'].pop()
    if fault=='manifest':seal['manifest_sha256']='wrong'
    if fault=='model':seal['model_sha256']='wrong'
    if fault=='freeze':seal['freeze_sha256']='wrong'
    with pytest.raises(ValueError):verify_seal(seal,p,caps,source,manifest,model,frozen)


def test_estimator_rejects_held_rows_before_accessing_any_outcome():
    with pytest.raises(ValueError,match='Held outcomes'):qc.seal_training([dict(stage='held_queue')],[],{},None)


@pytest.mark.parametrize('parent_mode',[False,True])
@pytest.mark.parametrize('fault',[None,'source','raw','report','freeze','model','seal','cpu','mode','runtime','summary'])
def test_independent_raw_audit_rebuilds_and_rejects_tampering(monkeypatch,tmp_path,fault,parent_mode):
    from scripts import check_prefill_queue_confirmation as checker,run_prefill_service as runner,run_linux_cpu_tests as cpu
    from prefill_lab import suite,causal
    from test_prefill_async_predictor import runtime_metadata
    p,rows,f=fixture();runtime=runtime_metadata();pins=json.loads((qc.ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    runtime['source']=dict(version='0.30.0',files=pins);runtime['workers'][0]['source']=runtime['source']
    sources={'exact':'source'};acceptance=dict(valid=True,tracked_clean=True,source_sha256=sources,tests=10);model={'sha256':'base'}
    for i,row in enumerate(rows):row.update(start=2.*i,end=2.*i+1.)
    report=qc.evaluate(rows,p,f)
    env={'PREFILL_LAB_'+k:'0' for k in ('QUEUE_PROBE','STAGES','EFFECTIVE_CALIBRATION','DELAYED_ARRIVALS','EVENT_CLOCK','LIVE_STATE')}
    env['PREFILL_LAB_QUEUE_CONFIRMATION']='1'
    if parent_mode:
        env['PREFILL_LAB_LIVE_STATE']='1';env['PREFILL_LAB_ENTRY_HTTP']='1'
    summary=dict(collection_complete=True,failures=[],completed=[t['trial_id'] for t in p['order']],candidate_passed=True,status='passed',**{k:False for k in qc.ACCEPTANCE})
    seal=dict(completed=summary['completed'][:88],next_trial='qc-089',at_monotonic=175.5,source_sha256=sources,
        manifest_sha256=digest({}),model_sha256=model['sha256'],freeze_sha256=f['sha256'])
    files={'queue-confirmation-plan.json':p,'confirmation.json':summary,'runtime.json':runtime,'runtime-final.json':runtime,
        'token-manifest.json':{},'collection-sources.json':sources,'base-model.json':model,'entry-freeze.json':f,'seal.json':seal,
        'confirmation-rows.json':rows,'confirmation-report.json':report,'native-cpu-acceptance/acceptance.json':acceptance,
        'launch.json':dict(source_sha256=sources,cpu_acceptance=acceptance,environment=env,scope='entry_http_state_bridge' if parent_mode else 'independent_queue_confirmation')}
    for name,value in files.items():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);save(path,value)
    for row in rows:save(tmp_path/(row['trial_id']+'.json'),row)
    monkeypatch.setattr(runner,'code_identity',lambda:sources);monkeypatch.setattr(cpu,'validate_report',lambda *a:dict(test_cases=10))
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None);monkeypatch.setattr(qc,'validate_manifest',lambda *a:None)
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True));monkeypatch.setattr(qc,'extract',lambda cap,*a:cap)
    def rebuild(data,caps,*a):
        assert len(data)==88 and len(caps)==24 and not any(r['stage'] in qc.HELD for r in data)
        return model,f
    monkeypatch.setattr(qc,'seal_training',rebuild)
    if fault:
        name={'source':'collection-sources.json','raw':'qc-093.json','report':'confirmation-report.json',
            'freeze':'entry-freeze.json','model':'base-model.json','seal':'seal.json','cpu':'native-cpu-acceptance/acceptance.json',
            'mode':'launch.json','runtime':'runtime-final.json','summary':'confirmation.json'}[fault]
        data=json.loads((tmp_path/name).read_text())
        if fault=='source':data['exact']='different'
        elif fault=='raw':data['clocks']['sentinel']['queue_residence_s']=.2
        elif fault=='report':data['candidate_passed']=False
        elif fault in ('freeze','model'):data['sha256']='different'
        elif fault=='seal':data['at_monotonic']=178.
        elif fault=='cpu':data['tests']=11
        elif fault=='mode':data['environment']['PREFILL_LAB_QUEUE_PROBE']='1'
        elif fault=='runtime':data['changed']=True
        elif fault=='summary':data['model_accepted']=True
        save(tmp_path/name,data)
        with pytest.raises(ValueError):checker.audit(tmp_path,entry_http=parent_mode)
    else:
        assert checker.audit(tmp_path,entry_http=parent_mode)==report
        if parent_mode:
            with pytest.raises(ValueError):checker.audit(tmp_path)


def test_host_audits_inconclusive_run_then_archives_without_retry(monkeypatch,tmp_path):
    import subprocess
    from scripts import prefill_host as host
    from test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    def command(args,**kw):
        calls.append(args)
        if '--queue-confirmation-only' in args:
            target=host.Path(args[args.index('--output')+1]);target.mkdir();save(target/'confirmation-report.json',dict(status='inconclusive'))
            return subprocess.CompletedProcess(args,2)
        assert args[1].endswith('check_prefill_queue_confirmation.py')
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(host.subprocess,'run',command)
    with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',queue_confirmation=True)
    assert len(calls)==2 and '--queue-confirmation-only' in calls[0]
    assert len(list((tmp_path/'pr32-queue-confirmations').glob('*.tar.gz')))==1


@pytest.mark.parametrize('fault',[None,'freeze','source','cancel','readiness'])
def test_collector_seals_once_before_all_held_trials_and_retains_failure(monkeypatch,tmp_path,fault):
    from prefill_lab import suite,causal
    from scripts import predict_prefill_async,run_prefill_service
    p,rows,frozen=fixture();lookup={r['trial_id']:r for r in rows};observed=[];fitted=[];identity={'a':'source'}
    monkeypatch.setattr(qc,'validate_manifest',lambda m:None)
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None);monkeypatch.setattr(suite,'host_identity',lambda:{})
    monkeypatch.setattr(predict_prefill_async,'runtime_inputs',lambda m:None)
    monkeypatch.setattr(run_prefill_service,'code_identity',lambda:copy.deepcopy(identity))
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True))
    def seal(data,caps,plan,runtime,deadline):
        assert len(data)==88 and len(caps)==24 and len(observed)==88
        assert not any(r['stage'] in qc.HELD for r in data)
        if fault=='readiness':
            rejected=dict(copy.deepcopy(frozen),readiness_passed=False,errors=[],first_commit_readiness=dict(checks=[dict(passed=False,trial_id='qc-070')]))
            raise qc.TrainingReadinessError({'sha256':'rejected-base'},rejected)
        fitted.append(1);return {'sha256':'base'},copy.deepcopy(frozen)
    monkeypatch.setattr(qc,'seal_training',seal);monkeypatch.setattr(qc,'extract',lambda cap,t,c,p:copy.deepcopy(lookup[t['trial_id']]))
    async def trial(client,t,deadline,path):
        if t['stage'] in qc.HELD:assert (tmp_path/'seal.json').exists() and len(fitted)==1
        observed.append(t['trial_id'])
        if len(observed)==89:
            if fault=='freeze':save(tmp_path/'entry-freeze.json',{})
            if fault=='source':identity['a']='changed'
            if fault=='cancel':raise asyncio.CancelledError()
        return dict(trial_id=t['trial_id'],start=time.monotonic(),end=time.monotonic())
    monkeypatch.setattr(suite,'controlled_trial',trial)
    class Client:
        async def command(self,operation):return dict(workers=[dict(dispatch_table={})])
    if fault:
        with pytest.raises(asyncio.CancelledError if fault=='cancel' else ValueError):asyncio.run(qc.run(Client(),None,None,{},tmp_path))
        summary=json.loads((tmp_path/'confirmation.json').read_text());assert not summary['collection_complete'] and summary['failures']
        assert len(observed)==(88 if fault=='readiness' else 89)
        if fault=='readiness':
            rejected=json.loads((tmp_path/'training-readiness-failure.json').read_text())
            assert not rejected['sealed_for_held'] and rejected['completed']==observed
            assert rejected['first_commit_failures']==[dict(passed=False,trial_id='qc-070')]
            assert json.loads((tmp_path/'rejected-base-model.json').read_text())==dict(sha256='rejected-base')
            assert not json.loads((tmp_path/'rejected-entry-candidate.json').read_text())['readiness_passed']
            assert all(not (tmp_path/n).exists() for n in ('base-model.json','entry-freeze.json','seal.json','confirmation-report.json'))
            assert all(rejected[k] is False for k in qc.ACCEPTANCE)
    else:
        summary=asyncio.run(qc.run(Client(),None,None,{},tmp_path));assert summary['collection_complete'] and summary['candidate_passed']
        assert len(observed)==240 and len(fitted)==1
    assert all(summary[k] is False for k in qc.ACCEPTANCE)
