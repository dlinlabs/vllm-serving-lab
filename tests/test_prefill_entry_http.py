"""CPU contracts for entry-return continuation and controlled-to-HTTP handoff."""
import asyncio
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from prefill_lab import entry_http as eh,entry_http_model as handoff,live_predictor as live,submit_clock,entry_clock,live_http
from prefill_lab.async_predictor import Request,Scope,simulate,CostLaw
from prefill_lab.contracts import digest,save
from test_prefill_live import empty,D,busy,synthetic_http_rows
from test_prefill_async_predictor import runtime_metadata


@pytest.mark.parametrize('tokens',[1,1024,4096,7168])
@pytest.mark.parametrize('prefill,decode',[(0.,0.),(.012,.0005),(.15,.012)])
def test_entry_continuation_matches_independent_from_empty_graph(tokens,prefill,decode):
    model=submit_clock.costs([.01,.1,.002,.003,.001,.004],eh.qc.protocol(),prefill,decode)
    request=Request('new',tokens,4);before=empty();result=live.forecast(before,request,model,D,until_first=False)
    reference=simulate([request],model,D)
    assert result['ttft_from_cut_s']==pytest.approx(reference['requests']['new']['ttft_s'],abs=1e-10)
    assert result['completion_from_cut_s']==pytest.approx(reference['requests']['new']['finished_s'],abs=1e-10)
    assert result['cost_sha256']==digest(model.to_dict()) and before==empty()


def test_pending_return_is_not_charged_again_but_future_cpu_overlap_is_charged():
    p=eh.qc.protocol();theta=[.01,.1,.002,.003,.001,.004]
    a=submit_clock.costs(theta,p,.01,.0005);b=submit_clock.costs(theta,p,.15,.012)
    state=busy(True);slow=live.forecast(state,Request('new',1024,2),b,D)
    fast=live.forecast(state,Request('new',1024,2),a,D)
    assert slow['initial_pending_residual_s']==fast['initial_pending_residual_s']==0
    assert slow['ttft_from_cut_s']>fast['ttft_from_cut_s']
    assert state==busy(True)


@pytest.mark.parametrize('prefill,decode',[(.012,.0005),(.15,.012)])
def test_busy_entry_continuation_matches_independent_graph_at_observed_cut(prefill,decode):
    model=submit_clock.costs([.01,.1,.002,.003,.001,.004],eh.qc.protocol(),prefill,decode)
    reference=simulate([Request('a',1024,8),Request('n',512,2,.005)],model,D)
    cut=reference['requests']['n']['enqueue_s'];at=next(s for s in reference['steps'] if s['scheduled_s']==cut)
    requests={k:{f:v[f] for f in ('prompt_tokens','max_tokens','num_tokens','computed','placeholders')} for k,v in at['state_before'].items() if k!='n'}
    pending=[s for s in reference['steps'] if s['scheduled_s']<cut and s['committed_s']>cut]
    state=dict(schema='live-engine-state-v1',clock_domain='test',observed_at=cut,
        last_commit_at=max((s['committed_s'] for s in reference['steps'] if s['committed_s']<=cut),default=None),running=list(requests),waiting=[],requests=requests,
        pending=[dict(work=s['work'],scheduled_at=s['scheduled_s'],ready=reference['nodes'][s['copy_node']]['end_s']<=cut) for s in pending])
    result=live.forecast(state,Request('n',512,2),model,D,until_first=False)
    assert result['ttft_from_cut_s']==pytest.approx(reference['requests']['n']['first_s']-cut,abs=1e-10)
    assert result['completion_from_cut_s']==pytest.approx(reference['requests']['n']['finished_s']-cut,abs=1e-10)


def test_extra_physical_stage_cost_is_not_silently_ignored():
    model=submit_clock.costs([.01,.1,.002,.003,.001,.004],eh.qc.protocol(),.01,.001)
    parts={k:dict(v) for k,v in model.launches.items()};parts['logits_sample']['cpu_after']=CostLaw({'bias':.01})
    bad=submit_clock.SubmitCosts(parts,dict(model.scalar),'unsupported extra stage',model.base_cost_document)
    with pytest.raises(ValueError,match='physical costs'):live.forecast(empty(),Request('n',1024,1),bad,D)


def test_native_tracker_records_entry_cost_and_configuration_before_add():
    from prefill_lab.live_runtime import LiveTracker
    model=submit_clock.costs([.01,.1,.002,.003,.001,.004],eh.qc.protocol(),.035,.002)
    core=NS(batch_queue=[],scheduler=NS(running=[],waiting=[],requests={}),_prefill_trial=NS(requests={'probe':{}}))
    tracker=LiveTracker(core,model,Scope(),D,clock=lambda:10.,domain='test',configuration_sha256='sealed')
    request=NS(request_id='probe',num_prompt_tokens=1024,max_tokens=128,sampling_params=NS(ignore_eos=True,
        extra_args=dict(prefill_lab_trace_id='a'*32,prefill_lab_clock_domain='test',prefill_lab_client_sent=9.99,prefill_lab_probe=True)))
    tracker.before_add(request);row=tracker.records.rows[0]
    assert row['status']=='predicted' and 'probe' not in row['snapshot']['requests']
    tracker.after_add(request);captured=tracker.capture()
    assert row['cut']<=row['prediction_finished']<=row['engine_enqueue']
    assert captured['configuration_sha256']=='sealed' and captured['cost_sha256']==digest(model.to_dict())


@pytest.mark.parametrize('fault',[None,'busy','runtime','old_mode'])
def test_plugin_configures_new_family_only_in_explicit_drained_entry_mode(monkeypatch,fault):
    from prefill_lab import plugin
    runtime=runtime_metadata();model=submit_clock.costs([.01,.1,.002,.003,.001,.004],eh.qc.protocol(),.035,.002)
    core=NS(_prefill_trial=None,batch_queue=[],scheduler=NS(has_requests=lambda:fault=='busy'),
        model_executor=NS(collective_rpc=lambda *a:runtime['workers']))
    monkeypatch.setattr(plugin,'core_runtime',lambda *a:None)
    monkeypatch.setattr(plugin,'resolved_config',lambda *a:{k:v for k,v in runtime.items() if k!='workers'})
    monkeypatch.setattr(handoff,'costs',lambda config,actual:model)
    monkeypatch.setenv('PREFILL_LAB_LIVE_STATE','1');monkeypatch.setenv('PREFILL_LAB_ENTRY_HTTP','0' if fault=='old_mode' else '1')
    supplied=copy.deepcopy(runtime)
    if fault=='runtime':supplied['process_id']=8
    payload=dict(operation='live_configure',frozen=dict(sha256='seal'),runtime=supplied)
    if fault:
        with pytest.raises((ValueError,RuntimeError)):plugin.command(core,payload)
    else:
        assert plugin.command(core,payload)==dict(configured=True,frozen_sha256='seal')
        assert core._prefill_live_config[0] is model and core._prefill_live_config_sha256=='seal'


def config_fixture(monkeypatch):
    p=eh.qc.protocol();runtime=runtime_metadata();pred={'synthetic':'prediction'};f0={'synthetic':'baseline'};source={'code':'hash'}
    base=dict(sha256='base',parameters_s=[.01,.1,.002,.003,.001,.004],measurement_origin='fresh_native_training_events')
    labels=dict(constants_s=dict(prefill=.035,decode=.002),training_trial_ids=sorted(t['trial_id'] for t in p['order'] if t['stage']=='train' and t['observer']=='light'))
    entry=dict(schema='entry-submit-clock-freeze-v1',readiness_passed=True,base_model_sha256='base',plan_sha256=digest(p),runtime_sha256=digest(runtime),
        implementation_sha256=hashlib.sha256(Path(entry_clock.__file__).read_bytes()).hexdigest(),labels=labels,predictions=pred,f0_predictions=f0)
    entry['sha256']=digest(entry)
    monkeypatch.setattr(handoff.clock,'predict_frozen',lambda *a:f0)
    monkeypatch.setattr(handoff.submit_clock,'own_rollouts',lambda *a:pred)
    monkeypatch.setattr(handoff.clock,'forecasts',lambda x:x)
    from scripts import run_prefill_service
    monkeypatch.setattr(run_prefill_service,'code_identity',lambda:source)
    monkeypatch.setattr(handoff,'clock_domain',lambda:'same-boot');monkeypatch.setattr(handoff.time,'monotonic',lambda:10.)
    config=handoff.configure(base,entry,dict(candidate_passed=True,frozen_sha256=entry['sha256']),runtime,finished_at=9.,http_manifest_sha256='manifest')
    return config,runtime


def test_handoff_preserves_fitted_cpu_and_mixed_gpu_costs(monkeypatch):
    config,runtime=config_fixture(monkeypatch);model=handoff.costs(config,runtime)
    assert model.launches['forward']['cpu_after'](dict(prefill=1))==.035
    assert model.launches['forward']['cpu_after'](dict(prefill=0))==.002
    assert model.launches['forward']['gpu_tail'].mixed>0


@pytest.mark.parametrize('fault',['boot','source','chronology','future','prerequisite','entry','base','runtime','implementation','train','prediction','f0'])
def test_resealed_handoff_cannot_override_its_boundaries(monkeypatch,fault):
    config,runtime=config_fixture(monkeypatch)
    if fault=='boot':config['clock_domain']='different-boot'
    if fault=='source':config['source_sha256']={}
    if fault=='chronology':config['controlled_finished_at']=11.
    if fault=='future':config['configured_at']=11.
    if fault=='prerequisite':config['controlled_candidate_passed']=False
    if fault=='entry':config['entry_freeze']['readiness_passed']=False
    if fault=='base':config['base_model']['sha256']='different'
    if fault=='runtime':runtime=copy.deepcopy(runtime);runtime['changed']=True
    if fault=='implementation':config['entry_freeze']['implementation_sha256']='different'
    if fault=='train':config['entry_freeze']['labels']['training_trial_ids'].pop()
    if fault=='prediction':config['entry_freeze']['predictions']={}
    if fault=='f0':config['entry_freeze']['f0_predictions']={}
    entry=config['entry_freeze'];entry['sha256']=digest({k:v for k,v in entry.items() if k!='sha256'})
    config['sha256']=digest({k:v for k,v in config.items() if k!='sha256'})
    with pytest.raises(ValueError):handoff.costs(config,runtime)


def test_numeric_protocol_retains_all_cells_and_new_content_namespace():
    p=eh.protocol();old=live_http.protocol();assert len(p['order'])==98
    assert [t['case'] for t in p['order']]==[t['case'] for t in old['order']]
    assert len({t['trial_id'] for t in p['order']})==98
    assert not {t['paired_content_key'] for t in p['order']} & {t['paired_content_key'] for t in old['order']}
    assert {c['sentinel']['arrival_ms'] for c in p['cases']}=={50,200,400,800}


def test_endpoint_modes_and_manifest_are_exactly_paired(monkeypatch,tmp_path):
    from prefill_lab.tokens import materialize
    from prefill_lab.endpoint import Endpoint
    p=eh.qc.protocol();manifest=materialize(p['cases'],p['order'],NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[]),model=p['model']['name'],revision=p['model']['revision'],tokenizer_sha256='a'*64)
    save(tmp_path/'tokens.json',manifest);monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(tmp_path/'tokens.json'))
    for flag in ('QUEUE_PROBE','EVENT_CLOCK','DELAYED_ARRIVALS','EFFECTIVE_CALIBRATION'):monkeypatch.setenv('PREFILL_LAB_'+flag,'0')
    for flag in ('QUEUE_CONFIRMATION','LIVE_STATE','ENTRY_HTTP'):monkeypatch.setenv('PREFILL_LAB_'+flag,'1')
    e=Endpoint();asyncio.run(e.init_state(NS(engine_core=NS(call_utility_async=lambda:None)),None,None))
    assert len(e.trials)==240 and len(e.live_trials)==98 and e.external_ids==set(e.live_trials)
    monkeypatch.setenv('PREFILL_LAB_ENTRY_HTTP','0')
    with pytest.raises(ValueError):asyncio.run(e.init_state(NS(engine_core=NS(call_utility_async=lambda:None)),None,None))


def test_host_binds_both_manifests_audits_failed_report_and_archives(monkeypatch,tmp_path):
    import subprocess
    from scripts import prefill_host as host
    from test_prefill_host import mock_host
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[];prepared=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path));monkeypatch.setattr(host,'prepare',lambda a:prepared.append(a))
    def command(args,**kwargs):
        calls.append(args)
        if len(calls)==1:
            assert '--entry-http-only' in args and '--http-manifest' in args
            root=Path(args[args.index('--output')+1]);root.mkdir();save(root/'entry-http-report.json',dict(candidate_passed=False))
            return subprocess.CompletedProcess(args,2)
        assert args[1].endswith('check_prefill_entry_http.py') and kwargs['timeout']==600
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(host.subprocess,'run',command)
    with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',entry_http=True)
    assert len(calls)==2 and prepared[0].queue_confirmation_only and prepared[1].entry_http_tokens
    assert len(list((tmp_path/'pr32-entry-http').glob('*.tar.gz')))==1


@pytest.mark.parametrize('fault',[None,'handoff','domain','manifest','rows','client','journal','transport','report','runtime','cpu','traffic'])
def test_independent_entry_http_audit_rebuilds_raw_boundaries(monkeypatch,tmp_path,fault):
    from scripts import check_prefill_entry_http as checker
    p=eh.protocol();oldp,data=synthetic_http_rows();rows=[dict(r,trial_id=t['trial_id'],clock_domain='recorded-domain') for r,t in zip(data,p['order'])]
    transport=live_http.fit_transport(rows,p);report=eh.score(rows,p,transport);runtime={'identity':'runtime'};base={};entry={};prerequisite=dict(candidate_passed=True)
    config=dict(sha256='config',controlled_finished_at=1.,configured_at=2.,http_manifest_sha256=digest({}),controlled_report_sha256=digest(prerequisite),
        base_model=base,entry_freeze=entry,clock_domain='recorded-domain',source_sha256={'source':'hash'})
    seal=dict(at_monotonic=61.,next_trial=p['order'][26]['trial_id'],completed=[t['trial_id'] for t in p['order'][:26]],transport_sha256=transport['sha256'],config_sha256='config',source_sha256=config['source_sha256'])
    controlled=tmp_path/'controlled';controlled.mkdir();journal=[];directory=tmp_path/'bridge';directory.mkdir()
    for n in ('launch.json','collection-sources.json','token-manifest.json','native-cpu-acceptance/acceptance.json','native-cpu-acceptance/junit.xml','native-cpu-acceptance/pytest.log'):
        for root in (tmp_path,controlled):
            target=root/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{}')
    for i,t in enumerate(p['order']):
        client=dict(rows=[dict(trace_id=f'{i:032x}')]);events=[dict(event='received',request_id=f'{i:032x}',run_id='test')];journal+=events
        cap=dict(start=10.+2*i,end=11.+2*i,trial_id=t['trial_id'])
        save(directory/(t['trial_id']+'.json'),dict(capture=cap,client=client,gateway=events));save(directory/(t['trial_id']+'-client.json'),client)
    gw=dict(admission_mode='protected',max_in_flight=32,max_waiting_capacity=0,shared_upstream_client=True,prefill_raw_completions=True,gateway_diagnostics=True,
        telemetry_dropped_records=0,telemetry_write_errors=0,telemetry_run_id='test',accepted_requests=live_http.expected_requests(p),rejected_requests=0,failed_requests=0,current_in_flight=0,current_waiting=0)
    files={'entry-http-plan.json':p,'entry-http-summary.json':dict(collection_complete=True,candidate_passed=True,failures=[],completed=[t['trial_id'] for t in p['order']],**{k:False for k in eh.qc.ACCEPTANCE}),
        'http-token-manifest.json':{},'entry-http-config.json':config,'http-runtime-final.json':runtime,'http-rows.json':rows,'transport-freeze.json':transport,'transport-seal.json':seal,
        'entry-http-report.json':report,'entry-http-configured.json':dict(configured=True,frozen_sha256='config'),'gateway-start.json':gw,'gateway-final.json':gw,
        'telemetry-status.json':dict(run_id='test',status='closed',dropped=0,write_errors=0,diagnostics=dict(complete=True))}
    for n,value in files.items():save(tmp_path/n,value)
    for n,value in [('runtime.json',runtime),('qc-240.json',dict(end=1.)),('base-model.json',base),('entry-freeze.json',entry)]:save(controlled/n,value)
    (tmp_path/'gateway.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in journal))
    def controlled_audit(root,**kwargs):
        assert kwargs==dict(entry_http=True);return prerequisite
    monkeypatch.setattr(checker,'audit_controlled',controlled_audit);monkeypatch.setattr(eh,'validate_manifest',lambda *a:None);monkeypatch.setattr(eh.old,'validate_pair',lambda *a:None)
    def verified_costs(c,runtime,**kwargs):
        assert kwargs==dict(now=10.,domain='recorded-domain');return None
    monkeypatch.setattr(handoff,'costs',verified_costs);monkeypatch.setattr(eh,'extract',lambda bundle,t,*a,**k:next(r for r in rows if r['trial_id']==t['trial_id']))
    from prefill_lab import causal
    # Native source/dispatch/hardware are boundary mocks; all raw sidecars,
    # journals, seal chronology, rows, transport and report comparisons are real.
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True));runtime['workers']=[dict(dispatch_table=D)]
    save(controlled/'runtime.json',runtime);save(tmp_path/'http-runtime-final.json',runtime)
    if fault:
        path={'handoff':'entry-http-config.json','domain':'http-rows.json','manifest':'http-token-manifest.json','rows':'http-rows.json','client':'bridge/eh-001-client.json',
            'journal':'bridge/eh-001.json','transport':'transport-seal.json','report':'entry-http-report.json','runtime':'http-runtime-final.json',
            'cpu':'native-cpu-acceptance/junit.xml','traffic':'gateway.jsonl'}[fault]
        target=tmp_path/path
        if fault=='cpu':target.write_text('different')
        elif fault=='traffic':target.write_text(target.read_text()+json.dumps(dict(event='received',request_id='extra',run_id='test'))+'\n')
        else:
            obj=json.loads(target.read_text())
            if fault=='handoff':obj['configured_at']=20.
            elif fault=='domain':obj[0]['clock_domain']='another-boot';rows[0]['clock_domain']='another-boot'
            elif fault=='manifest':obj['different']=True
            elif fault=='rows':obj.pop()
            elif fault=='client':obj['different']=True
            elif fault=='journal':obj['gateway']=[]
            elif fault=='transport':obj['at_monotonic']=63.
            elif fault=='report':obj['candidate_passed']=False
            elif fault=='runtime':obj['different']=True
            save(target,obj)
        with pytest.raises(ValueError):checker.audit(tmp_path)
    else:assert checker.audit(tmp_path)==report


@pytest.mark.parametrize('fault',[None,'controlled','generator','cancel','warmup','config'])
def test_combined_collection_seals_before_held_and_stops_on_failure(monkeypatch,tmp_path,fault):
    from prefill_lab import suite,causal
    from scripts import run_prefill_service
    p=eh.protocol();oldp,data=synthetic_http_rows();mapped=[]
    for t,row in zip(p['order'],data):mapped.append(dict(row,trial_id=t['trial_id']))
    runtime=dict(workers=[dict(dispatch_table=D)]);seen=[]
    for n,d in [('launch.json',{}),('collection-sources.json',{}),('token-manifest.json',{})]:save(tmp_path/n,d)
    (tmp_path/'native-cpu-acceptance').mkdir();save(tmp_path/'native-cpu-acceptance/acceptance.json',{})
    monkeypatch.setattr(eh,'validate_manifest',lambda x:None);monkeypatch.setattr(eh.qc,'validate_manifest',lambda x:None);monkeypatch.setattr(eh.old,'validate_pair',lambda *a:None)
    monkeypatch.setattr(run_prefill_service,'code_identity',lambda:{'code':'hash'})
    async def controlled(*args):
        directory=args[-1];seen.append('controlled')
        for name,value in [('runtime.json',runtime),('base-model.json',{}),('entry-freeze.json',{}),('confirmation-report.json',{}),('qc-240.json',dict(end=1.))]:save(directory/name,value)
        return dict(candidate_passed=fault!='controlled')
    monkeypatch.setattr(eh.qc,'run',controlled)
    config={'sha256':'configuration'};monkeypatch.setattr(handoff,'configure',lambda *a,**k:config);monkeypatch.setattr(handoff,'costs',lambda *a:None)
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True));monkeypatch.setattr(eh.old,'gateway_records',lambda *a:[])
    monkeypatch.setattr(eh,'extract',lambda b,t,*a,**k:next(dict(r,forecast_available=False) if fault=='warmup' else r for r in mapped if r['trial_id']==t['trial_id']))
    async def generator(path,tid,url,output,deadline):
        seen.append(tid);t=next(t for t in p['order'] if t['trial_id']==tid)
        if t['stage'] not in ('train','warmup'):assert (tmp_path/'transport-seal.json').exists()
        if fault=='generator':raise RuntimeError('generator failed')
        if fault=='cancel':raise asyncio.CancelledError()
        if fault=='config':save(tmp_path/'entry-http-config.json',{})
        return dict(rows=[])
    monkeypatch.setattr(eh.old,'isolated_trial',generator)
    async def finish(*a):return {}
    monkeypatch.setattr(suite,'finish_external',finish)
    gw=dict(admission_mode='protected',max_in_flight=32,max_waiting_capacity=0,shared_upstream_client=True,prefill_raw_completions=True,
        gateway_diagnostics=True,telemetry_dropped_records=0,telemetry_write_errors=0,telemetry_run_id='test',accepted_requests=live_http.expected_requests(p),
        rejected_requests=0,failed_requests=0,current_in_flight=0,current_waiting=0)
    save(tmp_path/'gateway-start.json',gw)
    async def command(op,**kwargs):
        seen.append(op)
        return runtime if op=='meta' else dict(frozen_sha256='configuration') if op=='live_configure' else {}
    async def get(*a):return NS(json=lambda:gw)
    if fault:
        with pytest.raises(asyncio.CancelledError if fault=='cancel' else (ValueError,RuntimeError)):
            asyncio.run(eh.run(NS(command=command),NS(get=get),'',{}, {},tmp_path,'',99999999))
        summary=json.loads((tmp_path/'entry-http-summary.json').read_text());assert not summary['collection_complete'] and summary['failures']
        if fault=='controlled':assert 'live_configure' not in seen and 'external_begin' not in seen
        if fault in ('generator','cancel'):assert 'abort' in seen and (tmp_path/'bridge/eh-001-partial.json').exists()
        if fault=='warmup':assert not (tmp_path/'transport-freeze.json').exists() and len(summary['completed'])==10
    else:
        summary=asyncio.run(eh.run(NS(command=command),NS(get=get),'',{}, {},tmp_path,'',99999999))
        assert summary['collection_complete'] and summary['candidate_passed'] and len(summary['completed'])==98
        assert seen.index('controlled')<seen.index('live_configure')<seen.index('external_begin')
    assert all(summary[k] is False for k in eh.qc.ACCEPTANCE)
