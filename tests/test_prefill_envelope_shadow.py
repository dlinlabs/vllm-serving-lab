"""Shadow measurement and arithmetic, entirely CPU/mock; no native claims."""
import asyncio
import copy
import json
import time
from types import SimpleNamespace as NS

import pytest

from prefill_lab import envelope_shadow as shadow, state_envelope as envelope, live_runtime
from prefill_lab.async_predictor import Request
from prefill_lab.contracts import digest, save
from test_prefill_async_predictor import runtime_metadata
from test_prefill_live import empty, busy


def assumptions():
    return dict(schema='envelope-shadow-assumptions-v1', gpu_lower_s=[.01,.1,.002,.003,.001,.004],
                gpu_upper_s=[.01,.1,.002,.003,.001,.004], prefill_return_s=[.002,.002],
                decode_return_s=[.001,.001], policy=dict(max_age_s=.005,max_width_s=.010))


def manifest():
    from prefill_lab.tokens import materialize
    p=shadow.protocol()
    tokenizer=NS(get_vocab=lambda: {'a': 1, 'b': 2}, all_special_ids=[])
    return materialize(p['cases'],p['order'],tokenizer,model=p['model']['name'],
                       revision=p['model']['revision'],tokenizer_sha256='synthetic')


def config_fixture(*,gpu_upper_factor=1.):
    runtime=runtime_metadata()
    inputs=manifest()
    bounds=assumptions()
    bounds['gpu_upper_s']=[v*gpu_upper_factor for v in bounds['gpu_lower_s']]
    support=shadow.resolve_support(bounds,runtime)
    config=shadow.configure(support,runtime,inputs,now=1.,domain='test',sources={'code':'hash'})
    return runtime,inputs,config


def test_short_plan_balances_controls_and_has_disjoint_content():
    p=shadow.protocol()
    assert len(p['cases'])==4 and len(p['order'])==20
    assert len({t['trial_id'] for t in p['order']})==20
    assert len([t for t in p['order'] if t['stage']=='warmup'])==4
    for case in p['cases']:
        for block,order in [(1,['off','light']),(2,['light','off'])]:
            pair=[t for t in p['order'] if t['case']==case['id'] and t['block']==block]
            assert [t['observer'] for t in pair]==order
            assert len({t['paired_content_key'] for t in pair})==1
    shadow.validate_manifest(manifest())


@pytest.mark.parametrize('fault',['model','revision','forced','namespace','missing'])
def test_manifest_cannot_change_scope_or_request_inventory(fault):
    m=manifest()
    if fault in ('model','revision'): m[fault]='wrong'
    if fault=='forced': m['forced_outputs']=False
    if fault=='namespace': m['trials']['es-001'][0]['token_key']='unrelated'
    if fault=='missing': m['trials'].pop('es-001')
    with pytest.raises((ValueError,KeyError)): shadow.validate_manifest(m)


def test_prepared_context_detaches_inputs_and_does_not_revalidate_per_probe(monkeypatch):
    runtime=runtime_metadata()
    support=shadow.resolve_support(assumptions(),runtime)
    before=copy.deepcopy(support)
    expected=envelope.forecast(empty(),Request('z',512,128),support,runtime,now=10.,domain='test')
    context=envelope.PreparedEnvelope(support,runtime)
    support['gpu_lower_s'][0]=100.
    runtime['workers'].clear()
    exported=context.support
    exported['policy']['max_age_s']=100
    monkeypatch.setattr(envelope,'models',lambda *a:pytest.fail('revalidated support in hot path'))
    monkeypatch.setattr(envelope,'runtime_inputs',lambda *a:pytest.fail('reparsed runtime in hot path'))
    assert context.support==before
    assert context.forecast(empty(),Request('z',512,128),now=10.,domain='test')==expected


@pytest.mark.parametrize('fault',['clock','source','runtime','future','promote','tamper','policy'])
def test_configuration_is_bound_before_shadow_without_point_readiness(fault):
    runtime,m,config=config_fixture()
    now=2.;domain='test';sources={'code':'hash'}
    if fault=='clock': domain='other'
    if fault=='source': sources={'code':'changed'}
    if fault=='runtime': runtime['process_id']+=1
    if fault=='future': now=.5
    if fault=='promote': config['model_accepted']=True
    if fault=='tamper': config['manifest_sha256']='changed'
    if fault=='policy': config['support']['policy']['max_age_s']=-1
    with pytest.raises(ValueError): shadow.validate_config(config,runtime,now=now,domain=domain,sources=sources)


def test_old_empirical_support_cannot_be_rebound_to_another_host():
    runtime=runtime_metadata()
    support=shadow.resolve_support(assumptions(),runtime)
    runtime['process_id']+=1
    runtime['workers'][0]['pid']=runtime['process_id']
    with pytest.raises(ValueError): shadow.resolve_support(support,runtime)
    with pytest.raises(ValueError): shadow.resolve_support(dict(assumptions(),training='old-host'),runtime)


def tracker(clock, *, observed=True, state=None):
    runtime=runtime_metadata()
    context=envelope.PreparedEnvelope(shadow.resolve_support(assumptions(),runtime),runtime)
    core=NS(batch_queue=[],scheduler=NS(running=[],waiting=[],requests={}),
            _prefill_trial=NS(requests={'probe':{}}))
    result=live_runtime.EnvelopeTracker(core,context,clock=clock,domain='test',observed=observed,
                                        configuration_sha256='sealed')
    if state is not None:
        result.snapshot=lambda:copy.deepcopy(state)
    request=NS(request_id='probe',num_prompt_tokens=512,max_tokens=128,
               sampling_params=NS(ignore_eos=True,extra_args=dict(prefill_lab_trace_id='a'*32,
                 prefill_lab_client_sent=9.,prefill_lab_clock_domain='test',prefill_lab_probe=True)))
    return result,request


def test_finished_age_forces_fallback_even_when_supplied_snapshot_age_is_zero():
    ticks=iter([10.,10.,10.,10.,10.,10.006,10.007])
    observer,request=tracker(lambda:next(ticks))
    observer.before_add(request)
    observer.after_add(request)
    row=observer.records.rows[0]
    assert row['forecast']['available'] and not row['forecast']['fallback_required']
    assert row['completion_age_s']==pytest.approx(.006) and not row['completion_age_passed']
    assert row['fallback_required'] and row['engine_enqueue']==10.007
    assert not observer.capture()['actuation_authorized']


@pytest.mark.parametrize('kind',['snapshot_error','stale','overdue','wide','off'])
def test_shadow_retains_unavailable_and_wide_observations_without_actuation(kind):
    state=empty()
    if kind=='stale': state['observed_at']=9.9
    if kind=='overdue': state=busy();state['last_commit_at']=9.;state['pending'][0]['scheduled_at']=9.
    observer,request=tracker(lambda:10.,state=state,observed=kind!='off')
    if kind=='snapshot_error':
        def fail(): raise ValueError('unobserved pending batch')
        observer.snapshot=fail
    if kind=='wide':
        document=assumptions();document['prefill_return_s']=[.001,.15]
        runtime=runtime_metadata()
        observer.predictor=envelope.PreparedEnvelope(shadow.resolve_support(document,runtime),runtime)
    observer.before_add(request);observer.after_add(request)
    row=observer.records.rows[0]
    if kind=='off': assert row['status']=='off' and row['forecast'] is None
    else:
        assert row['fallback_required']
        if kind=='snapshot_error': assert row['error']=='ValueError: unobserved pending batch'
        elif kind=='wide': assert row['forecast']['available'] and row['forecast']['reason']=='interval_too_wide'
        else: assert not row['forecast']['available']
    assert observer.core._prefill_trial.requests['probe']['enqueue']==10.


def synthetic_rows():
    return [dict(**{k:t[k] for k in ('trial_id','stage','case','block','observer')},
                 available=t['observer']=='light',covered=t['observer']=='light',narrow=True,useful=t['observer']=='light',
                 fallback_required=False,reason=None,engine_from_cut_s=.1,measurement_valid=True)
            for t in shadow.protocol()['order']]


def test_scoring_keeps_misses_unavailable_wide_and_unattempted_denominators():
    rows=synthetic_rows()
    lights=[r for r in rows if r['stage']=='shadow' and r['observer']=='light']
    lights[0].update(available=False,covered=False,narrow=False,useful=False,fallback_required=True,reason='missing')
    lights[1].update(covered=False,useful=False)
    lights[2].update(narrow=False,useful=False,fallback_required=True,reason='interval_too_wide')
    report=shadow.score(rows,[r['trial_id'] for r in rows],[])
    result=report['groups']['shadow']
    assert (result['planned'],result['available'],result['unavailable'],result['misses'],result['wide'])==(8,7,1,1,1)
    assert result['covered_per_planned']==6/8 and result['useful_per_planned']==5/8
    assert report['collection_complete'] and all(report[k] is False for k in shadow.flags())
    prefix=shadow.score(rows[:5],[r['trial_id'] for r in rows[:6]],[{'message':'stopped'}])
    assert prefix['groups']['shadow']['planned']==8 and prefix['groups']['shadow']['not_measured']==8
    assert not prefix['collection_complete']


@pytest.mark.parametrize('fault',[None,'generator','cancel','source','ack','drift'])
def test_collector_seals_once_and_preserves_partial_failures_without_retries(monkeypatch,tmp_path,fault):
    from prefill_lab import causal
    from scripts import run_prefill_service
    runtime,m,_=config_fixture()
    seen=[]
    monkeypatch.setattr(shadow,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(shadow.live_http,'clock_domain',lambda:'test')
    monkeypatch.setattr(run_prefill_service,'code_identity',
                        lambda:{'code':'changed' if fault=='source' and 'es-001' in seen else 'hash'})
    monkeypatch.setattr(causal,'dispatcher_parity',lambda *a:dict(matches=True))
    async def command(op,**kw):
        seen.append(op)
        if op=='meta':
            return dict(runtime,drift=True) if fault=='drift' and seen.count('meta')==2 else runtime
        if op=='shadow_configure':
            assert not (tmp_path/'shadow-seal.json').exists()
            return dict(configured=True,frozen_sha256='bad' if fault=='ack' else kw['frozen']['sha256'])
        return {}
    async def generate(path,tid,url,output,deadline):
        assert (tmp_path/'shadow-seal.json').exists()
        seen.append(tid)
        save(output,dict(rows=[]))
        if fault=='generator': raise RuntimeError('generator failed')
        if fault=='cancel': raise asyncio.CancelledError()
        return {}
    async def finish(*a): return {'partial':fault in ('generator','cancel')}
    monkeypatch.setattr(shadow.live_experiment,'isolated_trial',generate)
    monkeypatch.setattr(shadow,'finish_external',finish)
    operation=shadow.run(NS(command=command),None,'',m,assumptions(),tmp_path,'',time.monotonic()+600)
    if fault:
        with pytest.raises(asyncio.CancelledError if fault=='cancel' else (ValueError,RuntimeError)):
            asyncio.run(operation)
    else: assert asyncio.run(operation)['collection_complete']
    summary=json.loads((tmp_path/'shadow-summary.json').read_text())
    assert all(summary[k] is False for k in shadow.flags())
    assert not any(name in seen for name in ('start','live_configure'))
    if fault in ('generator','cancel'):
        assert seen.count('es-001')==1 and 'abort' in seen
        assert json.loads((tmp_path/'es-001.json').read_text())['partial']
        assert summary['attempted']==['es-001'] and not summary['completed']
    elif fault=='ack': assert not summary['attempted']
    elif not fault: assert len(summary['completed'])==20


def raw_bundle(trial, inputs, config, runtime, *, base=10.):
    from prefill_lab import submit_clock, queue_confirmation
    from prefill_lab.async_predictor import simulate
    context=envelope.PreparedEnvelope(config['support'],runtime)
    spec=inputs['trials'][trial['trial_id']][0]
    rid='native-'+trial['trial_id']
    trace=digest(['live-http',trial['trial_id'],spec['id']])[:32]
    point=submit_clock.costs([v*1.01 for v in config['support']['gpu_lower_s']],
                             queue_confirmation.protocol(),.002,.001)
    graph=simulate([Request(rid,512,128)],point,context._dispatch)
    state=empty();state['observed_at']=base
    prediction=context.forecast(state,Request(rid,512,128),now=base,domain='test')
    first=base+graph['requests'][rid]['first_s']
    done=base+graph['requests'][rid]['finished_s']
    record=dict(trace_id=trace,engine_id=rid,clock_domain='test',cut=base,engine_enqueue=base,
                status='off',forecast=None,client_sent=base-.01)
    if trial['observer']=='light':
        record.update(status='predicted',forecast=prediction,snapshot=state,prediction_started=base,
                      evaluation_at=base,prediction_finished=base,snapshot_width_s=0.,
                      completion_age_s=0.,observer_elapsed_s=0.,completion_age_passed=True,fallback_required=False,anchor_count=0)
    steps=[];workers=[];anchors=[]
    if trial['observer']=='light':
        for row in graph['steps']:
            counts={w['request_id']:w['tokens'] for w in row['work']}
            steps.append(dict(row['features'],step_id=row['step_id'],work=row['work'],parity=dict(matches=True),
                              start=base+row['scheduled_s']+(1e-8 if row['scheduled_s']>0 else 0),
                              end=base+row['committed_s']))
            workers.append(dict(step_id=row['step_id'],counts=counts,executed_counts=counts,
                                executed_positions={w['request_id']:w['position'] for w in row['work']},
                                actual_tokens=sum(counts.values()),gpu_forward_s=.001,
                                runtime_mode=row['features']['mode'],padded_tokens=sum(counts.values())))
            batch=len(steps)
            anchors.append(dict(kind='scheduled',batch=batch,scheduled_at=steps[-1]['start'],
                                observed_at=steps[-1]['start'],work=copy.deepcopy(row['work'])))
            anchors.append(dict(kind='committed',batch=batch,observed_at=steps[-1]['end']))
        anchors.sort(key=lambda event:(event['observed_at'],event['kind']=='scheduled'))
    cap=dict(trial_id=trial['trial_id'],observer=trial['observer'],status='complete',start=base-.1,end=done+.01,
             errors=[],overflow=0,pending_step_ids=[],steps=steps,workers=[dict(rows=workers,pending=0,overflow=0)],
             requests={rid:dict(role='sentinel',input_tokens=512,max_tokens=128,output_tokens=128,
                               first=first,finished=done,enqueue=base,trace_id=trace,
                               input_ids_sha256=inputs['pools'][spec['token_key']]['sha256'])},
             live=dict(schema='envelope-shadow-records-v1',records=[record],errors=[],overflow=0,
                       observed=trial['observer']=='light',configuration_sha256=config['sha256'],
                       support_sha256=config['support']['sha256'],clock_domain='test',
                       anchor_events=anchors,anchor_overflow=0,
                       cost_sha256=digest(context._lower.to_dict()),dispatcher_sha256=digest(context._dispatch),
                       actuation_authorized=False,native_coverage_qualified=False))
    client=dict(clock_domain='test',release=base-.01,rows=[dict(id='sentinel',role='sentinel',trace_id=trace,
                input_ids_sha256=inputs['pools'][spec['token_key']]['sha256'],clock_domain='test',
                scheduled=base-.01,sent=base-.01,first=first+.001,finished=done+.001,
                status='complete',stream_complete=True)])
    journal=[dict(event=event,monotonic=at,request_id=trace,run_id='gateway') for event,at in
             [('received',base-.009),('accepted',base-.008),('backend_sent',base-.007),
              ('first_content',first+.0005),('completed',done+.005)]]
    journal.append(dict(event='bridge_clock',monotonic=base-.008,request_id=trace,run_id='gateway',clock_domain='test'))
    return dict(capture=cap,client=client,gateway=journal)


def raw_fixture():
    runtime,inputs,config=config_fixture(gpu_upper_factor=1.02)
    trial=shadow.protocol()['order'][0]
    return raw_bundle(trial,inputs,config,runtime),trial,inputs,runtime,config


def test_raw_interval_work_identity_and_http_joins_are_independently_rebuilt():
    bundle,trial,m,runtime,config=raw_fixture()
    result=shadow.extract(bundle,trial,m,runtime,config)
    assert result['available'] and result['covered'] and result['useful'] and result['structure_matches']
    assert result['completion_age_s']==0 and result['snapshot_width_s']==0
    assert not result['actuation_authorized']


@pytest.mark.parametrize('fault',['interval','age','state','config','clock','input','output','extra_request','forecast_time','fallback','work','invented_pending','step_window'])
def test_raw_evidence_tampering_is_not_a_passing_measurement(fault):
    bundle,trial,m,runtime,config=raw_fixture()
    cap=bundle['capture'];row=cap['live']['records'][0]
    if fault=='interval': row['forecast']['ttft_from_cut_s'][1]+=1
    if fault=='age': row['completion_age_s']=.001
    if fault=='state': row['snapshot']['observed_at']=9.
    if fault=='config': cap['live']['configuration_sha256']='other'
    if fault=='clock': row['clock_domain']='another-boot'
    if fault=='input': bundle['client']['rows'][0]['input_ids_sha256']='wrong'
    if fault=='output': next(iter(cap['requests'].values()))['output_tokens']=127
    if fault=='extra_request': cap['requests']['unexpected']=dict(next(iter(cap['requests'].values())))
    if fault=='forecast_time': row['prediction_finished']=row['engine_enqueue']+.001
    if fault=='fallback': row['fallback_required']=True
    if fault=='work': cap['steps'][0]['work'][0]['tokens']-=1
    if fault=='step_window': cap['steps'][0]['start']=.5
    if fault=='invented_pending':
        row['snapshot']=busy(False)
        context=envelope.PreparedEnvelope(config['support'],runtime)
        row['forecast']=context.forecast(row['snapshot'],Request(row['engine_id'],512,128),now=10.,domain='test')
    with pytest.raises(ValueError): shadow.extract(bundle,trial,m,runtime,config)


def test_raw_stale_unavailable_and_wide_intervals_remain_in_measurements():
    bundle,trial,m,runtime,config=raw_fixture()
    row=bundle['capture']['live']['records'][0]
    config['support']['policy']['max_width_s']=1e-9
    config['support']['sha256']=digest({k:v for k,v in config['support'].items() if k!='sha256'})
    config['sha256']=digest({k:v for k,v in config.items() if k!='sha256'})
    bundle=raw_bundle(trial,m,config,runtime)
    # raw_bundle defaults to no fallback; reconstruct the actual runtime decision.
    row=bundle['capture']['live']['records'][0];row['fallback_required']=True
    result=shadow.extract(bundle,trial,m,runtime,config)
    assert result['covered'] and not result['narrow'] and result['fallback_required']
    row.update(evaluation_at=10.006,prediction_finished=10.007,engine_enqueue=10.007,
               completion_age_s=10.007-10.,observer_elapsed_s=10.007-10.,completion_age_passed=False)
    context=envelope.PreparedEnvelope(config['support'],runtime)
    rid=row['engine_id']
    bundle['capture']['requests'][rid]['enqueue']=10.007
    row['forecast']=context.forecast(row['snapshot'],Request(rid,512,128),now=10.006,domain='test')
    row['status']='unavailable'
    result=shadow.extract(bundle,trial,m,runtime,config)
    assert not result['available'] and result['reason']=='stale_snapshot'


def write_audit_fixture(monkeypatch,root):
    from scripts import check_prefill_envelope_shadow as checker
    from prefill_lab import suite
    p=shadow.protocol()
    p['cases']=p['cases'][:1]
    p['order']=[t for t in p['order'] if t['case']=='empty']
    monkeypatch.setattr(shadow,'protocol',lambda:copy.deepcopy(p))
    bundle,_,m,runtime,config=raw_fixture()
    monkeypatch.setattr(checker,'native_evidence',lambda *a:{'code':'hash'})
    monkeypatch.setattr(suite,'runtime_preflight',lambda *a:None)
    save(root/'shadow-plan.json',p);save(root/'runtime.json',runtime);save(root/'runtime-final.json',runtime)
    save(root/'token-manifest.json',m);save(root/'http-token-manifest.json',m)
    save(root/'support-input.json',config['support']);save(root/'shadow-config.json',config)
    save(root/'shadow-seal.json',dict(at_monotonic=2.,config_sha256=config['sha256'],completed=[],source_sha256={'code':'hash'}))
    save(root/'shadow-configured.json',dict(configured=True,frozen_sha256=config['sha256']))
    ids=[t['trial_id'] for t in p['order']]
    save(root/'shadow-summary.json',dict(attempted=ids,completed=ids,failures=[],collection_complete=True,**shadow.flags()))
    journal=[]
    for i,t in enumerate(p['order']):
        b=raw_bundle(t,m,config,runtime,base=10.+i*3)
        save(root/(t['trial_id']+'.json'),b['capture'])
        save(root/(t['trial_id']+'-client.json'),b['client'])
        journal.extend(b['gateway'])
    (root/'gateway.jsonl').write_text('\n'.join(json.dumps(r) for r in journal)+'\n')
    metrics=dict(admission_mode='protected',max_in_flight=32,max_waiting_capacity=0,shared_upstream_client=True,
                 prefill_raw_completions=True,gateway_diagnostics=True,telemetry_dropped_records=0,
                 telemetry_write_errors=0,telemetry_run_id='gateway',accepted_requests=0,rejected_requests=0,
                 failed_requests=0,current_in_flight=0,current_waiting=0)
    save(root/'gateway-start.json',metrics)
    save(root/'gateway-final.json',dict(metrics,accepted_requests=len(ids)))
    save(root/'telemetry-status.json',dict(run_id='gateway',status='closed',dropped=0,write_errors=0,diagnostics=dict(complete=True)))
    return checker,p


def test_closed_raw_auditor_reconstructs_complete_report_not_saved_scores(monkeypatch,tmp_path):
    checker,p=write_audit_fixture(monkeypatch,tmp_path)
    report=checker.run(tmp_path)
    assert report['collection_complete'] and not report['integrity_errors']
    assert report['groups']['shadow']['covered']==2 and report['measured_trials']==5
    assert len(report['paired_controls'])==2 and all(r['comparable'] for r in report['paired_controls'])
    assert not (tmp_path/'shadow-report.json').exists()


@pytest.mark.parametrize('fault',['seal','membership','extra','native_ack','input','telemetry','runtime','raw'])
def test_closed_audit_rejects_or_explicitly_counts_corruption(monkeypatch,tmp_path,fault):
    checker,p=write_audit_fixture(monkeypatch,tmp_path)
    names=dict(seal='shadow-seal.json',membership='shadow-summary.json',native_ack='shadow-configured.json',
               input='support-input.json',telemetry='telemetry-status.json',runtime='runtime-final.json',raw='es-001.json')
    if fault=='extra': save(tmp_path/'es-999.json',{})
    else:
        path=tmp_path/names[fault];data=json.loads(path.read_text())
        if fault=='seal': data['at_monotonic']=1000.
        if fault=='membership': data['attempted'].reverse()
        if fault=='native_ack': data['frozen_sha256']='wrong'
        if fault=='input': data['gpu_lower_s'][0]+=1.
        if fault=='telemetry': data['status']='open'
        if fault=='runtime': data['process_id']+=1
        if fault=='raw': data['live']['records'][0]['forecast']['ttft_from_cut_s'][1]+=1.
        save(path,data)
    if fault=='raw':
        report=checker.run(tmp_path)
        assert not report['collection_complete'] and len(report['integrity_errors'])==1
        assert report['groups']['warmup']['planned']==1 and report['groups']['warmup']['not_measured']==1
    else:
        with pytest.raises(ValueError): checker.run(tmp_path)


@pytest.mark.parametrize('fault',[None,'scope','cpu','source','junit','flag','old_mode','installed_source'])
def test_native_audit_keeps_source_cpu_and_mode_gates(monkeypatch,tmp_path,fault):
    from scripts import check_prefill_envelope_shadow as checker, run_prefill_service, run_linux_cpu_tests
    source={'code':'hash'}
    monkeypatch.setattr(run_prefill_service,'code_identity',lambda:source)
    monkeypatch.setattr(run_linux_cpu_tests,'validate_report',lambda *a:dict(test_cases=2 if fault=='junit' else 1))
    env={'PREFILL_LAB_ENABLED':'1','VLLM_PLUGINS':'prefill_lab'}
    env.update({'PREFILL_LAB_'+k:v for k,v in dict(ENVELOPE_SHADOW='1',LIVE_STATE='1',ENTRY_HTTP='0',
        QUEUE_CONFIRMATION='0',QUEUE_PROBE='0',STAGES='0',EFFECTIVE_CALIBRATION='0',DELAYED_ARRIVALS='0',EVENT_CLOCK='0').items()})
    acceptance=dict(source_sha256=source,valid=True,tracked_clean=True,tests=1)
    launch=dict(source_sha256=source,cpu_acceptance=acceptance,scope='envelope_shadow',environment=env)
    runtime=runtime_metadata()
    pins=json.loads((checker.ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    runtime['source']=dict(version='0.30.0',files=pins)
    runtime['workers'][0]['source']=copy.deepcopy(runtime['source'])
    if fault=='scope': launch['scope']='entry_http_state_bridge'
    if fault=='cpu': acceptance['valid']=False
    if fault=='source': launch['source_sha256']={}
    if fault=='flag': env['PREFILL_LAB_ENVELOPE_SHADOW']='0'
    if fault=='old_mode': env['PREFILL_LAB_QUEUE_CONFIRMATION']='1'
    if fault=='installed_source': runtime['workers'][0]['source']['files']={}
    save(tmp_path/'collection-sources.json',source);save(tmp_path/'launch.json',launch)
    (tmp_path/'native-cpu-acceptance').mkdir()
    save(tmp_path/'native-cpu-acceptance/acceptance.json',acceptance)
    if fault:
        with pytest.raises(ValueError): checker.native_evidence(tmp_path,runtime)
    else: assert checker.native_evidence(tmp_path,runtime)==source


def test_partial_audit_counts_failed_last_trial_and_every_unattempted_probe(monkeypatch,tmp_path):
    checker,p=write_audit_fixture(monkeypatch,tmp_path)
    first=p['order'][0]['trial_id']
    save(tmp_path/'shadow-summary.json',dict(attempted=[first],completed=[],failures=[dict(type='CancelledError',message='')],
                                           collection_complete=False,**shadow.flags()))
    cap=json.loads((tmp_path/(first+'.json')).read_text())
    cap['status']='invalid';cap['errors']=['cancelled']
    save(tmp_path/(first+'.json'),cap)
    for t in p['order'][1:]:
        (tmp_path/(t['trial_id']+'.json')).unlink()
        (tmp_path/(t['trial_id']+'-client.json')).unlink()
    journal=[json.loads(line) for line in (tmp_path/'gateway.jsonl').read_text().splitlines()]
    trace=digest(['live-http',first,'sentinel'])[:32]
    (tmp_path/'gateway.jsonl').write_text('\n'.join(json.dumps(r) for r in journal if r['request_id']==trace)+'\n')
    final=json.loads((tmp_path/'gateway-final.json').read_text());final['accepted_requests']=1
    save(tmp_path/'gateway-final.json',final)
    report=checker.run(tmp_path)
    assert not report['collection_complete'] and report['measured_trials']==0
    assert report['groups']['shadow']['planned']==2 and report['groups']['shadow']['not_measured']==2
    assert report['groups']['warmup']['not_measured']==1 and len(report['failures'])==2


def test_resealed_capture_windows_cannot_hide_probes_before_configuration(monkeypatch,tmp_path):
    checker,p=write_audit_fixture(monkeypatch,tmp_path)
    config=json.loads((tmp_path/'shadow-config.json').read_text())
    config['configured_at']=99.
    config['sha256']=digest({k:v for k,v in config.items() if k!='sha256'})
    save(tmp_path/'shadow-config.json',config)
    save(tmp_path/'shadow-configured.json',dict(configured=True,frozen_sha256=config['sha256']))
    save(tmp_path/'shadow-seal.json',dict(at_monotonic=99.5,config_sha256=config['sha256'],completed=[],source_sha256={'code':'hash'}))
    for i,t in enumerate(p['order']):
        path=tmp_path/(t['trial_id']+'.json');cap=json.loads(path.read_text())
        cap.update(start=100.+i*3,end=102.+i*3)
        cap['live']['configuration_sha256']=config['sha256']
        save(path,cap)
    report=checker.run(tmp_path)
    assert not report['collection_complete'] and len(report['integrity_errors'])==5


def test_interrupted_generator_without_client_sidecar_keeps_planned_denominator(monkeypatch,tmp_path):
    checker,p=write_audit_fixture(monkeypatch,tmp_path)
    first=p['order'][0]['trial_id']
    save(tmp_path/'shadow-summary.json',dict(attempted=[first],completed=[],failures=[dict(type='CancelledError',message='')],
                                           collection_complete=False,**shadow.flags()))
    (tmp_path/(first+'-client.json')).unlink()
    for t in p['order'][1:]:
        (tmp_path/(t['trial_id']+'.json')).unlink()
        (tmp_path/(t['trial_id']+'-client.json')).unlink()
    trace=digest(['live-http',first,'sentinel'])[:32]
    journal=[json.loads(line) for line in (tmp_path/'gateway.jsonl').read_text().splitlines()]
    (tmp_path/'gateway.jsonl').write_text('\n'.join(json.dumps(r) for r in journal if r['request_id']==trace)+'\n')
    final=json.loads((tmp_path/'gateway-final.json').read_text());final['accepted_requests']=1
    save(tmp_path/'gateway-final.json',final)
    report=checker.run(tmp_path)
    assert not report['collection_complete'] and report['measured_trials']==0
    assert report['groups']['warmup']['not_measured']==1 and report['groups']['shadow']['not_measured']==2


@pytest.mark.parametrize('fault',[None,'commit','scheduled','missing_event','overflow'])
def test_snapshot_pending_timing_is_rebuilt_from_complete_anchor_log(fault):
    work=[dict(request_id='a',position=512,tokens=1)]
    first=[dict(request_id='a',position=0,tokens=512)]
    steps=[dict(start=10.,end=10.1,work=first),dict(start=10.02,end=10.2,work=work)]
    events=[dict(kind='scheduled',batch=1,scheduled_at=10.,observed_at=10.,work=first),
            dict(kind='scheduled',batch=2,scheduled_at=10.02,observed_at=10.02,work=work),
            dict(kind='committed',batch=1,observed_at=10.0999),
            dict(kind='committed',batch=2,observed_at=10.1999)]
    record=dict(cut=10.15,anchor_count=3,snapshot=dict(last_commit_at=10.0999,
                pending=[dict(work=work,scheduled_at=10.02,ready=False)]))
    observed=dict(anchor_events=events,anchor_overflow=0)
    if fault=='commit': record['snapshot']['last_commit_at']-=.005
    if fault=='scheduled': record['snapshot']['pending'][0]['scheduled_at']-=.005
    if fault=='missing_event': events.pop()
    if fault=='overflow': observed['anchor_overflow']=1
    if fault:
        with pytest.raises(ValueError): shadow.validate_anchors(dict(start=9.,end=11.,steps=steps),observed,record)
    else: shadow.validate_anchors(dict(start=9.,end=11.,steps=steps),observed,record)


def test_empty_batch_anchors_are_not_fabricated_from_positive_worker_steps(monkeypatch):
    observer,_=tracker(lambda:10.)
    output=object()
    monkeypatch.setattr(live_runtime,'scheduled_work',lambda output:[])
    observer.scheduled(output,9.9);observer.committed(output)
    recorded=observer.capture()
    assert [e['kind'] for e in recorded['anchor_events']]==['scheduled','committed']
    assert not recorded['anchor_events'][0]['work']
    shadow.validate_anchors(dict(start=9.,end=11.,steps=[]),recorded,
                            dict(cut=10.,anchor_count=2,snapshot=dict(last_commit_at=10.,pending=[])))


@pytest.mark.parametrize('fault',[None,'overflow','missing','incomplete'])
def test_snapshot_failure_still_requires_complete_capture_anchor_evidence(fault):
    bundle,trial,m,runtime,config=raw_fixture()
    observed=bundle['capture']['live'];row=observed['records'][0]
    row.pop('snapshot');row.pop('anchor_count')
    row.update(status='unavailable',forecast=None,error='ValueError: snapshot unavailable',fallback_required=True)
    if fault=='overflow': observed['anchor_overflow']=1
    if fault=='missing': observed.pop('anchor_events')
    if fault=='incomplete': observed['anchor_events'].pop()
    if fault:
        with pytest.raises(ValueError): shadow.extract(bundle,trial,m,runtime,config)
    else:
        result=shadow.extract(bundle,trial,m,runtime,config)
        assert not result['available'] and result['fallback_required']


def test_actual_plugin_hook_order_matches_anchor_audit_inequalities(monkeypatch):
    from prefill_lab import plugin
    ticks=iter([1.,2.,3.,4.,5.])
    clock=lambda:next(ticks)
    output=NS(num_scheduled_tokens=1)
    work=[dict(request_id='a',position=0,tokens=512)]
    monkeypatch.setattr(plugin,'installed_sources',lambda:None)
    monkeypatch.setattr(plugin.sys,'setswitchinterval',lambda value:None)
    monkeypatch.setattr(plugin.time,'monotonic',clock)
    monkeypatch.setattr(plugin,'_WORKER',None)
    monkeypatch.setattr(live_runtime,'scheduled_work',lambda output:work)
    core=NS(scheduler=NS(schedule=lambda:output,update_from_output=lambda output:{}),batch_queue=[])
    plugin.core_runtime(core)
    steps=[]
    def after_schedule(output,before):
        steps.append(dict(start=before[0],end=None,work=work))
    def after_output(output,results):
        steps[-1]['end']=clock()
    core._prefill_trial=NS(mode='light',before_schedule=lambda:(clock(),{}),after_schedule=after_schedule,
                           after_output=after_output,errors=[])
    runtime=runtime_metadata()
    context=envelope.PreparedEnvelope(shadow.resolve_support(assumptions(),runtime),runtime)
    core._prefill_live=live_runtime.EnvelopeTracker(core,context,clock=clock,domain='test')
    core.scheduler.schedule();core.scheduler.update_from_output(output)
    evidence=core._prefill_live.capture()
    assert steps[0]['start']==1. and evidence['anchor_events'][0]['scheduled_at']==2.
    assert evidence['anchor_events'][1]['observed_at']==4. and steps[0]['end']==5.
    assert not core._prefill_trial.errors
    shadow.validate_anchors(dict(start=0.,end=6.,steps=steps),evidence)
