"""Dedicated shadow launch and native boundaries, mocked CPU evidence only."""
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace as NS

import httpx
import pytest

from prefill_lab import envelope_shadow as shadow, plugin
from prefill_lab.contracts import digest, save
from prefill_lab.tokens import materialize
from scripts import prefill_host as host, run_prefill_service as runner
from test_prefill_async_predictor import runtime_metadata


LEGACY=('STAGES','EFFECTIVE_CALIBRATION','DELAYED_ARRIVALS','EVENT_CLOCK',
        'QUEUE_PROBE','QUEUE_CONFIRMATION','ENTRY_HTTP')
EMPTY_GATEWAY=dict(accepted_requests=0,rejected_requests=0,failed_requests=0,current_in_flight=0,current_waiting=0)


def enable_shadow(monkeypatch):
    monkeypatch.setenv('PREFILL_LAB_ENVELOPE_SHADOW','1')
    monkeypatch.setenv('PREFILL_LAB_LIVE_STATE','1')
    for name in LEGACY:monkeypatch.setenv('PREFILL_LAB_'+name,'0')


def manifest():
    p=shadow.protocol()
    tokenizer=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    return materialize(p['cases'],p['order'],tokenizer,model=p['model']['name'],
                       revision=p['model']['revision'],tokenizer_sha256='synthetic')


def assumptions():
    return dict(schema='envelope-shadow-assumptions-v1',gpu_lower_s=[.01,.1,.002,.003,.001,.004],
                gpu_upper_s=[.02,.2,.004,.006,.002,.008],prefill_return_s=[.001,.002],
                decode_return_s=[.001,.002],policy=dict(max_age_s=.005,max_width_s=.01))


def native_fixture(monkeypatch,tmp_path):
    enable_shadow(monkeypatch)
    m=manifest();save(tmp_path/'manifest.json',m)
    monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(tmp_path/'manifest.json'))
    runtime=runtime_metadata();runtime['token_manifest_sha256']=digest(m)
    config=shadow.configure(shadow.resolve_support(assumptions(),runtime),runtime,m,
                            now=1.,domain='test',sources={'code':'hash'})
    core=NS(_prefill_trial=None,_prefill_external=False,batch_queue=[],
            scheduler=NS(has_requests=lambda:False,running=[],waiting=[],requests={}),
            model_executor=NS(collective_rpc=lambda *a,**kw:runtime['workers']))
    monkeypatch.setattr(plugin,'core_runtime',lambda *a:None)
    monkeypatch.setattr(plugin,'resolved_config',lambda *a:{k:v for k,v in runtime.items() if k!='workers'})
    monkeypatch.setattr(runner,'code_identity',lambda:{'code':'hash'})
    from prefill_lab import live_http,live_runtime
    monkeypatch.setattr(live_http,'clock_domain',lambda:'test')
    monkeypatch.setattr(live_runtime,'clock_domain',lambda:'test')
    return core,dict(operation='shadow_configure',runtime=copy.deepcopy(runtime),frozen=config)


def test_native_shadow_configuration_prepares_once_without_point_readiness(monkeypatch,tmp_path):
    from prefill_lab.live_runtime import EnvelopeTracker
    from prefill_lab.state_envelope import PreparedEnvelope
    core,payload=native_fixture(monkeypatch,tmp_path)
    answer=plugin.command(core,payload)
    assert answer==dict(configured=True,frozen_sha256=payload['frozen']['sha256'])
    prepared=core._prefill_shadow_config
    assert isinstance(prepared,PreparedEnvelope)
    monkeypatch.setattr(shadow,'validate_config',lambda *a,**kw:pytest.fail('revalidated on hot path'))
    plugin.command(core,dict(operation='external_begin',trial_id='es-001',observer='light',expected_foreground=1))
    assert isinstance(core._prefill_live,EnvelopeTracker) and core._prefill_live.predictor is prepared
    assert core._prefill_live.configuration_sha256==payload['frozen']['sha256']
    assert core._prefill_external and core._prefill_live.observed


@pytest.mark.parametrize('fault',['disabled','live_disabled','busy','batch','trial','external','legacy_config',
    'runtime','worker','domain','source','manifest','future',*LEGACY])
def test_native_shadow_rejects_conflicts_and_runtime_drift_before_install(monkeypatch,tmp_path,fault):
    core,payload=native_fixture(monkeypatch,tmp_path)
    if fault=='disabled':monkeypatch.setenv('PREFILL_LAB_ENVELOPE_SHADOW','0')
    elif fault=='live_disabled':monkeypatch.setenv('PREFILL_LAB_LIVE_STATE','0')
    elif fault=='busy':core.scheduler.has_requests=lambda:True
    elif fault=='batch':core.batch_queue=[object()]
    elif fault=='trial':core._prefill_trial=object()
    elif fault=='external':core._prefill_external=True
    elif fault=='legacy_config':core._prefill_live_config=object()
    elif fault=='runtime':payload['runtime']['process_id']+=1
    elif fault=='worker':payload['runtime']['workers'][0]['pid']+=1
    elif fault in LEGACY:monkeypatch.setenv('PREFILL_LAB_'+fault,'1')
    else:
        frozen=payload['frozen']
        if fault=='domain':frozen['clock_domain']='another-boot'
        elif fault=='source':frozen['source_sha256']={'code':'changed'}
        elif fault=='manifest':frozen['manifest_sha256']='other-manifest'
        elif fault=='future':frozen['configured_at']=1e99
        frozen['sha256']=digest({k:v for k,v in frozen.items() if k!='sha256'})
    with pytest.raises((ValueError,RuntimeError)):plugin.command(core,payload)
    assert not hasattr(core,'_prefill_shadow_config')


@pytest.mark.parametrize('fault',['unconfigured','legacy_unconfigured','disabled','bad_flag','live_disabled',*LEGACY])
def test_external_begin_config_failure_never_marks_engine_external(monkeypatch,tmp_path,fault):
    core,payload=native_fixture(monkeypatch,tmp_path)
    if fault not in ('unconfigured','legacy_unconfigured'):plugin.command(core,payload)
    if fault in ('disabled','legacy_unconfigured'):monkeypatch.setenv('PREFILL_LAB_ENVELOPE_SHADOW','0')
    elif fault=='bad_flag':monkeypatch.setenv('PREFILL_LAB_ENVELOPE_SHADOW','true')
    elif fault=='live_disabled':monkeypatch.setenv('PREFILL_LAB_LIVE_STATE','0')
    elif fault in LEGACY:monkeypatch.setenv('PREFILL_LAB_'+fault,'1')
    with pytest.raises((ValueError,RuntimeError)):
        plugin.command(core,dict(operation='external_begin',trial_id='es-001',observer='light'))
    assert not core._prefill_external and core._prefill_trial is None


@pytest.mark.parametrize('operation',['live_configure','start'])
def test_native_shadow_cannot_enter_legacy_modes(monkeypatch,tmp_path,operation):
    core,payload=native_fixture(monkeypatch,tmp_path)
    with pytest.raises(RuntimeError):plugin.command(core,dict(payload,operation=operation))
    assert not core._prefill_external


def test_native_shadow_off_arm_keeps_tracker_without_forecasting(monkeypatch,tmp_path):
    from prefill_lab.live_runtime import EnvelopeTracker
    core,payload=native_fixture(monkeypatch,tmp_path);plugin.command(core,payload)
    plugin.command(core,dict(operation='external_begin',trial_id='es-005',observer='off'))
    assert isinstance(core._prefill_live,EnvelopeTracker) and not core._prefill_live.observed


def test_endpoint_enforces_exact_shadow_membership_observer_and_foreground_counts(monkeypatch,tmp_path):
    from fastapi import FastAPI
    from prefill_lab.endpoint import Endpoint
    enable_shadow(monkeypatch);m=manifest();save(tmp_path/'manifest.json',m)
    monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(tmp_path/'manifest.json'))
    key='test-only-'*4;monkeypatch.setenv('PREFILL_LAB_KEY',key)
    seen=[]
    async def utility(name,payload):seen.append(payload);return dict(ok=True)
    async def run():
        endpoint=Endpoint()
        await endpoint.init_state(NS(engine_core=NS(call_utility_async=utility)),None,None)
        assert endpoint.external_ids==set(m['trials']) and not endpoint.trials
        app=FastAPI();endpoint.attach_router(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as http:
            assert (await http.post('/prefill-lab/command',json={'operation':'shadow_configure'})).status_code==401
            http.headers['authorization']='Bearer '+key
            async def post(payload):return await http.post('/prefill-lab/command',json=payload)
            assert (await post(dict(operation='shadow_configure',frozen={},runtime={}))).status_code==200
            for trial in shadow.protocol()['order']:
                tid=trial['trial_id'];payload=dict(operation='external_begin',trial_id=tid,observer=trial['observer'])
                assert (await post(payload)).status_code==200
                assert seen[-1]['expected_foreground']==sum(r['role']!='background' for r in m['trials'][tid])
                assert (await post(dict(payload,observer='off' if trial['observer']=='light' else 'light'))).status_code==400
            for payload in (dict(operation='external_begin',trial_id='not-shadow',observer='light'),
                            dict(operation='external_begin',trial_id='es-005'),
                            dict(operation='start',trial_id='es-001'),
                            dict(operation='live_configure',frozen={},runtime={}),
                            dict(operation='shadow_configure',frozen={}),
                            dict(operation='status',runtime={}),
                            dict(operation='external_begin',trial_id='es-001',expected_foreground=0)):
                assert (await post(payload)).status_code==400
    asyncio.run(run())


@pytest.mark.parametrize('fault',['disabled',*LEGACY])
def test_endpoint_shadow_configuration_and_mode_fail_closed(monkeypatch,tmp_path,fault):
    from fastapi import FastAPI
    from prefill_lab.endpoint import Endpoint
    enable_shadow(monkeypatch);save(tmp_path/'manifest.json',manifest())
    monkeypatch.setenv('PREFILL_LAB_MANIFEST',str(tmp_path/'manifest.json'))
    if fault!='disabled':
        monkeypatch.setenv('PREFILL_LAB_'+fault,'1')
        with pytest.raises(ValueError):
            asyncio.run(Endpoint().init_state(NS(engine_core=NS(call_utility_async=lambda *a:None)),None,None))
        return
    endpoint=Endpoint();endpoint.live_enabled=False;endpoint.shadow_enabled=False
    app=FastAPI();endpoint.attach_router(app);key='test-only-'*4;monkeypatch.setenv('PREFILL_LAB_KEY',key)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',
                                    headers={'authorization':'Bearer '+key}) as http:
            response=await http.post('/prefill-lab/command',json=dict(operation='shadow_configure',frozen={},runtime={}))
            assert response.status_code==400
    asyncio.run(run())


@pytest.mark.parametrize('extra,valid',[
    (['--envelope-shadow','--support','support.json'],True),
    (['--trace','trace.json'],True),
    (['--envelope-shadow'],False),(['--support','support.json'],False),([],False),
    (['--envelope-shadow','--support','support.json','--trace','trace.json'],False),
    (['--envelope-shadow','--support','support.json','--calibration-only'],False),
    (['--envelope-shadow','--support','support.json','--live-http-only'],False),
    (['--envelope-shadow','--support','support.json','--http-manifest','http.json'],False),
])
def test_collect_parser_requires_support_only_in_dedicated_shadow_mode(monkeypatch,extra,valid):
    argv=['run_prefill_service.py','collect','--manifest','tokens.json','--tokenizer','model',
          '--cpu-acceptance','acceptance.json','--output','capture',*extra]
    monkeypatch.setattr(sys,'argv',argv);seen=[]
    async def collect(args):seen.append(args);return 0
    monkeypatch.setattr(runner,'collect',collect)
    if valid:
        assert runner.main()==0 and len(seen)==1
        assert bool(seen[0].support)==seen[0].envelope_shadow
    else:
        with pytest.raises(SystemExit) as error:runner.main()
        assert error.value.code==2 and not seen


def test_prepare_parser_materializes_dedicated_shadow_tokens(monkeypatch,tmp_path):
    tokenizer=NS(get_vocab=lambda:{'a':1,'b':2},all_special_ids=[])
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=NS(from_pretrained=lambda *a,**kw:tokenizer)))
    p=shadow.protocol()
    monkeypatch.setattr(sys,'argv',['run_prefill_service.py','prepare','--tokenizer',str(tmp_path),
        '--revision',p['model']['revision'],'--output',str(tmp_path/'tokens.json'),'--envelope-shadow-tokens'])
    assert runner.main()==0
    result=json.loads((tmp_path/'tokens.json').read_text())
    shadow.validate_manifest(result)
    assert set(result['trials'])=={t['trial_id'] for t in p['order']}


def collect_fixture(monkeypatch,tmp_path):
    from diagnostic_sampling import ObservationSampler
    import trace_workload
    from scripts import setup_v1
    monkeypatch.setattr(runner.platform,'system',lambda:'Linux')
    monkeypatch.setattr(runner,'sys',NS(version_info=(3,12),executable=sys.executable))
    monkeypatch.setattr(runner,'runtime_environment',lambda:{})
    monkeypatch.setattr(ObservationSampler,'process_resources',lambda *a:dict(available=True))
    monkeypatch.setattr(runner,'code_identity',lambda:{'code':'hash'})
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'commit')
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=NS(from_pretrained=lambda *a,**kw:object())))
    monkeypatch.setattr(trace_workload,'load_trace',lambda *a:pytest.fail('shadow must not load Azure trace'))
    monkeypatch.setattr(trace_workload,'verify_prompts',lambda *a:pytest.fail('shadow must not verify Azure trace'))
    monkeypatch.setattr(setup_v1,'check_port',lambda *a:None)
    m=manifest();m['tokenizer_files']={};save(tmp_path/'tokens.json',m)
    save(tmp_path/'support.json',assumptions())
    acceptance=tmp_path/'cpu';acceptance.mkdir()
    save(acceptance/'acceptance.json',dict(valid=True,commit='commit',source_sha256={'code':'hash'}))
    (acceptance/'junit.xml').write_text('<testsuites/>');(acceptance/'pytest.log').write_text('native CPU proof')
    args=NS(envelope_shadow=True,support=tmp_path/'support.json',manifest=tmp_path/'tokens.json',
            tokenizer=tmp_path,cpu_acceptance=acceptance/'acceptance.json',output=tmp_path/'capture',
            startup_timeout=900)
    events=[];processes=[]
    class Process:
        def __init__(self,argv,**kw):
            self.argv,self.env,self.log=argv,kw['env'],kw['stdout']
            self.pid=100+len(processes);self.stopped=False;processes.append(self)
        def poll(self):return 0 if self.stopped else None
        def wait(self,**kw):
            events.append(('closed',self.pid));self.stopped=True
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:events.append(('signal',pid)),raising=False)
    monkeypatch.setattr(runner.signal,'SIGKILL',9,raising=False)
    class HTTP:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def get(self,url):
            return NS(status_code=200,json=lambda:dict(EMPTY_GATEWAY),raise_for_status=lambda:None)
        async def post(self,*a,**kw):return NS(raise_for_status=lambda:None,json=lambda:{})
    monkeypatch.setattr(httpx,'AsyncClient',HTTP)
    from prefill_lab import live_http
    monkeypatch.setattr(live_http,'gateway_contract',lambda *a:None)
    return args,events,processes


@pytest.mark.parametrize('failure',[None,'error','timeout','cancel'])
def test_shadow_collect_needs_no_trace_forwards_support_and_closes_processes(monkeypatch,tmp_path,failure):
    args,events,processes=collect_fixture(monkeypatch,tmp_path)
    async def run(client,http,url,m,support,output,gateway_url,deadline):
        assert support==assumptions() and gateway_url=='http://127.0.0.1:8080'
        assert shadow.validate_manifest(m)
        assert json.loads((output/'support-input.json').read_text())==support
        assert 0<deadline-runner.time.monotonic()<=shadow.protocol()['hard_seconds']-runner.SHADOW_CLEANUP_SECONDS
        assert not (output/'trace-identity.json').exists()
        assert (output/'native-cpu-acceptance/junit.xml').read_text()=='<testsuites/>'
        events.append(('run',None))
        if failure=='error':raise RuntimeError('synthetic shadow failure')
        if failure=='timeout':raise TimeoutError('synthetic shadow failure')
        if failure=='cancel':raise asyncio.CancelledError('synthetic shadow failure')
        return dict(collection_complete=True)
    monkeypatch.setattr(shadow,'run',run)
    if failure:
        error={'error':RuntimeError,'timeout':TimeoutError,'cancel':asyncio.CancelledError}[failure]
        with pytest.raises(error,match='synthetic shadow failure'):asyncio.run(runner.collect(args))
        assert (args.output/'startup-or-run-failure.json').exists()
    else:assert asyncio.run(runner.collect(args))==0
    assert len(processes)==2 and all(p.stopped and p.log.closed for p in processes)
    assert events==[('run',None),('signal',101),('closed',101),('signal',100),('closed',100)]
    env=processes[0].env
    assert env['PREFILL_LAB_ENVELOPE_SHADOW']==env['PREFILL_LAB_LIVE_STATE']=='1'
    assert all(env['PREFILL_LAB_'+name]=='0' for name in LEGACY)
    assert processes[1].env['ADMISSION_MODE']=='protected'
    assert processes[1].env['MAX_IN_FLIGHT']=='32' and processes[1].env['MAX_WAITING']=='0'
    assert json.loads((args.output/'launch.json').read_text())['scope']=='envelope_shadow'
    assert json.loads((args.output/'gateway-final.json').read_text())==EMPTY_GATEWAY


@pytest.mark.parametrize('counter',list(EMPTY_GATEWAY))
def test_shadow_gateway_requires_fresh_zero_counters(monkeypatch,tmp_path,counter):
    args,_,processes=collect_fixture(monkeypatch,tmp_path)
    get=httpx.AsyncClient.get
    async def stale(self,url):
        result=await get(self,url)
        if url.endswith('/metrics'):result.json=lambda:dict(EMPTY_GATEWAY,**{counter:1})
        return result
    monkeypatch.setattr(httpx.AsyncClient,'get',stale)
    async def run(*a):pytest.fail('nonempty gateway must not start shadow trials')
    monkeypatch.setattr(shadow,'run',run)
    with pytest.raises(ValueError,match='startup'):asyncio.run(runner.collect(args))
    assert all(p.stopped and p.log.closed for p in processes)


@pytest.mark.parametrize('primary_failure',[False,True])
def test_final_gateway_failure_is_bounded_and_preserves_primary_error(monkeypatch,tmp_path,primary_failure):
    args,_,processes=collect_fixture(monkeypatch,tmp_path)
    get=httpx.AsyncClient.get;metrics=0
    async def broken_final(self,url):
        nonlocal metrics
        if url.endswith('/metrics'):
            metrics+=1
            if metrics==2:raise httpx.ReadTimeout('final metrics timeout')
        return await get(self,url)
    monkeypatch.setattr(httpx.AsyncClient,'get',broken_final)
    wait_for=asyncio.wait_for;timeouts=[]
    async def observed_wait(awaitable,timeout):
        timeouts.append(timeout)
        return await wait_for(awaitable,timeout)
    monkeypatch.setattr(asyncio,'wait_for',observed_wait)
    async def run(*args):
        if primary_failure:raise RuntimeError('primary shadow failure')
        return dict(collection_complete=True)
    monkeypatch.setattr(shadow,'run',run)
    with pytest.raises(RuntimeError if primary_failure else httpx.ReadTimeout,
                       match='primary shadow failure' if primary_failure else 'final metrics timeout'):
        asyncio.run(runner.collect(args))
    assert timeouts[-1]==5. and (args.output/'gateway-final-failure.json').exists()
    assert all(p.stopped and p.log.closed for p in processes)
    failure=json.loads((args.output/'startup-or-run-failure.json').read_text())
    assert failure['type']==('RuntimeError' if primary_failure else 'ReadTimeout')


@pytest.mark.parametrize('failure',['grace','kill','primary'])
def test_shadow_shutdown_waits_are_bounded_and_cleanup_continues_on_timeout(monkeypatch,tmp_path,failure):
    args,_,processes=collect_fixture(monkeypatch,tmp_path)
    wait=runner.subprocess.Popen.wait;waits=[]
    def timed_wait(self,*,timeout):
        waits.append((self.pid,timeout))
        if self.pid==101 and (timeout==10 or failure!='grace'):
            raise subprocess.TimeoutExpired(self.argv,timeout)
        return wait(self,timeout=timeout)
    monkeypatch.setattr(runner.subprocess.Popen,'wait',timed_wait)
    async def run(*args):
        if failure=='primary':raise RuntimeError('primary shadow failure')
        return dict(collection_complete=True)
    monkeypatch.setattr(shadow,'run',run)
    if failure=='grace':assert asyncio.run(runner.collect(args))==0
    else:
        with pytest.raises(RuntimeError,match='primary shadow failure' if failure=='primary' else 'processes did not close'):
            asyncio.run(runner.collect(args))
        cleanup=json.loads((args.output/'cleanup-failure.json').read_text())
        assert cleanup['failures'][0]['pid']==101 and not cleanup['complete']
    assert waits==[(101,10),(101,2),(100,10)]
    assert all(p.log.closed for p in processes) and processes[0].stopped
    assert (args.output/'gateway-final.json').exists()


def test_shadow_collect_still_requires_current_native_cpu_source_proof(monkeypatch,tmp_path):
    args,_,processes=collect_fixture(monkeypatch,tmp_path)
    save(args.cpu_acceptance,dict(valid=True,commit='commit',source_sha256={'code':'old'}))
    with pytest.raises(RuntimeError,match='cpu-check'):asyncio.run(runner.collect(args))
    assert not processes and not args.output.exists()


def host_fixture(monkeypatch,tmp_path):
    out=tmp_path/'inputs';seen=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(host.subprocess,'check_output',lambda args,**kw:'' if 'diff' in args else 'a'*40)
    monkeypatch.setattr(host,'code_identity',lambda:{'code':'hash'})
    monkeypatch.setitem(sys.modules,'huggingface_hub',NS(HfApi=lambda:NS(model_info=lambda model:NS(sha='b'*40)),
        snapshot_download=lambda **kw:str(tmp_path/'snapshot')))
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=NS(from_pretrained=lambda *a,**kw:object())))
    monkeypatch.setattr(host,'load_trace',lambda *a:pytest.fail('shadow preparation must not load trace'))
    monkeypatch.setattr(host,'verify_prompts',lambda *a:pytest.fail('shadow preparation must not verify trace'))
    def prepare(args):
        assert args.envelope_shadow_tokens
        seen.append('tokens');save(args.output,manifest())
    monkeypatch.setattr(host,'prepare',prepare)
    def cpu(path):
        seen.append('cpu');path.mkdir(parents=True)
        save(path/'acceptance.json',dict(valid=True,commit='a'*40,source_sha256={'code':'hash'}))
        return 0
    monkeypatch.setattr(host,'cpu_check',cpu)
    return out,seen


def test_shadow_host_preparation_is_independent_and_reuses_only_exact_acceptance(monkeypatch,tmp_path):
    out,seen=host_fixture(monkeypatch,tmp_path)
    first=host.prepare_host(out,envelope_shadow=True)
    assert seen==['tokens','cpu'] and first['mode']=='envelope-shadow'
    assert 'trace' not in first and not (out/'ready.json').exists()
    assert host.prepare_host(out,envelope_shadow=True)==first and seen==['tokens','cpu','tokens']
    assert (out/'shadow-ready.json').exists()
    monkeypatch.setattr(host,'code_identity',lambda:{'code':'changed'})
    assert not host.accepted(Path(first['cpu_acceptance']),'a'*40)


@pytest.mark.parametrize('failure',[None,'collector','checker'])
def test_host_forwards_support_checks_after_child_exit_and_archives_failures(monkeypatch,tmp_path,failure):
    out,_=host_fixture(monkeypatch,tmp_path);host.prepare_host(out,envelope_shadow=True)
    support=tmp_path/'support.json';save(support,assumptions());calls=[]
    def run(argv,**kw):
        calls.append(argv)
        if 'collect' in argv:
            capture=Path(argv[argv.index('--output')+1]);capture.mkdir()
            save(capture/'shadow-summary.json',dict(collection_complete=failure!='collector'))
            save(capture/'support-input.json',assumptions())
            if failure=='collector':raise subprocess.CalledProcessError(2,argv)
            (capture/'closed-gateway.jsonl').write_text('closed')
        else:
            assert len(calls)==2 and '--capture' in argv
            capture=Path(argv[-1]);assert (capture/'closed-gateway.jsonl').exists()
            save(capture/'shadow-report.json',dict(collection_complete=failure!='checker'))
            if failure=='checker':raise subprocess.CalledProcessError(2,argv)
        return NS(returncode=0)
    monkeypatch.setattr(host.subprocess,'run',run)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):host.collect_host(out,envelope_shadow=True,support=support)
    else:host.collect_host(out,envelope_shadow=True,support=support)
    argv=calls[0]
    assert '--envelope-shadow' in argv and '--trace' not in argv and '--http-manifest' not in argv
    assert argv[argv.index('--support')+1]==str(support.resolve())
    assert argv[argv.index('--manifest')+1]==str(out/'envelope-shadow-tokens.json')
    archives=list((tmp_path/'pr32-envelope-shadows').glob('*.tar.gz'));assert len(archives)==1
    with tarfile.open(archives[0]) as archive:
        assert any(name.endswith('/shadow-summary.json') for name in archive.getnames())
        if failure!='collector':assert any(name.endswith('/shadow-report.json') for name in archive.getnames())
    assert len(calls)==(1 if failure=='collector' else 2)


def test_host_shadow_parser_forwards_options_and_rejects_legacy_mix(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(host,'collect_host',lambda output,**kw:calls.append((output,kw)))
    monkeypatch.setattr(sys,'argv',['prefill_host.py','envelope-shadow','--support',str(tmp_path/'support.json'),
                                  '--output-dir',str(tmp_path)])
    host.main()
    assert calls==[(tmp_path,dict(envelope_shadow=True,support=tmp_path/'support.json'))]
    monkeypatch.setattr(host,'prepare_host',lambda output,**kw:calls.append((output,kw)))
    monkeypatch.setattr(sys,'argv',['prefill_host.py','prepare','--envelope-shadow','--output-dir',str(tmp_path)])
    host.main();assert calls[-1]==(tmp_path,dict(envelope_shadow=True))
    for argv in (['envelope-shadow'],['collect','--support','support.json'],['collect','--envelope-shadow']):
        monkeypatch.setattr(sys,'argv',['prefill_host.py',*argv])
        with pytest.raises(SystemExit):host.main()


@pytest.mark.parametrize('audit_failure',[None,'exit','timeout'])
def test_host_audits_closed_partial_capture_without_replacing_collector_failure(monkeypatch,tmp_path,audit_failure):
    out,_=host_fixture(monkeypatch,tmp_path);host.prepare_host(out,envelope_shadow=True)
    support=tmp_path/'support.json';save(support,assumptions());calls=[]
    def run(argv,**kw):
        calls.append(argv)
        if 'collect' in argv:
            capture=Path(argv[argv.index('--output')+1]);capture.mkdir()
            for name in ('shadow-summary.json','shadow-config.json','shadow-seal.json','runtime.json',
                         'gateway-start.json','gateway-final.json','telemetry-status.json'):
                save(capture/name,dict(collection_complete=False))
            (capture/'gateway.jsonl').write_text('closed')
            raise subprocess.CalledProcessError(7,argv)
        assert len(calls)==2 and kw['timeout']==300
        capture=Path(argv[-1]);assert (capture/'gateway.jsonl').read_text()=='closed'
        save(capture/'shadow-report.json',dict(collection_complete=False))
        if audit_failure=='exit':raise subprocess.CalledProcessError(2,argv)
        if audit_failure=='timeout':raise subprocess.TimeoutExpired(argv,300)
    monkeypatch.setattr(host.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        host.collect_host(out,envelope_shadow=True,support=support)
    assert error.value.returncode==7 and error.value.cmd==calls[0]
    archives=list((tmp_path/'pr32-envelope-shadows').glob('*.tar.gz'));assert len(archives)==1
    with tarfile.open(archives[0]) as archive:
        assert any(name.endswith('/shadow-report.json') for name in archive.getnames())
        assert any(name.endswith('/shadow-audit-failure.json') for name in archive.getnames())==bool(audit_failure)


@pytest.mark.parametrize('with_support',[False,True])
def test_setup_shadow_selection_prepares_without_trace_or_collection(monkeypatch,tmp_path,capsys,with_support):
    from scripts import setup_prefill_calibration as setup
    monkeypatch.setattr(setup,'ROOT',tmp_path);(tmp_path/'.venv').mkdir()
    monkeypatch.setattr(setup.Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(setup.platform,'system',lambda:'Linux')
    monkeypatch.setattr(setup,'sys',NS(version_info=(3,12),executable=sys.executable))
    monkeypatch.setattr(setup.subprocess,'check_output',lambda *a,**kw:'')
    calls=[];monkeypatch.setattr(setup.subprocess,'run',lambda argv,**kw:calls.append(argv))
    support=tmp_path/'support.json';save(support,assumptions())
    setup.main(operation='envelope-shadow',support=support if with_support else None)
    assert calls[-1][-3:]==['scripts/prefill_host.py','prepare','--envelope-shadow']
    assert not any('collect' in argv for argv in calls)
    output=capsys.readouterr().out
    assert 'envelope-shadow --support' in output
    if with_support:assert str(support) in output


def test_setup_rejects_support_on_legacy_operation_before_installing(monkeypatch,tmp_path):
    from scripts import setup_prefill_calibration as setup
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected installation'))
    with pytest.raises(ValueError,match='only for envelope shadow'):setup.main(support=tmp_path/'support.json')
    with pytest.raises(ValueError,match='existing JSON'):setup.main(operation='envelope-shadow',support=tmp_path/'missing.json')
