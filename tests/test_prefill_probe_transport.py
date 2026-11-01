"""Prepared HTTP offers: wire equivalence and deterministic release ordering."""
import asyncio
import copy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from prefill_lab import live_http as transport
from scripts import prefill_live_http_trial as generator


IDENTIFIER='a'*32
DOMAIN='test:"\\\nclock'


def payload():
    return dict(model='model"\\\n\u2603',prompt=['escaped "\\\n', 'prefill_lab_client_sent":123}}'],
        max_tokens=2,stream=True,request_id='old-request',
        vllm_xargs=dict(prefill_lab_trace_id='old-trace',prefill_lab_clock_domain='old-domain',
            prefill_lab_client_sent=-1,prefill_lab_role='sentinel',prefill_lab_probe=1,
            custom={'nested':[True,None,{'prefill_lab_client_sent':'do not replace'}]}))


def expected(source,sent):
    result=copy.deepcopy(source)
    result['request_id']=IDENTIFIER
    result['vllm_xargs']=dict(result.get('vllm_xargs',{}),prefill_lab_trace_id=IDENTIFIER,
        prefill_lab_clock_domain=DOMAIN,prefill_lab_client_sent=sent)
    return result


def decode(body):
    def unique(pairs):
        assert len(dict(pairs))==len(pairs), 'Duplicate JSON object fields'
        return dict(pairs)
    return json.loads(body,object_pairs_hook=unique)


@pytest.mark.parametrize('source',[{},dict(vllm_xargs={}),payload()])
def test_prepared_body_is_equivalent_json_with_fresh_metadata(source):
    before=copy.deepcopy(source)
    prepared=transport.prepare_probe(source,IDENTIFIER,DOMAIN)
    for sent in (10.125,12.75,13):
        body=prepared.body(sent)
        assert isinstance(body,bytes)
        assert decode(body)==expected(before,sent)
    assert source==before
    assert prepared.identifier==IDENTIFIER and prepared.domain==DOMAIN
    with pytest.raises(FrozenInstanceError):
        prepared.prefix=b'changed'


def test_prepared_body_freezes_static_values_and_only_encodes_timestamp(monkeypatch):
    source=payload()
    prepared=transport.prepare_probe(source,IDENTIFIER,DOMAIN)
    before=copy.deepcopy(source)
    source['prompt'].append('too late')
    source['vllm_xargs']['custom']['nested'].append('too late')
    canonical=transport.canonical
    encoded=[]
    def scalar_only(value):
        assert type(value) in (int,float)
        encoded.append(value)
        return canonical(value)
    monkeypatch.setattr(transport,'canonical',scalar_only)
    assert decode(prepared.body(10.125))==expected(before,10.125)
    assert decode(prepared.body(11.25))==expected(before,11.25)
    assert encoded==[10.125,11.25]


@pytest.mark.parametrize('sent',[float('nan'),float('inf'),-float('inf'),True,False,None,'1',[],{}])
def test_prepared_body_rejects_nonfinite_or_nonnumeric_timestamp(sent):
    prepared=transport.prepare_probe({},IDENTIFIER,DOMAIN)
    with pytest.raises(ValueError,match='timestamp'):
        prepared.body(sent)


def test_prepare_validates_trace_identifier():
    with pytest.raises(ValueError,match='correlation'):
        transport.prepare_probe({},'not-a-trace-id',DOMAIN)


@pytest.mark.parametrize('identifier,domain,message',[
    ('b'*32,DOMAIN,'identifier'),
    (IDENTIFIER,'another-clock','domain'),
])
def test_probe_rejects_prepared_identity_or_domain_mismatch(identifier,domain,message):
    async def run():
        def unexpected(request):
            pytest.fail('A mismatched prepared offer must not be sent')
        prepared=transport.prepare_probe({},IDENTIFIER,DOMAIN)
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as http:
            with pytest.raises(ValueError,match=message):
                await transport.probe(http,'http://test',prepared,identifier,0,domain=domain)
    asyncio.run(run())


@pytest.mark.parametrize('prepared',[False,True])
def test_probe_posts_content_and_headers_using_actual_sent_timestamp(monkeypatch,prepared):
    async def run():
        source=payload()
        item=transport.prepare_probe(source,IDENTIFIER,DOMAIN) if prepared else source
        ready=asyncio.Event()
        ticks=iter([9.,10.,10.1,10.2,10.3,10.4])
        requests=[];observed_clocks=[]
        canonical=transport.canonical
        def encode(value):
            if isinstance(value,dict):assert not observed_clocks, 'Static encoding entered the timed path'
            return canonical(value)
        def clock():
            value=next(ticks);observed_clocks.append(value)
            return value
        monkeypatch.setattr(transport,'canonical',encode)
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                assert not ready.is_set()
                yield b'data: {"choices":[{"text":"hello"}]}\n\n'
                assert ready.is_set()
                yield b'data: [DONE]\n\n'
        def handle(request):
            requests.append(request)
            assert request.method=='POST'
            assert request.headers['content-type']=='application/json'
            assert request.headers['x-benchmark-request-id']==IDENTIFIER
            assert decode(request.content)==expected(source,10.)
            return httpx.Response(200,stream=Body(),headers={
                'content-type':'text/event-stream','x-request-id':IDENTIFIER})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            initial={'id':'sentinel'}
            row=await transport.probe(http,'http://test',item,IDENTIFIER,8.,domain=DOMAIN,
                clock=clock,row=initial,ready=ready)
        assert row is initial and len(requests)==1
        assert row['scheduled']==8. and row['sent']==10.
        assert row['first']==10.2 and row['finished']==10.4
        assert row['status']=='complete' and row['stream_complete']
        assert source==payload()
    asyncio.run(run())


@pytest.mark.parametrize('status,expected_status',[(503,'rejected'),(500,'failed')])
def test_prepared_probe_preserves_rejection_rows(status,expected_status):
    async def run():
        ready=asyncio.Event()
        def handle(request):return httpx.Response(status)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            row=await transport.probe(http,'http://test',
                transport.prepare_probe({},IDENTIFIER,DOMAIN),IDENTIFIER,0.,domain=DOMAIN,ready=ready)
        assert row['status']==expected_status and row['http_status']==status
        assert not row['stream_complete'] and row['first'] is None and not ready.is_set()
        assert row['finished']>=row['sent']
    asyncio.run(run())


@pytest.mark.parametrize('fault',['read-error','content-type','response-id','body-size','deadline'])
def test_prepared_probe_keeps_stream_failure_rows_and_bounds(fault):
    async def run():
        attempts=[];now=NS(value=10.)
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                if fault=='read-error':raise httpx.ReadError('stream failed')
                if fault=='deadline':now.value+=46
                if fault=='body-size':yield b'x'*(8*1024*1024+1)
                else:yield b'data: {"choices":[{"text":"hello"}]}\n\ndata: [DONE]\n\n'
        def handle(request):
            attempts.append(request)
            return httpx.Response(200,stream=Body(),headers={
                'content-type':'text/plain' if fault=='content-type' else 'text/event-stream',
                'x-request-id':'b'*32 if fault=='response-id' else IDENTIFIER})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            row=await transport.probe(http,'http://test',transport.prepare_probe({},IDENTIFIER,DOMAIN),
                IDENTIFIER,0.,domain=DOMAIN,clock=lambda:now.value)
        assert row['status']=='failed' and len(attempts)==1
        assert row['finished']>=row['sent']
        if fault!='response-id':
            assert not row['stream_complete'] and row['first'] is None
            assert row['error']==('ReadError' if fault=='read-error' else 'ValueError')
    asyncio.run(run())


def test_prepared_probe_preserves_cancelled_rows():
    async def run():
        entered=asyncio.Event()
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b''
        def handle(request):
            return httpx.Response(200,stream=Body(),headers={'content-type':'text/event-stream'})
        row={};ready=asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            task=asyncio.create_task(transport.probe(http,'http://test',
                transport.prepare_probe({},IDENTIFIER,DOMAIN),IDENTIFIER,0.,domain=DOMAIN,row=row,ready=ready))
            await asyncio.wait_for(entered.wait(),1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
        assert row['status']=='cancelled' and row['finished']>=row['sent']
        assert not row['stream_complete'] and not ready.is_set()
    asyncio.run(run())


@pytest.mark.parametrize('background',[False,True])
def test_all_hashes_and_wire_preparation_precede_offers_and_release(monkeypatch,background):
    specs=[
        dict(id='foreground-1',role='foreground',token_key='long',max_tokens=2,at=.002),
        dict(id='sentinel',role='sentinel',token_key='short',max_tokens=1,at=.006),
        dict(id='foreground-2',role='foreground',token_key='long',max_tokens=2,at=.012),
    ]
    if background:specs.insert(0,dict(id='background',role='background',token_key='long',max_tokens=4,at=0))
    manifest=dict(model='test',pools=dict(long=dict(input_ids=list(range(7168))),
        short=dict(input_ids=[1,2])),trials={'trial':specs})
    now=NS(value=100.)
    history=[];saved=[];prepared=[];offered=[]
    digest=generator.digest
    prepare=transport.prepare_probe
    def slow_digest(value):
        history.append('hash');now.value+=.025
        return digest(value)
    def slow_prepare(source,identifier,domain):
        history.append('prepare');now.value+=.05
        item=prepare(source,identifier,domain);prepared.append(item)
        return item
    async def offer(http,url,item,identifier,scheduled,*,domain,row,ready=None):
        assert isinstance(item,transport.PreparedProbe)
        assert len(prepared)==len(specs)
        assert history.count('hash')==2*len(specs)
        history.append('offer');offered.append((row.copy(),item,scheduled))
        row.update(status='complete')
        if ready:
            now.value+=.125
            ready.set()
    monkeypatch.setattr(generator,'clock_domain',lambda:DOMAIN)
    monkeypatch.setattr(generator,'time',NS(monotonic=lambda:now.value))
    monkeypatch.setattr(generator,'digest',slow_digest)
    monkeypatch.setattr(generator,'prepare_probe',slow_prepare)
    monkeypatch.setattr(generator,'probe',offer)
    monkeypatch.setattr(generator,'protocol',lambda:dict(trial_seconds=45))
    monkeypatch.setattr(generator,'save',lambda output,data:saved.append(data))
    asyncio.run(generator.run(None,'http://test',manifest,'trial','unused.json'))
    preparation_finished=100.+len(specs)*.1
    release=preparation_finished+(.125 if background else 0)
    assert saved[0]['release']==pytest.approx(release)
    assert all(step!='offer' for step in history[:3*len(specs)])
    assert len(offered)==len(specs)
    for row,item,scheduled in offered:
        spec=next(r for r in specs if r['id']==row['id'])
        assert row['input_ids_sha256']==digest(manifest['pools'][spec['token_key']]['input_ids'])
        body=decode(item.body(101.))
        assert body['prompt']==manifest['pools'][spec['token_key']]['input_ids']
        assert item.identifier==digest(['live-http','trial',spec['id']])[:32]
        assert type(body['vllm_xargs']['prefill_lab_probe']) is int
        assert body['vllm_xargs']['prefill_lab_probe']==int(spec['role']=='sentinel')
        assert scheduled==pytest.approx(preparation_finished if spec['role']=='background' else release+spec['at'])


def test_background_ready_barrier_and_foreground_offsets_are_preserved(monkeypatch):
    async def run():
        specs=[dict(id='background-1',role='background',token_key='pool',max_tokens=4,at=0),
            dict(id='background-2',role='background',token_key='pool',max_tokens=4,at=0),
            dict(id='foreground',role='foreground',token_key='pool',max_tokens=1,at=.004),
            dict(id='sentinel',role='sentinel',token_key='pool',max_tokens=1,at=.008)]
        manifest=dict(model='test',pools={'pool':dict(input_ids=[1,2])},trials={'trial':specs})
        started={r['id']:asyncio.Event() for r in specs}
        allow={r['id']:asyncio.Event() for r in specs if r['role']=='background'}
        sent_ready={key:asyncio.Event() for key in allow}
        finish=asyncio.Event();saved=[];scheduled={};cancelled=[]
        async def offer(http,url,item,identifier,at,*,domain,row,ready=None):
            key=row['id'];scheduled[key]=at;started[key].set()
            try:
                if ready:
                    await allow[key].wait()
                    ready.set();sent_ready[key].set()
                await finish.wait()
                row['status']='complete'
            except asyncio.CancelledError:
                cancelled.append(key);raise
        monkeypatch.setattr(generator,'clock_domain',lambda:DOMAIN)
        monkeypatch.setattr(generator,'probe',offer)
        monkeypatch.setattr(generator,'protocol',lambda:dict(trial_seconds=45))
        monkeypatch.setattr(generator,'save',lambda output,data:saved.append(data))
        task=asyncio.create_task(generator.run(None,'http://test',manifest,'trial','unused.json'))
        try:
            await asyncio.wait_for(asyncio.gather(*(started[key].wait() for key in allow)),1)
            await asyncio.sleep(0)
            assert not started['foreground'].is_set() and not started['sentinel'].is_set()
            allow['background-1'].set()
            await asyncio.wait_for(sent_ready['background-1'].wait(),1)
            await asyncio.sleep(0)
            assert not started['foreground'].is_set() and not started['sentinel'].is_set()
            allow['background-2'].set()
            await asyncio.wait_for(asyncio.gather(started['foreground'].wait(),started['sentinel'].wait()),1)
            finish.set()
            await asyncio.wait_for(task,1)
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        assert not cancelled
        assert scheduled['foreground']==saved[0]['release']+.004
        assert scheduled['sentinel']==saved[0]['release']+.008
        assert all(scheduled[key]<=saved[0]['release'] for key in allow)
    asyncio.run(run())
