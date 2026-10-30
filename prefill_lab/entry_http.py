"""Fresh controlled confirmation then frozen pre-ADD HTTP qualification."""
import asyncio
import copy
import json
from pathlib import Path
import shutil
import time
from . import queue_confirmation as qc,live_http,live_experiment as old,entry_http_model
from .contracts import digest,save

ROOT=Path(__file__).resolve().parents[1]


def protocol():
    spec=json.loads((ROOT/'experiments/prefill-entry-http-v1.json').read_text());base=live_http.protocol()
    if spec['base_http_plan_sha256']!=digest(base) or spec['base_confirmation_plan_sha256']!=digest(qc.protocol()):raise ValueError('Entry HTTP base protocols changed')
    p=copy.deepcopy(base);p.pop('base_clock_plan_sha256',None)
    for i,t in enumerate(p['order']):
        t['trial_id']=f'eh-{i+1:03d}';t['paired_content_key']='entry-http-v1-'+t['paired_content_key']
    p.update(schema='entry-http-plan-v1',scope=spec['scope'],base_confirmation_plan_sha256=spec['base_confirmation_plan_sha256'],policy_sha256=digest(spec),hard_seconds=spec['hard_seconds'])
    if len(p['order'])!=98:raise ValueError('Entry HTTP denominator changed')
    if digest(p)!=digest(json.loads((ROOT/'experiments/prefill-entry-http-v1.plan.json').read_text())):raise ValueError('Entry HTTP numeric plan changed')
    return p


def validate_manifest(manifest):
    from .tokens import validate
    p=protocol()
    if (manifest['model'],manifest['revision'])!=(p['model']['name'],p['model']['revision']):raise ValueError('Entry HTTP model/revision differs')
    validate(manifest,p['cases'],p['order'])
    for t in p['order']:
        if any(r['token_key']!=f'pair:{t["paired_content_key"]}:{r["id"]}:{r["input_tokens"]}' for r in manifest['trials'][t['trial_id']]):
            raise ValueError('Entry HTTP token namespace differs')
    return digest(manifest)


def extract(bundle,trial,manifest,runtime,config,*,model=None):
    model=entry_http_model.costs(config,runtime) if model is None else model
    if bundle['capture']['live'].get('configuration_sha256')!=config['sha256']:raise ValueError('HTTP native configuration identity differs')
    row=old.extract(bundle,trial,manifest,runtime,config['base_model'],plan=protocol(),model=model,manifest_validator=validate_manifest)
    row.update(f0_predicted_engine_s=None,f0_unavailable_reason=None)
    if row['forecast_available']:
        from .async_predictor import Request
        from .live_predictor import forecast
        from scripts.predict_prefill_async import runtime_inputs
        scope,dispatch=runtime_inputs(runtime);record=next(r for r in bundle['capture']['live']['records'] if r['trace_id']==row['trace_id'])
        request=bundle['capture']['requests'][record['engine_id']]
        try:
            f0=handoff_f0(config)
            row['f0_predicted_engine_s']=forecast(record['snapshot'],Request(record['engine_id'],request['input_tokens'],request['max_tokens']),f0,dispatch,scope=scope)['ttft_from_cut_s']
        except (ValueError,RuntimeError) as exc:row['f0_unavailable_reason']=str(exc)
    return row


def handoff_f0(config):
    return entry_http_model.clock.costs(config['base_model']['parameters_s'],qc.protocol(),entry_http_model.clock.specification())


def score(rows,p,transport):
    from .contracts import finite
    report=live_http.score(rows,p,transport);checks=[]
    for r in rows:
        if r['stage']=='warmup' or r['observer']!='light':continue
        f0=r.get('f0_predicted_engine_s')
        for metric,observed,estimate in (('engine',r['engine_from_cut_s'],f0),('http',r['client_ttft_s'],None if not finite(f0) else r['elapsed_to_cut_s']+f0+transport['egress_s'])):
            limit=max(p['gates'][metric+'_absolute_s'],p['gates']['relative_error']*observed)
            checks.append(dict(trial_id=r['trial_id'],stage=r['stage'],case=r['case'],metric=metric,observed_s=observed,predicted_s=estimate,
                passed=bool(finite(estimate) and abs(estimate-observed)<=limit),limit_s=limit))
    report.update(schema='entry-http-qualification-v1',scope=p['scope'],f0_snapshot_ablation=checks,
        ablation_scope='offline replay of identical observed LIGHT snapshots; no extra online observer overhead or held-dependent selection')
    return report


async def run(client,http,url,controlled_manifest,http_manifest,output,gateway_url,deadline):
    from .suite import finish_external
    from .causal import dispatcher_parity
    from scripts.run_prefill_service import code_identity
    output=Path(output);p=protocol();validate_manifest(http_manifest);qc.validate_manifest(controlled_manifest);old.validate_pair(controlled_manifest,http_manifest)
    started=time.monotonic();rows=[];active=None;transport=None;sources=code_identity()
    summary=dict(collection_complete=False,candidate_passed=False,completed=[],failures=[],scope=p['scope'],**{k:False for k in qc.ACCEPTANCE})
    save(output/'entry-http-plan.json',p);save(output/'http-token-manifest.json',http_manifest)
    directory=output/'bridge';directory.mkdir();controlled=output/'controlled';controlled.mkdir()
    # The sub-audit records the actual combined launch modes, not fabricated
    # standalone flags. It has a separate explicit parent-scope audit path.
    for name in ('launch.json','collection-sources.json','token-manifest.json'):
        shutil.copyfile(output/name,controlled/name)
    shutil.copytree(output/'native-cpu-acceptance',controlled/'native-cpu-acceptance')
    try:
        prerequisite=await asyncio.wait_for(qc.run(client,http,url,controlled_manifest,controlled),max(.001,deadline-time.monotonic()))
        if not prerequisite['candidate_passed']:raise ValueError('Controlled confirmation failed/inconclusive; HTTP phase not attempted')
        load=lambda n:json.loads((controlled/n).read_text())
        runtime=load('runtime.json');base=load('base-model.json');entry=load('entry-freeze.json');report=load('confirmation-report.json')
        final_trial=load(qc.protocol()['order'][-1]['trial_id']+'.json')
        config=entry_http_model.configure(base,entry,report,runtime,finished_at=final_trial['end'],http_manifest_sha256=digest(http_manifest))
        save(output/'entry-http-config.json',config);model=entry_http_model.costs(config,runtime)
        answer=await client.command('live_configure',frozen=config,runtime=runtime)
        if answer.get('frozen_sha256')!=config['sha256']:raise ValueError('Native entry HTTP configuration acknowledgement differs')
        save(output/'entry-http-configured.json',answer)
        for t in p['order']:
            if time.monotonic()>=deadline:raise TimeoutError('Entry HTTP combined deadline')
            if code_identity()!=sources or digest(json.loads((output/'entry-http-config.json').read_text()))!=digest(config):raise ValueError('Entry HTTP source/model changed during collection')
            if t['stage']=='train' and not any(r['stage']=='train' for r in rows):live_http.validate_warmup(rows,p)
            if t['stage'] not in ('train','warmup') and transport is None:
                transport=live_http.fit_transport(rows,p);save(output/'transport-freeze.json',transport)
                save(output/'transport-seal.json',dict(at_monotonic=time.monotonic(),next_trial=t['trial_id'],completed=list(summary['completed']),
                    transport_sha256=transport['sha256'],config_sha256=config['sha256'],source_sha256=sources))
            tid=t['trial_id'];print(f'Entry HTTP {len(rows)+1}/98: {tid} {t["stage"]} {t["observer"]}',flush=True)
            await client.command('external_begin',trial_id=tid,observer=t['observer']);active=tid
            received=await old.isolated_trial(output/'http-token-manifest.json',tid,gateway_url+'/v1/completions',directory/(tid+'-client.json'),deadline)
            capture=await finish_external(client,min(deadline,time.monotonic()+5));active=None
            bundle=dict(capture=capture,client=received,gateway=[]);save(directory/(tid+'.json'),bundle)
            if t['observer']=='light' and not dispatcher_parity(capture,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('Entry HTTP dispatcher mismatch')
            until=min(deadline,time.monotonic()+5)
            while True:
                try:bundle['gateway']=old.gateway_records(output/'gateway.jsonl',{r['trace_id'] for r in received['rows']});break
                except (ValueError,FileNotFoundError):
                    if time.monotonic()>=until:raise
                    await asyncio.sleep(.05)
            save(directory/(tid+'.json'),bundle);rows.append(extract(bundle,t,http_manifest,runtime,config,model=model));save(output/'http-rows.json',rows)
            summary['completed'].append(tid);save(output/'entry-http-summary.json',summary)
        final=await client.command('meta');save(output/'http-runtime-final.json',final)
        if digest(final)!=digest(runtime):raise ValueError('Runtime drift during entry HTTP qualification')
        metrics=(await http.get(gateway_url+'/metrics')).json();save(output/'gateway-final.json',metrics);live_http.gateway_drained(metrics,p)
        if live_http.gateway_contract(metrics)!=live_http.gateway_contract(json.loads((output/'gateway-start.json').read_text())):raise ValueError('Gateway run identity drift')
        result=score(rows,p,transport);save(output/'entry-http-report.json',result)
        summary.update(collection_complete=True,candidate_passed=result['candidate_passed'],
            rejection_reasons=[dict(gate='prediction',**r) for r in result['checks'] if not r['passed']]+[dict(gate='observer',**r) for r in result['observer_checks'] if not r['passed']])
    except BaseException as exc:
        summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        if active is not None:
            try:
                await client.command('abort',reason='Entry HTTP failed or cancelled')
                partial=await finish_external(client,time.monotonic()+2);save(directory/(active+'-partial.json'),partial)
            except BaseException as cleanup:summary['cleanup_error']=type(cleanup).__name__+': '+str(cleanup)
        raise
    finally:summary['elapsed_seconds']=time.monotonic()-started;save(output/'entry-http-summary.json',summary)
    return summary
