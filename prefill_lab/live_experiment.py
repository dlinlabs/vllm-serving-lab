"""Frozen HTTP/live-state qualification; same-host clock pilot precedes it."""
import asyncio
import json
import os
from pathlib import Path
import sys
import httpx
import time

from . import event_clock_calibration as clock
from .contracts import audit_capture,digest,save
from .live_http import protocol,join,fit_transport,score,gateway_contract,gateway_drained,validate_warmup
from .live_predictor import forecast
from .async_predictor import Request
from scripts.predict_prefill_async import runtime_inputs


def validate_manifest(manifest):
    from .tokens import validate
    p=protocol()
    if p['base_clock_plan_sha256']!=digest(clock.protocol()):raise ValueError('Base clock plan changed')
    if (manifest['model'],manifest['revision'])!=(p['model']['name'],p['model']['revision']):raise ValueError('Live HTTP model differs')
    return validate(manifest,p['cases'],p['order'])


def extract(bundle,trial,manifest,runtime,frozen):
    p=protocol();validate_manifest(manifest);capture=bundle['capture'];client=bundle['client'];gateway=bundle['gateway']
    workers=[r for w in capture['workers'] for r in w['rows']]
    audited=audit_capture(capture,workers,require_steps=trial['observer']=='light')
    live=capture['live']
    if (not audited['valid'] or capture.get('pending_step_ids') or live['errors'] or live['overflow']
            or capture['trial_id']!=trial['trial_id'] or capture['observer']!=trial['observer']
            or live['observed']!=(trial['observer']=='light')):raise ValueError('Invalid native live HTTP capture')
    specs=manifest['trials'][trial['trial_id']];rows=client['rows'];by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!={r['id'] for r in specs}:raise ValueError('HTTP request inventory differs')
    native={r['trace_id']:(rid,r) for rid,r in capture['requests'].items()}
    observed={r['trace_id']:r for r in live['records']}
    expected={digest(['live-http',trial['trial_id'],r['id']])[:32] for r in specs}
    if (set(native)!=expected or set(observed)!=expected or len(native)!=len(capture['requests'])
            or len(observed)!=len(live['records'])):raise ValueError('Native correlation inventory differs')
    scope,dispatch=runtime_inputs(runtime);model=clock.costs(frozen['parameters_s'],clock.protocol(),clock.specification())
    if live['cost_sha256']!=digest(model.to_dict()) or live['dispatcher_sha256']!=digest(dispatch):raise ValueError('Live model identity changed')
    for spec in specs:
        row=by_id[spec['id']];tid=digest(['live-http',trial['trial_id'],spec['id']])[:32];rid,native_row=native[tid]
        if (row['trace_id']!=tid or row['input_ids_sha256']!=manifest['pools'][spec['token_key']]['sha256']
                or native_row['input_ids_sha256']!=row['input_ids_sha256'] or native_row['input_tokens']!=spec['input_tokens']
                or native_row['max_tokens']!=spec['max_tokens'] or native_row['role']!=spec['role']):raise ValueError('Exact native HTTP inputs differ')
        if row['sent']<row['scheduled'] or row['sent']-row['scheduled']>p['posting_lag_s']:raise ValueError('HTTP posting lag exceeds plan')
        if spec['role']!='background' and abs(row['scheduled']-(client['release']+spec['at']))>1e-9:raise ValueError('HTTP target differs')
        if observed[tid]['engine_id']!=rid:raise ValueError('Engine correlation differs')
        join(row,gateway,observed[tid],native_row)
    foreground_done=max(r['finished'] for r in capture['requests'].values() if r['role']!='background')
    if any(r['finished']<foreground_done for r in capture['requests'].values() if r['role']=='background'):raise ValueError('HTTP backgrounds finished early')
    probe_spec=next(r for r in specs if r['role']=='sentinel');c=by_id[probe_spec['id']];rid,n=native[c['trace_id']];record=observed[c['trace_id']]
    result=join(c,gateway,record,n)
    result.update(**{k:trial[k] for k in ('trial_id','stage','case','block','observer')},forecast_available=False,structure_matches=False,
                  capture_sha256=digest(bundle),snapshot_width_s=record.get('snapshot_width_s'),prediction_age_s=None,predicted_engine_s=None)
    if trial['observer']=='light' and record['status']=='predicted':
        if any(r['enqueue']>record['cut'] for key,r in capture['requests'].items() if key!=rid):raise ValueError('Unmodelled arrivals after the snapshot')
        regenerated=forecast(record['snapshot'],Request(rid,n['input_tokens'],n['max_tokens']),model,dispatch,scope=scope)
        if digest(regenerated)!=digest(record['forecast']):raise ValueError('Live forecast differs from raw snapshot')
        actual=[s['work'] for s in capture['steps'] if record['engine_enqueue']<=s['start']<=n['first']]
        short=lambda work:[(w['request_id'],w['position'],w['tokens']) for w in work]
        structure=[short(s['work']) for s in regenerated['steps']]==[short(w) for w in actual]
        result.update(forecast_available=True,structure_matches=structure,predicted_engine_s=regenerated['ttft_from_cut_s'],
            prediction_age_s=record['engine_enqueue']-record['cut'])
    return result


def gateway_records(path,ids):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    found={r.get('request_id') for r in rows if r.get('event')=='completed'}
    if not ids<=found:raise ValueError('Gateway journal not fully flushed')
    return [r for r in rows if r.get('request_id') in ids]


async def isolated_trial(manifest_path,tid,url,output,deadline):
    output=Path(output);log=output.with_suffix('.log').open('wb');process=None
    try:
        process=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/prefill_live_http_trial.py'),
            '--manifest',str(manifest_path),'--trial',tid,'--url',url,'--output',str(output),stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        await asyncio.wait_for(process.wait(),min(50,max(.001,deadline-time.monotonic())))
        if process.returncode:raise RuntimeError('Isolated HTTP generator failed; see '+str(output.with_suffix('.log')))
        return json.loads(output.read_text())
    finally:
        if process and process.returncode is None:
            process.terminate()
            try:await asyncio.wait_for(process.wait(),2)
            except asyncio.TimeoutError:process.kill();await process.wait()
        log.close()


async def run(client,http,url,clock_manifest,http_manifest,output,gateway_url,deadline):
    from .delayed_calibration import run as clock_run
    from .suite import finish_external
    from .causal import dispatcher_parity
    output=Path(output);p=protocol();validate_manifest(http_manifest);validate_pair(clock_manifest,http_manifest);start=time.monotonic();rows=[]
    active_tid=None
    summary=dict(collection_complete=False,candidate_passed=False,model_accepted=False,completed=[],failures=[],scope=p['scope'])
    save(output/'bridge-plan.json',p);save(output/'http-token-manifest.json',http_manifest);directory=output/'bridge';directory.mkdir()
    try:
        calibration=await asyncio.wait_for(clock_run(client,http,url,clock_manifest,output,event_clock=True,shared_deadline=deadline),max(.001,deadline-time.monotonic()))
        if not calibration['candidate_passed']:raise ValueError('Same-host clock confirmation failed; HTTP phase not attempted')
        frozen=json.loads((output/'frozen.json').read_text());runtime=json.loads((output/'runtime.json').read_text())
        await client.command('live_configure',frozen=frozen,runtime=runtime);transport=None
        for t in p['order']:
            if time.monotonic()>=deadline:raise TimeoutError('Combined live HTTP deadline')
            if t['stage']=='train' and not any(r['stage']=='train' for r in rows):
                validate_warmup(rows,p)
            if t['stage'] not in ('warmup','train') and transport is None:
                transport=fit_transport(rows,p);save(output/'transport-freeze.json',transport)
            tid=t['trial_id'];print(f'Live HTTP {len(summary["completed"])+1}/{len(p["order"])}: {tid} {t["stage"]} {t["observer"]}',flush=True)
            await client.command('external_begin',trial_id=tid,observer=t['observer']);active_tid=tid
            received=await isolated_trial(output/'http-token-manifest.json',tid,gateway_url+'/v1/completions',directory/(tid+'-client.json'),deadline)
            capture=await finish_external(client,min(deadline,time.monotonic()+5));active_tid=None
            bundle=dict(capture=capture,client=received,gateway=[])
            save(directory/(tid+'.json'),bundle)
            if t['observer']=='light' and not dispatcher_parity(capture,runtime['workers'][0]['dispatch_table'])['matches']:raise ValueError('Native HTTP dispatcher differs')
            until=min(deadline,time.monotonic()+5)
            while True:
                try:bundle['gateway']=gateway_records(output/'gateway.jsonl',{r['trace_id'] for r in received['rows']});break
                except (ValueError,FileNotFoundError):
                    if time.monotonic()>until:raise
                    await asyncio.sleep(.05)
            save(directory/(tid+'.json'),bundle);rows.append(extract(bundle,t,http_manifest,runtime,frozen));save(output/'http-rows.json',rows)
            summary['completed'].append(tid);save(output/'bridge-summary.json',summary)
        if digest(await client.command('meta'))!=digest(runtime):raise ValueError('Runtime drift after HTTP qualification')
        gateway_final=(await http.get(gateway_url+'/metrics')).json();save(output/'gateway-final.json',gateway_final);gateway_drained(gateway_final,p)
        if gateway_contract(gateway_final)!=gateway_contract(json.loads((output/'gateway-start.json').read_text())):
            raise ValueError('Gateway identity drift')
        report=score(rows,p,transport);save(output/'bridge-report.json',report)
        if time.monotonic()>=deadline:raise TimeoutError('Live HTTP deadline during scoring')
        summary.update(collection_complete=True,candidate_passed=report['candidate_passed'],
            rejection_reasons=[dict(gate='prediction',**r) for r in report['checks'] if not r['passed']]
                +[dict(gate='observer',**r) for r in report['observer_checks'] if not r['passed']])
    except BaseException as exc:
        summary['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        if active_tid is not None:
            try:
                await client.command('abort',reason='Live HTTP trial failed or cancelled')
                until=time.monotonic()+2
                while True:
                    try:
                        partial=await client.command('external_end')
                        if partial.get('status')!='gpu_events_pending':
                            save(directory/(active_tid+'-partial.json'),partial);break
                    except httpx.HTTPStatusError:
                        if time.monotonic()>until:raise
                    if time.monotonic()>until:raise TimeoutError('Partial HTTP capture did not drain')
                    await asyncio.sleep(.01)
            except BaseException as cleanup:
                summary['cleanup_error']=type(cleanup).__name__+': '+str(cleanup)
        raise
    finally:
        summary['elapsed_seconds']=time.monotonic()-start;save(output/'bridge-summary.json',summary)
    return summary


def validate_pair(clock_manifest,http_manifest):
    for key in ('model','revision','tokenizer_sha256','tokenizer_files'):
        if key not in clock_manifest or clock_manifest[key]!=http_manifest.get(key):
            raise ValueError('Clock/HTTP input identities differ: '+key)
