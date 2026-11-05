"""Clock-safe HTTP probe, joins and training-only transport correction."""
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
import numpy as np

from telemetry import SSEObserver
from .contracts import canonical,digest,finite

ROOT=Path(__file__).resolve().parents[1]


def protocol():return json.loads((ROOT/'experiments/prefill-live-http-v1.json').read_text())


def clock_domain():
    # Monotonic values may only be subtracted within the same boot/time namespace.
    return 'linux-monotonic:'+Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+os.readlink('/proc/self/ns/time')


def trace_id(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{32}',value):raise ValueError('Invalid bridge correlation ID')
    return value


@dataclass(frozen=True)
class PreparedProbe:
    """Immutable static JSON prefix; only the actual send timestamp remains."""
    identifier: str
    domain: str
    prefix: bytes

    def body(self,sent):
        if not finite(sent):raise ValueError('Invalid client sent timestamp')
        return self.prefix+canonical(sent)+b'}}'


def prepare_probe(payload,identifier,domain):
    """Serialize large prompts before offers, without timestamp placeholders.

    Injected IDs make both objects nonempty. Remove only their final braces,
    then append one fresh client timestamp and close both objects at send time.
    """
    identifier=trace_id(identifier)
    outer=dict(payload,request_id=identifier)
    xargs=dict(outer.pop('vllm_xargs',{}),prefill_lab_trace_id=identifier,prefill_lab_clock_domain=domain)
    xargs.pop('prefill_lab_client_sent',None)
    prefix=canonical(outer)[:-1]+b',"vllm_xargs":'+canonical(xargs)[:-1]+b',"prefill_lab_client_sent":'
    return PreparedProbe(identifier,domain,prefix)


async def probe(http,url,payload,identifier,scheduled,*,domain=None,clock=time.monotonic,row=None,ready=None):
    identifier=trace_id(identifier);domain=domain or clock_domain()
    if isinstance(payload,PreparedProbe):
        if payload.identifier!=identifier:raise ValueError('Prepared probe identifier mismatch')
        if payload.domain!=domain:raise ValueError('Prepared probe clock domain mismatch')
        prepared=payload
    else:prepared=prepare_probe(payload,identifier,domain)
    await asyncio.sleep(max(0.,scheduled-clock()));sent=clock()
    row={} if row is None else row
    row.update(trace_id=identifier,clock_domain=domain,scheduled=scheduled,sent=sent,first=None,finished=None,
             stream_complete=False,status='attempted');observer=SSEObserver(completions=True);size=0
    try:
        async with http.stream('POST',url,content=prepared.body(sent),
                headers={'x-benchmark-request-id':identifier,'content-type':'application/json'}) as response:
            row['http_status']=response.status_code;row['response_trace_id']=response.headers.get('x-request-id')
            if response.status_code!=200:row['status']='rejected' if response.status_code==503 else 'failed';return row
            if response.headers.get('content-type','').split(';')[0]!='text/event-stream':raise ValueError('Expected SSE')
            async for chunk in response.aiter_bytes():
                size+=len(chunk)
                if size>8*1024*1024 or clock()-sent>45:raise ValueError('HTTP body/deadline bound')
                observer.feed(chunk)
                if observer.has_content and row['first'] is None:
                    row['first']=clock()
                    if ready:ready.set()
            row['stream_complete']=observer.done and not observer.invalid and not observer.buffer.strip() and not observer.data
            row['status']='complete' if row['stream_complete'] and row['first'] is not None and row['response_trace_id']==identifier else 'failed'
    except asyncio.CancelledError:row['status']='cancelled';raise
    except Exception as exc:row.update(status='failed',error=type(exc).__name__)
    finally:row['finished']=clock()
    return row


def join(client,gateway,record,engine):
    """All operands are same-domain causal milestones; no wall-time subtraction."""
    tid=trace_id(client['trace_id'])
    if record['trace_id']!=tid or record['client_sent']!=client['sent'] or client['status']!='complete' or not client['stream_complete']:
        raise ValueError('Incomplete/mismatched HTTP outcome')
    events=[r for r in gateway if r.get('request_id')==tid];by={}
    for r in events:
        if r['event'] in ('received','accepted','backend_sent','first_content','completed','bridge_clock'):
            if r['event'] in by:raise ValueError('Duplicate gateway milestone')
            by[r['event']]=r
    if set(by)!={'received','accepted','backend_sent','first_content','completed','bridge_clock'}:
        raise ValueError('Missing gateway milestones')
    if len({r['run_id'] for r in by.values()})!=1:raise ValueError('Gateway restarted/mixed runs')
    domain=client['clock_domain']
    if not domain or by['bridge_clock']['clock_domain']!=domain or record['clock_domain']!=domain:
        raise ValueError('Clock domains differ')
    if engine.get('trace_id')!=tid or not finite(engine.get('first')) or not finite(engine.get('finished')):
        raise ValueError('Missing native output identity')
    if record['status']=='predicted' and (record['snapshot']['observed_at']!=record['cut']
            or record['forecast']['state_sha256']!=digest(record['snapshot'])
            or not record['cut']<=record['prediction_finished']<=record['engine_enqueue']):
        raise ValueError('Forecast was not sealed before ADD')
    times=[client['sent'],by['received']['monotonic'],by['accepted']['monotonic'],by['backend_sent']['monotonic'],
           record['cut'],record['engine_enqueue'],engine['first'],by['first_content']['monotonic'],client['first'],client['finished']]
    if not all(finite(v) for v in times) or times!=sorted(times) or engine['finished']>client['finished']:
        raise ValueError('Clock chronology invalid')
    if by['completed']['monotonic']<by['first_content']['monotonic']:raise ValueError('Invalid gateway completion')
    sent,received,admitted,backend,cut,enqueued,first,gfirst,cfirst,done=times
    parts=dict(client_to_gateway_s=received-sent,gateway_before_backend_s=backend-received,
               backend_to_engine_s=enqueued-backend,engine_s=first-enqueued,egress_s=cfirst-first)
    return dict(trace_id=tid,client_ttft_s=cfirst-sent,engine_from_cut_s=first-cut,
        elapsed_to_cut_s=cut-sent,egress_s=cfirst-first,parts=parts,clock_domain=domain)


def validate_warmup(rows,plan):
    wanted={t['trial_id'] for t in plan['order'] if t['stage']=='warmup'}
    found=[r for r in rows if r['stage']=='warmup']
    if (not wanted or len(found)!=len(wanted) or {r['trial_id'] for r in found}!=wanted
            or any(r.get('forecast_available') is not True or r.get('structure_matches') is not True for r in found)):
        raise ValueError('Live warmup state/debt coverage failed; preserve diagnostics before full HTTP qualification')
    return True


def fit_transport(rows,plan):
    wanted={t['trial_id'] for t in plan['order'] if t['stage']=='train' and t['observer']=='off'}
    found={}
    for r in rows:
        if r['trial_id'] not in wanted:continue
        if r['trial_id'] in found:raise ValueError('Duplicate transport training trial')
        if r['stage']!='train' or r['observer']!='off' or not finite(r['egress_s']) or r['egress_s']<0:
            raise ValueError('Invalid transport training label')
        found[r['trial_id']]=r
    if set(found)!=wanted:raise ValueError('Incomplete transport training split')
    values=[found[k]['egress_s'] for k in sorted(found)];median=float(np.median(values))
    if max(values)-min(values)>plan['gates']['transport_spread_s']:raise ValueError('Transport training spread too large')
    frozen=dict(schema='live-http-transport-freeze-v1',egress_s=median,plan_sha256=digest(plan),
        training_sha256=digest([found[k] for k in sorted(found)]),held_labels_used=False,model_accepted=False)
    frozen['sha256']=digest(frozen);return frozen


def score(rows,plan,frozen):
    if frozen['sha256']!=digest({k:v for k,v in frozen.items() if k!='sha256'}) or frozen['plan_sha256']!=digest(plan):
        raise ValueError('Transport freeze changed')
    wanted={t['trial_id']:t for t in plan['order'] if t['stage']!='warmup'};found={}
    for r in rows:
        tid=r['trial_id']
        if tid not in wanted:continue
        if tid in found or any(r[k]!=wanted[tid][k] for k in ('stage','case','observer','block')):raise ValueError('Bridge split membership mismatch')
        found[tid]=r
    if set(found)!=set(wanted):raise ValueError('Incomplete bridge evidence')
    # Rebind the freeze to exact training inputs; unrelated held labels cannot refit it.
    if fit_transport(list(found.values()),plan)!=frozen:raise ValueError('Transport training identity changed')
    checks=[];quality=[];groups={}
    for tid,r in found.items():
        if r['observer']=='off':continue
        t=wanted[tid];pair=[v for v in found.values() if v['stage']==r['stage'] and v['case']==r['case'] and v['block']==r['block'] and v['observer']=='off']
        if len(pair)!=1:raise ValueError('Missing OFF/LIGHT pair')
        delta=abs(r['client_ttft_s']-pair[0]['client_ttft_s'])
        quality.append(dict(trial_id=tid,delta_s=delta,passed=delta<=max(plan['gates']['observer_absolute_s'],plan['gates']['observer_relative']*pair[0]['client_ttft_s'])))
        valid=r.get('forecast_available') is True and r.get('structure_matches') is True
        valid=valid and all(finite(r.get(k)) and 0<=r[k]<=plan['gates'][k] for k in ('snapshot_width_s','prediction_age_s'))
        for metric,observed,predicted,floor in (
            ('engine',r['engine_from_cut_s'],r.get('predicted_engine_s'),plan['gates']['engine_absolute_s']),
            ('http',r['client_ttft_s'],None if not finite(r.get('predicted_engine_s')) else r['elapsed_to_cut_s']+r['predicted_engine_s']+frozen['egress_s'],plan['gates']['http_absolute_s'])):
            limit=max(floor,plan['gates']['relative_error']*observed)
            checks.append(dict(trial_id=tid,case=t['case'],stage=t['stage'],metric=metric,observed_s=observed,predicted_s=predicted,
                limit_s=limit,passed=bool(valid and finite(predicted) and abs(predicted-observed)<=limit)))
        groups.setdefault(r['case'],[]).append(r)
    return dict(schema='live-http-qualification-v1',candidate_passed=bool(checks) and all(r['passed'] for r in checks+quality),
        checks=checks,observer_checks=quality,transport_sha256=frozen['sha256'],
        unavailable_predictions=sum(not r.get('forecast_available',False) for r in found.values() if r['observer']=='light'),
        scope='post-arrival EngineCore pre-ADD prediction; elapsed HTTP path is known, future egress uses TRAIN-only correction',
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


def gateway_contract(metrics):
    expected=dict(admission_mode='protected',max_in_flight=32,max_waiting_capacity=0,
                  shared_upstream_client=True,prefill_raw_completions=True,gateway_diagnostics=True,
                  telemetry_dropped_records=0,telemetry_write_errors=0)
    if any(type(metrics.get(k)) is not type(v) or metrics[k]!=v for k,v in expected.items()):
        raise ValueError('Gateway does not match the frozen fixed32 bridge')
    if not isinstance(metrics.get('telemetry_run_id'),str) or not metrics['telemetry_run_id']:
        raise ValueError('Missing gateway run identity')
    return metrics['telemetry_run_id']


def expected_requests(plan):
    cases={c['id']:c for c in plan['cases']}
    return sum(cases[t['case']]['background_decoders']+len(cases[t['case']]['foreground_tokens'])+1 for t in plan['order'])


def gateway_drained(metrics,plan):
    if (metrics.get('accepted_requests')!=expected_requests(plan) or any(metrics.get(k)!=0 for k in
            ('rejected_requests','failed_requests','current_in_flight','current_waiting'))):
        raise ValueError('Gateway request totals/drain differ from the complete frozen offers')
