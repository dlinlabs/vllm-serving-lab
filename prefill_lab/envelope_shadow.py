"""Bounded shadow measurements; no calibration, admission or acceptance."""
import asyncio
import copy
import json
from collections import Counter
from pathlib import Path
import time
import httpx
import numpy as np

from . import live_http, live_experiment, state_envelope, tokens
from .async_predictor import Request
from .contracts import audit_capture, digest, finite, save
from .suite import finish_external, runtime_preflight
from scripts.predict_prefill_async import runtime_inputs


def flags():
    return dict(actuation_authorized=False, native_coverage_qualified=False,
                online_compute_age_qualified=False, **{k: False for k in state_envelope.FLAGS})


def protocol():
    old = live_http.protocol()
    cases = [
        dict(id='empty', background_decoders=0, foreground_tokens=[], foreground_offsets_ms=[],
             sentinel=dict(input_tokens=512, arrival_ms=0)),
        dict(id='decode-D16', background_decoders=16, foreground_tokens=[], foreground_offsets_ms=[],
             sentinel=dict(input_tokens=512, arrival_ms=0)),
        *[copy.deepcopy(c) for c in old['cases'] if c['id'] in
          ('mixed-plus-sentinel-at50-D0', 'mixed-plus-sentinel-at50-D16')],
    ]
    order = []
    for stage, block in [('warmup', 0), ('shadow', 1), ('shadow', 2)]:
        for case in cases:
            arms = ('light',) if stage == 'warmup' else ('off', 'light') if block == 1 else ('light', 'off')
            for observer in arms:
                order.append(dict(trial_id=f'es-{len(order)+1:03d}', case=case['id'], stage=stage,
                                  block=block, observer=observer,
                                  paired_content_key=f'envelope-shadow-v1-{block}-{case["id"]}'))
    return dict(schema='envelope-shadow-plan-v1', model=old['model'], cases=cases, order=order,
                hard_seconds=600, trial_seconds=old['trial_seconds'], posting_lag_s=old['posting_lag_s'],
                snapshot_limit_s=.001, observer_absolute_s=.002, observer_relative=.1,
                scope='fixed protected32 pre-ADD engine intervals; diagnostic only', **flags())


def validate_manifest(manifest):
    p = protocol()
    if (manifest['model'], manifest['revision']) != (p['model']['name'], p['model']['revision']):
        raise ValueError('Shadow model/revision differs')
    if manifest.get('ignore_eos') is not True or manifest.get('forced_outputs') is not True:
        raise ValueError('Shadow requires forced outputs')
    tokens.validate(manifest, p['cases'], p['order'])
    for trial in p['order']:
        for request in manifest['trials'][trial['trial_id']]:
            if request['token_key'] != f'pair:{trial["paired_content_key"]}:{request["id"]}:{request["input_tokens"]}':
                raise ValueError('Shadow content namespace differs')
    return digest(manifest)


def configure(support, runtime, manifest, *, now, domain, sources):
    state_envelope.PreparedEnvelope(support, runtime)
    if not finite(now) or now < 0 or not isinstance(domain, str) or not domain or not sources:
        raise ValueError('Invalid shadow configuration clock/source')
    config = dict(schema='envelope-shadow-config-v1', support=copy.deepcopy(support),
                  runtime_sha256=digest(runtime), manifest_sha256=validate_manifest(manifest),
                  plan_sha256=digest(protocol()), configured_at=now, clock_domain=domain,
                  source_sha256=copy.deepcopy(sources), **flags())
    config['sha256'] = digest(config)
    return config


def resolve_support(document, runtime):
    if document.get('schema') == 'state-time-support-v1':
        state_envelope.PreparedEnvelope(document, runtime)
        return copy.deepcopy(document)
    required = {'schema', 'gpu_lower_s', 'gpu_upper_s', 'prefill_return_s', 'decode_return_s', 'policy'}
    if set(document) != required or document['schema'] != 'envelope-shadow-assumptions-v1':
        raise ValueError('Use exact-runtime support or explicit shadow timing assumptions')
    if set(document['policy']) != {'max_age_s', 'max_width_s'}:
        raise ValueError('Explicit shadow age/width policy required')
    return state_envelope.make_support(document['gpu_lower_s'], document['gpu_upper_s'],
        document['prefill_return_s'], document['decode_return_s'], runtime, **document['policy'])


def validate_config(config, runtime, *, now, domain, sources):
    required = {'schema', 'support', 'runtime_sha256', 'manifest_sha256', 'plan_sha256',
                'configured_at', 'clock_domain', 'source_sha256', 'sha256', *flags()}
    if (set(config) != required or config['schema'] != 'envelope-shadow-config-v1'
            or config['sha256'] != digest({k: v for k, v in config.items() if k != 'sha256'})
            or config['runtime_sha256'] != digest(runtime) or config['plan_sha256'] != digest(protocol())
            or config['source_sha256'] != sources or config['clock_domain'] != domain
            or any(config[k] is not False for k in flags())):
        raise ValueError('Shadow configuration identity differs')
    if not finite(now) or not finite(config['configured_at']) or not 0 <= config['configured_at'] <= now:
        raise ValueError('Shadow configuration chronology differs')
    return state_envelope.PreparedEnvelope(config['support'], runtime)


def validate_anchors(cap, observed, record=None):
    events = observed.get('anchor_events')
    if not isinstance(events, list) or observed.get('anchor_overflow') != 0:
        raise ValueError('Shadow anchor log overflow')
    count = None if record is None else record['anchor_count']
    if record is not None and (type(count) is not int or not 0 <= count <= len(events)):
        raise ValueError('Shadow snapshot anchor position differs')
    scheduled, pending, last, previous = {}, {}, None, cap['start']
    steps = {digest(sorted((w['request_id'], w['position'], w['tokens']) for w in s['work'])): s for s in cap['steps']}
    seen_steps = set()
    snapshot_pending, snapshot_last = {}, None
    if count == 0:
        snapshot_pending, snapshot_last = {}, None
    for index, event in enumerate(events):
        at = event['observed_at']
        if not finite(at) or not previous <= at <= cap['end']:
            raise ValueError('Shadow anchor event chronology differs')
        previous = at
        batch = event['batch']
        if type(batch) is not int or batch < 1:
            raise ValueError('Shadow anchor batch identity differs')
        if event['kind'] == 'scheduled':
            if set(event) != {'kind', 'batch', 'scheduled_at', 'observed_at', 'work'} or batch in scheduled:
                raise ValueError('Duplicate/invalid shadow scheduling anchor')
            if not finite(event['scheduled_at']) or not cap['start'] <= event['scheduled_at'] <= at:
                raise ValueError('Shadow scheduling anchor time differs')
            if event['work']:
                key = digest(sorted((w['request_id'], w['position'], w['tokens']) for w in event['work']))
                if key not in steps or key in seen_steps or not steps[key]['start'] <= event['scheduled_at'] <= at <= steps[key]['end']:
                    raise ValueError('Shadow scheduling anchor/native step differs')
                seen_steps.add(key)
            scheduled[batch] = event
            pending[batch] = event
        elif event['kind'] == 'committed':
            if set(event) != {'kind', 'batch', 'observed_at'} or batch not in pending or batch != next(iter(pending)):
                raise ValueError('Shadow output anchor order differs')
            job = pending.pop(batch)
            if job['work']:
                key = digest(sorted((w['request_id'], w['position'], w['tokens']) for w in job['work']))
                if not steps[key]['start'] <= at <= steps[key]['end']:
                    raise ValueError('Shadow commit anchor is outside native output span')
            last = at
        else:
            raise ValueError('Unknown shadow anchor event')
        if index+1 == count:
            snapshot_pending, snapshot_last = dict(pending), last
    if pending or seen_steps != set(steps):
        raise ValueError('Shadow anchor/native lifetime inventory differs')
    if record is None:
        return
    cut = record['cut']
    if any(e['observed_at'] > cut for e in events[:count]) or any(e['observed_at'] < cut for e in events[count:]):
        raise ValueError('Shadow snapshot cut differs from anchor position')
    snapshot = record['snapshot']
    jobs = [{k: job[k] for k in ('work', 'scheduled_at')} for job in snapshot_pending.values()]
    if snapshot['last_commit_at'] != snapshot_last or [{k: job[k] for k in ('work', 'scheduled_at')} for job in snapshot['pending']] != jobs:
        raise ValueError('Shadow pending time/commit anchors do not reproduce')


def extract(bundle, trial, manifest, runtime, config):
    """Rebuild interval arithmetic and joins, retaining misses and abstentions."""
    validate_manifest(manifest)
    cap, client, journal = bundle['capture'], bundle['client'], bundle['gateway']
    if not all(finite(cap.get(k)) for k in ('start', 'end')) or not config['configured_at'] <= cap['start'] <= cap['end']:
        raise ValueError('Shadow capture/configuration chronology differs')
    capture_audit = audit_capture(cap, [r for w in cap['workers'] for r in w['rows']],
                                  require_steps=trial['observer'] == 'light')
    if any(not all(finite(s.get(k)) for k in ('start', 'end'))
           or not cap['start'] <= s['start'] <= s['end'] <= cap['end'] for s in cap['steps']):
        raise ValueError('Shadow native step timestamps are outside capture')
    if any(a['start'] > b['start'] for a, b in zip(cap['steps'], cap['steps'][1:])):
        raise ValueError('Shadow native scheduling order differs')
    observed = cap['live']
    if (not capture_audit['valid'] or cap.get('pending_step_ids') or observed['errors'] or observed['overflow']
            or cap['trial_id'] != trial['trial_id'] or cap['observer'] != trial['observer']
            or observed['schema'] != 'envelope-shadow-records-v1'
            or observed['observed'] != (trial['observer'] == 'light')
            or observed['configuration_sha256'] != config['sha256']
            or observed['support_sha256'] != config['support']['sha256']
            or observed['clock_domain'] != config['clock_domain']
            or observed['actuation_authorized'] is not False or observed['native_coverage_qualified'] is not False):
        raise ValueError('Shadow native capture integrity differs')
    validate_anchors(cap, observed)
    predictor = validate_config(config, runtime, now=cap['start'], domain=observed['clock_domain'],
                                sources=config['source_sha256'])
    if observed['cost_sha256'] != digest(predictor._lower.to_dict()) or observed['dispatcher_sha256'] != digest(predictor._dispatch):
        raise ValueError('Shadow native cost/dispatcher differs')
    specs = manifest['trials'][trial['trial_id']]
    native = {r['trace_id']: (rid, r) for rid, r in cap['requests'].items()}
    records = {r['trace_id']: r for r in observed['records']}
    clients = {r['id']: r for r in client['rows']}
    expected = {digest(['live-http', trial['trial_id'], r['id']])[:32] for r in specs}
    if (set(native) != expected or set(records) != expected or len(native) != len(cap['requests'])
            or len(records) != len(observed['records']) or len(clients) != len(client['rows'])
            or set(clients) != {r['id'] for r in specs}):
        raise ValueError('Shadow request/correlation inventory differs')
    for spec in specs:
        row = clients[spec['id']]
        trace = digest(['live-http', trial['trial_id'], spec['id']])[:32]
        rid, request = native[trace]
        if (row['trace_id'] != trace or row['input_ids_sha256'] != manifest['pools'][spec['token_key']]['sha256']
                or request['input_ids_sha256'] != row['input_ids_sha256'] or request['input_tokens'] != spec['input_tokens']
                or request['max_tokens'] != spec['max_tokens'] or request['role'] != spec['role']
                or records[trace]['engine_id'] != rid or records[trace]['engine_enqueue'] != request['enqueue']):
            raise ValueError('Shadow exact HTTP/native input differs')
        native_record = records[trace]
        times = [row['sent'], row['first'], row['finished'], native_record['cut'],
                 native_record['engine_enqueue'], request['enqueue'], request['first'], request['finished']]
        times += [native_record[k] for k in ('prediction_started', 'evaluation_at', 'prediction_finished') if k in native_record]
        if not all(finite(at) and cap['start'] <= at <= cap['end'] for at in times):
            raise ValueError('Shadow probe/native timestamps are outside capture')
        if not 0 <= row['sent']-row['scheduled'] <= protocol()['posting_lag_s']:
            raise ValueError('Shadow posting lag outside scope')
        if spec['role'] != 'background' and abs(row['scheduled']-(client['release']+spec['at'])) > 1e-9:
            raise ValueError('Shadow offered schedule differs')
        live_http.join(row, journal, records[trace], request)
    probe = clients['sentinel']
    rid, request = native[probe['trace_id']]
    record = records[probe['trace_id']]
    foreground_done = max(r['finished'] for r in cap['requests'].values() if r['role'] != 'background')
    if any(r['finished'] < foreground_done for r in cap['requests'].values() if r['role'] == 'background'):
        raise ValueError('Shadow backgrounds finished before foreground')
    joined = live_http.join(probe, journal, record, request)
    result = dict(**{k: trial[k] for k in ('trial_id', 'case', 'stage', 'block', 'observer')},
                  **joined, available=False, covered=False, narrow=False, useful=False,
                  fallback_required=True, reason='off_control', width_s=None,
                  completion_age_s=None, enqueue_age_s=None, observer_elapsed_s=None,
                  snapshot_width_s=None, structure_matches=False, **flags())
    if trial['observer'] == 'off':
        if any(r['status'] != 'off' or r['forecast'] is not None for r in records.values()):
            raise ValueError('OFF control executed a forecast')
        return result
    start, cut, evaluated, finished, enqueued = (record.get(k) for k in
        ('prediction_started', 'cut', 'evaluation_at', 'prediction_finished', 'engine_enqueue'))
    if not all(finite(v) for v in (start, cut, finished, enqueued)) or not start <= cut <= finished <= enqueued:
        raise ValueError('Shadow prediction chronology differs')
    age, elapsed = finished-cut, finished-start
    age_ok = age <= config['support']['policy']['max_age_s']
    if (record['completion_age_s'] != age or record['observer_elapsed_s'] != elapsed
            or record['completion_age_passed'] != age_ok):
        raise ValueError('Shadow completion-age evidence differs')
    result.update(completion_age_s=age, enqueue_age_s=enqueued-cut, observer_elapsed_s=elapsed,
                  reason=record.get('error', 'unavailable'))
    if 'snapshot' not in record:
        if record['status'] != 'unavailable' or record['forecast'] is not None or not record.get('error') or record['fallback_required'] is not True:
            raise ValueError('Missing snapshot without retained observation failure')
        return result
    if not finite(evaluated) or not cut <= evaluated <= finished or record['snapshot']['observed_at'] != cut:
        raise ValueError('Shadow snapshot/evaluation chronology differs')
    snapshot = record['snapshot']
    if set(snapshot) != {'running', 'waiting', 'requests'}:
        validate_anchors(cap, observed, record)
        active = {key: r for key, r in cap['requests'].items()
                  if key != rid and r['enqueue'] <= cut < r['finished']}
        if set(snapshot.get('requests', {})) != set(active):
            raise ValueError('Shadow snapshot/native active inventory differs')
        for key, state in snapshot['requests'].items():
            if state['prompt_tokens'] != active[key]['input_tokens'] or state['max_tokens'] != active[key]['max_tokens']:
                raise ValueError('Shadow snapshot/native request shape differs')
            scheduled = [w for s in cap['steps'] if s['start'] < cut for w in s['work'] if w['request_id'] == key]
            committed = [w for s in cap['steps'] if s['end'] <= cut for w in s['work'] if w['request_id'] == key]
            outputs = sum(w['position']+w['tokens'] >= state['prompt_tokens'] for w in committed)
            if state['computed'] != sum(w['tokens'] for w in scheduled) or state['num_tokens'] != state['prompt_tokens']+outputs:
                raise ValueError('Shadow snapshot/native progress differs')
        pending_work = [[{k: w[k] for k in ('request_id', 'position', 'tokens')} for w in s['work']]
                        for s in cap['steps'] if s['start'] < cut < s['end']]
        if [job['work'] for job in snapshot.get('pending', []) if job['work']] != pending_work:
            raise ValueError('Shadow snapshot/native pending work differs')
    width = cut-start
    if record['snapshot_width_s'] != width:
        raise ValueError('Shadow snapshot width differs')
    try:
        rebuilt = predictor.forecast(record['snapshot'], Request(rid, request['input_tokens'], request['max_tokens']),
                                     now=evaluated, domain=config['clock_domain'])
    except (ValueError, RuntimeError) as exc:
        if record['status'] != 'unavailable' or record['forecast'] is not None or record.get('error') != type(exc).__name__+': '+str(exc):
            raise ValueError('Shadow unavailable forecast does not reproduce') from exc
        if record['fallback_required'] is not True:
            raise ValueError('Unavailable forecast suppressed fallback')
        result.update(snapshot_width_s=width)
        return result
    if (rebuilt != record['forecast'] or record['status'] != ('predicted' if rebuilt['available'] else 'unavailable')
            or record['fallback_required'] != (not age_ok or rebuilt['fallback_required'])):
        raise ValueError('Shadow forecast does not reproduce')
    result.update(available=rebuilt['available'], snapshot_width_s=width, reason=rebuilt['reason'])
    if not rebuilt['available']:
        return result
    lo, hi = rebuilt['ttft_from_cut_s']
    later = any(r['enqueue'] > cut for key, r in cap['requests'].items() if key != rid)
    actual = [[{k: w[k] for k in ('request_id', 'position', 'tokens')} for w in s['work']]
              for s in cap['steps'] if enqueued <= s['start'] < request['first']]
    structure = digest(actual) == rebuilt['work_sha256']
    narrow = rebuilt['width_s'] <= config['support']['policy']['max_width_s']
    usable = narrow and age_ok and enqueued-cut <= config['support']['policy']['max_age_s'] and width <= protocol()['snapshot_limit_s'] and not later and structure
    reasons = []
    for failed, name in [(not narrow, 'interval_too_wide'), (not age_ok, 'completion_age_exceeded'),
                         (enqueued-cut > config['support']['policy']['max_age_s'], 'enqueue_age_exceeded'),
                         (width > protocol()['snapshot_limit_s'], 'snapshot_too_wide'),
                         (later, 'unmodelled_arrivals'), (not structure, 'work_mismatch')]:
        if failed:
            reasons.append(name)
    result.update(ttft_bounds_s=[lo, hi], width_s=hi-lo, covered=not later and structure and lo <= joined['engine_from_cut_s'] <= hi,
                  narrow=narrow, useful=usable and lo <= joined['engine_from_cut_s'] <= hi,
                  fallback_required=not usable, reason=','.join(reasons) or None, structure_matches=structure)
    return result


def score(rows, attempted, failures):
    p = protocol()
    wanted = [t['trial_id'] for t in p['order']]
    if attempted != wanted[:len(attempted)] or len(attempted) != len(set(attempted)):
        raise ValueError('Shadow attempted prefix differs')
    ids = [r['trial_id'] for r in rows]
    if ids != attempted[:len(ids)] or len(ids) != len(set(ids)):
        raise ValueError('Shadow measured prefix differs')
    lookup = {t['trial_id']: t for t in p['order']}
    for row in rows:
        if any(row[k] != lookup[row['trial_id']][k] for k in ('stage', 'case', 'block', 'observer')):
            raise ValueError('Shadow scored membership differs')
    groups = {}
    for stage in ('warmup', 'shadow'):
        planned = [t for t in p['order'] if t['stage'] == stage and t['observer'] == 'light']
        light = [r for r in rows if r['stage'] == stage and r['observer'] == 'light']
        measured = [r for r in light if r.get('measurement_valid', True)]
        available = sum(r['available'] for r in light)
        covered = sum(r['covered'] for r in light)
        groups[stage] = dict(planned=len(planned), measured=len(measured), not_measured=len(planned)-len(measured),
            available=available, unavailable=sum(not r['available'] for r in measured),
            covered=covered, misses=sum(r['available'] and not r['covered'] for r in light),
            wide=sum(r['available'] and not r['narrow'] for r in light),
            useful=sum(r['useful'] for r in light), fallback=sum(r['fallback_required'] for r in light),
            covered_per_planned=covered/len(planned), useful_per_planned=sum(r['useful'] for r in light)/len(planned),
            coverage_given_available=None if not available else covered/available,
            reasons=dict(Counter(r['reason'] for r in light if r['reason'])))
        timing = {}
        for name in ('width_s', 'snapshot_width_s', 'completion_age_s', 'enqueue_age_s', 'observer_elapsed_s'):
            values = [r[name] for r in measured if finite(r.get(name))]
            timing[name] = dict(samples=len(values), p50=None if not values else float(np.percentile(values, 50)),
                                p95=None if not values else float(np.percentile(values, 95)),
                                maximum=None if not values else max(values))
        groups[stage]['timing'] = timing
    controls = []
    for trial in p['order']:
        if trial['stage'] != 'shadow' or trial['observer'] != 'light':
            continue
        pair = [r for r in rows if (r['stage'], r['case'], r['block']) == ('shadow', trial['case'], trial['block'])]
        off = next((r for r in pair if r['observer'] == 'off'), None)
        light = next((r for r in pair if r['observer'] == 'light'), None)
        if off is None or light is None or not off.get('measurement_valid', True) or not light.get('measurement_valid', True):
            controls.append(dict(trial_id=trial['trial_id'], comparable=False, reason='missing_pair'))
        else:
            delta = light['engine_from_cut_s']-off['engine_from_cut_s']
            limit = max(p['observer_absolute_s'], p['observer_relative']*off['engine_from_cut_s'])
            controls.append(dict(trial_id=trial['trial_id'], comparable=True, signed_delta_s=delta,
                                 limit_s=limit, within_budget=abs(delta) <= limit,
                                 same_initial_state_established=False, observer_effect_identified=False))
    return dict(schema='envelope-shadow-report-v1', collection_complete=ids == wanted and not failures,
                planned_trials=len(wanted), attempted_trials=len(attempted), measured_trials=sum(r.get('measurement_valid', True) for r in rows),
                failures=copy.deepcopy(failures), groups=groups, paired_controls=controls,
                paired_control_scope='same offered content with balanced arm order; not identical native state or causal overhead identification',
                rows=rows, inference='descriptive short shadow study, not statistical coverage qualification',
                **flags())


async def run(client, http, url, manifest, support, output, gateway_url, deadline):
    from scripts.run_prefill_service import code_identity
    from .causal import dispatcher_parity
    output = Path(output)
    p = protocol()
    validate_manifest(manifest)
    deadline = min(deadline, time.monotonic()+p['hard_seconds'])
    summary = dict(attempted=[], completed=[], failures=[], collection_complete=False, **flags())
    active = False
    save(output/'shadow-plan.json', p)
    save(output/'http-token-manifest.json', manifest)
    try:
        runtime = await asyncio.wait_for(client.command('meta'), max(.001, min(10, deadline-time.monotonic())))
        runtime_preflight(runtime, manifest)
        runtime_inputs(runtime)
        save(output/'runtime.json', runtime)
        sources = code_identity()
        save(output/'support-input.json', support)
        support = resolve_support(support, runtime)
        config = configure(support, runtime, manifest, now=time.monotonic(),
                           domain=live_http.clock_domain(), sources=sources)
        save(output/'shadow-config.json', config)
        answer = await asyncio.wait_for(client.command('shadow_configure', frozen=config, runtime=runtime),
                                        max(.001, min(10, deadline-time.monotonic())))
        if answer.get('configured') is not True or answer.get('frozen_sha256') != config['sha256']:
            raise ValueError('Shadow native acknowledgement differs')
        save(output/'shadow-configured.json', answer)
        save(output/'shadow-seal.json', dict(at_monotonic=time.monotonic(), config_sha256=config['sha256'],
                                           completed=[], source_sha256=sources))
        for trial in p['order']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Shadow collection deadline')
            if code_identity() != sources or json.loads((output/'shadow-config.json').read_text()) != config:
                raise ValueError('Shadow source/configuration changed during collection')
            tid = trial['trial_id']
            summary['attempted'].append(tid)
            save(output/'shadow-summary.json', summary)
            active = True
            await asyncio.wait_for(client.command('external_begin', trial_id=tid, observer=trial['observer']),
                                   max(.001, deadline-time.monotonic()))
            await live_experiment.isolated_trial(output/'http-token-manifest.json', tid,
                gateway_url+'/v1/completions', output/(tid+'-client.json'), deadline)
            capture = await asyncio.wait_for(finish_external(client, min(deadline, time.monotonic()+5)),
                                             max(.001, min(5, deadline-time.monotonic())))
            active = False
            save(output/(tid+'.json'), capture)
            if trial['observer'] == 'light' and not dispatcher_parity(capture, runtime['workers'][0]['dispatch_table'])['matches']:
                raise ValueError('Shadow native dispatcher differs')
            summary['completed'].append(tid)
            save(output/'shadow-summary.json', summary)
        final = await asyncio.wait_for(client.command('meta'), max(.001, min(10, deadline-time.monotonic())))
        save(output/'runtime-final.json', final)
        if final != runtime or code_identity() != sources or json.loads((output/'shadow-config.json').read_text()) != config:
            raise ValueError('Shadow final runtime/source/configuration drift')
        summary['collection_complete'] = True
    except BaseException as exc:
        summary['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        try:
            if active:
                await asyncio.wait_for(client.command('abort', reason='Shadow collection interrupted'), 2)
                partial = await asyncio.wait_for(finish_external(client, time.monotonic()+3), 3)
                save(output/(summary['attempted'][-1]+'.json'), partial)
        except (ValueError, RuntimeError, TimeoutError, OSError, httpx.HTTPError) as exc:
            summary['failures'].append(dict(type=type(exc).__name__, message='Shadow cleanup: '+str(exc)))
        except asyncio.CancelledError:
            summary['failures'].append(dict(type='CancelledError', message='Shadow cleanup interrupted'))
            raise
        finally:
            save(output/'shadow-summary.json', summary)
    return summary
