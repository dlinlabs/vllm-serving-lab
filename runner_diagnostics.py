"""Opt-in runner observations; no request/token logging and no GC policy changes."""
import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from telemetry import GatewayDiagnostics, Telemetry

SCHEMA = 1
FILES = ('runner_observations.jsonl', 'runner_observations_status.json')
LOOP_INTERVAL = .01


class RunnerDiagnostics:
    """Reuse the bounded primitive GC inbox; JSON writes run off the arrival loop."""
    def __init__(self, output, run_id, capacity=4096):
        self.output = Path(output)
        self.telemetry = Telemetry()
        self.telemetry.enabled = True
        self.telemetry.run_id = run_id
        self.counts = Counter()
        self.diagnostics = GatewayDiagnostics(self.emit, capacity=capacity)
        self.stop_event, self.ready = asyncio.Event(), asyncio.Event()
        self.tasks = []
        self.closed = False
        self.started = False

    def emit(self, event, request_id=None, **fields):
        fields.setdefault('process_id', os.getpid())
        fields['source'] = 'runner'
        self.counts[event] += 1
        self.telemetry.emit(event, request_id, **fields)

    async def monitor(self):
        self.emit('loop_monitor_started', interval_seconds=LOOP_INTERVAL)
        try:
            while not self.stop_event.is_set():
                due = time.monotonic() + LOOP_INTERVAL
                self.ready.set()
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=LOOP_INTERVAL)
                except asyncio.TimeoutError:
                    resumed = time.monotonic()
                    self.emit('event_loop_lag', scheduled_at=due, resumed_at=resumed,
                              lag_seconds=max(0., resumed-due))
        finally:
            self.emit('loop_monitor_stopped')

    async def collect(self):
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=1.)
            except asyncio.TimeoutError:
                pass
            self.diagnostics.sample()
            await self.telemetry.flush(self.output / FILES[0])

    async def start(self):
        (self.output / FILES[0]).touch(exist_ok=False)
        self.diagnostics.start()
        self.started = True
        # Warm up the writer before measurement; never write JSON in a GC callback.
        await self.telemetry.flush(self.output / FILES[0])
        if self.telemetry.write_errors or self.telemetry.dropped:
            raise RuntimeError('Runner diagnostics initial persistence failed')
        self.tasks = [asyncio.create_task(self.monitor()), asyncio.create_task(self.collect())]
        await self.ready.wait()

    async def close(self):
        if self.closed or not self.started:
            return
        self.stop_event.set()
        errors = await asyncio.gather(*self.tasks, return_exceptions=True)
        errors = [type(e).__name__ for e in errors if isinstance(e, BaseException)]
        await self.diagnostics.stop()
        await self.telemetry.flush(self.output / FILES[0])
        status = dict(schema_version=SCHEMA, run_id=self.telemetry.run_id, source='runner',
                      process_id=os.getpid(), status='closed', closed_at=time.monotonic(),
                      counts=dict(self.counts), dropped=self.telemetry.dropped,
                      write_errors=self.telemetry.write_errors, task_errors=errors,
                      health=self.diagnostics.health())
        temporary = self.output / (FILES[1]+'.tmp')
        temporary.write_text(json.dumps(status, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(self.output / FILES[1])
        self.closed = True
        if errors or status['dropped'] or status['write_errors'] or not status['health']['complete']:
            raise RuntimeError('Runner diagnostics did not close with complete evidence')


def audit_runner(output, manifest, clients=(), warmup=()):
    """Separate evidence gate. Legacy/disabled captures retain their old reports."""
    output = Path(output)
    config = manifest.get('config', {})
    enabled = config.get('runner_diagnostics', False) if isinstance(config, dict) else None
    present = any((output/name).exists() for name in FILES)
    declared = 'runner_diagnostics' in manifest
    if enabled is False and not present and not declared:
        return None
    issues = []
    result = dict(valid=False, issues=issues, source='runner', gc_coverage_complete=False,
                  note='Timing overlap is not exclusive CPU attribution; runner GC/OS scheduling are distinct.')

    def require(test, message):
        if not test and message not in issues:
            issues.append(message)

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    try:
        require(enabled is True, 'Runner diagnostics artifacts/declaration without enabled option')
        require(type(manifest['runner_process_id']) is int and manifest['runner_process_id']>0,
                'Invalid runner process ID')
        declaration = manifest['runner_diagnostics']
        require(declaration == dict(schema_version=SCHEMA, enabled=True,
                                   process_id=manifest['runner_process_id'], interval_seconds=LOOP_INTERVAL),
                'Runner diagnostics declaration mismatch')
        fingerprint = manifest['code_sha256'].get('runner_diagnostics.py')
        require(isinstance(fingerprint, str) and len(fingerprint)==64 and
                all(c in '0123456789abcdef' for c in fingerprint), 'Missing runner diagnostics code fingerprint')
        rows = [json.loads(line) for line in (output/FILES[0]).read_text().splitlines() if line.strip()]
        status = json.loads((output/FILES[1]).read_text())
        require(bool(rows), 'Empty runner observations')
        require(all(r['run_id'] == manifest['run_id'] and r['source'] == 'runner' and
                    r['process_id'] == manifest['runner_process_id'] for r in [*rows, status]),
                'Runner observation identity mismatch')
        require(status['schema_version'] == SCHEMA and status['status'] == 'closed' and
                finite(status['closed_at']), 'Runner observations did not close normally')
        require(status['dropped'] == status['write_errors'] == 0 and not status['task_errors'],
                'Runner observations lost records or worker failed')
        require(all(finite(r['monotonic']) and r['monotonic'] <= status['closed_at'] for r in rows) and
                all(a['monotonic'] <= b['monotonic'] for a,b in zip(rows, rows[1:])),
                'Invalid runner emission timestamps/order')
        counts = Counter(r['event'] for r in rows)
        require(dict(counts) == status['counts'], 'Runner record counts disagree with status')
        allowed = {'diagnostics_started','diagnostics_stopped','gc_span','gc_unmatched',
                   'diagnostic_cpu','loop_monitor_started','loop_monitor_stopped','event_loop_lag'}
        require(not counts.keys()-allowed, 'Unknown runner observation event')
        require(all(counts[k] == 1 for k in ['diagnostics_started','diagnostics_stopped',
                                           'loop_monitor_started','loop_monitor_stopped']),
                'Missing or duplicate runner coverage boundaries')
        first = next(r for r in rows if r['event']=='diagnostics_started')
        last = next(r for r in rows if r['event']=='diagnostics_stopped')
        require(first['diagnostics_schema_version']==last['diagnostics_schema_version']==1 and
                first['python_implementation']=='cpython' and first['gil_enabled'] is True and
                type(first['loop_thread_id']) is int and first['loop_thread_id']>0,
                'Invalid runner GC implementation/schema/thread identity')
        monitor_start = next(r for r in rows if r['event']=='loop_monitor_started')
        monitor_stop = next(r for r in rows if r['event']=='loop_monitor_stopped')
        a,b = first['coverage_start'],last['coverage_end']
        latest = max([manifest['measurement_end']] +
                     [r['finished_at'] for r in [*clients,*warmup] if finite(r.get('finished_at'))])
        require(finite(a) and finite(b) and a <= manifest['measurement_start'] <= latest <= b and
                last['coverage_start'] == a and b <= last['monotonic'], 'Incomplete runner GC coverage')
        require(monitor_start['interval_seconds'] == LOOP_INTERVAL and
                a <= monitor_start['monotonic'] <= manifest['measurement_start'] and
                latest <= monitor_stop['monotonic'] <= last['monotonic'], 'Incomplete runner loop coverage')
        health = status['health']
        require(health == last['health'] and health['complete'] is True and
                health['closed'] is True and health['quiesced'] is True and
                health['registration_missing'] is False and
                all(health[k] == 0 for k in ['lost_callbacks','buffered_callbacks','inflight_callbacks',
                   'callback_errors','pending_spans','unmatched_boundaries','undelivered_callbacks']) and
                health['callbacks_produced'] == health['callbacks_received'] == 2*counts['gc_span'] and
                health['gc_spans'] == counts['gc_span'] and health['cpu_samples'] == counts['diagnostic_cpu'] and
                counts['gc_unmatched'] == 0, 'Incomplete runner GC callback accounting')
        spans = [r for r in rows if r['event']=='gc_span']
        tickets=[]
        for r in spans:
            require(r['diagnostics_schema_version'] == 1 and type(r['generation']) is int and 0<=r['generation']<=2 and
                    all(finite(r[k]) for k in ['span_start','span_end','callback_thread_cpu_seconds']) and
                    a<=r['span_start']<=r['span_end']<=b and r['span_end']<=r['monotonic'] and
                    r['callback_thread_cpu_seconds']>=0 and
                    type(r['on_loop_thread']) is bool and
                    r['on_loop_thread'] == (r['callback_thread_id']==first['loop_thread_id']),
                    'Invalid runner GC span')
            require(type(r['start_sequence']) is int and type(r['stop_sequence']) is int and
                    0<=r['start_sequence']<r['stop_sequence'], 'Invalid runner GC boundary sequence')
            tickets.extend([r['start_sequence'],r['stop_sequence']])
        # CallbackCounter.read() consumes private probe tickets. Sequence values
        # are unique but need not be contiguous; callback counts exclude probes.
        require(len(tickets)==len(set(tickets))==health['callbacks_produced'],
                'Missing/duplicate runner GC boundary sequences')
        cpu=[r for r in rows if r['event']=='diagnostic_cpu']
        require(len(cpu)>=2, 'Missing runner CPU boundary samples')
        require(bool(cpu) and cpu[0]['sampled_at']<=manifest['measurement_start'] and
                cpu[-1]['sampled_at']>=latest, 'Incomplete runner CPU coverage')
        for r in cpu:
            require(r['loop_thread_id']==first['loop_thread_id'] and
                    all(finite(r[k]) for k in ['sampled_at','loop_thread_cpu_seconds','process_cpu_seconds']) and
                    a<=r['sampled_at']<=last['monotonic'], 'Invalid runner CPU sample')
        require(all(x['sampled_at']<=y['sampled_at'] and
                    x['loop_thread_cpu_seconds']<=y['loop_thread_cpu_seconds'] and
                    x['process_cpu_seconds']<=y['process_cpu_seconds'] for x,y in zip(cpu,cpu[1:])),
                'Runner CPU counters decreased')
        monitors=[r for r in rows if r['event']=='event_loop_lag']
        require(bool(monitors), 'No runner loop observations')
        for r in monitors:
            require(all(finite(r[k]) for k in ['scheduled_at','resumed_at','lag_seconds']) and
                    monitor_start['monotonic']<=r['scheduled_at']<=r['resumed_at']<=r['monotonic']<=monitor_stop['monotonic'] and
                    abs(r['lag_seconds']-(r['resumed_at']-r['scheduled_at']))<1e-6,
                    'Invalid runner loop observation')
        result.update(coverage_start=a, coverage_end=b, gc_spans=len(spans), loop_samples=len(monitors),
                      max_loop_lag=max((r['lag_seconds'] for r in monitors),default=None))
    except (OSError, ValueError, TypeError, KeyError, StopIteration, OverflowError):
        require(False, 'Missing or malformed runner diagnostic evidence')
    result['valid'] = result['gc_coverage_complete'] = not issues
    return result
