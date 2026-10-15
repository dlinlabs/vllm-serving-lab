"""Bounded, opt-in automatic cyclic-GC deferral in the request runner only."""
import asyncio
import gc
import json
import math
import os
from pathlib import Path
import time

FILE = 'runner_gc_control.json'
INTERVAL = .1
MAX_SECONDS = 600.
MAX_SAMPLE_GAP = .5


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def gc_spec(config, duration):
    policy = config.get('runner_gc_policy', 'automatic')
    rss = config.get('runner_max_rss_mib')
    if policy == 'automatic':
        if rss is not None:
            raise ValueError('RSS bound is only supported with deferred runner GC')
        return None
    if policy != 'defer' or not finite(rss) or rss <= 0:
        raise ValueError('Deferred runner GC needs a finite positive --runner-max-rss-mib')
    if (config.get('runner_diagnostics') is not True or config.get('sampler_mode') != 'process' or
            not (config.get('gateway_diagnostics') or config.get('engine_observations'))):
        raise ValueError('Deferred runner GC requires runner diagnostics and an enabled process sampler')
    drain = config.get('drain_timeout')
    if not finite(duration) or not finite(drain) or min(duration, drain) <= 0:
        raise ValueError('Invalid runner GC duration/drain bound')
    bound = duration + drain + 1.
    if bound > MAX_SECONDS:
        raise ValueError('Deferred runner GC window including drain must not exceed 600 seconds')
    if not math.isfinite(rss * 1048576):
        raise ValueError('Runner RSS limit is too large')
    limit = int(rss * 1048576)
    if limit < 1:
        raise ValueError('Runner RSS limit is less than one byte')
    return dict(schema_version=1, policy='defer', max_rss_bytes=limit,
                max_disabled_seconds=bound, rss_interval_seconds=INTERVAL,
                max_rss_poll_gap_seconds=MAX_SAMPLE_GAP)


def resources():
    # Keep the existing Linux PID/procfs guard; never remap or ignore identity.
    from diagnostic_sampling import ObservationSampler
    return ObservationSampler.process_resources(os.getpid())


class RunnerGCControl:
    def __init__(self, output, run_id, spec, *, _probe=None):
        self.output, self.probe = Path(output), _probe or resources
        self.spec = spec
        self.state = dict(schema_version=1, run_id=run_id, process_id=os.getpid(),
            spec=spec, status='preparing', initial_enabled=gc.isenabled(),
            initial_thresholds=list(gc.get_threshold()), samples=[], completed=False,
            disabled_at=None, restore_started_at=None, restored_at=None, failure=None)

    def fail(self, reason):
        self.state['failure'] = self.state['failure'] or reason
        raise RuntimeError('Runner GC control: '+reason)

    async def sample(self, phase):
        try:
            row = await asyncio.wait_for(asyncio.to_thread(self.probe), timeout=1.)
        except Exception:
            self.fail('RSS probe failed or timed out')
        row = dict(row, phase=phase, gc_enabled=gc.isenabled())
        self.state['samples'].append(row)
        if len(self.state['samples']) > int(MAX_SECONDS/INTERVAL)+10:
            self.fail('RSS record capacity exceeded')
        if (row.get('available') is not True or row.get('pid') != os.getpid() or
                type(row.get('rss_bytes')) is not int or row['rss_bytes'] < 0 or
                not all(finite(row.get(k)) for k in ('process_created_unix','sample_start','sample_end')) or
                row['sample_end'] < row['sample_start']):
            self.fail('RSS probe unavailable or invalid')
        first = self.state['samples'][0]
        if row['process_created_unix'] != first['process_created_unix']:
            self.fail('RSS process identity changed')
        if row['rss_bytes'] > self.spec['max_rss_bytes']:
            self.fail('RSS limit exceeded')
        if phase == 'active':
            previous = self.state['samples'][-2]
            if previous['phase']=='active' and row['sample_end']-previous['sample_end']>MAX_SAMPLE_GAP:
                self.fail('RSS monitor gap exceeded')
            if gc.isenabled() or list(gc.get_threshold()) != self.state['initial_thresholds']:
                self.fail('GC policy changed inside deferred window')
            if time.monotonic()-self.state['disabled_at'] > self.spec['max_disabled_seconds']:
                self.fail('Deferred GC duration exceeded')

    async def prepare(self):
        if not self.state['initial_enabled']:
            self.fail('GC was already disabled before intervention')
        await self.sample('before_gc')
        self.state['precollect_start'] = time.monotonic()
        self.state['precollect_objects'] = gc.collect(2)
        self.state['precollect_end'] = time.monotonic()
        await self.sample('after_gc')
        gc.disable()
        self.state['disabled_at'] = time.monotonic()
        self.state['status'] = 'active'
        # Complete a guarded resource read before choosing measurement_start.
        await self.sample('active')

    def restore(self):
        if self.state['restored_at'] is not None:
            return
        self.state['restore_started_at'] = time.monotonic()
        # This runs synchronously before any cancellation-prone await.
        if list(gc.get_threshold()) != self.state['initial_thresholds']:
            gc.set_threshold(*self.state['initial_thresholds'])
        if self.state['initial_enabled']:
            gc.enable()
        else:
            gc.disable()
        self.state['restored_at'] = time.monotonic()
        self.state['restored_enabled'] = gc.isenabled()
        self.state['restored_thresholds'] = list(gc.get_threshold())

    async def supervise(self, work):
        async def watch():
            while True:
                await asyncio.sleep(INTERVAL)
                await self.sample('active')
        main, watcher = asyncio.create_task(work), asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait([main, watcher], return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                watcher.result()
                self.fail('RSS watcher exited unexpectedly')
            result = main.result()
            watcher.cancel()
            joined = await asyncio.gather(watcher, return_exceptions=True)
            if isinstance(joined[0], BaseException) and not isinstance(joined[0], asyncio.CancelledError):
                raise joined[0]
            await self.sample('active')
            self.state['completed'] = True
            return result
        except BaseException as exc:
            self.state['failure'] = self.state['failure'] or type(exc).__name__
            raise
        finally:
            self.restore()
            for task in (main, watcher):
                task.cancel()
            async def join():
                await asyncio.gather(main, watcher, return_exceptions=True)
            cleanup = asyncio.create_task(join())
            cancellation = None
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            cleanup.result()
            if cancellation is not None:
                raise cancellation

    def close(self):
        self.restore()
        self.state.update(status='closed', closed_at=time.monotonic())
        temporary = self.output / (FILE+'.tmp')
        temporary.write_text(json.dumps(self.state, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(self.output / FILE)


def audit_gc_control(output, manifest, clients=(), runner_verdict=None):
    """Never upgrade a failed experiment or accept missing memory/restore evidence."""
    output = Path(output)
    config = manifest.get('config', {})
    if not isinstance(config, dict):
        config = {}
    enabled = config.get('runner_gc_policy', 'automatic') != 'automatic'
    if not enabled and 'runner_gc_control' not in manifest and not (output/FILE).exists():
        return None
    issues = []
    def require(test, message):
        if not test and message not in issues:
            issues.append(message)
    result = dict(valid=False, issues=issues, policy=config.get('runner_gc_policy'),
                  note='RSS is sampled, not an OS memory ceiling. GC deferral is not a latency guarantee.')
    try:
        duration = json.loads((output/'trace.json').read_text())['duration_seconds']
        spec = gc_spec(config, duration)
        require(spec is not None and spec==manifest['runner_gc_control'], 'GC declaration/config mismatch')
        fingerprint = manifest['code_sha256'].get('runner_gc_control.py')
        require(isinstance(fingerprint,str) and len(fingerprint)==64 and
                all(c in '0123456789abcdef' for c in fingerprint), 'Missing GC control fingerprint')
        data = json.loads((output/FILE).read_text())
        require(data['schema_version']==1 and data['spec']==spec and data['run_id']==manifest['run_id'] and
                type(data['process_id']) is int and data['process_id']==manifest['runner_process_id'],
                'GC evidence identity/spec mismatch')
        require(data['status']=='closed' and data['completed'] is True and data['failure'] is None,
                'GC intervention did not complete normally')
        require(data['initial_enabled'] is True and data['restored_enabled'] is True and
                data['initial_thresholds']==data['restored_thresholds'], 'GC state was not restored')
        keys=['precollect_start','precollect_end','disabled_at','restore_started_at','restored_at','closed_at']
        require(all(finite(data[k]) for k in keys) and
                [data[k] for k in keys]==sorted(data[k] for k in keys), 'Invalid GC lifecycle timestamps')
        start,end=manifest['measurement_start'],manifest['measurement_end']
        latest=max([end]+[r['finished_at'] for r in clients if finite(r.get('finished_at'))])
        require(data['disabled_at']<=start<=latest<=data['restore_started_at'] and
                data['restore_started_at']-data['disabled_at']<=spec['max_disabled_seconds'],
                'Incomplete or overlong deferred GC coverage')
        require(runner_verdict is not None and runner_verdict.get('valid') is True,
                'Runner diagnostics cannot verify deferred GC')
        samples=data['samples']
        require(4<=len(samples)<=int(MAX_SECONDS/INTERVAL)+10 and
                [s['phase'] for s in samples[:2]]==['before_gc','after_gc'] and
                all(s['phase']=='active' for s in samples[2:]), 'Missing or invalid RSS sample phases')
        for s in samples:
            require(s['available'] is True and type(s['pid']) is int and s['pid']==data['process_id'] and
                    s['process_created_unix']==samples[0]['process_created_unix'] and
                    finite(s['process_created_unix']) and s['process_created_unix']>0 and
                    type(s['rss_bytes']) is int and 0<=s['rss_bytes']<=spec['max_rss_bytes'] and
                    finite(s['sample_start']) and finite(s['sample_end']) and
                    s['sample_start']<=s['sample_end']<=data['closed_at'], 'Invalid/unavailable/over-limit RSS sample')
            require(s['gc_enabled'] is (s['phase']!='active'), 'Observed GC policy differs from intervention')
        require(all(a['sample_end']<=b['sample_start'] for a,b in zip(samples,samples[1:])),
                'RSS samples overlap or are out of order')
        require(all(b['sample_end']-a['sample_end']<=MAX_SAMPLE_GAP for a,b in zip(samples[2:],samples[3:])),
                'RSS observation gap exceeded declared bound')
        require(samples[0]['sample_end']<=data['precollect_start'] and
                data['precollect_end']<=samples[1]['sample_start'] and
                samples[1]['sample_end']<=data['disabled_at'] and
                samples[2]['sample_end']<=start and samples[-1]['sample_start']>=latest,
                'RSS boundary samples do not cover measurement/drain')
        rows=[json.loads(s) for s in (output/'runner_observations.jsonl').read_text().splitlines() if s.strip()]
        overlap=[r for r in rows if r['event']=='gc_span' and
                 r['span_start']<data['restore_started_at'] and r['span_end']>data['disabled_at']]
        require(not overlap, 'A GC collection overlapped deferred window')
        result.update(samples=len(samples), peak_rss_mib=max(s['rss_bytes'] for s in samples)/1048576,
                      max_rss_mib=spec['max_rss_bytes']/1048576,
                      disabled_seconds=data['restore_started_at']-data['disabled_at'],
                      measurement_max_runner_loop_lag=max((r['lag_seconds'] for r in rows
                          if r['event']=='event_loop_lag' and start<=r['scheduled_at']<end),default=None),
                      restored=data['restored_enabled']==data['initial_enabled'])
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
        require(False, 'Missing or malformed GC control evidence')
    result['valid'] = not issues
    return result
