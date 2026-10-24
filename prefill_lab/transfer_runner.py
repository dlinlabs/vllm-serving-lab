"""One fresh request-generator process per frozen Azure transfer attempt.

No automatic retries, backend restart or relaxed fidelity gate. Cyclic GC is
deferred only in this child, with the existing native RSS/time guard and restore.
The controller's retained calibration/capture heap never enters the arrival loop.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import httpx

from .contracts import save

ROOT=Path(__file__).resolve().parents[1]
KEY_ENV='PREFILL_TRANSFER_AUTHORIZATION'
GC_SPEC=dict(schema_version=1,policy='defer',max_rss_bytes=1536*1048576,
             max_disabled_seconds=91.,rss_interval_seconds=.1,max_rss_poll_gap_seconds=.5)


def audit_transfer_gc(directory,manifest):
    """Validate the transfer-specific declaration without claiming a legacy sampler.

    RunnerGCControl validates live native resource samples. Independently check
    its persisted window/restoration and the existing complete GC/loop evidence.
    """
    from runner_diagnostics import audit_runner
    directory=Path(directory);issues=[]
    runner=audit_runner(directory,manifest)
    if not runner['valid']:issues.append('Runner diagnostics invalid')
    result=dict(valid=False,issues=issues,runner_diagnostics=runner,
                process_id=manifest['runner_process_id'],run_id=manifest['run_id'])
    try:
        data=json.loads((directory/'runner_gc_control.json').read_text())
        def require(test,message):
            if not test:issues.append(message)
        require(data['spec']==GC_SPEC and manifest['transfer_gc_control']==GC_SPEC,
                'GC declaration/spec mismatch')
        require(data['run_id']==manifest['run_id'] and data['process_id']==manifest['runner_process_id'],
                'GC process/run identity mismatch')
        require(data['status']=='closed' and data['completed'] and data['failure'] is None,
                'GC guard did not complete')
        require(data['initial_enabled'] is True and data['restored_enabled'] is True and
                data['initial_thresholds']==data['restored_thresholds'],'GC policy not restored')
        start,end=manifest['measurement_start'],manifest['measurement_end']
        require(data['disabled_at']<=start<=end<=data['restore_started_at']<=data['restored_at']<=data['closed_at'] and
                data['restore_started_at']-data['disabled_at']<=GC_SPEC['max_disabled_seconds'],
                'GC window does not cover transfer or exceeds bound')
        samples=data['samples']
        require(len(samples)>=4 and [s['phase'] for s in samples[:2]]==['before_gc','after_gc'] and
                all(s['phase']=='active' for s in samples[2:]),'Invalid RSS sample phases')
        for s in samples:
            require(s['available'] is True and s['pid']==data['process_id'] and
                    s['process_created_unix']==samples[0]['process_created_unix'] and
                    type(s['rss_bytes']) is int and 0<=s['rss_bytes']<=GC_SPEC['max_rss_bytes'] and
                    s['sample_start']<=s['sample_end']<=data['closed_at'] and
                    s['gc_enabled'] is (s['phase']!='active'),'Invalid native RSS/policy sample')
        require(all(a['sample_end']<=b['sample_start'] for a,b in zip(samples,samples[1:])),
                'RSS samples out of order')
        require(all(b['sample_end']-a['sample_end']<=GC_SPEC['max_rss_poll_gap_seconds']
                    for a,b in zip(samples[2:],samples[3:])),'RSS monitor gap exceeded')
        require(samples[0]['sample_end']<=data['precollect_start']<=data['precollect_end']<=samples[1]['sample_start'] and
                samples[1]['sample_end']<=data['disabled_at'] and samples[2]['sample_end']<=start and
                samples[-1]['sample_start']>=end,'RSS boundaries do not cover transfer')
        events=[json.loads(line) for line in (directory/'runner_observations.jsonl').read_text().splitlines()]
        require(not any(e['event']=='gc_span' and e['span_start']<data['restore_started_at'] and
                        e['span_end']>data['disabled_at'] for e in events),'GC overlapped deferred window')
        result.update(peak_rss_mib=max(s['rss_bytes'] for s in samples)/1048576,
                      disabled_seconds=data['restore_started_at']-data['disabled_at'],restored=data['restored_enabled'])
    except (OSError,ValueError,KeyError,TypeError,IndexError):
        issues.append('Missing or malformed transfer GC evidence')
    result['valid']=not issues
    return result


async def stop_process(process):
    if process.returncode is not None:return
    try:process.terminate()
    except ProcessLookupError:return
    try:await asyncio.wait_for(process.wait(),5.)
    except asyncio.TimeoutError:
        try:process.kill()
        except ProcessLookupError:pass
        await process.wait()


async def isolated_transfer(http,url,attempt,trace,deadline,output):
    tid=attempt['attempt_id'];output=Path(output).resolve()
    directory=output/(tid+'-runner');directory.mkdir(exist_ok=False)
    # Immutable selection, original request order/content and natural-EOS cap128.
    selected=[r for r in trace['requests'] if attempt['replay_start']<=r['arrival_seconds']<attempt['score_end']]
    packet=dict(attempt=attempt,trace=dict(model=trace['model'],requests=selected),
                url=url,deadline=deadline,output=str(output))
    path=directory/'input.json';save(path,packet)
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    env=os.environ.copy();env[KEY_ENV]=http.headers['authorization']
    print(f'Transfer {tid}: starting isolated generator ({len(selected)} offers)',flush=True)
    process=None
    try:
        with (directory/'process.log').open('w') as log:
            process=await asyncio.create_subprocess_exec(sys.executable,'-m','prefill_lab.transfer_runner',
                '--input',str(path),cwd=ROOT,env=env,stdout=log,stderr=log)
            try:await asyncio.wait_for(process.wait(),max(.001,min(120.,deadline-time.monotonic())))
            except asyncio.TimeoutError as exc:raise TimeoutError(f'Transfer {tid} generator deadline expired') from exc
        if not (directory/'outcome.json').exists():
            raise RuntimeError(f'Transfer {tid} generator exited without an outcome; see {directory}/process.log')
        status=json.loads((directory/'outcome.json').read_text())
        if status['input_sha256']!=fingerprint or status['process_id']!=process.pid:
            raise RuntimeError('Transfer generator input/process identity mismatch')
        if process.returncode!=0 or status['status']!='complete':
            raise RuntimeError(f'Transfer {tid}: '+status.get('error','generator did not complete'))
        verdict=json.loads((directory/'gc-audit.json').read_text())
        if (status.get('gc_policy')!='bounded_runner_defer' or status.get('gc_spec')!=GC_SPEC or
                not verdict.get('valid') or verdict.get('process_id')!=process.pid or verdict.get('run_id')!=tid):
            raise RuntimeError('Transfer generator GC control evidence failed')
        data=json.loads((output/f'{tid}.json').read_text())
        if not data.get('offered_valid'):raise RuntimeError('Transfer generator returned invalid offers')
        maximum=max((r['sent']-r['scheduled'] for r in data['rows']),default=0)
        print(f'Transfer {tid}: valid, max offer lag {maximum*1000:.3f}ms',flush=True)
        return data['capture']
    finally:
        if process is not None:await stop_process(process)


async def worker(path):
    # Imports complete before transfer_local chooses its first offered timestamp.
    from runner_diagnostics import RunnerDiagnostics,SCHEMA,LOOP_INTERVAL
    from runner_gc_control import RunnerGCControl
    from .suite import Client,transfer_local
    raw=Path(path).read_bytes();packet=json.loads(raw);directory=Path(path).parent
    diagnostics=RunnerDiagnostics(directory,packet['attempt']['attempt_id'])
    guard=RunnerGCControl(directory,packet['attempt']['attempt_id'],dict(GC_SPEC))
    manifest=dict(run_id=packet['attempt']['attempt_id'],runner_process_id=os.getpid(),
        config=dict(runner_diagnostics=True),transfer_gc_control=dict(GC_SPEC),
        runner_diagnostics=dict(schema_version=SCHEMA,enabled=True,process_id=os.getpid(),interval_seconds=LOOP_INTERVAL),
        code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ('runner_diagnostics.py','runner_gc_control.py')})
    status=dict(status='failed',process_id=os.getpid(),parent_process_id=os.getppid(),
                input_sha256=hashlib.sha256(raw).hexdigest(),gc_policy='bounded_runner_defer',gc_spec=dict(GC_SPEC))
    task=asyncio.current_task();loop=asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM,task.cancel)
    try:
        authorization=os.environ.pop(KEY_ENV)
        try:
            await diagnostics.start()
            async with httpx.AsyncClient(headers={'authorization':authorization},timeout=45,trust_env=False,
                    limits=httpx.Limits(max_connections=None,max_keepalive_connections=100)) as http:
                try:
                    await transfer_local(Client(http,packet['url']),http,packet['url'],packet['attempt'],
                        packet['trace'],packet['deadline'],Path(packet['output']),gc_control=guard)
                finally:guard.restore()  # Before cancellation-prone HTTP cleanup.
                data=json.loads((Path(packet['output'])/(packet['attempt']['attempt_id']+'.json')).read_text())
                manifest['measurement_start']=data['offer_start']
                manifest['measurement_end']=data['drain_end']
        finally:
            try:guard.close()
            finally:await diagnostics.close()
        verdict=audit_transfer_gc(directory,manifest)
        save(directory/'gc-audit.json',verdict)
        if not verdict['valid']:raise RuntimeError('Transfer GC audit failed: '+str(verdict['issues']))
        status['status']='complete'
    except BaseException as exc:
        status.update(error=str(exc) or type(exc).__name__,error_type=type(exc).__name__)
        raise
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        save(directory/'gc-window.json',manifest)
        status['closed_at']=time.monotonic();save(directory/'outcome.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();asyncio.run(worker(args.input))


if __name__=='__main__':main()
