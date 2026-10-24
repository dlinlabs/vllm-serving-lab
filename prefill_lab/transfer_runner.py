"""One fresh request-generator process per frozen Azure transfer attempt.

No automatic retries, backend restart, GC policy change or relaxed fidelity gate.
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
        data=json.loads((output/f'{tid}.json').read_text())
        if not data.get('offered_valid'):raise RuntimeError('Transfer generator returned invalid offers')
        maximum=max((r['sent']-r['scheduled'] for r in data['rows']),default=0)
        print(f'Transfer {tid}: valid, max offer lag {maximum*1000:.3f}ms',flush=True)
        return data['capture']
    finally:
        if process is not None:await stop_process(process)


async def worker(path):
    # Imports complete before transfer_local chooses its first offered timestamp.
    from runner_diagnostics import RunnerDiagnostics
    from .suite import Client,transfer_local
    raw=Path(path).read_bytes();packet=json.loads(raw);directory=Path(path).parent
    diagnostics=RunnerDiagnostics(directory,packet['attempt']['attempt_id'])
    status=dict(status='failed',process_id=os.getpid(),parent_process_id=os.getppid(),
                input_sha256=hashlib.sha256(raw).hexdigest(),gc_policy='automatic_unchanged')
    task=asyncio.current_task();loop=asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM,task.cancel)
    try:
        authorization=os.environ.pop(KEY_ENV)
        await diagnostics.start()
        async with httpx.AsyncClient(headers={'authorization':authorization},timeout=45,trust_env=False,
                limits=httpx.Limits(max_connections=None,max_keepalive_connections=100)) as http:
            await transfer_local(Client(http,packet['url']),http,packet['url'],packet['attempt'],
                packet['trace'],packet['deadline'],Path(packet['output']))
        await diagnostics.close()
        status['status']='complete'
    except BaseException as exc:
        status.update(error=str(exc) or type(exc).__name__,error_type=type(exc).__name__)
        raise
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        try:await diagnostics.close()
        finally:
            status['closed_at']=time.monotonic();save(directory/'outcome.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();asyncio.run(worker(args.input))


if __name__=='__main__':main()
