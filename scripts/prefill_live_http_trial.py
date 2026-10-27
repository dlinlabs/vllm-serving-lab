"""One isolated HTTP offer process; no retained calibration heap or retries."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prefill_lab.contracts import digest,save
from prefill_lab.live_http import probe,clock_domain,protocol


async def run(http,url,manifest,trial,output):
    tasks=[];rows=[];ready=[];domain=clock_domain();start=time.monotonic();release=None
    def launch(r,at,event=None):
        row={};rows.append(row);ids=manifest['pools'][r['token_key']]['input_ids']
        payload=dict(model=manifest['model'],prompt=ids,max_tokens=r['max_tokens'],temperature=0,ignore_eos=True,stream=True,
            stream_options=dict(include_usage=True),vllm_xargs=dict(prefill_lab_role=r['role'],prefill_lab_probe=r['role']=='sentinel'))
        tid=digest(['live-http',trial,r['id']])[:32]
        row.update(id=r['id'],role=r['role'],input_ids_sha256=digest(ids))
        tasks.append(asyncio.create_task(probe(http,url,payload,tid,at,domain=domain,row=row,ready=event)))
    try:
        specs=manifest['trials'][trial]
        for r in specs:
            if r['role']=='background':
                event=asyncio.Event();ready.append(event);launch(r,time.monotonic(),event)
        if ready:await asyncio.wait_for(asyncio.gather(*(e.wait() for e in ready)),protocol()['trial_seconds'])
        release=time.monotonic()
        for r in specs:
            if r['role']!='background':launch(r,release+r['at'])
        await asyncio.wait_for(asyncio.gather(*tasks),protocol()['trial_seconds']-min(time.monotonic()-start,protocol()['trial_seconds']-.001))
        if any(r['status']!='complete' for r in rows):raise RuntimeError('HTTP request failed or was rejected')
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        save(output,dict(schema='live-http-client-v1',clock_domain=domain,release=release,rows=rows))


def main():
    import httpx
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--trial',required=True);p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    async def execute():
        async with httpx.AsyncClient(timeout=45,trust_env=False,limits=httpx.Limits(max_connections=None,max_keepalive_connections=32)) as http:
            response=await http.get(a.url.removesuffix('/v1/completions')+'/metrics');response.raise_for_status()
            await run(http,a.url,json.loads(a.manifest.read_text()),a.trial,a.output)
    asyncio.run(execute())


if __name__=='__main__':main()
