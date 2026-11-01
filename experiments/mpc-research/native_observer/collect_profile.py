"""Bounded direct-HTTP profiling smoke; max8 truncation is intentional."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx
import pr33_native_observer
from transformers import AutoTokenizer

from collect_interventions import ROOT, messages
from mpc_lab.http_capture import capture_chat
from mpc_lab.runtime import local_clock_domain


async def main():
    directory = ROOT/"native-profile-http"
    directory.mkdir(exist_ok=False)
    config = json.loads(Path("/root/pr33-native-control.json").read_text())
    domain = local_clock_domain()
    report = dict(schema="native-direct-http-profiler-smoke-v1",completed=False,
        observer_sha256=hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        groups=[],scope="max8 truncated profiling probes; not natural-EOS system identification or benchmark",
        calibration_accepted=False,performance_accepted=False)
    def persist():
        (directory/"profile-http.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"],local_files_only=True)
    async def request_id_hook(request):
        if "x-benchmark-request-id" in request.headers:
            request.headers["x-request-id"] = request.headers["x-benchmark-request-id"]
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000",timeout=30,
            event_hooks={"request":[request_id_hook]}) as client:
            engine_run = None
            async def sample():
                nonlocal engine_run
                reply = await client.get("/pr33-native/state",
                                          headers={"x-observer-key":config["ADMIN_API_KEY"]})
                reply.raise_for_status()
                frame = reply.json()
                if (frame["clock_domain"]!=domain or not frame["complete"]
                        or frame["observer_sha256"]!=report["observer_sha256"]):
                    raise ValueError("profiler observer source/clock")
                now = time.monotonic()
                if (not 0 <= now-frame["sampled_at"] <= .5
                        or not 0 <= frame["frontend_sampled_at"]-frame["sampled_at"] <= .05):
                    raise ValueError("profiler observer stale/wide")
                engine_run = engine_run or frame["run_id"]
                if frame["run_id"]!=engine_run:
                    raise ValueError("engine epoch changed")
                return {**frame,"received_at":now}
            deadline = time.monotonic()+180
            while True:
                try:
                    await sample()
                    break
                except httpx.HTTPError:
                    if time.monotonic()>deadline:
                        raise
                    await asyncio.sleep(2)
            async def drain():
                checks,deadline = [],time.monotonic()+20
                while time.monotonic()<deadline:
                    f = await sample()
                    checks = checks+[f] if all(f[k]==0 for k in (
                        "running","waiting","pending_batches","frontend_unfinished")) else []
                    if len(checks)==2:
                        return checks
                    await asyncio.sleep(.05)
                raise ValueError("profile native drain failed")
            async def group(prompt,batch,prelude):
                g = dict(prompt_tokens=prompt,offered_batch=batch,prelude=prelude,
                         rows=[],native_frames=[],initial_drain=await drain())
                report["groups"].append(g)
                body = dict(model="Qwen/Qwen3-4B-Instruct-2507",temperature=0,max_tokens=8,
                    messages=messages(tokenizer,prompt,28,"ocean currents"))
                g["start"] = time.monotonic()+.1
                finished = asyncio.Event()
                async def observe():
                    while not finished.is_set():
                        g["native_frames"].append(await sample())
                        await asyncio.sleep(.02)
                async def offers():
                    try:
                        await asyncio.gather(*(capture_chat(client,"/v1/chat/completions",
                            body,uuid.uuid4().hex,g["start"],enabled=True,timeout_s=20,
                            record=g["rows"].append) for _ in range(batch)))
                    finally:
                        finished.set()
                observer,requests = asyncio.create_task(observe()),asyncio.create_task(offers())
                try:
                    done,_ = await asyncio.wait((observer,requests),return_when=asyncio.FIRST_EXCEPTION)
                    for task in done:
                        task.result()
                finally:
                    if not requests.done():
                        requests.cancel()
                    await asyncio.gather(requests,return_exceptions=True)
                    finished.set()
                    await observer
                    persist()
                g["final_drain"] = await drain()
                if len(g["rows"])!=batch or any(
                    r["status"]!="complete" or r["response_request_id"]!=r["request_id"]
                    or r["usage"]["prompt_tokens"]!=prompt for r in g["rows"]):
                    raise ValueError("direct HTTP profile probe invalid")
                persist()
            for prompt in (128,4096):
                await group(prompt,1,True)
            started = await client.post("/start_profile")
            started.raise_for_status()
            report["profile_start_received_at"] = time.monotonic()
            try:
                for repetition in range(2):
                    for prompt,batch in ((128,1),(4096,4),(4096,1),(128,4)):
                        await group(prompt,batch,False)
            finally:
                stopped = await client.post("/stop_profile")
                report["profile_stop_status"] = stopped.status_code
                stopped.raise_for_status()
            report["completed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        persist()


if __name__=="__main__":
    asyncio.run(main())
