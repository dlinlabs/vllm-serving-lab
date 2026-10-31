"""Sparse Train-only first-residence identification; no adaptive actuation."""
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import httpx
import pr33_native_observer
from transformers import AutoTokenizer

from collect_interventions import ROOT, messages
from mpc_lab.actuator import GuardedActuator, IdentificationPermit
from mpc_lab.contracts import CostSpec, Decision, ModelParameters, implementation_hash
from mpc_lab.http_capture import capture_chat, drain_evidence
from mpc_lab.runtime import local_clock_domain, snapshot_from_dict


async def main():
    directory = ROOT/"native-isolated-train"
    directory.mkdir(exist_ok=True)
    output = directory/"isolated.json"
    if output.exists():
        raise ValueError("isolated evidence already exists")
    config = json.loads(Path("/root/pr33-native-control.json").read_text())
    domain = local_clock_domain()
    observer_sha = hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest()
    report = dict(schema="native-isolated-train-v1",completed=False,episodes=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_hash=implementation_hash(),observer_sha256=observer_sha,
        specification=dict(seeds=[160009,160010],cells=[[128,128],[128,256],[4096,128],[4096,256]],
            repetitions=3,prelude_per_cell=1,timeout_s=20,
            purpose="Observed-isolated backend-send to first-content proxy, not pure GPU kernel time.",
            isolation="Two consecutive drained observations before and after each serial offer.",
            fitting="Seed-balanced nonnegative overhead + prompt token slope; grouped leave-one-seed-out.",
            acceptance="No calibrated control or performance acceptance from this pilot."),
        calibration_accepted=False,performance_accepted=False)
    def persist():
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    persist()
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"],local_files_only=True)
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8080",timeout=30,
            headers={"x-admin-api-key":config["ADMIN_API_KEY"]}) as gateway, \
            httpx.AsyncClient(base_url="http://127.0.0.1:8000",timeout=5) as native:
            initial = (await gateway.get("/admin/mpc/state")).json()
            parameters = ModelParameters(cost_spec=CostSpec(**initial["cost_spec"]),
                model_version="native-id-unfitted-v1",calibration_version="unfitted-not-validated")
            if snapshot_from_dict(initial).identity != parameters.identity:
                raise ValueError("gateway model identity")
            driver = GuardedActuator(gateway,parameters,directory/"actuator-pending.json",
                permit=IdentificationPermit(parameters.identity,approved=True,
                    run_id=initial["run_id"],clock_domain=domain,expires_at=time.monotonic()+300),
                enabled=True,max_age_s=.5)
            engine_run = None
            async def sample():
                nonlocal engine_run
                g = await gateway.get("/admin/mpc/state")
                g.raise_for_status()
                received_g = time.monotonic()
                n = await native.get("/pr33-native/state",
                                     headers={"x-observer-key":config["ADMIN_API_KEY"]})
                n.raise_for_status()
                received_n = time.monotonic()
                document, frame = g.json(), n.json()
                if (frame["clock_domain"] != domain or document["clock_domain"] != domain
                        or frame["observer_sha256"] != observer_sha or not frame["complete"]
                        or document["run_id"] != initial["run_id"]):
                    raise ValueError("source or clock changed")
                engine_run = engine_run or frame["run_id"]
                if (frame["run_id"] != engine_run or not 0 <= received_n-frame["sampled_at"] <= .5
                        or not 0 <= frame["frontend_sampled_at"]-frame["sampled_at"] <= .05
                        or not 0 <= received_n-document["sampled_at"] <= .5):
                    raise ValueError("stale, wide or foreign-epoch sample")
                return {**frame,"received_at":received_n,"gateway_received_at":received_g,
                        "gateway_snapshot":document}
            async def drain():
                evidence, deadline = [],time.monotonic()+20
                while time.monotonic() < deadline:
                    frame = await sample()
                    snapshot = snapshot_from_dict(frame["gateway_snapshot"])
                    proof = drain_evidence(snapshot,frame,identity=parameters.identity,
                        gateway_run=initial["run_id"],engine_run=engine_run,domain=domain,
                        now=time.monotonic(),pending_operation=driver.path.exists())
                    evidence = evidence+[proof] if proof["drained"] else []
                    if len(evidence)==2:
                        return evidence
                    await asyncio.sleep(.05)
                raise ValueError("drain failed")
            report["initial_drain"] = await drain()
            fence = await driver.reconcile()
            if fence["status"]=="pending":
                raise ValueError("unresolved pending command")
            state = snapshot_from_dict((await gateway.get("/admin/mpc/state")).json())
            reply = await driver.submit(state,Decision(16,None,"identification_assignment",
                False,None,None,0,0,0,parameters.identity))
            report["assignment"] = reply
            if reply["status"]!="applied":
                raise ValueError("fixed16 assignment unverified")
            for seed in report["specification"]["seeds"]:
                rng = random.Random(seed)
                cells = list(report["specification"]["cells"])
                rng.shuffle(cells)
                for prompt,maximum in cells:
                    for repetition in range(4):
                        episode = dict(seed=seed,split="train",budget=16,
                            episode_id=f"isolated-{seed}-{prompt}-{maximum}-{repetition}",
                            prompt_tokens=prompt,max_tokens=maximum,prelude=repetition==0,
                            rows=[],native_frames=[])
                        report["episodes"].append(episode)
                        episode["initial_drain"] = await drain()
                        episode["initial"] = (await gateway.get("/admin/mpc/state")).json()
                        episode["start"] = time.monotonic()
                        body = dict(model="Qwen/Qwen3-4B-Instruct-2507",temperature=0,
                            max_tokens=maximum,messages=messages(tokenizer,prompt,
                                rng.choice((28,32,36) if maximum==128 else (65,80,95)),
                                rng.choice(("ocean currents","the solar system","volcanoes"))))
                        finished = asyncio.Event()
                        async def observe():
                            while not finished.is_set():
                                episode["native_frames"].append(await sample())
                                await asyncio.sleep(.02)
                        async def offer():
                            try:
                                await capture_chat(gateway,"/v1/chat/completions",body,uuid.uuid4().hex,
                                    time.monotonic(),enabled=True,timeout_s=20,
                                    record=episode["rows"].append)
                            finally:
                                finished.set()
                        observer, request = asyncio.create_task(observe()),asyncio.create_task(offer())
                        try:
                            done,_ = await asyncio.wait((observer,request),
                                                        return_when=asyncio.FIRST_EXCEPTION)
                            for task in done:
                                task.result()
                        finally:
                            if not request.done():
                                request.cancel()
                            await asyncio.gather(request,return_exceptions=True)
                            finished.set()
                            await observer
                            persist()
                        episode["final_drain"] = await drain()
                        row = episode["rows"][0]
                        if (row["status"]!="complete" or row["finish_reason"]!="stop"
                                or row["usage"]["prompt_tokens"]!=prompt):
                            raise ValueError("incomplete or truncated isolated outcome")
                        persist()
            report["completed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        persist()


if __name__=="__main__":
    asyncio.run(main())
