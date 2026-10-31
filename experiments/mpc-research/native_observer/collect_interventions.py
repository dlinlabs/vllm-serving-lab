"""Bounded Train-only native intervention collection, not MPC activation."""
import asyncio
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path("/root/pr33-native-0588707")
sys.path.insert(0, str(ROOT))
import httpx
from transformers import AutoTokenizer
import pr33_native_observer
from mpc_lab.actuator import GuardedActuator, IdentificationPermit
from mpc_lab.contracts import CostSpec, Decision, ModelParameters, implementation_hash
from mpc_lab.http_capture import capture_chat, drain_evidence
from mpc_lab.runtime import local_clock_domain, snapshot_from_dict


def offered_tape(seed, reference, duration=90):
    rng, tape = random.Random(seed), []
    for start, end, multiple in ((0,30,.6),(30,60,1.1),(60,duration,.6)):
        at = float(start)
        while True:
            at += rng.expovariate(reference*multiple)
            if at >= end:
                break
            tape.append(at)
    return tape


def messages(tokenizer, target, words, topic):
    prefix = f"Write approximately {words} words of factual prose about {topic}. Ignore the filler. Filler:"
    def count(text):
        return len(tokenizer.apply_chat_template([{"role":"user","content":text}],
            tokenize=True,add_generation_prompt=True,return_dict=False))
    repeats = max(0,target-count(prefix))
    for _ in range(4):
        text = prefix+" hello"*repeats
        n = count(text)
        if n == target:
            return [{"role":"user","content":text}]
        repeats += target-n
    raise ValueError("prompt token preparation failed")


async def main(args):
    if not 160001 <= args.first_seed <= args.last_seed <= 160012:
        raise ValueError("only the frozen Train seed range is executable")
    if len(Path(args.output_directory).parts) != 1 or Path(args.output_directory).is_absolute():
        raise ValueError("output must be a named directory inside the native deployment")
    directory = ROOT/args.output_directory
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"interventions.json").exists():
        raise ValueError("existing intervention evidence must not be overwritten")
    cfg = json.loads(Path("/root/pr33-native-control.json").read_text())
    protocol_path = ROOT/"results/mpc-native-integration-cpu/identification-protocol-v2.json"
    protocol = json.loads(protocol_path.read_text())
    domain = local_clock_domain()
    sha = hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest()
    references = ({"p128-m128":43.0,"p128-m256":14.0,"p4096-m128":3.0,"p4096-m256":2.25}
        if args.reference=="scout32" else
        {"p128-m128":307/12,"p128-m256":94/12,"p4096-m128":27/12,"p4096-m256":18/12})
    report = dict(schema="native-train-intervention-v1",scope="TRAIN only; no MPC or calibration acceptance",
        source_commit="0588707949f651039d44a7a034c12bd35b8c8047",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_hash=implementation_hash(),observer_sha256=sha,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        rate_reference=references,rate_reference_scope="rough Train scout cohort capacity, not window throughput",
        reference_budget=32 if args.reference=="scout32" else 16,
        phases=[[0,30,.6],[30,60,1.1],[60,90,.6]],episodes=[],clock_domain=domain,
        performance_accepted=False,calibration_accepted=False)
    output = directory/"interventions.json"
    def persist():
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"],local_files_only=True)
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8080",timeout=30,
            headers={"x-admin-api-key":cfg["ADMIN_API_KEY"]}) as client, \
            httpx.AsyncClient(base_url="http://127.0.0.1:8000",timeout=5) as native:
            deadline = time.monotonic()+180
            while True:
                try:
                    reply = await native.get("/pr33-native/state",headers={"x-observer-key":cfg["ADMIN_API_KEY"]})
                    reply.raise_for_status()
                    break
                except httpx.HTTPError:
                    if time.monotonic() > deadline:
                        raise
                    await asyncio.sleep(2)
            doc = (await client.get("/admin/mpc/state")).json()
            initial = snapshot_from_dict(doc)
            params = ModelParameters(cost_spec=CostSpec(**doc["cost_spec"]),
                model_version="native-id-unfitted-v1",calibration_version="unfitted-not-validated")
            if initial.identity != params.identity:
                raise ValueError("gateway identity mismatch")
            driver = GuardedActuator(client,params,directory/"actuator-pending.json",
                permit=IdentificationPermit(params.identity,approved=True,run_id=initial.run_id,
                    clock_domain=domain,expires_at=time.monotonic()+1200),
                enabled=True,max_age_s=.5)
            engine_run = None
            async def sample():
                nonlocal engine_run
                gateway_reply = await client.get("/admin/mpc/state")
                gateway_reply.raise_for_status()
                gateway_received_at = time.monotonic()
                gateway_document = gateway_reply.json()
                if (gateway_document.get("clock_domain") != domain
                        or gateway_document.get("run_id") != initial.run_id):
                    raise ValueError("gateway sample clock or epoch changed")
                r = await native.get("/pr33-native/state",headers={"x-observer-key":cfg["ADMIN_API_KEY"]})
                r.raise_for_status()
                f = r.json()
                if f["clock_domain"] != domain or f["observer_sha256"] != sha or not f["complete"]:
                    raise ValueError("native source/clock/completeness mismatch")
                engine_run = engine_run or f["run_id"]
                if f["run_id"] != engine_run:
                    raise ValueError("native epoch changed")
                now = time.monotonic()
                if not (0 <= now-f["sampled_at"] <= .5
                        and 0 <= f["frontend_sampled_at"]-f["sampled_at"] <= .05):
                    raise ValueError("native stale or wide")
                if not 0 <= now-gateway_document["sampled_at"] <= .5:
                    raise ValueError("gateway sample stale")
                return {**f,"received_at":now,"gateway_snapshot":gateway_document,
                        "gateway_received_at":gateway_received_at}
            async def drain():
                checks, end = [],time.monotonic()+20
                while time.monotonic() < end:
                    s = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                    f = await sample()
                    e = drain_evidence(s,f,identity=params.identity,gateway_run=initial.run_id,
                        engine_run=engine_run,domain=domain,now=time.monotonic(),
                        pending_operation=driver.path.exists())
                    checks = checks+[e] if e["drained"] else []
                    if len(checks) == 2:
                        return checks
                    await asyncio.sleep(.05)
                raise ValueError("native drain not established")
            async def assign(budget, attempts):
                for _ in range(3):
                    fence = await driver.reconcile()
                    if fence["status"] == "pending":
                        await asyncio.sleep(.11)
                        continue
                    s = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                    result = await driver.submit(s,Decision(budget,None,"identification_assignment",
                        False,None,None,0,0,0,params.identity))
                    attempts.append({"fence":fence,"reply":result,"received_at":time.monotonic()})
                    if result["status"] == "applied":
                        return attempts
                    await asyncio.sleep(.11)
                raise ValueError("assignment could not be acknowledged and fenced")
            tapes, payloads = {},{}
            selected = [e for e in protocol["episodes"] if e["split"]=="train"
                        and args.first_seed <= e["seed_block"] <= args.last_seed]
            for e in selected:
                seed = e["seed_block"]
                if seed in tapes:
                    continue
                tapes[seed] = offered_tape(seed,references[e["cell"]])
                rng = random.Random(f"native-prompts:{seed}")
                topics = ("the solar system","ocean currents","plant photosynthesis","volcanoes")
                payloads[seed] = [dict(model="Qwen/Qwen3-4B-Instruct-2507",
                    messages=messages(tokenizer,e["prompt_tokens"],
                        rng.choice((28,32,36) if e["max_tokens"]==128 else (65,80,95)),
                        rng.choice(topics)),temperature=0,max_tokens=e["max_tokens"])
                    for _ in tapes[seed]]
            report["offered_tapes"] = tapes
            report["prompt_recipe"] = {"topics":["the solar system","ocean currents","plant photosynthesis","volcanoes"],
                "words_short":[28,32,36],"words_long":[65,80,95],"seed_rule":"native-prompts:<seed>"}
            persist()
            for specification in selected:
                e = {**specification,"rows":[],"native_frames":[],"actions":[]}
                report["episodes"].append(e)
                e["initial_drain"] = await drain()
                e["preassignment"] = []
                await assign(e["schedule"][0]["budget"], e["preassignment"])
                e["initial"] = (await client.get("/admin/mpc/state")).json()
                start = time.monotonic()+.5
                e["start"] = start
                physical_ids = [uuid.uuid4().hex for _ in tapes[e["seed_block"]]]
                e["expected_ids"] = physical_ids
                persist()
                finished = asyncio.Event()
                async def observe():
                    while not finished.is_set():
                        e["native_frames"].append(await sample())
                        await asyncio.sleep(.05)
                async def controls():
                    for action in e["schedule"][1:]:
                        await asyncio.sleep(max(0,start+action["at_s"]-time.monotonic()))
                        receipt = {**action,"attempts":[]}
                        e["actions"].append(receipt)
                        await assign(action["budget"], receipt["attempts"])
                async def offers():
                    await asyncio.gather(*(capture_chat(client,"/v1/chat/completions",payload,rid,
                        start+at,enabled=True,timeout_s=20,
                        record=lambda row:e["rows"].append(row))
                        for rid,at,payload in zip(physical_ids,tapes[e["seed_block"]],payloads[e["seed_block"]])))
                observer_task = asyncio.create_task(observe())
                control_task = asyncio.create_task(controls())
                offer_task = asyncio.create_task(offers())
                async def finish_work():
                    try:
                        await asyncio.gather(control_task,offer_task)
                    finally:
                        finished.set()
                work_task = asyncio.create_task(finish_work())
                try:
                    done,_ = await asyncio.wait((work_task,observer_task),
                                                return_when=asyncio.FIRST_EXCEPTION)
                    for task in done:
                        task.result()
                finally:
                    for task in (control_task,offer_task):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(control_task,offer_task,return_exceptions=True)
                    finished.set()
                    if not work_task.done():
                        work_task.cancel()
                    await asyncio.gather(work_task,return_exceptions=True)
                    await observer_task
                    persist()
                e["final_drain"] = await drain()
                e["final"] = (await client.get("/admin/mpc/state")).json()
                e["metrics"] = (await client.get("/metrics")).json()
                if len(e["rows"]) != len(physical_ids):
                    raise ValueError("offered cohort incomplete")
                completed = [r for r in e["rows"] if r["status"]=="complete"]
                e["summary"] = dict(offered=len(physical_ids),complete=len(completed),
                    rejected=sum(r["status"]=="rejected" for r in e["rows"]),
                    other=sum(r["status"] not in ("complete","rejected") for r in e["rows"]),
                    ttft_good=sum(r["first_content_at"]-r["scheduled_at"] <= .5 for r in completed),
                    natural_eos=sum(r["finish_reason"]=="stop" for r in completed))
                persist()
                print(e["episode_id"],json.dumps(e["summary"]),flush=True)
                if e["summary"]["other"] or e["summary"]["natural_eos"] != len(completed):
                    raise ValueError("incomplete or truncated native outcome; retained, not accepted")
            report["completed"] = True
    except Exception as exc:
        report["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        persist()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed",type=int,default=160005)
    parser.add_argument("--last-seed",type=int,default=160008)
    parser.add_argument("--reference",choices=("scout16","scout32"),default="scout16")
    parser.add_argument("--output-directory",default="native-id-low-train")
    asyncio.run(main(parser.parse_args()))
