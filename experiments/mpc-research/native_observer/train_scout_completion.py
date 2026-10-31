"""Train-only saturation scout; not the frozen identification or policy test."""
import asyncio
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


def prompt(tokenizer, target, words):
    prefix = (f"Write approximately {words} words of factual prose about the solar system. "
              "Do not discuss the filler. Filler:")
    count = lambda text: len(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, return_dict=False))
    repeats = target - count(prefix)
    for _ in range(4):
        text = prefix + " hello" * repeats
        actual = count(text)
        if actual == target:
            return [{"role": "user", "content": text}]
        repeats += target - actual
    raise ValueError("could not prepare exact native token count")


def arrivals(seed, rate, duration):
    rng, tape, at = random.Random(seed), [], 0.0
    while True:
        at += rng.expovariate(rate)
        if at >= duration:
            return tape
        tape.append(at)


async def main():
    cfg = json.loads(Path("/root/pr33-native-control.json").read_text())
    out = ROOT / "native-evidence/train-saturation-scout-completion.json"
    domain = local_clock_domain()
    observer_hash = hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest()
    report = dict(scope="train-only rate/action-binding scout; no fitted model or performance acceptance",
                  source_commit="0588707949f651039d44a7a034c12bd35b8c8047",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  implementation_hash=implementation_hash(), observer_sha256=observer_hash,
                  clock_domain=domain, episodes=[], performance_accepted=False)
    predecessor = ROOT / "native-evidence/train-saturation-scout-v1.json"
    report["predecessor_sha256"] = hashlib.sha256(predecessor.read_bytes()).hexdigest()
    persist = lambda: out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], local_files_only=True)
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=30,
                headers={"x-admin-api-key": cfg["ADMIN_API_KEY"]}) as client, \
                httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=5) as native:
            initial = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
            document = (await client.get("/admin/mpc/state")).json()
            params = ModelParameters(cost_spec=CostSpec(**document["cost_spec"]),
                model_version="native-id-unfitted-v1", calibration_version="unfitted-not-validated")
            if initial.identity != params.identity:
                raise ValueError("gateway identity mismatch")
            driver = GuardedActuator(client, params, ROOT / "native-evidence/scout-pending.json",
                permit=IdentificationPermit(params.identity, approved=True, run_id=initial.run_id,
                    clock_domain=domain, expires_at=time.monotonic()+600),
                enabled=True, max_age_s=0.5)
            pending = driver.path
            report["pending_before_reconciliation"] = json.loads(pending.read_text()) if pending.exists() else None
            report["reconciliation"] = await driver.reconcile()
            if report["reconciliation"]["status"] not in ("clear", "expired", "fenced"):
                raise ValueError("old assignment has no verified fence")
            engine_run = None
            async def sample():
                nonlocal engine_run
                response = await native.get("/pr33-native/state",
                    headers={"x-observer-key": cfg["ADMIN_API_KEY"]})
                response.raise_for_status()
                frame = response.json()
                if (frame["clock_domain"] != domain or frame["observer_sha256"] != observer_hash
                        or not frame["complete"] or frame["vllm_version"] != "0.30.0"):
                    raise ValueError("observer identity or completeness failed")
                engine_run = engine_run or frame["run_id"]
                if frame["run_id"] != engine_run:
                    raise ValueError("engine epoch changed")
                now = time.monotonic()
                if not (0 <= now-frame["sampled_at"] <= .5
                        and 0 <= frame["frontend_sampled_at"]-frame["sampled_at"] <= .05):
                    raise ValueError("observer stale or sample too wide")
                return {**frame, "received_at": now}

            async def drain():
                checks = []
                deadline = time.monotonic()+20
                while time.monotonic() < deadline:
                    state = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                    frame = await sample()
                    evidence = drain_evidence(state, frame, identity=params.identity,
                        gateway_run=initial.run_id, engine_run=engine_run, domain=domain,
                        now=time.monotonic(), pending_operation=driver.path.exists())
                    checks = checks+[evidence] if evidence["drained"] else []
                    if len(checks) == 2:
                        return checks
                    await asyncio.sleep(.05)
                raise ValueError("could not establish two fresh drain samples")

            cells = [(4096,256,16)]
            report["frozen_scout_cells"] = cells
            for index, (p, m, rate) in enumerate(cells, start=3):
                messages = prompt(tokenizer, p, 30 if m == 128 else 80)
                tape = arrivals(159901+index, rate, 12)
                report.setdefault("offered_tapes", {})[str(159901+index)] = tape
                for budget in ((16,32) if index % 2 == 0 else (32,16)):
                    episode = dict(seed=159901+index, prompt_tokens=p, max_tokens=m,
                        arrival_rate=rate, offers_s=12, budget=budget, rows=[], native_frames=[])
                    report["episodes"].append(episode)
                    episode["initial_drain"] = await drain()
                    state = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                    episode["action"] = await driver.submit(state, Decision(
                        budget,None,"identification_assignment",False,None,None,0,0,0,params.identity))
                    if episode["action"]["status"] != "applied":
                        raise ValueError("scout fixed-budget assignment did not apply")
                    episode["initial"] = (await client.get("/admin/mpc/state")).json()
                    start = time.monotonic()+.2
                    episode["start"] = start
                    stopped = asyncio.Event()
                    async def observe():
                        while not stopped.is_set():
                            episode["native_frames"].append(await sample())
                            await asyncio.sleep(.05)
                    observer_task = asyncio.create_task(observe())
                    payload = dict(model="Qwen/Qwen3-4B-Instruct-2507",messages=messages,
                                   temperature=0,max_tokens=m)
                    try:
                        await asyncio.gather(*(capture_chat(client,"/v1/chat/completions",
                            payload,uuid.uuid4().hex,start+at,enabled=True,timeout_s=20,
                            record=lambda row: episode["rows"].append(row)) for at in tape))
                    finally:
                        stopped.set()
                        await observer_task
                        persist()
                    episode["final_drain"] = await drain()
                    episode["final"] = (await client.get("/admin/mpc/state")).json()
                    episode["metrics"] = (await client.get("/metrics")).json()
                    rows = episode["rows"]
                    completed = [r for r in rows if r["status"] == "complete"]
                    if (len(rows) != len(tape) or len({r["request_id"] for r in rows}) != len(tape)
                            or any(r["status"] not in ("complete","rejected") for r in rows)
                            or any(r["usage"]["prompt_tokens"] != p for r in completed)):
                        raise ValueError("scout HTTP cohort incomplete or token shape differs")
                    ttft = sorted(r["first_content_at"]-r["scheduled_at"] for r in completed)
                    good = sum(t <= .5 for t in ttft)
                    summary = dict(offered=len(tape),complete=len(completed),
                        rejected=sum(r["status"]=="rejected" for r in rows),ttft_good=good,
                        cohort_ttft_good_per_offering_s=good/12,
                        ttft_p95_s=ttft[min(len(ttft)-1,int(.95*len(ttft)))] if ttft else None,
                        natural_eos=sum(r["finish_reason"]=="stop" for r in completed),
                        completion_tokens=sorted(r["usage"]["completion_tokens"] for r in completed),
                        max_running=max(f["running"] for f in episode["native_frames"]),
                        max_waiting=max(f["waiting"] for f in episode["native_frames"]))
                    episode["summary"] = summary
                    print(json.dumps({k:v for k,v in {**episode,**summary}.items()
                        if k in ("seed","budget","prompt_tokens","max_tokens","offered","complete",
                                 "rejected","ttft_good","ttft_p95_s","natural_eos","max_running")}),flush=True)
                    persist()
            report["completed"] = True
    except Exception as exc:
        report["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        persist()


if __name__ == "__main__":
    asyncio.run(main())
