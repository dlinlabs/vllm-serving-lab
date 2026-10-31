"""Prespecified native interventions, not fitted MPC or a superiority benchmark."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "native-v3-state-response-id-v1"
CELLS = {
    "p128-m128": (128, 128, 43.0),
    "p128-m256": (128, 256, 14.0),
    "p4096-m128": (4096, 128, 10.0),
    "p4096-m256": (4096, 256, 3.75),
    "mixed": (None, None, 6.0),
}


def protocol_for(source, cost_spec, smoke=False):
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("exact source commit required")
    count, duration = (1, 12) if smoke else (40, 120)
    cells = ("p128-m128", "mixed") if smoke else tuple(CELLS)
    episodes = []
    for block in range(count):
        seed = (180001 if smoke else 181001) + block
        split = "train" if smoke or block < 16 else "calibration" if block < 36 else "validation"
        for cell in cells:
            rng = random.Random(f"action-sequence:{seed}:{cell}")
            actions = [rng.choice((16, 32)) for _ in range(duration)]
            for arm in (0, 1):
                episodes.append(dict(episode_id=f"nv3-{seed}-{cell}-arm{arm}",
                    seed_block=seed, split=split, cell=cell, duration_s=duration, arm=arm,
                    schedule=[dict(at_s=t, budget=48-a if arm else a)
                              for t, a in enumerate(actions)]))
    return dict(schema=SCHEMA, source_commit=source, cost_spec=cost_spec,
        scope="Open-loop identification only; no native calibration or MPC acceptance",
        identity_scope="Legacy ModelParameters used ONLY for actuator identity; "
                       "default serial service coefficients are never evaluated.",
        split_unit="Independent arrival/action seed block; cells and arms are correlated",
        contrasts="Whole assigned sequences, not same-state pointwise causal differences",
        observer_scope="Native observer samples; no CUDA timing or pristine-hook claim",
        final_test_used=False, planned_offered_window_s=sum(e["duration_s"] for e in episodes),
        minimum_requested_experiment_s=36000 if not smoke else None, episodes=episodes)


def tape_for(seed, cell, duration):
    if cell not in CELLS or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid cell or duration")
    rng, tape = random.Random(f"arrivals:{seed}:{cell}"), []
    for lo, hi, multiple in ((0, duration/3, .6),
                             (duration/3, 2*duration/3, 1.1),
                             (2*duration/3, duration, .6)):
        at = lo
        while True:
            at += rng.expovariate(CELLS[cell][2] * multiple)
            if at >= hi:
                break
            tape.append(at)
    return tape


def persist(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def summary(rows, start, duration):
    complete = [r for r in rows if r["status"] == "complete"]
    natural = [r for r in complete if r["finish_reason"] == "stop"]
    def good(row):
        output = row["usage"]["completion_tokens"]
        return (output > 1 and 0 <= row["first_content_at"]-row["scheduled_at"] <= .5
                and 0 <= (row["terminal_at"]-row["first_content_at"])/(output-1) <= .05)
    return dict(offered=len(rows), complete=len(complete), natural_eos=len(natural),
        capped=sum(r["finish_reason"] == "length" for r in complete),
        rejected=sum(r["status"] == "rejected" for r in rows),
        other=sum(r["status"] not in ("complete", "rejected") for r in rows),
        natural_cohort_good=sum(good(r) for r in natural),
        natural_in_window_good=sum(good(r) and r["terminal_at"] <= start+duration for r in natural),
        metric_scope="Identification labels, not a frozen MPC superiority outcome")


async def run(args, protocol):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT/"experiments/mpc-research/native_observer"))
    import httpx
    import importlib.metadata
    import pr33_native_observer
    from transformers import AutoTokenizer
    from collect_interventions import messages
    from mpc_lab.actuator import GuardedActuator, IdentificationPermit
    from mpc_lab.contracts import CostSpec, Decision, ModelParameters, implementation_hash
    from mpc_lab.http_capture import capture_chat, drain_evidence
    from mpc_lab.runtime import local_clock_domain, snapshot_from_dict

    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"results") or output.exists():
        raise ValueError("fresh output directory under results required")
    output.mkdir(parents=True, mode=0o700)
    config = json.loads(args.config.read_text())
    domain = local_clock_domain()
    observer_hash = hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest()
    parameters = ModelParameters(cost_spec=CostSpec(**protocol["cost_spec"]),
        model_version="native-id-unfitted-v1", calibration_version="unfitted-not-validated")
    index = dict(schema=SCHEMA, source_commit=protocol["source_commit"],
        implementation_hash=implementation_hash(), protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_version=parameters.model_version, controller_version="prespecified-cap-sequence-v1",
        model_identity=parameters.identity.__dict__, clock_domain=domain,
        actual_offered_window_s=0, experiment_started_at=time.monotonic(),
        gpu_cost_per_hour=None, gpu_cost_reason="Lifecycle credential unavailable; price not verified",
        gpu_environment=dict(model_snapshot=config["MODEL_NAME"],
            software={p:importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
            native_launch=json.loads(args.vllm_launch.read_text())),
        calibrated=False, mpc_active=False, episodes=[])
    persist(output/"index.json", index)
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], local_files_only=True)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=30,
        headers={"x-admin-api-key":config["ADMIN_API_KEY"]}) as client, \
        httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=5) as native:
        reply = await client.get("/admin/mpc/state")
        reply.raise_for_status()
        initial = snapshot_from_dict(reply.json())
        if initial.identity != parameters.identity or initial.clock_domain != domain:
            raise ValueError("gateway identity or clock mismatch")
        driver = GuardedActuator(client, parameters, output/"actuator-pending.json",
            permit=IdentificationPermit(parameters.identity, approved=True, run_id=initial.run_id,
                clock_domain=domain, expires_at=time.monotonic()+protocol["planned_offered_window_s"]+14400),
            enabled=True, max_age_s=.5)
        engine_run = None
        payload_cache = {}
        async def sample():
            nonlocal engine_run
            g = await client.get("/admin/mpc/state")
            g.raise_for_status()
            received = time.monotonic()
            gateway = g.json()
            r = await native.get("/pr33-native/state",
                headers={"x-observer-key":config["ADMIN_API_KEY"]})
            r.raise_for_status()
            frame = r.json()
            engine_run = engine_run or frame["run_id"]
            if (gateway["clock_domain"] != domain or frame["clock_domain"] != domain
                    or gateway["run_id"] != initial.run_id or frame["run_id"] != engine_run
                    or frame["observer_sha256"] != observer_hash):
                raise ValueError("source, clock or serving epoch changed")
            now = time.monotonic()
            available = (frame["complete"] and gateway["complete"]
                and 0 <= now-frame["sampled_at"] <= .5
                and 0 <= frame["frontend_sampled_at"]-frame["sampled_at"] <= .05
                and 0 <= now-gateway["sampled_at"] <= .5)
            return {**frame, "received_at":now, "gateway_snapshot":gateway,
                    "gateway_received_at":received, "available":available}
        async def drain():
            checks, end = [], time.monotonic()+30
            while time.monotonic() < end:
                frame = await sample()
                evidence = drain_evidence(snapshot_from_dict(frame["gateway_snapshot"]), frame,
                    identity=parameters.identity, gateway_run=initial.run_id, engine_run=engine_run,
                    domain=domain, now=time.monotonic(), pending_operation=driver.path.exists())
                checks = checks+[evidence] if evidence["drained"] else []
                if len(checks) == 2:
                    return checks
                await asyncio.sleep(.05)
            raise ValueError("verified native drain not established")
        async def assign(budget, attempts):
            for _ in range(3):
                fence = await driver.reconcile()
                if fence["status"] == "pending":
                    await asyncio.sleep(.11)
                    continue
                response = await client.get("/admin/mpc/state")
                response.raise_for_status()
                snapshot = snapshot_from_dict(response.json())
                result = await driver.submit(snapshot, Decision(budget, None, "identification_assignment",
                    False, None, None, 0, 0, 0, parameters.identity))
                attempts.append(dict(fence=fence, reply=result, received_at=time.monotonic()))
                if result["status"] == "applied":
                    return
                await asyncio.sleep(.11)
            raise ValueError("bounded assignment failed; no unacknowledged action accepted")
        try:
            for specification in protocol["episodes"]:
                episode = {**specification, "rows":[], "native_frames":[], "actions":[]}
                episode_path = output/(specification["episode_id"]+".json")
                episode["initial_drain"] = await drain()
                episode["preassignment"] = []
                await assign(specification["schedule"][0]["budget"], episode["preassignment"])
                tape = tape_for(specification["seed_block"], specification["cell"], specification["duration_s"])
                rng = random.Random(f"payloads:{specification['seed_block']}:{specification['cell']}")
                payloads = []
                for _ in tape:
                    cell = specification["cell"]
                    if cell == "mixed":
                        cell = rng.choice(("p128-m256", "p4096-m128"))
                    prompt, maximum, _ = CELLS[cell]
                    words = rng.choice((28, 32, 36) if maximum == 128 else (65, 80, 95))
                    topic = rng.choice(("the solar system", "ocean currents", "photosynthesis", "volcanoes"))
                    key = prompt, maximum, words, topic
                    if key not in payload_cache:
                        payload_cache[key] = messages(tokenizer, prompt, words, topic)
                    payloads.append(dict(model="Qwen/Qwen3-4B-Instruct-2507",
                        messages=payload_cache[key],
                        temperature=0, max_tokens=maximum))
                ids = [uuid.uuid4().hex for _ in tape]
                episode.update(start=time.monotonic()+.5, offered_tape=tape, expected_ids=ids)
                persist(episode_path, episode)
                event_stream = episode_path.with_suffix(".events.jsonl").open("x")
                last_sync = time.monotonic()
                def event(kind, value):
                    nonlocal last_sync
                    event_stream.write(json.dumps(dict(kind=kind, value=value), allow_nan=False)+"\n")
                    if time.monotonic()-last_sync >= 1:
                        event_stream.flush()
                        os.fsync(event_stream.fileno())
                        last_sync = time.monotonic()
                def record(row):
                    episode["rows"].append(row)
                    event("http_outcome", row)
                stop = asyncio.Event()
                async def observe():
                    while not stop.is_set():
                        frame = await sample()
                        episode["native_frames"].append(frame)
                        event("native_frame", frame)
                        await asyncio.sleep(.1)
                async def controls():
                    for action in specification["schedule"][1:]:
                        await asyncio.sleep(max(0, episode["start"]+action["at_s"]-time.monotonic()))
                        receipt = {**action, "attempts":[]}
                        episode["actions"].append(receipt)
                        await assign(action["budget"], receipt["attempts"])
                        event("action", receipt)
                async def offers():
                    await asyncio.gather(*(capture_chat(client, "/v1/chat/completions", payload, rid,
                        episode["start"]+at, enabled=True, timeout_s=30,
                        record=record) for rid, at, payload in zip(ids, tape, payloads)))
                    await asyncio.sleep(max(0, episode["start"]+specification["duration_s"]-time.monotonic()))
                async def gpu():
                    while not stop.is_set():
                        process = await asyncio.create_subprocess_exec("nvidia-smi",
                            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE)
                        stdout, _ = await process.communicate()
                        if process.returncode:
                            raise ValueError("GPU resource monitor unavailable")
                        event("gpu", dict(received_at=time.monotonic(), csv=stdout.decode().strip()))
                        await asyncio.sleep(5)
                tasks = [asyncio.create_task(fn()) for fn in (observe, controls, offers, gpu)]
                try:
                    work = asyncio.gather(tasks[1], tasks[2])
                    done, _ = await asyncio.wait((work, tasks[0], tasks[3]),
                        return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    stop.set()
                    episode["final_drain"] = await drain()
                    if len(episode["rows"]) != len(ids) or {r["request_id"] for r in episode["rows"]} != set(ids):
                        raise ValueError("all-offered cohort identity or completeness mismatch")
                    episode["summary"] = summary(episode["rows"], episode["start"], specification["duration_s"])
                    if episode["summary"]["other"]:
                        raise ValueError("failed HTTP evidence retained; not admitted as valid identification")
                    episode["completed_at"] = time.monotonic()
                    index["actual_offered_window_s"] += specification["duration_s"]
                    index["episodes"].append(dict(episode_id=specification["episode_id"],
                        split=specification["split"], summary=episode["summary"], artifact=episode_path.name))
                    print(specification["episode_id"], episode["summary"], flush=True)
                finally:
                    stop.set()
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    if not work.done():
                        work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
                    event_stream.flush()
                    os.fsync(event_stream.fileno())
                    event_stream.close()
                    persist(episode_path, episode)
                    persist(output/"index.json", index)
            index["completed"] = True
        except Exception as exc:
            index["error"] = type(exc).__name__+": "+str(exc)
            raise
        finally:
            index["experiment_finished_at"] = time.monotonic()
            persist(output/"index.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vllm-launch", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        sys.path.insert(0, str(ROOT))
        import httpx
        config = json.loads(args.config.read_text())
        with httpx.Client(timeout=5) as client:
            response = client.get("http://127.0.0.1:8080/admin/mpc/state",
                headers={"x-admin-api-key":config["ADMIN_API_KEY"]})
            response.raise_for_status()
        protocol = protocol_for(args.source_commit, response.json()["cost_spec"], args.smoke)
        with args.protocol.open("x") as stream:
            json.dump(protocol, stream, indent=2, allow_nan=False)
    else:
        protocol = json.loads(args.protocol.read_text())
        smoke = protocol["minimum_requested_experiment_s"] is None
        if protocol != protocol_for(protocol["source_commit"], protocol["cost_spec"], smoke):
            raise ValueError("protocol not identical to prespecified factory")
        asyncio.run(run(args, protocol))


if __name__ == "__main__":
    main()
