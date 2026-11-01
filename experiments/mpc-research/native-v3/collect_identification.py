"""Prespecified native interventions, not fitted MPC or a superiority benchmark."""
import argparse
import asyncio
from datetime import datetime, timezone
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


def protocol_for(source, cost_spec, smoke=False, completed_blocks=0,
                 first_train_block=0, gateway_ledger=False):
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("exact source commit required")
    count, duration = (1, 12) if smoke else (40, 120)
    if type(completed_blocks) is not int or not 0 <= completed_blocks < 16 or (
            smoke and completed_blocks):
        raise ValueError("continuation may skip only complete Train seed blocks")
    if (type(first_train_block) is not int or not 0 <= first_train_block < 16
            or first_train_block and (smoke or completed_blocks)):
        raise ValueError("fresh Train start cannot assert completed blocks or skip calibration")
    if type(gateway_ledger) is not bool:
        raise ValueError("gateway ledger capture must be boolean")
    cells = ("p128-m128", "mixed") if smoke else tuple(CELLS)
    episodes = []
    for block in range(first_train_block or completed_blocks, count):
        seed = (180001 if smoke else 181001) + block
        split = "train" if smoke or block < 16 else "calibration" if block < 36 else "validation"
        for cell in cells:
            rng = random.Random(f"action-sequence:{seed}:{cell}")
            actions = [rng.choice((16, 32)) for _ in range(duration)]
            order = [0, 1]
            random.Random(f"arm-order:{seed}:{cell}").shuffle(order)
            for arm in order:
                episodes.append(dict(episode_id=f"nv3-{seed}-{cell}-arm{arm}",
                    seed_block=seed, split=split, cell=cell, duration_s=duration, arm=arm,
                    schedule=[dict(at_s=t, budget=48-a if arm else a)
                              for t, a in enumerate(actions)]))
    protocol = dict(schema=SCHEMA, source_commit=source, cost_spec=cost_spec,
        scope="Open-loop identification only; no native calibration or MPC acceptance",
        identity_scope="Legacy ModelParameters used ONLY for actuator identity; "
                       "default serial service coefficients are never evaluated.",
        split_unit="Independent arrival/action seed block; cells and arms are correlated",
        contrasts="Whole assigned sequences, not same-state pointwise causal differences",
        assignment_order="Independent prespecified randomized arm order per block/cell",
        completed_seed_blocks=completed_blocks,
        assignment_execution="Changed budgets require guarded applied ACK; unchanged budgets "
                             "require a fresh same-version observation of the prior applied ACK",
        gateway_epoch_episode_limit=2,
        epoch_scope="Drained paired-episode rollover; retain the production 1024-operation "
                    "idempotency ceiling; do not evict outcomes or restart native vLLM",
        observer_scope="Native observer samples; no CUDA timing or pristine-hook claim",
        final_test_used=False, planned_offered_window_s=sum(e["duration_s"] for e in episodes),
        minimum_requested_experiment_s=36000 if not smoke else None, episodes=episodes)
    if first_train_block:
        protocol.update(first_train_block=first_train_block,
            resumption_scope="Fresh source experiment; skipped blocks are NOT asserted complete. "
                             "Prior distinct valid exposure is retained and accounted separately.")
    if gateway_ledger:
        protocol["gateway_ledger_capture"] = True
    return protocol


def continuation_evidence(protocol, previous):
    count = protocol["completed_seed_blocks"]
    if not count or not isinstance(previous, dict):
        raise ValueError("continuation requires a prior complete-block index")
    expected = protocol_for(protocol["source_commit"], protocol["cost_spec"])["episodes"][:count*10]
    if (previous.get("schema") != SCHEMA
            or previous.get("actual_offered_window_s") != count*1200
            or [e.get("episode_id") for e in previous.get("episodes", [])]
            != [e["episode_id"] for e in expected]):
        raise ValueError("previous index does not establish the complete prescribed blocks")
    return dict(previous_source_commit=previous["source_commit"],
                complete_seed_blocks=count, completed_offered_window_s=count*1200,
                scope="Only complete blocks skipped; failed partial evidence remains in prior run")


def held_budget_evidence(snapshot, acknowledgment, parameters, domain, now):
    from mpc_lab.controller import validate_snapshot

    reason = validate_snapshot(snapshot, now=now, clock_domain=domain,
        expected_identity=parameters.identity, max_age_s=.5)
    if reason:
        raise ValueError(reason)
    if (snapshot.run_id != acknowledgment["run_id"]
            or snapshot.clock_domain != acknowledgment["clock_domain"]
            or snapshot.budget_version != acknowledgment["budget_version"]
            or snapshot.budget != acknowledgment["budget"]
            or snapshot.budget not in parameters.supported_budgets):
        raise ValueError("held budget changed or lost its applied provenance")
    if any(mark.cost != parameters.cost_spec.estimate(mark.prompt_tokens, mark.max_tokens)
           for mark in snapshot.requests):
        raise ValueError("cost_spec_mismatch")
    if not acknowledgment["applied_at"] <= snapshot.sampled_at <= now:
        raise ValueError("held budget chronology differs")
    return dict(status="confirmed_hold", budget=snapshot.budget,
                run_id=snapshot.run_id, clock_domain=snapshot.clock_domain,
                budget_version=snapshot.budget_version, confirmed_at=now,
                reference_operation_id=acknowledgment["operation_id"],
                scope="Existing applied budget observed; no new command or ACK manufactured")


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
    continuation = None
    if protocol["completed_seed_blocks"]:
        if args.previous_index is None:
            raise ValueError("continuation requires the preserved previous index")
        raw_previous = args.previous_index.read_bytes()
        previous = json.loads(raw_previous)
        continuation = continuation_evidence(protocol, previous)
        if previous.get("model_identity") != parameters.identity.__dict__:
            raise ValueError("continuation cost/core/model identity differs")
        continuation["previous_index_sha256"] = hashlib.sha256(raw_previous).hexdigest()
    index = dict(schema=SCHEMA, source_commit=protocol["source_commit"],
        implementation_hash=implementation_hash(), protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_version=parameters.model_version, controller_version="prespecified-cap-sequence-v2-confirmed-holds",
        model_identity=parameters.identity.__dict__, clock_domain=domain,
        actual_offered_window_s=0, experiment_started_at=time.monotonic(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        gpu_cost_per_hour=None, gpu_cost_reason="Lifecycle credential unavailable; price not verified",
        gpu_environment=dict(model_snapshot=config["MODEL_NAME"],
            software={p:importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
            native_launch=json.loads(args.vllm_launch.read_text())),
        calibrated=False, mpc_active=False, continuation=continuation, episodes=[])
    persist(output/"index.json", index)
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], local_files_only=True)
    managed_gateway = None
    async def start_gateway():
        epoch = len(index.setdefault("gateway_epochs", []))+1
        evidence = output/f"gateway-epoch-{epoch:04d}"
        evidence.mkdir()
        with (evidence/"process.log").open("x") as stream:
            process = await asyncio.create_subprocess_exec(sys.executable,
                str(ROOT/"experiments/mpc-research/native-v2/launch_batch_service.py"),
                "gateway", "--config", str(args.config.resolve()), "--plan",
                str(args.gateway_plan.resolve()), "--python", sys.executable,
                "--evidence", str(evidence), "--identification-protocol",
                str(args.protocol.resolve()), stdout=stream, stderr=asyncio.subprocess.STDOUT)
        index["gateway_epochs"].append(dict(epoch=epoch, pid=process.pid,
            source_commit=protocol["source_commit"], evidence=evidence.name))
        persist(output/"index.json", index)
        return process
    async def stop_gateway():
        if managed_gateway is not None and managed_gateway.returncode is None:
            managed_gateway.terminate()
            try:
                await asyncio.wait_for(managed_gateway.wait(), 15)
            except TimeoutError:
                managed_gateway.kill()
                await managed_gateway.wait()
    if args.gateway_plan:
        managed_gateway = await start_gateway()
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=30,
        headers={"x-admin-api-key":config["ADMIN_API_KEY"]}) as client, \
        httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=5) as native:
        async def ready_gateway():
            for attempt in range(90):
                if managed_gateway is not None and managed_gateway.returncode is not None:
                    raise ValueError("owned gateway exited during startup")
                try:
                    reply = await client.get("/admin/mpc/state")
                    reply.raise_for_status()
                    snapshot = snapshot_from_dict(reply.json())
                    if snapshot.identity != parameters.identity or snapshot.clock_domain != domain:
                        raise ValueError("gateway identity or clock mismatch")
                    return snapshot
                except httpx.HTTPError:
                    if attempt == 89:
                        raise
                    await asyncio.sleep(1)
            raise ValueError("gateway readiness not established")
        expires_at = time.monotonic()+protocol["planned_offered_window_s"]+14400
        class MeasuredActuator(GuardedActuator):
            def _save(self, command):
                started = time.monotonic()
                super()._save(command)
                self.last_journal_timing = dict(started_at=started,
                    finished_at=time.monotonic(), not_after=command["not_after"])
        def actuator(snapshot):
            return MeasuredActuator(client, parameters, output/"actuator-pending.json",
                permit=IdentificationPermit(parameters.identity, approved=True, run_id=snapshot.run_id,
                    clock_domain=domain, expires_at=expires_at), enabled=True, max_age_s=.5)
        try:
            initial = await ready_gateway()
            driver = actuator(initial)
        except BaseException:
            await stop_gateway()
            raise
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
                driver.last_journal_timing = None
                result = await driver.submit(snapshot, Decision(budget, None, "identification_assignment",
                    False, None, None, 0, 0, 0, parameters.identity))
                attempt = dict(fence=fence, reply=result, received_at=time.monotonic(),
                               journal_timing=driver.last_journal_timing)
                if result["status"] == "pending" and driver.path.exists():
                    attempt["retained_pending_command"] = json.loads(driver.path.read_text())
                attempts.append(attempt)
                if result["status"] == "applied":
                    return result["acknowledgment"]
                await asyncio.sleep(.11)
            raise ValueError("bounded assignment failed; no unacknowledged action accepted")
        try:
            for position, specification in enumerate(protocol["episodes"]):
                limit = protocol["gateway_epoch_episode_limit"]
                if args.gateway_plan and position and position % limit == 0:
                    await drain()
                    await stop_gateway()
                    managed_gateway = await start_gateway()
                    initial = await ready_gateway()
                    driver = actuator(initial)
                episode = {**specification, "rows":[], "native_frames":[], "actions":[],
                           "held_assignments":[],
                           "gateway_run_id":initial.run_id}
                episode_path = output/(specification["episode_id"]+".json")
                episode["initial_drain"] = await drain()
                episode["preassignment"] = []
                first_ack = await assign(specification["schedule"][0]["budget"], episode["preassignment"])
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
                    last_ack = first_ack
                    for action in specification["schedule"][1:]:
                        await asyncio.sleep(max(0, episode["start"]+action["at_s"]-time.monotonic()))
                        if action["budget"] == last_ack["budget"]:
                            if driver.path.exists():
                                raise ValueError("pending command cannot establish a held budget")
                            snapshot = await driver._state()
                            hold = {**action, **held_budget_evidence(snapshot, last_ack,
                                parameters, domain, time.monotonic()),
                                "state_observation":driver.last_state_observation}
                            episode["held_assignments"].append(hold)
                            event("held_assignment", hold)
                            continue
                        receipt = {**action, "attempts":[]}
                        episode["actions"].append(receipt)
                        last_ack = await assign(action["budget"], receipt["attempts"])
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
            await stop_gateway()
            index["experiment_finished_at"] = time.monotonic()
            index["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            persist(output/"index.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vllm-launch", type=Path)
    parser.add_argument("--gateway-plan", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--completed-blocks", type=int, default=0)
    parser.add_argument("--first-train-block", type=int, default=0)
    parser.add_argument("--gateway-ledger", action="store_true")
    parser.add_argument("--previous-index", type=Path)
    args = parser.parse_args()
    if args.prepare:
        sys.path.insert(0, str(ROOT))
        import httpx
        config = json.loads(args.config.read_text())
        with httpx.Client(timeout=5) as client:
            response = client.get("http://127.0.0.1:8080/admin/mpc/state",
                headers={"x-admin-api-key":config["ADMIN_API_KEY"]})
            response.raise_for_status()
        protocol = protocol_for(args.source_commit, response.json()["cost_spec"],
                                args.smoke, args.completed_blocks,
                                args.first_train_block, args.gateway_ledger)
        with args.protocol.open("x") as stream:
            json.dump(protocol, stream, indent=2, allow_nan=False)
    else:
        protocol = json.loads(args.protocol.read_text())
        smoke = protocol["minimum_requested_experiment_s"] is None
        if protocol != protocol_for(protocol["source_commit"], protocol["cost_spec"], smoke,
                                    protocol["completed_seed_blocks"],
                                    protocol.get("first_train_block", 0),
                                    protocol.get("gateway_ledger_capture", False)):
            raise ValueError("protocol not identical to prespecified factory")
        if args.gateway_plan is None:
            raise ValueError("campaign requires prespecified drained gateway epoch rollover")
        asyncio.run(run(args, protocol))


if __name__ == "__main__":
    main()
