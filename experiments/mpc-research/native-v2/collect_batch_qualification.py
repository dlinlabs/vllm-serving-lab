"""Bounded standard-Chat batch/commit qualification, never an MPC benchmark."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from batch_capture_bridge import source_hash, validate_plan
from mpc_lab.contracts import implementation_hash
from mpc_lab.http_capture import capture_chat, drain_evidence
from mpc_lab.runtime import local_clock_domain, snapshot_from_dict


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_new(path, value):
    data = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def qualification_plan(commit):
    trials = []
    for seed in (170001, 170002):
        cells = [(p, n, mode) for p in (128, 4096) for n in (1, 4)
                 for mode in ("off", "light")]
        random.Random(seed).shuffle(cells)
        for prompt, cohort, mode in cells:
            tid = f"nv2-qual-{seed}-p{prompt}-n{cohort}-{mode}"
            trials.append(dict(trial_id=tid, seed=seed, observer=mode,
                prompt_tokens=prompt, cohort=cohort, max_tokens=128, words=28,
                request_ids=[hashlib.sha256(f"{tid}-{i}".encode()).hexdigest()[:32]
                             for i in range(cohort)]))
    plan = dict(schema="native-v2-batch-capture-plan-v1", partition="train",
        source_commit=commit, trials=trials, experiment_id="native-v3-async-batch-qualification",
        capture_audit_version="native-v3-async-terminal-accounting-v1",
        pod_id="24nm6zrh2sej8c", static_cost_budget=32,
        hypothesis="Existing PR32 passive capture can join normal Gateway Chat requests,"
            "scheduler work,worker spans and host commits on graph/async vLLM0.30."
            " Preserve strict v2 results;account separately for an independently proved"
            " pre-terminal one-placeholder executed suffix beyond delivered-token minimum.",
        workload="16 short drained arms;2 TRAIN seeds;p128/4096;cohort1/4;OFF/light;"
            "natural EOS with max128,not forced outputs. Startup/first-cell effects retained.",
        completion_rule="Stop on first invalid identity,failed outcome,overflow,source/"
            "clock change or drain failure. Timeout20s/request,20s/drain,10s/event resolution."
            " No performance equivalence,service calibration or MPC acceptance.",
        artifact_directory="native-v3-async-batch-qualification")
    validate_plan(plan)
    return plan


class CaptureClient:
    def __init__(self, client, directory, plan, domain):
        self.client, self.directory, self.plan, self.domain = client, directory, plan, domain
        self.plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
        self.run_id = None
        self.sequence = 0

    async def call(self, operation, **fields):
        reply = await self.client.post("/pr33-native/batch",
                                       json=dict(operation=operation, **fields))
        received = time.monotonic()
        self.sequence += 1
        write_new(self.directory/f"utility-{self.sequence:04}-{operation}.json",
                  dict(operation=operation, received_at=received, status=reply.status_code,
                       body=reply.text))
        reply.raise_for_status()
        value = reply.json()
        if (value.get("clock_domain") != self.domain
                or value.get("source_commit") != self.plan["source_commit"]
                or value.get("plan_sha256") != self.plan_sha
                or value.get("bridge_sha256") != source_hash()
                or (self.run_id is not None and value.get("run_id") != self.run_id)):
            raise ValueError("Capture source,plan,clock or epoch changed")
        if not isinstance(value.get("run_id"), str) or not value["run_id"]:
            raise ValueError("Missing capture epoch")
        self.run_id = value["run_id"]
        return value

    async def finish(self, trial_id):
        deadline = time.monotonic() + 10
        while True:
            result = await self.call("end")
            if result.get("status") != "gpu_events_pending":
                break
            if time.monotonic() >= deadline:
                raise ValueError("GPU events did not resolve within the frozen bound")
            await asyncio.sleep(.02)
        if (result.get("trial_id") != trial_id or "capture" not in result
                or hashlib.sha256(canonical(result["capture"])).hexdigest()
                != result.get("capture_sha256")):
            raise ValueError("Returned capture digest or trial differs")
        path = self.directory/f"{trial_id}-capture.json"
        write_new(path, result)
        sealed = json.loads(path.read_bytes())
        if hashlib.sha256(canonical(sealed["capture"])).hexdigest() != result["capture_sha256"]:
            raise ValueError("Persisted capture differs;do not acknowledge")
        await self.call("ack", trial_id=trial_id, capture_sha256=result["capture_sha256"])
        if result.get("capture_valid") is not True:
            raise ValueError("Invalid native capture retained and acknowledged,not accepted")
        return result


async def collect(plan, config_path, directory):
    validate_plan(plan)
    if plan != qualification_plan(plan["source_commit"]):
        raise ValueError("Only the frozen fixed32 qualification protocol is supported")
    directory.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(config_path.read_text())
    domain = local_clock_domain()
    from transformers import AutoTokenizer
    import pr33_native_observer
    sys.path.insert(0, str(ROOT/"experiments/mpc-research/native_observer"))
    from collect_interventions import messages
    from native_prediction import NATIVE_ID
    sys.path.insert(0, str(ROOT/"experiments/mpc-research/native-v3"))
    from audit_async_capture import audit_async_capture
    observer_sha = hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], local_files_only=True)
    write_new(directory/"manifest.json", dict(plan=plan, clock_domain=domain,
        implementation_hash=implementation_hash(), observer_sha256=observer_sha,
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Qualification only;OFF inside enabled hooks is not literal disabled."
              " Post-drain GPU labels are not online features. No overhead claim."))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=30,
            headers={"x-admin-api-key":cfg["ADMIN_API_KEY"]}) as gateway, \
            httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30,
            headers={"x-observer-key":cfg["ADMIN_API_KEY"]}) as native:
        initial = await gateway.get("/admin/mpc/state")
        initial.raise_for_status()
        initial = initial.json()
        identity = snapshot_from_dict(initial).identity
        engine_run, drain_sequence = None, 0
        if initial.get("budget") != 32:
            raise ValueError("Qualification requires an actually applied fixed32 budget")
        bridge = CaptureClient(native, directory, plan, domain)
        await bridge.call("metadata")

        async def drain():
            nonlocal engine_run, drain_sequence
            deadline, clear = time.monotonic() + 20, []
            while time.monotonic() < deadline:
                g, n = await gateway.get("/admin/mpc/state"), await native.get("/pr33-native/state")
                g.raise_for_status()
                n.raise_for_status()
                document, frame = g.json(), n.json()
                drain_sequence += 1
                write_new(directory/f"drain-{drain_sequence:04}.json",
                          dict(gateway=document, engine=frame, received_at=time.monotonic()))
                if (frame.get("clock_domain") != domain or document.get("clock_domain") != domain
                        or frame.get("observer_sha256") != observer_sha
                        or frame.get("complete") is not True):
                    raise ValueError("Unqualified native source/clock")
                engine_run = engine_run or frame["run_id"]
                proof = drain_evidence(snapshot_from_dict(document), frame,
                    identity=identity, gateway_run=initial["run_id"], engine_run=engine_run,
                    domain=domain, now=time.monotonic(), pending_operation=False)
                clear = clear + [proof] if proof["drained"] else []
                if len(clear) == 2:
                    return
                await asyncio.sleep(.05)
            raise ValueError("Gateway/frontend/native drain failed")

        for trial in plan["trials"]:
            await drain()
            payload = dict(model="Qwen/Qwen3-4B-Instruct-2507", temperature=0,
                max_tokens=trial["max_tokens"],
                messages=messages(tokenizer, trial["prompt_tokens"], trial["words"], "ocean currents"))
            await bridge.call("begin", trial_id=trial["trial_id"])
            rows, start = [], time.monotonic() + .1
            def record(row):
                rows.append(row)
                write_new(directory/f"{row['request_id']}-http.json", row)
            await asyncio.gather(*(capture_chat(gateway, "/v1/chat/completions", payload, rid,
                start, enabled=True, domain=domain, timeout_s=20, record=record)
                for rid in trial["request_ids"]))
            await drain()
            result = await bridge.finish(trial["trial_id"])
            if (len(rows) != trial["cohort"] or any(r["status"] != "complete"
                    or not isinstance(r.get("usage"), dict)
                    or r["usage"]["prompt_tokens"] != trial["prompt_tokens"]
                    or r["usage"]["completion_tokens"] <= 0 for r in rows)):
                raise ValueError("HTTP failures/rejections retained;qualification stopped")
            expected = {r["request_id"] for r in rows}
            actual = set(result["capture"].get("requests", {}))
            matches = [NATIVE_ID.fullmatch(rid) for rid in actual]
            if (any(m is None for m in matches) or len(actual) != len(expected)
                    or {m[1] for m in matches} != expected):
                raise ValueError("Gateway-to-native request identity requires qualification")
            audit = audit_async_capture(result["capture"], result["capture"]["workers"][0]["rows"],
                                       require_steps=trial["observer"] == "light")
            write_new(directory/f"{trial['trial_id']}-audit.json", audit)
            if audit["valid"] is not True:
                raise ValueError("Work/progress parity unqualified;raw capture retained")
        write_new(directory/"completed.json", dict(completed=True, experiment_id=plan["experiment_id"],
            source_commit=plan["source_commit"], trials=len(plan["trials"]),
            capture_audit_version=plan["capture_audit_version"],
            scope="V3 observed physical/delivered async work accounting,join/HTTP/drain only."
                  " Strict v2 results remain recorded;not performance or MPC acceptance."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-plan")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.make_plan:
        print(json.dumps(qualification_plan(args.make_plan), indent=2))
    else:
        if None in (args.plan, args.config, args.output):
            parser.error("--plan,--config and --output are required")
        asyncio.run(collect(json.loads(args.plan.read_text()), args.config, args.output))


if __name__ == "__main__":
    main()
