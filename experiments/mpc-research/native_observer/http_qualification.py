"""Bounded functional qualification, not training or a controller benchmark."""
import asyncio
from dataclasses import asdict
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path("/root/pr33-native-0588707")
sys.path.insert(0, str(ROOT))
import httpx
from transformers import AutoTokenizer
from mpc_lab.actuator import GuardedActuator, IdentificationPermit
from mpc_lab.contracts import CostSpec, Decision, ModelParameters, implementation_hash
from mpc_lab.http_capture import capture_chat, drain_evidence
from mpc_lab.runtime import local_clock_domain, require_guarded_capability, snapshot_from_dict

spec = importlib.util.spec_from_file_location(
    "preparation", ROOT / "experiments/mpc-research/native_id_preparation.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


async def main():
    cfg = json.loads(Path("/root/pr33-native-control.json").read_text())
    output = ROOT / "native-evidence/http-qualification.json"
    report = {"source_commit": "0588707949f651039d44a7a034c12bd35b8c8047",
              "implementation_hash": implementation_hash(), "scope": "functional-only",
              "clock_domain": local_clock_domain(), "actions": [], "rows": [],
              "native_engine_freshness_qualified": False, "performance_accepted": False}
    native_frames = []
    report["native_frames"] = native_frames
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], local_files_only=True)
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=10,
                                     headers={"x-admin-api-key": cfg["ADMIN_API_KEY"]}) as client:
            deadline = time.monotonic() + 30
            while True:
                try:
                    response = await client.get("/admin/mpc/state")
                    response.raise_for_status()
                    document = response.json()
                    break
                except httpx.HTTPError:
                    if time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(0.2)
            require_guarded_capability(document)
            initial = snapshot_from_dict(document)
            report["initial"] = document
            cost = CostSpec(**document["cost_spec"])
            if ";available=True;" not in cost.tokenizer_identity:
                raise ValueError("native tokenizer unavailable")
            params = ModelParameters(cost_spec=cost, model_version="native-id-unfitted-v1",
                                     calibration_version="unfitted-not-validated")
            if params.identity != initial.identity:
                raise ValueError("native identity mismatch")
            permit = IdentificationPermit(params.identity, approved=True, run_id=initial.run_id,
                                          clock_domain=local_clock_domain(),
                                          expires_at=time.monotonic() + 180)
            driver = GuardedActuator(client, params, ROOT / "native-evidence/guarded-pending.json",
                                     permit=permit, enabled=True, max_age_s=0.5)
            for budget in (32, 16):
                state = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                decision = Decision(budget, None, "identification_assignment", False,
                                    None, None, 0, 0, 0, params.identity)
                result = await driver.submit(state, decision)
                report["actions"].append(result)
                if result["status"] != "applied":
                    raise ValueError("identification assignment did not apply")

            planned = []
            start = time.monotonic() + 0.2
            for repetition in range(2):
                for target, maximum in ((128, 128), (128, 256), (4096, 128), (4096, 256)):
                    prefix = "Ignore the filler and answer only OK. Filler: "
                    base = len(tokenizer.apply_chat_template(
                        [{"role": "user", "content": prefix}], tokenize=True,
                        add_generation_prompt=True, return_dict=False))
                    text = prefix + " hello" * max(0, target - base)
                    messages = [{"role": "user", "content": text}]
                    actual = len(tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_dict=False))
                    if actual != target:
                        raise ValueError(f"prepared prompt {actual} differs from target {target}")
                    planned.append((uuid.uuid4().hex, start + len(planned) * 0.3, {
                        "model": "Qwen/Qwen3-4B-Instruct-2507", "messages": messages,
                        "temperature": 0, "max_tokens": maximum,
                    }, {"repetition": repetition, "nominal_prompt_tokens": target,
                        "prepared_chat_tokens": actual, "role": "functional"}))
            planned.append((uuid.uuid4().hex, start + len(planned) * 0.3, {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "messages": [{"role": "user", "content": "Reply OK."}],
                "temperature": 0, "max_tokens": 4096,
            }, {"role": "deterministic_admission_rejection"}))
            report["expected_ids"] = [p[0] for p in planned]

            async def run(item):
                rid, scheduled, payload, metadata = item
                row = await capture_chat(client, "/v1/chat/completions", payload, rid,
                                         scheduled, enabled=True,
                                         record=lambda r: report["rows"].append({**r, **metadata}))
                row.update(metadata)
                return row
            import pr33_native_observer
            observer_hash = hashlib.sha256(Path(pr33_native_observer.__file__).read_bytes()).hexdigest()
            native_run = None
            async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=5) as native:
                if (await native.get("/pr33-native/state")).status_code != 401:
                    raise ValueError("native observer authentication failed")
                async def sample():
                    nonlocal native_run
                    reply = await native.get("/pr33-native/state",
                                             headers={"x-observer-key": cfg["ADMIN_API_KEY"]})
                    reply.raise_for_status()
                    frame = reply.json()
                    if (frame["clock_domain"] != report["clock_domain"]
                            or frame["observer_sha256"] != observer_hash
                            or not frame["complete"] or frame["vllm_version"] != "0.30.0"):
                        raise ValueError("native observer identity or completeness failed")
                    if native_run is None:
                        native_run = frame["run_id"]
                    if frame["run_id"] != native_run:
                        raise ValueError("native engine epoch changed")
                    now = time.monotonic()
                    if not (0 <= now - frame["sampled_at"] <= 0.5
                            and 0 <= frame["frontend_sampled_at"] - frame["sampled_at"] <= 0.05):
                        raise ValueError("native observer sample stale or too wide")
                    native_frames.append({**frame, "received_at": now})
                    return frame
                finished = asyncio.Event()
                async def observe_requests():
                    while not finished.is_set():
                        await sample()
                        await asyncio.sleep(0.05)
                observer_task = asyncio.create_task(observe_requests())
                try:
                    await asyncio.gather(*(run(p) for p in planned))
                finally:
                    finished.set()
                    await observer_task
                report["drain_checks"] = []
                for _ in range(2):
                    gateway_state = snapshot_from_dict((await client.get("/admin/mpc/state")).json())
                    engine_state = await sample()
                    report["drain_checks"].append(drain_evidence(
                        gateway_state, engine_state, identity=params.identity,
                        gateway_run=initial.run_id, engine_run=native_run,
                        domain=report["clock_domain"], now=time.monotonic(),
                        pending_operation=driver.path.exists()))
                    await asyncio.sleep(0.05)
            report["native_engine_freshness_qualified"] = (
                any(frame["running"] > 0 for frame in native_frames)
                and all(check["drained"] for check in report["drain_checks"]))
            await asyncio.sleep(2)
            report["final"] = (await client.get("/admin/mpc/state")).json()
            report["metrics"] = (await client.get("/metrics")).json()
            report["http_audit"] = preparation.audit_http(
                report["rows"], report["expected_ids"], report["clock_domain"])
            report["gateway_empty"] = (not report["final"]["requests"]
                                       and report["metrics"]["current_in_flight"] == 0)
            report["prompt_tokens_match"] = all(
                row["prepared_chat_tokens"] == row["usage"]["prompt_tokens"]
                == row["nominal_prompt_tokens"] for row in report["rows"]
                if row["role"] == "functional" and row["status"] == "complete")
            report["functional_pass"] = (
                report["http_audit"]["valid"] and report["gateway_empty"]
                and report["prompt_tokens_match"]
                and all(r["status"] == "complete" for r in report["rows"]
                        if r["role"] == "functional")
                and all(r["status"] == "rejected" for r in report["rows"]
                        if r["role"] == "deterministic_admission_rejection"))
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        report["functional_pass"] = False
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report.get(k) for k in (
            "functional_pass", "http_audit", "gateway_empty", "error")}, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
