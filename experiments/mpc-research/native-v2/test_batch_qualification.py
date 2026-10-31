import asyncio
import hashlib
import json
import os
import shutil
import sys

import httpx
import pytest

import collect_batch_qualification as collector
import launch_batch_service as launcher


def test_launcher_exposes_selected_interpreter_tools_to_native_subprocesses(tmp_path, monkeypatch):
    binary = tmp_path / "venv" / "bin"
    binary.mkdir(parents=True)
    ninja = binary / "ninja"
    ninja.write_text("#!/bin/sh\nexit 0\n")
    ninja.chmod(0o755)
    config, plan = tmp_path / "config.json", tmp_path / "plan.json"
    config.write_text(json.dumps({"MODEL_NAME": "test-model", "ADMIN_API_KEY": "test-key"}))
    plan.write_text(json.dumps(collector.qualification_plan("a" * 40)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", {**os.environ, "PATH": "/missing-system-tools"})
    monkeypatch.setattr(sys, "argv", ["launch", "vllm", "--config", str(config),
        "--plan", str(plan), "--python", str(binary / "python"),
        "--evidence", str(tmp_path / "evidence")])
    observed = []
    monkeypatch.setattr(os, "execv", lambda *args: observed.append(shutil.which("ninja")))
    launcher.main()
    assert observed == [str(ninja)]


@pytest.mark.parametrize("valid", [True, False])
def test_capture_is_on_disk_before_ack_even_when_invalid(tmp_path, valid):
    plan = collector.qualification_plan("a" * 40)
    trial = plan["trials"][0]["trial_id"]
    raw = {"errors": [] if valid else ["worker failure"], "retained": [1, 2]}
    digest = hashlib.sha256(collector.canonical(raw)).hexdigest()
    common = dict(clock_domain="test", source_commit=plan["source_commit"],
        plan_sha256=hashlib.sha256(collector.canonical(plan)).hexdigest(),
        bridge_sha256=collector.source_hash(), run_id="epoch")
    operations = []
    def handle(request):
        body = json.loads(request.content)
        operations.append(body["operation"])
        if body["operation"] == "ack":
            saved = json.loads((tmp_path/f"{trial}-capture.json").read_bytes())
            assert saved["capture"] == raw
            assert body["capture_sha256"] == digest
            value = dict(common, acknowledged=True)
        else:
            value = dict(common, trial_id=trial, capture=raw,
                         capture_sha256=digest, capture_valid=valid)
        return httpx.Response(200, json=value)
    async def run():
        async with httpx.AsyncClient(base_url="http://native",
                transport=httpx.MockTransport(handle)) as client:
            driver = collector.CaptureClient(client, tmp_path, plan, "test")
            if valid:
                assert (await driver.finish(trial))["capture"] == raw
            else:
                with pytest.raises(ValueError, match="Invalid native capture"):
                    await driver.finish(trial)
    asyncio.run(run())
    assert operations == ["end", "ack"]


@pytest.mark.parametrize("failure", ["digest", "epoch"])
def test_unqualified_reply_retained_without_ack(tmp_path, failure):
    plan = collector.qualification_plan("a" * 40)
    trial = plan["trials"][0]["trial_id"]
    raw = {"retained": "do not erase rejected evidence"}
    operations = []
    def handle(request):
        operations.append(json.loads(request.content)["operation"])
        return httpx.Response(200, json=dict(clock_domain="test",
            source_commit=plan["source_commit"],
            plan_sha256=hashlib.sha256(collector.canonical(plan)).hexdigest(),
            bridge_sha256=collector.source_hash(),
            run_id="changed" if failure == "epoch" else "epoch", trial_id=trial,
            capture=raw, capture_sha256="wrong", capture_valid=True))
    async def run():
        async with httpx.AsyncClient(base_url="http://native",
                transport=httpx.MockTransport(handle)) as client:
            driver = collector.CaptureClient(client, tmp_path, plan, "test")
            driver.run_id = "epoch"
            with pytest.raises(ValueError):
                await driver.finish(trial)
    asyncio.run(run())
    assert operations == ["end"]
    assert "do not erase" in next(tmp_path.glob("utility-*-end.json")).read_text()
    assert not list(tmp_path.glob("*-capture.json"))
