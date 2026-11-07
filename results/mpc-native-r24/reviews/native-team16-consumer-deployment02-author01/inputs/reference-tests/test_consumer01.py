import asyncio
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest

import common01 as c
from consumer01 import CutConsumer
from fixture01 import SourceFixture
import ipc01 as ipc

SECRET = "source-fixture-not-a-real-credential-" + "0" * 28


def new(directory):
    return CutConsumer(directory, enabled=True, evidence_mode="cpu-source-fixture", redaction_secret=SECRET)


async def setup(monkeypatch, path, *, ghost=False):
    fixture = await SourceFixture(monkeypatch, path / "producer").initialize(ghost=ghost)
    owner = new(path / "consumer")
    try:
        await owner.start()
        owner.activate()
        for kind, evidence in (
            ("bootstrap_confirmed", fixture.case.startup),
            ("public_history_published", {"memory": sorted(fixture.case.public_memory)}),
            ("offer_intent", {"fixture_ordinal": 2})):
            owner.publish(kind, evidence)
    except BaseException:
        await owner.close(cancelled=True)
        raise
    return fixture, owner


def run(test):
    return asyncio.run(test())


async def finish(owner, **options):
    result = await owner.close(**options)
    c.write(owner.output / "returned-cleanup.json", result)
    return result


def test_default_off_has_no_file_or_helper(tmp_path):
    async def scenario():
        owner = CutConsumer(tmp_path / "must-not-exist")
        assert (await owner.start())["status"] == "disabled"
        assert owner.begin(ordinal=2, pending_at_trigger=True)["status"] == "disabled"
        assert (await owner.handoff(None))["status"] == "disabled"
        assert (await owner.close())["all_joined"]
        assert not owner.output.exists()
    run(scenario)


@pytest.mark.parametrize("ghost", [False, True])
def test_actual_consumer_fresh_clock_generation_and_exact_stock(monkeypatch, tmp_path, ghost):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path, ghost=ghost)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            source = await fixture.collect()
            result = await owner.handoff(source)
            assert result["generation"]["generation"] == 3
            assert source["current"]["journal"]["cut"]["owner_prefix"]["budget_version"] == 1
            clocks = result["clocks"]
            assert clocks["collection_boundary_started_at"] < clocks["collector_returned_at"]
            assert result["source"]["clocks"]["consumer_dequeue_at"] >= clocks["collector_returned_at"]
            assert clocks["handoff_return_guard_at"] < result["deadline"]
            stock = result["source"]["full_stock"]
            assert stock["base"]["running"] == stock["base"]["waiting"] == 0
            if ghost:
                assert stock["base"]["pending_batches"] == len(stock["queue_oldest_first"]) == 2
                assert len(stock["queued_ghost_request_ids"]) == 1 and stock["queue_oldest_first"][1]["work"] == []
            assert result["actual_application_at"] is result["planner_entry_at"] is result["decision_at"] is None
            assert not result["source"]["source_state_is_current_state"]
        finally:
            cleanup = await finish(owner)
        assert cleanup["component_pass"] and cleanup["all_joined"] and cleanup["postflight"]["status"] == "verified"
        assert not cleanup["events"]
    run(scenario)


@pytest.mark.parametrize("ordinal,pending", [(1, True), (2, False)])
def test_no_fallback_after_missing_designated_handoff(monkeypatch, tmp_path, ordinal, pending):
    async def scenario():
        _, owner = await setup(monkeypatch, tmp_path)
        try:
            with pytest.raises(ValueError, match="designated_ordinal2"):
                owner.begin(ordinal=ordinal, pending_at_trigger=pending)
            with pytest.raises(ValueError, match="no_fallback"):
                owner.begin(ordinal=2, pending_at_trigger=True)
        finally:
            result = await finish(owner)
        assert result["all_joined"] and not result["component_pass"]
    run(scenario)


@pytest.mark.parametrize("defect", ["source_hash", "epoch", "lifecycle", "publication", "prefix_bytes", "future_receipt", "stale"])
def test_source_unknown_and_stale_refuse(monkeypatch, tmp_path, defect):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path, ghost=True)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            source = await fixture.collect()
            cut = source["current"]["journal"]["cut"]
            if defect == "source_hash":
                cut["native_observation"]["native_phase"]["source_sha256"] = "0" * 64
            elif defect == "epoch":
                cut["common_clock_proof"]["native_run_id"] = "wrong-run"
            elif defect == "lifecycle":
                del cut["native_observation"]["native_phase"]["lifecycle"]
            elif defect == "publication":
                del cut["native_observation"]["native_phase"]["publications"]
            elif defect == "prefix_bytes":
                source["current"]["prefix_base64"] = ""
            elif defect == "future_receipt":
                source["current"]["journal"]["native_record"]["client_receive_at"] = time.monotonic() + 1
            else:
                cut["cut_at"] -= 1
            if defect in ("source_hash", "lifecycle", "publication"):
                raw = c.canonical(cut["native_observation"])
                source["current"]["journal"]["native_record"].update(
                    response_body_base64=base64.b64encode(raw).decode(), response_body_sha256=c.sha(raw))
            reasons = {"source_hash":"native_phase_exact_source_binding", "epoch":"source_clock_join",
                       "lifecycle":"missing_or_incomplete_phase_history",
                       "publication":"missing_or_incomplete_phase_history",
                       "prefix_bytes":"source_raw_inventory_binding",
                       "future_receipt":"receipt_after_permitted_cutoff", "stale":"source_deadline_before_boundary"}
            with pytest.raises(ValueError, match=reasons[defect]):
                await owner.handoff(source)
        finally:
            result = await finish(owner)
        assert result["all_joined"] and not result["component_pass"]
    run(scenario)


def test_exact_point_refusal_despite_valid_initial_interval(monkeypatch, tmp_path):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            with pytest.raises(ValueError, match="exact_application_time_unobserved"):
                await owner.handoff(await fixture.collect(), require_exact_application_time=True)
        finally:
            result = await finish(owner)
        assert result["all_joined"] and not result["component_pass"]
    run(scenario)


async def direct(owner, source, kind):
    returned = time.monotonic()
    owner.authority.tighten(source["current"]["journal"]["cut"]["cut_at"])
    lease = owner.authority.check(owner.lease["generation"])
    value = dict(schema="team16-cut-consumer-input-v1", namespace=c.NAMESPACE,
        run_id=owner.authority.run_id, ordinal=2, generation=owner.lease["generation"],
        boundary_started_at=owner.lease["begin"], active_deadline=lease["active_deadline"],
        deadline=lease["deadline"], collector_returned_at=returned, source=source,
        require_exact_application_time=False)
    if kind == "changed_generation":
        owner.publish("offer_outcome", {"observed_after_export": True})
    elif kind == "budget_as_generation":
        value["generation"] = source["current"]["journal"]["cut"]["owner_prefix"]["budget_version"]
    elif kind == "unknown_generation":
        value["generation"] = None
    elif kind == "namespace":
        value["run_id"] = "unowned-run"
    started = time.monotonic()
    raw = b"{" if kind == "malformed" else c.canonical(value)
    finished = time.monotonic()
    payload = ipc.frame(raw, sent=time.monotonic(), serial_started=started, serial_finished=finished)
    if kind == "partial":
        payload = payload[:ipc.HEADER.size + 10]
    elif kind == "duplicate":
        payload += payload
    elif kind == "oversized":
        payload = ipc.HEADER.pack(ipc.MAGIC, c.MAX_BYTES + 1, bytes(32), finished, started, finished)
    c.raw_write(owner.output / "adversarial-offered-frame.bin", payload)
    send_receipt = {}
    try:
        await ipc.write_async(owner.send_fd, payload, lease["deadline"], send_receipt)
    except BrokenPipeError:
        assert kind == "duplicate"
        send_receipt["bounded_rejection_during_duplicate_send"] = True
    c.write(owner.output / "adversarial-send.json", send_receipt)
    os.close(owner.send_fd)
    owner.send_fd = None
    result, _, _ = await ipc.receive_async(owner.recv_fd, time.monotonic() + .2, owner.received)
    c.write(owner.output / "adversarial-result.json", result)
    return result


@pytest.mark.parametrize("kind", ["changed_generation", "budget_as_generation", "unknown_generation", "namespace",
                                  "malformed", "partial", "duplicate", "oversized"])
def test_actual_worker_checks_authority_and_malformed_IPC(monkeypatch, tmp_path, kind):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            result = await direct(owner, await fixture.collect(), kind)
            assert result["status"] == "refused"
            if kind in ("changed_generation", "budget_as_generation"):
                assert result["reason"] == "caller_generation_changed"
        finally:
            cleanup = await finish(owner)
        assert cleanup["all_joined"] and not cleanup["component_pass"]
    run(scenario)


@pytest.mark.parametrize("phase", ["collection", "partial_dequeue", "result", "journal"])
def test_absolute_80ms_includes_every_boundary(monkeypatch, tmp_path, phase):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            source = await fixture.collect()
            if phase == "collection":
                await asyncio.sleep(.095)
            elif phase == "partial_dequeue":
                original = ipc.write_async
                async def delayed(fd, raw, deadline, receipt):
                    await original(fd, raw[:12], deadline, receipt)
                    await asyncio.sleep(.095)
                    await original(fd, raw[12:], deadline, receipt)
                monkeypatch.setattr(ipc, "write_async", delayed)
            elif phase == "result":
                original = ipc.receive_async
                async def delayed(*args):
                    value = await original(*args)
                    await asyncio.sleep(.095)
                    return value
                monkeypatch.setattr(ipc, "receive_async", delayed)
            else:
                original = c.write
                def delayed(path, value):
                    original(path, value)
                    if Path(path).name == "release-candidate.json":
                        time.sleep(.095)
                monkeypatch.setattr(c, "write", delayed)
            with pytest.raises((TimeoutError, ValueError, BrokenPipeError, EOFError)):
                await owner.handoff(source)
            assert owner.release is None
            if phase == "journal":
                assert (owner.output / "release-candidate.json").is_file()
            elif phase == "result":
                assert len(owner.received) > ipc.HEADER.size
        finally:
            result = await finish(owner)
        assert result["all_joined"] and not result["component_pass"]
    run(scenario)


def test_parent_rechecks_authoritative_change_after_worker_validation(monkeypatch, tmp_path):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path)
        original = owner._fresh
        calls = 0
        def changed():
            nonlocal calls
            calls += 1
            if calls == 5:
                owner.publish("public_history_published", {"changed_after_journal": True})
            return original()
        monkeypatch.setattr(owner, "_fresh", changed)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            with pytest.raises(ValueError, match="caller_generation_changed"):
                await owner.handoff(await fixture.collect())
        finally:
            result = await finish(owner)
        assert calls == 5 and result["all_joined"] and not result["component_pass"]
    run(scenario)


def test_cancel_blocked_consumer_and_cancelled_cleanup_still_join(monkeypatch, tmp_path):
    async def scenario():
        _, owner = await setup(monkeypatch, tmp_path)
        owner.begin(ordinal=2, pending_at_trigger=True)
        task = asyncio.create_task(owner.close(cancelled=True))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.closed and owner.cleanup["all_joined"]
        assert not owner.cleanup["component_pass"]
    run(scenario)


@pytest.mark.parametrize("field", ["pid", "start_ticks", "parent_pid", "pid_namespace", "clock_domain", "boot_id"])
def test_wrong_reported_consumer_identity_refuses_but_only_owned_PID_is_joined(monkeypatch, tmp_path, field):
    async def scenario():
        original = ipc.receive_async
        async def changed(*args):
            value, raw, header = await original(*args)
            if field in ("clock_domain", "boot_id"):
                value["identity"][field] = "linux-monotonic:unknown:time:[0]"
            elif field == "pid_namespace":
                value["identity"]["process"][field] = "pid:[0]"
            else:
                value["identity"]["process"][field] += 1000000000
            return value, raw, header
        monkeypatch.setattr(ipc, "receive_async", changed)
        owner = new(tmp_path / "consumer")
        try:
            with pytest.raises(ValueError, match="actual_consumer_ready_process_source_clock_binding"):
                await owner.start()
        finally:
            result = await finish(owner, cancelled=True)
        assert result["all_joined"] and len(result["owned_pid_start_ticks"]) == 1
    run(scenario)


@pytest.mark.parametrize("kind", ["unknown_status", "unknown_epoch", "stock_projection", "raw_inventory",
                                 "future_dequeue", "extra_field"])
def test_unknown_consumer_output_cannot_be_released(monkeypatch, tmp_path, kind):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path, ghost=True)
        original = ipc.receive_async
        async def changed(*args):
            value, raw, header = await original(*args)
            if kind == "unknown_status":
                value["status"] = "unknown"
            elif kind == "unknown_epoch":
                value["source_phases"]["native_run_id"] = "unknown"
            elif kind == "stock_projection":
                value["full_stock"]["queue_oldest_first"] = []
            elif kind == "raw_inventory":
                value["raw_inventory"]["current"]["prefix"]["sha256"] = "0" * 64
            elif kind == "future_dequeue":
                value["clocks"]["consumer_dequeue_at"] = time.monotonic() + 1
            else:
                value["unrecognized"] = True
            return value, raw, header
        monkeypatch.setattr(ipc, "receive_async", changed)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            with pytest.raises(ValueError):
                await owner.handoff(await fixture.collect())
        finally:
            result = await finish(owner)
        assert result["all_joined"] and not result["component_pass"]
    run(scenario)


@pytest.mark.parametrize("defect", ["zero_deadline", "malformed_postflight"])
def test_cleanup_refusal_retains_evidence_and_does_not_ignore_zero_deadline(monkeypatch, tmp_path, defect):
    async def scenario():
        _, owner = await setup(monkeypatch, tmp_path)
        os.close(owner.send_fd)
        owner.send_fd = None
        limit = time.monotonic() + 1
        try:
            while owner.process.poll() is None and time.monotonic() < limit:
                owner._drain()
                await asyncio.sleep(.005)
            assert owner.process.poll() is not None
            if defect == "zero_deadline":
                with pytest.raises(ValueError, match="cleanup_incomplete_or_late"):
                    await owner.close(hard_deadline=0)
                assert owner.cleanup["deadline"] == 0
            else:
                original = c.load
                def malformed(path):
                    if Path(path).name == "postflight.json":
                        return []
                    return original(path)
                monkeypatch.setattr(c, "load", malformed)
                await finish(owner)
                assert owner.cleanup["postflight"]["status"] == "refused"
            assert owner.closed and owner.cleanup["all_joined"] and not owner.cleanup["component_pass"]
            assert (owner.output / "component.json").is_file()
        finally:
            if not owner.closed:
                await owner.close(cancelled=True)
    run(scenario)


def test_final_cleanup_guard_refusal_cannot_become_pass_on_repeated_close(monkeypatch, tmp_path):
    async def scenario():
        fixture, owner = await setup(monkeypatch, tmp_path)
        try:
            owner.begin(ordinal=2, pending_at_trigger=True)
            await owner.handoff(await fixture.collect())
            original = c.require
            def refuse(value, reason):
                if reason == "cleanup_final_return_late":
                    raise ValueError("injected_final_guard_refusal")
                return original(value, reason)
            monkeypatch.setattr(c, "require", refuse)
            with pytest.raises(ValueError, match="injected_final_guard_refusal"):
                await owner.close()
            repeated = await owner.close()
            assert repeated["all_joined"] and not repeated["component_pass"]
            assert repeated["status"] != "complete"
            c.write(owner.output / "final-guard-negative.json", dict(
                fault="explicit_negative_final_guard_injection", repeated_close= repeated))
        finally:
            if not owner.closed:
                await owner.close(cancelled=True)
    run(scenario)
