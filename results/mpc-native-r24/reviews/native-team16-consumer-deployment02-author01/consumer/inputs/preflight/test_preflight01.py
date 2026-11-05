"""Synthetic CPU controls and explicitly historical Native02 subset replay."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

import frozen01 as frozen
import preflight01 as api
from synthetic01 import DOMAIN, Fixture, rewrite


@pytest.fixture
def case(monkeypatch, tmp_path):
    value = Fixture(monkeypatch, tmp_path)
    asyncio.run(value.initialize())
    yield value
    value.preserve("synthetic-startup-inputs.json", value.startup)
    value.preserve("synthetic-journal.json", value.journal)


def test_current_source_route_really_refuses_and_proposed_route_is_not_live_authority():
    result = api.source_feasibility()
    assert result["current_route_refusal"] == "prepared_copy:prepared_copy_exclusive_route_required"
    assert result["source_phase_hooks_present"] and result["pinned_SDK_files"] == 38
    assert not result["installed_entrypoints_verified"] and not result["live_launcher_ready"]
    assert not result["native_execution_authorized"]
    assert not result["observer_source_shim_required"]


def test_default_off_never_touches_callbacks_and_cli_has_no_live_mode(capsys):
    async def disabled():
        return await api.collect_initial(native_get=lambda: pytest.fail("native callback"))
    assert asyncio.run(disabled())["status"] == "disabled"
    from run_preflight01 import main
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "disabled"
    with pytest.raises(SystemExit) as failure:
        main(["--native"])
    assert failure.value.code != 0


def test_actual_source_CAS_publication_phase_and_flushed_prefix_close_initial_cut(case):
    closure, report = asyncio.run(case.initial())
    assert closure.initial_sequence == 1
    assert closure.native_sources == frozen.sources()
    assert report["status"] == "initial_closure_contract_satisfied"
    assert report["cut"]["gateway_process_proof"]["owned_chain"][0]["pid"] == 200
    assert len(report["cut"]["native_process_proof"]["owned_chain"]) == 3
    assert report["confirmation"]["actual_application_interval"][1] < closure.capture_started_at
    assert report["observation"]["gateway_budget_version"] == 1
    assert report["acquisition_token"] is report["request_token"] is None
    assert report["actual_application_at"] is None and not report["native_mpc_permitted"]
    assert report["cut"]["owner_prefix"]["prefix_bytes"] > 0
    case.preserve("successful-synthetic-cut.json", report)


@pytest.mark.parametrize("change", ["phase", "publication", "lifecycle", "source", "capture", "epoch"])
def test_missing_or_changed_native_history_never_becomes_a_closed_cut(case, change):
    def mutate(body):
        phase = body["native_phase"]
        if change == "phase":
            del body["native_phase"]
        elif change == "publication":
            phase["publications"] = {}
        elif change == "lifecycle":
            del phase["lifecycle"]
        elif change == "source":
            phase["source_sha256"] = "0" * 64
        elif change == "capture":
            phase["capture"]["trial_id"] = "other"
        else:
            phase["run_id"] = "other"
    case.mutate_native = mutate
    with pytest.raises((ValueError, KeyError)):
        asyncio.run(case.initial())
    assert case.journal["status"] == "refused" and "native_record" in case.journal
    case.preserve("refused-native-history.json", case.journal)


@pytest.mark.parametrize("change", ["future_receipt", "unknown_interval", "clock", "SDK", "batch_epoch"])
def test_startup_receipts_metadata_and_capture_boundaries_are_not_invented(case, change):
    if change == "future_receipt":
        case.startup["gateway_confirm_record"]["client_receive_at"] = case.clock() + 2
    elif change == "unknown_interval":
        record = case.startup["cas_command_record"]
        body = frozen.control().body(record)
        body["applied_at"] = None
        rewrite(record, body)
    else:
        key = "capture_metadata_record" if change == "SDK" else "capture_begin_record"
        record = case.startup[key]
        body = frozen.control().body(record)
        if change == "clock":
            body["clock_domain"] = "unknown"
        elif change == "SDK":
            body["runtime"]["source"]["files"]["v1/engine/core.py"] = "0" * 64
        else:
            body["run_id"] = "other-batch"
        rewrite(record, body)
    with pytest.raises((ValueError, TypeError)):
        asyncio.run(case.initial())
    assert case.journal["status"] == "refused"
    assert not case.records[-1]["path"].endswith("phase-state")


def test_prewrite_guard_alone_passes_old_closure_but_not_full_receipt_boundary(case):
    ack = frozen.control().body(case.startup["cas_command_record"])
    upper = case.startup["cas_command_record"]["client_receive_at"]
    case.capture.started = (ack["applied_at"] + upper) / 2
    with pytest.raises(ValueError, match="confirmed_receipt_interval_must_precede_capture"):
        asyncio.run(case.initial())
    cut = case.journal["cut"]
    old = frozen.module("capture_owner_closure").establish_initial_capture_closure(
        cut, native_startup=case.startup["native_owned"], gateway_startup=case.startup["gateway_owned"])
    assert old.capture_started_at < upper
    case.preserve("prewrite-is-not-application-witness.json", case.journal)


@pytest.mark.parametrize("change", ["namespace", "reuse", "serving_child"])
def test_process_clock_and_actual_serving_PID_are_bound(case, change):
    if change == "namespace":
        case.proc.rows[111] = replace(case.proc.rows[111], time_namespace="time:[99]")
    elif change == "reuse":
        def mutate(_):
            case.proc.rows[111] = replace(case.proc.rows[111], start_ticks=31)
        case.mutate_native = mutate
    else:
        case.gateway_process["pid"] = 201
        case.gateway_process["start_ticks"] = 41
    with pytest.raises(ValueError, match="process_clock|gateway_serving_pid"):
        asyncio.run(case.initial())
    assert case.journal["status"] == "refused"
    case.preserve("process-refusal.json", case.journal)


@pytest.mark.parametrize("kind", ["legacy_pending", "zero_work", "ghost"])
def test_initial_empty_gateway_or_RUNNING_does_not_erase_native_stock(case, kind):
    record = case.startup["drain_records"][1]
    body = frozen.control().body(record)
    body["base"]["pending_batches"] = 1
    body["base"]["orphaned"] = None
    if kind != "legacy_pending":
        work = [] if kind == "zero_work" else [
            dict(request_id="ghost", position=9, tokens=1, prompt_length=None)]
        body["queue_oldest_first"] = [dict(step_id="c" * 64, work=work)]
        body["queued_ghost_request_ids"] = [] if kind == "zero_work" else ["ghost"]
    rewrite(record, body)
    with pytest.raises(ValueError, match="full_queue_count_conservation|physical_stock"):
        asyncio.run(case.initial())
    assert case.journal["status"] == "refused"


@pytest.mark.parametrize("kind", ["missing", "producer_loss"])
def test_missing_or_unflushed_ledger_cannot_establish_owner_prefix(case, monkeypatch, kind):
    if kind == "producer_loss":
        case.log.dropped = 1
    else:
        async def missing(_):
            case.log.records.clear()
        monkeypatch.setattr(case.log, "flush", missing)
    with pytest.raises((ValueError, OSError, case.httpx.HTTPStatusError)):
        asyncio.run(case.initial())
    assert case.journal["status"] == "refused"


def test_late_acquisition_retains_raw_receipt_without_returning_closure(case):
    def delayed(_):
        case.clock.advance(1)
    case.mutate_native = delayed
    with pytest.raises(ValueError, match="late_receipt|chronology|stale"):
        asyncio.run(case.initial())
    assert case.journal["native_record"]["client_receive_at"] > case.deadline
    assert case.journal["status"] == "refused"


def test_native_receipt_in_the_future_is_not_consumed_even_with_fresh_source(case, monkeypatch):
    original = case.native_record
    def future(phase=False):
        record = original(phase)
        if phase:
            record["client_receive_at"] += 0.1
        return record
    monkeypatch.setattr(case, "native_record", future)
    with pytest.raises(ValueError, match="receipt_after_permitted_cutoff"):
        asyncio.run(case.initial())
    assert case.journal["native_record"]["client_receive_at"] > case.clock()


def test_phase_source_must_be_inside_its_actual_HTTP_brackets(case, monkeypatch):
    original = case.native_record
    def before_send(phase=False):
        record = original(phase)
        if phase:
            body = frozen.control().body(record)
            body["native_phase"]["source_at"] = record["client_send_at"] - 0.00001
            rewrite(record, body)
        return record
    monkeypatch.setattr(case, "native_record", before_send)
    with pytest.raises(ValueError, match="phase_source_outside_collector_brackets"):
        asyncio.run(case.initial())


def test_flush_source_cannot_postdate_its_actual_collector_receipt(case, monkeypatch):
    options = case.options()
    original = options["gateway_async"]
    class FutureBarrier:
        async def post(self, *args, **kwargs):
            response = await original.post(*args, **kwargs)
            body = response.json()
            for key in ("source_at", "flush_started_at", "flush_finished_at", "sample_finished_at"):
                body[key] += 1
            return case.httpx.Response(200, json=body, request=response.request)
    options["gateway_async"] = FutureBarrier()
    monkeypatch.setattr(case, "options", lambda: options)
    with pytest.raises(ValueError, match="flush_source_outside_collector_brackets"):
        asyncio.run(case.initial())
    case.preserve("out-of-bracket-flush-witness.json", case.journal)


def test_untrusted_auth_header_capture_is_refused_before_body_consumption(case):
    record = deepcopy(case.startup["capture_begin_record"])
    record["auth_values_captured"] = True
    record["response_headers"]["authorization"] = "[NOT_A_REAL_CREDENTIAL]"
    with pytest.raises(ValueError, match="credential_safe_capture_required"):
        api.receipt_body(record, method="POST", path="/pr33-native/batch", cutoff=case.clock())


def test_later_source_join_preserves_released_ghost_and_zero_work_stock(case):
    async def run():
        closure, _ = await case.initial()
        rid, nid = await case.issue_then_release_with_physical_stock()
        cut = await case.later(closure)
        view = api.inspect_cut(cut, consume_at=case.clock(), deadline=case.deadline)
        phases = frozen.module("source_owner_phases").classify_source_owners(
            cut, initial_closure=closure, native_startup=case.startup["native_owned"],
            public_memory=case.public_memory, now=case.clock(), expected_clock_domain=DOMAIN)
        return closure, rid, nid, cut, view, phases
    closure, rid, nid, cut, view, phases = asyncio.run(run())
    assert cut["owner_prefix"]["owners"] == [] and phases.owners == ()
    assert phases.released_ghost_ids == (nid,)
    assert view["full_stock"]["base"]["running"] == 0
    assert view["full_stock"]["scheduler_state"]["requests"] == {}
    assert len(view["full_stock"]["queue_oldest_first"]) == 2
    assert view["full_stock"]["queue_oldest_first"][1]["work"] == []
    assert view["full_stock"]["queued_ghost_request_ids"] == [nid]
    assert view["gateway_budget_version"] == 1
    assert view["actual_application_at"] is None and view["acquisition_token"] is None
    case.preserve("synthetic-released-physical-stock.json",
                  dict(cut=cut, view=view, phases=phases.evidence(), closure=asdict(closure)))


@pytest.mark.parametrize("kind", ["before_receipt", "stale", "absolute_deadline"])
def test_a_source_cut_is_not_consumable_before_receipt_or_after_its_bound(case, kind):
    _, report = asyncio.run(case.initial())
    cut = report["cut"]
    now, deadline = case.clock(), case.deadline
    if kind == "before_receipt":
        now = (cut["cut_at"] + cut["validated_at"]) / 2
    elif kind == "stale":
        now = cut["cut_at"] + 0.6
        deadline = now + 1
    else:
        deadline = cut["validated_at"] - 0.00001
    with pytest.raises(ValueError, match="cut_stale_future_or_after_absolute_deadline"):
        api.inspect_cut(cut, consume_at=now, deadline=deadline)


def test_publication_writer_race_is_not_a_complete_empty_phase(case, monkeypatch):
    real = case.ledger.original_put
    seen = []
    def in_flight(item, block=True, timeout=None):
        result = real(item, block=block, timeout=timeout)
        seen.append(case.phase.phase_snapshot(case.core)["native_phase"])
        return result
    monkeypatch.setattr(case.ledger, "original_put", in_flight)
    from types import SimpleNamespace
    case.ledger.put(("ADD", (SimpleNamespace(request_id="between-publication-and-marker"), 0)))
    assert not seen[0]["complete"]
    assert "input_publication_in_flight" in seen[0]["refusal_reasons"]


def test_prior_Native02_empty_frames_are_not_retrospective_owner_closure():
    result = api.historical_native02()
    assert result["legacy_pending_batches"] == 1
    assert result["full_empty"] == [True, True]
    assert result["native_phase_present"] == [False, False]
    assert not result["owner_bootstrap_admissible"]
    assert result["actual_application_at"] is None and not result["native_mpc_permitted"]
