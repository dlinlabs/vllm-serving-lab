"""Fake procfs and authenticated-acquisition stand-ins; no native processes."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def case(monkeypatch):
    monkeypatch.syspath_prepend(str(
        Path(__file__).resolve().parents[1] / "experiments/mpc-research/native-r23"))
    import process_clock_provenance as api

    boot = "11111111-1111-1111-1111-111111111111"
    domain = f"linux-monotonic:{boot}:time:[42]"

    class Proc:
        self_pid = 900

        def __init__(self):
            self.boot = boot
            self.rows = {
                900: api.Process(900, 1, 900, 900, 5, "time:[42]", "pid:[7]"),
                100: api.Process(100, 900, 100, 100, 10, "time:[42]", "pid:[7]"),
                110: api.Process(110, 100, 100, 100, 20, "time:[42]", "pid:[7]"),
                111: api.Process(111, 110, 100, 100, 30, "time:[42]", "pid:[7]"),
            }

        def boot_id(self):
            return self.boot

        def read(self, pid):
            return self.rows[pid]

    process = {"pid": 111, "start_ticks": 30, "pid_namespace": "pid:[7]"}

    def startup(role):
        frame = dict(run_id=role + "-epoch", clock_domain=domain, complete=True)
        ready = (
            dict(snapshot=frame, gateway_run_id=frame["run_id"], clock_domain=domain)
            if role == "gateway" else dict(frames=[dict(frame), dict(frame)])
        )
        return dict(pid=100, start_ticks="10", process_group=100, started_at=1,
                    native_ready=True, cleanup_completed=False, readiness=ready)

    def run(role="native", *, proc=None, receipt=None, mutate=None, frame_changes=None,
            metadata=None, times=(10, 10.1, 10.4, 10.5), max_age_s=0.5):
        proc = Proc() if proc is None else proc
        clock = iter(times)
        frame = dict(run_id=role + "-epoch", clock_domain=domain)
        frame.update(
            dict(process_identity=dict(process), source_at=10.2, sample_finished_at=10.3)
            if role == "native" else dict(sampled_at=10.2))
        frame.update(frame_changes or {})

        def acquire():
            if mutate:
                mutate(proc)
            return frame

        return api.prove_owned_process_clock(
            startup(role) if receipt is None else receipt,
            role=role, acquire=acquire, proc=proc,
            native_process=(process if metadata is None else metadata) if role == "native" else None,
            clock=lambda: next(clock), max_age_s=max_age_s)

    return SimpleNamespace(api=api, Proc=Proc, startup=startup, run=run,
                           process=process, boot=boot, domain=domain)


def test_native_descendant_and_gateway_source_derived_clock_join(case):
    native, gateway = case.run(), case.run("gateway")
    assert [p["pid"] for p in native["owned_chain"]] == [111, 110, 100]
    assert gateway["owned_chain"][0]["pid"] == 100
    assert native["source_age_s"] == pytest.approx(0.3)
    joined = case.api.join_process_clocks(gateway, native, now=10.6, max_age_s=0.5)
    assert joined["clock_domain"] == case.domain
    assert joined["native_run_id"] == "native-epoch"
    assert not joined["capture_verified"] and not joined["native_mpc_permitted"]


@pytest.mark.parametrize("pid,changes", [
    (100, {"start_ticks": 11}),
    (111, {"start_ticks": 31}),
    (111, {"parent_pid": 900}),
    (111, {"parent_pid": 111}),
    (111, {"process_group": 999}),
    (111, {"session": 999}),
    (110, {"start_ticks": 40}),
    (111, {"time_namespace": "time:[99]"}),
    (111, {"pid_namespace": "pid:[99]"}),
    (900, {"time_namespace": "time:[99]"}),
])
def test_wrong_process_parent_and_namespace_refuse(case, pid, changes):
    proc = case.Proc()
    proc.rows[pid] = replace(proc.rows[pid], **changes)
    with pytest.raises(ValueError):
        case.run(proc=proc)


@pytest.mark.parametrize("pid,changes", [
    (100, {"start_ticks": 11}),
    (111, {"start_ticks": 31}),
    (111, {"parent_pid": 100}),
    (111, {"time_namespace": "time:[99]"}),
    (110, {"start_ticks": 19}),
    (900, {"start_ticks": 6}),
])
def test_before_after_reuse_reparent_or_namespace_changes_refuse(case, pid, changes):
    def mutate(proc):
        proc.rows[pid] = replace(proc.rows[pid], **changes)

    with pytest.raises(ValueError):
        case.run(mutate=mutate)


def test_wrong_host_or_boot_change_refuses(case):
    proc = case.Proc()
    proc.boot = "22222222-2222-2222-2222-222222222222"
    with pytest.raises(ValueError, match="host_or_boot"):
        case.run(proc=proc)
    with pytest.raises(ValueError, match="host_or_boot"):
        case.run(mutate=lambda proc: setattr(
            proc, "boot", "22222222-2222-2222-2222-222222222222"))


def test_missing_core_identity_is_not_replaced_by_owned_launcher(case):
    with pytest.raises(ValueError, match="core_process_metadata"):
        case.run(metadata={})
    with pytest.raises(ValueError, match="core_process_metadata"):
        case.run(frame_changes={"process_identity": None})
    with pytest.raises(ValueError, match="core_identity"):
        case.run(metadata={**case.process, "pid_namespace": "pid:[99]"})
    with pytest.raises(ValueError, match="reply_core_identity"):
        case.run(frame_changes={"process_identity": {
            **case.process, "pid": 110, "start_ticks": 20}})


def test_readiness_and_live_epoch_fences(case):
    for change in [
        {"native_ready": False}, {"cleanup_completed": True},
        {"finished_at": 10}, {"returncode": 0}, {"process_group": 101},
    ]:
        with pytest.raises(ValueError):
            case.run(receipt={**case.startup("native"), **change})
    receipt = case.startup("native")
    receipt["readiness"]["frames"][1]["run_id"] = "another-engine"
    with pytest.raises(ValueError, match="readiness_epoch"):
        case.run(receipt=receipt)
    for change in [
        {"run_id": "another-engine"},
        {"clock_domain": "caller-entered-same-host"},
    ]:
        with pytest.raises(ValueError, match="live_epoch_or_domain"):
            case.run(frame_changes=change)


def test_acquisition_and_validation_latency_are_charged(case):
    with pytest.raises(ValueError, match="stale_after_validation"):
        case.run(times=(10, 10.1, 10.4, 11))
    with pytest.raises(ValueError, match="chronology"):
        case.run(frame_changes={"source_at": 10.05})
    with pytest.raises(ValueError, match="chronology"):
        case.run(frame_changes={"sample_finished_at": 10.6})
    with pytest.raises(ValueError, match="invalid_time"):
        case.run(times=(10, 10.1, 10.4, float("nan")))
    with pytest.raises(ValueError, match="join_stale"):
        case.api.join_process_clocks(case.run("gateway"), case.run(), now=11, max_age_s=0.5)


def test_common_clock_join_rejects_other_host_or_collector(case):
    gateway, native = case.run("gateway"), case.run()
    for key, value in [("boot_id", "other-host"), ("clock_domain", "other-domain"),
                       ("collector", {**native["collector"], "start_ticks": 6})]:
        other = {**native, key: value}
        with pytest.raises(ValueError, match="collector_mismatch"):
            case.api.join_process_clocks(gateway, other, now=10.6, max_age_s=0.5)


def stat_text(pid=111, *, start=30, state="S"):
    fields = [state, "110", "100", "100", *(["0"] * 15), str(start)]
    return f"{pid} (core (worker) with ) name) " + " ".join(fields)


def test_real_proc_reader_on_fake_tree_and_minimal_metadata(case, tmp_path):
    path = tmp_path / "111"
    (path / "ns").mkdir(parents=True)
    (path / "stat").write_text(stat_text())
    (path / "ns/time").symlink_to("time:[42]")
    (path / "ns/pid").symlink_to("pid:[7]")
    boot = tmp_path / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text(case.boot + "\n")
    proc = case.api.LinuxProc(tmp_path, self_pid=111)
    assert proc.boot_id() == case.boot
    assert proc.read(111).start_ticks == 30
    assert case.api.self_process_metadata(proc) == case.process
    (path / "stat").write_text(stat_text(state="Z"))
    with pytest.raises(ValueError, match="dead_proc_stat"):
        proc.read(111)


def test_pid_changes_while_reading_proc_record_refuse(case, tmp_path, monkeypatch):
    path = tmp_path / "111/ns"
    path.mkdir(parents=True)
    (path / "time").symlink_to("time:[42]")
    (path / "pid").symlink_to("pid:[7]")
    proc = case.api.LinuxProc(tmp_path, self_pid=111)
    records = iter([stat_text(start=30), stat_text(start=31)])
    monkeypatch.setattr(proc, "text", lambda path: next(records))
    with pytest.raises(ValueError, match="changed_during_read"):
        proc.read(111)


def test_host_proc_mount_with_foreign_pid_view_refuses(case, tmp_path, monkeypatch):
    (tmp_path / "self").symlink_to("222")
    monkeypatch.setattr(case.api.os, "getpid", lambda: 111)
    with pytest.raises(ValueError, match="procfs_pid_view"):
        case.api.LinuxProc(tmp_path)


@pytest.mark.parametrize("raw", [
    "111 (missing tail)", stat_text(pid=112), stat_text(start=0), stat_text(state="X"),
])
def test_bad_proc_stat_refuses(case, raw):
    with pytest.raises(ValueError, match="proc_stat"):
        case.api.parse_stat(raw, 111)
