"""Accept retained engineering and partial coverage without changing the pass rule."""

import base64
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = ROOT / ".runtime/r24-team16-owner-inflight-gpu01-mayor"
RAW = WORK / "received01"
NATIVE = RAW / "native-output"
SEAL = "07971a037254cba76224e3c69b6d294101f3e097bfe6f628347c3df5f9d8a8a4"


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert sha(RAW / "RAW-SHA256SUMS") == SEAL
    expected = {"RAW-SHA256SUMS"}
    for line in (RAW / "RAW-SHA256SUMS").read_text().splitlines():
        digest, raw_name = line.split("  ", 1)
        name = Path(raw_name)
        assert not name.is_absolute() and ".." not in name.parts and name.as_posix() not in expected
        assert sha(RAW / name) == digest
        expected.add(name.as_posix())
    assert len(expected) == 243
    assert {p.relative_to(RAW).as_posix() for p in RAW.rglob("*") if p.is_file()} == expected
    gate, supervision, result = (load(NATIVE / name) for name in
                                ("export-gate.json", "supervision.json", "client/result.json"))
    assert gate["safe_to_export"] and not (NATIVE / "DO_NOT_EXPORT.json").exists()
    assert supervision["phase_bounds"] and supervision["sources_unchanged"]
    assert supervision["source_before"] == supervision["source_after"]
    assert supervision["candidate_unchanged"] and supervision["private_config_removed_after_join"]
    assert supervision["cleanup"]["all_joined"] and supervision["cleanup"]["capture_complete"]
    assert not supervision["errors"] and not supervision["cleanup"]["surviving_owned_pids"]
    assert result["engineering_pass"] and not result["failures"]
    assert result["nonempty_overlap_count"] == 1 and not result["coverage_complete"]
    assert gate["component_pass"] is supervision["component_pass"] is result["component_pass"] is False
    assert load(RAW / "mayor/launcher-exit.json")["returncode"] == 1
    assert (WORK / "remote-execution01.exit").read_text().strip() == "1"
    first, second = [offer["inflight"] for offer in result["offers"]]
    assert first["status"] == "empty_or_unattributed_stock" and not first["nonempty_overlap"]
    assert first["native_id"] is None and not first["target_scheduler_or_issued_stock"]
    assert second["status"] == "nonempty_available_while_pending" and second["nonempty_overlap"]
    assert second["native_kind"] == "running" and second["target_scheduler_or_issued_stock"]
    assert second["stock_counts"]["running"] == second["stock_counts"]["pending_batches"] == 1
    assert second["stock_counts"]["issued_request_ids"] == second["stock_counts"]["issued_routes"] == 1
    assert second["stock_counts"]["ghosts"] == 0
    assert second["client_send_at"] < second["source_at"] <= second["available_at"] < second["client_receive_at"]
    assert second["actual_application_at"] is None and not second["native_mpc_permitted"]
    cuts = {}
    for path in (NATIVE / "client").glob("*-cut.json"):
        journal = load(path)
        observation = journal["observation"]
        assert observation["actual_application_at"] is None
        assert not any(observation[key] for key in
                       ("deployable", "qualified_model", "prediction_qualified", "native_mpc_permitted"))
        age = observation["consumed_at"] - observation["source_at"]
        assert 0 <= age <= 0.5
        cuts[path.name] = dict(source_to_consumed_s=age, source_at=observation["source_at"],
                               consumed_at=observation["consumed_at"])
    assert len(cuts) == 5
    records = [load(p) for p in (NATIVE / "client/http").glob("*.json")]
    assert len(records) == 90
    generation = sorted((r for r in records if r["path"] == "/v1/chat/completions"),
                        key=lambda r: r["sequence"])
    assert len(generation) == 3
    completions = []
    for row in generation:
        request = json.loads(row["request_body"])
        response = json.loads(base64.b64decode(row["response_body_base64"], validate=True))
        assert request["max_tokens"] == 16
        completions.append(dict(role=row["role"], HTTP_sequence=row["sequence"],
                                usage=response["usage"], finish_reason=response["choices"][0]["finish_reason"]))
    assert [row["usage"]["completion_tokens"] for row in completions] == [11, 16, 16]
    assert [row["finish_reason"] for row in completions[1:]] == ["length", "length"]
    capture = load(NATIVE / "client/retained-capture.json")
    assert len(capture["requests"]) == 2 and len(capture["steps"]) == 32 and len(capture["workers"]) == 1
    assert (WORK / "replay-py31401.json").read_bytes() == (WORK / "replay-py31201.json").read_bytes()
    replay = load(WORK / "replay-py31401.json")
    assert replay["replay_consistent"] and replay["reported_engineering_pass"]
    assert not replay["component_evidence_consistent"] and replay["nonempty_overlap_count"] == 1
    assert not replay["live_procfs_reproved"] and not replay["model_qualified"]
    postcheck = load(WORK / "postcheck01.json")
    assert postcheck["boot_id"] == "2bc1567a-40a4-401d-b5a1-3afb0ef3b954"
    assert not postcheck["compute_apps"] and not postcheck["listeners"] and postcheck["private_config_absent"]
    assert len(postcheck["known_owned_identities"]) == 24
    assert all(row["original_identity_absent"] for row in postcheck["known_owned_identities"])
    index = load(WORK / "handoff-index.json")
    assert index["local_preregistration_commit"] == "d6ded2b25a0667028f6ab9f2cc08fe76df1c3315"
    files = {}
    for path in WORK.rglob("*"):
        assert not path.is_symlink() and (path.is_dir() or path.is_file())
        if path.is_file():
            files[path.relative_to(WORK).as_posix()] = sha(path)
    target = HERE / "native-team16-owner-inflight-gpu01"
    target.mkdir()
    for name, digest in files.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORK / name, destination)
        assert sha(destination) == digest
    report = dict(
        status="ACCEPTED_NATIVE_ENGINEERING_AND_ONE_TRUE_INFLIGHT_CUT_COMPONENT_COVERAGE_FAIL",
        preregistration_commit=index["local_preregistration_commit"],
        raw_manifest_sha256=SEAL, raw_files=242, archived_files=files,
        Native_experiments=1, attempts=1, retries=0,
        elapsed_s=supervision["finished_at"] - supervision["started_at"],
        engineering_pass=True, component_pass=False, prescribed_overlap_target=2,
        actual_nonempty_overlap_count=1, first_miss=first, observed_nonempty_cut=second,
        HTTP_records=90, generations=completions, actual_completion_tokens=43,
        requested_completion_cap=48, captured_requests=2, captured_steps=32, workers=1,
        source_cuts=cuts, full_sources_and_SDK_unchanged=True, safe_export=True,
        independent_live_owned_identities_gone=24, offline_replay=replay,
        offline_replays_are_not_new_Native_trials=True, actual_application_at=None,
        copy_ready_timing_qualification_inherited=False,
        model_prediction_mixing_P2_or_MPC_qualification_promoted=False, Native14="FAIL unchanged",
        interpretation="First actual usable target running/issued stock; 1/2 is NOT a component PASS. No persistent, ghost, copy-ready or predictive-accuracy claim.")
    write(HERE / "native-team16-owner-inflight-gpu01-results.json", report)
    print(json.dumps({key: report[key] for key in
                      ("status", "elapsed_s", "actual_nonempty_overlap_count", "component_pass",
                       "actual_completion_tokens", "captured_steps", "raw_manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
