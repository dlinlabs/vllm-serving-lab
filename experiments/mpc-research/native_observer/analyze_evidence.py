"""Audit immutable native evidence; diagnostics are not online model acceptance."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import re
import statistics
import tarfile

from mpc_lab.identification import reserved_cost_at
from mpc_lab.runtime import snapshot_from_dict


def analyze(archive):
    with tarfile.open(archive, "r:gz") as bundle:
        def text(name):
            with bundle.extractfile("./" + name) as stream:
                return stream.read().decode()
        def document(name):
            return json.loads(text(name))

        qualification = document("http-qualification.json")
        first = document("train-saturation-scout-v1.json")
        completion = document("train-saturation-scout-completion.json")
        ledger = [json.loads(line) for line in text("gateway-observed-telemetry.jsonl").splitlines()]
        ledger = [frame for frame in ledger if frame.get("event") == "mpc_ledger"]
        assert qualification["functional_pass"] and qualification["prompt_tokens_match"]
        assert qualification["native_engine_freshness_qualified"]
        assert all(check["drained"] for check in qualification["drain_checks"])
        assert completion["completed"]
        assert completion["reconciliation"]["status"] == "expired"
        assert completion["reconciliation"]["observed_version"] == completion[
            "pending_before_reconciliation"]["expected_version"]
        result = dict(
            schema="native-functional-and-train-scout-v1",
            archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            source_commit=qualification["source_commit"],
            implementation_hash=qualification["implementation_hash"],
            environment=document("environment.json"),
            functional_http=qualification["http_audit"],
            native_freshness_qualified=True,
            failed_preoffer_assignment=first["episodes"][-1]["action"],
            pending_resolution=completion["reconciliation"],
            independent_train_seed_blocks=4,
            performance_accepted=False,
            fitted_model_accepted=False,
            limitations=[
                "Four scout seed blocks, one per cell; no inferential policy comparison.",
                "TTFT<=0.5s cohort counts include outcome drain and are not window throughput.",
                "Inter-token timestamps, TPOT-tail SLO and composite goodput are unqualified.",
                "Persistence error is retrospective train-only, not an online prediction result.",
                "Scheduler num_computed_tokens is bookkeeping, not proof of completed GPU work.",
                "Hardware is a final point sample; no clocks/power/temperature workload series.",
                "Fixed budgets are reservation cost units, not request concurrency.",
            ], episodes=[])
        assert hashlib.sha256(text("native-observer-v1.py").encode()).hexdigest() == result[
            "environment"]["observer_sha256"]
        episodes = [e for e in first["episodes"] if "summary" in e] + completion["episodes"]
        assert len(episodes) == 8
        all_ids = set()
        for episode in episodes:
            initial, final = map(snapshot_from_dict, (episode["initial"], episode["final"]))
            frames = [f for f in ledger if f["run_id"] == initial.run_id
                      and initial.sequence < f["sequence"] <= final.sequence]
            metrics = episode["metrics"]
            reserved_cost_at(initial, frames, final, initial.sampled_at+1,
                dropped_records=metrics["telemetry_dropped_records"],
                write_errors=metrics["telemetry_write_errors"])
            assert initial.reserved == final.reserved == 0
            assert all(check["drained"] for check in episode["initial_drain"] + episode["final_drain"])
            assert episode["action"]["status"] == "applied"
            rows = episode["rows"]
            tape = (first if episode["seed"] != 159904 else completion)["offered_tapes"][str(episode["seed"])]
            assert len(rows) == len(tape)
            for actual, planned in zip(sorted(r["scheduled_at"] for r in rows),
                                       (episode["start"]+at for at in tape)):
                assert abs(actual-planned) < 1e-9
            assert all(r["clock_domain"] == initial.clock_domain for r in rows)
            ids = {r["request_id"] for r in rows}
            assert len(ids) == len(rows) and not all_ids.intersection(ids)
            all_ids.update(ids)
            assert all(r["status"] in ("complete", "rejected") for r in rows)
            for row in rows:
                assert row["request_id"] == row["response_request_id"]
                if row["status"] == "complete":
                    assert row["stream_done"] and not row["stream_error"]
                    assert row["finish_reason"] == "stop"
                    assert row["usage"]["prompt_tokens"] == episode["prompt_tokens"]
                else:
                    assert row["admission_rejection"]
            times = [initial.sampled_at] + [f["effective_at"] for f in frames]
            values = [initial.reserved] + [f["reserved_cost"] for f in frames]
            def cost_at(at):
                return values[bisect_right(times, at)-1]
            errors = [abs(cost_at(episode["start"]+t+1)-cost_at(episode["start"]+t))/32
                      for t in range(1, 11)]
            summary = {k: episode[k] for k in ("seed", "budget", "prompt_tokens", "max_tokens")}
            summary.update(episode["summary"])
            summary["ledger_frames"] = len(frames)
            summary["offline_persistence_r1_normalized_mae"] = statistics.mean(errors)
            completed = [r for r in rows if r["status"] == "complete"]
            summary["request_mean_tpot_median_s"] = statistics.median(
                (r["last_content_at"]-r["first_content_at"])/(r["usage"]["completion_tokens"]-1)
                for r in completed)
            native = episode["native_frames"]
            assert all(f["clock_domain"] == initial.clock_domain
                       and f["observer_sha256"] == result["environment"]["observer_sha256"]
                       and f["complete"] for f in native)
            assert all(0 <= f["received_at"]-f["sampled_at"] <= .5
                       and 0 <= f["frontend_sampled_at"]-f["sampled_at"] <= .05 for f in native)
            completed_by_id = {r["request_id"]: r for r in completed}
            observed_ids, last_output, leads = set(), {}, []
            for frame in native:
                for request in frame["requests"]:
                    match = re.fullmatch(r"chatcmpl-([0-9a-f]{32})-([0-9a-f]{8})", request["request_id"])
                    assert match and match[1] in completed_by_id
                    rid = match[1]
                    assert request["prompt_tokens"] == episode["prompt_tokens"]
                    assert request["maximum_output_tokens"] == episode["max_tokens"]
                    output = request["output_tokens"]
                    assert last_output.get(rid, 0) <= output <= completed_by_id[rid]["usage"]["completion_tokens"]
                    last_output[rid] = output
                    observed_ids.add(rid)
                    leads.append(request["computed_tokens"]-request["prompt_tokens"]-output)
            assert observed_ids == set(completed_by_id)
            summary["joined_native_requests"] = len(observed_ids)
            summary["maximum_scheduler_credit_lead_tokens"] = max(leads)
            summary["native_samples"] = len(native)
            summary["maximum_sample_width_s"] = max(
                f["frontend_sampled_at"]-f["sampled_at"] for f in native)
            summary["completion_token_range"] = [
                min(summary.pop("completion_tokens")), max(r["usage"]["completion_tokens"] for r in completed)]
            result["episodes"].append(summary)
        result["physical_offers"] = len(all_ids)
        result["complete"] = sum(e["complete"] for e in result["episodes"])
        result["rejected"] = sum(e["rejected"] for e in result["episodes"])
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.evidence)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in (
        "physical_offers", "complete", "rejected", "performance_accepted")}, indent=2))
