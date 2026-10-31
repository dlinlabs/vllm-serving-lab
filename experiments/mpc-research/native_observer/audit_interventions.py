"""Audit actual native intervention assignments and the entire offered cohort."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

from mpc_lab.identification import reserved_cost_at
from mpc_lab.runtime import snapshot_from_dict
from state_sample import join_observations


def composite_good(row):
    count = row["usage"]["completion_tokens"]
    mean_tpot = ((row["terminal_at"]-row["first_content_at"])/(count-1)
                 if count>1 else 0)
    return row["first_content_at"]-row["scheduled_at"] <= .5 and mean_tpot <= .05


def audit_isolated(directory):
    directory = Path(directory)
    report = json.loads((directory/"isolated.json").read_text())
    if not report.get("completed") or report["assignment"]["status"]!="applied":
        raise ValueError("isolated collection or assignment incomplete")
    if hashlib.sha256((directory/"collector-measured.py").read_bytes()).hexdigest() != report["script_sha256"]:
        raise ValueError("isolated measured source mismatch")
    expected = {f"isolated-{s}-{p}-{m}-{r}" for s in (160009,160010)
                for p,m in ((128,128),(128,256),(4096,128),(4096,256)) for r in range(4)}
    episodes = report["episodes"]
    if len(episodes)!=len(expected) or {e["episode_id"] for e in episodes}!=expected:
        raise ValueError("isolated planned cohort differs")
    ids, samples, preludes = set(),0,0
    for episode in episodes:
        if (episode["split"]!="train" or episode["budget"]!=16 or len(episode["rows"])!=1
                or not episode["native_frames"]
                or episode["initial"]["requests"] or episode["initial"]["budget"]!=16
                or len(episode["initial_drain"])!=2 or len(episode["final_drain"])!=2
                or not all(d["drained"] for d in episode["initial_drain"]+episode["final_drain"])):
            raise ValueError("isolated boundary or observation missing")
        row = episode["rows"][0]
        if (row["request_id"] in ids or row["response_request_id"]!=row["request_id"]
                or row["clock_domain"]!=episode["initial"]["clock_domain"]
                or row["status"]!="complete" or not row["stream_done"] or row["stream_error"]
                or row["finish_reason"]!="stop"
                or row["usage"]["prompt_tokens"]!=episode["prompt_tokens"]
                or not row["scheduled_at"] <= row["sent_at"] <= row["first_content_at"]
                     <= row["last_content_at"] <= row["terminal_at"]):
            raise ValueError("isolated HTTP outcome invalid")
        ids.add(row["request_id"])
        snapshot = snapshot_from_dict(episode["initial"])
        for frame in episode["native_frames"]:
            join_observations(frame["gateway_snapshot"],frame,identity=snapshot.identity,
                gateway_run=snapshot.run_id,native_run=episode["native_frames"][0]["run_id"],
                observer_sha=report["observer_sha256"],domain=snapshot.clock_domain,
                gateway_received_at=frame["gateway_received_at"],
                native_received_at=frame["received_at"],now=frame["received_at"])
            if frame["running"]>1 or frame["waiting"]>0:
                raise ValueError("isolated native work was shared")
            samples += 1
        preludes += episode["prelude"]
    return dict(schema="native-isolated-audit-v1",physical_offers=len(ids),
        natural_eos_complete=len(ids),preludes_retained=preludes,fit_requests=len(ids)-preludes,
        receipt_stamped_samples=samples,independent_train_seeds=2,
        raw_json_sha256=hashlib.sha256((directory/"isolated.json").read_bytes()).hexdigest(),
        calibration_accepted=False,performance_accepted=False)


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory/"interventions.json").read_text())
    if not report.get("completed") or len(report["episodes"]) != 8:
        raise ValueError("intervention collection incomplete; preserve failed evidence")
    ledger = [json.loads(line) for line in (directory/"gateway-telemetry.jsonl").open()]
    ledger = [f for f in ledger if f.get("event") == "mpc_ledger"]
    hardware_path = directory/"hardware-partial.jsonl"
    hardware = [json.loads(line) for line in hardware_path.read_text().splitlines()] if hardware_path.exists() else []
    summaries, all_ids, groups = [], set(), {}
    for episode in report["episodes"]:
        seed = episode["seed_block"]
        if episode["split"] != "train" or not 160001 <= seed <= 160012:
            raise ValueError("unexpected split or seed")
        groups.setdefault(seed, set()).add(episode["arm"])
        initial, final = map(snapshot_from_dict,(episode["initial"],episode["final"]))
        if initial.identity.implementation_hash != report["implementation_hash"]:
            raise ValueError("source identity mismatch")
        frames = [f for f in ledger if f["run_id"] == initial.run_id
                  and initial.sequence < f["sequence"] <= final.sequence]
        metrics = episode["metrics"]
        reserved_cost_at(initial,frames,final,initial.sampled_at+1,
            dropped_records=metrics["telemetry_dropped_records"],write_errors=metrics["telemetry_write_errors"])
        if initial.reserved or final.reserved or not all(
                d["drained"] for d in episode["initial_drain"]+episode["final_drain"]):
            raise ValueError("drain evidence incomplete")
        preassignment = episode["preassignment"][-1]["reply"]
        prescribed_initial = episode["schedule"][0]["budget"]
        if (preassignment["status"] != "applied" or initial.budget != prescribed_initial
                or preassignment["acknowledgment"]["budget"] != prescribed_initial
                or preassignment["acknowledgment"]["run_id"] != initial.run_id
                or preassignment["acknowledgment"]["clock_domain"] != initial.clock_domain
                or preassignment["command"]["identity"] != asdict(initial.identity)
                or preassignment["acknowledgment"]["budget_version"] != initial.budget_version
                or not preassignment["acknowledgment"]["applied_at"] <
                       preassignment["command"]["not_after"]):
            raise ValueError("initial assignment unverified")
        expected = episode["expected_ids"]
        ids = {r["request_id"] for r in episode["rows"]}
        if len(expected) != len(set(expected)) or ids != set(expected) or all_ids & ids:
            raise ValueError("physical offered cohort differs")
        all_ids.update(ids)
        scheduled = {rid:episode["start"]+at for rid,at in zip(expected,report["offered_tapes"][str(seed)])}
        for row in episode["rows"]:
            if row["clock_domain"] != initial.clock_domain:
                raise ValueError("HTTP clock domain")
            if abs(row["scheduled_at"]-scheduled[row["request_id"]]) > 1e-9:
                raise ValueError("offered tape differs")
            if row["response_request_id"] != row["request_id"]:
                raise ValueError("response identity differs")
            if row["status"] == "complete":
                if (not row["stream_done"] or row["stream_error"] or row["finish_reason"] != "stop"
                        or row["usage"]["prompt_tokens"] != episode["prompt_tokens"]):
                    raise ValueError("native completion or prompt shape invalid")
                if (not 0 <= row["scheduled_at"] <= row["sent_at"] <= row["first_content_at"]
                        <= row["last_content_at"] <= row["terminal_at"]
                        or type(row["usage"]["completion_tokens"]) is not int
                        or row["usage"]["completion_tokens"] < 1):
                    raise ValueError("native completion chronology or output count")
            elif row["status"] != "rejected" or not row["admission_rejection"]:
                raise ValueError("failed or unresolved offered request")
        lag, retries = [], 0
        if len(episode["actions"]) != len(episode["schedule"])-1:
            raise ValueError("assigned schedule incomplete")
        for prescribed, actual in zip(episode["schedule"][1:],episode["actions"]):
            if any(prescribed[k] != actual[k] for k in ("at_s","budget")):
                raise ValueError("prescribed action differs")
            attempts = actual["attempts"]
            retries += len(attempts)-1
            reply = attempts[-1]["reply"]
            if reply["status"] != "applied":
                raise ValueError("actual assignment unacknowledged")
            command, ack = reply["command"], reply["acknowledgment"]
            if command["identity"] != asdict(initial.identity) or ack["budget"] != prescribed["budget"]:
                raise ValueError("actuator identity or budget")
            if command["clock_domain"] != initial.clock_domain:
                raise ValueError("actuator clock domain")
            if not command["not_after"] > ack["applied_at"]:
                raise ValueError("actuator expiry violated")
            applied_lag = ack["applied_at"]-(episode["start"]+prescribed["at_s"])
            if applied_lag < 0:
                raise ValueError("assignment applied before prescribed time")
            lag.append(applied_lag)
        native = episode["native_frames"]
        if not native:
            raise ValueError("no native state evidence")
        receipts = dict(recorded=0,joined=0,with_gateway_pending_work=0,exact_work_join=0)
        native_run = native[0]["run_id"]
        for frame in native:
            if (frame["clock_domain"] != initial.clock_domain or not frame["complete"]
                    or frame["observer_sha256"] != report["observer_sha256"]
                    or frame["run_id"] != native_run):
                raise ValueError("native identity or completeness")
            if not (0 <= frame["received_at"]-frame["sampled_at"] <= .5
                    and 0 <= frame["frontend_sampled_at"]-frame["sampled_at"] <= .05):
                raise ValueError("native stale or wide")
            if "gateway_snapshot" in frame:
                observation = join_observations(frame["gateway_snapshot"],frame,
                    identity=initial.identity,gateway_run=initial.run_id,native_run=native_run,
                    observer_sha=report["observer_sha256"],domain=initial.clock_domain,
                    gateway_received_at=frame["gateway_received_at"],
                    native_received_at=frame["received_at"],now=frame["received_at"])
                receipts["recorded"] += 1
                receipts["joined"] += 1
                unknown = any(w.physical_prompt_tokens is None for w in observation.observations)
                receipts["with_gateway_pending_work"] += unknown
                receipts["exact_work_join"] += (
                    not unknown and observation.running==len(observation.observations))
        completed = [r for r in episode["rows"] if r["status"]=="complete"]
        computed_summary = dict(offered=len(episode["rows"]),complete=len(completed),
            rejected=sum(r["status"]=="rejected" for r in episode["rows"]),other=0,
            ttft_good=sum(r["first_content_at"]-r["scheduled_at"] <= .5 for r in completed),
            natural_eos=len(completed))
        if episode["summary"] != computed_summary:
            raise ValueError("reported summary differs from complete offered evidence")
        composite_count = sum(composite_good(r) for r in completed)
        sample_gaps = [b["received_at"]-a["received_at"] for a,b in zip(native,native[1:])]
        if any(gap < 0 for gap in sample_gaps):
            raise ValueError("native receipt chronology")
        phases = []
        for begin,end in ((0,30),(30,60),(60,90)):
            offered = [r for r in episode["rows"] if begin <= r["scheduled_at"]-episode["start"] < end]
            phases.append(dict(begin_s=begin,end_s=end,offered=len(offered),
                rejected=sum(r["status"]=="rejected" for r in offered),
                complete=sum(r["status"]=="complete" for r in offered)))
        summaries.append(dict(episode_id=episode["episode_id"],seed_block=seed,arm=episode["arm"],
            cell=episode["cell"],**episode["summary"],ledger_frames=len(frames),native_samples=len(native),
            assigned_changes=sum(a["budget"]!=b["budget"] for a,b in zip(episode["schedule"],episode["schedule"][1:])),
            receipt_observations=receipts,composite_good=composite_count,
            hardware_samples_inside_offer_window=sum(
                episode["start"] <= h["sample_begin"] <= episode["start"]+90 for h in hardware),
            maximum_applied_lag_s=max(lag),median_applied_lag_s=statistics.median(lag),assignment_retries=retries,
            maximum_native_receipt_gap_s=max(sample_gaps) if sample_gaps else None,
            effective_native_samples_per_s=len(native)/(native[-1]["received_at"]-native[0]["received_at"])
                if len(native)>1 else None,
            gaps_longer_than_freshness=sum(gap>.5 for gap in sample_gaps),phases=phases,
            actual_output_range=[min(r["usage"]["completion_tokens"] for r in completed),
                                 max(r["usage"]["completion_tokens"] for r in completed)]))
    if any(arms != {0,1} for arms in groups.values()) or len(groups) != 4:
        raise ValueError("independent paired assignment blocks incomplete")
    return dict(schema="native-intervention-audit-v1",source_commit=report["source_commit"],
        raw_json_sha256=hashlib.sha256((directory/"interventions.json").read_bytes()).hexdigest(),
        independent_train_blocks=4,physical_offers=len(all_ids),episodes=summaries,
        calibration_accepted=False,mpc_performance_accepted=False,
        contrast="whole assigned complementary sequence; not same-state pointwise action effects",
        hardware_coverage="partial timeline; inspect per-episode overlap, not full coverage" if hardware else "not recorded",
        research_slo=dict(ttft_s=.5,request_mean_tpot_s=.05,single_token_tpot_s=0),
        limitations=["Only four new Train blocks; no Calibration/Validation/Test execution.",
                     "Receipt joins are replayed from actual capture, not forecasts issued online.",
                     "Composite cohort counts are not controller performance comparisons or causal model acceptance."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--isolated",action="store_true")
    args = parser.parse_args()
    result = audit_isolated(args.directory) if args.isolated else audit(args.directory)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"physical_offers":result["physical_offers"],
                      "mpc_performance_accepted":False},indent=2))
