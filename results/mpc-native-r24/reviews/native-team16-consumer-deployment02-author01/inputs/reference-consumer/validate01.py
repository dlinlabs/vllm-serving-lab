"""Read-only literal source validators plus explicit byte/clock/namespace joins."""

from dataclasses import asdict
import time

import common01 as c
from ipc01 import strict_json


def prepare():
    started = time.monotonic()
    pre = c.preflight()
    modules = [pre.frozen.control()]
    modules.extend(pre.frozen.module(name) for name in (
        "input_post_observer", "live_owner_prefix", "capture_owner_closure", "source_owner_phases"))
    return dict(started_at=started, finished_at=time.monotonic(),
                module_files=[str(module.__file__) for module in modules],
                observation_consumed=False)


def strict_native_body(record):
    raw = c.raw64(record["response_body_base64"])
    c.require(c.sha(raw) == record["response_body_sha256"], "strict_native_body_hash")
    frame = strict_json(raw)
    c.require(type(frame) is dict, "strict_native_body_object")
    c.canonical(frame)
    return frame


def _raw_cut(bundle, *, startup, closure, memory, now, deadline, initial=False):
    pre = c.preflight()
    api = pre.frozen.control()
    cut = bundle["journal"]["cut"]
    record = bundle["journal"]["native_record"]
    frame = strict_native_body(record)
    literal = pre.receipt_body(record, method="GET", path="/pr33-native/phase-state", cutoff=now)
    c.require(c.canonical(frame) == c.canonical(literal), "strict_native_literal_binding")
    c.require(c.canonical(frame) == c.canonical(cut["native_observation"]), "native_raw_byte_binding")
    checked, _ = api.full_queue_observation(
        record, dict(pre.frozen.settings(), final_queue_path="/pr33-native/phase-state"),
        native_run=startup["native_owned"]["readiness"]["frames"][0]["run_id"], deadline=now)
    c.require(checked == frame and record["client_send_at"] <= cut["cut_at"]
              <= frame["native_phase"]["sample_finished_at"] <= record["client_receive_at"]
              <= cut["validated_at"] <= now, "source_receipt_cutoff")
    view = pre.inspect_cut(cut, consume_at=now, deadline=deadline)
    prefix = cut["owner_prefix"]
    raw = {name: c.raw64(bundle[name + "_base64"]) for name in ("snapshot", "prefix", "barrier")}
    c.require(c.sha(raw["barrier"]) == prefix["barrier_sha256"]
              and api.canonical(strict_json(raw["barrier"])) == api.canonical(prefix["barrier"])
              and api.canonical(strict_json(raw["snapshot"])) == api.canonical(prefix["snapshot"])
              and len(raw["prefix"]) == prefix["prefix_bytes"], "source_raw_inventory_binding")
    prefix_api = pre.frozen.module("live_owner_prefix")
    phase = frame["native_phase"]
    identifiers = (set(phase["publications"]) | set(phase["lifecycle"])) - set(cut["baseline_native_ids"])
    replay = prefix_api.validate_live_owner_prefix(
        raw["snapshot"], raw["prefix"], receipt=prefix_api.Receipt(**prefix["receipt"]),
        now=now, max_age_s=.5, cut_at=cut["cut_at"],
        expected_run_id=startup["gateway_owned"]["readiness"]["gateway_run_id"],
        expected_clock_domain=cut["common_clock_proof"]["clock_domain"],
        expected_identity=startup["gateway_owned"]["readiness"]["snapshot"]["identity"],
        initial_budget=32, public_memory=memory, native_ids=tuple(sorted(identifiers)),
        native_run_id=phase["run_id"], expected_native_run_id=phase["run_id"],
        include_intervening_events="intervening_gateway_events" in prefix)
    for key in ("run_id", "clock_domain", "cut_at", "snapshot_at", "closed_through_sequence",
                "budget", "budget_version", "owners", "released_native_ids", "native_run_id"):
        c.require(c.canonical(replay[key]) == c.canonical(prefix[key]), "source_prefix_replay:" + key)
    c.require(prefix["budget"] == 32 and prefix["snapshot"]["budget"] == 32
              and prefix["budget_version"] == prefix["snapshot"]["budget_version"] == 1,
              "fixed32_version1_component")
    proc = pre.frozen.module("process_clock_provenance")
    joined = proc.join_process_clocks(
        cut["gateway_process_proof"], cut["native_process_proof"],
        now=cut["validated_at"], max_age_s=.5)
    c.require(c.canonical(joined) == c.canonical(cut["common_clock_proof"]), "source_clock_join")
    if not initial:
        phases = pre.frozen.module("source_owner_phases").classify_source_owners(
            cut, initial_closure=closure, native_startup=startup["native_owned"],
            public_memory=memory, now=now, expected_clock_domain=joined["clock_domain"])
    else:
        phases = None
    inventory = {name: dict(bytes=len(data), sha256=c.sha(data)) for name, data in raw.items()}
    native_raw = c.raw64(record["response_body_base64"])
    inventory["native_HTTP_body"] = dict(bytes=len(native_raw), sha256=c.sha(native_raw))
    return view, phases, inventory


def validate(value, *, config, now):
    c.require(type(value) is dict and set(value) == {
        "schema", "namespace", "run_id", "ordinal", "generation", "boundary_started_at",
        "active_deadline", "deadline", "collector_returned_at", "source",
        "require_exact_application_time"}, "consumer_input_fields")
    c.require(value["schema"] == "team16-cut-consumer-input-v1" and value["namespace"] == c.NAMESPACE
              and value["run_id"] == config["authority"]["run_id"] and value["ordinal"] == 2
              and type(value["generation"]) is int and value["generation"] >= 0
              and type(value["require_exact_application_time"]) is bool, "consumer_input_namespace")
    source = value["source"]
    c.require(set(source) == {"startup", "initial", "current", "closure", "public_memory"},
              "source_bundle_inventory")
    current = source["current"]["journal"]["cut"]
    domain = config["authority"]["parent"]["clock_domain"]
    c.require(current["common_clock_proof"]["clock_domain"] == domain
              and current["native_observation"]["native_phase"]["capture"]["trial_id"] == c.CAPTURE,
              "consumer_source_domain_capture")
    c.require(c.number(value["boundary_started_at"]) <= c.number(value["collector_returned_at"])
              <= now < c.number(value["deadline"])
              and value["deadline"] == min(value["boundary_started_at"] + .080,
                                            current["cut_at"] + .5, value["active_deadline"]),
              "whole_pipeline_deadline_or_future_receipt")
    c.require(current["validated_at"] <= value["collector_returned_at"], "collection_return_before_validation")
    c.require(value["boundary_started_at"] <=
              source["current"]["journal"]["native_record"]["client_send_at"],
              "designated_collection_not_fresh_after_boundary")
    pre = c.preflight()
    c.verify_code(literal_cwd=True)
    initial = source["initial"]["journal"]
    initial_at = initial["observation"]["consumed_at"]
    startup = source["startup"]
    for record in [startup["warmup_record"], *startup["drain_records"],
                   startup["capture_metadata_record"], startup["capture_begin_record"]]:
        strict_native_body(record)
    bootstrap = c.candidate_bootstrap(startup, cutoff=initial_at)
    initial_view, _, initial_inventory = _raw_cut(
        source["initial"], startup=startup, closure=None, memory={}, now=initial_at,
        deadline=initial_at, initial=True)
    closure = pre.frozen.module("capture_owner_closure").establish_initial_capture_closure(
        initial["cut"], native_startup=startup["native_owned"], gateway_startup=startup["gateway_owned"])
    c.require(asdict(closure) == source["closure"] and closure.capture_id == c.CAPTURE,
              "initial_closure_binding")
    capture_started = closure.capture_started_at
    c.require(bootstrap["confirmation"]["actual_applied_at"] is None
              and bootstrap["confirmation"]["actual_application_interval"][1] <= capture_started
              and bootstrap["confirmation_available_at"] <= bootstrap["begin_record"]["client_send_at"]
              <= capture_started <= bootstrap["begin_record"]["client_receive_at"]
              <= closure.initial_cut_at <= current["cut_at"], "initial_real_confirmation_capture_chronology")
    prefix_api = pre.frozen.module("live_owner_prefix")
    memory = {rid: prefix_api.PublicRequest(**row) for rid, row in source["public_memory"].items()}
    if config["evidence_mode"] == "owned-native":
        c.require(set(memory) <= {row["request_id"] for row in c.load(c.HERE / "OFFERS.json")["rows"]},
                  "prospective_public_offer_inventory")
        runtime = bootstrap["metadata"]["runtime"]
        c.require(runtime["max_num_batched_tokens"] == 2048 and runtime["max_num_seqs"] == 32,
                  "native_formation_configuration_not_gateway_budget")
        proc = c.proc_api()
        host = proc.LinuxProc()
        boot, ns = proc.domain_parts(domain)
        for role in ("native", "gateway"):
            proof = current[role + "_process_proof"]
            original = proof["owned_chain"]
            _, chain = proc.inspect_chain(host, original[-1]["pid"], original[-1]["start_ticks"],
                                           original[0]["pid"], boot, ns)
            c.require([asdict(p) for p in chain] == original, "actual_source_process_identity_changed")
    else:
        c.require(config["evidence_mode"] == "cpu-source-fixture", "unknown_evidence_mode")
    view, phases, inventory = _raw_cut(
        source["current"], startup=startup, closure=closure, memory=memory,
        now=now, deadline=value["deadline"])
    c.require(view["actual_application_at"] is None, "point_timestamp_not_permitted")
    if value["require_exact_application_time"]:
        raise ValueError("exact_application_time_unobserved")
    stock = view["full_stock"]
    return dict(full_stock=stock, full_stock_sha256=c.sha(c.canonical(stock)),
                source_cut_sha256=c.sha(c.canonical(current)), source_at=current["cut_at"],
                raw_inventory={"initial": initial_inventory, "current": inventory},
                source_phases=phases.evidence(), initial_closure_sha256=c.sha(c.canonical(asdict(closure))),
                initial_point_gate=initial_view["exact_application_time_gate"],
                source_state_is_current_state=False,
                evidence_mode=config["evidence_mode"], **c.permissions())
