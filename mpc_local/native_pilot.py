"""H16 deployment mechanics; operator authority is not runtime qualification."""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
import platform
from pathlib import Path
import re
import signal
import shlex
import sys
import tarfile
import time

from mpc_lab.actuator import ActuationPermit, GuardedActuator, IdentificationPermit
from mpc_lab.contracts import CostSpec, Decision, ModelParameters, finite
from mpc_lab.runtime import local_clock_domain
from mpc_local.control import ROOT, SOURCE_HASHES, audit_owner_log, qualify_snapshot, qualify_source


EVIDENCE = {"native_state", "native_actions", "calibration", "native_http", "fallback"}
EXTENSION = "experiments/mpc-research/native-v3/prospective_identification.py"
LOCAL_PROTOCOL = "experiments/mpc-research/native-v3/r21-local-protocol.json"
SERVING_COMMIT = "104e65a69df73997b977bce2588b655ae714f45d"
DEPLOYMENT_COMMIT = "7357358d9801102f82c1aff61ca6f5aa88b3a1c0"
H16_PROTOCOL_SHA = "ab883c90e85b9ec61967499841ce9de5942f11d916372f1b40824866750608da"
POST_CONSUMER_SHA = "e66f3a17550e212f1cdd31f4e4b64dfc8d9dcf5653cb7493d794424031ad9630"
PROVENANCE_SHAS = {
    "65ace5ba17f2dba4f527f68c5ba5debfd4d2e7521dadbf4c24e3d8b90adea0a6",
    "1009ccf4f7e2b5f6bd08cc7d06f31b1b37e59e54a1d55d29735b9d4b1793d3c0",
}
NATIVE_VERSIONS = {"vllm": "0.30.0", "torch": "2.13.0", "transformers": "5.22.1"}


def wall_valid(record, *, now, duration):
    start, end = record.get("valid_from_utc_s"), record.get("valid_until_utc_s")
    return (
        record.get("validity_clock") == "unix_utc"
        and all(finite(v) for v in (start, end, now, duration))
        and 0 <= start <= now and duration > 0 and now + duration <= end
    )


def artifact(record):
    path = Path(record["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("artifact_outside_owned_root")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("artifact_hash")
    return json.loads(raw)


def extension():
    qualify_source()
    spec = importlib.util.spec_from_file_location("r21_owned_prospective", ROOT / EXTENSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_h16(cost_spec):
    return extension().protocol_for(
        "h16", "104e65a69df73997b977bce2588b655ae714f45d", asdict(cost_spec),
    )


def preflight(config, protocol, *, now=None, wall_now=None, measurement=False, worker_phase=False):
    now = time.monotonic() if now is None else now
    wall_now = time.time() if wall_now is None else wall_now
    reasons = []
    sources = qualify_source()
    local = json.loads((ROOT / LOCAL_PROTOCOL).read_bytes())
    try:
        extension().validate_protocol(protocol)
        if protocol.get("prospective_reservation") != "h16":
            raise ValueError("h16_only")
    except (ValueError, KeyError, TypeError) as exc:
        reasons.append("protocol:" + str(exc))
    try:
        parameters = ModelParameters(
            cost_spec=CostSpec(**protocol["cost_spec"]),
            model_version="native-id-unfitted-v1", calibration_version="unfitted-not-validated",
        )
        permit_data = config["identification_permit"]
        permit = IdentificationPermit(
            parameters.identity, approved=permit_data["approved"],
            run_id=permit_data["run_id"], clock_domain=permit_data["clock_domain"],
            expires_at=permit_data["expires_at"],
        )
        if (
            not permit.allows(parameters.identity) or not finite(now)
            or permit.clock_domain != local_clock_domain()
            or now < 0 or now >= permit.expires_at
            or config.get("identity") != asdict(parameters.identity)
            or not finite(config.get("wall_ceiling_s"), positive=True)
            or permit.expires_at < now + config["wall_ceiling_s"]
        ):
            raise ValueError("explicit_unfitted_identification_permit")
        if not measurement:
            pairs = []
            supplied = config["evidence"]
            if not isinstance(supplied, dict) or set(supplied) != EVIDENCE:
                raise ValueError("original_five_evidence")
            for key, record in supplied.items():
                path = Path(record["path"]).resolve()
                if not path.is_relative_to(ROOT):
                    raise ValueError("evidence_outside_owned_root")
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha != record["sha256"]:
                    raise ValueError("evidence_hash")
                pairs.append((key, sha))
            if not ActuationPermit(parameters.identity, tuple(pairs), approved=permit_data["approved"]).allows(
                parameters.identity,
            ):
                raise ValueError("original_five_evidence")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        reasons.append("permit_evidence:" + str(exc))
    attestations = () if measurement else (
        "backend_free", "epoch_qualified", "q1_q5_qualified",
        "both_bundles_qualified", "written_owner_closed", "billing_stop_control",
    )
    for name in attestations:
        if config.get(name) is not True:
            reasons.append("operator_attestation:" + name)
    try:
        bundles = config["bundle_evidence"]
        if not isinstance(bundles, list) or len(bundles) != 2 or (
            bundles[0]["sha256"] == bundles[1]["sha256"]
        ):
            raise ValueError("both_distinct_bundles")
        for record in bundles:
            artifact(record)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        reasons.append("bundle_evidence:" + str(exc))
    if config.get("MPC_OWNER_LEDGER_ENABLED") != "true":
        reasons.append("owner_writer_not_enabled")
    limits = local["future_native_pilot"]
    for name, ceiling in (
        ("active_ceiling_s", limits["active_time_ceiling_s"]),
        ("wall_ceiling_s", limits["total_wall_ceiling_s"]),
        ("spend_ceiling_usd", limits["spend_ceiling_usd"]),
    ):
        value = config.get(name)
        if not finite(value, positive=True) or value > ceiling:
            reasons.append("ceiling:" + name)
    planned = protocol.get("planned_offered_window_s")
    if (
        not finite(planned, positive=True)
        or not finite(config.get("active_ceiling_s"), positive=True)
        or planned > config["active_ceiling_s"]
    ):
        reasons.append("legacy_protocol_exceeds_active_ceiling")
    rate = config.get("provider_hourly_usd")
    try:
        if worker_phase:
            return worker_preflight_result(config, protocol, reasons, sources, now, wall_now)
        price = artifact(config["provider_rate_evidence"])
        if (
            not finite(rate, positive=True) or price.get("hourly_rate_usd") != rate
            or price.get("verified") is not True or price.get("existing_pod") is not True
            or not isinstance(config.get("existing_pod_id"), str) or not config["existing_pod_id"]
            or price.get("pod_id") != config["existing_pod_id"]
            or price.get("run_id") != config["identification_permit"]["run_id"]
            or not wall_valid(price, now=wall_now, duration=config["wall_ceiling_s"])
        ):
            raise ValueError("actual_provider_rate_missing")
        stop_evidence = artifact(config["billing_stop_evidence"])
        if (
            stop_evidence.get("verified_control") is not True
            or stop_evidence.get("stop_argv") != config.get("stop_argv")
            or stop_evidence.get("pod_id") != config.get("existing_pod_id")
            or stop_evidence.get("run_id") != config["identification_permit"]["run_id"]
            or not wall_valid(stop_evidence, now=wall_now, duration=config["wall_ceiling_s"])
        ):
            raise ValueError("billing_stop_control_missing")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        reasons.append("provider_evidence:" + str(exc))
    if not finite(rate, positive=True):
        reasons.append("actual_provider_rate_missing")
    elif rate > 1.0:
        reasons.append("provider_rate_above_one_dollar_per_hour")
    elif finite(config.get("wall_ceiling_s"), positive=True) and finite(
        config.get("spend_ceiling_usd"), positive=True,
    ) and rate * config["wall_ceiling_s"] / 3600 > config["spend_ceiling_usd"]:
        reasons.append("spend_ceiling")
    release = config.get("operator_release", {})
    if (
        not isinstance(release, dict) or release.get("approved") is not True
        or release.get("scope") != "h16_open_loop_measurement"
        or release.get("source_commit") != protocol.get("source_commit")
        or release.get("pod_id") != config.get("existing_pod_id")
        or release.get("run_id") != config.get("identification_permit", {}).get("run_id")
        or release.get("clock_domain") != local_clock_domain()
        or release.get("protocol_sha256") != hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        or not finite(release.get("expires_at")) or not finite(now)
        or release["expires_at"] <= now
    ):
        reasons.append("separate_operator_release_missing")
    elif not finite(config.get("wall_ceiling_s"), positive=True) or (
        release["expires_at"] < now + config["wall_ceiling_s"]
    ):
        reasons.append("release_expires_before_wall_ceiling")
    for key in ("stop_argv",):
        value = config.get(key)
        if not isinstance(value, list) or not value or any(
            not isinstance(part, str) or not part for part in value
        ):
            reasons.append("operator_stop_command_missing")
    return {
        "status": "blocked" if reasons else "operator_preflight_only",
        "reasons": reasons, "sources": sources,
        "delegate_entry": str(ROOT / EXTENSION),
        "evidence_role": "operator attestations and byte hashes, not native qualification proof",
        "permit_scope": "IdentificationPermit open-loop assignments only; never MPC",
        "native_dispatched": False, "mpc_permit": False, "calibrated": False,
        "source_comparator": SOURCE_HASHES["gateway.py"],
        "r21_protocol_alone_authorizes_native": False,
        "active_clock": "conservative entire delegate lifetime including startup/drain",
        "authority_clock": local_clock_domain(),
        "provider_validity_clock": "unix_utc; must cover full authorized wall window",
    }


def worker_preflight_result(config, protocol, reasons, sources, now, wall_now):
    """Provider control stays external; native identification retains its own clock."""
    release = config.get("operator_release", {})
    authority = config.get("external_controller_authorization", {})
    placement = config.get("native_execution", {})
    if any(not isinstance(value, dict) for value in (release, authority, placement)):
        return {"status": "blocked", "reasons": [*reasons, "external_authority_schema"],
                "native_dispatched": False, "mpc_permit": False}
    if (
        release.get("approved") is not True or release.get("scope") != "h16_open_loop_measurement"
        or release.get("clock_domain") != local_clock_domain()
        or release.get("pod_id") != config.get("existing_pod_id")
        or release.get("run_id") != config.get("identification_permit", {}).get("run_id")
        or release.get("source_commit") != protocol.get("source_commit")
        or not finite(release.get("expires_at"))
        or release["expires_at"] < now + config.get("wall_ceiling_s", float("inf"))
        or release.get("protocol_sha256") != hashlib.sha256(json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        or authority.get("approved") is not True
        or authority.get("scope") != "external_watchdog_existing_pod_h16"
        or authority.get("pod_id") != config.get("existing_pod_id")
        or authority.get("run_id") != config.get("identification_permit", {}).get("run_id")
        or authority.get("protocol_sha256") != H16_PROTOCOL_SHA
        or not wall_valid(authority, now=wall_now, duration=config.get("wall_ceiling_s", 0))
        or not isinstance(authority.get("hostname"), str)
        or authority["hostname"] in ("UNKNOWN", platform.node())
        or not isinstance(authority.get("clock_domain"), str)
        or authority["clock_domain"] in ("UNKNOWN", local_clock_domain())
        or placement.get("mode") != "native_host"
    ):
        reasons.append("external_controller_and_native_authority_binding")
    return {"status": "blocked" if reasons else "native_worker_preflight_only",
            "reasons": reasons, "sources": sources, "native_dispatched": False,
            "mpc_permit": False, "calibrated": False}


def source_manifest():
    qualify_source()
    paths = set(SOURCE_HASHES)
    for directory in ("mpc_lab", "prefill_lab", "experiments/mpc-research/native_observer"):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).glob("*.py"))
    paths.update((
        "mpc_local/native_pilot.py", "mpc_local/control.py",
        "experiments/mpc-research/native_observer/collect_interventions.py",
        "experiments/mpc-research/native-v2/launch_batch_service.py",
        "experiments/mpc-research/native-v2/batch_capture_bridge.py",
    ))
    return {
        "schema": "r21-h16-client-source-manifest-v1",
        "serving_commit": SERVING_COMMIT,
        "serving_gateway_sha256": SOURCE_HASHES["gateway.py"],
        "deployment_commit": DEPLOYMENT_COMMIT,
        "deployment_binding": "provenance only, not actual runtime qualification",
        "client_sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sorted(paths)
        },
        "native_versions": NATIVE_VERSIONS,
        "native_qualified": False,
        "deployment_instruction": (
            "Copy exactly client_sources preserving relative paths into an operator-approved "
            "native-host ROOT; no credentials in this manifest. Use that host's pinned native "
            "interpreter. Bind fresh config/receipts to its actual host/boot clock domain. "
            "Set PYTHONPATH=ROOT:ROOT/experiments/mpc-research:"
            "ROOT/experiments/mpc-research/native_observer. "
            "On the distinct external controller run CONTROLLER_PYTHON -m "
            "mpc_local.native_pilot --mode controller-template, then explicitly supply all "
            "UNKNOWN inputs. Only its --mode controller --operator-release --config CONFIG "
            "--protocol ORIGINAL_H16 starts the watchdog before strict-key SSH native-worker "
            "prelaunch. Legacy measure/worker refuse Pod self-stop. Native interpreter/source/"
            "boot and model-free empty-owner/CAS readiness remain worker-owned. External "
            "controller retrieves artifacts before same-Pod STOP and TLS-authenticated GET. "
            "EXITED observes compute exit, cost is a tariff, and retained storage stays billed. "
            "Retained mounts require a verified prelaunch Pod/path/size descriptor, not a "
            "global volume UUID. The generated provider-stop adapter uses only explicit "
            "controller-bound --execute POST /v2/pods/id/action with action stop. "
            "Full billing closure is not proved by GET. Original ab883 and dual provenance "
            "remain required; Q1-Q5 are postcollection labels."
        ),
    }


def fixture_document(value):
    if isinstance(value, dict):
        if any(value.get(k) is True for k in ("fixture", "synthetic", "cpu_fixture")):
            return True
        if value.get("native") is False:
            return True
        return any(fixture_document(v) for v in value.values())
    return isinstance(value, list) and any(fixture_document(v) for v in value)


def source_bundle(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "results"):
        raise ValueError("bundle_outside_owned_results")
    manifest = source_manifest()
    with path.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, expected in manifest["client_sources"].items():
            raw = (ROOT / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("bundle_source_changed")
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        raw = json.dumps(manifest, sort_keys=True, allow_nan=False).encode() + b"\n"
        info = tarfile.TarInfo("h16-client-source-manifest.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_manifest": manifest, "native_qualified": False}


def native_entry_reasons(config, protocol, *, worker_phase=False):
    """Never promote dry-run flags/hash checks to a real measurement release."""
    reasons = []
    if fixture_document(config) or fixture_document(protocol):
        reasons.append("fixture_document_not_native")
    evidence, bundles = config.get("evidence"), config.get("bundle_evidence")
    if evidence is not None and not isinstance(evidence, dict) or not isinstance(bundles, list):
        reasons.append("native_evidence_missing_or_unknown")
    for record in (
        *(evidence.values() if isinstance(evidence, dict) else ()),
        *(bundles if isinstance(bundles, list) else ()),
        *((config.get("provider_rate_evidence"), config.get("billing_stop_evidence"))
          if not worker_phase else ()),
    ):
        try:
            if fixture_document(artifact(record)):
                reasons.append("fixture_document_not_native")
        except (TypeError, KeyError, ValueError, OSError, AttributeError):
            reasons.append("native_evidence_missing_or_unknown")
    placement = config.get("native_execution", {})
    if not isinstance(placement, dict):
        placement = {}
    if (
        placement.get("mode") != "native_host"
        or placement.get("root") != str(ROOT)
        or placement.get("python") != str(Path(sys.executable).resolve())
        or placement.get("hostname") != platform.node()
        or placement.get("clock_domain") != local_clock_domain()
    ):
        reasons.append("native_host_interpreter_or_boot_binding")
    for name, expected in NATIVE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual is None or actual.partition("+")[0] != expected:
            reasons.append("native_environment:" + name)
            continue
        try:
            spec = importlib.util.find_spec(name)
            distribution_root = Path(importlib.metadata.distribution(name).locate_file("")).resolve()
            if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(
                distribution_root,
            ):
                reasons.append("native_import_placement:" + name)
        except (ValueError, ImportError, OSError, importlib.metadata.PackageNotFoundError):
            reasons.append("native_import_placement:" + name)
    try:
        manifest = artifact(placement["client_manifest"])
        if manifest.get("client_sources") != source_manifest()["client_sources"]:
            raise ValueError("client_source_manifest_mismatch")
    except (TypeError, KeyError, ValueError, OSError):
        reasons.append("native_client_source_manifest")
    try:
        if {r["sha256"] for r in bundles} != PROVENANCE_SHAS:
            raise ValueError("original_dual_provenance")
        launch = json.loads(Path(config["vllm_launch"]).read_bytes())
        if (
            launch.get("serving_base_commit") != SERVING_COMMIT
            or launch.get("deployment_bundle_commit") != DEPLOYMENT_COMMIT
            or launch.get("r21_client_source_manifest_sha256") != placement["client_manifest"]["sha256"]
        ):
            raise ValueError("launch_three_source_identities")
    except (TypeError, KeyError, ValueError, OSError):
        reasons.append("original_dual_provenance_or_launch_identity")
    try:
        if worker_phase:
            return sorted(set(reasons + postconsumer_reasons(config)))
        path = Path(config["post_stop_receipt_path"]).resolve()
        if not path.is_relative_to(ROOT / "results") or path.exists():
            raise ValueError("post_stop_receipt_must_be_fresh")
    except (TypeError, KeyError, ValueError, OSError):
        reasons.append("fresh_owned_post_stop_receipt_required")
    try:
        consumer = Path(config["postcollection_consumer"]).resolve()
        if not consumer.is_relative_to(ROOT) or hashlib.sha256(
            consumer.read_bytes(),
        ).hexdigest() != POST_CONSUMER_SHA:
            raise ValueError("postcollection_consumer")
    except (TypeError, KeyError, ValueError, OSError):
        reasons.append("pinned_postcollection_consumer_required")
    return sorted(set(reasons))


def postconsumer_reasons(config):
    try:
        consumer = Path(config["postcollection_consumer"]).resolve()
        if not consumer.is_relative_to(ROOT) or hashlib.sha256(
            consumer.read_bytes(),
        ).hexdigest() != POST_CONSUMER_SHA:
            raise ValueError("postcollection_consumer")
        return []
    except (TypeError, KeyError, ValueError, OSError):
        return ["pinned_postcollection_consumer_required"]


async def basic_runtime_probe(client, parameters, *, pending_path, authorization):
    if (not isinstance(authorization, IdentificationPermit)
            or not authorization.allows(parameters.identity)
            or authorization.clock_domain != local_clock_domain()
            or time.monotonic() >= authorization.expires_at):
        raise ValueError("basic_probe_explicit_identification_authority")
    response = await client.get("/admin/mpc/state")
    response.raise_for_status()
    document = response.json()
    snapshot = qualify_snapshot(
        document, parameters, run_id=document["run_id"],
        clock_domain=local_clock_domain(), now=time.monotonic(),
    )
    if snapshot.requests or snapshot.reserved != 0:
        raise ValueError("basic_probe_nonempty_owner")
    permit = IdentificationPermit(
        parameters.identity, approved=True, run_id=snapshot.run_id,
        clock_domain=local_clock_domain(), expires_at=authorization.expires_at,
    )
    driver = GuardedActuator(client, parameters, pending_path, permit=permit, enabled=True)
    decision = Decision(snapshot.budget, None, "identification_assignment", False,
                        None, None, 0.0, 0, 0, parameters.identity)
    reply = await driver.submit(snapshot, decision)
    if reply.get("status") != "applied":
        raise ValueError("basic_probe_unknown_or_refused_cas:" + reply.get("status", "unknown"))
    after = await driver._state()
    if after.requests or after.budget != snapshot.budget or (
        after.budget_version != snapshot.budget_version + 1
    ):
        raise ValueError("basic_probe_readback_mismatch")
    return {
        "schema": "r21-h16-basic-runtime-probe-v1", "status": "basic_ready",
        "gateway_run_id": snapshot.run_id, "clock_domain": snapshot.clock_domain,
        "hostname": platform.node(), "identity": asdict(parameters.identity),
        "campaign_run_id": authorization.run_id,
        "snapshot": document, "same_budget_ack": reply,
        "owner_inventory_empty": True, "q1_q5_role": "postcollection only",
        "native_visibility_proved": False, "authentication_proved": False,
        "mpc_permit": False,
    }


def postcollection(config, protocol_path):
    path = Path(config["postcollection_consumer"]).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != POST_CONSUMER_SHA:
        raise ValueError("postcollection_consumer_hash")
    spec = importlib.util.spec_from_file_location("r21_pinned_h16_consumer", path)
    consumer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = consumer
    spec.loader.exec_module(consumer)
    binding = consumer.h16_source_binding(protocol_path, ROOT, ROOT / EXTENSION)
    output = Path(config["output"]).resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("postcollection_output_scope")
    labels = consumer.verify_h16_run(binding, output)
    index = json.loads((output / "index.json").read_bytes())
    launch = index.get("gpu_environment", {}).get("native_launch", {})
    expected = {
        "serving_base_commit": SERVING_COMMIT, "deployment_bundle_commit": DEPLOYMENT_COMMIT,
        "r21_client_source_manifest_sha256": config["native_execution"]["client_manifest"]["sha256"],
    }
    failures = []
    if any(launch.get(k) != v for k, v in expected.items()):
        failures.append("actual_run_index_three_source_identities")
    if labels.get("itt_slots", {}).get("noncompliant", 0):
        failures.append("all_assigned_noncompliance_retained")
    return {
        "status": "retrospective_labels" if labels["ok"] and not failures else "nonqualification",
        "consumer_sha256": POST_CONSUMER_SHA, "consumer_labels": labels,
        "additional_failures": failures, "authentication": False, "fsync": False,
        "online_cut": False, "native_claim": False, "mpc_permit": False,
    }


def provider_stop_receipt(path, *, pod_id, run_id, started_utc, finished_utc):
    result = {"verified": False, "reason": "provider_stop_receipt_missing",
              "provider_state": "unknown", "billing_state": "unknown"}
    try:
        raw = Path(path).read_bytes()
        receipt = json.loads(raw)
        result["sha256"] = hashlib.sha256(raw).hexdigest()
        if not isinstance(receipt, dict):
            raise ValueError("provider_receipt_schema")
        result["provider_state"] = receipt.get("provider_state", "unknown")
        result["billing_state"] = receipt.get("billing_state", "unknown")
        if (
            fixture_document(receipt)
            or receipt.get("schema") != "r21-provider-post-stop-envelope-v1"
            or receipt.get("pod_id") != pod_id or receipt.get("run_id") != run_id
            or receipt.get("receipt_source") != "provider"
            or receipt.get("verified_provider_observation") is not True
            or not finite(receipt.get("observed_at_utc_s"))
            or not started_utc <= receipt["observed_at_utc_s"] <= finished_utc
            or receipt.get("provider_state") != "STOPPED"
            or receipt.get("billing_state") != "STOPPED"
        ):
            raise ValueError("provider_state_or_billing_not_verified")
        result.update(verified=True, reason=None)
    except (OSError, ValueError, TypeError, AttributeError):
        if "sha256" in result:
            result["reason"] = "provider_state_or_billing_not_verified"
    return result


async def bounded_delegate(argv, *, stop_argv, active_s, wall_s, spend_usd, rate,
                           expires_at, log_path, grace_s=5.0, provider_receipt_path=None,
                           pod_id=None, run_id=None, env=None):
    """Owned process group; caller must separately release native measurement."""
    values = (active_s, wall_s, spend_usd, rate, expires_at, grace_s)
    if any(not finite(v, positive=True) for v in values):
        raise ValueError("invalid_watchdog_limits")
    started = time.monotonic()
    started_utc = time.time()
    allowed = min(active_s, wall_s, spend_usd / rate * 3600, expires_at - started)
    if allowed <= grace_s:
        raise ValueError("watchdog_shutdown_reserve")
    for command in (argv, stop_argv):
        if not isinstance(command, list) or not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise ValueError("invalid_delegate_command")
    status, returncode, stop_code = "completed", None, None
    stop_started_utc = started_utc
    with Path(log_path).open("xb") as stream:
        child = None
        try:
            child = await asyncio.create_subprocess_exec(
                *argv, stdout=stream, stderr=stream, start_new_session=True, env=env,
            )
            await asyncio.wait_for(child.wait(), allowed - grace_s)
            returncode = child.returncode
            if returncode:
                status = "delegate_failed"
        except TimeoutError:
            status = "watchdog_stop"
        except OSError:
            status = "delegate_spawn_failed"
        finally:
            # The group is created above; no unrelated service PID is targeted.
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(), grace_s / 2)
                except TimeoutError:
                    os.killpg(child.pid, signal.SIGKILL)
                    await child.wait()
                returncode = child.returncode
            try:
                stop_started_utc = time.time()
                stop = await asyncio.create_subprocess_exec(*stop_argv, stdout=stream, stderr=stream)
                try:
                    await asyncio.wait_for(stop.wait(), grace_s / 2)
                except TimeoutError:
                    stop.kill()
                    await stop.wait()
                stop_code = stop.returncode
            except OSError:
                stop_code = -1
            if stop_code:
                status = "operator_stop_failed"
    finished = time.monotonic()
    finished_utc = time.time()
    if finished - started > allowed:
        status = "ceiling_exceeded"
    receipt = provider_stop_receipt(
        provider_receipt_path, pod_id=pod_id, run_id=run_id,
        started_utc=stop_started_utc, finished_utc=finished_utc,
    ) if provider_receipt_path is not None else {"verified": False, "reason": "not_requested"}
    if status == "completed" and provider_receipt_path is not None and not receipt["verified"]:
        status = "provider_stop_unverified"
    return {
        "status": status, "delegate_returncode": returncode,
        "operator_stop_returncode": stop_code,
        "started_at": started, "finished_at": finished,
        "active_charging": "entire delegate lifetime conservatively bounded",
        "provider_billing_stopped": receipt["verified"], "physical_shutdown_proved": False,
        "provider_stop_receipt": receipt,
        "provider_proof_role": "operator-supplied provider observation; not provider authentication",
        "started_at_utc_s": started_utc, "finished_at_utc_s": finished_utc,
        "estimated_delegate_charge_usd": rate * (finished - started) / 3600,
        "hard_spend_containment_proved": False,
        "authority_deadline": expires_at,
        "legacy_internal_permit_rewritten": False,
        "mpc_permit": False,
    }


def measurement_command(config_path, protocol_path, config, *, watchdog_open=False):
    command = [sys.executable, str(ROOT / EXTENSION), "--execute",
               "--config", str(config_path), "--protocol", str(protocol_path)]
    for flag, key in (("--output", "output"), ("--vllm-launch", "vllm_launch"),
                      ("--gateway-plan", "gateway_plan")):
        path = Path(config[key]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("delegate_path_outside_owned_root")
        command.extend((flag, str(path)))
    log_path = Path(config["watchdog_log"]).resolve()
    if not log_path.is_relative_to(ROOT / "results") or (log_path.exists() and not watchdog_open):
        raise ValueError("fresh_owned_watchdog_log_required")
    command.extend(("--backend-free", "--epoch-qualified"))
    return command


async def collection_worker(config_path, protocol_path, config, protocol, *, external=False):
    import httpx

    if (not external and os.getenv("MPC_H16_WATCHDOG_PARENT_PID") != str(os.getppid())
            or os.getsid(0) != os.getpid()):
        raise ValueError("owned_watchdog_handoff_required")
    for port in (8000, 8080):
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
        except ConnectionRefusedError:
            continue
        writer.close()
        await writer.wait_closed()
        raise ValueError("preexisting_loopback_service_not_backend_free:" + str(port))
    output = Path(config["output"]).resolve()
    probe_dir = output.with_name(output.name + "-basic-runtime-probe")
    if not probe_dir.is_relative_to(ROOT / "results") or probe_dir.exists():
        raise ValueError("fresh_owned_basic_probe_directory")
    probe_dir.mkdir()
    parameters = ModelParameters(
        cost_spec=CostSpec(**protocol["cost_spec"]), model_version="native-id-unfitted-v1",
        calibration_version="unfitted-not-validated",
    )
    command = [
        sys.executable, str(ROOT / "experiments/mpc-research/native-v2/launch_batch_service.py"),
        "gateway", "--config", str(config_path), "--plan", str(Path(config["gateway_plan"]).resolve()),
        "--python", sys.executable, "--evidence", str(probe_dir),
        "--identification-protocol", str(protocol_path),
    ]
    with (probe_dir / "gateway-probe-console.txt").open("xb") as stream:
        child = await asyncio.create_subprocess_exec(*command, stdout=stream, stderr=stream)
        try:
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:8080",
                headers={"x-admin-api-key": config["ADMIN_API_KEY"]},
                trust_env=False, follow_redirects=False, timeout=1,
            ) as client:
                async def ready():
                    while True:
                        if child.returncode is not None:
                            raise ValueError("basic_gateway_start_failed")
                        try:
                            return await basic_runtime_probe(
                                client, parameters, pending_path=probe_dir / "probe-pending.json",
                                authorization=IdentificationPermit(
                                    parameters.identity,
                                    approved=config["identification_permit"]["approved"],
                                    run_id=config["identification_permit"]["run_id"],
                                    clock_domain=config["identification_permit"]["clock_domain"],
                                    expires_at=min(config["operator_release"]["expires_at"],
                                                   config["identification_permit"]["expires_at"]),
                                ),
                            )
                        except httpx.ConnectError:
                            await asyncio.sleep(.05)
                receipt = await asyncio.wait_for(ready(), 10)
                (probe_dir / "basic-runtime-receipt.json").write_text(
                    json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n",
                )
        finally:
            if child.returncode is None:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), 2)
                except TimeoutError:
                    child.kill()
                    await child.wait()
    if child.returncode not in (0, -signal.SIGTERM):
        raise ValueError("basic_gateway_shutdown_failed")
    journal = audit_owner_log(
        (probe_dir / "gateway-telemetry.jsonl").read_bytes(),
        json.loads((probe_dir / "gateway-telemetry-status.json").read_bytes()),
        run_id=receipt["gateway_run_id"], clock_domain=local_clock_domain(),
        identity=parameters.identity, request_costs={},
    )
    (probe_dir / "basic-closed-owner-receipt.json").write_text(
        json.dumps(journal, sort_keys=True, allow_nan=False) + "\n",
    )
    return measurement_command(config_path, protocol_path, config, watchdog_open=True)


def external_controller_template():
    return {
        "schema": "r21-existing-pod-external-controller-v1",
        "controller_execution": {"root": "UNKNOWN", "python": "UNKNOWN",
                                 "hostname": "UNKNOWN", "clock_domain": "UNKNOWN"},
        "native_execution": {"root": "UNKNOWN", "python": "UNKNOWN",
                             "hostname": "UNKNOWN", "clock_domain": "UNKNOWN"},
        "existing_pod_id": "UNKNOWN", "run_id": "UNKNOWN",
        "ssh_target": "UNKNOWN", "known_hosts": {"path": "UNKNOWN", "sha256": "UNKNOWN"},
        "native_config_path": "UNKNOWN", "native_config_sha256": "UNKNOWN",
        "native_protocol_path": "UNKNOWN", "native_output": "UNKNOWN",
        "native_owner_receipt": "UNKNOWN", "output": "UNKNOWN",
        "authorization": {"approved": False, "scope": "UNKNOWN", "pod_id": "UNKNOWN",
                          "run_id": "UNKNOWN", "protocol_sha256": H16_PROTOCOL_SHA,
                          "validity_clock": "unix_utc", "valid_from_utc_s": "UNKNOWN",
                          "valid_until_utc_s": "UNKNOWN"},
        "provider_rate_evidence": {"path": "UNKNOWN", "sha256": "UNKNOWN"},
        "provider_control_evidence": {"path": "UNKNOWN", "sha256": "UNKNOWN"},
        "provider_stop_url": "UNKNOWN", "provider_readback_url": "UNKNOWN",
        "provider_api_token_env": "UNKNOWN",
        "provider_hourly_usd": "UNKNOWN", "active_ceiling_s": 2700,
        "wall_ceiling_s": 3600, "spend_ceiling_usd": 1,
        "shutdown_stage_s": 30,
        "prelaunch_mount_evidence": {"path": "UNKNOWN", "sha256": "UNKNOWN"},
        "prelaunch_mount_descriptor_required": {
            "schema": "r21-prelaunch-persistent-mount-v1", "pod_id": "UNKNOWN",
            "run_id": "UNKNOWN", "path": "UNKNOWN", "size": "UNKNOWN",
            "verified_prelaunch": False, "authenticated_provider": False,
            "receipt_source": "provider_https", "response_sha256": "UNKNOWN",
            "observed_at_utc_s": "UNKNOWN", "validity_clock": "unix_utc",
            "valid_from_utc_s": "UNKNOWN", "valid_until_utc_s": "UNKNOWN",
            "identity_semantics": "Pod-local persistent path/size, not global volume UUID",
        },
        "persistent_native_evidence": "UNKNOWN",
        "native_worker_required_inputs": {
            "worker_owner_receipt": "UNKNOWN",
            "external_controller_authorization": {
                "approved": False, "scope": "external_watchdog_existing_pod_h16",
                "hostname": "UNKNOWN", "clock_domain": "UNKNOWN",
                "pod_id": "UNKNOWN", "run_id": "UNKNOWN",
                "protocol_sha256": H16_PROTOCOL_SHA, "validity_clock": "unix_utc",
                "valid_from_utc_s": "UNKNOWN", "valid_until_utc_s": "UNKNOWN",
            },
            "remaining_native_inputs": "Original explicit native permit/release/source/protocol/paths required",
        },
    }


def controller_preflight(config, protocol_sha, *, wall_now=None):
    wall_now = time.time() if wall_now is None else wall_now
    reasons = []
    try:
        if fixture_document(config):
            raise ValueError("fixture_config_not_authority")
        controller, native = config["controller_execution"], config["native_execution"]
        if controller != {"root": str(ROOT), "python": str(Path(sys.executable).resolve()),
                          "hostname": platform.node(), "clock_domain": local_clock_domain()}:
            raise ValueError("actual_controller_placement")
        if (native["hostname"] in ("UNKNOWN", controller["hostname"])
                or native["clock_domain"] in ("UNKNOWN", controller["clock_domain"])
                or not isinstance(native["hostname"], str)
                or not isinstance(native["clock_domain"], str)):
            raise ValueError("distinct_native_clock_and_host")
        for key in ("root", "python"):
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+", native[key]) or ".." in Path(native[key]).parts:
                raise ValueError("native_path")
        for key in ("native_config_path", "native_protocol_path", "native_output", "native_owner_receipt"):
            value = config[key]
            if (not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
                    or not Path(value).is_relative_to(native["root"])
                    or ".." in Path(value).parts):
                raise ValueError("remote_path_scope")
        for key in ("native_output", "native_owner_receipt"):
            if not Path(config[key]).is_relative_to(Path(native["root"]) / "results"):
                raise ValueError("native_evidence_scope")
        owner_dir = Path(config["native_owner_receipt"]).parent
        if (owner_dir == Path(native["root"]) / "results"
                or owner_dir == Path(config["native_output"])
                or owner_dir.is_relative_to(Path(config["native_output"]))
                or Path(config["native_output"]).is_relative_to(owner_dir)):
            raise ValueError("narrow_separate_worker_evidence_directory")
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9][A-Za-z0-9.-]*", config["ssh_target"])
                or not re.fullmatch(r"[0-9a-f]{64}", config["native_config_sha256"])):
            raise ValueError("ssh_target_or_native_config_binding")
        host_record = config["known_hosts"]
        host_path = Path(host_record["path"]).resolve()
        if (not host_path.is_relative_to(ROOT) or not host_path.is_file()
                or hashlib.sha256(host_path.read_bytes()).hexdigest() != host_record["sha256"]):
            raise ValueError("pinned_known_hosts")
        output = Path(config["output"]).resolve()
        if not output.is_relative_to(ROOT / "results") or output.exists():
            raise ValueError("fresh_controller_evidence_required")
        pod, run = config["existing_pod_id"], config["run_id"]
        if any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", v)
               or v == "UNKNOWN" for v in (pod, run)):
            raise ValueError("existing_pod_and_run_binding")
        auth = config["authorization"]
        if (auth.get("approved") is not True or auth.get("pod_id") != pod
                or auth.get("run_id") != run or protocol_sha != H16_PROTOCOL_SHA
                or auth.get("protocol_sha256") != protocol_sha
                or auth.get("scope") not in (
                    "h16_existing_pod_full_billing_stop",
                    "h16_existing_pod_stop_gpu_keep_preexisting_storage")
                or not wall_valid(auth, now=wall_now, duration=config["wall_ceiling_s"])):
            raise ValueError("scoped_existing_pod_portable_authorization")
        for key, ceiling in (("active_ceiling_s", 2700), ("wall_ceiling_s", 3600),
                             ("spend_ceiling_usd", 1), ("provider_hourly_usd", 1),
                             ("shutdown_stage_s", 30)):
            if not finite(config[key], positive=True) or config[key] > ceiling:
                raise ValueError("campaign_ceiling:" + key)
        if (config["active_ceiling_s"] < 2400
                or config["wall_ceiling_s"] < config["active_ceiling_s"]
                or 3 * config["shutdown_stage_s"] >= config["active_ceiling_s"]
                or config["provider_hourly_usd"] * config["wall_ceiling_s"] / 3600
                > config["spend_ceiling_usd"]):
            raise ValueError("original_offered_window_or_campaign_spend")
        price = artifact(config["provider_rate_evidence"])
        control = artifact(config["provider_control_evidence"])
        for record in (price, control):
            if (fixture_document(record) or record.get("pod_id") != pod
                    or record.get("run_id") != run
                    or not wall_valid(record, now=wall_now, duration=config["wall_ceiling_s"])):
                raise ValueError("provider_artifact_scope")
        if (price.get("verified") is not True
                or price.get("existing_pod") is not True
                or price.get("includes_compute_storage_fees") is not True
                or price.get("hourly_rate_usd") != config["provider_hourly_usd"]):
            raise ValueError("total_campaign_rate_unknown")
        if (config.get("provider_readback_url") !=
                "https://api.runpod.io/v2/pods/" + pod
                or not re.fullmatch(r"[A-Z][A-Z0-9_]*", config.get("provider_api_token_env", ""))
                or config["provider_api_token_env"] == "UNKNOWN"
                or not os.environ.get(config["provider_api_token_env"])
                or control.get("provider_readback_url") != config["provider_readback_url"]):
            raise ValueError("scoped_authenticated_provider_readback")
        if (config.get("provider_stop_url") != config["provider_readback_url"] + "/action"
                or control.get("provider_stop_url") != config["provider_stop_url"]
                or control.get("stop_body") != {"action": "stop"}
                or control.get("stop_adapter") != "owned_http_stop_v1"):
            raise ValueError("owned_stop_only_HTTP_not_explicitly_bound")
        if (control.get("operation") != "stop_existing_pod"
                or control.get("verified_control") is not True
                or control.get("ssh_target") != config["ssh_target"]
                or control.get("native_execution") != native
                or control.get("native_config_sha256") != config["native_config_sha256"]):
            raise ValueError("no_start_terminate_delete_authority")
        if auth["scope"] == "h16_existing_pod_stop_gpu_keep_preexisting_storage":
            prelaunch_mount_descriptor(config, now=wall_now, duration=config["wall_ceiling_s"])
            if (price.get("prelaunch_mount_evidence") != config["prelaunch_mount_evidence"]
                    or control.get("prelaunch_mount_evidence") != config["prelaunch_mount_evidence"]
                    or price.get("retained_storage_metered") is not True):
                raise ValueError("explicit_preexisting_storage_metering")
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        reasons.append(str(exc))
    return {"status": "blocked" if reasons else "controller_preflight_only",
            "reasons": reasons, "native": False, "mpc_permit": False}


def controller_commands(config, *, stop_authorization=None):
    native = config["native_execution"]
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "UserKnownHostsFile=" + str(Path(config["known_hosts"]["path"]).resolve()),
               "-o", "ConnectTimeout=10"]
    remote = ["env", "PYTHONPATH=" + ":".join((
        native["root"], native["root"] + "/experiments/mpc-research",
        native["root"] + "/experiments/mpc-research/native_observer",
    )), native["python"], "-m", "mpc_local.native_pilot", "--mode", "native-worker",
        "--operator-release", "--config", config["native_config_path"],
        "--protocol", config["native_protocol_path"],
        "--expected-config-sha256", config["native_config_sha256"]]
    output = Path(config["output"])
    return {
        "worker": ["ssh", *options, "-T", "--", config["ssh_target"], "exec " + shlex.join(remote)],
        "retrieve": ["scp", *options, "-r", "--",
                     config["ssh_target"] + ":" + config["native_output"],
                     config["ssh_target"] + ":" + str(Path(config["native_owner_receipt"]).parent),
                     config["ssh_target"] + ":" + config["native_output"] + "-basic-runtime-probe",
                     str(output / "native-artifacts")],
        "stop": [sys.executable, "-m", "mpc_local.native_pilot", "--mode", "provider-stop",
                 "--execute", "--config",
                 stop_authorization["path"] if stop_authorization else "UNKNOWN",
                 "--controller-bound-sha256",
                 stop_authorization["sha256"] if stop_authorization else "UNKNOWN",
                 "--pod-id", config["existing_pod_id"], "--run-id", config["run_id"]],
    }


async def controller_command(argv, timeout_s, log_path, *, provider_operation=False, env=None):
    child, timed_out = None, False
    with log_path.open("xb") as stream:
        try:
            child = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.DEVNULL if provider_operation else stream,
                stderr=asyncio.subprocess.DEVNULL if provider_operation else stream,
                start_new_session=True, env=env,
            )
            await asyncio.wait_for(child.wait(), timeout_s)
        except TimeoutError:
            timed_out = True
        except OSError:
            return {"returncode": None, "timed_out": False, "spawn_failed": True}
        finally:
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(), .5)
                except TimeoutError:
                    os.killpg(child.pid, signal.SIGKILL)
                    await child.wait()
    return {"returncode": child.returncode, "timed_out": timed_out,
            "local_owned_pid": child.pid, "remote_stop_proved": False}


async def authenticated_provider_readback(config, *, client=None):
    import httpx

    token = os.environ.get(config["provider_api_token_env"])
    if not token:
        raise ValueError("provider_authentication_missing")
    owned_client = client is None
    if owned_client:
        client = httpx.AsyncClient(verify=True, trust_env=False, follow_redirects=False, timeout=10)
    try:
        response = await client.get(
            config["provider_readback_url"], headers={"Authorization": "Bearer " + token},
        )
        response.raise_for_status()
        if response.status_code != 200 or len(response.content) > 1024 * 1024:
            raise ValueError("provider_response_unavailable")
        body = response.json()
        if (not isinstance(body, dict) or body.get("id") != config["existing_pod_id"]
                or "runtime" not in body):
            raise ValueError("provider_pod_identity")
        safe = {key: provider_safe_value(body[key]) for key in (
            "id", "status", "actions", "gpu", "mounts", "cost",
        ) if key in body}
        safe["runtime"] = None if body.get("runtime") is None else {"present": True}
        return {
            "schema": "r21-external-post-stop-v1", "receipt_source": "provider_https",
            "authenticated_provider": owned_client,
            "pod_id": body["id"], "run_id": config["run_id"],
            "observed_at_utc_s": time.time(), "provider_state": safe.get("status"),
            "compute_stopped": safe.get("status") == "EXITED" and safe["runtime"] is None,
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "provider_fields": safe, "cost_semantics": "hourly_compute_tariff_not_live_billing",
            "all_billing_stopped": False,
            "billing_observation": "GET Pod is not proof of total billing closure",
        }
    except httpx.HTTPError:
        raise ValueError("provider_readback_HTTP_error") from None
    finally:
        if owned_client:
            await client.aclose()


def provider_safe_value(value, *, depth=0):
    if depth > 3:
        return {"omitted": True}
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, list):
        return [provider_safe_value(item, depth=depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {key: provider_safe_value(item, depth=depth + 1) for key, item in value.items()
                if key in {"id", "name", "type", "count", "memory", "mountPath", "path",
                           "size", "sizeGb", "sizeInGb", "storageId", "volumeId",
                           "networkVolumeId", "persistent", "readOnly",
                           "amount", "currency", "unit", "rate", "value"}}
    return {"omitted": True}


def prelaunch_mount_descriptor(config, *, now, duration=0):
    record = artifact(config["prelaunch_mount_evidence"])
    start, end = record.get("valid_from_utc_s"), record.get("valid_until_utc_s")
    path, size, observed = record.get("path"), record.get("size"), record.get("observed_at_utc_s")
    if (fixture_document(record)
            or record.get("schema") != "r21-prelaunch-persistent-mount-v1"
            or record.get("pod_id") != config["existing_pod_id"]
            or record.get("run_id") != config["run_id"]
            or record.get("verified_prelaunch") is not True
            or record.get("authenticated_provider") is not True
            or record.get("receipt_source") != "provider_https"
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("response_sha256", ""))
            or record.get("validity_clock") != "unix_utc"
            or any(not finite(v) for v in (start, end, observed, now, duration, size))
            or not 0 <= start <= observed <= now <= now + duration <= end
            or size <= 0 or not isinstance(path, str) or not path.startswith("/")
            or ".." in Path(path).parts or path == "/"):
        raise ValueError("verified_prelaunch_pod_mount_descriptor_required")
    return record


def retained_mount_visible(mounts, descriptor, *, pod_id):
    if not isinstance(mounts, dict) or not isinstance(descriptor, dict):
        return False
    persistent = mounts.get("persistent")
    return (
        isinstance(persistent, dict) and descriptor.get("pod_id") == pod_id
        and finite(persistent.get("size"), positive=True)
        and persistent["size"] == descriptor.get("size")
        and persistent.get("path") == descriptor.get("path")
    )


def external_stop_observation(raw, config, *, started_utc, finished_utc):
    result = {"compute_stopped": False, "provider_billing_stopped": False,
              "storage_billing_continues": None, "accepted_scope": False,
              "authenticated_provider": False, "hard_spend_containment_proved": False}
    try:
        receipt = json.loads(raw)
        if (fixture_document(receipt)
                or receipt.get("schema") != "r21-external-post-stop-v1"
                or receipt.get("pod_id") != config["existing_pod_id"]
                or receipt.get("run_id") != config["run_id"]
                or receipt.get("receipt_source") != "provider_https"
                or receipt.get("authenticated_provider") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("response_sha256", ""))
                or not finite(receipt.get("observed_at_utc_s"))
                or not started_utc <= receipt["observed_at_utc_s"] <= finished_utc
                or receipt.get("provider_state") != "EXITED"
                or receipt.get("compute_stopped") is not True):
            raise ValueError("compute_stop_not_observed")
        result["compute_stopped"] = True
        result["authenticated_provider"] = True
        fields = receipt.get("provider_fields", {})
        descriptor = prelaunch_mount_descriptor(config, now=finished_utc)
        storage = retained_mount_visible(
            fields.get("mounts"), descriptor, pod_id=config["existing_pod_id"],
        )
        result["storage_billing_continues"] = True if storage else None
        if config["authorization"]["scope"] == "h16_existing_pod_full_billing_stop":
            result["accepted_scope"] = result["provider_billing_stopped"]
        else:
            result["accepted_scope"] = (
                storage and receipt.get("all_billing_stopped") is False)
        result["reason"] = None if result["accepted_scope"] else "billing_scope_not_proved"
    except (ValueError, KeyError, TypeError, AttributeError):
        result["reason"] = "compute_or_billing_observation_unknown"
    return result


def write_controller_stop_authorization(config, output):
    authorization = config["authorization"]
    document = {
        "schema": "r21-controller-bound-provider-stop-v1",
        "controller_execution": config["controller_execution"],
        "controller_pid": os.getpid(), "pod_id": config["existing_pod_id"],
        "run_id": config["run_id"], "provider_stop_url": config["provider_stop_url"],
        "provider_api_token_env": config["provider_api_token_env"],
        "provider_control_evidence": config["provider_control_evidence"],
        "authorization": {key: authorization[key] for key in (
            "approved", "scope", "pod_id", "run_id", "protocol_sha256",
            "validity_clock", "valid_from_utc_s", "valid_until_utc_s",
        )},
    }
    path = output / "controller-bound-stop.json"
    raw = json.dumps(document, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


async def provider_stop_adapter(document, *, pod_id, run_id, execute=False, client=None):
    import httpx

    blocked = {"status": "blocked", "request_sent": False, "native": False, "mpc_permit": False}
    try:
        if execute is not True:
            raise ValueError("explicit_stop_execute_required")
        auth = document["authorization"]
        actual = {"root": str(ROOT), "python": str(Path(sys.executable).resolve()),
                  "hostname": platform.node(), "clock_domain": local_clock_domain()}
        now = time.time()
        if (fixture_document(document)
                or document.get("schema") != "r21-controller-bound-provider-stop-v1"
                or document.get("controller_execution") != actual
                or type(document.get("controller_pid")) is not int
                or document["controller_pid"] != os.getppid()
                or any(not isinstance(v, str) or v == "UNKNOWN"
                       or not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in (pod_id, run_id))
                or document.get("pod_id") != pod_id or document.get("run_id") != run_id
                or auth.get("approved") is not True
                or auth.get("scope") not in (
                    "h16_existing_pod_full_billing_stop",
                    "h16_existing_pod_stop_gpu_keep_preexisting_storage")
                or auth.get("pod_id") != pod_id or auth.get("run_id") != run_id
                or auth.get("protocol_sha256") != H16_PROTOCOL_SHA
                or auth.get("validity_clock") != "unix_utc"
                or not finite(auth.get("valid_from_utc_s"))
                or not finite(auth.get("valid_until_utc_s"))
                or not 0 <= auth["valid_from_utc_s"] <= now < auth["valid_until_utc_s"]
                or document.get("provider_stop_url") !=
                "https://api.runpod.io/v2/pods/" + pod_id + "/action"):
            raise ValueError("controller_bound_existing_pod_stop_authority")
        control = artifact(document["provider_control_evidence"])
        if (fixture_document(control) or control.get("verified_control") is not True
                or control.get("operation") != "stop_existing_pod"
                or control.get("pod_id") != pod_id or control.get("run_id") != run_id
                or control.get("validity_clock") != "unix_utc"
                or not finite(control.get("valid_from_utc_s"))
                or not finite(control.get("valid_until_utc_s"))
                or not 0 <= control["valid_from_utc_s"] <= now < control["valid_until_utc_s"]
                or control.get("provider_stop_url") != document["provider_stop_url"]
                or control.get("stop_adapter") != "owned_http_stop_v1"
                or control.get("stop_body") != {"action": "stop"}):
            raise ValueError("verified_stop_only_control_binding")
        env_name = document["provider_api_token_env"]
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env_name) or env_name == "UNKNOWN":
            raise ValueError("provider_token_env_required")
        token = os.environ.get(env_name)
        if not token:
            raise ValueError("provider_token_env_required")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {**blocked, "reason": str(exc)}
    owned_client = client is None
    if owned_client:
        client = httpx.AsyncClient(verify=True, trust_env=False, follow_redirects=False, timeout=10)
    try:
        response = await client.post(
            document["provider_stop_url"], json={"action": "stop"},
            headers={"Authorization": "Bearer " + token},
        )
        return {
            "status": "stop_acknowledged" if 200 <= response.status_code < 300 else "stop_HTTP_failed",
            "request_sent": True, "http_status": response.status_code,
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "authenticated_transport": owned_client, "body_archived": False,
            "pod_id": pod_id, "run_id": run_id, "compute_stop_proved": False,
            "provider_billing_stopped": False, "native": False, "mpc_permit": False,
        }
    except httpx.HTTPError:
        return {"status": "stop_transport_unknown", "request_sent": None,
                "compute_stop_proved": False, "provider_billing_stopped": False}
    finally:
        if owned_client:
            await client.aclose()


async def external_controller(config, protocol_sha, *, runner=None, observer=None):
    result = controller_preflight(config, protocol_sha)
    if result["status"] == "blocked":
        return result
    if runner is None:
        async def runner(argv, timeout_s, log_path):
            env = dict(os.environ)
            provider_operation = log_path.stem == "stop"
            if not provider_operation:
                env.pop(config["provider_api_token_env"], None)
            return await controller_command(
                argv, timeout_s, log_path, provider_operation=provider_operation, env=env,
            )
    observer = authenticated_provider_readback if observer is None else observer
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True)
    (output / "native-artifacts").mkdir()
    stop_authorization = write_controller_stop_authorization(config, output)
    commands = controller_commands(config, stop_authorization=stop_authorization)
    started, started_utc = time.monotonic(), time.time()
    allowed = min(config["active_ceiling_s"], config["wall_ceiling_s"],
                  config["spend_ceiling_usd"] / config["provider_hourly_usd"] * 3600)
    grace = config["shutdown_stage_s"]
    events = [{"event": "external_watchdog_started", "at": started,
               "deadline": started + allowed, "clock_domain": local_clock_domain()}]
    with (output / "controller-events.jsonl").open("x") as stream:
        def event(name):
            row = {"event": name, "at": time.monotonic(), "clock_domain": local_clock_domain()}
            events.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

        stream.write(json.dumps(events[0], sort_keys=True) + "\n")
        stream.flush()
        worker = {"returncode": None}
        stages = {}
        try:
            event("native_transport_before_prelaunch")
            worker_budget = allowed - 3 * grace - 2
            worker = await asyncio.wait_for(
                runner(commands["worker"], worker_budget, output / "worker.txt"), worker_budget,
            )
        except (OSError, ValueError, TimeoutError) as exc:
            worker["error"] = type(exc).__name__
        finally:
            for stage in ("retrieve", "stop", "observe"):
                event(stage + "_started")
                if stage == "stop":
                    stop_started_utc = time.time()
                try:
                    if stage == "observe":
                        receipt = await asyncio.wait_for(observer(config), grace)
                        with (output / "observe.txt").open("x") as readback_stream:
                            readback_stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
                        stages[stage] = {"returncode": 0}
                    else:
                        stages[stage] = await asyncio.wait_for(
                            runner(commands[stage], grace, output / (stage + ".txt")), grace,
                        )
                except (OSError, ValueError, TimeoutError) as exc:
                    stages[stage] = {"returncode": None, "error": type(exc).__name__}
                event(stage + "_finished")
    finished_utc, finished = time.time(), time.monotonic()
    readback = output / "observe.txt"
    observation = external_stop_observation(
        readback.read_bytes() if readback.exists() and readback.stat().st_size <= 65536 else b"",
        config, started_utc=stop_started_utc, finished_utc=finished_utc,
    )
    clock_join = False
    try:
        owner_path = Path(config["native_owner_receipt"])
        owner = json.loads((output / "native-artifacts" / owner_path.parent.name /
                           owner_path.name).read_bytes())
        native = config["native_execution"]
        clock_join = (
            owner.get("schema") == "r21-native-worker-owner-v1"
            and owner.get("run_id") == config["run_id"]
            and owner.get("pod_id") == config["existing_pod_id"]
            and all(owner.get(key) == native[key] for key in ("hostname", "clock_domain", "python", "root"))
            and type(owner.get("pid")) is int and owner["pid"] > 0
            and type(owner.get("session_id")) is int
            and owner.get("session_id") == owner["pid"]
            and finite(owner.get("native_started_at")) and owner["native_started_at"] >= 0
            and owner.get("native_config_sha256") == config["native_config_sha256"])
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    retrieval = stages["retrieve"].get("returncode") == 0 and clock_join
    ok = (worker.get("returncode") == 0 and retrieval and observation["accepted_scope"]
          and all(stages[key].get("returncode") == 0 for key in ("stop", "observe"))
          and finished - started <= allowed)
    result.update(
        status="completed_compute_only" if ok and not observation["provider_billing_stopped"]
        else "completed_controller_mechanics" if ok else "incomplete",
        worker=worker, stages=stages, events=events, provider=observation,
        evidence_retrieved=retrieval, distinct_clock_join=clock_join,
        remote_stop_proved=observation["compute_stopped"],
        started_utc_s=started_utc, finished_utc_s=finished_utc,
        estimated_campaign_charge_usd=config["provider_hourly_usd"] * (finished - started) / 3600,
        continuing_storage_after_campaign="unbounded future cost; not covered by campaign ceiling",
        hard_spend_containment_proved=False, native=False, mpc_permit=False,
    )
    with (output / "controller-receipt.json").open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return result


async def native_worker(config_path, protocol_path, config, protocol, config_sha):
    if os.getsid(0) != os.getpid():
        os.setsid()
    owner_path = Path(config["worker_owner_receipt"]).resolve()
    if not owner_path.is_relative_to(ROOT / "results"):
        raise ValueError("native_owner_receipt_scope")
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"schema": "r21-native-worker-owner-v1", "pid": os.getpid(),
             "session_id": os.getsid(0), "native_config_sha256": config_sha,
             "run_id": config["identification_permit"]["run_id"],
             "pod_id": config["existing_pod_id"], "hostname": platform.node(),
             "clock_domain": local_clock_domain(), "python": str(Path(sys.executable).resolve()),
             "root": str(ROOT), "native_started_at": time.monotonic(), "native": False}
    with owner_path.open("x") as stream:
        stream.write(json.dumps(owner, sort_keys=True) + "\n")
    checked = preflight(config, protocol, measurement=True, worker_phase=True)
    checked["reasons"].extend(native_entry_reasons(config, protocol, worker_phase=True))
    with owner_path.with_suffix(".preflight.json").open("x") as stream:
        stream.write(json.dumps(checked, sort_keys=True, allow_nan=False) + "\n")
    if checked["reasons"]:
        return {**checked, "status": "blocked"}
    command = await collection_worker(config_path, protocol_path, config, protocol, external=True)
    child = await asyncio.create_subprocess_exec(
        *command, env={**os.environ, "MPC_OWNER_LEDGER_ENABLED": "true"},
    )
    with owner_path.with_suffix(".collector.json").open("x") as stream:
        stream.write(json.dumps({"pid": child.pid, "session_id": os.getsid(0),
                                 "run_id": owner["run_id"]}, sort_keys=True) + "\n")
    code = await child.wait()
    if code:
        return {"status": "native_worker_failed", "returncode": code, "native": False}
    labels = postcollection(config, protocol_path)
    with owner_path.with_suffix(".postcollection.json").open("x") as stream:
        stream.write(json.dumps(labels, sort_keys=True, allow_nan=False) + "\n")
    return {"status": "native_worker_collected", "returncode": code,
            "postcollection": labels, "native": False, "provider_stop": "external_only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(
        "dry-run", "preflight", "measure", "worker", "manifest", "bundle", "postcollection",
        "controller", "native-worker", "controller-template",
        "provider-stop",
    ),
                        default="dry-run")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--controller-bound-sha256")
    parser.add_argument("--pod-id")
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-release", action="store_true",
                        help="explicit separate measurement authority; never an MPC permit")
    args = parser.parse_args(argv)
    if args.mode == "controller-template":
        print(json.dumps(external_controller_template(), sort_keys=True, allow_nan=False))
        return 0
    if args.mode == "manifest":
        print(json.dumps(source_manifest(), sort_keys=True, allow_nan=False))
        return 0
    if args.mode == "bundle":
        if args.bundle_output is None:
            parser.error("--bundle-output is required in bundle mode")
        print(json.dumps(source_bundle(args.bundle_output), sort_keys=True, allow_nan=False))
        return 0
    if args.mode == "provider-stop":
        if not args.execute or args.config is None:
            print(json.dumps({"status": "blocked", "reason": "explicit_stop_execute_config_required"}))
            return 2
        path = args.config.resolve()
        if not path.is_relative_to(ROOT / "results"):
            print(json.dumps({"status": "blocked", "reason": "owned_controller_stop_config_required"}))
            return 2
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.controller_bound_sha256:
            print(json.dumps({"status": "blocked", "reason": "controller_stop_config_hash"}))
            return 2
        result = asyncio.run(provider_stop_adapter(
            json.loads(raw), pod_id=args.pod_id, run_id=args.run_id, execute=args.execute,
        ))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 2 if result["status"] == "blocked" else 0 if result["status"] == "stop_acknowledged" else 1
    if args.mode == "controller" and not args.operator_release:
        print(json.dumps({"status": "blocked", "reason": "explicit_operator_release_required"}))
        return 2
    if args.config is None or args.protocol is None:
        parser.error("--config and --protocol are required outside manifest mode")
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    protocol = json.loads(args.protocol.read_bytes())
    if args.mode == "controller":
        result = asyncio.run(external_controller(
            config, hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        ))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 2 if result["status"] == "blocked" else 1 if result["status"] == "incomplete" else 0
    if args.mode == "native-worker":
        config_sha = hashlib.sha256(config_raw).hexdigest()
        if (not args.operator_release or args.expected_config_sha256 != config_sha
                or hashlib.sha256(args.protocol.read_bytes()).hexdigest() != H16_PROTOCOL_SHA):
            result = {"status": "blocked", "reasons": ["native_worker_explicit_source_authority"]}
        else:
            result = asyncio.run(native_worker(args.config, args.protocol, config, protocol, config_sha))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 2 if result["status"] == "blocked" else result.get("returncode", 1)
    if args.mode == "postcollection":
        result = postcollection(config, args.protocol)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "retrospective_labels" else 1
    result = preflight(config, protocol, measurement=args.mode in ("measure", "worker"))
    if args.mode in ("measure", "worker"):
        result["reasons"].append("external_controller_required_no_pod_self_stop")
        if hashlib.sha256(args.protocol.read_bytes()).hexdigest() != H16_PROTOCOL_SHA:
            result["reasons"].append("original_ab883_protocol_required")
        result["reasons"].extend(native_entry_reasons(config, protocol))
        if result["reasons"]:
            result["status"] = "blocked"
        if not args.operator_release:
            result["status"] = "blocked"
            result["reasons"].append("explicit_operator_release_switch_required")
        if result["status"] != "blocked":
            measurement_command(args.config, args.protocol, config, watchdog_open=args.mode == "worker")
            if args.mode == "worker":
                command = asyncio.run(collection_worker(args.config, args.protocol, config, protocol))
                os.execv(command[0], command)
            command = [
                sys.executable, "-m", "mpc_local.native_pilot", "--mode", "worker",
                "--operator-release", "--config", str(args.config.resolve()),
                "--protocol", str(args.protocol.resolve()),
            ]
            result["watchdog"] = asyncio.run(bounded_delegate(
                command, stop_argv=config["stop_argv"], active_s=config["active_ceiling_s"],
                wall_s=config["wall_ceiling_s"], spend_usd=config["spend_ceiling_usd"],
                rate=config["provider_hourly_usd"],
                expires_at=min(config["operator_release"]["expires_at"],
                               config["identification_permit"]["expires_at"]),
                log_path=config["watchdog_log"],
                provider_receipt_path=config["post_stop_receipt_path"],
                pod_id=config["existing_pod_id"],
                run_id=config["identification_permit"]["run_id"],
                env={**os.environ, "MPC_OWNER_LEDGER_ENABLED": "true",
                     "MPC_H16_WATCHDOG_PARENT_PID": str(os.getpid()),
                     "PYTHONPATH": os.pathsep.join((
                    str(ROOT), str(ROOT / "experiments/mpc-research"),
                    str(ROOT / "experiments/mpc-research/native_observer"),
                ))},
            ))
            result["native_delegate_started"] = (
                result["watchdog"]["delegate_returncode"] is not None
            )
            result["native_dispatched"] = (
                None if result["native_delegate_started"] else False
            )
            if (result["watchdog"]["status"] != "completed"
                    or not result["watchdog"]["provider_billing_stopped"]):
                result["status"] = "failed"
            if result["watchdog"]["delegate_returncode"] == 0:
                try:
                    result["postcollection"] = postcollection(config, args.protocol)
                except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
                    result["postcollection"] = {"status": "nonqualification", "error": str(exc)}
                if result["postcollection"]["status"] != "retrospective_labels":
                    result["status"] = "failed"
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 2 if result["status"] == "blocked" else 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
