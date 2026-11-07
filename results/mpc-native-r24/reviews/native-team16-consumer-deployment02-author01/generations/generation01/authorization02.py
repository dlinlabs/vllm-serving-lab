"""Read-only preactivation gate; author never creates accepted approval receipts."""

import json
from pathlib import Path
import re
import stat

import composition_common01 as c

SCHEMA = "mayor-cut-consumer-deployment02-preregistration-v1"
SCOPE = "one_model_free_ordinal2_consumer_component_only"
ACCEPTANCES = ("successor_Mayor_engineering", "successor_independent_delta",
               "deployment_Mayor_review", "observational_policy_Mayor_review")
FIELDS = {"schema","scope","execute_once","deployment_manifest_sha256","protocol_sha256",
          "native_plan_sha256","offers_sha256","successor_manifest_sha256","capture_id",
          "operation_id","gpu_uuid","source_root","native_python","preregistration_commit",
          "existing_Mayor_exclusive_GPU_lock_held","fresh_host_receipt","fresh_host_receipt_sha256",
          "acceptances","actual_actuation_authorized","model_qualification_authorized"}


def regular(path, cap):
    path = Path(path)
    info = path.lstat()
    c.require(not path.is_symlink() and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and info.st_size <= cap, "regular_bounded_approval_or_receipt")
    return path


def strict(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            c.require(key not in value, "duplicate_approval_key")
            value[key] = item
        return value
    def constant(value):
        raise ValueError("nonfinite_approval")
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    json.dumps(result, allow_nan=False)
    return result


def require_activation(enabled, path, *, role, expected_sha256=None):
    c.require(type(enabled) is bool and role in ("launch","client","service"), "explicit_activation_role")
    if not enabled:
        return None
    c.require(path is not None, "separate_Mayor_acceptance_and_preregistration_required")
    path = regular(path, 65536)
    raw = path.read_bytes()
    digest = c.sha_bytes(raw)
    c.require(expected_sha256 is None or digest == expected_sha256, "owned_approval_bytes_changed")
    value = strict(raw)
    c.require(type(value) is dict and set(value) == FIELDS and value.get("schema") == SCHEMA
              and value.get("scope") == SCOPE and value.get("execute_once") is True
              and value.get("actual_actuation_authorized") is False
              and value.get("model_qualification_authorized") is False,
              "new_Mayor_approval_required_not_author_template")
    c.require(value["successor_manifest_sha256"] == c.CONSUMER_SEAL
              and value["capture_id"] == "nv2-team16-cut-consumer02-fixed32"
              and value["operation_id"] == "opteam16-cut-consumer02"
              and value["gpu_uuid"] == "GPU-8610519d-3680-ae5f-cdfc-40037bfb5013"
              and value["source_root"] == str(c.ROOT) and value["native_python"] == str(c.PYTHON)
              and value["existing_Mayor_exclusive_GPU_lock_held"] is True
              and isinstance(value["preregistration_commit"], str)
              and re.fullmatch("[a-f0-9]{40}", value["preregistration_commit"]),
              "exact_deployment02_authority_scope")
    fingerprints = {key:c.sha(c.HERE / name) for key, name in (
        ("deployment_manifest_sha256","SHA256SUMS"), ("protocol_sha256","PROTOCOL.json"),
        ("native_plan_sha256","NATIVE_PLAN.json"), ("offers_sha256","OFFERS.json"))}
    c.require(all(value[key] == fingerprint for key, fingerprint in fingerprints.items()),
              "exact_frozen_deployment02_authority")
    receipts = value["acceptances"]
    c.require(type(receipts) is dict and set(receipts) == set(ACCEPTANCES),
              "separate_dependency_delta_deployment_policy_receipts_required")
    for kind in ACCEPTANCES:
        proof = receipts[kind]
        subject = (c.CONSUMER_SEAL if kind.startswith("successor_") else
                   value["protocol_sha256"] if kind.startswith("observational_") else
                   value["deployment_manifest_sha256"])
        c.require(type(proof) is dict and set(proof) == {"accepted","path","sha256","subject_sha256"}
                  and proof["accepted"] is True and proof["subject_sha256"] == subject
                  and isinstance(proof["sha256"], str)
                  and re.fullmatch("[a-f0-9]{64}", proof["sha256"]),
                  "separate_acceptance_pending:" + kind)
        c.require(c.sha(regular(proof["path"], 16 * 1024 * 1024)) == proof["sha256"],
                  "separate_acceptance_bytes_changed:" + kind)
    c.require(c.sha(regular(value["fresh_host_receipt"], 16 * 1024 * 1024))
              == value["fresh_host_receipt_sha256"], "fresh_host_receipt_bytes_changed")
    return dict(document=value, approval_path=str(path.resolve()), approval_sha256=digest, role=role)
