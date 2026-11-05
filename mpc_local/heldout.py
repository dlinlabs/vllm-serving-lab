"""Consume the frozen CPU test set before exposure, even if the run then fails."""

import json
import os
from pathlib import Path
import subprocess
import time

from .seal import file_hash, verify_candidate


def reserve(root, candidate_receipt, reviews, seeds):
    root = Path(root).resolve()
    candidate = verify_candidate(root, candidate_receipt)
    expected = list(range(2103001, 2103013))
    if candidate["heldout_seeds"] != expected or list(seeds) != expected:
        raise ValueError("complete_frozen_heldout_required")
    if set(reviews) != {"opus", "astra", "sol", "gemini"}:
        raise ValueError("four_final_reviews_required")
    if file_hash(root / "mpc_local/heldout.py") != file_hash(__file__):
        raise ValueError("heldout_code_context_mismatch")
    candidate_hash = file_hash(candidate_receipt)
    review_hashes, models = {}, set()
    for reviewer, path in reviews.items():
        path = Path(path).resolve()
        relative = path.relative_to(root)
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(relative)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "diff", "--name-only", "HEAD", "--", str(relative)],
            text=True,
        ).strip()
        review = json.loads(path.read_text())
        if (
            dirty or review.get("schema") != "r21-cpu-final-review-v1"
            or review.get("reviewer") != reviewer
            or review.get("candidate_receipt_sha256") != candidate_hash
            or review.get("source_sha256") != candidate["source_sha256"]
            or review.get("heldout_seen") is not False
            or review.get("native_mpc_permitted") is not False
            or review.get("blocking_issues") != []
            or not isinstance(review.get("model"), str) or not review["model"]
        ):
            raise ValueError("unresolved_or_misbound_final_review")
        models.add(review["model"])
        review_hashes[str(relative)] = file_hash(path)
    if len(models) != 4:
        raise ValueError("four_distinct_review_models_required")
    directory = root / "results/mpc-local-r21"
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root):
        raise ValueError("heldout_ledger_outside_owned_root")
    ledger = directory / "formal-heldout-consumption.json"
    common = Path(subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        text=True,
    ).strip()).resolve()
    shared_marker = common / "r21-cpu-heldout-consumption.json"
    record = {
        "schema": "r21-cpu-heldout-consumption-v1",
        "status": "consumed_before_execution",
        "candidate": candidate["candidate"],
        "candidate_receipt_sha256": candidate_hash,
        "source_sha256": candidate["source_sha256"],
        "review_files": review_hashes,
        "seeds": expected,
        "reserved_at_utc_s": time.time(),
        "failure_or_partial_run_still_consumes": True,
        "reviewer_identity_is_declared_not_authenticated": True,
        "native_mpc_permitted": False,
    }
    # The shared marker fences every worktree; partial writes also consume the set.
    for path in (shared_marker, ledger):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise ValueError("heldout_already_consumed") from error
        with os.fdopen(descriptor, "w") as handle:
            json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return record
