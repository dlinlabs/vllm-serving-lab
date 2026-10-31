"""Opus Round3 revision: focused CPU checks only (stdlib, TRAIN artifacts only).

1. nested_knee_loco      -- Round1 chose the knee on all batches before LOCO (peer
                            Astra O2 / Sol O3). Re-select the knee inside each fold.
2. step_count_lower_bound -- TRAIN lower bound on engine output-commit steps per
                            episode versus the bridge's 8192-row default Buffer.
3. bridge witnesses       -- executed against the exact Mayor bridge source
                            (git object 1cbe251, sha256-pinned, loaded in memory):
                            unexpected request accepted, metadata during an active
                            trial, non-idempotent ACK after a lost response.

Nothing here is a GPU observation, an overhead measurement, or a calibration.
Held-out seeds 160021-160024 are refused by the Round1 loader.
"""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round1"))
sys.path.insert(0, str(HERE.parent / "round2"))
import opus_batch_models as r1  # noqa: E402  frozen Round1, read-only reuse
import opus_counterexamples as r2  # noqa: E402  frozen Round2, read-only reuse

ROOT = HERE.parents[3]
BRIDGE_COMMIT = "1cbe251"
BRIDGE_PATH = "experiments/mpc-research/native-v2/batch_capture_bridge.py"
BRIDGE_SHA256 = "0e24eb2c00d3b3da08f01a0d235e4a397551eee17f6853f5fe23dca295498f4a"
BUFFER_CAPACITY = 8192  # prefill_lab/contracts.py Buffer default (core steps and worker rows/pending)


# ------------------------------------------------------- 1. nested knee LOCO

def nested_knee_loco(batches):
    """Knee chosen by r1.knee_profile on the training cells only, per held-out cell."""
    cells = defaultdict(list)
    for b in batches:
        cells[r1.condition_key(b)].append(b)
    err, knees = [], []
    for key, held in sorted(cells.items()):
        train = [b for k, g in cells.items() if k != key for b in g]
        knee = r1.knee_profile(train)["best_knee"]
        law = r1.fit_law(train, knee)
        knees.append(knee)
        err += [abs(r1.predict_ms(law, b) - b["kernel_busy_union_us"] / 1000) for b in held]
    leaked_knee = r1.knee_profile(batches)["best_knee"]
    leaked = r1.leave_one_condition_out(batches, leaked_knee)["hinge_batch_law"]
    loglinear = r2.loglinear_vs_hinge_loco(batches)["loglinear"]
    return dict(nested=dict(mae_ms=statistics.mean(err), max_abs_ms=max(err), batches=len(err),
                            fold_knees=sorted(set(knees))),
                round1_leaked_knee=dict(knee=leaked_knee, mae_ms=leaked["mae_ms"], max_abs_ms=leaked["max_abs_ms"]),
                loglinear_loco=loglinear,
                scope="kernel_busy_union target from max8 profile; fitted mean, not a certified lower envelope")


# ------------------------------------------- 2. step-count lower bound vs 8192

def step_count_lower_bound(episodes):
    """Without speculation a request gains at most one output token per engine step,
    so between two complete snapshots at least max_r Delta(output_tokens) steps occur.
    Summed over consecutive complete snapshots this lower-bounds output-commit steps;
    prefill-only steps, pre-first-frame, post-last-frame and drain steps are excluded."""
    rows = []
    for e in episodes:
        frames = [f for f in e["native_frames"] if f.get("complete") and f.get("requests") is not None]
        steps = 0
        for a, b in zip(frames, frames[1:]):
            ra = {r["request_id"]: r["output_tokens"] for r in a["requests"]}
            gains = [r["output_tokens"] - ra[r["request_id"]] for r in b["requests"] if r["request_id"] in ra]
            steps += max([g for g in gains if g > 0], default=0)
        rows.append(dict(episode_id=e["episode_id"], cell=e["cell"], arm=e["arm"],
                         seed_block=e["seed_block"], step_lower_bound=steps,
                         fraction_of_buffer=steps / BUFFER_CAPACITY,
                         observed_span_s=frames[-1]["sampled_at"] - frames[0]["sampled_at"] if frames else 0.0))
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r["step_lower_bound"])
    return dict(buffer_capacity=BUFFER_CAPACITY, episodes=rows,
                by_cell={c: dict(min=min(v), max=max(v), max_fraction=max(v) / BUFFER_CAPACITY)
                         for c, v in sorted(by_cell.items())},
                assumption="no speculative decoding (bridge features() flags speculation fail-closed)")


def max_safe_tape_s(bound, margin=0.5):
    """Tape duration at which the worst observed step rate fills `margin` of the buffer."""
    rates = [r["step_lower_bound"] / r["observed_span_s"] for r in bound["episodes"] if r["observed_span_s"] > 0]
    return margin * BUFFER_CAPACITY / max(rates)


# --------------------------------------------------- 3. bridge source witnesses

def load_bridge():
    source = subprocess.check_output(["git", "show", f"{BRIDGE_COMMIT}:{BRIDGE_PATH}"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != BRIDGE_SHA256:
        raise ValueError("bridge source differs from the reviewed object")
    module = types.ModuleType("reviewed_batch_capture_bridge")
    module.__file__ = "<git:%s:%s>" % (BRIDGE_COMMIT, BRIDGE_PATH)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.source_hash = lambda: BRIDGE_SHA256  # on-disk re-read is replaced by the pinned object hash
    return module


def _plan():
    return dict(schema="native-v2-batch-capture-plan-v1", partition="train", source_commit="a" * 40,
                trials=[dict(trial_id="nv2-r3-light", observer="light", seed=170001),
                        dict(trial_id="nv2-r3-off", observer="off", seed=170002)])


def _capture(requests):
    return dict(schema="prefill-capture-v1", trial_id="nv2-r3-light", status="complete", errors=[],
                overflow=0, pending_step_ids=[], requests=requests, steps=[{"step_id": "s0"}],
                workers=[dict(pending=0, overflow=0, rows=[dict(step_id="s0", gpu_forward_s=.012)])])


def _harness(bridge, reply):
    calls = []
    clocks = [dict(clock_domain="boot:ns", process_id=7, bridge_sha256=BRIDGE_SHA256)]
    core = types.SimpleNamespace(model_executor=types.SimpleNamespace(collective_rpc=lambda name: clocks))

    def command(_core, payload):
        calls.append(payload["operation"])
        if payload["operation"] == "meta":
            return dict(tensor_parallel_size=1, pipeline_parallel_size=1)
        if payload["operation"] == "external_begin":
            return {"external": True}
        return copy.deepcopy(reply)
    return bridge.BatchCapture(core, _plan(), command, clock=lambda: 1.0, domain="boot:ns"), calls


NATIVE_ID = re.compile(r"chatcmpl-([0-9a-f]{32})-[0-9a-f]{8}")  # native_observer/native_prediction.py:27


def _offer_identity(engine_id, row):
    if row.get("trace_id") is not None:
        return row["trace_id"]
    match = NATIVE_ID.fullmatch(engine_id)
    return match.group(1) if match else None


def reconcile_requests(capture, expected_trace_ids):
    """Proposed minimal fix: exact set equality between the frozen accepted-offer
    manifest and the engine's captured identities (trace_id, else the Chat
    x-request-id join chatcmpl-<offer>-<suffix>)."""
    seen = [_offer_identity(k, r) for k, r in capture.get("requests", {}).items()]
    errors = []
    if None in seen:
        errors.append("request_without_offer_identity")
    if len(seen) != len(set(seen)) or set(seen) - {None} != set(expected_trace_ids):
        errors.append("offered_engine_request_set_mismatch")
    return errors


def bridge_witnesses(bridge=None):
    bridge = bridge or load_bridge()
    out = {}
    # (a) a stray Chat request with no trace id (health probe, retry, other client) is accepted.
    stray = _capture({"engine-a": dict(finished=3.0, trace_id="t" * 32),
                      "engine-stray": dict(finished=3.5)})
    out["unexpected_request"] = dict(
        bridge_errors=bridge.capture_errors(stray, "light"),
        reconciliation_errors=reconcile_requests(stray, ["t" * 32]))
    # (b) metadata (38-file hashing + worker dispatch-table meta) is executed mid-trial.
    capture, calls = _harness(bridge, _capture({"engine-a": dict(finished=3.0)}))
    capture.invoke({"operation": "metadata"})
    capture.invoke({"operation": "begin", "trial_id": "nv2-r3-light"})
    before = len(calls)
    try:
        capture.invoke({"operation": "metadata"})
        accepted = True
    except ValueError:
        accepted = False
    out["metadata_during_active_trial"] = dict(accepted=accepted, active=capture.active,
                                               engine_meta_calls_during_trial=calls[before:].count("meta"))
    # (c) the ACK response is lost; the collector retries the identical ACK.
    result = capture.invoke({"operation": "end"})
    ack = dict(operation="ack", trial_id=result["trial_id"], capture_sha256=result["capture_sha256"])
    capture.invoke(ack)
    try:
        capture.invoke(ack)
        retry = "acknowledged"
    except ValueError as error:
        retry = "ValueError: " + str(error)
    out["ack_retry_after_lost_response"] = dict(
        retry=retry, status_unacknowledged=capture.invoke({"operation": "status"})["unacknowledged_trial"])
    return out


# ------------------------------------------------------------------- driver

def run(profile_path, archive_path):
    batches = json.loads(Path(profile_path).read_text())["batches"]
    bound = step_count_lower_bound(r1.load_train_archive(archive_path))
    bound_summary = {k: v for k, v in bound.items() if k != "episodes"}
    return dict(schema="opus-round3-revision-v1", calibration_accepted=False, performance_accepted=False,
                gpu_observation=False, nested_knee_loco=nested_knee_loco(batches),
                step_count_lower_bound=bound_summary,
                max_tape_s_at_half_buffer=max_safe_tape_s(bound),
                bridge_sha256=BRIDGE_SHA256, bridge_witnesses=bridge_witnesses())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    R = "results/mpc-native-4090-scout"
    p.add_argument("--profile", default=f"{R}/native-profile-diagnostic.json")
    p.add_argument("--train-archive", default=f"{R}/native-identification-train-evidence.tar.gz")
    a = p.parse_args()
    print(json.dumps(run(a.profile, a.train_archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
