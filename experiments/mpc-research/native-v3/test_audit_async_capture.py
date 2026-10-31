from copy import deepcopy

import pytest

from audit_async_capture import audit_async_capture


def capture():
    steps, workers = [], []
    for i, (position, count) in enumerate(((0, 2), (2, 1), (3, 1))):
        key = f"step-{i}"
        steps.append(dict(step_id=key, start=i / 2, end=i + 1,
            parity={"matches": True}, work=[dict(request_id="r", position=position, tokens=count)],
            state_before={"requests": {"r": dict(computed=position, num_tokens=position,
                placeholders=1 if i == 2 else 0)}}))
        workers.append(dict(step_id=key, counts={"r": count}, executed_counts={"r": count},
            actual_tokens=count, executed_positions={"r": position},
            gpu_forward_s=.01, host_start=i / 2 + .01))
    return dict(status="complete", errors=[], overflow=0, steps=steps, workers=[],
        requests={"r": dict(input_tokens=2, output_tokens=2, max_tokens=8,
                           finished=2, finish_reason="stop")}), workers


def test_real_async_shape_keeps_failed_v2_audit_and_all_executed_work():
    document, workers = capture()
    original = deepcopy(document)
    result = audit_async_capture(document, workers)
    assert result["valid"]
    assert not result["strict_v2_audit"]["valid"]
    assert result["request_work"]["r"]["terminal_overhang_tokens"] == 1
    assert result["request_work"]["r"]["placeholder_worker_step_ids"] == ["step-2"]
    assert document == original


@pytest.mark.parametrize("fault", ["placeholder", "post_terminal", "worker_position",
                                  "extra_two", "zero_output", "nonfinite_terminal"])
def test_overhang_is_not_a_generic_progress_tolerance(fault):
    document, workers = capture()
    if fault == "placeholder":
        document["steps"][-1]["state_before"]["requests"]["r"]["placeholders"] = 0
    elif fault == "post_terminal":
        document["requests"]["r"]["finished"] = .9
    elif fault == "worker_position":
        workers[-1]["executed_positions"]["r"] = 99
    elif fault == "extra_two":
        document["steps"][-1]["work"][0]["tokens"] = 2
        workers[-1].update(counts={"r": 2}, executed_counts={"r": 2}, actual_tokens=2)
    elif fault == "zero_output":
        document["requests"]["r"]["output_tokens"] = 0
    else:
        document["requests"]["r"]["finished"] = float("nan")
    assert not audit_async_capture(document, workers)["valid"]
