from mpc_lab.benchmark import make_trace, outcome_metrics
from mpc_lab.contracts import Outcome


def test_goodput_keeps_rejection_late_failure_and_unresolved_in_offered_denominator():
    rows = (
        Outcome("good", 0, True, 0.2, 1, True, 1, 10, 10),
        Outcome("late-fail", 0, True, 0.2, 1, False, 1, 10, 10),
        Outcome("rejected", 0, False, None, None, False, 1, 10, 10),
        Outcome("unresolved", 0, True, 0.2, None, None, 1, 10, 10),
    )
    result = outcome_metrics(rows, 2)
    assert result["good"] == 1
    assert result["successful_goodput_rps"] == 0.5
    assert result["all_offered_slo_nonattainment"] == 0.75
    assert result["rejection_rate"] == result["failure_rate"] == 0.25
    assert result["unresolved"] == 1


def test_synthetic_trace_is_reproducible_and_eos_hidden_from_request_marks():
    trace = make_trace(101, "shift", 12)
    assert trace == make_trace(101, "shift", 12)
    assert trace != make_trace(102, "shift", 12)
    assert all(1 <= row.output_tokens <= row.max_tokens for row in trace)
    assert len({row.output_tokens for row in trace}) > 1
    assert all(a.scheduled_at < b.scheduled_at for a, b in zip(trace, trace[1:]))
