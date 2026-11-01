import pytest

from mpc_lab.contracts import CostSpec, finite


def test_unrepresentable_numeric_input_is_rejected_without_overflowing_validator():
    assert not finite(10**1000)
    with pytest.raises(ValueError, match="nonfinite_request_cost"):
        CostSpec().estimate(10**1000, 128)


def test_boolean_cost_and_token_marks_cannot_silently_change_reservation_semantics():
    with pytest.raises(ValueError, match="invalid_cost_spec"):
        CostSpec(input_per_token=True)
    with pytest.raises(ValueError, match="invalid_prompt_tokens"):
        CostSpec().estimate(True, 128)
