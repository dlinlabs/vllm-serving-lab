"""Finite hidden-EOS witnesses, not a simulator or a native performance model.

The fixture grants an oracle the TOTAL remaining decode work. Even that stronger
projection cannot determine whole-request reservation release. Unit token rate,
zero transport delay and arbitrary cost units are analytic assumptions only.
"""

import argparse
from dataclasses import dataclass
import json
import math


def _number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError("expected finite nonnegative number")
    return value


@dataclass(frozen=True)
class DecodeWorld:
    costs: tuple[float, ...]
    output_so_far: tuple[int, ...]
    output_caps: tuple[int, ...]
    ages_s: tuple[float, ...]
    remaining_tokens: tuple[int, ...]

    def __post_init__(self):
        fields = (self.costs, self.output_so_far, self.output_caps,
                  self.ages_s, self.remaining_tokens)
        if any(type(x) is not tuple for x in fields):
            raise ValueError("immutable tuples required")
        if not self.costs or any(len(x) != len(self.costs) for x in fields):
            raise ValueError("nonempty aligned request fields required")
        for cost, output, cap, age, remaining in zip(*fields):
            _number(cost, positive=True)
            _number(age)
            if (any(type(x) is not int for x in (output, cap, remaining))
                    or output < 1 or remaining < 1 or output + remaining > cap):
                raise ValueError("invalid active decode work or output cap")

    @property
    def projection(self):
        """Same per-request observations, plus otherwise unobservable total work."""
        return (self.costs, self.output_so_far, self.output_caps,
                self.ages_s, sum(self.remaining_tokens))

    def reserved_after(self, horizon_s, token_rate):
        _number(horizon_s)
        _number(token_rate, positive=True)
        return math.fsum(cost for cost, remaining in
                         zip(self.costs, self.remaining_tokens)
                         if remaining / token_rate > horizon_s)


def release_certificate(worlds, *, horizon_s, token_rate, offer_cost, budgets):
    """Exact finite-witness minimax point-error bound for one fixed projection.

    Releases at the horizon occur before the hypothetical offer. No probabilities,
    action-optimality, history-equivalence or physical vLLM feasibility are inferred.
    """
    worlds, budgets = tuple(worlds), tuple(budgets)
    if len(worlds) < 2:
        raise ValueError("at least two possible worlds required")
    if any(w.projection != worlds[0].projection for w in worlds[1:]):
        raise ValueError("worlds do not share the declared projection")
    _number(offer_cost, positive=True)
    if not budgets:
        raise ValueError("at least one budget required")
    for budget in budgets:
        _number(budget, positive=True)
    if len(set(budgets)) != len(budgets):
        raise ValueError("duplicate budgets")
    reserved = [w.reserved_after(horizon_s, token_rate) for w in worlds]
    low, high = min(reserved), max(reserved)
    return {
        "scope": "analytic finite-witness obstruction, not native calibration",
        "same_projection": True,
        "total_remaining_tokens": worlds[0].projection[-1],
        "horizon_s": horizon_s,
        "token_rate_per_request_s": token_rate,
        "reserved_cost_by_world": reserved,
        "minimax_absolute_point_error_cost": (high - low) / 2,
        "minimax_point_prediction_cost": low + (high - low) / 2,
        "offer_cost": offer_cost,
        "offer_accepted_by_budget": [
            {"budget": budget, "accepted_by_world": [
                r + offer_cost <= budget for r in reserved]}
            for budget in budgets
        ],
    }


def available_at_decision(*, sampled_at, received_at, decision_at,
                          source_clock, decision_clock):
    """Source-time causality alone does not establish online input availability."""
    for value in (sampled_at, received_at, decision_at):
        _number(value)
    if not source_clock or source_clock != decision_clock:
        raise ValueError("clock-domain mismatch")
    if received_at < sampled_at:
        raise ValueError("receipt precedes source sample")
    return received_at <= decision_at


def diagnostic(horizon_s=2):
    common = dict(costs=(8., 8.), output_so_far=(1, 1),
                  output_caps=(16, 16), ages_s=(1., 1.))
    a = DecodeWorld(**common, remaining_tokens=(1, 9))
    b = DecodeWorld(**common, remaining_tokens=(5, 5))
    return {
        "schema": "sol-hidden-state-counterexamples-v1",
        "native_data_used": False,
        "native_performance_claim": False,
        "assumptions": [
            "two parallel decoders, each one token/second irrespective of membership",
            "no arrivals before the hypothetical offer; zero cleanup delay",
            "whole cost released at EOS, before an offer at an equal timestamp",
            "remaining lengths are alternative hidden futures, not online features",
            "costs are abstract reservation units, not native fitted coefficients",
        ],
        "hidden_remaining_tokens": [a.remaining_tokens, b.remaining_tokens],
        "release_alias": release_certificate(
            (a, b), horizon_s=horizon_s, token_rate=1,
            offer_cost=4, budgets=(16, 32)),
        "availability_counterexample": {
            "sampled_at": 1.0, "decision_at": 1.1, "received_at": 1.2,
            "source_time_only_would_pass": True,
            "qualified_online_input": available_at_decision(
                sampled_at=1., received_at=1.2, decision_at=1.1,
                source_clock="same-boot", decision_clock="same-boot"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-s", type=float, default=2)
    args = parser.parse_args()
    try:
        result = diagnostic(args.horizon_s)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
