"""Shadow-only continuation of the sealed planner15 complete-panel evaluator."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from fractions import Fraction
import hashlib
from itertools import islice, product
import json
import math
import sys
import time

from policy_planner15 import (
    NativePanelAcquisition, NativeResponsePlanner, NativeScenario, OwnerSourceAudit,
    PlanningEntry, PolicyPlanner15,
)
from historical_owner_scenario import HistoricalOwnerScenario
from sequential_publication_world import SequentialPublicationWorld


GUARD_SCHEMA = "r24-team16-common-panel-guards-v1"
OBJECTIVE = "all-offered-natural-good_then-http-wall_then-transitions_sum_lex-v1"
RANKED_FIELDS = (
    "rankings", "best_multistep", "best_fixed", "best_one_move", "selected_plan",
)


def _finite(value, *, positive=False):
    try:
        valid = (
            type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("policy16_finite_nonnegative_value_required")
    return value


def _canonical(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "fields": {field.name: _canonical(getattr(value, field.name))
                       for field in fields(value)},
        }
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("policy16_snapshot_string_mapping_keys_required")
        return {"dict": {key: _canonical(item) for key, item in value.items()}}
    if type(value) in (tuple, list):
        return {type(value).__name__: [_canonical(item) for item in value]}
    if type(value) in (set, frozenset):
        items = [_canonical(item) for item in value]
        return {type(value).__name__: sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, allow_nan=False, separators=(",", ":")))}
    if type(value) is SequentialPublicationWorld:
        return {"SequentialPublicationWorld": _canonical(vars(value))}
    raise ValueError("policy16_unsupported_snapshot_type:" + type(value).__name__)


def _fingerprint(value):
    raw = json.dumps(
        _canonical(value), sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _bounded(items, name):
    try:
        rows = tuple(islice(iter(items), 17))
    except TypeError as exc:
        raise ValueError("policy16_explicit_" + name + "_required") from exc
    if not 1 <= len(rows) <= 16:
        raise ValueError("policy16_complete_" + name + "_one_to_sixteen_required")
    return rows


def _scaled_weights(weights):
    weights = tuple(_finite(weight, positive=True) for weight in weights)
    # Exact common scaling preserves weights and avoids an overflowing sum.
    # Products near underflow still need exact representability checks below.
    shift = math.frexp(max(weights))[1] + (len(weights) - 1).bit_length()
    scaled = tuple(math.ldexp(weight, -shift) for weight in weights)
    if any(weight <= 0 for weight in scaled):
        raise ValueError("policy16_positive_weight_lost_by_scaling")
    factor = Fraction(2) ** -shift
    if any(Fraction(after) != Fraction(before) * factor
           for before, after in zip(weights, scaled)):
        raise ValueError("policy16_weight_ratio_not_exactly_representable")
    return scaled, shift


def _rank(row):
    return (
        -row["expected_natural_slo_good"], row["expected_cohort_wall_s"],
        row["transitions"], sum(row["path"]), tuple(row["path"]),
    )


def _validate_complete(result):
    try:
        contract = result["planner_contract"]
        budgets, offsets = contract["budgets"], contract["step_offsets_s"]
        scenarios = result["scenario_count"]
        if (
            result["status"] != "model_plan_only"
            or result["guard_schema"] != GUARD_SCHEMA
            or contract["objective"] != OBJECTIVE
            or result["planner_contract_sha256"] != _fingerprint(contract)
            or not 1 <= scenarios <= 16 or not 2 <= len(offsets) <= 4
            or not 1 <= len(budgets) <= 3 or len(set(budgets)) != len(budgets)
            or result["native_mpc_permitted"] is not False
            or result["deployable"] is not False
            or result.get("refusals") or result.get("policy_guard_refusals")
            or result.get("timing_refused")
            or result["cooperative_deadline_exhausted"]
        ):
            raise ValueError("policy16_complete_unqualified_panel_required")
        paths = set(product(budgets, repeat=len(offsets)))
        rows = result["rankings"]
        if (
            len(rows) != len(paths) or {tuple(row["path"]) for row in rows} != paths
            or result["candidate_paths"] != len(paths)
            or result["evaluations_expected"] != len(paths) * scenarios
            or result["evaluations_attempted"] != len(paths) * scenarios
            or len(result["all_offered_per_scenario"]) != scenarios
            or len(result["scenario_weights_scaled"]) != scenarios
            or result["received_budget"] not in budgets
        ):
            raise ValueError("policy16_partial_panel_never_ranked")
        weights = tuple(
            _finite(weight, positive=True) for weight in result["scenario_weights_scaled"])
        denominator = math.fsum(weights)
        for row in rows:
            samples = row["per_scenario"]
            if len(samples) != scenarios:
                raise ValueError("policy16_partial_panel_never_ranked")
            for index, sample in enumerate(samples):
                good, offered = sample["natural_slo_good"], sample["all_offered"]
                if (
                    type(good) is not int or type(offered) is not int
                    or not 0 <= good <= offered
                    or offered != result["all_offered_per_scenario"][index]
                ):
                    raise ValueError("policy16_all_offered_objective_required")
                _finite(sample["cohort_wall_s"])
            for sample_key, expected_key in (
                ("natural_slo_good", "expected_natural_slo_good"),
                ("cohort_wall_s", "expected_cohort_wall_s"),
            ):
                terms = [weight * sample[sample_key]
                         for weight, sample in zip(weights, samples)]
                # Include min normal: an inexact subnormal product can round up to it.
                if any(
                    not math.isfinite(term) or sample[sample_key] > 0 and term == 0
                    or 0 < term <= sys.float_info.min
                    and Fraction(term) != Fraction(weight) * Fraction(sample[sample_key])
                    for term, weight, sample in zip(terms, weights, samples)
                ):
                    raise ValueError("policy16_unrepresentable_weighted_objective")
                expected = math.fsum(terms) / denominator
                if _finite(row[expected_key]) != expected:
                    raise ValueError("policy16_weighted_objective_mismatch")
            transitions = sum(
                left != right
                for left, right in zip([result["received_budget"]] + row["path"], row["path"]))
            if row["transitions"] != transitions:
                raise ValueError("policy16_received_budget_transition_mismatch")
        if (
            rows != sorted(rows, key=_rank)
            or result["best_multistep"] != rows[0]
            or result["best_fixed"] != min(
                (row for row in rows if len(set(row["path"])) == 1), key=_rank)
            or result["best_one_move"] != min(
                (row for row in rows if row["transitions"] <= 1), key=_rank)
        ):
            raise ValueError("policy16_complete_policy_objective_mismatch")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("policy16_invalid_complete_panel") from exc


@dataclass(frozen=True)
class FrozenFixedChoice:
    budget: int
    planner_contract_sha256: str
    training_panel_sha256: tuple[str, ...]
    training_weights: tuple[int | float, ...]
    candidate_objectives: tuple[tuple, ...]
    split: str = "train"

    def __post_init__(self):
        def sha(value):
            return (type(value) is str and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value))

        if (
            type(self.budget) is not int or self.budget not in (16, 32)
            or self.split != "train" or not sha(self.planner_contract_sha256)
            or type(self.training_panel_sha256) is not tuple
            or not 1 <= len(self.training_panel_sha256) <= 16
            or not all(sha(value) for value in self.training_panel_sha256)
            or type(self.training_weights) is not tuple
            or len(self.training_weights) != len(self.training_panel_sha256)
            or type(self.candidate_objectives) is not tuple
            or len(self.candidate_objectives) != 2
        ):
            raise ValueError("policy16_immutable_train_choice_fields_required")
        for weight in self.training_weights:
            _finite(weight, positive=True)
        for budget, row in zip((16, 32), self.candidate_objectives):
            if type(row) is not tuple or len(row) != 4 or row[0] != budget:
                raise ValueError("policy16_immutable_train_candidate_rows_required")
            for value in row[1:]:
                _finite(value)


def freeze_fixed_choice(training_results, *, split, panel_weights=None):
    """Caller-declared train-only evidence, never heldout selection or attestation."""
    if split != "train":
        raise ValueError("policy16_fixed_selection_train_only")
    received = _bounded(training_results, "training_panels")
    received_sha = _fingerprint(received)
    results = deepcopy(received)
    if _fingerprint(received) != received_sha or _fingerprint(results) != received_sha:
        raise ValueError("policy16_training_panels_changed_during_snapshot")
    for result in results:
        _validate_complete(result)
    contracts = {result["planner_contract_sha256"] for result in results}
    if len(contracts) != 1:
        raise ValueError("policy16_training_common_objective_horizon_and_actions_required")
    weights = (
        (1.0,) * len(results) if panel_weights is None
        else _bounded(panel_weights, "training_weights"))
    if len(weights) != len(results):
        raise ValueError("policy16_training_weight_inventory")
    exact_weights = tuple(Fraction(_finite(weight, positive=True)) for weight in weights)
    denominator = sum(exact_weights)
    horizon = len(results[0]["planner_contract"]["step_offsets_s"])
    candidates = []
    for budget in (16, 32):
        selected = []
        for result in results:
            row = next(
                (row for row in result["rankings"] if row["path"] == [budget] * horizon),
                None)
            if row is None:
                raise ValueError("policy16_both_original_training_fixed_candidates_required")
            selected.append(row)
        good, wall, transitions = (
            sum(weight * Fraction(row[key]) for weight, row in zip(exact_weights, selected))
            / denominator
            for key in ("expected_natural_slo_good", "expected_cohort_wall_s", "transitions")
        )
        candidates.append((budget, good, wall, transitions))
    chosen = min(candidates, key=lambda row: (
        -row[1], row[2], row[3], row[0] * horizon, (row[0],) * horizon))
    fingerprints = tuple(_fingerprint({
        "common_panel_sha256": result["common_panel_sha256"],
        "source_cut_canonical_sha256": result["source_cut_canonical_sha256"],
        "planner_contract_sha256": result["planner_contract_sha256"],
        "rankings": result["rankings"],
    }) for result in results)
    return FrozenFixedChoice(
        chosen[0], results[0]["planner_contract_sha256"], fingerprints,
        tuple(weights),
        tuple((budget, float(good), float(wall), float(transitions))
              for budget, good, wall, transitions in candidates))


class PolicyPlanner16(PolicyPlanner15):
    def __init__(
        self, *, cost_spec, budgets, step_offsets_s, solver_budget_s,
        actuation_budget_s, policy, fixed_budget=None, training_choice=None,
    ):
        if training_choice is not None:
            if (
                type(training_choice) is not FrozenFixedChoice
                or policy != "train_selected_fixed" or training_choice.split != "train"
                or type(training_choice.budget) is not int
                or training_choice.budget not in (16, 32)
                or fixed_budget is not None and fixed_budget != training_choice.budget
            ):
                raise ValueError("policy16_explicit_frozen_training_choice_required")
            fixed_budget = training_choice.budget
        super().__init__(
            cost_spec=cost_spec, budgets=budgets, step_offsets_s=step_offsets_s,
            solver_budget_s=solver_budget_s, actuation_budget_s=actuation_budget_s,
            policy=policy, fixed_budget=fixed_budget)
        self._solver_limit16 = self.solver_budget_s
        self._contract16 = self._contract()
        self._contract_sha16 = _fingerprint(self._contract16)
        if (
            training_choice is not None
            and training_choice.planner_contract_sha256 != self._contract_sha16
        ):
            raise ValueError("policy16_frozen_training_configuration_mismatch")
        self._selection16 = (self.policy, self.fixed_budget)
        self._training_choice16 = training_choice
        self._last_source16 = None
        self._last_version16 = None
        self._busy16 = False

    def _contract(self):
        return {
            "budgets": list(self.budgets), "step_offsets_s": list(self.offsets),
            "cost_spec": asdict(self.cost_spec),
            "solver_limit_s": self._solver_limit16,
            "actuation_budget_s": self.actuation_budget_s,
            "objective": OBJECTIVE, "first_action_source_age_limit_s": 0.5,
        }

    def _configuration_guard(self):
        if (
            _fingerprint(self._contract()) != self._contract_sha16
            or (self.policy, self.fixed_budget) != self._selection16
            or _finite(self.solver_budget_s, positive=True) > self._solver_limit16
        ):
            raise ValueError("policy16_frozen_configuration_changed")

    @staticmethod
    def _refuse(result, reason, *, timing=False):
        for key in RANKED_FIELDS:
            result.pop(key, None)
        result.update(
            status="refused", first_action=None,
            native_mpc_permitted=False, deployable=False,
            timing_refused=timing or result.get("timing_refused", False))
        result.setdefault("refusals", [])
        result.setdefault("policy_guard_refusals", []).append(reason)
        return result

    def plan(self, scenarios, *, clock_now=None):
        return self._execute(
            scenarios, clock_now=clock_now, started=time.monotonic(),
            version=None, version_now=None, acquisition_entry=None)

    def plan_acquisition(self, acquisition, *, version_now, clock_now=None):
        """Optional same-type token adapter; the runtime still owns IPC/CAS/TTL."""
        started = time.monotonic()
        if (
            not isinstance(acquisition, NativePanelAcquisition)
            or not isinstance(acquisition.entry, PlanningEntry)
            or type(acquisition.observation_version) is not int
            or acquisition.observation_version < 0 or not callable(version_now)
        ):
            raise ValueError("policy16_typed_versioned_acquisition_required")
        return self._execute(
            acquisition.scenarios,
            clock_now=clock_now if clock_now is not None else lambda: acquisition.entry,
            started=started, version=acquisition.observation_version,
            version_now=version_now, acquisition_entry=acquisition.entry)

    def _execute(
        self, scenarios, *, clock_now, started, version, version_now, acquisition_entry,
    ):
        if self._busy16:
            raise ValueError("policy16_single_inflight_plan_required")
        _finite(started)
        self._busy16 = True
        try:
            self._configuration_guard()
            effective_budget = self.solver_budget_s
            if not callable(clock_now):
                raise ValueError("native_planner_common_domain_entry_clock_required")
            entry = clock_now()
            if not isinstance(entry, PlanningEntry):
                raise ValueError("native_planner_common_domain_entry_clock_required")
            _finite(entry.at)
            if acquisition_entry is not None and (
                entry.clock_domain != acquisition_entry.clock_domain
                or entry.at < _finite(acquisition_entry.at)
            ):
                raise ValueError("policy16_acquisition_entry_domain_or_order")
            if version is not None:
                actual = version_now()
                if type(actual) is not int or actual != version:
                    raise ValueError("policy16_observation_changed_before_planning")
                if self._last_version16 is not None and version <= self._last_version16:
                    raise ValueError("policy16_reobserve_newer_version_required")
                self._last_version16 = version
            received = _bounded(scenarios, "scenarios")
            if any(not isinstance(row, NativeScenario) for row in received):
                raise ValueError("native_planner_explicit_common_scenarios_required")
            first = received[0]
            if not isinstance(first.frontier, HistoricalOwnerScenario):
                raise ValueError("native_planner_historical_native_origin_required")
            source = first.frontier.source
            source_at = _finite(source.source_at)
            source_key = (
                source.clock_domain, source.native_run_id, source.gateway_run_id,
                source.capture_id)
            if (
                entry.clock_domain != source.clock_domain
                or entry.at < _finite(first.frontier.decision_at)
            ):
                raise ValueError("native_planner_entry_clock_domain_or_order")
            if self._last_source16 is not None and (
                source_key != self._last_source16[0] or source_at <= self._last_source16[1]
            ):
                raise ValueError("native_planner_reobserve_new_source_before_replanning")
            self._last_source16 = (source_key, source_at)
            received_sha = _fingerprint(received)
            owned = deepcopy(received)
            if (
                _fingerprint(received) != received_sha
                or _fingerprint(owned) != received_sha
            ):
                raise ValueError("policy16_panel_changed_during_snapshot")
            weights, shift = _scaled_weights(row.weight for row in owned)
            panel = tuple(replace(row, weight=weight) for row, weight in zip(owned, weights))
            panel_sha = _fingerprint(panel)
            last_tick = _finite(time.monotonic())
            if last_tick < started:
                raise ValueError("policy16_nonmonotonic_clock")
            with OwnerSourceAudit(reuse_received_prefix=True) as source_audit:
                result = super()._plan(
                    panel, clock_now=lambda: entry, started=started,
                    source_audit=source_audit)
                result.update(
                    policy=self.policy, fixed_budget=self.fixed_budget,
                    guard_schema=GUARD_SCHEMA,
                    planner_contract=deepcopy(self._contract16),
                    planner_contract_sha256=self._contract_sha16,
                    common_panel_sha256=received_sha,
                    evaluated_panel_sha256=panel_sha,
                    scenario_weights_original=[row.weight for row in owned],
                    scenario_weights_scaled=list(weights),
                    weight_binary_scale_exponent=shift,
                    received_budget=owned[0].frontier.source_cut["owner_prefix"]["snapshot"]["budget"],
                    observation_version=version,
                    observation_version_guard=(
                        "trusted_callback_entry_and_return" if version is not None
                        else "legacy_source_only_no_observation_version_attestation"),
                    training_fixed_choice=(
                        asdict(self._training_choice16) if self._training_choice16 else None),
                    fixed_choice_provenance=(
                        "frozen_caller_declared_train_only_evidence"
                        if self._training_choice16 else "constructor_frozen_caller_choice"),
                )
                if result["status"] == "model_plan_only":
                    try:
                        _validate_complete(result)
                    except ValueError as exc:
                        self._refuse(result, str(exc))
                    else:
                        self._select_policy(result)
            result["static_source_audit"] = "single-plan-entry-and-exit-v1"
            result["policy_selection_timing"] = (
                "entry_snapshot_full_panel_selection_audits_version_and_json_preparation"
                "_cooperative_only")
            if _fingerprint(panel) != panel_sha:
                self._refuse(result, "policy16_common_panel_mutated")
            try:
                self._configuration_guard()
                if self.solver_budget_s != effective_budget:
                    raise ValueError("policy16_solver_budget_changed_during_plan")
            except ValueError as exc:
                self._refuse(result, str(exc))
            if version is not None:
                actual = version_now()
                if type(actual) is not int or actual != version:
                    self._refuse(result, "policy16_observation_changed_during_planning")
            try:
                json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
            except (ValueError, OverflowError):
                self._refuse(result, "policy16_nonfinite_output")
            if version is not None:
                actual = version_now()
                if type(actual) is not int or actual != version:
                    self._refuse(result, "policy16_observation_changed_during_output")
            try:
                self._configuration_guard()
                if self.solver_budget_s != effective_budget:
                    raise ValueError("policy16_solver_budget_changed_during_output")
            except ValueError as exc:
                self._refuse(result, str(exc))
            now = time.monotonic()
            if type(now) not in (int, float) or not math.isfinite(now) or now < last_tick:
                result["solver_elapsed_s"] = None
                return self._refuse(result, "policy16_nonmonotonic_clock", timing=True)
            elapsed = now - started
            result["solver_elapsed_s"] = elapsed
            offset = self._contract16["step_offsets_s"][0]
            actuation = self._contract16["actuation_budget_s"]
            age = entry.at - source_at + max(offset, elapsed + actuation)
            result["first_action_source_age_s"] = age
            if (
                elapsed > effective_budget or age > 0.5
                or elapsed + actuation > offset
            ):
                self._refuse(result, "policy16_full_entry_solver_actuation_deadline", timing=True)
            return result
        finally:
            self._busy16 = False
