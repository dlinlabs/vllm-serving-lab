"""Fair 5-arm comparative evaluation harness for local R21 MPC research.

This harness enforces:
- Identical frozen arrival/payload tapes across all arms in each seed block
- Identical causal PublicState observations and non-anticipating information access
- Explicit support for five registered arms:
  * fixed16: constant budget 16.0
  * fixed32: constant budget 32.0
  * reactive: non-anticipating stock threshold rule (stock > 16.0 -> 16.0, else 32.0)
  * strongOneMove: 1-step lookahead optimizer sharing objective and continuation
  * MPC: 2-step receding-horizon lookahead optimizer sharing objective and continuation
- Stand-alone operation for fixed/reactive baselines without requiring model/planner
- Graceful deferred integration for model-based arms
- All-assigned cohort and in-window scoring with full failure and delay accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from contextlib import nullcontext
import importlib
import math
import random
from typing import Any, Callable

from mpc_local.contracts import Choice, Config, PublicState
from mpc_local.simulator import WorkItem, compute_tape_hash, generate, simulate


# ---------------------------------------------------------------------------
# Policy Implementations
# ---------------------------------------------------------------------------

def fixed16_policy(state: PublicState) -> Choice:
    """Fixed-cost 16.0 baseline policy."""
    return Choice(
        budget=16.0,
        plan=(16.0, 16.0),
        scores=(),
        status="ok",
        elapsed_s=0.0,
    )


def fixed32_policy(state: PublicState) -> Choice:
    """Fixed-cost 32.0 baseline policy."""
    return Choice(
        budget=32.0,
        plan=(32.0, 32.0),
        scores=(),
        status="ok",
        elapsed_s=0.0,
    )


def reactive_policy(state: PublicState, threshold: float = 16.0) -> Choice:
    """Reactive threshold policy.

    Per r21-local-protocol.json:
    'shared frozen nonanticipating reactive rule: cost stock above 16 chooses 16, otherwise 32'
    """
    current_stock = math.fsum(r.cost for r in state.requests)
    budget = 16.0 if current_stock > threshold else 32.0
    return Choice(
        budget=budget,
        plan=(budget, budget),
        scores=(),
        status="ok",
        elapsed_s=0.0,
    )


def make_prbs_policy(seed: int = 99175, slot_s: float = 1.0, duration_s: float = 20.0) -> Callable[[PublicState], Choice]:
    """Prespecified open-loop independent random 16/32 per 1s slot from PRBS(seed)."""
    import random
    rng = random.Random(seed)
    num_slots = int(math.ceil(duration_s / slot_s)) + 10
    schedule = [float(rng.choice((16, 32))) for _ in range(num_slots)]

    def _policy(state: PublicState) -> Choice:
        idx = int(state.now // slot_s)
        budget = schedule[idx] if 0 <= idx < len(schedule) else 16.0
        return Choice(
            budget=budget,
            plan=(budget, budget),
            scores=(),
            status="ok",
            elapsed_s=0.0,
        )

    return _policy


def build_policy(
    arm: str,
    model: Any = None,
    planner: Any = None,
    config: Config | None = None,
    frozen_prefix: bool = False,
) -> Callable[[PublicState], Choice | float]:
    """Build policy callable for a named arm.

    Args:
        arm: One of 'fixed16', 'fixed32', 'reactive', 'strongOneMove', 'MPC', 'prbs99175'.
        model: Optional fitted model instance.
        planner: Optional planner module or object exposing choose().
        config: Optional Config instance.

    Returns:
        Callable mapping PublicState to Choice or float budget.
    """
    cfg = config or Config()

    if arm == "fixed16":
        return fixed16_policy

    if arm == "fixed32":
        return fixed32_policy

    if arm in ("prbs99175", "identification"):
        return make_prbs_policy(seed=99175, slot_s=cfg.slot_s, duration_s=cfg.offered_end_s or 20.0)

    if arm == "reactive":
        return lambda state: reactive_policy(state, threshold=cfg.reactive_threshold)

    if arm in ("strongOneMove", "MPC"):
        if model is None:
            raise RuntimeError(f"Arm '{arm}' requires planner integration and a fitted model.")
        mode = "one_move" if arm == "strongOneMove" else "mpc"

        # Check if planner is explicitly passed
        p_module = planner
        if p_module is None:
            try:
                p_module = importlib.import_module("mpc_local.planner")
            except ImportError as err:
                raise RuntimeError(
                    f"Arm '{arm}' requires planner integration. "
                    "mpc_local.planner is not yet integrated in this clean worktree."
                ) from err

        choose_fn = getattr(p_module, "choose", None)
        if choose_fn is None:
            raise RuntimeError(f"planner module {p_module} has no choose() function.")

        first = None
        first_version = None

        def _model_based_policy(state: PublicState) -> Choice:
            nonlocal first, first_version
            if frozen_prefix and first is not None:
                if first.status != "ok":
                    return first
                if state.version == first_version:
                    return replace(first, status="frozen_first_not_applied", plan=())
                slot = int(state.now // cfg.slot_s)
                budget = first.plan[slot] if slot < len(first.plan) else reactive_policy(
                    state, threshold=cfg.reactive_threshold,
                ).budget
                return Choice(budget, first.plan, first.scores, "ok", 0.0)
            result = choose_fn(
                state=state,
                model=model,
                config=cfg,
                mode=mode,
            )
            if frozen_prefix:
                first = result
                first_version = state.version
            return result

        return _model_based_policy

    raise ValueError(f"unknown_arm: {arm}. Must be one of ('fixed16', 'fixed32', 'reactive', 'strongOneMove', 'MPC').")


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------

def evaluate_arm(
    items: list[WorkItem],
    arm: str,
    duration_s: float = 20.0,
    control_delay_s: float = 0.0,
    solver_delay_s: float = 0.0,
    model: Any = None,
    planner: Any = None,
    config: Config | None = None,
    service_model: str = "v2_event_driven",
    parallel: bool = False,
    frozen_prefix: bool = False,
) -> dict[str, Any]:
    """Evaluate a single arm on a frozen set of WorkItems.

    Args:
        items: List of WorkItems (arrival tape).
        arm: Named arm to evaluate.
        duration_s: Offered arrival duration in seconds.
        control_delay_s: Actuator delay in seconds.
        solver_delay_s: Solver execution delay in seconds.
        model: Optional model for model-based arms.
        planner: Optional planner for model-based arms.
        config: Configuration instance.
        service_model: 'v2_event_driven' (default) or 'v1_admission_fixed'.

    Returns:
        Simulation result dict with added 'arm' field in metrics.
    """
    cfg = config or Config()
    if cfg.offered_end_s is None:
        cfg = replace(cfg, offered_end_s=duration_s)
    arm_model = copy.deepcopy(model) if model is not None else None
    context = nullcontext(arm_model)
    if parallel and arm in ("strongOneMove", "MPC"):
        from mpc_local.parallel_predictor import ParallelPredictor
        context = ParallelPredictor(arm_model)
    with context as backend:
        policy = build_policy(arm=arm, model=backend, planner=planner, config=cfg,
                              frozen_prefix=frozen_prefix)
        readiness = backend.metadata if parallel and arm in ("strongOneMove", "MPC") else None
        result = simulate(
            items=items, policy=policy, duration_s=duration_s,
            control_delay_s=control_delay_s,
            solver_delay_s=solver_delay_s if arm in ("strongOneMove", "MPC") else 0.0,
            config=cfg, service_model=service_model,
        )
        after_simulation = backend.metadata if readiness is not None else None
    if readiness is not None:
        result["backend_metadata"] = {
            "ready": readiness, "after_simulation": after_simulation, "closed": backend.metadata,
        }
    backend_failed = after_simulation is not None and (
        not after_simulation["ready"] or after_simulation["last_error"] is not None
    )
    result["metrics"]["backend_failed_mid_run"] = backend_failed
    result["metrics"]["optimizer_execution_complete"] = (
        not backend_failed and result["metrics"]["solver_failures"] == 0
    ) if arm in ("strongOneMove", "MPC") else None
    result["metrics"]["dispatch_execution_complete"] = all(
        decision["applied"] for decision in result["decisions"]
    )
    result["metrics"]["arm"] = arm
    return result


def evaluate_seed_block(
    seed: int,
    arms: list[str] | tuple[str, ...] = ("fixed16", "fixed32", "reactive"),
    duration_s: float = 20.0,
    regime: str = "mixed",
    control_delay_s: float = 0.0,
    solver_delay_s: float = 0.0,
    model: Any = None,
    planner: Any = None,
    config: Config | None = None,
    service_model: str = "v2_event_driven",
    parallel: bool = False,
    frozen_prefix: bool = False,
) -> dict[str, Any]:
    """Evaluate multiple arms on the EXACT SAME arrival tape for a given seed.

    Guarantees arrival tape identity across all evaluated arms.

    Args:
        seed: Random seed for workload generation.
        arms: Sequence of arm names to evaluate.
        duration_s: Offered arrival duration in seconds.
        regime: Workload regime ('mixed', etc.).
        control_delay_s: Transport/actuator delay.
        solver_delay_s: Optimizer solver delay.
        model: Optional model for model-based arms.
        planner: Optional planner for model-based arms.
        config: Configuration instance.
        service_model: 'v2_event_driven' (default) or 'v1_admission_fixed'.

    Returns:
        Dict with seed, arrival_tape_hash, arm_metrics, and pairwise contrasts.
    """
    cfg = config or Config()
    items = generate(seed=seed, duration_s=duration_s, regime=regime)
    tape_hash = compute_tape_hash(items)

    arm_results: dict[str, dict[str, Any]] = {}
    execution_order = list(arms)
    random.Random(seed ^ 0xA21F).shuffle(execution_order)
    for arm in execution_order:
        res = evaluate_arm(
            items=items,
            arm=arm,
            duration_s=duration_s,
            control_delay_s=control_delay_s,
            solver_delay_s=solver_delay_s,
            model=model,
            planner=planner,
            config=cfg,
            service_model=service_model,
            parallel=parallel,
            frozen_prefix=frozen_prefix,
        )
        # Verify tape hash invariant
        arm_tape_hash = res["metrics"]["arrival_tape_hash"]
        if arm_tape_hash != tape_hash:
            raise RuntimeError(
                f"Tape hash mismatch on arm {arm}: expected {tape_hash}, got {arm_tape_hash}"
            )
        arm_results[arm] = res

    # Compute pairwise contrasts for cohort_good
    contrasts: dict[str, float] = {}
    arm_names = list(arms)
    for i in range(len(arm_names)):
        for j in range(i + 1, len(arm_names)):
            a1 = arm_names[i]
            a2 = arm_names[j]
            diff = arm_results[a1]["metrics"]["cohort_good"] - arm_results[a2]["metrics"]["cohort_good"]
            contrasts[f"{a1}_minus_{a2}"] = diff

    return {
        "seed": seed,
        "regime": regime,
        "duration_s": duration_s,
        "arrival_tape_hash": tape_hash,
        "num_items_offered": len(items),
        "arms": {arm: res["metrics"] for arm, res in arm_results.items()},
        "contrasts": contrasts,
        "arm_results": arm_results,
        "arm_execution_order": execution_order,
    }
