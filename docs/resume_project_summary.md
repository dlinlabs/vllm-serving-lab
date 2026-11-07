# LLM serving control: evidence-driven project summary

Status: **A: Resume Ready (scoped, independently audited); B: native MPC P0 open**.
Evidence cutoff: 2026-11-04; final wording audit: 2026-11-05.
Negative results are acceptable; unsupported claims are not.

## Independent deliverables

**A — Resume Evidence Package:** the gateway, instrumentation, replay and
CPU-planner claims in [resume bullets](resume_bullets.md) stand on existing
evidence. They do not wait for native MPC success. This is a prototype-safe
package, not a declaration that all native P0 gates passed. The
[final independent audit and Mayor acceptance](../results/mpc-native-r24/reviews/resume-evidence-audit-a-parent02.json)
record the original failures and their approved corrections.

**B — Native Closed-loop MPC Validation:** incomplete. Require at least two
consecutive timestamped observe/predict/plan/first-action-apply/reobserve cycles,
an observed change in actual admission behavior, Train-only baseline selection,
a preregistered held-out comparison, reproduction and independent audit.
A changed planner output or budget ACK alone is insufficient.

## Problem and contribution

Heterogeneous prompts and output caps make request-count limits an imperfect
proxy for serving load. This project investigates admission decisions in front
of vLLM, balancing completed-request SLO goodput, overload rejection and latency.

The original work is the gateway admission/accounting layer, lifecycle and
control-timing instrumentation, auditable workload/replay tooling, and
experimental prediction, planning and guarded-actuation components. It is
**not** an implementation of vLLM's continuous batching, PagedAttention, model
kernels or KV-cache manager.

The proposed controller observes versioned state, predicts candidate budget
trajectories, optimizes over a finite horizon, executes the first action and
reobserves. The actuator changes gateway reservation budget, not the native
engine's token budget. [Architecture](architecture.md) distinguishes working
subsystems from the unqualified integrated loop.

## Immediate evidence report

| Defensible now | Evidence and boundary |
|---|---|
| Instrumented, audited real-GPU serving captures | E1: archived request outcomes, state reconstruction and applied-control timing. These are scheduled identification runs, not MPC runs. |
| Implemented a frozen policy-planner component | E2: Parent-accepted exact candidate ordering, with full-action Fixed-reference scope and CPU deadline/version checks. Runtime integration is not accepted. |
| Implemented and tested observation/consumer deployment components | E3: bounded CPU integration evidence; subsequent D2-F1 blocks live release of that composition. |
| Preserved negative results and release blockers | E4: component coverage FAIL (one of two required observations). E5: source-identified cleanup-bypass path. Neither is relabelled PASS. |
| Corrected baseline timing and validated replay inputs | E6: Fixed no longer inherits solver-only waiting/deadlines. E7: real-tokenizer Train conversion; this was CPU work, not GPU inference. |
| Implemented causal Train-only forecast generation | E8: bounded CPU component acceptance; nominal assumptions, not measured native accuracy. |

Trace identities, archive locations, commands, configurations and recorded
commits are indexed in
[benchmark results](benchmark_results.md). Historical audits are not represented
as new replications performed for this document.

## MVP completion gate

Native deliverable B is complete only when all rows below have reviewable evidence. Winning every
baseline, convergence of the Native14 posterior, and production deployment are
**not** requirements.

| Priority | Deliverable / acceptance | Current status |
|---|---|---|
| P0 | Explain original work, actuator, states, model limitations, objective and constraints without conflating gateway and engine state. | A's scoped claim audit and approved wording corrections PASS. B's final native-results audit remains open. |
| P0 | One runnable, source-pinned path demonstrating repeated real observation, prediction, optimization, first-action application and reobservation. Retain refusals, timeouts and actual applied-state evidence. | **Missing native end-to-end evidence.** Shadow output and ACK-only evidence do not qualify. |
| P0 | Safe bounded operation: queue/admission limits, overload outcomes, stale-state refusal, unknown-command handling, timeout cleanup and private-config handling. | New isolated service ownership and runtime integration pending. The legacy D2-F1/R1 launcher remains blocked and is not the selected path. |
| P0 | Comparable descriptive GPU results for MPC, strong development-selected Fixed, FIFO and strong OneMove; same workload, model, resource limits and scoring. Predeclare evaluation data disjoint from predictor/scenario fitting and all policy tuning/selection. | **Not collected.** Train/Eval generation is frozen; empirical Fixed selection, actual backend-order proof and corrected metric comparability remain open. |
| P0 | Report TTFT p50/p95/p99, SLO goodput, throughput, rejection, fairness and full controller overhead, with sample counts and failures. | Historical measurements are partial; current MPC table is unfilled. |
| P0 | Portable commands/configurations, raw-to-table reconstruction and exact code/data commits; audit every quantitative resume claim. | Historical bundles exist; clean-checkout MVP reproduction remains unverified. |
| P1 | Better decision-relevant forecasts, overhead reduction, broader workload/load coverage and repeatability estimates. | Only if a P0 result justifies the change. |
| P2 | Native14 mixing research, full empirical-pool quantiles, new proposal geometry, extensive posterior qualification, large confirmatory campaigns and additional hardware. | Separate research backlog; must not delay P0 or applications. |

A replacement simple frozen predictor is permissible only under its own
explicit experimental scope, support and validation. It cannot inherit
Native14 qualification, waive safety checks, or be presented as a calibrated
posterior. Existing scientific gates remain unchanged.

## Smallest remaining execution path

1. Reuse accepted planner/accounting components. Finish the isolated runtime,
   request recorder, service ownership and metric-audit fixes; do not reopen
   the legacy launcher or another model family.
2. Freeze one MVP protocol: exact commands, state/action interface, development-
   only baseline selection, FIFO meaning, disjoint development/evaluation
   workload identities, resource bounds, metric definitions, failure handling
   and stopping budget. Scenario panels may use only information available at
   decision time, never the scored workload's future arrivals. Resolve
   portability before GPU work.
3. Perform the smallest bounded real-loop engineering qualification. Require
   two consecutive complete timestamped cycles and an actual admission-behavior
   change, not a same-budget no-op, simulated receipt or cap ACK alone.
4. If the smoke passes, run the predeclared sequential policy comparison on the
   authorized GPU. Publish descriptive results even if MPC loses.
5. Reconstruct the table, audit claims and commit the evidence. Declare
   **Native Closed-loop MPC Validated** only after B closes. Package A and
   applications do not wait for that declaration. Move optional research to P1/P2.

No experiment is launched by this document. Before each proposed run, Mayor must
record: the specific P0 claim it establishes, why existing evidence is
insufficient, fixed scope/budget, outcomes that stop work, and resulting
trace/config/command/commit. A run with no new P0 answer is deferred.
The former open-ended iteration objective is superseded by this gate.

The E1 Azure trace is development data, not held-out evaluation. A replay of it
may be an explicitly labelled in-sample engineering pilot, but cannot satisfy
the held-out comparison row or support a generalization claim. Freeze the Fixed
selection method and all predictor/scenario/policy settings before evaluation;
do not tune on evaluation outcomes.

## Resume-safe wording

- Built an instrumented vLLM serving gateway and auditable trace-replay
  tooling for admission-control and overload analysis, preserving request
  outcomes and auditable control-timing evidence.
- Implemented finite-horizon policy-planner prototypes with CPU-tested candidate
  ordering and deadline/version checks, and corrected a Fixed-baseline timing
  bias in the current nominal planner while retaining explicit native-integration
  and comparison limitations.

These are prototype claims, not production deployment, guaranteed latency, or
MPC superiority. Role-specific alternatives are in [resume bullets](resume_bullets.md).
The [initial independent audit and Mayor disposition](../results/mpc-native-r24/reviews/resume-mvp-claims-parent01.json)
record wording corrections, bounded evidence and unresolved P0 requirements.

## Seven-agent ownership

| Role | Bounded resume-focused responsibility |
|---|---|
| Model Identification | State/parameter/model/support table tied to accepted evidence. |
| Sampling & Mixing | Estimation reproducibility and numerical limitations; no convergence campaign for P0. |
| Model Improvement | Only validated improvements that change or explain controller decisions. |
| Independent Validation | Claim ledger, leakage audit, counterexamples and unsupported wording. |
| MPC Planner | Objective, horizon, constraints, actions and Fixed/OneMove distinction. |
| Closed-loop Runtime | Actual first-action loop and bounded failure/cleanup behavior. |
| Fair Comparison | Comparable baseline definitions, metrics, overhead and all failure cases. |

Mayor owns integration, acceptance, GPU serialization and local commits.
