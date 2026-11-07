# LLM serving control: evidence-driven project summary

Status: **research prototype; not Resume Ready for a real closed-loop MPC claim**.
Evidence cutoff: 2026-11-04. Negative results are acceptable; unsupported claims are not.

## Problem and contribution

Heterogeneous prompts and output caps make request-count limits an imperfect
proxy for serving load. This project investigates admission decisions in front
of vLLM, balancing completed-request SLO goodput, overload rejection and latency.

The original work is the gateway admission/accounting layer, lifecycle and
control-timing instrumentation, reproducible workload/audit tooling, and
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
| Implemented finite-horizon policy-planning components | E2: independently accepted CPU/shadow tests. No native performance claim follows. |
| Implemented and tested observation/consumer deployment components | E3: bounded CPU integration evidence; subsequent D2-F1 blocks live release of that composition. |
| Preserved negative results and release blockers | E4/E5: incomplete native observation coverage and cleanup failure path; neither is relabelled PASS. |

Full traces, commands, configurations and exact commits are indexed in
[benchmark results](benchmark_results.md). Historical audits are not represented
as new replications performed for this document.

## MVP completion gate

P0 is complete only when all rows below have reviewable evidence. Winning every
baseline, convergence of the Native14 posterior, and production deployment are
**not** requirements.

| Priority | Deliverable / acceptance | Current status |
|---|---|---|
| P0 | Explain original work, actuator, states, model limitations, objective and constraints without conflating gateway and engine state. | Documented; final independent claim audit pending. |
| P0 | One runnable, source-pinned path demonstrating repeated real observation, prediction, optimization, first-action application and reobservation. Retain refusals, timeouts and actual applied-state evidence. | **Missing native end-to-end evidence.** Shadow output and ACK-only evidence do not qualify. |
| P0 | Safe bounded operation: queue/admission limits, overload outcomes, stale-state refusal, unknown-command handling, timeout cleanup and private-config handling. | Components tested; **D2-F1 repair/acceptance remains required** for the affected launcher. |
| P0 | Comparable descriptive GPU results for MPC, strong development-selected Fixed, FIFO and strong OneMove; same workload, model, resource limits and scoring. | **Not collected.** FIFO identity and missing throughput/fairness reporting require explicit resolution. |
| P0 | Report TTFT p50/p95/p99, SLO goodput, throughput, rejection, fairness and full controller overhead, with sample counts and failures. | Historical measurements are partial; current MPC table is unfilled. |
| P0 | Portable commands/configurations, raw-to-table reconstruction and exact code/data commits; audit every quantitative resume claim. | Historical bundles exist; clean-checkout MVP reproduction remains unverified. |
| P1 | Better decision-relevant forecasts, overhead reduction, broader workload/load coverage and repeatability estimates. | Only if a P0 result justifies the change. |
| P2 | Native14 mixing research, full empirical-pool quantiles, new proposal geometry, extensive posterior qualification, large confirmatory campaigns and additional hardware. | Separate research backlog; must not delay P0 or applications. |

A replacement simple frozen predictor is permissible only under its own
explicit experimental scope, support and validation. It cannot inherit
Native14 qualification, waive safety checks, or be presented as a calibrated
posterior. Existing scientific gates remain unchanged.

## Smallest remaining execution path

1. Reuse accepted planner/accounting components. Resolve the known cleanup and
   scorer blockers; finish one small integration path, not another model family.
2. Freeze one MVP protocol: exact commands, state/action interface, baseline
   selection, FIFO meaning, workload, resource bounds, metric definitions,
   failure handling and stopping budget. Resolve portability before GPU work.
3. Perform one bounded real-loop engineering smoke. Require actual application
   and reobservation, not a same-budget no-op or simulated action receipt.
4. If the smoke passes, run the predeclared sequential policy comparison on the
   authorized GPU. Publish descriptive results even if MPC loses.
5. Reconstruct the table, audit claims, commit the evidence and declare
   **Resume Ready** only after P0 closes. Move all optional research to P1/P2.

No experiment is launched by this document. Before each proposed run, Mayor must
record: the specific P0 claim it establishes, why existing evidence is
insufficient, fixed scope/budget, outcomes that stop work, and resulting
trace/config/command/commit. A run with no new P0 answer is deferred.
The former open-ended iteration objective is superseded by this gate.

## Safe resume wording today

- Built an instrumented vLLM serving gateway and reproducible trace-replay
  tooling for admission-control and overload analysis, preserving request
  outcomes and auditable control-timing evidence.
- Implemented and CPU-validated finite-horizon MPC planning components with
  versioned-state, deadline and failure checks; documented the remaining native
  closed-loop and baseline-comparison gaps.

These are prototype claims, not production deployment, guaranteed latency, or
MPC superiority. Role-specific alternatives are in [resume bullets](resume_bullets.md).

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
