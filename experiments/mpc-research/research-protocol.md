# Native four-model MPC research protocol

Source freeze: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Owner: Mayor. Epic: `pr-mpc`. Convoy: `hq-cv-8tdlf`, caller-owned, local only.
This protocol does not select an MPC architecture before independent research.

## Common task and independence

All four researchers answer the same complete problem. Models are
`claude-opus-5.5`, `gpt-6-astra`, `gemini-3.8-flash` and `gpt-6.1-sol`,
launched with their native Gas Town profiles and separate Polecat worktrees.
Profile text, process arguments and actual model usage must be distinguished.

Read the current-state audit, then inspect current source and primary reports.
Audit statements are starting evidence, not constraints on scientific conclusions.
In round 1, do not read peer proposals or Mayor design preferences.

Each proposal must contain:

1. Minimal state, controls, disturbances and observation equations; for each
   candidate state justify necessity, observability, delay/noise, identifiability,
   incremental predictive value and an ablation that can reject its inclusion.
2. Explicit serving dynamics, units, conservation and cap-reduction semantics.
   Distinguish current gateway actuators from proposed scheduler changes.
3. Objective, hard/soft/terminal constraints, feasibility/stability conditions,
   sampling/horizon choices, solver/observation/actuation budgets and fallback.
   Do not claim deterministic latency guarantees under unbounded disturbances.
4. Comparison of fixed concurrency/cost caps, heuristics, hysteresis, queue
   control, linear/nonlinear/robust/stochastic/learning/hierarchical MPC.
5. Tail-risk/uncertainty calibration, OOD behavior, unknown decode length and
   nonstationary arrivals. State exact assumptions behind statistical claims.
6. Identification and experimental design, action support, chronological
   train/calibration/validation/final splits, counterfactual limitations,
   repeated-run inference, competing baselines and negative-result criteria.
7. Implementable module boundaries/interfaces and targeted falsification tests.
   Costs/weights need SLO/units/Pareto or development-data justification.
8. A recommended architecture, alternatives and unresolved hypotheses; cite
   source locations and label deductions, measured facts and proposed experiments.

## Three rounds

- Round 1: independent proposal, saved as a separate report. Stop at the barrier.
- Round 2: read all peer proposals only after Mayor releases the barrier.
  Produce adversarial critique with equations, concrete counterexamples,
  failure cases and discriminating experiments rather than endorsements.
- Round 3: revise after all critiques; preserve evidence-unresolved disagreements.
  Mayor adjudicates by evidence and executable counterexamples, not majority vote.

Reports are `experiments/mpc-research/proposals/<model>-round<N>.md`
inside the researcher's own worktree. Save each round separately.
Notify Mayor using native mail and remain available for the next round.

## Execution and safety

Only pr32 worktrees and native tracking may change. No sandbox rig modifications.
No push, merge, PR comments, history rewrite, paid GPU/external compute or
credentials in artifacts. Do not execute `gt done`: its normal completion path
can push. Do not auto-run Git finishing/landing recipes. Research rounds may
write their reports, but must not edit product code or commit.

After architecture review, Mayor creates disjoint implementation Beads/Polecats,
freezes module interfaces, integrates local changes and runs required checks.
Do not silently expand the available actuators or enable live actuation.

## Deliverables and stopping

Produce current-state audit, mathematical model, identification report,
controller comparison, MPC specification, ADR, three-round consensus evidence,
implementation/test evidence, benchmark evidence ledger and limitations.
CPU-only artifacts must explicitly leave GPU/HTTP/performance acceptance false.

Stop development iterations that add no new evidence; retain negative results.
Paid experiments stay blocked until an explicit plan, time/cost ceiling and
human authorization exist. Lack of GPU evidence is a real remaining gate, not a
reason to fabricate benchmark results or claim mission completion.
