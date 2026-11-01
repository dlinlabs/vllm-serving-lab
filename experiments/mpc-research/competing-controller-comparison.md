# Competing controller comparison

The purpose is to test whether prediction and future control freedom add value
beyond good observations. No method is selected as a native winner.

| Architecture | Useful mechanism | Key failure/identification risk | PR32 decision |
|---|---|---|---|
| Fixed cap | minimal overhead and predictable admission | workload-dependent optimum, no adaptation | mandatory tuned development baseline, frozen before test |
| Heuristic adaptive | reacts to backlog/latency cheaply | delayed/censored labels, thresholds and oscillation | mandatory same-state baseline |
| Threshold/hysteresis | suppresses noisy switching | lag, threshold dependence, stale budget | W_prompt hysteresis; tune only on development |
| Queue-based | backlog stability under declared service assumptions | no gateway waiting queue; engine queue is latent | backlog feedback is feasible; do not claim MaxWeight theorem |
| Linear MPC | cheap approximate multistep forecasts | mode changes, admission thresholds, nonnegative work and mixed units | later identified local-linear challenger, not assumed globally valid |
| Nonlinear MPC | captures contention and discrete admission | mismatch and solver latency; false precision | small finite candidate lifecycle rollout rather than general nonlinear solver |
| Robust MPC | explicit worst-case uncertainty | uncalibrated set or wide infeasible bounds; no observed KV | retain guards, not an unsupported robust-SLO certificate |
| Stochastic/chance MPC | expected reward and calibrated risk | dependence/OOD, unobservable counterfactual calibration, tail sample cost | scenarios for exploratory scoring; no chance constraints without valid scores |
| Learning-based MPC | nonlinear prediction and online adaptation | causal coverage, drift, leakage and unsafe identification | freeze model per epoch; no initial RLS/neural/RL dependency |
| Hybrid/hierarchical | exact phase events plus slower budget decisions | confusing admission with inner engine scheduling | selected CPU challenger: hybrid lifecycle + discrete gateway budget; no inner scheduler controller |

The implemented comparison must expose fixed caps16/32, development-selected
backlog feedback, a same-model one-control-move policy and3-move MPC. All receive
the same past information, cost convention and action set. Evaluation cohorts
and continuation remain fixed when varying control freedom; changing the
prediction window is a separate ablation.

Goodput, rejection, all-offered nonattainment, conditional TTFT/TPOT/E2E and
controller cost must be shown together. Reducing admitted-only latency by
rejecting everything is not success. No TPOT/E2E SLO or GPU-utilization weight is
fabricated when the experiment has not defined/qualified one. They remain
reported outcomes and later Pareto constraints. Initial switch/risk weights are
zero, with deterministic tie preference for hold/fewer changes.

Fair native policy testing needs current-host randomized action excitation,
frozen independent calibration/validation/test episodes, complete outcomes and
paired block-level uncertainty. AUC of factual TTFT, an apparently physical
equation, model consensus or a successful CPU suite cannot rank native policies.
If simple feedback matches the planner, report that negative result and prefer
the cheaper deployable controller. Unresolved horizon, cadence, fallback and
model-richness choices are experiments, not matters for a four-model vote.
