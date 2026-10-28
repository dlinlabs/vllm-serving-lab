# Bounded route from controlled latency to a minimal admission-budget MPC

The current evidence includes one passing fixed B2048 controlled-arrival
experiment, with identical F0/entry forecasts, and a later native attempt
rejected at TRAIN first-commit readiness before held/HTTP. The latter has a
planned case whose repeat budgets cannot share one fixed prediction; matching
recorded release scheduler state omits pending readiness/debt and physical
CPU/GPU state. Cross-host robustness is unestablished. Neither run
shows that changing an action improves the outcome. This is the remaining gap
to MPC, and HTTP validation alone does not close it.

## Keep the control variables distinct

- B=2048 is the vLLM maximum batched-token budget, fixed in the present model.
- C is the weighted admission-cost budget (`MAX_ADMITTED_COST`) in the
  gateway cost_aware mode, not a request count. Start with the existing studied
  C=16 and C=32 actions; do not add an unconstrained budget search yet.
- Protected32 in the HTTP bridge is a separate maximum concurrent-request
  limit. It does not exercise the cost_aware C16/C32 admission policy.
- D16 is a background decoder population, not an admission cap.

A continuation observed inside EngineCore before ADD occurs after upstream
admission. That observation cannot retroactively justify the earlier gateway
admission decision. A controller needs a separately qualified observable state
feed and its age at the actual cost-aware admission decision point. The existing
authenticated gateway budget-update endpoint already provides actuation; it
does not establish that the model predicts the consequences of those updates. Known client-to-cut time is not
a prediction of future ingress or input-queue waiting.

## Three completion gates

| Gate | Smallest useful result | Stop condition / next decision |
|---|---|---|
| 1. HTTP state bridge | Same-host controlled prerequisite; prospectively sealed pre-ADD continuation and client TTFT; all raw joins/work/overhead gates hold | If warmups fail, preserve one diagnostic archive and identify snapshot, pending-debt, forecast-age or transport failure before another rental; do not refit per held case |
| 2. Action validity | Under matched offered trace prefixes and qualified states, C16/C32 outcome ranking and drain transients are predicted on fresh held episodes | If one fixed cap wins throughout, or ranking differences cannot be distinguished from repeated-run variation, report that evidence; do not assert an adaptive/MPC advantage |
| 3. Minimal closed loop | A predeclared finite-horizon controller beats replicated fixed16/fixed32 and a myopic/hysteresis baseline on fresh offered traces | Retain all rejections/failures and transition windows; no MPC acceptance from offline oracle or simulator-only success |

CPU HTTP integration and a separate conditional state-envelope prototype are
implemented. The envelope permits independently varying future service times
and explicit unavailable/wide-interval fallback; it has no native coverage or
actuation qualification. See prefill-state-envelope-v1.md. The old failed TRAIN
point gate is unchanged, and no native shadow collector is added by that CPU
layer. Action validity can be designed and its CPU runner built before demanding
production-grade HTTP accuracy; no additional open-ended calibration stage is
inserted by this plan.
Native action and closed-loop claims still require their own frozen experiments.

## Gate 2: decision-focused experiment

Reuse the pinned model, engine B2048, same GPU and existing C16/C32 study.
Qualify the cost_aware request/state path explicitly; the protected raw-token
HTTP bridge cannot silently substitute for it. Feed
identical offered request identities/arrival schedules from a common drained
initial condition. Use a common offered prefix up to each decision so the
initial action state is actually comparable. Record native state correspondence;
identical trace timestamps alone do not establish identical states. Replicate
and balance action/run order; the old single fixed reference runs are not
matched causal controls.

Predict each action using only information available at the decision boundary:
current requests/progress, current admitted cost, known queued/offered state
and a TRAIN-derived arrival/output model. The fixed per-request admission-cost
formula must use online-known inputs and remain frozen; C16 does not mean 16
requests.
Forced128/256 controlled requests establish a narrow measurement contract;
actual future EOS lengths and future recorded Azure arrivals must not become
online controller features. An offline oracle may be reported only as an upper
bound. State/arrival uncertainty must lead to a predeclared fallback, not held
parameter tuning.

On a C32-to-C16 reduction, existing requests continue. A new request is
admitted only when current_admitted_cost + request_cost <= C; otherwise the
existing cost-aware policy rejects it. Outstanding cost is released as existing
requests finish. Do not simulate an immediate population shrink, cancellation
or an invented waiting queue. Validate action ranking, ranking gaps versus
replicate variation and regret, plus initial transition and later intervals.
The objective remains successful complete streams with scheduled-arrival
TTFT <=0.5s divided by the entire offered-arrival window, following the existing
low-cap study. Rejected/failed offers and early transition time remain in the
denominator. Successful-only p99 is not a substitute for this objective.

Freeze episode splits, uncertainty/fallback, ranking/gain requirements and
analysis before collecting new outcomes. State-dependent action reversals and
held ranking evidence are needed to justify adaptation; an accurate fixed-cap
latency fit does not provide them.

## Gate 3: minimal controller scope

Once action validity passes, begin with only C16/C32 and a short discrete
horizon, not a general optimizer. Freeze control period, horizon, switching
penalty, state-age and compute budgets, overload constraints and fallback from
TRAIN/action-development evidence before closed-loop held trials. At each
step, predict candidate admission/drain sequences, choose the feasible sequence
with the frozen SLO-goodput objective, execute only its first action, then
observe and replan. This is the receding-horizon part that makes it MPC.

Compare replicated fixed16, fixed32, a myopic/hysteresis policy and the MPC on
matched offered held traces. Keep natural output/arrival uncertainty, all
rejections/failures, control lag, forecast overhead and drain transients. A
controller that cannot outperform the best fixed cap in its predeclared scope
is a useful negative result, not a reason to keep expanding calibration until
one metric wins. No live controller or adaptive-cap acceptance is implemented
by the HTTP integration itself.
