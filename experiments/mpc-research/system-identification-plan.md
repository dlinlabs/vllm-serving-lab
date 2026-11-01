# Causal serving-system identification plan

Source freeze: `8e92371`. Proposed experiments below have not been executed.
This plan does not choose an MPC family before independent review.

## Existing information and its limits

Historical fixed-cap and cap-switch traces contain some action variation, but
not randomized actions at every relevant state/workload. A factual forecast error
does not identify the sign of a same-state cap intervention. Closed-loop cap
selection can confound workload and action; adding more correlated features does
not repair missing treatment support.

The gateway reservation cost is based on a legacy RTX3090 token-cost fit
(`gateway.py:29-37`), not a direct measurement of RTX4090 GPU work or KV memory.
Its units and cap semantics must be preserved while evaluating its predictive
adequacy on the current hardware. C32 never means 32 requests.

The cost-aware gateway rejects immediately rather than maintaining an admission
waiting queue. `awaiting_first_content` is an admitted-request proxy, not measured
vLLM queue length; `streaming_requests` is not a token-work counter.
`telemetry.py:350-357` supplies these causal proxies. EngineCore observations
after admission do not supply a pre-admission oracle.

## Online feature qualification

For each retained feature record: producer/source line, physical units, sampling
and arrival timestamps, clock domain, update frequency, missingness, measurement
age, noise/repeatability, and whether it is available before the decision.

| Candidate | Availability and required test |
|---|---|
| Active/running count | Gateway active/in-flight counts exist; compare lifecycle conservation and distinguish engine-running from gateway-active |
| Waiting/prefill backlog | Awaiting-first count/oldest age are online proxies; distinguish buffering, transport and engine wait |
| Decode backlog/remaining work | Streaming count is online; output caps are known but remaining EOS work is latent; test conditional survival estimates without future lengths |
| Reserved cost/budget | Available with budget version; verify admission and non-evicting cap-reduction conservation |
| Arrival rate/composition | Use received-offer history, not accepted-only history; prompt/cap features require an online observation contract |
| KV/memory pressure | Only retain a qualified timestamped runtime channel; do not reconstruct online features from completed offline captures |
| Service rate/saturation | Infer causally from completed observations; validate workload dependence, delay and closed-loop confounding |
| TTFT/TPOT/E2E history | Outcome observations are delayed and censored by active requests; successful-only summaries have selection bias |
| Prediction-error history | Only predictions whose labels have arrived enter the state; never score unfinished predictions as known residuals |

Use nested, episode-separated ablations: a variable stays only if it improves
multi-step prediction or constraint discrimination beyond uncertainty and
measurement overhead. Test counts, workload summaries, age distributions and
pressure independently; do not keep both an algebraically redundant aggregate
and all of its components without justification.

## Interventions and safe coverage

First qualify fixed endpoints and cap transitions using the real
`/admin/budget` actuator (`gateway.py:588-630`). Existing reservations survive
cap decreases. Preserve actual application time/version and distinguish intent,
HTTP acknowledgement and observed application. Unacknowledged actions are not
silently treated as applied.

Within an approved native pilot, use a balanced seeded screening design over:

- Qualified cap endpoints and both transition directions; an intermediate cap is
  added to the online action set only after its own qualification.
- Offered rates below, near and above measured saturation.
- Short/long and mixed prompt populations with known prepared prompt counts.
- Different declared output caps and natural EOS, separating prefill-heavy and
  decode-heavy composition.
- Low/high observed KV pressure where the runtime channel is qualified.
- Bursts, recovery, workload mixture switches and output-distribution shifts.

The bounded pilot cannot estimate every interaction in this factorial space.
Choose a rank/conditioning-aware screening subset on development assumptions,
publish its coverage matrix before collection, and label uncovered interactions.
Randomize feasible actions/transition times within blocks independently of
unobserved future arrivals. Include cooldown/reset and repeated reference cells.
Safety limits may truncate exploration; that truncation limits identifiability
rather than justifying extrapolation.

Log intended actions, applied actions, states, offers, all terminal outcomes,
feature/version fingerprints and timing/integrity errors. Retain rejected offers.
Do not relax the existing 5ms shadow-posting scope or discard its adverse samples.

## Splits and model assessment

Assign whole episodes/seeds/workload families to training, calibration and
validation before fitting. Reserve independent final testing, including trace C,
without opening it. Gap adjacent time segments when shared histories create
leakage; fit normalization, feature selection and model hyperparameters on
training/development folds only.

Assess one-step error and free-running multi-step error; evolve endogenous
history variables rather than freezing their future values. Assess counts/work
conservation, cap-draining transients, action ranking and regret on supported
interventions, TTFT/TPOT-tail risk, violation discrimination, residual coverage,
and sensitivity to service-rate/output-length misspecification.

Calibration units must match the intended joint claim. Marginal per-step coverage
does not imply simultaneous horizon or closed-loop policy coverage. Use blocked
calibration appropriate to dependent serving episodes and state its assumptions.
Exchangeability, action support and distribution stability cannot be asserted
under arbitrary feedback or OOD workloads. Until native validation establishes
transfer, uncertainty sets are engineering stress scenarios, not certified
physical bounds or formal SLO guarantees.

OOD evaluation includes unseen composition, arrival bursts, service degradation,
missing/stale observations, clock changes, tokenizer/cost/model mismatch and
solver/actuator failure. The correct result can be an observable fallback rather
than a predicted advantage. A model that cannot distinguish cap actions cannot
justify MPC merely by beating a trailing predictor on factual data.
