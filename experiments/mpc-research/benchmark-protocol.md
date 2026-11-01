# MPC evidence and benchmark protocol

Source freeze: `8e92371`. This is a preregistration framework, not GPU results.
Architecture-dependent settings must be frozen after round 3 and before the
first policy benchmark. A changed setting starts a new development experiment;
it never retroactively changes the final-test contract.

## Evidence layers

| Layer | Permitted conclusion |
|---|---|
| Deterministic unit/property tests | Implementation satisfies the tested invariants |
| CPU closed-loop simulation | Behavior under the explicitly declared synthetic plant |
| CPU HTTP/mock integration | Transport, observation and actuator semantics under a mock |
| Native GPU identification/qualification | Measured dynamics and instrumentation on the pinned runtime |
| Held-out HTTP policy experiment | Performance of frozen policies under the measured offered workloads |

Neither simulation nor a forecast trained on historical data identifies a live
policy advantage. An HTTP mock is not a GPU/HTTP serving acceptance run.

## Outcomes and denominators

Primary goodput is the number of eventually successful complete streams with
scheduled-arrival TTFT <= 0.5s, divided by the full offered-arrival window in
seconds. Drain requests after that window and preserve their final outcomes.
An unfinished, failed, cancelled or rejected request is not successful goodput.
Do not terminate a trial early because a policy admits fewer requests.

Report, for every arm: offered and accepted count/rate, complete-stream count,
goodput, all-offered SLO attainment, rejection and failure rates; TTFT p50/p95/p99;
per-request TPOT and inter-token-gap distributions with their precise definitions;
E2E latency; observed GPU utilization; controller CPU, solver/decision latency;
cap changes, reversals, recovery time and outstanding cost after cap reduction.
Successful-request latency is conditional on success and is never the sole score.
Keep token-gap-weighted and request-weighted TPOT summaries separate.

The existing primary contract defines no TPOT or E2E deadline. Do not invent one
and report it as an established SLO. If a new joint SLO is needed, freeze it,
its rationale and an additional all-offered denominator before new experiments.

## Comparators and fairness

Include every qualified fixed cost cap in the development action set; select a
best fixed baseline only on development data and retain its per-cap results.
Include a predeclared hysteresis/threshold policy and a queue/proxy feedback
policy when their actual observations exist. Include same-model horizon-one
greedy control to test whether lookahead, rather than a better predictor, helps.
Compare the final MPC candidate with these under identical action bounds.

No policy may read future trace arrivals, realized output lengths, eventual
completion, future budget-application timestamps or EngineCore fields unavailable
at its decision time. Client-declared output caps are known; actual EOS is not.
Give every arm the same causal arrival estimator and history availability.

Freeze GPU model/UUID, driver, CUDA, vLLM, Torch, tokenizer/model revisions,
gateway source/configuration, cost semantics, one-worker operation, scheduler
B2048/context8192, caching/speculation, offered payloads/times, warmup and drain
rules. Compare no protected32 raw bridge with cost-aware C32 as if identical.
Keep CPU affinity, monitoring frequency and logging equal across arms.

Use seeded randomized blocks, with an entire independently reset episode as the
unit of replication. Pair arms within workload/seed blocks and randomize order.
Requests or five-second windows inside one run are not independent replicates.
Record thermal/power/clock drift and counterbalance order; never remove a slow
run after observing its score. Infrastructure-invalid episodes remain in the
ledger with reasons and any predeclared replacement rule.

## Inference and stopping

Separate training, uncertainty calibration, development validation and a final
locked test. Existing trace C remains unopened during this work. Do not repeatedly
tune against a final set and continue calling it held out.

Begin with a separately labeled variance pilot. Use its episode-level paired
variance to preregister repetitions/power for the minimum operationally useful
goodput difference. That difference and admissible rejection/latency regressions
require an explicit operational contract; they cannot be inferred from whichever
result makes MPC win. If that contract is absent, report a Pareto frontier and
uncertainty, not an operational superiority claim.

Report paired episode-level mean differences and uncertainty intervals, with
resampling at the independent episode/block level. Predeclare comparison
families and simultaneous inference if selecting among multiple MPC candidates.
Do not treat an interval containing zero as proof of equivalence; equivalence
needs a preregistered margin and adequate precision.

The default scientific negative outcome is retained when fixed/feedback/greedy
control dominates, the MPC effect remains unresolved, or model/solver overhead
erases its benefit. Identify workload conditions rather than changing the metric,
dropping adverse trials or relaxing timing gates.

## Paid-resource gate and staged budget

No paid resource is launched by this protocol. CPU work must complete first.
The existing rented host is not assumed to be available or authorized now.

First request a bounded qualification/identification pilot, not an open-ended
full benchmark: one RTX4090, up to 2.5 billable hours including setup/download,
instrumentation checks, and at most 48 development episodes averaging 90 seconds
including warmup/drain. Stop before a new episode if the remaining ceiling cannot
cover it. Preserve early-stop/incomplete status; no held final policy score is
promised by this pilot.

At rate R per GPU-hour, the compute ceiling is 2.5R, plus explicitly quoted
storage/network/minimum-billing charges. A dollar ceiling and the provider quote
must be approved before launch; no assumed public rental price is an authorization.
Long outputs or slow drain reduce the episode count, not the integrity gates.
After the pilot, provide an observed-duration/variance-based budget for the
independently authorized, repeated native HTTP policy comparison.
