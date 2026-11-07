# Interview defense

Use the [evidence ledger](benchmark_results.md) with each quantitative answer.
Current title: **LLM serving admission-control and MPC research prototype**.

## What did you build beyond vLLM?

An admission/accounting gateway, workload replay and lifecycle audits, control-
application timing instrumentation, and experimental model/planner/runtime
components. vLLM supplies native inference, batching and KV management; those
are not my claimed inventions.

## What does the controller actually control?

A positive weighted reservation budget at the gateway. It decides which new
requests may be admitted. It does not directly set native batch tokens, evict
active requests or guarantee the backend has stopped cancelled computation.

## What makes the design MPC rather than a threshold rule?

It evaluates future action sequences against a prediction model and constrained
objective, executes the first action and reobserves before replanning.
CPU-tested planner components exist; E2 accepts exact candidate ordering within
its full-action Fixed-reference scope, not runtime integration or native MPC.
A future complete-loop claim needs real application and reobservation receipts.
The current nominal continuation adds the E6 Fixed-timing correction and E8
Train-only scenario generation; those are component results, not native-loop
qualification.

## What is the objective?

The reference specification maximizes expected eventual successful requests
meeting the TTFT SLO over a fixed offered cohort. Rejected and failed requests
cannot improve the score by disappearing. Tail/unresolved credit and tie
breaking must be fixed across policies. Settings are version-specific.
The nominal MVP uses model-resolved successful TTFT counts as an uncalibrated
planning proxy; actual natural-SLO measurements use the separately pinned
scorer. They must not be described as the same validated objective.

## What are the state and prediction limitations?

Gateway reserved work, lifecycle phases and ages are observable proxies.
Remaining GPU work is partly latent. Source time differs from availability.
Native14 did not meet its sampling qualification; its uncertainty is not
calibrated. A source-faithful model or retrospective replay is not a validated
forecast or counterfactual action-ranking model.

## Why not require an elaborate Bayesian model for the MVP?

MPC needs an explicit predictive model, not necessarily that model family.
A simpler frozen experimental predictor may demonstrate the engineering loop
if separately scoped, checked and labelled. It cannot inherit qualification or
justify accuracy guarantees. Advanced mixing/quantiles belong in P2 unless
they answer a concrete P0 question.

## What is the difference from Fixed, FIFO and OneMove?

Fixed holds a development-selected budget. FIFO specifies service order and
needs an explicit admission/resource policy; otherwise it may duplicate Fixed.
Strong OneMove searches a best allowed single transition with the same model
and evaluation horizon. MPC permits more action changes and replans. Do not
weaken the baselines or give MPC privileged future information.

## How do you prevent evaluation leakage?

Freeze disjoint development and evaluation workload identities before scoring.
Fit models, construct scenario distributions, select Fixed and tune policies
on development data only. At each decision, scenarios may use observed history,
not the evaluation trace's future arrivals. The E1 Azure trace was development
data; replaying it is an in-sample engineering pilot, not held-out validation.
No currently published result establishes out-of-sample MPC performance.

## Why report rejection next to tail latency?

Successful-request latency can improve simply by rejecting hard requests.
All-offered SLO goodput, rejection, throughput and group outcomes expose that
trade-off. Conditional p99 is descriptive, not a service-level guarantee.

## Where can tail latency come from?

Admission waiting, backend scheduling/prefill/decode interference, transport,
event-loop stalls, GC and measurement delay. An overlapping pause is not
exclusive causal attribution. Separate scheduled-arrival, send, first-content,
terminal and application clocks.

## What failure handling matters?

Versioned state, expiry checks, bounded planning, unknown-command fencing,
exact reservation release and owned-process cleanup. D2-F1 shows why truthful
failure reporting alone is insufficient when cleanup can be skipped. The
affected composition remains blocked pending repair.
The selected nominal MVP uses a new isolated service-ownership path instead;
its actual cleanup and integration evidence is still pending.

## What do the existing measurements establish?

E1 provides audited historical native serving measurements under scheduled
budgets; E2/E3 provide bounded CPU component evidence. E4 is a partial native
observation diagnostic whose component coverage failed. None establishes a fair
native MPC win. A/B were on different hosts and are not a causal comparison.
Each offered 7,643 requests in 300 seconds: A rejected 5,667, B rejected 5,584;
572 and 736, respectively, met the TTFT SLO. Those deliberately varied,
scheduled-budget overload captures are identification data, not controller
performance.

## What if MPC loses?

Publish the actual table and scope, then explain model error, overhead, control
authority or workload sensitivity only where evidence supports the explanation.
The accounting, instrumentation, bounded runtime components and audit trail
remain engineering contributions; portable clean-checkout reproduction is
still a P0 item. A negative result is better than an inflated claim.

## Can this go on a resume before native MPC succeeds?

Yes, using deliverable A's [scoped bullets](resume_bullets.md): implemented
gateway/accounting, instrumentation, replay and CPU-tested planning, with their
evidence boundaries. Do not claim a native MPC deployment, measured speedup or
complete clean-checkout benchmark reproduction.

Deliverable B separately requires two consecutive timestamped real control
cycles, actual admission-behavior evidence, a Train-frozen strong Fixed
baseline, held-out comparison, reproduction and independent audit.
Superiority, production deployment and a universal SLO theorem are not
prerequisites. B's incompleteness does not block A or applications.
