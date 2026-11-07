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
CPU/shadow planning exists; current evidence does not establish an integrated
native MPC deployment. A future complete-loop claim needs real application and
reobservation receipts.

## What is the objective?

The reference specification maximizes expected eventual successful requests
meeting the TTFT SLO over a fixed offered cohort. Rejected and failed requests
cannot improve the score by disappearing. Tail/unresolved credit and tie
breaking must be fixed across policies. Settings are version-specific.

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

## What do the existing measurements establish?

E1 provides audited historical native serving measurements under scheduled
budgets; E2/E3 provide bounded CPU component evidence. E4 is a partial native
observation diagnostic. None establishes a fair native MPC win. A/B were on
different hosts and are not a causal policy comparison.

## What if MPC loses?

Publish the actual table and scope, then explain model error, overhead, control
authority or workload sensitivity only where evidence supports the explanation.
The accounting, instrumentation, bounded runtime and reproducibility remain
engineering contributions. A negative result is better than an inflated claim.

## What would make this Resume Ready?

The P0 gate in [the project summary](resume_project_summary.md): a reproducible
real loop, safe failure paths, comparable descriptive baseline results and an
audited claim-to-trace/command/config/commit ledger. Superiority, production
deployment and a universal SLO theorem are not prerequisites.
