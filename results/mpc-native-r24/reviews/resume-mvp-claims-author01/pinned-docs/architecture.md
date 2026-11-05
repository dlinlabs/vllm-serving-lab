# Serving and control architecture

Status: implemented serving/instrumentation and CPU research components;
the complete native MPC loop is **not yet qualified**.

```mermaid
flowchart LR
    W[Fixed offered workload] --> G[Gateway admission and reservation ledger]
    G -->|admitted request| V[vLLM inference engine]
    G -->|overload rejection| O[All-offered outcome log]
    V -->|stream and terminal outcome| G
    G --> T[Versioned lifecycle and control telemetry]
    V -. experimental native observations .-> T
    T --> Q[Clock, identity, freshness and coherence checks]
    Q --> M[Frozen prediction model and shared scenarios]
    M --> P[Bounded finite-horizon planner]
    P --> A[First-action actuator and application check]
    A -. native MPC integration not qualified .-> G
    T --> O
    O --> B[Offline audits and benchmark table]
```

Solid arrows describe serving flow or implemented component interfaces, not
proof that the full diagram has run as an integrated controller. The dashed
actuator edge is the principal missing native closed-loop evidence.

## Ownership and extension points

| Layer | Responsibility | Repository evidence |
|---|---|---|
| vLLM | Model execution, native batching and cache management. | External backend; do not claim these algorithms as project contributions. |
| Gateway | Reservation accounting, admission/rejection, streaming lifecycle and cleanup. | [`gateway.py`](../gateway.py) |
| Measurement | Scheduled arrivals, actual applications, request/state/diagnostic records. | [`trace_control.py`](../trace_control.py), [`telemetry.py`](../telemetry.py), E1 in [results](benchmark_results.md) |
| Prediction/planning | Coherent scenario rollouts and policy selection. | [`mpc_lab/model.py`](../mpc_lab/model.py), [`mpc_lab/controller.py`](../mpc_lab/controller.py), E2 frozen planner |
| Runtime/actuation | Observation guards, pending-command handling and first-action coordination. | [`mpc_lab/runtime.py`](../mpc_lab/runtime.py), [`mpc_lab/actuator.py`](../mpc_lab/actuator.py); native qualification pending |
| Evaluation | All-offered scoring, failure retention and baseline accounting. | [`mpc_lab/benchmark.py`](../mpc_lab/benchmark.py), accepted research evidence; no completed current native comparison |

The public library and frozen team16 prototypes are separate versions. Do not
combine their settings, interfaces or test totals into one allegedly deployed
implementation.

## State and actuator semantics

Gateway observations include immutable request reservation costs, reserved work
R, awaiting-first-content Q, streaming D, ages and arrival/outcome history.
These are gateway state/proxies, **not observed remaining GPU work**.
Native scheduler/copy-ready information needs its own source-clock and
availability qualification.

The controlled value is a gateway weighted reservation budget u. Admission
requires `R + request_cost <= u`; terminal cleanup releases that request's
reservation. A lower budget does not evict already admitted work. An increase
does not itself submit a request. Gateway release is not proof that cancelled
backend computation stopped.

## Prediction and optimization

The reference specification maximizes expected completed SLO-good request count
over a fixed offered cohort using a common scenario panel. Rejected, failed or
unresolved requests receive no positive success credit. Candidate plans use the
same model, objective, resource constraints and scoring population.

Fixed holds one budget. Strong OneMove can choose the best permitted single
transition, including a delayed transition; it is not a deliberately weak
one-step forecast. MPC can consider multiple future actions but executes only
the first, then reobserves. The accepted frozen planner's exact limits belong to
its own PLAN, not a universal horizon setting.

See the [mathematical specification](../experiments/mpc-research/mpc-mathematical-specification.md)
and E2 for configuration-specific details. Native14 is a separate failed-
qualification model; a source-faithful formula is not evidence of predictive
accuracy or calibrated uncertainty.

## Runtime contracts and unresolved failure paths

Reject stale, incoherent, wrong-clock or wrong-version observations. Bound
search and account for observation, serialization, IPC, solver, logging and
application delay. An unresolved command timeout means unknown application,
not cancellation; do not send speculative replacements.

Process exit must include owned roots/leaves/readers/helpers and a truthful
final receipt. Pre-return metadata is not proof of completed cleanup. D2-F1
currently shows how a pending alarm or failed cleanup-anchor publication can
skip cleanup in one deployment composition. That launcher is not live-eligible
until its repair is accepted.

Admission/resource bounds are not end-to-end latency guarantees, universal
queueing stability, physical KV safety or production reliability certification.
