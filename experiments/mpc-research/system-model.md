# First-principles serving model and observability

Source freeze: `8e92371`; implementation changes are tracked on the local
research branch. A state is useful only if it changes a conditional prediction,
is available before acting, and improves a held-out ablation.

The physical plant is hybrid: ordered admission/first-content/terminal events
couple continuous remaining work and discrete active membership. The gateway
actuator changes future membership, not previously admitted work or GPU speed.
Exact ledger L and an approximate latent-work belief b form controller state:

    x_k = (L_k, b_k, applied_budget, version, pending_command, arrival_history).
    x_{k+1} = f(x_k, u_k, w_k; theta), y_j = h_j(x(s_j)) + noise_j.

Observations have distinct source and availability times. The control information
set includes only values available by the decision. Static code/cost/calibration
identity and clock domain are feasibility conditions, not learned features.
The reduced regressor uses W_prompt, D and ages only as supported by ablation;
the exact marked ledger is retained for release/outcome correctness.

| Candidate | Why it might predict | Online availability / noise | Inclusion test and current decision |
|---|---|---|---|
| Running requests | contention and reservation persistence | gateway active set exact locally; not exact engine-running set | retain ledger, ablate count as learned feature |
| Waiting queue | waiting work and overload | gateway has no waiting queue; awaiting-first proxy is not engine queue | no invented queue state; qualify engine observer before adding |
| KV occupancy | allocation pressure/preemption | no qualified current-host feed; allocator/prefix effects unknown | exclude hard/chance constraint; test only with qualified runtime sensor |
| GPU memory pressure | OOM/headroom | no qualified online sensor or mapping | report unknown, not zero; separate future allocation model |
| Prefill backlog | strong factual TTFT signal | original prompt sum before observed first; remaining tokens latent | retain W_prompt; AUC/MAE reanalysis is development evidence |
| Decode backlog | interference and future holding time | streaming count observable; remaining EOS length unknown | retain phase/age bookkeeping; test D incrementally |
| Remaining work | release forecast/opportunity cost | latent, especially natural EOS; output maximum is only a bound | scenario belief; residual-lifetime calibration remains required |
| Arrival rate | future admission pressure | offered-event history possible; active count alone insufficient | use past offered history or explicit prior, never future tape |
| Workload composition | token service demand/cost distribution | prompt estimate/output maximum at arrival; tokenization fallback | bind estimator identity; estimate from past marks, test shift |
| Prefill/decode service rates | work-to-time dynamics | cannot infer causal rate from cost coefficients or client TTFT alone | frozen surrogate parameters until intervention identification |
| Throughput saturation | nonlinear service degradation | historical conditional curves, not physical ceilings | diagnose supported regimes; no extrapolated universal bound |
| Recent TTFT | model drift and backlog risk | gateway first-minus-receipt differs from client first-minus-schedule | separate labels, availability and censored streams |
| Recent TPOT | decode slowdown | SSE chunk is not necessarily one token; terminal labels delayed | no invented token-level sensor; qualify token timing |
| E2E latency | residence and reservation persistence | complete-stream success/time arrives late; cancellation may orphan work | retain terminal outcomes and censoring, not first-hit proxy |
| Prediction error | drift/OOD and fallback | only factual executed-policy error is observed | use declared episode scores; no all-plan counterfactual residual oracle |
| Reserved cost R | exact admission feasibility | exact serialized reservation; source defaults are not GPU seconds | mandatory accounting; weak factual TTFT feature alone |
| Oldest awaiting age | imminent miss / hidden residual progress | receipt age historically; new marked admission age needed | distinguish clocks/phase entry and ablate conditional residual prediction |

Noise/latency magnitudes are not established for new native observations.
Existing telemetry sampled about1s does not qualify a20Hz controller. CPU fixtures
provide exact ordered observations only inside their declared synthetic scope.
Current-state audit and identification report separate measured numbers from
unknown sensor delay, orphan work, EOS and action effects.

## Actuators and disturbances

The current gateway controls one positive weighted budget and immediate
accept/reject on arrival. Lowering it never evicts existing streams. Fixed
request-count concurrency belongs to a different protected path; it is not the
same unit. Queue capacity/priority require a new gateway queue and change the
plant. Prefill token budget, decode scheduling, KV allocation and inner
prefill/decode prioritization require vLLM runtime changes and fresh observation/
actuation qualification. They are not silently exposed by this MPC prototype.

Disturbances include marked arrivals, bursts, prompt/output distribution shift,
hidden EOS, service-rate drift, transport delay, failures and backend cancellation.
Offers, marks and service uncertainty are sampled jointly within each scenario.
Actions are the same across scenarios before their observations are revealed.
No adversarial-robust or probabilistic guarantee follows merely from sampling.

## Feasibility and stability

Positive bounded budgets and nonnegative whole-cost releases imply
R<=max(R_initial,u_max); c_min>0 also bounds reserved membership. This is a
gateway reservation proof only. There is no queue-stability proof for upstream
clients, physical-KV bound, terminal-drain theorem, Lyapunov guarantee or SLO
safety theorem. Reject-all has zero goodput and cannot improve the primary
all-offered successful-goodput objective.
