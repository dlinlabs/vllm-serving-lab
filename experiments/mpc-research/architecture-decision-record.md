# ADR: bounded gateway-admission MPC challenger, default off

Source baseline: `8e92371`. Decision follows four independent proposals, four
adversarial critiques and four Round3 revisions. Agreement is not runtime evidence.

Implement a shared immutable marked ledger, explicit reservation accounting,
a compact event/lifecycle CPU surrogate, small discrete planning and a separate
guarded transport. Use only the weighted admission budget; do not claim gateway
control over vLLM token scheduling, KV allocation or physical concurrency.
The reduced model is an engineering challenger, not an identified GPU law.
The more detailed kernel-inspired model remains an alternative, not a prerequisite.

Keep prediction/evaluation window fixed at5s while varying future control
freedom over1/3/5 one-second slots. Start with3 slots,8 plans and common scenarios.
After the control horizon use a declared fixed backup, with continuing arrivals.
Score unresolved old plus fixed-window new cohorts exactly once, including
late failure. Report unresolved reward bounds; initial selection maximizes
the resolved lower score, so long unresolved streams can legitimately bias it.
Tail-length sensitivity and native identification must test this approximation.
No hard terminal-drain, KV chance, SLO-safety or universal stability claim exists.

Compare best development fixed cap, same-state backlog feedback, same-model
one-control-move rollout and multistep planning. A separate genuinely myopic
prediction-horizon ablation changes the objective and must be labeled as such.
No weak baseline, future workload tape, oracle EOS or held-out tuning is allowed.

The initial CPU event model uses request phases and remaining-work scenarios,
not independent cost/queue heads. All costs remain reserved until terminal;
cap reductions do not evict or clip reservations. Model coefficients and
cost/tokenizer semantics, source implementation and calibration are sealed.
Forecasts from incomplete, stale, cross-clock or mismatched state fall back
observably. A small state vector is distinct from the ledger needed for correctness.

The CLI is shadow-only. A separately named, explicitly enabled server endpoint
must enforce run/clock/cost/model identity, operation identity, CAS and expiry
atomically. Sending fields to the legacy endpoint is not capability negotiation.
Timeout means unknown application; a GET is not a fence. The old budget endpoint
and default admission behavior stay unchanged.

Rejected: unsupported physical ceilings, mean-release pathwise bounds,
unobservable KV risk certification, hard terminal feasibility, independent
cost-out regressions, compulsory per-decision significance and online RLS before
causal identification. Nonlinear/robust/chance/learning MPC remain candidates,
not implemented superiority claims. Fixed fallback16 versus32, useful horizon,
residual-lifetime sufficiency, natural EOS, intervention calibration and native
performance remain unresolved. CPU work must finish before a separately
authorized native identification/HTTP/GPU experiment.
