# Astra controller implementation evidence

Assignment: `pr-mpc-control`. Scope: `mpc_lab/controller.py`,
`tests/test_mpc_controller.py`, and this note. The owner ADR and mathematical
specification govern this implementation; Round1/2/3 remain frozen.

## Implemented boundary

- `validate_snapshot(snapshot, *, now, clock_domain, expected_identity,
  max_age_s=2.0)` returns a diagnostic reason or `None`. Clock-domain equality
  precedes observed-time comparisons. It validates the declared complete ledger,
  identity, run/request IDs, integer versions/sequence/token counts, finite
  positive costs/budget, freshness, no future observations, receipt/admission/
  first-content ordering, scheduled-before-receipt ordering, and finite summed
  reservations. Numeric booleans are rejected. It does not impose `R <= budget`.
- `choose_plan(snapshot, parameters, scenarios, config, *, now=None,
  clock_domain=None, predictor=None)` checks parameter/configuration support,
  model/calibration/support/cost/source identity, exact cost-formula consistency
  to floating-point tolerance (`rel_tol=abs_tol=1e-12`), bounded active ledger,
  supplied scenario validity, and prediction bounds. The active-ledger count
  limit is checked before traversing marks. Default prediction lazily imports
  `mpc_lab.model.rollout(..., deadline=deadline)`. Without an explicit domain it
  obtains the local domain from the existing `prefill_lab.live_http.clock_domain`
  helper, never by trusting the snapshot's domain.
- `feedback_budget(snapshot, *, low=16, high=32, backlog_low=2000,
  backlog_high=6000)` uses only observed prompt tokens of requests still awaiting
  first content. At or above the high threshold it requests the low budget; at
  or below the low threshold it requests the high budget; within the band it
  retains the current budget bounded to `[low, high]`. Already-streaming prompts
  do not contribute. It needs no future arrivals or realized output lengths.

The feedback API validates structural coherence at the snapshot's own timestamp;
it does not independently qualify freshness, host identity, or model support.
Its caller must apply those checks before treating a baseline as actionable.
Likewise, `validate_snapshot` cannot reconstruct a `CostSpec` from an identity
hash: formula consistency is checked by `choose_plan`, which has the parameters.
The contracts contain no separate recorded reservation total or proof of atomic
snapshot capture. Checking a declared complete ledger is not proving that the
observation adapter actually captured every active request coherently.

## Bounded search and objective

Supported development settings are 1/3/5 one-second control slots, with a fixed
5-second evaluation window, 10-second tail, and 0.5-second TTFT threshold.
Unique finite actions and the continuation budget must be in model support.
All Cartesian action sequences are evaluated if their count fits `max_plans`;
otherwise the result is `plan_limit_exceeded`, not a biased partial search.
For `{16,32}` the complete counts are 2/8/32. The same supplied scenario objects
and order are reused for every candidate. No scenario-conditioned future actions
are introduced.

The selection criterion is mean resolved-good lower count. Mean upper count is
reported for the selected candidate, not used as success credit. Equal lower
scores prefer holding the current first action, then fewer changes from the
current budget and between optimized slots, then configured action order.
The mandated continuation switch is not included in this tie-break count.
The predictor owns the disjoint old/new cohort accounting and eventual-success
semantics; the controller validates finite ordered score bounds and their cohort
ceiling, rather than constructing a second lifecycle simulator.

Configuration, scenario count, offer count, active request count, rollout event
count, and wall-time limits have explicit fallback paths. Deadline checks occur
during offer validation, before and between rollouts, immediately after each
rollout, and before selection is returned; the absolute monotonic deadline is
also passed into the predictor. A late or invalid rollout discards the search,
including any earlier favorable candidate. `evaluated_plans` counts only plans
whose complete set of scenario evaluations finished.

Synchronous predictor execution is cooperative, not forcibly preempted. A
predictor that ignores the deadline can return late or fail to return; this
controller is not an OS real-time guarantee. No `asyncio.wait_for` wrapper is
used. Production timing and freshness must be rechecked by the separately
guarded actuator.

## Executed checks

Initial requested interpreter attempt:

```text
PYTHONDONTWRITEBYTECODE=1 /home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/pr32-mpc-py312/bin/python -m pytest -q tests/test_mpc_controller.py
/bin/bash: line 1: /home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/pr32-mpc-py312/bin/python: No such file or directory
exit 127
```

The system-interpreter attempt reported `Python 3.14.4` followed by
`/usr/bin/python3: No module named pytest` (exit 1). No dependency installation or
sandbox change was performed. Mayor then supplied the existing CPU environment.

Successful final focused command:

```text
PYTHONDONTWRITEBYTECODE=1 /home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/pr32-cpu-venv/bin/python -m pytest -q tests/test_mpc_controller.py
108 passed in 0.18s
exit 0
```

The same executable's `--version` reported `Python 3.14.4`. An earlier revision
passed 106 cases before adding explicit final-order tie cases and strengthening
the overflow/count-bound and equal-clock counterexamples. Test wall time is
test-runner evidence, not a planner benchmark or universal compute bound.

Coverage includes cross-boot timestamps with local `now` smaller/equal/larger;
all five identity components; cost formula/seal mismatches and cost overflow;
missing/incomplete/stale/future/invalid snapshots; strict bool/nonfinite/type and
phase checks; no eviction on budget reduction below reservations; exact plan
counts and shared draws; old-cohort terminal-success scoring through the injected
predictor contract; lower versus unresolved-upper scoring; scenario means;
deterministic tie ordering; invalid limits/scenarios/predictions; cooperative
deadline rejection including a partially evaluated search; lazy predictor
loading; default local clock mismatch; and same-state hysteresis.

`git diff --check` passed. Frozen report SHA-256 values remain:

```text
dbc98841590575cf7d2bbeceb7aa04a09cf79871b309580b0930503bf30754d0  astra-round1.md
af7814d3cb372b9c2c9baa897380ed45c1bf3de649a794438e717eec6918e939  astra-round2.md
9e8257e3cb51c5cc4ef108be29dbb96a003baac1b5613e3b2ebbd19eb6a7a24b  astra-round3.md
```

## Acceptance limits and integration

This is default-off CPU/shadow controller evidence using injected predictions.
It is not a test of Opus's actual model, Sol's runtime/transport, native HTTP,
GPU behavior, physical KV safety, terminal feasibility, SLO safety, calibrated
uncertainty, or policy superiority. Full CPU and supported host Python3.12
acceptance remain for Mayor; neither is claimed here.

Identity includes source hashes, so incorporating peer modules or the canonical
source-compatible `finite()` overflow correction changes the implementation
seal. Integrated snapshots must use freshly derived identities. Canonical
contracts and peer-owned files were not edited by this worker.

Every fallback is a diagnostic desired budget, with no selected plan or success
score. Even a finite configured fallback does not prove support when configuration
validation has failed, or prove delivery or safety. There are no admin writes,
capability claims, transport fencing, expiry/CAS guarantees, or live-actuation
authorization in this module. Those belong to the separate qualified command
path. Lowering a desired budget neither clips the observed reservation ledger
nor evicts an admitted request.
