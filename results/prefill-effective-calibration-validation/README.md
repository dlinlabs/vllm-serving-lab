# Effective endpoint calibration implementation and self-review

This change reaches the first GPU-dependent qualification step. A separate
138-trial zero-offset B2048 pilot now collects matched OFF/LIGHT endpoints,
fits four effective service parameters from training only, freezes forecasts
before fresh held trials, evaluates preregistered pilot gates and retains a
self-contained evidence archive. It does not qualify the21 physical stage costs,
activate admission control or establish MPC benefit.

Final local validation: **329 related CPU tests passed in45.44s**, zero failures,
errors or skips; JUnit is retained here. There are36 added cases relative to the
prior293-test related set. Tests cover own-rollout synthetic recovery, held-label
poison isolation, training-label isolation at evaluation, missing/duplicate/
nonfinite labels, sensitivity/bound/convergence/multistart rejection, observer/
repeat/endpoint/structure rejection, frozen/runtime/prediction integrity,
endpoint chronology, independent background lifetime and pending-output checks,
fixed split/content plan, endpoint mode isolation, freeze-before-held failure
retention, host manifest/archive routing and setup pin/import/command boundaries.
Python compilation and whitespace checks pass.

The first focused test attempt ran34 pass/1 fail: the synthetic generator reused
endpoint dictionaries across trials, so a deliberately corrupted LIGHT fixture
also changed its OFF counterpart. Each synthetic trial now owns a deep copy.
No native observation, measurement threshold or numerical fitting rule was
changed to resolve it. The subsequent focused37-case run, broader325-case run
and329-case pre-final run passed. Final329 includes explicit synthetic/native
provenance and the latest registered source hashes.

The standalone final demo uses only the previously archived runtime metadata
(scope and dispatcher); all labels are newly generated synthetic endpoints.
Its report/freeze are retained under synthetic-report.json and
synthetic-freeze.json. Maximum parameter error is1.2401538129758194e-14s and the normalized local
sensitivity condition is3.839305674041866. Numerical recovery is not measured
GPU performance.
No prior light/deep held-composition timings enter fitting or parameter choice.

## Self-review corrections and limits

- The physical21-component model remains unidentifiable from the existing stage
  intervals. The new candidate explicitly collapses CPU/GPU service into one
  effective lane. Held failures can reject this approximation; coefficients are
  never described as exclusive GPU forward or copy measurements.
- Four coefficients have an explicit fixed name order. JSON key sorting cannot
  swap coefficients. Only training rows enter fitting, relative scales and local
  sensitivity checks. Held trials begin only after freeze and forecast files exist.
- LIGHT work signatures are independently compared with the model's own ordered
  work/positions. Existing native scheduler/dispatcher and request-progress gates
  are preserved. The extractor also recomputes background lifetime/barrier and
  foreground completion boundaries instead of trusting an empty errors list.
- OFF has residual private-control overhead. The paired gate qualifies only
  incremental LIGHT overhead; no production-overhead or HTTP-latency claim.
- Distinct manifests and an explicit endpoint environment flag isolate this
  pilot. Normal launches force that flag off. No GPU instrumentation wrapper,
  scheduler, full-suite rule, serial overlap guard, gateway or cap policy changed.
- Source/CPU acceptance, original JUnit/log, model/token manifest, raw captures,
  endpoints, runtime, frozen candidate and held report survive in the downloadable
  archive. Offline audit checks consistency and recomputes results. Digests are
  not signatures and do not prove wall-clock collection order against forgery.
- Training/held phase ordering is explicit; later training-cell references test
  drift but do not prove stationarity. Rank/condition and multistart agreement are
  local numerical safeguards, not global identification or statistical power.
- This pilot excludes delayed sentinels, changing batch budgets, frontend/gateway
  overhead, broader mode/support claims and production workload validation. Even
  candidate_passed=true leaves model/physical/full-experiment/cap/MPC acceptance
  false. Fresh delayed-arrival and HTTP qualification must precede cap comparison.

Local Torch/vLLM are absent and the native PID/procfs resource probe is unavailable.
No native full acceptance or GPU run was performed or bypassed. The next action
is a fresh RTX4090/Python3.12 host, the documented preparation helper, then the
single calibrate command. Actual native startup, observation quality, parameter
identifiability and held prediction accuracy can still fail. Every failed attempt
is retained; there is no automatic retry or post-result threshold adjustment.
