# PR32 ground-truth audit

Audit date: 2026-10-29. Source: `8e92371f342be98a50682bf4e2ddbb014665498b`.
This is a new source audit, not the earlier four-model review of `13e783c`.

## Identity and boundaries

- GitHub PR32 is open/draft on `experiments/prefill-service-design-v1`.
- Stacked base is `6a3db67f6d57e67d6eafe02527b22937f2a2a56a`
  (`analysis/cap-miss-causes-v1`), not main.
- Fresh API enumeration found 47 commits and 15 issue comments; no formal
  review or inline review-comment rows were returned. The PR body still
  describes the older `13e783c` collector state and is not current evidence.
- The newest commit fixes probe transport and prepares static HTTP payload
  work before release. It explicitly claims no new native run.
- Research branch: `research/pr32-mpc-8e92371`; all work stays local.
- Native tracking: epic `pr-mpc`, audit `pr-mpc-audit`, caller-owned convoy
  `hq-cv-8tdlf`. The GPU/HTTP benchmark bead remains blocked for authorization.

## Reproduced issues: current disposition

| Issue | Current source and regression | Disposition |
|---|---|---|
| Same support hash despite changed cost semantics | `prefill_lab/state_envelope.py` seals specification, plan features, dependency hashes and generated costs; `tests/test_prefill_state_envelope.py:192-230` changes scales, features, costs and dependencies | Fixed in `b7e007b`; targeted regressions pass at current HEAD |
| Cross-boot chronology compared before domain | Domain guard precedes same-domain comparison; tests at `tests/test_prefill_state_envelope.py:156-189` check smaller/equal/larger uptime and actual CLI fallback artifacts | Fixed in `b7e007b`; targeted regressions pass |
| JSON boolean probe converted to integer and silently treated as OFF | Current producer uses integer marker; runtime validates bool/int 0/1 and role; wire/transport regressions exercise native ADD path | Source fix in `8e92371`; CPU checks pass, installed-vLLM schema check skipped locally |
| First reported shadow posting lags exceed 5ms | Current producer prepares hashes/static JSON before release; original gate unchanged | Mitigation implemented, native timing improvement unverified |
| Old flow rollout freezes endogenous histories | `results/flow-diagnosis-20261020/report.md` | Known modeling approximation, not repaired by the newest commits |

The original rejected run must not be rescored as passing. Its full raw archive
is not available locally: the shadow diagnosis in
`experiments/prefill-envelope-shadow-v1.md:139-165` is explicitly based on supplied
excerpts, not a fresh independent raw audit.

## Evidence ledger

| Layer | Evidence | What it does not establish |
|---|---|---|
| Current local CPU | 289 passed, 1 skipped in 12.58s, five selected test files; XML and log under `results/mpc-research-cpu/` | Full native CPU acceptance, GPU timing, HTTP accuracy or MPC benefit |
| Unfiltered baseline, Python 3.14 | 1718 passed, 4 failed, 2 skipped in 222.31s; `baseline-full-junit.xml` | Not a green baseline: four setup tests require Python 3.12; rerun in an isolated 3.12 environment without changing tests |
| Unfiltered baseline, Python 3.12.16 | 1723 passed, 1 skipped in 359.03s; `baseline-py312-junit.xml` | Native vLLM schema regression remains skipped; no GPU/HTTP/MPC acceptance |
| Historical native controlled `2ebb185` | Source-bound report: 240 trials; 144 held queue samples pass; F0/entry forecasts identical | Correction gain or cross-host robustness |
| Historical native TRAIN `efca3e4` | Source-bound reconstruction: 40 warmups + 48 TRAIN; 21/24 readiness pass, errors 14.225/11.242/21.795ms | Held controlled or HTTP acceptance; none were attempted |
| Shadow `b7e007b` | Supplied evidence: prediction chronology failures from skipped LIGHT forecasts and posting-lag failures | Native interval coverage/usefulness; absent predictions are not accuracy misses |
| Low-cap factual forecasting | 288 development windows: MAE 0.7290 versus trailing baseline 0.9676; cap32 optimistic bias +0.3163 req/s | Same-state counterfactual ranking, policy gain, independent final test |
| Dynamic admission/MPC | Existing open-loop budget schedules and actuator; no accepted live MPC | Deployment or performance superiority |

Primary historical sources:
`results/prefill-queue-confirmation-2ebb185/report.md`,
`results/prefill-entry-http-train-rejection-efca3e4/report.md`,
`results/low-cap-frozen-20261021/report.md`,
`results/state-model-20261021/report.md`,
`results/flow-diagnosis-20261020/report.md`.
Their archived evidence and reconstructions are historical claims with stated
provenance, not GPU experiments rerun in this audit.

## Real actuator, observations and modeling gaps

- `gateway.py:301-330` performs weighted-cost admission. A new request is
  rejected when reserved cost plus request cost exceeds the applied budget.
- `gateway.py:588-630` authenticates and applies budget changes, records a
  version and application time. Existing reservations survive cap reduction.
  Run one gateway worker. There is no cost-aware gateway waiting queue.
- Fixed protected concurrency and waiting limits are startup settings, not
  the dynamic cost-budget action. Raw-token completions require protected mode
  (`gateway.py:75-77`); protected32 is not C32.
- `/metrics` supplies gateway state and causal telemetry with sample time.
  This is not a qualified EngineCore progress/state channel.
- Engine B2048, weighted C16/C32, and background decoder population D16 are
  different quantities. Scheduler resource allocation/prioritization needs
  separate vLLM runtime work and cannot be claimed as an existing actuator.
- Pre-ADD snapshots occur after gateway admission; they cannot retrospectively
  justify that admission. Pending ready bits and observation age matter.
- Conditional state envelopes assume fixed membership, no later arrivals,
  forced outputs and valid service support. TRAIN extrema are not probability
  calibration or physical certainty. Native coverage remains unqualified.
- Prior flow models do not evolve all their endogenous history predictors.
  Good factual prediction does not identify intervention effects.
- Engine observations from completed offline captures are not automatically
  available to a live controller at the same timestamps.

## Environment

The current native protocol targets one RTX4090, Qwen/Qwen3-4B-Instruct-2507,
vLLM 0.30.0, Python 3.12, torch 2.13.0+cu130, B2048/context8192, prefix caching
and speculation disabled, TP1, preserved async/graph behavior. These are protocol
pins, not a claim that every historical capture has this exact environment.
See `experiments/prefill-service-v1.design.json` runtime and source pins.

This audit runs on WSL Linux, Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1,
pytest 9.2.0, pydantic 2.14.1, httpx 0.28.1 and fastapi 0.162.0.
Neither vLLM nor torch is installed in the selected CPU environment.
The installed-vLLM schema regression is therefore skipped, not passed.
No GPU resources or inference backend were started.

The unfiltered baseline exposed a toolchain mismatch, not a reason to weaken the
Python 3.12 setup contract. A separate CPython 3.12.16 environment was restored
using uv after the required interpreter was found missing. Its dependencies come
from the unchanged `requirements-cpu-tests.txt`; the complete rerun passed all
1723 runnable CPU/mock tests with the installed-vLLM case explicitly skipped.
The installed-vLLM requirement remains separate. Exact installed packages are
preserved in `results/mpc-research-cpu/python312-requirements.txt`.

## Native execution qualification

The four new native Polecats are obsidian/Opus, quartz/Astra, jasper/Gemini and
onyx/Sol. Actual Copilot usage records, not just profile names, identify all four
requested models. Round reports remain independent until all four are present.

Worker command sandboxes deny Host-loopback Dolt access even though Mayor can
read/write the same database. No sandbox was disabled or bypassed. Mayor delivered
the common offline assignment through native `gt nudge`; workers read local
source and write their own reports while Mayor performs authorized Beads updates.
Initial generic startup recipes attempted `gt done` before the assignment could
be loaded. Their generated local configuration auto-saves are not research
implementation and will not be integrated; no history is rewritten.

Convoy creation persisted nine `tracks` edges but initially stored issue type
`task`. A native `bd update --type convoy` repaired this record. `gt convoy status`
now finds the convoy, but displays 0/0 because the current dependency listing
omits targets in another database. Raw native dependency records retain all nine
PR32 IDs. Completion will be reconciled against the actual PR32 Beads, not this
misleading aggregate. No global daemon, model profile or sandbox-rig setting was
changed.

## Problem statement and acceptance separation

Determine whether an observable, causally available serving state supports
useful action-conditioned multi-step predictions and whether receding-horizon
control improves all-offered successful SLO goodput over qualified fixed and
simple adaptive policies on fresh repeated episodes.

Preserve the existing primary metric: eventually successful complete streams
with scheduled-arrival TTFT <=0.5s, divided by the full offered-arrival window.
Report rejection, failure, TPOT/ITL, E2E and transition behavior separately.
An E2E deadline is not invented where none was previously declared.

The research must separate (1) mathematical/model assumptions, (2) executable
CPU behavior, (3) native state/action qualification, and (4) independent policy
performance. No vote, synthetic win, oracle or passing CPU suite substitutes for
the latter two. Existing development data may be reused with explicit scope;
do not inspect or tune on the reserved final trace C.
