# Implementation and test evidence

PR source: `8e92371f342be98a50682bf4e2ddbb014665498b`.
Integrated implementation: `14b7cbd`; current-source manifest repair: `0af6e16`;
causal synthetic-grid correction/final source: `de36581`.
Branch: `research/pr32-mpc-8e92371`. This is a default-off research prototype,
not an accepted native controller.

## Module boundaries

| Module | Implemented contract |
|---|---|
| `mpc_lab/contracts.py` | Immutable marked state, complete cost/model/source identity and bounded configuration |
| `mpc_lab/model.py` | Event/lifecycle surrogate; full reservation residence, fixed-cohort success scoring and unresolved reward bounds |
| `mpc_lab/controller.py` | Clock-first validation, common-scenario finite-plan search, deadline fallback, hold/switch tie breaking and same-state feedback baseline |
| `gateway.py` | Default-off marked ledger and separate authenticated, same-lock run/clock/identity/CAS/expiry/idempotent budget endpoint |
| `mpc_lab/runtime.py` | GET-only shadow CLI, strict decoding, real model/controller calls and reproducibility metadata |
| `mpc_lab/actuator.py` | Separately opt-in operator-permitted client, atomic durable pending command, exclusive writer lock and conservative unknown-write reconciliation |
| `mpc_lab/benchmark.py` | Distinct synthetic truth plant, development-only baseline selection, fixed offered traces and paired exploratory reporting |

The gateway can change a weighted admission budget. It does not dynamically
change vLLM token scheduling, KV allocations, physical concurrency or a gateway
waiting queue. A lower budget never evicts existing reservations. Mathematical
and observational limits are in `mpc-mathematical-specification.md`.

The actuator permit's five native reference hashes are operator attestations,
not an automatic certificate. Tests use fictitious references and ASGI/mock
transports. There is no native approval or CLI live-write switch.

## Reproduced defects and fixes during integration

Mayor checks found and corrected scenario-generation bounds; incomplete
client-lifecycle credit; old-mark cost/seal inconsistency; a nonexistent planner
API call; and incompatible process-nonce clock domains. All are covered by
focused model/controller/runtime/integration regressions. Real decision calls,
not mock planner-only paths, are exercised in `tests/test_mpc_integration.py`.
Durable transport cases additionally cover unknown acknowledgment, restart,
CAS/expiry fencing, cross-domain uptime, writer exclusion and unsupported action.

Initial integrated Host Python3.12 checks:276 passed,20.82s
(`integrated-junit.xml`). After adding durable transport, its15 cases plus
two real-model integration cases passed in0.29s. Worker implementation notes
contain earlier Python3.14 results; those are not relabeled as Python3.12.

## Preserve the failed complete run

The first unfiltered integrated suite returned1973 passed,1 failed,1 skipped
in328.00s. Failure:
`test_committed_design_metadata_and_generator_outputs_are_current`.
The current pending-native design still referenced the old runner, packaging
and gateway source hashes after those files changed.

`0af6e16` changes only those three current-source entries. It does not change
workload, revision, thresholds, generated predictions, historical qualification
manifests or test assertions. The targeted design suite then passed30 tests.
Original failed XML/log bytes remain as
`results/mpc-research-cpu/pre-manifest-fix-{junit.xml,pytest.log}`.
The exact captured pytest whitespace is preserved, not edited for a cosmetic
diff check. Source/config whitespace checks are separate from raw evidence.

The manifest-corrected complete suite passed1974 tests,1 native skip, in322.23s.
The first benchmark command then stopped during development baseline selection,
before any held-out policy outcome was inspected: the fluid plant could expose
a first-token timestamp just beyond the next grid sample, or admit a
tiny-future arrival through an epsilon. Two constructed regressions reproduce
both failures on the old plant. The correction uses the same canonical grid
boundary for interval-end events and never admits an arrival before its time.
Chronology validators, policies, metrics, tuning grid, wall limit and held-out
seeds are unchanged. See `development-benchmark-failure.json` and preserved
before-fix logs. The1974-pass run is retained as `pre-grid-fix-full-*`.

## Acceptance layers

The final unfiltered result is recorded in
`results/mpc-research-cpu/final-full-junit.xml` and its matching pytest log.
On the final source, CPython3.12.16 completed **1976 passed,0 failed,1 skipped,
1 existing TestClient deprecation warning in304.02s**. No selector excluded
failures. The final wheel contains all seven MPC Python modules with bytes
matching source; `wheel-build-evidence.json` records its hash. This is package
content validation, not native deployment qualification.
Native vLLM is not installed: its installed-schema case is explicitly skipped,
not passed. CPU functional tests, mock-ASGI behavior, synthetic policy outcomes,
native GPU validation, real vLLM HTTP qualification and deployed MPC superiority
are separate claims.

No new GPU/native/real-vLLM HTTP experiment has run. No gain, native
uncertainty coverage, SLO-safety, stability or production readiness is inferred
from these CPU checks. The native Bead remains authorization-blocked.

## Execution integrity

The no-push boundary was violated by Onyx's three native `gt done` calls.
See `execution-incident.md` for exact branch/SHA/timestamps. All four native
workers were terminated; the pr32 merge queue is empty and their tmux sessions
are absent. Their retained registry entries are idle/recovery-needed, not
running models. Remaining engineering and validation are Mayor-local.
The remote incident branch was not deleted; PR/main remain unchanged.
