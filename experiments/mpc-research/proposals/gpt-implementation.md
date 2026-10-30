# pr-mpc-transport: guarded server and shadow runtime

Status: scoped CPU implementation; focused Python 3.14 checks passed.
This is not Python 3.12/native acceptance, model qualification, or permission
for autonomous live actuation. Research Round1/2/3 reports are preserved.

## Files and ownership

- `gateway.py`: default-off marked ledger and separately named guarded routes.
- `mpc_lab/runtime.py`: strict shared-contract snapshot adapters and GET-only CLI.
- `tests/test_mpc_gateway.py`: mocked gateway, admission, command and lifecycle tests.
- `tests/test_mpc_runtime.py`: adapter, negotiation and shadow-only tests.
- This evidence note.

Read the owner ADR, corrected mathematical specification, `contracts.py` and
`__init__.py`. No telemetry, shared-contract or packaging edits were made.
The owner's source-compatible `finite()` overflow correction may be copied
later without changing this API; the adapter already rejects overflowing numbers.

## Feature flag and marked state

`MPC_LAB_ENABLED` defaults to `false`; only `true` or `1` enables the new paths.
Disabled routes return 404, allocate no request marks, and do not increment the
MPC sequence. Existing metrics fields, admission decisions, forwarding and
`POST /admin/budget` validation/response remain unchanged.

Both new routes use the same `x-admin-api-key` and `ADMIN_API_KEY` as the legacy
admin endpoint. Missing configured key returns 503; incorrect key returns 401.
The feature requires cost-aware mode. Non-single-worker `WEB_CONCURRENCY` or
`UVICORN_WORKERS` values are rejected. This is a per-process implementation;
environment checks cannot discover a contradictory `uvicorn --workers` flag.
The deployment owner must actually enforce a single worker.

`GET /admin/mpc/state` returns the shared Snapshot fields, `schema`, exact
`capabilities: ["mpc-budget-cas-expiry-v1"]`, and the actual `cost_spec`.
`snapshot_from_dict()` accepts this envelope; `snapshot_to_dict()` emits the
canonical schema-plus-Snapshot representation. JSON roundtrips preserve
RequestMark and Identity fields and tuple reconstruction.

Ledger admission, first-content mutation, release, snapshot copying and budget
updates all use the existing `state_lock`. Marks are frozen dataclasses:
immutable cost, estimated prompt length, actual reserved output maximum,
receipt and admission timestamps, plus once-only first-content timestamp.
First content replaces the frozen mark rather than changing admission marks.
It means successful observed SSE content, not GPU execution or client receipt.
`scheduled_at` remains None: the gateway cannot infer client scheduled time.
The ledger ends at gateway reservation release; it is not an outcome archive.

Run identity is telemetry's fresh process run ID. Clock domain uses the existing
`prefill_lab.live_http.clock_domain()` Linux boot-ID/time-namespace identity,
not a per-process nonce. Co-located processes can therefore validate the same
monotonic domain; restart fencing remains the separate run ID. All
marks/sample/application times use monotonic seconds.
The sequence advances on admissions, first observations, releases and either
kind of budget update, not merely on polling. Snapshot completeness requires
both request-count and reservation-sum reconciliation with live admission
state. Direct legacy helper admissions without marks produce `complete=false`
while outstanding, rather than fabricated observations.

Cost identity seals the configured intercept/input/output/minimum coefficients,
default output maximum, estimator version, model name, tokenizer availability,
configured tokenizer identity and heuristic fallback identity. Tokenizer,
model, calibration and support defaults explicitly say `unqualified`.
The latter three use `MPC_MODEL_VERSION`, `MPC_CALIBRATION_VERSION` and
`MPC_SUPPORT_HASH`; tokenizer identity uses `MPC_TOKENIZER_IDENTITY`.
These strings are declared seals, not evidence that native fitting occurred.
Implementation identity comes from the unchanged shared `implementation_hash()`
helper at enabled gateway import. Integration/source changes require a fresh
process and identity; do not hot-edit files and assume the old seal is current.

## Guarded command contract

`POST /admin/mpc/budget` accepts exactly these JSON keys:

```python
{
    "schema": snapshot_document["schema"],
    "run_id": snapshot_document["run_id"],
    "clock_domain": snapshot_document["clock_domain"],
    "identity": snapshot_document["identity"],
    "expected_version": snapshot_document["budget_version"],
    "operation_id": "unique-operation-within-this-boot",
    "budget": 16.0,
    "not_after": server_clock_deadline,
}
```

This is documentation of the server contract, not executable client actuation.
A client cannot treat its own monotonic clock as `server_clock_deadline`.

The server rejects malformed shapes, missing/extra fields, booleans in numeric
fields, nonfinite/overflowing values, nonpositive budgets and invalid versions.
Inside the **same admission/budget lock**, it checks schema/run/clock and every
source/model/calibration/support/cost identity before any uptime comparison.
It then resolves idempotency, checks CAS and outcome capacity, and checks
expiry immediately before constructing/applying the mutation. Equality at
`not_after` is expired. Queued commands therefore cannot rely on checks made
before awaiting the lock.

Success increments the shared legacy budget version exactly once, records
application time and returns operation/run/clock identity, original previous
budget and new budget/version. Lowering the budget does not evict, clip or
release existing reservations. A subsequent legacy update also invalidates a
guarded command's old expected version.

The operation cache retains successful outcomes, bounded at 1024 operations.
An identical canonical payload retry returns the original outcome without
mutation, even after its original deadline or later budget changes. This is
an acknowledgement of an earlier application, not a new application or a
claim that its budget is still current. Changed payload under the same retained
operation ID returns 409. Rejections do not mutate admission state, sequence,
version or operation cache.

When full, new operations return 503; existing retries remain resolvable.
There is deliberately no silent LRU eviction and no unsafe ID-reuse window.
Long-running deployments need an explicitly designed durable/fenced retention
protocol before using this as an operational transport. Restart changes the
boot/clock identity; old commands fail identity validation.

No client posts to the legacy path. The guarded capability is separate from
native-policy authorization. A timeout is still unknown application, not
cancellation; a state GET does not cancel/fence a queued command. No guarded
actuation client is provided, because a persistent uncertainty/resume state
machine and its qualification are outside this minimal implementation.

## Shadow runtime and controller integration

```sh
ADMIN_API_KEY=... python -m mpc_lab.runtime --url http://127.0.0.1:8080
```

The runner makes one GET to `/admin/mpc/state`, requires the exact capability,
validates identity/mark shapes and timestamp ordering within the declared
server clock, and prints one JSON record containing the shared Decision,
snapshot run/clock/sequence/version and `mode="shadow", applied=false`.
It neither compares server uptime to client uptime nor writes any budget.
Redirect following and environment-proxy use are disabled.

The corrected lazy integration calls canonical `validate_snapshot(...)`, then
`model.generate_scenarios(...)`, then
`choose_plan(snapshot, parameters, scenarios, config, now=..., clock_domain=...)`.
There is no `controller.decide` call. Until the modules are integrated, missing
controller/model produces explicit `controller_unavailable`/`predictor_unavailable`
fallback. Missing transitive dependencies or API errors are not silently
relabeled as missing modules. An injectable one-argument planner remains
available, after canonical snapshot validation; it does not generate scenarios.

Parameters/config default to the shared synthetic `ModelParameters()` and
`PlanningConfig()`. Explicit `SyntheticPrior()` defaults are arrival rate 1/s,
prompt/max marks `(128,)`, 16 scenarios and service scales `(0.8, 1.0, 1.2)`;
generated outputs use the declared maximum, no failures and zero transport
delay. Seed defaults to 0; `--seed` changes it. `--max-wall-s` configures the
controller deadline target (default 0.1s). These are declared development
priors/settings, not fits to the current native state.

The runtime resolves the actual local boot/time namespace, rejects domain
mismatch before cross-domain subtraction, and supplies current local
monotonic time to canonical validation before generating any scenarios.
Stale, future, incomplete and identity-mismatched states do not forecast.
`choose_plan` gets refreshed local time after generation, so scenario
generation cannot disguise a state that became stale. The server's domain is
never adopted as the caller's expected domain. Local lookup failure is an
observable `clock_domain_unavailable` shadow fallback.

Expected identity comes from the explicit **local parameters**, never from
rewriting/adopting native server identity. Controller-returned identity and
positive finite budget are checked before serialization. Native unqualified
or mismatched seals therefore remain observable fallback, not synthetic
forecast success.

JSON `reproducibility` records include full parameters/config/prior and their
fingerprints, seed, fixed scenario duration, snapshot fingerprint, generated
scenario-batch fingerprint/count, expected and observed identity, actual local
domain, validation time and injected/default planner mode. No-generation
fallbacks explicitly have a null scenario fingerprint and zero scenarios.
No live transport or native-qualification flag exists in this CLI.

## Executed checks and toolchain provenance

The initially supplied `pr32-mpc-py312/bin/python` failed with:

```text
/bin/bash: line 1: /home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/pr32-mpc-py312/bin/python: No such file or directory
```

System Python was 3.14.4 and lacked pytest (`No module named pytest`).
Before the owner's corrective toolchain message, an isolated dependency
installation attempt failed through the proxy with `403 Forbidden`;
no dependencies were installed. The temporary environment was removed.
No further installs, policy changes or infrastructure repairs were attempted.

The owner then supplied the existing CPU environment. Exact interpreter:

```text
/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/pr32-cpu-venv/bin/python
Python 3.14.4
```

Executed, with that interpreter substituted for `$PY`:

```sh
$PY -m pytest -q tests/test_mpc_runtime.py tests/test_mpc_gateway.py tests/test_dynamic_budget.py tests/test_gateway_diagnostics.py tests/test_shared_upstream_client.py
```

Result: **118 passed in 16.18s**.

```sh
$PY -m pytest -q tests/test_telemetry.py tests/test_trace_replay.py tests/test_prefill_live.py -k 'gateway or lifespan_periodic_logging or shutdown_does_not_hide_final_flush_failure'
```

Result: **6 passed, 73 deselected, 1 warning in 2.11s**. The warning was the
installed Starlette/FastAPI httpx test-client deprecation, not a test failure.

The focused checks cover default-off/auth/mode/worker gates; configured cost
identity; truthful incomplete snapshots; concurrent marked admissions and
SSE first/cleanup on success/failure/cancellation/connect failure; lowering a
cap without eviction; cross-boot smaller/equal/larger uptimes; stale/queued
CAS and queued expiry; idempotency/payload reuse/capacity refusal; malformed,
boolean and nonfinite/overflowing input; legacy endpoint compatibility;
capability refusal; and GET-only shadow/CLI behavior. All backend traffic in
these tests is mocked. `git diff --check` also passed.

No GPU/backend/model download, paid compute, final C, native experiment, policy
gain, full-suite result or host Python 3.12 acceptance is claimed. Owner must
integrate the controller/packaging, run host Python 3.12 focused/full checks,
and separately qualify any live deployment.

## Same-task canonical integration correction

Corrected the nonexistent `decide` API and process-nonce clock domain following
owner review. Only the same five owned files changed. Earlier research reports,
shared contracts and peer sources were not edited.

For real integration testing only, copied the canonical controller/model
unchanged into this package, then removed these temporary copies and their
bytecode. They were never staged or included in the scoped implementation
commit. SHA256 before and after the final tests matched both peer originals:

```text
controller.py 863fc26145190e37eb9ed40db48cb6fc26b48449023d354de7dafd588ec47d8e
model.py      eadbfe1718caf86144277ebbf8604bbf245fd80174293c1aaf12d12962468aec
```

Final executed commands used the same Python 3.14.4 interpreter and selectors
listed above. Updated results:

```text
127 passed in 14.49s
6 passed, 73 deselected, 1 warning in 1.90s
```

The seven parametrized **real CLI entrypoint** integration cases use actual
canonical `validate_snapshot`, `choose_plan`, `generate_scenarios` and `rollout`,
with only HTTP transport mocked. A valid same-host synthetic snapshot executes
all eight plans over 16 scenarios and emits a selected recommendation without
POST. Stale state, incompatible domain (including deliberately incomparable
uptime), and changed support/cost/calibration/implementation seals return
explicit fallback; spies prove neither generation nor prediction ran for
these invalid cases. The successful scenario batch is exactly reproducible
from the emitted prior/seed/duration, and the snapshot/parameter/config/prior/
scenario fingerprints are checked. The CLI test explicitly raises its
development solver target to 2s; this is not a timing claim.

A restart test confirms two gateway instances share the real host domain but
not the run identity. Existing run/domain/CAS/queued-expiry/idempotency and
default-off tests remain passing. The one warning is the same installed
Starlette/FastAPI httpx test-client deprecation. `git diff --check` passed.

The canonical-module integration cases explicitly skip when those peer files
have not yet been integrated; they did **not** skip during the executed
127-test run. No claim is made about newer peer revisions. Owner must run
final integrated host Python 3.12 checks; native qualification remains false.
