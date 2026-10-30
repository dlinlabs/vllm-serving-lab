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

Run identity is telemetry's process run ID. Clock domain contains a fresh
process-boot nonce and all marks/sample/application times use monotonic seconds.
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

The lazy integration interface is `mpc_lab.controller.decide(snapshot) -> Decision`.
Until that module is integrated, the CLI emits an explicit
`controller_unavailable` shadow fallback. Incomplete state skips the planner
with `incomplete_snapshot`. Controller-returned identity and positive finite
budget are checked before serialization. Missing controller dependencies or
API errors are not silently relabeled as an unavailable controller.

Model-specific observation age, support, scenario generation and forecast
validation remain the controller's responsibility. The shadow adapter alone
does not establish an age bound or resolve the unknown network/clock offset.
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
