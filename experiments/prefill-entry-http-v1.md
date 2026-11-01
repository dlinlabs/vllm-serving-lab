# Entry-return HTTP integration: implemented CPU contract, native test pending

This command connects the frozen entry family to actual localhost HTTP through
the protected gateway. CPU/mock validation is separate from GPU/native HTTP
accuracy. Historical controlled and HTTP policies/results remain unchanged.

## Scope and decision point

B2048 is the engine token budget; protected32 is gateway admission capacity.
Both remain fixed. This experiment chooses neither a cap nor an MPC action.

The prediction cut is EngineCore immediately before ADD of the proposed request.
Client-to-cut time has already elapsed. Predict future time from that observed
state to the first engine output; add TRAIN-only future egress to reconstruct
client TTFT. No future schedule, completion, output length or native phase is
supplied to a forecast. This is not a forecast made before gateway admission.

The engine snapshot contains current running/waiting progress, at most one
pending batch at this input-poll boundary, last observed output commit, and a
nonblocking CUDA copy-ready bit. An unready batch overdue under the declared
effective clock remains unavailable; it is not silently assigned zero debt.
The pending residual is still a commit-anchored effective approximation, not an
identified physical GPU service duration. No CUDA synchronize/Future.result is
introduced by the observer.

## CPU-return continuation

The existing six GPU-lane terms remain fixed after same-host TRAIN. The entry
family's prefill/decode return constants are retained as forward.cpu_after.
For each future submitted batch at CPU time t:

- main GPU completion becomes max(previous GPU completion,t)+GPU service;
- CPU return becomes t+P for prefill, or t+D for pure decode;
- the oldest output can commit only when both CPU return and that output's
  readiness allow it; the normal two-slot submit/reap order remains.

The already-pending batch has already returned before the pre-ADD snapshot.
Do not charge its CPU return again. CPU return overlaps GPU service; it is not
added to GPU duration. Nonzero unreviewed stages are rejected. Independent
from-empty EventGraph tests cover GPU-dominant and CPU-dominant costs, empty
and busy snapshots, first output, full completion and in-flight progress.

## One fresh same-host command

The combined workflow has 338 trials: the existing 240-trial controlled
confirmation, then a separate 98-trial HTTP protocol. The HTTP protocol retains
the original cases, stages, arm pairing and budgets but uses new eh identities
and disjoint content namespaces. Both numeric manifests are committed and
validated before collection. No historical absolute coefficients are reused on
a fresh host. No family is selected based on new held outcomes.

1. Verify the exact current native CPU checkout, installed source/runtime,
   pinned model and both exact-token manifests.
2. Run the existing 240-trial controlled confirmation. Failed or inconclusive
   readiness/confirmation stops before HTTP, with raw evidence preserved.
3. While drained on that same boot/time namespace, hand off the original base
   model, entry freeze and passing report. Bind source, runtime, clock domain,
   HTTP manifest, completion time and model hashes. The native configuration
   reconstructs both F0 and entry own forecasts before accepting the costs.
4. Run 10 HTTP LIGHT warmups. Every warmup needs an available forecast and
   matching own/native work. Failure stops before transport TRAIN/held work.
5. Collect 16 HTTP OFF/LIGHT TRAIN trials. Freeze only future egress from TRAIN
   OFF and seal before any held anchor. Keep source/model/manifest identity.
6. Run four before anchors, 64 held OFF/LIGHT trials and four after anchors.
   Every request must complete with exact input/correlation identity; an
   unavailable prediction or failed stream is retained, not excluded.
7. Drain the engine/gateway. Independently rebuild the controlled sub-audit,
   same-boot handoff, every native/client/gateway join, predictions, transport
   seal, scores, gateway totals and closed telemetry from raw evidence.

The new explicit entry launch mode permits the controlled prerequisite and
subsequent live observation in one engine process. The ordinary controlled
checker still rejects that parent launch unless invoked by the dedicated entry
HTTP audit; actual launch flags are copied without alteration. Old live-http
continues to use its original six-term protocol.

## Frozen gates and evidence

Original gates remain: engine max(10ms,20%); HTTP max(15ms,20%); incremental
observer max(2ms,10%); snapshot width <=1ms; prediction age <=5ms; TRAIN egress
spread <=10ms. The 3600s combined collection deadline is a resource bound, not
a relaxed prediction budget. Startup, dependency installation, full native CPU
acceptance and the final offline audit are outside that collection deadline.

All same-domain milestones close client TTFT without adding overlapping spans.
Timing from different boot/time namespaces is rejected. Every native record is
bound to the configuration hash and effective cost document. Any other arrival
after the snapshot which is absent from the model invalidates that forecast.
Dispatcher/source/work/output/lifetime checks and gateway rejection/drain totals
remain mandatory. The exact raw-token localhost path does not cover external
network, chat templating, production request lengths or unknown natural EOS.

F0 is replayed offline on the same recorded LIGHT snapshots. Those descriptive
ablation scores neither add online blocking nor choose/refit a model after held
outcomes. They are not a causal OFF performance comparison or proof of MPC gain.

The helper archives success, qualification failure, partial collection and
cancellation without automatic retries. A completed failed report is audited
before its nonzero exit is propagated. Package/model preparation can be reused
only with matching source acceptance; changing code requires full native CPU
acceptance again. No passing CPU test sets native HTTP/cap/MPC flags true.

## Future rental entry point

After CPU review, from the project directory on RTX4090/Linux/Python3.12:

```bash
python3 scripts/setup_prefill_calibration.py --next entry-http
nohup .venv/bin/python scripts/prefill_host.py entry-http > ~/pr32-entry-http.log 2>&1 < /dev/null &
```

The archive is stored under ~/pr32-entry-http. Use the new entry-http command;
the old live-http command does not silently upgrade its model family. A fresh
native run is still required; there is no HTTP accuracy claim in this revision.

Dependency installation has its own configurable wall-time limit, default
7200s, independent of collection. On a slow download, use
`--install-timeout 7200` (or another positive number of seconds) on the setup
command. This is a ceiling, not a duration estimate or a network-speed fix.
Setup preserves the existing venv and pip cache; completed cached files may be
reused, while an interrupted download may restart. Installation timeout stops
before native acceptance/collection, without automatic retry. Keep the original
log when restarting setup; no experiment gate or collection deadline is relaxed.

## Native attempt at efca3e4: rejected during TRAIN

Archive 1793156148422216020 completes the first 88 controlled trials, then
fails TRAIN first-commit readiness: 21/24 checks pass, three fail, and phase
construction has no errors. No held controlled or HTTP trial is attempted.
Its 1414-test native CPU acceptance and 199 captured source files are verified;
an original-source, matching-numerics reconstruction reproduces the rejection.
See `results/prefill-entry-http-train-rejection-efca3e4/report.md`.

For single-L7168-D16 the four observed first commits are 51.639, 50.539,
25.161 and 49.974ms. Their original error-budget intervals have empty common
intersection. Changing one case-constant prediction cannot make all four pass.
Their normalized recorded release scheduler states match, but those states
exclude pending-copy readiness/debt and physical CPU/GPU state. This evidence
does not identify host contention or show that a state-conditioned predictor
is impossible. First commit may consume a previously submitted background
batch; it is not the foreground request's first token or full batch completion.

The collector now preserves rejected candidate artifacts and failed checks
without writing an accepted freeze/seal, retrying, or advancing into held/HTTP.
The next model design must address decision-time observable state and/or
explicit uncertainty, then freeze a separate test; it cannot silently pick a
different constant from these outcomes or bypass the original failed gate.
