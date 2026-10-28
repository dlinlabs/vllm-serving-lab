# Entry-return HTTP integration: CPU validation

The new `entry-http` command and independent raw checker are implemented.
**348 related CPU tests pass, with zero failures, errors or skips (93.85s).**
These are one overlapping selection, not additional native acceptance cases.
No GPU collection or native HTTP accuracy result is produced by this change.

## What is covered

- Independent EventGraph parity from empty and busy cuts, first output and full completion, GPU/CPU overlap, and no duplicate charge for an already-pending return.
- Fresh controlled-to-HTTP model handoff with source, runtime, boot/time namespace, TRAIN membership, original prediction/freeze identities and chronology checks.
- Explicit combined native launch; old standalone and HTTP mode isolation; 240 controlled plus 98 HTTP numeric trials and exact-token namespaces.
- Transport sealed after HTTP TRAIN and before held trials; all stages use the frozen costs. Same-snapshot F0 replay is descriptive and offline.
- Raw-audit reconstruction and tampering of source-bound handoff, clock domain, manifests, client sidecars, gateway journal, rows, transport seal, report, runtime, CPU proof and unexpected traffic.
- Mocked prerequisite, warmup, generator, cancellation and changed-model failures; partial cleanup, failed-report audit and archive preservation.
- Existing live, controlled confirmation, receipt, entry/submit/event clock, async predictor and analysis regressions; committed design/hash regeneration.

Native hardware, process/source boundary mocks and synthetic fixtures do not
establish native performance or replace full exact-checkout acceptance. The
future host setup runs the full native suite before collection. Historical
1331-test native acceptance and 240-trial controlled evidence at 2ebb185 remain
unchanged and separate. On that prior host F0 and entry forecasts were identical;
there is still no demonstrated correction advantage or cross-host guarantee.

## Remaining evidence

The new command requires a fresh same-host 240-trial controlled confirmation,
then 98 HTTP trials. Collection has a 3600s resource bound, excluding setup,
startup and final audit. Fixed B2048/protected32 and pre-ADD after admission do
not validate cost-aware C16/C32 actions. Native HTTP qualification, action/drain
ranking and a closed-loop benefit remain separate unpassed gates. No live MPC
controller is implemented here.

See `experiments/prefill-entry-http-v1.md` for the command and unchanged gates,
and `experiments/prefill-to-mpc-roadmap.md` for the bounded decision experiment.

`cpu-validation.json` binds this test selection to current source SHA256 values
and the exact JUnit/log/report bytes. It records local numerical versions and
explicitly keeps native HTTP, model, adaptive-cap and MPC acceptance false.
