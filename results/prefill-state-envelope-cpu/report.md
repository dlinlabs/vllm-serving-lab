# State timing envelope: conditional CPU prototype

**285 related CPU tests plus one metadata/hash check pass (286 total, no
failures/errors/skips).** This is a separate conditional interval interface,
not acceptance of the old point model, HTTP, cap or MPC. No native collector,
new GPU run, native coverage or end-to-end compute-age qualification is added.

## Implemented behavior

- Use real pre-ADD request progress, pending work/time/readiness and an explicit
  service box. Every future batch can independently vary inside CPU/GPU ranges;
  no common future P/D scalar is required.
- Propagate extremal times through the fixed logical two-slot graph. Independent
  EventGraph tests use varying per-batch GPU and CPU services, empty/busy cuts,
  ready/inflight outputs, CPU/GPU overlap, TTFT and full completion.
- Keep missing/stale/mixed-clock/overdue unready input unavailable. Never infer a
  missing ready bit or debt from future observed commits. Preserve wide bounds
  with fallback=true rather than selecting a favorable point or dropping them.
- Bind scope, runtime, dispatcher, implementation and support identities; reject
  tampering and preserve the original public point forecast's overdue behavior.
- The support freezes age <=5ms and width <=10ms arithmetic policies. The
  optional deadline relation is engine-cut-relative only, not client SLO safety
  or a pre-admission decision. All outputs explicitly forbid actuation and keep
  native/model/HTTP/adaptive-cap/MPC qualification false.

## Reuse of the immutable native archive

From the original efca3e4 archive (115 files, SHA256
`a281a36e0939a7181233f8c7edbc5dc86bec598cd33377c83a641d26f51e4d8f`),
reconstruct the original 24 TRAIN LIGHT event identities and validate them
against the original reconstructed base model. From 3280 past prefix worker
host spans, prefill support is **10.732867–55.953486ms** and decode support is
**0.738216–4.336616ms**. GPU coefficients stay fixed at the development fit.

These empirical extrema are not calibrated coverage/confidence bounds or
identified physical CPU/GPU costs. The old 3/24 point-readiness failure remains.
A trajectory outside the supplied box need not lie inside the computed envelope.

All **24** checked legacy TRAIN release states lack pending/time information and
return missing_pending_time_state. This is an input-sufficiency result, not a
latency/coverage score. No future ready bits, completion times or held labels
are substituted for absent state. Every original raw file and historical verdict
is unchanged. The native 1414-case CPU proof remains separate.

## Limits and remaining native work

Conditional bounds rely on fixed pre-ADD membership, no later arrivals, forced
lengths, monotone nonnegative max-plus costs and the declared effective pending
anchor. They do not establish a physical pending-debt bound on this GPU. The
CLI can construct past support and replay snapshots; it does not launch or
upgrade entry-http. Synthetic examples are labeled synthetic even though the
provided runtime metadata comes from the archived host.

A separate short shadow collection contract/entrypoint is still needed before
another rental: prospectively sealed support, actual nonblocking pre-ADD state,
observer controls, actual forecast age, missed/wide/unavailable cases retained,
and held coverage/usefulness analysis. No safe admission or native model claim
follows from the synthetic bounds. C16/C32 action validity and closed-loop gain
remain separate. See experiments/prefill-state-envelope-v1.md for equations,
inputs, CPU commands and the unresolved native requirements.

The 286-case selection overlaps previous local selections; do not sum it with
those or with native acceptance. Current code/artifact hashes are recorded in
cpu-validation.json. There is no new full native CPU acceptance run here.
