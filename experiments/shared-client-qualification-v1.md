# Shared upstream client: bounded development qualification

**Qualification completed.** The declared smoke/A captures pass all six enabled
gates; exact runtime d0ccb2c has 476 native CPU cases passing on two hosts.
[Complete archives, verification and model limitations](../results/shared-client-20261018/report.md)
are published. The original frozen JSON is unchanged; these run names are consumed.

The original excitation-v1 A capture on 2026-10-17 stopped at the 210 s
96-to-48 step: send lag 0.961 ms, application lag 52.884 ms (limit 50 ms).
Within the 51.923 ms send-to-application interval, recorded synchronous gateway
phases overlap 50.347 ms, including a 39.960 ms per-request HTTP client
constructor. These are elapsed spans, not exclusive CPU attribution. No gateway
GC overlaps that interval. The runner's 282.916 ms GC starts after the failed
acknowledgement, following restoration of automatic GC. This supports removing
per-request transport construction, without establishing that reuse guarantees
the control deadline. Original failure evidence remains unchanged.

`run_trace_pilot.py --shared-upstream-client` explicitly opts into one gateway
lifespan-owned `httpx.AsyncClient`. It is constructed before requests, shares
connections, and closes on application shutdown. Each request closes only its
own response and releases its reservation, including cancellation/error paths.
Response cookies are not persisted between users. The shared client connects
directly (`trust_env=False`); pool connections have no additional concurrency
cap, while idle keep-alive connections are capped at 100. Per-request timeout
remains 60 s. Admission rules and the 50 ms load/control gates are unchanged.
Legacy per-request mode remains the default.

In shared mode, per-request `client_construction` and `client_close` phases are
`not_reached`; they must not be interpreted as missing shutdown cleanup. Metrics,
launch settings and the runner manifest record and verify the selected mode.

Use the separate [frozen qualification declaration](shared-client-qualification-v1.json).
It declares a single fixed-96 smoke and a single A replay on existing development
inputs, in that order, with identical instrumentation and 50 ms limits. Every
enabled integrity/fidelity gate must pass before proceeding. Keep failures and
stop; no retry-until-pass or held-out run is authorized by this declaration.
This protocol has no model-acceptance section: passing it qualifies measurement
in this mode, not forecast quality or MPC benefit. Do not append these captures
to excitation-v1 or silently pool old/new gateway modes for model acceptance.

First run the complete native CPU entry point on a clean checkout of the exact
new revision with ports 8000/8080 free. Its full suite includes a shared-client
real-subprocess mock replay (success, rejection, upstream failure and scheduled
controls). Existing virtual environments can be reused on the same host; no
new CUDA installation is required. Then restart the same configured vLLM backend
and keep that launch for smoke and A. Archive its actual launch and host records.

For each run, use the declared trace and schedule, the existing gc-deferred
replay settings, and both:

```text
--shared-upstream-client
--protocol experiments/shared-client-qualification-v1.json --protocol-run RUN_NAME
```

The output directory basename must equal RUN_NAME. The implementation alone was
not GPU evidence; the archived completed captures now qualify measurement.
The unused held-out C window stays unused;
any later model acceptance needs a separately declared protocol for the new mode.
