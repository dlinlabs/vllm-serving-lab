# Bounded runner GC deferral

The October 15 runner diagnostic capture identified three launches over 50 ms
inside one 169.74 ms runner gen-2 GC. This feature tests that specific load
generator problem. It is **opt-in**; `automatic` remains the default. No change
is made to gateway/vLLM GC, admission policy, model, trace or fidelity thresholds.
There is no measured latency improvement from this change yet.

## Policy and protection

Use `--runner-gc-policy defer --runner-max-rss-mib 1536` alongside
`--runner-diagnostics --gateway-diagnostics --sampler-mode process`.

- The sampler must be a separate process, started before deferral. This avoids
  changing its GC policy together with the runner. Runner diagnostics are required.
- After warmup has drained and the initial budget is acknowledged, validate the
  current runner RSS, run `gc.collect(2)`, read RSS again, then disable automatic
  cyclic GC **before choosing the measurement clock**.
- Keep automatic GC deferred through the arrival window and client-request
  drain. Reference counting still runs. Restore the original enabled state and
  thresholds synchronously when work finishes, errors, or is cancelled; the
  existing protected cleanup also restores before gateway finalization.
- The RSS allowance is explicit, not inferred from this host's free memory.
  1536 MiB is a proposed bound for this diagnostic, above the prior observed
  approximately 839 MiB runner peak. It is not a proven deferred-mode peak.
- A worker-thread resource read uses the **unchanged native PID/procfs guard**.
  Check every 100 ms; terminate the attempt if a read is unavailable/invalid,
  RSS exceeds the declared cap, GC policy changes, or active RSS sample end times
  are separated by more than 500 ms. Reader waits have a one-second timeout;
  cancelling one cannot forcibly stop an underlying OS read.
- The maximum disabled duration is predeclared as trace duration + drain timeout
  + one second, with a maximum supported bound of 600 seconds. The extra second
  covers choosing the clock and the initial 0.2-second measurement lead-in.
  Samples are capped; failures cancel traffic/control work, do not retry commands,
  and leave planned outcomes in the capture.

These are sampled fail-closed checks, **not** an OS hard memory limit or hard
real-time deadline. A stalled interpreter can delay the watchdog. Deferral can
retain reference cycles, so it must not become an unbounded service default.

## Evidence and compatibility

`replay_plan.json` binds the GC policy, absolute RSS limit, duration bound and
RSS polling bounds to the immutable trace/config. `runner_gc_control.json`
contains the pre-collection span, policy transitions, native RSS samples, work
completion/failure and restoration. It is atomically persisted after restoration,
outside the arrival window; no periodic JSON writes are added to that window.

`report.json.runner_gc_integrity` is a separate gate. Normal completion, exact
identity/spec, RSS coverage/bounds, restored policy and qualified runner
diagnostics are required. A GC span inside the declared deferred window fails
this gate, even if some library explicitly called `gc.collect()`.
CLI success and offline comparison qualification require the gate when enabled.
Invalid/missing evidence is not converted to a successful run.

`runner_diagnostic_integrity.max_loop_lag` still spans the whole capture and may
include the deliberate **pre-measurement** full collection. The new
`runner_gc_integrity.measurement_max_runner_loop_lag` restricts due times to the
arrival window. Load fidelity still uses actual request launch delays at the
unchanged 50 ms limit. Legacy/default-automatic captures retain their old reports.

## Rental-host checks

Record and freeze the revision; do not update code between comparison arms.
Before backend startup, prefer full exact-revision native CPU acceptance:

```bash
.venv-cpu-tests/bin/python scripts/run_linux_cpu_tests.py
```

If the backend is already running, the focused CPU/mock tests do not use its
ports and may precede a diagnostic capture. They are **not** the full native
matrix and do not establish an RSS/latency improvement on the GPU host:

```bash
.venv-cpu-tests/bin/python -m pytest -q tests/test_runner_gc_control.py tests/test_runner_diagnostics.py
```

The unit tests inject resource samples at a private reader seam to test failure
handling; the production CLI has no alternate reader or namespace override.
The real process-sampler/scheduled-control integration cases also exercise this
mode on a compatible native host, inside the unchanged five-case CPU matrix.

## First capture: smoke only

Wait for the existing backend READY message. Use the byte-identical checked-in
trace; model setup/context/cache policy are unchanged. Choose a fresh directory:

```bash
nohup .venv/bin/python -u run_trace_pilot.py \
  --trace results/trace-smoke-96/trace.json \
  --output-dir results/gc-defer-smoke-96-r1 \
  --budget 96 --ttft-slo 0.5 --max-scheduling-lag 0.05 \
  --cache-policy disabled --gateway-diagnostics --sampler-mode process \
  --runner-diagnostics --runner-gc-policy defer --runner-max-rss-mib 1536 \
  > /workspace/gc-defer-smoke-96-r1.log 2>&1 < /dev/null &
```

```bash
cat results/gc-defer-smoke-96-r1/report.json
```

Inspect all six gates, including the new GC gate, RSS peak and restoration.
Then run one five-minute diagnostic with the **same** flags, changing only:

```text
--trace results/trace-experiment-96-r1/trace.json
--output-dir results/gc-defer-fixed-96-r1
```

Use a matching new log filename too. Stop on an RSS/gap/policy/fidelity failure;
keep the failed capture and do not rerun until one passes. Do not raise the RSS
cap or change trace/50 ms gate after seeing the result and call that the same run.

This first capture qualifies the harness, not a causal improvement claim against
a different host. A later automatic/deferred comparison needs the same revision,
host, trace, backend and observer settings with all attempts retained. Once the
load generator qualifies, use the same harness policy for 96/64 and down/up
identification; gateway GC remains a realistic plant disturbance.

Download complete raw captures, setup provenance, native test artifacts and
`runner_gc_control.json` before returning the machine.
