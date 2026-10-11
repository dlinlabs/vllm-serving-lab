# RTX 4090 Azure trace: budget-96 diagnosis

The smoke run and both five-minute repeats are uploaded. All three raw integrity
reports reproduce exactly. The long repeats remain **load-fidelity FAILED** under
the predeclared 50 ms maximum client scheduling-lag limit. They are diagnostic
runs, not qualified MPC benchmarks. No controller was tested.

## What was replayed

- GPU: RTX 4090, 24,564 MiB; driver 580.159.04.
- Qwen/Qwen3-4B-Instruct-2507, max context 8,192; vLLM 0.30.0,
  PyTorch 2.13.0+cu130, Python 3.12.3.
- Fixed admission budget 96; TTFT SLO 0.5 s; output cap 128, natural EOS.
- Smoke: a fixed 30 s source window. Repeats: the same fixed 300 s window
  beginning 2024-05-12 00:01:00 UTC; 7,652 source rows, nine too-short contexts
  excluded, 7,643 retained; no thinning or context-overflow exclusions.
- Arrival times and input token counts come from the Azure conversation trace.
  Prompt content is synthesized to match input length. Historical output lengths
  are retained for audit, not replayed or exposed as an admission oracle.
- Both repeat trace files have SHA-256
  `62f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e`.
  All five recorded runtime source hashes match the analyzed checkout. The
  manifests record commit `a3cdf68bcce2d2d933a575538da5889b538c5366`
  and a dirty working tree; source hashes establish these recorded files' identity,
  not the identity of every unrecorded environment file.
- `cache_policy=disabled` is self-reported. vLLM launch logs and
  `gateway-process.log` are absent from this upload, so backend cache settings
  and engine warnings cannot be independently checked here.

## Results

| Measurement | Smoke | r1 | r2 |
|---|---:|---:|---:|
| Arrival window (s) | 30 | 300 | 300 |
| Offered requests | 787 | 7,643 | 7,643 |
| Successful / rejected / failed | 248 / 539 / 0 | 2,239 / 5,404 / 0 | 2,220 / 5,423 / 0 |
| Integrity | PASS | PASS | PASS |
| Load fidelity | PASS | **FAILED** | **FAILED** |
| Client scheduling lag P99 (ms) | 5.50 | 6.61 | 5.97 |
| Client scheduling lag maximum (ms) | 8.41 | 208.07 | 175.38 |
| Requests exceeding 50 ms | 0 | 5 | 6 |
| Client TTFT P99, successful only (s) | 2.689 | 2.489 | 2.945 |
| Scheduled-arrival TTFT P99, successful only (s) | 2.691 | 2.514 | 2.945 |
| Successful requests meeting scheduled-arrival TTFT SLO | 115 | 721 | 659 |
| SLO goodput / arrival-window second | 3.833 | 2.403 | 2.197 |
| SLO successes / all offered | 14.61% | 9.43% | 8.62% |
| Gateway loop-monitor maximum lag (ms) | 53.75 | 804.08 | 779.07 |

Gateway completed counters include three warmup requests: 251 / 2,242 / 2,223.
They agree with the lifecycle and client records; those warmups are excluded from
performance and arrival plots. No client transport failures, telemetry loss,
state-poll errors, or undrained final reservations were recorded.

The two long runs differ by about 18.3% in client P99 TTFT and 8.6% in SLO
success count. Two repeats do not establish statistical confidence or identify the
cause of that variation. Only 5/7,643 and 6/7,643 requests exceeded the fidelity
limit, but that does not bound downstream queue effects. Do not delete these rows
or loosen the threshold after seeing results.

## Where the latency occurs

The decomposition uses each successful request's shared-host monotonic timestamps:
client start → handler `received` → `accepted` → `backend_sent` → gateway
`first_content` → client first content. Each decomposition sums to that request's
client TTFT within 1 microsecond. Stage percentiles are not added or subtracted.

![Same-cohort tail latency decomposition](latency_stages.png)

| Mean stage for the slowest ceil(1%) successful requests | r1 (23 requests) | r2 (23 requests) |
|---|---:|---:|
| Client start → handler | 52.11 ms | 54.72 ms |
| Handler → accepted | 1.56 ms | 1.65 ms |
| Accepted → dispatch | 10.47 ms | 10.62 ms |
| Dispatch → gateway first content | 2,749.69 ms | 3,128.84 ms |
| Gateway → client first content | 1.10 ms | 1.10 ms |
| Total client TTFT | 2,814.93 ms | 3,196.93 ms |
| Post-dispatch share | **97.68%** | **97.87%** |

Across all successful requests, accepted → dispatch averages 11.66 / 11.39 ms,
with maxima 27.78 / 22.91 ms. Thus a long pause *between those two events* is
not the primary tail component in these runs.

However, `backend_sent` is emitted immediately **before awaiting the HTTP send**.
Post-dispatch includes connection/transport, vLLM queue/prefill, and gateway
scheduling/response processing. It is not a GPU-only timer. The evidence does not
exclude an HTTP-client effect during send, and cannot separate GPU work from
engine queue time. The earlier RTX 3090 synthetic pilot had a different tail
profile; different hardware and workloads prevent a causal cross-study comparison.

## What telemetry says about gateway pauses

The exploratory selection below uses loop-monitor delays greater than 300 ms.
This selection is independent of the original 50 ms client-fidelity limit. The
monitor delay is late execution of a scheduled callback, not a direct CPU or GC
measurement. Its due→resumed interval is not a precisely measured continuous
blocking interval. We also find the largest adjacent lifecycle gap overlapping
that interval or its preceding 200 ms.

| Run | Monitor due (s) | Monitor lag (ms) | Nearby lifecycle gap (ms) | Event before → after gap | Accepted count at gap start, incl. warmup |
|---|---:|---:|---:|---|---:|
| r1 | 58.471 | 650.19 | 635.82 | backend_sent → received | 447 |
| r1 | 124.696 | 804.08 | 812.31 | backend_sent → received | 919 |
| r1 | 190.305 | 790.44 | 761.54 | backend_sent → received | 1,409 |
| r1 | 256.054 | 790.94 | 753.10 | backend_sent → received | 1,912 |
| r2 | 62.826 | 623.41 | 666.46 | backend_sent → received | 490 |
| r2 | 135.493 | 779.07 | 734.30 | received → accepted, same request | 1,014 |
| r2 | 206.993 | 473.74 | 566.55 | backend_sent → received | 1,514 |
| r2 | 272.493 | 484.09 | 535.61 | backend_sent → received | 2,016 |

Use `summary.json` for full-precision values and request IDs. Times are relative to
each run's measurement start. All backend_sent → received pairs refer to
*different requests*, so these gaps cannot be assigned as a request-stage duration.

The accepted-count increments between gaps are **472, 490, 503** in r1 and
**524, 500, 502** in r2. Time spacings are **66.225, 65.609, 65.749 s** in r1
and **72.667, 71.500, 65.500 s** in r2. Roughly 500 admissions is a useful
work-count-related hypothesis to investigate, not proof of an allocation, GC,
HTTP-client, or engine mechanism. Count pacing may also reflect throughput.

![Replay, scheduling, stage latency, and gateway reservations](stall_alignment.png)

All 23 tail requests in each long run have **zero overlap between their
post-dispatch stage and the selected monitor due→resumed intervals**. This is a
narrow timing observation: earlier queue effects, smaller monitor delays, and
unobserved work are still possible. It does show that the large monitor intervals
do not directly cover these tail stages.

Only one of r1's five late client intervals intersects a selected gateway monitor
interval; none of r2's six do. Client schedule qualification and gateway pause
diagnosis therefore need separate checks. Both use the same clock, but their
processes do not have to stall together.

The source's largest planned one-second bin has 43 requests in both repeats;
gateway receipt peaks are 50 and 49. Some gateway-side peaks reflect accumulated
requests arriving at the handler after a gap. Preserve this service-side distortion
as a system disturbance; separately verify that the generator supplied the
predeclared external workload. The one-second bins are visualization, not a
replacement for the per-request fidelity audit.

## What this means for MPC and the next rental

The current runs expose real disturbances and candidate state signals. We do not
need a perfectly smooth gateway to design MPC. We do need a defensible boundary:
planned arrivals are external workload, generator lateness is replay error, and
handler/engine delays are part of the system under control.

1. **Finish CPU-only work before renting again.** Preserve complete server logs;
   revalidate raw records; exercise the generator against a mock endpoint with
   this trace's burst pattern. Such a mock qualifies a code path, not the shared
   host under GPU contention. Predeclare the next experiment settings and limits.
2. **Collect a qualified fixed-budget sweep on one hardware/configuration.** A
   candidate set is 32, 48, 64, 96, with balanced run order and at least three
   repeats. These are proposed excitation points, not established optimum budgets.
   Keep the existing 50 ms client limit, separate or provision the generator if
   needed, and retain every diagnostic failed run. Log actual launch flags so
   cache policy is verified. The old GPU's cost coefficients are a proxy, not a
   measured RTX 4090 service-time model.
3. **Investigate pauses with targeted timing.** CPU/RSS and GC start/end logging,
   explicit timings around body parse/tokenization and upstream send, and vLLM
   queue/prefill observations would discriminate causes. Compare a shared upstream
   client against the current per-request client under matched conditions rather
   than attributing pauses from adjacency alone. Current telemetry has lifecycle,
   loop lag, state-poll RTT, and counters, but no CPU stack or GC duration.
4. **Identify dynamics before implementing MPC.** Fixed budget 96 cannot estimate
   the effect of changing budget. After the sweep, excite budget over time and
   model awaiting-first-content requests, streaming requests, and reserved cost
   as gateway state proxies; include arrival rate and input mix as disturbances.
   Validate predictions on a predeclared separate source window and on unseen
   repeats before controller comparisons. Fit coarse intervals only after checking
   timing and prediction error; these runs do not establish an update period.
5. **Cover decode variation explicitly.** Every successful measurement request in
   all three runs generated exactly 128 tokens. This is a controlled fixed-output
   study, not evidence of robustness to variable output demand. A later protocol
   should predeclare several output caps/mixes and measure realized lengths,
   without revealing historical future output length to the controller.

Compare MPC with the best validated fixed budget and a simple reactive controller.
Use scheduled-arrival SLO goodput over **all offered requests**, with rejection
rates and per-input-size results alongside it; optimizing successful-only P99 can
reward rejecting almost everything. Report controller solve time, state freshness,
and fallback behavior as measured overhead. No MPC benefit or optimal control
frequency follows from these two budget-96 runs.

## Reproduction and provenance

```bash
python analyze_trace_dynamics.py \
  results/trace-smoke-96 \
  results/trace-experiment-96-r1 \
  results/trace-experiment-96-r2 \
  --output-dir results/trace-budget96-analysis
```

Requires NumPy and Matplotlib plus the project's replay-auditor dependencies.
`summary.json` includes hashes of ten files per run, the exact re-audited reports,
stage distributions, cohort IDs, arrival bins, and monitor/gap details. The
`*-tail-and-late.jsonl` files retain joined rows for the selected tail and late
client requests; add `--all-requests` to export every joined measurement row.
No raw reports or figures are overwritten. Audit outputs are written to a
temporary directory. Raw hashes are checked before and after processing.
