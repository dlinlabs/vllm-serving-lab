# RTX 4090 diagnostic capture — 2026-10-12

## Outcome and qualification

This round validates PR #12's diagnostic capture and narrows the explanation of
periodic gateway delays. It does not evaluate an MPC controller or a budget change.
The long replay passes data integrity and diagnostic integrity, but **fails the
predeclared 50 ms maximum client scheduling-lag gate**. Preserve that result.

| Capture | Offered | Successful | Rejected | Failed | Data integrity | Load fidelity | Diagnostic integrity | Max client scheduling lag |
|---|---:|---:|---:|---:|---|---|---|---:|
| Diagnostic smoke, 30 s | 787 | 233 | 554 | 0 | PASS | PASS | PASS | 5.30 ms |
| Diagnostic experiment r1, 300 s | 7,643 | 2,063 | 5,580 | 0 | PASS | **FAIL** | PASS | 146.53 ms |

The long run has 409 SLO successes, goodput
1.363 requests per arrival second,
and client TTFT P99 2.814 s. Rejected and failed arrivals
remain in the offered denominator. TTFT SLO means successful stream completion
and first content within 0.5 s of the **scheduled arrival**.
Three successful warmups are excluded from performance but included in gateway
counters: 2,066 gateway completions versus 2,063 measurement successes is expected.

## Experiment boundary and provenance

- Code commit: `f1f7cef3dbea01cda0ab38ce7fa13b929dcdc1aa` (merged PR #12).
- Host: RTX 4090, driver 580.178.04, 24,564 MiB; Python 3.12.3;
  vLLM 0.30.0, Torch **2.13.0+cu130**, CUDA runtime 13.0.
- Model: Qwen/Qwen3-4B-Instruct-2507; verified context limit 8,192.
- Fixed admission budget **96**; output cap 128 with natural EOS.
- Prefix caching disabled, supported by preserved
  [launch argv](setup/launch.json), rather than solely the runner's self-report.
- Same Azure preset/window as the earlier experiment repeats:
  2024-05-12 00:01:00 UTC for 300 s; synthetic prompts reproduce input lengths
  and arrivals. Source output lengths are audit-only, not replayed.
- Trace SHA-256:
  `62f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e`.
- Run ID: `1b094e42de6a49e2a6362980099d4210`; measurement monotonic start
  `93351.875394356`. Normalize each capture by its own start.
- Manifest records `git_dirty=true`; all 11 recorded runtime source/dependency
  file hashes match the capture commit. Do not present the manifest as a clean checkout.
- Gateway diagnostics and external engine/resource sampling were both enabled;
  GPU utilization and request-linked GPU/prefill execution were **not collected**.

The unmodified [complete archive parts](raw-bundle/) is 21,123,273 bytes
and contains 81 regular files, including both real captures, setup/install/model
logs, native validation proofs, and the user's original offline analysis.
Archive SHA-256:
`32676c162da7182e1b22693f1e0eff899ad7179e10bf28a3074b7b9c0324c454`.
[Machine-readable facts](facts.json) include all 81 original file hashes.
Small reports/manifests are also exposed outside the archive for GitHub browsing;
these directories are **not standalone raw captures** until the archive is extracted.

## 1. Long generation-2 GC is a measured part of gateway pauses

All four long generation-2 GC spans occurred on the gateway event-loop thread.
Their thread CPU duration is almost equal to wall duration.

| GC start, measurement second | Monitor overdue ms | GC wall ms | GC thread CPU ms | Actual interval overlap / overdue |
|---:|---:|---:|---:|---:|
| 58.518 | 587.7 | 384.4 | 384.4 | 62.8% |
| 120.406 | 691.4 | 411.2 | 411.1 | 56.4% |
| 186.842 | 677.5 | 417.9 | 417.9 | 61.7% |
| 249.716 | 552.9 | 484.6 | 484.6 | 87.6% |

![Gateway monitor delays and actual GC overlap](gc_overlap.png)

Generation-2 GC overlaps **66.2%** of the
summed four monitor-overdue intervals. This is interval intersection using actual
GC start/end, not the later emission time. There are 650 GC spans across the whole
capture (warmup/drain included): 592 generation 0, 54 generation 1, and four
generation 2. The four long spans total 1.698 s.

The observation supports GC as an important component of these pauses, not the
only cause. Monitor overdue is not an exact continuous CPU-blocking boundary;
GC CPU can include finalizers and callbacks. The overlap fraction is **not a
predicted speedup from disabling GC**. No allocation stacks or reclaimed object
types were captured, and this does not retrospectively prove the cause of older
r1/r2 pauses.

GC start intervals were 61.888, 66.436, and 62.874 s, not a fixed timer. Accepted
increments between them were 427, 462, and 422, not a fixed request-count trigger.
Observed thresholds were `(700, 10, 10)`. A relatively stable allocation rate is a
plausible reason automatic collection looks periodic, not a proven object source.
The second long GC begins within a recorded cost-estimation phase and the fourth
within client construction; that does not identify where collected objects originated.

## 2. Multi-second TTFT peaks have substantial post-dispatch delay

For the slowest 21 successful requests (approximately 1%), the mean client TTFT
is 3.260 s and **93.72%**
of their summed TTFT occurs between gateway `backend_sent` and first content.
None of those 21 post-dispatch intervals directly overlaps a long generation-2
GC span. GC may still indirectly amplify a subsequent admission burst.

During [186,189) s, 43 requests with 71,461 estimated input tokens were admitted,
and second 187 has 33 completions. Reserved cost stays near 95 while the gateway
awaiting-first proxy rises from 4 to 30. The sampled vLLM waiting gauge rises from
0 at 187.815 s to 24 at 188.816 s. Initial admissions also produce a multi-second
TTFT peak before any long generation-2 GC.

![TTFT, engine backlog, gateway GC and client scheduling](diagnostic_overview.png)

This supports an admission/refill wave plus engine backlog as a candidate
mechanism. `backend_sent` is recorded before awaiting HTTP send: this span also
includes transport, frontend work, engine queueing and observation delays. It is
**not exclusive GPU execution time**, nor proof of GPU saturation. Observed
preemptions counter delta is zero.

## 3. Client scheduling failure is separately visible

Nine launches exceed 50 ms, clustered around 97.7 s and 178 s. Neither cluster
coincides with the four long gateway GC spans. Eight are rejected; one succeeds
within the TTFT SLO. No requests are excluded to pass the gate.

Near the second cluster, collector elapsed time from receiving the HTTP body to
starting host-resource reads is 166.7 ms and overlaps delayed launches. This
suggests runner-side parsing/GC/GIL/scheduling as candidates, not an attribution
to any one of them. The first cluster starts before that sample's HTTP request.
A sampler worker thread still shares the runner interpreter; process isolation
is a future experiment, not a demonstrated fix.

Per-admitted-request HTTP client construction was measured 2,066 times: median
17.81 ms elapsed, maximum 502.81 ms. Reusing a client is a reasonable optimization
to test, but these records do not prove it caused all GC objects or pauses.

## Native validation completed

- [Full Linux CPU test log](native-validation/pr12-linux-tests.log): **135 passed**,
  31.10 s, no skipped tests in the recorded run.
- [External-only log](native-validation/pr12-external-only.log) and
  [validation status](native-validation/validation_status.json): success, rejection,
  and backend-failure fake-backend captures all pass; gateway diagnostics disabled,
  external observations enabled. This is **not** a real GPU performance experiment.
- Saved external-only raw captures precede the test's deliberate corruption
  checks. All three are included in the archive.
- Independent read-only re-audit exactly reproduced five saved reports
  (real smoke/long replay plus three fake-backend cases) and the uploaded dynamics
  summary. All 11 recorded runtime hashes matched. Raw originals were not rewritten.

## Reproduce without a GPU

From the repository root, with its CPU analysis dependencies installed:

```bash
bundle=$(mktemp -d)
python results/diagnostics-20261012/extract_bundle.py --output-dir "$bundle"
python results/diagnostics-20261012/build_report.py \
  --bundle-dir "$bundle" --output-dir "$bundle/review"
```

The builder calls the versioned dynamics analyzer, writes to a new output
directory, then produces facts, two plots and a self-contained HTML report.
The archived `results/analysis-diag-96-r1` also preserves the original analysis.
Use the capture commit when checking runtime source hashes; subsequent code
changes can legitimately differ from those captured.

## Next experiment toward MPC

First compare fixed budgets on the same trace, with the same instrumentation,
model, cache policy, SLO and original fidelity gate. Initial candidate budgets
48/64/80/96 are a proposal, **not measured results**. Repeat and vary execution
order, retain every run and mark qualification explicitly. Historical budget-96
results can provide context but do not substitute for matched new repeats.

Then predeclare a budget step schedule and log actual applied times to identify
response delay and state dynamics. Fixed-budget data alone cannot determine MPC
control gain, update interval, prediction horizon, or benefit. GC diagnosis need
not become a prerequisite to obtaining budget-response data.

There is no matched diagnostics-off/on study, so the difference from older r1/r2
SLO counts cannot be assigned to observer overhead. No admission-policy or MPC
implementation changes are included in this publication.
