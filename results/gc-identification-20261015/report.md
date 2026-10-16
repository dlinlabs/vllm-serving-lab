# RTX 4090: qualified GC-deferred budget identification

The fixed-96, fixed-64 and scheduled 96→64→96 captures all pass the six enabled
integrity/fidelity gates. Fixed 64 has the highest **observed** SLO goodput in this
single run per arm. The intervention shows an admission/first-token backlog after
raising the cap, but substantial variation exists before any switch. This does
not establish an optimal budget, causal improvement percentage or MPC benefit.

## Data and provenance

- Experiment revision: `33a06087ff880746f4951f00bd9ccb69814739c5` (PR #18 merge).
- Host: `13dc91588215`; RTX 4090, driver 580.159.04, 24,564 MiB.
- Python 3.12.3; vLLM 0.30.0; Torch 2.13.0+cu130; Transformers 5.20.0.
- Model endpoint: Qwen/Qwen3-4B-Instruct-2507; max_model_len=8192.
- Each long run: 7,643 offered requests over 300 s; identical trace SHA-256
  `62f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e`.
- All successful measurement requests generated exactly 128 tokens. Azure
  arrival/input lengths are replayed with synthetic prompts; original output
  lengths are not replayed.
- Three warmup requests per run are excluded from performance and included in
  gateway cumulative completions and full-capture engine counter differences.
- TTFT SLO=0.5 s from scheduled arrival, requiring complete successful streams.
  No E2E SLO was declared; this is not joint TTFT/E2E goodput.
- Both client and budget lateness gates remain 50 ms. Runner cyclic GC is
  deferred through replay/drain, RSS allowance=1536 MiB, process sampler and
  gateway/runner diagnostics enabled. Automatic GC remains the program default.

All four captures, including smoke, were re-audited in temporary copies. Reports
reproduced exactly and raw hashes were unchanged. All 14 recorded source/spec
hashes match the experiment revision. `git_dirty=true` is retained. The samples
share one backend exporter process-start identity, supporting backend continuity.
The setup log confirms READY/context 8192, but **backend launch.json/actual argv
are not included**: prefix-cache policy remains self-reported in this bundle.
Exact-revision full native CPU acceptance is not included or claimed.

## Overall outcomes

| Strategy | Success | Rejected | Failed | SLO successes | SLO goodput | Client TTFT P99 | Client E2E P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed 96 | 2095 | 5548 | 0 | 466 | 1.553/s | 3.210 s | 9.346 s |
| Fixed 64 | 1814 | 5829 | 0 | 638 | 2.127/s | 2.400 s | 6.655 s |
| 96→64→96 | 1972 | 5671 | 0 | 433 | 1.443/s | 3.077 s | 8.796 s |

Fixed 64 has 172 more SLO successes (36.9% observed increase) but 281 fewer
successful completions than fixed 96. Rejection rises from about 72.59% to 76.27%.
Successful-request latency is conditional on admission and completion; the
all-offered SLO denominator retains all rejection/failure outcomes.

## Match the same trace windows before interpreting switches

SLO success counts by **scheduled arrival** cohort, with intervals [start,end):

| Strategy | 0–100 s (2457 offered) | 100–200 s (2571) | 200–300 s (2615) |
|---|---:|---:|---:|
| Fixed 96 | 161 | 146 | 159 |
| Fixed 64 | 148 | 240 | 250 |
| 96→64→96 | 95 | 187 | 151 |

Before any switch, both the fixed-96 and scheduled runs use cap=96, yet SLO
counts already differ by 66 (161 versus 95). Their full-run difference is only
33; phase differences are **−66, +41, −8**. The lower scheduled-run total cannot
be attributed solely to switching. Fixed 64 also does not win every phase.
There is one run per arm and no randomized/counterbalanced repetition; requests
are dependent observations, not thousands of independent experimental repeats.

![Matched offered load, SLO goodput, first-token backlog and admissions](matched-windows.png)

## Actual transition evidence

**Down at 100.021989 s:** application lateness=21.99 ms. Reserved cost was
already **62.242**, below the new cap 64. There were 21 gateway completions during
[99,100). This transition does **not** measure drain from a reservation above the
new cap. No universal drain/settling time can be fitted from it.

**Up at 200.001329 s:** application lateness=1.33 ms. Reserved cost was 36.436.
During the next two seconds the gateway admitted **27 requests / 44,698 input
tokens**, with no completion. All 27 eventually completed but missed the 0.5 s
scheduled TTFT SLO. Gateway awaiting-first-content rose from 6 at 199.808 s to 17
at 200.825 s; sampled engine waiting went from 3 to 9 and 11. The engine sample
returns to zero at 203.765 s, but that is not a validated settling-time estimate.
No selected >300 ms gateway stall occurs near 200 s.

![Actual budget application and local state/request response](budget-switches.png)

## Remaining gateway pauses and backend evidence

There are 5 / 5 / 4 selected gateway monitor delays over 300 ms. Gateway loop
gen-2 GC overlaps about **67.4% / 63.5% / 64.7%** of their summed overdue windows.
These are temporal intersections, not exclusive CPU attribution or expected
speedups. Fixed 64's 321.8 ms monitor delay near 23.58 s has no gen-2 overlap.

The slowest approximately 1% successful requests have **97.13% / 93.71% / 97.10%**
of their combined TTFT after dispatch. Their post-dispatch spans do not directly
overlap those selected monitor-delay windows. Indirect effects remain possible:
the step run's largest TTFT (3.846 s, scheduled at 251.287 s) follows a gateway
stall; [250,253) contains 33 admissions / 54,295 input tokens. Timing alone does
not prove a single cause.

Post-dispatch includes transport, frontend, internal queue/prefill and observation
delay. It is **not per-request GPU execution time**.

Full-capture histogram counter differences, including warmup/drain:

| Strategy | Completed count Δ | Queue mean | Prefill mean | Decode mean | Sampled waiting max | Preemptions Δ |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 96 | 2098 | 0.581 s | 0.309 s | 5.714 s | 31 | 0 |
| Fixed 64 | 1817 | 0.333 s | 0.268 s | 4.308 s | 20 | 0 |
| 96→64→96 | 1975 | 0.531 s | 0.309 s | 5.230 s | 30 | 0 |

These aggregate elapsed-time means are not request-linked/exclusive GPU timing.
Their sum is not client TTFT. No sampled preemption increment does not prove an
unsaturated GPU. Runner GC spans are absent throughout deferral; whole-capture
runner loop maxima of 357–406 ms occur after restoration/drain, while the
measurement-window maxima remain 14.79–25.94 ms.

## Model implications and next step

Budget is a useful actuator, but reserved cost alone does not describe work
phase: even at ≥95% of cap, awaiting-first-content ranges over 0–33 / 0–23 / 0–32
in the three runs. Candidate causal features include reserved cost, awaiting-first
count/age, streaming count, available engine waiting, and recent arrival/input-token
mix. Targets should include admission/completion and **all-offered** SLO outcomes.
Future arrivals and eventual output lengths are not online features.

Start an offline feature table and persistence/static forecast baseline. Preserve
observation sample/receive times; the current offline sampler is not an online
feature-delivery channel. Before claiming an MPC-ready model, collect a small,
predeclared repeat/counterbalanced set, cover both control directions including
over-cap downward drain, and validate multi-step forecasts on untouched
chronological windows/unseen control sequences. Do not select optimal caps,
controller periods or uncertainty margins from these three runs alone.

Use the same GPU for baseline-versus-MPC comparisons. A later RTX 3090 portability
study must rerun its own fixed baseline and MPC with recalibration/validation;
old different-hardware captures are not a direct causal baseline.

## Complete evidence and CPU-only reproduction

- [Full original archive parts + hashes](raw-bundle/) — **78 original files**,
  including raw gateway/runner/engine/client records, traces, all reports, original
  analysis ledgers/summary/figures, schedule and setup log. No source file omitted.
- [Computed facts and every original file hash](facts.json).
- [Independent audit verification](verification.json).
- [Self-contained Chinese report](gc-defer-ident-analysis-20261015.html).
- Browsable capture subsets: [smoke](../gc-defer-smoke-96-r1/),
  [fixed 96](../gc-defer-fixed-96-r1/), [fixed 64](../gc-defer-fixed-64-r1/),
  [step](../gc-defer-step-96-64-96-r1/). Extract the bundle before auditing.

From a checkout of the experiment revision or this data-only publication:

```bash
python3 results/gc-identification-20261015/extract_bundle.py \
  --output-dir /tmp/gc-identification-20261015

python analyze_trace_dynamics.py \
  /tmp/gc-identification-20261015/results/gc-defer-smoke-96-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-fixed-96-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-fixed-64-r1 \
  /tmp/gc-identification-20261015/results/gc-defer-step-96-64-96-r1 \
  --output-dir /tmp/gc-identification-reaudit

python results/gc-identification-20261015/analyze_capture.py \
  --capture-root /tmp/gc-identification-20261015 \
  --output-dir /tmp/gc-identification-findings
```

Extraction needs Python 3.12+ standard library only. Offline audit/findings need
the repository CPU dependencies (including NumPy/Matplotlib/HTTPX), no GPU or vLLM
server. Use fresh output directories. Do not rewrite original reports or change
the 50 ms thresholds. The original archive SHA-256 is
`08901e9e18c180d53a1008f21e989708205faa777d6af40b01b2899fe5ea3eec`.
