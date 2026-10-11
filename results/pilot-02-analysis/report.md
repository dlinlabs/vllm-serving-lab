# pilot-02 dynamics diagnosis

Run ID: `eb6102dad507470b88b2709bdf8f1690`. Source code commit: `e1e2cf9716f460cabdcc661db9df677656e4817f`.

## Scope and integrity

Single RTX 3090; Qwen/Qwen3-4B-Instruct-2507; max_model_len 8192; vLLM 0.30.0; Python 3.12.3. Open-loop 30 RPS for 150 seconds, budgets 80/96/112/96/80 for 30 seconds each, after 15 warmup requests at 1 RPS and a separate drain. Temperature 0; natural EOS. All 4,515 warmup + measurement IDs are unique and match gateway lifecycles. Measurement: 4,500 attempted, 3,402 completed, 1,098 rejected (24.4%), zero failed. Telemetry loss, write errors and tokenizer fallback are zero; final active work is zero. Last measurement request terminates 1.931 seconds after arrivals stop. Raw input is not modified; SHA-256 provenance is in summary.json.

## Latency is concentrated in two different paths

Each successful request is joined by ID and decomposed as:

`client TTFT = client start→gateway received + received→accepted + accepted→backend_sent + backend_sent→gateway first_content + gateway first_content→client first_content`

Scheduling lag is separate because client TTFT starts at actual client start. All processes share this host's monotonic clock. Gateway emit timestamps approximate observer timestamps; no engine-internal timestamps exist. Maximum per-request decomposition residual is 0 seconds.

| Segment | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- |
| ingress_s | 1.5 | 50.9 | 1293.7 |
| received_to_accepted_s | 0.4 | 2.1 | 3.2 |
| accepted_to_sent_s | 18.3 | 19.8 | 21.5 |
| sent_to_first_s | 105.7 | 595.6 | 703.4 |
| delivery_s | 3.1 | 4.5 | 5.6 |
| gateway_ttft_s | 125.1 | 614.7 | 722.4 |
| client_ttft_s | 130.1 | 730.7 | 1853.5 |

Quantiles from different segments cannot be added or subtracted. Instead, the slowest 35 successful requests (ceil of 1%) have mean client TTFT 2.021s, mean ingress 1.431s (70.8% of aggregate TTFT), and mean backend_sent→first_content 0.570s. The latter includes upstream connection/HTTP, vLLM processing/queueing, and gateway observation scheduling; it is not pure GPU time.

![Per-request latency diagnosis](latency_diagnosis.png)

## Five gateway lifecycle event gaps

| Gap start s | Gap end s | Duration s | Client starts inside | Distance to nearest budget step s |
| --- | --- | --- | --- | --- |
| 20.456 | 21.130 | 0.674 | 20 | 9.6 |
| 46.890 | 47.983 | 1.093 | 33 | 13.1 |
| 71.598 | 72.549 | 0.951 | 29 | 11.6 |
| 97.488 | 98.532 | 1.045 | 31 | 7.5 |
| 123.891 | 124.774 | 0.884 | 27 | 3.9 |

Gap-start spacings are 26.43s, 24.71s, 25.89s, 26.40s. They do not coincide with budget steps. No snapshots were emitted inside any of the five gaps. Requests continue to start in the separate load process during these gaps; scheduling lag over all arrivals has P99 7.2ms and maximum 16.4ms. A temporary slowdown of gateway processing/host scheduling is consistent with the evidence. The exact cause is unproven: no CPU profiles, event-loop lag, GC timing, or vLLM internal timing were recorded. Transport queueing and resource contention are still possible contributors.

Gateway code creates a fresh httpx.AsyncClient for each admitted request. accepted→backend_sent has a median 18.3ms. This span includes client construction and request building, but does not separately time each operation. Reusing a lifespan-owned client is a plausible experiment; it is not an established fix for the recurring gaps. Other hypotheses include GC pauses and synchronous work. Instrument them before drawing causal conclusions.

As a sensitivity analysis only, excluding starts from 1 second before each event-gap start to 2 seconds after its end leaves 2962 successes with client P99 TTFT 0.193s. These windows were chosen after observing this run. This is not a replacement performance result and exclusions must not be used to claim MPC improvement.

## Budget effects and downward-step behavior

Rows are cohorts by scheduled arrival time, with latency on successful requests only. They include transitions and spikes. Snapshot state means are unweighted sample means, not time-weighted integrals. One run cannot establish steady state, causal latency effects, an optimal budget, or a settling-time constant.

| Window s | Budget | Reject % | Client P99 TTFT s | Client P99 E2E s | Mean cost | Mean streaming |
| --- | --- | --- | --- | --- | --- | --- |
| 0-30 | 80 | 22.8 | 1.429 | 4.104 | 75.6 | 52.8 |
| 30-60 | 96 | 25.7 | 1.826 | 4.949 | 92.3 | 66.5 |
| 60-90 | 112 | 22.1 | 2.196 | 5.572 | 109.0 | 76.2 |
| 90-120 | 96 | 25.7 | 1.886 | 4.845 | 94.3 | 67.6 |
| 120-150 | 80 | 25.8 | 1.106 | 3.869 | 78.0 | 57.7 |

Higher budgets coincide with more reserved work and more streaming sequences. Rejection is not monotonic across stages, and client tail latency is confounded by the event gaps. No throughput benefit from increasing budget is established.

| Applied s | Budget change | Cost at update | Time until cost ≤ new budget s |
| --- | --- | --- | --- |
| 30.058 | 80 → 96 | 79.28 | 0.000 |
| 60.021 | 96 → 112 | 94.72 | 0.000 |
| 90.021 | 112 → 96 | 110.57 | 0.393 |
| 120.027 | 96 → 80 | 94.25 | 0.317 |

The short downward-step delay measures reservation release until cost first fits the new limit, not latency settling. Lowering the budget preserves running requests; new requests are rejected until their projected cost fits. Current admitted cost is the original cost reservation held until completion, not remaining GPU work.

![Budget and gateway arrival behavior](budget_diagnosis.png)

## Output cap and workload selection

| Workload | Reject % | Output cap | Mean actual output | Mean reserved cost |
| --- | --- | --- | --- | --- |
| long_context | 21.7 | 128 | 128.0 | 1.405 |
| long_output | 51.6 | 512 | 93.9 | 2.943 |
| medium | 15.9 | 128 | 128.0 | 0.996 |
| short_interactive | 8.4 | 64 | 64.0 | 0.610 |

long_output does not generate 512 tokens on average; natural EOS ends it much earlier. Admission still reserves its 512-token cap. This is evidence of cap/realized-length mismatch and unequal rejection by class. It does not establish true GPU cost, and successful-output distributions are selected by admission. Rejected requests have no realized token usage. Do not replace output caps with future actual output lengths during online evaluation.

## Candidate MPC signals

Signal correlations use only snapshot features available at snapshot time and the P90 latency of successful client starts in a future 5-second window. Lead is the delay before the label window starts. Correlations are descriptive: windows overlap, samples are not independent, only one fixed arrival/mix schedule exists, and phase/time trends confound the results. Phase demeaning removes phase means, not all confounders.

| Feature | 1-second lead client P90 r | Within-phase r |
| --- | --- | --- |
| current_admitted_cost | 0.154 | 0.094 |
| awaiting_first_content | -0.087 | -0.101 |
| streaming_requests | 0.185 | 0.140 |
| oldest_awaiting_first_seconds | -0.088 | -0.099 |

No signal is validated for out-of-sample MPC prediction by these correlations. Awaiting/oldest age may react to an ongoing slowdown rather than warn before it; gateway states cannot see requests still delayed before handler receipt. Cost and streaming counts carry the controlled operating level but are not independent measurements of GPU work. Estimated prompt/output features for rejected traffic are absent from gateway events, so total offered token pressure cannot be recovered exactly from this telemetry alone. Exact-second state reconstruction is an offline view of lifecycle history, not proof that an external controller could fetch fresh state during a gateway stall. Snapshot state counts match lifecycle reconstruction for all 166 snapshots, with maximum cost difference 1.71e-13 (floating-point rounding).

## Next experiment

1. Add gateway event-loop lag, process CPU/RSS and GC pause measurements; keep the real workload and original trace as the baseline. Run an isolated repeated A/B of per-request versus lifespan-owned upstream HTTP clients, keeping pool limits and all other workload/runtime choices documented and controlled. Preserve every request and include failures.
2. After diagnosing the path, collect repeated budget excitation with non-periodic step order/hold durations, and separately vary RPS and request mix. Keep a held-out run for prediction validation; evaluate 1–5-second forecasts against persistence. Identify a usable operating range before enabling MPC.
3. Add online summaries of total offered prompt tokens/output caps before admission (including rejected requests), and define the latency SLO before calculating SLO goodput. Gateway first-content state is a proxy, not actual prefill/decode queue state.
4. Start control with a 1-second interval only as a hypothesis. Current snapshot intervals drift and can be stale; use actual dt and a stale-data fallback. Run shadow mode, measure controller overhead, then compare fixed budget and simple feedback against MPC on identical traces and repeated trials.

## Reproduce

```bash
python analyze_budget_dynamics.py results/pilot-02 --output-dir results/pilot-02-analysis
```

Requires numpy and matplotlib. Artifacts: requests.csv (ID-joined latency decomposition), aligned_seconds.csv (state at second boundaries and explicitly future labels), stages.csv, workloads.csv, event_gaps.csv, budget_responses.csv, signal_correlations.csv, summary.json, and two figures. Missing numerical values are blank in CSV. All raw input hashes are included in summary.json; the script refuses incomplete lifecycle data. This report is one-run diagnosis, not MPC performance evidence.
