# First budget step pilot

This is a system-identification **pilot**, not a comparison proving MPC superiority.
It changes one gateway budget under fixed, deterministic heterogeneous traffic.
It uses the existing four workload shapes and allows natural EOS, as the final V1
heterogeneous experiments did. No controller is implemented here.

## Before renting a GPU

The orchestration and analysis tests can run on CPU:

```bash
python -m pip install fastapi httpx pytest uvicorn matplotlib
python -m pytest -q
```

The integration test starts a real gateway and a mock vLLM HTTP server, with a
stub tokenizer only in its isolated subprocess environment. It validates files,
request correlation, control calls, shutdown, and rendering; its latency values
are NOT GPU benchmark data. Ports 8000/8080 must be free for that test.

## On the rented machine

Use the same RTX 3090, Qwen model and vLLM configuration as V1. Activate the same
Python environment used by vLLM so the real tokenizer and model cache are available.
Keep vLLM running in another terminal:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 --max-model-len 8192
```

Install the lightweight additional plotting dependency if needed:

```bash
python -m pip install matplotlib
```

Do NOT start a separate gateway. The runner owns a single gateway process on 8080
and generates its admin key internally. The runner, gateway and telemetry file
must live on the same host (analysis uses their shared monotonic clock).

Start with a short, low-load smoke run:

```bash
python run_budget_pilot.py --output-dir results/pilot-smoke \
  --rps 1 --budgets 80 96 --step-seconds 5 --warmup-seconds 8
```

If `report.json` is valid and there are no failures, run the full pilot:

```bash
python run_budget_pilot.py --output-dir results/pilot-01
```

Defaults: low-load warmup at 1 RPS for 15 seconds, drain warmup requests, then
30 RPS continuously across budgets **80 → 96 → 112 → 96 → 80**, 30 seconds each.
After the 150-second arrival window, drain requests (maximum 120 seconds), collect
a post-drain snapshot, and gracefully stop only the gateway started by this run.
The vLLM process stays running. Each output directory must be new.

## Outputs

- `manifest.json`: settings, package versions, GPU/driver metadata when available,
  git revision, run ID, timing bounds, initial/final metrics and collection status.
  Record the exact vLLM launch command separately; the manifest does not discover
  all vLLM scheduler settings or prove a clean working tree.
- `warmup.json`: excluded from measured latency summaries.
- `client.json`: measurement requests, IDs, outcomes, scheduling lag and latency.
- `budget_changes.json`: scheduled control times and acknowledgement timings.
- `gateway.jsonl`: request lifecycle, actual budget changes and snapshots.
- `gateway-process.log`: startup/runtime diagnostics.
- `report.json`: integrity checks and whole-window descriptive statistics.
- `timeseries.png`: actual budget/cost, pending/streaming requests, interval rates,
  per-request gateway TTFT and oldest pending age. Dotted line ends arrivals;
  data to the right show draining. A snapshot interval crossing warmup/measurement
  boundary is omitted from the rate plot.

The key is not saved to the manifest. Existing gateways are never stopped. Failure
preserves partial client/control files and marks the manifest failed; forced
shutdown makes the run invalid. Keep all files together when downloading them.

## Read the outcome

Validation checks unique client IDs, matching gateway events, one terminal outcome
per request, expected measurement count, budget sequence, final drain state,
telemetry losses and tokenizer fallback. The arrival scheduler is open loop;
inspect reported scheduling lag to detect a client that cannot sustain the load.
Inspect actual control timing too; a delayed HTTP update is not an on-time step.

`valid: true` means data integrity passed, not that 30 seconds reached steady state,
that admission was binding, or that the controller is useful. No rejection and cost
far below all budgets means the input did not meaningfully excite admission; adjust
the offered load after inspection. If responses have not settled, extend segment
length. The full-window P99 mixes different operating points and must not be used
as a per-budget causal comparison. Low-load warmup verifies readiness but does not
guarantee thermal or overloaded steady state.

Copy the result folder before releasing the rental. Analysis is CPU-only:

```bash
python analyze_budget_pilot.py results/pilot-01
```

This is one exploratory run. Repeats, randomized/persistently exciting inputs,
held-out workloads and baseline comparisons are later experiment stages.
