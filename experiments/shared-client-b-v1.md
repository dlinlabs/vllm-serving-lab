# Shared-client B on a fresh rental host

This is a development collection declaration, not model or MPC acceptance.
Keep the existing qualification A capture and its declaration unchanged. Run one
new fixed-96 smoke, then one B capture if every enabled smoke gate passes. Stop
and retain artifacts on any failure; do not retry until a passing run appears.
C remains unused. No held-out inputs need to be prepared for this collection.

Use the exact collection runtime commit
`d0ccb2c0d3711e1b2a69bb4ec248c61961ce3586`. The JSON declaration is delivered
as an external file so a documentation-only commit does not replace the runtime
whose full native suite was validated. The runner accepts an absolute protocol
path and records its hash. The `collection_runtime` metadata is an operator
requirement, not a new automatically enforced check. Verify a clean exact runtime
checkout, run native CPU acceptance on this host, and archive the actual host,
backend launch and dependency records.

After cloning the repository and checking out that runtime commit:

```bash
cd /workspace/vllm-serving-lab
python3.12 -m venv .venv-cpu-tests
.venv-cpu-tests/bin/python -m pip install -r requirements-cpu-tests.txt
nohup .venv-cpu-tests/bin/python -u scripts/run_linux_cpu_tests.py \
  > /workspace/shared-b-cpu-tests.log 2>&1 < /dev/null &
```

Keep ports 8000 and 8080 free until `Native CPU acceptance passed` appears in
that log. If the suite fails, stop. Then start the backend:

```bash
cd /workspace/vllm-serving-lab
nohup python3.12 -u scripts/setup_v1.py --disable-prefix-caching \
  > /workspace/shared-b-vllm-setup.log 2>&1 < /dev/null &
```

This creates `.venv`, installs the recorded vLLM/Torch core versions, sets the
child PATH, checks CUDA, and starts Qwen with max context 8192 and prefix cache
disabled. Wait for `READY` before replay. Keep the same backend launch for smoke
and B; archive the corresponding `results/setup-*` directory.

Fetch this declaration without switching the runtime checkout:

```bash
git fetch origin experiments/shared-client-b-v1
git rev-parse FETCH_HEAD > /workspace/shared-client-b-protocol-source.txt
git show FETCH_HEAD:experiments/shared-client-b-v1.json > /workspace/shared-client-b-v1.json
sha256sum /workspace/shared-client-b-v1.json
```

The expected SHA256 is
`6610d3839bb16db0c8b18e90dc11718d9adc2ae7f56ed1be82bbe024e6a4625f`.
Stop if the hash differs. The traces and B schedule already exist in the runtime
checkout; do not regenerate them. The runner checks their declared content.

Run the fresh-host smoke once:

```bash
nohup .venv/bin/python -u run_trace_pilot.py \
  --trace results/trace-smoke-96/trace.json \
  --output-dir results/shared-client-b-smoke-96-r1 \
  --budget 96 --ttft-slo 0.5 \
  --max-scheduling-lag 0.05 --max-budget-lag 0.05 \
  --cache-policy disabled --gateway-diagnostics --sampler-mode process \
  --runner-diagnostics --runner-gc-policy defer --runner-max-rss-mib 1536 \
  --shared-upstream-client \
  --protocol /workspace/shared-client-b-v1.json \
  --protocol-run shared-client-b-smoke-96-r1 \
  > /workspace/shared-client-b-smoke-96-r1.log 2>&1 < /dev/null &
```

Wait for completion. Inspect `results/shared-client-b-smoke-96-r1/report.json`;
`valid` and the `valid` fields of `load_fidelity`, `control_fidelity`,
`diagnostic_integrity`, `runner_diagnostic_integrity`, and `runner_gc_integrity`
must all be true. Only then run B once, with no other replay running:

```bash
nohup .venv/bin/python -u run_trace_pilot.py \
  --trace results/trace-experiment-96-r1/trace.json \
  --output-dir results/shared-client-b-r1 \
  --budget-schedule results/budget-schedules/ident-multilevel-30s-v1-b.json \
  --ttft-slo 0.5 --max-scheduling-lag 0.05 --max-budget-lag 0.05 \
  --cache-policy disabled --gateway-diagnostics --sampler-mode process \
  --runner-diagnostics --runner-gc-policy defer --runner-max-rss-mib 1536 \
  --shared-upstream-client \
  --protocol /workspace/shared-client-b-v1.json \
  --protocol-run shared-client-b-r1 \
  > /workspace/shared-client-b-r1.log 2>&1 < /dev/null &
```

The B sequence is `96,64,48,96,80,48,64,80,64,96`, with 30-second holds.
Apply the same six gates to B. A passing capture supplies development model data;
it does not establish model quality or MPC improvement. Use shared-client A/B for
this series, keeping legacy per-request captures as diagnostic history. The new
rental may differ in driver, CPU and contention: record those differences, and do
not interpret reversed A/B aggregate scores as an isolated causal cap comparison.

Before returning the machine, preserve both complete run directories, native CPU
evidence, the exact external protocol and its source commit file, setup records,
and all outer logs. Offline analysis does not require retaining the GPU.
