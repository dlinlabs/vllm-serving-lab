# Runner GC and event-loop diagnosis

This opt-in capture investigates client launch delays; it does **not** fix them
or change admission, budget, the trace, GC policy, output cap or the 50 ms gate.
Use `--runner-diagnostics` with the existing `--gateway-diagnostics
--sampler-mode process` configuration. No new dependencies are required.

## New evidence

- `runner_observations.jsonl`: process/run/source identity, GC generation and
  monotonic spans, callback thread identity and thread CPU time, 10 ms loop
  monitor due/resume timestamps, and approximately one-second process/loop CPU
  counters. These observe the original request runner, not the external sampler.
- `runner_observations_status.json`: closed status, counts, callback loss and
  unmatched boundaries, write/drop/task errors. The existing bounded GC inbox
  is reused; GC callbacks append primitives and never write files. JSON batches
  are written by a thread, warmed before measurement. Observer overhead is not
  assumed negligible, and that writer still shares the runner's process/GIL.
- `report.json.runner_diagnostic_integrity`: independent evidence gate. Missing,
  damaged, incomplete or foreign observations fail this gate and CLI exit status
  without rewriting client/load/control outcomes. Coverage includes measurement
  and request drain. Cleanup is inside the runner's existing cancellation shield.
- Offline `summary.json.runner_diagnostics` aligns **every** launch over the
  original threshold, including rejections, with GC spans and monitor overdue
  windows. Invalid evidence produces unknown/null attribution, not zero delay.

Legacy captures with no runner diagnostics retain their original report fields.
New failures must be preserved; do not delete late rejections or relax limits.
Loop delay with no GC overlap still does not distinguish synchronous runner
work from OS scheduling/cgroup throttling. No OS scheduler trace is collected.

## Rental host: first verify the new revision

Use native Linux x86_64 / CPython 3.12. Record `git rev-parse HEAD`; preserve the
acceptance directory from the previous revision too. For this unmerged change:

```bash
cd /workspace/vllm-serving-lab
git fetch origin feat/runner-diagnostics-20261015
git switch --detach FETCH_HEAD
git rev-parse HEAD
```

Before starting vLLM/gateway (ports 8000/8080 must be free), run the exact-revision
acceptance in a separate CPU environment:

```bash
python3.12 -m venv .venv-cpu-tests
.venv-cpu-tests/bin/python -m pip install -r requirements-cpu-tests.txt
nohup .venv-cpu-tests/bin/python -u scripts/run_linux_cpu_tests.py \
  > /workspace/runner-native-cpu.log 2>&1 < /dev/null &
tail -n 30 /workspace/runner-native-cpu.log
```

Full native acceptance requires `Native CPU acceptance passed` for this exact
revision. An older commit's pass does not validate this change. The full suite includes synthetic loop blocking,
actual GC, cancellation, loss/corruption, plus runner-to-auditor integration;
the unchanged five sampler configurations remain required. No GPU/model is
needed for those tests. If the backend is already running, the focused
in-process CPU check below can run without stopping it, but is **not** full native
acceptance. After that check passes, the smoke/long diagnostic captures below
may investigate this instrumentation; keep full native acceptance explicitly
pending, and do not label that route a completed policy-comparison acceptance:

```bash
.venv-cpu-tests/bin/python -m pytest -q tests/test_runner_diagnostics.py
```

## Start the unchanged backend

From the repository directory, first-time setup creates `.venv`, installs the
V1 runtime, and checks NVIDIA GPU/driver, CUDA and the model context length:

```bash
cd /workspace/vllm-serving-lab
nohup python3.12 -u /workspace/vllm-serving-lab/scripts/setup_v1.py \
  --disable-prefix-caching > /workspace/vllm-setup.log 2>&1 < /dev/null &
tail -n 30 /workspace/vllm-setup.log
```

Wait for READY and verified `max_model_len=8192`. On later starts of an already
installed environment add `--skip-install`; do not launch a second backend.
Keep the complete `results/setup-*` launch provenance.

## Reuse byte-identical prepared inputs

The repository already contains the same traces used by the October 15 capture.
There is no need to redownload or regenerate prompts for this diagnostic run:

```bash
sha256sum results/trace-smoke-96/trace.json results/trace-experiment-96-r1/trace.json
```

Expected smoke SHA-256:
`9998c82da5174ec41484985f7b5cf77a658992cee5011fc38bb7155caf065294`.
Expected experiment SHA-256:
`62f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e`.
The runner also verifies prompts against the cached tokenizer before measurement.

Run smoke first in a fresh output directory:

```bash
nohup .venv/bin/python -u run_trace_pilot.py \
  --trace results/trace-smoke-96/trace.json \
  --output-dir results/runner-diag-smoke-96-r1 \
  --budget 96 --ttft-slo 0.5 --max-scheduling-lag 0.05 \
  --cache-policy disabled --gateway-diagnostics \
  --sampler-mode process --runner-diagnostics \
  > /workspace/runner-diag-smoke-96-r1.log 2>&1 < /dev/null &
```

Inspect its `report.json`: client integrity, load fidelity, control fidelity,
gateway/external diagnostic integrity, and **runner diagnostic integrity** must
all pass. Then run one five-minute diagnostic capture; never run concurrently:

```bash
nohup .venv/bin/python -u run_trace_pilot.py \
  --trace results/trace-experiment-96-r1/trace.json \
  --output-dir results/runner-diag-fixed-96-r1 \
  --budget 96 --ttft-slo 0.5 --max-scheduling-lag 0.05 \
  --cache-policy disabled --gateway-diagnostics \
  --sampler-mode process --runner-diagnostics \
  > /workspace/runner-diag-fixed-96-r1.log 2>&1 < /dev/null &
```

```bash
cat results/runner-diag-fixed-96-r1/report.json
.venv/bin/python analyze_trace_dynamics.py \
  results/runner-diag-fixed-96-r1 --output-dir results/analysis-runner-diag-96-r1
```

If load fidelity fails, retain this capture and inspect the runner overlaps;
do not repeat until one passes. This is a diagnostic capture, not evidence of
MPC benefit or a qualified budget comparison. Gateway GC may remain a realistic
plant disturbance; it need not be eliminated to identify a controller.

Before returning the host, download this archive:

```bash
tar -czf /workspace/runner-diag-01.tar.gz \
  cpu-ci-artifacts results/setup-* results/runner-diag-* results/analysis-runner-diag-96-r1
```
