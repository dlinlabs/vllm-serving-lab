# Low-cap range/phase identification v1

[Frozen declaration](low-cap-switch-v1.json). This is **development identification/calibration**,
not an independent validation of #28, a cost-formula refit, an optimum search or MPC execution.
The earlier fixed 16/32 captures exist; bidirectional low-cap transients and phase-swapped repeats
do not. Collect on one exclusive Linux NVIDIA rental host with one unchanged backend.

## Question and design

Does the useful operating region lie below the earlier 48–96 range? Does a cap effect persist
when cap assignments exchange positions on the same arrival trace? How long does admitted work
remain after lowering the cap? Use the existing full 300 s / 7,643-arrival trace, unchanged input
lengths/timing and output cap 128; source output lengths are not replayed.

| Arrival seconds | 0–60 | 60–120 | 120–180 | 180–240 | 240–300 |
|---|---:|---:|---:|---:|---:|
| A | 16 | 32 | 16 | 32 | 16 |
| B | 32 | 16 | 32 | 16 | 32 |

Each cap gets equal total exposure across A+B at every arrival phase. Keep the V1 cost formula
0.259 + 0.000758 I + 0.0052083333 O fixed. Lowering the cap does not cancel admitted work: if
reserved cost exceeds the new cap, existing work drains before more can enter.

Frozen execution order (one full command, no manual cap edits):

1. Smoke: fixed 32, 30 s.
2. Fixed 32 reference, 300 s.
3. A repeat 1, 300 s.
4. B repeat 1, 300 s.
5. Fixed 16 reference, 300 s.
6. B repeat 2, 300 s.
7. A repeat 2, 300 s.

The A/B execution order reverses on repeat (ABBA) and their mean ordinal positions match.
This balances linear order drift only, not nonlinear drift or all carryover. Fixed references
have one run each and are descriptive context, not independently replicated causal controls.
The four switching runs contain eight 16→32 and eight 32→16 changes. Each hold lasts 60 s to
allow observation beyond the several-second request lifetimes seen previously, but this is
**not a guarantee that state settles**.

Same GPU UUID, backend/API/engine processes, package versions, source commit and protocol bytes
are checked throughout. Every replay uses a fresh gateway, three sequential warmups, acknowledged
initial cap and three consecutive idle engine samples before/after. Prefix caching is disabled;
Qwen context is explicitly 8192. Draining does not reset all allocator, clock or cache state.
Shared upstream HTTP client, gateway/runner diagnostics, process sampler and bounded runner GC
deferral remain enabled. Load/control lag limits stay **50 ms**. Any gate failure stops the series
and preserves evidence; no automatic retry/resume or threshold relaxation.

## Rental-host steps

Use the collection branch/commit linked by its PR, based on main; the #26/#28 modeling branches
are not required. Fetch and check out that exact revision **before** native CPU acceptance.
Do not edit tracked scripts/protocol during collection. The scripts use the repository `.venv`
directly, so no shell activation is required.

```bash
cd /workspace/vllm-serving-lab
git fetch origin experiments/low-cap-switch-v1
git switch --detach origin/experiments/low-cap-switch-v1
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3.12 --version

python3.12 -m venv .venv-cpu-tests
.venv-cpu-tests/bin/python -m pip install -r requirements-cpu-tests.txt
nohup .venv-cpu-tests/bin/python -u scripts/run_linux_cpu_tests.py \
  > /workspace/low-cap-cpu-tests.log 2>&1 < /dev/null &
```

Wait for `Native CPU acceptance passed` in `/workspace/low-cap-cpu-tests.log`. The CPU tests
need ports 8000/8080; complete them before starting vLLM. Symlink venvs may need those exact
two local paths in `.git/info/exclude`; keep code/protocol visible to Git.

```bash
cd /workspace/vllm-serving-lab
nohup python3.12 -u scripts/setup_v1.py --disable-prefix-caching \
  > /workspace/low-cap-vllm-setup.log 2>&1 < /dev/null &
```

Add `--skip-install` only if this machine already has the pinned V1 `.venv`. Wait for `READY`:
the setup checks NVIDIA/Python/CUDA, installs packages, fixes child PATH for ninja, starts Qwen
and verifies `max_model_len=8192`. Reuse an existing correctly launched backend; do not launch
two. Keep the same backend for the whole suite.

Inspect the plan (no GPU requests or output writes):

```bash
.venv/bin/python run_cap_sweep.py --protocol experiments/low-cap-switch-v1.json \
  --stage full --output-dir results/low-cap-switch-v1 --plan-only
```

Run the complete collection, including smoke:

```bash
nohup .venv/bin/python -u run_cap_sweep.py \
  --protocol experiments/low-cap-switch-v1.json \
  --stage full --output-dir results/low-cap-switch-v1 \
  > /workspace/low-cap-switch-v1.log 2>&1 < /dev/null &
```

Expect **30 minutes of full-trace measurement**, plus smoke, warmups, drain and checks. If a
separate smoke is desired, use `--stage smoke` with a fresh output name; it is not reused by full.

```bash
tail -n 25 /workspace/low-cap-switch-v1.log
cat /workspace/vllm-serving-lab/results/low-cap-switch-v1/suite.json
pgrep -af '[r]un_cap_sweep.py|[r]un_trace_pilot.py'
```

A report appears after a capture finishes; an initially empty log is not evidence of failure.
On reconnect, use these commands; avoid starting a duplicate suite. Existing outputs are refused.

## Analysis and what the evidence can establish

The analysis re-audits all original capture reports in private temporary outputs, verifies raw
file hashes, host/code/packages and collection binding, and refuses a partial/failed suite.
It needs no GPU and does not overwrite raw files or fit any model.

```bash
.venv/bin/python analyze_low_cap_switch.py results/low-cap-switch-v1 \
  --output-dir results/analysis-low-cap-switch-v1
```

Outputs: `summary.json`, `report.md`, `phase-response.png`.

- Primary: successful complete streams with first content ≤0.5 s from **scheduled arrival**,
  divided by all arrival seconds. Include rejected/failed requests and every hold's first 15 s.
- Compare cap32 minus cap16 at identical offered-arrival phases in A/B, separately by repeat.
  Same arrivals do not mean identical engine state/history, so these are descriptive comparisons,
  not isolated causal treatment effects.
- Separate first 15 s and remaining 45 s as prespecified early/late descriptions. Do not treat
  15 s as a measured settling time or optimize this exclusion after seeing results.
- Record 5 s arrival cohorts, and A2−A1 / B2−B1 at the same cap and phase to contextualize variation.
  Adjacent bins are correlated, not independent experiments; two repeats do not identify a noise
  floor or support a strong significance claim.
- Inspect dynamics/lifecycle/engine telemetry afterward to explain delayed response. A lower cap
  winning supports the low operating-region hypothesis on this workload, not a universal optimum.
- All this new data is development/calibration data. Save pre-fit #28 as a diagnostic comparator;
  if the model is revised using these runs, independent acceptance needs separate future data.
  No #28 retraining or forecast acceptance is performed by this collector/analyzer. C stays unused.

Before returning the GPU/storage, retain the entire suite, setup and native evidence:

```bash
cd /workspace/vllm-serving-lab
tar -czf /workspace/low-cap-switch-v1-evidence.tar.gz \
  results/low-cap-switch-v1 results/setup-* cpu-ci-artifacts
sha256sum /workspace/low-cap-switch-v1-evidence.tar.gz
```

Download/upload and confirm retention before releasing storage. Retain partial failures too.
