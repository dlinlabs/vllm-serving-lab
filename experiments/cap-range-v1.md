# Same-host fixed-cap range study v1

Collection-only development declaration: [cap-range-v1.json](cap-range-v1.json).
This explores whether the previous 48–96 range missed a useful operating region
and how much full-run results vary on repeat. It does not replace the historical
excitation/shared-client declarations, fit a new model, collect C, or run MPC.

## Design

- One Linux rental host, GPU, live vLLM backend and unchanged code revision.
- Original five-minute development trace: 7,643 offered arrivals, original timing
  and input lengths, synthetic prompts, output cap 128 with natural EOS. Original
  output lengths are not replayed. No arrival stretching or thinning.
- Unchanged V1 cost coefficients: 0.259 / 0.000758 / 0.0052083333. Cap units are
  reservations, not requests or measured GPU work. New caps are exploration points.
- TTFT SLO 0.5 s; original maximum load/control lag limits 50 ms. Shared upstream
  client, gateway/runner diagnostics, external process sampler and bounded runner
  GC deferral (1536 MiB) remain enabled.
- New smoke at fixed 96, then eight **separate fixed-cap** 300 s replays:

| Block | Execution order |
|---|---|
| 1 | 64 → 16 → 96 → 32 |
| 2 | 32 → 96 → 16 → 64 |

The second block mirrors the first: each cap's two ordinal positions have the
same mean. This balances linear order drift, not arbitrary thermal/host drift or
all carryover effects; the order is prespecified, not randomized. Two repeats are
a preliminary repeatability check, not a reliable variance estimate/noise floor.
Expect 40 minutes of measurement plus smoke, warmups, drain and checks.

Each replay creates a fresh gateway. The supervisor requires three consecutive
backend waiting/running=0 samples before and after the replay; the child performs
three sequential warmups, drains them, and acknowledges its initial cap before
measurement. The backend API/EngineCore PIDs and creation times, GPU UUID/driver,
host boot ID, source revision and protocol bytes must remain unchanged. This
prevents outstanding requests from the preceding run entering the next one;
it does not reset every engine allocator/cache/clock state. Use an exclusive GPU
and no other traffic. Prefix caching must be explicitly disabled in the live
local backend command; the supervisor also verifies model/context.

## Rental-host commands

Use a checkout containing this change. If the host is already set up, keep its
venvs and model cache; do not reinstall or redownload unnecessarily. First run
native CPU acceptance on the **exact collection revision**, before starting vLLM
(the CPU integration tests use ports 8000 and 8080):

```bash
cd /workspace/vllm-serving-lab
python3.12 -m venv .venv-cpu-tests
.venv-cpu-tests/bin/python -m pip install -r requirements-cpu-tests.txt
nohup .venv-cpu-tests/bin/python -u scripts/run_linux_cpu_tests.py \
  > /workspace/cap-range-cpu-tests.log 2>&1 < /dev/null &
```

Wait for `Native CPU acceptance passed` in `/workspace/cap-range-cpu-tests.log`.
The supervisor finds an acceptance record for its exact HEAD automatically.
The native script requires a clean checkout; do not edit tracked code/protocol.
If `.venv` or `.venv-cpu-tests` is a symlink, add that exact local path to
`.git/info/exclude` rather than committing or deleting the venv.

Start Qwen from the repository directory:

```bash
nohup python3.12 -u scripts/setup_v1.py --disable-prefix-caching \
  > /workspace/cap-range-vllm-setup.log 2>&1 < /dev/null &
```

If V1 packages are already installed, add `--skip-install`. Wait for `READY`;
Qwen3-4B and `max_model_len=8192` must be verified. An already running server
launched with these exact flags can be reused. Do not start two backends.

Review the complete plan without a GPU, output files or requests:

```bash
.venv/bin/python run_cap_sweep.py --stage full \
  --output-dir results/cap-range-v1 --plan-only
```

Then launch the full suite (includes its own smoke); **one command**:

```bash
nohup .venv/bin/python -u run_cap_sweep.py --stage full \
  --output-dir results/cap-range-v1 \
  > /workspace/cap-range-v1.log 2>&1 < /dev/null &
```

For a separate smoke-only trial use `--stage smoke --output-dir
results/cap-range-v1-smoke-check`; it is not reused by the full suite.
Existing output directories are refused: no automatic resume, retry or deletion.

Monitor from any reconnected terminal; activating a venv is not required:

```bash
cd /workspace/vllm-serving-lab
tail -n 25 /workspace/cap-range-v1.log
cat results/cap-range-v1/suite.json
pgrep -af '[r]un_cap_sweep.py|[r]un_trace_pilot.py'
```

Each capture's full stdout/stderr is in the suite root as `<run-name>.log`.
On any failed child, missing gate, identity change or non-drained state, retain
everything and stop; a failed attempt is not silently replaced. SIGTERM/SIGINT
asks the owned child to clean up, not the shared backend. SIGKILL/pod loss cannot
guarantee cleanup; a non-collected suite must not be treated as complete.

## Read-only offline analysis

After all nine captures pass, analysis needs no GPU. It checks original artifact
hashes and re-runs the existing capture auditor in private temporary outputs;
it does not rewrite the raw reports. Use a fresh output outside the suite:

```bash
.venv/bin/python analyze_cap_sweep.py results/cap-range-v1 \
  --output-dir results/analysis-cap-range-v1
```

Outputs: `summary.json`, `report.md`, `goodput.png`. Primary goodput is successful
complete streams with first content within 0.5 s **of scheduled arrival**, divided
by all 300 arrival seconds. The first 30 seconds are included; no post-hoc
transient exclusion. Secondary data retain rejection, TTFT, input-size strata,
two-repeat differences per cap, and within-block differences versus cap64 using
all offered trace IDs. Thirty-second bins use scheduled-arrival cohorts, not
completion bins; their later outcomes are counted and rejected/failed requests
are not dropped. Bins overlap the same system history and are not independent
replicates. Matched arrivals do not make the underlying states counterfactual
matches. No significance test, claimed global optimum, model acceptance or MPC
benefit is produced. The old dynamic A/B remains separate evidence.

Before returning the GPU, archive the full suite plus actual setup/native records:

```bash
tar -czf /workspace/cap-range-v1-evidence.tar.gz \
  results/cap-range-v1 results/setup-* cpu-ci-artifacts
sha256sum /workspace/cap-range-v1-evidence.tar.gz
```

Download/upload the archive and confirm it is retained before releasing storage.
If a run fails, archive the partial attempt too; do not claim eight-run results.
