# Effective async endpoint calibration pilot

Status: implementation ready for its first native GPU qualification. CPU tests
and synthetic recovery do not establish real accuracy. This is a separate,
reduced candidate, not completion of the 21-channel physical stage estimator.
The old serial overlap guard and the full-suite statistical protocol are intact.

## What is identified

The prior diagnostic cannot separate the 21 physical CPU/GPU service components.
This pilot instead fits four **effective batch-service** coefficients to request
endpoints: bias, prefill tokens /2048, decode context /32768 and causal attention
pairs /2097152. Parameters are seconds per normalized feature. Their placement
in forward.gpu_tail is an adapter to the existing simulator's single effective
service lane, NOT a claim of exclusive forward GPU duration. Other modeled
service components are zero. CPU enqueue, sampling, copy and commit costs are
collapsed, and their independent overlaps are not recovered. A held endpoint
failure is evidence against this reduction; do not remove that failure by
reinterpreting coefficients as physical stage costs.

The simulator retains its own request progress, native two-slot submit-first/
FIFO-reap sequence, first-background-output barrier, KV bounds and conditional
1ms idle yields. It reads no observed future schedule, release state, graph mode
or timestamps. Native work is compared only after independent prediction.
Measurements are EngineCore queue-post to first output commit (TTFT), and
queue-post to forced final output commit (completion). These exclude HTTP,
frontend detokenization and gateway latency. Background timestamps are not fit
labels. No overlapping wall interval or wait-inclusive copy event is summed.

## Frozen scope and order

The machine-readable plan is `prefill-effective-calibration-v1.json`. It binds
Qwen/Qwen3-4B-Instruct-2507 at cdbee75f17c01a7cc42f958dc650907174af0554,
vLLM0.30.0/Torch2.13.0+cu130, RTX4090, V2/same-process UniProcExecutor,
TP/PP/DP1, B2048, seq32, context8192, two slots, no prefix/speculation/KV transfer.
The existing 38-file installed-source guard and clean exact-source native CPU
acceptance remain required. No runner/eager/async override is introduced.

The first pilot deliberately tests only the 14 zero-offset cells. Six training
cells are single1024/single7168/uniform4x4096 at D0/D16. Eight held cells are
single2048/single4096/mixed/reverse at D0/D16. Each cell has four fresh OFF/LIGHT
pairs with matched exact token content, balanced pair direction and a fixed
randomized case order within blocks. It is a conditional pilot, not a delayed-
arrival or production trace claim. The sentinel cells remain in the original
protocol and are excluded here in advance, not removed after observing failure.

138 trials execute once on one loaded engine:

1. 14 LIGHT warmups, unscored.
2. 48 training trials: six cells x four OFF/LIGHT pairs.
3. Fit only the 24 OFF training endpoint sets, gate training quality and own-work
   agreement, write frozen.json/costs.json and pre-held-predictions.json.
4. 64 fresh held trials: eight cells x four OFF/LIGHT pairs.
5. 12 fresh held-reference trials: one OFF/LIGHT pair per training cell, used for
   post-freeze drift/error checks only. Never refit from these labels.
6. Recheck runtime identity and produce held-report.json and calibration.json.

Training and held phases are sequential to make the freeze unambiguous. Fresh
training-cell references check drift after the held phase. This does not establish
stationarity between every trial or independent errors. No deep profiler or
extra seven-stage timers run. OFF still executes the private control hooks and
some scheduler inspection; OFF/LIGHT pairs estimate incremental LIGHT overhead,
not zero-instrumentation production overhead.

## Predeclared pilot rejection rules

Training uses bounded nonnegative least squares on **end-to-end causal rollout**,
not a regression on captured intermediate states. Residuals are relative to each
observed endpoint (with 5ms TTFT/10ms completion floors), normalized by request
count so uniform four-request trials do not dominate single-request trials.
Each training case has the same four repeats. Two fixed optimizer starts,
60 evaluations per start, bounds [1e-7,0.5] seconds per normalized parameter.

Both starts must converge and agree within5% (denominator floor1us); parameters
cannot hit bounds. The column-normalized local finite-difference sensitivity
matrix must have rank4 and condition<=1000. This is a local numerical check,
not proof of global identifiability or confidence intervals. Basis order is
explicit and cannot change when JSON keys are sorted.

For every foreground request/metric, OFF repeat max-minus-min must be <=max(
5ms TTFT or10ms completion,20% of its median). Every matched LIGHT/OFF difference
must be <=max(2ms,10% of OFF latency). Training and held median prediction errors
must each be <=max(5ms TTFT or10ms completion,20% of measured median), per cell
and request; no aggregate average can hide one failing request. Training and
held LIGHT work sequences must match independently generated ordered request
IDs/positions/counts, and native scheduler/dispatcher audits must pass. D16
backgrounds must outlive foregrounds both natively and in the prediction.

These engineering thresholds are preregistered for this separate feasibility
pilot. They do not replace PR32's original structural-group, noise, comparator,
HTTP-transfer or cap/MPC acceptance tests. `candidate_passed` applies only to
this fixed runtime/workload pilot; `model_accepted`, physical-cost identification,
complete-experiment, adaptive-cap and MPC acceptance stay false.

Any incomplete/mismatched capture, invalid labels, unresolved events, failed
source/runtime/quality/sensitivity gate or deadline stops with retained evidence.
No automatic retry, best-run selection, threshold adjustment or cross-run stitch.
The active pilot is bounded to1800s after server startup, including fitting;
server startup has its separate900s bound. Installation/download/CPU acceptance
are separate and can take longer. GPU is held during CPU fitting so the held
phase uses the same engine. This is a first native attempt, so no measured runtime
estimate or guaranteed success is advertised.

## Rental-host commands

New native Linux/Python3.12 RTX4090 host, after cloning the PR32 branch:

```bash
cd ~/vllm-serving-lab-pr32
nohup python3 scripts/setup_prefill_calibration.py > ~/pr32-prepare.log 2>&1 &
tail -f ~/pr32-prepare.log
```

The helper creates/reuses .venv, installs the pinned runtime plus CPU dependencies
and ninja, checks pip/CUDA/source files, pre-pins the model revision and delegates
to exact-source `prefill_host.py prepare`. It does not start experiments.
After `PREPARED` appears, Ctrl+C exits only tail, then:

```bash
nohup .venv/bin/python scripts/prefill_host.py calibrate > ~/pr32-calibration.log 2>&1 &
tail -f ~/pr32-calibration.log
```

An already installed/updated host can run prepare then calibrate directly. The
calibration command generates its own token manifest and explicitly enables the
calibration endpoint plan. Normal collect/diagnose/stages retain their own plan.
No gateway/Azure replay runs in the pilot. Inherited calibration flags are reset
by normal launches. Progress prints trial/total, split, case and observer.

Outputs: ~/pr32-calibrations/<timestamp>/ plus a sibling <timestamp>.tar.gz,
automatically packed even after a failed collection. Preserve/download the
archive before returning the machine. No automatic upload is performed.

- `collection_complete=true`: all138 trials plus final analysis completed.
- `candidate_passed=true`: this pilot's quality, own-work and endpoint gates pass.
- A complete but failing candidate returns exit2; invalid/incomplete runs return
  nonzero with the original error and retained partial captures. The wrapper's
  CalledProcessError is secondary; inspect calibration.json/held-report.json.

Independent offline audit on the exact recorded checkout:

```bash
python scripts/check_prefill_effective_calibration.py audit /path/to/capture
```

It re-extracts raw endpoints, checks work/dispatcher, installed source pins,
manifest/runtime/CPU-source identity, training hashes, frozen costs/predictions
and recomputes the held report. This detects inconsistent retained artifacts;
hashes are not signatures or protection against deliberate forgery. The original
native acceptance JSON, JUnit and pytest log are copied into the capture archive
and the offline audit validates the original JUnit coverage.

CPU-only noiseless recovery (saved metadata supplies scope/dispatcher only):

```bash
python scripts/check_prefill_effective_calibration.py demo --runtime /path/to/runtime.json --output /tmp/new-demo
```

The first native pilot is the next GPU-dependent step. Passing it would justify
fresh delayed-arrival/mode/support and HTTP validation, then fixed-cap versus
state-aware-cap tests. It does not enable a controller or justify MPC deployment.
