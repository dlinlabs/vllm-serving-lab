# Receipt-state calibration probes

The native 4c89188 clock run and 26e01e4 phase-margin development remain rejected.
This collection supplies missing state evidence; it does not claim to repair
queue accuracy using previously held outcomes.

## Frozen acquisition

Retain all original cases, pinned Qwen revision, B2048, seq32 and existing numeric
budgets. First collect the original 16 LIGHT warmups and 48 TRAIN OFF/LIGHT
references with new qr-clock- IDs/content and no receipt recorder. Save the 24
TRAIN LIGHT event records. This permits offline same-host clock fitting without
another rental. A failed fit cannot prevent acquisition of diagnostic probes.

Then collect 168 probes at the original 50/200/400/800 ms offsets × D0/D16:
24 warmups followed by six blocks of 24. For each case/block, three arms share
exact token IDs: baseline OFF, receipt-only OFF, LIGHT plus receipt. All six arm
permutations are used across the scored blocks; case traversal reverses on
alternate blocks. Blocks use distinct token namespaces disjoint from ct-/qp-.

Total: **232 trials, 40 warmups, 48 TRAIN references, 144 scored probes**.
There are 48 receipt/LIGHT pairs. Every sample here is calibration/development
data; none is independent model confirmation. The protocol is saved before
trials and source-bound by full native CPU acceptance. The collection deadline
is 1800 s after startup, with the existing separate startup limit, per-trial 45 s
limit and additional bounded cleanup/archival time. No automatic retries occur.

## Measurements and controls

An opt-in EngineCore-thread recorder observes CPU schedule-return and
update-from-output-return boundaries, the foreground release and sentinel
pre-ADD state. It tracks the same outstanding SchedulerOutput object until
commit and serializes ordinal IDs. No GPU event, synchronization or scheduler
replacement is added. The existing v0.30.0 core.py source pin covers this path.

Record relative schedule/commit progress, pending batches, running/waiting order,
computed/token/placeholder counts, original queue-post/pre-ADD times and snapshot
duration. The report includes preceding/following CPU events around queue-post.
These are not exact input-poll timestamps or physical GPU completion. As in the
existing collector, enqueue is sampled before request construction/add, not at
add return. The baseline OFF phase remains unobserved. A receipt is after queue
wait; it is explanatory evidence and never a pre-arrival forecast input.

Report all per-arm min/median/max/ranges and state counts. Matching normalized
receipt-state pairs retain max(10 ms,20% receipt-arm residence). Different-state
pairs stay unassessable, with counts. Report coverage separately for each cell;
zero comparable pairs in any cell blocks the whole same-state control verdict. Equal
observed state does not prove equal physical phase. Baseline/receipt and
baseline/LIGHT endpoint controls use original observer limits independently.
Six repeats do not establish statistical coverage or identify perturbation's
physical cause. No median between modes can hide a sample failure.

Overflow, unmatched/reordered commit, invalid chronology, missing/duplicate
receipt, posting lag, incomplete outputs, dispatcher drift or wrong background
lifetime reject integrity. LIGHT events must align with its existing step rows.
Observer exceptions invalidate evidence without swallowing native exceptions.
Controls that fail remain findings and do not silently delete later samples.

The host stops the backend, audits raw-derived rows/report, then archives every
completed or failed attempt. Audit checks exact source and original full CPU
JUnit coverage, runtime equality, installed vLLM pins, manifest pairing/freshness,
TRAIN events and summary. Audit success is not model acceptance. All model,
HTTP, adaptive-cap and MPC flags remain false.

## Commands

Prepare a clean updated rental checkout:

```bash
nohup python3 scripts/setup_prefill_calibration.py --next queue-probe > ~/pr32-prepare.log 2>&1 < /dev/null &
```

After PREPARED:

```bash
nohup .venv/bin/python scripts/prefill_host.py queue-probe > ~/pr32-queue-probe.log 2>&1 < /dev/null &
tail -f ~/pr32-queue-probe.log
```

The exact `/root/pr32-queue-probes/<run-id>.tar.gz` download path is printed.
Audit a retained directory using its exact source checkout:

```bash
.venv/bin/python scripts/check_prefill_queue_probe.py /path/to/capture
```

After downloading evidence, GPU is not needed for the separate offline fit:

```bash
.venv/bin/python scripts/fit_prefill_queue_probe.py /path/to/capture --output /path/to/new-fit
```

The fitter audits first and explicitly selects only 48 TRAIN rows plus TRAIN
LIGHT events, excluding every delayed probe label. It saves a six-term training
freeze and forecasts, or a structured fit failure. Output must be outside the
immutable capture directory. A training fit is not held confirmation.

The next design is [model selection and independent confirmation](prefill-queue-confirmation-v2.md).
Do not rerun unchanged live-http to acquire these missing measurements.
