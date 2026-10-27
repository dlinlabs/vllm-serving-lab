# Effective batch-event clock: development and fresh confirmation

The first fresh native confirmation now passes its controlled pilot gates. See [the audited native result](../results/prefill-clock-33876f3-confirmation/report.md). Full-model/HTTP/cap/MPC acceptance remains false.

The original39d0c7d delayed pilot is complete and rejected. Its frozen model,
raw archive and max(10ms,20%) queue rule remain unchanged. This is a separately
identified model developed after examining that failure. Old delayed cases are
now development data; passing them cannot establish independent validation.

## Model and objective

Keep the existing from-empty, FCFS, B2048/seq32/context8192, two-slot predictor.
It generates its own work, modes, pending outputs, queue-consumption boundaries,
background barrier and event dependencies. No recorded future schedule, state,
mode, release time or latency enters a prediction. Source pins and serial-overlap
guard are unchanged.

The effective server lane is nonnegative:

`bias + prefill*q/2048 + decode_context*K/32768 + attention_pairs*Z/2097152
 + decode*d/16 + mixed*q*d/32768`.

The first four terms are the original basis; the two new terms represent decode
count and simultaneous prefill/decode interference. All covariates come from the
model's own scheduled work. This is an effective latency reduction, not an
identification of exclusive CPU/GPU/copy service or a causal hardware explanation.
The separate effective-event-clock-costs-v1 schema includes the mixed term;
generic PipelineCosts.from_dict intentionally rejects it rather than dropping it.
Use event_clock_calibration.costs/predict_frozen to reconstruct this candidate.

Only the original six TRAIN cases fit coefficients. Each has four OFF/LIGHT
pairs:24 OFF endpoint trials and24 LIGHT event trials. Keep OFF request/metric
relative endpoint weights from v1. Add native schedule-start and matched
output-commit event points, relative to that trial's observed foreground release;
subtract the independently predicted release from model points. Clock-origin
translation cancels. **No overlapping wall interval is added to another.**

The event window is the ordered prefix from initial background work through the
commit completing the foreground. Exclude subsequent background-only drain.
Native integrity/request/chronology checks precede extraction. Full ordered work
signatures, prefill/decode groups and foreground completion index must match the
own rollout before any event residual can be formed. Labels never provide work
or cost covariates.

For each event, normalize by max(10ms,2% of absolute time from release).
Each training trial gives equal total squared weight to its prefill and decode
phases, and equal weight to starts and commits. Endpoint and event groups have
equal normalized total weight. This prevents127 decode steps from overwhelming
a few admission-sensitive prefill boundaries. These scales are fit weights,
not replacements for any validation threshold.

Two fixed starts, positive bounds1e-7 to0.5s and80 evaluations are declared in
prefill-event-clock-v1.json. Require convergence, <=5% multistart disagreement,
local normalized sensitivity rank6/condition<=1000, no boundary coefficient,
training endpoint/work/observer/repeat gates. Keep individual optimizer histories,
training row/event hashes, runtime/plan/spec hashes and an immutable freeze.
The local rank does not prove global identification. LIGHT event timing has not
been separately qualified as uninstrumented production or exclusive service;
OFF endpoints anchor the effective fit but do not remove that limitation.

## Retained-data development

Run calibrate_prefill_event_clock.py with --capture, --archive and a fresh
--output directory. It checks archive bytes/inventory and executes the original
39d0c7d audit from verified Git sources. Its training reader opens only frozen
TRAIN file names. Fit and save frozen.json and all development forecasts before
loading retrospective delayed scoring rows. Four training-only leave-one-block-
out fits test repeat stability. The development scorer reuses the unchanged
endpoint, queue, work/lifetime, observer and separate before/after anchor rules;
it has no accepted-candidate field. It always leaves model/cap/MPC false.

The model design was informed by previously examined delayed failures. A
training-only coefficient fit does not make those delayed rows fresh again.
The accompanying report records unsuccessful earlier exploratory candidates.
There is no best-run selection, threshold relaxation or modification of the
historical archive/freeze. Strict archive checking rejects unrelated extra files;
analysis uses a clean byte-identical archive copy when needed.

## Prospectively frozen confirmation

prefill-event-clock-confirmation-v1.json declares the same16 workload cases,
136-trial phase counts, forced outputs and numeric validation rules. New ct-
trial IDs and clock-v1 paired-content keys produce new exact token pools. Native
collection must train all six coefficients afresh on the new host, freeze before
any confirmation/anchor trial, and retain all failures. It never imports the
archived coefficient vector. Future use of the same workload cases tests fresh
measurements/content on a new host; it does not establish generalization to new
arrival patterns or serving configurations.

The dedicated clock operation creates clock-tokens.json, uses an isolated
endpoint flag and records all training event labels. All launches reset that
flag, along with the existing calibration/arrivals flags. Conflicting modes or
source/CPU acceptance mismatch fail. Normal collection and legacy v1/v2 pilots
keep their existing model and scoring behavior. No GPU worker hook is added.

On the fresh native Linux/Python3.12 RTX4090 host, from the PR32 checkout:

```bash
nohup python3 scripts/setup_prefill_calibration.py --next clock > ~/pr32-prepare.log 2>&1 < /dev/null &
tail -f ~/pr32-prepare.log
```

After PREPARED, exit tail with Ctrl+C:

```bash
nohup .venv/bin/python scripts/prefill_host.py clock > ~/pr32-clock.log 2>&1 < /dev/null &
tail -f ~/pr32-clock.log
```

Active collection/fitting/scoring has the existing1800s limit after a separate
900s startup allowance. Install/download/native CPU acceptance are separate.
Outputs go to ~/pr32-clocks/<timestamp>; a sibling .tar.gz is created on success
or failure. Download that evidence before returning the machine. No automatic
retry, GPU rental or upload occurs.

On the exact source checkout, independently recompute:

```bash
.venv/bin/python scripts/check_prefill_event_clock.py /root/pr32-clocks/TIMESTAMP
```

The audit checks original native CPU JUnit/source/runtime/token identities,
re-extracts raw requests and training events, verifies freeze/forecast identities
and recomputes all unchanged pilot gates. Valid rejected candidates exit2;
incomplete/integrity failures exit nonzero. candidate_passed, if eventually true,
only qualifies this effective controlled pilot. Physical/full-model/full-suite,
HTTP/state-observation, adaptive cap and MPC acceptance remain false.
