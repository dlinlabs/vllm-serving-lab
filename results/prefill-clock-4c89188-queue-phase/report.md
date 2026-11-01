# Native clock prerequisite: queue-phase rejection

The 4c89188 native attempt collected all 136 clock trials in 609.033 seconds, then
stopped before HTTP. Independent audit of the exact 167 recorded source files and
the original 1180-case native CPU acceptance succeeds with **candidate_passed=false**.
The raw archive, six fitted coefficients, original reports and numerical limits
are unchanged. No HTTP, production-model, cap or MPC acceptance follows.

## What failed

The original report rejected one queue point: mixed-plus-sentinel-at400-D16.
Its predicted input-queue residence was 6.698 ms; the four OFF measurements had a
91.939 ms median. Error was -85.240 ms against an 18.388 ms budget. The remaining
seven queue medians and original endpoint/observer/repeat/structure/anchor gates
passed. Queue residence here means **queue-post to EngineCore ADD**, not waiting
for the first scheduler service or first output.

| Pair | OFF residence (ms) | LIGHT residence (ms) |
| --- | ---: | ---: |
| ct-069 / ct-070 | 160.867 | 4.370 |
| ct-086 / ct-085 | 20.398 | 8.625 |
| ct-107 / ct-108 | 163.909 | 9.793 |
| ct-118 / ct-117 | 23.010 | 10.294 |

The OFF median lies between two groups. Its 143.511 ms range cannot be treated as
a repeatable 92 ms queue response. Original quality checks examined TTFT and
completion only; those stable endpoints did not establish queue-clock stability.

Retrospectively applying the new stricter queue controls finds six failed
OFF/LIGHT pairs across 400 ms/D16, 800 ms/D0 and 50 ms/D16, and two failed OFF ranges
(400 ms/D16 and 50 ms/D16). These findings are descriptive development results;
they do not rewrite the original report or claim preregistered validation.

## Mechanism and evidence boundary

The from-empty model polls new arrivals once per iteration. Submission has
priority over FIFO output reap; an input arriving during a blocking future wait
is consumed at a later iteration. Small clock differences near that poll boundary
can change queue residence by an entire batch interval without comparably large
changes to final endpoints.

For this case the model's nearby output-commit boundaries are 406.698 ms and
575.183 ms after its own predicted release. Its 400 ms input enters at 406.698 ms.
The four held LIGHT captures show the corresponding commit at 404.161–410.147 ms
and ADD at 404.489–410.415 ms. Their observed work signatures match the independently
generated model. The LIGHT warmup instead commits at 393.619 ms and 552.170 ms;
its 400 ms input enters at 552.389 ms. This directly demonstrates different observed
queue-consumption phases within one case.

OFF captures do not record internal steps. Their exact blocking/poll boundaries
are therefore **not measured**. The interpretation is consistent with the native
loop contract and LIGHT evidence; it does not identify a physical kernel, GC,
thermal, host or instrumentation cause for each OFF sample. Paired differences
alone do not prove LIGHT causally created every phase change.

A retrospective sensitivity check scales all frozen coefficients together;
it does not optimize or use held labels to choose a replacement vector:

| Scale | Predicted residence (ms) |
| ---: | ---: |
| 0.97 | 157.928 |
| 0.98 | 163.680 |
| 0.99 | 2.631 |
| 1.00 | 6.698 |
| 1.01 | 10.765 |
| 1.02 | 14.832 |
| 1.03 | 18.899 |

This shows the sensitivity of the current deterministic point rollout. It is
not a calibrated confidence interval, new fitted model or robustness guarantee.

## Correction in this change

`delayed_calibration.queue_quality` now independently checks each paired queue
delta and each case's OFF queue range, using the existing absolute 10 ms/relative 20%
queue budget. The result participates in future candidate acceptance and reports
all failed pairs/ranges explicitly. An accurate median with unstable samples
cannot pass through endpoint-only quality checks. These stricter checks were
added after this rejection; old numeric limits are not relaxed.

The frozen six-term law, optimizer, training split, requests, runtime source pins
and raw evidence are unchanged. The HTTP prerequisite remains blocking. The
underlying point-model accuracy is **not repaired** by adding a quality guard.
Further work must identify how queue-phase uncertainty or an independently
observed state should enter a prospectively frozen prediction. This capture is
development evidence for that design, not its independent confirmation.
The current HTTP predictor starts after input-queue waiting has already elapsed;
that conditioning does not by itself forecast the earlier queue interval or
qualify a pre-admission cap decision. The failed prerequisite is not bypassed.

## Reproduction and provenance

Original file: 1793086097723591624.tar.gz, 13032945 bytes, SHA256
`2e9ef84b537857652251cb17e89b485a5d6835a333a630ee2a96dd78241649fd`.
The unchanged archive is retained as
[the uploaded evidence](https://chatgpt.com/library?file_id=libfile_b8dd1381722c81918782bc67d3f8e2ec&account_id=personal).
It contains 164 files and 119054968 uncompressed bytes. `input-files.json` binds
every raw file. `native/` preserves selected byte-identical reports, inputs and
CPU proof; the full binary archive is not duplicated in Git.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analyze_prefill_queue_phase.py \
  --capture /path/to/1793086097723591624 \
  --archive /path/to/1793086097723591624.tar.gz \
  --output /new/output/directory
```

The analyzer reconstructs the original audited source from its pinned Git commit,
then checks archive bytes, reports all cases' retrospective queue controls and
the failed point's native/model windows. It preserves a valid rejected verdict.
Source tampering, incomplete original collection or identity drift fails audit.
Prediction source hashes must still match the recorded implementation.

See `cpu-validation.json` for the local correction tests. Local CPU regressions
are distinct from the captured original native 1180-case acceptance. No local GPU
execution or new native qualification was performed for this correction.

The correction passes 564 related local CPU cases (one warning, 190.60 seconds)
and 55 focused queue/audit/design cases (3.19 seconds), with no errors, failures
or skips. Counts overlap and are not summed. The 13 new parameterized cases cover
stable endpoints hiding bad queue controls, invalid evidence and rejected-source
audit integrity. These tests establish guard behavior, not model accuracy.
