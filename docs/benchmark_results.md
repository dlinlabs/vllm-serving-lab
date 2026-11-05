# Recorded results and benchmark evidence

Status: **no qualified native MPC-versus-baseline performance result**.
Historical serving measurements, CPU tests and current component diagnostics
are intentionally separate. No new experiment was run to write this document.

## E1: historical native serving measurements, not MPC

Hardware/configuration as recorded by the runner: RTX 4090, reported 24,564 MiB,
driver 595.91.07; configured model Qwen3-4B-Instruct-2507; runner environment
vLLM 0.30.0, Torch 2.13.0+cu130, Python 3.12.3. The manifests explicitly say
these package records are not proof of the remote backend version and do not
observe backend launch provenance; the historical report states the same core
runtime versions.
Both captures replay the same Azure-derived arrivals/input lengths with
synthetic prompts, output cap 128, context 8192 and 300-second arrival windows.
Source output lengths are not replayed. TTFT SLO is 0.5 seconds.

| Capture | Offered | Completed success | Rejected | Failed | SLO-good completions | SLO goodput (req/s) | Client TTFT p99 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared A | 7,643 | 1,976 | 5,667 | 0 | 572 | 1.906667 | 2.412583 |
| Shared B | 7,643 | 2,059 | 5,584 | 0 | 736 | 2.453333 | 2.202635 |

TTFT p99 is conditional on successful requests; rejection is shown alongside
it. SLO goodput counts eventual successful completions whose first content
meets the scheduled-arrival TTFT deadline, divided by offered duration.
The arrival window is not total wall time including startup/drain.

These are **scheduled identification captures on different hosts**, with
different protocol declarations, budget-schedule orders, replay-plan hashes
and gateway virtual-environment paths. They are not MPC, Fixed/OneMove
comparisons, causal schedule effects or proof of an optimal cap.
Historical integrity/load/control qualification does not mean every request
met its SLO. Recorded `git_dirty=true` is retained; source hashes, not a claim
of a pristine checkout, support the archived runtime identity.

### E1 evidence chain and reconstruction limits

| Item | Exact evidence |
|---|---|
| Reports | [A report](../results/shared-client-20261018/captures/shared-client-a-r1/report.json), [B report](../results/shared-client-20261018/captures/shared-client-b-r1/report.json) |
| Configuration and trace identity | [A manifest](../results/shared-client-20261018/captures/shared-client-a-r1/manifest.json), [B manifest](../results/shared-client-20261018/captures/shared-client-b-r1/manifest.json) |
| Complete raw archives and file hashes | [Raw-bundle index](../results/shared-client-20261018/raw-bundle/index.json); browsing subsets are not complete raw captures |
| Trace SHA256 | `62f3f182bcb889c846b5a46a05a53766ef20c5f79008e9050a4fe33707bc4e2e` |
| Archived independent audits | [verification.json](../results/shared-client-20261018/verification.json) |
| Exact collection declarations and commands | [Historical report](../results/shared-client-20261018/report.md), [A runbook](../experiments/shared-client-qualification-v1.md), [B runbook](../experiments/shared-client-b-v1.md) |
| Recorded runtime commit | `d0ccb2c0d3711e1b2a69bb4ec248c61961ce3586` |
| Evidence publication commit | `aea7694a0af2d1dadc883b4b9e33d1a9bdb61098` |

Historical offline feature/forecast reconstruction command, after using the
report's extraction helper and recorded analysis environment:

```bash
python results/shared-client-20261018/reproduce.py \
  --raw-root /tmp/shared-client-captures \
  --output-dir /tmp/shared-client-analysis --extended \
  --code-root /path/to/recorded-runtime-source
```

This command does not regenerate the E1 outcome table. `reproduce.py` checks
the 15 manifest-recorded runtime files, then runs `identification_features.py`
and `forecast_baselines.py` from `--code-root` without checking their hashes.
It is not a fully source-pinned reproduction path. E1 table reconstruction
evidence is the archived raw-to-report comparison in `verification.json`,
not a fresh replay performed for these documents.

The runtime commit was not resolvable in the current local object database
during this documentation audit. A source tree matching the archived hashes is
required; the example path is a required input, not a working shipped checkout.
This is a P0 portability gap. No dependencies were installed and no captures
were re-extracted/replayed for this documentation change.

## Other evidence: engineering, not MPC performance

| ID | Verified result / limitation | Trace, configuration and command | Exact evidence commit |
|---|---|---|---|
| E2 | Frozen policy planner: 130 CPU cases passed per Python in accepted Parent replays. Acceptance is exact candidate ordering with full-action Fixed-reference scope; runtime integration and native actuation are not accepted. | [Acceptance index](../results/mpc-native-r24/reviews/native-team16-planner-revision02-parent-seal01.json) records PLAN, source hashes and archived command/result paths. | `2b4ed5bc754060cba11c16852b7798b975bc03eb` |
| E3 | Deployment composition: nine CPU integration cases passed per Python, with bounded acceptance and independent full-composition review still pending at that acceptance; never live-eligible. Later blocked by D2-F1 (E5). | [Acceptance/config index](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-parent01.json), [exact archived command](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-parent01/attempts/mayor-py314.command.json). Absolute paths require relocation; not a portable CLI claim. | `c00d29d2296bda7e249578f2de7c08650a4d79c9` |
| E4 | Native observation diagnostic, one approximately 67-second GPU run: engineering accepted; component coverage **FAIL** (one of two required nonempty in-flight observations). 90 HTTP records; three generations (one warmup and two managed); 43 completion tokens including 11 warmup tokens. Not a throughput or prediction benchmark. | [Result/hash index](../results/mpc-native-r24/reviews/native-team16-owner-inflight-gpu01-results.json), [preregistration](../results/mpc-native-r24/reviews/native-team16-owner-inflight-gpu01-preregistration.json), [command/config/raw bundle](../results/mpc-native-r24/reviews/native-team16-owner-inflight-gpu01/). | Results `52b2b244f5904306f730ab6ca26a3d572343e2b2`; preregistration `d6ded2b25a0667028f6ab9f2cc08fe76df1c3315` |
| E5 | Source-backed D2-F1 cleanup bypass; no observed live orphan incident asserted. Live use of the affected launcher is blocked. | [Mayor disposition](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-independent-parent01.json), [independent review](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-independent-author01/REVIEW.md). No live reproduction command was run. | `26139be5f90ec7386e2c578c95aa331aa0052244` |

No row above establishes calibrated prediction, actual MPC benefit, production
deployment, deterministic SLO compliance or a general OS cleanup theorem.

## Current nominal MVP component evidence

These are additional evidence for resume package A, not results for native
deliverable B. Counts below describe the named CPU controls only; they are not
summed across Python versions or treated as performance measurements.

| ID | Verified result / boundary | Trace, configuration and command | Exact evidence commit |
|---|---|---|---|
| E6 | Fixed-policy solver-wait/deadline bias corrected; Mayor's combined component controls passed. Model-ranked Fixed is not empirical baseline selection. | [Component acceptance](../results/mpc-native-r24/reviews/resume-p0-components-parent02.json), [CPU plan/command](../results/mpc-native-r24/reviews/resume-p0-parent-cpu01-plan.json), [original negative witness result](../results/mpc-native-r24/reviews/resume-p0-nominal-planner01-fixed-delay-result.json). | Correction/evidence `f645a66a44bcbe0e4894cab4213d7f9349ae1481`; negative witness `814a9b5aa38dd8783ae643270b5aed59e1bc12b7` |
| E7 | All 88 frozen Train prompts matched requested chat-template token counts with the real cached tokenizer. CPU conversion on the existing host, no inference. | [Result](../results/mpc-native-r24/reviews/resume-p0-tokenizer02-result.json), [plan/command](../results/mpc-native-r24/reviews/resume-p0-tokenizer02-plan.json), [Train replay](../results/mpc-native-r24/reviews/resume-p0-tokenizer02-raw/returned/train.replay.json). Replay SHA256 `5e418b8939179bac16d146d08f7acfa4d617b78adc4c324caca8ec0893ba52e7`. The first offline repository-ID lookup failed and is retained in the predecessor evidence. | `02b292b4dbe55c274eba0d0b435bc5f9fc2bac9e`; local-asset successor plan `1e9540caa9dd37f927d5389d05be3d973d234bf7` |
| E8 | Train-only causal forecast and permit-dependency CPU component accepted; 14 controls passed on Python 3.12.16 from a scoped Git materialization. Not full end-to-end reproduction. | [Acceptance/result](../results/mpc-native-r24/reviews/resume-p0-scenario-parent01-result.json), [plan/command](../results/mpc-native-r24/reviews/resume-p0-scenario-parent01-plan.json), [frozen forecast source](../experiments/mpc-research/resume-mvp/nominal_scenario.py). | `6240906706d26c4ce8e5f2a9e70f5e54ac9d52d0` |

The nominal forecast uses the frozen Train arrival-rate/mark distribution with
uncalibrated service assumptions. Its stationary rolling 15-second horizon can
extend beyond the public five-second workload window. Neither the tokenizer
checks nor the causal-input controls establish forecast accuracy.

The [independent metrics audit and Mayor disposition](../results/mpc-native-r24/reviews/resume-p0-metrics-independent-parent01.json)
block claim-qualified Train selection/comparison on metrics01: common-environment
identity, frozen selection provenance, complete offered cohorts and timestamp
bounds need correction and recheck. Existing single-run arithmetic controls do
not override those blockers. No new native MPC measurements are available.

## Required MVP comparison: currently unmeasured

| Policy | TTFT p50/p95/p99 | SLO goodput | Throughput | Rejection | Fairness | Controller overhead |
|---|---|---|---|---|---|---|
| Strong development-selected Fixed | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Explicit FIFO baseline | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Strong OneMove | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Receding-horizon MPC | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

The historical accepted comparison harness has no separate FIFO arm, throughput
metric or fairness index. These are P0 implementation/specification gaps, not
zero-valued measured results. FIFO is a queue discipline, not automatically a
distinct admission controller: define its hard bounds and distinguish it from
Fixed, or explicitly disclose identical arms.

Before collection, freeze the strongest Fixed selection on development data
only; use identical action/resource support, offered requests, prompt/output
rules, warmup/cache settings and failure accounting across policies. OneMove
must use the same model/scenario panel and a meaningful best single-transition
search. Charge controller delays without slowing the offered workload to hide
overhead. Serialize GPU arms; record ordering, resets and contamination checks.

Predeclare development and evaluation workload identities, hashes and split.
The evaluation workload must be disjoint from every input used to fit the
predictor, build scenario panels, select Fixed or tune MPC/OneMove. Record
whether Fixed is selected by development GPU measurements or model ranking.
Freeze that choice before evaluation. Scenario panels may use only information
available at each decision, never the scored workload's own future arrivals.
E1 A/B both use the same development trace; held-out C was not collected.
Reusing that trace permits only an explicitly labelled in-sample engineering
pilot, not the held-out P0 comparison or an out-of-sample accuracy claim.

Required metric definitions:

- TTFT: report scheduled-arrival and client-send origins separately, estimator,
  completed-request sample counts and missing/censored outcomes.
- SLO goodput: successful, SLO-meeting completions divided by the fixed offered
  interval; every rejected/failed/unknown offer remains in the outcome ledger.
- Throughput: declare completed requests/s versus output tokens/s and whether
  warmup/drain are included. Never substitute HTTP-record count for generations.
- Rejection/failure: report each outcome over all offered requests, retaining
  unknown application and incomplete cleanup.
- Fairness: predefine workload groups and report per-group offered-normalized
  SLO attainment/rejection. Define any Jain index, zero-denominator convention
  and group sample counts prospectively; no starvation guarantee follows.
- Overhead: include observation, model, IPC, optimization, logging, actuation,
  missed deadlines and end-to-end delay; isolated CPU solver time is insufficient.

A descriptive pilot may support a bounded engineering claim without statistical
superiority. Do not turn that scope into a PASS on an older confirmatory gate.

### Prospective metric decisions, not measurements

The [Role7 proposal and Mayor disposition](../results/mpc-native-r24/reviews/resume-p0-metrics-contract-parent01.json)
record definitions pending the single producer/runtime contract. No metric
implementation or native eligibility is accepted by that proposal.

The earlier contract proposal used boundaries 512 and 2048 tokens; the current
frozen nominal workload has exact prompt classes 256, 1024 and 3072 tokens,
verified on Train in E7. Preserve its declared grouping and scorer through
Eval; do not retrospectively regroup to improve results. Keep SLO
attainment (good/all offered, unitless) distinct from goodput (good/second).
Report cohort-normalized rates using the offered window, alongside actual
wall-envelope rates including setup/drain; neither is a steady-state GPU
capacity estimate. MPC and strong OneMove must have identical solver and
actuation time budgets as well as action/model/scenario support.

A missing-first-content penalty of infinity may define an all-offered
operational TTFT metric, but is not an identified survival-analysis quantile of
right-censored latency. Label that estimand and null/censor reasons explicitly.
Do not infer native FIFO service from the CPU plant or infer maximum budget
from the term FIFO. Any static-cap alias needs an explicit definition and
verified ordering; disclose duplicate policies rather than claim four
independent algorithms.
