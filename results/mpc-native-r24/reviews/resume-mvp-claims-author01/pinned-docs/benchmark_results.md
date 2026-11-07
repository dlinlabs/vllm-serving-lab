# Verified results and benchmark evidence

Status: **no qualified native MPC-versus-baseline performance result**.
Historical serving measurements, CPU tests and current component diagnostics
are intentionally separate. No new experiment was run to write this document.

## E1: historical native serving measurements, not MPC

Hardware/configuration: RTX 4090, reported 24,564 MiB, driver 595.91.07;
Qwen3-4B-Instruct-2507, vLLM 0.30.0, Torch 2.13.0+cu130, Python 3.12.3.
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

These are **scheduled identification captures on different hosts**, not MPC,
Fixed/OneMove comparisons, causal schedule effects or proof of an optimal cap.
Historical integrity/load/control qualification does not mean every request
met its SLO. Recorded `git_dirty=true` is retained; source hashes, not a claim
of a pristine checkout, support the archived runtime identity.

### E1 evidence chain and reproduction

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

Historical offline reconstruction command, after using the report's extraction
helper and pinned analysis environment:

```bash
python results/shared-client-20261018/reproduce.py \
  --raw-root /tmp/shared-client-captures \
  --output-dir /tmp/shared-client-analysis --extended \
  --code-root /path/to/recorded-runtime-source
```

The runtime commit was not resolvable in the current local object database
during this documentation audit. A source tree matching the archived hashes is
required; the example path is a required input, not a working shipped checkout.
This is a P0 portability gap. No dependencies were installed and no captures
were re-extracted/replayed for this documentation change.

## Other evidence: engineering, not MPC performance

| ID | Verified result / limitation | Trace, configuration and command | Exact evidence commit |
|---|---|---|---|
| E2 | Frozen policy planner: 130 CPU cases passed per Python in accepted Parent replays; no native inference or actuation qualification. | [Acceptance index](../results/mpc-native-r24/reviews/native-team16-planner-revision02-parent-seal01.json) records PLAN, source hashes and archived command/result paths. | `2b4ed5bc754060cba11c16852b7798b975bc03eb` |
| E3 | Deployment composition: nine CPU integration cases passed per Python; this predates the subsequent cleanup blocker. | [Acceptance/config index](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-parent01.json), [exact archived command](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-parent01/attempts/mayor-py314.command.json). Absolute paths require relocation; not a portable CLI claim. | `c00d29d2296bda7e249578f2de7c08650a4d79c9` |
| E4 | Native observation diagnostic: 90 HTTP records, 43 completion tokens and incomplete nonempty coverage (one of two required observations). Not a throughput or prediction benchmark. | [Result/hash index](../results/mpc-native-r24/reviews/native-team16-owner-inflight-gpu01-results.json), [preregistration](../results/mpc-native-r24/reviews/native-team16-owner-inflight-gpu01-preregistration.json), [command/config/raw bundle](../results/mpc-native-r24/reviews/native-team16-owner-inflight-gpu01/). | Results `52b2b244f5904306f730ab6ca26a3d572343e2b2`; preregistration `d6ded2b25a0667028f6ab9f2cc08fe76df1c3315` |
| E5 | Source-backed D2-F1 cleanup bypass; no observed live orphan incident asserted. Live use of the affected launcher is blocked. | [Mayor disposition](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-independent-parent01.json), [independent review](../results/mpc-native-r24/reviews/native-team16-consumer-deployment02-independent-author01/REVIEW.md). No live reproduction command was run. | `26139be5f90ec7386e2c578c95aa331aa0052244` |

No row above establishes calibrated prediction, actual MPC benefit, production
deployment, deterministic SLO compliance or a general OS cleanup theorem.

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
