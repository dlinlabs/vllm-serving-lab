# Prefill service identification v1 — proposed next GPU experiment

Status: **design draft; collector and version-specific hooks are not implemented**.
The accompanying design, cases and order files are reviewable proposals, not an
executable collection protocol. Freeze their final hashes and an accepted collector
commit before renting a GPU for collection. Existing trace protocols and results stay historical.

## Question and evidence

Determine whether the cap-32 forecast error is better explained by work/metric timing,
scheduler/chunk composition, or a nonlinear prefill service term. The output is a
measured service/state representation for the next simulator iteration, not a cap
optimum or MPC-benefit experiment.

* PR26/27: one-step flow consistency did not make multistep goodput reliable;
  evolving endogenous work/history matters, and additional counter inputs alone did not fix it.
* PR28: FIFO prefill, 8192 scheduled tokens per step and partial-prefill progress are
  approximations. The verified context limit does not verify the scheduler token budget.
* PR29/30: fixed/switch 16/32 data exist. State calibration improves mean forecast
  error, but leaves large segment bias and unreliable hypothetical action rankings.
* PR31: recent accepted prompt volume is associated with short-request misses on
  both hosts. A linear timing control explains much of the apparent squared-term gain.
  Uniformly spreading tokens over accept-to-first spans uses future observations;
  it is diagnostic and cannot become an online feature.

## Hypotheses to distinguish

| Hypothesis | Discriminating observation | Consequence |
|---|---|---|
| H1: counter timing obscures the workload that actually ran | Scheduled prompt-token progress reconciles with request state before aggregate counter increments | Attribute work to executed steps; model pending prefill explicitly |
| H2: shape/context adds service cost beyond total scheduled tokens | A context/chunk term improves predictions for held-out lengths and mixtures after controlling decode context | Retain that term only if the prespecified validation supports it |
| H3: scheduler/backlog changes make short requests wait | Same demand delivered as a burst versus staggered arrivals changes measured queue/chunk history and sentinel TTFT | Correct scheduler/state approximation rather than simply raising a global delay |
| H4: observation changes the system materially | Paired observer-off/light captures differ by more than the declared overhead allowance | Fix the observer before using timing captures |

These are competing/overlapping explanations. Failure to establish H2 is a valid result.
No requirement says the model must prefer cap16 in a named Azure segment.

## Environment and version contract

One RTX4090, one host, one loaded EngineCore for the entire benchmark. Match current
Qwen/Qwen3-4B-Instruct-2507, Python3.12, vLLM0.30.0 and Torch2.13.0+cu130; pin model
and tokenizer revisions, package lock, installed-source hashes and observer patch.
Save driver/CUDA, GPU UUID/clocks/power/temperature, CPU affinity/quota and resolved
engine config. GPU clocks are observed, not silently changed.

Required resolved config includes model/context8192, max_num_batched_tokens,
max_num_seqs, chunked-prefill and partial-prefill limits, long-prefill threshold,
attention backend/dtype, KV capacity, prefix-cache flag, graph/compile settings,
async scheduling, worker mode, speculative decoding and tensor parallelism.
The new run measures its actual setting; it cannot retroactively establish old defaults.
Keep normal graph/async behavior. Do not silently force eager/serial mode to simplify hooks.
Require prefix caching off, speculative decoding off, TP1 and no external traffic.
If these cannot be reconciled with the selected runtime, stop and revise the draft before collection.

## Three distinct measurement layers

1. **Engine-side enqueue barrier:** submit a foreground cohort before the next
   scheduler pass. This fixes logical arrivals while leaving actual chunking to
   vLLM. Separate admission wall time, scheduling decisions and execution.
2. **Per-step observer:** request identity, prompt length, prior computed position,
   scheduled prompt/decode tokens, context lengths, running/waiting IDs, preemptions,
   graph mode, scheduler/execute timing and result visibility, joined by step ID.
3. **Short independent trace:** use supported profiler facilities on named cases
   to validate GPU timing and stream/graph coverage. Deep traces are never the
   primary performance measurements.

The official v0.30.0 scheduler output exposes request-level scheduled-token counts
and computed positions; the adapter must inspect the exact installed implementation
to reconcile progress, including async in-flight work. A scheduler decision alone
is not proof the tokens completed. Reconcile scheduled, executed, committed and
metric-reported work at trial drain.

Record CPU monotonic iteration wall time and CUDA event spans as different targets.
CUDA launches are asynchronous: a Python function's return is not GPU completion.
Use stream-correct event recording with delayed resolution, no per-step global
cuda.synchronize. Confirm multi-stream/graph boundaries against short traces.
If there is no unambiguous GPU span, do not label a CPU span as GPU time. Async
overlapping steps cannot be converted into a serial service model by adding spans.

Use bounded in-memory records and deferred flush after drain. An overflow, dropped
event, restart, unexpected worker or failed reconciliation invalidates the trial.
Store prefill chunks and positions, not just sums of entire admitted prompt lengths.

## Workloads: 20 cells, five randomized complete blocks

All lengths mean **actual engine input token IDs**, including any template/special
tokens. Prebuild and hash IDs before GPU collection; exclude special/EOS IDs from
the synthetic body. Do not approximate length by repeated text. Materialize the
100 per-trial prompt manifests from the fixed content seed before freezing.

Foreground outputs are forced to exactly128 tokens (ignore EOS for this mechanistic
study, temperature0). Report this difference from historical natural-EOS replay.
For D8, eight background requests have input256 and forced output512; wait until
all are decoding before releasing the foreground barrier. Record actual live count
and context of each background request; eight is a target at release, not assumed
constant during subsequent steps. Wait for **all** work to drain before the next trial.
Input plus requested output must stay within8192 for every foreground/background/sentinel request.

| Family | Foreground at relative time0 | Extra request | Background at release | Cells |
|---|---|---|---|---:|
| Single-length | one of1024/2048/4096/7168 tokens | none | D0 or D8 | 8 |
| Equal total/number | four4096, or1024/1024/7168/7168, or reverse order | none | D0 or D8 | 6 |
| Sentinel | none, or four4096 as one burst, or four4096 at0/50/100/150ms | one512-token sentinel at200ms | D0 or D8 | 6 |

The equal-total cases each have four requests and16384 prompt tokens; sum-of-squares
and admission order differ. This controls offered demand/count, **not** actual
per-step composition, which must be observed. Contrasting the two heterogeneous
orders helps distinguish order effects. Same token content is paired within each
block for observer checks and sentinel burst/staggered comparisons.

For sentinel trials, target offsets are engine enqueue times from a common release
epoch. Maximum offset error is5ms; simultaneous groups are inserted at one barrier,
not four independent HTTP sends. Retain target and actual times. The sentinel's
200ms arrival means all four staggered requests should already have arrived.

Gateway cap is bypassed in this engine experiment. It is not an admission-policy
comparison: cap16 versus32 and SLO goodput will be tested later through the gateway.
No synthetic case is called an Azure replay or held-out C workload.

## Fixed order, warmup, observer check and resource budget

The generated order uses Python3.12 Random seed20261022. Each of five main blocks
contains every cell once, randomized before collection; do not reorder after seeing
latency. A block, not a GPU step, is the replication unit. All traces share one host.

* 40 unscored warmups: two passes through all20 cells, to cover long prompts,
  background decoding and graph shapes. Any compilation first seen during scored
  work invalidates that affected trial; do not silently discard/retry it.
* 40 observer trials: two demanding cells ×10 off/light pairs. Pair order is
  balanced five off-first/five light-first, randomized in advance.
* 100 main light-observer trials: five complete20-cell blocks.
* Four deep diagnostic trials: two named cells ×two traces, after the main series;
  trace only five actual execution steps beginning at foreground release, with config/source
  support verified beforehand. Profiler flush time is recorded separately.

Total planned:184 trials, of which100 are model/association trials. Each trial has
a45s timeout including background/drain; no hidden cancellations to speed up runs.
After timeout, preserve the partial trial, stop, and leave the suite incomplete.
Target active collection is15–30min after setup, **not a guarantee**. Hard collection
budget is1800s including warmup/observer checks and deep flush, excluding installation
and initial model loading. A budget expiry preserves an incomplete attempt; it does
not authorize a smaller successful suite. No auto-retry or adaptive cell selection.

Observer allowance: paired case wall time and foreground TTFT are measured in both
off/light modes. For each of the two cells, the upper95% paired-bootstrap interval
(10000 pair resamples, seed20261023) on median light/off relative slowdown must be
<=5% for both metrics. This is an engineering measurement allowance, not a theorem
or proof of zero observer effect. If inconclusive/failing, stop before the100 main
trials and improve the observer. Deep-profile latency is separately labelled.

## Outputs and qualification

Save suite/order/case/prompt/source hashes; environment/config/source mapping;
per-request target/enqueue/first/completion times and output counts; per-step
scheduler/execution records; raw metrics with send/receive timestamps; short traces;
GPU/CPU resource samples; gate decisions, stdout/stderr and complete failure evidence.

Mandatory before interpreting service coefficients:

* Exact-revision CPU/mock acceptance, plus adapter tests for token accounting,
  chunked/partial prefill, async IDs, duplicate timestamps, cancellation and output
  completion, missing events and future-observation isolation.
* One stable engine identity/config, all declared184 trials accounted for, no
  silent retries, no OOM/preemptions/prefix hits, clean drain between trials.
* Every request identity, and every light/deep-observed step identity, reconciles.
  Observer-off trials retain request/outcome evidence without pretending to have step logs.
  Completed foreground/sentinel output128,
  background512. All scheduled prompt progress reconciles with input IDs at drain;
  counter semantics and any offsets are documented from source and observed traces.
* Target D8 backgrounds are all decoding at release; measured context is retained.
  Injected timing error <=5ms, no dropped/overwritten logs and observer gate passes.

Existing replay50ms load/control gates remain unchanged for future Azure runs.
They are not automatically applicable to this new engine-side harness.

## Fitting and validation, declared before collection

The cases file assigns eight calibration cells: single1024/4096, uniform four4096,
and no-burst sentinel, each at D0/D8. Fit only blocks1–3 for those cells (24 trials).
Blocks4–5 of those cells are repeat checks (16 trials). Other lengths, heterogeneous
orders and burst/staggered sentinel cases form60 held-case trials across all five
blocks. Do not inspect their timing summaries before freezing the candidate fit.
These are controlled-case transfer checks, not independent production acceptance.

Fit three fixed nonnegative regression specifications, separately for valid GPU
spans and valid engine iteration wall-time targets; never mix target units/scopes:

* M0: intercept + scheduled decode-token count + scheduled prefill-token count.
* M1: M0 + sum of decode context lengths.
* M2: M1 + attention-pair proxy Z = sum(q*l + q*(q+1)/2), where q is the **executed
  prompt chunk** and l its pre-chunk computed prompt position. This is a candidate
  explanatory proxy, not a physical law or squared full-prompt cost per chunk.

Train-only scaling; no fitted request/run/time-of-segment identities. Choose one
of M0/M1/M2 using leave-one-calibration-cell-out MAE inside the24 training trials,
with simpler model winning ties. Store all training fits, source and chosen-model
hash before opening held cases. Report every model's held-case results once;
changing the candidate afterwards creates a new development iteration, not a pass.

Report per-cell/block service-time MAE and signed bias, next-prefill/first-token
timing, remaining-work error, sentinel TTFT and queue wait, output throughput,
measured chunk/decoder-context distribution, and observer effect. GPU steps within
a trial are not independent replicates. Use cell/block summaries and disclose five
blocks; avoid claims of precise population significance from thousands of steps.

The fixed simulator from PR28/30 remains an archived reference. Its coefficients
were learned from interval wall time, so comparing them with CUDA time requires
matching the target scope, not reusing the old1.0286/0.729 forecast thresholds.
Any end-to-end first-token replay uses only injected arrivals, pre-decision request
attributes and predicted state progress. Future first-content times and measured
future chunk decisions are confined to oracle diagnostics. A GPU-time fit alone
does not supply a valid end-to-end rollout, particularly with async overlap.

Decision rules:

1. If only timing/accounting changes help, fix work attribution/prefill state and
   keep the simpler service form. Do not add a square term just because its sign is positive.
2. A selected context/nonlinear model passes the service-time screen if held-case
   wall-time MAE improves by>=10% versus refitted M0, with no per-case MAE increase
   greater than max(10% of that case's M0 MAE, 1ms). The floor prevents unstable
   ratios when reference error is near zero. It is then a **service-model candidate**;
   its causal scheduler rollout must separately predict the sentinel cases before
   Azure development replay. No measured future chunks may be used to pass that check.
   These margins are engineering screening choices, not MPC acceptance guarantees.
3. Otherwise report mixed/negative results and the unresolved scheduler/overlap
   behavior. Do not relax gates or force cap rankings.

After this experiment: implement causal pending-prefill state and the supported
service/queue update, compare it with frozen baselines on existing development
replays, then freeze a separate same-host repeated fixed/reactive/MPC evaluation.
This experiment alone cannot establish dynamic cap benefit; C remains unused.

## Sources and preparation checklist

* [PR31 diagnosis](https://github.com/dlinlabs/vllm-serving-lab/pull/31), with PR26–30
  reports as linked there.
* [vLLM v0.30.0 profiling](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/):
  detailed profilers add overhead; short traces validate scope, not benchmark latency.
* [v0.30.0 scheduler output](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/output.py):
  inspect against installed wheel source, not current-main assumptions.
* [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html):
  asynchronous execution requires appropriate event/stream timing. Keep the chosen Torch build pinned.

Before collection, implement/review the opt-in adapter and runner, materialize token
manifests, verify resolved config and step/metric source mapping, validate buffering/
async timing contracts, run CPU acceptance, replace draft placeholders with exact
hashes, and freeze a new executable protocol. This design intentionally supplies
no pretend runnable vLLM command for hooks that do not yet exist.
