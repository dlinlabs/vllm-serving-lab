# CPU causal async predictor prototype

Status: executable, synthetic/unfitted. This implements the next engineering
milestone in prefill-async-pipeline-v1.md. It does not freeze a calibration or
confirmatory collection protocol, replace the guarded serial fitter, or qualify
CUDA scope, latency accuracy, adaptive cap or MPC. No new GPU hooks are added.

## Run without a GPU

From the repository root, in the CPU dependency environment:

```sh
python scripts/predict_prefill_async.py --synthetic-demo --output /tmp/pr32-async-demo
```

This generates all 22 declared controlled cases. The output directory must be
new, so a prior result cannot be overwritten. `--case single-L7168-D16` selects
one cell. `--timeline` additionally writes its generated states, work and event
graph. The default demonstration assumes eager dispatch and explicit synthetic
costs from prefill-async-synthetic-costs-v1.json. Its TTFT numbers are illustrations,
not forecasts for a 4090. No timing coefficients come from the development traces.

`--runtime /path/to/runtime.json` instead uses saved configuration and the frozen
V2 dispatcher. This metadata scope check is not the native acceptance/source
gate. `--costs /path/to/costs.json` accepts explicit coefficient documents and
requires runtime metadata; it still returns an unaccepted prototype. The output
records the scope and hashes of costs, dispatcher and case document. Supplying coefficients
does not establish calibration provenance or physical-model acceptance.

For one case, `--compare-structure /path/to/trial.json` independently compares the
finished prediction with a separately audited capture. The comparator reads
captured work/positions, pre-schedule request counters, executed mode/padding,
step-bound committed outputs and final output counts. Complete pipeline output
evidence is required; a missing sampling tail cannot pass. It does not read
captured starts, ends or first-output times.
The capture never enters rollout or changes its coefficients. Exact agreement
in zero-offset development cases is scheduling evidence, not held latency
validation. Nonzero-offset work may differ when synthetic service times change
which engine iteration receives an arrival.

## State transition and two slots

The simulator starts with an empty engine. Backgrounds enter in lexicographic ID
order with 1024 input tokens and 256 forced outputs. Foregrounds enter only after
all background first outputs have been committed. It simulates that barrier and
any batch already in flight; no observed release snapshot is consumed.
Foregrounds have 128 forced outputs, with declared offsets relative to the
predicted barrier. The generic Request API also supports smaller forced lengths
for hand-computable tests.

Each schedule uses the existing restricted running-first FCFS reference and its
own computed positions, committed tokens and placeholders. Scheduling advances
computed/in-flight tokens; a non-partial batch adds an output placeholder.
Committing its output appends a token and removes a placeholder. Intermediate
prefill chunks emit no committed output, even if a newer in-flight chunk has
already reached the end of that prompt. The terminal guard prevents scheduling
more forced outputs than requested.

The native two-slot queue is submit-first, FIFO-reap. A ready older output does
not preempt submission into an available second slot. Empty scheduler results
also occupy native futures and are retained in the simulation, though they do
not appear as executed worker steps. When an empty iteration leaves requests,
the simulator accounts for the native 1ms idle yield without actually sleeping.
Only engine-loop boundaries consume ideal timer arrivals; a CPU blocked on an
old future cannot inject a sentinel in the middle of that wait.

Block-rounded scheduled positions track KV demand. Insufficient capacity raises
an error instead of approximating preemption. Prompt plus forced output must fit
the model context. Default scope is B2048, 32 sequences, context8192, two slots,
16-token blocks and 5594 blocks; the default demonstration labels these as
assumptions. Supplied runtime metadata provides its actual declared values.

## Resource graph and cost meaning

Nodes have ready time, explicit earlier dependencies, a resource lane, and a
nonnegative cost. Start is the maximum of ready time and dependency/resource
completion. Lanes represent availability of the EngineCore thread, the GPU main
stream and the GPU copy stream; lane occupancy is not hardware utilization.

Each forward/logits/post-update launch has five explicit cost laws:

| Part | Meaning |
| --- | --- |
| cpu_to_first | Host work before the first enqueue |
| cpu_between | Host submission between first and last enqueue |
| cpu_after | Host work after last enqueue before return |
| gpu_prefix | Main-stream service eligible after first enqueue |
| gpu_tail | Remaining service eligible after prefix and last enqueue |

The GPU prefix can overlap CPU submission. The tail prevents GPU completion
before the final enqueue. This is a two-segment approximation to a multi-launch
program, not an inference that whole CPU spans precede whole GPU spans. Current
aggregate stage spans do not uniquely identify these five parts. The approximation
needs qualification and an identifiable fitting protocol before physical claims.

Forward, logits/sample and post-update preserve main-stream FIFO ordering.
After CPU copy enqueue, copy service waits on logits completion and copy-stream
availability. Post-update is submitted after copy enqueue and waits on the main
stream, **not copy completion**. Thus copy and post-update can overlap. Future
forward work waits on main-stream post-update. CPU output materialization waits
on its own batch's copy completion; commit follows materialization and previous
CPU work. Batch identity remains stable while newer submissions exist.

`copy_service` is an explicit wait-exclusive service parameter. The measured
copy-enqueue CUDA elapsed label includes a native stream wait and cannot populate
this parameter directly. The nested `sample_tokens` parent is not an additional
service node when its children are already costed. No automatic event-span
conversion or fitting is provided.

Cost laws are immutable nonnegative affine maps over bias, prefill tokens,
decode requests/context, incremental attention pairs, padding excess and request
count. Modes are chosen from predicted work with the existing frozen V2
dispatcher. This prototype shares coefficients across modes; it does not claim
mode-specific accuracy or support qualification. Empty submission, scheduling,
copy enqueue/service, output materialization and commit have separate laws.

Request `ttft_s` is ideal input-queue posting to first scheduler commit;
`engine_ttft_s` starts at actual predicted engine injection. Their difference is
`input_queue_wait_s`. HTTP/gateway, frontend detokenization and network latency
are outside this scope. CPU drain and all-resource drain are reported separately.

## Explicit limits and remaining work

Scope is reviewed vLLM0.30.0 V2, same-process UniProcExecutor, TP/PP/DP1,
ordinary autoregressive forced outputs, no prefix caching, speculation, remote
KV, preemption, structured output, encoder, multimodal or LoRA. Arbitrary generic
Request inputs are a simulator interface, not an expansion of native qualification.
The simulator does not model final idle-worker cleanup, abort/EOS paths, timer
posting jitter, CPU process contention, observer overhead, graph warmup or hidden
device streams. In-flight KV allocation follows the no-sharing/no-lookahead scope;
allocator/cleanup latency is not independently established.

The synthetic smoke run drains all 22 cells. With its illustrative costs, three
late-arrival D16 sentinel cells let backgrounds finish before foregrounds. That
condition is reported as false rather than keeping decoders alive artificially;
it does not satisfy the controlled experiment's background-outliving condition.
Smoke completion and structural development agreement do not mean those cells
are physically valid or latency-qualified.

Next: preregister a cost parameterization that can actually be identified from
new calibration measurements, with observer checks, supported mode coverage,
whole-case/repeat splits and dependence handling. Freeze coefficients and
selection before fresh held TTFT validation. Keep measured wait labels separate
from service parameters. Only an accepted causal latency model should feed a
fixed-cap versus state-aware-cap comparison; MPC follows that separate milestone.

## Reviewed source boundaries

The three full upstream files below were fetched at v0.30.0 and checked against
the existing capture source pins before implementation:

- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py
  (`step_with_batch_queue`, input polling, native empty-iteration yield).
- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py
  (running-first selection, terminal guard, computed-token progress).
- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/async_scheduler.py
  (placeholder increments/decrements and PP eligibility; PP1 only here).

`next_decode_eligible_step=current_step+1` adds no extra skipped schedule in PP1;
larger PP is rejected. GPU launch/copy/update ordering uses the source-reviewed
pipeline proposal and ab24416 diagnostic. These source and development checks
do not replace a new exact-revision native acceptance run before future collection.
