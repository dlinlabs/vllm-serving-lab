# Astra Round3: a reduced outcome model, not a qualified GPU plant

Status: bounded revised proposal and CPU source witnesses; no native performance
claim. Frozen review base: `9b02f6ebdbd6681763b562bc3c37aa564b8b330a`.
I read the complete Opus, Gemini and Sol R2 reports in that integration and their
criticisms of my R1/R2. Their findings are not accepted by vote. No GPU, provider,
SSH, final held-out, deployment or observer-overhead experiment was performed.

## 1. Adjudication and changes to my proposal

Paths below are relative to `experiments/mpc-research/native-v2/` unless stated
otherwise. R1/R2 files remain frozen.

| Criticism / source | Disposition and consequence |
|---|---|
| Opus `round2/opus-critique.md`, Astra review; `opus_counterexamples.py:288-319`: exponential duration law extrapolates poorly. | **Accept the practical objection:** withdraw an unconstrained global log-duration law as my default candidate. Its five-feature busy-time surrogate is evidence against that fit, not qualification of another family or a refutation of every supported log-link model. |
| Opus's extrapolation curve and FLOP-floor argument | **Partly reject the claimed scope.** Training uses `prefill_query_key_product_sum`, but line315 substitutes `q*(q+1)/2`. Existing `results/mpc-native-4090-scout/native-profile-diagnostic.json` has pure-prefill `(q, feature)` pairs `(128,16384)` and `(2048,4194304)`, not `(128,8256)` and `(2048,2098176)`. This is a feature-convention change, not a consistent extrapolation. A 4096-token single step is also outside the frozen 2048 scheduler-token budget. Assumed FLOP throughput is not a measured native lower envelope. These limitations do not rescue my default law; they limit the refutation. |
| Opus: cadence/window unsupported; short measured decoder residence limits planning | **Accept** that my suggested timing was unearned and that computation must justify itself. **Reject** the universal inference that short individual residence removes a multistep opportunity: rolling offers, pending admissions and future membership can couple decisions. This establishes possibility, not gain. |
| Opus's suggested irreversible TPOT pruning | **Conditionally useful**, not required now: with a verified final-token upper bound `m>=2`, at least two outputs already observed, and `(cut-first)/(m-1) > 0.05`, eventual mean TPOT cannot pass. Future output/terminal information must not enter the prefix. |
| Gemini `round2/gemini-critique.md`, Astra discussion: Poisson/no new information means no multistep value | **Reject attribution and theorem.** My claim concerned actionable information, not an iff condition for horizon gain. Even deterministic known future arrivals can make optimizing a second action useful. An uncertain state can conversely have no useful action-dependent information. |
| Gemini: finite-tail lower-bound maximization causes underadmission | **Reject attribution.** `round1/astra_horizon_diagnostic.py` computes retrospective cohort intervals; it is not an admission policy. My R1 warned against lower-score bias. No maximization of settled successes is proposed here. |
| Gemini: exact future dependency DAG is intractable | **Accept unmeasured computational risk**, but reject treating my event representation as a proved claim to solve a clairvoyant static DAG. Remove event-tree optimization from the first candidate altogether. |
| Sol `round2/sol-critique.md`, Astra section: clean archive schema, availability, joint uncertainty, one-switch search, and terminal calibration | **Accept.** The R1 complete/rejected TRAIN loader is not a general failure evaluator. Future failed/cancelled offers remain zero-valued outcomes, not dropped rows. Prefix availability needs the actual consumer clock. Tail intervals do not calibrate prediction error. A pruned search needs comparison with the complete small action set before use. |
| Sol's release-state alias and lower-score counterexamples | **Accept within stated scope.** They motivate marked costs/residual uncertainty and confirm the accounting warning. They do not prove different optimal native budgets, equivalence of complete histories, or native multistep gain. |

Other explicit rulings: **withdraw Gemini R2's 4.5us/450us/1200us and zero
serving-jitter assertions from usable evidence**. They are assumed fixture
constants (`round2/gemini_counterexamples.py:182-213`), not measurements; I do not
endorse them. Falsifying the HTTP serial-service interpretation or an affine
law does not select a roofline/hinge family. Opus's fitted mean busy-time law is
not a certified lower envelope, and survivor delta-output per snapshot gap is
an effective commit-rate proxy until participation and async parity qualify it.
Equal whole-sequence Fixed16/32 scores do not exclude switching or heterogeneous
effects. Physical memory implies neither gain nor an optimal horizon; low
decision disagreement frequency gives no impact bound.

My R2 nested-knee comparison remains the appropriate comparison to a globally
selected knee: 1.706663ms versus the reported 1.361494ms MAE, on the same limited
profiler evidence. Neither number is native control validation. I retain the
finite-cohort bounds and strong common-continuation One-move comparison, not
any claim that they alone supply a controller.

## 2. Concrete bridge review and the one required acceptance fix

Reviewed `batch_capture_bridge.py` at the integration above; it is byte-identical
to Mayor's `1cbe251` version, SHA256
`0e24eb2c00d3b3da08f01a0d235e4a397551eee17f6853f5fe23dca295498f4a`.
Also read its focused tests and the reused PR32 source. Mayor's 80 CPU passes /
one installed-vLLM skip are a supplied receipt, not my rerun or native evidence.

**Sound scope and lifecycle.** Module loading is stdlib-only. `register` and
`Endpoint.__init__` return when disabled, before installing native hooks/routes;
the source-guarded PR32 imports are lazy. Enabled operation requires the passive
flag combination, frozen TRAIN170001-170012 plan and authenticated endpoint.
`metadata` checks TP=PP=1, one worker, source hash, clock domain and positive
worker PID; `begin` rechecks the identity tuple before native BEGIN.
`end` flags changed identity and preserves invalid results. These are useful
process/source checks, not proof of installed deployment or zero observer effect.
The plan's `source_commit` is a declaration: the collector must compare actual
installed 38-source evidence and bridge/plan hashes with its frozen expectations.

`BatchCapture.invoke:153-182` retains pending state and replays the exact completed
END reply until matching trial/SHA ACK. Failed BEGIN invalidates the capture
rather than pretending it never happened. Raw invalid records must be persisted,
not simply excluded without an exclusion receipt. Retention is explicitly
in-memory, not crash durability. A crash before persistence leaves a missing
capture, not permission to reconstruct a successful one.

**Executed CPU witness / proved validation gap, not an observed native mismatch:**
`capture_errors:48-60` compares engine/worker step-key sets but does not compare
scheduled versus executed counts or input positions. The owned witness supplies
one finished natural-EOS request, a 128-token scheduled step, matching keys, and
either `executed_counts={"r":127}` or `executed_positions={"r":1}`. Both receive
`capture_valid=true`, no validation errors, and preserved raw mismatch. Thus
this flag is only a structural transport gate; its name must not become a
certificate of executed work.

**Smallest required fix before accepting numerical labels:** reuse the existing
`prefill_lab/contracts.py:audit_capture` executed-count, position, actual-total
and request-progress checks in the collector acceptance gate. For successful
natural-EOS request audits set `forced_outputs=False`; enforce step evidence for
LIGHT, not OFF. Keep failed/cancelled requests in the all-offered outcome ledger
and audit their recorded partial work separately: the existing success-output
predicate must not erase their workload or outcome. Do not require the restricted
FIFO reference as the definition of normal continuous batching; distinguish a
reference-policy mismatch from scheduler/worker disagreement. Add the two exact
mismatch cases to the collector's qualification tests. This is a small reuse
requirement, not a new observer pipeline or an implemented change to Mayor's code.

**ACK lost response is not a demonstrated data-loss defect.** ACK retries currently
raise after the first ACK cleared retention (`invoke:123-129`). My witness confirms
this, exact END replay without another native END, and recovery information in
`status`. A single collector must fsync raw capture and its SHA receipt before
ACK, and, after a lost ACK response, verify the same run/plan/bridge identity,
trial in `used_trials`, no active trial and no unacknowledged trial. Then continue.
Changed run or contradictory state must stop/reconcile, not resend workload.
An idempotent last-ACK tombstone would simplify the client, but is not a required
bridge fix if this status path is implemented and tested. No other collector may
concurrently consume that capture. BEGIN/END response-loss recovery also needs
qualification in the actual collector, beyond these in-process cases.

**Observer and scope caveats.** `prefill_lab/plugin.py:core_runtime` changes
`sys.setswitchinterval(5e-4)` when metadata initializes the runtime; installed
hooks, request bookkeeping and worker calls exist even in enabled/OFF.
Literal disabled and enabled/OFF are different treatments. `WorkerRecorder`
uses current-stream preparation/forward events and deferred resolution: its
elapsed span includes that stream's ordering/waits, excludes unjoined other-stream
work, and is neither kernel-busy union nor whole-pipeline service. Host execute
return is not GPU completion; schedule-to-output-commit spans may overlap.
`cuda_scope_qualified=False` is an honest boundary, not a reason to rename the span.
Both engine and worker 8192 buffers invalidate on overflow; never silently drop
old rows. Bound capture duration and inspect row totals/pending/drain/overflow
before any fit. Do not add serving-path synchronization or invented GPU/host
microsecond alignment.

The ordinary HTTP capture is not itself a frozen request-ID manifest. Require
external offered ID -> Gateway reservation/admit/reject -> engine request ID ->
scheduled work/step key -> executed worker work -> committed output -> client
terminal joins. Include accepted-but-not-native-visible work as unknown/pending,
not zero. A clean bridge flag without these joins is insufficient. Transport-valid
captures can still have invalid labels; valid failed-request outcomes can still
be required for policy evaluation.

## 3. Smallest predictive candidate I would now test

**Target, not hidden GPU reconstruction.** Estimate a conditional distribution of
whole-reservation release trajectories and final composite-good counts for a
fixed offered cohort, under a small randomized two-command policy family and
common continuation. Start with an empirical stratified response table:
action sequence, pre-action reserved-cost bin, and a coarse short/long pending
composition stratum. Compare with action-only and hold/no-change predictors.
Use unsmoothed block-level empirical distributions initially; no interpolation
across unsupported cells or extrapolation to new scheduler settings. If this is
too sparse, retain action-only whole-policy estimates, not fabricated dynamics.
This is a support-limited reduced predictive model, not a qualified Markov state.

The full logged online history contains per-request original cost, prompt class,
admit state, age, prefix output count and first-content status if actually
available, pending/unknown native work, reservation total, recent arrivals,
applied cap and command age/fence. These are observations, not clairvoyant
remaining tokens. Only fields delivered to the controller before its decision
enter features. End-of-drain CUDA labels and later output callbacks are offline
labels, never backdated online features. EOS, remaining generation, hidden KV
occupancy and unobserved batch participation stay latent. Batch annotations may
explain residuals offline; they need not be online inputs to the first model.
Disturbances include future offers/prompt mix, EOS and system/GPU jitter. A frozen
experimental tape conditions this pilot; it is not a future-arrival oracle for
an eventual online controller.

For a cut `t`, define the cohort as outstanding earlier offers plus all offers
scheduled in `[t,t+D)`. Predict release event times/costs and the final good
indicator for each relevant outcome, or their jointly sampled cohort totals;
do not predict success by replacing each terminal time with a mean. Score the
same cohort and continuation for every action. Success requires complete
response, scheduled-client TTFT <=0.5s and request-mean TPOT <=0.05s; one output
token has TPOT=0. Reject/error/cancel is zero. Administrative censoring remains
unresolved until drain/deadline classification, not implicitly successful.
Bootstrap/resample complete episode/seed blocks jointly, preserving correlations
between timing, EOS, release and success; independent per-token draws are invalid.

State evolution has an exact accounting part:
`R_next = R + sum(newly_admitted original_cost) - sum(terminal original_cost)`.
Here terminal means Gateway reservation finalization/cleanup, not merely model
EOS or the first client terminal notification; use the actual release event.
Admission requires `R+c <= applied_cap`. Reducing 32 to16 while `R=32` retains
all reservations. The history-max bound has no extra-offer allowance under the
serialized projected-cost check. Neither cap is concurrency or a vLLM scheduler
token actuator. KV risk is not cost in disguise: unknown KV/preemption/OOM behavior
limits support and causes fallback/abstention, not a claimed safety guarantee.

Randomizing complete command sequences identifies their intention-to-treat
policy effects within the tape/support, including actual command latency. It
does **not** randomize intermediate batch membership, isolate a prefill causal
coefficient, or identify an adaptive policy at every rolling state. Actual applied
sequences/fences must be logged; conditioning on successful application can bias
the randomized comparison. Report assignment and application deviations together.
The first response table is qualified only at the randomized decision cuts.

**Control comparison and computation.** Initially use `D=5s`, two possible
1s command slots, budgets {16,32}, and an explicitly shared continuation after
slot two; these are pilot design choices, not optimized native timings.
Enumerate all four pairs. Strong One-move optimizes its first slot over the same
5s outcomes and old cohort, then uses a continuation selected on development
data from hold, Fixed16/32 and a frozen strong reactive rule. Multistep uses the
same continuation after its optimized slots. Both eventually replan at the same
rate; an open-loop rollout is only a plan-scoring experiment, not certification
of a deployed receding-horizon policy.

The pilot's four fixed sequences support only its fixed continuation, not a
reactive continuation or arbitrary rolling replans. The initial table must not
score the latter by stitching unrelated rows. Use additional randomized support
for the frozen strong continuation before that comparison; otherwise report the
strong One-move question unresolved and do not promote the model to control.

Do not declare multistep gain from different future actions alone. On independent
TRAIN blocks require an uncertainty-resolved difference in first action versus
strong One-move, subsequent action actually capable of affecting admissions,
and improved all-offered score after measured decision/actuation cost. A later
direct policy comparison must randomize the complete replanning algorithms.
Rank supported plans by expected final composite-good count over that fixed
cohort, not settled lower-bound count; permit a change only when joint uncertainty
and measured overhead clear the frozen gain margin while accounting/support
constraints hold. No uncalibrated chance constraint is a KV-safety certificate.
If first actions coincide, keep One-move; record conditional impact at rare
disagreements rather than equating their frequency with lost goodput. Fixed16,
Fixed32 and the frozen reactive rule remain required deployment baselines.

A response table evaluates four candidate paths, not an exponential scenario
tree or nonlinear solver. For `M` coherent empirical trajectories per candidate
and `K` scored grid points, evaluation is O(4MK); unsupported-cell checks precede
ranking. Measure end-to-end feature/update/evaluation time, not just scoring.
Provisional decision deadline is 100ms at a 1s cadence, subject to measured
actuation headroom. Timeout, stale observations, unresolved safety/support or
ambiguous gain -> no new command / the frozen previously qualified fallback,
with existing TTL/fence semantics respected. Never mint an unfenced cap update.

Evaluate one-slot release/outcome calibration, two-slot joint path calibration,
and final cohort ranking on separate TRAIN blocks; use one frozen alternative
short/long mixture as an OOD challenge later, not the final held-out IDs.
Any OOD goodput claim requires new randomized data. Final tail scoring uses
complete outcomes where available and Astra's `[L1-U0,U1-L0]` contrast otherwise,
with common maximum drain and censor reporting. Tail uncertainty and model
uncertainty are separate; neither is repaired by increasing the optimized horizon.

## 4. One highest-information bounded native TRAIN pilot

**A measurement-qualified randomized two-slot sequence experiment**, not a live
predictor trial. Mayor alone authorizes execution; none was executed here.
Freeze source/config/tape/plan hashes and all IDs before serving. Use two
independent allowed seed blocks from170001-170012, assigned by Mayor. Within each
block reuse a frozen 12s mixed short/long ordinary-Chat offer tape across arms,
with the same offered times, prompt/max-output parameters and request-ID mapping.
Preserve natural EOS. Same-sequence repeats separate stochastic repeatability
from treatment differences; same seeds are not proof of identical token streams.

Use four sequences LL/LH/HL/HH, L=16/H=32. First2s use the common frozen baseline;
assign slot1 at2s and slot2 at3s; after4s use the common frozen continuation.
A simple initial sequence-identification continuation is Fixed16, expressly not
the final strong One-move benchmark. The 5s prediction cohort at2s includes old
outstanding offers and new offers in[2,7); all offers in the full12s tape also
receive a separate complete policy outcome ledger.

Compare each sequence in **literal disabled**, **enabled/OFF**, and **LIGHT**
process configurations. Literal disabled makes no facade metadata/begin/end
calls and uses the same external client/Gateway receipts. Enabled modes perform
the same lifecycle; only LIGHT emits batch CUDA/host rows. Restart/reset between
literal disabled and enabled as needed so the GIL/hook state cannot leak across
them. Match warmup/reset policy and randomize process-group order across blocks;
report unavoidable order confounding rather than pretending launches are free.

Per block: four sequences x three modes, plus one independent LIGHT repeat of
each sequence =16 episodes, **32 episodes total**. Randomize sequence/replicate
order within mode groups and retain every assignment. Freeze a30s post-offer
drain limit; still-unfinished offers are censored/failed according to the frozen
evaluation rule, and capture cannot be called complete unless actually drained.
This is a bounded acquisition design, not an assertion that 8192 rows suffice:
overflow/pending/failed drain invalidates numerical labels and triggers revision,
while preserving all client/Gateway outcomes. Do not silently shorten the tape
after looking at scores. No final160021-160024 access or tuning.

Mandatory receipt per episode: mode/source/config/plan/run identity; offered
manifest and actual client arrival; assigned/applied cap timeline, ACK and fence;
each original cost and exact release; every terminal success/failure; raw
engine/worker capture and its fsynced SHA before ACK for enabled modes.
Separate event occurrence clock/domain from observer availability timestamp.
No cross-clock subtraction without qualified mapping; current-stream CUDA
durations need no fabricated absolute timestamp. Join scheduled/committed work
through existing step keys and executed positions/counts, not nearest timestamps.

Primary questions: do disabled->OFF and OFF->LIGHT effects or data loss prevent
trusting the labels; are releases/outcomes repeatable at a fixed sequence; and
does switching show a supported effect that fixed-arm averages miss?
Report mode-specific block contrasts for good/offer, TTFT/TPOT, rejects/failures,
command lag, raw row counts and scoped forward/commit labels, with complete-tail
and unresolved-tail versions. Use block, not request, as the independent unit;
the four LIGHT repeats in one seed do not create four independent seeds.
Two blocks are an initial falsification/engineering pilot, not adequate evidence
for a0.5good/s threshold or a statistically certified controller win.

Continue only if identities/action/outcome joins and loss recovery pass, no
unexplained capture loss remains, and mode effects are bounded tightly enough
for the predeclared application margin. An inconclusive overhead interval is
not zero overhead: extend independent TRAIN blocks before numerical acceptance.
If exact released-cost accounting fails, stop. If executed parity fails, retain
raw records and fix the acceptance/instrumentation boundary. If observation
changes serving materially, revise LIGHT or model the measurement treatment,
rather than subtracting a fitted constant. If supported sequence outcomes are
indistinguishable relative to noise, keep simpler control and improve identification.
Only after useful policy effects and cross-block predictive ranking appear
should the strong One-move/multistep replanning discriminator be attempted.

## 5. Reproducible CPU receipt and remaining disagreements

Owned `round3/astra_revision.py` loads only the SHA-pinned frozen bridge from local
Git and stubs the core RPC/capture. Its source-hash function returns the verified
blob hash because the file is read from Git, not installed. No clock/RPC fixture
is evidence of a real engine, CUDA, HTTP timeout, throughput or overhead.
`round3/test_astra_revision.py` has four focused cases: wrong executed work,
exact END replay, lost-ACK status reconciliation, and metadata-to-BEGIN worker
epoch change. Tests intentionally demonstrate the frozen validator's false
acceptance; they do not claim the collector fix has been implemented.

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/mpc-research/native-v2/round3 -p test_astra_revision.py -v
PYTHONDONTWRITEBYTECODE=1 python3 experiments/mpc-research/native-v2/round3/astra_revision.py
```

Result: **4 tests passed**; both wrong-work fixtures returned `capture_valid=true`
and empty errors, with raw mismatches retained. Python3.14.4; stdlib only.
No installed-vLLM or GPU tests run. Frozen R1/R2 and raw archives were not changed.
Remaining disagreements are scientific, not deployment vetoes: neither a
particular roofline family, an exact delta-output iteration clock, a certified
busy lower envelope, nor a memory-based multistep theorem follows from current
evidence. A support-limited whole-policy model can be attempted now; qualification
and adaptive-control advantage remain open experiments.
