# Sol Round3: a support-limited release/progress predictor, not a GPU twin

## 1. Decision and scope

**No qualified native MPC model or performance winner exists.** I would first
qualify measurement and randomized **budget sequences**, then attempt a small
history-conditioned predictor of cost release and composite-good completions.
Perfect GPU decomposition is not a prerequisite. Unsupported internals remain
latent; a physical lower envelope and a calibrated native EOS model are absent.

Review sources are the complete peer R2 reports at common commit
`9b02f6ebdbd6681763b562bc3c37aa564b8b330a`, and concrete bridge commit
`1cbe251ced4d54bec8f56cfbce95983081abc16a`. Paths below are repository-relative;
`round2/` means `experiments/mpc-research/native-v2/round2/`.
No new GPU observation, installed-vLLM execution, held-out access, or native
performance measurement was performed. Mayor's 80 CPU passes/one installed
skip are reported evidence, not reproduced here and not native qualification.

## 2. Response to every substantive criticism of my design

| Peer/source | Adjudication and revision |
|---|---|
| Opus, `round2/opus-critique.md:243-250`: toy timescale versus <=0.92s short-cell decode residence | **Accept the scope warning.** The finite witness does not establish native decision relevance. Full old-cohort release before the next relevant decision removes that particular ambiguity; continuing admissions can still matter. Mixed long cells and actual binding-state incidence must decide relevance, not toy seconds. |
| Opus, same file `251-256`: unspecified log-time law and sparse binding states | **Accept underspecification and support concern.** Drop the proposed microscopic law from the first candidate; count eligible, binding contrasts before promising local causal estimates or power. **Do not accept** a fitted mean busy-time curve as a certified physical floor. |
| Astra, `round2/astra-critique.md:365-407`: no optimal-action proof, unobserved high-dimensional belief, retransformation, availability and tail | **Accept.** Costs and residual splits prove a projection obstruction, not opposite optimal caps; cap32 accepts the toy offer in both worlds. No current adapter identifies my full R1 belief. Replace it with the small predictive target below. Do not equate `exp(E log tau)` with mean time. Preserve joint residual dependence and continuing-traffic scoring. |
| Gemini, `round2/gemini-critique.md:83-87`: no executable predictive controller | **Accept the implementation gap; reject the attributed conclusion.** R1's batch/semi-Markov proposal and R2's reduced-policy recommendation do not forbid predictive control. None was implemented or qualified. This revision proposes a deliberately smaller first fit, not a claim that it already works. |
| Gemini, same lines: independent-rate witness invalid under lockstep | **Reject as a refutation of the limited proof; native magnitude unresolved.** In discrete lockstep, give each surviving decoder one token per common step: residuals `(1,9)` versus `(5,5)` still release cost8 versus0 after two steps, with identical initial request counts/costs and total residual10. That is a finite construction, not a native batch law. Gemini's one-request versus two-request illustration at `75-77` is not my same-projection witness. |

I also revise my R2 pilot: fixed-cap comparisons alone are insufficient.
Opus `round2/opus-critique.md:392-396` incorrectly concludes that equal Fixed16/32
scores exclude all switching gains. A two-period reward table with per-period
rewards `(2,0)` for cap16 and `(0,2)` for cap32 gives each fixed policy total2,
while switching16->32 gives4. This is a mathematical counterexample, not a
native mechanism or fitted reward table. Physical memory does not imply useful
multistep control, either; low disagreement frequency does not bound its impact.

I explicitly withdraw from consideration Gemini R2's **4.5us, 450us, 1200us,
negligible overhead and zero serving-jitter claims**
(`round2/gemini-critique.md:117-137`). They are unmeasured fixture assumptions,
not facts I can adopt. Post-drain resolution avoids one source of active
synchronization, not all observer effects. Nor does falsifying an affine law
validate a roofline family. Opus's newly fitted decode delta-output relation
remains an **effective commit-rate proxy**, not actual iteration time without
participation/step/async qualification. Astra's finite-tail intervals remain
sound accounting diagnostics; I do not attribute an admission policy to them.

## 3. Concrete bridge review and the one required qualification fix

All bridge line references here refer to `1cbe251:experiments/mpc-research/native-v2/batch_capture_bridge.py`.

**Useful, source-supported behavior.** `register:201-215` and
`register_endpoint:218-243` return before enabled-only loading/routing when
disabled. The authenticated endpoint exposes passive operations, not request
generation or controls. `BatchCapture.invoke:100-118` qualifies TP/PP1, worker
count, positive PID, source digest and clock domain; BEGIN rechecks the worker
identity (`130-152`). Marking a failed BEGIN active/used and invalid is safer
than silently retrying an ambiguous start. END rechecks epoch, retains pending
event state, and retains invalid raw results (`157-183`). A changed core yields
a different bridge run identity; the collector must reject cross-run joins.
These checks do not prove an immutable GPU generation merely from a PID.

**Loss/replay protocol.** Lost END response: repeat END and persist the exact
retained packet. ACK only after durable write, fsync and independent digest
verification. Lost BEGIN response: reconcile status in the same run; do not
reoffer the tape blindly. Duplicate ACK currently raises after the first ACK
has cleared memory (`123-129`). This need not require a bridge change: after
an ambiguous ACK, a same-run status with no unacknowledged trial and that trial
still used, plus the collector's durable packet/ACK journal, can reconcile it.
A changed run or a different retained trial is not that evidence. The SHA
acknowledges the raw capture, not every envelope field; bind plan/source/run
identities separately. No process-crash durability is promised.

**Default-off is not enabled/OFF.** Reused `prefill_lab/plugin.py:38-43,147-150`
installs hooks and sets `sys.setswitchinterval(5e-4)` upon metadata/core-runtime
initialization. Observer OFF avoids batch events, not this GIL change, Python
hooks or request/output bookkeeping. This is a real configuration difference,
not a measured slowdown. Separate literal-disabled, enabled/OFF and LIGHT
runs are required; do not silently restore/change the interval to claim parity.

**Buffers and timing scope.** Reused `WorkerRecorder:251-290,325-345` has8192
bounded entries/pending events; overflow invalidates evidence. At an assumed
100 steps/s, 90s already needs9000 entries before drain. This arithmetic is
capacity planning, not measured native cadence. Short pilot windows and explicit
post-drain counts are preferable to increasing buffers speculatively. Current
stream preparation/forward event spans do not include every other stream,
sampling or the whole pipeline. Host schedule-start/output-commit spans may
overlap and must not be summed as exclusive service. No hot-path sync or
microsecond client/server clock mapping is needed.

**Executable scope counterexample, not an observed native failure.**
`capture_errors:36-67` checks IDs, terminal presence and finite spans, but not
actual work/position parity. My hash-pinned CPU fixture sets scheduler work128
and worker `executed_counts=64`, with the same step ID: END returns
`capture_valid=True`, preserves raw data and reports no validation errors.
The existing `prefill_lab/contracts.py:audit_capture` rejects it as
`Scheduler/worker work mismatch`. A second fixture sets scheduler
`parity.matches=False`: the bridge flag remains true while the existing audit
rejects restricted FIFO parity. This flag therefore cannot certify the
downstream batch model. A FIFO mismatch disqualifies that restricted model,
not necessarily all useful coarse policy measurements.

**Only required fix proposed:** before marking LIGHT batch labels
restricted-model-qualified,
the collector must reuse the existing post-drain
`audit_capture(capture, rows, forced_outputs=False)`, retain its problems and
async/scope labels, and reconcile the frozen offered/Gateway/native identity
map. Keep `capture_valid` explicitly a thin bridge check. No new hot-path
validator, synchronizer or capture pipeline is needed. If the prepared collector
already does this, no bridge source change is required. Request terminals alone
are not complete successful Chat responses. Bridge `validate_plan:17-33`
freezes trial IDs/seeds/modes, **not** the offered request manifest: the collector
must enforce that separately, including no extra native IDs and every accepted
request accounted for. Gateway rejects belong in outcomes, not expected native
work. Do not force natural EOS to max_tokens; that is why the audit needs
`forced_outputs=False`. Exact stream/step parity still needs installed/native
qualification beyond this CPU case.

## 4. Smallest candidate I would attempt after qualification

**Plant and exact state.** Gateway cap16/32 is reservation COST, not concurrency,
KV bytes or vLLM scheduler-token budget. Keep the actual marked reservation
ledger exact: admit iff `R+c<=u_applied`; release the original cost once at
terminal cleanup. A cap reduction changes neither old cost nor old work.
With atomic projected check/add there is no extra one-offer overshoot term;
`R` can exceed the current reduced cap. Scheduler token budgets remain fixed.

**Predictive target, not a decomposition.** At a decision cut t, predict jointly
(a) cumulative release of the then-live marked cost by fixed offsets, bounded
in `[0,R(t)]`, and (b) complete composite-good response counts in a fixed future
window. Good means complete successful response, scheduled-origin TTFT<=0.5s
and request-mean TPOT<=0.05s (single-output-token convention:0), not token-tail
P99. These are native terminal/commit outcomes, not exclusive GPU service.
EOS/remaining work stay latent; censored outcomes stay unresolved, not zero
residual or assumed success. A history-conditioned distribution may represent
the alias without per-request particles.

Start with a small regularized direct regression and empirical **paired**
release/count residuals on supported histories and assigned sequences, not an
unfitted kernel simulator. Features: R and marked cost/age bins, available
waiting prompt mix, decode/output-progress bins, recent offered mix/rate,
current cap plus acknowledged control history, observation age and unknown-join
counts. Ablate the native phase/progress bins against ledger/history-only and
hold-current baselines. Do not add a KV regime, hardware mode or pending-step
DAG that the adapter cannot observe. Missing fields are explicit; support
failure invokes fallback, not invented zeros. Small block counts can make even
this fit infeasible; that is a legitimate pilot result.

**Identification and availability.** Labels require full post-drain joins;
features must have been received by the logged decision cut, with provenance
and known clock domain. Later CUDA spans and future tape contents are never
features. Native observation source time alone is insufficient. Use client
scheduled-origin TTFT/request-mean TPOT and complete success from the client
clock; do not subtract a worker timestamp from an unrelated client timestamp.
For cross-domain joins use request/step IDs and receipt brackets, not invented
precise offsets. Randomized whole-sequence assignment identifies its policy
effect under the declared block/randomization design. It does **not** identify
a direct causal coefficient of batch composition. Conditional forecasts need
actual overlap/support; assignment randomization does not create every
state/action/history contrast. The first pilot below supports its assigned
A/B programs, not arbitrary replanning sequences or all eight three-move plans.

**Potential controller, not yet deployed.** A later supported candidate could
enumerate up to eight three-move cap sequences at1s cuts, mask unsupported
plans, issue only the first move,
and score all candidates over the same5s window plus the same declared
continuation/tail. These periods are hypotheses pending measured release and
full-path timing. The objective is all-offered composite goodput, not accepted
only throughput or a token-rate surrogate; report rejection, successful
completion and late-drain counts separately. Respect exact applied caps/fences,
existing timeout/safety rules and qualified operating support. No forecast
certifies KV safety: respect existing memory guards and fall back outside
qualified operating support, without requiring an unavailable KV allocation
feature or pretending that its absence means zero occupancy.
Fallback is the best frozen development fixed/reactive policy, with cap16 as
the conservative degradation action only where that behavior is qualified;
lowering it does not evict already accepted work.

Measure observation receipt, feature/update, enumeration, command/ACK and
deadline misses together. Continue only with a predeclared timing margin below
both the control interval and existing permit expiry; eight cheap evaluations
are not a latency measurement. Missed/stale/uncalibrated/OOD cases take fallback.
Use block-held-out prediction error, baseline comparison and joint trajectory
coverage diagnostics. Empirical residuals from a few seeds are not certified
tail/chance constraints.

**Actual multistep discriminator.** Strong One-move chooses its first action
with the **same5s prediction window**, future traffic law, uncertainty samples,
tail and development-frozen continuation as the three-move candidate; it is
not a short-sighted one-slot baseline. One-move frees only its first move and
then uses the backup; three-move frees three moves and then uses that same
backup. Both actually replan at the same cuts. Continuation must compete with
the best development fixed/reactive backup, not an intentionally weak hold.
Log first-action disagreement, binding incidence and affected offered mass,
then evaluate realized paired policy effects and continuing-traffic tails.
On sufficiently supported pre-action disagreement states, randomize the two
first actions and thereafter use a common frozen continuation to test first
action value separately from whole-policy rollout benefit. If such states are
rare, no powered local test is promised. Compare Fixed16, Fixed32 and reactive
on the same tape/cohort too. No memory theorem, disagreement percentage, longer
scoring window or favorable synthetic horizon can substitute for this test.

## 5. One highest-information bounded TRAIN pilot

**Proposal, not executed:** a measurement/sequence-identification pilot, with
no live predictor or synthetic requests. Freeze three independent TRAIN seed
blocks170001-170003 (inside the allowed170001-170012), each a12s standard-Chat
mixed short/long tape with burst phases, sampling seeds, prompt hashes,
scheduled times, costs, max_tokens and logical IDs. Repeats reuse that tape,
not new independent block identities. Outputs terminate naturally.

For each block run three literal modes: bridge/PR32 capture hooks disabled
(no metadata activation/GIL change); enabled with observer OFF; enabled LIGHT.
Keep any separately qualified read-only observer cadence identical across
modes, and declare that common observer overhead; this is not a claim of
wholly uninstrumented vLLM. Use clean processes for mode changes, randomized
mode order, and frozen model/software/hardware settings/source38 receipts.
No actual process or provider changes are authorized by this report.

Within each mode randomize order of two assigned cap sequences, each repeated
twice: A=`(16,32,16)` repeated four times; B=`(32,16,32)` repeated four times,
one cap assignment per1s slot. Total36 episodes,432s offered time plus drain
and process setup. This deliberately tests history-dependent policy support;
it is not a full factorial state-local randomization or an MPC comparison.
Any unsupported application cadence must be qualified first, not silently
clipped. Record intended/applied/fenced assignments and ACK lag; never select
episodes post hoc according to favorable actual application. Analyze assigned
sequence effects, with application failures retained and separately flagged.

Start empty, use identical continuing traffic through the12s boundary, then
drain with a frozen15s deadline. Retain all offered rejections, errors, client
cancellations, incomplete/late outcomes, first/last/token timestamps and exact
reservation releases. Primary score counts complete composite-good responses
by12s; separately report eventual complete-good count for the fixed offered
cohort. Deadline-unresolved outcomes remain explicit accounting intervals and
exclude the episode from *qualified training*, not from failure reporting.
Short windows do not guarantee8192 safety: inspect counts/overflow after drain.

Reconcile source/plan/run/core/worker identities, request mappings, pending
events, full work/position/commit joins and both audit scopes before using
LIGHT labels. Preserve invalid raw packets and replay/ACK journals. Native
batch labels are diagnostic mediators, not identified causal contributions.
Predeclare practically relevant observer-effect margins from the serving SLO
and collect per-block disabled->OFF and OFF->LIGHT contrasts, deadline/TTFT/TPOT
changes, step-count distributions and application lag. Three independent blocks
plus repeats cannot establish zero effect or a performance winner: use exact
randomization consistent with the blocked assignment, show paired contrasts
and uncertainty, and do not inflate n with episodes/requests/steps.
Shared sampling seeds do not guarantee identical outputs or a common native
random-number path under different batch histories.

**Continue** only if identity/audits/application and terminal accounting pass,
instrumentation effect is sufficiently bounded for the intended conclusion,
and there is enough binding mixed-history support to beat simple release/count
prediction baselines on blocked TRAIN validation. Otherwise **revise** the
measurement scope, cadence or workload support; if uncertainty is too wide,
declare it inconclusive rather than accept equivalence. If policy sequence
contrasts exist without usable batch decomposition, proceed with the reduced
policy predictor. If they do not, report the tested-support limitation, not
"no possible switching gain." Reserve controller discrimination for a later,
separately frozen development experiment. No final held-out160021-160024 use.

## 6. Reproducible CPU receipt and remaining disagreements

Owned `round3/sol_revision.py` loads the exact frozen bridge source and verifies
its SHA256, then executes its real lifecycle on fake core/worker replies.
It AST-loads only the unchanged `finite`/`audit_capture` functions from the
hash-pinned contracts source, avoiding NumPy/native imports. Source identity
plumbing is explicitly mocked; these are neither installation tests nor GPU
observations. The matching natural-EOS control passes the existing audit;
both malformed-scope fixtures retain raw evidence but fail that audit.
Lost-END replay succeeds; lost-ACK reconciliation requires status, not blind
duplicate ACK success.

Commands from repository root (Python3.14.4):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/mpc-research/native-v2/round3 -p 'test_sol_revision.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 experiments/mpc-research/native-v2/round3/sol_revision.py
```

Result: **4 focused tests pass**; diagnostic emits both scope counterexamples
and the loss/replay receipt. No new fitted model, solver-runtime claim, native
overhead estimate or winner is implied. Remaining disagreements: fitted floors,
four-scalar safety sufficiency and fixed-policy-equality impossibility are
unsupported; the native size of marked-release ambiguity and incremental
multistep value remain unresolved.

## 7. Frozen collector addendum: qualification gate now present

Subsequent narrow source review at
`ab8c3920a1751be1fe3974f378719d523f0627e0`, limited to
`experiments/mpc-research/native-v2/collect_batch_qualification.py` and its
focused persistence tests:

**The required collector-side gate from section3 is implemented.**
`collect:192-207` requires complete HTTP outcomes/usage, reconciles the exact
offered ID set with native IDs, and invokes the existing audit with
`require_steps=(observer=="light"), forced_outputs=False`. It retains the audit
and stops on failure. Thus the bridge's thin `capture_valid` flag is not its
sole qualification criterion. No additional bridge validator is required by
my two CPU scope witnesses.

**Persistence order is sound within the stated fail-stop workflow.**
`CaptureClient.call:73-92` durably records raw utility responses before parsing
or accepting source/plan/clock/run identities. `finish:94-115` resolves pending
events, checks trial/digest, exclusively creates the capture file, flushes and
fsyncs its contents and containing directory (`write_new:26-36`), reads it back
and verifies the raw-capture digest, then ACKs. Invalid bridge captures are
also saved before ACK and then rejected. ACK preceding the later scientific
audit is acceptable: it releases the bridge's in-memory copy, not the durable
collector evidence, and does not by itself mark the experiment accepted.

There is no automatic lost-END/ACK retry or process-crash resume here.
Timeouts, ambiguous ACK responses and exclusive-file collisions stop collection;
they are not silently accepted trials. Recovery requires the saved packet and
same-run reconciliation described above, rather than rerunning the tape or
blindly rerunning `finish`. One portable power-loss durability caveat remains:
`collect:122` creates a new output directory, but its own parent directory is
not explicitly fsynced. If surviving loss of that newly created directory entry
is part of the durability guarantee, fsync that parent after creation (and any
new ancestor entries), or start from an already durably provisioned directory.
This is a filesystem-contract caveat, not an observed power-loss failure.

**Scope remains appropriately narrow.** The frozen16-arm plan covers two
TRAIN ordering seeds, p128/4096, cohorts1/4 and enabled OFF/LIGHT at fixed32.
It is a join/drain/natural-output smoke, not my later sequence-identification
pilot, literal-disabled comparison, overhead equivalence, service calibration,
or performance/MPC acceptance. The manifest/completion scope explicitly agrees
(`133-137,208-210`). No demonstrated critical defect in the reviewed scope
requires declaring this smoke invalid in advance.

The four collector persistence cases were read, not rerun; they cover
valid/invalid save-before-ACK and digest/epoch rejection, not power-loss
durability or installed/native execution. This addendum changes documentation
only; the four owned CPU cases in section6 are unchanged.
