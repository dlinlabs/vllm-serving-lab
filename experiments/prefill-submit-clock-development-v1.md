# Submit-return uncertainty ablation: development only

The 232-trial receipt archive at 8bd96cb validates the original measurement
protocol, not the queue predictor. A fresh TRAIN-only six-term fit passes the
probe endpoint score, but retains four queue misses: three baseline OFF and one
LIGHT sample at 50 ms/D16. Its common-scale envelope is only ±3.5034% and does
not declare this cell ambiguous. OFF receipt state is not measured.

## Evidence motivating the hypothesis

TRAIN LIGHT records independently show the first post-release background commit
at about 38–52 ms, whereas F0 predicts 13.215 ms. The same TRAIN prefix has
nonzero execute_model host spans, while F0 explicitly sets all CPU launch parts
to zero. One observed fast LIGHT probe receives at 50.899 ms before schedule 12;
a paired late receipt probe has already scheduled 12 and receives at 128.320 ms
after commit 11. These are different observed states, not evidence of a 77 ms
same-state observer shift. Their actual queue-post times are 50.136/50.114 ms.

These data motivate omitted CPU return unavailability as a hypothesis. They do
not identify a physical CPU-only cost: execute_model's host span may contain
waits and overlapping GPU enqueue work. Nor do the observations prove how an
unobserved baseline OFF trial reached its branch or rule out observer effects.

## Explicit effective model

`prefill_lab.submit_clock` retains the already TRAIN-fitted six GPU-lane
coefficients, including the mixed term. It changes only forward.cpu_after:

- P seconds when the own predicted batch has prefill tokens;
- D seconds for a pure decode batch.

GPU work and CPU unavailability may overlap in the existing max-plus event graph.
No duration is added directly to a measured queue outcome. Work, modes, release,
polls and branches still come from the model's own causal rollout. No native
future receipt state, native future schedule or measured probe queue is supplied.

Estimate P and D from TRAIN LIGHT only. For each of the 24 TRAIN LIGHT trials,
join the audited scheduler/worker step identities and retain the prefix through
foreground completion, including the pre-release background work. Validate host
start/end ordering inside its matched schedule/commit span. Take a median host
span per trial/class, then a median across those trial medians. No post-foreground
background drain or probe label enters the estimator.

The uncertainty domain for P is the minimum/maximum of all retained TRAIN
prefill host spans, bounded to [0,100 ms] as a computational readiness restriction.
D stays at its TRAIN median and the GPU coefficients stay fixed. This is an
empirical, shared-parameter sensitivity family, not confidence coverage or a
model of independent per-batch jitter. CPU/GPU common-scale uncertainty is not
combined with it to manufacture a wider union.

On the supplied archive, P is 35.155 ms, D is 1.686 ms and P's domain is
14.985–57.354 ms. These values come solely from TRAIN. Their physical placement
in cpu_after is an effective-model assumption. The development family was chosen
after inspecting calibration failures; it is not prospectively confirmed.

## Phase construction and readiness

Build a no-sentinel reference with all foreground offsets zero. Propagate exact
piecewise-affine max-plus times over the full P domain: a prefill forward.cpu_after
node has duration P and other durations are fixed. Subtract the reference's OWN
release curve. Retain every crossing, parameter endpoint and assignment tie.
There is no finite scaling grid certifying the phase result.

Allow at most two consumed input iterations. Freeze each branch as its nominal
no-probe poll boundary minus arrival, clipped at zero. A branch set is a set of
points, not a continuous interval accepting all intermediate residuals. Verify
that the independent full point rollout equals the reference nominal point.
Excessive complexity, unsupported readiness, degenerate domain, ambiguous commit
anchor, drained reference or more than two branches fails readiness. Tests compare
analytic curves with independent own rollouts; no native qualification follows.

The freeze binds the base model, plan, runtime, all TRAIN host samples, rule and
implementation hash. Verification rebuilds the entire TRAIN-bound freeze rather
than accepting a resealed self hash. Custom cost serialization retains both CPU
constants and the original GPU mixed term; generic pipeline loading rejects it.

## Scoring and limits

Report F0 and this ablation independently. Score every calibration-probe sample
against its nearest own frozen branch with unchanged max(10 ms,20% observed
residence); unresolved assignments fail. Preserve per-arm denominators, endpoint
scores, own/native LIGHT work signatures, raw measurement report and zero-count
branches. Observation assignments are not measured OFF phases or probabilities.

On this calibration archive, the point model still misses the original four
queue samples. The ablation's two-point phase score fits all 144 probe samples,
with endpoint and work-signature checks unchanged. Its nominal first post-release
commit is 35.155 ms, still 2.816–17.011 ms earlier than the TRAIN observations.
Thus the developmental phase score is not proof of a repaired nominal clock.

The ablation predicts ambiguity at both 50 ms/D0 and 50 ms/D16. The extra early
D0 branch is unobserved; receipt-only D16 also has an unobserved early branch.
Those branches are not validated by point-budget success on other samples.
A passing in-sample developmental queue score is not acceptance, cross-validation,
model selection, physical identification, independent held confirmation or cap/MPC
qualification. The code sets all production/HTTP/control acceptance flags false.

No native collector or original policy/gate is changed by this ablation. No new
GPU run is required to reproduce this analysis. Before a fresh confirmation,
review/admit or reject the hypothesis, freeze the estimator and source, complete
its independent confirmation collector and preserve the separate measurement,
queue, endpoint/anchor and integrity gates in the existing v2 design. A fixed cap
and one GPU host cannot establish action validity or MPC benefit.
