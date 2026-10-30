# TRAIN-bound phase-margin proposal

This implements the CPU part of [the review on f7085a9](https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6437204879).
The original native rejection remains unchanged. The new proposal also rejects
the retained development data. It is not enabled as a native clock/HTTP gate and
is not a reason to rent another GPU for the unchanged experiment.

## Calibration and freeze

Fit the existing six-term law using the original TRAIN OFF endpoints and TRAIN
LIGHT events. The phase layer changes no coefficient, fitting objective, endpoint
normalization, request offset or numeric error budget. For each TRAIN OFF endpoint,
compute observed/predicted minus one. The maximum absolute residual over all
TRAIN cases, repeats, positions and TTFT/completion metrics is the symmetric
common-scale half-width `s`. There is no held-derived floor, quantile, alternative
width selection or search for a passing score. On the retained 4c89188 data this
gives s=0.04626523891083423, from calibration only.

This is an empirical calibration envelope for a shared coefficient scale. It is
not a confidence interval, population coverage claim or bound on independent
CPU/GPU/network errors. A width above the declared 0.1 maximum fails readiness.
The 0.1 limit is a complexity/scope restriction, not a tolerance used to accept
held errors.

Seal policy, plan, runtime, model, original point predictions, exact TRAIN rows,
width and branch predictions before scoring held outcomes. Evaluation rebuilds
the entire freeze from TRAIN; rehashing a wider branch set cannot manufacture
coverage. Unknown timing sources, graph assumptions, more than two branches,
collapsed points, excessive piece counts or arrival after reference foreground
completion fail readiness. Six timing-source hashes pin the reviewed algebra.

## Continuous phase classification

Generate an independent no-sentinel reference from the offered burst/backgrounds.
All foreground offsets must be zero in that reference. The effective lane has
only a forward GPU-tail law and the fixed native 1 ms idle yield. Under a shared
coefficient scale, every own max-plus event time is piecewise affine in scale.

Propagate exact affine pieces through dependencies/resource order, retaining
linear crossings and the fixed idle duration. Subtract the model's own barrier
curve. Partition the entire [1-s,1+s] interval at every poll/arrival intersection
and curve breakpoint, including endpoint/tie assignments. This determines which
input iteration consumes the sentinel over the continuum; a finite scaling grid
does not certify absence of a flip. Nominal graph times and the independent
original point rollout must agree. The six source pins and reference restrictions
are necessary: this is not a generic physical pipeline uncertainty solver.

If one consumed iteration is possible, keep point classification. If two are
possible, freeze `phase_ambiguous` with both nominal no-probe poll boundaries
minus the declared arrival; clip negative residence to zero. These are two fixed
points, not a continuous interval that accepts any value between them. More than
two possible branches is unsupported. The proposed 136-trial plan retains all
original hard offsets and budgets, with new qp- IDs/content keys. Moving arrivals
to easy mid-interval positions is not used to hide the boundary behavior.

## Separate predictions from observation qualification

Each held OFF and LIGHT sample is scored against its nearest frozen point using
the existing max(10 ms,20% observed residence) budget. Stable cells also retain
their old point-median condition. There is no cancellation across samples.

Report these outcomes separately:

- **Expected phase difference:** paired samples fit different frozen branches.
  Their raw large delta and cross-branch spread are diagnostic, not automatic
  failures of a model that predicted the ambiguity.
- **Same-phase observer shift:** model-assigned paired samples choose the same
  branch but their delta exceeds the existing queue budget. This fails measurement
  qualification independently of the prediction-residual flag.
- **Unexplained prediction-budget failure:** a sample misses every declared point.
  This is a prediction failure; it does not establish a physical root cause.
  Cross-branch pairs containing such failures are not excused as expected switches.

Require within-model-assigned-branch OFF repeat stability. Report one-sample
groups as underpowered and zero-count branches as unobserved; neither validates
an unseen branch. Assignments and counts refer to nearest model points, not
directly measured OFF phase or estimated branch probabilities. Exact assignment
ties remain unresolved and fail measurement qualification.

Endpoint/TTFT, separate anchor and structural scores remain unchanged. Overall
proposal candidacy requires endpoint/structure, queue prediction and measurement
qualification; a measurement failure is not mislabeled as proof of a bad service
law. Original strict raw-queue diagnostics and point verdict remain in the report.
Production-model/HTTP/cap/MPC flags stay false even for a passing CPU fixture.

## Current development result and next evidence

See [the development result and review response](../results/prefill-phase-margin-development/report.md).
The TRAIN-derived envelope declares 400 ms/D16 and 800 ms/D0 phase-ambiguous.
It explains the late 400 ms samples as another model branch, but the two early
samples still miss by 13.70 and 16.31 ms. The 50 ms/D16 zero-residence outlier
is not a predicted flip under this common-scale family and remains a failure.
Same-model-phase observer shifts also remain. No width or branch is added from
these held observations.

The existing native `clock` and `live-http` workflows retain their strict gates;
this CPU proposal cannot unblock them. Before a fresh confirmation, design
training probes or receipt-state observation that can identify same-phase shifts
and phase-origin errors, which zero-offset endpoint residuals do not establish.
The current OFF captures lack internal poll boundaries. Retained held data are
development evidence for that next design, not its independent validation.

CPU reproduction with the unchanged supplied archive:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/develop_prefill_phase_margin.py \
  --capture /path/to/1793086097723591624 \
  --archive /path/to/1793086097723591624.tar.gz \
  --output /new/output/directory
```

This verifies archive bytes and reconstructs the original source audit, then saves
the phase freeze, scored development report, policy and proposed inputs. The CLI
is read-only with respect to raw data. Output is explicitly retrospective; the
policy was not frozen before the original GPU run. No native entry point or new
GPU result is introduced by this proposal.
