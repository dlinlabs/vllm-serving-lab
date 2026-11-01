# Async calibration CPU implementation and self-review

Implemented a strict 21-component service-label contract, trial-balanced/scaled
NNLS estimation, rank/condition rejection, immutable-by-digest model freeze,
separate held evaluation, marginal support/mode reporting, reusable PipelineCosts
export and short CLI commands. See ../../experiments/prefill-async-calibration-v1.md.

293 related CPU tests passed in13.76s, zero failures/errors/skips, including23
new calibration cases. The earlier targeted calibration/predictor run passed69.
The local runtime initially lacked pytest; declared CPU dependencies were
installed. An initial demo correctly rejected NumPy scalar feature values;
the synthetic generator now explicitly serializes builtin float values.
Compilation and registered source checks pass. No native resource/full-suite
acceptance or GPU execution occurred.

The deterministic synthetic demo recovers its known coefficients with maximum
absolute error2.4729317146630186e-18; maximum held component MAE is
2.2768245622195593e-18 seconds. These numbers describe algebraic noiseless
recovery, not hardware timing accuracy. A test runs fitted versus ground-truth
costs through the causal predictor with a long prompt and delayed sentinel and
checks TTFT equality within1e-10 seconds. The generated input is reproducible
with `python scripts/calibrate_prefill_async.py demo --output NEW_DIRECTORY`.
Committed artifacts contain the plan, frozen costs, scores, JUnit and source hashes.

## Self-review findings and disposition

1. **Native identifiability remains BLOCKED.** Existing CPU/CUDA aggregate spans
   do not separately identify first/last enqueue, GPU prefix/tail service, copy
   wait versus copy service, or output blocking versus materialization. No native
   extraction/qualification implementation exists yet. ALL native-origin inputs
   are rejected, and every model/native-collection acceptance flag stays false.
   This is an explicit remaining engineering dependency, not a completed GPU
   calibration or a claim that the old diagnostic can train the new model.
2. **Held leakage protected.** Split membership comes from the frozen plan;
   unrelated split features/labels are never inspected. Poisoned held values do
   not change the freeze; poisoned training labels do not change held scoring.
3. **Identifiability enforced.** Zero, rank-deficient and nearly-collinear bases
   fail. RMS scales use trial-balanced training weights, so replicating steps in
   one trial does not increase that trial's influence or change the scaling rule.
4. **Timing scope protected.** Nested/wait-inclusive labels, negative/nonfinite
   data, missing/duplicate/unknown trial IDs, changed runtime/plan and corrupted
   freezes fail. Feature support and unseen modes are reported without refitting.
5. **Claims bounded.** Marginal support is not joint support. Synthetic stage-fit
   accuracy is not measured TTFT accuracy. No data-driven acceptance threshold,
   cap adaptation, MPC or statistical qualification is silently introduced.

Remaining work: an identifiable native parameterization and qualified service
label extractor, followed by a concrete preregistered native manifest/overhead/
repeat/dependence/held-TTFT protocol. Do not rent a GPU or rerun the existing stage
collector expecting it to satisfy this new label contract. Existing serial guard,
GPU hooks and workload/statistical rules remain unchanged.
