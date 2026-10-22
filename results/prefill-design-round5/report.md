# Calibrated design screen — revision 7

CPU-only synthetic checks. No new GPU data, physical coefficient fit, C use or MPC result.

Every scenario generates calibration-only step labels, refits nonnegative models with
case-balanced weights and training-only scaling, selects by training-only LOCO,
then predicts held sentinel timings with its own causal FIFO progress.
Structural components use the existing calibrated-comparator gates unchanged.

| Design | k | Assumed jitter | Selected/base | Eligible versus M0 | Structural component |
|---|---:|---:|---|---:|---|
| revision6_uniform | 0 | 0ms | M0/M0 | N/A | not_applicable |
| revision6_uniform | 1e-09 | 0ms | M2/M0 | 0 | inconclusive |
| revision6_uniform | 1.5e-09 | 0ms | M2/M0 | 0 | inconclusive |
| revision6_uniform | 1.5e-09 | 10ms | M2/M0 | 0 | inconclusive |
| revision6_uniform | 1.5e-09 | 15ms | M2/M0 | 0 | inconclusive |
| revision6_uniform | 3.15e-09 | 0ms | M2/M0 | 8 | components_passed |
| revision7_mixed | 0 | 0ms | M0/M0 | N/A | not_applicable |
| revision7_mixed | 1e-09 | 0ms | M2/M0 | 0 | inconclusive |
| revision7_mixed | 1.5e-09 | 0ms | M2/M0 | 4 | components_passed |
| revision7_mixed | 1.5e-09 | 10ms | M2/M0 | 4 | components_passed |
| revision7_mixed | 1.5e-09 | 15ms | M2/M0 | 0 | inconclusive |
| revision7_mixed | 3.15e-09 | 0ms | M2/M0 | 8 | components_passed |

## Change and limits

Replace the four-request foreground at 50/400ms with [1024,1024,7168,7168].
Keep four requests and 16384 input tokens. Keep uniform four-at800 as a policy tie
control, and keep one7168-at200. IDs remain stable within this versioned draft;
the complete case content and hashes, not IDs alone, identify the protocol.
Only 50-vs-400 now has matched composition for offset-order contrasts.

At k=1.5e-9, recalibrating M0 reduces the old eight gaps below25ms even without
measurement noise. The mixed replacement produces four gaps about36.7–36.8ms;
it resolves the existing component at hypothetical0/10ms jitter, but NOT15ms.
k=1e-9 remains unresolved. This is an honest sensitivity limit, not a reason to
lower the25ms floor, change eligibility after collection or treat correlated cells as independent.

M0Z is fitted on the same calibration folds and reported alongside M0/M1/M2.
When M2 beats M0 after M1 was rejected, report joint feature benefit; do not
attribute it uniquely to Z. These ablations are conditional predictive evidence,
not proof of a kernel mechanism. Selection remains M0/M1/M2.

Pure linear ±10% rescaling controls select M0, so recalibration does not become
a structural success. Context-only and combined context/Z controls are archived
in JSON, including required calibrated-M1 comparisons where M1 is retained.

## Azure trace and the next experiment

Keep Azure development transfer. prepare_trace.py preserves retained arrival
timestamps and input lengths, synthesizes prompt content and uses output cap128
with natural EOS. Source GeneratedTokens are audit-only. This is trace-driven
replay. Synthetic prompt text is appropriate for this trace: Azure releases
timestamps/input/output sizes, not prompt text. Azure describes forcing the
recorded output size; our fixed128 cap instead changes output work. Source:
https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md

Realistic inputs test usefulness; they do not automatically separate correlated
prefill, decode-context and queue effects. Assess feature support on measured
training rows before claiming the trace can or cannot identify a mechanism.
Controlled cells separate candidate explanations; unchanged Azure development
slices then test transfer without fitting on their outcomes. Independent C and
fixed/reactive/MPC evaluation remain reserved for later.

Before executable freeze, scheduler parity must compare scheduled request IDs,
chunk sizes, computed positions and completion/admission events using the exact
installed vLLM/config. Logging these fields alone is not a parity test.
Native scheduler adapter, collector, mode-aware causal rollout and token manifests
remain blockers; this synthetic tool is not their implementation.

Later cap-policy validation must compare matched offered traces under fixed caps
and cap transitions, score action ranking/regret and transient draining, and
account for unknown future output lengths. D16 is a decoder population, not cap16.

## Limitations

- No new GPU observations or measured coefficient estimates.
- Five repeated errors are identical deterministic entries, not replicates.
- Four eligible cells share composition and are not four independent effects.
- Serial FIFO toy parity is not actual vLLM scheduler/async/mode parity.
- Hypothetical jitter thresholds do not model noise or estimate power.
- M0Z is a reporting ablation, not an additional selectable family.
- Component success does not satisfy bias/order, observer or complete experiment gates.
