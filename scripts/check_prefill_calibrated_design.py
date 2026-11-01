"""Reproduce the revision-6 blind spot and revision-7 calibrated design screen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_prefill_design import cases
from prefill_planning import legacy_cases, synthetic_check


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checks():
    cells = cases()
    scenarios = []
    for design, cs in [('revision6_uniform', legacy_cases(cells)), ('revision7_mixed', cells)]:
        for k, jitter in [(0., 0.), (1e-9, 0.), (1.5e-9, 0.), (1.5e-9, .010),
                          (1.5e-9, .015), (3.15e-9, 0.)]:
            scenarios.append(dict(design=design, **synthetic_check(cs, quadratic=k, jitter_s=jitter)))
    controls = [synthetic_check(cells, quadratic=0., scale=s) for s in (.9, 1.1)]
    controls += [synthetic_check(cells, quadratic=k, context_cost=1e-7) for k in (0., 1.5e-9)]
    sources = ['prefill_planning.py', 'prefill_design.py', 'prefill_identifiability.py',
               'prefill_validation.py', 'scripts/build_prefill_design.py',
               'scripts/check_prefill_calibrated_design.py', 'experiments/prefill-service-v1.cases.json']
    return dict(schema='calibrated-prefill-planning-v1', role='deterministic synthetic feasibility, not statistical power',
                source_sha256={p: digest(ROOT / p) for p in sources}, scenarios=scenarios, controls=controls,
                limitations=['No new GPU observations or measured coefficient estimates.',
                             'Five repeated errors are identical deterministic entries, not replicates.',
                             'Four eligible cells share composition and are not four independent effects.',
                             'Serial FIFO toy parity is not actual vLLM scheduler/async/mode parity.',
                             'Hypothetical jitter thresholds do not model noise or estimate power.',
                             'M0Z is a reporting ablation, not an additional selectable family.',
                             'Component success does not satisfy bias/order, observer or complete experiment gates.'])


def report(data):
    lines = ['# Calibrated design screen — revision 7', '',
             'CPU-only synthetic checks. No new GPU data, physical coefficient fit, C use or MPC result.', '',
             'Every scenario generates calibration-only step labels, refits nonnegative models with',
             'case-balanced weights and training-only scaling, selects by training-only LOCO,',
             'then predicts held sentinel timings with its own causal FIFO progress.',
             'Structural components use the existing calibrated-comparator gates unchanged.', '',
             '| Design | k | Assumed jitter | Selected/base | Eligible versus M0 | Structural component |',
             '|---|---:|---:|---|---:|---|']
    for r in data['scenarios']:
        comparisons = r['comparison_plan']['required_structural_comparators']
        eligible = len(comparisons['calibrated_M0']['eligible_cells']) if comparisons else 'N/A'
        lines.append(f"| {r['design']} | {r['quadratic']:g} | {r['jitter_s']*1000:g}ms | {r['selection']['selected']}/{r['selection']['retained_base']} | {eligible} | {r['gate']['structural_status']} |")
    lines += ['', '## Change and limits', '',
              'Replace the four-request foreground at 50/400ms with [1024,1024,7168,7168].',
              'Keep four requests and 16384 input tokens. Keep uniform four-at800 as a policy tie',
              'control, and keep one7168-at200. IDs remain stable within this versioned draft;',
              'the complete case content and hashes, not IDs alone, identify the protocol.',
              'Only 50-vs-400 now has matched composition for offset-order contrasts.', '',
              'At k=1.5e-9, recalibrating M0 reduces the old eight gaps below25ms even without',
              'measurement noise. The mixed replacement produces four gaps about36.7–36.8ms;',
              'it resolves the existing component at hypothetical0/10ms jitter, but NOT15ms.',
              'k=1e-9 remains unresolved. This is an honest sensitivity limit, not a reason to',
              'lower the25ms floor, change eligibility after collection or treat correlated cells as independent.', '',
              'M0Z is fitted on the same calibration folds and reported alongside M0/M1/M2.',
              'When M2 beats M0 after M1 was rejected, report joint feature benefit; do not',
              'attribute it uniquely to Z. These ablations are conditional predictive evidence,',
              'not proof of a kernel mechanism. Selection remains M0/M1/M2.', '',
              'Pure linear ±10% rescaling controls select M0, so recalibration does not become',
              'a structural success. Context-only and combined context/Z controls are archived',
              'in JSON, including required calibrated-M1 comparisons where M1 is retained.', '',
              '## Azure trace and the next experiment', '',
              'Keep Azure development transfer. prepare_trace.py preserves retained arrival',
              'timestamps and input lengths, synthesizes prompt content and uses output cap128',
              'with natural EOS. Source GeneratedTokens are audit-only. This is trace-driven',
              'replay. Synthetic prompt text is appropriate for this trace: Azure releases',
              'timestamps/input/output sizes, not prompt text. Azure describes forcing the',
              'recorded output size; our fixed128 cap instead changes output work. Source:',
              'https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md', '',
              'Realistic inputs test usefulness; they do not automatically separate correlated',
              'prefill, decode-context and queue effects. Assess feature support on measured',
              'training rows before claiming the trace can or cannot identify a mechanism.',
              'Controlled cells separate candidate explanations; unchanged Azure development',
              'slices then test transfer without fitting on their outcomes. Independent C and',
              'fixed/reactive/MPC evaluation remain reserved for later.', '',
              'Before executable freeze, scheduler parity must compare scheduled request IDs,',
              'chunk sizes, computed positions and completion/admission events using the exact',
              'installed vLLM/config. Logging these fields alone is not a parity test.',
              'Native scheduler adapter, collector, mode-aware causal rollout and token manifests',
              'remain blockers; this synthetic tool is not their implementation.', '',
              'Later cap-policy validation must compare matched offered traces under fixed caps',
              'and cap transitions, score action ranking/regret and transient draining, and',
              'account for unknown future output lengths. D16 is a decoder population, not cap16.', '',
              '## Limitations', ''] + [f'- {s}' for s in data['limitations']]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = checks()
    (args.output_dir / 'calibrated-planning.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'report.md').write_text(report(data))
    print(json.dumps(dict(scenarios=len(data['scenarios']), controls=len(data['controls']))))
