"""Revision9 synthetic noise screen. Uses no GPU captures or held measurements."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_prefill_design import cases
from prefill_design import COEF
from prefill_planning import collect_training
from prefill_noise import (fit_scenario, cases_at_revision, evaluate_draws, held_noise,
                           freeze_plan, NOISE_PROFILES)
from prefill_grouped_validation import ALL_CELLS, T95_DF4, MIN_MEAN_GAIN_S

SEED = 20261022


def compact_fit(fitted):
    out = {k: v for k, v in fitted.items() if k != 'selection'}
    s = fitted['selection']
    out['selection'] = {k: v for k, v in s.items() if k != 'loco_errors'}
    out['selection']['mean_loco_errors'] = {
        f: {target: float(np.mean([r[target] for r in rows.values()])) for target in ('decode', 'prefill', 'cumulative')}
        for f, rows in s['loco_errors'].items()}
    return out


def legacy_rates(fitted, sigma, draws):
    """Reviewer reproduction: cell tally only, not the new grouped protocol."""
    selected = fitted['selection']['selected']
    pred = fitted['predictions']
    cand = np.asarray([pred[selected][k] for k in ALL_CELLS])
    base = np.asarray([pred['M0'][k] for k in ALL_CELLS])
    true = np.asarray([fitted['truth'][k] for k in ALL_CELLS])
    observed = true[None, :, None]+held_noise(SEED, draws, sigma, 'iid')
    a, b = np.abs(cand[None, :, None]-observed), np.abs(base[None, :, None]-observed)
    gain = (b-a).mean(axis=-1)
    eligible = np.abs(cand-base) > max(3*sigma, .025)
    need = int(np.ceil(.75*eligible.sum()))
    regression = (a.mean(axis=-1)-b.mean(axis=-1) > np.maximum(.1*b.mean(axis=-1), .025)).any(axis=-1)
    margins = {'revision7_margin': max(3*sigma, .025),
               'reviewer_cell_t': T95_DF4*(b-a).std(axis=-1, ddof=1)/np.sqrt(5)}
    result = {}
    for name, margin in margins.items():
        passed = ((gain > margin)[..., eligible].sum(axis=-1) >= need) & ~regression & (eligible.sum() >= 4)
        result[name] = int(passed.sum())
    return dict(sigma=sigma, draws=draws, eligible_cells=int(eligible.sum()), **result)


def run(draws=20000, training_seeds=40, held_draws_per_fit=200):
    current = cases()
    held_rows, null_rows, reproduction = [], [], []
    exact_fits = {}
    for revision in (7, 8):
        fitted = fit_scenario(cases_at_revision(current, revision), 1.5e-9)
        exact_fits[str(revision)] = compact_fit(fitted)
        if revision == 7:
            reproduction = [legacy_rates(fitted, sigma, draws) for sigma in (.008, .010, .012)]
        for sigma in (.008, .010, .012, .015):
            for profile in NOISE_PROFILES:
                held_rows.append(dict(revision=revision, sigma=sigma, profile=profile,
                                      **evaluate_draws(fitted, sigma, draws=draws, seed=SEED, profile=profile)))
        if revision == 8:
            for sigma in (.008, .010, .012, .015):
                for null in ('midpoint', 'baseline_truth'):
                    truth = {k: (fitted['predictions']['M2'][k]+fitted['predictions']['M0'][k])/2
                             if null == 'midpoint' else fitted['predictions']['M0'][k] for k in ALL_CELLS}
                    for profile in NOISE_PROFILES:
                        null_rows.append(dict(null=null, sigma=sigma, profile=profile,
                                              **evaluate_draws(fitted, sigma, draws=draws, seed=SEED+1,
                                                               profile=profile, truth_override=truth)))
    print('Held noise, correlated blocks and midpoint/baseline controls complete.', flush=True)

    label_rows = []
    for quadratic in (0., 1.5e-9):
        training = collect_training(current, [*COEF, 0., 2*quadratic])
        for label_profile in ('iid', 'shared_trial'):
            for cv in (.01, .02, .05):
                records = []
                for index in range(training_seeds):
                    seed = SEED+index
                    fitted = fit_scenario(current, quadratic, cv, seed, label_profile, training=training)
                    evaluations = {}
                    for label, sigma, profile, actual_sigma in (
                            ('label_only', .010, 'iid', 0.),
                            ('combined_iid_10ms', .010, 'iid', .010),
                            ('combined_shared_ar_10ms', .010, 'shared_block_ar1', .010),
                            ('combined_iid_15ms', .015, 'iid', .015)):
                        evaluations[label] = evaluate_draws(fitted, sigma, draws=held_draws_per_fit,
                                                            seed=SEED+10000+index, profile=profile, actual_sigma=actual_sigma)
                    records.append(dict(**compact_fit(fitted), evaluations=evaluations))
                selections = dict(Counter(r['selection']['selected'] for r in records))
                summaries = {}
                for label in records[0]['evaluations']:
                    rates = [r['evaluations'][label]['passed']/held_draws_per_fit for r in records]
                    nominal = [r['evaluations'][label]['nominal_passed']/held_draws_per_fit for r in records]
                    summaries[label] = dict(mean_pass_fraction=float(np.mean(rates)), nominal_mean_pass_fraction=float(np.mean(nominal)),
                                            min_seed_fraction=min(rates), max_seed_fraction=max(rates),
                                            status_counts={status: sum(r['evaluations'][label][status] for r in records)
                                                           for status in ('passed', 'failed', 'inconclusive', 'not_applicable')})
                label_rows.append(dict(quadratic=quadratic, cv=cv, label_profile=label_profile,
                                       training_seeds=training_seeds, selections=selections,
                                       summaries=summaries, records=records))
                print(f'k={quadratic:g} CV={cv:g} {label_profile}: {selections}', flush=True)
    sources = ['prefill_grouped_validation.py', 'prefill_noise.py', 'prefill_planning.py',
               'prefill_design.py', 'prefill_identifiability.py', 'prefill_validation.py',
               'scripts/build_prefill_design.py', 'scripts/check_prefill_noise.py',
               'experiments/prefill-service-v1.cases.json']
    return dict(schema='prefill-noise-screen-v9', minimum_mean_gain_s=MIN_MEAN_GAIN_S, seed=SEED, draws=draws,
                training_seeds=training_seeds, held_draws_per_fit=held_draws_per_fit,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                exact_fits=exact_fits, reviewer_reproduction=reproduction,
                held_noise=held_rows, null_controls=null_rows, label_noise=label_rows,
                complete_experiment_passed=False, measured_power=False,
                noise_contract=dict(held='additive Gaussian, same realization for candidate and comparators',
                                    profiles=NOISE_PROFILES,
                                    training='mean-one lognormal step labels; three trials/cell; marginal CV; iid or shared-trial normal correlation0.8',
                                    endpoints='Timing-label noise does not alter true executed work, features or causal arrival schedule.',
                                    limitations='Truth is inside M2; no out-of-family misspecification, config error, noisy features or non-Gaussian held tails. Rates are scenario-conditional; not actual GPU power.'),
                dependence_note='Many held draws share one training fit. Report ranges across training seeds; do not claim they are all independent full experiments.')


def report(data):
    lines = ['# Revision9: paired structural groups and explicit noise', '',
             'CPU-only synthetic results. No GPU collection, C use or MPC validation.', '',
             '## Protocol change', '',
             'Mixed bursts at50/400ms now use [512,512,7680,7680]; single-at200 uses7680.',
             'Uniform4x4096-at800 remains a control. Counts remain22 cells/212 trials.',
             'The frozen linear budget is1518.29s versus1517.78s previously, below1800s.',
             '7680+128 leaves384 tokens below8192; exact input token manifests are still required.',
             'The single7680 test extends beyond the7168 calibration maximum: bounded length',
             'extrapolation, not pure interpolation. Mode-aware actual feature support remains a gate.', '',
             'Eligibility is unchanged per cell: prediction gap >max(3*jitter,25ms).',
             'Count TWO fixed groups, mixed_burst and single_long. Every constituent cell must',
             'be eligible; the uniform control is never another structural contrast.',
             'Within each group/block average paired absolute-error improvement across member',
             'cells, then compute the mean and sample SD over FIVE blocks. Both groups need',
             'mean gain >max(t(.95,4)*SD/sqrt(5),5ms), with1e-12s numerical tolerance.',
             'The5ms engineering floor is frozen before this rerun, not fitted to its outcomes.',
             'Equality and deterministic1us gains are non-wins. This is NOT lower-bound>5ms.',
             'All-eight-cell regression guards remain. Workload revision8 is unchanged;',
             'table revision numbers identify workloads, all scored with the revision9 gate.',
             'Calibrated M0 and retained M1 must each satisfy this rule. M0 is structural N/A.',
             'The .75 cell-win tally and reuse of3*jitter as a required MAE gain are superseded.', '',
             'This is a one-sided paired-t engineering screen. Independent approximately normal',
             'block differences are assumptions; cells inside a group are not independent n.',
             'Cross-block correlation is explicitly stressed below, not fixed by grouping.',
             'The public gate defaults block_dependence_status to unresolved and cannot pass.',
             'Qualified status needs a preregistered calibration-only dependence assessment;',
             'unresolved/violated status blocks acceptance while preserving nominal t diagnostics.',
             'The synthetic AR1 profile is known violated. Iid/shared-within-block profiles',
             'are independent across blocks by construction, not evidence about actual GPU noise.',
             'No full-experiment5% false-positive claim is made. Source for paired differences:',
             'https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm', '',
             '## Held measurement noise (no training noise)', '',
             '| Revision | Sigma | Noise | Eligible groups vs M0 | Nominal t pass | Qualified pass |',
             '|---|---:|---|---|---:|---:|']
    for r in data['held_noise']:
        groups = ', '.join(r['eligible_groups'].get('calibrated_M0', [])) or 'none'
        lines.append(f"| {r['revision']} | {r['sigma']*1000:g}ms | {r['profile']} | {groups} | {r['nominal_passed']/r['draws']:.2%} | {r['passed']/r['draws']:.2%} |")
    lines += ['', 'At10ms the single-long gap is only about30.2ms against a30ms eligibility',
              'threshold. This is fragile. At12–15ms it is ineligible and the two-group gate',
              'remains inconclusive even with a perfect M2. Do not lower thresholds or expand',
              'the matrix after viewing held measurements. Historical revision7 had one group.', '',
              '## Equal-performance and wrong-model controls', '',
              '| Null | Sigma | Noise | Nominal t pass | Qualified pass |',
              '|---|---:|---|---:|---:|']
    for r in data['null_controls']:
        lines.append(f"| {r['null']} | {r['sigma']*1000:g}ms | {r['profile']} | {r['nominal_passed']/r['draws']:.2%} | {r['passed']/r['draws']:.2%} |")
    lines += ['', 'Cross-block AR1 midpoint errors have a high nominal false-pass rate (about14%).',
              'They are blocked by the dependence qualification; do not quote an iid result',
              'as assurance about a correlated GPU run. Five blocks cannot establish independence.', '',
              '## Calibration-label noise and joint sensitivity', '',
              'Three calibration trials per cell are generated. LOCO holds out the entire',
              'cell including all its repeats; scales and fits never see that cell.',
              'Multiplicative noise is mean-one lognormal (positive labels), unlike an exact',
              'reproduction of the review Gaussian-label experiment. Only labels change;',
              'true work positions and first-token membership stay fixed. This isolates label',
              'uncertainty and does not simulate physical slowdowns changing scheduling.', '',
              '| k | CV | Labels | M2 selections | Label-only pass | Joint iid10ms | Joint shared/AR10ms nominal | Joint iid15ms |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in data['label_noise']:
        s = r['summaries']
        values = [s[k]['nominal_mean_pass_fraction' if k == 'combined_shared_ar_10ms' else 'mean_pass_fraction'] for k in ('label_only','combined_iid_10ms','combined_shared_ar_10ms','combined_iid_15ms')]
        lines.append(f"| {r['quadratic']:g} | {r['cv']:.0%} | {r['label_profile']} | {r['selections'].get('M2',0)}/{r['training_seeds']} | " + ' | '.join(f'{v:.2%}' for v in values) + ' |')
    lines += ['', 'Shared/AR rates in the table are nominal diagnostics; qualified pass is zero.', '',
              'The JSON archives every training seed, selected family, coefficients, LOCO',
              'mean errors, predictions, eligibility and held-result counts. Shared training',
              'fits mean held Monte Carlo draws are not independent full experimental runs.', '',
              '## Calibration-only readiness', '',
              'After the first three calibration blocks and before model/held-score freeze,',
              'compute repeat CV within matched executed-work, request-position, mode and',
              'padding strata. Freeze expected strata before opening labels; do not pool',
              'different chunk sizes or discard unsupported/missing strata. Use one duration',
              'per independent trial, not individual token gaps as repeat observations.',
              '`calibration_noise_readiness` rejects incomplete/nonfinite repeats and stops',
              'structural claims if any matched-stratum CV exceeds the preregistered5% limit.',
              'Above2% is flagged. This is a conservative engineering limit with only three',
              'repeats, not a population-CV confidence bound. Within range never accepts a model.',
              'Keep held timing sealed; no resampling/retry or threshold tuning from held results.',
              'Measured sigma/CV and actual two-group support must accompany any later claim.',
              'A calibration-only serial-dependence assessment must be specified and reviewed',
              'before executable freeze. Five held blocks cannot qualify independence; lack of',
              'a significant autocorrelation test is not evidence of independence. Until that',
              'assessment exists and supports the assumptions, dependence remains unresolved.', '',
              'Native collector/source mapping, matched-stratum identity, scheduler parity,',
              'mode-aware causal rollout and exact token manifests remain pre-freeze blockers.',
              'Preserve Azure development transfer; controlled tests and real-load transfer',
              'remain complementary. The fixed128-output replay scope is unchanged.', '',
              'Reproduce: `python scripts/check_prefill_noise.py --output-dir /tmp/prefill-noise-fresh`.',
              'All previous reports are historical snapshots at their recorded revisions.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = run()
    (args.output_dir/'noise-screen.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(report(data))
    print('Noise screen saved.', flush=True)
