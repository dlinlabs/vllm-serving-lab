"""Reproducible synthetic label/held-noise sensitivities, not measured power."""
import copy
import math
import numpy as np

from prefill_design import COEF
from prefill_identifiability import features
from prefill_planning import collect_training, fit, rollout, FAMILIES
from prefill_grouped_validation import (ALL_CELLS, grouped_comparison_plan,
                                       paired_components)

NOISE_PROFILES = {
    'iid': (0., 0.),
    'shared_block': (.8, 0.),
    'shared_block_ar1': (.8, .6),
}


def held_noise(seed, draws, sigma, profile):
    if profile not in NOISE_PROFILES or not math.isfinite(sigma) or sigma < 0 or type(draws) is not int or draws < 1:
        raise ValueError('Invalid held-noise configuration')
    rng = np.random.default_rng(seed)
    rho, phi = NOISE_PROFILES[profile]
    z = math.sqrt(rho)*rng.normal(size=(draws, 1, 5)) + math.sqrt(1-rho)*rng.normal(size=(draws, 8, 5))
    for block in range(1, 5):
        z[..., block] = phi*z[..., block-1] + math.sqrt(1-phi**2)*z[..., block]
    return sigma*z


def training_labels(training, cv, seed, profile='iid'):
    """Three independent calibration trials per cell; labels do not move work.

    Mean-one lognormal multiplier has declared marginal CV. Shared-trial profile
    correlates underlying normal errors at .8 within a trial. It is not additive
    CPU launch overhead, physical clock drift, or noisy feature measurement.
    """
    if not math.isfinite(cv) or cv < 0 or profile not in ('iid', 'shared_trial'):
        raise ValueError('Invalid label-noise configuration')
    rng = np.random.default_rng(seed)
    variance = math.log1p(cv**2)
    out = {}
    for name in sorted(training):
        _, steps = training[name]
        y = np.asarray([s['end']-s['start'] for s in steps])
        z = rng.normal(size=(3, len(steps)))
        if profile == 'shared_trial':
            z = math.sqrt(.8)*rng.normal(size=(3, 1)) + math.sqrt(.2)*z
        out[name] = y[None, :] * np.exp(math.sqrt(variance)*z-variance/2)
    return out


def fit_labels(training, labels, family):
    # Repeated X has identical work rows. Fitting mean(y) is exactly equivalent
    # to fitting all3 repeats with total squared weight1 per case (not per step).
    if set(training) != set(labels):
        raise ValueError('Label/training cells differ')
    observed = {}
    for name, (result, steps) in training.items():
        y = np.asarray(labels[name], float)
        if y.shape != (3, len(steps)) or not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError('Expected three positive label vectors per case')
        observed[name] = result, [dict(s, end=s['start']+value) for s, value in zip(steps, y.mean(axis=0))]
    return fit(observed, family)


def label_errors(observed, labels, coefficients):
    result, steps = observed
    x = np.asarray([features(s) for s in steps])
    predicted = x @ coefficients
    prefill = np.asarray([s['prefill'] > 0 for s in steps])
    residual = predicted[None, :]-labels
    cumulative = []
    for r in result['foreground'].values():
        # Membership uses unperturbed work IDs/endpoints, not noisy label endpoints.
        mask = np.asarray([s['end'] <= result['release_s']+r['first_s']+1e-12 for s in steps])
        cumulative.extend(np.abs(residual[:, mask].sum(axis=1)))
    return dict(decode=float(np.abs(residual[:, ~prefill]).mean()),
                prefill=float(np.abs(residual[:, prefill]).mean()), cumulative=float(np.mean(cumulative)))


def select_labels(training, labels):
    errors = {f: {} for f in FAMILIES}
    for held in training:
        train = {k: v for k, v in training.items() if k != held}
        train_labels = {k: labels[k] for k in train}
        for family in FAMILIES:
            errors[family][held] = label_errors(training[held], labels[held], fit_labels(train, train_labels, family))

    def retain(candidate, base, primary, guards):
        for target in primary:
            a = np.mean([v[target] for v in errors[candidate].values()])
            b = np.mean([v[target] for v in errors[base].values()])
            if b <= 1e-12 or b-a < .1*b:
                return False
        return all(errors[candidate][case][target]-v <= max(.1*v, .001)
                   for case, row in errors[base].items() for target, v in row.items() if target in guards)

    base = 'M1' if retain('M1', 'M0', ('decode',), ('prefill', 'cumulative')) else 'M0'
    selected = 'M2' if retain('M2', base, ('prefill', 'cumulative'), ('decode', 'prefill', 'cumulative')) else base
    return dict(selected=selected, retained_base=base, loco_errors=errors)


def predictions(cases, coefficients):
    return {c['id']: rollout(c, coefficients)[0]['foreground']['sentinel']['ttft_s']
            for c in cases if c['sentinel']}


def fit_scenario(cases, quadratic, cv=0., seed=0, label_profile='iid', training=None):
    truth = np.asarray([*COEF, 0., 2*quadratic])
    if training is None:
        training = collect_training(cases, truth)
    labels = training_labels(training, cv, seed, label_profile)
    selection = select_labels(training, labels)
    coefficients = {f: fit_labels(training, labels, f) for f in FAMILIES}
    needed = set(('M0', selection['selected']))
    if selection['retained_base'] == 'M1':
        needed.add('M1')
    pred = {f: predictions(cases, coefficients[f]) for f in sorted(needed)}
    return dict(quadratic=quadratic, cv=cv, training_seed=seed, label_profile=label_profile,
                selection=selection, coefficients={f: x.tolist() for f, x in coefficients.items()},
                predictions=pred, truth=predictions(cases, truth), raw=predictions(cases, [*COEF, 0., 0.]))


def freeze_plan(fitted, sigma):
    selected, base = fitted['selection']['selected'], fitted['selection']['retained_base']
    p = fitted['predictions']
    return grouped_comparison_plan(selected, base, p[selected], p['M0'], fitted['raw'],
                                   {k: sigma for k in ALL_CELLS}, calibrated_m1=p.get('M1'))


def evaluate_draws(fitted, sigma, *, draws, seed, profile, actual_sigma=None, truth_override=None):
    plan = freeze_plan(fitted, sigma)
    required = plan['required_structural_comparators']
    if not required:
        return dict(draws=draws, passed=0, nominal_passed=0, failed=0, inconclusive=0, not_applicable=draws, eligible_groups={}, block_dependence_status='not_applicable')
    true = fitted['truth'] if truth_override is None else truth_override
    noise = held_noise(seed, draws, sigma if actual_sigma is None else actual_sigma, profile)
    observed = np.asarray([true[k] for k in ALL_CELLS])[None, :, None] + noise
    selected = fitted['selection']['selected']
    candidate = np.asarray([fitted['predictions'][selected][k] for k in ALL_CELLS])[None, :, None]-observed
    passed, failed = np.ones(draws, bool), np.zeros(draws, bool)
    groups = {}
    for name, e in required.items():
        baseline = fitted['predictions'][name.removeprefix('calibrated_')]
        b = np.asarray([baseline[k] for k in ALL_CELLS])[None, :, None]-observed
        r = paired_components(e, candidate, b)
        passed &= r['passed']; failed |= r['failed']
        groups[name] = e['eligible_groups']
    nominal_passed = int(passed.sum())
    # Correlation across blocks violates this paired-t model. Nominal diagnostics
    # stay visible, but cannot qualify a structural component. Synthetic iid and
    # shared-within-block profiles have independent blocks by construction.
    dependence = 'violated' if NOISE_PROFILES[profile][1] else 'qualified'
    if dependence != 'qualified':
        passed[:] = False
    return dict(draws=draws, passed=int(passed.sum()), nominal_passed=nominal_passed,
                block_dependence_status=dependence, failed=int(failed.sum()),
                inconclusive=int((~passed & ~failed).sum()), not_applicable=0, eligible_groups=groups)


def cases_at_revision(cases, revision):
    out = copy.deepcopy(cases)
    if revision not in (6, 7, 8):
        raise ValueError('Unknown design revision')
    for c in out:
        if not c['sentinel']:
            continue
        if len(c['foreground_tokens']) == 1:
            c['foreground_tokens'] = [7680 if revision == 8 else 7168]
        elif c['sentinel']['arrival_ms'] in (50, 400):
            c['foreground_tokens'] = {6: [4096]*4, 7: [1024,1024,7168,7168], 8: [512,512,7680,7680]}[revision]
    return out
