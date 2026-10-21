"""Grouped development ablations. No causal action accuracy or policy value is estimated."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import nnls, lsq_linear
import evaluate_low_cap_frozen as lf
import evaluate_greybox as ev
import greybox_simulator as gb
import state_model as sm

ROOT = Path(__file__).resolve().parent
ORIGINS = np.arange(60., 296., 5.)
SERVICE = ('frozen', 'prefix_nnls', 'prefix_shrink', 'quadratic_025', 'quadratic_050')
CALIBRATION = {'margin': [0], 'running': [0, 1], 'prefill': list(range(7)), 'state': list(range(9)),
               'recent': [0, 1, 9, 10, 11, 12, 13], 'state_recent': list(range(17)),
               'nonlinear': list(range(9)) + list(range(17, 21)), 'tree': list(range(9))}
PENALTIES = (1., 10., 100.)


def write_json(path, value):
    Path(path).write_text(json.dumps(ev.clean(value), indent=2, allow_nan=False) + '\n')


def prefix_coefficients(scrapes, frozen):
    intervals = gb.step_intervals(scrapes, end=60.)
    x = np.c_[intervals['steps'], intervals['gen'], intervals['prompt']]; y = intervals['dt']
    if len(y) < 20 or np.linalg.matrix_rank(x) < 3:
        return dict(nnls=frozen, shrink=frozen, available=False, reason='insufficient_prefix_rank')
    scale = np.maximum(np.sqrt((x * x).mean(axis=0)), 1e-12); z = x / scale
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    constrained = nnls(z, y)[0] / scale
    penalty = 20.
    shrunk = lsq_linear(np.r_[z, np.sqrt(penalty) * np.eye(3)],
        np.r_[y, np.sqrt(penalty) * np.array(frozen) * scale], bounds=(0, np.inf), tol=1e-12).x / scale
    if constrained[0] <= 0 or shrunk[0] <= 0:
        raise ValueError('Nonpositive step overhead')
    return dict(nnls=constrained, shrink=shrunk, available=True, intervals=len(y), ols=ols,
                ols_nonphysical=bool((ols < 0).any()), scaled_condition=float(np.linalg.cond(z)),
                prior_penalty=penalty, latest_available_s=float(scrapes['recv'][scrapes['recv'] <= 60.].max()))


def instrumented_residuals(run, coef, quadratic, overhead):
    index = {rid: i for i, rid in enumerate(run.req['request_id'])}
    rows, point_x, point_y, point_times, point_ids, point_ttft = [], [], [], [], [], []
    for t in np.arange(5., 296., 1.):
        end = t + gb.HORIZON + gb.EXTRA
        view = run.observed_prefix(end)
        out, info = sm.simulate(view, t, view.oracle_arrivals(t), coef, quadratic=quadratic)
        bias = gb.online_bias(np.array(rows, float).reshape(-1, 8), overhead, t)
        if t % 5 == 0:
            kept, features = sm.point_features(out, info, sm.observed_state(run, t), overhead, bias)
            for o, feature in zip(kept, features):
                i = index[o.request_id]
                if run.req['admitted'][i] and math.isfinite(view.req['ttft'][i]):
                    point_x.append(feature); point_y.append(float(run.req['met'][i])); point_times.append(t)
                    point_ids.append(o.request_id)
                    point_ttft.append([o.first - o.sched, float(run.req['ttft'][i]), float(overhead.mean(o.running)) + bias])
        for o in out:
            i = index[o.request_id]
            if o.admitted and math.isfinite(o.first) and view.req['admitted'][i] and math.isfinite(view.req['ttft'][i]):
                rows.append((t, o.sched, o.first - o.sched, view.req['ttft'][i], o.running,
                             view.req['c_first'][i], run.cap(view.req['acc'][i]), end))
    return np.array(rows, float).reshape(-1, 8), dict(x=np.array(point_x, float).reshape(-1, len(sm.POINT_NAMES)),
        y=np.array(point_y, float), t=np.array(point_times), ids=point_ids, ttft=np.array(point_ttft, float).reshape(-1, 3))


def service_run(capture, variant, frozen, overhead, reference):
    prefix = prefix_coefficients(capture['scrapes'], frozen)
    coef = tuple(prefix['nnls'] if variant == 'prefix_nnls' else prefix['shrink'] if variant == 'prefix_shrink' else frozen)
    quadratic = .25 if variant == 'quadratic_025' else .5 if variant == 'quadratic_050' else 0.
    run = gb.Run(capture['name'], capture['meta'], capture['req'], capture['scrapes'], pre=reference['pre_s'])
    res, points = instrumented_residuals(run, coef, quadratic, overhead)
    rows, draws = [], []
    for t in ORIGINS:
        state, obs, bias = sm.observed_state(run, t), lf.actual(run, t), gb.online_bias(res, overhead, t)
        rng = np.random.default_rng(lf.SEED + int(t)); packs, predictions, admissions = [], [], []
        for _ in range(lf.BOOTS):
            out, info = sm.simulate(run, t, run.bootstrap_arrivals(t, rng), coef, quadratic=quadratic)
            _, x = sm.point_features(out, info, state, overhead, bias)
            packs.append(x); predictions.append(gb.expected_met(out, overhead, bias) / gb.HORIZON)
            admissions.append(sum(bool(o.admitted) for o in out) / gb.HORIZON)
        rows.append(dict(run=run.name, t=t, cap=run.cap(t), actual=obs['goodput'], actual_admission=obs['admission'],
            admission=float(np.mean(admissions)), predicted=float(np.mean(predictions)), nowcast=ev.nowcast(run.req, t),
            near_change=ev.near_change(run, t), state=state))
        draws.append(np.concatenate(packs))
    return dict(name=run.name, variant=variant, rows=rows, draws=draws, points=points, prefix=prefix, coefficients=coef)


def rate(calibrator, matrices):
    return np.array([calibrator.predict(x).sum() / (lf.BOOTS * gb.HORIZON) if len(x) else 0. for x in matrices])


def fit_calibration(data, names, kind, penalty):
    if not names:
        raise ValueError('Empty calibration training set')
    x = np.concatenate([data[n]['points']['x'] for n in names]); y = np.concatenate([data[n]['points']['y'] for n in names])
    factory = sm.TreeCalibration if kind == 'tree' else sm.LogisticCalibration
    return factory(CALIBRATION[kind], penalty).fit(x, y)


def schedule_group(name):
    return 'a' if '-a-' in name else 'b' if '-b-' in name else 'fixed16' if 'fixed-16' in name else 'fixed32'


def nested_calibration(data, kind, grouped=False):
    """Outer labels cannot select a penalty; inner folds also hold out entire units."""
    names, predictions, folds = list(data), {}, []
    units = list(dict.fromkeys(schedule_group(n) for n in names)) if grouped else names
    for held in units:
        held_names = [n for n in names if schedule_group(n) == held] if grouped else [held]
        train = [n for n in names if n not in held_names]; scores = {}
        for penalty in PENALTIES:
            errors = []
            inner_units = list(dict.fromkeys(schedule_group(n) for n in train)) if grouped else train
            for inner in inner_units:
                validate = [n for n in train if schedule_group(n) == inner] if grouped else [inner]
                model = fit_calibration(data, [n for n in train if n not in validate], kind, penalty)
                for n in validate:
                    errors.extend(abs(rate(model, data[n]['draws']) - np.array([r['actual'] for r in data[n]['rows']])))
            scores[penalty] = float(np.mean(errors))
        chosen = min(PENALTIES, key=lambda p: (scores[p], -p))
        model = fit_calibration(data, train, kind, chosen)
        for n in held_names:
            predictions[n] = rate(model, data[n]['draws'])
        folds.append(dict(held_runs=held_names, training_runs=train, chosen_penalty=chosen,
                          inner_mae={str(k): v for k, v in scores.items()}, parameters=model.parameters()))
    return predictions, folds


def summarize(rows, predictions):
    merged = [dict(r, predicted=float(p)) for n, ps in predictions.items() for r, p in zip(rows[n], ps)]
    def metric(v):
        e = np.array([r['predicted'] - r['actual'] for r in v])
        return dict(n=len(v), mae=float(np.abs(e).mean()), bias=float(e.mean()))
    return dict(overall=metric(merged), by_cap={str(c): metric([r for r in merged if r['cap'] == c]) for c in (16, 32)},
        near_change=metric([r for r in merged if r['near_change']]),
        per_run={n: metric([r for r in merged if r['run'] == n]) for n in rows},
        by_segment=[dict(start=s, cap=c, **metric([r for r in merged if s <= r['t'] < s + 60 and r['cap'] == c]))
                    for s in (60, 120, 180, 240) for c in (16, 32)], predictions=merged)


def diagnosis(captures):
    output = []
    for c in captures:
        r, start, events, samples = c['req'], c['meta']['start'], {}, []
        for line in (c['path'] / 'gateway.jsonl').open():
            e = json.loads(line)
            if e.get('request_id'):
                events.setdefault(e['request_id'], {})[e['event']] = e
        for line in (c['path'] / 'engine_observations.jsonl').open():
            e = json.loads(line); samples.append((e['sent_at'] - start, e.get('received_at', math.inf) - start, e.get('raw_metrics') or ''))
        for a in (0, 60, 120, 180, 240):
            cohort = (r['sched'] >= a) & (r['sched'] < a + 60); admitted = cohort & r['admitted']
            phases = {'schedule_to_received': [], 'received_to_accepted': [], 'dispatch_to_first': []}
            for i in np.flatnonzero(admitted):
                e = events[r['request_id'][i]]
                if 'received' in e:
                    phases['schedule_to_received'].append(e['received']['monotonic'] - start - r['sched'][i])
                    phases['received_to_accepted'].append(r['acc'][i] - (e['received']['monotonic'] - start))
                if 'backend_sent' in e and 'first_content' in e:
                    phases['dispatch_to_first'].append(e['first_content']['monotonic'] - e['backend_sent']['monotonic'])
            ss = [(t, v, raw) for t, v, raw in samples if a <= t and v < a + 60]; gauges, hist = {}, {}
            for name in ('kv_cache_usage_perc', 'num_requests_waiting', 'num_requests_running', 'num_preemptions_total'):
                vals = [gb.metric(raw, name) for _, _, raw in ss]; vals = [x for x in vals if math.isfinite(x)]
                gauges[name] = dict(mean=float(np.mean(vals)), max=max(vals)) if vals else None
            for name in gb.HISTOGRAMS:
                count = gb.metric(ss[-1][2], name + '_count') - gb.metric(ss[0][2], name + '_count')
                total = gb.metric(ss[-1][2], name + '_sum') - gb.metric(ss[0][2], name + '_sum')
                hist[name] = dict(count=count, mean_s=total / count if count > 0 else None)
            buckets = np.histogram(r['acc'][admitted], bins=np.arange(a, a + 61))[0]
            output.append(dict(run=c['name'], from_s=a, offered=int(cohort.sum()), admitted=int(admitted.sum()),
                completed=int((cohort & r['success']).sum()), met=int((cohort & r['met']).sum()),
                offered_tokens=float(np.sum(r['est_in'][cohort & np.isfinite(r['est_in'])])),
                admitted_tokens=float(np.sum(r['est_in'][admitted])), admitted_prompt_mean=float(np.mean(r['est_in'][admitted])),
                admitted_long_fraction=float(np.mean(r['est_in'][admitted] >= 2048)), admitted_peak_per_second=int(buckets.max()),
                gauges=gauges, histogram_intervals=hist, lifecycle_mean_s={k: float(np.mean(v)) if v else None for k, v in phases.items()}))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--suite-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path); args = p.parse_args()
    if args.output_dir.exists():
        p.error('Choose a fresh output directory')
    guarded = ('state_model.py', 'iterate_state_model.py', 'greybox_simulator.py', 'evaluate_low_cap_frozen.py')
    before = {f: ev.code_sha256(ROOT / f) for f in guarded}
    captures, audit = lf.audit_suite(args.suite_dir); coef, overhead, reference, _ = lf.frozen_parameters()
    args.output_dir.mkdir(parents=True); write_json(args.output_dir / 'diagnosis.json', diagnosis(captures))
    variants = {}
    for variant in SERVICE:
        data = {}
        for c in captures:
            data[c['name']] = service_run(c, variant, coef, overhead, reference)
            print('Prepared', variant, c['name'], flush=True)
        variants[variant] = data
    rows = {n: x['rows'] for n, x in variants['frozen'].items()}; scores, folds = {}, {}
    for variant, data in variants.items():
        scores[variant] = summarize(rows, {n: [r['predicted'] for r in d['rows']] for n, d in data.items()})
    old = json.loads(gzip.decompress((ROOT / 'results/low-cap-frozen-20261021/full-report.json.gz').read_bytes()))
    expected = {(r['run'], r['t']): r['simulator'] for r in old['predictions']}
    if any(abs(r['predicted'] - expected[r['run'], r['t']]) > 1e-10 for r in scores['frozen']['predictions']):
        raise ValueError('Instrumentation changed frozen forecasts')
    for kind in CALIBRATION:
        pred, fit = nested_calibration(variants['frozen'], kind)
        scores['calibration_' + kind], folds[kind] = summarize(rows, pred), fit
        print('Calibrated', kind, scores['calibration_' + kind]['overall'], flush=True)
    for kind in ('running', 'state', 'recent', 'state_recent', 'nonlinear', 'tree'):
        pred, fit = nested_calibration(variants['frozen'], kind, grouped=True)
        scores['schedule_holdout_' + kind], folds['schedule_holdout_' + kind] = summarize(rows, pred), fit
        print('Schedule-held-out', kind, scores['schedule_holdout_' + kind]['overall'], flush=True)
    if before != {f: ev.code_sha256(ROOT / f) for f in guarded}:
        raise ValueError('Sources changed during evaluation')
    report = dict(protocol=dict(role='development_ablation', independent_acceptance=False,
        outer_split='leave one complete run out', inner_split='leave one remaining complete run out',
        penalties=PENALTIES, service_variants=SERVICE, source_model_unchanged=True, calibration_features=sm.POINT_NAMES,
        controller_enabled=False, time_or_run_identity_features=False, same_trace_shared_across_folds=True),
        audit=audit, code_sha256=before, scores=scores, calibration_folds=folds,
        prefix_service_fits={n: d['prefix'] for n, d in variants['frozen'].items()})
    raw = json.dumps(ev.clean(report), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    (args.output_dir / 'full-report.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    summary = {k: v for k, v in report.items() if k not in ('scores', 'calibration_folds')}
    summary['scores'] = {k: {a: b for a, b in v.items() if a != 'predictions'} for k, v in scores.items()}
    summary['full_report_sha256'] = hashlib.sha256(raw).hexdigest(); write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: v['overall'] for k, v in scores.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
