"""Audit new low-cap captures and test the archived PR #28 model without refitting.

This is development diagnosis, not off-policy evaluation or a controller experiment.
The learned service coefficients and overhead curves stay byte-identical to the old
report. A fixed old-reference transport delay replaces per-new-capture calibration.
Online bias is the existing causal rule, reported beside the no-bias ablation.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

import greybox_simulator as gb
import evaluate_greybox as ev
from analyze_cap_sweep import audit_copy
from experiment_protocol import load_protocol
from run_cap_sweep import gate_issues, qualify_capture

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'results/greybox-forecast-20261020/full-report.json.gz'
CALIBRATION = ROOT / 'experiments/low-cap-frozen-v1-calibration.json'
MODEL_COMMIT = 'ca919b0526bda1f32218b5f202b4a4dc7a7f2b2d'
PROTOCOL_SHA = '0538ff46e3206346a899c34f25b01d23fe6ba93043580cd0b0f38360030da5fe'
SUITE_COMMIT = '719dbde128ab91ecee576dae0d4acbde36ac4050'
START, STOP, STRIDE, BOOTS, SEED = 60, 295, 5, 16, 1


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen_parameters():
    reference = read(CALIBRATION)
    if reference['source_model_commit'] != MODEL_COMMIT or sha(FROZEN) != reference['source_report_sha256']:
        raise ValueError('Frozen model provenance differs')
    archive = json.loads(gzip.decompress(FROZEN.read_bytes()))
    for name, digest in archive['code_sha256'].items():
        if ev.code_sha256(ROOT / name) != digest:
            raise ValueError('Frozen source differs: ' + name)
    # Reconstitute saved arrays directly. Calling the fitting constructor would refit.
    overhead = object.__new__(gb.Overhead)
    for attr, key in (('knots', 'knots'), ('values', 'mean'),
                      ('sigma_knots', 'sigma_knots'), ('sigma_values', 'sigma')):
        setattr(overhead, attr, np.asarray(archive['overhead'][key], float))
    coef = tuple(archive['step_coefficients'][k] for k in
                 ('step_s', 'per_decode_token_s', 'per_prompt_token_s'))
    return coef, overhead, reference, archive


def load_capture_compatible(path):
    """Retain transport failures using the existing client ID; never invent a gateway event.

    PR #28's loader requires request_id even for failed-before-headers rows. The
    collector also records client_request_id, which preserves identity in that case.
    Only the private reader input is normalized; original evidence is read-only.
    """
    client = read(path / 'client.json')
    replacements = 0
    for row in client['requests']:
        if row.get('workload') != 'trace' or row.get('request_id'):
            continue
        if row.get('success') or row.get('rejected') or not row.get('client_request_id'):
            raise ValueError('Missing identity is allowed only for an identified transport failure')
        row['request_id'] = row['client_request_id']
        replacements += 1
    if not replacements:
        return (*gb.load_capture(path), 0)
    with tempfile.TemporaryDirectory(prefix='frozen-loader-') as directory:
        reader = Path(directory)
        for name in ('manifest.json', 'replay_plan.json', 'gateway.jsonl'):
            (reader / name).symlink_to((path / name).resolve())
        (reader / 'client.json').write_text(json.dumps(client))
        meta, req = gb.load_capture(reader)
    return meta, req, replacements


def audit_suite(source):
    """Verify original inventory, identity, CPU evidence, and independently repeat all gates."""
    source = Path(source).resolve()
    suite = read(source / 'suite.json')
    document, digest = load_protocol(source / 'protocol.json')
    if (suite.get('valid') is not True or suite.get('status') != 'collected'
            or suite.get('stage') != 'full' or suite.get('commit') != SUITE_COMMIT
            or digest != PROTOCOL_SHA or suite['protocol_sha256'] != digest
            or document['protocol'] != 'low-cap-switch-v1' or document['status'] != 'frozen'):
        raise ValueError('Expected the frozen, qualified low-cap full suite')
    order = document['qualification']['order']
    if suite['planned'] != order or [x['name'] for x in suite['runs']] != order:
        raise ValueError('Missing, reordered or duplicate suite runs')
    acceptance = suite['cpu_acceptance']
    if acceptance['acceptance'].get('valid') is not True or acceptance['acceptance']['commit'] != suite['commit']:
        raise ValueError('Native CPU acceptance must cover the collection commit')
    evidence_root = source.parents[1]
    cpu = [p for p in (evidence_root / 'cpu-ci-artifacts').glob('*/acceptance.json')
           if sha(p) == acceptance['acceptance_sha256']]
    if len(cpu) != 1 or sha(cpu[0].parent / 'junit.xml') != acceptance['junit_sha256']:
        raise ValueError('Original CPU acceptance and JUnit evidence are required')
    definitions = {x['name']: x for x in document['runs']}
    captures, expected_code, checked = [], None, 0
    for record in suite['runs']:
        name = record['name']
        path = source / name
        if (record['status'] != 'passed' or record['exitcode'] != 0
                or record['backend'] != suite['backend']):
            raise ValueError(name + ': collection/backend identity differs')
        actual = {p.name: sha(p) for p in path.iterdir() if p.is_file()}
        if actual != record['artifact_sha256']:
            raise ValueError(name + ': original inventory/hash differs')
        checked += len(actual)
        manifest, saved = qualify_capture(path, document, digest, name, suite['commit'], expected_code)
        expected_code = manifest['code_sha256']
        if manifest['host'] != suite['backend']['host'] or manifest['packages'] != suite['packages']:
            raise ValueError(name + ': host/packages differ')
        audited = audit_copy(path)
        if gate_issues(audited) or audited != saved:
            raise ValueError(name + ': independent six-gate audit differs')
        trace = read(path / 'trace.json')
        rows = read(path / 'client.json')['requests']
        if definitions[name]['trace'] == 'smoke':
            continue
        ids = [r['trace_id'] for r in trace['requests']]
        if (len(ids) != len(set(ids)) or len(rows) != len(ids)
                or Counter(r['trace_id'] for r in rows) != Counter(ids)):
            raise ValueError(name + ': offered trace IDs differ')
        t0 = manifest['measurement_start']
        arrivals = {r['trace_id']: r['arrival_seconds'] for r in trace['requests']}
        if any(abs(r['scheduled_at'] - t0 - arrivals[r['trace_id']]) > 1e-8 for r in rows):
            raise ValueError(name + ': scheduled arrivals differ from the trace')
        meta, req, replacements = load_capture_compatible(path)
        captures.append(dict(name=name, path=path, meta=meta, req=req,
                             scrapes=gb.load_scrapes(path, t0), report=saved,
                             definition=definitions[name], identity_adaptations=replacements))
    trace_hashes = {read(c['path'] / 'manifest.json')['trace_sha256'] for c in captures}
    if len(captures) != 6 or len(trace_hashes) != 1:
        raise ValueError('Expected six development captures of the same trace')
    return captures, dict(artifact_files_verified=checked, six_gates_independently_reproduced=7,
                          suite_sha256=sha(source / 'suite.json'), protocol_sha256=digest,
                          cpu_test_cases=acceptance['acceptance']['test_cases'],
                          transport_failure_client_ids_used={c['name']: c['identity_adaptations'] for c in captures},
                          backend=suite['backend'], input_sha256={
                              r['name']: r['artifact_sha256'] for r in suite['runs']})


def actual(run, t):
    r = run.req
    cohort = (r['sched'] >= t) & (r['sched'] < t + gb.HORIZON)
    return dict(goodput=float((cohort & r['met']).sum() / gb.HORIZON),
                admission=float((cohort & r['admitted']).sum() / gb.HORIZON),
                offered=int(cohort.sum()))


def forecast(run, t, coef, overhead, residuals, seed=SEED):
    """Identical past-window draws for the actual plan and both hypothetical caps."""
    rng = np.random.default_rng(seed)
    bias = gb.online_bias(residuals, overhead, t)
    predictions = {key: [] for key in ('simulator', 'no_bias', 'admission', 'cap16', 'cap32')}
    for _ in range(BOOTS):
        arrivals = run.bootstrap_arrivals(t, rng)
        out = gb.simulate(run, t, arrivals, coef)
        predictions['simulator'].append(gb.expected_met(out, overhead, bias) / gb.HORIZON)
        predictions['no_bias'].append(gb.expected_met(out, overhead) / gb.HORIZON)
        predictions['admission'].append(sum(bool(x.admitted) for x in out) / gb.HORIZON)
        for cap in (16, 32):
            counterfactual = gb.simulate(run, t, arrivals, coef, cap_fn=lambda at, c=cap: c)
            predictions['cap' + str(cap)].append(gb.expected_met(counterfactual, overhead, bias) / gb.HORIZON)
    return dict(online_bias_s=bias, nowcast=ev.nowcast(run.req, t),
                **{key: float(np.mean(values)) for key, values in predictions.items()})


def oracle_diagnostic(run, coef, overhead, residuals, sims, times):
    """Actual future arrivals, never a deployable predictor; paired admitted subset TTFT."""
    index = {rid: i for i, rid in enumerate(run.req['request_id'])}
    output = []
    for t in times:
        out = sims[float(t)]
        bias = gb.online_bias(residuals, overhead, t)
        matched = []
        for o in out:
            i = index[o.request_id]
            if o.admitted and math.isfinite(o.first) and run.req['admitted'][i] and math.isfinite(run.req['ttft'][i]):
                pred = o.first - o.sched + float(overhead.mean(o.running)) + bias
                matched.append((pred, float(run.req['ttft'][i])))
        output.append(dict(t=t, admission=sum(bool(o.admitted) for o in out) / gb.HORIZON,
                           goodput=gb.expected_met(out, overhead, bias) / gb.HORIZON,
                           ttft_matched_n=len(matched),
                           ttft_absolute_error_sum=sum(abs(p - a) for p, a in matched),
                           ttft_signed_error_sum=sum(p - a for p, a in matched)))
    return output


def metrics(rows, pred='simulator', target='actual_goodput'):
    error = np.array([r[pred] - r[target] for r in rows])
    return dict(n=len(rows), mae=float(np.abs(error).mean()), bias=float(error.mean()))


def matched_phase_differences(rows, captures):
    """Factual A/B gap only; engine states/history differ, so no action-accuracy claim."""
    definitions = {c['name']: c['definition'] for c in captures}
    lookup = {(r['run'], r['t']): r for r in rows}
    result = []
    for rep in (1, 2):
        a, b = [next(n for n, d in definitions.items() if d['schedule'] == s and d['replicate'] == rep)
                for s in ('a', 'b')]
        for t in range(START, STOP + 1, STRIDE):
            left, right = lookup[a, t], lookup[b, t]
            high, low = (left, right) if left['cap'] == 32 else (right, left)
            observed = high['actual_goodput'] - low['actual_goodput']
            predicted = high['simulator'] - low['simulator']
            result.append(dict(replicate=rep, t=t, observed_gap=observed, predicted_gap=predicted,
                               same_direction=None if observed == 0 or predicted == 0 else bool(observed * predicted > 0)))
    return result


def evaluate(captures, coef, overhead, reference):
    times = list(range(START, STOP + 1, STRIDE))
    rows, oracles, measured = [], [], []
    for c in captures:
        name = c['name']
        run = gb.Run(name, c['meta'], c['req'], c['scrapes'], pre=reference['pre_s'])
        # Historical simulations are censored at each own horizon + drain boundary.
        residuals, sims = gb.residuals(run, coef, np.arange(5., STOP + 1, 1.))
        diagnostic = oracle_diagnostic(run, coef, overhead, residuals, sims, times)
        for t, oracle in zip(times, diagnostic):
            obs = actual(run, t)
            p = forecast(run, t, coef, overhead, residuals, seed=SEED + int(t))
            rows.append(dict(run=name, t=t, cap=run.cap(t), near_change=ev.near_change(run, t),
                             actual_goodput=obs['goodput'], actual_admission=obs['admission'],
                             offered=obs['offered'], **p))
            oracles.append(dict(run=name, **oracle))
        perf = c['report']['performance']
        measured.append(dict(run=name, schedule=c['definition']['schedule'],
                             goodput=perf['slo_goodput_per_arrival_second'],
                             successful=perf['success'], failed=perf['failed']))
        print('Scored:', name, flush=True)
    scores = {key: metrics(rows, key) for key in ('simulator', 'no_bias', 'nowcast')}
    scores['admission'] = metrics(rows, 'admission', 'actual_admission')
    oracle_rows = [dict(actual_goodput=r['actual_goodput'], actual_admission=r['actual_admission'],
                        simulator=o['goodput'], admission=o['admission']) for r, o in zip(rows, oracles)]
    scores['oracle_arrivals_goodput'] = metrics(oracle_rows)
    scores['oracle_arrivals_admission'] = metrics(oracle_rows, 'admission', 'actual_admission')
    n = sum(o['ttft_matched_n'] for o in oracles)
    oracle_ttft = dict(matched_pairs=n,
                       mae_s=sum(o['ttft_absolute_error_sum'] for o in oracles) / n if n else None,
                       bias_s=sum(o['ttft_signed_error_sum'] for o in oracles) / n if n else None,
                       scope='Oracle arrivals; accepted in both simulation and observation; survivor-selected diagnostic')
    per_run = {c['name']: {k: metrics([r for r in rows if r['run'] == c['name']], k)
                           for k in ('simulator', 'no_bias', 'nowcast')} for c in captures}
    subsets = {}
    for label, sub in [('near_change', [r for r in rows if r['near_change']]),
                       ('not_near_change', [r for r in rows if not r['near_change']]),
                       ('cap16', [r for r in rows if r['cap'] == 16]),
                       ('cap32', [r for r in rows if r['cap'] == 32])]:
        subsets[label] = {k: metrics(sub, k) for k in ('simulator', 'no_bias', 'nowcast')}
    gaps = matched_phase_differences(rows, captures)
    eligible = [g for g in gaps if g['same_direction'] is not None]
    gap_score = dict(n=len(gaps), nonzero_pairs=len(eligible),
                     descriptive_sign_agreement=sum(g['same_direction'] for g in eligible) / len(eligible),
                     always_positive_gap_sign_agreement=sum(g['observed_gap'] > 0 for g in eligible) / len(eligible),
                     always_negative_gap_sign_agreement=sum(g['observed_gap'] < 0 for g in eligible) / len(eligible),
                     gap_mae=float(np.mean([abs(g['predicted_gap'] - g['observed_gap']) for g in gaps])),
                     interpretation='Factual matched-arrival gaps across different states; NOT counterfactual action accuracy')
    return dict(scores=scores, per_run=per_run, subsets=subsets, oracle_ttft=oracle_ttft,
                observed_whole_runs=measured, factual_gap_score=gap_score,
                predictions=rows, oracle_diagnostics=oracles, factual_phase_gaps=gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', required=True, type=Path,
                        help='.../results/low-cap-switch-v1, with original cpu-ci-artifacts alongside results')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source, output = args.suite_dir.resolve(), args.output_dir.resolve()
    if output.exists() or source in output.parents or output in source.parents:
        parser.error('Choose a fresh output directory outside the original capture suite')
    guarded = ['greybox_simulator.py', 'evaluate_greybox.py', 'evaluate_low_cap_frozen.py',
               'experiments/low-cap-frozen-v1-calibration.json']
    code = {f: ev.code_sha256(ROOT / f) for f in guarded}
    coef, overhead, reference, archive = frozen_parameters()
    captures, audit = audit_suite(source)
    result = evaluate(captures, coef, overhead, reference)
    if code != {f: ev.code_sha256(ROOT / f) for f in guarded}:
        raise ValueError('Scoring source changed during evaluation')
    result.update(protocol=dict(model_commit=MODEL_COMMIT, collection_commit=SUITE_COMMIT,
                                 role='development_frozen_model_transfer_diagnosis',
                                 outcome_summary_seen_before_analysis_design=True,
                                 independent_acceptance=False, controller_executed=False,
                                 horizons_s=gb.HORIZON, origins_start_s=START, origins_end_s=STOP,
                                 origins_stride_s=STRIDE, boot_k=BOOTS, seed=SEED,
                                 batch_tokens_assumption=gb.B, output_tokens_assumption=gb.OUT,
                                 source_parameters_refitted=False, frozen_pre_s=reference['pre_s'],
                                 online_rule='Original causal bias; each complete historical window must be available'),
                  audit=audit, code_sha256=code, frozen_step_coefficients=archive['step_coefficients'],
                  model_readiness='not_established',
                  limits=['New host; same synthetic-prompt trace and output cap; no independent C.',
                          'First 60 s supply causal history; forecast scoring covers 60-300 s only. Whole-run measurements retain all 300 s.',
                          'Step/overhead arrays and reference transport delay are frozen; original online bias is an explicit ablation.',
                          'Simulator predicts first-token hits; successfully completed cohorts are the target; terminal failures are not modeled.',
                          'TTFT oracle diagnostic conditions on admission in both records and is not all-offered accuracy.',
                          'Candidate cap16/cap32 use the same current state and draws but have no observed counterfactual labels.',
                          'Factual A/B gap agreement cannot establish action accuracy, policy value, or a gain over fixed cap.',
                          'Five-second forecasts, including nearby origins, share history; no independent-sample confidence claim.'])
    output.mkdir(parents=True)
    full = json.dumps(ev.clean(result), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    (output / 'full-report.json.gz').write_bytes(gzip.compress(full, mtime=0))
    summary = {k: v for k, v in result.items() if k not in ('predictions', 'oracle_diagnostics', 'factual_phase_gaps')}
    summary['full_report_sha256'] = hashlib.sha256(full).hexdigest()
    (output / 'summary.json').write_text(json.dumps(ev.clean(summary), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(scores=result['scores'], factual_gap_score=result['factual_gap_score']), indent=2))


if __name__ == '__main__':
    main()
