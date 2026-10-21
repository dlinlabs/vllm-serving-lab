"""Re-audit low-cap identification captures and compare matched arrival phases, without fitting a model."""
import argparse
from pathlib import Path

from analyze_cap_sweep import audit_copy, describe, paired
from experiment_protocol import expected_schedule, load_protocol
from run_cap_sweep import ROOT, gate_issues, qualify_capture, read, save
from trace_workload import sha256_file


def offered_outcomes(rows, slo):
    result = {}
    for row in rows:
        key = row['trace_id']
        if key in result:
            raise ValueError('Duplicate offered trace IDs')
        first = row.get('first_content_at')
        result[key] = bool(row['success'] and row.get('stream_complete') and first is not None
                           and first - row['scheduled_at'] <= slo)
    return result


def cohorts(capture, start, end):
    arrivals = capture['arrivals']
    return [r for r in capture['rows'] if start <= arrivals[r['trace_id']] < end]


def cap_at(capture, t):
    return next(s['budget'] for s in reversed(capture['schedule']) if s['at_seconds'] <= t)


def paired_windows(left, right, windows, slo):
    return [dict(from_s=start, to_s=end, **paired(offered_outcomes(cohorts(left, start, end), slo),
                                                   offered_outcomes(cohorts(right, start, end), slo), end - start))
            for start, end in windows]


def compare(captures, document):
    """All windows are prespecified arrival cohorts, with eventual successful outcomes."""
    hold = document['design']['analysis']['hold_seconds']
    early = document['design']['analysis']['transient_seconds']
    sub = document['design']['analysis']['sub_bin_seconds']
    slo = document['replay']['ttft_slo']
    full, holds, bins, phase_pairs, repeats = [], [], [], [], []
    for c in captures:
        full.append(dict(name=c['name'], schedule=c['definition']['schedule'],
                         replicate=c['definition']['replicate'], **describe(c['rows'], 300., slo)))
        for start in range(0, 300, hold):
            for label, a, b in (('all', start, start + hold), ('early', start, start + early),
                                ('late', start + early, start + hold)):
                holds.append(dict(name=c['name'], schedule=c['definition']['schedule'],
                                  cap=cap_at(c, start), from_s=a, to_s=b, phase=label,
                                  **describe(cohorts(c, a, b), b - a, slo)))
        for start in range(0, 300, sub):
            bins.append(dict(name=c['name'], cap=cap_at(c, start), from_s=start, to_s=start + sub,
                             **describe(cohorts(c, start, start + sub), sub, slo)))
    for rep in (1, 2):
        a, b = (next(c for c in captures if c['definition']['schedule'] == s
                     and c['definition']['replicate'] == rep) for s in ('a', 'b'))
        for start in range(0, 300, hold):
            high, low = (a, b) if cap_at(a, start) == 32 else (b, a)
            if cap_at(high, start) != 32 or cap_at(low, start) != 16:
                raise ValueError('A/B must assign opposite low caps at every phase')
            for label, lo, hi in (('all', start, start + hold), ('early', start, start + early),
                                  ('late', start + early, start + hold)):
                phase_pairs.append(dict(replicate=rep, phase=label, cap32_run=high['name'], cap16_run=low['name'],
                                        **paired_windows(high, low, [(lo, hi)], slo)[0]))
    for schedule in ('a', 'b'):
        first, second = (next(c for c in captures if c['definition']['schedule'] == schedule
                             and c['definition']['replicate'] == rep) for rep in (1, 2))
        for row in paired_windows(second, first, [(s, s + hold) for s in range(0, 300, hold)], slo):
            repeats.append(dict(schedule=schedule, left=second['name'], right=first['name'], **row))
    return dict(per_run=full, arrival_holds=holds, arrival_5s_bins=bins,
                cap32_minus_cap16_matched_phases=phase_pairs, same_schedule_repeat2_minus_repeat1=repeats)


def analyze(source, output, plot=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('Use a fresh analysis output outside the raw suite')
    suite = read(source / 'suite.json')
    if suite.get('stage') != 'full' or suite.get('valid') is not True or suite.get('status') != 'collected':
        raise ValueError('A complete qualified full suite is required; preserve incomplete attempts separately')
    document, sha = load_protocol(source / 'protocol.json')
    if document.get('protocol') != 'low-cap-switch-v1' or document.get('status') != 'frozen' or sha != suite['protocol_sha256']:
        raise ValueError('Frozen collection protocol/hash differs')
    order = document['qualification']['order']
    if suite['planned'] != order or [r['name'] for r in suite['runs']] != order:
        raise ValueError('Missing, duplicate or reordered runs')
    acceptance = suite.get('cpu_acceptance', {}).get('acceptance', {})
    if acceptance.get('valid') is not True or acceptance.get('commit') != suite['commit']:
        raise ValueError('Native CPU acceptance must cover the exact collection revision')
    captures, expected_code = [], None
    definitions = {r['name']: r for r in document['runs']}
    for record in suite['runs']:
        name, path = record['name'], source / record['name']
        if record.get('status') != 'passed' or record.get('exitcode') != 0 or record.get('backend') != suite['backend']:
            raise ValueError(f'{name}: capture failed or backend/host changed')
        actual = {p.name: sha256_file(p) for p in path.iterdir() if p.is_file()}
        if not actual or actual != record.get('artifact_sha256'):
            raise ValueError(f'{name}: original artifact hashes/files differ')
        manifest, saved = qualify_capture(path, document, sha, name, suite['commit'], expected_code)
        expected_code = manifest['code_sha256']
        if manifest.get('host') != suite['backend']['host'] or not suite.get('packages') or manifest.get('packages') != suite['packages']:
            raise ValueError(f'{name}: host or package identity differs')
        audited = audit_copy(path)
        if gate_issues(audited) or audited != saved:
            raise ValueError(f'{name}: independent capture audit failed or differs')
        definition = definitions[name]
        trace = read(path / 'trace.json')
        _, schedule = expected_schedule(document['schedules'][definition['schedule']], trace['duration_seconds'])
        label = (schedule[0]['budget'] if len(schedule) == 1 else [s['budget'] for s in schedule])
        if record['cap'] != label:
            raise ValueError(f'{name}: cap label differs from declared schedule')
        if definition['trace'] == 'smoke':
            continue
        if trace['duration_seconds'] != 300.:
            raise ValueError('Development replay duration must be 300 seconds')
        rows = [r for r in read(path / 'client.json')['requests'] if r.get('workload') == 'trace']
        arrivals = {r['trace_id']: r['arrival_seconds'] for r in trace['requests']}
        if len(arrivals) != len(trace['requests']) or set(offered_outcomes(rows, document['replay']['ttft_slo'])) != set(arrivals):
            raise ValueError(f'{name}: offered trace IDs do not match')
        captures.append(dict(name=name, definition=definition, schedule=schedule, rows=rows, arrivals=arrivals,
                             trace_sha256=manifest['trace_sha256']))
    if len({c['trace_sha256'] for c in captures}) != 1:
        raise ValueError('All development runs must replay the same trace')
    report = dict(valid=True, protocol='low-cap-switch-v1', protocol_sha256=sha, commit=suite['commit'],
                  role='development_identification', model_readiness='not_established', **compare(captures, document),
                  limits=['One host and one trace; not a global cap optimum or MPC result.',
                          'Fixed references have one capture each; no variance estimate from those alone.',
                          'A/B order balances linear ordinal drift only; state/history and nonlinear drift remain.',
                          'Matched arrivals do not imply matched system state or a causal counterfactual.',
                          'Early/late windows are prespecified descriptions, not a verified settling time.',
                          'No model is fitted or independently accepted here; C stays unused.'])
    output.mkdir(parents=True)
    save(output / 'summary.json', report)
    lines = ['# Low-cap phase/transition identification', '',
             '| Run | Schedule | SLO goodput |', '|---|---|---:|']
    lines += [f'| {r["name"]} | {r["schedule"]} | {r["slo_goodput_per_arrival_second"]:.3f} |' for r in report['per_run']]
    lines += ['', 'Cap32 minus cap16 at matched arrival phases (all 60 seconds):', '',
              '| Repeat | Arrival seconds | Difference (req/s) |', '|---:|---|---:|']
    lines += [f'| {r["replicate"]} | {r["from_s"]}–{r["to_s"]} | {r["left_minus_right_goodput"]:+.3f} |'
              for r in report['cap32_minus_cap16_matched_phases'] if r['phase'] == 'all']
    lines += ['', *['- ' + line for line in report['limits']]]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 7), height_ratios=(2, 1))
        for c in captures:
            if c['definition']['schedule'] not in ('a', 'b'):
                continue
            bins = [r for r in report['arrival_5s_bins'] if r['name'] == c['name']]
            top.plot([r['from_s'] for r in bins], [r['slo_goodput_per_arrival_second'] for r in bins],
                     label=c['definition']['schedule'] + str(c['definition']['replicate']), alpha=.7)
            if c['definition']['replicate'] == 1:
                bottom.step([s['at_seconds'] for s in c['schedule']] + [300],
                            [s['budget'] for s in c['schedule']] + [c['schedule'][-1]['budget']],
                            where='post', label=c['definition']['schedule'])
        top.set(ylabel='SLO goodput / 5 s arrival cohort', title='Same arrivals, opposite cap phases; eventual completed outcomes')
        bottom.set(xlabel='Scheduled arrival second', ylabel='Cost cap', yticks=[16, 32])
        top.legend(); bottom.legend()
        fig.tight_layout(); fig.savefig(output / 'phase-response.png', dpi=140); plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    report = analyze(args.suite_dir, args.output_dir, not args.no_plot)
    import json
    print(json.dumps(dict(valid=report['valid'], per_run=report['per_run']), indent=2))


if __name__ == '__main__':
    main()
