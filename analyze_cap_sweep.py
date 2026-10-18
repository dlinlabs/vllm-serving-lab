"""Re-audit the fixed-cap suite and report descriptive repeats/matched arrival cohorts."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile

from experiment_protocol import load_protocol
from run_cap_sweep import ROOT, gate_issues, qualify_capture, read, save
from trace_workload import sha256_file


def audit_copy(source):
    # The existing auditor writes report.json even with plot=False. Give it a
    # private output and read-only input links; never replace the capture report.
    from analyze_trace_pilot import analyze
    with tempfile.TemporaryDirectory(prefix='cap-sweep-audit-') as directory:
        target = Path(directory)
        for path in source.iterdir():
            if path.name not in ('report.json', 'timeseries.png'):
                (target / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
        return analyze(target, plot=False)


def outcomes(rows, slo):
    return {row['trace_id']: bool(row['success'] and row['first_content_at'] - row['scheduled_at'] <= slo)
            for row in rows}


def paired(left, right, duration):
    if set(left) != set(right):
        raise ValueError('Paired comparison needs exactly the same offered trace IDs')
    both = sum(left[key] and right[key] for key in left)
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(right[key] and not left[key] for key in left)
    return dict(offered=len(left), both_met=both, left_only_met=left_only, right_only_met=right_only,
                left_minus_right_slo_successes=left_only - right_only,
                left_minus_right_goodput=(left_only - right_only) / duration,
                interpretation='Matched arrivals, not matched engine state or a causal counterfactual')


def describe(rows, duration, slo):
    from analyze_trace_pilot import stats
    result = stats(rows, duration, slo)
    successful = [row for row in rows if row['success']]
    result['reject_fraction_of_offered'] = result['rejected'] / len(rows) if rows else None
    result['slo_fraction_of_successful'] = result['slo_successes'] / len(successful) if successful else None
    result['scheduled_p50_ttft'] = (statistics.median(row['first_content_at'] - row['scheduled_at']
                                                    for row in successful) if successful else None)
    return result


def analyze_suite(source, output, plot=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('Use a fresh analysis output outside the raw suite directory')
    suite = read(source / 'suite.json')
    if suite.get('stage') != 'full' or suite.get('valid') is not True or suite.get('status') != 'collected':
        raise ValueError('A complete, qualified full suite is required; incomplete attempts stay diagnostic only')
    acceptance = suite.get('cpu_acceptance', {}).get('acceptance', {})
    if acceptance.get('valid') is not True or acceptance.get('commit') != suite['commit']:
        raise ValueError('Native CPU evidence must cover the collection revision')
    document, protocol_sha = load_protocol(source / 'protocol.json')
    if protocol_sha != suite['protocol_sha256']:
        raise ValueError('Archived protocol differs from collection bytes')
    # Use the archived collection declaration, not a later edited copy on main.
    order = document['qualification']['order']
    if suite['planned'] != order or [row['name'] for row in suite['runs']] != order:
        raise ValueError('Missing, duplicated or reordered captures')
    definitions = {run['name']: run for run in document['runs']}
    captures, expected_code = [], None
    for record in suite['runs']:
        name, path = record['name'], source / record['name']
        if record.get('status') != 'passed' or record.get('exitcode') != 0:
            raise ValueError(f'{name}: collection did not pass')
        if record.get('backend') != suite['backend']:
            raise ValueError(f'{name}: backend/host identity differs')
        declared_files = record.get('artifact_sha256', {})
        actual_files = {p.name: sha256_file(p) for p in path.iterdir() if p.is_file()}
        if not declared_files or actual_files != declared_files:
            raise ValueError(f'{name}: original artifact hashes/files differ')
        manifest, saved = qualify_capture(path, document, protocol_sha, name, suite['commit'], expected_code)
        expected_code = manifest['code_sha256']
        if manifest.get('host') != suite['backend']['host']:
            raise ValueError(f'{name}: capture host differs from suite')
        if not suite.get('packages') or manifest.get('packages') != suite['packages']:
            raise ValueError(f'{name}: runtime package versions differ')
        for filename, digest in expected_code.items():
            if sha256_file(ROOT / filename) != digest:
                raise ValueError(f'Use the capture runtime to audit: {filename} differs')
        audited = audit_copy(path)
        if gate_issues(audited) or audited != saved:
            raise ValueError(f'{name}: independent audit failed or differs from saved report')
        trace = read(path / 'trace.json')
        rows = read(path / 'client.json')['requests']
        run = definitions[name]
        cap = document['schedules'][run['schedule']]['fixed_budget']
        if record['cap'] != cap:
            raise ValueError(f'{name}: suite cap label differs from declared capture')
        captures.append(dict(name=name, trace_role=run['trace'], cap=cap,
                             replicate=run.get('replicate'), trace=trace, rows=rows, report=audited,
                             manifest=manifest))
    development = [run for run in captures if run['trace_role'] == 'development']
    caps = sorted(schedule['fixed_budget'] for schedule in document['schedules'].values())
    keys = [(run['cap'], run['replicate']) for run in development]
    if len(keys) != len(set(keys)) or set(keys) != {(cap, rep) for cap in caps for rep in (1, 2)}:
        raise ValueError('Need exactly two declared development repeats per fixed cap')
    if len({run['manifest']['trace_sha256'] for run in development}) != 1:
        raise ValueError('Development runs must replay the same trace')
    per_run, by_cap, bins, pairings = [], [], [], []
    for run in development:
        duration, slo = run['trace']['duration_seconds'], run['manifest']['config']['ttft_slo']
        performance = describe(run['rows'], duration, slo)
        per_run.append(dict(name=run['name'], cap=run['cap'], replicate=run['replicate'], **performance,
                            performance_by_input_size=run['report']['performance_by_input_size']))
        arrivals = {row['trace_id']: row['arrival_seconds'] for row in run['trace']['requests']}
        for index in range(int(duration / 30)):
            cohort = [row for row in run['rows'] if index * 30 <= arrivals[row['trace_id']] < (index + 1) * 30]
            bins.append(dict(name=run['name'], cap=run['cap'], replicate=run['replicate'],
                             from_s=index * 30, to_s=(index + 1) * 30, **describe(cohort, 30, slo)))
        run['outcomes'] = outcomes(run['rows'], slo)
    for cap in caps:
        repeats = [run for run in development if run['cap'] == cap]
        values = [run['report']['performance']['slo_goodput_per_arrival_second'] for run in repeats]
        left, right = sorted(repeats, key=lambda run: run['replicate'])
        by_cap.append(dict(cap=cap, repeats=len(repeats), goodput_values=values,
                           mean_goodput=statistics.mean(values), min_goodput=min(values), max_goodput=max(values),
                           repeat_2_minus_1=paired(right['outcomes'], left['outcomes'], duration)))
    for rep in (1, 2):
        reference = next(run for run in development if run['cap'] == 64 and run['replicate'] == rep)
        for run in development:
            if run['replicate'] == rep and run['cap'] != 64:
                pairings.append(dict(left=run['name'], right=reference['name'],
                                     **paired(run['outcomes'], reference['outcomes'], duration)))
    report = dict(schema_version=1, valid=True, evaluation_role='development',
                  model_readiness='not_established', protocol_sha256=protocol_sha, runtime=suite['commit'],
                  qualification='All capture reports independently reproduced, original artifact hashes matched',
                  per_run=per_run, by_cap=by_cap, matched_30s_cohorts=bins, paired_to_cap64=pairings,
                  limits=['Two repeats per cap; ranges are descriptive, not confidence intervals or a noise floor',
                          'Mirrored order balances linear ordinal drift, not arbitrary nonlinear drift/carryover',
                          'Same offered trace IDs do not ensure equal admitted work or engine state',
                          'One host, one 300-second workload window, output cap 128; no global optimum established',
                          'No held-out C, forecast acceptance or closed-loop MPC evaluation'])
    output.mkdir(parents=True)
    save(output / 'summary.json', report)
    lines = ['# Fixed-cap range: descriptive development results', '',
             '| Cap | Repeat 1 goodput | Repeat 2 goodput | Mean |', '|---:|---:|---:|---:|']
    for row in by_cap:
        runs = sorted((r for r in per_run if r['cap'] == row['cap']), key=lambda r: r['replicate'])
        lines.append(f'| {row["cap"]:g} | {runs[0]["slo_goodput_per_arrival_second"]:.3f} | '
                     f'{runs[1]["slo_goodput_per_arrival_second"]:.3f} | {row["mean_goodput"]:.3f} |')
    lines.extend(['', *[f'- {item}.' for item in report['limits']], '',
                  'Thirty-second bins use scheduled-arrival cohorts, retaining rejected/failed requests. '
                  'They are correlated subdivisions, not independent replicates. Whole-run primary '
                  'goodput includes the first 30 seconds; no post-hoc warmup exclusion.'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rep in (1, 2):
            runs = sorted((r for r in per_run if r['replicate'] == rep), key=lambda r: r['cap'])
            ax.plot([r['cap'] for r in runs], [r['slo_goodput_per_arrival_second'] for r in runs],
                    'o--', label=f'Repeat {rep}')
        ax.set(xlabel='Fixed admission cost cap', ylabel='SLO goodput (requests / arrival second)',
               title='Same-host development sweep — two repeats, no confidence intervals', xticks=caps,
               ylim=(0, None))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'goodput.png', dpi=160)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite_dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    report = analyze_suite(args.suite_dir, args.output_dir, not args.no_plot)
    print(json.dumps(dict(valid=report['valid'], by_cap=report['by_cap']), indent=2))


if __name__ == '__main__':
    main()
