"""Re-audit every collected run and describe the incomplete cap-range attempt."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import tempfile

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
from analyze_trace_pilot import analyze as audit
from experiment_protocol import capture_identity, capture_mismatches
from run_cap_sweep import qualify_capture

GATES = ('load_fidelity', 'control_fidelity', 'diagnostic_integrity',
         'runner_diagnostic_integrity', 'runner_gc_integrity')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def good(row, slo):
    return bool(row['success'] and row.get('first_content_at') is not None and
                row['first_content_at'] - row['scheduled_at'] <= slo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    raw, output = args.raw_root.resolve(), args.output_dir.resolve()
    if output.exists() or any(output == p or p in output.parents for p in (raw, CODE)):
        parser.error('Use a fresh output outside the raw and code trees')
    bundle = read(HERE / 'raw-bundle/manifest.json')
    for name, expected in bundle['files'].items():
        p = raw / name
        if not p.is_file() or p.stat().st_size != expected['bytes'] or digest(p) != expected['sha256']:
            raise ValueError('Original extracted file differs: ' + name)
    suite_root = raw / 'results/cap-range-v1'
    suite = read(suite_root / 'suite.json')
    protocol = read(suite_root / 'protocol.json')
    if digest(raw / 'experiments/cap-range-v1.json') != suite['protocol_sha256']:
        raise ValueError('Protocol differs from suite')
    if protocol != read(raw / 'experiments/cap-range-v1.json'):
        raise ValueError('Archived protocol copies differ')
    if suite['status'] != 'failed' or suite['valid'] is not False:
        raise ValueError('This publication preserves a failed, incomplete suite')
    names = [r['name'] for r in suite['runs']]
    if names != suite['planned'][:len(names)]:
        raise ValueError('Attempt order differs from declaration')
    if suite['planned'] != protocol['qualification']['order']:
        raise ValueError('Planned runs differ from frozen declaration')
    definitions = {run['name']: run for run in protocol['runs']}
    from scripts.run_linux_cpu_tests import validate_report
    cpu_dir = raw / 'cpu-ci-artifacts' / Path(suite['cpu_acceptance']['path']).parent.name
    cpu = read(cpu_dir / 'acceptance.json')
    coverage = validate_report(cpu_dir / 'junit.xml')
    if (cpu != suite['cpu_acceptance']['acceptance'] or cpu['valid'] is not True or
            cpu['commit'] != suite['commit'] or cpu['test_cases'] != coverage['test_cases']):
        raise ValueError('Exact-revision native CPU evidence differs')
    for filename, key in [('acceptance.json','acceptance_sha256'),('junit.xml','junit_sha256')]:
        if digest(cpu_dir/filename) != suite['cpu_acceptance'][key]:
            raise ValueError('Supervisor CPU fingerprint differs')
    summaries, clients, manifests = [], {}, {}
    recorded_hashes = 0
    for declared in suite['runs']:
        name = declared['name']
        p = suite_root / name
        manifest, saved = read(p / 'manifest.json'), read(p / 'report.json')
        trace, replay = read(p/'trace.json'), read(p/'replay_plan.json')
        identity = capture_identity(manifest['config'], trace, digest(p/'trace.json'), replay,
                                    manifest.get('experiment_protocol'))
        if capture_mismatches(protocol, name, identity):
            raise ValueError('Frozen capture identity differs: ' + name)
        binding = manifest['experiment_protocol']
        expected_cap = protocol['schedules'][definitions[name]['schedule']]['fixed_budget']
        if (declared['backend'] != suite['backend'] or manifest['host'] != suite['backend']['host'] or
                declared['cap'] != expected_cap or binding['sha256'] != suite['protocol_sha256'] or
                binding['run'] != name or binding['role'] != 'development'):
            raise ValueError('Backend/cap/protocol identity differs: ' + name)
        for path, expected in manifest['code_sha256'].items():
            if digest(CODE / path) != expected:
                raise ValueError('Use code matching the capture: ' + path)
        if manifest['git_commit'] != suite['commit'] or manifest['packages'] != suite['packages']:
            raise ValueError('Capture source/packages differ')
        for path, expected in declared.get('artifact_sha256', {}).items():
            if digest(p / path) != expected:
                raise ValueError('Supervisor artifact fingerprint differs: ' + name + '/' + path)
            recorded_hashes += 1
        # The original auditor writes only to the temporary view; raw reports stay immutable.
        with tempfile.TemporaryDirectory(prefix='cap-range-audit-') as directory:
            temporary = Path(directory)
            for f in p.iterdir():
                if f.is_file() and f.name not in ('report.json', 'timeseries.png'):
                    (temporary / f.name).symlink_to(f)
            regenerated = audit(temporary, plot=False)
        if regenerated != saved:
            raise ValueError('Original report did not reproduce: ' + name)
        qualified = saved['valid'] and all(saved[k]['valid'] for k in GATES)
        if qualified != (declared['status'] == 'passed'):
            raise ValueError('Qualification differs from supervisor: ' + name)
        if qualified:
            qualify_capture(p, protocol, suite['protocol_sha256'], name, suite['commit'])
        capture_clients = read(p / 'client.json')['requests']
        arrivals = {r['trace_id']: r['arrival_seconds'] for r in trace['requests']}
        slo, duration = manifest['config']['ttft_slo'], trace['duration_seconds']
        counts = [Counter() for _ in range(int(duration / 30))]
        for row in capture_clients:
            cohort = counts[int(arrivals[row['trace_id']] // 30)]
            cohort.update(offered=1, success=int(row['success']), rejected=int(row['rejected']),
                          good=int(good(row, slo)))
        summaries.append(dict(name=name, cap=declared['cap'], qualified=qualified,
            supervisor_status=declared['status'], raw_report_reproduced=True,
            gate_verdicts=dict(integrity=saved['valid'], **{k:saved[k]['valid'] for k in GATES}),
            gate_issues={k:saved[k]['issues'] for k in GATES if not saved[k]['valid']},
            performance=saved['performance'], performance_by_input_size=saved['performance_by_input_size'],
            load_fidelity=saved['load_fidelity'], timing=saved['timing'],
            duration_seconds=duration, cohorts_30s=[dict(c) for c in counts],
            git_dirty_self_reported=manifest['git_dirty']))
        clients[name], manifests[name] = capture_clients, manifest
        print('Reproduced:', name, 'qualified=' + str(qualified), flush=True)
    fixed = [r for r in summaries if r['qualified'] and '-fixed-' in r['name']]
    by_cap = {}
    for r in fixed:
        by_cap.setdefault(str(int(r['cap'])), []).append(r['performance']['slo_goodput_per_arrival_second'])
    paired = []
    for rep in ('r1', 'r2'):
        left_name = 'cap-range-v1-fixed-32-' + rep
        left = next(r for r in fixed if r['name'] == left_name)
        for cap in (16, 64, 96):
            right_name = f'cap-range-v1-fixed-{cap}-{rep}'
            right = next((r for r in fixed if r['name'] == right_name), None)
            if right is None:
                continue
            a = {r['trace_id']: good(r, .5) for r in clients[left_name]}
            b = {r['trace_id']: good(r, .5) for r in clients[right_name]}
            if a.keys() != b.keys():
                raise ValueError('Offered trace IDs differ')
            deltas = [x['good']-y['good'] for x,y in zip(left['cohorts_30s'], right['cohorts_30s'])]
            paired.append(dict(left=left_name, right=right_name, both_good=sum(a[k] and b[k] for k in a),
                only_left_good=sum(a[k] and not b[k] for k in a),
                only_right_good=sum(not a[k] and b[k] for k in a),
                cohort_good_count_deltas=deltas, winning_cohorts=sum(d>0 for d in deltas)))
    failed_name = 'cap-range-v1-fixed-16-r2'
    p = suite_root / failed_name
    manifest, gateway = manifests[failed_name], rows(p / 'gateway.jsonl')
    runner, engine = rows(p / 'runner_observations.jsonl'), rows(p / 'engine_observations.jsonl')
    start = manifest['measurement_start']
    loop = max((e for e in gateway if e['event']=='event_loop_lag' and
                start<=e['scheduled_at']<=start+300), key=lambda e:e['lag_seconds'])
    due, resumed = loop['scheduled_at'], loop['monotonic']
    gc = [e for e in gateway if e['event']=='gc_span' and e['span_start']<resumed and e['span_end']>due]
    cpu = [e for e in gateway if e['event']=='diagnostic_cpu']
    before = max((e for e in cpu if e['sampled_at']<=due),key=lambda e:e['sampled_at'])
    after = min((e for e in cpu if e['sampled_at']>=resumed),key=lambda e:e['sampled_at'])
    late = [r for r in clients[failed_name] if r['started_at']-r['scheduled_at']>.05]
    errors = [e for e in engine if e.get('error_type')]
    incident = dict(run=failed_name, gateway_due_second=due-start, gateway_resumed_second=resumed-start,
        gateway_monitor_delay_seconds=loop['lag_seconds'], overlapping_gateway_gc_spans=len(gc),
        gateway_cpu_sample_elapsed=after['sampled_at']-before['sampled_at'],
        gateway_process_cpu_delta=after['process_cpu_seconds']-before['process_cpu_seconds'],
        gateway_loop_cpu_delta=after['loop_thread_cpu_seconds']-before['loop_thread_cpu_seconds'],
        metrics_errors=[dict(sent_second=e['sent_at']-start, received_second=e['received_at']-start,
                             error_type=e['error_type'], http_status=e.get('http_status')) for e in errors],
        late_launch_count=len(late), late_scheduled_first_second=min(e['scheduled_at']-start for e in late),
        late_scheduled_last_second=max(e['scheduled_at']-start for e in late),
        late_launch_outcomes=dict(Counter('success' if e['success'] else 'rejected' if e['rejected'] else 'failed' for e in late)),
        max_runner_measurement_loop_lag=max(e['lag_seconds'] for e in runner if
            e['event']=='event_loop_lag' and start<=e['scheduled_at']<=start+300),
        note='Temporal overlap is not causal/exclusive CPU attribution; no syscall/scheduler trace was collected.')
    p = suite_root / 'cap-range-v1-fixed-64-r1'
    failures = [r for r in clients[p.name] if not r['success'] and not r['rejected']]
    failed_ids = {r['request_id'] for r in failures}
    failures64 = dict(clients=failures, lifecycle=[e for e in rows(p/'gateway.jsonl') if e.get('request_id') in failed_ids])
    summary = dict(schema_version=1, attempt_status='failed_incomplete', original_suite_valid=False,
        capture_commit=suite['commit'], native_cpu_test_cases=coverage['test_cases'],
        original_files_verified=len(bundle['files']), supervisor_artifact_hashes_verified=recorded_hashes,
        source_archive_sha256=bundle['archive_sha256'], raw_reports_reproduced=len(summaries),
        qualified_fixed_runs=len(fixed), missing_runs=[n for n in suite['planned'] if n not in names],
        runs=summaries, qualified_goodput_by_cap=by_cap, paired_qualified_runs=paired,
        cap32_vs96_mean_goodput_increase_pct=100*(mean(by_cap['32'])/mean(by_cap['96'])-1),
        diagnostic_incident=incident, cap64_failed_request=failures64,
        model_readiness='not_established', MPC_run=False, C_used=False,
        limitations=['Development data; one host, two repeats at most, fixed mirrored order.',
          'Cap32 repeats are adjacent; arbitrary order/thermal drift is not controlled.',
          'Matched arrivals do not match internal states; cohorts are not independent replicates.',
          'Cap16-r2 excluded from qualified comparisons by unchanged gates; raw outcomes retained.',
          'All full-run results include the first 30 seconds and all offered requests.',
          'Synthetic prompts, original arrival/input lengths, output cap128 with natural EOS.',
          'Fixed-cap evidence does not identify low-cap switch dynamics or a global optimum.'])
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    render(summary, output)
    print(json.dumps({k:summary[k] for k in ('attempt_status','raw_reports_reproduced','qualified_fixed_runs','missing_runs')}))


def render(summary, output):
    fixed = [r for r in summary['runs'] if r['qualified'] and '-fixed-' in r['name']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4.5))
    for rep, marker in [('r1','o'),('r2','s')]:
        selected = sorted((r for r in fixed if r['name'].endswith(rep)),key=lambda r:r['cap'])
        ax.plot([r['cap'] for r in selected],[r['performance']['slo_goodput_per_arrival_second'] for r in selected],
                marker=marker,label=rep+' (qualified captures)',linewidth=1.5)
    ax.set(xlabel='Fixed reservation cap',ylabel='SLO goodput (requests / arrival second)',
           title='Same-host fixed-cap exploration: incomplete attempt',xticks=[16,32,64,96],ylim=(0,6))
    ax.legend(); ax.grid(alpha=.2)
    fig.text(.5,.015,'Missing cap64-r2; cap16-r2 failed gates. Lines guide the eye; no confidence intervals.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(output/'goodput.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    colors = {16:'C0',32:'C1',64:'C2',96:'C3'}
    for ax,rep in zip(axes,('r1','r2')):
        for cap in (16,32,64,96):
            r=next((r for r in fixed if r['name']==f'cap-range-v1-fixed-{cap}-{rep}'),None)
            if r:ax.plot([15+30*i for i in range(10)],[x['good']/30 for x in r['cohorts_30s']],marker='.',label=f'cap {cap}',color=colors[cap])
        ax.set(ylabel='SLO goodput / s',title=rep+' scheduled-arrival cohorts',ylim=(0,7));ax.legend(ncol=4);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Scheduled trace arrival time (seconds)')
    fig.text(.5,.015,'Later completion outcomes assigned to original arrival cohorts; bins are not independent trials.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(output/'arrival-cohorts.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
