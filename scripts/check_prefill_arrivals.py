"""CPU dry-run or immutable offline audit of the delayed-arrival pilot."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import digest, save
from prefill_lab.delayed_calibration import protocol, forecasts as predictions, evaluate, extract
from prefill_lab.effective_calibration import fit


def synthetic_rows(plan, runtime, theta):
    forecasts=predictions(theta,plan,runtime)
    rows=[dict(trial_id=t['trial_id'],case=t['case'],observer=t['observer'],stage=t['stage'],
                 block=t['block'],endpoints=copy.deepcopy(forecasts[t['case']]['endpoints']),
                 work_signature=forecasts[t['case']]['work_signature'] if t['observer']=='light' else None,
                 capture_sha256='synthetic-not-native') for t in plan['order']]
    for r in rows:
        if 'sentinel_queue_residence_s' in forecasts[r['case']]:
            q=forecasts[r['case']]['sentinel_queue_residence_s'];ttft=r['endpoints']['sentinel']['ttft_s']
            r['clocks']={'sentinel':dict(queue_residence_s=q,posting_lag_s=0.,posted_ttft_s=ttft,target_ttft_s=ttft)}
    return rows


def audit(directory):
    from prefill_lab.tokens import validate
    from prefill_lab.suite import runtime_preflight
    from prefill_lab.causal import dispatcher_parity
    load=lambda name:json.loads((directory/name).read_text())
    p=protocol();saved=load('calibration-plan.json')
    if digest(saved)!=digest(p):raise ValueError('Protocol differs from this checkout')
    summary=load('calibration.json')
    if (not summary['collection_complete'] or summary['failures'] or
            summary['completed']!=[t['trial_id'] for t in p['order']]):
        raise ValueError('Incomplete native collection')
    runtime=load('runtime.json')
    if digest(runtime)!=digest(load('runtime-final.json')):raise ValueError('Runtime drift')
    manifest=load('token-manifest.json');validate(manifest,p['cases'],p['order']);runtime_preflight(runtime,manifest)
    from prefill_lab.delayed_calibration import validate_manifest
    validate_manifest(manifest)
    pins=json.loads((ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    for source in [runtime['source'],*[w['source'] for w in runtime['workers']]]:
        if source['version']!='0.30.0' or source['files']!=pins:raise ValueError('Installed source pins differ')
    launch=load('launch.json');acceptance=launch['cpu_acceptance'];sources=load('collection-sources.json')
    if (not acceptance.get('valid') or acceptance.get('source_sha256')!=sources or
            launch['source_sha256']!=sources or launch['scope']!='effective_delayed_arrivals_pilot'):
        raise ValueError('Source/CPU acceptance identity mismatch')
    if load('native-cpu-acceptance/acceptance.json')!=acceptance:
        raise ValueError('Original CPU acceptance differs from launch record')
    from scripts.run_linux_cpu_tests import validate_report
    coverage=validate_report(directory/'native-cpu-acceptance/junit.xml')
    if coverage['test_cases']!=acceptance['tests'] or not acceptance.get('tracked_clean'):
        raise ValueError('Original JUnit count/clean-source acceptance mismatch')
    # Offline analysis must use the same source content as the native run.
    from scripts.run_prefill_service import code_identity
    if code_identity()!=sources:raise ValueError('Use the exact recorded source checkout for offline audit')
    rows=[];cases={c['id']:c for c in p['cases']}
    for t in p['order']:
        capture=load(t['trial_id']+'.json')
        rows.append(extract(capture,t,cases[t['case']],p))
        if t['observer']=='light' and not dispatcher_parity(capture,runtime['workers'][0]['dispatch_table'])['matches']:
            raise ValueError('Dispatcher mismatch')
    if digest(rows)!=digest(load('endpoints.json')):raise ValueError('Saved endpoints differ from raw captures')
    frozen=load('frozen.json')
    if frozen['data_origin']!='native_controlled_endpoints':raise ValueError('Not a native endpoint freeze')
    from prefill_lab.effective_calibration import select
    if frozen['training_sha256']!=digest(select(rows,p,{'train'},'off')):
        raise ValueError('Frozen training digest mismatch')
    result=evaluate(rows,p,runtime,frozen,load('pre-held-predictions.json'))
    if digest(result)!=digest(load('held-report.json')):raise ValueError('Saved report differs from independent audit')
    if result['candidate_passed']!=summary['candidate_passed']:raise ValueError('Summary disagrees with report')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('demo');p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('audit');p.add_argument('directory',type=Path)
    args=parser.parse_args()
    if args.command=='audit':
        result=audit(args.directory);print(json.dumps(dict(audit_valid=True,candidate_passed=result['candidate_passed'],model_accepted=False)))
        return 0 if result['candidate_passed'] else 2
    args.output.mkdir(parents=True,exist_ok=False)
    p=protocol();runtime=json.loads(args.runtime.read_text());truth=[.00935,.1006,.00733,.00873]
    rows=synthetic_rows(p,runtime,truth)
    frozen=fit(rows,p,runtime);forecast=predictions(frozen['parameters_s'],p,runtime)
    report=evaluate(rows,p,runtime,frozen,forecast)
    report.update(origin='synthetic',native_collection_performed=False,truth_parameters_s=truth)
    save(args.output/'frozen.json',frozen);save(args.output/'report.json',report)
    error=max(abs(a-b) for a,b in zip(truth,frozen['parameters_s']))
    print(json.dumps(dict(synthetic_candidate_passed=report['candidate_passed'],max_parameter_error_s=error,
                         condition=frozen['local_sensitivity']['condition'],model_accepted=False)))
    return 0 if report['candidate_passed'] and error<1e-6 else 1


if __name__=='__main__':raise SystemExit(main())
