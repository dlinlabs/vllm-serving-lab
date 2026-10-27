"""Immutable source-bound confirmation analysis; no model/control promotion."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import statistics
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import digest,save
from scripts.analyze_prefill_effective_capture import archive_identity,source_paths,endpoint_summary
from scripts.analyze_prefill_queue_probe import EXTRAS as PROBE_EXTRAS

COMMIT='2ebb18566b0e4c4751bded54ab6f18fab73abcbb'
EXTRAS=PROBE_EXTRAS|{'experiments/prefill-queue-confirmation-v2.json','experiments/prefill-queue-confirmation-v2.plan.json'}


def snapshot(capture,destination):
    load=lambda n:json.loads((capture/n).read_text());launch=load('launch.json')
    if launch['cpu_acceptance']['commit']!=COMMIT or launch['scope']!='independent_queue_confirmation':raise ValueError('Unsupported confirmation source/scope')
    def git(*args):
        try:return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:raise ValueError('Pinned source unavailable; run git fetch --depth 1 origin '+COMMIT) from exc
    names=git('ls-tree','-r','--name-only',COMMIT).decode().splitlines();sources=load('collection-sources.json')
    if set(sources)!=source_paths(names)|EXTRAS:raise ValueError('Confirmation source inventory differs')
    for name,sha in sources.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe source path')
        raw=git('show',COMMIT+':'+name)
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Git/native source mismatch: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return sources


def original_audit(capture):
    import numpy,scipy
    native=json.loads((capture/'host-before.json').read_text())['packages']
    if (numpy.__version__,scipy.__version__)!=(native['numpy'],native['scipy']):
        raise ValueError('Exact numerical replay requires captured numpy='+native['numpy']+' scipy='+native['scipy'])
    with tempfile.TemporaryDirectory(prefix='pr32-confirmation-source-') as temp:
        source=Path(temp);sources=snapshot(capture,source)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source)+os.pathsep+os.environ.get('PYTHONPATH',''))
        result=subprocess.run([sys.executable,str(source/'scripts/check_prefill_queue_confirmation.py'),str(capture)],
            cwd=source,env=env,capture_output=True,text=True,timeout=300)
        if result.returncode:raise ValueError('Original strict confirmation audit failed: '+result.stdout+result.stderr)
        verdict=json.loads(result.stdout)
        if verdict.get('audit_valid') is not True:raise ValueError('Original confirmation audit incomplete')
    return dict(commit=COMMIT,matched_source_files=len(sources),numerical_versions=dict(numpy=numpy.__version__,scipy=scipy.__version__),**verdict)


def summarize(report,frozen):
    samples=report['sample_checks']
    if len(samples)!=144 or len({c['trial_id'] for c in samples})!=144:raise ValueError('Confirmation sample denominator differs')
    groups=defaultdict(list)
    for c in samples:groups[c['case'],c['arm']].append(c)
    cases={c['case'] for c in samples}
    if len(cases)!=8 or set(groups)!={(case,arm) for case in cases for arm in ('baseline','receipt','light')} or any(len(v)!=6 for v in groups.values()):
        raise ValueError('Confirmation cell/arm denominator differs')
    table=[]
    for (case,arm),rows in sorted(groups.items()):
        q=[r['observed_s'] for r in rows];point=frozen['predictions'][case]['sentinel_queue_residence_s']
        table.append(dict(case=case,arm=arm,samples=len(rows),observed_min_s=min(q),observed_median_s=statistics.median(q),observed_max_s=max(q),
            predicted_s=point,mae_s=statistics.mean(abs(point-v) for v in q),max_absolute_error_s=max(abs(point-v) for v in q),
            signed_errors_s=[point-v for v in q],maximum_budget_fraction=max(r['point']['error_s']/r['point']['limit_s'] for r in rows),
            f0_point_failures=sum(not r['f0_point']['passed'] for r in rows),entry_point_failures=sum(not r['point']['passed'] for r in rows),
            phase_failures=sum(not r['phase']['passed'] or r['phase']['branch'] is None for r in rows),
            eligible_branches=len(frozen['cells'][case]['branches'])))
    first=frozen['first_commit_readiness']['checks'];d16=[c for c in first if c['case'].endswith('D16')]
    controls=report['measurement'];comparable=[c for c in controls['receipt_pair_checks'] if c['assessable']]
    return dict(schema='queue-confirmation-descriptive-summary-v1',status=report['status'],candidate_passed=report['candidate_passed'],
        arm_scores=report['arm_scores'],queue_cells=table,branch_coverage=report['branch_coverage'],
        eligible_branches={k:len(v['branches']) for k,v in frozen['cells'].items()},
        f0_and_entry_forecasts_identical=frozen['predictions']==frozen['f0_predictions'],
        correction_benefit_established=False,previous_host_phase_switch_confirmed=False,baseline_native_phase_observed=False,
        endpoint_scores={k:endpoint_summary(v['cells']) for k,v in report['endpoint_scores'].items()},
        first_commit_readiness=dict(passed=frozen['first_commit_readiness']['passed'],samples=len(first),d16_samples=len(d16),
            max_absolute_error_s=max(abs(c['error_s']) for c in first),d16_max_absolute_error_s=max(abs(c['error_s']) for c in d16)),
        observer_controls=dict(comparable_pairs=len(comparable),different_state_pairs=controls['different_state_pairs'],
            max_same_state_queue_delta_s=max(c['delta_s'] for c in comparable),
            max_endpoint_delta_s=max(c['delta_s'] for c in controls['controls']),
            max_endpoint_budget_fraction=max(c['delta_s']/c['limit_s'] for c in controls['controls'])),
        anchor_max_delta_s=max(abs(c['delta_s']) for c in report['anchor_drift']),
        anchor_max_budget_fraction=max(abs(c['delta_s'])/c['limit_s'] for c in report['anchor_drift']),
        scope='fixed B2048 controlled arrivals on one host; not a correction-win, phase-switch, HTTP, action-validity or MPC result',
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


def run(capture,archive,output):
    capture,archive,output=map(lambda p:Path(p).resolve(),(capture,archive,output))
    if output==capture or capture in output.parents:raise ValueError('Output must be outside immutable capture')
    identity,index=archive_identity(archive,capture);output.mkdir(parents=True,exist_ok=False)
    save(output/'input-files.json',index);save(output/'archive.json',identity)
    proof=original_audit(capture);save(output/'original-audit.json',proof)
    read=lambda n:json.loads((capture/n).read_text())
    summary=summarize(read('confirmation-report.json'),read('entry-freeze.json'))
    save(output/'summary.json',summary);return dict(original_audit=proof,archive=identity,summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=run(args.capture,args.archive,args.output)
    print(json.dumps(dict(audit_valid=result['original_audit']['audit_valid'],candidate_passed=result['summary']['candidate_passed'],
        correction_benefit_established=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)))
