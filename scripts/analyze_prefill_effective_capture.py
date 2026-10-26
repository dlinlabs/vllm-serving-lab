"""Read-only, source-bound analysis of the first native effective endpoint pilot.

Run the ORIGINAL audit from verified Git source in an isolated temporary tree,
then describe fresh-held errors separately from held training-cell references.
No capture/threshold/frozen model is modified and no new acceptance is granted.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import os

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import digest, save

COMMIT='dc9847e33d93541071a0bbc4ea38e4fff4623131'
EXPLICIT={'experiments/prefill-vllm-v030-source-map.json','experiments/prefill-stage-diagnostic-v1.json',
          'experiments/prefill-effective-calibration-v1.json','pyproject.toml'}


def read(directory,name):return json.loads((directory/name).read_text())


def source_paths(names):
    def included(name):
        p=PurePosixPath(name)
        return (name in EXPLICIT or (p.suffix=='.py' and (len(p.parts)==1 or
                len(p.parts)==2 and p.parts[0] in ('prefill_lab','scripts','tests')))
                or p.parent.as_posix()=='experiments' and p.name.startswith('prefill-service-v1.'))
    return {name for name in names if included(name)}


def snapshot_sources(repository,capture,destination):
    acceptance=read(capture,'launch.json')['cpu_acceptance']
    if acceptance['commit']!=COMMIT:raise ValueError('Unsupported capture revision')
    sources=read(capture,'collection-sources.json')
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=repository,text=True).splitlines()
    if set(sources)!=source_paths(names):raise ValueError('Captured source set differs from Git revision')
    for name,expected in sources.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe source path')
        raw=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=repository)
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Git/capture source hash mismatch: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return sources


def original_audit(repository,capture):
    with tempfile.TemporaryDirectory(prefix='pr32-native-effective-audit-') as temp:
        root=Path(temp);sources=snapshot_sources(repository,capture,root)
        env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
        result=subprocess.run([sys.executable,str(root/'scripts/check_prefill_effective_calibration.py'),
                               'audit',str(capture)],cwd=root,env=env,text=True,capture_output=True,timeout=120)
        if result.returncode:raise ValueError('Original source audit failed: '+result.stdout+result.stderr)
        verdict=json.loads(result.stdout)
        if verdict!={'audit_valid':True,'candidate_passed':True,'model_accepted':False}:
            raise ValueError('Original audit verdict differs')
    return dict(commit=COMMIT,matched_source_files=len(sources),**verdict)


def archive_identity(archive,capture):
    """Bind every extracted file to the immutable uploaded archive; no extraction."""
    files={p.relative_to(capture).as_posix():p for p in capture.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in capture.rglob('*')):raise ValueError('Linked capture path')
    matched=set();total=0
    with tarfile.open(archive) as bundle:
        for member in bundle:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk() or member.isdev():
                raise ValueError('Unsafe archive member')
            if not name.parts or name.parts[0]!=capture.name:raise ValueError('Archive root mismatch')
            if member.isdir():continue
            if not member.isfile():raise ValueError('Unsupported archive member')
            relative=PurePosixPath(*name.parts[1:]).as_posix()
            if relative in matched or relative not in files:raise ValueError('Duplicate/missing archive file')
            total+=member.size
            if total>512*1024*1024:raise ValueError('Archive exceeds analysis bound')
            raw=bundle.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest()!=hashlib.sha256(files[relative].read_bytes()).hexdigest():
                raise ValueError('Extracted file differs from archive: '+relative)
            matched.add(relative)
    if matched!=set(files):raise ValueError('Extra extracted file')
    index={name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
           for name,path in sorted(files.items())}
    return dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                file_count=len(index),uncompressed_bytes=total),index


def endpoint_summary(cells):
    """Request-position median errors; also report equal weight per case."""
    if not cells:raise ValueError('No score cells')
    result={}
    for metric in ('ttft_s','completion_s'):
        selected=[c for c in cells if c['metric']==metric]
        if not selected:raise ValueError('Missing endpoint metric')
        cases=defaultdict(list)
        for c in selected:cases[c['case']].append(abs(c['error_s']))
        result[metric]=dict(request_positions=len(selected),cases=len(cases),
            mean_absolute_error_s=sum(abs(c['error_s']) for c in selected)/len(selected),
            case_balanced_mae_s=sum(sum(v)/len(v) for v in cases.values())/len(cases),
            max_absolute_error_s=max(abs(c['error_s']) for c in selected),
            max_relative_error=max(abs(c['error_s'])/c['measured_median_s'] for c in selected),
            signed_mean_error_s=sum(c['error_s'] for c in selected)/len(selected),
            all_pilot_gates_pass=all(c['passed'] for c in selected))
    return result


def training_sensitivity(capture):
    """Post-run training-only leave-one-block-out diagnostic, never a new model."""
    for name in ('prefill_lab/effective_calibration.py','prefill_lab/async_predictor.py','prefill_lab/contracts.py',
                 'prefill_lab/causal.py','scripts/predict_prefill_async.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=read(capture,'collection-sources.json')[name]:
            raise ValueError('Training diagnostic implementation differs from captured source: '+name)
    from prefill_lab.effective_calibration import fit
    p=read(capture,'calibration-plan.json');rows=read(capture,'endpoints.json');runtime=read(capture,'runtime.json')
    result=[]
    for block in (1,2,3,4):
        subset=dict(p,order=[t for t in p['order'] if not(t['stage']=='train' and t['block']==block)])
        try:
            f=fit(rows,subset,runtime,origin='native_controlled_endpoints')
            result.append(dict(excluded_train_block=block,status='fitted',parameters_s=f['parameters_s'],
                               local_sensitivity=f['local_sensitivity'],training_sha256=f['training_sha256']))
        except ValueError as exc:result.append(dict(excluded_train_block=block,status='failed',reason=str(exc)))
    return result


def analyze(capture,archive,repository=ROOT,*,sensitivity=True):
    identity,index=archive_identity(archive,capture)
    audit=original_audit(repository,capture)
    p=read(capture,'calibration-plan.json');f=read(capture,'frozen.json');h=read(capture,'held-report.json')
    new_cases={c['id'] for c in p['cases'] if c['split']=='held_case'}
    fresh=[c for c in h['endpoint_scores']['cells'] if c['case'] in new_cases]
    references=[c for c in h['endpoint_scores']['cells'] if c['case'] not in new_cases]
    modes=Counter();requests=steps=light=off=0
    for t in p['order']:
        c=read(capture,t['trial_id']+'.json');requests+=len(c['requests'])
        if t['observer']=='light':light+=1
        else:off+=1
        for w in c['workers']:
            steps+=len(w['rows']);modes.update(r['runtime_mode'] for r in w['rows'])
    # Recheck all LIGHT work signatures, including unscored warmups.
    forecasts=read(capture,'pre-held-predictions.json');rows=read(capture,'endpoints.json')
    if any(r['work_signature']!=forecasts[r['case']]['work_signature'] for r in rows if r['observer']=='light'):
        raise ValueError('Own-work signature mismatch in scored or warmup LIGHT trial')
    result=dict(schema='native-effective-analysis-v1',archive=identity,source_audit=audit,
        native_cpu_tests=read(capture,'native-cpu-acceptance/acceptance.json')['tests'],
        collection=read(capture,'calibration.json'),trial_counts=dict(light=light,off=off),
        request_instances=requests,native_observed_steps=steps,mode_counts=dict(modes),
        fresh_held=endpoint_summary(fresh),
        fresh_held_by_background={str(d):endpoint_summary([c for c in fresh if c['case'].endswith('-D'+str(d))]) for d in (0,16)},
        held_training_cell_references=endpoint_summary(references),
        fresh_held_cells=fresh,reference_cells=references,parameters_s=f['parameters_s'],
        parameter_names=['bias','prefill/2048','decode_context/32768','attention_pairs/2097152'],
        local_sensitivity=f['local_sensitivity'],frozen_sha256=f['sha256'],
        training_leave_one_block_out=training_sensitivity(capture) if sensitivity else [],
        effective_pilot_passed=True,physical_costs_identified=False,model_accepted=False,
        complete_experiment_passed=False,adaptive_cap_accepted=False,mpc_accepted=False)
    return result,index


def arrival_proposal(capture):
    """Development-only frozen-model screen; this is not a native collection plan."""
    from prefill_lab.async_predictor import rollout
    from prefill_lab.effective_calibration import costs
    from scripts.predict_prefill_async import runtime_inputs
    p=read(capture,'calibration-plan.json');f=read(capture,'frozen.json')
    scope,dispatch=runtime_inputs(read(capture,'runtime.json'));cases=[]
    for d in (0,16):
        for offset in (50,200,400,800):
            case=dict(id=f'mixed-plus-sentinel-at{offset}-D{d}',background_decoders=d,
                      foreground_tokens=[1024,1024,7168,7168],foreground_offsets_ms=[0,0,0,0],
                      sentinel=dict(input_tokens=512,arrival_ms=offset))
            pred=rollout(case,costs(f['parameters_s'],p),dispatch,scope=scope)
            cases.append(dict(case=case,development_prediction=dict(
                sentinel_target_to_first_s=pred['requests']['sentinel']['ttft_s'],
                sentinel_queue_residence_s=pred['requests']['sentinel']['input_queue_wait_s'],
                backgrounds_outlive_foreground=pred['background_outlives_foreground'])))
    return dict(schema='effective-delayed-arrival-proposal-v2',status='design_only_not_collection_ready',
                native_collection_implemented=False,source_frozen_sha256=f['sha256'],
                source_runtime_sha256=f['runtime_sha256'],cost_basis_changed=False,development_only=True,
                cases=cases,proposed_repeats_per_case=4,proposed_observers=['off','light'],
                primary_endpoint='declared target-to-first output; consistent with ideal-arrival forecast',
                secondary_endpoints=['actual queue-post-to-first','queue-post-to-engine-enqueue','posting lag'],
                maximum_absolute_posting_lag_s=.005,
                new_host_rule='Fresh original six-cell training and freeze on each new host; archived coefficients are not portable acceptance',
                required_before_native=['source-bound exact-token order/endpoint routing',
                    'freeze before all scored delayed trials, no refitting from delayed labels',
                    'frozen workload/quality/drift gates and before/after zero-offset anchors',
                    'own-work/output/dispatcher checks and recorded effective CPU/GPU identities',
                    'unit/end-to-end fixture tests of target versus posted versus enqueue clocks'],
                model_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)


def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    cells=result['fresh_held_cells']
    for ax,metric,label in zip(axes,('ttft_s','completion_s'),('First output','Final output')):
        rows=[c for c in cells if c['metric']==metric]
        maximum=max(max(c['predicted_s'],c['measured_median_s']) for c in rows)*1.07
        ax.plot([0,maximum],[0,maximum],color='.5',linestyle='--',linewidth=1,label='Perfect prediction')
        for d,color in ((0,'#2878B5'),(16,'#D1495B')):
            group=[c for c in rows if c['case'].endswith('-D'+str(d))]
            ax.scatter([c['measured_median_s'] for c in group],[c['predicted_s'] for c in group],
                       color=color,s=38,alpha=.8,label=f'{d} background decoders')
        summary=result['fresh_held'][metric]
        ax.set(title=f'{label}: MAE {summary["mean_absolute_error_s"]*1000:.1f} ms',
               xlabel='Measured median (s)',ylabel='Frozen prediction (s)',xlim=(0,maximum),ylim=(0,maximum))
        ax.grid(alpha=.2);ax.set_aspect('equal')
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Eight fresh held workload cells · four OFF repeats per request position\nFixed B2048 effective candidate; zero-offset arrivals only',fontsize=11)
    fig.savefig(output/'held-endpoints.png',dpi=180);fig.savefig(output/'held-endpoints.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result,index=analyze(args.capture.resolve(),args.archive.resolve())
    save(output/'analysis.json',result);save(output/'input-files.json',index)
    save(output/'arrival-proposal.json',arrival_proposal(args.capture.resolve()));plot(result,output)
    print(json.dumps(dict(audit_valid=True,candidate_passed=result['effective_pilot_passed'],
                         fresh_held=result['fresh_held'],model_accepted=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
