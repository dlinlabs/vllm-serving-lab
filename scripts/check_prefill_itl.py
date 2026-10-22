"""CPU ITL mechanism sensitivity and conditional transfer preregistration, no refitting.

Historical arrivals, admissions and realized outputs are oracle development inputs.
Transfer counts freeze those paths, NOT predictions of a new host's admission/EOS.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate_low_cap_frozen as lf
from prefill_design import simulate
from scripts.build_prefill_design import digest,write
from scripts.check_prefill_budget import itl_tail
from prefill_histogram import exporter_histogram,simulated_histogram,discrepancy,shared_step_reference,support_and_magnitude
from scripts.reproduce_cap_miss_causes import verify,old_files,new_files

GUARD=('prefill_design.py','scripts/check_prefill_itl.py','greybox_simulator.py',
       'evaluate_low_cap_frozen.py','scripts/check_prefill_budget.py','scripts/reproduce_cap_miss_causes.py',
       'scripts/build_prefill_design.py','prefill_identifiability.py','prefill_histogram.py','prefill_validation.py')
SCENARIOS=[(8192,0.),(2048,0.),(2048,1e-9),(2048,1.5e-9),(2048,3.15e-9)]


def accepted_requests(path, *, replay_start=0., replay_end=300., forced_output=None):
    _,table,_=lf.load_capture_compatible(path)
    trace=json.loads((path/'trace.json').read_text())
    tokens={r['trace_id']:r['input_tokens'] for r in trace['requests']}
    client={(r.get('request_id') or r.get('client_request_id')):r for r in json.loads((path/'client.json').read_text())['requests'] if r.get('workload')=='trace'}
    requests=[]
    for i,rid in enumerate(table['request_id']):
        if not table['admitted'][i]:continue
        # Use scheduled arrival to select the offered cohort; preserve accepted-at delay.
        if not replay_start <= table['sched'][i] < replay_end:continue
        row=client[rid];output=forced_output if forced_output is not None else row['output_tokens']
        if not row['success'] or not isinstance(output,int) or output<1:
            raise ValueError('Expected successful admitted requests with positive realized outputs')
        requests.append(dict(id=rid,tokens=int(tokens[table['trace_id'][i]]),
            at=float(table['acc'][i]-replay_start),output=output))
        if requests[-1]['at']<0:raise ValueError('Acceptance before scheduled replay start')
    return requests


def predict(requests,window):
    return {f'B{b}-fifo-k{k}':simulate(requests,b,quadratic=k,itl_window=window)['itl'] for b,k in SCENARIOS}


def histogram_predictions(requests,observed,rebase=0.):
    window=[t-rebase for t in observed['actual_window_s']]
    rows={}
    for b,k in SCENARIOS:
        steps=[];simulate(requests,b,quadratic=k,steps_out=steps)
        histogram=simulated_histogram(steps,window,observed['upper_bounds_s'])
        rows[f'B{b}-fifo-k{k}']=dict(histogram=histogram)
    reference=shared_step_reference(rows['B2048-fifo-k0.0']['histogram'])
    # Freeze the same common-positive mask for EVERY scenario; do not select
    # flattering bins independently for each candidate. Primary still keeps zeros.
    common=[i for i,a in enumerate(observed['counts']) if a>0 and all(rows[k]['histogram']['counts'][i]>0
            for k in ('B2048-fifo-k0.0','B2048-fifo-k1.5e-09'))]
    for row in rows.values():
        row['joint_discrepancy']=support_and_magnitude(observed,row['histogram'],reference,common)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-evidence',type=Path,required=True)
    p.add_argument('--new-evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():p.error('Choose a fresh output file')
    before={n:digest(ROOT/n) for n in GUARD}
    verified=dict(old=verify(a.old_evidence,old_files()),new=verify(a.new_evidence,new_files()))
    sources={};historical=[]
    pairs=[(a.old_evidence/'results/cap-range-v1',n) for n in ('cap-range-v1-fixed-32-r1','cap-range-v1-fixed-32-r2','cap-range-v1-fixed-16-r1')]
    pairs += [(a.new_evidence/'results/low-cap-switch-v1',n) for n in ('low-cap-switch-v1-fixed-32-r1','low-cap-switch-v1-fixed-16-r1')]
    for root,name in pairs:
        path=root/name
        sources[name]={n:digest(path/n) for n in ('trace.json','client.json','gateway.jsonl','manifest.json','engine_observations.jsonl')}
        requests=accepted_requests(path);observed=exporter_histogram(path,60,300)
        historical.append(dict(capture=name,observed_exporter=itl_tail(path),
            output_lengths=sorted({r['output'] for r in requests}),accepted_requests=len(requests),
            simulated=predict(requests,(60.,300.)),full_histogram=dict(observed=observed,
                scenarios=histogram_predictions(requests,observed))))
    path=a.new_evidence/'results/low-cap-switch-v1/low-cap-switch-v1-fixed-32-r1'
    transfers=[]
    for score_start,score_end in ((60,120),(180,240)):
        replay_start=score_start-20
        requests=accepted_requests(path,replay_start=replay_start,replay_end=score_end)
        observed=exporter_histogram(path,score_start,score_end)
        forced=accepted_requests(path,replay_start=replay_start,replay_end=score_end,forced_output=128)
        transfers.append(dict(full_histogram=dict(observed=observed,scenarios=histogram_predictions(requests,observed,replay_start)),
            forced128_equals_realized=all(r['output']==128 for r in requests),
            replay_start=replay_start,score_start=score_start,score_end=score_end,cap=32,
            lead_in_seconds=20,score_window_local=[20,80],
            realized_output_oracle=predict(requests,(20.,80.)),
            forced128_output_sensitivity=predict(forced,(20.,80.))))
    references=[]
    for i,j,label in ((0,1,'same_host_old32_replicates'),(0,3,'cross_host_32'),(2,4,'cross_host_16')):
        left=historical[i];right=historical[j]
        reference=left['full_histogram']['scenarios']['B2048-fifo-k0.0']['joint_discrepancy']['shared_reference']
        references.append(dict(id=label,left=left['capture'],right=right['capture'],
            requested_window_s=[60,300],same_host=i==0 and j==1,
            shared_step_floor_factor=discrepancy(left['full_histogram']['observed'],right['full_histogram']['observed'],count_floor=reference['count_floor'])['worst_bin_rate_factor'],
            legacy_floor5_factor=discrepancy(left['full_histogram']['observed'],right['full_histogram']['observed'])['worst_bin_rate_factor'],
            shared_reference=reference,
            role='One descriptive pair, not a calibrated envelope; cross-host differences are not within-host repeat noise. Endpoints differ slightly.'))
    for row in historical+transfers:
        row['replicate_reference_ids']=[r['id'] for r in references]
        row['replicate_reference_caveat']='240s reference pairs, not window-matched transfer replicates. One same-host pair cannot calibrate factor2 or accept a model.'
    result=dict(role='development mechanism diagnostic and conditional transfer preregistration; no GPU collection',
        historical_working_budget=2048,historical_config_verified=False,verified_source_files=verified,
        code_sha256=before,
        capture_sha256=sources,historical=historical,transfer_slices=transfers,replicate_references=references,
        limitations=[
            'Running-first FIFO assumes every live decoder emits one token in each executed step; counter/async semantics must be checked.',
            'A gap is counted by its right token timestamp: start < timestamp <= end. Exporter scrapes cover slightly different endpoints.',
            'All five historical admitted paths have output128; forced128 is duplicate evidence, not independent sensitivity. Admission timing remains retrospective; future natural EOS may differ.',
            'Transfer simulation starts empty at the lead-in, not with reconstructed historical state. Boundary state must be measured.',
            'Neither k nor a k bound is supported by a selected tail bin; the full neighboring-bin discrepancies are reported.',
            'Support flags and shared toy-equivalent-step magnitude replace legacy floor5 headlines. Floor1/5 and common-positive bins are sensitivities, not acceptance.',
            'Gaps share executed steps. Future uncertainty must resample step/time blocks, not independent gaps; no effective sample size is inferred from toy steps.',
            'A new transfer run also needs a frozen conditional replay on its actual accepted arrivals/outputs; this cannot pass a causal forecasting gate.',
            'Tail mismatches can falsify the joint scenario; matching tails do not uniquely establish scheduler budget or nonlinear cost.'
        ])
    if before != {n:digest(ROOT/n) for n in GUARD}:raise ValueError('Source changed during analysis')
    write(a.output,result)
    print(json.dumps(dict(historical=len(historical),transfer_slices=len(transfers),verified=verified),indent=2))


if __name__=='__main__':main()
