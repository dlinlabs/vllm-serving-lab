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
from scripts.reproduce_cap_miss_causes import verify,old_files,new_files

GUARD=('prefill_design.py','scripts/check_prefill_itl.py','greybox_simulator.py',
       'evaluate_low_cap_frozen.py','scripts/check_prefill_budget.py','scripts/reproduce_cap_miss_causes.py',
       'scripts/build_prefill_design.py','prefill_identifiability.py')
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
        historical.append(dict(capture=name,observed_exporter=itl_tail(path),
            simulated=predict(accepted_requests(path),(60.,300.))))
    path=a.new_evidence/'results/low-cap-switch-v1/low-cap-switch-v1-fixed-32-r1'
    transfers=[]
    for score_start,score_end in ((60,120),(180,240)):
        replay_start=score_start-20
        transfers.append(dict(replay_start=replay_start,score_start=score_start,score_end=score_end,cap=32,
            lead_in_seconds=20,score_window_local=[20,80],
            realized_output_oracle=predict(accepted_requests(path,replay_start=replay_start,replay_end=score_end),(20.,80.)),
            forced128_output_sensitivity=predict(accepted_requests(path,replay_start=replay_start,replay_end=score_end,forced_output=128),(20.,80.))))
    result=dict(role='development mechanism diagnostic and conditional transfer preregistration; no GPU collection',
        historical_working_budget=2048,historical_config_verified=False,verified_source_files=verified,
        code_sha256=before,
        capture_sha256=sources,historical=historical,transfer_slices=transfers,
        limitations=[
            'Running-first FIFO assumes every live decoder emits one token in each executed step; counter/async semantics must be checked.',
            'A gap is counted by its right token timestamp: start < timestamp <= end. Exporter scrapes cover slightly different endpoints.',
            'Historical realized output lengths and admissions are retrospective oracle inputs; transfer counts do not predict fresh admission/EOS.',
            'Transfer simulation starts empty at the lead-in, not with reconstructed historical state. Boundary state must be measured.',
            'k values are sensitivities with unchanged linear coefficients, not fitted parameters or physical upper bounds.',
            'A new transfer run also needs a frozen conditional replay on its actual accepted arrivals/outputs; this cannot pass a causal forecasting gate.',
            'Tail mismatches can falsify the joint scenario; matching tails do not uniquely establish scheduler budget or nonlinear cost.'
        ])
    if before != {n:digest(ROOT/n) for n in GUARD}:raise ValueError('Source changed during analysis')
    write(a.output,result)
    print(json.dumps(dict(historical=len(historical),transfer_slices=len(transfers),verified=verified),indent=2))


if __name__=='__main__':main()
