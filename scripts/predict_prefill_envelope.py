"""CPU support construction / snapshot replay; no native or cap acceptance."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prefill_lab import state_envelope as envelope,queue_confirmation as qc
from prefill_lab.async_predictor import Request
from prefill_lab.contracts import save
from prefill_lab.live_http import clock_domain


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='operation',required=True)
    fit=commands.add_parser('support',help='Empirical support from past TRAIN only; not a confidence interval')
    for name in ('controlled','base-model','output'):fit.add_argument('--'+name,type=Path,required=True)
    predict=commands.add_parser('predict',help='Conditional envelope from a real pre-ADD snapshot')
    for name in ('snapshot','request','runtime','support','output'):predict.add_argument('--'+name,type=Path,required=True)
    predict.add_argument('--replay-at',type=float,help='Explicit recorded evaluation time for offline replay; clock domain is not independently verified')
    predict.add_argument('--deadline',type=float,help='Seconds from the snapshot cut, not from a later admission decision')
    a=parser.parse_args(argv);load=lambda p:json.loads(p.read_text())
    if a.operation=='support':
        plan=qc.protocol();runtime=load(a.controlled/'runtime.json')
        captures=[load(a.controlled/(t['trial_id']+'.json')) for t in plan['order'] if t['stage']=='train' and t['observer']=='light']
        result=envelope.from_training(captures,plan,runtime,load(a.base_model));name='support.json'
    else:
        snapshot=load(a.snapshot);offline=a.replay_at is not None
        result=envelope.forecast(snapshot,Request(**load(a.request)),load(a.support),load(a.runtime),
            now=a.replay_at if offline else time.monotonic(),domain=snapshot.get('clock_domain') if offline else clock_domain(),deadline_s=a.deadline)
        result['clock_check']='recorded_metadata_only' if offline else 'current_boot_and_monotonic_clock'
        result['online_compute_age_qualified']=False;name='envelope.json'
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/name,result)
    print(json.dumps(dict(output=str(a.output/name),native_coverage_qualified=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)))


if __name__=='__main__':main()
