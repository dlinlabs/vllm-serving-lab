"""Reproduce entry-clock development after immutable original-source audit/refit."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_prefill_queue_probe as original
from prefill_lab import entry_clock
from prefill_lab.contracts import save


def run(capture,archive,output):
    capture,archive,output=map(Path,(capture,archive,output))
    proof=original.run(capture,archive,output/'predecessors')
    read=lambda n:json.loads((capture/n).read_text())
    plan=read('queue-probe-plan.json');runtime=read('runtime.json')
    model=json.loads((output/'predecessors/native-fit/frozen.json').read_text())
    captures=[read(t['trial_id']+'.json') for t in plan['order'] if t['stage']=='train' and t['observer']=='light']
    frozen=entry_clock.freeze(captures,plan,runtime,model);save(output/'freeze.json',frozen)
    report=original.score(read('queue-probe-rows.json'),plan,frozen['predictions'],frozen['cells'])
    save(output/'development-report.json',report)
    summary=dict(schema='entry-clock-development-v1',original_audit=proof['original_audit'],
        archive=proof['archive'],readiness_passed=frozen['readiness_passed'],calibration_phase_score_passed=report['queue_budget_passed'],
        first_commit_max_absolute_error_s=max(abs(c['error_s']) for c in frozen['first_commit_readiness']['checks']),
        fresh_confirmation_performed=False,model_selected=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    save(output/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.capture,args.archive,args.output)))
